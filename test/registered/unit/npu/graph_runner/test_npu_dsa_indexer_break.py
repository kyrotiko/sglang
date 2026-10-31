"""CPU regression tests for the NPU DSA indexer's prefill graph boundary."""

import ast
import functools
import unittest
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_SRT = Path(__file__).resolve().parents[5] / "python/sglang/srt"


def _load_symbols(path, names, namespace):
    # Execute the real bodies without importing accelerator-only dependencies.
    path = _SRT / path
    nodes = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(nodes) == len(names)
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def _batch(lengths, *, prefix=0, slot=16, prefill=True):
    return SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.int64),
        out_cache_loc=torch.arange(slot, slot + 4),
        seq_lens=torch.tensor([prefix + n for n in lengths]),
        extend_seq_lens=torch.tensor(lengths),
        global_num_token_non_padded_cpu=sum(lengths),
        attn_cp_metadata=None,
        forward_mode=SimpleNamespace(
            is_extend_without_speculative=lambda: prefill,
            is_extend=lambda: prefill,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
        ),
    )


class _Stream:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def wait_stream(self, stream):
        self.events.append((self.name, "wait_stream", stream.name))

    def record_event(self):
        return self.name

    def wait_event(self, event):
        self.events.append((self.name, "wait_event", event))


class TestNpuDSAIndexerBreak(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(
            batch=_batch([4]), graph=True, prefill=True, dcp=False
        )
        self.events, self.gathers, self.writes, self.kernels = [], [], [], []
        self.main_stream = _Stream("main", self.events)
        self.q_stream = _Stream("query", self.events)
        self.weight_stream = _Stream("weights", self.events)
        self.multi_stream = False
        self.pool = SimpleNamespace(
            start_layer=0,
            dtype=torch.bfloat16,
            indexer_hadamard_128=None,
            index_k_scale_buffer=None,
            get_index_k_buffer=lambda _: torch.zeros(1),
            get_index_k_scale_buffer=lambda _: torch.ones(1),
            set_index_k_buffer=lambda layer, loc, k: self.writes.append(
                (loc.clone(), k.clone())
            ),
            set_index_k_scale_buffer=Mock(),
        )
        self.backend = SimpleNamespace(is_draft_worker=False)
        self._refresh_metadata()
        self.capture_var = ContextVar("dsa_test_capture", default=None)
        bcg = _load_symbols(
            "model_executor/runner_backend_utils/breakable_cuda_graph/"
            "breakable_cuda_graph.py",
            {
                "_weak_ref_if_tensor",
                "_EagerOnGraphCallable",
                "_copy_into_output",
                "eager_on_graph",
                "BreakableCUDAGraph",
            },
            {
                "torch": torch,
                "functools": functools,
                "_current_capture_var": self.capture_var,
                "_current_stream_var": ContextVar("dsa_test_stream", default=None),
                "get_device_module": lambda: SimpleNamespace(
                    current_stream=lambda: self.main_stream
                ),
            },
        )
        self.graph = bcg["BreakableCUDAGraph"]()

        def collective(out, local):
            self.gathers.append((local.shape[0], out.shape[0]))
            out.copy_(torch.cat([local, local]))

        def kernel(*, quant=False, **kwargs):
            self.kernels.append(kwargs)
            n = kwargs["query"].shape[0]
            value = int(kwargs["actual_seq_lengths_key"].sum())
            out = torch.full((n, 1, 2), value, dtype=torch.int32)
            return out if quant else (out, None)

        self.ns = _load_symbols(
            "layers/attention/dsa/dsa_npu_indexer.py",
            {"bcg_npu_dsa_indexer", "DSANPUIndexerMixin", "scattered_to_tp_attn_full"},
            {
                "torch": torch,
                "eager_on_graph": bcg["eager_on_graph"],
                "is_in_breakable_cuda_graph": lambda: self.state.graph,
                "is_in_prefill_graph_scope": lambda: self.state.prefill,
                "get_forward_batch": lambda: self.state.batch,
                "get_attn_backend": lambda: self.backend,
                "get_token_to_kv_pool": lambda: self.pool,
                "get_parallel": lambda: SimpleNamespace(dcp_enabled=self.state.dcp),
                "_use_ag_after_qlora": False,
                "attn_tp_all_gather_into_tensor": collective,
                "get_indexer_weight_stream": lambda: self.weight_stream,
                "envs": SimpleNamespace(
                    SGLANG_NPU_USE_MULTI_STREAM=SimpleNamespace(
                        get=lambda: self.multi_stream
                    )
                ),
                "_quantize_npu_indexer_activation": lambda x, *_: (
                    x,
                    torch.ones(x.shape[:-1]),
                ),
                "torch_npu": SimpleNamespace(
                    npu_rotary_mul=lambda x, *_: x,
                    npu_lightning_indexer=kernel,
                    npu_quant_lightning_indexer=functools.partial(kernel, quant=True),
                ),
            },
        )
        self.indexer = self.ns["DSANPUIndexerMixin"]()
        self.indexer.rotary_emb = SimpleNamespace(
            is_neox_style=True, cos_sin_cache=torch.ones(64, 64)
        )
        self.indexer.alt_stream = None
        self.indexer.hidden_size = self.indexer.head_dim = 128
        self.indexer.rope_head_dim = 64
        self.indexer.n_heads, self.indexer.index_topk = 1, 2
        self.indexer.dsa_enable_prefill_cp = False
        self.indexer.wq_b = self.indexer.wk = lambda x: (
            x[0] if isinstance(x, tuple) else x,
            None,
        )
        self.indexer.k_norm = lambda x: x
        self.indexer.weights_proj = lambda x: (torch.ones(len(x), 1), None)
        rotary_patch = patch.object(
            torch.ops.npu, "npu_rotary_mul", lambda x, *_: x, create=True
        )
        rotary_patch.start()
        self.addCleanup(rotary_patch.stop)

    def _refresh_metadata(self):
        self.backend.forward_metadata = SimpleNamespace(
            seq_lens_cpu_int=None,
            seq_lens=self.state.batch.seq_lens,
            block_tables=torch.full((len(self.state.batch.seq_lens), 2), 7),
        )

    @contextmanager
    def _capture(self):
        owner = self

        class Capture:
            cuda_graph = owner.graph

            def _end_current_segment(self):
                self.cuda_graph._append_segment(
                    SimpleNamespace(replay=lambda: None), False
                )

            def _begin_new_segment(self):
                pass

        capture = Capture()
        token = self.capture_var.set(capture)
        try:
            yield
            capture._end_current_segment()
        finally:
            self.capture_var.reset(token)

    def _run(self, *, gather=False, dynamic_scale=None):
        return self.indexer.forward_npu(
            torch.ones(2 if gather else 4, 128, dtype=torch.bfloat16),
            torch.ones(4, 128, dtype=torch.bfloat16),
            torch.arange(4),
            self.state.batch,
            0,
            gather,
            dynamic_scale,
        )

    def test_replay_reads_live_batch_and_preserves_output_address(self):
        with self._capture():
            out = self._run()
        self.assertEqual(len(self.graph._segments), 2)
        self.assertEqual(len(self.graph._break_fns), 1)
        self.assertEqual(self.kernels, [])
        self.assertEqual(self.writes, [])
        address = out.data_ptr()
        for lengths, prefix, slot in (([3], 6, 16), ([1, 1], 8, 32), ([0], 0, 48)):
            self.state.batch = _batch(lengths, prefix=prefix, slot=slot)
            self._refresh_metadata()
            self.graph.replay()
            n = sum(lengths)
            self.assertEqual(out.data_ptr(), address)
            self.assertTrue((out[n:] == -1).all())
            if n:
                self.assertEqual(
                    self.writes[-1][0].tolist(), list(range(slot, slot + n))
                )
                self.assertEqual(self.writes[-1][1].shape[0], n)
                call = self.kernels[-1]
                self.assertEqual(call["query"].shape[0], n)
                self.assertEqual(call["weights"].shape[0], n)
                self.assertEqual(
                    call["actual_seq_lengths_query"].tolist(),
                    self.state.batch.extend_seq_lens.cumsum(0).tolist(),
                )
                self.assertEqual(
                    call["block_table"].data_ptr(),
                    self.backend.forward_metadata.block_tables.data_ptr(),
                )
                self.assertTrue((out[:n] == int(self.state.batch.seq_lens.sum())).all())
        self.assertEqual(len(self.writes), 2)
        self.assertEqual(len(self.kernels), 2)

    def test_tp_gathers_keep_bucket_shape_before_trimming_quantized_kv(self):
        self.ns["_use_ag_after_qlora"] = True
        self.pool.index_k_scale_buffer = object()
        with self._capture():
            out = self._run(gather=True, dynamic_scale=torch.ones(4, 1))
        self.state.batch = _batch([3], slot=32)
        self._refresh_metadata()
        self.graph.replay()
        self.assertEqual(self.gathers, [(2, 4), (2, 4)])
        self.assertEqual(self.writes[0][0].tolist(), [32, 33, 34])
        self.assertEqual(self.writes[0][1].shape[0], 3)
        _, loc, scale = self.pool.set_index_k_scale_buffer.call_args.args
        self.assertEqual(loc.tolist(), [32, 33, 34])
        self.assertEqual(scale.shape[0], 3)
        self.assertEqual(self.kernels[0]["query_dequant_scale"].shape[0], 3)
        self.assertTrue((out[3:] == -1).all())

    def test_multistream_projection_dependencies_are_joined(self):
        self.multi_stream = True
        self.indexer.alt_stream = self.q_stream
        fake_npu = SimpleNamespace(
            current_stream=lambda: self.main_stream, stream=lambda _: nullcontext()
        )
        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(torch.Tensor, "record_stream", return_value=None),
            self._capture(),
        ):
            self._run()
        self.state.batch = _batch([3])
        self._refresh_metadata()
        with (
            patch.object(torch, "npu", fake_npu, create=True),
            patch.object(torch.Tensor, "record_stream", return_value=None),
        ):
            self.graph.replay()
        self.assertIn(("main", "wait_event", "query"), self.events)
        self.assertIn(("main", "wait_event", "weights"), self.events)
        self.assertEqual(self.kernels[0]["query"].shape[0], 3)

    def test_interleaved_rope_with_padding_and_both_indexer_dtypes(self):
        self.indexer.rotary_emb = Mock(
            is_neox_style=False,
            cos_sin_cache=torch.ones(64, 64),
            side_effect=lambda positions, q, k: (q, k),
        )
        for quant in (False, True):
            with self.subTest(quant=quant):
                self.graph._segments.clear()
                self.graph._break_fns.clear()
                self.kernels.clear()
                self.writes.clear()
                self.pool.index_k_scale_buffer = object() if quant else None
                with self._capture():
                    out = self._run()
                self.state.batch = _batch([1, 2], prefix=4, slot=32)
                self._refresh_metadata()
                self.graph.replay()
                self.assertEqual(len(self.kernels), 1)
                self.assertEqual(self.kernels[0]["query"].shape[0], 3)
                self.assertEqual(self.writes[0][0].tolist(), [32, 33, 34])
                self.assertTrue((out[:3] == 11).all())
                self.assertTrue((out[3:] == -1).all())

    def test_native_dispatch_for_eager_decode_and_speculative_modes(self):
        native = Mock(return_value=object())
        self.indexer._forward_npu_eager = native
        for graph, prefill_scope, prefill_mode in (
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ):
            with self.subTest(graph=graph, scope=prefill_scope, mode=prefill_mode):
                self.state.graph, self.state.prefill = graph, prefill_scope
                self.state.batch = _batch([4], prefill=prefill_mode)
                self.assertIs(self._run(), native.return_value)
                self.assertIs(native.call_args.args[3], self.state.batch)
                self.assertEqual(native.call_args.kwargs, {})
        self.assertEqual(self.graph._break_fns, [])

    def test_cp_and_dcp_are_rejected_before_capture(self):
        for cp, dcp in ((True, False), (False, True)):
            with self.subTest(cp=cp, dcp=dcp):
                self.indexer.dsa_enable_prefill_cp, self.state.dcp = cp, dcp
                with self.assertRaisesRegex(RuntimeError, "CP/DCP"):
                    self._run()
        self.assertEqual(self.kernels, [])

    def test_invalid_live_token_count_does_not_write_kv(self):
        with self._capture():
            self._run()
        self.state.batch.global_num_token_non_padded_cpu = 5
        with self.assertRaises(AssertionError):
            self.graph.replay()
        self.assertEqual(self.writes, [])


if __name__ == "__main__":
    unittest.main()

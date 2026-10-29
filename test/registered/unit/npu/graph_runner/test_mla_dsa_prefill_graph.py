"""CPU regression coverage for NPU MLA/DSA graph boundaries and dummy KV."""

import ast
import functools
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.graph_runner.mla_prefill_capture import (
    MLAPrefillCapture,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[5] / "python/sglang/srt"


def _load_symbols(path, names, namespace):
    # These units exercise the real Python bodies without importing NPU kernels.
    path = _ROOT / path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(nodes) == len(names)
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


class _Breaks:
    """Record eager callbacks; graph mechanics are covered by backend tests."""

    def __init__(self):
        self.enabled = True
        self.capturing = False
        self.calls = []

    def decorate(self, fn):
        def call(*args, **kwargs):
            if self.capturing:
                self.calls.append(functools.partial(fn, *args, **kwargs))
            else:
                return fn(*args, **kwargs)

        return call


def _batch(lengths, *, prefix=0, slot=16, bucket=4):
    n = sum(lengths)
    return SimpleNamespace(
        batch_size=len(lengths),
        extend_seq_lens_cpu=lengths,
        extend_seq_lens=torch.tensor(lengths),
        seq_lens=torch.tensor([prefix + length for length in lengths]),
        input_ids=torch.zeros(bucket, dtype=torch.int64),
        positions=torch.arange(bucket),
        req_pool_indices=torch.empty(len(lengths), dtype=torch.int64),
        out_cache_loc=torch.arange(slot, slot + bucket),
        global_num_token_non_padded_cpu=n,
        attn_cp_metadata=None,
        forward_mode=SimpleNamespace(
            is_extend_without_speculative=lambda: True,
            is_extend=lambda: True,
            is_draft_extend_v2=lambda: False,
            is_target_verify=lambda: False,
        ),
    )


class _Pool:
    def __init__(self):
        self.free_slots = [1, 2, 3]
        self.req_to_token = torch.full((4, 16), 99, dtype=torch.int32)

    def alloc_rows(self, count):
        if count > len(self.free_slots):
            return None
        rows = self.free_slots[:count]
        del self.free_slots[:count]
        return rows

    def write(self, indices, values):
        self.req_to_token[indices] = values.to(torch.int32)

    def free_rows(self, rows):
        self.free_slots.extend(rows)


class TestMLAPrefillCapture(unittest.TestCase):
    def test_request_tables_bucket_reuse_and_cleanup(self):
        pool = _Pool()
        loc = torch.tensor([8, 9, 10, 11, 16, 17, 18])
        allocator = SimpleNamespace(
            page_size=2, alloc_extend=Mock(return_value=loc), free=Mock()
        )
        capture = MLAPrefillCapture(allocator, pool, [4, 3])
        self.addCleanup(capture.close)
        for lengths in ([4, 3], [4, 1], [3]):
            batch = _batch(lengths, bucket=sum(lengths))
            capture.prepare_batch(batch)
            torch.testing.assert_close(batch.out_cache_loc, loc[: sum(lengths)])
            self.assertEqual(batch.req_pool_indices.tolist(), [1, 2][: len(lengths)])
            torch.testing.assert_close(
                batch.positions, torch.cat([torch.arange(n) for n in lengths])
            )
        self.assertEqual(pool.req_to_token[1, :4].tolist(), [8, 9, 10, 11])
        self.assertEqual(pool.req_to_token[2, :3].tolist(), [16, 17, 18])
        allocator.alloc_extend.assert_called_once()
        self.assertTrue((pool.req_to_token[0] == 99).all())
        self.assertTrue((pool.req_to_token[3] == 99).all())
        capture.close()
        capture.close()
        allocator.free.assert_called_once_with(loc)
        self.assertEqual(sorted(pool.free_slots), [1, 2, 3])
        self.assertTrue((pool.req_to_token[1:3] == 0).all())

    def test_allocation_and_table_write_failure_release_owned_resources(self):
        for fail_write in (False, True):
            with self.subTest(fail_write=fail_write):
                pool = _Pool()
                allocator = SimpleNamespace(
                    page_size=2,
                    alloc_extend=Mock(
                        return_value=torch.arange(8, 12) if fail_write else None
                    ),
                    free=Mock(),
                )
                if fail_write:
                    pool.write = Mock(side_effect=RuntimeError("table write failed"))
                with self.assertRaises(RuntimeError):
                    MLAPrefillCapture(allocator, pool, [4])
                self.assertEqual(sorted(pool.free_slots), [1, 2, 3])
                self.assertEqual(allocator.free.call_count, int(fail_write))

    def test_page_size_one_and_request_exhaustion(self):
        pool = _Pool()
        allocator = SimpleNamespace(
            page_size=1, alloc=Mock(return_value=torch.arange(3)), free=Mock()
        )
        capture = MLAPrefillCapture(allocator, pool, [3])
        capture.close()
        allocator.alloc.assert_called_once_with(3)
        pool.free_slots = []
        with self.assertRaisesRegex(RuntimeError, "request rows"):
            MLAPrefillCapture(allocator, pool, [3])
        self.assertEqual(allocator.alloc.call_count, 1)

    def test_backend_selects_mla_or_dsv4_and_reuses_allocation(self):
        mla, dsv4 = Mock(), Mock()
        parallel = SimpleNamespace(attn_cp_size=1, attn_dcp_size=1)
        ns = _load_symbols(
            "hardware_backend/npu/graph_runner/breakable_npu_graph_backend.py",
            {"BreakableNPUGraphBackend"},
            {
                "BreakableCudaGraphBackend": object,
                "contextmanager": __import__("contextlib").contextmanager,
                "is_deepseek_v4": lambda config: config.v4,
                "get_parallel": lambda: parallel,
                "MLAPrefillCapture": mla,
                "DSV4PrefillCapture": dsv4,
            },
        )
        cls = ns["BreakableNPUGraphBackend"]
        for v4 in (False, True):
            backend = cls.__new__(cls)
            backend._capture_kv = None
            backend._cuda_graph_runner = SimpleNamespace(
                max_num_tokens=4, max_context_size=None
            )
            backend._model_runner = SimpleNamespace(
                model_config=SimpleNamespace(
                    context_len=3, hf_config=SimpleNamespace(v4=v4)
                ),
                attn_backend=SimpleNamespace(use_mla=True),
                token_to_kv_pool_allocator=object(),
                req_to_token_pool=object(),
            )
            cls.prepare_capture_batch(backend, "first")
            cls.prepare_capture_batch(backend, "second")
            factory = dsv4 if v4 else mla
            factory.assert_called_once_with(
                backend._model_runner.token_to_kv_pool_allocator,
                backend._model_runner.req_to_token_pool,
                [3, 1],
            )
            self.assertEqual(factory.return_value.prepare_batch.call_count, 2)
        backend._model_runner.model_config.hf_config.v4 = False
        parallel.attn_dcp_size = 2
        with self.assertRaisesRegex(NotImplementedError, "CP/DCP"):
            cls.prepare_capture_batch(backend, "unsupported")


class TestMLACacheBreaks(unittest.TestCase):
    def setUp(self):
        self.breaks = _Breaks()
        self.context = SimpleNamespace(batch=_batch([4]))
        self.pool = SimpleNamespace(
            get_kv_buffer=Mock(return_value=(object(), object())), set_kv_buffer=Mock()
        )
        self.fused_calls = []

        def fused(latent, weight, cos, sin, slots, *caches, **kwargs):
            self.fused_calls.append((slots.clone(), kwargs["cache_mode"]))
            return None, None, latent[..., 2:].clone(), latent[..., :2].clone()

        self.ns = _load_symbols(
            "hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py",
            {
                "bcg_mla_kv_norm_rope_cache",
                "bcg_mla_store_kv",
                "forward_mha_prepare_npu",
            },
            {
                "torch": torch,
                "eager_on_graph": self.breaks.decorate,
                "get_forward_batch": lambda: self.context.batch,
                "get_token_to_kv_pool": lambda: self.pool,
                "is_fia_nz": lambda: True,
                "torch_npu": SimpleNamespace(
                    npu_kv_rmsnorm_rope_cache=fused, npu_interleave_rope=lambda q, *_: q
                ),
                "is_in_breakable_cuda_graph": lambda: self.breaks.enabled,
                "is_in_prefill_graph_scope": lambda: True,
            },
        )
        self.model = SimpleNamespace(
            layer_id=0,
            kv_lora_rank=2,
            qk_rope_head_dim=2,
            kv_a_layernorm=SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6),
        )

    def test_fused_cache_break_uses_live_slots_and_static_outputs(self):
        latent = torch.arange(16, dtype=torch.float32).view(4, 1, 4)
        rope_out, kv_out = torch.empty(4, 1, 2), torch.empty(4, 2)
        self.breaks.capturing = True
        self.ns["bcg_mla_kv_norm_rope_cache"](
            self.model, latent, torch.ones(4, 2), torch.zeros(4, 2), rope_out, kv_out
        )
        self.assertEqual(self.fused_calls, [])
        for n, slot in ((3, 16), (1, 32), (0, 48)):
            self.context.batch = _batch([n], slot=slot)
            self.breaks.calls[0]()
            torch.testing.assert_close(rope_out[:n], latent[:n, :, 2:])
            torch.testing.assert_close(kv_out[:n], latent[:n, 0, :2])
            self.assertTrue((rope_out[n:] == 0).all())
            self.assertTrue((kv_out[n:] == 0).all())
            if n:
                self.assertEqual(self.fused_calls[-1], (unittest.mock.ANY, "PA_NZ"))
                self.assertEqual(
                    self.fused_calls[-1][0].tolist(), list(range(slot, slot + n))
                )
        self.assertEqual(len(self.fused_calls), 2)

    def test_unfused_store_has_no_padding_or_idle_writes(self):
        kv, rope = torch.ones(4, 2), torch.ones(4, 1, 2)
        self.breaks.capturing = True
        self.ns["bcg_mla_store_kv"](self.model, kv, rope)
        for n, slot in ((2, 16), (1, 32), (0, 48)):
            self.context.batch = _batch([n], slot=slot)
            self.breaks.calls[0]()
            if n:
                args = self.pool.set_kv_buffer.call_args.args
                self.assertEqual(args[1].tolist(), list(range(slot, slot + n)))
                self.assertEqual(args[2].shape[0], n)
        self.assertEqual(self.pool.set_kv_buffer.call_count, 2)

    def test_mha_prepare_keeps_eager_results_with_both_rope_paths(self):
        class Norm:
            weight = torch.ones(2)
            variance_epsilon = 1e-6

            def __call__(self, x):
                return x

        class Rope:
            def __call__(self, positions, q, k):
                return q, k

            def get_cos_sin_cache(self, positions, dtype, offsets=None):
                return torch.ones(len(positions), 2), torch.zeros(len(positions), 2)

        model = self.model
        model.q_lora_rank, model.num_local_heads = None, 1
        model.qk_head_dim = 4
        model.qk_nope_head_dim, model.v_head_dim = 2, 2
        model.kv_a_layernorm, model.rotary_emb = Norm(), Rope()
        model.q_proj = model.kv_a_proj_with_mqa = lambda x: (x[:, :4].clone(), None)
        model.kv_b_proj = lambda x: (torch.cat([x, x], dim=-1), None)
        model._concat_and_cast_mha_k = lambda k, rope, batch: torch.cat(
            [k, rope], dim=-1
        )
        x = torch.arange(32, dtype=torch.float32).view(4, 8)
        for yarn in (False, True):
            with self.subTest(yarn=yarn):
                model.use_deepseek_yarn_rope = yarn
                batch = _batch([3])
                self.context.batch = batch
                self.breaks.enabled = True
                graph_result = self.ns["forward_mha_prepare_npu"](
                    model, batch.positions, x, batch, None, False
                )
                self.breaks.enabled = False
                batch.out_cache_loc = batch.out_cache_loc[:3]
                eager_result = self.ns["forward_mha_prepare_npu"](
                    model, batch.positions[:3], x[:3], batch, None, False
                )
                for graph_tensor, eager_tensor in zip(
                    graph_result[:3], eager_result[:3]
                ):
                    torch.testing.assert_close(graph_tensor[:3], eager_tensor)
                if yarn:
                    self.assertTrue((graph_result[1][3:] == 0).all())
                    self.assertTrue((graph_result[2][3:] == 0).all())


class TestDSAIndexerBreak(unittest.TestCase):
    def _setup_indexer(self, *, neox, gather, quant):
        breaks, context = _Breaks(), SimpleNamespace(batch=_batch([4]))
        cache_writes, gathers, kernel_calls = [], [], []
        pool = SimpleNamespace(
            start_layer=0,
            dtype=torch.bfloat16,
            indexer_hadamard_128=None,
            index_k_scale_buffer=object() if quant else None,
            get_index_k_buffer=lambda _: torch.zeros(1),
            get_index_k_scale_buffer=lambda _: torch.ones(1),
            set_index_k_buffer=lambda layer, loc, k: cache_writes.append(
                (loc.clone(), k.clone())
            ),
            set_index_k_scale_buffer=Mock(),
        )
        backend = SimpleNamespace(
            is_draft_worker=False,
            forward_metadata=SimpleNamespace(
                seq_lens_cpu_int=None,
                seq_lens=torch.tensor([4]),
                block_tables=torch.ones(2, 2, dtype=torch.int32),
            ),
        )

        def collective(out, local):
            gathers.append((local.shape[0], out.shape[0]))
            out.copy_(torch.cat([local, local], dim=0))

        def indexer_kernel(**kwargs):
            kernel_calls.append(kwargs)
            n = kwargs["query"].shape[0]
            value = int(kwargs["actual_seq_lengths_key"].sum())
            out = torch.full((n, 1, 2), value, dtype=torch.int32)
            return out if quant else (out, None)

        class Rope:
            is_neox_style = neox
            cos_sin_cache = torch.ones(64, 64)

            def __call__(self, positions, q, k):
                return q, k

        ns = _load_symbols(
            "layers/attention/dsa/dsa_npu_indexer.py",
            {"bcg_npu_dsa_indexer", "DSANPUIndexerMixin", "scattered_to_tp_attn_full"},
            {
                "torch": torch,
                "eager_on_graph": breaks.decorate,
                "get_forward_batch": lambda: context.batch,
                "is_in_breakable_cuda_graph": lambda: breaks.enabled,
                "is_in_prefill_graph_scope": lambda: True,
                "get_attn_backend": lambda: backend,
                "get_token_to_kv_pool": lambda: pool,
                "get_parallel": lambda: SimpleNamespace(dcp_enabled=False),
                "_use_ag_after_qlora": gather,
                "attn_tp_all_gather_into_tensor": collective,
                "envs": SimpleNamespace(
                    SGLANG_NPU_USE_MULTI_STREAM=SimpleNamespace(get=lambda: False)
                ),
                "_quantize_npu_indexer_activation": lambda x, *_: (
                    x,
                    torch.ones(x.shape[:-1]),
                ),
                "torch_npu": SimpleNamespace(
                    npu_rotary_mul=lambda x, *_: x,
                    npu_lightning_indexer=indexer_kernel,
                    npu_quant_lightning_indexer=indexer_kernel,
                ),
            },
        )
        indexer = ns["DSANPUIndexerMixin"]()
        indexer.rotary_emb, indexer.alt_stream = Rope(), None
        indexer.hidden_size, indexer.head_dim, indexer.rope_head_dim = 128, 128, 64
        indexer.n_heads, indexer.index_topk, indexer.dsa_enable_prefill_cp = 1, 2, False
        indexer.wq_b = indexer.wk = lambda x: (
            (x[0] if isinstance(x, tuple) else x),
            None,
        )
        indexer.k_norm = lambda x: x
        indexer.weights_proj = lambda x: (torch.ones(len(x), 1), None)
        return indexer, breaks, context, pool, gathers, cache_writes, kernel_calls

    def test_live_requests_padding_quantization_and_padded_tp_gather(self):
        with patch.object(
            torch.ops.npu, "npu_rotary_mul", lambda x, *_: x, create=True
        ):
            for neox, gather, quant in (
                (True, False, False),
                (True, True, False),
                (False, False, False),
                (True, True, True),
                (False, False, True),
            ):
                with self.subTest(neox=neox, gather=gather, quant=quant):
                    indexer, breaks, context, pool, gathers, writes, kernels = (
                        self._setup_indexer(neox=neox, gather=gather, quant=quant)
                    )
                    x = torch.ones(2 if gather else 4, 128, dtype=torch.bfloat16)
                    q = torch.ones(4, 128, dtype=torch.bfloat16)
                    positions = torch.arange(4)
                    breaks.capturing = True
                    out = indexer.forward_npu(x, q, positions, context.batch, 0, gather)
                    self.assertEqual(kernels, [])
                    address = out.data_ptr()
                    for lengths, prefix, slot in (
                        ([3], 6, 16),
                        ([1, 1], 8, 32),
                        ([0], 0, 48),
                    ):
                        context.batch = _batch(lengths, prefix=prefix, slot=slot)
                        breaks.calls[0]()
                        n = sum(lengths)
                        self.assertEqual(out.data_ptr(), address)
                        self.assertTrue((out[n:] == -1).all())
                        if n:
                            self.assertEqual(
                                writes[-1][0].tolist(), list(range(slot, slot + n))
                            )
                            self.assertEqual(writes[-1][1].shape[0], n)
                            self.assertEqual(kernels[-1]["query"].shape[0], n)
                            self.assertEqual(kernels[-1]["weights"].shape[0], n)
                            self.assertEqual(
                                kernels[-1]["actual_seq_lengths_query"].tolist(),
                                list(torch.tensor(lengths).cumsum(0).tolist()),
                            )
                            self.assertTrue(
                                (out[:n] == int(context.batch.seq_lens.sum())).all()
                            )
                    self.assertEqual(len(writes), 2)
                    self.assertEqual(len(kernels), 2)
                    self.assertEqual(gathers, [(2, 4)] * 4 if gather else [])
                    self.assertEqual(
                        pool.set_index_k_scale_buffer.call_count, 2 if quant else 0
                    )

    def test_non_graph_dispatch_keeps_native_path(self):
        with patch.object(
            torch.ops.npu, "npu_rotary_mul", lambda x, *_: x, create=True
        ):
            indexer, breaks, context, _, _, writes, kernels = self._setup_indexer(
                neox=True, gather=False, quant=False
            )
            breaks.enabled = False
            out = indexer.forward_npu(
                torch.ones(4, 128),
                torch.ones(4, 128),
                torch.arange(4),
                context.batch,
                0,
            )
            self.assertEqual(tuple(out.shape), (4, 2))
            self.assertEqual(breaks.calls, [])
            self.assertEqual(len(writes), 1)
            self.assertEqual(len(kernels), 1)


if __name__ == "__main__":
    unittest.main()

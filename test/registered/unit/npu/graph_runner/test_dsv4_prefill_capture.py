"""CPU coverage for DSV4 prefill capture allocation and request-table wiring."""

import ast
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.hardware_backend.npu.graph_runner.dsv4_prefill_capture import (
    DSV4PrefillCapture,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclass
class _Bundle:
    out_full_loc: torch.Tensor
    out_swa_loc: torch.Tensor
    out_c4_loc: torch.Tensor
    out_c128_loc: torch.Tensor


class _Allocator:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = 0
        self.c128_attn_allocator = SimpleNamespace(page_size=2, free=Mock())
        self.c128_page_refcount = torch.zeros(20, dtype=torch.int32)
        self.free = Mock()
        self.cache = SimpleNamespace(clear_request_scoped_state=Mock())

    def get_kvcache(self):
        return self.cache

    def alloc_extend(self, prefix, prefix_cpu, seq, seq_cpu, last, count, **kwargs):
        self.calls += 1
        torch.testing.assert_close(prefix, torch.zeros_like(seq))
        torch.testing.assert_close(prefix_cpu, torch.zeros_like(seq_cpu))
        torch.testing.assert_close(seq, seq_cpu)
        torch.testing.assert_close(last, torch.full_like(seq, -1))
        assert count == int(seq.sum())
        assert kwargs["req_to_token_pool"] is not None
        assert len(kwargs["req_pool_indices"]) == len(seq)
        if self.fail:
            return None
        # Distinct, nonzero physical pages, including a partially filled tail.
        full = torch.cat(
            [torch.arange(8 + i * 512, 8 + i * 512 + n) for i, n in enumerate(seq)]
        )
        c128 = torch.cat(
            [torch.arange(2 + i * 4, 2 + i * 4 + n // 128) for i, n in enumerate(seq)]
        )
        return _Bundle(full, full + 1000, full[full % 4 == 3] // 4, c128)

    def release_c128_pages(self, pages):
        self.c128_page_refcount[pages] -= 1
        self.c128_attn_allocator.free(pages * 2)


class _ReqPool:
    def __init__(self, allocator, *, fail_write=None):
        self.allocator = allocator
        self.fail_write = fail_write
        self.free_slots = [1, 2, 3]
        self.req_to_token = torch.full((4, 512), 999, dtype=torch.int32)
        self.req_to_c128_sidecar = torch.zeros((4, 8), dtype=torch.int32)

    def alloc_rows(self, count):
        if count > len(self.free_slots):
            return None
        rows = self.free_slots[:count]
        del self.free_slots[:count]
        return rows

    def free_rows(self, rows):
        self.free_slots.extend(rows)

    def write(self, indices, values):
        self.req_to_token[indices] = values.to(torch.int32)

    def write_c128(self, indices, values):
        row, positions = indices
        if row == self.fail_write:
            raise RuntimeError("injected sidecar write failure")
        pages = values[::2] // 2
        self.req_to_c128_sidecar[row, : len(pages)] = pages.to(torch.int32)
        self.allocator.c128_page_refcount[pages] += 1


def _batch(lengths):
    return SimpleNamespace(
        batch_size=len(lengths),
        extend_seq_lens_cpu=lengths,
        req_pool_indices=torch.empty(len(lengths), dtype=torch.int64),
        out_cache_loc=torch.empty(sum(lengths), dtype=torch.int64),
        positions=torch.empty(sum(lengths), dtype=torch.int64),
    )


class TestDSV4PrefillCapture(unittest.TestCase):
    def test_multi_request_tables_and_bucket_reuse(self):
        allocator = _Allocator()
        pool = _ReqPool(allocator)
        capture = DSV4PrefillCapture(allocator, pool, [256, 132])
        self.addCleanup(capture.close)
        batch = _batch([256, 132])
        capture.prepare_batch(batch)
        self.assertEqual(batch.req_pool_indices.tolist(), [1, 2])
        torch.testing.assert_close(
            pool.req_to_token[1, :256].long(), batch.out_cache_loc[:256]
        )
        torch.testing.assert_close(
            pool.req_to_token[2, :132].long(), batch.out_cache_loc[256:]
        )
        self.assertEqual(pool.req_to_c128_sidecar[1, 0].item(), 1)
        self.assertEqual(pool.req_to_c128_sidecar[2, 0].item(), 3)
        torch.testing.assert_close(
            batch.positions, torch.cat([torch.arange(256), torch.arange(132)])
        )
        bundle = batch.out_cache_loc_dsv4
        self.assertIs(bundle.out_full_loc, batch.out_cache_loc)
        torch.testing.assert_close(bundle.out_swa_loc, batch.out_cache_loc + 1000)
        self.assertEqual(bundle.out_c4_loc.numel(), 97)
        self.assertEqual(bundle.out_c128_loc.numel(), 3)
        small = _batch([4])
        capture.prepare_batch(small)
        self.assertEqual(allocator.calls, 1)
        self.assertEqual(small.out_cache_loc_dsv4.out_c4_loc.numel(), 1)
        self.assertEqual(small.out_cache_loc_dsv4.out_c128_loc.numel(), 0)
        tail = _batch([256, 5])
        capture.prepare_batch(tail)
        self.assertEqual(tail.out_cache_loc_dsv4.out_c128_loc.tolist(), [2, 3])
        # The padding row and other free rows must not be touched.
        self.assertTrue((pool.req_to_token[0] == 999).all())
        self.assertTrue((pool.req_to_token[3] == 999).all())

    def test_close_releases_full_swa_c128_and_rows_once(self):
        allocator = _Allocator()
        pool = _ReqPool(allocator)
        capture = DSV4PrefillCapture(allocator, pool, [256, 132])
        batch = _batch([4])
        capture.prepare_batch(batch)
        capture.close()
        capture.close()
        allocator.free.assert_called_once()
        self.assertEqual(sorted(pool.free_slots), [1, 2, 3])
        self.assertTrue((allocator.c128_page_refcount == 0).all())
        self.assertTrue((pool.req_to_c128_sidecar == 0).all())
        self.assertTrue((pool.req_to_token[1:3] == 0).all())
        freed = torch.cat(
            [call.args[0] for call in allocator.c128_attn_allocator.free.call_args_list]
        )
        self.assertEqual(sorted(freed.tolist()), [2, 6])
        # Capture batch owns its tensor views after the temporary pages are freed.
        self.assertEqual(batch.out_cache_loc_dsv4.out_c4_loc.numel(), 1)

    def test_allocation_failure_returns_request_rows(self):
        allocator = _Allocator(fail=True)
        pool = _ReqPool(allocator)
        with self.assertRaisesRegex(RuntimeError, "FULL/SWA/C128"):
            DSV4PrefillCapture(allocator, pool, [256])
        self.assertEqual(sorted(pool.free_slots), [1, 2, 3])
        allocator.free.assert_not_called()

    def test_partial_sidecar_failure_releases_all_pages(self):
        allocator = _Allocator()
        pool = _ReqPool(allocator, fail_write=2)
        with self.assertRaisesRegex(RuntimeError, "injected sidecar"):
            DSV4PrefillCapture(allocator, pool, [256, 132])
        self.assertEqual(sorted(pool.free_slots), [1, 2, 3])
        self.assertTrue((allocator.c128_page_refcount == 0).all())
        allocator.free.assert_called_once()
        freed = torch.cat(
            [call.args[0] for call in allocator.c128_attn_allocator.free.call_args_list]
        )
        self.assertEqual(sorted(freed.tolist()), [2, 6])

    def test_request_exhaustion_leaves_pool_untouched(self):
        allocator = _Allocator()
        pool = _ReqPool(allocator)
        pool.free_slots = []
        with self.assertRaisesRegex(RuntimeError, "request rows"):
            DSV4PrefillCapture(allocator, pool, [256])
        self.assertEqual(allocator.calls, 0)
        self.assertTrue((pool.req_to_token == 999).all())

    def test_short_bucket_has_empty_compressed_tensors(self):
        allocator = _Allocator()
        pool = _ReqPool(allocator)
        capture = DSV4PrefillCapture(allocator, pool, [3])
        self.addCleanup(capture.close)
        batch = _batch([3])
        capture.prepare_batch(batch)
        self.assertEqual(batch.out_cache_loc_dsv4.out_c4_loc.numel(), 0)
        self.assertEqual(batch.out_cache_loc_dsv4.out_c128_loc.numel(), 0)


class TestDSV4NPUPrefillBreaks(unittest.TestCase):
    def setUp(self):
        # Load just the Python break bodies: the model's imports need NPU/CUDA
        # extensions, whereas selecting the live request must be testable on CPU.
        source = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/srt/models/deepseek_v4.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"bcg_dsv4_npu_store_cache", "bcg_dsv4_npu_sources"}
        bodies = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.assertEqual(len(bodies), 2)
        self.context = SimpleNamespace(forward_batch=None)
        self.backend = SimpleNamespace(
            store_cache=Mock(), forward_core_compressor=Mock()
        )
        self.functions = {
            "torch": torch,
            "eager_on_graph": lambda fn: fn,
            "get_tc_piecewise_forward_context": lambda: self.context,
            "get_attn_backend": lambda: self.backend,
        }
        exec(
            compile(ast.Module(body=bodies, type_ignores=[]), str(source), "exec"),
            self.functions,
        )

    def test_breaks_follow_live_requests_and_ignore_padding(self):
        layer = SimpleNamespace(layer_id=7, indexer=Mock(), compressor=object())
        x = torch.arange(16).view(4, 4)
        stored = []

        def store_cache(*, layer_id, swa_k, forward_batch):
            stored.append((forward_batch, forward_batch.out_cache_loc.clone(), swa_k))

        self.backend.store_cache.side_effect = store_cache
        for count, base in ((3, 16), (1, 32)):
            batch = SimpleNamespace(
                global_num_token_non_padded_cpu=count,
                out_cache_loc=torch.arange(base, base + 4),
                out_cache_loc_dsv4=object(),
            )
            original_loc = batch.out_cache_loc
            self.context.forward_batch = batch
            self.functions["bcg_dsv4_npu_store_cache"](7, x)
            self.functions["bcg_dsv4_npu_sources"](layer, x, x)
            self.assertIs(batch.out_cache_loc, original_loc)
            self.assertIs(stored[-1][0], batch)
            self.assertEqual(stored[-1][1].tolist(), list(range(base, base + count)))
            torch.testing.assert_close(stored[-1][2], x[:count])
            kwargs = layer.indexer.call_args.kwargs
            self.assertIs(kwargs["forward_batch"], batch)
            self.assertEqual(len(kwargs["x"]), count)
            self.assertEqual(len(kwargs["q_lora"]), count)
            args = self.backend.forward_core_compressor.call_args.args
            self.assertIs(args[1], batch)
            self.assertEqual(len(args[0]), count)
        self.context.forward_batch.global_num_token_non_padded_cpu = 0
        self.functions["bcg_dsv4_npu_store_cache"](7, x)
        self.functions["bcg_dsv4_npu_sources"](layer, x, x)
        self.assertEqual(len(stored), 2)
        self.assertEqual(layer.indexer.call_count, 2)
        self.assertEqual(self.backend.forward_core_compressor.call_count, 2)

    def test_cache_error_restores_original_location_tensor(self):
        original_loc = torch.arange(4)
        self.context.forward_batch = SimpleNamespace(
            global_num_token_non_padded_cpu=2, out_cache_loc=original_loc
        )
        self.backend.store_cache.side_effect = RuntimeError("injected store error")
        with self.assertRaisesRegex(RuntimeError, "injected store"):
            self.functions["bcg_dsv4_npu_store_cache"](7, torch.zeros(4, 2))
        self.assertIs(self.context.forward_batch.out_cache_loc, original_loc)


if __name__ == "__main__":
    unittest.main()

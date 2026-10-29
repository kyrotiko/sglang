"""Temporary, allocator-owned DSV4 KV storage for prefill graph warmup."""

from __future__ import annotations

import torch


class DSV4PrefillCapture:
    """Share the largest dummy allocation across capture buckets.

    Cache writes and compression run in eager breaks with the live batch at
    replay. Only the location tensors need to survive capture; the temporary
    request rows and physical KV pages can be returned after capture finishes.
    """

    def __init__(self, allocator, req_pool, seq_lens: list[int]):
        self.allocator = allocator
        self.req_pool = req_pool
        self.seq_lens = seq_lens
        self.rows = None
        self.bundle = None
        try:
            self.rows = req_pool.alloc_rows(len(seq_lens))
            if self.rows is None:
                raise RuntimeError(
                    "Not enough request rows for DSV4 NPU prefill capture"
                )
            device = req_pool.req_to_token.device
            self.indices = torch.tensor(self.rows, dtype=torch.int64, device=device)
            req_pool.req_to_token[self.indices] = 0
            req_pool.req_to_c128_sidecar[self.indices] = 0
            seq_cpu = torch.tensor(seq_lens, dtype=torch.int64, device="cpu")
            prefix_cpu = torch.zeros_like(seq_cpu)
            self.bundle = allocator.alloc_extend(
                prefix_cpu.to(device),
                prefix_cpu,
                seq_cpu.to(device),
                seq_cpu,
                torch.full_like(seq_cpu, -1, device=device),
                sum(seq_lens),
                req_pool_indices=self.indices,
                req_to_token_pool=req_pool,
            )
            if self.bundle is None:
                raise RuntimeError(
                    "Not enough FULL/SWA/C128 KV pages for DSV4 NPU prefill capture "
                    f"({sum(seq_lens)} tokens); lower --cuda-graph-max-bs-prefill"
                )
            token_offset = compressed_offset = 0
            for row, length in zip(self.rows, seq_lens):
                req_pool.write(
                    (row, slice(0, length)),
                    self.bundle.out_full_loc[token_offset : token_offset + length],
                )
                compressed_length = length // 128
                req_pool.write_c128(
                    (row, slice(0, compressed_length)),
                    self.bundle.out_c128_loc[
                        compressed_offset : compressed_offset + compressed_length
                    ],
                )
                allocator.get_kvcache().clear_request_scoped_state(row)
                token_offset += length
                compressed_offset += compressed_length
        except BaseException:
            self.close()
            raise

    def prepare_batch(self, batch) -> None:
        lengths = batch.extend_seq_lens_cpu
        num_tokens = sum(lengths)
        remaining = num_tokens
        expected_lengths = []
        for capacity in self.seq_lens:
            if remaining == 0:
                break
            length = min(capacity, remaining)
            expected_lengths.append(length)
            remaining -= length
        if remaining or lengths != expected_lengths:
            raise ValueError("DSV4 capture bucket does not fit the dummy KV allocation")
        batch.req_pool_indices.copy_(self.indices[: batch.batch_size])
        batch.out_cache_loc.copy_(self.bundle.out_full_loc[:num_tokens])
        batch.out_cache_loc_dsv4 = type(self.bundle)(
            out_full_loc=batch.out_cache_loc,
            out_swa_loc=self.bundle.out_swa_loc[:num_tokens],
            out_c4_loc=self.bundle.out_c4_loc[: sum(n // 4 for n in lengths)],
            out_c128_loc=self.bundle.out_c128_loc[: sum(n // 128 for n in lengths)],
        )
        # Positions restart at zero for each synthetic request.
        offset = 0
        for length in lengths:
            batch.positions[offset : offset + length].copy_(
                torch.arange(length, device=batch.positions.device)
            )
            offset += length

    def close(self) -> None:
        if self.rows is None:
            return
        allocator, req_pool = self.allocator, self.req_pool
        if self.bundle is not None:
            # A failed table write may leave some new C128 pages unregistered.
            # Registered pages go through the refcounted release; the remaining
            # pages are still exclusively owned by this dummy allocation.
            page_size = allocator.c128_attn_allocator.page_size
            pages = torch.unique(self.bundle.out_c128_loc.long() // page_size)
            retained = allocator.c128_page_refcount[pages] > 0
            allocator.c128_attn_allocator.free(pages[~retained] * page_size)
            allocator.release_c128_pages(pages[retained])
            allocator.free(self.bundle.out_full_loc)
            self.bundle = None
        for row in self.rows:
            allocator.get_kvcache().clear_request_scoped_state(row)
            req_pool.req_to_token[row].zero_()
            req_pool.req_to_c128_sidecar[row].zero_()
        req_pool.free_rows(self.rows)
        self.rows = None

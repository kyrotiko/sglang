"""Temporary request/KV storage for ordinary MLA and DSA prefill capture."""

from __future__ import annotations

import torch


class MLAPrefillCapture:
    def __init__(self, allocator, req_pool, seq_lens: list[int]):
        self.allocator = allocator
        self.req_pool = req_pool
        self.seq_lens = seq_lens
        self.rows = None
        self.loc = None
        try:
            self.rows = req_pool.alloc_rows(len(seq_lens))
            if self.rows is None:
                raise RuntimeError(
                    "Not enough request rows for MLA/DSA prefill capture"
                )
            device = req_pool.req_to_token.device
            self.indices = torch.tensor(self.rows, dtype=torch.int64, device=device)
            req_pool.req_to_token[self.indices] = 0
            seq_cpu = torch.tensor(seq_lens, dtype=torch.int64, device="cpu")
            prefix_cpu = torch.zeros_like(seq_cpu)
            if allocator.page_size == 1:
                self.loc = allocator.alloc(sum(seq_lens))
            else:
                self.loc = allocator.alloc_extend(
                    prefix_cpu.to(device),
                    prefix_cpu,
                    seq_cpu.to(device),
                    seq_cpu,
                    torch.full_like(seq_cpu, -1, device=device),
                    sum(seq_lens),
                )
            if self.loc is None:
                raise RuntimeError(
                    "Not enough KV pages for MLA/DSA prefill capture "
                    f"({sum(seq_lens)} tokens); lower --cuda-graph-max-bs-prefill"
                )
            offset = 0
            for row, length in zip(self.rows, seq_lens):
                req_pool.write(
                    (row, slice(0, length)), self.loc[offset : offset + length]
                )
                offset += length
        except BaseException:
            self.close()
            raise

    def prepare_batch(self, batch) -> None:
        lengths = batch.extend_seq_lens_cpu
        remaining = sum(lengths)
        expected = []
        for capacity in self.seq_lens:
            if remaining == 0:
                break
            length = min(capacity, remaining)
            expected.append(length)
            remaining -= length
        if remaining or lengths != expected:
            raise ValueError(
                "MLA/DSA capture bucket does not fit the dummy KV allocation"
            )
        batch.req_pool_indices.copy_(self.indices[: batch.batch_size])
        batch.out_cache_loc.copy_(self.loc[: sum(lengths)])
        offset = 0
        for length in lengths:
            batch.positions[offset : offset + length].copy_(
                torch.arange(length, device=batch.positions.device)
            )
            offset += length

    def close(self) -> None:
        if self.rows is None:
            return
        if self.loc is not None:
            self.allocator.free(self.loc)
            self.loc = None
        for row in self.rows:
            self.req_pool.req_to_token[row].zero_()
        self.req_pool.free_rows(self.rows)
        self.rows = None

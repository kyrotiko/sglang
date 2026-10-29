# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Breakable NPU graph backend: one torch.npu.NPUGraph per segment,
attention (and other host-input ops) eager between segments.

The parent capture context, capture loop and replay path are reused.
MLA/DSA/DSV4 warmup also needs temporary request/KV allocations. The per-segment
graph class is device-specific (_device_graph_cls in breakable_cuda_graph.py).
Segment graphs use bare capture_begin/capture_end — the graph.update channel
is not used because every op that consumes host-side values runs at
the eager break points, outside the captured segments.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import is_deepseek_v4
from sglang.srt.hardware_backend.npu.graph_runner.dsv4_prefill_capture import (
    DSV4PrefillCapture,
)
from sglang.srt.hardware_backend.npu.graph_runner.mla_prefill_capture import (
    MLAPrefillCapture,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.runtime_context import get_flags, get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class BreakableNPUGraphBackend(BreakableCudaGraphBackend):
    """NPU prefill graphs split at eager_on_graph break points."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        super().__init__(
            cuda_graph_runner,
            enable_memory_saver=enable_memory_saver,
            debug_eager=debug_eager,
        )
        # torch.compile cannot contain the eager break points this backend
        # depends on (NPU decode still compiles through NPUGraphRunner).
        assert not get_flags().capture.enable_torch_compile, (
            "breakable NPU prefill graph does not support torch.compile; "
            "disable --enable-torch-compile or prefill graph"
        )
        self._capture_kv = None

    def prepare_capture_batch(self, forward_batch: ForwardBatch) -> None:
        runner = self._cuda_graph_runner
        if is_deepseek_v4(self._model_runner.model_config.hf_config):
            capture_cls = DSV4PrefillCapture
        else:
            attn_backend = self._model_runner.attn_backend
            attn_backend = getattr(attn_backend, "prefill_backend", attn_backend)
            if not getattr(attn_backend, "use_mla", False):
                return
            parallel = get_parallel()
            if parallel.attn_cp_size > 1 or parallel.attn_dcp_size > 1:
                raise NotImplementedError(
                    "NPU MLA/DSA breakable prefill does not support CP/DCP yet"
                )
            capture_cls = MLAPrefillCapture
        if self._capture_kv is None:
            context_len = self._model_runner.model_config.context_len
            context_len = min(runner.max_context_size or context_len, context_len)
            lengths = [
                min(context_len, runner.max_num_tokens - start)
                for start in range(0, runner.max_num_tokens, context_len)
            ]
            self._capture_kv = capture_cls(
                self._model_runner.token_to_kv_pool_allocator,
                self._model_runner.req_to_token_pool,
                lengths,
            )
        self._capture_kv.prepare_batch(forward_batch)

    def _release_capture_kv(self) -> None:
        capture = getattr(self, "_capture_kv", None)
        if capture is not None:
            self._device_module.synchronize()
            capture.close()
            # Finish state/table clearing before another stream reuses the rows.
            self._device_module.synchronize()
            self._capture_kv = None

    @contextmanager
    def capture_session(self, stream):
        try:
            with super().capture_session(stream):
                yield
        finally:
            self._release_capture_kv()

    def close(self) -> None:
        super().close()
        self._release_capture_kv()

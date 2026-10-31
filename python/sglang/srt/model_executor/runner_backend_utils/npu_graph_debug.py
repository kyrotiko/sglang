"""Opt-in host diagnostics. No device synchronization or device tensor reads."""

from __future__ import annotations

import functools
import itertools
import json
import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_context = ContextVar("npu_graph_debug", default={})
_runs = itertools.count(1)
_events = itertools.count(1)


def _cpu_values(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            return _describe(value)
        return value.reshape(-1)[:8].tolist()
    if isinstance(value, (tuple, list)):
        return [_describe(item) for item in value[:8]]
    return _describe(value)


def _describe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, torch.Tensor):
        result = dict(
            shape=list(value.shape), dtype=str(value.dtype), device=str(value.device)
        )
        # Shape and storage addresses are host metadata, including for NPU tensors.
        try:
            result["ptr"] = hex(value.data_ptr())
        except Exception:
            result["ptr"] = "unavailable"
        return result
    if isinstance(value, (tuple, list)):
        return [_describe(item) for item in value[:8]]
    if isinstance(value, dict):
        return {key: _describe(item) for key, item in value.items()}
    if isinstance(value, functools.partial):
        result = {"fn": getattr(value.func, "__qualname__", type(value.func).__name__)}
        if value.args:
            result["layer"] = _describe(getattr(value.args[0], "layer_id", None))
        return result
    if hasattr(value, "forward_mode") and hasattr(value, "input_ids"):
        return dict(
            id=hex(id(value)),
            mode=getattr(value.forward_mode, "name", type(value.forward_mode).__name__),
            bs=getattr(value, "batch_size", None),
            real_tokens=getattr(value, "global_num_token_non_padded_cpu", None),
            seq_lens=_describe(getattr(value, "seq_lens", None)),
            seq_lens_cpu=_cpu_values(getattr(value, "seq_lens_cpu", None)),
            extend_lens_cpu=_cpu_values(getattr(value, "extend_seq_lens_cpu", None)),
            prefix_lens_cpu=_cpu_values(getattr(value, "extend_prefix_lens_cpu", None)),
            global_tokens_cpu=_cpu_values(
                getattr(value, "global_num_tokens_cpu", None)
            ),
            req_pool_indices=_describe(getattr(value, "req_pool_indices", None)),
            cache_loc=_describe(getattr(value, "out_cache_loc", None)),
        )
    return getattr(value, "__qualname__", type(value).__name__)


def npu_graph_debug(event: str, **fields):
    if not envs.SGLANG_NPU_GRAPH_DEBUG.get():
        return
    context = _context.get()
    if (
        context.get("phase") in ("warmup", "capture")
        and not envs.SGLANG_NPU_GRAPH_DEBUG_CAPTURE.get()
        and not event.endswith(".error")
    ):
        return
    # Diagnostics must never replace the original execution failure.
    try:
        ranks = {}
        try:
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
            for name in ("tp_rank", "attn_dp_rank", "attn_tp_rank", "moe_ep_rank"):
                ranks[name] = getattr(parallel, name)
        except Exception:
            pass
        payload = {
            "event": event,
            "seq": next(_events),
            "host_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            **ranks,
            **context,
            **fields,
        }
        logger.warning("[NPU_GRAPH_DEBUG] %s", json.dumps(_describe(payload)))
    except Exception:
        logger.warning("[NPU_GRAPH_DEBUG] event=%s metadata_unavailable", event)


@contextmanager
def npu_graph_debug_scope(phase: str, **fields):
    if not envs.SGLANG_NPU_GRAPH_DEBUG.get():
        yield
        return
    parent = _context.get()
    token = _context.set(
        {
            **parent,
            "run": parent["run"] if "run" in parent else next(_runs),
            "phase": phase,
            **fields,
        }
    )
    npu_graph_debug("phase.begin")
    try:
        yield
    except Exception as exc:
        npu_graph_debug("phase.error", error_type=type(exc).__name__, error=str(exc))
        raise
    else:
        npu_graph_debug("phase.return")
    finally:
        _context.reset(token)


def npu_graph_debug_call(event, fn, *args, **kwargs):
    if not envs.SGLANG_NPU_GRAPH_DEBUG.get():
        return fn(*args, **kwargs)
    npu_graph_debug(event + ".begin", args=args, kwargs=kwargs)
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        npu_graph_debug(event + ".error", error_type=type(exc).__name__, error=str(exc))
        raise
    npu_graph_debug(event + ".return", result=result)
    return result

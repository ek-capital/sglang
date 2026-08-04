# SPDX-License-Identifier: Apache-2.0
"""Low-overhead semantic NVTX ranges for production-faithful profiles.

Tensor replay capture and semantic profiling are mutually exclusive by design.
The ranges contain no CUDA synchronization or tensor inspection and may remain
in production code while disabled.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_ENABLED = os.environ.get("SGLANG_HOTLOOP_PROFILE", "0").lower() in {
    "1",
    "true",
    "yes",
}
_ARCHITECTURE_CONTEXT: ContextVar[dict[str, object]] = ContextVar(
    "sglang_hotloop_architecture_context", default={}
)


def hotloop_profile_enabled() -> bool:
    return _ENABLED


@contextmanager
def architecture_scope(**dimensions: object) -> Iterator[None]:
    """Attach stable architectural dimensions to nested semantic ranges.

    Speculative algorithms use this for ``phase`` and ``model_role`` while
    model adapters add dimensions such as ``block_type`` and ``layer``.  The
    context is inert when profiling is disabled and does not inspect tensors.
    """

    parent = _ARCHITECTURE_CONTEXT.get()
    merged = {
        **parent,
        **{key: value for key, value in dimensions.items() if value is not None},
    }
    token = _ARCHITECTURE_CONTEXT.set(merged)
    try:
        yield
    finally:
        _ARCHITECTURE_CONTEXT.reset(token)


@contextmanager
def semantic_range(name: str, **dimensions: object) -> Iterator[None]:
    if not _ENABLED or not torch.cuda.is_available():
        yield
        return
    merged_dimensions = {**_ARCHITECTURE_CONTEXT.get(), **dimensions}
    suffix = ",".join(
        f"{key}={merged_dimensions[key]}" for key in sorted(merged_dimensions)
    )
    label = f"sglang.hotloop/{name}" + (f"/{suffix}" if suffix else "")
    # NVTX is consumed by Nsight Systems. record_function emits the same
    # semantic boundary into Torch Chrome traces, which otherwise contain only
    # opaque kernel names when GPU-only activities are requested.
    with torch.profiler.record_function(label):
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

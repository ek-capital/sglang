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

import torch

_ENABLED = os.environ.get("SGLANG_HOTLOOP_PROFILE", "0").lower() in {
    "1",
    "true",
    "yes",
}


def hotloop_profile_enabled() -> bool:
    return _ENABLED


@contextmanager
def semantic_range(name: str, **dimensions: object) -> Iterator[None]:
    if not _ENABLED or not torch.cuda.is_available():
        yield
        return
    suffix = ",".join(f"{key}={dimensions[key]}" for key in sorted(dimensions))
    label = f"sglang.hotloop/{name}" + (f"/{suffix}" if suffix else "")
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

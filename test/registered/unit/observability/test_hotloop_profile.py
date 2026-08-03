# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).parents[4] / "python/sglang/srt/observability/hotloop_profile.py"
)
_SPEC = importlib.util.spec_from_file_location("hotloop_profile", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
hotloop_profile = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hotloop_profile)


def test_semantic_range_is_noop_when_disabled():
    with hotloop_profile.semantic_range("moe.layer", layer=1, phase="decode"):
        value = 42
    assert value == 42

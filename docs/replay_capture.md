# Utility-first replay capture

SGLang replay capture records small, complete operation boundaries for kernel
and runtime optimization. It is model-neutral: a model adapter registers the
semantic operations it supports, while the shared transport enforces a hard
byte ceiling and writes atomic safetensors bundles asynchronously.

Replay capture is intentionally separate from performance profiling. Capture
may select an unfused path and performs device-to-host copies; never use timing
from an armed capture run. For production-faithful Nsight profiles, leave
capture disabled, keep CUDA graphs enabled, and set:

```bash
export SGLANG_HOTLOOP_PROFILE=1
```

This emits `sglang.hotloop/*` NVTX ranges for speculative draft/verify,
KDA/MLA, MoE, AttnRes, and fused residual collectives without synchronizing the
GPU or inspecting tensors.

## Capture plan

Select complete semantic boundaries, not individual internal tensors:

```bash
export SGLANG_REPLAY_CAPTURE_DIR=/captures/run-001
export SGLANG_REPLAY_CAPTURE_PLAN=examples/replay_capture/kimi_k3_puzzles.json
export SGLANG_REPLAY_CAPTURE_MAX_GIB=4
export SGLANG_K3_CAPTURE_RANKS=all
export SGLANG_K3_CAPTURE_ASYNC=1
export SGLANG_K3_CAPTURE_LOCAL_DIR=/local-nvme/run-001
export SGLANG_K3_CAPTURE_ARM_FILE=/tmp/sglang-capture-arm
```

Replay plans fail closed unless the arm sentinel is configured outside the
capture destination. Create the sentinel only after startup and health checks;
the health endpoint may execute a small prefill. Async replay plans also
require an explicit local staging directory, so a FUSE/network destination
cannot accidentally become the shard hot-write path.

`SGLANG_REPLAY_CAPTURE_MAX_GIB` is a hard **per-rank** ceiling. The plan can
set a lower default and per-operation row limits. TP collective plans require
`SGLANG_K3_CAPTURE_RANKS=all`; validators must reject a collective dataset
unless the same `collective_sequence` exists on every rank.
Non-collective bundles are written only by global rank zero even in such a
mixed plan, preventing an unnecessary TP-sized multiplier in captured bytes.

Each bundle carries a semantic operation, schema version, stable bundle ID,
policy (`single`, `paired`, `sequence`, or `collective`), shapes, strides,
dtypes, execution metadata, and `timing_valid=false`. Missing required tensors
raise before bytes are queued, preventing large but unreplayable datasets.

Validate manifests before downloading payloads:

```bash
python -m sglang.srt.debug_utils.replay_capture_validate /captures/run-001 \
  --require speculative.draft_round \
  --require moe.tail \
  --require residual.attnres \
  --require collective.tp_residual \
  --minimum-bundles 20
```

The validator streams manifests, hashes shards with an ordered worker pool,
requires each rank's transactional close marker, checks schema completeness,
minimum bundle counts, consistent world size, and that every collective
sequence is present on every rank. Use `--allow-open` only for inspecting an
active run and `--skip-hashes` only for a quick non-integrity audit.

## Adding a model

1. Register semantic `ReplaySpec` entries in
   `sglang.srt.debug_utils.replay_capture` or a model adapter.
2. Attach `replay_capture_bundle()` at the narrow operation boundary.
   Call `replay_capture_set_forward_context()` once at the model-forward entry
   if the common runner has not already supplied the forward context.
3. Include all inputs, outputs, decisions, and mutable state before/after that
   a standalone replay needs. Reference checkpoint weights by revision/hash
   rather than copying them into every bundle.
4. Add a small plan under `examples/replay_capture/`.
5. Run a depth-reduced preflight and prove that every selected bundle replays
   without executing preceding layers before renting full-model hardware.

DeepSeek and Qwen can reuse the writer, plan, schemas, timing ranges, and
collective policy; only their model adapters and semantic operation specs are
model-specific.

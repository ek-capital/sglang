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

This emits matching `sglang.hotloop/*` NVTX and Torch `record_function` ranges
for speculative draft/verify/accept/commit, KDA projections/recurrence/norm/
output projection, MLA, MoE front/experts/reduce/tail, AttnRes, and fused
residual collectives without synchronizing the GPU or inspecting tensors.
Validate an exported Torch trace before trusting it:

```bash
python -m sglang.srt.observability.hotloop_profile_validate trace.json.gz \
  --requirements-file examples/replay_capture/kimi_k3_hotloop_profile_sections.txt
```

## Profile-guided two-pass workflow

Do not begin with a broad tensor dump. Each GPU topology gets an independent
run because its fused kernels, communication and bottleneck ordering may
differ.

1. Run the real serving recipe with captures disabled and representative,
   concurrency-saturating traffic. Exclude warmup and prefill from the steady
   decode trace.
2. Attribute every CUDA kernel to exactly one leaf section and build the timing
   table. Kernel-name ownership remains available under CUDA graph replay,
   while `sglang.hotloop/*` semantic spans provide stronger evidence when an
   exporter preserves them.
3. Select the top two or three actionable, capture-ready sections.
4. Perform a tiny armed preflight to measure bytes per complete replay case.
   Compile the selection and measured sizes into a hard-bounded plan.
5. Relaunch for capture. Timing from this pass is invalid. Validate hashes,
   rank completeness and replay correctness before uploading the package.
6. Optimize each extracted section offline. Every candidate must consume the
   same captured inputs and state, reproduce discrete outputs exactly and meet
   declared floating-point tolerances before its CUDA-event latency is compared.

Build the exclusive report and selection manifest:

```bash
python -m sglang.srt.observability.hotloop_report traces/rank*.json.gz \
  --model-family kimi_k3 --top-k 3 \
  --json-out profile.json --markdown-out profile.md \
  --selection-out selection.json
```

After measuring one complete case for every selected section, compile the
capture plan. The compiler refuses missing estimates and plans above the 5 TB
hard ceiling:

```bash
python -m sglang.srt.debug_utils.profile_guided_plan selection.json \
  --world-size 16 --cases-per-section 16 \
  --case-bytes speculative.draft=123456789 \
  --case-bytes attention.kda=234567890 \
  --case-bytes collective.tp_ep=345678901 \
  --out capture-plan.json
```

The speculative path is deliberately divided into draft generation, planning,
acceptance and state commit. The selector uses the smaller
`speculative.draft_generation@1`, `speculative.verify_plan@1`, or
`speculative.acceptance@1` contract when one of those leaves wins. The complete
`speculative.draft_round@2` contract remains available for an end-to-end change:
it ties proposal tokens and corrected draft logits to target verification
logits, the exact acceptance uniforms and masks, accepted/bonus/committed
tokens, and draft/target mutable state. This avoids paying for target vocabulary
logits when only draft generation or verify planning is under optimization.

Run an extracted replay package with:

```bash
python -m sglang.srt.debug_utils.replay_harness extracted/speculative.draft \
  --device cuda --benchmark --warmup 25 --iterations 200
```

### Kimi K3 topology presets

The command generator encodes the topology-specific official serving choices,
but requires the workload-derived Mamba memory ratio. It prints JSON containing
the argv, environment and a shell-safe command; it does not launch anything.

```bash
# One node, 8x B300: TP8/DCP8
python -m sglang.srt.observability.k3_profile_topology b300-8 \
  --model-path /mnt/kimi3 --draft-model-path /mnt/kimi3-draft \
  --mamba-full-memory-ratio 0.20 --profile

# Run on both nodes, changing --node-rank. 16x H200: TP16/EP16.
python -m sglang.srt.observability.k3_profile_topology h200-16 \
  --model-path /mnt/kimi3 --draft-model-path /mnt/kimi3-draft \
  --mamba-full-memory-ratio 0.20 --node-rank 0 \
  --dist-init-addr 10.0.0.1:5000 --profile
```

For H200, set `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`, and
`SGLANG_HOST_IP` to the actual cross-node interface/address on both machines.
Profile and capture are mutually exclusive modes in the generator.

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
Replicated non-collective bundles are written only by global rank zero even in
such a mixed plan. A `rank_local` bundle is intentionally written by every
rank when its inputs or state differ across TP ranks.

Plans may also select model-specific fine-grained `points` alongside complete
semantic `operations`, with independent `max_rows_per_point` limits. This is
intended for bounded discovery corpora such as K3 decode-hotloop work; a point
does not replace the complete operation bundle needed for standalone replay.

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
For expensive captures, pass `--coverage <json>` to additionally require exact
ranks, modes, layers, distinct forward IDs, and/or teacher-token totals. The
two narrow K3 supplement plans have matching coverage files under
`examples/replay_capture/kimi_k3_missing_*`; they contain no MoE, AttnRes, or
generic activation points and therefore do not recapture the existing corpus.

`attention.kda_target_verify@1` includes dense verify inputs, convolution and
recurrent state before the call, every candidate state needed for rollback,
ReplaySSM ring state when enabled, output-norm inputs when fused, layer-local
convolution/decay parameters, and the recurrence output. This is deliberately
rank-local: TP ranks hold different heads and state.

`speculative.draft_round@2` preserves a complete proposal/verify/accept/commit
round. It accepts checkpoints with or without an explicit confidence head and
records the derived confidence when available, corrected draft logits, target
logits, actual sampling-accept uniforms, proposal tokens, acceptance lengths,
bonus/committed tokens, hidden-state handoff, temperatures, and request/position
identity. Full vocabulary tensors are bounded by an event quota rather than
silently row-sampled.

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

## DSpark teacher-data capture

`speculative.dspark_training_window@1` stores complete teacher-forced windows
at the exact K3 auxiliary-stream boundary consumed by DSpark. Unlike ordinary
activation points, sequence bundles are never row-sampled: all token IDs,
positions, shifted labels, masks, packed-sequence offsets, request pool IDs,
and concatenated target hidden states remain aligned.

Use `examples/replay_capture/kimi_k3_dspark_training.json`; its row limit is a
window/event limit because every selected sequence bundle is preserved whole.

Arm it through the normal replay plan and specify the checkpoint ABI's ordered
target layers:

```bash
export SGLANG_DSPARK_CAPTURE_TARGET_LAYER_IDS=7,23,51,67,83
export SGLANG_DSPARK_CAPTURE_DRAFT_REVISION=<draft-checkpoint-sha>
export SGLANG_DSPARK_CAPTURE_TOKENIZER_REVISION=<tokenizer-sha>
export SGLANG_DSPARK_CAPTURE_CHAT_TEMPLATE_SHA256=<render-template-sha256>
```

The exporter captures extend/prefill forwards. The last token of each packed
request is assigned target ID `-100` and masked because its successor is not
present in that forward; labels never cross request boundaries. Request IDs are
stored as metadata when the scheduler provides them. Model, tokenizer/corpus,
and SGLang revisions should be pinned through the existing capture run
environment variables before producing a training corpus.

# Architecture-level speculative decode profiling

This profiler answers: **how many milliseconds of latency does an average
request spend generating its next committed token, and where does that time go?**
It is separate from replay capture. Keep replay capture disabled while timing.

## Metric

For speculative round `r`, let `B_r` be the number of live requests,
`T_r,c` the GPU elapsed time for component `c`, and `commit_len_r,i` the tokens
committed for request `i`:

```text
avg_ms_per_next_token(c) =
  sum_r(B_r * T_r,c) / sum_r(sum_i(commit_len_r,i))
```

This is request-latency normalized. The report also emits
`gpu_service_ms_per_token = sum_r(T_r) / committed_tokens`, which is a
throughput/service-cost metric and must not be labelled next-token latency.

## DSpark collection

Enable the phase timer and per-request commit lengths:

```bash
export SGLANG_DSPARK_DEBUG_DUMP=core,reqs,phase_gpu_times
```

Run only representative steady-state decode traffic. Exclude warmup, prefill,
idle steps and partial first/last speculative rounds. Obtain
`dspark_info_record` through the existing internal-state endpoint, then run:

```bash
python -m sglang.srt.observability.decode_latency_report dspark-info.json \
  --json-out decode-latency.json \
  --markdown-out decode-latency.md
```

The additive top-level phases are:

1. prepare verify window;
2. draft generation;
3. confidence and verify-budget planning;
4. verify layout scheduling;
5. target-model verification;
6. acceptance and output finalization;
7. recurrent/KV/draft-hidden state commit;
8. runtime gaps and unattributed time.

`runtime_unattributed` bridges measured phase time to the complete speculative
round. `phase_timing_coverage` must be inspected before trusting percentages.

## Architecture attribution contract

`architecture_scope()` in `hotloop_profile.py` carries structured dimensions
through nested semantic ranges without inspecting tensors or synchronizing the
GPU. Speculative workers set:

- `phase`: `prepare`, `draft`, `plan`, `target_verify`, `accept`, or
  `state_commit`;
- `model_role`: `draft`, `target`, or `runtime`.

Model adapters should add:

- `block_type`, such as `kda`, `mla`, `dense`, or `moe`;
- `layer`;
- a stable operation name such as `attention.recurrence`, `experts.w13`, or
  `output_projection.tp_allreduce`.

Common operation names should be reused across model families. A model adapter
only names genuinely model-specific blocks or state transitions.

Kimi-K3 target verification should roll up as:

```text
target_verify
  kda: projections, recurrence, output_norm, output_projection
  mla: qkv/latent projection, attention_core, gate, output_projection
  moe: router, dispatch, experts.w13, experts.w2, combine, shared_experts
  attnres: aggregate, bank_update
  output: final_norm, lm_head
```

Communication belongs under the operation that caused it. For example, report
`kda.output_projection.tp_allreduce`, not a free-standing `collective.tp_ep`.
An alternate communication-only view can aggregate those same leaves.

## CUDA graphs and parallel streams

Top-level phase CUDA events measure latency. Fine-grained kernel attribution is
exclusive GPU work and is not automatically additive when streams overlap.
Parallel regions report parent elapsed latency plus child GPU work.

NVTX/Python nesting is not reliably replayed by a full CUDA graph. A complete
future implementation should record graph node ID to architecture path during
graph capture and join CUPTI/Nsight graph-node activity to that sidecar during
replay. Kernel-name regexes remain a diagnostic fallback only; reports must
show their attribution coverage and fail closed below the required threshold.


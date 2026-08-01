# Kimi-K3 tensor capture

The K3 capture path is inert unless `SGLANG_K3_CAPTURE_DIR` is set. Scalable
capture samples deterministic position strata on GPU, copies selected rows on
a dedicated CUDA stream into pinned host memory, packs entries into 128-512 MB
safetensors shards on local NVMe, then uploads complete atomic shards to the
persistent capture directory in a second background thread. The manifest
retains tensor shapes, dtypes, row counts, phase/cache strata, TP rank and
SHA-256 digests. The legacy synchronous writer remains available for smoke
tests by leaving `SGLANG_K3_CAPTURE_ASYNC` unset.

## Canonical collection settings

Use PP=1, disable CUDA graphs, and do not enable speculative decoding. Keep
DP-attention and expert parallelism off for the first canonical corpus so token
rows are not silently sharded or remapped.

```bash
export SGLANG_K3_CAPTURE_DIR=/data/k3-capture
export SGLANG_K3_CAPTURE_ARM_FILE=/data/k3-capture.armed
export SGLANG_K3_CAPTURE_RUN_ID=k3-full-$(date -u +%Y%m%dT%H%M%SZ)
export SGLANG_K3_CAPTURE_MODEL_REVISION=9f62e4e9fffbd0a83ddd60e1c209d828994b3569
export SGLANG_K3_CAPTURE_SGLANG_REVISION=52522121501
# Set these to the SHA-256 of the prompt-corpus manifest and the fixed sampler seed.
export SGLANG_K3_CAPTURE_CORPUS_SHA256=<sha256>
export SGLANG_K3_CAPTURE_SAMPLING_SEED=20260801
export SGLANG_K3_CAPTURE_LAYERS=all
export SGLANG_K3_CAPTURE_RANKS=0
export SGLANG_K3_CAPTURE_MAX_ROWS=512
export SGLANG_K3_CAPTURE_MAX_ROWS_ROUTING=8192
export SGLANG_K3_CAPTURE_MAX_ROWS_ROUTING_STATIC=1
export SGLANG_K3_CAPTURE_MAX_ROWS_ATTN_RES=128
export SGLANG_K3_CAPTURE_MAX_ROWS_ATTN_RES_STATE=128
export SGLANG_K3_CAPTURE_MAX_ROWS_ATTN_RES_STATIC=1
export SGLANG_K3_CAPTURE_POINTS=all
export SGLANG_K3_CAPTURE_ASYNC=1
export SGLANG_K3_CAPTURE_LOCAL_DIR=/captures/k3-full
export SGLANG_K3_CAPTURE_SHARD_MB=256
export SGLANG_K3_CAPTURE_MAX_PENDING=8
export SGLANG_K3_CAPTURE_DELETE_LOCAL_AFTER_UPLOAD=1

python -m sglang.launch_server \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 8 \
  --language-only \
  --disable-cuda-graph
```

Wait until the server is healthy, then create the arm file before sending the
first corpus request. This prevents profiling and warm-up forwards from
consuming the row limits with synthetic inputs:

```bash
touch /data/k3-capture.armed
```

Send a stratified prompt corpus to the normal server endpoint. Capture keeps
adding rows until each `(point, layer, rank)` reaches its configured cap.

For TP-local KDA/MLA kernel work, run a second, smaller pass with all ranks:

```bash
export SGLANG_K3_CAPTURE_DIR=/data/k3-capture-tp-local
export SGLANG_K3_CAPTURE_RANKS=all
export SGLANG_K3_CAPTURE_MAX_ROWS=256
export SGLANG_K3_CAPTURE_POINTS=kda,kda_state_decode,kda_state_extend,mla_gate
```

The main pass captures every logical layer's stream boundaries, attention and
MoE boundaries, router logits, correction bias, top-k IDs/weights, routed
latent input, KDA projections/gates, MLA output gate, and final LM-head input
and next-token logits. It also retains the MLA packed Q/KV latent projection
and exact AttnRes kernel inputs, targets, weights, and bounded bank-state
snapshots as independent capture points. The TP-local
pass adds exact recurrent KDA state before
and after decode/extend kernel calls.

For activation discovery, configure independent phase quotas. A practical
TP8 rank-0 pass uses 8,192 prefill and 4,096 decode rows for layer streams,
32,768 routing rows per phase, an expert-assignment quota of 256-512, 4,096
KDA transitions per phase, and 8,192 MLA rows. `expert_output` stores the fused
routed-expert result together with routed input and top-k IDs/weights; sampling
prioritizes tokens assigned to experts whose quota is not yet satisfied.

```bash
export SGLANG_K3_CAPTURE_MAX_ROWS_LAYER_INPUT_PREFILL=8192
export SGLANG_K3_CAPTURE_MAX_ROWS_LAYER_INPUT_DECODE=4096
export SGLANG_K3_CAPTURE_MAX_ROWS_LAYER_OUTPUT_PREFILL=8192
export SGLANG_K3_CAPTURE_MAX_ROWS_LAYER_OUTPUT_DECODE=4096
export SGLANG_K3_CAPTURE_MAX_ROWS_ROUTING_PREFILL=32768
export SGLANG_K3_CAPTURE_MAX_ROWS_ROUTING_DECODE=32768
export SGLANG_K3_CAPTURE_MAX_ROWS_EXPERT_OUTPUT=65536
export SGLANG_K3_CAPTURE_EXPERT_QUOTA=512
export SGLANG_K3_CAPTURE_NUM_EXPERTS=896
export SGLANG_K3_CAPTURE_MAX_ROWS_KDA_PREFILL=4096
export SGLANG_K3_CAPTURE_MAX_ROWS_KDA_DECODE=4096
export SGLANG_K3_CAPTURE_MAX_ROWS_KDA_STATE_EXTEND_PREFILL=4096
export SGLANG_K3_CAPTURE_MAX_ROWS_KDA_STATE_DECODE_DECODE=4096
export SGLANG_K3_CAPTURE_MAX_ROWS_MLA_GATE=8192
export SGLANG_K3_CAPTURE_MAX_ROWS_MLA_LATENT=8192
```

## Output layout

```text
/data/k3-capture/
  rank-00000/
    manifest.jsonl
    shard-000000.safetensors
    shard-000001.safetensors
    ...
```

No prompt text or tokenizer IDs are written. Request metadata is limited to
forward mode, layer/type, rank, tensor geometry, and capture counters.

Verify hashes and layer coverage before moving or deleting the instance:

```bash
python scripts/k3_capture_summary.py /data/k3-capture
python scripts/k3_capture_summary.py /data/k3-capture-tp-local
python scripts/k3_capture_summary.py /data/k3-capture --write-complete
python scripts/k3_capture_summary.py /data/k3-capture-tp-local --write-complete
```

Only terminate the pod after both directories have a validated `COMPLETE.json`
and that file has been copied with the shards. It commits to every manifest and
shard using a corpus-level SHA-256 root.

The default 512-row all-point pass is a plumbing corpus. For the authoritative
routing dataset, collect at least 8k-16k diverse prefill rows per layer and
128-512 decode invocations while retaining complete invocation boundaries.
Expect tens of GiB. Use a persistent volume of at least 4 TiB so the 1.56 TB
checkpoint, container/cache overhead, and raw capture corpus can coexist.

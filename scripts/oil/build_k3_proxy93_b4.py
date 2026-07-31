#!/usr/bin/env python3
"""Build the language-only Kimi-K3 Proxy93-B4 systems checkpoint.

The source checkpoint is MXFP4 safetensors. This builder parses each source
shard header, selects tensors, and copies only their HTTP byte ranges into a
new safetensors file. Tensor payloads are never decoded or requantized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_SOURCE = "moonshotai/Kimi-K3"
DEFAULT_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
EXPERT_MARKER = ".block_sparse_moe.experts."
SOURCE_EXPERT_LAYERS = (1, 24, 48, 72)
EXPERT_BANK_OWNERS = (1, 2, 3, 4)
OMIT_PREFIXES = ("vision_tower.", "mm_projector.")


def session(token: str | None) -> requests.Session:
    retry = Retry(
        total=8,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    result = requests.Session()
    result.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    result.headers["User-Agent"] = "open-inference-league-k3-proxy-builder/1"
    if token:
        result.headers["Authorization"] = f"Bearer {token}"
    return result


def resolve_url(repo_id: str, revision: str, filename: str) -> str:
    return (
        "https://huggingface.co/"
        f"{quote(repo_id, safe='/')}/resolve/{quote(revision, safe='')}/"
        f"{quote(filename, safe='/')}?download=true"
    )


def get_json(client: requests.Session, url: str) -> dict[str, Any]:
    response = client.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def get_range(
    client: requests.Session, url: str, start: int, end: int
) -> requests.Response:
    response = client.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        stream=True,
        timeout=(30, 900),
    )
    response.raise_for_status()
    expected = end - start + 1
    if response.status_code != 206:
        response.close()
        raise RuntimeError(
            f"server ignored byte range {start}-{end} (HTTP {response.status_code})"
        )
    content_range = response.headers.get("Content-Range", "")
    if not content_range.startswith(f"bytes {start}-{end}/"):
        response.close()
        raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
    length = response.headers.get("Content-Length")
    if length is not None and int(length) != expected:
        response.close()
        raise RuntimeError(f"range length {length} != {expected}")
    return response


def read_safetensors_header(
    client: requests.Session, url: str
) -> tuple[int, dict[str, Any]]:
    with get_range(client, url, 0, 7) as response:
        prefix = response.content
    if len(prefix) != 8:
        raise RuntimeError(f"invalid safetensors prefix length: {len(prefix)}")
    header_len = struct.unpack("<Q", prefix)[0]
    with get_range(client, url, 8, 8 + header_len - 1) as response:
        raw_header = response.content
    if len(raw_header) != header_len:
        raise RuntimeError("truncated safetensors header")
    return header_len, json.loads(raw_header.rstrip(b" "))


def layer_number(name: str) -> int | None:
    marker = "language_model.model.layers."
    if not name.startswith(marker):
        return None
    rest = name[len(marker) :]
    head, separator, _ = rest.partition(".")
    return int(head) if separator and head.isdigit() else None


def destination_name(name: str) -> str | None:
    if name.startswith(OMIT_PREFIXES):
        return None
    if EXPERT_MARKER not in name:
        return name
    source_layer = layer_number(name)
    if source_layer not in SOURCE_EXPERT_LAYERS:
        return None
    owner = EXPERT_BANK_OWNERS[SOURCE_EXPERT_LAYERS.index(source_layer)]
    source_prefix = f"language_model.model.layers.{source_layer}."
    owner_prefix = f"language_model.model.layers.{owner}."
    return owner_prefix + name[len(source_prefix) :]


def padded_header(entries: list[tuple[str, dict[str, Any]]], metadata: Any) -> bytes:
    header: dict[str, Any] = {}
    offset = 0
    for name, source_info in entries:
        size = source_info["data_offsets"][1] - source_info["data_offsets"][0]
        header[name] = {
            "dtype": source_info["dtype"],
            "shape": source_info["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    return raw + b" " * ((8 - len(raw) % 8) % 8)


def contiguous_groups(
    entries: list[tuple[str, dict[str, Any]]],
) -> Iterable[tuple[int, int]]:
    if not entries:
        return
    start, end = entries[0][1]["data_offsets"]
    for _, info in entries[1:]:
        next_start, next_end = info["data_offsets"]
        if next_start == end:
            end = next_end
        else:
            yield start, end
            start, end = next_start, next_end
    yield start, end


def bounded_ranges(start: int, end: int, chunk_size: int = 256 * 1024 * 1024):
    while start < end:
        next_end = min(end, start + chunk_size)
        yield start, next_end
        start = next_end


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_shard(
    client: requests.Session,
    url: str,
    selected: dict[str, str],
    output: Path,
) -> dict[str, Any]:
    source_header_len, source_header = read_safetensors_header(client, url)
    entries: list[tuple[str, dict[str, Any]]] = []
    for source_name, destination in selected.items():
        if source_name not in source_header:
            raise KeyError(f"{source_name} missing from shard header")
        entries.append((destination, source_header[source_name]))
    entries.sort(key=lambda item: item[1]["data_offsets"][0])
    if len({name for name, _ in entries}) != len(entries):
        raise RuntimeError(f"duplicate destination tensor in {output.name}")

    header = padded_header(entries, source_header.get("__metadata__"))
    digest = hashlib.sha256()
    expected_data_bytes = sum(
        info["data_offsets"][1] - info["data_offsets"][0] for _, info in entries
    )
    written_data_bytes = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("wb") as handle:
        prefix = struct.pack("<Q", len(header))
        handle.write(prefix)
        handle.write(header)
        digest.update(prefix)
        digest.update(header)
        data_base = 8 + source_header_len
        source_entries = sorted(
            ((source, source_header[source]) for source in selected),
            key=lambda item: item[1]["data_offsets"][0],
        )
        for relative_start, relative_end in contiguous_groups(source_entries):
            for bounded_start, bounded_end in bounded_ranges(
                relative_start, relative_end
            ):
                absolute_start = data_base + bounded_start
                absolute_end = data_base + bounded_end - 1
                with get_range(client, url, absolute_start, absolute_end) as response:
                    range_bytes = 0
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            digest.update(chunk)
                            range_bytes += len(chunk)
                expected_range_bytes = bounded_end - bounded_start
                if range_bytes != expected_range_bytes:
                    raise RuntimeError(
                        f"truncated data range in {output.name}: "
                        f"{range_bytes} != {expected_range_bytes}"
                    )
                written_data_bytes += range_bytes
        handle.flush()
        os.fsync(handle.fileno())
    if written_data_bytes != expected_data_bytes:
        raise RuntimeError(
            f"wrong shard payload size: {written_data_bytes} != {expected_data_bytes}"
        )
    temporary.replace(output)
    return {
        "sha256": digest.hexdigest(),
        "size": output.stat().st_size,
        "tensor_count": len(entries),
        "tensor_bytes": expected_data_bytes,
    }


def download_file(
    client: requests.Session, url: str, destination: Path
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    size = 0
    with client.get(url, stream=True, timeout=(30, 900)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
    temporary.replace(destination)
    return {"sha256": digest.hexdigest(), "size": size}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"files": {}}
    return json.loads(path.read_text())


def proxy_readme(source: str, revision: str, sglang_commit: str) -> str:
    return f"""---
license: other
library_name: transformers
base_model: {source}
---

# Kimi-K3 Proxy93-B4

This is a **systems-performance proxy**, not a quality model. It preserves all
93 logical language layers, full 7168 hidden width, the original 69 KDA / 24
MLA schedule, 896 routed-expert slots, top-16 routing, shared experts, routers,
and attention-residual tensors. Routed expert payloads are retained from four
source layers (1, 24, 48, 72) and physically shared across logical layers by a
patched SGLang loader.

The checkpoint is intended for single-node inference engineering where the
full Kimi-K3 checkpoint is too large. Its generated text is not meaningful and
it must not be used for accuracy, acceptance-rate, or final performance claims.
Validate winning changes against full Kimi-K3.

## Reproducibility

- Source: `{source}`
- Source revision: `{revision}`
- Required SGLang revision: `{sglang_commit}` on the Open Inference League fork
- Physical expert-bank owners: layers 1, 2, 3, 4
- Source expert banks: layers 1, 24, 48, 72
- Vision tower and multimodal projector: omitted

Run SGLang with `--language-only`. See `proxy_manifest.json` for the exact
logical-to-physical mapping and SHA-256 digest of every generated file.

The upstream Kimi-K3 license and attribution are preserved in `LICENSE`.
"""


def build(args: argparse.Namespace) -> None:
    token = os.environ.get("HF_TOKEN")
    client = session(token)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / ".proxy-build-state.json"
    state = read_state(state_path)

    api_url = (
        f"https://huggingface.co/api/models/{args.source}/revision/{args.revision}"
    )
    model_info = get_json(client, api_url)
    resolved_sha = model_info["sha"]
    if resolved_sha != args.revision:
        raise RuntimeError(
            f"revision resolved to {resolved_sha}, expected {args.revision}"
        )

    index_url = resolve_url(args.source, args.revision, "model.safetensors.index.json")
    index_response = client.get(index_url, timeout=180)
    index_response.raise_for_status()
    index_bytes = index_response.content
    source_index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    source_index = json.loads(index_bytes)

    shard_selection: dict[str, dict[str, str]] = defaultdict(dict)
    output_weight_map: dict[str, str] = {}
    for source_name, shard in source_index["weight_map"].items():
        destination = destination_name(source_name)
        if destination is None:
            continue
        if destination in output_weight_map:
            raise RuntimeError(f"duplicate output tensor: {destination}")
        shard_selection[shard][source_name] = destination
        output_weight_map[destination] = shard

    print(
        f"selected {len(output_weight_map):,} of "
        f"{len(source_index['weight_map']):,} tensors across "
        f"{len(shard_selection)} shards",
        flush=True,
    )
    files = state.setdefault("files", {})
    for position, shard in enumerate(sorted(shard_selection), 1):
        destination = output / shard
        if shard in files and destination.exists():
            actual = file_sha256(destination)
            if actual == files[shard]["sha256"]:
                print(
                    f"[{position:02d}/{len(shard_selection)}] resume {shard}",
                    flush=True,
                )
                continue
        print(f"[{position:02d}/{len(shard_selection)}] build {shard}", flush=True)
        files[shard] = build_shard(
            client,
            resolve_url(args.source, args.revision, shard),
            shard_selection[shard],
            destination,
        )
        atomic_json(state_path, state)
        print(
            f"  {files[shard]['tensor_count']:,} tensors, "
            f"{files[shard]['size'] / 1e9:.3f} GB",
            flush=True,
        )

    # Copy the small config/tokenizer/license files at the same pinned revision.
    skipped = {"README.md", "model.safetensors.index.json"}
    for sibling in model_info["siblings"]:
        filename = sibling["rfilename"]
        if filename.endswith(".safetensors") or filename in skipped:
            continue
        target = output / filename
        print(f"metadata {filename}", flush=True)
        files[filename] = download_file(
            client, resolve_url(args.source, args.revision, filename), target
        )

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config.setdefault("text_config", {})
    layer_to_bank = [0] + [1 + ((layer - 1) % 4) for layer in range(1, 93)]
    text_config.update(
        {
            "oil_proxy_schema": 1,
            "oil_proxy_name": "Kimi-K3-Proxy93-B4",
            "oil_proxy_layer_to_bank": layer_to_bank,
            "oil_proxy_expert_bank_owners": list(EXPERT_BANK_OWNERS),
            "oil_proxy_source_expert_layers": list(SOURCE_EXPERT_LAYERS),
            "oil_proxy_language_only": True,
            "oil_proxy_sglang_commit": args.sglang_commit,
        }
    )
    atomic_json(config_path, config)

    output_index = {
        "metadata": {
            "total_size": sum(
                item["tensor_bytes"]
                for item in files.values()
                if "tensor_bytes" in item
            )
        },
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    atomic_json(output / "model.safetensors.index.json", output_index)
    (output / "README.md").write_text(
        proxy_readme(args.source, args.revision, args.sglang_commit)
    )

    # Recompute every generated/copy digest after config/index/README mutation.
    manifest_files: dict[str, dict[str, Any]] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial") or path == state_path:
            continue
        relative = path.relative_to(output).as_posix()
        manifest_files[relative] = {
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }

    manifest = {
        "schema": 1,
        "name": "Kimi-K3-Proxy93-B4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_id": args.source,
            "revision": args.revision,
            "index_sha256": source_index_sha256,
        },
        "selection": {
            "logical_layers": 93,
            "expert_bank_count": 4,
            "expert_bank_owners": list(EXPERT_BANK_OWNERS),
            "source_expert_layers": list(SOURCE_EXPERT_LAYERS),
            "logical_layer_to_bank": layer_to_bank,
            "omitted_prefixes": list(OMIT_PREFIXES),
        },
        "sglang": {
            "repository": "https://github.com/ek-capital/sglang",
            "commit": args.sglang_commit,
        },
        "tensor_count": len(output_weight_map),
        "tensor_bytes": output_index["metadata"]["total_size"],
        "files": manifest_files,
        "warning": "Systems proxy only; generated outputs are not meaningful.",
    }
    atomic_json(output / "proxy_manifest.json", manifest)
    state_path.unlink(missing_ok=True)
    print(
        f"complete: {manifest['tensor_count']:,} tensors, "
        f"{manifest['tensor_bytes'] / 1e9:.3f} GB payload",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--sglang-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        build(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)

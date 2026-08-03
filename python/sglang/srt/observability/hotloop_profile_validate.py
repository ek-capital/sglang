# SPDX-License-Identifier: Apache-2.0
"""Fail closed when a Torch Chrome trace lacks required hotloop sections."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def validate_hotloop_trace(
    path: Path, required_sections: list[str], minimum_occurrences: int = 1
) -> dict[str, Any]:
    payload = _read_trace(path)
    names = Counter(
        str(event.get("name", ""))
        for event in payload.get("traceEvents", [])
        if str(event.get("name", "")).startswith("sglang.hotloop/")
    )
    counts = {
        section: sum(
            count
            for name, count in names.items()
            if name == f"sglang.hotloop/{section}"
            or name.startswith(f"sglang.hotloop/{section}/")
        )
        for section in required_sections
    }
    missing = {
        section: count
        for section, count in counts.items()
        if count < minimum_occurrences
    }
    return {
        "valid": not missing,
        "path": str(path),
        "required_counts": counts,
        "missing": missing,
        "semantic_event_names": dict(sorted(names.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--requirements-file", type=Path)
    parser.add_argument("--minimum-occurrences", type=int, default=1)
    args = parser.parse_args()
    required = list(args.require)
    if args.requirements_file is not None:
        required.extend(
            line.strip()
            for line in args.requirements_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    report = validate_hotloop_trace(args.trace, required, args.minimum_occurrences)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()

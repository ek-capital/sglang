# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness and latency harness for extracted replay sections."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import load_file


@dataclass(frozen=True)
class Tolerance:
    atol: float = 0.0
    rtol: float = 0.0
    max_relative_l2: float | None = None
    exact: bool = False


def _load_callable(reference: str) -> Callable[..., Any]:
    try:
        module_name, function_name = reference.split(":", 1)
    except ValueError as exc:
        raise ValueError("runner must use module:function syntax") from exc
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"runner is not callable: {reference}")
    return function


def _load_group(
    case_dir: Path, name: str, device: torch.device
) -> dict[str, torch.Tensor]:
    path = case_dir / f"{name}.safetensors"
    if not path.exists():
        return {}
    return {key: value.to(device) for key, value in load_file(path).items()}


def _clone(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in values.items()}


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return float((diff / denominator).item())


def _compare(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    tolerance: Tolerance,
) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "valid": False,
            "error": (
                f"{name}: shape/dtype {tuple(actual.shape)}/{actual.dtype} != "
                f"{tuple(expected.shape)}/{expected.dtype}"
            ),
        }
    exact = tolerance.exact or not (actual.is_floating_point() or actual.is_complex())
    close = (
        torch.equal(actual, expected)
        if exact
        else torch.allclose(actual, expected, atol=tolerance.atol, rtol=tolerance.rtol)
    )
    relative_l2 = None if exact else _relative_l2(actual, expected)
    if tolerance.max_relative_l2 is not None:
        close = close and relative_l2 <= tolerance.max_relative_l2
    max_abs = (
        0.0
        if actual.numel() == 0
        else float((actual.float() - expected.float()).abs().max().item())
    )
    return {
        "valid": bool(close),
        "exact": exact,
        "max_abs": max_abs,
        "relative_l2": relative_l2,
    }


def _tolerance(contract: dict[str, Any], name: str) -> Tolerance:
    raw = {
        **contract.get("default_tolerance", {}),
        **contract.get("tolerances", {}).get(name, {}),
    }
    return Tolerance(**raw)


def validate_case(
    contract: dict[str, Any], case_dir: Path, *, device: torch.device
) -> dict[str, Any]:
    runner = _load_callable(contract["runner"])
    inputs = _load_group(case_dir, "inputs", device)
    state_before = _load_group(case_dir, "state-before", device)
    static = _load_group(case_dir.parent.parent, "static", device)
    expected_outputs = _load_group(case_dir, "expected-output", device)
    expected_state = _load_group(case_dir, "state-after", device)
    control_path = case_dir / "control.json"
    control = {} if not control_path.exists() else json.loads(control_path.read_text())
    with torch.inference_mode():
        result = runner(
            inputs=_clone(inputs),
            state=_clone(state_before),
            static=static,
            control=control,
        )
    if not isinstance(result, dict):
        raise TypeError("replay runner must return a dict")
    actual_outputs = result.get("outputs", result)
    actual_state = result.get("state", {})
    errors: list[str] = []
    comparisons: dict[str, Any] = {}
    for group_name, actual, expected in (
        ("outputs", actual_outputs, expected_outputs),
        ("state", actual_state, expected_state),
    ):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing:
            errors.append(f"{group_name} missing: {missing}")
        if extra and group_name == "outputs":
            errors.append(f"{group_name} undeclared: {extra}")
        for name in sorted(set(expected) & set(actual)):
            comparison = _compare(
                name, actual[name], expected[name], _tolerance(contract, name)
            )
            comparisons[f"{group_name}.{name}"] = comparison
            if not comparison["valid"]:
                errors.append(f"{group_name}.{name} failed tolerance")
    return {
        "case": case_dir.name,
        "valid": not errors,
        "comparisons": comparisons,
        "errors": errors,
    }


def benchmark_case(
    contract: dict[str, Any],
    case_dir: Path,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if device.type != "cuda":
        raise ValueError("latency benchmarking requires a CUDA device")
    runner = _load_callable(contract["runner"])
    inputs = _load_group(case_dir, "inputs", device)
    state_before = _load_group(case_dir, "state-before", device)
    static = _load_group(case_dir.parent.parent, "static", device)
    control_path = case_dir / "control.json"
    control = {} if not control_path.exists() else json.loads(control_path.read_text())

    def invoke() -> Any:
        # Stateful operations must begin every measured iteration from the exact
        # captured checkpoint; state reset is outside the timed region.
        state = _clone(state_before)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = runner(inputs=inputs, state=state, static=static, control=control)
        end.record()
        return result, start, end

    with torch.inference_mode():
        for _ in range(warmup):
            invoke()
        torch.cuda.synchronize(device)
        samples: list[float] = []
        for _ in range(iterations):
            _, start, end = invoke()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    ordered = sorted(samples)
    return {
        "case": case_dir.name,
        "iterations": iterations,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "min_ms": min(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    contract = json.loads((args.package / "contract.json").read_text())
    if int(contract.get("format_version", 0)) != 1:
        raise ValueError("unsupported replay contract format_version")
    device = torch.device(args.device)
    reports = []
    valid = True
    for case_dir in sorted((args.package / "cases").iterdir()):
        if not case_dir.is_dir():
            continue
        correctness = validate_case(contract, case_dir, device=device)
        valid &= correctness["valid"]
        report: dict[str, Any] = {"correctness": correctness}
        if args.benchmark and correctness["valid"]:
            report["benchmark"] = benchmark_case(
                contract,
                case_dir,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        reports.append(report)
    print(json.dumps({"valid": valid, "cases": reports}, indent=2, sort_keys=True))
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()

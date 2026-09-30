#!/usr/bin/env python3
"""Record a reproducible, hardware-qualified PATACON performance baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repository", type=Path, default=Path("."))
    arguments = parser.parse_args()
    if arguments.runs < 1:
        parser.error("--runs must be positive")
    if arguments.seed < 0:
        parser.error("--seed must be nonnegative")
    return arguments


def capture(command: list[str], repository: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=repository,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_rank(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(proportion * len(ordered)))
    return ordered[rank - 1]


def summary(values: Iterable[float], scale: float = 1.0) -> dict[str, float]:
    scaled = [float(value) / scale for value in values]
    if not scaled:
        raise ValueError("cannot summarize an empty sample")
    return {
        "min": min(scaled),
        "median": statistics.median(scaled),
        "mean": statistics.fmean(scaled),
        "p95": nearest_rank(scaled, 0.95),
        "max": max(scaled),
    }


def cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return None


def run_case(
    executable: Path,
    repository: Path,
    case: dict[str, Any],
    runs: int,
    seed: int,
    temporary_directory: Path,
) -> dict[str, Any]:
    result_path = temporary_directory / f"{case['robot']}.json"
    command_arguments = [
        case["robot"],
        case["problem"],
        str(case["problem_index"]),
        "--runs",
        str(runs),
        "--seed",
        str(seed),
        "--save-json",
        str(result_path),
        "--no-print-path",
    ]
    completed = subprocess.run(
        [str(executable), *command_arguments],
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max(120, runs * 30),
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{case['robot']} benchmark exited with {completed.returncode}"
        )
    with result_path.open("r", encoding="utf-8") as stream:
        bundle = json.load(stream)
    results = bundle["results"] if runs > 1 else [bundle]
    if len(results) != runs:
        raise RuntimeError(f"{case['robot']} produced {len(results)}/{runs} results")

    return {
        "robot": case["robot"],
        "problem": case["problem"],
        "problem_index": case["problem_index"],
        "command_arguments": command_arguments[:-3] + [
            "--save-json", "<temporary-result.json>", "--no-print-path"
        ],
        "runs": runs,
        "solved_runs": sum(bool(result["solved"]) for result in results),
        "kernel_ms": summary(
            (result["kernel_ns"] for result in results),
            scale=1.0e6,
        ),
        "wall_ms": summary(
            (result["wall_ns"] for result in results),
            scale=1.0e6,
        ),
        "cost": summary(result["cost"] for result in results),
        "path_waypoint_count": summary(
            result["path_waypoint_count"] for result in results
        ),
        "iterations": summary(result["iters"] for result in results),
    }


def main() -> int:
    arguments = parse_args()
    repository = arguments.repository.resolve()
    executable = arguments.executable.resolve()
    cases_path = (repository / arguments.cases).resolve()
    output_path = (repository / arguments.output).resolve()
    with cases_path.open("r", encoding="utf-8") as stream:
        cases = json.load(stream)["cases"]

    commit = capture(["git", "rev-parse", "HEAD"], repository)
    status = capture(["git", "status", "--porcelain"], repository)
    gpu = capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        repository,
    )
    cuda = capture(["nvcc", "--version"], repository)

    with tempfile.TemporaryDirectory(prefix="patacon-baseline-") as directory:
        temporary_directory = Path(directory)
        case_results = [
            run_case(
                executable,
                repository,
                case,
                arguments.runs,
                arguments.seed,
                temporary_directory,
            )
            for case in cases
        ]

    payload = {
        "format": "PATACON_performance_baseline_v1",
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "commit": commit,
            "working_tree_dirty": bool(status),
            "executable_sha256": sha256(executable),
        },
        "system": {
            "platform": platform.platform(),
            "cpu": cpu_model(),
            "gpu": gpu,
            "cuda": cuda.splitlines()[-1] if cuda else None,
        },
        "method": {
            "cases_file": str(arguments.cases),
            "runs_per_case": arguments.runs,
            "first_seed": arguments.seed,
            "seeds": f"{arguments.seed}..{arguments.seed + arguments.runs - 1}",
            "primary_timing_field": "kernel_ns",
            "warmup": "single_mbm CUDA warmup before the first measured run",
            "execution": "cases run serially on one GPU",
        },
        "cases": case_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(f"baseline saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot cumulative successful pRRTC runs against planning time."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def positive_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def planning_seconds(record: dict[str, Any]) -> float | None:
    seconds = positive_number(record.get("planning_sec"))
    if seconds is not None:
        return seconds
    kernel_ns = positive_number(record.get("kernel_ns"))
    if kernel_ns is not None:
        return kernel_ns / 1.0e9
    return None


def load_runs(path: Path) -> tuple[list[tuple[float, bool]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object")

    raw_results = payload.get("results")
    records = raw_results if isinstance(raw_results, list) else [payload]
    if not records:
        raise ValueError("input JSON contains no runs")

    runs: list[tuple[float, bool]] = []
    for ordinal, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"run {ordinal} is not a JSON object")
        elapsed = planning_seconds(record)
        if elapsed is None:
            raise ValueError(
                f"run {ordinal} has no positive planning_sec or kernel_ns"
            )
        runs.append((elapsed, bool(record.get("solved", False))))

    metadata = {
        key: payload[key]
        for key in ("planner", "robot", "problem_name", "problem_idx")
        if key in payload
    }
    return runs, metadata


def inferred_title(metadata: dict[str, Any]) -> str:
    planner = str(metadata.get("planner", "pRRTC"))
    robot = metadata.get("robot")
    problem = metadata.get("problem_name")
    problem_idx = metadata.get("problem_idx")
    title = planner
    if robot is not None and problem is not None:
        title += f" - {robot} / {problem}"
    if problem_idx is not None:
        title += f" #{problem_idx}"
    return title


def plot_ecdf(
    runs: list[tuple[float, bool]],
    title: str,
) -> plt.Figure:
    all_times = [elapsed for elapsed, _ in runs]
    solved_times = sorted(elapsed for elapsed, solved in runs if solved)
    minimum = min(all_times)
    maximum = max(all_times)
    if math.isclose(minimum, maximum, rel_tol=1.0e-12, abs_tol=0.0):
        x_min = minimum / 2.0
        x_max = maximum * 2.0
    else:
        x_min = minimum / 1.25
        x_max = maximum * 1.25

    if solved_times:
        x_values = [x_min, *solved_times, x_max]
        y_values = [0, *range(1, len(solved_times) + 1), len(solved_times)]
    else:
        x_values = [x_min, x_max]
        y_values = [0, 0]

    figure, axis = plt.subplots(figsize=(10, 6.5))
    axis.step(
        x_values,
        y_values,
        where="post",
        linewidth=3.0,
        color="#2a9d8f",
        label="pRRTC",
    )
    axis.set_xscale("log")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(0, max(1, len(runs)) * 1.05)
    axis.set_xlabel("Planning Time (seconds)", fontsize=15)
    axis.set_ylabel("Solved Runs", fontsize=15)
    axis.set_title(title, fontsize=17)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", alpha=0.15)
    axis.legend(loc="lower right")
    axis.text(
        0.02,
        0.96,
        f"Solved: {len(solved_times)}/{len(runs)}",
        transform=axis.transAxes,
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    figure.tight_layout()
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the cumulative number of successful pRRTC runs against "
            "their kernel planning time."
        )
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title")
    args = parser.parse_args()

    if not args.input_path.is_file():
        parser.error(f"input file does not exist: {args.input_path}")

    try:
        runs, metadata = load_runs(args.input_path)
        figure = plot_ecdf(runs, args.title or inferred_title(metadata))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    output = args.output or args.input_path.with_name(
        args.input_path.stem + "_ecdf.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)

    solved_count = sum(solved for _, solved in runs)
    print(f"runs: {len(runs)}")
    print(f"solved: {solved_count}/{len(runs)}")
    print(f"saved_plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

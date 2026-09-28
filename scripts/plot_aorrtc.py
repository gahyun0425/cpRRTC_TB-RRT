#!/usr/bin/env python3
"""Plot AORRTC success and anytime-cost convergence.

The preferred input is the JSON written by single_mbm --save-json because it
contains every solution update. The human-readable single_mbm log is also
accepted, but it only contains the initial and final solution endpoints.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
RUN_PATTERN = re.compile(r"^run\s*:\s*(\d+)\s*$", re.IGNORECASE)
TITLE_FONT_SIZE = 26
AXIS_LABEL_FONT_SIZE = 22
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 18
NOTE_FONT_SIZE = 15


@dataclass
class RunHistory:
    run_id: str
    events: list[tuple[float, float]]
    planning_sec: float | None
    solved: bool
    complete_history: bool


def positive_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def normalized_events(events: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort events, merge identical times, and enforce best-so-far cost."""
    valid = [
        (float(time_sec), float(cost))
        for time_sec, cost in events
        if positive_float(time_sec) is not None
        and positive_float(cost) is not None
    ]
    valid.sort(key=lambda item: item[0])

    output: list[tuple[float, float]] = []
    best_cost = math.inf
    for time_sec, cost in valid:
        best_cost = min(best_cost, cost)
        if output and math.isclose(time_sec, output[-1][0], rel_tol=0.0, abs_tol=1e-12):
            output[-1] = (time_sec, min(output[-1][1], best_cost))
        else:
            output.append((time_sec, best_cost))
    return output


def endpoint_events(record: dict[str, Any]) -> list[tuple[float, float]]:
    initial_time = positive_float(
        record.get("initial_solution_sec", record.get("aorrtc_initial_solution_sec"))
    )
    initial_cost = positive_float(
        record.get("initial_cost", record.get("aorrtc_initial_cost"))
    )
    best_time = positive_float(
        record.get("best_solution_sec", record.get("aorrtc_best_solution_sec"))
    )
    best_cost = positive_float(record.get("cost"))

    events: list[tuple[float, float]] = []
    if initial_time is not None and initial_cost is not None:
        events.append((initial_time, initial_cost))
    if best_time is not None and best_cost is not None:
        events.append((best_time, best_cost))
    return normalized_events(events)


def deduplicate_runs(
    runs: list[RunHistory], policy: str
) -> list[RunHistory]:
    positions: dict[str, int] = {}
    output: list[RunHistory] = []
    duplicate_ids: list[str] = []

    for run in runs:
        if run.run_id not in positions or policy == "all":
            positions.setdefault(run.run_id, len(output))
            output.append(run)
            continue

        duplicate_ids.append(run.run_id)
        if policy == "error":
            raise ValueError(f"duplicate run id: {run.run_id}")
        if policy == "last":
            output[positions[run.run_id]] = run
        # The first occurrence is already in output for policy == "first".

    if duplicate_ids:
        def run_id_order(run_id: str) -> tuple[int, int | str]:
            return (0, int(run_id)) if run_id.isdigit() else (1, run_id)

        unique_ids = ", ".join(sorted(set(duplicate_ids), key=run_id_order))
        print(
            f"warning: duplicate run id(s) {unique_ids}; "
            f"duplicate policy '{policy}' was applied",
            file=sys.stderr,
        )
    return output


def parse_summary_log(path: Path, duplicate_policy: str) -> list[RunHistory]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finish_current() -> None:
        nonlocal current
        if current is not None and len(current) > 1:
            records.append(current)
        current = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        run_match = RUN_PATTERN.match(line)
        if run_match:
            finish_current()
            current = {"run_idx": run_match.group(1)}
            continue
        if line.lower() == "failed!":
            if current is None:
                current = {"run_idx": str(len(records) + 1)}
            current["failed"] = True
            continue
        if ":" not in line:
            continue

        key, raw_value = (part.strip() for part in line.split(":", 1))
        if key not in {
            "cost",
            "planning_s",
            "aorrtc_initial_cost",
            "aorrtc_solution_updates",
            "aorrtc_initial_solution_sec",
            "aorrtc_best_solution_sec",
        }:
            continue
        if current is None:
            current = {"run_idx": str(len(records) + 1)}
        if NUMBER_PATTERN.match(raw_value):
            current[key] = float(raw_value)

    finish_current()
    if not records:
        raise ValueError("no AORRTC run records were found in the text log")

    runs: list[RunHistory] = []
    for record in records:
        events = endpoint_events(record)
        failed = bool(record.get("failed", False))
        updates = positive_float(record.get("aorrtc_solution_updates"))
        solved = not failed and bool(events) and (updates is None or updates >= 1)
        runs.append(
            RunHistory(
                run_id=str(record["run_idx"]),
                events=events if solved else [],
                planning_sec=positive_float(record.get("planning_s")),
                solved=solved,
                complete_history=False,
            )
        )
    return deduplicate_runs(runs, duplicate_policy)


def planning_time_from_json(record: dict[str, Any]) -> float | None:
    planning_sec = positive_float(record.get("planning_sec"))
    if planning_sec is not None:
        return planning_sec
    planning_ns = positive_float(record.get("planning_ns"))
    if planning_ns is not None:
        return planning_ns / 1.0e9
    settings = record.get("settings", {})
    if isinstance(settings, dict):
        return positive_float(settings.get("time_limit_sec"))
    return None


def parse_json(path: Path, duplicate_policy: str) -> tuple[list[RunHistory], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict):
        raw_results = payload.get("results")
        records = raw_results if isinstance(raw_results, list) else [payload]
        metadata = {
            key: payload[key]
            for key in ("planner", "robot", "problem_name", "problem_idx")
            if key in payload
        }
        settings = payload.get("settings")
        if isinstance(settings, dict):
            time_limit_sec = positive_float(settings.get("time_limit_sec"))
            if time_limit_sec is not None:
                metadata["time_limit_sec"] = time_limit_sec
    else:
        raise ValueError("JSON input must be an object or an array")

    runs: list[RunHistory] = []
    for ordinal, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"JSON result {ordinal} is not an object")
        raw_history = record.get("solution_history", [])
        events: list[tuple[float, float]] = []
        if isinstance(raw_history, list):
            for update in raw_history:
                if not isinstance(update, dict):
                    continue
                time_sec = positive_float(update.get("found_sec"))
                if time_sec is None:
                    found_ns = positive_float(update.get("found_ns"))
                    time_sec = None if found_ns is None else found_ns / 1.0e9
                cost = positive_float(update.get("cost"))
                if time_sec is not None and cost is not None:
                    events.append((time_sec, cost))

        events = normalized_events(events)
        complete_history = bool(events) and not bool(
            record.get("solution_history_overflow", False)
        )
        expected_updates = record.get("solution_updates")
        if isinstance(expected_updates, int) and expected_updates > len(events):
            complete_history = False

        solved = bool(record.get("solved", bool(events)))
        if solved and not events:
            events = endpoint_events(record)
            complete_history = False
        runs.append(
            RunHistory(
                run_id=str(record.get("run_idx", ordinal)),
                events=events if solved else [],
                planning_sec=planning_time_from_json(record),
                solved=solved and bool(events),
                complete_history=complete_history,
            )
        )

    if not runs:
        raise ValueError("JSON input contains no runs")
    return deduplicate_runs(runs, duplicate_policy), metadata


def parse_csv(path: Path) -> list[RunHistory]:
    events: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames or not {"time_sec", "cost"}.issubset(reader.fieldnames):
            raise ValueError("CSV input requires time_sec and cost columns")
        for row in reader:
            time_sec = positive_float(row.get("time_sec"))
            cost = positive_float(row.get("cost"))
            if time_sec is not None and cost is not None:
                events.append((time_sec, cost))
    events = normalized_events(events)
    if not events:
        raise ValueError("CSV input contains no valid AORRTC updates")
    return [
        RunHistory(
            run_id="1",
            events=events,
            planning_sec=events[-1][0],
            solved=True,
            complete_history=True,
        )
    ]


def load_runs(
    path: Path, input_format: str, duplicate_policy: str
) -> tuple[list[RunHistory], dict[str, Any]]:
    if input_format == "auto":
        prefix = path.read_text(encoding="utf-8")[:4096].lstrip()
        if prefix.startswith("{") or prefix.startswith("["):
            input_format = "json"
        elif re.search(r"(?:^|\n)\s*time_sec\s*,\s*cost\s*(?:\n|$)", prefix):
            input_format = "csv"
        else:
            input_format = "log"

    if input_format == "json":
        return parse_json(path, duplicate_policy)
    if input_format == "csv":
        return parse_csv(path), {}
    return parse_summary_log(path, duplicate_policy), {}


def cost_at_time(run: RunHistory, time_sec: float) -> float | None:
    event_times = [event[0] for event in run.events]
    index = bisect.bisect_right(event_times, time_sec) - 1
    return None if index < 0 else run.events[index][1]


def convergence_statistics(runs: list[RunHistory]) -> dict[str, np.ndarray]:
    event_times = sorted({time for run in runs for time, _ in run.events})
    if not event_times:
        raise ValueError("none of the runs contains a valid solution")

    # Start at the beginning of the requested planning budget.
    first_solution_time = event_times[0]
    event_times.insert(0, 0.0)

    horizons = [
        run.planning_sec
        for run in runs
        if run.planning_sec is not None and run.planning_sec >= first_solution_time
    ]
    horizon = max(horizons + [event_times[-1]])
    if horizon > event_times[-1]:
        event_times.append(horizon)

    success: list[float] = []
    medians: list[float] = []
    minima: list[float] = []
    maxima: list[float] = []

    for time_sec in event_times:
        costs = [
            cost
            for run in runs
            if (cost := cost_at_time(run, time_sec)) is not None
        ]
        success.append(100.0 * len(costs) / len(runs))
        if costs:
            medians.append(float(np.median(costs)))
            minima.append(float(np.min(costs)))
            maxima.append(float(np.max(costs)))
        else:
            medians.append(math.nan)
            minima.append(math.nan)
            maxima.append(math.nan)

    return {
        "time": np.asarray(event_times),
        "success": np.asarray(success),
        "median": np.asarray(medians),
        "minimum": np.asarray(minima),
        "maximum": np.asarray(maxima),
    }


def inferred_title(metadata: dict[str, Any]) -> str:
    robot = str(metadata.get("robot", "")).strip()
    if robot == "g1":
        return "G1 whole body"
    if robot == "franka_single":
        return "Franka single"
    return robot.replace("_", " ").strip().title() or "AORRTC"


def plot_convergence(
    runs: list[RunHistory],
    metadata: dict[str, Any],
    title: str | None,
) -> plt.Figure:
    statistics = convergence_statistics(runs)
    color = "#4b8f29"

    figure, (success_axis, cost_axis) = plt.subplots(
        2,
        1,
        figsize=(12.5, 7.0),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )

    time = statistics["time"]
    success_axis.step(
        time,
        statistics["success"],
        where="post",
        color=color,
        linewidth=3.5,
        label="Success rate",
    )
    success_axis.set_ylabel("Solved Problems [%]", fontsize=AXIS_LABEL_FONT_SIZE)
    success_axis.set_ylim(0.0, 102.0)
    success_axis.set_yticks([0, 25, 50, 75, 100])
    success_axis.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE)

    cost_axis.fill_between(
        time,
        statistics["minimum"],
        statistics["maximum"],
        step="post",
        color=color,
        alpha=0.18,
        linewidth=0.0,
        label="Min–max range",
        zorder=1,
    )
    cost_axis.step(
        time,
        statistics["median"],
        where="post",
        color=color,
        linewidth=3.5,
        label="Median path cost",
        zorder=2,
    )
    median_cost = statistics["median"]
    finite_indices = np.flatnonzero(np.isfinite(median_cost))
    update_indices: list[int] = []
    previous_index: int | None = None
    for index in finite_indices:
        if previous_index is None or not np.isclose(
            median_cost[index],
            median_cost[previous_index],
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            update_indices.append(int(index))
        previous_index = int(index)
    cost_axis.scatter(
        time[update_indices],
        median_cost[update_indices],
        s=52,
        marker="o",
        facecolors="white",
        edgecolors=color,
        linewidths=2.2,
        zorder=3,
        label="Cost update time",
    )
    cost_axis.set_ylabel("Path cost", fontsize=AXIS_LABEL_FONT_SIZE)
    cost_axis.set_xlabel(
        "Computation time [s]", fontsize=AXIS_LABEL_FONT_SIZE
    )
    cost_axis.yaxis.set_major_locator(MultipleLocator(2))
    cost_axis.yaxis.set_minor_locator(MultipleLocator(1))
    cost_axis.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)

    time_limit_sec = positive_float(metadata.get("time_limit_sec"))
    time_horizon = (
        time_limit_sec
        if time_limit_sec is not None
        else float(statistics["time"][-1])
    )
    cost_axis.set_xlim(0.0, time_horizon)
    cost_axis.set_xticks(np.linspace(0.0, time_horizon, 6))
    if time_horizon > 0.0:
        cost_axis.xaxis.set_minor_locator(MultipleLocator(time_horizon / 10.0))

    for axis in (success_axis, cost_axis):
        axis.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
        axis.tick_params(axis="both", which="minor", labelsize=TICK_FONT_SIZE)
        axis.grid(True, which="major", alpha=0.32, linewidth=0.7)
        axis.grid(True, which="minor", alpha=0.15, linewidth=0.5)

    complete = all(run.complete_history for run in runs if run.solved)
    if not complete:
        cost_axis.text(
            0.01,
            0.03,
            "Endpoint-only approximation: intermediate solution updates "
            "were not present in the input.",
            transform=cost_axis.transAxes,
            fontsize=NOTE_FONT_SIZE,
            color="#8a3b12",
            bbox={"facecolor": "white", "edgecolor": "#d8aa91", "alpha": 0.9},
        )

    figure.suptitle(
        title or inferred_title(metadata), fontsize=TITLE_FONT_SIZE
    )
    return figure


def print_summary(runs: list[RunHistory], output: Path | None) -> None:
    solved_runs = [run for run in runs if run.solved]
    print(f"runs: {len(runs)}")
    print(f"solved: {len(solved_runs)}/{len(runs)}")
    if solved_runs:
        initial_times = [run.events[0][0] for run in solved_runs]
        initial_costs = [run.events[0][1] for run in solved_runs]
        final_costs = [run.events[-1][1] for run in solved_runs]
        print(f"median_initial_solution_sec: {np.median(initial_times):.6g}")
        print(f"median_initial_cost: {np.median(initial_costs):.6g}")
        print(f"median_final_cost: {np.median(final_costs):.6g}")
    if output is not None:
        print(f"saved_plot: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a success and median path-cost convergence plot "
            "from single_mbm JSON, text logs, or a legacy time_sec,cost CSV."
        )
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument(
        "--input-format",
        choices=("auto", "json", "log", "csv"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, help="PNG, PDF, or SVG output path")
    parser.add_argument("--title")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--duplicate-runs",
        choices=("last", "first", "all", "error"),
        default="last",
        help="how repeated run ids in text logs are handled (default: last)",
    )
    args = parser.parse_args()

    if not args.input_path.is_file():
        parser.error(f"input file does not exist: {args.input_path}")

    try:
        runs, metadata = load_runs(
            args.input_path, args.input_format, args.duplicate_runs
        )
        figure = plot_convergence(
            runs, metadata, args.title
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=220, bbox_inches="tight")
    print_summary(runs, args.output)

    if args.show or args.output is None:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

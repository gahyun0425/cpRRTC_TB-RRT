#!/usr/bin/env python3
"""Compare planner run JSON files with one solved-runs ECDF and table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, MultipleLocator


UNIT_TO_SECONDS = {
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "ns": 1.0e-9,
}
COMPARISON_KEYS = ("robot", "problem_name", "problem_idx")


@dataclass(frozen=True)
class Run:
    run_idx: int
    seed: int | None
    solved: bool
    planning_sec: float | None


@dataclass
class Series:
    label: str
    path: Path
    time_field: str
    time_unit: str
    metadata: dict[str, Any]
    runs: list[Run]

    @property
    def solved_runs(self) -> list[Run]:
        return [run for run in self.runs if run.solved]

    @property
    def solved_times(self) -> list[float]:
        return sorted(
            run.planning_sec
            for run in self.solved_runs
            if run.planning_sec is not None
        )


def finite_nonnegative(value: Any, context: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{context} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a number") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{context} must be finite and nonnegative")
    return number


def load_series(
    label: str,
    path: Path,
    time_field: str,
    time_unit: str,
) -> Series:
    if time_unit not in UNIT_TO_SECONDS:
        raise ValueError(
            f"{label}: unsupported time unit {time_unit!r}; "
            "use s, ms, us, or ns"
        )
    if not path.is_file():
        raise ValueError(f"{label}: input file does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: input JSON must be an object")
    raw_results = payload.get("results")
    records = raw_results if isinstance(raw_results, list) else [payload]
    if not records:
        raise ValueError(f"{label}: input JSON contains no run records")

    factor = UNIT_TO_SECONDS[time_unit]
    runs: list[Run] = []
    for ordinal, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{label}: run {ordinal} is not a JSON object")
        solved = record.get("solved")
        if not isinstance(solved, bool):
            raise ValueError(f"{label}: run {ordinal} has no boolean solved field")

        raw_elapsed = record.get(time_field)
        planning_sec: float | None = None
        if raw_elapsed is not None:
            planning_sec = finite_nonnegative(
                raw_elapsed,
                f"{label}: run {ordinal} field {time_field}",
            ) * factor
        elif solved:
            raise ValueError(
                f"{label}: solved run {ordinal} has no {time_field!r} field"
            )

        raw_seed = record.get("seed")
        seed = (
            raw_seed
            if isinstance(raw_seed, int) and not isinstance(raw_seed, bool)
            else None
        )
        raw_run_idx = record.get("run_idx", ordinal)
        run_idx = raw_run_idx if isinstance(raw_run_idx, int) else ordinal
        runs.append(Run(run_idx, seed, solved, planning_sec))

    metadata = {
        key: payload[key]
        for key in (*COMPARISON_KEYS, "planner", "format", "timeout_sec", "timing_scope")
        if key in payload
    }
    return Series(label, path, time_field, time_unit, metadata, runs)


def validate_comparability(
    series_list: list[Series],
    allow_unequal_runs: bool,
    allow_mixed_problems: bool,
) -> None:
    if len(series_list) < 2:
        return

    run_counts = {len(series.runs) for series in series_list}
    if len(run_counts) > 1 and not allow_unequal_runs:
        details = ", ".join(
            f"{series.label}={len(series.runs)}" for series in series_list
        )
        raise ValueError(
            f"run counts differ ({details}); use the same number of trials or "
            "pass --allow-unequal-runs"
        )

    if not allow_mixed_problems:
        for key in COMPARISON_KEYS:
            values = {
                json.dumps(series.metadata[key], sort_keys=True)
                for series in series_list
                if key in series.metadata
            }
            if len(values) > 1:
                details = ", ".join(
                    f"{series.label}={series.metadata.get(key, '<missing>')}"
                    for series in series_list
                )
                raise ValueError(
                    f"input metadata differs for {key} ({details}); "
                    "pass --allow-mixed-problems only if this is intentional"
                )


def percentile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summary_row(series: Series) -> dict[str, Any]:
    times = series.solved_times
    solved = len(series.solved_runs)
    row: dict[str, Any] = {
        "planner": series.label,
        "runs": len(series.runs),
        "solved": solved,
        "success_rate_percent": 100.0 * solved / len(series.runs),
        "time_field": series.time_field,
        "time_unit": series.time_unit,
        "min_sec": None,
        "median_sec": None,
        "mean_sec": None,
        "p95_sec": None,
        "max_sec": None,
    }
    if times:
        row.update(
            {
                "min_sec": min(times),
                "median_sec": statistics.median(times),
                "mean_sec": statistics.fmean(times),
                "p95_sec": percentile(times, 0.95),
                "max_sec": max(times),
            }
        )
    return row


def format_ms(seconds: Any) -> str:
    if seconds is None:
        return "-"
    return f"{float(seconds) * 1.0e3:.3f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Planner | Runs | Solved | Success | Min (ms) | Median (ms) | Mean (ms) | P95 (ms) | Max (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = str(row["planner"]).replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row["runs"]),
                    str(row["solved"]),
                    f"{row['success_rate_percent']:.1f}%",
                    format_ms(row["min_sec"]),
                    format_ms(row["median_sec"]),
                    format_ms(row["mean_sec"]),
                    format_ms(row["p95_sec"]),
                    format_ms(row["max_sec"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_combined_json(path: Path, series_list: list[Series]) -> None:
    payload = {
        "format": "planner_comparison_runs_v1",
        "series": [
            {
                "label": series.label,
                "source": str(series.path.resolve()),
                "source_time_field": series.time_field,
                "source_time_unit": series.time_unit,
                "metadata": series.metadata,
                "runs": [
                    {
                        "run_idx": run.run_idx,
                        "seed": run.seed,
                        "solved": run.solved,
                        "planning_sec": run.planning_sec,
                    }
                    for run in series.runs
                ],
            }
            for series in series_list
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_ecdf(
    series_list: list[Series],
    title: str | None,
    logarithmic_x: bool,
    percent_y: bool,
    figure_width: float,
    figure_height: float,
    colors: dict[str, str],
    linestyles: dict[str, str],
    legend_columns: int,
    font_scale: float,
    legend_counts: bool,
    line_width: float,
    uniform_font_size: bool,
    hide_first_x_tick_label: bool,
) -> plt.Figure:
    known_times = [
        run.planning_sec
        for series in series_list
        for run in series.runs
        if run.planning_sec is not None and run.planning_sec > 0.0
    ]
    if not known_times:
        raise ValueError("no positive planning times were found")

    minimum = min(known_times)
    maximum = max(known_times)
    if logarithmic_x:
        minimum_decade = math.floor(math.log10(minimum))
        maximum_decade = math.ceil(math.log10(maximum))
        if minimum_decade == maximum_decade:
            maximum_decade += 1
        x_min = 10.0**minimum_decade
        x_max = 10.0**maximum_decade
    elif math.isclose(minimum, maximum, rel_tol=1.0e-12, abs_tol=0.0):
        x_min = minimum / 2.0
        x_max = maximum * 2.0
    else:
        span = maximum - minimum
        x_min = max(0.0, minimum - span * 0.05)
        x_max = maximum + span * 0.05

    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    for series in series_list:
        solved_times = [value for value in series.solved_times if value > 0.0]
        if solved_times:
            x_values = [x_min, *solved_times, x_max]
            if percent_y:
                denominator = len(series.runs)
                solved_percentages = [
                    100.0 * solved_count / denominator
                    for solved_count in range(1, len(solved_times) + 1)
                ]
                y_values = [0.0, *solved_percentages, solved_percentages[-1]]
            else:
                y_values = [0, *range(1, len(solved_times) + 1), len(solved_times)]
        else:
            x_values = [x_min, x_max]
            y_values = [0, 0]
        legend_label = series.label
        if legend_counts:
            legend_label += f" ({len(series.solved_runs)}/{len(series.runs)})"
        axis.step(
            x_values,
            y_values,
            where="post",
            linewidth=line_width,
            color=colors.get(series.label),
            linestyle=linestyles.get(series.label, "solid"),
            label=legend_label,
        )

    if logarithmic_x:
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(base=10.0))
    axis.set_xlim(x_min, x_max)
    title_font_size = 16 * font_scale
    axis_font_size = title_font_size if uniform_font_size else 14 * font_scale
    tick_font_size = title_font_size if uniform_font_size else 11 * font_scale
    legend_font_size = title_font_size if uniform_font_size else 9 * font_scale

    axis.set_xlabel("Planning Time (seconds)", fontsize=axis_font_size)
    if percent_y:
        axis.set_ylim(0, 105)
        axis.set_ylabel("Solved Problems (%)", fontsize=axis_font_size)
        axis.yaxis.set_major_locator(MultipleLocator(25))
        axis.grid(True, which="major", linestyle="--", alpha=0.55)
        axis.grid(True, which="minor", linestyle=":", alpha=0.2)
    else:
        axis.set_ylim(0, max(len(series.runs) for series in series_list) * 1.05)
        axis.set_ylabel("Solved Runs", fontsize=axis_font_size)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(True, which="major", alpha=0.35)
        axis.grid(True, which="minor", alpha=0.15)
    if title:
        axis.set_title(title, fontsize=title_font_size)
    axis.tick_params(axis="both", which="major", labelsize=tick_font_size)
    axis.legend(
        loc="lower right",
        ncol=legend_columns,
        fontsize=legend_font_size,
    )
    if hide_first_x_tick_label:
        figure.canvas.draw()
        visible_x_minimum, visible_x_maximum = axis.get_xlim()
        for location, label in zip(axis.get_xticks(), axis.get_xticklabels()):
            if (
                location >= visible_x_minimum * (1.0 - 1.0e-12)
                and location <= visible_x_maximum * (1.0 + 1.0e-12)
            ):
                label.set_visible(False)
                break
    figure.tight_layout()
    return figure


def inferred_title(series_list: list[Series]) -> str:
    first = series_list[0].metadata
    robot = first.get("robot")
    problem = first.get("problem_name")
    problem_idx = first.get("problem_idx")
    if robot is None or problem is None:
        return "Planner Runtime Comparison"
    title = f"{robot} / {problem}"
    if problem_idx is not None:
        title += f" #{problem_idx}"
    return title


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create one solved-runs-versus-time ECDF, summary table, and "
            "normalized JSON from multiple planner result JSON files."
        )
    )
    parser.add_argument(
        "--series",
        action="append",
        nargs=4,
        required=True,
        metavar=("LABEL", "JSON", "TIME_FIELD", "UNIT"),
        help=(
            "planner label, JSON path, per-run time field, and its unit "
            "(s/ms/us/ns); repeat once per planner"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/planner_comparison_ecdf.png"),
        help="combined ECDF image path",
    )
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--combined-json", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument(
        "--percent-y",
        action="store_true",
        help="show solved runs as a percentage of all runs",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="omit the plot title",
    )
    parser.add_argument("--fig-width", type=float, default=11.0)
    parser.add_argument("--fig-height", type=float, default=7.0)
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument(
        "--uniform-font-size",
        action="store_true",
        help="use the title font size for axes, ticks, and legend text",
    )
    parser.add_argument(
        "--hide-first-x-tick-label",
        action="store_true",
        help="hide the leftmost x-axis label to avoid overlap with the y-axis zero",
    )
    parser.add_argument("--line-width", type=float, default=2.4)
    parser.add_argument(
        "--color",
        action="append",
        nargs=2,
        default=[],
        metavar=("LABEL", "COLOR"),
        help="set a series color; repeat for multiple series",
    )
    parser.add_argument(
        "--linestyle",
        action="append",
        nargs=2,
        default=[],
        metavar=("LABEL", "STYLE"),
        help="set a series line style; repeat for multiple series",
    )
    parser.add_argument("--legend-columns", type=int, default=1)
    parser.add_argument(
        "--legend-no-counts",
        action="store_true",
        help="omit solved/total run counts from legend labels",
    )
    parser.add_argument("--allow-unequal-runs", action="store_true")
    parser.add_argument("--allow-mixed-problems", action="store_true")
    args = parser.parse_args()

    try:
        series_list = [
            load_series(label, Path(path), time_field, unit)
            for label, path, time_field, unit in args.series
        ]
        labels = [series.label for series in series_list]
        if len(labels) != len(set(labels)):
            raise ValueError("planner labels must be unique")
        colors = dict(args.color)
        linestyles = dict(args.linestyle)
        styled_labels = set(colors) | set(linestyles)
        unknown_labels = styled_labels - set(labels)
        if unknown_labels:
            raise ValueError(
                "style specified for unknown series: "
                + ", ".join(sorted(unknown_labels))
            )
        if (
            args.fig_width <= 0.0
            or args.fig_height <= 0.0
            or args.font_scale <= 0.0
            or args.line_width <= 0.0
        ):
            raise ValueError(
                "figure width, height, font scale, and line width must be positive"
            )
        if args.legend_columns <= 0:
            raise ValueError("legend columns must be positive")
        validate_comparability(
            series_list,
            args.allow_unequal_runs,
            args.allow_mixed_problems,
        )
        rows = [summary_row(series) for series in series_list]
        figure = plot_ecdf(
            series_list,
            None if args.no_title else (args.title or inferred_title(series_list)),
            not args.linear_x,
            args.percent_y,
            args.fig_width,
            args.fig_height,
            colors,
            linestyles,
            args.legend_columns,
            args.font_scale,
            not args.legend_no_counts,
            args.line_width,
            args.uniform_font_size,
            args.hide_first_x_tick_label,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    output = args.output
    stem = output.with_suffix("")
    summary_csv = args.summary_csv or stem.with_name(stem.name + "_summary.csv")
    summary_md = args.summary_md or stem.with_name(stem.name + "_summary.md")
    combined_json = args.combined_json or stem.with_name(stem.name + "_runs.json")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    write_summary_csv(summary_csv, rows)
    table = markdown_table(rows)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text(table, encoding="utf-8")
    write_combined_json(combined_json, series_list)

    print(table, end="")
    print(f"saved_plot: {output}")
    print(f"saved_summary_csv: {summary_csv}")
    print(f"saved_summary_md: {summary_md}")
    print(f"saved_combined_json: {combined_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

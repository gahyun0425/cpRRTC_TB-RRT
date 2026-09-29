#!/usr/bin/env python3
"""Plot planning time measurements from PATACON range-sweep JSON files."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RANGE_FILE_RE = re.compile(r"range_([-+]?(?:\d+(?:\.\d*)?|\.\d+))\.json$")


def load_sweep(sweep_dir: Path):
    series = []

    for json_path in sweep_dir.glob("range_*.json"):
        match = RANGE_FILE_RE.fullmatch(json_path.name)
        if match is None:
            continue

        with json_path.open("r", encoding="utf-8") as file:
            document = json.load(file)

        results = document.get("results", [])
        if not results:
            continue

        range_value = float(
            results[0].get("settings", {}).get("range", match.group(1))
        )
        planning_times = np.asarray(
            [result["kernel_ns"] / 1.0e9 for result in results if "kernel_ns" in result],
            dtype=float,
        )
        if planning_times.size == 0:
            continue

        solved_count = sum(result.get("solved") is True for result in results)
        series.append((range_value, planning_times, len(results), solved_count))

    if not series:
        raise ValueError(f"No range_*.json files with kernel_ns found in {sweep_dir}")

    return sorted(series, key=lambda item: item[0])


def write_summary_csv(output_path: Path, datasets):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "planner",
                "range",
                "runs",
                "solved_runs",
                "mean_planning_s",
                "median_planning_s",
                "std_planning_s",
                "min_planning_s",
                "p25_planning_s",
                "p75_planning_s",
                "max_planning_s",
            ]
        )
        for label, series in datasets:
            for range_value, times, run_count, solved_count in series:
                writer.writerow(
                    [
                        label,
                        f"{range_value:g}",
                        run_count,
                        solved_count,
                        f"{np.mean(times):.9f}",
                        f"{np.median(times):.9f}",
                        f"{np.std(times, ddof=1):.9f}",
                        f"{np.min(times):.9f}",
                        f"{np.percentile(times, 25):.9f}",
                        f"{np.percentile(times, 75):.9f}",
                        f"{np.max(times):.9f}",
                    ]
                )


def plot(output_path: Path, datasets, title: str):
    all_ranges = np.asarray(
        sorted({item[0] for _, series in datasets for item in series})
    )
    if len(all_ranges) > 1:
        spacing = float(np.min(np.diff(all_ranges)))
    else:
        spacing = max(abs(float(all_ranges[0])) * 0.25, 0.05)

    dataset_count = len(datasets)
    if dataset_count == 1:
        offsets = np.zeros(1)
        box_width = spacing * 0.34
    else:
        offsets = np.linspace(-spacing * 0.20, spacing * 0.20, dataset_count)
        box_width = spacing * 0.26

    colors = ["#E68613", "#4C78A8", "#54A24B", "#B279A2", "#E45756"]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    rng = np.random.default_rng(0)
    total_runs = 0
    for dataset_idx, ((label, series), offset) in enumerate(zip(datasets, offsets)):
        color = colors[dataset_idx % len(colors)]
        ranges = np.asarray([item[0] for item in series])
        time_groups = [item[1] for item in series]
        positions = ranges + offset
        means = np.asarray([np.mean(times) for times in time_groups])
        total_runs += sum(len(times) for times in time_groups)

        for position, times in zip(positions, time_groups):
            jitter = rng.uniform(-box_width * 0.42, box_width * 0.42, times.size)
            ax.scatter(
                np.full(times.size, position) + jitter,
                times,
                s=8,
                alpha=0.10,
                color=color,
                linewidths=0,
                zorder=1,
            )

        box = ax.boxplot(
            time_groups,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            zorder=2,
        )
        for patch in box["boxes"]:
            patch.set(facecolor=color, edgecolor=color, alpha=0.24)
        for median in box["medians"]:
            median.set(color=color, linewidth=2.0)
        for element_name in ("whiskers", "caps"):
            for element in box[element_name]:
                element.set(color=color, linewidth=1.2)

        ax.plot(
            positions,
            means,
            color=color,
            marker="o",
            markersize=5.5,
            linewidth=2.1,
            label=label,
            zorder=3,
        )

    ax.set_xlabel("Range")
    ax.set_ylabel("Planning time [s]")
    ax.set_title(title)
    ax.set_xticks(all_ranges)
    ax.set_xticklabels([f"{value:g}" for value in all_ranges])
    ax.set_xlim(
        all_ranges[0] - spacing * 0.55, all_ranges[-1] + spacing * 0.55
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.99,
        0.02,
        f"Boxes: IQR, center: median, circles: mean\nn = {total_runs:,} runs",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot range versus planning time from PATACON range-sweep JSON files."
    )
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Legend labels in the same order as SWEEP_DIRS",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path (default: SWEEP_DIR/range_vs_planning_time.png)",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="CSV output path (default: SWEEP_DIR/range_vs_planning_time.csv)",
    )
    parser.add_argument(
        "--title",
        help="Plot title",
    )
    args = parser.parse_args()

    sweep_dirs = [path.resolve() for path in args.sweep_dirs]
    if args.labels is not None and len(args.labels) != len(sweep_dirs):
        parser.error("--labels must contain one label for each SWEEP_DIR")
    labels = args.labels or [path.name for path in sweep_dirs]
    datasets = [
        (label, load_sweep(sweep_dir))
        for label, sweep_dir in zip(labels, sweep_dirs)
    ]

    is_comparison = len(datasets) > 1
    plot_name = (
        "range_vs_planning_time_comparison.png"
        if is_comparison
        else "range_vs_planning_time.png"
    )
    csv_name = (
        "range_vs_planning_time_comparison.csv"
        if is_comparison
        else "range_vs_planning_time.csv"
    )
    output_path = (args.output or sweep_dirs[0] / plot_name).resolve()
    csv_path = (args.summary_csv or sweep_dirs[0] / csv_name).resolve()
    title = args.title or (
        "Range Sweep: Planning Time Comparison"
        if is_comparison
        else "PATACON Range Sweep: Planning Time"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot(output_path, datasets, title)
    write_summary_csv(csv_path, datasets)

    print(f"Wrote plot: {output_path}")
    print(f"Wrote summary: {csv_path}")
    for label, series in datasets:
        for range_value, times, run_count, solved_count in series:
            print(
                f"{label}: range={range_value:g}  runs={run_count}  "
                f"solved={solved_count}  mean={np.mean(times):.6f}s  "
                f"median={np.median(times):.6f}s"
            )


if __name__ == "__main__":
    main()

import argparse
import csv

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument(
        "--title",
        default="AORRTC Cost Convergence"
    )
    args = parser.parse_args()

    times = []
    costs = []

    with open(args.csv_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            times.append(float(row["time_sec"]))
            costs.append(float(row["cost"]))

    if not times:
        print("No AORRTC solution history to plot.")
        return

    plt.figure(figsize=(8, 5))

    # 새로운 best path가 발견될 때 cost가 계단식으로 감소
    plt.step(
        times,
        costs,
        where="post",
        label="Best path cost"
    )

    # 실제 solution이 갱신된 시점을 점으로 표시
    plt.scatter(
        times,
        costs,
        s=35,
        zorder=3
    )

    plt.xlabel("Solution update time from planning start [s]")
    plt.ylabel("Path cost")
    plt.title(args.title)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
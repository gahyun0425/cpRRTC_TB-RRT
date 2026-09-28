#!/usr/bin/env python3
"""Replay a saved G1 planner-result JSON in a MuJoCo environment XML."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

import visualize_g1
from mujoco_video import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    add_video_view_argument,
    validate_video_views,
)


EMPTY_DYNAMIC_ENVIRONMENT = {"sphere": [], "cylinder": [], "box": []}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        type=Path,
        required=True,
        help="MuJoCo XML containing both the G1 model and the static scene.",
    )
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="Planner result JSON containing path_start_to_goal or path.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=visualize_g1.default_urdf_path(),
        help="G1 URDF providing per-joint velocity limits.",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--velocity-scale",
        "--speed",
        dest="velocity_scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to the 2x-playback URDF joint and floating-base "
            "velocity limits; must be in (0, 1] (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=visualize_g1.DEFAULT_TRAJECTORY_ACCELERATION,
        help=(
            "Maximum planning-coordinate acceleration per second squared "
            f"(default: "
            f"{visualize_g1.DEFAULT_TRAJECTORY_ACCELERATION:g})."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate both files without opening a viewer window.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Render one simulation-timed replay directly to this MP4 file.",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=os.environ.get("PRRTC_VIDEO_WIDTH", str(DEFAULT_VIDEO_WIDTH)),
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=os.environ.get("PRRTC_VIDEO_HEIGHT", str(DEFAULT_VIDEO_HEIGHT)),
    )
    add_video_view_argument(parser)
    args = parser.parse_args()
    if (
        not math.isfinite(args.fps)
        or not math.isfinite(args.velocity_scale)
        or not math.isfinite(args.acceleration)
        or args.fps <= 0.0
        or args.velocity_scale <= 0.0
        or args.velocity_scale > 1.0
        or args.acceleration <= 0.0
    ):
        parser.error(
            "--fps and --acceleration must be positive, and "
            "--velocity-scale must be in (0, 1]"
        )
    if args.validate_only and args.video is not None:
        parser.error("--validate-only and --video cannot be used together")
    if args.video is not None and args.video.suffix.lower() != ".mp4":
        parser.error("--video output must use the .mp4 extension")
    if (
        args.video_width <= 0
        or args.video_height <= 0
        or args.video_width % 2 != 0
        or args.video_height % 2 != 0
    ):
        parser.error(
            "--video-width and --video-height must be positive even integers"
        )
    validate_video_views(parser, args)
    return args


def load_result_path(path: Path) -> list[list[float]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("planner result must be a JSON object")
    if document.get("robot", "g1") != "g1":
        raise ValueError(f"planner result robot is not g1: {document.get('robot')!r}")
    if document.get("solved") is False:
        raise ValueError("planner result is not solved")

    values = document.get("path_start_to_goal")
    source = "path_start_to_goal"
    if not isinstance(values, list) or len(values) < 2:
        values = document.get("path")
        source = "path"
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(
                "planner result must contain at least two path_start_to_goal "
                "or path waypoints"
            )
        orientation = document.get("path_orientation")
        if orientation == "goal_to_start":
            values = list(reversed(values))
        elif orientation not in (None, "start_to_goal"):
            raise ValueError(f"unsupported path_orientation: {orientation!r}")

    waypoints = [
        visualize_g1.finite_vector(
            waypoint,
            visualize_g1.G1_CONFIGURATION_DIMENSION,
            f"{source} waypoint {index}",
        )
        for index, waypoint in enumerate(values)
    ]

    expected_start = document.get("start")
    if expected_start is not None:
        start = visualize_g1.finite_vector(
            expected_start,
            visualize_g1.G1_CONFIGURATION_DIMENSION,
            "planner result start",
        )
        if max(abs(left - right) for left, right in zip(waypoints[0], start)) > 1.0e-5:
            raise ValueError("selected result path is not in start-to-goal order")
    return waypoints


def load_result_planning_time(path: Path) -> float | None:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("planner result must be a JSON object")
    if document.get("planning_sec") is not None:
        planning_time_sec = float(document["planning_sec"])
    elif document.get("planning_ns") is not None:
        planning_time_sec = float(document["planning_ns"]) / 1.0e9
    elif document.get("kernel_ns") is not None:
        planning_time_sec = float(document["kernel_ns"]) / 1.0e9
    else:
        return None
    if not math.isfinite(planning_time_sec) or planning_time_sec < 0.0:
        raise ValueError("planner result planning time must be finite and nonnegative")
    return planning_time_sec


def validate(
    environment_path: Path,
    waypoints: list[list[float]],
    fps: float,
    velocity_limits: np.ndarray,
    acceleration: float,
) -> None:
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo Python package is required: python3 -m pip install mujoco"
        ) from error

    model = visualize_g1.build_kinematic_model(mujoco, environment_path, None)
    data = mujoco.MjData(model)
    base_address, joint_addresses = visualize_g1.resolve_model_layout(
        mujoco, model
    )
    visualize_g1.validate_joint_limits(mujoco, model, waypoints)
    trajectory = visualize_g1.time_parameterize_waypoints(
        waypoints,
        velocity_limits,
        acceleration,
        fps,
    )
    for configuration in visualize_g1.time_parameterized_frames(
        trajectory,
        fps,
    ):
        visualize_g1.apply_configuration(
            mujoco,
            model,
            data,
            base_address,
            joint_addresses,
            configuration,
        )
    if not np.isfinite(data.qpos).all():
        raise ValueError("replayed qpos contains a non-finite value")
    print(
        f"validated {len(waypoints)} time-parameterized G1 waypoints "
        f"({trajectory.duration:.3f} s, "
        f"peak velocity={trajectory.peak_velocity:.6g}, "
        f"peak acceleration={trajectory.peak_acceleration:.6g}, "
        f"limit utilization="
        f"{100.0 * trajectory.velocity_limit_utilization:.1f}% velocity / "
        f"{100.0 * trajectory.acceleration_limit_utilization:.1f}% "
        f"acceleration) against "
        f"{environment_path} (nq={model.nq}, ngeom={model.ngeom})"
    )


def main() -> int:
    args = parse_args()
    environment_path = args.environment.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    if not environment_path.is_file():
        raise FileNotFoundError(f"MuJoCo environment not found: {environment_path}")
    if not result_path.is_file():
        raise FileNotFoundError(f"planner result not found: {result_path}")

    waypoints = load_result_path(result_path)
    planning_time_sec = load_result_planning_time(result_path)
    velocity_limits = visualize_g1.load_planning_velocity_limits(
        urdf_path,
        args.velocity_scale,
    )
    if args.validate_only:
        validate(
            environment_path,
            waypoints,
            args.fps,
            velocity_limits,
            args.acceleration,
        )
        return 0
    if args.video is not None:
        visualize_g1.save_video(
            environment_path,
            waypoints,
            EMPTY_DYNAMIC_ENVIRONMENT,
            None,
            args.video,
            args.fps,
            velocity_limits,
            args.acceleration,
            "qpos",
            1.0,
            args.video_width,
            args.video_height,
            args.video_views,
            planning_time_sec,
        )
        return 0

    visualize_g1.replay(
        environment_path,
        waypoints,
        EMPTY_DYNAMIC_ENVIRONMENT,
        None,
        args.fps,
        velocity_limits,
        args.acceleration,
        "qpos",
        1.0,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}")
        raise SystemExit(1)

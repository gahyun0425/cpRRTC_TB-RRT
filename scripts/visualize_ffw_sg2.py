#!/usr/bin/env python3
"""Replay a 15-DoF or right-arm-only pRRTC FFW-SG2 MuJoCo trajectory."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


DUAL_ARM_PLANNING_JOINTS = (
    "lift_joint",
    "arm_l_joint1",
    "arm_l_joint2",
    "arm_l_joint3",
    "arm_l_joint4",
    "arm_l_joint5",
    "arm_l_joint6",
    "arm_l_joint7",
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
)
SINGLE_ARM_PLANNING_JOINTS = (
    "lift_joint",
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
)
LEFT_ARM_JOINTS = tuple(f"arm_l_joint{index}" for index in range(1, 8))
SUPPORTED_JOINT_ORDERS = {
    DUAL_ARM_PLANNING_JOINTS,
    SINGLE_ARM_PLANNING_JOINTS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.7,
        help="Maximum configuration-coordinate change per second.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate without opening a viewer window.",
    )
    args = parser.parse_args()
    if args.fps <= 0.0 or args.speed <= 0.0:
        parser.error("--fps and --speed must be positive")
    return args


def load_trajectory(path: Path) -> tuple[tuple[str, ...], list[list[float]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    joint_names = tuple(document.get("joint_names", ()))
    if joint_names not in SUPPORTED_JOINT_ORDERS:
        raise ValueError(
            "trajectory joint order does not match a supported FFW-SG2 planning order"
        )

    waypoints = document.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise ValueError("trajectory must contain at least two waypoints")
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, list) or len(waypoint) != len(joint_names):
            raise ValueError(
                f"waypoint {index} is not a {len(joint_names)}-DoF configuration"
            )
        if not all(math.isfinite(float(value)) for value in waypoint):
            raise ValueError(f"waypoint {index} contains a non-finite value")
    normalized = [[float(value) for value in waypoint] for waypoint in waypoints]

    expected_start = document.get("start")
    if not isinstance(expected_start, list) or len(expected_start) != len(joint_names):
        raise ValueError(
            f"trajectory is missing its {len(joint_names)}-DoF start configuration"
        )
    if max(
        abs(normalized[0][index] - float(expected_start[index]))
        for index in range(len(joint_names))
    ) > 1.0e-5:
        raise ValueError("trajectory was not converted to start-to-goal order")
    return joint_names, normalized


def resolve_qpos_addresses(mujoco, model, joint_names) -> list[int]:
    addresses: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing planning joint: {name}")
        joint_type = int(model.jnt_type[joint_id])
        scalar_types = {
            int(mujoco.mjtJoint.mjJNT_SLIDE),
            int(mujoco.mjtJoint.mjJNT_HINGE),
        }
        if joint_type not in scalar_types:
            raise ValueError(f"planning joint is not scalar: {name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def validate_joint_limits(
    mujoco, model, joint_names, waypoints: list[list[float]]
) -> None:
    tolerance = 1.0e-6
    for waypoint_index, waypoint in enumerate(waypoints):
        for name, value in zip(joint_names, waypoint):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if not model.jnt_limited[joint_id]:
                continue
            lower, upper = model.jnt_range[joint_id]
            if value < lower - tolerance or value > upper + tolerance:
                raise ValueError(
                    f"waypoint {waypoint_index}, {name}={value} is outside "
                    f"[{lower}, {upper}]"
                )


def apply_configuration(mujoco, model, data, addresses, configuration) -> None:
    for address, value in zip(addresses, configuration):
        data.qpos[address] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def interpolated_frames(waypoints, fps: float, speed: float):
    for start, goal in zip(waypoints, waypoints[1:]):
        max_change = max(abs(goal[i] - start[i]) for i in range(len(start)))
        frame_count = max(2, math.ceil(max_change / speed * fps))
        for frame_index in range(frame_count):
            ratio = (frame_index + 1) / frame_count
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            yield [
                start[i] + (goal[i] - start[i]) * smooth_ratio
                for i in range(len(start))
            ]


def replay(model_path: Path, joint_names, waypoints, fps: float, speed: float) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo Python package is required: python3 -m pip install mujoco"
        ) from error

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = resolve_qpos_addresses(mujoco, model, joint_names)
    validate_joint_limits(mujoco, model, joint_names, waypoints)
    if joint_names == SINGLE_ARM_PLANNING_JOINTS:
        left_arm_addresses = resolve_qpos_addresses(mujoco, model, LEFT_ARM_JOINTS)
        for address in left_arm_addresses:
            data.qpos[address] = 0.0
    apply_configuration(mujoco, model, data, addresses, waypoints[0])

    print("MuJoCo viewer: start -> goal 경로를 반복 재생합니다.")
    print("창을 닫으면 single_mbm 실행이 종료됩니다.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.45, 0.0, 0.9)
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        viewer.sync()

        frame_period = 1.0 / fps
        while viewer.is_running():
            apply_configuration(mujoco, model, data, addresses, waypoints[0])
            viewer.sync()
            time.sleep(0.75)
            deadline = time.perf_counter()
            for configuration in interpolated_frames(waypoints, fps, speed):
                if not viewer.is_running():
                    return
                apply_configuration(mujoco, model, data, addresses, configuration)
                viewer.sync()
                deadline += frame_period
                time.sleep(max(0.0, deadline - time.perf_counter()))
            time.sleep(1.0)


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    trajectory_path = args.trajectory.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory not found: {trajectory_path}")

    joint_names, waypoints = load_trajectory(trajectory_path)
    validate_only = args.validate_only or os.environ.get(
        "PRRTC_MUJOCO_VALIDATE_ONLY"
    ) == "1"
    if validate_only:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(model_path))
        resolve_qpos_addresses(mujoco, model, joint_names)
        validate_joint_limits(mujoco, model, joint_names, waypoints)
        print(
            f"validated {len(waypoints)} waypoints, {len(joint_names)} joints, "
            f"model nq={model.nq}"
        )
        return 0

    replay(model_path, joint_names, waypoints, args.fps, args.speed)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        print(f"visualization error: {error}", file=sys.stderr)
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # MuJoCo 3.6 can leave a native viewer thread alive during Python teardown.
    # This script is an isolated visualization subprocess, so terminate it after
    # the viewer context has released its resources and all output is flushed.
    os._exit(exit_code)

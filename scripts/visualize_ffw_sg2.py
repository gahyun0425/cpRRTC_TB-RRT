#!/usr/bin/env python3
"""Replay a 15-DoF, mobile-base, or right-arm-only pRRTC FFW-SG2 trajectory."""

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
MOBILITY_BASE_JOINTS = ("base_x", "base_y", "base_yaw")
MOBILITY_BASE_LIMITS = {
    "base_x": (-0.5, 0.5),
    "base_y": (-0.5, 0.5),
    "base_yaw": (-3.14, 3.14),
}
MOBILITY_PLANNING_JOINTS = MOBILITY_BASE_JOINTS + DUAL_ARM_PLANNING_JOINTS
OBJECT_FREEJOINT = "object_freejoint"
GRIPPER_SITE_NAMES = (
    "gripper_l_rh_p12_rn_base",
    "gripper_r_rh_p12_rn_base",
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
    MOBILITY_PLANNING_JOINTS,
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


def resolve_qpos_addresses(mujoco, model, joint_names) -> list[int | None]:
    addresses: list[int | None] = []
    for name in joint_names:
        if name in MOBILITY_BASE_JOINTS:
            addresses.append(None)
            continue
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
            if name in MOBILITY_BASE_LIMITS:
                lower, upper = MOBILITY_BASE_LIMITS[name]
                if value < lower - tolerance or value > upper + tolerance:
                    raise ValueError(
                        f"waypoint {waypoint_index}, {name}={value} is outside "
                        f"[{lower}, {upper}]"
                    )
                continue
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if not model.jnt_limited[joint_id]:
                continue
            lower, upper = model.jnt_range[joint_id]
            if value < lower - tolerance or value > upper + tolerance:
                raise ValueError(
                    f"waypoint {waypoint_index}, {name}={value} is outside "
                    f"[{lower}, {upper}]"
                )


def resolve_mobility_base_body(mujoco, model, joint_names):
    if joint_names != MOBILITY_PLANNING_JOINTS:
        return None
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if body_id < 0:
        raise ValueError("MuJoCo model is missing mobile base body: base_link")
    origin = tuple(float(value) for value in model.body_pos[body_id])
    return body_id, origin


def resolve_object_freejoint(mujoco, model, joint_names) -> int | None:
    if joint_names not in (DUAL_ARM_PLANNING_JOINTS, MOBILITY_PLANNING_JOINTS):
        return None
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        OBJECT_FREEJOINT,
    )
    if joint_id < 0:
        return None
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"{OBJECT_FREEJOINT} is not a MuJoCo freejoint")
    return int(model.jnt_qposadr[joint_id])


def dot(left, right) -> float:
    return sum(left[index] * right[index] for index in range(3))


def cross(left, right) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def norm(vector) -> float:
    return math.sqrt(dot(vector, vector))


def normalize(vector) -> tuple[float, float, float] | None:
    length = norm(vector)
    if length <= 1.0e-9:
        return None
    return tuple(value / length for value in vector)


def subtract_projection(vector, axis) -> tuple[float, float, float]:
    scale = dot(vector, axis)
    return tuple(vector[index] - scale * axis[index] for index in range(3))


def site_matrix_column(data, site_id: int, column: int) -> tuple[float, float, float]:
    matrix = data.site_xmat[site_id]
    return (
        float(matrix[column]),
        float(matrix[3 + column]),
        float(matrix[6 + column]),
    )


def matrix_to_quat(matrix) -> tuple[float, float, float, float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * scale,
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        )
    if matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        return (
            (matrix[2][1] - matrix[1][2]) / scale,
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
        )
    if matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        return (
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
        )
    scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
    return (
        (matrix[1][0] - matrix[0][1]) / scale,
        (matrix[0][2] + matrix[2][0]) / scale,
        (matrix[1][2] + matrix[2][1]) / scale,
        0.25 * scale,
    )


def object_quat_from_grippers(data, left_site: int, right_site: int):
    left_pos = data.site_xpos[left_site]
    right_pos = data.site_xpos[right_site]
    y_axis = normalize(
        tuple(float(left_pos[index] - right_pos[index]) for index in range(3))
    )
    if y_axis is None:
        return (1.0, 0.0, 0.0, 0.0)

    left_z = site_matrix_column(data, left_site, 2)
    right_z = site_matrix_column(data, right_site, 2)
    x_hint = normalize(
        tuple(left_z[index] + right_z[index] for index in range(3))
    )
    if x_hint is None:
        x_hint = (1.0, 0.0, 0.0)

    x_axis = normalize(subtract_projection(x_hint, y_axis))
    if x_axis is None:
        z_fallback = (0.0, 0.0, 1.0)
        x_axis = normalize(cross(y_axis, z_fallback))
    if x_axis is None:
        x_axis = (1.0, 0.0, 0.0)

    z_axis = normalize(cross(x_axis, y_axis))
    if z_axis is None:
        return (1.0, 0.0, 0.0, 0.0)

    left_y = site_matrix_column(data, left_site, 1)
    right_y = site_matrix_column(data, right_site, 1)
    z_hint = normalize(
        tuple(right_y[index] - left_y[index] for index in range(3))
    )
    if z_hint is not None and dot(z_axis, z_hint) < 0.0:
        x_axis = tuple(-value for value in x_axis)
        z_axis = tuple(-value for value in z_axis)

    rotation = (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )
    return matrix_to_quat(rotation)


def place_object_at_gripper_pose(mujoco, model, data, object_qposadr) -> None:
    site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        for site_name in GRIPPER_SITE_NAMES
    ]
    if min(site_ids) < 0:
        raise ValueError("MuJoCo model is missing gripper sites for object replay")

    for axis in range(3):
        data.qpos[object_qposadr + axis] = 0.5 * (
            data.site_xpos[site_ids[0]][axis] + data.site_xpos[site_ids[1]][axis]
        )
    data.qpos[object_qposadr + 3 : object_qposadr + 7] = object_quat_from_grippers(
        data,
        site_ids[0],
        site_ids[1],
    )
    mujoco.mj_forward(model, data)


def apply_configuration(
    mujoco,
    model,
    data,
    joint_names,
    addresses,
    configuration,
    base_body,
    object_qposadr,
) -> None:
    for address, value in zip(addresses, configuration):
        if address is None:
            continue
        data.qpos[address] = value
    if base_body is not None:
        body_id, origin = base_body
        base_x = configuration[joint_names.index("base_x")]
        base_y = configuration[joint_names.index("base_y")]
        base_yaw = configuration[joint_names.index("base_yaw")]
        model.body_pos[body_id][:] = (
            origin[0] + base_x,
            origin[1] + base_y,
            origin[2],
        )
        half_yaw = 0.5 * base_yaw
        model.body_quat[body_id][:] = (
            math.cos(half_yaw),
            0.0,
            0.0,
            math.sin(half_yaw),
        )
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if object_qposadr is not None:
        place_object_at_gripper_pose(mujoco, model, data, object_qposadr)


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
    base_body = resolve_mobility_base_body(mujoco, model, joint_names)
    object_qposadr = resolve_object_freejoint(mujoco, model, joint_names)
    validate_joint_limits(mujoco, model, joint_names, waypoints)
    if joint_names == SINGLE_ARM_PLANNING_JOINTS:
        left_arm_addresses = resolve_qpos_addresses(mujoco, model, LEFT_ARM_JOINTS)
        for address in left_arm_addresses:
            data.qpos[address] = 0.0
    apply_configuration(
        mujoco,
        model,
        data,
        joint_names,
        addresses,
        waypoints[0],
        base_body,
        object_qposadr,
    )

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
            apply_configuration(
                mujoco,
                model,
                data,
                joint_names,
                addresses,
                waypoints[0],
                base_body,
                object_qposadr,
            )
            viewer.sync()
            time.sleep(0.75)
            deadline = time.perf_counter()
            for configuration in interpolated_frames(waypoints, fps, speed):
                if not viewer.is_running():
                    return
                apply_configuration(
                    mujoco,
                    model,
                    data,
                    joint_names,
                    addresses,
                    configuration,
                    base_body,
                    object_qposadr,
                )
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
        data = mujoco.MjData(model)
        addresses = resolve_qpos_addresses(mujoco, model, joint_names)
        base_body = resolve_mobility_base_body(mujoco, model, joint_names)
        object_qposadr = resolve_object_freejoint(mujoco, model, joint_names)
        validate_joint_limits(mujoco, model, joint_names, waypoints)
        apply_configuration(
            mujoco,
            model,
            data,
            joint_names,
            addresses,
            waypoints[0],
            base_body,
            object_qposadr,
        )
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

#!/usr/bin/env python3
"""Replay a 15-DoF, mobile-base, or right-arm-only PATACON FFW-SG2 trajectory."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from mujoco_primitive_environment import (
    add_primitive_environment,
    normalize_primitive_environment,
    primitive_count,
)
from mujoco_video import (
    DEFAULT_TRAJECTORY_ACCELERATION,
    DEFAULT_TRAJECTORY_SPEED,
    GeometricPathWaypoints,
    MultiViewVideoWriter,
    add_video_arguments,
    configure_camera,
    configure_model_render_quality,
    continuous_trajectory_frames,
    validate_video_arguments,
    video_view_azimuth,
    video_view_elevation,
)


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
GRIPPER_JOINT_NAMES = tuple(
    f"gripper_{side}_joint{index}"
    for side in ("l", "r")
    for index in range(1, 5)
)
ATTACHED_OBJECT_FRAME_OFFSET = (0.1, 0.0, 0.0)
ACTUATOR_NAME_BY_JOINT = {}
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
FRANKA_FLOOR_RGBA = (0.48, 0.48, 0.48, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_TRAJECTORY_SPEED,
        help="Maximum configuration-coordinate change per second.",
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=DEFAULT_TRAJECTORY_ACCELERATION,
        help="Maximum configuration-coordinate acceleration per second squared.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("qpos", "ctrl"),
        default="qpos",
        help="Replay with direct qpos assignment or MuJoCo position-actuator targets.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=12,
        help="MuJoCo steps advanced after each ctrl target update.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate without opening a viewer window.",
    )
    add_video_arguments(parser, "PATACON_FFW_SG2_VIDEO")
    args = parser.parse_args()
    if (
        not math.isfinite(args.fps)
        or not math.isfinite(args.speed)
        or not math.isfinite(args.acceleration)
        or args.fps <= 0.0
        or args.speed <= 0.0
        or args.acceleration <= 0.0
    ):
        parser.error("--fps, --speed, and --acceleration must be positive")
    if args.settle_steps < 0:
        parser.error("--settle-steps must be greater than or equal to 0")
    validate_video_arguments(parser, args)
    return args


def parse_attached_object_frame_offset(document) -> tuple[float, float, float]:
    value = document.get("attached_object_frame_offset", ATTACHED_OBJECT_FRAME_OFFSET)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("attached_object_frame_offset must be [x, y, z]")
    offset = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in offset):
        raise ValueError("attached_object_frame_offset contains a non-finite value")
    return offset


def normalize_trajectory(document, joint_names, label):
    if not isinstance(document, dict):
        raise ValueError(f"{label} is not a JSON object")
    waypoints = document.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise ValueError(f"{label} must contain at least two waypoints")
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, list) or len(waypoint) != len(joint_names):
            raise ValueError(
                f"{label} waypoint {index} is not a "
                f"{len(joint_names)}-DoF configuration"
            )
        if not all(math.isfinite(float(value)) for value in waypoint):
            raise ValueError(f"{label} waypoint {index} contains a non-finite value")
    normalized = [[float(value) for value in waypoint] for waypoint in waypoints]

    expected_start = document.get("start")
    if not isinstance(expected_start, list) or len(expected_start) != len(joint_names):
        raise ValueError(
            f"{label} is missing its {len(joint_names)}-DoF start configuration"
        )
    if max(
        abs(normalized[0][index] - float(expected_start[index]))
        for index in range(len(joint_names))
    ) > 1.0e-5:
        raise ValueError(f"{label} was not converted to start-to-goal order")
    return (
        GeometricPathWaypoints(
            normalized,
            document.get("geometric_path"),
            document.get("path_smoothing", True),
        ),
        parse_attached_object_frame_offset(document),
    )


def load_trajectories(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("trajectory document must be a JSON object")
    joint_names = tuple(document.get("joint_names", ()))
    if joint_names not in SUPPORTED_JOINT_ORDERS:
        raise ValueError(
            "trajectory joint order does not match a supported FFW-SG2 planning order"
        )
    raw_trajectories = document.get("trajectories")
    if raw_trajectories is None:
        raw_trajectories = [document]
    if not isinstance(raw_trajectories, list) or not raw_trajectories:
        raise ValueError("trajectory bundle must contain at least one trajectory")
    trajectories = []
    environment = None
    for index, trajectory in enumerate(raw_trajectories):
        label = str(trajectory.get("label", f"trajectory {index + 1}"))
        waypoints, offset = normalize_trajectory(
            trajectory, joint_names, label
        )
        trajectory_environment = normalize_primitive_environment(
            trajectory.get("environment"), label
        )
        if environment is None:
            environment = trajectory_environment
        elif trajectory_environment != environment:
            raise ValueError(
                "all FFW-SG2 trajectories in one replay must use the same "
                "primitive environment"
            )
        trajectories.append((label, waypoints, offset))
    return joint_names, trajectories, environment


def build_model(mujoco, model_path: Path, environment: dict[str, list]):
    spec = mujoco.MjSpec.from_file(str(model_path))
    add_primitive_environment(mujoco, spec, environment, physical=True)
    return spec.compile()


def apply_visual_style(mujoco, model, joint_names: tuple[str, ...]) -> None:
    """Apply the common neutral color to the visualization floor."""
    del joint_names
    floor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )
    if floor_id < 0:
        raise ValueError("MuJoCo model is missing floor geom: floor")
    model.geom_matid[floor_id] = -1
    model.geom_rgba[floor_id] = FRANKA_FLOOR_RGBA


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


def resolve_ctrl_addresses(mujoco, model, joint_names) -> list[int | None]:
    addresses: list[int | None] = []
    for joint_name in joint_names:
        if joint_name in MOBILITY_BASE_JOINTS:
            addresses.append(None)
            continue
        actuator_name = ACTUATOR_NAME_BY_JOINT.get(joint_name, joint_name)
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        if actuator_id < 0:
            raise ValueError(f"MuJoCo model is missing actuator: {actuator_name}")
        addresses.append(int(actuator_id))
    return addresses


def close_grippers(mujoco, model, data) -> None:
    for joint_name in GRIPPER_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing gripper joint: {joint_name}")

        closed_value = float(model.jnt_range[joint_id][1])
        data.qpos[int(model.jnt_qposadr[joint_id])] = closed_value

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            joint_name,
        )
        if actuator_id >= 0:
            lower, upper = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = min(max(closed_value, lower), upper)


def validate_joint_limits(
    mujoco,
    model,
    joint_names,
    waypoints: list[list[float]],
    label: str = "trajectory",
) -> None:
    # The planner stores limits as float while MuJoCo reads XML values as
    # double. Allow only that representation gap; never alter the trajectory.
    tolerance = 1.0e-4
    for waypoint_index, waypoint in enumerate(waypoints):
        for name, value in zip(joint_names, waypoint):
            if name in MOBILITY_BASE_LIMITS:
                lower, upper = MOBILITY_BASE_LIMITS[name]
            else:
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                if not model.jnt_limited[joint_id]:
                    continue
                lower, upper = model.jnt_range[joint_id]
            if value < lower - tolerance or value > upper + tolerance:
                raise ValueError(
                    f"{label}: waypoint {waypoint_index}, {name}={value} is outside "
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


def object_axes_from_grippers(data, left_site: int, right_site: int):
    left_pos = data.site_xpos[left_site]
    right_pos = data.site_xpos[right_site]
    y_axis = normalize(
        tuple(float(left_pos[index] - right_pos[index]) for index in range(3))
    )
    if y_axis is None:
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)

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
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)

    left_y = site_matrix_column(data, left_site, 1)
    right_y = site_matrix_column(data, right_site, 1)
    z_hint = normalize(
        tuple(right_y[index] - left_y[index] for index in range(3))
    )
    if z_hint is not None and dot(z_axis, z_hint) < 0.0:
        x_axis = tuple(-value for value in x_axis)
        z_axis = tuple(-value for value in z_axis)

    return x_axis, y_axis, z_axis


def rotation_from_axes(x_axis, y_axis, z_axis):
    return (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )


def object_quat_from_grippers(data, left_site: int, right_site: int):
    x_axis, y_axis, z_axis = object_axes_from_grippers(data, left_site, right_site)
    return matrix_to_quat(rotation_from_axes(x_axis, y_axis, z_axis))


def place_object_at_gripper_pose(
    mujoco,
    model,
    data,
    object_qposadr,
    attached_object_frame_offset,
) -> None:
    site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        for site_name in GRIPPER_SITE_NAMES
    ]
    if min(site_ids) < 0:
        raise ValueError("MuJoCo model is missing gripper sites for object replay")

    x_axis, y_axis, z_axis = object_axes_from_grippers(data, site_ids[0], site_ids[1])
    frame_offset = attached_object_frame_offset
    world_offset = tuple(
        x_axis[axis] * frame_offset[0]
        + y_axis[axis] * frame_offset[1]
        + z_axis[axis] * frame_offset[2]
        for axis in range(3)
    )

    for axis in range(3):
        center = 0.5 * (
            data.site_xpos[site_ids[0]][axis] + data.site_xpos[site_ids[1]][axis]
        )
        data.qpos[object_qposadr + axis] = center + world_offset[axis]
    data.qpos[object_qposadr + 3 : object_qposadr + 7] = matrix_to_quat(
        rotation_from_axes(x_axis, y_axis, z_axis)
    )
    mujoco.mj_forward(model, data)


def apply_configuration(
    mujoco,
    model,
    data,
    joint_names,
    qpos_addresses,
    ctrl_addresses,
    configuration,
    base_body,
    object_qposadr,
    attached_object_frame_offset,
    input_mode,
    settle_steps,
    seed_qpos=False,
) -> None:
    for index, (qpos_address, value) in enumerate(
        zip(qpos_addresses, configuration)
    ):
        if qpos_address is None:
            continue
        if input_mode == "ctrl":
            data.ctrl[ctrl_addresses[index]] = value
            if seed_qpos:
                data.qpos[qpos_address] = value
        else:
            data.qpos[qpos_address] = value
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
    close_grippers(mujoco, model, data)
    if input_mode == "ctrl" and not seed_qpos:
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        if settle_steps == 0:
            mujoco.mj_forward(model, data)
    else:
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    if object_qposadr is not None:
        place_object_at_gripper_pose(
            mujoco,
            model,
            data,
            object_qposadr,
            attached_object_frame_offset,
        )


def interpolated_frames(
    waypoints,
    fps: float,
    speed: float,
    acceleration: float = DEFAULT_TRAJECTORY_ACCELERATION,
):
    yield from continuous_trajectory_frames(
        waypoints, fps, speed, acceleration
    )


def configure_replay_camera(
    mujoco, camera, joint_names, view: str = "front"
) -> None:
    if tuple(joint_names) == MOBILITY_PLANNING_JOINTS:
        configure_camera(
            mujoco,
            camera,
            (0.48, 0.0, 0.9),
            3.4,
            video_view_azimuth(315.0, view),
            video_view_elevation(-18.0, view),
        )
    else:
        configure_camera(
            mujoco,
            camera,
            (0.3, 0.0, 1.0),
            3.0,
            video_view_azimuth(45.0, view),
            video_view_elevation(-15.0, view),
        )


def save_video(
    model_path: Path,
    joint_names,
    waypoints,
    attached_object_frame_offset,
    environment,
    fps: float,
    speed: float,
    input_mode: str,
    settle_steps: int,
    output_path: Path,
    width: int,
    height: int,
    views: tuple[str, ...],
    acceleration: float = DEFAULT_TRAJECTORY_ACCELERATION,
) -> None:
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo is required for MP4 rendering: python3 -m pip install mujoco"
        ) from error

    model = build_model(mujoco, model_path, environment)
    apply_visual_style(mujoco, model, joint_names)
    configure_model_render_quality(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    qpos_addresses = resolve_qpos_addresses(mujoco, model, joint_names)
    ctrl_addresses = (
        resolve_ctrl_addresses(mujoco, model, joint_names)
        if input_mode == "ctrl"
        else []
    )
    base_body = resolve_mobility_base_body(mujoco, model, joint_names)
    object_qposadr = resolve_object_freejoint(mujoco, model, joint_names)
    validate_joint_limits(mujoco, model, joint_names, waypoints)
    if joint_names == SINGLE_ARM_PLANNING_JOINTS:
        left_arm_addresses = resolve_qpos_addresses(
            mujoco, model, LEFT_ARM_JOINTS
        )
        for address in left_arm_addresses:
            data.qpos[address] = 0.0
    apply_configuration(
        mujoco,
        model,
        data,
        joint_names,
        qpos_addresses,
        ctrl_addresses,
        waypoints[0],
        base_body,
        object_qposadr,
        attached_object_frame_offset,
        input_mode,
        settle_steps,
        seed_qpos=input_mode == "ctrl",
    )

    cameras = {}
    for view in views:
        camera = mujoco.MjvCamera()
        configure_replay_camera(mujoco, camera, joint_names, view)
        cameras[view] = camera
    renderer = mujoco.Renderer(model, height=height, width=width)
    output_path = output_path.expanduser().resolve()
    start_frame_count = max(1, math.ceil(0.75 * fps))
    end_frame_count = max(1, math.ceil(1.0 * fps))

    def write_frame(writer: MultiViewVideoWriter) -> None:
        for view, camera in cameras.items():
            renderer.update_scene(data, camera=camera)
            writer.write(view, renderer.render())

    print(
        f"rendering {len(views)} FFW-SG2 MP4 view(s) at {fps:g} fps "
        f"({width}x{height}, {input_mode} mode): {', '.join(views)}"
    )
    try:
        with MultiViewVideoWriter(
            output_path, views, width, height, fps
        ) as writer:
            for _ in range(start_frame_count):
                write_frame(writer)
            for configuration in interpolated_frames(
                waypoints, fps, speed, acceleration
            ):
                apply_configuration(
                    mujoco,
                    model,
                    data,
                    joint_names,
                    qpos_addresses,
                    ctrl_addresses,
                    configuration,
                    base_body,
                    object_qposadr,
                    attached_object_frame_offset,
                    input_mode,
                    settle_steps,
                )
                write_frame(writer)
            for _ in range(end_frame_count):
                write_frame(writer)
            frame_count = writer.frame_count
    finally:
        renderer.close()

    print(
        "saved FFW-SG2 MP4: "
        + ", ".join(str(path) for path in writer.output_paths.values())
        + f" ({frame_count} frames each, {frame_count / fps:.3f} s)"
    )


def replay(
    model_path: Path,
    joint_names,
    trajectories,
    environment,
    fps: float,
    speed: float,
    input_mode: str,
    settle_steps: int,
    acceleration: float = DEFAULT_TRAJECTORY_ACCELERATION,
) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo Python package is required: python3 -m pip install mujoco"
        ) from error

    model = build_model(mujoco, model_path, environment)
    apply_visual_style(mujoco, model, joint_names)
    data = mujoco.MjData(model)
    qpos_addresses = resolve_qpos_addresses(mujoco, model, joint_names)
    ctrl_addresses = (
        resolve_ctrl_addresses(mujoco, model, joint_names)
        if input_mode == "ctrl"
        else []
    )
    base_body = resolve_mobility_base_body(mujoco, model, joint_names)
    object_qposadr = resolve_object_freejoint(mujoco, model, joint_names)
    for label, waypoints, _ in trajectories:
        validate_joint_limits(mujoco, model, joint_names, waypoints, label)
    if joint_names == SINGLE_ARM_PLANNING_JOINTS:
        left_arm_addresses = resolve_qpos_addresses(mujoco, model, LEFT_ARM_JOINTS)
        for address in left_arm_addresses:
            data.qpos[address] = 0.0
    print(
        f"MuJoCo viewer: start -> goal 경로 {len(trajectories)}개를 "
        "순서대로 반복 재생합니다."
    )
    print("창을 닫으면 MuJoCo 시각화가 종료됩니다.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_replay_camera(mujoco, viewer.cam, joint_names)
        viewer.sync()

        frame_period = 1.0 / fps
        while viewer.is_running():
            for trajectory_index, (label, waypoints, offset) in enumerate(
                trajectories
            ):
                if not viewer.is_running():
                    return
                print(
                    f"[{trajectory_index + 1}/{len(trajectories)}] {label}",
                    flush=True,
                )
                mujoco.mj_resetData(model, data)
                apply_configuration(
                    mujoco,
                    model,
                    data,
                    joint_names,
                    qpos_addresses,
                    ctrl_addresses,
                    waypoints[0],
                    base_body,
                    object_qposadr,
                    offset,
                    input_mode,
                    settle_steps,
                    seed_qpos=input_mode == "ctrl",
                )
                viewer.sync()
                time.sleep(0.75)
                deadline = time.perf_counter()
                for configuration in interpolated_frames(
                    waypoints, fps, speed, acceleration
                ):
                    if not viewer.is_running():
                        return
                    apply_configuration(
                        mujoco,
                        model,
                        data,
                        joint_names,
                        qpos_addresses,
                        ctrl_addresses,
                        configuration,
                        base_body,
                        object_qposadr,
                        offset,
                        input_mode,
                        settle_steps,
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

    joint_names, trajectories, environment = load_trajectories(trajectory_path)
    validate_only = args.validate_only or os.environ.get(
        "PATACON_MUJOCO_VALIDATE_ONLY"
    ) == "1"
    if validate_only:
        import mujoco

        model = build_model(mujoco, model_path, environment)
        apply_visual_style(mujoco, model, joint_names)
        data = mujoco.MjData(model)
        qpos_addresses = resolve_qpos_addresses(mujoco, model, joint_names)
        ctrl_addresses = (
            resolve_ctrl_addresses(mujoco, model, joint_names)
            if args.input_mode == "ctrl"
            else []
        )
        base_body = resolve_mobility_base_body(mujoco, model, joint_names)
        object_qposadr = resolve_object_freejoint(mujoco, model, joint_names)
        waypoint_count = 0
        for label, waypoints, offset in trajectories:
            mujoco.mj_resetData(model, data)
            validate_joint_limits(
                mujoco, model, joint_names, waypoints, label
            )
            apply_configuration(
                mujoco,
                model,
                data,
                joint_names,
                qpos_addresses,
                ctrl_addresses,
                waypoints[0],
                base_body,
                object_qposadr,
                offset,
                args.input_mode,
                args.settle_steps,
                seed_qpos=args.input_mode == "ctrl",
            )
            waypoint_count += len(waypoints)
        print(
            f"validated {len(trajectories)} trajectories, "
            f"{waypoint_count} waypoints, {len(joint_names)} joints, "
            f"{primitive_count(environment)} environment primitives, "
            f"model nq={model.nq}"
        )
        return 0

    if args.video is not None:
        if len(trajectories) != 1:
            raise ValueError("FFW-SG2 bundle MP4 rendering is not supported")
        _, waypoints, attached_object_frame_offset = trajectories[0]
        save_video(
            model_path,
            joint_names,
            waypoints,
            attached_object_frame_offset,
            environment,
            args.fps,
            args.speed,
            args.input_mode,
            args.settle_steps,
            args.video,
            args.video_width,
            args.video_height,
            args.video_views,
            args.acceleration,
        )
        return 0

    replay(
        model_path,
        joint_names,
        trajectories,
        environment,
        args.fps,
        args.speed,
        args.input_mode,
        args.settle_steps,
        args.acceleration,
    )
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

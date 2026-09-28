#!/usr/bin/env python3
"""Validate or replay a Franka single/dual cpRRTC trajectory in MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

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


SINGLE_JOINTS = tuple(f"panda0_joint{index}" for index in range(1, 8))
DUAL_JOINTS = SINGLE_JOINTS + tuple(
    f"panda1_joint{index}" for index in range(1, 8)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--speed", type=float, default=DEFAULT_TRAJECTORY_SPEED
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=DEFAULT_TRAJECTORY_ACCELERATION,
        help="Maximum configuration-coordinate acceleration per second squared.",
    )
    parser.add_argument("--validate-only", action="store_true")
    add_video_arguments(parser, "PRRTC_FRANKA_VIDEO")
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
    validate_video_arguments(parser, args)
    return args


def load_trajectory(path: Path) -> tuple[tuple[str, ...], list[list[float]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    joint_names = tuple(document.get("joint_names", ()))
    if joint_names not in (SINGLE_JOINTS, DUAL_JOINTS):
        raise ValueError("trajectory joint order is not Franka single or dual")
    waypoints = document.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise ValueError("trajectory must contain at least two waypoints")
    normalized: list[list[float]] = []
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, list) or len(waypoint) != len(joint_names):
            raise ValueError(
                f"waypoint {index} is not a {len(joint_names)}-DoF configuration"
            )
        row = [float(value) for value in waypoint]
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"waypoint {index} contains a non-finite value")
        normalized.append(row)
    expected_start = document.get("start")
    if not isinstance(expected_start, list) or len(expected_start) != len(joint_names):
        raise ValueError("trajectory is missing its start configuration")
    if max(
        abs(normalized[0][index] - float(expected_start[index]))
        for index in range(len(joint_names))
    ) > 1.0e-5:
        raise ValueError("trajectory was not converted to start-to-goal order")
    return joint_names, GeometricPathWaypoints(
        normalized,
        document.get("geometric_path"),
        document.get("path_smoothing", True),
    )


def resolve_addresses(mujoco, model, joint_names: tuple[str, ...]) -> list[int]:
    addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing planning joint: {name}")
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"planning joint is not a hinge: {name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def validate_limits(mujoco, model, joint_names, waypoints) -> None:
    for waypoint_index, waypoint in enumerate(waypoints):
        for name, value in zip(joint_names, waypoint):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lower, upper = model.jnt_range[joint_id]
            if value < lower - 1.0e-6 or value > upper + 1.0e-6:
                raise ValueError(
                    f"waypoint {waypoint_index}, {name}={value} is outside "
                    f"[{lower}, {upper}]"
                )


def set_grippers(mujoco, model, data, dual: bool) -> None:
    opening = 0.0 if dual else 0.015
    arm_indices = (0, 1) if dual else (0,)
    for arm in arm_indices:
        for finger in (1, 2):
            name = f"panda{arm}_finger_joint{finger}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                data.qpos[int(model.jnt_qposadr[joint_id])] = opening


def apply_configuration(data, addresses, configuration) -> None:
    for address, value in zip(addresses, configuration):
        data.qpos[address] = value


def interpolated_frames(
    waypoints,
    fps: float,
    speed: float,
    acceleration: float = DEFAULT_TRAJECTORY_ACCELERATION,
):
    yield from continuous_trajectory_frames(
        waypoints, fps, speed, acceleration
    )


def resolve_attached_object(
    mujoco, model, data, joint_names, addresses, start
) -> dict[str, object]:
    object_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "object"
    )
    if object_id < 0:
        raise ValueError("MuJoCo model is missing attached body: object")
    if int(model.body_jntnum[object_id]) != 1:
        raise ValueError("attached object must have exactly one free joint")
    joint_id = int(model.body_jntadr[object_id])
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("attached object joint is not free")
    qpos_address = int(model.jnt_qposadr[joint_id])

    # Both source models name the left/single attachment site end_effector;
    # the dual model names only the right site end_effector1.
    site_name = "end_effector"
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, site_name
    )
    if site_id < 0:
        raise ValueError(f"MuJoCo model is missing attachment site: {site_name}")

    apply_configuration(data, addresses, start)
    mujoco.mj_forward(model, data)
    ee_position = np.asarray(data.site_xpos[site_id]).copy()
    ee_rotation = np.asarray(data.site_xmat[site_id]).reshape(3, 3).copy()
    object_position = np.asarray(data.xpos[object_id]).copy()
    object_rotation = np.asarray(data.xmat[object_id]).reshape(3, 3).copy()
    return {
        "site_id": site_id,
        "qpos_address": qpos_address,
        "relative_position": ee_rotation.T @ (object_position - ee_position),
        "relative_rotation": ee_rotation.T @ object_rotation,
    }


def apply_attached_configuration(
    mujoco, model, data, addresses, configuration, attachment
) -> None:
    apply_configuration(data, addresses, configuration)
    mujoco.mj_forward(model, data)
    site_id = int(attachment["site_id"])
    ee_position = np.asarray(data.site_xpos[site_id])
    ee_rotation = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
    object_position = (
        ee_position + ee_rotation @ attachment["relative_position"]
    )
    object_rotation = ee_rotation @ attachment["relative_rotation"]
    object_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(object_quaternion, object_rotation.reshape(9))
    address = int(attachment["qpos_address"])
    data.qpos[address : address + 3] = object_position
    data.qpos[address + 3 : address + 7] = object_quaternion
    mujoco.mj_forward(model, data)


def configure_replay_camera(
    mujoco,
    camera,
    joint_names,
    view: str = "front",
) -> None:
    base_azimuth = 180.0 if tuple(joint_names) == DUAL_JOINTS else 0.0
    configure_camera(
        mujoco,
        camera,
        (0.15, 0.0, 0.75),
        2.5,
        video_view_azimuth(base_azimuth, view),
        video_view_elevation(-18.0, view),
    )


def save_video(
    model_path: Path,
    joint_names,
    waypoints,
    fps: float,
    speed: float,
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

    model = mujoco.MjModel.from_xml_path(str(model_path))
    configure_model_render_quality(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    addresses = resolve_addresses(mujoco, model, joint_names)
    validate_limits(mujoco, model, joint_names, waypoints)
    set_grippers(mujoco, model, data, joint_names == DUAL_JOINTS)
    attachment = resolve_attached_object(
        mujoco, model, data, joint_names, addresses, waypoints[0]
    )
    apply_attached_configuration(
        mujoco, model, data, addresses, waypoints[0], attachment
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
        f"rendering {len(views)} Franka MP4 view(s) at {fps:g} fps "
        f"({width}x{height}): {', '.join(views)}"
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
                apply_attached_configuration(
                    mujoco,
                    model,
                    data,
                    addresses,
                    configuration,
                    attachment,
                )
                write_frame(writer)
            for _ in range(end_frame_count):
                write_frame(writer)
            frame_count = writer.frame_count
    finally:
        renderer.close()

    print(
        "saved Franka MP4: "
        + ", ".join(str(path) for path in writer.output_paths.values())
        + f" ({frame_count} frames each, {frame_count / fps:.3f} s)"
    )


def replay(
    model_path,
    joint_names,
    waypoints,
    fps,
    speed,
    acceleration=DEFAULT_TRAJECTORY_ACCELERATION,
) -> None:
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = resolve_addresses(mujoco, model, joint_names)
    validate_limits(mujoco, model, joint_names, waypoints)
    set_grippers(mujoco, model, data, joint_names == DUAL_JOINTS)
    attachment = resolve_attached_object(
        mujoco, model, data, joint_names, addresses, waypoints[0]
    )
    frame_period = 1.0 / fps
    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_replay_camera(mujoco, viewer.cam, joint_names)
        apply_attached_configuration(
            mujoco, model, data, addresses, waypoints[0], attachment
        )
        viewer.sync()
        for configuration in interpolated_frames(
            waypoints, fps, speed, acceleration
        ):
            if not viewer.is_running():
                return
            apply_attached_configuration(
                mujoco, model, data, addresses, configuration, attachment
            )
            viewer.sync()
            time.sleep(frame_period)
        while viewer.is_running():
            viewer.sync()
            time.sleep(frame_period)


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    joint_names, waypoints = load_trajectory(args.trajectory)
    if args.validate_only:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        addresses = resolve_addresses(mujoco, model, joint_names)
        validate_limits(mujoco, model, joint_names, waypoints)
        set_grippers(mujoco, model, data, joint_names == DUAL_JOINTS)
        attachment = resolve_attached_object(
            mujoco, model, data, joint_names, addresses, waypoints[0]
        )
        apply_attached_configuration(
            mujoco, model, data, addresses, waypoints[0], attachment
        )
        print(
            f"validated {len(waypoints)} waypoints, {len(joint_names)} joints, "
            f"model nq={model.nq}"
        )
        return 0
    if args.video is not None:
        save_video(
            model_path,
            joint_names,
            waypoints,
            args.fps,
            args.speed,
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
        waypoints,
        args.fps,
        args.speed,
        args.acceleration,
    )
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:
        print(f"visualization error: {error}", file=sys.stderr)
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

#!/usr/bin/env python3
"""Validate or replay a Franka single/dual PATACON trajectory in MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

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


SINGLE_JOINTS = tuple(f"fer0_joint{index}" for index in range(1, 8))
DUAL_JOINTS = SINGLE_JOINTS + tuple(
    f"fer1_joint{index}" for index in range(1, 8)
)

# These are the fixed T_EE_object transforms used by
# src/robots/franka_collision.cuh. They must not be reconstructed from the
# object's default XML pose because random problem starts have different
# world poses.
SINGLE_ATTACHED_ROTATION = (
    5.65184217e-06,
    4.04983458e-06,
    -1.0,
    7.46365351e-06,
    1.0,
    4.04987676e-06,
    1.0,
    -7.46367640e-06,
    5.65181194e-06,
)
SINGLE_ATTACHED_TRANSLATION = (
    -0.0299951539,
    -3.27119653e-06,
    -0.0234040056,
)
DUAL_ATTACHED_ROTATION = (
    1.0,
    -1.12881255e-05,
    4.01251945e-06,
    -1.12881331e-05,
    -1.0,
    1.89331703e-06,
    4.01249807e-06,
    -1.89336233e-06,
    -1.0,
)
DUAL_ATTACHED_TRANSLATION = (
    -7.26699543e-08,
    0.124998270,
    0.0465992695,
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
    add_video_arguments(parser, "PATACON_FRANKA_VIDEO")
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


def load_trajectories(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("trajectory document must be a JSON object")
    joint_names = tuple(document.get("joint_names", ()))
    if joint_names not in (SINGLE_JOINTS, DUAL_JOINTS):
        raise ValueError("trajectory joint order is not Franka single or dual")

    raw_trajectories = document.get("trajectories")
    if raw_trajectories is None:
        raw_trajectories = [document]
    if not isinstance(raw_trajectories, list) or not raw_trajectories:
        raise ValueError("trajectory bundle must contain at least one trajectory")

    trajectories = []
    environment = None
    for trajectory_index, trajectory in enumerate(raw_trajectories):
        if not isinstance(trajectory, dict):
            raise ValueError(f"trajectory {trajectory_index} is not a JSON object")
        label = str(
            trajectory.get("label", f"trajectory {trajectory_index + 1}")
        )
        waypoints = trajectory.get("waypoints")
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            raise ValueError(f"{label} must contain at least two waypoints")
        normalized: list[list[float]] = []
        for waypoint_index, waypoint in enumerate(waypoints):
            if not isinstance(waypoint, list) or len(waypoint) != len(joint_names):
                raise ValueError(
                    f"{label} waypoint {waypoint_index} is not a "
                    f"{len(joint_names)}-DoF configuration"
                )
            row = [float(value) for value in waypoint]
            if not all(math.isfinite(value) for value in row):
                raise ValueError(
                    f"{label} waypoint {waypoint_index} contains a non-finite value"
                )
            normalized.append(row)
        expected_start = trajectory.get("start")
        if (
            not isinstance(expected_start, list)
            or len(expected_start) != len(joint_names)
        ):
            raise ValueError(f"{label} is missing its start configuration")
        if max(
            abs(normalized[0][index] - float(expected_start[index]))
            for index in range(len(joint_names))
        ) > 1.0e-5:
            raise ValueError(f"{label} was not converted to start-to-goal order")
        trajectory_environment = normalize_primitive_environment(
            trajectory.get("environment"), label
        )
        if environment is None:
            environment = trajectory_environment
        elif trajectory_environment != environment:
            raise ValueError(
                "all Franka trajectories in one replay must use the same "
                "primitive environment"
            )
        trajectories.append((
            label,
            GeometricPathWaypoints(
                normalized,
                trajectory.get("geometric_path"),
                trajectory.get("path_smoothing", True),
            ),
        ))
    return joint_names, trajectories, environment


def build_model(mujoco, model_path: Path, environment: dict[str, list]):
    spec = mujoco.MjSpec.from_file(str(model_path))
    add_primitive_environment(mujoco, spec, environment, physical=False)
    return spec.compile()


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
            name = f"fer{arm}_finger_joint{finger}"
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


def resolve_attached_object(mujoco, model, joint_names) -> dict[str, object]:
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

    # Both generated models name the left/single attachment site end_effector.
    # The dual model additionally names its right attachment site end_effector1;
    # the planner parents the payload to the left site.
    site_name = "end_effector"
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, site_name
    )
    if site_id < 0:
        raise ValueError(f"MuJoCo model is missing attachment site: {site_name}")

    if joint_names == SINGLE_JOINTS:
        relative_position = np.asarray(
            SINGLE_ATTACHED_TRANSLATION, dtype=np.float64
        )
        relative_rotation = np.asarray(
            SINGLE_ATTACHED_ROTATION, dtype=np.float64
        ).reshape(3, 3)
    else:
        relative_position = np.asarray(
            DUAL_ATTACHED_TRANSLATION, dtype=np.float64
        )
        relative_rotation = np.asarray(
            DUAL_ATTACHED_ROTATION, dtype=np.float64
        ).reshape(3, 3)
    return {
        "site_id": site_id,
        "qpos_address": qpos_address,
        "relative_position": relative_position,
        "relative_rotation": relative_rotation,
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
    trajectories,
    environment,
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

    model = build_model(mujoco, model_path, environment)
    configure_model_render_quality(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    addresses = resolve_addresses(mujoco, model, joint_names)
    for _, waypoints in trajectories:
        validate_limits(mujoco, model, joint_names, waypoints)
    attachment = resolve_attached_object(mujoco, model, joint_names)

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
            for trajectory_index, (label, waypoints) in enumerate(trajectories):
                print(
                    f"[{trajectory_index + 1}/{len(trajectories)}] {label}",
                    flush=True,
                )
                mujoco.mj_resetData(model, data)
                set_grippers(mujoco, model, data, joint_names == DUAL_JOINTS)
                apply_attached_configuration(
                    mujoco, model, data, addresses, waypoints[0], attachment
                )
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
    trajectories,
    environment,
    fps,
    speed,
    acceleration=DEFAULT_TRAJECTORY_ACCELERATION,
) -> None:
    import mujoco
    import mujoco.viewer

    model = build_model(mujoco, model_path, environment)
    data = mujoco.MjData(model)
    addresses = resolve_addresses(mujoco, model, joint_names)
    for _, waypoints in trajectories:
        validate_limits(mujoco, model, joint_names, waypoints)
    attachment = resolve_attached_object(mujoco, model, joint_names)
    frame_period = 1.0 / fps
    print(
        f"MuJoCo viewer: Franka start -> goal 경로 "
        f"{len(trajectories)}개를 순서대로 반복 재생합니다."
    )
    print("창을 닫으면 MuJoCo 시각화가 종료됩니다.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_replay_camera(mujoco, viewer.cam, joint_names)
        while viewer.is_running():
            for trajectory_index, (label, waypoints) in enumerate(trajectories):
                if not viewer.is_running():
                    return
                print(
                    f"[{trajectory_index + 1}/{len(trajectories)}] {label}",
                    flush=True,
                )
                mujoco.mj_resetData(model, data)
                set_grippers(mujoco, model, data, joint_names == DUAL_JOINTS)
                apply_attached_configuration(
                    mujoco, model, data, addresses, waypoints[0], attachment
                )
                viewer.sync()
                time.sleep(0.75)
                deadline = time.perf_counter()
                for configuration in interpolated_frames(
                    waypoints, fps, speed, acceleration
                ):
                    if not viewer.is_running():
                        return
                    apply_attached_configuration(
                        mujoco,
                        model,
                        data,
                        addresses,
                        configuration,
                        attachment,
                    )
                    viewer.sync()
                    deadline += frame_period
                    time.sleep(max(0.0, deadline - time.perf_counter()))
                time.sleep(1.0)


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()
    joint_names, trajectories, environment = load_trajectories(args.trajectory)
    if args.validate_only:
        import mujoco

        model = build_model(mujoco, model_path, environment)
        data = mujoco.MjData(model)
        addresses = resolve_addresses(mujoco, model, joint_names)
        attachment = resolve_attached_object(mujoco, model, joint_names)
        waypoint_count = 0
        for _, waypoints in trajectories:
            mujoco.mj_resetData(model, data)
            validate_limits(mujoco, model, joint_names, waypoints)
            set_grippers(mujoco, model, data, joint_names == DUAL_JOINTS)
            apply_attached_configuration(
                mujoco, model, data, addresses, waypoints[0], attachment
            )
            waypoint_count += len(waypoints)
        print(
            f"validated {len(trajectories)} trajectories, "
            f"{waypoint_count} waypoints, {len(joint_names)} joints, "
            f"{primitive_count(environment)} "
            f"environment primitives, model nq={model.nq}"
        )
        return 0
    if args.video is not None:
        save_video(
            model_path,
            joint_names,
            trajectories,
            environment,
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
        trajectories,
        environment,
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

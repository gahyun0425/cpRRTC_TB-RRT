#!/usr/bin/env python3
"""Replay a 35-DoF pRRTC G1 trajectory with its planning obstacles."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


G1_ACTUATED_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_CONFIGURATION_DIMENSION = 35
G1_BASE_ACTUATORS = (
    "planner_base_x",
    "planner_base_y",
    "planner_base_z",
    "planner_base_roll",
    "planner_base_pitch",
    "planner_base_yaw",
)

# The planner's foot end-effector frames are constrained to z=0, while the
# MuJoCo sole contact points extend about 35 mm below those frames.
G1_PLANNING_FLOOR_HEIGHT = -0.0351

# Torque-PD gains in G1_ACTUATED_JOINTS order.  The native XML actuators are
# motors, so their ctrl values are torques rather than target angles.
G1_JOINT_KP = np.asarray(
    [
        50, 50, 30, 60, 20, 15,
        50, 50, 30, 60, 20, 15,
        30, 25, 30,
        20, 20, 15, 20, 5, 3, 3,
        20, 20, 15, 20, 5, 3, 3,
    ],
    dtype=np.float64,
)
G1_JOINT_KD = np.asarray(
    [
        5, 5, 3, 6, 2, 1,
        5, 5, 3, 6, 2, 1,
        3, 2, 3,
        2, 2, 1.5, 2, 0.5, 0.3, 0.3,
        2, 2, 1.5, 2, 0.5, 0.3, 0.3,
    ],
    dtype=np.float64,
)
G1_BASE_KP = np.asarray([500, 500, 800, 200, 200, 200], dtype=np.float64)
G1_BASE_KD = np.asarray([100, 100, 120, 50, 50, 50], dtype=np.float64)


def default_model_path() -> Path:
    vamp_root = os.environ.get("VAMP_ROOT")
    if vamp_root:
        root = Path(vamp_root).expanduser()
    else:
        root = Path.home() / "gh_ws" / "vamp"
    return (
        root
        / "third_party"
        / "unitree_ros"
        / "robots"
        / "g1_description"
        / "g1_29dof.xml"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.7,
        help="Maximum planning-coordinate change per second.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("ctrl", "qpos"),
        default=os.environ.get("PRRTC_G1_CONTROL_MODE", "ctrl"),
        help=(
            "ctrl uses torque-PD actuators and physics (default); qpos keeps "
            "the legacy kinematic replay"
        ),
    )
    parser.add_argument(
        "--gain-scale",
        type=float,
        default=1.0,
        help="Scale all ctrl-mode proportional and derivative gains.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate without opening a viewer window.",
    )
    args = parser.parse_args()
    if args.control_mode not in ("ctrl", "qpos"):
        parser.error("--control-mode must be ctrl or qpos")
    if args.fps <= 0.0 or args.speed <= 0.0 or args.gain_scale <= 0.0:
        parser.error("--fps, --speed, and --gain-scale must be positive")
    return args


def finite_vector(value, dimension: int, description: str) -> list[float]:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{description} must contain {dimension} values")
    normalized = [float(component) for component in value]
    if not all(math.isfinite(component) for component in normalized):
        raise ValueError(f"{description} contains a non-finite value")
    return normalized


def load_trajectory(path: Path) -> tuple[list[list[float]], dict[str, list]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    waypoints_value = document.get("waypoints")
    if not isinstance(waypoints_value, list) or len(waypoints_value) < 2:
        raise ValueError("trajectory must contain at least two waypoints")
    waypoints = [
        finite_vector(
            waypoint,
            G1_CONFIGURATION_DIMENSION,
            f"waypoint {index}",
        )
        for index, waypoint in enumerate(waypoints_value)
    ]
    expected_start = finite_vector(
        document.get("start"),
        G1_CONFIGURATION_DIMENSION,
        "trajectory start",
    )
    if max(
        abs(waypoints[0][index] - expected_start[index])
        for index in range(G1_CONFIGURATION_DIMENSION)
    ) > 1.0e-5:
        raise ValueError("trajectory was not converted to start-to-goal order")

    environment = document.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("environment must be a JSON object")
    for primitive in ("sphere", "cylinder", "box"):
        values = environment.get(primitive, [])
        if not isinstance(values, list):
            raise ValueError(f"environment.{primitive} must be a list")
        environment[primitive] = values
    return waypoints, environment


def resolve_model_layout(mujoco, model) -> tuple[int, list[int]]:
    base_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "floating_base_joint",
    )
    if base_id < 0:
        raise ValueError("MuJoCo model is missing floating_base_joint")
    if int(model.jnt_type[base_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("floating_base_joint is not a free joint")

    addresses: list[int] = []
    for name in G1_ACTUATED_JOINTS:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing G1 joint: {name}")
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"G1 joint is not a hinge: {name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return int(model.jnt_qposadr[base_id]), addresses


def roll_pitch_yaw_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def apply_configuration(
    mujoco,
    model,
    data,
    base_address: int,
    joint_addresses: list[int],
    configuration: list[float],
) -> None:
    data.qpos[base_address : base_address + 3] = configuration[0:3]
    data.qpos[base_address + 3 : base_address + 7] = roll_pitch_yaw_quaternion(
        configuration[3],
        configuration[4],
        configuration[5],
    )
    for address, value in zip(joint_addresses, configuration[6:]):
        data.qpos[address] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def validate_joint_limits(mujoco, model, waypoints: list[list[float]]) -> None:
    tolerance = 1.0e-5
    for waypoint_index, waypoint in enumerate(waypoints):
        for name, value in zip(G1_ACTUATED_JOINTS, waypoint[6:]):
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            if not model.jnt_limited[joint_id]:
                continue
            lower, upper = model.jnt_range[joint_id]
            if value < lower - tolerance or value > upper + tolerance:
                raise ValueError(
                    f"waypoint {waypoint_index}, {name}={value} is outside "
                    f"[{lower}, {upper}]"
                )


def euler_xyz_matrix(angles: list[float]) -> np.ndarray:
    roll, pitch, yaw = angles
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ],
        dtype=np.float64,
    )


def validate_environment(environment: dict[str, list]) -> None:
    for index, sphere in enumerate(environment["sphere"]):
        finite_vector(sphere.get("position"), 3, f"sphere {index} position")
        radius = float(sphere.get("radius"))
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"sphere {index} radius must be positive")
    for index, cylinder in enumerate(environment["cylinder"]):
        finite_vector(cylinder.get("position"), 3, f"cylinder {index} position")
        finite_vector(
            cylinder.get("orientation_euler_xyz"),
            3,
            f"cylinder {index} orientation",
        )
        radius = float(cylinder.get("radius"))
        length = float(cylinder.get("length"))
        if not all(math.isfinite(value) and value > 0.0 for value in (radius, length)):
            raise ValueError(f"cylinder {index} dimensions must be positive")
    for index, box in enumerate(environment["box"]):
        finite_vector(box.get("position"), 3, f"box {index} position")
        finite_vector(
            box.get("orientation_euler_xyz"),
            3,
            f"box {index} orientation",
        )
        half_extents = finite_vector(
            box.get("half_extents"),
            3,
            f"box {index} half extents",
        )
        if any(value <= 0.0 for value in half_extents):
            raise ValueError(f"box {index} half extents must be positive")


def add_physical_environment(mujoco, spec, environment: dict[str, list]) -> None:
    for index, sphere in enumerate(environment["sphere"]):
        spec.worldbody.add_geom(
            name=f"planning_sphere_{index}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=sphere["position"],
            size=[float(sphere["radius"]), 0.0, 0.0],
            contype=0,
            conaffinity=1,
            density=0.0,
            rgba=[0.85, 0.25, 0.20, 0.65],
        )

    for index, cylinder in enumerate(environment["cylinder"]):
        spec.worldbody.add_geom(
            name=f"planning_cylinder_{index}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=cylinder["position"],
            euler=cylinder["orientation_euler_xyz"],
            size=[
                float(cylinder["radius"]),
                0.5 * float(cylinder["length"]),
                0.0,
            ],
            contype=0,
            conaffinity=1,
            density=0.0,
            rgba=[0.85, 0.45, 0.15, 0.65],
        )

    for index, box in enumerate(environment["box"]):
        is_obstacle = box.get("name") == "obstacle"
        color = (
            [0.85, 0.20, 0.15, 0.65]
            if is_obstacle
            else [0.45, 0.32, 0.18, 0.65]
        )
        spec.worldbody.add_geom(
            name=f"planning_box_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=box["position"],
            euler=box["orientation_euler_xyz"],
            size=box["half_extents"],
            contype=0,
            conaffinity=1,
            density=0.0,
            rgba=color,
        )


def build_control_model(mujoco, model_path: Path, environment: dict[str, list]):
    spec = mujoco.MjSpec.from_file(str(model_path))

    for geom in spec.geoms:
        if geom.name == "floor":
            geom.pos = [0.0, 0.0, G1_PLANNING_FLOOR_HEIGHT]
            geom.contype = 0
            geom.conaffinity = 1
            continue
        if geom.meshname in ("left_rubber_hand", "right_rubber_hand"):
            # The source XML marks the rubber hands as visual-only.  ctrl mode
            # makes them physical so a rendered hand/obstacle overlap produces
            # an actual MuJoCo contact.
            geom.name = f"collision_{geom.meshname}"
            geom.contype = 1
            geom.conaffinity = 0
        elif geom.contype or geom.conaffinity:
            # Match the planner split: robot geoms interact with the physical
            # environment, while self-collision remains governed by the
            # planner's VAMP sphere-pair table rather than MuJoCo's mesh pairs.
            geom.contype = 1
            geom.conaffinity = 0

    add_physical_environment(mujoco, spec, environment)

    for axis, name in enumerate(G1_BASE_ACTUATORS):
        gear = [0.0] * 6
        gear[axis] = 1.0
        actuator = spec.add_actuator(
            name=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target="floating_base_joint",
            gear=gear,
        )
        actuator.set_to_motor()
        actuator.ctrllimited = True
        actuator.ctrlrange = (
            [-2000.0, 2000.0]
            if axis < 3
            else [-500.0, 500.0]
        )

    model = spec.compile()
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.timestep = min(float(model.opt.timestep), 0.001)
    return model


def resolve_control_layout(mujoco, model):
    base_address, joint_qpos_addresses = resolve_model_layout(mujoco, model)
    base_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "floating_base_joint",
    )
    base_dof_address = int(model.jnt_dofadr[base_id])
    joint_dof_addresses: list[int] = []
    joint_actuator_ids: list[int] = []

    for name in G1_ACTUATED_JOINTS:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        if actuator_id < 0:
            raise ValueError(f"MuJoCo model is missing G1 actuator: {name}")
        joint_dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        joint_actuator_ids.append(actuator_id)

    base_actuator_ids = []
    for name in G1_BASE_ACTUATORS:
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        if actuator_id < 0:
            raise ValueError(f"MuJoCo model is missing base actuator: {name}")
        base_actuator_ids.append(actuator_id)

    return (
        base_address,
        base_dof_address,
        np.asarray(joint_qpos_addresses, dtype=np.int32),
        np.asarray(joint_dof_addresses, dtype=np.int32),
        np.asarray(joint_actuator_ids, dtype=np.int32),
        np.asarray(base_actuator_ids, dtype=np.int32),
    )


def configure_control_damping(model, layout, gain_scale: float) -> None:
    _, base_dof_address, _, joint_dof_addresses, _, _ = layout
    damping_scale = math.sqrt(gain_scale)
    model.dof_damping[base_dof_address : base_dof_address + 6] += (
        damping_scale * G1_BASE_KD
    )
    model.dof_damping[joint_dof_addresses] += damping_scale * G1_JOINT_KD


def add_environment_geometries(mujoco, viewer, environment: dict[str, list]) -> None:
    scene = viewer.user_scn
    primitive_count = sum(len(environment[key]) for key in ("sphere", "cylinder", "box"))
    if primitive_count > len(scene.geoms):
        raise ValueError("too many environment primitives for the MuJoCo user scene")
    scene.ngeom = 0

    for sphere in environment["sphere"]:
        radius = float(sphere["radius"])
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([radius, 0.0, 0.0]),
            np.asarray(sphere["position"], dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray([0.85, 0.25, 0.20, 0.65], dtype=np.float32),
        )
        scene.ngeom += 1

    for cylinder in environment["cylinder"]:
        radius = float(cylinder["radius"])
        half_length = 0.5 * float(cylinder["length"])
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.asarray([radius, half_length, 0.0]),
            np.asarray(cylinder["position"], dtype=np.float64),
            euler_xyz_matrix(cylinder["orientation_euler_xyz"]),
            np.asarray([0.85, 0.45, 0.15, 0.65], dtype=np.float32),
        )
        scene.ngeom += 1

    for box in environment["box"]:
        is_obstacle = box.get("name") == "obstacle"
        color = [0.85, 0.20, 0.15, 0.65] if is_obstacle else [0.45, 0.32, 0.18, 0.65]
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(box["half_extents"], dtype=np.float64),
            np.asarray(box["position"], dtype=np.float64),
            euler_xyz_matrix(box["orientation_euler_xyz"]),
            np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1


def interpolated_frames(waypoints: list[list[float]], fps: float, speed: float):
    for start, goal in zip(waypoints, waypoints[1:]):
        maximum_change = max(
            abs(goal[index] - start[index])
            for index in range(G1_CONFIGURATION_DIMENSION)
        )
        frame_count = max(2, math.ceil(maximum_change / speed * fps))
        for frame_index in range(frame_count):
            ratio = (frame_index + 1) / frame_count
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            yield [
                start[index]
                + (goal[index] - start[index]) * smooth_ratio
                for index in range(G1_CONFIGURATION_DIMENSION)
            ]


def apply_ctrl_reference(
    mujoco,
    model,
    data,
    layout,
    reference: list[float],
    gain_scale: float,
    orientation_error: np.ndarray,
) -> None:
    (
        base_qpos_address,
        base_dof_address,
        joint_qpos_addresses,
        joint_dof_addresses,
        joint_actuator_ids,
        base_actuator_ids,
    ) = layout

    target_quaternion = roll_pitch_yaw_quaternion(*reference[3:6])
    mujoco.mju_subQuat(
        orientation_error,
        target_quaternion,
        data.qpos[base_qpos_address + 3 : base_qpos_address + 7],
    )
    base_error = np.concatenate(
        (
            np.asarray(reference[0:3])
            - data.qpos[base_qpos_address : base_qpos_address + 3],
            orientation_error,
        )
    )

    base_dofs = slice(base_dof_address, base_dof_address + 6)
    data.ctrl[base_actuator_ids] = (
        gain_scale * G1_BASE_KP * base_error
        + data.qfrc_bias[base_dofs]
    )

    joint_error = (
        np.asarray(reference[6:]) - data.qpos[joint_qpos_addresses]
    )
    data.ctrl[joint_actuator_ids] = (
        gain_scale * G1_JOINT_KP * joint_error
        + data.qfrc_bias[joint_dof_addresses]
    )


def replay(
    model_path: Path,
    waypoints: list[list[float]],
    environment: dict[str, list],
    fps: float,
    speed: float,
    control_mode: str,
    gain_scale: float,
) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo Python package is required: python3 -m pip install mujoco"
        ) from error

    if control_mode == "ctrl":
        model = build_control_model(mujoco, model_path, environment)
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    base_address, joint_addresses = resolve_model_layout(mujoco, model)
    control_layout = (
        resolve_control_layout(mujoco, model)
        if control_mode == "ctrl"
        else None
    )
    if control_layout is not None:
        configure_control_damping(model, control_layout, gain_scale)
    validate_joint_limits(mujoco, model, waypoints)
    apply_configuration(
        mujoco,
        model,
        data,
        base_address,
        joint_addresses,
        waypoints[0],
    )

    print(
        "MuJoCo viewer: G1 start -> goal 경로를 "
        f"{control_mode} 모드로 반복 재생합니다."
    )
    if control_mode == "ctrl":
        print(
            "29개 관절 motor와 6개 floating-base virtual motor에 "
            "PD+중력보상 torque를 입력합니다."
        )
        print("shelf/obstacle/rubber hand는 실제 MuJoCo contact에 참여합니다.")
    else:
        print("qpos를 직접 적용하는 기존 kinematic replay입니다.")
    print("갈색은 shelf, 빨간색은 obstacle입니다. 창을 닫으면 종료됩니다.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            if control_mode == "qpos":
                add_environment_geometries(mujoco, viewer, environment)
            viewer.cam.lookat[:] = (0.25, 0.0, 0.75)
            viewer.cam.distance = 2.8
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
        viewer.sync()

        frame_period = 1.0 / fps
        control_substeps = max(
            1,
            math.ceil(frame_period / float(model.opt.timestep)),
        )
        orientation_error = np.zeros(3, dtype=np.float64)
        while viewer.is_running():
            mujoco.mj_resetData(model, data)
            apply_configuration(
                mujoco,
                model,
                data,
                base_address,
                joint_addresses,
                waypoints[0],
            )
            viewer.sync()
            time.sleep(0.75)
            deadline = time.perf_counter()
            for configuration in interpolated_frames(waypoints, fps, speed):
                if not viewer.is_running():
                    return
                if control_mode == "ctrl":
                    for _ in range(control_substeps):
                        apply_ctrl_reference(
                            mujoco,
                            model,
                            data,
                            control_layout,
                            configuration,
                            gain_scale,
                            orientation_error,
                        )
                        mujoco.mj_step(model, data)
                    if not np.isfinite(data.qpos).all():
                        raise RuntimeError("ctrl simulation became non-finite")
                else:
                    apply_configuration(
                        mujoco,
                        model,
                        data,
                        base_address,
                        joint_addresses,
                        configuration,
                    )
                viewer.sync()
                deadline += frame_period
                time.sleep(max(0.0, deadline - time.perf_counter()))
            time.sleep(1.0)


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    trajectory_path = args.trajectory.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"G1 MuJoCo model not found: {model_path}; set VAMP_ROOT if needed"
        )
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory not found: {trajectory_path}")

    waypoints, environment = load_trajectory(trajectory_path)
    validate_environment(environment)
    validate_only = args.validate_only or os.environ.get(
        "PRRTC_MUJOCO_VALIDATE_ONLY"
    ) == "1"
    if validate_only:
        import mujoco

        if args.control_mode == "ctrl":
            model = build_control_model(mujoco, model_path, environment)
        else:
            model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        base_address, joint_addresses = resolve_model_layout(mujoco, model)
        validate_joint_limits(mujoco, model, waypoints)
        for configuration in waypoints:
            apply_configuration(
                mujoco,
                model,
                data,
                base_address,
                joint_addresses,
                configuration,
            )
        if args.control_mode == "ctrl":
            control_layout = resolve_control_layout(mujoco, model)
            configure_control_damping(model, control_layout, args.gain_scale)
            orientation_error = np.zeros(3, dtype=np.float64)
            apply_configuration(
                mujoco,
                model,
                data,
                base_address,
                joint_addresses,
                waypoints[0],
            )
            for _ in range(10):
                apply_ctrl_reference(
                    mujoco,
                    model,
                    data,
                    control_layout,
                    waypoints[0],
                    args.gain_scale,
                    orientation_error,
                )
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("ctrl validation rollout became non-finite")
        primitive_count = sum(
            len(environment[key]) for key in ("sphere", "cylinder", "box")
        )
        print(
            f"validated {len(waypoints)} waypoints, "
            f"35 planning coordinates -> model nq={model.nq}, "
            f"nu={model.nu}, control_mode={args.control_mode}, "
            f"{primitive_count} environment primitives"
        )
        return 0

    replay(
        model_path,
        waypoints,
        environment,
        args.fps,
        args.speed,
        args.control_mode,
        args.gain_scale,
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        print(f"G1 visualization error: {error}", file=sys.stderr)
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)

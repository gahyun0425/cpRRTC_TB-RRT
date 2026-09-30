#!/usr/bin/env python3
"""Replay a 35-DoF PATACON G1 trajectory with its planning obstacles."""

from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from mujoco_video import (
    MultiViewVideoWriter,
    add_video_text_overlay,
    add_video_arguments,
    configure_model_render_quality,
    validate_video_arguments,
    video_view_azimuth,
    video_view_elevation,
)


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
REPOSITORY_DIR = Path(__file__).resolve().parents[1]
REPLAN_PROBLEM_NAME = "g1_continuous_replan"
REPLAN_RESPONSE_PREFIX = b"G1_REPLAN_RESULT "
REPLAN_START_TOLERANCE = 0.1
# Treat planner nodes within 3 milliradians/millimetres as duplicates. Retaining
# them creates a two-frame segment and a large acceleration spike at the next
# meaningful edge even though the geometric path is effectively unchanged.
REPLAN_PROGRESS_TOLERANCE = 3.0e-3
# Keep only a short fixed floor.  The loop adds the observed planner latency
# and the tracking-stability window below, so faster planners can hand off a
# replacement sooner instead of waiting behind a fixed one-second gate.
CONTINUOUS_REPLAN_LOOKAHEAD_SEC = 0.10
REPLANNING_SPEED_SCALE = 0.3
REPLANNING_ACCELERATION_LIMIT = 4.0
CONTINUOUS_CANDIDATE_MAX_START_ERROR = 0.15
HANDOFF_JOINT_POSITION_TOLERANCE_RAD = 0.05
HANDOFF_JOINT_VELOCITY_ERROR_TOLERANCE_RAD_PER_SEC = 0.20
HANDOFF_TRACKING_STABLE_SEC = 0.10
HANDOFF_TRACKING_CHECK_INTERVAL = 1
HANDOFF_ACTUAL_RECOVERY_POSITION_ERROR_RAD = 0.15
FINAL_APPROACH_REMAINING_PATH_SEC = 1.0
FINAL_GOAL_NODE_TOLERANCE = 1.0e-4
PING_PONG_GOAL_HOLD_SEC = 1.0
OBJECT_PATH_PREVIEW_MAX_POINTS = 96
OBJECT_PATH_TRAIL_MAX_POINTS = 180
OBJECT_PATH_TRAIL_MIN_DISTANCE_M = 0.002
OBJECT_PATH_OVERLAY_REFRESH_FRAMES = 3
OBJECT_PATH_LINE_RADIUS_M = 0.004
OBJECT_PATH_ACTIVE_RGBA = (0.10, 0.90, 0.25, 0.88)
OBJECT_PATH_ACTUAL_RGBA = (0.05, 0.42, 1.00, 0.96)
MOUSE_OBSTACLE_NAME = "mouse_dynamic_obstacle"
MOUSE_OBSTACLE_BODY_NAME = "mouse_dynamic_obstacle_body"
DYNAMIC_OBSTACLE_Y_MIN_OFFSET_M = -0.20
DYNAMIC_OBSTACLE_Y_MAX_OFFSET_M = 0.00
DYNAMIC_OBSTACLE_Y_SPEED_M_PER_SEC = 0.05
DYNAMIC_OBSTACLE_ENDPOINT_HOLD_SEC = 2.0
G1_SUPPORT_FOOT_BODIES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
# The planner's foot end-effector frames are constrained to z=0, while the
# MuJoCo sole contact points extend about 35 mm below those frames.
G1_PLANNING_FLOOR_HEIGHT = -0.0351
FRANKA_FLOOR_RGB = (0.48, 0.48, 0.48)
FRANKA_BACKGROUND_RGB = (0.72, 0.72, 0.72)
REPLANNING_CAMERA_LOOKAT = (0.25, 0.0, 0.60)
G1_FALL_MIN_PELVIS_HEIGHT_ABOVE_FLOOR_M = 0.20
G1_FALL_MIN_UPRIGHT_COSINE = math.cos(math.radians(60.0))

# Torque-PD gains in G1_ACTUATED_JOINTS order.  The native XML actuators are
# motors, so their ctrl values are torques rather than target angles.  The
# left wrist is tuned more firmly because it carries the attached payload.
G1_JOINT_KP = np.asarray(
    [
        37.5, 37.5, 37.5, 37.5, 37.5, 37.5,
        37.5, 37.5, 37.5, 37.5, 37.5, 37.5,
        30, 25, 30,
        20, 20, 20, 20, 8, 6, 10,
        20, 20, 15, 20, 5, 3, 3,
    ],
    dtype=np.float64,
)
G1_JOINT_KD = np.asarray(
    [
        20, 20, 12, 24, 8, 4,
        20, 20, 12, 24, 8, 4,
        7.5, 5, 7.5,
        4, 4, 3.5, 4, 2, 1.5, 2,
        4, 4, 3, 4, 1, 0.6, 0.6,
    ],
    dtype=np.float64,
)

# Contact-aware balance gains.  The support-force solve supplies the ground
# reaction needed by the unactuated floating base.
G1_COM_POSITION_KP = 12.0
G1_COM_VELOCITY_KD = 7.0
G1_BASE_ORIENTATION_KP = np.asarray([100.0, 100.0, 30.0])
G1_BASE_ORIENTATION_KD = np.asarray([20.0, 20.0, 8.0])
G1_SUPPORT_FRICTION_COEFFICIENT = 1.2
G1_SUPPORT_CONTACT_FRICTION = np.asarray(
    [G1_SUPPORT_FRICTION_COEFFICIENT, 0.005, 0.0001],
    dtype=np.float64,
)
G1_PAYLOAD_POSITION_KP = 600.0
G1_PAYLOAD_VELOCITY_KD = 70.0


@dataclass(frozen=True)
class G1ControlLayout:
    base_qpos_address: int
    base_dof_address: int
    joint_qpos_addresses: np.ndarray
    joint_dof_addresses: np.ndarray
    joint_actuator_ids: np.ndarray
    pelvis_body_id: int
    foot_contact_geom_ids: np.ndarray


@dataclass(frozen=True)
class G1BalanceReference:
    support_center_xy: np.ndarray
    base_quaternion: np.ndarray
    total_mass_kg: float


@dataclass(frozen=True)
class G1PayloadControlTarget:
    position: np.ndarray
    linear_velocity: np.ndarray


@dataclass(frozen=True)
class ReplanningSettings:
    planner_executable: Path
    base_seed: int
    aorrtc: bool
    time_limit_sec: float
    projection_smoothness: bool
    axis: bool
    mouse_obstacle_radius_m: float
    mouse_obstacle_collision_radius_m: float
    mouse_obstacle_initial_position: tuple[float, float, float]


@dataclass(frozen=True)
class MouseObstacleRuntime:
    body_id: int
    mocap_id: int
    collision_radius_m: float


@dataclass(frozen=True)
class ReplannedPath:
    waypoints: list[list[float]]
    planning_time_sec: float | None


def default_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
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
        default=os.environ.get("PATACON_G1_CONTROL_MODE", "qpos"),
        help=(
            "qpos directly replays the planned configurations (default); "
            "ctrl uses torque-PD actuators and physics"
        ),
    )
    parser.add_argument(
        "--replanning",
        action="store_true",
        help="Continuously replan while following the G1 path in ctrl mode.",
    )
    parser.add_argument(
        "--gain-scale",
        type=float,
        default=16.0,
        help=(
            "Scale all ctrl-mode proportional and derivative gains "
            "(default: 16.0)."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate without opening a viewer window.",
    )
    snapshot_default = os.environ.get("PATACON_G1_SNAPSHOT")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(snapshot_default) if snapshot_default else None,
        help="Render one close-up frame to a PNG instead of opening a viewer.",
    )
    parser.add_argument(
        "--snapshot-waypoint",
        type=int,
        default=int(os.environ.get("PATACON_G1_SNAPSHOT_WAYPOINT", "-1")),
        help="Waypoint to render; -1 selects the middle waypoint.",
    )
    add_video_arguments(parser, "PATACON_G1_VIDEO")
    args = parser.parse_args()
    if args.control_mode not in ("ctrl", "qpos"):
        parser.error("--control-mode must be ctrl or qpos")
    if args.fps <= 0.0 or args.speed <= 0.0 or args.gain_scale <= 0.0:
        parser.error("--fps, --speed, and --gain-scale must be positive")
    if args.replanning and args.control_mode != "ctrl":
        parser.error("--replanning requires --control-mode ctrl")
    if args.snapshot is not None and args.video is not None:
        parser.error("--snapshot and --video cannot be used together")
    if args.video is not None and not args.replanning:
        parser.error("--video in this visualizer requires --replanning")
    validate_video_arguments(parser, args)
    return args


def finite_vector(value, dimension: int, description: str) -> list[float]:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{description} must contain {dimension} values")
    normalized = [float(component) for component in value]
    if not all(math.isfinite(component) for component in normalized):
        raise ValueError(f"{description} contains a non-finite value")
    return normalized


def load_trajectory(
    path: Path,
) -> tuple[
    list[list[float]],
    dict[str, list],
    dict | None,
    list[float] | None,
    dict,
    ReplanningSettings | None,
    float | None,
]:
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

    payload = document.get("payload")
    if payload is not None and not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    goal_value = document.get("goal")
    goal = (
        finite_vector(goal_value, G1_CONFIGURATION_DIMENSION, "trajectory goal")
        if goal_value is not None
        else None
    )
    constraints = document.get("constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be a JSON object")
    planning_time_value = document.get("planning_time_sec")
    planning_time_sec = (
        float(planning_time_value)
        if planning_time_value is not None
        else None
    )
    if planning_time_sec is not None and (
        not math.isfinite(planning_time_sec) or planning_time_sec < 0.0
    ):
        raise ValueError("planning_time_sec must be finite and nonnegative")

    replanning_value = document.get("replanning", {})
    if not isinstance(replanning_value, dict):
        raise ValueError("replanning must be a JSON object")
    replanning = None
    if bool(replanning_value.get("enabled", False)):
        executable = replanning_value.get("planner_executable")
        if not isinstance(executable, str) or not executable:
            raise ValueError("replanning.planner_executable must be a path")
        base_seed = int(replanning_value.get("base_seed", 1))
        time_limit_sec = float(replanning_value.get("time_limit_sec", 5.0))
        radius = float(replanning_value.get("mouse_obstacle_radius_m", 0.040))
        collision_radius = float(
            replanning_value.get("mouse_obstacle_collision_radius_m", 0.040)
        )
        position = tuple(
            finite_vector(
                replanning_value.get(
                    "mouse_obstacle_initial_position",
                    [0.400, 0.200, 0.800],
                ),
                3,
                "replanning.mouse_obstacle_initial_position",
            )
        )
        if not math.isfinite(time_limit_sec) or time_limit_sec <= 0.0:
            raise ValueError("replanning.time_limit_sec must be positive")
        if base_seed < 0 or base_seed > 2**64 - 1:
            raise ValueError("replanning.base_seed must fit in uint64")
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("replanning.mouse_obstacle_radius_m must be positive")
        if not math.isfinite(collision_radius) or collision_radius <= 0.0:
            raise ValueError(
                "replanning.mouse_obstacle_collision_radius_m must be positive"
            )
        if collision_radius < radius:
            raise ValueError(
                "replanning.mouse_obstacle_collision_radius_m must be at least "
                "mouse_obstacle_radius_m"
            )
        replanning = ReplanningSettings(
            planner_executable=Path(executable).expanduser().resolve(),
            base_seed=base_seed,
            aorrtc=bool(replanning_value.get("aorrtc", False)),
            time_limit_sec=time_limit_sec,
            projection_smoothness=bool(
                replanning_value.get("projection_smoothness", True)
            ),
            axis=bool(
                replanning_value.get("axis", False)
            ),
            mouse_obstacle_radius_m=radius,
            mouse_obstacle_collision_radius_m=collision_radius,
            mouse_obstacle_initial_position=position,
        )
    return (
        waypoints,
        environment,
        payload,
        goal,
        constraints,
        replanning,
        planning_time_sec,
    )


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


def resolve_support_foot_geometries(mujoco, model) -> tuple[list[int], list[int]]:
    foot_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in G1_SUPPORT_FOOT_BODIES
    ]
    if any(body_id < 0 for body_id in foot_body_ids):
        raise ValueError("MuJoCo model is missing a G1 support-foot body")
    foot_contact_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in foot_body_ids
        and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    ]
    if not foot_contact_geoms:
        raise ValueError("MuJoCo model has no G1 foot contact spheres")
    return foot_body_ids, foot_contact_geoms


def rpy_chain_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rx @ ry @ rz


def rpy_chain_angles(rotation: np.ndarray) -> list[float]:
    """Invert ``rpy_chain_rotation`` for a 3-by-3 rotation matrix."""
    pitch = math.asin(float(np.clip(rotation[0, 2], -1.0, 1.0)))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) > 1.0e-8:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[2, 2]))
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[0, 0]))
    else:
        # At the XYZ gimbal lock, choose yaw=0 and preserve the observable
        # combined X/Z rotation in roll.
        yaw = 0.0
        roll_sign = 1.0 if pitch >= 0.0 else -1.0
        roll = math.atan2(
            roll_sign * float(rotation[1, 0]),
            float(rotation[1, 1]),
        )
    return [roll, pitch, yaw]


def actual_g1_configuration(mujoco, data, layout: G1ControlLayout) -> list[float]:
    """Return the live MuJoCo floating-base and actuated-joint state."""
    quaternion = data.qpos[
        layout.base_qpos_address + 3 : layout.base_qpos_address + 7
    ]
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quaternion)
    return (
        data.qpos[
            layout.base_qpos_address : layout.base_qpos_address + 3
        ].astype(float).tolist()
        + rpy_chain_angles(rotation.reshape(3, 3))
        + data.qpos[layout.joint_qpos_addresses].astype(float).tolist()
    )


def apply_actuated_joint_configuration(
    mujoco,
    model,
    data,
    joint_addresses: list[int],
    configuration: list[float],
) -> None:
    for address, value in zip(joint_addresses, configuration[6:]):
        data.qpos[address] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def apply_configuration(
    mujoco,
    model,
    data,
    base_address: int,
    joint_addresses: list[int],
    configuration: list[float],
) -> None:
    data.qpos[base_address : base_address + 3] = configuration[:3]
    quaternion = np.empty(4, dtype=np.float64)
    rotation = rpy_chain_rotation(*configuration[3:6])
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    data.qpos[base_address + 3 : base_address + 7] = quaternion
    apply_actuated_joint_configuration(
        mujoco, model, data, joint_addresses, configuration
    )


def initialize_floating_base_from_feet(
    mujoco,
    model,
    data,
    base_address: int,
    joint_addresses: list[int],
    configuration: list[float],
) -> None:
    """Initialize the free base once from joint FK and ground contact."""
    # Ground contact alone does not determine world X/Y. Preserve the XML
    # convention for those coordinates and solve orientation/height only.
    data.qpos[base_address : base_address + 2] = model.qpos0[
        base_address : base_address + 2
    ]
    data.qpos[base_address + 2] = 0.0
    data.qpos[base_address + 3 : base_address + 7] = [1.0, 0.0, 0.0, 0.0]
    apply_actuated_joint_configuration(
        mujoco, model, data, joint_addresses, configuration
    )

    foot_body_ids, foot_contact_geoms = resolve_support_foot_geometries(
        mujoco, model
    )

    # Find the closest rotation to the two foot-frame orientations, then
    # rotate the base so their average contact plane is horizontal.
    mean_foot_rotation = sum(
        data.xmat[body_id].reshape(3, 3) for body_id in foot_body_ids
    )
    left_singular, _, right_singular = np.linalg.svd(mean_foot_rotation)
    handedness = np.linalg.det(left_singular @ right_singular)
    average_foot_rotation = (
        left_singular
        @ np.diag([1.0, 1.0, 1.0 if handedness >= 0.0 else -1.0])
        @ right_singular
    )
    base_rotation = average_foot_rotation.T
    base_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(base_quaternion, base_rotation.reshape(-1))
    data.qpos[base_address + 3 : base_address + 7] = base_quaternion
    mujoco.mj_forward(model, data)

    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    if floor_id < 0:
        raise ValueError("MuJoCo model is missing the floor geom")
    floor_height = float(model.geom_pos[floor_id, 2])
    lowest_sole = min(
        float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0])
        for geom_id in foot_contact_geoms
    )
    data.qpos[base_address + 2] += floor_height - lowest_sole
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


def validate_payload(payload: dict | None) -> None:
    if payload is None:
        return
    mass_kg = float(payload.get("mass_kg"))
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("payload.mass_kg must be positive")
    half_extents = finite_vector(
        payload.get("half_extents"), 3, "payload half extents"
    )
    if any(value <= 0.0 for value in half_extents):
        raise ValueError("payload half extents must be positive")
    rgba = finite_vector(payload.get("rgba"), 4, "payload rgba")
    if any(value < 0.0 or value > 1.0 for value in rgba):
        raise ValueError("payload rgba components must be in [0, 1]")
    finite_vector(
        payload.get("left_hand_center_offset"),
        3,
        "payload left-hand center offset",
    )
    payload_collision_spheres(payload)


def payload_collision_spheres(
    payload: dict | None,
) -> list[tuple[list[float], float]]:
    if payload is None:
        return []
    counts_value = payload.get("collision_spheres_per_axis")
    radius_value = payload.get("collision_sphere_radius")
    if counts_value is None and radius_value is None:
        return []
    if not isinstance(counts_value, list) or len(counts_value) != 3:
        raise ValueError("payload collision_spheres_per_axis must contain 3 integers")
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count <= 0
        for count in counts_value
    ):
        raise ValueError("payload collision sphere counts must be positive integers")
    radius = float(radius_value)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("payload collision_sphere_radius must be positive")

    half_extents = finite_vector(
        payload.get("half_extents"), 3, "payload half extents"
    )
    cell_half_extents = [
        half_extents[axis] / counts_value[axis] for axis in range(3)
    ]
    spheres: list[tuple[list[float], float]] = []
    for z in range(counts_value[2]):
        for y in range(counts_value[1]):
            for x in range(counts_value[0]):
                indices = (x, y, z)
                center = [
                    -half_extents[axis]
                    + (2.0 * indices[axis] + 1.0) * cell_half_extents[axis]
                    for axis in range(3)
                ]
                spheres.append((center, radius))
    return spheres


def add_payload_to_spec(mujoco, spec, payload: dict | None) -> None:
    if payload is None:
        return
    left_wrist = spec.body("left_wrist_yaw_link")
    if left_wrist is None:
        raise ValueError("MuJoCo model is missing left_wrist_yaw_link")
    payload_body = left_wrist.add_body(
        name="planning_payload",
        pos=payload["left_hand_center_offset"],
    )
    box_rgba = list(payload["rgba"])
    visual_half_extents = list(payload["half_extents"])
    visual_half_extents[0] = 0.07
    visual_half_extents[2] = 0.03
    payload_body.add_geom(
        name="planning_payload_collision_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=payload["half_extents"],
        mass=float(payload["mass_kg"]),
        # Participate in payload-vs-world contacts. Robot geoms use the same
        # contype and therefore remain filtered from this attached payload.
        contype=1,
        conaffinity=0,
        priority=1,
        solref=[0.006, 0.35],
        friction=[0.8, 0.005, 0.0001],
        rgba=[0.0, 0.0, 0.0, 0.0],
    )
    payload_body.add_geom(
        name="planning_payload_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=visual_half_extents,
        mass=0.0,
        contype=0,
        conaffinity=0,
        rgba=box_rgba,
    )


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


def add_mouse_obstacle_to_spec(
    mujoco,
    spec,
    replanning: ReplanningSettings | None,
) -> None:
    if replanning is None:
        return
    body = spec.worldbody.add_body(
        name=MOUSE_OBSTACLE_BODY_NAME,
        pos=replanning.mouse_obstacle_initial_position,
    )
    body.mocap = True
    body.add_geom(
        name=MOUSE_OBSTACLE_NAME,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        # Render and physically collide with exactly the sphere sent to the
        # planner instead of showing the former, smaller proxy sphere.
        size=[replanning.mouse_obstacle_collision_radius_m, 0.0, 0.0],
        mass=0.0,
        contype=0,
        conaffinity=1,
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.90, 0.04, 0.04, 1.0],
    )


def apply_franka_scene_appearance(mujoco, spec) -> None:
    """Match the floor and background colors used by the Franka XML scenes."""
    floor = spec.geom("floor")
    if floor is None:
        raise ValueError("MuJoCo model is missing the floor geom")

    floor_texture_names: set[str] = set()
    if floor.material:
        floor_material = spec.material(floor.material)
        if floor_material is not None:
            floor_material.reflectance = 0.0
            floor_texture_names.update(
                name for name in floor_material.textures if name
            )

    found_floor_texture = False
    found_skybox = False
    for texture in spec.textures:
        if texture.name in floor_texture_names:
            texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
            texture.rgb1 = FRANKA_FLOOR_RGB
            texture.rgb2 = FRANKA_FLOOR_RGB
            texture.mark = mujoco.mjtMark.mjMARK_NONE
            found_floor_texture = True
        if int(texture.type) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
            texture.rgb1 = FRANKA_BACKGROUND_RGB
            texture.rgb2 = FRANKA_BACKGROUND_RGB
            found_skybox = True

    if not found_floor_texture:
        floor.material = ""
        floor.rgba = (*FRANKA_FLOOR_RGB, 1.0)
    if not found_skybox:
        spec.add_texture(
            name="franka_style_skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=FRANKA_BACKGROUND_RGB,
            rgb2=FRANKA_BACKGROUND_RGB,
            width=256,
            height=1536,
        )

    spec.visual.rgba.haze = (*FRANKA_BACKGROUND_RGB, 1.0)


def build_control_model(
    mujoco,
    model_path: Path,
    environment: dict[str, list],
    payload: dict | None,
    replanning: ReplanningSettings | None = None,
):
    spec = mujoco.MjSpec.from_file(str(model_path))
    apply_franka_scene_appearance(mujoco, spec)
    add_payload_to_spec(mujoco, spec, payload)

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
            # planner's validated sphere-pair table rather than MuJoCo's
            # mesh pairs.
            geom.contype = 1
            geom.conaffinity = 0

    physical_environment = environment
    if replanning is not None:
        # The environment snapshot also contains the red obstacle used by the
        # initial plan.  Keep the static world physical, but let the mocap body
        # below be the red obstacle's only MuJoCo representation.
        physical_environment = copy.deepcopy(environment)
        physical_environment["sphere"] = [
            sphere
            for sphere in physical_environment["sphere"]
            if sphere.get("name") != MOUSE_OBSTACLE_NAME
        ]
    add_physical_environment(mujoco, spec, physical_environment)
    add_mouse_obstacle_to_spec(mujoco, spec, replanning)

    model = spec.compile()
    _, foot_contact_geom_ids = resolve_support_foot_geometries(mujoco, model)
    floor_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )
    if floor_geom_id < 0:
        raise ValueError("MuJoCo model is missing the floor geom")
    model.geom_friction[np.asarray(foot_contact_geom_ids, dtype=int)] = (
        G1_SUPPORT_CONTACT_FRICTION
    )
    model.geom_friction[floor_geom_id] = G1_SUPPORT_CONTACT_FRICTION
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.timestep = min(float(model.opt.timestep), 0.001)
    return model


def build_kinematic_model(mujoco, model_path: Path, payload: dict | None):
    spec = mujoco.MjSpec.from_file(str(model_path))
    apply_franka_scene_appearance(mujoco, spec)
    add_payload_to_spec(mujoco, spec, payload)
    floor = spec.geom("floor")
    if floor is None:
        raise ValueError("MuJoCo model is missing the floor geom")
    floor.pos = [0.0, 0.0, G1_PLANNING_FLOOR_HEIGHT]
    return spec.compile()


def resolve_control_layout(mujoco, model):
    base_qpos_address, joint_qpos_addresses = resolve_model_layout(
        mujoco, model
    )
    base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
    )
    pelvis_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    if pelvis_body_id < 0:
        raise ValueError("MuJoCo model is missing the G1 pelvis body")
    _, foot_contact_geom_ids = resolve_support_foot_geometries(mujoco, model)
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

    return G1ControlLayout(
        base_qpos_address=base_qpos_address,
        base_dof_address=int(model.jnt_dofadr[base_id]),
        joint_qpos_addresses=np.asarray(
            joint_qpos_addresses, dtype=np.int32
        ),
        joint_dof_addresses=np.asarray(
            joint_dof_addresses, dtype=np.int32
        ),
        joint_actuator_ids=np.asarray(joint_actuator_ids, dtype=np.int32),
        pelvis_body_id=pelvis_body_id,
        foot_contact_geom_ids=np.asarray(
            foot_contact_geom_ids, dtype=np.int32
        ),
    )


def configure_control_damping(
    model, layout: G1ControlLayout, gain_scale: float
) -> None:
    # Keep the negative-velocity half of the derivative term in MuJoCo's
    # implicit passive damping for contact-stable integration.  The matching
    # desired-velocity feed-forward is added in apply_ctrl_reference.
    damping_scale = math.sqrt(gain_scale)
    model.dof_damping[layout.joint_dof_addresses] += (
        damping_scale * G1_JOINT_KD
    )


def make_balance_reference(
    model,
    data,
    layout: G1ControlLayout,
) -> G1BalanceReference:
    support_center_xy = np.mean(
        data.geom_xpos[layout.foot_contact_geom_ids, :2], axis=0
    )
    return G1BalanceReference(
        support_center_xy=np.asarray(support_center_xy).copy(),
        base_quaternion=data.qpos[
            layout.base_qpos_address + 3 : layout.base_qpos_address + 7
        ].copy(),
        total_mass_kg=float(model.body_subtreemass[layout.pelvis_body_id]),
    )


def g1_fall_reason(data, layout: G1ControlLayout) -> str | None:
    """Return a diagnostic when the pelvis is too low or no longer upright."""
    pelvis_height = float(
        data.xpos[layout.pelvis_body_id, 2] - G1_PLANNING_FLOOR_HEIGHT
    )
    pelvis_up_z = float(data.xmat[layout.pelvis_body_id, 8])
    if pelvis_height < G1_FALL_MIN_PELVIS_HEIGHT_ABOVE_FLOOR_M:
        return (
            f"pelvis height {pelvis_height:.3f} m < "
            f"{G1_FALL_MIN_PELVIS_HEIGHT_ABOVE_FLOOR_M:.3f} m"
        )
    if pelvis_up_z < G1_FALL_MIN_UPRIGHT_COSINE:
        return (
            f"pelvis upright cosine {pelvis_up_z:.3f} < "
            f"{G1_FALL_MIN_UPRIGHT_COSINE:.3f}"
        )
    return None


def solve_support_forces(
    base_wrench_map: np.ndarray,
    desired_base_wrench: np.ndarray,
) -> np.ndarray:
    """Distribute a base wrench over unilateral, friction-limited feet."""
    contact_count = base_wrench_map.shape[1] // 3
    active_contacts = list(range(contact_count))
    forces = np.zeros(3 * contact_count, dtype=np.float64)
    tolerance = 1.0e-6

    while active_contacts:
        columns = np.asarray(
            [
                3 * contact + axis
                for contact in active_contacts
                for axis in range(3)
            ],
            dtype=np.int32,
        )
        forces[:] = 0.0
        forces[columns] = np.linalg.lstsq(
            base_wrench_map[:, columns],
            desired_base_wrench,
            rcond=1.0e-8,
        )[0]

        violations: list[tuple[float, int]] = []
        for contact in active_contacts:
            offset = 3 * contact
            normal_force = float(forces[offset + 2])
            tangent_force = float(np.linalg.norm(forces[offset : offset + 2]))
            friction_limit = (
                G1_SUPPORT_FRICTION_COEFFICIENT * max(normal_force, 0.0)
            )
            violation = max(-normal_force, tangent_force - friction_limit)
            if violation > tolerance:
                violations.append((violation, contact))
        if not violations:
            forces[2::3] = np.maximum(forces[2::3], 0.0)
            return forces
        _, worst_contact = max(violations)
        active_contacts.remove(worst_contact)

    raise RuntimeError("G1 balance wrench has no feasible foot-force solution")


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
        # Cubic smoothstep reaches 1.5 times its average rate at the midpoint.
        # Lengthen the edge so `speed` is the actual peak-rate limit.
        frame_count = max(
            2,
            math.ceil(1.5 * maximum_change / speed * fps),
        )
        for frame_index in range(frame_count):
            ratio = (frame_index + 1) / frame_count
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            yield [
                start[index]
                + (goal[index] - start[index]) * smooth_ratio
                for index in range(G1_CONFIGURATION_DIMENSION)
            ]


def make_payload_control_target(
    mujoco,
    model,
    reference_data,
    layout: G1ControlLayout,
    payload_body_id: int,
    reference: list[float],
    reference_velocity: list[float] | np.ndarray | None,
) -> G1PayloadControlTarget | None:
    if payload_body_id < 0:
        return None
    velocity = (
        np.zeros(G1_CONFIGURATION_DIMENSION, dtype=np.float64)
        if reference_velocity is None
        else np.asarray(reference_velocity, dtype=np.float64)
    )
    apply_configuration(
        mujoco,
        model,
        reference_data,
        layout.base_qpos_address,
        list(layout.joint_qpos_addresses),
        reference,
    )
    desired_qvel = np.zeros(model.nv, dtype=np.float64)
    desired_qvel[
        layout.base_dof_address : layout.base_dof_address + 6
    ] = velocity[:6]
    desired_qvel[layout.joint_dof_addresses] = velocity[6:]
    position_jacobian = np.empty((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(
        model,
        reference_data,
        position_jacobian,
        None,
        payload_body_id,
    )
    return G1PayloadControlTarget(
        position=reference_data.xpos[payload_body_id].copy(),
        linear_velocity=position_jacobian @ desired_qvel,
    )


def apply_ctrl_reference(
    mujoco,
    model,
    data,
    layout: G1ControlLayout,
    balance_reference: G1BalanceReference,
    reference: list[float],
    gain_scale: float,
    reference_velocity: list[float] | np.ndarray | None = None,
    payload_body_id: int = -1,
    payload_target: G1PayloadControlTarget | None = None,
) -> None:
    if reference_velocity is None:
        reference_velocity_array = np.zeros(
            G1_CONFIGURATION_DIMENSION, dtype=np.float64
        )
    else:
        reference_velocity_array = np.asarray(
            reference_velocity, dtype=np.float64
        )
    joint_error = (
        np.asarray(reference[6:])
        - data.qpos[layout.joint_qpos_addresses]
    )
    joint_torque = (
        gain_scale * G1_JOINT_KP * joint_error
        + math.sqrt(gain_scale)
        * G1_JOINT_KD
        * reference_velocity_array[6:]
        + data.qfrc_bias[layout.joint_dof_addresses]
    )

    contact_jacobian = np.empty(
        (3 * len(layout.foot_contact_geom_ids), model.nv),
        dtype=np.float64,
    )
    for contact, geom_id in enumerate(layout.foot_contact_geom_ids):
        rows = slice(3 * contact, 3 * contact + 3)
        body_id = int(model.geom_bodyid[geom_id])
        mujoco.mj_jac(
            model,
            data,
            contact_jacobian[rows],
            None,
            data.geom_xpos[geom_id],
            body_id,
        )

    base_dofs = slice(
        layout.base_dof_address, layout.base_dof_address + 6
    )
    desired_base_wrench = data.qfrc_bias[base_dofs].copy()

    com_jacobian = np.empty((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSubtreeCom(
        model, data, com_jacobian, layout.pelvis_body_id
    )
    com_velocity = com_jacobian @ data.qvel
    com_position = data.subtree_com[layout.pelvis_body_id]
    desired_base_wrench[:2] += balance_reference.total_mass_kg * (
        G1_COM_POSITION_KP
        * (balance_reference.support_center_xy - com_position[:2])
        - G1_COM_VELOCITY_KD * com_velocity[:2]
    )

    orientation_error = np.empty(3, dtype=np.float64)
    current_base_quaternion = data.qpos[
        layout.base_qpos_address + 3 : layout.base_qpos_address + 7
    ]
    mujoco.mju_subQuat(
        orientation_error,
        balance_reference.base_quaternion,
        current_base_quaternion,
    )
    desired_base_wrench[3:6] += (
        G1_BASE_ORIENTATION_KP * orientation_error
        - G1_BASE_ORIENTATION_KD
        * data.qvel[
            layout.base_dof_address + 3 : layout.base_dof_address + 6
        ]
    )

    base_wrench_map = contact_jacobian[:, base_dofs].T
    support_forces = solve_support_forces(
        base_wrench_map, desired_base_wrench
    )
    # Contact forces supply the unactuated-base wrench.  Subtract their joint
    # generalized forces from inverse-dynamics bias to obtain support torque.
    joint_torque -= (
        contact_jacobian[:, layout.joint_dof_addresses].T @ support_forces
    )

    if payload_target is not None and payload_body_id >= 0:
        payload_position_jacobian = np.empty(
            (3, model.nv), dtype=np.float64
        )
        mujoco.mj_jacBody(
            model,
            data,
            payload_position_jacobian,
            None,
            payload_body_id,
        )
        payload_linear_velocity = payload_position_jacobian @ data.qvel
        payload_force = (
            G1_PAYLOAD_POSITION_KP
            * (payload_target.position - data.xpos[payload_body_id])
            + G1_PAYLOAD_VELOCITY_KD
            * (payload_target.linear_velocity - payload_linear_velocity)
        )
        joint_torque += (
            payload_position_jacobian[:, layout.joint_dof_addresses].T
            @ payload_force
        )
    data.ctrl[layout.joint_actuator_ids] = joint_torque


def configuration_max_error(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def resolve_mouse_obstacle(
    mujoco,
    model,
    settings: ReplanningSettings,
) -> MouseObstacleRuntime:
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        MOUSE_OBSTACLE_BODY_NAME,
    )
    if body_id < 0:
        raise ValueError("MuJoCo model is missing the replanning obstacle body")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError("replanning obstacle body is not a MuJoCo mocap body")
    return MouseObstacleRuntime(
        body_id=int(body_id),
        mocap_id=mocap_id,
        collision_radius_m=settings.mouse_obstacle_collision_radius_m,
    )


def select_mouse_obstacle(viewer, runtime: MouseObstacleRuntime) -> None:
    with viewer.lock():
        viewer.perturb.select = runtime.body_id
        viewer.perturb.flexselect = -1
        viewer.perturb.skinselect = -1
        viewer.perturb.localpos[:] = 0.0
    viewer.sync()


def snapshot_replanning_environment(
    environment: dict[str, list],
    data,
    viewer,
    runtime: MouseObstacleRuntime,
) -> tuple[dict[str, list], list[float]]:
    with viewer.lock():
        position = [
            float(value) for value in data.mocap_pos[runtime.mocap_id]
        ]
    # Preserve every static world primitive and replace the red obstacle's
    # initial pose with its current mocap pose for this planning request.
    snapshot = copy.deepcopy(environment)
    snapshot["sphere"] = [
        sphere
        for sphere in snapshot["sphere"]
        if sphere.get("name") != MOUSE_OBSTACLE_NAME
    ]
    snapshot["sphere"].append(
        {
            "name": MOUSE_OBSTACLE_NAME,
            "position": position,
            "radius": runtime.collision_radius_m,
        }
    )
    return snapshot, position


def dynamic_obstacle_y_offset(elapsed_sec: float) -> float:
    """Move spawn (maximum offset) -> minimum, then shuttle with holds."""
    minimum = DYNAMIC_OBSTACLE_Y_MIN_OFFSET_M
    maximum = DYNAMIC_OBSTACLE_Y_MAX_OFFSET_M
    speed = DYNAMIC_OBSTACLE_Y_SPEED_M_PER_SEC
    hold_sec = DYNAMIC_OBSTACLE_ENDPOINT_HOLD_SEC
    span = maximum - minimum
    travel_sec = span / speed
    time_sec = max(elapsed_sec, 0.0)

    # The configured spawn is the maximum-offset endpoint. Move immediately
    # toward -Y; endpoint holds begin after reaching the first endpoint.
    if time_sec < travel_sec:
        return maximum - speed * time_sec

    phase = math.fmod(
        time_sec - travel_sec,
        2.0 * (hold_sec + travel_sec),
    )
    if phase < hold_sec:
        return minimum
    phase -= hold_sec
    if phase < travel_sec:
        return minimum + speed * phase
    phase -= travel_sec
    if phase < hold_sec:
        return maximum
    phase -= hold_sec
    return maximum - speed * phase


def update_dynamic_obstacle_position(
    data,
    viewer,
    runtime: MouseObstacleRuntime,
    center: tuple[float, float, float],
    elapsed_sec: float,
) -> list[float]:
    position = [float(value) for value in center]
    position[1] += dynamic_obstacle_y_offset(elapsed_sec)
    with viewer.lock():
        data.mocap_pos[runtime.mocap_id] = position
    return position


def replanning_problem(
    start: list[float],
    goal: list[float],
    environment: dict[str, list],
    constraints: dict,
) -> dict:
    problem = {
        "valid": True,
        "start": list(start),
        "goals": [list(goal)],
        "axis_endpoints": {
            "start": list(start),
            "goals": [list(goal)],
        },
        "sphere": copy.deepcopy(environment["sphere"]),
        "cylinder": copy.deepcopy(environment["cylinder"]),
        "box": copy.deepcopy(environment["box"]),
        "constraints": copy.deepcopy(constraints),
    }
    return {"problems": {REPLAN_PROBLEM_NAME: [problem]}}


def replanner_server_command(
    settings: ReplanningSettings,
    problem_path: Path,
) -> list[str]:
    command = [
        str(settings.planner_executable),
        "g1",
        REPLAN_PROBLEM_NAME,
        "1",
        "--problem-file",
        str(problem_path),
        "--replan-server",
        "--no-print-path",
        "--seed",
        str(settings.base_seed),
    ]
    if settings.aorrtc:
        command.extend(("--aorrtc", "--time", str(settings.time_limit_sec)))
    if not settings.projection_smoothness:
        command.append("--no-waypoint-smoothing")
    if settings.axis:
        command.append("--axis")
    return command


class PersistentReplanner:
    def __init__(
        self,
        settings: ReplanningSettings,
        start: list[float],
        goal: list[float],
        environment: dict[str, list],
        constraints: dict,
    ) -> None:
        self.settings = settings
        self._initial_problem = replanning_problem(
            start, goal, environment, constraints
        )
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._directory: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._pending_request_index: int | None = None
        self._output_buffer = bytearray()
        self._response_queue: list[dict] = []
        self._log_lines: list[str] = []
        self._protocol_error: str | None = None

    def __enter__(self) -> PersistentReplanner:
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="g1_replan_server_"
        )
        self._directory = Path(self._temporary_directory.name)
        problem_path = self._directory / "problem.json"
        problem_path.write_text(
            json.dumps(self._initial_problem, indent=2) + "\n",
            encoding="utf-8",
        )
        self._process = subprocess.Popen(
            replanner_server_command(self.settings, problem_path),
            cwd=REPOSITORY_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise RuntimeError("failed to open persistent replanner pipes")
        os.set_blocking(self._process.stdout.fileno(), False)
        planner_name = "AORRTC" if self.settings.aorrtc else "PATACON"
        print(
            f"Persistent {planner_name} G1 replanner를 시작했습니다; "
            "CUDA warmup은 이 프로세스에서 한 번만 실행합니다."
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def request_pending(self) -> bool:
        return self._pending_request_index is not None

    def log_tail(self, line_count: int = 12) -> str:
        self._drain_output()
        return "\n".join(self._log_lines[-line_count:])

    def _record_log_line(self, line: bytes) -> None:
        self._log_lines.append(line.decode("utf-8", errors="replace"))
        if len(self._log_lines) > 200:
            del self._log_lines[:-200]

    def _drain_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        while True:
            try:
                chunk = os.read(process.stdout.fileno(), 65536)
            except BlockingIOError:
                break
            except OSError:
                break
            if not chunk:
                break
            self._output_buffer.extend(chunk)

        while True:
            newline = self._output_buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(self._output_buffer[:newline]).rstrip(b"\r")
            del self._output_buffer[: newline + 1]
            if not line.startswith(REPLAN_RESPONSE_PREFIX):
                self._record_log_line(line)
                continue
            try:
                response = json.loads(line[len(REPLAN_RESPONSE_PREFIX) :])
                if not isinstance(response, dict):
                    raise ValueError("response is not a JSON object")
                self._response_queue.append(response)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                self._protocol_error = (
                    f"invalid replanner pipe response: {error}"
                )

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            deadline = time.perf_counter() + 2.0
            while process.poll() is None and time.perf_counter() < deadline:
                self._drain_output()
                time.sleep(0.01)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self._drain_output()
        if process is not None and process.stdout is not None:
            process.stdout.close()
        self._process = None
        self._pending_request_index = None
        self._response_queue.clear()
        self._protocol_error = None
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
        self._directory = None

    def submit(
        self,
        start: list[float],
        goal: list[float],
        environment: dict[str, list],
        request_index: int,
    ) -> bool:
        if self._directory is None or self._process is None:
            raise RuntimeError("persistent replanner is not running")
        if self.request_pending:
            raise RuntimeError("persistent replanner already has a request")
        if self._process.poll() is not None or self._process.stdin is None:
            print("Persistent replanner가 종료되었습니다.\n" + self.log_tail())
            return False

        request = {
            "request_index": request_index,
            "start": list(start),
            "goal": list(goal),
            "environment": copy.deepcopy(environment),
        }
        try:
            self._process.stdin.write(
                (json.dumps(request) + "\n").encode("utf-8")
            )
            self._process.stdin.flush()
        except BrokenPipeError:
            print("Persistent replanner와의 연결이 종료되었습니다.\n" + self.log_tail())
            return False

        self._pending_request_index = request_index
        return True

    def poll(self) -> tuple[int, dict] | None:
        if not self.request_pending:
            return None
        if self._pending_request_index is None or self._process is None:
            raise RuntimeError("persistent replanner request state is incomplete")

        self._drain_output()
        request_index = self._pending_request_index
        if self._protocol_error is not None:
            result = {
                "solved": False,
                "server_error": self._protocol_error,
            }
            self._protocol_error = None
        elif self._response_queue:
            result = self._response_queue.pop(0)
            response_index = result.get("request_index")
            if response_index != request_index:
                result = {
                    "solved": False,
                    "server_error": (
                        "replanner pipe response index mismatch: "
                        f"expected {request_index}, received {response_index}"
                    ),
                }
        elif self._process.poll() is None:
            return None
        else:
            result = {
                "solved": False,
                "server_error": (
                    "persistent replanner exited without a pipe response\n"
                    + self.log_tail()
                ),
            }

        self._pending_request_index = None
        return request_index, result


def parse_replanned_result(
    result: dict,
    settings: ReplanningSettings,
    requested_start: list[float],
    goal: list[float],
    request_index: int,
) -> ReplannedPath | None:
    if "server_error" in result:
        print(
            f"Replan #{request_index}: persistent server 오류: "
            f"{result['server_error']}"
        )
        return None
    if not bool(result.get("solved", False)):
        print(f"Replan #{request_index}: 해를 찾지 못했습니다.")
        return None
    raw_path = result.get("path_start_to_goal")
    if not isinstance(raw_path, list) or len(raw_path) < 2:
        print(f"Replan #{request_index}: 유효한 경로가 없습니다.")
        return None
    path = [
        finite_vector(value, G1_CONFIGURATION_DIMENSION, f"replan waypoint {index}")
        for index, value in enumerate(raw_path)
    ]
    if configuration_max_error(path[0], requested_start) > REPLAN_START_TOLERANCE:
        print(f"Replan #{request_index}: 경로와 요청 start가 다릅니다.")
        return None
    if configuration_max_error(path[-1], goal) > REPLAN_START_TOLERANCE:
        print(f"Replan #{request_index}: 반환 경로가 지정 goal에서 끝나지 않습니다.")
        return None
    timing_key = "planning_ns" if settings.aorrtc else "kernel_ns"
    timing_value = result.get(timing_key)
    planning_seconds = None
    if isinstance(timing_value, (int, float)) and not isinstance(
        timing_value, bool
    ):
        candidate_seconds = float(timing_value) / 1.0e9
        if math.isfinite(candidate_seconds) and candidate_seconds >= 0.0:
            planning_seconds = candidate_seconds
    seed = result.get("seed", "unknown")
    planning_description = (
        f"{planning_seconds:.4f}" if planning_seconds is not None else "unknown"
    )
    print(
        f"Replan #{request_index}: solved candidate, "
        f"seed={seed}, edges={len(path) - 1}, "
        f"planning_s={planning_description}"
    )
    return ReplannedPath(path, planning_seconds)


class VelocityContinuousPathFollower:
    def __init__(
        self,
        path: list[list[float]],
        current: list[float],
        fps: float,
        speed: float,
    ) -> None:
        self.fps = fps
        self.speed = speed
        self.current = list(current)
        self.path: list[list[float]] = []
        self.target_index = 1
        self.frame_index = 0
        self.frame_count = 1
        self.segment_duration = 1.0 / fps
        self.segment_start = list(current)
        self.segment_start_velocity = [0.0] * G1_CONFIGURATION_DIMENSION
        self.segment_end_velocity = [0.0] * G1_CONFIGURATION_DIMENSION
        self.velocity = [0.0] * G1_CONFIGURATION_DIMENSION
        self.replace(path, current, self.velocity)

    @property
    def reached(self) -> bool:
        return self.target_index >= len(self.path)

    def replace(
        self,
        path: list[list[float]],
        current: list[float],
        initial_velocity: list[float],
    ) -> None:
        normalized = [list(current)]
        for waypoint in path[1:]:
            if configuration_max_error(normalized[-1], waypoint) > (
                REPLAN_PROGRESS_TOLERANCE
            ):
                normalized.append(list(waypoint))
        self.path = normalized
        self.current = list(current)
        self.velocity = list(initial_velocity)
        self.target_index = 1
        self._begin_edge()

    def hold(self, current: list[float]) -> None:
        self.path = [list(current)]
        self.current = list(current)
        self.velocity = [0.0] * G1_CONFIGURATION_DIMENSION
        self.target_index = 1
        self._begin_edge()

    def _edge_duration(self, start: list[float], target: list[float]) -> float:
        maximum_change = configuration_max_error(start, target)
        return max(
            2.0 / self.fps,
            maximum_change / self.speed,
            math.sqrt(
                6.0 * maximum_change / REPLANNING_ACCELERATION_LIMIT
            ),
        )

    def _target_velocity(self, target: list[float]) -> list[float]:
        if self.target_index + 1 >= len(self.path):
            return [0.0] * G1_CONFIGURATION_DIMENSION
        next_target = self.path[self.target_index + 1]
        next_duration = self._edge_duration(target, next_target)
        total_duration = self.segment_duration + next_duration
        velocity = [0.0] * G1_CONFIGURATION_DIMENSION
        for index in range(G1_CONFIGURATION_DIMENSION):
            incoming = (
                target[index] - self.segment_start[index]
            ) / self.segment_duration
            outgoing = (next_target[index] - target[index]) / next_duration
            if incoming * outgoing > 0.0:
                velocity[index] = (
                    next_target[index] - self.segment_start[index]
                ) / total_duration
        return velocity

    def _peak_segment_acceleration(
        self,
        target: list[float],
        end_velocity: list[float],
    ) -> float:
        inverse_duration = 1.0 / self.segment_duration
        inverse_duration_squared = inverse_duration * inverse_duration
        peak = 0.0
        for index in range(G1_CONFIGURATION_DIMENSION):
            displacement = target[index] - self.segment_start[index]
            start_acceleration = (
                6.0 * displacement * inverse_duration_squared
                - (
                    4.0 * self.segment_start_velocity[index]
                    + 2.0 * end_velocity[index]
                )
                * inverse_duration
            )
            end_acceleration = (
                -6.0 * displacement * inverse_duration_squared
                + (
                    2.0 * self.segment_start_velocity[index]
                    + 4.0 * end_velocity[index]
                )
                * inverse_duration
            )
            peak = max(
                peak,
                abs(start_acceleration),
                abs(end_acceleration),
            )
        return peak

    def _peak_segment_velocity(
        self,
        target: list[float],
        end_velocity: list[float],
    ) -> float:
        peak = 0.0
        duration = self.segment_duration
        for index in range(G1_CONFIGURATION_DIMENSION):
            displacement = target[index] - self.segment_start[index]
            start_velocity = self.segment_start_velocity[index]
            target_velocity = end_velocity[index]
            coefficient_a = (
                -2.0 * displacement
                + duration * (start_velocity + target_velocity)
            )
            coefficient_b = (
                3.0 * displacement
                - duration * (2.0 * start_velocity + target_velocity)
            )
            coefficient_c = duration * start_velocity
            ratios = [0.0, 1.0]
            if abs(coefficient_a) > 1.0e-12:
                stationary_ratio = -coefficient_b / (3.0 * coefficient_a)
                if 0.0 < stationary_ratio < 1.0:
                    ratios.append(stationary_ratio)
            for ratio in ratios:
                velocity = (
                    3.0 * coefficient_a * ratio * ratio
                    + 2.0 * coefficient_b * ratio
                    + coefficient_c
                ) / duration
                peak = max(peak, abs(velocity))
        return peak

    def _begin_edge(self) -> None:
        self.frame_index = 0
        self.segment_start = list(self.current)
        self.segment_start_velocity = list(self.velocity)
        if self.reached:
            self.frame_count = 1
            self.segment_duration = 1.0 / self.fps
            self.segment_end_velocity = [0.0] * G1_CONFIGURATION_DIMENSION
            self.velocity = list(self.segment_end_velocity)
            return
        target = self.path[self.target_index]
        requested_duration = self._edge_duration(self.segment_start, target)
        for _ in range(12):
            self.frame_count = max(2, math.ceil(requested_duration * self.fps))
            self.segment_duration = self.frame_count / self.fps
            self.segment_end_velocity = self._target_velocity(target)
            peak_velocity = self._peak_segment_velocity(
                target, self.segment_end_velocity
            )
            peak_acceleration = self._peak_segment_acceleration(
                target, self.segment_end_velocity
            )
            duration_scale = max(
                peak_velocity / self.speed,
                math.sqrt(
                    peak_acceleration / REPLANNING_ACCELERATION_LIMIT
                ),
            )
            if duration_scale <= 1.0 + 1.0e-12:
                break
            requested_duration = self.segment_duration * duration_scale
        else:
            raise RuntimeError(
                "failed to enforce replanning path velocity/acceleration limits"
            )

    def advance(self) -> list[float]:
        if self.reached:
            return list(self.current)
        target = self.path[self.target_index]
        self.frame_index += 1
        ratio = min(1.0, self.frame_index / self.frame_count)
        ratio_squared = ratio * ratio
        ratio_cubed = ratio_squared * ratio
        h00 = 2.0 * ratio_cubed - 3.0 * ratio_squared + 1.0
        h10 = ratio_cubed - 2.0 * ratio_squared + ratio
        h01 = -2.0 * ratio_cubed + 3.0 * ratio_squared
        h11 = ratio_cubed - ratio_squared
        self.current = [
            h00 * self.segment_start[index]
            + h10 * self.segment_duration * self.segment_start_velocity[index]
            + h01 * target[index]
            + h11 * self.segment_duration * self.segment_end_velocity[index]
            for index in range(G1_CONFIGURATION_DIMENSION)
        ]
        self.velocity = [
            (6.0 * ratio_squared - 6.0 * ratio)
            / self.segment_duration
            * self.segment_start[index]
            + (3.0 * ratio_squared - 4.0 * ratio + 1.0)
            * self.segment_start_velocity[index]
            + (-6.0 * ratio_squared + 6.0 * ratio)
            / self.segment_duration
            * target[index]
            + (3.0 * ratio_squared - 2.0 * ratio)
            * self.segment_end_velocity[index]
            for index in range(G1_CONFIGURATION_DIMENSION)
        ]
        if self.frame_index >= self.frame_count:
            self.current = list(target)
            self.velocity = list(self.segment_end_velocity)
            self.target_index += 1
            self._begin_edge()
        return list(self.current)


@dataclass
class ScheduledPathReplacement:
    handoff_frame: int
    request_index: int
    follower: VelocityContinuousPathFollower
    object_points: list[np.ndarray]
    planning_time_sec: float | None


@dataclass(frozen=True)
class ScheduledReplanContext:
    commit_index: int
    start: list[float]
    velocity: list[float]
    goal: list[float]
    handoff_frame: int
    actual_state_start: bool
    submitted_frame: int


def path_reaches_goal_within_frames(
    follower: VelocityContinuousPathFollower,
    goal: list[float],
    frame_count: int,
) -> bool:
    if configuration_max_error(follower.path[-1], goal) > (
        FINAL_GOAL_NODE_TOLERANCE
    ):
        return False
    predicted = copy.deepcopy(follower)
    for _ in range(max(0, frame_count)):
        if predicted.reached:
            return True
        predicted.advance()
    return predicted.reached


def predict_scheduled_path_state(
    follower: VelocityContinuousPathFollower,
    current_frame: int,
    minimum_target_frame: int,
    goal: list[float],
    final_approach_frame_count: int,
) -> tuple[list[float], list[float], int, bool]:
    """Predict the commanded state at the latency-safe handoff frame.

    A handoff used to wait for the first planner waypoint after the minimum
    lookahead.  Replanning edges are commonly longer than one second, so that
    policy hid the planner's actual latency and capped path updates near 1 Hz.
    The follower already supplies a continuous position and velocity along
    each edge; using that state lets the replacement preserve both at a much
    earlier handoff.
    """
    predicted = copy.deepcopy(follower)
    frame = current_frame
    while frame < minimum_target_frame:
        frame += 1
        predicted.advance()
    return (
        list(predicted.current),
        list(predicted.velocity),
        frame,
        path_reaches_goal_within_frames(
            predicted,
            goal,
            final_approach_frame_count,
        ),
    )


def candidate_path_from_current(
    candidate: list[list[float]],
    current: list[float],
) -> list[list[float]] | None:
    nearest_index = min(
        range(len(candidate)),
        key=lambda index: configuration_max_error(candidate[index], current),
    )
    if configuration_max_error(candidate[nearest_index], current) > (
        CONTINUOUS_CANDIDATE_MAX_START_ERROR
    ):
        return None
    suffix = [list(waypoint) for waypoint in candidate[nearest_index:]]
    if suffix and configuration_max_error(suffix[0], current) <= (
        REPLAN_PROGRESS_TOLERANCE
    ):
        suffix = suffix[1:]
    return [list(current)] + suffix


def decimate_sequence(values: list, max_count: int) -> list:
    if max_count <= 0 or len(values) <= max_count:
        return list(values)
    if max_count == 1:
        return [values[0]]
    last_index = len(values) - 1
    return [
        values[int(round(index * last_index / (max_count - 1)))]
        for index in range(max_count)
    ]


def sampled_follower_configurations(
    follower: VelocityContinuousPathFollower,
    max_count: int = OBJECT_PATH_PREVIEW_MAX_POINTS,
) -> list[list[float]]:
    predicted = copy.deepcopy(follower)
    configurations = [list(predicted.current)]
    frame_limit = 100_000
    for _ in range(frame_limit):
        if predicted.reached:
            break
        configurations.append(predicted.advance())
    else:
        raise RuntimeError("object-path preview exceeded its frame limit")
    return decimate_sequence(configurations, max_count)


def payload_path_points(
    mujoco,
    model,
    preview_data,
    base_qpos_address: int,
    joint_qpos_addresses: list[int],
    payload_body_id: int,
    configurations: list[list[float]],
) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for configuration in configurations:
        apply_configuration(
            mujoco,
            model,
            preview_data,
            base_qpos_address,
            joint_qpos_addresses,
            configuration,
        )
        point = np.asarray(preview_data.xpos[payload_body_id]).copy()
        if not points or np.linalg.norm(point - points[-1]) > 1.0e-6:
            points.append(point)
    return points


def append_scene_connector(
    mujoco,
    scene,
    start,
    goal,
    radius: float,
    rgba,
) -> None:
    start_position = np.asarray(start, dtype=np.float64)
    goal_position = np.asarray(goal, dtype=np.float64)
    if (
        scene.ngeom >= len(scene.geoms)
        or np.linalg.norm(goal_position - start_position) < 1.0e-8
    ):
        return
    color = np.asarray(rgba, dtype=np.float32)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        color,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        start_position,
        goal_position,
    )
    geom.rgba[:] = color
    scene.ngeom += 1


def draw_object_path_overlay(
    mujoco,
    viewer,
    active_points: list[np.ndarray],
    actual_points: list[np.ndarray],
) -> None:
    scene = viewer.user_scn
    if scene is None:
        return
    scene.ngeom = 0
    paths = [
        (active_points, OBJECT_PATH_ACTIVE_RGBA),
        (actual_points, OBJECT_PATH_ACTUAL_RGBA),
    ]
    nonempty_path_count = sum(len(points) >= 2 for points, _ in paths)
    if nonempty_path_count == 0 or len(scene.geoms) == 0:
        return
    points_per_path = max(
        2,
        len(scene.geoms) // nonempty_path_count + 1,
    )
    for points, color in paths:
        visible_points = decimate_sequence(points, points_per_path)
        for start, goal in zip(visible_points, visible_points[1:]):
            append_scene_connector(
                mujoco,
                scene,
                start,
                goal,
                OBJECT_PATH_LINE_RADIUS_M,
                color,
            )


def enable_contact_visualization(mujoco, option) -> None:
    """Show contact points without contact-force arrows in replanning views."""
    option.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT)] = True
    option.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE)] = False


class OffscreenReplanningVideo:
    """Viewer-compatible target that streams replanning frames to MP4."""

    def __init__(
        self,
        mujoco,
        model,
        data,
        output_path: Path,
        views: tuple[str, ...],
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.views = views
        configure_model_render_quality(model)
        model.vis.global_.offwidth = max(
            int(model.vis.global_.offwidth), width
        )
        model.vis.global_.offheight = max(
            int(model.vis.global_.offheight), height
        )
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.writer = MultiViewVideoWriter(
            output_path,
            views,
            width,
            height,
            fps,
        )
        self.cameras = {}
        for view in views:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = REPLANNING_CAMERA_LOOKAT
            camera.distance = 2.8
            # Match the interactive viewer's front-facing camera.  The old
            # 315-degree base was exactly opposite and recorded the robot's
            # back for the "front" view.
            camera.azimuth = video_view_azimuth(135.0, view)
            camera.elevation = video_view_elevation(-18.0, view)
            self.cameras[view] = camera
        self.opt = mujoco.MjvOption()
        enable_contact_visualization(mujoco, self.opt)
        # These fields mirror the passive viewer interface used by the
        # replanning loop. The recording cameras above remain deterministic.
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.perturb = mujoco.MjvPerturb()
        mujoco.mjv_defaultPerturb(self.perturb)
        self.user_scn = mujoco.MjvScene(model, maxgeom=1024)
        self._planning_time_overlay = ""

    @property
    def frame_count(self) -> int:
        return self.writer.frame_count

    @property
    def output_paths(self) -> dict[str, Path]:
        return self.writer.output_paths

    def __enter__(self) -> OffscreenReplanningVideo:
        try:
            self.writer.__enter__()
        except Exception:
            self.renderer.close()
            raise
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        try:
            return self.writer.__exit__(exception_type, exception, traceback)
        finally:
            self.renderer.close()

    def lock(self):
        return contextlib.nullcontext()

    def is_running(self) -> bool:
        return True

    def set_active_path_planning_time(
        self,
        path_label: str,
        planning_time_sec: float | None,
    ) -> None:
        timing = (
            f"{planning_time_sec:.4f} s"
            if planning_time_sec is not None
            else "unavailable"
        )
        self._planning_time_overlay = (
            f"{path_label}\nPlanning time: {timing}"
        )

    def _append_user_geometries(self) -> None:
        scene = self.renderer.scene
        for index in range(self.user_scn.ngeom):
            if scene.ngeom >= len(scene.geoms):
                break
            source = self.user_scn.geoms[index]
            target = scene.geoms[scene.ngeom]
            self.mujoco.mjv_initGeom(
                target,
                source.type,
                np.asarray(source.size, dtype=np.float64),
                np.asarray(source.pos, dtype=np.float64),
                np.asarray(source.mat, dtype=np.float64).reshape(-1),
                np.asarray(source.rgba, dtype=np.float32),
            )
            scene.ngeom += 1

    def sync(self) -> None:
        for view, camera in self.cameras.items():
            self.renderer.update_scene(
                self.data,
                camera=camera,
                scene_option=self.opt,
            )
            self._append_user_geometries()
            frame = self.renderer.render()
            if self._planning_time_overlay:
                frame = add_video_text_overlay(
                    frame,
                    self._planning_time_overlay,
                )
            self.writer.write(view, frame)


def save_snapshot(
    model_path: Path,
    waypoints: list[list[float]],
    environment: dict[str, list],
    payload: dict | None,
    output_path: Path,
    waypoint_index: int,
) -> None:
    try:
        import mujoco
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo and Pillow are required for PNG snapshots"
        ) from error

    model = build_control_model(mujoco, model_path, environment, payload)
    data = mujoco.MjData(model)
    base_address, joint_addresses = resolve_model_layout(mujoco, model)
    validate_joint_limits(mujoco, model, waypoints)
    selected_index = (
        len(waypoints) // 2 if waypoint_index == -1 else waypoint_index
    )
    if selected_index < 0 or selected_index >= len(waypoints):
        raise ValueError(
            f"snapshot waypoint {selected_index} is outside "
            f"[0, {len(waypoints) - 1}]"
        )
    apply_configuration(
        mujoco,
        model,
        data,
        base_address,
        joint_addresses,
        waypoints[selected_index],
    )

    payload_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "planning_payload"
    )
    if payload_body_id < 0:
        raise ValueError("snapshot requires a planning payload")
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = data.xpos[payload_body_id]
    camera.distance = 0.72
    camera.azimuth = 138.0
    camera.elevation = -18.0

    # The stock G1 XML allocates a 640x480 offscreen framebuffer.
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render().copy()
    finally:
        renderer.close()

    image = Image.fromarray(pixels)
    caption_height = 42
    canvas = Image.new(
        "RGB", (image.width, image.height + caption_height), (22, 24, 28)
    )
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (14, image.height + 13),
        f"Attached payload object    waypoint {selected_index}",
        fill=(242, 244, 248),
    )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"saved G1 payload snapshot: {output_path}")


def replay_continuous_replanning(
    model_path: Path,
    waypoints: list[list[float]],
    environment: dict[str, list],
    payload: dict | None,
    goal: list[float],
    constraints: dict,
    settings: ReplanningSettings,
    fps: float,
    speed: float,
    gain_scale: float,
    video_path: Path | None = None,
    video_width: int = 1920,
    video_height: int = 1080,
    video_views: tuple[str, ...] = ("front",),
    initial_planning_time_sec: float | None = None,
) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo Python package is required: python3 -m pip install mujoco"
        ) from error

    if not settings.planner_executable.is_file():
        raise FileNotFoundError(
            f"replanning executable not found: {settings.planner_executable}"
        )
    model = build_control_model(
        mujoco,
        model_path,
        environment,
        payload,
        settings,
    )
    data = mujoco.MjData(model)
    layout = resolve_control_layout(mujoco, model)
    mouse_obstacle = resolve_mouse_obstacle(mujoco, model, settings)
    configure_control_damping(model, layout, gain_scale)
    validate_joint_limits(mujoco, model, waypoints)
    initialize_floating_base_from_feet(
        mujoco,
        model,
        data,
        layout.base_qpos_address,
        list(layout.joint_qpos_addresses),
        waypoints[0],
    )
    balance_reference = make_balance_reference(model, data, layout)

    home = list(waypoints[0])
    outward_goal = list(goal)
    active_goal = list(outward_goal)
    moving_outward = True
    current_command = list(home)
    follower = VelocityContinuousPathFollower(
        waypoints,
        current_command,
        fps,
        speed,
    )
    payload_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "planning_payload",
    )
    preview_data = mujoco.MjData(model) if payload_body_id >= 0 else None
    preview_joint_addresses = list(layout.joint_qpos_addresses)

    def object_points_for_configurations(
        configurations: list[list[float]],
    ) -> list[np.ndarray]:
        if preview_data is None or payload_body_id < 0:
            return []
        return payload_path_points(
            mujoco,
            model,
            preview_data,
            layout.base_qpos_address,
            preview_joint_addresses,
            payload_body_id,
            configurations,
        )

    def object_points_for_follower(
        path_follower: VelocityContinuousPathFollower,
    ) -> list[np.ndarray]:
        return object_points_for_configurations(
            sampled_follower_configurations(path_follower),
        )

    frame_period = 1.0 / fps
    control_substeps = max(
        1,
        math.ceil(frame_period / float(model.opt.timestep)),
    )

    def step_ctrl(
        reference: list[float],
        reference_velocity: list[float] | np.ndarray | None = None,
    ) -> None:
        payload_target = make_payload_control_target(
            mujoco,
            model,
            preview_data,
            layout,
            payload_body_id,
            reference,
            reference_velocity,
        )
        for _ in range(control_substeps):
            apply_ctrl_reference(
                mujoco,
                model,
                data,
                layout,
                balance_reference,
                reference,
                gain_scale,
                reference_velocity,
                payload_body_id,
                payload_target,
            )
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("G1 ctrl simulation became non-finite")

    planner_name = "AORRTC" if settings.aorrtc else "PATACON"
    print(
        f"G1 continuous replanning: CTRL로 active path를 실행하면서 "
        f"{planner_name}을 반복합니다. 요청 제한시간은 "
        f"{settings.time_limit_sec:.3f}초입니다."
    )
    print(
        "Replanning world: 고정 world obstacle collision은 제거하고 "
        "바닥과 빨간 동적 구 collision만 유지합니다."
    )
    print(
        f"Replanning path speed: {speed:.3f} "
        f"(기본 speed의 {REPLANNING_SPEED_SCALE:.1f}배)"
    )
    print(
        "Tracking control: implicit joint damping + desired joint velocity "
        "feedback, payload world-position feedback, and "
        f"{REPLANNING_ACCELERATION_LIMIT:.1f} coordinate/s^2 edge timing limit."
    )
    print(
        "Moving handoff gate: one tracking checkpoint per "
        f"{HANDOFF_TRACKING_CHECK_INTERVAL} accepted handoffs; actual "
        "joints to track position within "
        f"{HANDOFF_JOINT_POSITION_TOLERANCE_RAD:.3f} rad and "
        "reference velocity within "
        f"{HANDOFF_JOINT_VELOCITY_ERROR_TOLERANCE_RAD_PER_SEC:.3f} rad/s "
        f"for {HANDOFF_TRACKING_STABLE_SEC:.2f}s."
    )
    print(
        "Large-error recovery: live MuJoCo floating-base and joint qpos are "
        "used only when handoff position error reaches "
        f"{HANDOFF_ACTUAL_RECOVERY_POSITION_ERROR_RAD:.3f} rad."
    )
    print(
        f"최소 {CONTINUOUS_REPLAN_LOOKAHEAD_SEC:.2f}초 뒤의 continuous path state를 "
        "재계획 start와 예약 handoff 지점으로 사용하고, 실제 응답 "
        "latency와 tracking 안정화 시간에 맞춰 lookahead를 자동 조정합니다."
    )
    print(
        "Final approach: active path의 남은 재생 시간이 "
        f"{FINAL_APPROACH_REMAINING_PATH_SEC:.2f}초 이하이면 진행 중인 "
        "재계획 결과를 폐기하고 해당 path로 목표까지 실행합니다."
    )
    if video_path is None:
        print(
            "Endpoint에 도달하면 1초간 hold한 뒤 start↔goal 방향을 바꾸어 "
            "계속 재계획합니다."
        )
    else:
        print(
            "Video mode: 첫 goal reference 완료 후 최소 "
            f"{PING_PONG_GOAL_HOLD_SEC:.1f}초간 hold하고, 실제 joint "
            "추종이 안정되면 녹화를 종료합니다."
        )
    print(
        "빨간 구형 장애물은 world Y축으로 스폰 위치 기준 "
        f"{DYNAMIC_OBSTACLE_Y_MIN_OFFSET_M:+.2f}~"
        f"{DYNAMIC_OBSTACLE_Y_MAX_OFFSET_M:+.2f} m를 "
        f"{DYNAMIC_OBSTACLE_Y_SPEED_M_PER_SEC:.2f} m/s로 왕복하며, "
        f"각 끝점에서 {DYNAMIC_OBSTACLE_ENDPOINT_HOLD_SEC:.1f}초 멈춥니다."
    )
    print(
        "동적 장애물 표시/물리 collision 지름="
        f"{2.0 * settings.mouse_obstacle_collision_radius_m:.3f} m, "
        "planner collision 지름="
        f"{2.0 * settings.mouse_obstacle_collision_radius_m:.3f} m."
    )
    if payload_body_id >= 0:
        print(
            "Payload path: 현재부터 handoff까지의 active 구간과 새 replan "
            "구간을 하나의 초록색 path로 연결하고, 실제 이동 trail은 "
            "파란색으로 표시합니다."
        )

    video_target = (
        OffscreenReplanningVideo(
            mujoco,
            model,
            data,
            video_path,
            video_views,
            video_width,
            video_height,
            fps,
        )
        if video_path is not None
        else None
    )
    if video_target is not None:
        video_target.set_active_path_planning_time(
            "Initial path",
            initial_planning_time_sec,
        )
    viewer_context = (
        video_target
        if video_target is not None
        else mujoco.viewer.launch_passive(model, data)
    )
    if video_target is not None:
        print(
            f"Recording {len(video_views)} replanning MP4 view(s) at "
            f"{fps:g} fps ({video_width}x{video_height}): "
            + ", ".join(video_views)
        )
        print("Recording continues through the first goal hold unless G1 falls.")

    def recording_stop_reason() -> str | None:
        if video_target is None:
            return None
        fall_reason = g1_fall_reason(data, layout)
        if fall_reason is not None:
            return f"G1 fall detected ({fall_reason})"
        return None

    with (
        PersistentReplanner(
            settings,
            current_command,
            active_goal,
            environment,
            constraints,
        ) as replanner,
        viewer_context as viewer,
    ):
        with viewer.lock():
            viewer.cam.lookat[:] = REPLANNING_CAMERA_LOOKAT
            viewer.cam.distance = 2.8
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            enable_contact_visualization(mujoco, viewer.opt)
        stop_reason: str | None = None
        settle_deadline = time.perf_counter()
        settle_frames = math.ceil(0.75 / frame_period)
        for _ in range(settle_frames):
            if not viewer.is_running():
                return
            step_ctrl(current_command)
            viewer.sync()
            stop_reason = recording_stop_reason()
            if stop_reason is not None:
                print(f"Recording stopped: {stop_reason}.")
                break
            settle_deadline += frame_period
            time.sleep(max(0.0, settle_deadline - time.perf_counter()))

        request_count = 0
        commit_count = 0
        execution_frame = 0
        lookahead_frames = max(
            1,
            math.ceil(CONTINUOUS_REPLAN_LOOKAHEAD_SEC * fps),
        )
        final_approach_frames = max(
            1,
            math.ceil(FINAL_APPROACH_REMAINING_PATH_SEC * fps),
        )
        handoff_stable_frames_required = max(
            1,
            math.ceil(HANDOFF_TRACKING_STABLE_SEC * fps),
        )
        adaptive_lookahead_frames = lookahead_frames
        pending_context: dict[int, ScheduledReplanContext] = {}
        active_commit: ScheduledReplanContext | None = None
        scheduled_replacement: ScheduledPathReplacement | None = None
        tracking_stable_frames = 0
        handoffs_since_tracking_check = 0
        accepted_handoff_count = 0
        first_handoff_frame: int | None = None
        first_handoff_wall_time: float | None = None
        motion_held = False
        final_approach = False
        goal_hold_until_frame: int | None = None
        active_object_points = object_points_for_follower(follower)
        actual_object_points = (
            [np.asarray(data.xpos[payload_body_id]).copy()]
            if payload_body_id >= 0
            else []
        )
        overlay_dirty = True
        overlay_last_frame = -OBJECT_PATH_OVERLAY_REFRESH_FRAMES

        def refresh_object_path_overlay(force: bool = False) -> None:
            nonlocal overlay_dirty, overlay_last_frame
            if payload_body_id < 0:
                return
            if (
                not force
                and not overlay_dirty
                and execution_frame - overlay_last_frame
                < OBJECT_PATH_OVERLAY_REFRESH_FRAMES
            ):
                return
            with viewer.lock():
                draw_object_path_overlay(
                    mujoco,
                    viewer,
                    active_object_points,
                    actual_object_points,
                )
            overlay_dirty = False
            overlay_last_frame = execution_frame

        refresh_object_path_overlay(force=True)

        def hold_actual_state_for_replan(reason: str) -> None:
            nonlocal active_commit, scheduled_replacement
            nonlocal motion_held, final_approach, tracking_stable_frames
            nonlocal handoffs_since_tracking_check
            nonlocal current_command
            nonlocal active_object_points
            nonlocal overlay_dirty
            actual_configuration = actual_g1_configuration(
                mujoco,
                data,
                layout,
            )
            joint_gaps = np.abs(
                np.asarray(current_command[6:])
                - np.asarray(actual_configuration[6:])
            )
            worst_joint_index = int(np.argmax(joint_gaps))
            active_commit = None
            scheduled_replacement = None
            tracking_stable_frames = 0
            handoffs_since_tracking_check = 0
            current_command = actual_configuration
            follower.hold(current_command)
            motion_held = True
            final_approach = False
            active_object_points = object_points_for_follower(follower)
            overlay_dirty = True
            print(
                f"{reason} 실제 qpos를 CTRL hold와 다음 replanning start로 "
                f"사용합니다 (command_gap={joint_gaps[worst_joint_index]:.6f} "
                f"rad at {G1_ACTUATED_JOINTS[worst_joint_index]})."
            )

        def discard_replan_and_continue(reason: str) -> None:
            nonlocal active_commit, scheduled_replacement, final_approach
            nonlocal active_object_points, overlay_dirty
            active_commit = None
            scheduled_replacement = None
            final_approach = False
            active_object_points = object_points_for_follower(follower)
            overlay_dirty = True
            action = (
                "actual-state hold를 유지합니다."
                if motion_held
                else "기존 active path를 중단 없이 계속 실행합니다."
            )
            print(f"{reason} {action}")

        def update_tracking_stability() -> None:
            nonlocal tracking_stable_frames
            reference_joint_velocity = (
                np.zeros(len(G1_ACTUATED_JOINTS), dtype=np.float64)
                if motion_held
                else np.asarray(follower.velocity[6:])
            )
            joint_error = float(
                np.max(
                    np.abs(
                        np.asarray(current_command[6:])
                        - data.qpos[layout.joint_qpos_addresses]
                    )
                )
            )
            joint_velocity_error = float(
                np.max(
                    np.abs(
                        reference_joint_velocity
                        - data.qvel[layout.joint_dof_addresses]
                    )
                )
            )
            if (
                joint_error <= HANDOFF_JOINT_POSITION_TOLERANCE_RAD
                and joint_velocity_error
                <= HANDOFF_JOINT_VELOCITY_ERROR_TOLERANCE_RAD_PER_SEC
            ):
                tracking_stable_frames += 1
            else:
                tracking_stable_frames = 0

        def try_activate_due_replacement() -> None:
            nonlocal follower, current_command, active_commit
            nonlocal scheduled_replacement, motion_held, final_approach
            nonlocal tracking_stable_frames
            nonlocal handoffs_since_tracking_check
            nonlocal accepted_handoff_count
            nonlocal first_handoff_frame, first_handoff_wall_time
            nonlocal active_object_points
            nonlocal overlay_dirty
            if (
                scheduled_replacement is None
                or scheduled_replacement.handoff_frame > execution_frame
            ):
                return
            replacement = scheduled_replacement
            joint_errors = np.abs(
                np.asarray(replacement.follower.current[6:])
                - data.qpos[layout.joint_qpos_addresses]
            )
            joint_velocity_errors = np.abs(
                np.asarray(replacement.follower.velocity[6:])
                - data.qvel[layout.joint_dof_addresses]
            )
            worst_error_index = int(np.argmax(joint_errors))
            worst_velocity_error_index = int(np.argmax(joint_velocity_errors))
            actual_joint_error = float(joint_errors[worst_error_index])
            actual_joint_velocity_error = float(
                joint_velocity_errors[worst_velocity_error_index]
            )
            tracking_ready = (
                actual_joint_error
                <= HANDOFF_JOINT_POSITION_TOLERANCE_RAD
                and actual_joint_velocity_error
                <= HANDOFF_JOINT_VELOCITY_ERROR_TOLERANCE_RAD_PER_SEC
                and tracking_stable_frames >= handoff_stable_frames_required
            )
            tracking_check_due = (
                handoffs_since_tracking_check
                >= HANDOFF_TRACKING_CHECK_INTERVAL - 1
            )
            if actual_joint_error >= HANDOFF_ACTUAL_RECOVERY_POSITION_ERROR_RAD:
                diagnostic = (
                    f"Scheduled handoff #{replacement.request_index}: moving "
                    f"tracking emergency at frame={execution_frame} "
                    f"(q_error={actual_joint_error:.6f} rad, "
                    f"{G1_ACTUATED_JOINTS[worst_error_index]}, "
                    f"qdot_error={actual_joint_velocity_error:.6f} rad/s, "
                    f"{G1_ACTUATED_JOINTS[worst_velocity_error_index]}, "
                    f"stable={tracking_stable_frames}/"
                    f"{handoff_stable_frames_required})."
                )
                hold_actual_state_for_replan(
                    diagnostic + " 큰 위치 오차이므로 actual-qpos recovery."
                )
                return
            if tracking_check_due and not tracking_ready:
                diagnostic = (
                    f"Scheduled handoff #{replacement.request_index}: "
                    f"{HANDOFF_TRACKING_CHECK_INTERVAL}-handoff tracking "
                    f"checkpoint 불충족 at frame={execution_frame} "
                    f"(q_error={actual_joint_error:.6f} rad, "
                    f"{G1_ACTUATED_JOINTS[worst_error_index]}, "
                    f"qdot_error={actual_joint_velocity_error:.6f} rad/s, "
                    f"{G1_ACTUATED_JOINTS[worst_velocity_error_index]}, "
                    f"stable={tracking_stable_frames}/"
                    f"{handoff_stable_frames_required})."
                )
                discard_replan_and_continue(diagnostic)
                return

            scheduled_replacement = None
            active_commit = None
            position_gap = configuration_max_error(
                current_command,
                replacement.follower.current,
            )
            velocity_gap = configuration_max_error(
                follower.velocity,
                replacement.follower.velocity,
            )
            follower = replacement.follower
            current_command = list(follower.current)
            motion_held = False
            if tracking_check_due:
                handoffs_since_tracking_check = 0
                tracking_gate_status = "checked"
            else:
                handoffs_since_tracking_check += 1
                tracking_gate_status = (
                    f"deferred={handoffs_since_tracking_check}/"
                    f"{HANDOFF_TRACKING_CHECK_INTERVAL - 1}"
                )
            if video_target is not None:
                video_target.set_active_path_planning_time(
                    f"Replan #{replacement.request_index}",
                    replacement.planning_time_sec,
                )
            active_object_points = replacement.object_points
            overlay_dirty = True
            handoff_wall_time = time.perf_counter()
            accepted_handoff_count += 1
            if first_handoff_frame is None:
                first_handoff_frame = execution_frame
                first_handoff_wall_time = handoff_wall_time
            if (
                accepted_handoff_count >= 2
                and first_handoff_frame is not None
                and first_handoff_wall_time is not None
            ):
                simulation_elapsed = (
                    execution_frame - first_handoff_frame
                ) * frame_period
                wall_elapsed = handoff_wall_time - first_handoff_wall_time
                update_intervals = accepted_handoff_count - 1
                simulation_update_hz = (
                    update_intervals / simulation_elapsed
                    if simulation_elapsed > 0.0
                    else float("nan")
                )
                wall_update_hz = (
                    update_intervals / wall_elapsed
                    if wall_elapsed > 0.0
                    else float("nan")
                )
                update_rate_description = (
                    f", update_hz_sim={simulation_update_hz:.3f}, "
                    f"update_hz_wall={wall_update_hz:.3f}"
                )
            else:
                update_rate_description = ", update_hz=warming_up"
            print(
                f"Scheduled handoff #{replacement.request_index}: "
                f"frame={execution_frame}, q_gap={position_gap:.8f}, "
                f"qdot_gap={velocity_gap:.8f}, "
                f"actual_joint_error={actual_joint_error:.6f}, "
                f"actual_joint_velocity_error="
                f"{actual_joint_velocity_error:.6f}, "
                f"tracking_stable_frames={tracking_stable_frames}, "
                f"tracking_gate={tracking_gate_status}"
                f"{update_rate_description}"
            )

        def connected_object_points_for_replacement(
            handoff_frame: int,
            replacement_points: list[np.ndarray],
        ) -> list[np.ndarray]:
            predicted_active = copy.deepcopy(follower)
            prefix_configurations = [list(predicted_active.current)]
            for _ in range(max(0, handoff_frame - execution_frame)):
                if predicted_active.reached:
                    break
                prefix_configurations.append(predicted_active.advance())
            connected = object_points_for_configurations(prefix_configurations)
            for point in replacement_points:
                if (
                    not connected
                    or np.linalg.norm(point - connected[-1]) > 1.0e-6
                ):
                    connected.append(point)
            return connected

        def submit_latest_request() -> bool:
            nonlocal request_count, commit_count, active_commit, final_approach
            nonlocal follower, current_command
            nonlocal active_object_points, overlay_dirty
            if final_approach or goal_hold_until_frame is not None:
                return False
            if active_commit is None:
                if (
                    not motion_held
                    and path_reaches_goal_within_frames(
                        follower,
                        active_goal,
                        final_approach_frames,
                    )
                ):
                    final_approach = True
                    print(
                        "Final approach: 현재 active path가 "
                        f"{FINAL_APPROACH_REMAINING_PATH_SEC:.2f}초 이내 "
                        "목표에 도달하므로 추가 재계획을 중단합니다."
                    )
                    return False
                if motion_held:
                    actual_start = actual_g1_configuration(
                        mujoco,
                        data,
                        layout,
                    )
                    resnapshot_joint_gap = configuration_max_error(
                        current_command[6:],
                        actual_start[6:],
                    )
                    current_command = actual_start
                    follower.hold(current_command)
                    active_object_points = object_points_for_follower(follower)
                    overlay_dirty = True
                    predicted_start = list(actual_start)
                    predicted_velocity = [0.0] * G1_CONFIGURATION_DIMENSION
                    handoff_frame = execution_frame
                    final_approach_ready = False
                    actual_state_start = True
                    print(
                        "Actual-state recovery replan: live floating-base와 "
                        "joint qpos를 start로 캡처했습니다 "
                        f"(resnapshot_joint_gap={resnapshot_joint_gap:.6f} rad)."
                    )
                else:
                    (
                        predicted_start,
                        predicted_velocity,
                        handoff_frame,
                        final_approach_ready,
                    ) = predict_scheduled_path_state(
                        follower,
                        execution_frame,
                        execution_frame + adaptive_lookahead_frames,
                        active_goal,
                        final_approach_frames,
                    )
                    actual_state_start = False
                if final_approach_ready:
                    final_approach = True
                    print(
                        "Final approach: 예약 handoff부터 목표까지의 남은 "
                        f"재생 시간이 {FINAL_APPROACH_REMAINING_PATH_SEC:.2f}초 "
                        "이하이므로 추가 재계획을 중단합니다."
                    )
                    return False
                commit_count += 1
                active_commit = ScheduledReplanContext(
                    commit_index=commit_count,
                    start=list(predicted_start),
                    velocity=list(predicted_velocity),
                    goal=list(active_goal),
                    handoff_frame=handoff_frame,
                    actual_state_start=actual_state_start,
                    submitted_frame=execution_frame,
                )

            context = active_commit
            request_environment, obstacle_position = (
                snapshot_replanning_environment(
                    environment,
                    data,
                    viewer,
                    mouse_obstacle,
                )
            )
            request_count += 1
            if not replanner.submit(
                context.start,
                context.goal,
                request_environment,
                request_count,
            ):
                request_count -= 1
                return False
            pending_context[request_count] = context
            start_source = (
                "actual_qpos"
                if context.actual_state_start
                else "predicted_state"
            )
            print(
                f"Continuous replan #{request_count}: "
                f"commit={context.commit_index}, "
                f"start_source={start_source}, "
                f"obstacle_xyz=({obstacle_position[0]:.3f}, "
                f"{obstacle_position[1]:.3f}, {obstacle_position[2]:.3f}), "
                f"handoff_frame={context.handoff_frame}"
            )
            return True

        def enter_final_approach_if_ready() -> bool:
            nonlocal active_commit, final_approach
            if (
                final_approach
                or goal_hold_until_frame is not None
                or motion_held
                or scheduled_replacement is not None
                or not path_reaches_goal_within_frames(
                    follower,
                    active_goal,
                    final_approach_frames,
                )
            ):
                return False
            final_approach = True
            active_commit = None
            print(
                "Final approach: active path의 남은 재생 시간이 "
                f"{FINAL_APPROACH_REMAINING_PATH_SEC:.2f}초 이하입니다. "
                "진행 중인 재계획 결과는 폐기하고 현재 path로 목표까지 "
                "실행합니다."
            )
            return True

        deadline = time.perf_counter()
        while viewer.is_running() and stop_reason is None:
            update_dynamic_obstacle_position(
                data,
                viewer,
                mouse_obstacle,
                settings.mouse_obstacle_initial_position,
                execution_frame * frame_period,
            )
            enter_final_approach_if_ready()
            response = replanner.poll()
            if response is not None:
                request_index, result = response
                context = pending_context.pop(request_index, None)
                if context is None:
                    raise RuntimeError(
                        f"missing context for replan request {request_index}"
                    )
                observed_latency_frames = max(
                    1,
                    execution_frame - context.submitted_frame,
                )
                adaptive_lookahead_frames = max(
                    lookahead_frames,
                    observed_latency_frames
                    + handoff_stable_frames_required
                    + 1,
                )
                candidate = parse_replanned_result(
                    result,
                    settings,
                    context.start,
                    context.goal,
                    request_index,
                )
                response_matches_active_commit = (
                    active_commit is not None
                    and context.commit_index == active_commit.commit_index
                    and configuration_max_error(context.goal, active_goal)
                    <= REPLAN_START_TOLERANCE
                )
                if response_matches_active_commit:
                    if candidate is None:
                        discard_replan_and_continue(
                            "재계획 실패."
                        )
                    elif (
                        context.handoff_frame < execution_frame
                        and not context.actual_state_start
                    ):
                        discard_replan_and_continue(
                            f"재계획 결과 #{request_index}가 handoff보다 늦어 "
                            "폐기합니다."
                        )
                    else:
                        replacement_path = candidate_path_from_current(
                            candidate.waypoints,
                            context.start,
                        )
                        if replacement_path is None:
                            discard_replan_and_continue(
                                "재계획 경로를 handoff state에 연결할 수 없어 "
                                "폐기합니다."
                            )
                        else:
                            handoff_frame = (
                                execution_frame
                                if context.actual_state_start
                                else context.handoff_frame
                            )
                            handoff_velocity = list(context.velocity)
                            staged_follower = VelocityContinuousPathFollower(
                                replacement_path,
                                context.start,
                                fps,
                                speed,
                            )
                            staged_follower.replace(
                                replacement_path,
                                context.start,
                                handoff_velocity,
                            )
                            reaches_goal_after_handoff = (
                                path_reaches_goal_within_frames(
                                    staged_follower,
                                    active_goal,
                                    final_approach_frames,
                                )
                            )
                            replacement_object_points = (
                                object_points_for_follower(staged_follower)
                            )
                            scheduled_replacement = ScheduledPathReplacement(
                                handoff_frame=handoff_frame,
                                request_index=request_index,
                                follower=staged_follower,
                                object_points=replacement_object_points,
                                planning_time_sec=candidate.planning_time_sec,
                            )
                            active_object_points = (
                                connected_object_points_for_replacement(
                                    handoff_frame,
                                    replacement_object_points,
                                )
                            )
                            overlay_dirty = True
                            if reaches_goal_after_handoff:
                                final_approach = True
                                print(
                                    "Final approach 예약: 새 path가 handoff 후 "
                                    f"{FINAL_APPROACH_REMAINING_PATH_SEC:.2f}초 "
                                    "이내 목표에 도달하므로 재생성하지 않습니다."
                                )
                            print(
                                f"Replan #{request_index}: 새 path를 "
                                f"frame={handoff_frame}의 moving tracking "
                                "gate에서 속도를 유지해 "
                                "적용하도록 예약했습니다."
                            )
                else:
                    print(
                        f"Replan #{request_index}: 이미 지난 commit 결과이므로 "
                        "폐기합니다."
                    )

            if (
                goal_hold_until_frame is not None
                and execution_frame >= goal_hold_until_frame
            ):
                if video_target is not None and moving_outward:
                    if (
                        tracking_stable_frames
                        >= handoff_stable_frames_required
                    ):
                        print(
                            f"Goal hold {PING_PONG_GOAL_HOLD_SEC:.1f}초와 "
                            "실제 joint 추종 안정화를 완료했습니다; "
                            "start→goal 녹화를 종료합니다."
                        )
                        break
                else:
                    goal_hold_until_frame = None
                    if moving_outward:
                        active_goal = list(home)
                        moving_outward = False
                        direction = "goal→start"
                    else:
                        active_goal = list(outward_goal)
                        moving_outward = True
                        direction = "start→goal"
                    active_commit = None
                    scheduled_replacement = None
                    tracking_stable_frames = 0
                    handoffs_since_tracking_check = 0
                    final_approach = False
                    follower.hold(current_command)
                    motion_held = True
                    active_object_points = object_points_for_follower(follower)
                    overlay_dirty = True
                    print(
                        f"Goal hold {PING_PONG_GOAL_HOLD_SEC:.1f}초 완료; "
                        f"{direction} 재계획을 시작합니다."
                    )
            elif (
                goal_hold_until_frame is None
                and follower.reached
                and scheduled_replacement is None
                and not motion_held
            ):
                hold_frames = max(
                    1,
                    math.ceil(PING_PONG_GOAL_HOLD_SEC * fps),
                )
                goal_hold_until_frame = execution_frame + hold_frames
                active_commit = None
                scheduled_replacement = None
                tracking_stable_frames = 0
                handoffs_since_tracking_check = 0
                follower.hold(current_command)
                motion_held = True
                final_approach = True
                active_object_points = object_points_for_follower(follower)
                overlay_dirty = True
                endpoint_name = "goal" if moving_outward else "start"
                print(
                    f"G1 {endpoint_name} active path completed after "
                    f"{request_count} continuous replans; 최소 "
                    f"{PING_PONG_GOAL_HOLD_SEC:.1f}초간 CTRL target을 "
                    "유지합니다. 영상 모드에서는 실제 joint 추종이 "
                    "안정될 때까지 녹화를 계속합니다."
                )

            if (
                not replanner.request_pending
                and scheduled_replacement is None
            ):
                submit_latest_request()

            if not motion_held:
                current_command = follower.advance()
            execution_frame += 1
            enter_final_approach_if_ready()

            step_ctrl(
                current_command,
                None if motion_held else follower.velocity,
            )
            update_tracking_stability()
            try_activate_due_replacement()
            if payload_body_id >= 0:
                actual_point = np.asarray(
                    data.xpos[payload_body_id]
                ).copy()
                if (
                    not actual_object_points
                    or np.linalg.norm(actual_point - actual_object_points[-1])
                    >= OBJECT_PATH_TRAIL_MIN_DISTANCE_M
                ):
                    actual_object_points.append(actual_point)
                    if len(actual_object_points) > OBJECT_PATH_TRAIL_MAX_POINTS:
                        latest_actual_point = actual_object_points[-1]
                        actual_object_points = actual_object_points[::2]
                        if not np.array_equal(
                            actual_object_points[-1],
                            latest_actual_point,
                        ):
                            actual_object_points.append(latest_actual_point)
                refresh_object_path_overlay()
            viewer.sync()
            stop_reason = recording_stop_reason()
            if stop_reason is not None:
                print(f"Recording stopped: {stop_reason}.")
                break
            deadline += frame_period
            time.sleep(max(0.0, deadline - time.perf_counter()))

    if video_target is not None:
        print(
            "saved G1 replanning MP4: "
            + ", ".join(str(path) for path in video_target.output_paths.values())
            + f" ({video_target.frame_count} frames each, "
            f"{video_target.frame_count / fps:.3f} s)"
        )


def replay(
    model_path: Path,
    waypoints: list[list[float]],
    environment: dict[str, list],
    payload: dict | None,
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
        model = build_control_model(mujoco, model_path, environment, payload)
    else:
        model = build_kinematic_model(mujoco, model_path, payload)
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
    if control_layout is not None:
        initialize_floating_base_from_feet(
            mujoco,
            model,
            data,
            base_address,
            joint_addresses,
            waypoints[0],
        )
    else:
        apply_configuration(
            mujoco,
            model,
            data,
            base_address,
            joint_addresses,
            waypoints[0],
        )
    balance_reference = (
        make_balance_reference(model, data, control_layout)
        if control_layout is not None
        else None
    )

    print(
        "MuJoCo viewer: G1 start -> goal 경로를 "
        f"{control_mode} 모드로 반복 재생합니다."
    )
    if control_mode == "ctrl":
        print(
            "29개 관절 motor에 접촉력 분배, CoM/base 자세 feedback, "
            "PD torque를 입력합니다."
        )
        print(
            "shelf/obstacle/rubber hand/payload는 실제 MuJoCo contact에 "
            "참여합니다."
        )
    else:
        print(
            "planner의 floating-base 6축과 29개 관절 qpos를 "
            "매 프레임 직접 적용합니다."
        )
    if payload is not None:
        print(
            f"파란색 payload object는 {float(payload['mass_kg']):g} kg이며 "
            "양손 사이에 고정되어 있습니다."
        )
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
        while viewer.is_running():
            mujoco.mj_resetData(model, data)
            if control_layout is not None:
                initialize_floating_base_from_feet(
                    mujoco,
                    model,
                    data,
                    base_address,
                    joint_addresses,
                    waypoints[0],
                )
                balance_reference = make_balance_reference(
                    model, data, control_layout
                )
                # Settle with active physics instead of displaying a frozen
                # initial pose before trajectory tracking starts.
                settle_frames = math.ceil(0.75 / frame_period)
                settle_deadline = time.perf_counter()
                for _ in range(settle_frames):
                    if not viewer.is_running():
                        return
                    for _ in range(control_substeps):
                        apply_ctrl_reference(
                            mujoco,
                            model,
                            data,
                            control_layout,
                            balance_reference,
                            waypoints[0],
                            gain_scale,
                        )
                        mujoco.mj_step(model, data)
                    viewer.sync()
                    settle_deadline += frame_period
                    time.sleep(
                        max(0.0, settle_deadline - time.perf_counter())
                    )
            else:
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
                            balance_reference,
                            configuration,
                            gain_scale,
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
            f"G1 MuJoCo model not found: {model_path}; initialize the "
            "unitree_ros submodule"
        )
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory not found: {trajectory_path}")

    (
        waypoints,
        environment,
        payload,
        goal,
        constraints,
        replanning,
        planning_time_sec,
    ) = load_trajectory(trajectory_path)
    validate_environment(environment)
    validate_payload(payload)
    if args.replanning and replanning is None:
        raise ValueError(
            "--replanning requires metadata generated by single_mbm --replanning"
        )
    if args.replanning and goal is None:
        raise ValueError("--replanning trajectory is missing its goal")
    if args.replanning and not constraints:
        raise ValueError("--replanning trajectory is missing G1 constraints")
    if args.snapshot is not None:
        save_snapshot(
            model_path,
            waypoints,
            environment,
            payload,
            args.snapshot,
            args.snapshot_waypoint,
        )
        return 0
    validate_only = args.validate_only or os.environ.get(
        "PATACON_MUJOCO_VALIDATE_ONLY"
    ) == "1"
    if validate_only:
        import mujoco

        if args.control_mode == "ctrl":
            model = build_control_model(
                mujoco,
                model_path,
                environment,
                payload,
                replanning if args.replanning else None,
            )
        else:
            model = build_kinematic_model(mujoco, model_path, payload)
        data = mujoco.MjData(model)
        base_address, joint_addresses = resolve_model_layout(mujoco, model)
        validate_joint_limits(mujoco, model, waypoints)
        if args.control_mode == "qpos":
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
            initialize_floating_base_from_feet(
                mujoco,
                model,
                data,
                base_address,
                joint_addresses,
                waypoints[0],
            )
            balance_reference = make_balance_reference(
                model, data, control_layout
            )
            for _ in range(10):
                apply_ctrl_reference(
                    mujoco,
                    model,
                    data,
                    control_layout,
                    balance_reference,
                    waypoints[0],
                    args.gain_scale,
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
            f"{primitive_count} environment primitives, "
            f"payload_mass_kg={0.0 if payload is None else float(payload['mass_kg']):g}"
        )
        return 0

    if args.replanning:
        assert goal is not None
        assert replanning is not None
        replanning_speed = args.speed * REPLANNING_SPEED_SCALE
        replay_continuous_replanning(
            model_path,
            waypoints,
            environment,
            payload,
            goal,
            constraints,
            replanning,
            args.fps,
            replanning_speed,
            args.gain_scale,
            args.video,
            args.video_width,
            args.video_height,
            args.video_views,
            planning_time_sec,
        )
        return 0

    replay(
        model_path,
        waypoints,
        environment,
        payload,
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

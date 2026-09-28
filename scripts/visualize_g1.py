#!/usr/bin/env python3
"""Replay a 35-DoF pRRTC G1 trajectory with its planning obstacles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from xml.etree import ElementTree

import numpy as np

from mujoco_video import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    DEFAULT_TRAJECTORY_ACCELERATION,
    DEFAULT_TRAJECTORY_PLAYBACK_RATE,
    GeometricPathWaypoints,
    MultiViewVideoWriter,
    TimeParameterizedTrajectory,
    add_video_text_overlay,
    add_video_view_argument,
    configure_model_render_quality,
    validate_video_views,
    video_view_azimuth,
    video_view_elevation,
    time_parameterize_waypoints as toppra_time_parameterize_waypoints,
    time_parameterized_frames as toppra_time_parameterized_frames,
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
G1_SUPPORT_FOOT_BODIES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
# The planner's foot end-effector frames are constrained to z=0, while the
# MuJoCo sole contact points extend about 35 mm below those frames.
G1_PLANNING_FLOOR_HEIGHT = -0.0351
FRANKA_FLOOR_RGB = (0.48, 0.48, 0.48)
FRANKA_BACKGROUND_RGB = (0.72, 0.72, 0.72)

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

# Contact-aware balance gains.  The support-force solve supplies the ground
# reaction needed by the unactuated floating base; these feedback terms keep
# the projected CoM near the middle of the initial two-foot support and damp
# base orientation drift.
G1_COM_POSITION_KP = 12.0
G1_COM_VELOCITY_KD = 7.0
G1_BASE_ORIENTATION_KP = np.asarray([100.0, 100.0, 30.0])
G1_BASE_ORIENTATION_KD = np.asarray([20.0, 20.0, 8.0])
G1_SUPPORT_FRICTION_COEFFICIENT = 0.8
G1_BASE_VELOCITY_LIMITS = np.asarray(
    [0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
    dtype=np.float64,
)


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


def default_urdf_path() -> Path:
    return default_model_path().with_suffix(".urdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument(
        "--urdf",
        type=Path,
        default=default_urdf_path(),
        help="G1 URDF providing per-joint velocity limits.",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--velocity-scale",
        "--speed",
        dest="velocity_scale",
        type=float,
        default=os.environ.get("PRRTC_G1_VELOCITY_SCALE", "1.0"),
        help=(
            "Scale applied to the 2x-playback URDF joint and floating-base "
            "velocity limits; must be in (0, 1] (default: 1.0). "
            "PRRTC_G1_VELOCITY_SCALE provides the same option."
        ),
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=DEFAULT_TRAJECTORY_ACCELERATION,
        help=(
            "Maximum planning-coordinate acceleration per second squared "
            f"(default: {DEFAULT_TRAJECTORY_ACCELERATION:g})."
        ),
    )
    parser.add_argument(
        "--control-mode",
        choices=("ctrl", "qpos"),
        default=os.environ.get("PRRTC_G1_CONTROL_MODE", "ctrl"),
        help=(
            "ctrl uses torque-PD actuators and physics (default); "
            "qpos directly replays the planned configurations"
        ),
    )
    parser.add_argument(
        "--gain-scale",
        type=float,
        default=8.0,
        help=(
            "Scale all ctrl-mode proportional and derivative gains "
            "(default: 8.0)."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate without opening a viewer window.",
    )
    snapshot_default = os.environ.get("PRRTC_G1_SNAPSHOT")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(snapshot_default) if snapshot_default else None,
        help="Render one close-up frame to a PNG instead of opening a viewer.",
    )
    parser.add_argument(
        "--snapshot-waypoint",
        type=int,
        default=int(os.environ.get("PRRTC_G1_SNAPSHOT_WAYPOINT", "-1")),
        help="Waypoint to render; -1 selects the middle waypoint.",
    )
    video_default = os.environ.get("PRRTC_G1_VIDEO") or os.environ.get(
        "PRRTC_VIDEO"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(video_default) if video_default else None,
        help=(
            "Render one start-to-goal replay directly to an MP4 instead of "
            "opening a viewer. PRRTC_G1_VIDEO or PRRTC_VIDEO provides the "
            "same option."
        ),
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=os.environ.get("PRRTC_VIDEO_WIDTH", str(DEFAULT_VIDEO_WIDTH)),
        help=(
            f"Recorded video width in pixels (default: {DEFAULT_VIDEO_WIDTH}; "
            "PRRTC_VIDEO_WIDTH provides the same option)."
        ),
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=os.environ.get("PRRTC_VIDEO_HEIGHT", str(DEFAULT_VIDEO_HEIGHT)),
        help=(
            f"Recorded video height in pixels (default: {DEFAULT_VIDEO_HEIGHT}; "
            "PRRTC_VIDEO_HEIGHT provides the same option)."
        ),
    )
    add_video_view_argument(parser)
    args = parser.parse_args()
    if args.control_mode not in ("ctrl", "qpos"):
        parser.error("--control-mode must be ctrl or qpos")
    if (
        not math.isfinite(args.fps)
        or not math.isfinite(args.velocity_scale)
        or not math.isfinite(args.acceleration)
        or not math.isfinite(args.gain_scale)
        or args.fps <= 0.0
        or args.velocity_scale <= 0.0
        or args.velocity_scale > 1.0
        or args.acceleration <= 0.0
        or args.gain_scale <= 0.0
    ):
        parser.error(
            "--fps, --acceleration, and --gain-scale must be positive, and "
            "--velocity-scale must be in (0, 1]"
        )
    if args.snapshot is not None and args.video is not None:
        parser.error("--snapshot and --video cannot be used together")
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


def finite_vector(value, dimension: int, description: str) -> list[float]:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{description} must contain {dimension} values")
    normalized = [float(component) for component in value]
    if not all(math.isfinite(component) for component in normalized):
        raise ValueError(f"{description} contains a non-finite value")
    return normalized


def load_planning_velocity_limits(
    urdf_path: Path,
    velocity_scale: float,
) -> np.ndarray:
    if (
        not math.isfinite(velocity_scale)
        or velocity_scale <= 0.0
        or velocity_scale > 1.0
    ):
        raise ValueError("velocity scale must be in (0, 1]")
    if not urdf_path.is_file():
        raise FileNotFoundError(f"G1 URDF not found: {urdf_path}")

    root = ElementTree.parse(urdf_path).getroot()
    urdf_limits: dict[str, float] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None or limit.get("velocity") is None:
            continue
        velocity = float(limit.get("velocity"))
        if not math.isfinite(velocity) or velocity <= 0.0:
            raise ValueError(
                f"URDF joint {name} has an invalid velocity limit: {velocity}"
            )
        urdf_limits[name] = velocity

    missing = [name for name in G1_ACTUATED_JOINTS if name not in urdf_limits]
    if missing:
        raise ValueError(
            "G1 URDF is missing velocity limits for: " + ", ".join(missing)
        )
    return DEFAULT_TRAJECTORY_PLAYBACK_RATE * velocity_scale * np.concatenate(
        (
            G1_BASE_VELOCITY_LIMITS,
            np.asarray(
                [urdf_limits[name] for name in G1_ACTUATED_JOINTS],
                dtype=np.float64,
            ),
        )
    )


def load_trajectory(
    path: Path,
) -> tuple[
    list[list[float]],
    dict[str, list],
    dict | None,
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
    return (
        GeometricPathWaypoints(
            waypoints,
            document.get("geometric_path"),
            document.get("path_smoothing", True),
        ),
        environment,
        payload,
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
            # planner's VAMP sphere-pair table rather than MuJoCo's mesh pairs.
            geom.contype = 1
            geom.conaffinity = 0

    add_physical_environment(mujoco, spec, environment)

    model = spec.compile()
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
    damping_scale = math.sqrt(gain_scale)
    model.dof_damping[layout.joint_dof_addresses] += (
        damping_scale * G1_JOINT_KD
    )


def make_balance_reference(
    model, data, layout: G1ControlLayout
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


def append_environment_geometries(
    mujoco,
    scene,
    environment: dict[str, list],
) -> None:
    primitive_count = sum(
        len(environment[key]) for key in ("sphere", "cylinder", "box")
    )
    if scene.ngeom + primitive_count > len(scene.geoms):
        raise ValueError("too many environment primitives for the MuJoCo scene")

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
        color = (
            [0.85, 0.20, 0.15, 0.65]
            if is_obstacle
            else [0.45, 0.32, 0.18, 0.65]
        )
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(box["half_extents"], dtype=np.float64),
            np.asarray(box["position"], dtype=np.float64),
            euler_xyz_matrix(box["orientation_euler_xyz"]),
            np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1


def add_environment_geometries(
    mujoco,
    viewer,
    environment: dict[str, list],
) -> None:
    viewer.user_scn.ngeom = 0
    append_environment_geometries(mujoco, viewer.user_scn, environment)


def time_parameterize_waypoints(
    waypoints: list[list[float]],
    maximum_velocity: float | list[float] | np.ndarray,
    maximum_acceleration: float | list[float] | np.ndarray,
    fps: float | None = None,
) -> TimeParameterizedTrajectory:
    """Prepare either smoothed TOPP-RA or raw waypoint playback."""
    return toppra_time_parameterize_waypoints(
        waypoints,
        maximum_velocity,
        maximum_acceleration,
        fps,
    )


def time_parameterized_frames(
    trajectory: TimeParameterizedTrajectory,
    fps: float,
):
    yield from toppra_time_parameterized_frames(trajectory, fps)


def trajectory_timing_description(
    trajectory: TimeParameterizedTrajectory,
) -> str:
    if not getattr(trajectory, "uses_toppra", True):
        return (
            f"raw linear playback={trajectory.duration:.3f} s, "
            f"peak segment velocity={trajectory.peak_velocity:.6g}, "
            f"velocity utilization="
            f"{100.0 * trajectory.velocity_limit_utilization:.1f}%; "
            "shortcut/spline/TOPP-RA disabled"
        )
    return (
        f"TOPP-RA={trajectory.duration:.3f} s, "
        f"peak velocity={trajectory.peak_velocity:.6g}, "
        f"peak acceleration={trajectory.peak_acceleration:.6g}, "
        f"limit utilization="
        f"{100.0 * trajectory.velocity_limit_utilization:.1f}% velocity / "
        f"{100.0 * trajectory.acceleration_limit_utilization:.1f}% acceleration"
    )


def apply_ctrl_reference(
    mujoco,
    model,
    data,
    layout: G1ControlLayout,
    balance_reference: G1BalanceReference,
    reference: list[float],
    gain_scale: float,
) -> None:
    joint_error = (
        np.asarray(reference[6:])
        - data.qpos[layout.joint_qpos_addresses]
    )
    joint_torque = (
        gain_scale * G1_JOINT_KP * joint_error
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
    data.ctrl[layout.joint_actuator_ids] = joint_torque


def configure_replay_camera(mujoco, camera, view: str = "front") -> None:
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (0.25, 0.0, 0.75)
    camera.distance = 2.8
    # View the robot from the open side of the shelf.  The opposite 135-degree
    # direction is behind the shelf back panel and hides most of the G1.
    camera.azimuth = video_view_azimuth(315.0, view)
    camera.elevation = video_view_elevation(-18.0, view)


def advance_ctrl_to_time(
    mujoco,
    model,
    data,
    layout: G1ControlLayout,
    balance_reference: G1BalanceReference,
    reference: list[float],
    gain_scale: float,
    target_time: float,
) -> None:
    half_timestep = 0.5 * float(model.opt.timestep)
    while float(data.time) + half_timestep < target_time:
        apply_ctrl_reference(
            mujoco,
            model,
            data,
            layout,
            balance_reference,
            reference,
            gain_scale,
        )
        mujoco.mj_step(model, data)


def save_video(
    model_path: Path,
    waypoints: list[list[float]],
    environment: dict[str, list],
    payload: dict | None,
    output_path: Path,
    fps: float,
    velocity_limits: np.ndarray,
    acceleration: float,
    control_mode: str,
    gain_scale: float,
    width: int,
    height: int,
    views: tuple[str, ...] = ("front",),
    planning_time_sec: float | None = None,
) -> None:
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo is required for MP4 rendering: python3 -m pip install mujoco"
        ) from error

    if output_path.suffix.lower() != ".mp4":
        raise ValueError("video output path must use the .mp4 extension")
    if control_mode == "ctrl":
        model = build_control_model(mujoco, model_path, environment, payload)
    else:
        model = build_kinematic_model(mujoco, model_path, payload)
    configure_model_render_quality(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)

    data = mujoco.MjData(model)
    base_address, joint_addresses = resolve_model_layout(mujoco, model)
    validate_joint_limits(mujoco, model, waypoints)
    trajectory = time_parameterize_waypoints(
        waypoints,
        velocity_limits,
        acceleration,
        fps,
    )
    control_layout = (
        resolve_control_layout(mujoco, model)
        if control_mode == "ctrl"
        else None
    )
    if control_layout is not None:
        configure_control_damping(model, control_layout, gain_scale)
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
    else:
        apply_configuration(
            mujoco,
            model,
            data,
            base_address,
            joint_addresses,
            waypoints[0],
        )
        balance_reference = None

    cameras = {}
    for view in views:
        camera = mujoco.MjvCamera()
        configure_replay_camera(mujoco, camera, view)
        cameras[view] = camera
    renderer = mujoco.Renderer(model, height=height, width=width)
    frame_period = 1.0 / fps
    target_simulation_time = float(data.time)

    def advance(reference: list[float]) -> None:
        nonlocal target_simulation_time
        target_simulation_time += frame_period
        if control_layout is not None:
            assert balance_reference is not None
            advance_ctrl_to_time(
                mujoco,
                model,
                data,
                control_layout,
                balance_reference,
                reference,
                gain_scale,
                target_simulation_time,
            )
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("ctrl simulation became non-finite")
        else:
            apply_configuration(
                mujoco,
                model,
                data,
                base_address,
                joint_addresses,
                reference,
            )

    def write_frame(writer: MultiViewVideoWriter) -> None:
        for view, camera in cameras.items():
            renderer.update_scene(data, camera=camera)
            if control_mode == "qpos":
                append_environment_geometries(
                    mujoco, renderer.scene, environment
                )
            frame = renderer.render()
            if planning_time_sec is not None:
                frame = add_video_text_overlay(
                    frame,
                    f"Planning time: {planning_time_sec:.4f} s",
                )
            writer.write(view, frame)

    start_frame_count = max(1, math.ceil(0.75 * fps))
    end_frame_count = max(1, math.ceil(1.0 * fps))
    output_path = output_path.expanduser().resolve()
    print(
        f"rendering {len(views)} G1 MP4 view(s) at {fps:g} fps "
        f"({width}x{height}, {control_mode} mode): {', '.join(views)}"
    )
    print(
        f"prepared {len(waypoints)} waypoints: "
        f"{trajectory_timing_description(trajectory)}"
    )
    try:
        with MultiViewVideoWriter(
            output_path, views, width, height, fps
        ) as writer:
            write_frame(writer)
            for _ in range(start_frame_count - 1):
                advance(waypoints[0])
                write_frame(writer)
            for configuration in time_parameterized_frames(trajectory, fps):
                advance(configuration)
                write_frame(writer)
            for _ in range(end_frame_count - 1):
                write_frame(writer)
            frame_count = writer.frame_count
    finally:
        renderer.close()

    print(
        "saved G1 MP4: "
        + ", ".join(str(path) for path in writer.output_paths.values())
        + f" ({frame_count} frames each, {frame_count / fps:.3f} s)"
    )


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


def replay(
    model_path: Path,
    waypoints: list[list[float]],
    environment: dict[str, list],
    payload: dict | None,
    fps: float,
    velocity_limits: np.ndarray,
    acceleration: float,
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
    trajectory = time_parameterize_waypoints(
        waypoints,
        velocity_limits,
        acceleration,
        fps,
    )
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
    print(
        f"{len(waypoints)}개 waypoint 재생 경로 준비: "
        f"{trajectory_timing_description(trajectory)}"
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
            configure_replay_camera(mujoco, viewer.cam)
        viewer.sync()

        frame_period = 1.0 / fps
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
                target_simulation_time = float(data.time)
                for _ in range(settle_frames):
                    if not viewer.is_running():
                        return
                    target_simulation_time += frame_period
                    advance_ctrl_to_time(
                        mujoco,
                        model,
                        data,
                        control_layout,
                        balance_reference,
                        waypoints[0],
                        gain_scale,
                        target_simulation_time,
                    )
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
                target_simulation_time = float(data.time)
            deadline = time.perf_counter()
            for configuration in time_parameterized_frames(trajectory, fps):
                if not viewer.is_running():
                    return
                if control_mode == "ctrl":
                    target_simulation_time += frame_period
                    advance_ctrl_to_time(
                        mujoco,
                        model,
                        data,
                        control_layout,
                        balance_reference,
                        configuration,
                        gain_scale,
                        target_simulation_time,
                    )
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
    urdf_path = args.urdf.expanduser().resolve()
    trajectory_path = args.trajectory.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"G1 MuJoCo model not found: {model_path}; set VAMP_ROOT if needed"
        )
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory not found: {trajectory_path}")

    waypoints, environment, payload, planning_time_sec = load_trajectory(
        trajectory_path
    )
    validate_environment(environment)
    validate_payload(payload)
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
    velocity_limits = load_planning_velocity_limits(
        urdf_path,
        args.velocity_scale,
    )
    if args.video is not None:
        save_video(
            model_path,
            waypoints,
            environment,
            payload,
            args.video,
            args.fps,
            velocity_limits,
            args.acceleration,
            args.control_mode,
            args.gain_scale,
            args.video_width,
            args.video_height,
            args.video_views,
            planning_time_sec,
        )
        return 0
    validate_only = args.validate_only or os.environ.get(
        "PRRTC_MUJOCO_VALIDATE_ONLY"
    ) == "1"
    if validate_only:
        import mujoco

        if args.control_mode == "ctrl":
            model = build_control_model(
                mujoco, model_path, environment, payload
            )
        else:
            model = build_kinematic_model(mujoco, model_path, payload)
        data = mujoco.MjData(model)
        base_address, joint_addresses = resolve_model_layout(mujoco, model)
        validate_joint_limits(mujoco, model, waypoints)
        trajectory = time_parameterize_waypoints(
            waypoints,
            velocity_limits,
            args.acceleration,
            args.fps,
        )
        if args.control_mode == "qpos":
            for configuration in time_parameterized_frames(
                trajectory,
                args.fps,
            ):
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
            f"validated {len(waypoints)} prepared waypoints "
            f"({trajectory_timing_description(trajectory)}), "
            f"35 planning coordinates -> model nq={model.nq}, "
            f"nu={model.nu}, control_mode={args.control_mode}, "
            f"{primitive_count} environment primitives, "
            f"payload_mass_kg={0.0 if payload is None else float(payload['mass_kg']):g}"
        )
        return 0

    replay(
        model_path,
        waypoints,
        environment,
        payload,
        args.fps,
        velocity_limits,
        args.acceleration,
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

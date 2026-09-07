#!/usr/bin/env python3
"""Inspect the FFW-SG2 rack upper-to-lower MuJoCo scene with payload CoM markers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "ffw_lift" / "ffw_sg2_rack_upper_to_lower.xml"
DEFAULT_PAYLOAD_MASS_KG = 3.0
DEFAULT_SUPPORT_MARGIN_M = 0.05
DEFAULT_SETTLE_STEPS = 12
REAL_DEFAULT_SPEED = 0.10
REAL_DEFAULT_SETTLE_STEPS = 240
SPACE_KEY = 32
REAL_WHEEL_RADIUS_M = 0.09
REAL_BASE_KP_XY = 32.0
REAL_BASE_KP_YAW = 32.0
REAL_BASE_MAX_SPEED_MPS = 3.5
REAL_BASE_MAX_YAW_RATE_RPS = 6.0
REAL_STEER_DIRECTION_TOLERANCE_RAD = math.radians(5.0)
REAL_STEER_DELTA_WEIGHT = 0.35
REAL_STEER_DRIVE_FLIP_PENALTY = 0.25
REAL_STEER_RATE_LIMIT_RPS = 3.5
REAL_DRIVE_ACCEL_LIMIT_RPS2 = 70.0
REAL_WHEEL_LIFT_THRESHOLD_N = 1.0
REAL_INITIAL_SETTLE_STEPS = 1000
PATH_OVERLAY_MAX_POINTS = 220
PATH_OVERLAY_BASE_Z = 0.035
PATH_OVERLAY_LINE_RADIUS = 0.005
PATH_OVERLAY_ENDPOINT_RADIUS = 0.018
JOINT_TRACKING_REPORT_LIMIT = 5
PATH_OVERLAY_BASE_RGBA = np.array([0.0, 0.78, 1.0, 0.75], dtype=np.float32)
PATH_OVERLAY_PAYLOAD_RGBA = np.array([1.0, 0.75, 0.05, 0.8], dtype=np.float32)
PATH_OVERLAY_START_RGBA = np.array([0.0, 0.95, 0.35, 0.95], dtype=np.float32)
PATH_OVERLAY_GOAL_RGBA = np.array([1.0, 0.1, 0.2, 0.95], dtype=np.float32)
REFERENCE_PAYLOAD_GHOST_GEOM_NAMES = (
    "object_collision",
    "object_left_grasp",
    "object_right_grasp",
)
REFERENCE_PAYLOAD_GHOST_RGBA = np.array([1.0, 0.0, 0.95, 0.23], dtype=np.float32)
REFERENCE_PAYLOAD_GHOST_GRASP_RGBA = np.array([0.05, 0.05, 0.05, 0.28], dtype=np.float32)
OBJECT_BODY_NAMES = ("object_box", "payload_box")
OBJECT_FREEJOINT_NAMES = ("object_freejoint", "payload_freejoint")
ATTACHED_OBJECT_FRAME_OFFSET = (0.1, 0.0, 0.0)
REAL_PAYLOAD_MODE_RIGID = "rigid"
REAL_PAYLOAD_MODE_EQUALITY = "equality"
REAL_PAYLOAD_MODES = (REAL_PAYLOAD_MODE_RIGID, REAL_PAYLOAD_MODE_EQUALITY)
GRIPPER_SITE_NAMES = (
    "gripper_l_rh_p12_rn_base",
    "gripper_r_rh_p12_rn_base",
)
OBJECT_GRASP_SITE_NAMES = (
    "object_left_grasp_site",
    "object_right_grasp_site",
)
REAL_GRASP_HALF_WIDTH_M = 0.2
REAL_GRASP_SOLREF = "0.004 1"
REAL_GRASP_SOLIMP = "0.95 0.99 0.001"
REAL_GRASP_WELD_TORQUESCALE = "1.0"
REAL_ARM_ACTUATOR_KP = "32000"
REAL_ARM_ACTUATOR_FORCERANGE = "-12000 12000"
LEFT_OBJECT_GRASP_SITE_QUAT = (-0.5, 0.5, -0.5, 0.5)
RIGHT_OBJECT_GRASP_SITE_QUAT = (0.5, 0.5, 0.5, 0.5)
GRASP_EQUALITY_NAMES = {
    "object_left_grasp_connect",
    "object_left_grasp_weld",
    "object_right_grasp_connect",
    "object_right_grasp_weld",
}
GRIPPER_JOINT_NAMES = tuple(
    f"gripper_{side}_joint{index}"
    for side in ("l", "r")
    for index in range(1, 5)
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
MOBILITY_PLANNING_JOINTS = MOBILITY_BASE_JOINTS + DUAL_ARM_PLANNING_JOINTS
SUPPORTED_JOINT_ORDERS = {
    DUAL_ARM_PLANNING_JOINTS,
    MOBILITY_PLANNING_JOINTS,
}
ACTUATOR_NAME_BY_JOINT = {
    "lift_joint": "actuator_lift_joint",
}

SUPPORT_POLYGON_BASE_XY = np.array(
    [
        [0.1371, 0.2554],
        [-0.2899, 0.0],
        [0.1371, -0.2554],
    ],
    dtype=float,
)
HIDDEN_MARKER_POSITION = np.array([0.0, 0.0, -10.0], dtype=float)
REAL_WHEEL_MODULES = {
    "left": {
        "xy": (0.1371, 0.2554),
        "steer_joint": "left_wheel_steer_joint",
        "steer_actuator": "left_wheel_steer_act",
        "drive_actuator": "left_wheel_drive_act",
        "drive_body": "left_wheel_drive",
    },
    "right": {
        "xy": (0.1371, -0.2554),
        "steer_joint": "right_wheel_steer_joint",
        "steer_actuator": "right_wheel_steer_act",
        "drive_actuator": "right_wheel_drive_act",
        "drive_body": "right_wheel_drive",
    },
    "rear": {
        "xy": (-0.2899, 0.0),
        "steer_joint": "rear_wheel_steer_joint",
        "steer_actuator": "rear_wheel_steer_act",
        "drive_actuator": "rear_wheel_drive_act",
        "drive_body": "rear_wheel_drive",
    },
}


@dataclass
class RealBaseControl:
    base_body_id: int
    base_origin: np.ndarray
    freejoint_qposadr: int
    steer_qpos_addresses: dict[str, int]
    steer_ctrl_addresses: dict[str, int]
    steer_ctrl_ranges: dict[str, tuple[float, float]]
    drive_ctrl_addresses: dict[str, int]
    steer_command_state: dict[str, float]
    drive_command_state: dict[str, float]
    wheel_body_ids: dict[str, int]
    floor_geom_id: int


@dataclass(frozen=True)
class RealBaseTracking:
    kp_xy: float
    kp_yaw: float
    max_speed_mps: float
    max_yaw_rate_rps: float
    steer_rate_limit_rps: float
    drive_accel_limit_rps2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Maximum configuration-coordinate change per second. Defaults to 0.7, or 0.10 with --real.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("ctrl", "qpos"),
        default="ctrl",
        help="Use MuJoCo actuator ctrl targets or direct qpos assignment for trajectory replay.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=None,
        help=(
            "MuJoCo simulation steps advanced after each ctrl target update. "
            "Defaults to 12, or 240 with --real."
        ),
    )
    parser.add_argument(
        "--object-mass",
        type=float,
        default=DEFAULT_PAYLOAD_MASS_KG,
        help="Payload mass in kg. Defaults to 3.0.",
    )
    parser.add_argument(
        "--support-margin",
        type=float,
        default=DEFAULT_SUPPORT_MARGIN_M,
        help="Minimum signed distance from the support polygon edge.",
    )
    parser.add_argument(
        "--attach-payload",
        action="store_true",
        help="Kinematically place the payload at the gripper midpoint in non-real mode.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help=(
            "Use a physics-check model: free base, payload mass, "
            "wheel-contact drive tracking, and no "
            "kinematic base overwrite."
        ),
    )
    parser.add_argument(
        "--real-payload-mode",
        choices=REAL_PAYLOAD_MODES,
        default=REAL_PAYLOAD_MODE_EQUALITY,
        help=(
            "Payload model used by --real. 'rigid' makes the payload a jointless "
            "child body of the left gripper site parent. 'equality' keeps it as a "
            "free body constrained to both gripper sites."
        ),
    )
    parser.add_argument(
        "--payload-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Object-frame offset added to the attached payload midpoint.",
    )
    parser.add_argument(
        "--real-base-kp-xy",
        type=float,
        default=REAL_BASE_KP_XY,
        help="Real-mode proportional gain for planar base tracking.",
    )
    parser.add_argument(
        "--real-base-kp-yaw",
        type=float,
        default=REAL_BASE_KP_YAW,
        help="Real-mode proportional gain for base yaw tracking.",
    )
    parser.add_argument(
        "--real-base-max-speed",
        type=float,
        default=REAL_BASE_MAX_SPEED_MPS,
        help="Real-mode planar base speed limit in m/s.",
    )
    parser.add_argument(
        "--real-base-max-yaw-rate",
        type=float,
        default=REAL_BASE_MAX_YAW_RATE_RPS,
        help="Real-mode base yaw-rate limit in rad/s.",
    )
    parser.add_argument(
        "--real-steer-rate-limit",
        type=float,
        default=REAL_STEER_RATE_LIMIT_RPS,
        help="Real-mode wheel steering command rate limit in rad/s.",
    )
    parser.add_argument(
        "--real-drive-accel-limit",
        type=float,
        default=REAL_DRIVE_ACCEL_LIMIT_RPS2,
        help="Real-mode wheel drive command acceleration limit in rad/s^2.",
    )
    parser.add_argument(
        "--real-initial-settle-steps",
        type=int,
        default=REAL_INITIAL_SETTLE_STEPS,
        help="MuJoCo steps used to settle the initial real-mode grasp before validation or replay.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.speed is None:
        args.speed = REAL_DEFAULT_SPEED if args.real else 0.7
    if args.settle_steps is None:
        args.settle_steps = REAL_DEFAULT_SETTLE_STEPS if args.real else DEFAULT_SETTLE_STEPS
    if args.fps <= 0.0 or args.speed <= 0.0:
        parser.error("--fps and --speed must be positive")
    if args.settle_steps < 0:
        parser.error("--settle-steps must be greater than or equal to 0")
    if args.real_initial_settle_steps < 0:
        parser.error("--real-initial-settle-steps must be greater than or equal to 0")
    if args.object_mass <= 0.0 or not math.isfinite(args.object_mass):
        parser.error("--object-mass must be a finite value greater than 0")
    if args.support_margin < 0.0 or not math.isfinite(args.support_margin):
        parser.error("--support-margin must be a finite value greater than or equal to 0")
    real_tracking_options = (
        ("--real-base-kp-xy", args.real_base_kp_xy),
        ("--real-base-kp-yaw", args.real_base_kp_yaw),
        ("--real-base-max-speed", args.real_base_max_speed),
        ("--real-base-max-yaw-rate", args.real_base_max_yaw_rate),
        ("--real-steer-rate-limit", args.real_steer_rate_limit),
        ("--real-drive-accel-limit", args.real_drive_accel_limit),
    )
    for option, value in real_tracking_options:
        if value <= 0.0 or not math.isfinite(value):
            parser.error(f"{option} must be a finite value greater than 0")
    if args.real and args.input_mode != "ctrl":
        parser.error("--real requires --input-mode ctrl")
    return args


def parse_attached_object_frame_offset(document) -> tuple[float, float, float]:
    value = document.get("attached_object_frame_offset", ATTACHED_OBJECT_FRAME_OFFSET)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("attached_object_frame_offset must be [x, y, z]")
    offset = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in offset):
        raise ValueError("attached_object_frame_offset contains a non-finite value")
    return offset


def load_trajectory(path: Path) -> tuple[tuple[str, ...], list[list[float]], tuple[float, float, float]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    joint_names = tuple(document.get("joint_names", ()))
    if joint_names not in SUPPORTED_JOINT_ORDERS:
        raise ValueError("trajectory joint order is not supported by this scene")

    waypoints = document.get("waypoints", document.get("path"))
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise ValueError("trajectory must contain at least two waypoints")
    normalized: list[list[float]] = []
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, list) or len(waypoint) != len(joint_names):
            raise ValueError(f"waypoint {index} has the wrong dimension")
        values = [float(value) for value in waypoint]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"waypoint {index} contains a non-finite value")
        normalized.append(values)
    return joint_names, normalized, parse_attached_object_frame_offset(document)


def find_named_body(root: ET.Element, body_name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    raise ValueError(f"MuJoCo model is missing body: {body_name}")


def find_first_named_body(root: ET.Element, body_names: tuple[str, ...]) -> ET.Element:
    for body_name in body_names:
        for body in root.iter("body"):
            if body.get("name") == body_name:
                return body
    raise ValueError(
        "MuJoCo model is missing body: " + " or ".join(body_names)
    )


def find_parent(root: ET.Element, child: ET.Element) -> ET.Element | None:
    for candidate in root.iter():
        for nested in list(candidate):
            if nested is child:
                return candidate
    return None


def find_site_parent_body(root: ET.Element, site_name: str) -> ET.Element:
    for body in root.iter("body"):
        for child in body:
            if child.tag == "site" and child.get("name") == site_name:
                return body
    raise ValueError(f"MuJoCo model is missing site: {site_name}")


def has_freejoint(body: ET.Element) -> bool:
    for child in body:
        if child.tag == "freejoint":
            return True
        if child.tag == "joint" and child.get("type") == "free":
            return True
    return False


def remove_freejoints(body: ET.Element) -> None:
    for child in list(body):
        if child.tag == "freejoint":
            body.remove(child)
        elif child.tag == "joint" and child.get("type") == "free":
            body.remove(child)


def mujoco_vec(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def add_freejoint_if_missing(body: ET.Element, joint_name: str) -> None:
    if has_freejoint(body):
        return

    freejoint = ET.Element("freejoint", {"name": joint_name})
    insert_at = 0
    for index, child in enumerate(list(body)):
        if child.tag == "inertial":
            insert_at = index + 1
            break
    body.insert(insert_at, freejoint)


def upsert_site(
    body: ET.Element,
    name: str,
    position,
    rgba: str,
    quat=None,
) -> None:
    site = None
    for child in body:
        if child.tag == "site" and child.get("name") == name:
            site = child
            break
    if site is None:
        site = ET.Element("site", {"name": name})
        body.append(site)
    site.set("pos", mujoco_vec(position))
    site.set("type", "sphere")
    site.set("size", "0.008")
    site.set("rgba", rgba)
    if quat is not None:
        site.set("quat", mujoco_vec(quat))


def ensure_top_level_element(
    root: ET.Element,
    tag: str,
    before_tag: str | None = None,
) -> ET.Element:
    element = root.find(tag)
    if element is not None:
        return element

    element = ET.Element(tag)
    insert_at = len(list(root))
    if before_tag is not None:
        for index, child in enumerate(list(root)):
            if child.tag == before_tag:
                insert_at = index
                break
    root.insert(insert_at, element)
    return element


def remove_named_children(parent: ET.Element, names: set[str]) -> None:
    for child in list(parent):
        if child.get("name") in names:
            parent.remove(child)


def remove_real_grasp_equalities(root: ET.Element) -> None:
    equality = root.find("equality")
    if equality is not None:
        remove_named_children(equality, GRASP_EQUALITY_NAMES)


def configure_real_payload_grasp(
    root: ET.Element,
    payload_body: ET.Element,
    payload_offset: tuple[float, float, float],
) -> None:
    offset_x, offset_y, offset_z = payload_offset
    left_site_pos = (
        -offset_x,
        REAL_GRASP_HALF_WIDTH_M - offset_y,
        -offset_z,
    )
    right_site_pos = (
        -offset_x,
        -REAL_GRASP_HALF_WIDTH_M - offset_y,
        -offset_z,
    )
    upsert_site(
        payload_body,
        OBJECT_GRASP_SITE_NAMES[0],
        left_site_pos,
        "1 0.35 0.15 0.9",
        LEFT_OBJECT_GRASP_SITE_QUAT,
    )
    upsert_site(
        payload_body,
        OBJECT_GRASP_SITE_NAMES[1],
        right_site_pos,
        "0.15 0.35 1 0.9",
        RIGHT_OBJECT_GRASP_SITE_QUAT,
    )

    equality = ensure_top_level_element(root, "equality", before_tag="actuator")
    remove_named_children(equality, GRASP_EQUALITY_NAMES)
    equality.append(
        ET.Element(
            "weld",
            {
                "name": "object_left_grasp_weld",
                "site1": GRIPPER_SITE_NAMES[0],
                "site2": OBJECT_GRASP_SITE_NAMES[0],
                "torquescale": REAL_GRASP_WELD_TORQUESCALE,
                "solref": REAL_GRASP_SOLREF,
                "solimp": REAL_GRASP_SOLIMP,
            },
        )
    )
    equality.append(
        ET.Element(
            "connect",
            {
                "name": "object_right_grasp_connect",
                "site1": GRIPPER_SITE_NAMES[1],
                "site2": OBJECT_GRASP_SITE_NAMES[1],
                "solref": REAL_GRASP_SOLREF,
                "solimp": REAL_GRASP_SOLIMP,
            },
        )
    )


def configure_real_payload_rigid(
    root: ET.Element,
    payload_body: ET.Element,
) -> None:
    remove_real_grasp_equalities(root)
    remove_freejoints(payload_body)
    attach_parent = find_site_parent_body(root, GRIPPER_SITE_NAMES[0])
    payload_parent = find_parent(root, payload_body)
    if payload_parent is None:
        raise ValueError(f"{payload_body.get('name', 'payload')} has no parent body")
    if payload_parent is not attach_parent:
        payload_parent.remove(payload_body)
        attach_parent.append(payload_body)


def configure_real_arm_actuators(root: ET.Element) -> None:
    arm_joint_names = {
        joint_name
        for joint_name in DUAL_ARM_PLANNING_JOINTS
        if joint_name.startswith("arm_")
    }
    actuator = root.find("actuator")
    if actuator is None:
        return
    for element in actuator:
        if element.tag != "position":
            continue
        joint_name = element.get("joint", element.get("name", ""))
        if joint_name not in arm_joint_names:
            continue
        element.set("kp", REAL_ARM_ACTUATOR_KP)
        element.set("forcerange", REAL_ARM_ACTUATOR_FORCERANGE)


def write_real_model_xml(
    model_path: Path,
    payload_offset: tuple[float, float, float],
    payload_mode: str,
) -> Path:
    tree = ET.parse(model_path)
    root = tree.getroot()
    model_name = root.get("model", model_path.stem)
    root.set("model", f"{model_name}_real")

    base_body = find_named_body(root, "base_link")
    add_freejoint_if_missing(base_body, "floating_base")
    configure_real_arm_actuators(root)

    payload_body = find_first_named_body(root, OBJECT_BODY_NAMES)
    payload_parent = find_parent(root, payload_body)
    if payload_parent is None:
        raise ValueError(f"{payload_body.get('name', 'payload')} has no parent body")
    if payload_mode == REAL_PAYLOAD_MODE_RIGID:
        configure_real_payload_rigid(root, payload_body)
    elif payload_mode == REAL_PAYLOAD_MODE_EQUALITY:
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("MuJoCo model is missing worldbody")
        if payload_parent is not worldbody:
            payload_parent.remove(payload_body)
            worldbody.append(payload_body)
        add_freejoint_if_missing(payload_body, OBJECT_FREEJOINT_NAMES[0])
        configure_real_payload_grasp(root, payload_body, payload_offset)
    else:
        raise ValueError(f"unsupported real payload mode: {payload_mode}")

    ET.indent(tree, space="  ")
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f"{model_path.stem}_real_",
        suffix=".xml",
        dir=model_path.parent,
        delete=False,
    )
    with handle:
        tree.write(handle, encoding="utf-8", xml_declaration=False)
    return Path(handle.name)


def resolve_qpos_addresses(mujoco, model, joint_names: tuple[str, ...]) -> list[int | None]:
    addresses: list[int | None] = []
    for name in joint_names:
        if name in MOBILITY_BASE_JOINTS:
            addresses.append(None)
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint: {name}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def resolve_ctrl_addresses(mujoco, model, joint_names: tuple[str, ...]) -> list[int | None]:
    addresses: list[int | None] = []
    for name in joint_names:
        if name in MOBILITY_BASE_JOINTS:
            addresses.append(None)
            continue
        actuator_name = ACTUATOR_NAME_BY_JOINT.get(name, name)
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        if actuator_id < 0:
            raise ValueError(f"MuJoCo model is missing actuator: {actuator_name}")
        addresses.append(int(actuator_id))
    return addresses


def descendants_of(model, body_id: int) -> set[int]:
    descendants = {body_id}
    changed = True
    while changed:
        changed = False
        for candidate in range(1, model.nbody):
            if candidate in descendants:
                continue
            parent = int(model.body_parentid[candidate])
            if parent in descendants:
                descendants.add(candidate)
                changed = True
    return descendants


def qposadr_for_joint(mujoco, model, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo model is missing joint: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def qposadr_for_first_joint(mujoco, model, joint_names: tuple[str, ...]) -> int:
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            return int(model.jnt_qposadr[joint_id])
    raise ValueError(
        "MuJoCo model is missing joint: " + " or ".join(joint_names)
    )


def body_id_for_first_name(mujoco, model, body_names: tuple[str, ...]) -> int:
    for body_name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            return int(body_id)
    raise ValueError(
        "MuJoCo model is missing body: " + " or ".join(body_names)
    )


def set_payload_mass(mujoco, model, data, mass_kg: float) -> int:
    payload_id = body_id_for_first_name(mujoco, model, OBJECT_BODY_NAMES)

    current_mass = float(model.body_mass[payload_id])
    if current_mass <= 0.0:
        payload_name = model.body(payload_id).name or str(payload_id)
        raise ValueError(f"{payload_name} must have positive nominal mass")
    inertia_scale = mass_kg / current_mass
    model.body_mass[payload_id] = mass_kg
    model.body_inertia[payload_id][:] *= inertia_scale
    mujoco.mj_forward(model, data)
    return payload_id


def set_mocap_body(mujoco, model, data, body_name: str, position: np.ndarray) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        return
    data.mocap_pos[mocap_id][:] = position
    data.mocap_quat[mocap_id][:] = (1.0, 0.0, 0.0, 0.0)


def vector_dot(left, right) -> float:
    return sum(left[index] * right[index] for index in range(3))


def vector_cross(left, right) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def vector_norm(vector) -> float:
    return math.sqrt(vector_dot(vector, vector))


def vector_normalize(vector) -> tuple[float, float, float] | None:
    length = vector_norm(vector)
    if length <= 1.0e-9:
        return None
    return tuple(value / length for value in vector)


def subtract_projection(vector, axis) -> tuple[float, float, float]:
    scale = vector_dot(vector, axis)
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


def rotation_from_axes(x_axis, y_axis, z_axis):
    return (
        (x_axis[0], y_axis[0], z_axis[0]),
        (x_axis[1], y_axis[1], z_axis[1]),
        (x_axis[2], y_axis[2], z_axis[2]),
    )


def object_axes_from_grippers(data, left_site: int, right_site: int):
    left_pos = data.site_xpos[left_site]
    right_pos = data.site_xpos[right_site]
    y_axis = vector_normalize(
        tuple(float(left_pos[index] - right_pos[index]) for index in range(3))
    )
    if y_axis is None:
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)

    left_z = site_matrix_column(data, left_site, 2)
    right_z = site_matrix_column(data, right_site, 2)
    x_hint = vector_normalize(
        tuple(left_z[index] + right_z[index] for index in range(3))
    )
    if x_hint is None:
        x_hint = (1.0, 0.0, 0.0)

    x_axis = vector_normalize(subtract_projection(x_hint, y_axis))
    if x_axis is None:
        x_axis = vector_normalize(vector_cross(y_axis, (0.0, 0.0, 1.0)))
    if x_axis is None:
        x_axis = (1.0, 0.0, 0.0)

    z_axis = vector_normalize(vector_cross(x_axis, y_axis))
    if z_axis is None:
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)

    left_y = site_matrix_column(data, left_site, 1)
    right_y = site_matrix_column(data, right_site, 1)
    z_hint = vector_normalize(
        tuple(right_y[index] - left_y[index] for index in range(3))
    )
    if z_hint is not None and vector_dot(z_axis, z_hint) < 0.0:
        x_axis = tuple(-value for value in x_axis)
        z_axis = tuple(-value for value in z_axis)

    return x_axis, y_axis, z_axis


def gripper_site_ids(mujoco, model) -> tuple[int, int]:
    site_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        for site_name in GRIPPER_SITE_NAMES
    )
    if min(site_ids) < 0:
        raise ValueError("MuJoCo model is missing gripper sites")
    return site_ids


def object_pose_from_grippers(mujoco, model, data, offset) -> tuple[np.ndarray, np.ndarray]:
    left_site, right_site = gripper_site_ids(mujoco, model)
    x_axis, y_axis, z_axis = object_axes_from_grippers(data, left_site, right_site)
    midpoint = 0.5 * (
        np.array(data.site_xpos[left_site], dtype=float)
        + np.array(data.site_xpos[right_site], dtype=float)
    )
    frame_offset = tuple(float(component) for component in offset)
    world_offset = np.array(
        [
            x_axis[axis] * frame_offset[0]
            + y_axis[axis] * frame_offset[1]
            + z_axis[axis] * frame_offset[2]
            for axis in range(3)
        ],
        dtype=float,
    )
    object_quat = np.array(
        matrix_to_quat(rotation_from_axes(x_axis, y_axis, z_axis)),
        dtype=float,
    )
    return midpoint + world_offset, object_quat


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
    mujoco.mj_forward(model, data)


def transform_support_polygon(model, data, base_body_id: int) -> np.ndarray:
    base_pos = np.array(data.xpos[base_body_id], dtype=float)
    base_rot = np.array(data.xmat[base_body_id], dtype=float).reshape(3, 3)
    points = []
    for point_xy in SUPPORT_POLYGON_BASE_XY:
        local = np.array([point_xy[0], point_xy[1], 0.0], dtype=float)
        world = base_pos + base_rot @ local
        points.append(world[:2])
    return np.array(points)


def support_signed_distances(point_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    distances = []
    for index in range(len(polygon_xy)):
        start = polygon_xy[index]
        end = polygon_xy[(index + 1) % len(polygon_xy)]
        edge = end - start
        rel = point_xy - start
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        distances.append(cross / max(float(np.linalg.norm(edge)), 1.0e-9))
    return np.array(distances)


def compute_com_state(model, data, robot_body_ids: set[int], payload_body_ids: set[int]) -> dict:
    robot_mass = 0.0
    robot_weighted = np.zeros(3)
    for body_id in robot_body_ids:
        if body_id in payload_body_ids:
            continue
        mass = float(model.body_mass[body_id])
        if mass <= 0.0:
            continue
        robot_mass += mass
        robot_weighted += mass * np.array(data.xipos[body_id], dtype=float)

    if robot_mass <= 0.0:
        raise ValueError("robot subtree has no positive mass")

    payload_mass = 0.0
    payload_weighted = np.zeros(3)
    for payload_body_id in payload_body_ids:
        mass = float(model.body_mass[payload_body_id])
        if mass <= 0.0:
            continue
        payload_mass += mass
        payload_weighted += mass * np.array(data.xipos[payload_body_id], dtype=float)
    if payload_mass <= 0.0:
        raise ValueError("payload body has no positive mass")

    payload_com = payload_weighted / payload_mass
    robot_com = robot_weighted / robot_mass
    total_com = (robot_weighted + payload_mass * payload_com) / (robot_mass + payload_mass)
    return {
        "robot_mass": robot_mass,
        "payload_mass": payload_mass,
        "total_mass": robot_mass + payload_mass,
        "robot_com": robot_com,
        "payload_com": payload_com,
        "total_com": total_com,
    }


def place_payload_at_gripper_midpoint(mujoco, model, data, payload_qposadr: int, offset) -> None:
    position, quat = object_pose_from_grippers(mujoco, model, data, offset)
    data.qpos[payload_qposadr : payload_qposadr + 3] = position
    data.qpos[payload_qposadr + 3 : payload_qposadr + 7] = quat
    mujoco.mj_forward(model, data)


def grasp_site_error_distances(mujoco, model, data) -> dict[str, float]:
    errors = {}
    for side, gripper_site_name, object_site_name in zip(
        ("left", "right"),
        GRIPPER_SITE_NAMES,
        OBJECT_GRASP_SITE_NAMES,
    ):
        gripper_site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            gripper_site_name,
        )
        object_site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            object_site_name,
        )
        if gripper_site_id < 0 or object_site_id < 0:
            continue
        errors[side] = float(
            np.linalg.norm(
                np.array(data.site_xpos[gripper_site_id], dtype=float)
                - np.array(data.site_xpos[object_site_id], dtype=float)
            )
        )
    return errors


def joint_tracking_errors(
    data,
    qpos_addresses: list[int | None],
    joint_names: tuple[str, ...],
    target_configuration: list[float],
) -> dict[str, float]:
    errors = {}
    for qpos_address, joint_name, target in zip(
        qpos_addresses,
        joint_names,
        target_configuration,
    ):
        if qpos_address is None:
            continue
        error = float(data.qpos[qpos_address]) - float(target)
        if joint_name.startswith("arm_"):
            error = normalize_angle(error)
        errors[joint_name] = error
    return errors


def format_largest_joint_errors(errors: dict[str, float]) -> str:
    if not errors:
        return "none"
    largest = sorted(
        errors.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:JOINT_TRACKING_REPORT_LIMIT]
    return ", ".join(f"{name}:{error:.6f}" for name, error in largest)


def payload_target_position_error(
    mujoco,
    model,
    data,
    payload_body_id: int,
    payload_offset,
) -> tuple[float, np.ndarray]:
    target_position, _ = object_pose_from_grippers(mujoco, model, data, payload_offset)
    delta = np.array(data.xpos[payload_body_id], dtype=float) - target_position
    return float(np.linalg.norm(delta)), delta


def payload_tilt_degrees(data, payload_body_id: int) -> tuple[float, float]:
    rotation = np.array(data.xmat[payload_body_id], dtype=float).reshape(3, 3)
    x_axis = rotation[:, 0]
    z_axis = rotation[:, 2]
    level_tilt = math.degrees(
        math.acos(clamp(float(z_axis[2]), -1.0, 1.0))
    )
    forward_elevation = math.degrees(
        math.atan2(float(x_axis[2]), float(np.linalg.norm(x_axis[:2])))
    )
    return level_tilt, forward_elevation


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = normalize_quat(left)
    w1, x1, y1, z1 = normalize_quat(right)
    return normalize_quat(
        np.array(
            [
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ],
            dtype=float,
        )
    )


def base_pose_from_configuration(
    joint_names: tuple[str, ...],
    configuration: list[float],
) -> tuple[float, float, float]:
    base_x = 0.0
    base_y = 0.0
    base_yaw = 0.0
    for name, value in zip(joint_names, configuration):
        if name == "base_x":
            base_x = value
        elif name == "base_y":
            base_y = value
        elif name == "base_yaw":
            base_yaw = value
    return base_x, base_y, base_yaw


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw
    return math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def clamp_delta(target: float, current: float, max_delta: float) -> float:
    return current + clamp(target - current, -max_delta, max_delta)


def closest_periodic_angle(angle: float, reference: float) -> float:
    period = 2.0 * math.pi
    return angle + period * round((reference - angle) / period)


def continuous_steer_drive_command(
    raw_angle: float,
    wheel_speed: float,
    current_steer: float,
    steer_range: tuple[float, float],
    previous_steer: float,
    previous_drive_speed: float,
    timestep: float,
    steer_rate_limit_rps: float,
    drive_accel_limit_rps2: float,
) -> tuple[float, float]:
    if not math.isfinite(previous_steer):
        previous_steer = current_steer
    if not math.isfinite(previous_drive_speed):
        previous_drive_speed = 0.0

    base_drive_speed = wheel_speed / REAL_WHEEL_RADIUS_M
    candidates = []
    previous_drive_sign = 0.0
    if previous_drive_speed > 1.0e-6:
        previous_drive_sign = 1.0
    elif previous_drive_speed < -1.0e-6:
        previous_drive_sign = -1.0

    for drive_sign, axis_offset in ((1.0, 0.0), (-1.0, math.pi)):
        axis_angle = closest_periodic_angle(
            raw_angle + axis_offset,
            previous_steer,
        )
        steer_angle = clamp(axis_angle, steer_range[0], steer_range[1])
        commanded_direction = steer_angle if drive_sign > 0.0 else steer_angle + math.pi
        direction_error = abs(normalize_angle(raw_angle - commanded_direction))
        steer_delta = abs(steer_angle - previous_steer)
        drive_flip_cost = (
            REAL_STEER_DRIVE_FLIP_PENALTY
            if previous_drive_sign != 0.0 and drive_sign != previous_drive_sign
            else 0.0
        )
        score = (
            direction_error
            + REAL_STEER_DELTA_WEIGHT * steer_delta
            + drive_flip_cost
        )
        candidates.append(
            (
                score,
                direction_error,
                steer_delta,
                steer_angle,
                drive_sign,
            )
        )
    _, _, _, target_steer, drive_sign = min(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
    )

    steer_reference = clamp(previous_steer, steer_range[0], steer_range[1])
    max_steer_delta = steer_rate_limit_rps * timestep
    steer_angle = clamp_delta(target_steer, steer_reference, max_steer_delta)
    steer_angle = clamp(steer_angle, steer_range[0], steer_range[1])

    commanded_direction = steer_angle if drive_sign > 0.0 else steer_angle + math.pi
    direction_error = abs(normalize_angle(raw_angle - commanded_direction))
    alignment = 1.0
    if direction_error > REAL_STEER_DIRECTION_TOLERANCE_RAD:
        alignment = max(0.0, math.cos(direction_error))
    target_drive_speed = drive_sign * base_drive_speed * alignment
    target_drive_speed = clamp(target_drive_speed, -50.0, 50.0)

    max_drive_delta = drive_accel_limit_rps2 * timestep
    drive_speed = clamp_delta(
        target_drive_speed,
        previous_drive_speed,
        max_drive_delta,
    )
    return steer_angle, clamp(drive_speed, -50.0, 50.0)


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def inverse_quat(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def body_yaw(data, body_id: int) -> float:
    rotation = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
    return math.atan2(rotation[1, 0], rotation[0, 0])


def resolve_real_base_control(mujoco, model, base_body_id: int) -> RealBaseControl:
    freejoint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "floating_base",
    )
    if freejoint_id < 0:
        raise ValueError("real mode requires a floating_base freejoint")
    if int(model.jnt_type[freejoint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("floating_base is not a MuJoCo freejoint")

    steer_qpos_addresses: dict[str, int] = {}
    steer_ctrl_addresses: dict[str, int] = {}
    steer_ctrl_ranges: dict[str, tuple[float, float]] = {}
    drive_ctrl_addresses: dict[str, int] = {}
    steer_command_state: dict[str, float] = {}
    drive_command_state: dict[str, float] = {}
    wheel_body_ids: dict[str, int] = {}
    for key, module in REAL_WHEEL_MODULES.items():
        steer_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            module["steer_joint"],
        )
        steer_actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            module["steer_actuator"],
        )
        drive_actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            module["drive_actuator"],
        )
        drive_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            module["drive_body"],
        )
        if min(steer_joint_id, steer_actuator_id, drive_actuator_id, drive_body_id) < 0:
            raise ValueError(f"real mode is missing wheel model element: {key}")
        steer_qpos_addresses[key] = int(model.jnt_qposadr[steer_joint_id])
        steer_ctrl_addresses[key] = int(steer_actuator_id)
        lower, upper = model.actuator_ctrlrange[steer_actuator_id]
        steer_ctrl_ranges[key] = (float(lower), float(upper))
        drive_ctrl_addresses[key] = int(drive_actuator_id)
        steer_command_state[key] = 0.0
        drive_command_state[key] = 0.0
        wheel_body_ids[key] = int(drive_body_id)

    floor_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )
    if floor_geom_id < 0:
        raise ValueError("real mode requires floor geom contacts")

    return RealBaseControl(
        base_body_id=base_body_id,
        base_origin=np.array(model.body_pos[base_body_id], dtype=float),
        freejoint_qposadr=int(model.jnt_qposadr[freejoint_id]),
        steer_qpos_addresses=steer_qpos_addresses,
        steer_ctrl_addresses=steer_ctrl_addresses,
        steer_ctrl_ranges=steer_ctrl_ranges,
        drive_ctrl_addresses=drive_ctrl_addresses,
        steer_command_state=steer_command_state,
        drive_command_state=drive_command_state,
        wheel_body_ids=wheel_body_ids,
        floor_geom_id=int(floor_geom_id),
    )


def reset_real_base_command_state(data, real_base: RealBaseControl) -> None:
    for key in REAL_WHEEL_MODULES:
        real_base.steer_command_state[key] = float(
            data.qpos[real_base.steer_qpos_addresses[key]]
        )
        real_base.drive_command_state[key] = 0.0


def set_non_base_targets(
    data,
    qpos_addresses: list[int | None],
    ctrl_addresses: list[int | None],
    joint_names: tuple[str, ...],
    configuration: list[float],
    input_mode: str,
    seed_qpos: bool,
) -> None:
    for qpos_address, ctrl_address, name, value in zip(
        qpos_addresses,
        ctrl_addresses,
        joint_names,
        configuration,
    ):
        if name in MOBILITY_BASE_JOINTS:
            continue
        if input_mode == "ctrl":
            if ctrl_address is None:
                raise ValueError(f"MuJoCo model is missing ctrl address for joint: {name}")
            data.ctrl[ctrl_address] = value
            if seed_qpos and qpos_address is not None:
                data.qpos[qpos_address] = value
        elif qpos_address is not None:
            data.qpos[qpos_address] = value


def seed_real_configuration(
    mujoco,
    model,
    data,
    qpos_addresses: list[int | None],
    ctrl_addresses: list[int | None],
    joint_names: tuple[str, ...],
    configuration: list[float],
    real_base: RealBaseControl,
    input_mode: str,
) -> None:
    base_x, base_y, base_yaw = base_pose_from_configuration(joint_names, configuration)
    set_non_base_targets(
        data,
        qpos_addresses,
        ctrl_addresses,
        joint_names,
        configuration,
        input_mode,
        seed_qpos=True,
    )
    position = real_base.base_origin.copy()
    position[0] += base_x
    position[1] += base_y
    qposadr = real_base.freejoint_qposadr
    data.qpos[qposadr : qposadr + 3] = position
    data.qpos[qposadr + 3 : qposadr + 7] = yaw_to_quat(base_yaw)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def align_attached_payload_to_gripper_midpoint(
    mujoco,
    model,
    data,
    payload_body_id: int,
    offset,
) -> None:
    parent_id = int(model.body_parentid[payload_body_id])
    parent_pos = np.array(data.xpos[parent_id], dtype=float)
    parent_rot = np.array(data.xmat[parent_id], dtype=float).reshape(3, 3)
    position, object_quat = object_pose_from_grippers(mujoco, model, data, offset)
    model.body_pos[payload_body_id][:] = parent_rot.T @ (position - parent_pos)
    model.body_quat[payload_body_id][:] = quat_multiply(
        inverse_quat(np.array(data.xquat[parent_id], dtype=float)),
        object_quat,
    )
    mujoco.mj_forward(model, data)


def real_base_error_state(
    model,
    data,
    joint_names: tuple[str, ...],
    configuration: list[float],
    real_base: RealBaseControl,
) -> tuple[np.ndarray, float, float, float]:
    target_x, target_y, target_yaw = base_pose_from_configuration(
        joint_names,
        configuration,
    )
    target_x += float(real_base.base_origin[0])
    target_y += float(real_base.base_origin[1])

    base_pos = np.array(data.xpos[real_base.base_body_id], dtype=float)
    base_yaw = body_yaw(data, real_base.base_body_id)
    c = math.cos(base_yaw)
    s = math.sin(base_yaw)
    error_world_x = target_x - base_pos[0]
    error_world_y = target_y - base_pos[1]
    error_base = np.array(
        [
            c * error_world_x + s * error_world_y,
            -s * error_world_x + c * error_world_y,
        ],
        dtype=float,
    )
    yaw_error = normalize_angle(target_yaw - base_yaw)
    xy_error = math.hypot(error_world_x, error_world_y)
    return error_base, yaw_error, xy_error, abs(yaw_error)


def base_feedforward_from_configurations(
    joint_names: tuple[str, ...],
    previous_configuration: list[float] | None,
    configuration: list[float],
    duration: float,
) -> tuple[np.ndarray, float]:
    if previous_configuration is None or duration <= 0.0:
        return np.zeros(2, dtype=float), 0.0

    previous_x, previous_y, previous_yaw = base_pose_from_configuration(
        joint_names,
        previous_configuration,
    )
    target_x, target_y, target_yaw = base_pose_from_configuration(
        joint_names,
        configuration,
    )
    velocity_world = np.array(
        [
            (target_x - previous_x) / duration,
            (target_y - previous_y) / duration,
        ],
        dtype=float,
    )
    yaw_rate = normalize_angle(target_yaw - previous_yaw) / duration
    return velocity_world, yaw_rate


def update_real_base_controls(
    model,
    data,
    joint_names: tuple[str, ...],
    configuration: list[float],
    real_base: RealBaseControl,
    tracking: RealBaseTracking,
    feedforward_velocity_world: np.ndarray | None = None,
    feedforward_yaw_rate: float = 0.0,
) -> tuple[float, float]:
    error_base, yaw_error, xy_error, abs_yaw_error = real_base_error_state(
        model,
        data,
        joint_names,
        configuration,
        real_base,
    )
    velocity_base = tracking.kp_xy * error_base
    if feedforward_velocity_world is not None:
        base_yaw = body_yaw(data, real_base.base_body_id)
        c = math.cos(base_yaw)
        s = math.sin(base_yaw)
        velocity_base += np.array(
            [
                c * feedforward_velocity_world[0] + s * feedforward_velocity_world[1],
                -s * feedforward_velocity_world[0] + c * feedforward_velocity_world[1],
            ],
            dtype=float,
        )
    speed = float(np.linalg.norm(velocity_base))
    if speed > tracking.max_speed_mps:
        velocity_base *= tracking.max_speed_mps / speed

    yaw_rate = clamp(
        feedforward_yaw_rate + tracking.kp_yaw * yaw_error,
        -tracking.max_yaw_rate_rps,
        tracking.max_yaw_rate_rps,
    )

    for key, module in REAL_WHEEL_MODULES.items():
        wheel_x, wheel_y = module["xy"]
        wheel_velocity = np.array(
            [
                velocity_base[0] - yaw_rate * wheel_y,
                velocity_base[1] + yaw_rate * wheel_x,
            ],
            dtype=float,
        )
        wheel_speed = float(np.linalg.norm(wheel_velocity))
        current_steer = float(data.qpos[real_base.steer_qpos_addresses[key]])
        previous_steer = real_base.steer_command_state.get(key, current_steer)
        previous_drive_speed = real_base.drive_command_state.get(key, 0.0)
        if wheel_speed > 1.0e-6:
            steer_angle, drive_speed = continuous_steer_drive_command(
                math.atan2(wheel_velocity[1], wheel_velocity[0]),
                wheel_speed,
                current_steer,
                real_base.steer_ctrl_ranges[key],
                previous_steer,
                previous_drive_speed,
                float(model.opt.timestep),
                tracking.steer_rate_limit_rps,
                tracking.drive_accel_limit_rps2,
            )
        else:
            steer_angle = clamp(
                previous_steer,
                real_base.steer_ctrl_ranges[key][0],
                real_base.steer_ctrl_ranges[key][1],
            )
            drive_speed = clamp_delta(
                0.0,
                previous_drive_speed,
                tracking.drive_accel_limit_rps2 * float(model.opt.timestep),
            )
        real_base.steer_command_state[key] = steer_angle
        real_base.drive_command_state[key] = drive_speed
        data.ctrl[real_base.steer_ctrl_addresses[key]] = steer_angle
        data.ctrl[real_base.drive_ctrl_addresses[key]] = drive_speed

    return xy_error, abs_yaw_error


def step_real_configuration(
    mujoco,
    model,
    data,
    qpos_addresses: list[int | None],
    ctrl_addresses: list[int | None],
    joint_names: tuple[str, ...],
    configuration: list[float],
    real_base: RealBaseControl,
    tracking: RealBaseTracking,
    input_mode: str,
    settle_steps: int,
    previous_configuration: list[float] | None = None,
) -> tuple[float, float]:
    duration = max(1, settle_steps) * float(model.opt.timestep)
    feedforward_velocity_world, feedforward_yaw_rate = (
        base_feedforward_from_configurations(
            joint_names,
            previous_configuration,
            configuration,
            duration,
        )
    )

    set_non_base_targets(
        data,
        qpos_addresses,
        ctrl_addresses,
        joint_names,
        configuration,
        input_mode,
        seed_qpos=False,
    )
    max_xy_error = 0.0
    max_yaw_error = 0.0
    for _ in range(max(1, settle_steps)):
        xy_error, yaw_error = update_real_base_controls(
            model,
            data,
            joint_names,
            configuration,
            real_base,
            tracking,
            feedforward_velocity_world,
            feedforward_yaw_rate,
        )
        max_xy_error = max(max_xy_error, xy_error)
        max_yaw_error = max(max_yaw_error, yaw_error)
        mujoco.mj_step(model, data)
    return max_xy_error, max_yaw_error


def apply_configuration(
    mujoco,
    model,
    data,
    qpos_addresses: list[int | None],
    ctrl_addresses: list[int | None],
    joint_names: tuple[str, ...],
    configuration: list[float],
    base_body,
    input_mode: str,
    settle_steps: int,
    seed_qpos: bool = False,
) -> None:
    base_x = 0.0
    base_y = 0.0
    base_yaw = 0.0
    for qpos_address, ctrl_address, name, value in zip(
        qpos_addresses,
        ctrl_addresses,
        joint_names,
        configuration,
    ):
        if name == "base_x":
            base_x = value
        elif name == "base_y":
            base_y = value
        elif name == "base_yaw":
            base_yaw = value
        elif input_mode == "ctrl":
            if ctrl_address is None:
                raise ValueError(f"MuJoCo model is missing ctrl address for joint: {name}")
            data.ctrl[ctrl_address] = value
            if seed_qpos and qpos_address is not None:
                data.qpos[qpos_address] = value
        elif qpos_address is not None:
            data.qpos[qpos_address] = value

    if base_body is not None:
        body_id, origin = base_body
        model.body_pos[body_id][0] = origin[0] + base_x
        model.body_pos[body_id][1] = origin[1] + base_y
        model.body_pos[body_id][2] = origin[2]
        half_yaw = 0.5 * base_yaw
        model.body_quat[body_id][:] = (
            math.cos(half_yaw),
            0.0,
            0.0,
            math.sin(half_yaw),
        )

    if input_mode == "ctrl" and not seed_qpos:
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        if settle_steps == 0:
            mujoco.mj_forward(model, data)
        return

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def interpolated_frames(waypoints: list[list[float]], fps: float, speed: float):
    for start, goal in zip(waypoints, waypoints[1:]):
        max_change = max(abs(goal[index] - start[index]) for index in range(len(start)))
        frame_count = max(2, math.ceil(max_change / speed * fps))
        for frame_index in range(frame_count):
            ratio = (frame_index + 1) / frame_count
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            yield [
                start[index] + (goal[index] - start[index]) * smooth_ratio
                for index in range(len(start))
            ]


def decimate_sequence(values: list, max_count: int) -> list:
    if max_count <= 0 or len(values) <= max_count:
        return values
    if max_count == 1:
        return [values[0]]
    last_index = len(values) - 1
    return [
        values[int(round(index * last_index / (max_count - 1)))]
        for index in range(max_count)
    ]


def sampled_trajectory_configurations(
    waypoints: list[list[float]],
    fps: float,
    speed: float,
    max_count: int,
) -> list[list[float]]:
    if not waypoints:
        return []
    configurations = [waypoints[0]]
    configurations.extend(interpolated_frames(waypoints, fps, speed))
    return decimate_sequence(configurations, max_count)


def append_scene_sphere(mujoco, scene, position, radius: float, rgba) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    scene.ngeom += 1


def append_scene_box(mujoco, scene, half_extents, position, rotation, rgba) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray(half_extents, dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.asarray(rotation, dtype=np.float64).reshape(-1),
        rgba,
    )
    scene.ngeom += 1


def append_scene_connector(mujoco, scene, start, goal, radius: float, rgba) -> None:
    start_pos = np.asarray(start, dtype=np.float64)
    goal_pos = np.asarray(goal, dtype=np.float64)
    if scene.ngeom >= len(scene.geoms) or np.linalg.norm(goal_pos - start_pos) < 1.0e-8:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        start_pos,
        goal_pos,
    )
    scene.geoms[scene.ngeom].rgba[:] = rgba
    scene.ngeom += 1


def draw_path_polyline(mujoco, scene, points: list[np.ndarray], rgba) -> None:
    for start, goal in zip(points, points[1:]):
        append_scene_connector(
            mujoco,
            scene,
            start,
            goal,
            PATH_OVERLAY_LINE_RADIUS,
            rgba,
        )


def draw_path_overlay(
    mujoco,
    viewer,
    base_points: list[np.ndarray],
    payload_points: list[np.ndarray],
    reference_payload_geoms: list[dict] | None = None,
) -> None:
    scene = viewer.user_scn
    if scene is None:
        return
    scene.ngeom = 0
    capacity = len(scene.geoms)
    if capacity <= 0:
        return

    reference_payload_geoms = reference_payload_geoms or []

    def required_geom_count() -> int:
        count = max(0, len(base_points) - 1) + max(0, len(payload_points) - 1)
        if base_points:
            count += 2
        if payload_points:
            count += 2
        return count + len(reference_payload_geoms)

    while len(base_points) > 2 and len(payload_points) > 2 and required_geom_count() > capacity:
        next_count = max(2, len(base_points) - max(1, len(base_points) // 4))
        base_points = decimate_sequence(base_points, next_count)
        payload_points = decimate_sequence(payload_points, next_count)

    draw_path_polyline(mujoco, scene, base_points, PATH_OVERLAY_BASE_RGBA)
    draw_path_polyline(mujoco, scene, payload_points, PATH_OVERLAY_PAYLOAD_RGBA)
    if base_points:
        append_scene_sphere(
            mujoco,
            scene,
            base_points[0],
            PATH_OVERLAY_ENDPOINT_RADIUS,
            PATH_OVERLAY_START_RGBA,
        )
        append_scene_sphere(
            mujoco,
            scene,
            base_points[-1],
            PATH_OVERLAY_ENDPOINT_RADIUS,
            PATH_OVERLAY_GOAL_RGBA,
        )
    if payload_points:
        append_scene_sphere(
            mujoco,
            scene,
            payload_points[0],
            PATH_OVERLAY_ENDPOINT_RADIUS,
            PATH_OVERLAY_START_RGBA,
        )
        append_scene_sphere(
            mujoco,
            scene,
            payload_points[-1],
            PATH_OVERLAY_ENDPOINT_RADIUS,
            PATH_OVERLAY_GOAL_RGBA,
        )
    for ghost_geom in reference_payload_geoms:
        append_scene_box(
            mujoco,
            scene,
            ghost_geom["half_extents"],
            ghost_geom["position"],
            ghost_geom["rotation"],
            ghost_geom["rgba"],
        )


def update_com_markers(
    mujoco,
    model,
    data,
    com_state: dict,
    reference_com: np.ndarray | None = None,
) -> None:
    set_mocap_body(mujoco, model, data, "robot_com_marker", com_state["robot_com"])
    set_mocap_body(mujoco, model, data, "payload_com_marker", com_state["payload_com"])
    set_mocap_body(mujoco, model, data, "combined_com_marker", com_state["total_com"])
    set_mocap_body(
        mujoco,
        model,
        data,
        "reference_com_marker",
        reference_com if reference_com is not None else HIDDEN_MARKER_POSITION,
    )


def compute_reference_com_state_for_configuration(
    mujoco,
    model,
    data,
    qpos_addresses: list[int | None],
    ctrl_addresses: list[int | None],
    joint_names: tuple[str, ...],
    configuration: list[float],
    base_body,
    real_base: RealBaseControl | None,
    robot_body_ids,
    payload_body_ids,
    payload_body_id: int,
    payload_qposadr: int | None,
    payload_offset,
    input_mode: str,
    attach_payload: bool,
    real_payload_mode: str,
) -> dict:
    mujoco.mj_resetData(model, data)
    if data.ctrl.size:
        data.ctrl[:] = 0.0

    if real_base is not None:
        seed_real_configuration(
            mujoco,
            model,
            data,
            qpos_addresses,
            ctrl_addresses,
            joint_names,
            configuration,
            real_base,
            input_mode,
        )
    else:
        apply_configuration(
            mujoco,
            model,
            data,
            qpos_addresses,
            ctrl_addresses,
            joint_names,
            configuration,
            base_body,
            input_mode,
            0,
            seed_qpos=input_mode == "ctrl",
        )

    close_grippers(mujoco, model, data)
    if real_base is not None and real_payload_mode == REAL_PAYLOAD_MODE_RIGID:
        align_attached_payload_to_gripper_midpoint(
            mujoco,
            model,
            data,
            payload_body_id,
            payload_offset,
        )
    elif (real_base is not None or attach_payload) and payload_qposadr is not None:
        place_payload_at_gripper_midpoint(
            mujoco,
            model,
            data,
            payload_qposadr,
            payload_offset,
        )
    else:
        mujoco.mj_forward(model, data)

    return compute_com_state(model, data, robot_body_ids, payload_body_ids)


def format_vec(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"


def named_model_item(item, fallback: str) -> str:
    return item.name if item.name else fallback


def rack_contact_records(model, data) -> list[dict]:
    records = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 < 0 or geom2 < 0:
            continue
        geom1_name = named_model_item(model.geom(geom1), str(geom1))
        geom2_name = named_model_item(model.geom(geom2), str(geom2))
        body1_id = int(model.geom_bodyid[geom1])
        body2_id = int(model.geom_bodyid[geom2])
        body1_name = named_model_item(model.body(body1_id), str(body1_id))
        body2_name = named_model_item(model.body(body2_id), str(body2_id))
        names = (
            geom1_name.lower(),
            geom2_name.lower(),
            body1_name.lower(),
            body2_name.lower(),
        )
        if not any("rack" in name or "shelf" in name for name in names):
            continue
        records.append(
            {
                "geom_pair": (geom1_name, geom2_name),
                "body_pair": (body1_name, body2_name),
                "distance": float(contact.dist),
                "position": np.array(contact.pos, dtype=float),
            }
        )
    return records


def wheel_floor_normal_forces(mujoco, model, data, real_base: RealBaseControl) -> dict[str, float]:
    forces = {key: 0.0 for key in REAL_WHEEL_MODULES}
    contact_force = np.zeros(6, dtype=float)
    body_to_wheel = {
        body_id: key for key, body_id in real_base.wheel_body_ids.items()
    }
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 < 0 or geom2 < 0:
            continue
        if geom1 != real_base.floor_geom_id and geom2 != real_base.floor_geom_id:
            continue
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        wheel_key = body_to_wheel.get(body1) or body_to_wheel.get(body2)
        if wheel_key is None:
            continue
        mujoco.mj_contactForce(model, data, contact_index, contact_force)
        forces[wheel_key] += max(0.0, float(contact_force[0]))
    return forces


def stability_sample(
    mujoco,
    model,
    data,
    robot_body_ids: set[int],
    payload_body_ids: set[int],
    base_body_id: int,
    margin: float,
    real_base: RealBaseControl | None,
) -> dict:
    com_state = compute_com_state(model, data, robot_body_ids, payload_body_ids)
    support_polygon = transform_support_polygon(model, data, base_body_id)
    distances = support_signed_distances(com_state["total_com"][:2], support_polygon)
    sample = {
        "com_state": com_state,
        "support_min_signed_distance": float(np.min(distances)),
        "support_distances": distances,
        "wheel_forces": None,
    }
    if real_base is not None:
        sample["wheel_forces"] = wheel_floor_normal_forces(
            mujoco,
            model,
            data,
            real_base,
        )
    return sample


def print_com_report(
    mujoco,
    model,
    data,
    robot_body_ids,
    payload_body_ids,
    base_body_id,
    margin,
    real_base=None,
) -> None:
    sample = stability_sample(
        mujoco,
        model,
        data,
        robot_body_ids,
        payload_body_ids,
        base_body_id,
        margin,
        real_base,
    )
    com_state = sample["com_state"]
    min_distance = sample["support_min_signed_distance"]
    stable = min_distance >= margin

    print(f"robot_mass_kg: {com_state['robot_mass']:.4f}")
    print(f"payload_mass_kg: {com_state['payload_mass']:.4f}")
    print(f"total_mass_kg: {com_state['total_mass']:.4f}")
    print(f"robot_com_world: {format_vec(com_state['robot_com'])}")
    print(f"payload_com_world: {format_vec(com_state['payload_com'])}")
    print(f"combined_com_world: {format_vec(com_state['total_com'])}")
    print(f"support_margin_m: {margin:.4f}")
    print(f"support_min_signed_distance_m: {min_distance:.4f}")
    print(f"support_stable: {str(stable).lower()}")
    if sample["wheel_forces"] is not None:
        for key, force in sample["wheel_forces"].items():
            print(f"{key}_wheel_normal_force_n: {force:.4f}")


def validate_real_trajectory(
    mujoco,
    model,
    data,
    qpos_addresses,
    ctrl_addresses,
    joint_names,
    waypoints,
    real_base: RealBaseControl,
    tracking: RealBaseTracking,
    robot_body_ids,
    payload_body_ids,
    payload_body_id,
    base_body_id,
    payload_offset,
    args,
) -> None:
    min_support_distance = math.inf
    min_wheel_force = math.inf
    min_wheel_name = ""
    max_base_xy_error = 0.0
    max_base_yaw_error = 0.0
    samples = 0
    rack_contact_count = 0
    rack_contact_frames = 0
    max_rack_penetration = 0.0
    first_rack_contact = None
    worst_rack_contact = None
    max_grasp_site_error = 0.0
    max_payload_target_error = 0.0
    max_payload_target_z_error = 0.0
    final_payload_target_delta = np.zeros(3, dtype=float)
    max_payload_level_tilt_deg = 0.0
    max_abs_payload_forward_elevation_deg = 0.0
    final_payload_level_tilt_deg = 0.0
    final_payload_forward_elevation_deg = 0.0
    max_joint_tracking_errors = {
        joint_name: 0.0
        for qpos_address, joint_name in zip(qpos_addresses, joint_names)
        if qpos_address is not None
    }
    final_joint_tracking_errors = dict(max_joint_tracking_errors)

    def record_sample(target_configuration: list[float]) -> None:
        nonlocal min_support_distance, min_wheel_force, min_wheel_name, samples
        nonlocal rack_contact_count, rack_contact_frames
        nonlocal max_rack_penetration, first_rack_contact, worst_rack_contact
        nonlocal max_grasp_site_error, max_payload_target_error
        nonlocal max_payload_target_z_error, final_payload_target_delta
        nonlocal max_payload_level_tilt_deg, max_abs_payload_forward_elevation_deg
        nonlocal final_payload_level_tilt_deg, final_payload_forward_elevation_deg
        nonlocal final_joint_tracking_errors
        sample = stability_sample(
            mujoco,
            model,
            data,
            robot_body_ids,
            payload_body_ids,
            base_body_id,
            args.support_margin,
            real_base,
        )
        min_support_distance = min(
            min_support_distance,
            sample["support_min_signed_distance"],
        )
        wheel_forces = sample["wheel_forces"] or {}
        for key, force in wheel_forces.items():
            if force < min_wheel_force:
                min_wheel_force = force
                min_wheel_name = key
        grasp_errors = grasp_site_error_distances(mujoco, model, data)
        if grasp_errors:
            max_grasp_site_error = max(
                max_grasp_site_error,
                max(grasp_errors.values()),
            )
        payload_error, payload_delta = payload_target_position_error(
            mujoco,
            model,
            data,
            payload_body_id,
            payload_offset,
        )
        max_payload_target_error = max(max_payload_target_error, payload_error)
        max_payload_target_z_error = max(
            max_payload_target_z_error,
            abs(float(payload_delta[2])),
        )
        final_payload_target_delta = payload_delta
        level_tilt, forward_elevation = payload_tilt_degrees(data, payload_body_id)
        max_payload_level_tilt_deg = max(max_payload_level_tilt_deg, level_tilt)
        max_abs_payload_forward_elevation_deg = max(
            max_abs_payload_forward_elevation_deg,
            abs(forward_elevation),
        )
        final_payload_level_tilt_deg = level_tilt
        final_payload_forward_elevation_deg = forward_elevation
        final_joint_tracking_errors = joint_tracking_errors(
            data,
            qpos_addresses,
            joint_names,
            target_configuration,
        )
        for joint_name, error in final_joint_tracking_errors.items():
            if abs(error) > abs(max_joint_tracking_errors[joint_name]):
                max_joint_tracking_errors[joint_name] = error
        contacts = rack_contact_records(model, data)
        if contacts:
            rack_contact_frames += 1
            rack_contact_count += len(contacts)
            for contact in contacts:
                penetration = max(0.0, -contact["distance"])
                if first_rack_contact is None:
                    first_rack_contact = (samples, contact)
                if penetration > max_rack_penetration:
                    max_rack_penetration = penetration
                    worst_rack_contact = (samples, contact)
        samples += 1

    record_sample(waypoints[0])
    previous_configuration = waypoints[0]
    for configuration in interpolated_frames(waypoints, args.fps, args.speed):
        xy_error, yaw_error = step_real_configuration(
            mujoco,
            model,
            data,
            qpos_addresses,
            ctrl_addresses,
            joint_names,
            configuration,
            real_base,
            tracking,
            args.input_mode,
            args.settle_steps,
            previous_configuration,
        )
        previous_configuration = configuration
        max_base_xy_error = max(max_base_xy_error, xy_error)
        max_base_yaw_error = max(max_base_yaw_error, yaw_error)
        record_sample(configuration)

    if not math.isfinite(min_wheel_force):
        min_wheel_force = 0.0
    com_state = compute_com_state(model, data, robot_body_ids, payload_body_ids)
    print("real_mode: true")
    print(f"real_payload_mode: {args.real_payload_mode}")
    print(f"real_speed: {args.speed:.4f}")
    print(f"robot_mass_kg: {com_state['robot_mass']:.4f}")
    print(f"payload_mass_kg: {com_state['payload_mass']:.4f}")
    print(f"total_mass_kg: {com_state['total_mass']:.4f}")
    print(f"real_settle_steps: {args.settle_steps}")
    print(f"real_initial_settle_steps: {args.real_initial_settle_steps}")
    print(f"real_base_kp_xy: {tracking.kp_xy:.4f}")
    print(f"real_base_kp_yaw: {tracking.kp_yaw:.4f}")
    print(f"real_base_max_speed_mps: {tracking.max_speed_mps:.4f}")
    print(f"real_base_max_yaw_rate_rps: {tracking.max_yaw_rate_rps:.4f}")
    print(f"real_steer_rate_limit_rps: {tracking.steer_rate_limit_rps:.4f}")
    print(f"real_drive_accel_limit_rps2: {tracking.drive_accel_limit_rps2:.4f}")
    print(f"samples: {samples}")
    print(f"min_support_signed_distance_m: {min_support_distance:.4f}")
    print(f"support_margin_m: {args.support_margin:.4f}")
    print(
        "support_stable: "
        f"{str(min_support_distance >= args.support_margin).lower()}"
    )
    print(f"min_wheel_normal_force_n: {min_wheel_force:.4f}")
    print(f"min_wheel_normal_force_name: {min_wheel_name or 'none'}")
    print(
        "wheel_contact_stable: "
        f"{str(min_wheel_force >= REAL_WHEEL_LIFT_THRESHOLD_N).lower()}"
    )
    if args.real_payload_mode == REAL_PAYLOAD_MODE_EQUALITY:
        print(f"max_grasp_site_error_m: {max_grasp_site_error:.6f}")
    print(f"max_payload_target_error_m: {max_payload_target_error:.6f}")
    print(f"max_payload_target_z_error_m: {max_payload_target_z_error:.6f}")
    print(f"final_payload_target_delta_m: {format_vec(final_payload_target_delta)}")
    print(f"max_payload_level_tilt_deg: {max_payload_level_tilt_deg:.4f}")
    print(
        "max_abs_payload_forward_elevation_deg: "
        f"{max_abs_payload_forward_elevation_deg:.4f}"
    )
    print(f"final_payload_level_tilt_deg: {final_payload_level_tilt_deg:.4f}")
    print(
        "final_payload_forward_elevation_deg: "
        f"{final_payload_forward_elevation_deg:.4f}"
    )
    print(f"max_base_tracking_error_m: {max_base_xy_error:.4f}")
    print(f"max_base_yaw_error_rad: {max_base_yaw_error:.4f}")
    if max_joint_tracking_errors:
        max_joint_name, max_joint_error = max(
            max_joint_tracking_errors.items(),
            key=lambda item: abs(item[1]),
        )
        print(f"max_joint_tracking_error_name: {max_joint_name}")
        print(f"max_joint_tracking_error_rad_or_m: {max_joint_error:.6f}")
        print(
            "max_joint_tracking_errors_top: "
            f"{format_largest_joint_errors(max_joint_tracking_errors)}"
        )
        print(
            "final_joint_tracking_errors_top: "
            f"{format_largest_joint_errors(final_joint_tracking_errors)}"
        )
    print(f"rack_contact_count: {rack_contact_count}")
    print(f"rack_contact_frames: {rack_contact_frames}")
    print(f"max_rack_penetration_m: {max_rack_penetration:.6f}")
    if first_rack_contact is not None:
        sample_index, contact = first_rack_contact
        print(f"first_rack_contact_sample: {sample_index}")
        print(
            "first_rack_contact_pair: "
            f"{contact['geom_pair'][0]} <-> {contact['geom_pair'][1]}"
        )
        print(
            "first_rack_contact_bodies: "
            f"{contact['body_pair'][0]} <-> {contact['body_pair'][1]}"
        )
    if worst_rack_contact is not None:
        sample_index, contact = worst_rack_contact
        print(f"worst_rack_contact_sample: {sample_index}")
        print(
            "worst_rack_contact_pair: "
            f"{contact['geom_pair'][0]} <-> {contact['geom_pair'][1]}"
        )
        print(
            "worst_rack_contact_bodies: "
            f"{contact['body_pair'][0]} <-> {contact['body_pair'][1]}"
        )


def main() -> int:
    args = parse_args()
    payload_offset = (
        tuple(float(component) for component in args.payload_offset)
        if args.payload_offset is not None
        else ATTACHED_OBJECT_FRAME_OFFSET
    )
    joint_names: tuple[str, ...] = ()
    waypoints: list[list[float]] = []
    if args.trajectory:
        joint_names, waypoints, trajectory_payload_offset = load_trajectory(
            args.trajectory.resolve()
        )
        if args.payload_offset is None:
            payload_offset = trajectory_payload_offset
        if args.real and joint_names != MOBILITY_PLANNING_JOINTS:
            raise ValueError("--real requires base_x/base_y/base_yaw in the trajectory")

    real_base_tracking = RealBaseTracking(
        kp_xy=args.real_base_kp_xy,
        kp_yaw=args.real_base_kp_yaw,
        max_speed_mps=args.real_base_max_speed,
        max_yaw_rate_rps=args.real_base_max_yaw_rate,
        steer_rate_limit_rps=args.real_steer_rate_limit,
        drive_accel_limit_rps2=args.real_drive_accel_limit,
    )

    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("MuJoCo Python package is required: python3 -m pip install mujoco") from error

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    temporary_model_path: Path | None = None
    loaded_model_path = model_path
    if args.real:
        temporary_model_path = write_real_model_xml(
            model_path,
            payload_offset,
            args.real_payload_mode,
        )
        loaded_model_path = temporary_model_path

    try:
        model = mujoco.MjModel.from_xml_path(str(loaded_model_path))
        reference_model = (
            mujoco.MjModel.from_xml_path(str(loaded_model_path))
            if waypoints else None
        )
    finally:
        if temporary_model_path is not None:
            temporary_model_path.unlink(missing_ok=True)

    data = mujoco.MjData(model)
    payload_body_id = set_payload_mass(mujoco, model, data, args.object_mass)
    payload_body_ids = {payload_body_id}
    payload_qposadr = None
    if not (args.real and args.real_payload_mode == REAL_PAYLOAD_MODE_RIGID):
        payload_qposadr = qposadr_for_first_joint(
            mujoco,
            model,
            OBJECT_FREEJOINT_NAMES,
        )

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_body_id < 0:
        raise ValueError("MuJoCo model is missing body: base_link")
    real_base = resolve_real_base_control(mujoco, model, base_body_id) if args.real else None
    base_body = None if args.real else (
        base_body_id,
        tuple(float(value) for value in model.body_pos[base_body_id]),
    )
    robot_body_ids = descendants_of(model, base_body_id)

    qpos_addresses: list[int | None] = []
    ctrl_addresses: list[int | None] = []
    if waypoints:
        qpos_addresses = resolve_qpos_addresses(mujoco, model, joint_names)
        ctrl_addresses = resolve_ctrl_addresses(mujoco, model, joint_names)

    reference_data = None
    reference_payload_body_id = None
    reference_payload_body_ids = set()
    reference_payload_qposadr = None
    reference_base_body_id = None
    reference_real_base = None
    reference_base_body = None
    reference_robot_body_ids = set()
    reference_qpos_addresses: list[int | None] = []
    reference_ctrl_addresses: list[int | None] = []
    if reference_model is not None:
        reference_data = mujoco.MjData(reference_model)
        reference_payload_body_id = set_payload_mass(
            mujoco,
            reference_model,
            reference_data,
            args.object_mass,
        )
        reference_payload_body_ids = {reference_payload_body_id}
        if not (args.real and args.real_payload_mode == REAL_PAYLOAD_MODE_RIGID):
            reference_payload_qposadr = qposadr_for_first_joint(
                mujoco,
                reference_model,
                OBJECT_FREEJOINT_NAMES,
            )
        reference_base_body_id = mujoco.mj_name2id(
            reference_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base_link",
        )
        if reference_base_body_id < 0:
            raise ValueError("MuJoCo reference model is missing body: base_link")
        reference_real_base = (
            resolve_real_base_control(mujoco, reference_model, reference_base_body_id)
            if args.real else None
        )
        reference_base_body = None if args.real else (
            reference_base_body_id,
            tuple(float(value) for value in reference_model.body_pos[reference_base_body_id]),
        )
        reference_robot_body_ids = descendants_of(reference_model, reference_base_body_id)
        reference_qpos_addresses = resolve_qpos_addresses(
            mujoco,
            reference_model,
            joint_names,
        )
        reference_ctrl_addresses = resolve_ctrl_addresses(
            mujoco,
            reference_model,
            joint_names,
        )

    def reference_com_state(configuration: list[float] | None) -> dict | None:
        if (
            configuration is None or
            reference_model is None or
            reference_data is None or
            reference_payload_body_id is None or
            reference_base_body_id is None
        ):
            return None
        return compute_reference_com_state_for_configuration(
            mujoco,
            reference_model,
            reference_data,
            reference_qpos_addresses,
            reference_ctrl_addresses,
            joint_names,
            configuration,
            reference_base_body,
            reference_real_base,
            reference_robot_body_ids,
            reference_payload_body_ids,
            reference_payload_body_id,
            reference_payload_qposadr,
            payload_offset,
            args.input_mode,
            args.attach_payload,
            args.real_payload_mode,
        )

    def reference_total_com(configuration: list[float] | None) -> np.ndarray | None:
        state = reference_com_state(configuration)
        if state is None:
            return None
        return state["total_com"]

    def reference_payload_ghost_geoms(configuration: list[float] | None) -> list[dict]:
        if (
            configuration is None or
            reference_model is None or
            reference_data is None
        ):
            return []
        if reference_com_state(configuration) is None:
            return []
        geoms = []
        for geom_name in REFERENCE_PAYLOAD_GHOST_GEOM_NAMES:
            geom_id = mujoco.mj_name2id(
                reference_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            if geom_id < 0:
                continue
            if int(reference_model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            rgba = (
                REFERENCE_PAYLOAD_GHOST_GRASP_RGBA
                if "grasp" in geom_name
                else REFERENCE_PAYLOAD_GHOST_RGBA
            )
            geoms.append(
                {
                    "half_extents": np.array(reference_model.geom_size[geom_id][:3], dtype=float),
                    "position": np.array(reference_data.geom_xpos[geom_id], dtype=float),
                    "rotation": np.array(reference_data.geom_xmat[geom_id], dtype=float),
                    "rgba": rgba,
                }
            )
        return geoms

    def refresh_com_markers(reference_configuration: list[float] | None = None) -> None:
        com_state = compute_com_state(model, data, robot_body_ids, payload_body_ids)
        update_com_markers(
            mujoco,
            model,
            data,
            com_state,
            reference_total_com(reference_configuration),
        )
        mujoco.mj_forward(model, data)

    def reset_scene_to_start() -> None:
        mujoco.mj_resetData(model, data)
        if data.ctrl.size:
            data.ctrl[:] = 0.0

        if waypoints:
            if args.real:
                assert real_base is not None
                seed_real_configuration(
                    mujoco,
                    model,
                    data,
                    qpos_addresses,
                    ctrl_addresses,
                    joint_names,
                    waypoints[0],
                    real_base,
                    args.input_mode,
                )
            else:
                apply_configuration(
                    mujoco,
                    model,
                    data,
                    qpos_addresses,
                    ctrl_addresses,
                    joint_names,
                    waypoints[0],
                    base_body,
                    args.input_mode,
                    args.settle_steps,
                    seed_qpos=args.input_mode == "ctrl",
                )
            close_grippers(mujoco, model, data)

        if args.real:
            if args.real_payload_mode == REAL_PAYLOAD_MODE_RIGID:
                align_attached_payload_to_gripper_midpoint(
                    mujoco,
                    model,
                    data,
                    payload_body_id,
                    payload_offset,
                )
            else:
                assert payload_qposadr is not None
                place_payload_at_gripper_midpoint(
                    mujoco,
                    model,
                    data,
                    payload_qposadr,
                    payload_offset,
                )
            assert real_base is not None
            reset_real_base_command_state(data, real_base)
            if waypoints:
                for _ in range(args.real_initial_settle_steps):
                    step_real_configuration(
                        mujoco,
                        model,
                        data,
                        qpos_addresses,
                        ctrl_addresses,
                        joint_names,
                        waypoints[0],
                        real_base,
                        real_base_tracking,
                        args.input_mode,
                        1,
                    )
            else:
                for _ in range(args.real_initial_settle_steps):
                    mujoco.mj_step(model, data)
        elif args.attach_payload:
            assert payload_qposadr is not None
            place_payload_at_gripper_midpoint(
                mujoco,
                model,
                data,
                payload_qposadr,
                payload_offset,
            )

        refresh_com_markers(waypoints[0] if waypoints else None)

    reset_scene_to_start()

    validate_only = args.validate_only or os.environ.get("PRRTC_MUJOCO_VALIDATE_ONLY") == "1"
    if validate_only:
        if args.real and waypoints:
            assert real_base is not None
            validate_real_trajectory(
                mujoco,
                model,
                data,
                qpos_addresses,
                ctrl_addresses,
                joint_names,
                waypoints,
                real_base,
                real_base_tracking,
                robot_body_ids,
                payload_body_ids,
                payload_body_id,
                base_body_id,
                payload_offset,
                args,
            )
            return 0
        print_com_report(
            mujoco,
            model,
            data,
            robot_body_ids,
            payload_body_ids,
            base_body_id,
            args.support_margin,
            real_base,
        )
        return 0

    path_base_points: list[np.ndarray] = []
    path_payload_points: list[np.ndarray] = []
    if waypoints:
        path_configurations = sampled_trajectory_configurations(
            waypoints,
            args.fps,
            args.speed,
            PATH_OVERLAY_MAX_POINTS,
        )
        if args.real:
            assert real_base is not None
            base_origin = real_base.base_origin
        else:
            assert base_body is not None
            base_origin = np.asarray(base_body[1], dtype=float)
        for configuration in path_configurations:
            base_x, base_y, _ = base_pose_from_configuration(joint_names, configuration)
            path_base_points.append(
                np.asarray(
                    [
                        base_origin[0] + base_x,
                        base_origin[1] + base_y,
                        PATH_OVERLAY_BASE_Z,
                    ],
                    dtype=float,
                )
            )
            state = reference_com_state(configuration)
            if state is not None:
                path_payload_points.append(
                    np.asarray(state["payload_com"], dtype=float)
                )

    import mujoco.viewer

    print("MuJoCo viewer: FFW-SG2 rack upper-to-lower scene")
    print(
        "Blue: robot CoM, orange: payload CoM, "
        "green: current combined CoM, magenta: reference combined CoM"
    )
    if args.real:
        print(
            "Real physics mode: "
            f"free base, {args.real_payload_mode} payload, wheel-contact tracking"
        )
    replay_started = not bool(waypoints)
    replay_request_id = 0

    def key_callback(keycode: int) -> None:
        nonlocal replay_started, replay_request_id
        if keycode == SPACE_KEY:
            replay_started = True
            replay_request_id += 1

    if waypoints:
        print(f"Trajectory input mode: {args.input_mode}")
        print("Press Space to start/restart replay.")
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:
        viewer.cam.lookat[:] = (0.48, 0.0, 0.9)
        viewer.cam.distance = 3.4
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -18.0
        draw_path_overlay(
            mujoco,
            viewer,
            path_base_points,
            path_payload_points,
            reference_payload_ghost_geoms(waypoints[0] if waypoints else None),
        )
        viewer.sync()

        frame_period = 1.0 / args.fps
        if not waypoints:
            while viewer.is_running():
                if args.real:
                    mujoco.mj_step(model, data)
                com_state = compute_com_state(
                    model,
                    data,
                    robot_body_ids,
                    payload_body_ids,
                )
                update_com_markers(mujoco, model, data, com_state)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(frame_period)
            return 0

        while viewer.is_running() and not replay_started:
            if args.real:
                assert real_base is not None
                step_real_configuration(
                    mujoco,
                    model,
                    data,
                    qpos_addresses,
                    ctrl_addresses,
                    joint_names,
                    waypoints[0],
                    real_base,
                    real_base_tracking,
                    args.input_mode,
                    args.settle_steps,
                )
            com_state = compute_com_state(
                model,
                data,
                robot_body_ids,
                payload_body_ids,
            )
            update_com_markers(
                mujoco,
                model,
                data,
                com_state,
                reference_total_com(waypoints[0]),
            )
            draw_path_overlay(
                mujoco,
                viewer,
                path_base_points,
                path_payload_points,
                reference_payload_ghost_geoms(waypoints[0]),
            )
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(frame_period)

        handled_replay_request_id = replay_request_id
        while viewer.is_running():
            if replay_request_id != handled_replay_request_id:
                handled_replay_request_id = replay_request_id
            reset_scene_to_start()
            draw_path_overlay(
                mujoco,
                viewer,
                path_base_points,
                path_payload_points,
                reference_payload_ghost_geoms(waypoints[0]),
            )
            viewer.sync()
            time.sleep(0.75)
            replay_interrupted = False
            previous_configuration = waypoints[0]
            for configuration in interpolated_frames(waypoints, args.fps, args.speed):
                if not viewer.is_running():
                    return 0
                if replay_request_id != handled_replay_request_id:
                    handled_replay_request_id = replay_request_id
                    replay_interrupted = True
                    break
                if args.real:
                    assert real_base is not None
                    step_real_configuration(
                        mujoco,
                        model,
                        data,
                        qpos_addresses,
                        ctrl_addresses,
                        joint_names,
                        configuration,
                        real_base,
                        real_base_tracking,
                        args.input_mode,
                        args.settle_steps,
                        previous_configuration,
                    )
                    previous_configuration = configuration
                else:
                    apply_configuration(
                        mujoco,
                        model,
                        data,
                        qpos_addresses,
                        ctrl_addresses,
                        joint_names,
                        configuration,
                        base_body,
                        args.input_mode,
                        args.settle_steps,
                    )
                    if args.attach_payload:
                        assert payload_qposadr is not None
                        place_payload_at_gripper_midpoint(
                            mujoco,
                            model,
                            data,
                            payload_qposadr,
                            payload_offset,
                        )
                com_state = compute_com_state(
                    model,
                    data,
                    robot_body_ids,
                    payload_body_ids,
                )
                update_com_markers(
                    mujoco,
                    model,
                    data,
                    com_state,
                    reference_total_com(configuration),
                )
                draw_path_overlay(
                    mujoco,
                    viewer,
                    path_base_points,
                    path_payload_points,
                    reference_payload_ghost_geoms(configuration),
                )
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(frame_period)
            if replay_interrupted:
                continue
            time.sleep(0.75)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        print(f"simulation error: {error}", file=sys.stderr)
        exit_code = 1
    sys.exit(exit_code)

#!/usr/bin/env python3
"""Replay a 35-DoF IGRIS-C planner trajectory in a live MuJoCo viewer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import xacro

from mujoco_video import (
    DEFAULT_TRAJECTORY_ACCELERATION,
    DEFAULT_TRAJECTORY_PLAYBACK_RATE,
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


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    REPOSITORY_DIR
    / "igris_c_description_public"
    / "mujoco"
    / "igris_c_v2.xml.xacro"
)
CONFIGURATION_DIMENSION = 35
REAL_DEFAULT_SPEED = 0.10 * DEFAULT_TRAJECTORY_PLAYBACK_RATE
# The public IGRIS model runs at 2100 Hz. Thirty-five physics steps per
# 60 Hz target keeps the commanded trajectory speed in simulation time.
REAL_DEFAULT_SETTLE_STEPS = 35
REAL_DEFAULT_INITIAL_SETTLE_STEPS = 1000
DEFAULT_PAYLOAD_MASS_KG = 0.15
REAL_POSITION_KP = 5000.0
REAL_POSITION_FORCE_SCALE = 10.0
REAL_FOOT_FLOOR_FRICTION = "2 0.005 0.0001"
ACTUATED_JOINTS = (
    "l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee_pitch",
    "l_ankle_pitch", "l_ankle_roll", "r_hip_pitch", "r_hip_roll",
    "r_hip_yaw", "r_knee_pitch", "r_ankle_pitch", "r_ankle_roll",
    "waist_pitch", "waist_roll", "waist_yaw",
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",
    "l_elbow_pitch", "l_wrist_yaw", "l_wrist_roll", "l_wrist_pitch",
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
    "r_elbow_pitch", "r_wrist_yaw", "r_wrist_roll", "r_wrist_pitch",
)
FIXED_HAND_JOINTS_RAD = {
    "l_0_thumb_proximal": 0.65,
    "l_1_thumb_middle": 0.55,
    "l_2_thumb_distal": 0.45,
    "r_0_thumb_proximal": -0.65,
    "r_1_thumb_middle": 0.55,
    "r_2_thumb_distal": 0.45,
    **{
        f"{side}_{index}_{finger}_{segment}": angle
        for side in ("l", "r")
        for index, finger, segment, angle in (
            (3, "index", "middle", 0.55),
            (4, "index", "distal", 0.35),
            (5, "middle", "middle", 0.55),
            (6, "middle", "distal", 0.35),
            (7, "ring", "middle", 0.55),
            (8, "ring", "distal", 0.35),
            (9, "little", "middle", 0.55),
            (10, "little", "distal", 0.35),
        )
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help=(
            "Maximum planning-coordinate change per second. Defaults to "
            f"{DEFAULT_TRAJECTORY_SPEED:g}, or {REAL_DEFAULT_SPEED:g} with "
            "--real."
        ),
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=DEFAULT_TRAJECTORY_ACCELERATION,
        help="Maximum planning-coordinate acceleration per second squared.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help=(
            "Replay with MuJoCo dynamics, joint position actuators, a free "
            "floating base, ground/shelf contacts, and an equality-attached payload."
        ),
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=None,
        help="Simulation steps per trajectory target (default: 35 with --real).",
    )
    parser.add_argument(
        "--real-initial-settle-steps",
        type=int,
        default=REAL_DEFAULT_INITIAL_SETTLE_STEPS,
        help="Simulation steps used to settle the initial dynamic state.",
    )
    parser.add_argument(
        "--object-mass",
        type=float,
        default=DEFAULT_PAYLOAD_MASS_KG,
        help="Dynamic payload mass in kg (default: 0.15).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate without opening a viewer window.",
    )
    add_video_arguments(parser, "PATACON_IGRIS_C_VIDEO")
    args = parser.parse_args()
    if args.speed is None:
        args.speed = (
            REAL_DEFAULT_SPEED if args.real else DEFAULT_TRAJECTORY_SPEED
        )
    if args.settle_steps is None:
        args.settle_steps = REAL_DEFAULT_SETTLE_STEPS if args.real else 0
    if (
        not math.isfinite(args.fps)
        or not math.isfinite(args.speed)
        or not math.isfinite(args.acceleration)
        or args.fps <= 0.0
        or args.speed <= 0.0
        or args.acceleration <= 0.0
    ):
        parser.error("--fps, --speed, and --acceleration must be positive")
    if args.settle_steps < 0 or args.real_initial_settle_steps < 0:
        parser.error("settle-step counts must be greater than or equal to zero")
    if not math.isfinite(args.object_mass) or args.object_mass <= 0.0:
        parser.error("--object-mass must be a finite value greater than zero")
    validate_video_arguments(parser, args)
    return args


def finite_vector(value, dimension: int, description: str) -> list[float]:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{description} must contain {dimension} values")
    normalized = [float(component) for component in value]
    if not all(math.isfinite(component) for component in normalized):
        raise ValueError(f"{description} contains a non-finite value")
    return normalized


def load_trajectory(path: Path) -> tuple[list[list[float]], dict, dict, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    waypoints_value = document.get("waypoints")
    if not isinstance(waypoints_value, list) or len(waypoints_value) < 2:
        raise ValueError("trajectory must contain at least two waypoints")
    waypoints = [
        finite_vector(value, CONFIGURATION_DIMENSION, f"waypoint {index}")
        for index, value in enumerate(waypoints_value)
    ]
    expected_start = finite_vector(
        document.get("start"), CONFIGURATION_DIMENSION, "trajectory start"
    )
    if max(
        abs(waypoints[0][index] - expected_start[index])
        for index in range(CONFIGURATION_DIMENSION)
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
    task = document.get("task", {})
    constraints = document.get("constraints", {})
    if not isinstance(task, dict) or not isinstance(constraints, dict):
        raise ValueError("task and constraints must be JSON objects")
    return (
        GeometricPathWaypoints(
            waypoints,
            document.get("geometric_path"),
            document.get("path_smoothing", True),
        ),
        environment,
        task,
        constraints,
    )


def add_world_geom(
    worldbody: ET.Element,
    *,
    name: str,
    geom_type: str,
    position: list[float],
    size: list[float],
    rgba: str,
    euler: list[float] | None = None,
    contact: bool = False,
) -> None:
    attributes = {
        "name": name,
        "type": geom_type,
        "pos": " ".join(str(value) for value in position),
        "size": " ".join(str(value) for value in size),
        "rgba": rgba,
        "contype": "1" if contact else "0",
        "conaffinity": "3" if contact else "0",
    }
    if contact:
        attributes["friction"] = "1 0.005 0.0001"
    if euler is not None:
        attributes["euler"] = " ".join(str(value) for value in euler)
    ET.SubElement(worldbody, "geom", attributes)


def validate_environment(environment: dict) -> None:
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
            box.get("orientation_euler_xyz"), 3, f"box {index} orientation"
        )
        half_extents = finite_vector(
            box.get("half_extents"), 3, f"box {index} half extents"
        )
        if any(value <= 0.0 for value in half_extents):
            raise ValueError(f"box {index} half extents must be positive")


def build_viewer_xml(
    source: Path,
    environment: dict,
    task: dict,
    constraints: dict,
    *,
    real: bool = False,
    object_mass_kg: float = DEFAULT_PAYLOAD_MASS_KG,
) -> bytes:
    document = xacro.process_file(
        str(source),
        mappings={
            "base_type": "pelvis",
            "parallel": "false",
            "end_effector": "hand",
            "fixed": "false",
        },
    )
    root = ET.fromstring(document.toxml())
    # Keyframes use the public model's native coordinate order and are not used
    # for planner-path replay. Dynamic mode retains the native actuators; the
    # pelvis build has no matching IMU site, so its unused sensors are removed.
    removed_tags = (
        ("keyframe", "sensor")
        if real
        else ("actuator", "keyframe", "sensor")
    )
    for tag in removed_tags:
        for element in list(root.findall(tag)):
            root.remove(element)
    if real:
        # MuJoCo 3.6 rejects position actuators that specify both an explicit
        # kv and dampratio. The public file uses kv="0" together with
        # dampratio, so retain its dampratio tuning and remove the redundant kv.
        torque_limit_by_class = {
            "actuator_150": 150.0,
            "actuator_120": 120.0,
            "actuator_90": 90.0,
            "actuator_60": 60.0,
            "actuator_8": 8.0,
            "actuator_7": 7.0,
        }
        for position_actuator in root.findall("./actuator/position"):
            if "dampratio" in position_actuator.attrib:
                position_actuator.attrib.pop("kv", None)
            joint_name = position_actuator.get("joint")
            joint = (
                root.find(f".//joint[@name='{joint_name}']")
                if joint_name is not None
                else None
            )
            actuator_class = joint.get("class") if joint is not None else None
            torque_limit = (
                REAL_POSITION_FORCE_SCALE
                * torque_limit_by_class.get(actuator_class, 60.0)
            )
            position_actuator.set("forcelimited", "true")
            position_actuator.set("kp", str(REAL_POSITION_KP))
            position_actuator.set(
                "forcerange", f"{-torque_limit} {torque_limit}"
            )
        floor = root.find(".//geom[@name='floor']")
        if floor is None:
            raise ValueError("public IGRIS MuJoCo model has no floor geom")
        floor.set("contype", "1")
        floor.set("conaffinity", "3")
        floor.set("friction", REAL_FOOT_FLOOR_FRICTION)
        floor.set("priority", "1")
        for side in ("l", "r"):
            foot = root.find(
                f".//geom[@name='{side}_foot_original_collision']"
            )
            if foot is None:
                raise ValueError(
                    f"public IGRIS MuJoCo model has no {side!r} foot collision geom"
                )
            foot.set("friction", REAL_FOOT_FLOOR_FRICTION)
        collision_material = root.find(".//material[@name='collision_material']")
        if collision_material is not None:
            collision_material.set("rgba", "1.0 0.28 0.1 0.0")
    else:
        # Show the public visual geometry once; the planner collision backend is
        # validated separately from the live viewer.
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "geom" and child.get("class") == "collision":
                    parent.remove(child)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("public IGRIS MuJoCo model has no worldbody")
    for index, sphere in enumerate(environment["sphere"]):
        add_world_geom(
            worldbody,
            name=f"planning_sphere_{index}",
            geom_type="sphere",
            position=sphere["position"],
            size=[float(sphere["radius"])],
            rgba="0.85 0.20 0.15 0.72",
            contact=real,
        )
    for index, cylinder in enumerate(environment["cylinder"]):
        add_world_geom(
            worldbody,
            name=f"planning_cylinder_{index}",
            geom_type="cylinder",
            position=cylinder["position"],
            size=[float(cylinder["radius"]), 0.5 * float(cylinder["length"])],
            rgba="0.85 0.45 0.15 0.72",
            euler=cylinder["orientation_euler_xyz"],
            contact=real,
        )
    for index, box in enumerate(environment["box"]):
        add_world_geom(
            worldbody,
            name=f"planning_box_{index}",
            geom_type="box",
            position=box["position"],
            size=box["half_extents"],
            rgba="0.36 0.29 0.22 0.78",
            euler=box["orientation_euler_xyz"],
            contact=real,
        )

    com = constraints.get("com", {})
    for name, key, height, color in (
        ("raw_support", "raw_support_polygon", 0.0010, "0.20 0.55 0.95 0.25"),
        ("margin_support", "support_polygon", 0.0025, "0.15 0.85 0.35 0.55"),
    ):
        if key not in com:
            continue
        polygon = np.asarray(com[key], dtype=float).reshape(-1, 2)
        minimum = polygon.min(axis=0)
        maximum = polygon.max(axis=0)
        center = 0.5 * (minimum + maximum)
        half_size = 0.5 * (maximum - minimum)
        add_world_geom(
            worldbody,
            name=name,
            geom_type="box",
            position=[float(center[0]), float(center[1]), height],
            size=[float(half_size[0]), float(half_size[1]), 0.001],
            rgba=color,
        )

    for body_name, site_name in (
        ("l_wrist_connector", "l_grasp_planner"),
        ("r_wrist_connector", "r_grasp_planner"),
    ):
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"MuJoCo model is missing body {body_name!r}")
        ET.SubElement(
            body,
            "site",
            {
                "name": site_name,
                "type": "sphere",
                "pos": "-0.025025641714385472 0 -0.124",
                "size": "0.014",
                "rgba": "1.0 0.72 0.05 1.0",
            },
        )

    half_extents = finite_vector(
        task.get("temporary_box_half_extents_m", [0.105, 0.17, 0.125]),
        3,
        "task box half extents",
    )
    payload_attributes = {"name": "planner_payload", "pos": "0 0 0"}
    if not real:
        payload_attributes["mocap"] = "true"
    payload_body = ET.SubElement(worldbody, "body", payload_attributes)
    if real:
        ET.SubElement(payload_body, "freejoint", {"name": "planner_payload_free"})
    ET.SubElement(
        payload_body,
        "geom",
        {
            "name": "planner_payload_box",
            "type": "box",
            "size": " ".join(str(value) for value in half_extents),
            "rgba": "0.55 0.20 0.04 1.0",
            "mass": str(object_mass_kg) if real else "0",
            # Payload contacts the world (conaffinity 3) but not robot geoms
            # (contype/conaffinity 1), avoiding forces inside the grasp.
            "contype": "2" if real else "0",
            "conaffinity": "2" if real else "0",
            "friction": "0.8 0.005 0.0001",
        },
    )
    if real:
        bimanual_target = finite_vector(
            constraints.get("bimanual", {}).get(
                "target", [1.0, 0.0, 0.0, 0.0, 0.0, -0.34, 0.0]
            ),
            7,
            "bimanual target",
        )
        right_from_left = np.asarray(bimanual_target[4:7], dtype=float)
        for site_name, position in (
            ("planner_payload_left_grasp", -0.5 * right_from_left),
            ("planner_payload_right_grasp", 0.5 * right_from_left),
        ):
            ET.SubElement(
                payload_body,
                "site",
                {
                    "name": site_name,
                    "type": "sphere",
                    "pos": " ".join(str(float(value)) for value in position),
                    "size": "0.008",
                    "rgba": "1.0 0.72 0.05 0.8",
                },
            )

        equality = root.find("equality")
        if equality is None:
            equality = ET.Element("equality")
            actuator = root.find("actuator")
            insertion_index = (
                list(root).index(actuator) if actuator is not None else len(list(root))
            )
            root.insert(insertion_index, equality)
        equality.append(
            ET.Element(
                "weld",
                {
                    "name": "planner_payload_left_weld",
                    "site1": "l_grasp_planner",
                    "site2": "planner_payload_left_grasp",
                    "torquescale": "1.0",
                    "solref": "0.006 1",
                    "solimp": "0.95 0.99 0.001",
                },
            )
        )
        equality.append(
            ET.Element(
                "connect",
                {
                    "name": "planner_payload_right_connect",
                    "site1": "r_grasp_planner",
                    "site2": "planner_payload_right_grasp",
                    "solref": "0.006 1",
                    "solimp": "0.95 0.99 0.001",
                },
            )
        )
        for joint in root.findall(".//joint"):
            joint_name = joint.get("name", "")
            if not joint_name.endswith("_backlash"):
                continue
            equality.append(
                ET.Element(
                    "joint",
                    {
                        "name": f"{joint_name}_planning_lock",
                        "joint1": joint_name,
                        "polycoef": "0 0 0 0 0",
                        "solref": "0.001 1",
                        "solimp": "0.999 0.9999 0.0001",
                    },
                )
            )
        for joint_name, value in FIXED_HAND_JOINTS_RAD.items():
            equality.append(
                ET.Element(
                    "joint",
                    {
                        "name": f"{joint_name}_grasp_lock",
                        "joint1": joint_name,
                        "polycoef": f"{value} 0 0 0 0",
                        "solref": "0.004 1",
                        "solimp": "0.95 0.99 0.001",
                    },
                )
            )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def rpy_chain_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rx @ ry @ rz


def resolve_layout(mujoco, model) -> tuple[int, list[int], dict[str, int], int, int, int]:
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
    if free_id < 0 or int(model.jnt_type[free_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("MuJoCo model is missing floating_base free joint")
    joint_addresses: list[int] = []
    for name in ACTUATED_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing IGRIS joint {name!r}")
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))
    finger_addresses: dict[str, int] = {}
    for name in FIXED_HAND_JOINTS_RAD:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing hand joint {name!r}")
        finger_addresses[name] = int(model.jnt_qposadr[joint_id])
    payload_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "planner_payload")
    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "l_grasp_planner")
    right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_grasp_planner")
    if min(payload_id, left_site, right_site) < 0:
        raise ValueError("MuJoCo viewer model is missing planner payload/grasp markers")
    payload_mocap = int(model.body_mocapid[payload_id])
    return (
        int(model.jnt_qposadr[free_id]),
        joint_addresses,
        finger_addresses,
        payload_mocap,
        left_site,
        right_site,
    )


@dataclass(frozen=True)
class RealLayout:
    base_address: int
    joint_addresses: tuple[int, ...]
    control_addresses: tuple[int, ...]
    finger_addresses: dict[str, int]
    payload_address: int
    pelvis_body: int
    payload_body: int
    left_site: int
    right_site: int
    payload_left_site: int
    payload_right_site: int


def named_id(mujoco, model, object_type, name: str, description: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model is missing {description}: {name}")
    return int(object_id)


def resolve_real_layout(mujoco, model) -> RealLayout:
    free_id = named_id(
        mujoco,
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "floating_base",
        "joint",
    )
    if int(model.jnt_type[free_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("floating_base is not a MuJoCo free joint")

    joint_addresses = []
    control_addresses = []
    for name in ACTUATED_JOINTS:
        joint_id = named_id(
            mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name, "IGRIS joint"
        )
        actuator_id = named_id(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"{name}_Pos",
            "IGRIS position actuator",
        )
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))
        control_addresses.append(actuator_id)

    finger_addresses = {}
    for name in FIXED_HAND_JOINTS_RAD:
        joint_id = named_id(
            mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name, "hand joint"
        )
        finger_addresses[name] = int(model.jnt_qposadr[joint_id])

    payload_joint = named_id(
        mujoco,
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "planner_payload_free",
        "payload free joint",
    )
    if int(model.jnt_type[payload_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("planner_payload_free is not a MuJoCo free joint")

    return RealLayout(
        base_address=int(model.jnt_qposadr[free_id]),
        joint_addresses=tuple(joint_addresses),
        control_addresses=tuple(control_addresses),
        finger_addresses=finger_addresses,
        payload_address=int(model.jnt_qposadr[payload_joint]),
        pelvis_body=named_id(
            mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "pelvis", "body"
        ),
        payload_body=named_id(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "planner_payload",
            "body",
        ),
        left_site=named_id(
            mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "l_grasp_planner", "site"
        ),
        right_site=named_id(
            mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "r_grasp_planner", "site"
        ),
        payload_left_site=named_id(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "planner_payload_left_grasp",
            "site",
        ),
        payload_right_site=named_id(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "planner_payload_right_grasp",
            "site",
        ),
    )


def validate_joint_limits(mujoco, model, waypoints: list[list[float]]) -> None:
    tolerance = 1.0e-5
    for waypoint_index, waypoint in enumerate(waypoints):
        for name, value in zip(ACTUATED_JOINTS, waypoint[6:]):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if not model.jnt_limited[joint_id]:
                continue
            lower, upper = model.jnt_range[joint_id]
            if value < lower - tolerance or value > upper + tolerance:
                raise ValueError(
                    f"waypoint {waypoint_index}, {name}={value} is outside "
                    f"[{lower}, {upper}]"
                )


def apply_configuration(mujoco, model, data, layout, configuration: list[float]) -> None:
    (
        base_address,
        joint_addresses,
        finger_addresses,
        payload_mocap,
        left_site,
        right_site,
    ) = layout
    data.qpos[:] = model.qpos0
    data.qpos[base_address : base_address + 3] = configuration[:3]
    quaternion = np.empty(4, dtype=np.float64)
    rotation = rpy_chain_rotation(*configuration[3:6])
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    data.qpos[base_address + 3 : base_address + 7] = quaternion
    for address, value in zip(joint_addresses, configuration[6:]):
        data.qpos[address] = value
    for name, address in finger_addresses.items():
        data.qpos[address] = FIXED_HAND_JOINTS_RAD[name]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    data.mocap_pos[payload_mocap] = 0.5 * (
        data.site_xpos[left_site] + data.site_xpos[right_site]
    )
    mujoco.mju_mat2Quat(
        data.mocap_quat[payload_mocap],
        data.site_xmat[left_site],
    )
    mujoco.mj_forward(model, data)


def set_real_joint_targets(data, layout: RealLayout, configuration: list[float]) -> None:
    for control_address, value in zip(
        layout.control_addresses, configuration[6:]
    ):
        data.ctrl[control_address] = value


def seed_real_configuration(
    mujoco,
    model,
    data,
    layout: RealLayout,
    configuration: list[float],
) -> None:
    mujoco.mj_resetData(model, data)
    if data.ctrl.size:
        data.ctrl[:] = 0.0

    data.qpos[layout.base_address : layout.base_address + 3] = configuration[:3]
    quaternion = np.empty(4, dtype=np.float64)
    rotation = rpy_chain_rotation(*configuration[3:6])
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    data.qpos[layout.base_address + 3 : layout.base_address + 7] = quaternion
    for address, value in zip(layout.joint_addresses, configuration[6:]):
        data.qpos[address] = value
    for name, address in layout.finger_addresses.items():
        data.qpos[address] = FIXED_HAND_JOINTS_RAD[name]
    set_real_joint_targets(data, layout, configuration)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    data.qpos[layout.payload_address : layout.payload_address + 3] = 0.5 * (
        data.site_xpos[layout.left_site] + data.site_xpos[layout.right_site]
    )
    mujoco.mju_mat2Quat(
        data.qpos[layout.payload_address + 3 : layout.payload_address + 7],
        data.site_xmat[layout.left_site],
    )
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def step_real_target(
    mujoco,
    model,
    data,
    layout: RealLayout,
    configuration: list[float],
    steps: int,
) -> None:
    set_real_joint_targets(data, layout, configuration)
    for _ in range(max(1, steps)):
        mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise RuntimeError("IGRIS-C dynamic simulation produced a non-finite state")


def real_state_summary(model, data, layout: RealLayout, reference: list[float]) -> str:
    joint_errors = np.asarray(
        [
            float(data.qpos[address]) - target
            for address, target in zip(layout.joint_addresses, reference[6:])
        ],
        dtype=float,
    )
    base_error = np.asarray(
        data.qpos[layout.base_address : layout.base_address + 3], dtype=float
    ) - np.asarray(reference[:3], dtype=float)
    left_grasp_error = float(
        np.linalg.norm(
            data.site_xpos[layout.left_site]
            - data.site_xpos[layout.payload_left_site]
        )
    )
    right_grasp_error = float(
        np.linalg.norm(
            data.site_xpos[layout.right_site]
            - data.site_xpos[layout.payload_right_site]
        )
    )
    total_mass = float(np.sum(model.body_mass[1:]))
    combined_com = (
        np.sum(model.body_mass[1:, None] * data.xipos[1:], axis=0) / total_mass
    )
    return (
        f"base_xyz_error={np.linalg.norm(base_error):.5f} m, "
        f"max_joint_error={np.max(np.abs(joint_errors)):.5f} rad, "
        f"grasp_position_error=({left_grasp_error:.6f}, "
        f"{right_grasp_error:.6f}) m, "
        f"combined_com_xy=({combined_com[0]:.4f}, {combined_com[1]:.4f})"
    )


def interpolated_frames(
    waypoints: list[list[float]],
    fps: float,
    speed: float,
    acceleration: float = DEFAULT_TRAJECTORY_ACCELERATION,
):
    yield from continuous_trajectory_frames(
        waypoints, fps, speed, acceleration
    )


def configure_replay_camera(mujoco, camera, view: str = "front") -> None:
    configure_camera(
        mujoco,
        camera,
        (0.22, 0.0, 0.75),
        2.4,
        video_view_azimuth(270.0, view),
        video_view_elevation(-15.0, view),
    )


def save_video(
    mujoco,
    model,
    waypoints: list[list[float]],
    fps: float,
    speed: float,
    real: bool,
    settle_steps: int,
    initial_settle_steps: int,
    output_path: Path,
    width: int,
    height: int,
    views: tuple[str, ...],
    acceleration: float = DEFAULT_TRAJECTORY_ACCELERATION,
) -> None:
    configure_model_render_quality(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    if real:
        layout = resolve_real_layout(mujoco, model)
        seed_real_configuration(mujoco, model, data, layout, waypoints[0])
        for _ in range(initial_settle_steps):
            mujoco.mj_step(model, data)
    else:
        layout = resolve_layout(mujoco, model)
        apply_configuration(mujoco, model, data, layout, waypoints[0])

    cameras = {}
    for view in views:
        camera = mujoco.MjvCamera()
        configure_replay_camera(mujoco, camera, view)
        cameras[view] = camera
    renderer = mujoco.Renderer(model, height=height, width=width)
    output_path = output_path.expanduser().resolve()
    start_hold_seconds = 0.5 if real else 0.75
    start_frame_count = max(1, math.ceil(start_hold_seconds * fps))
    end_frame_count = max(1, math.ceil(1.0 * fps))

    def write_frame(writer: MultiViewVideoWriter) -> None:
        for view, camera in cameras.items():
            renderer.update_scene(data, camera=camera)
            writer.write(view, renderer.render())

    mode = "real dynamics" if real else "qpos"
    print(
        f"rendering {len(views)} IGRIS-C MP4 view(s) at {fps:g} fps "
        f"({width}x{height}, {mode}): {', '.join(views)}"
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
                if real:
                    step_real_target(
                        mujoco,
                        model,
                        data,
                        layout,
                        configuration,
                        settle_steps,
                    )
                    if not np.all(np.isfinite(data.qpos)):
                        raise RuntimeError(
                            "IGRIS-C real-mode simulation became non-finite"
                        )
                else:
                    apply_configuration(
                        mujoco, model, data, layout, configuration
                    )
                write_frame(writer)
            for _ in range(end_frame_count):
                write_frame(writer)
            frame_count = writer.frame_count
    finally:
        renderer.close()

    if real:
        print("Final dynamic state: " + real_state_summary(
            model, data, layout, waypoints[-1]
        ))
    print(
        "saved IGRIS-C MP4: "
        + ", ".join(str(path) for path in writer.output_paths.values())
        + f" ({frame_count} frames each, {frame_count / fps:.3f} s)"
    )


def replay(
    mujoco,
    model,
    waypoints: list[list[float]],
    fps: float,
    speed: float,
    acceleration: float = DEFAULT_TRAJECTORY_ACCELERATION,
) -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    layout = resolve_layout(mujoco, model)
    apply_configuration(mujoco, model, data, layout, waypoints[0])
    print("MuJoCo viewer: IGRIS-C planning 경로를 qpos 모드로 반복 재생합니다.")
    print("손가락 22축은 지정한 grasp 자세로 고정되고 박스는 양손을 따라갑니다.")
    print("갈색은 shelf, 파랑/초록은 support 영역입니다. 창을 닫으면 종료됩니다.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            configure_replay_camera(mujoco, viewer.cam)
        viewer.sync()
        frame_period = 1.0 / fps
        while viewer.is_running():
            apply_configuration(mujoco, model, data, layout, waypoints[0])
            viewer.sync()
            time.sleep(0.75)
            deadline = time.perf_counter()
            for configuration in interpolated_frames(
                waypoints, fps, speed, acceleration
            ):
                if not viewer.is_running():
                    return
                apply_configuration(mujoco, model, data, layout, configuration)
                viewer.sync()
                deadline += frame_period
                time.sleep(max(0.0, deadline - time.perf_counter()))
            time.sleep(1.0)


def replay_real(
    mujoco,
    model,
    waypoints: list[list[float]],
    fps: float,
    speed: float,
    settle_steps: int,
    initial_settle_steps: int,
    acceleration: float = DEFAULT_TRAJECTORY_ACCELERATION,
) -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    layout = resolve_real_layout(mujoco, model)
    seed_real_configuration(mujoco, model, data, layout, waypoints[0])
    for _ in range(initial_settle_steps):
        mujoco.mj_step(model, data)
    print("MuJoCo viewer: IGRIS-C planning 경로를 --real 동역학 모드로 재생합니다.")
    print("floating base는 시작 시에만 초기화하며 경로 중에는 직접 덮어쓰지 않습니다.")
    payload_mass = float(model.body_mass[layout.payload_body])
    print(
        "29개 관절은 position actuator로 추종하고 "
        f"{payload_mass:.3f}kg 박스는 양손 equality로 연결됩니다."
    )
    print("Initial dynamic state: " + real_state_summary(model, data, layout, waypoints[0]))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            configure_replay_camera(mujoco, viewer.cam)
        viewer.sync()
        while viewer.is_running():
            seed_real_configuration(mujoco, model, data, layout, waypoints[0])
            for settle_index in range(initial_settle_steps):
                if not viewer.is_running():
                    return
                mujoco.mj_step(model, data)
                if settle_index % 20 == 0:
                    viewer.sync()
            time.sleep(0.5)
            for configuration in interpolated_frames(
                waypoints, fps, speed, acceleration
            ):
                if not viewer.is_running():
                    return
                step_real_target(
                    mujoco,
                    model,
                    data,
                    layout,
                    configuration,
                    settle_steps,
                )
                viewer.sync()
            print("Final dynamic state: " + real_state_summary(model, data, layout, waypoints[-1]))
            time.sleep(1.0)


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    trajectory_path = args.trajectory.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"IGRIS MuJoCo model not found: {model_path}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory not found: {trajectory_path}")

    waypoints, environment, task, constraints = load_trajectory(trajectory_path)
    validate_environment(environment)
    viewer_xml = build_viewer_xml(
        model_path,
        environment,
        task,
        constraints,
        real=args.real,
        object_mass_kg=args.object_mass,
    )
    temporary_name: str | None = None
    try:
        # Keep the expanded XML beside the source so relative mesh paths resolve.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".xml",
            prefix="igris_c_viewer_",
            dir=model_path.parent,
            delete=False,
        ) as temporary:
            temporary.write(viewer_xml)
            temporary_name = temporary.name
        import mujoco

        model = mujoco.MjModel.from_xml_path(temporary_name)
        validate_joint_limits(mujoco, model, waypoints)
        validate_only = args.validate_only or os.environ.get(
            "PATACON_MUJOCO_VALIDATE_ONLY"
        ) == "1"
        if validate_only:
            data = mujoco.MjData(model)
            if args.real:
                layout = resolve_real_layout(mujoco, model)
                seed_real_configuration(mujoco, model, data, layout, waypoints[0])
                validation_steps = min(args.real_initial_settle_steps, 1000)
                for _ in range(validation_steps):
                    mujoco.mj_step(model, data)
                if not np.all(np.isfinite(data.qpos)):
                    raise RuntimeError(
                        "IGRIS-C dynamic validation produced a non-finite state"
                    )
                print(
                    "PASS: IGRIS-C MuJoCo --real input validated "
                    f"({len(waypoints)} waypoints, 29 position-controlled joints, "
                    f"free floating base, payload={args.object_mass:.3f} kg)"
                )
                print("Settled start: " + real_state_summary(
                    model, data, layout, waypoints[0]
                ))
            else:
                layout = resolve_layout(mujoco, model)
                for configuration in waypoints:
                    apply_configuration(mujoco, model, data, layout, configuration)
                print(
                    "PASS: IGRIS-C live MuJoCo viewer input validated "
                    f"({len(waypoints)} waypoints, 35 DoF, 22 fixed finger joints)"
                )
            return 0
        if args.video is not None:
            save_video(
                mujoco,
                model,
                waypoints,
                args.fps,
                args.speed,
                args.real,
                args.settle_steps,
                args.real_initial_settle_steps,
                args.video,
                args.video_width,
                args.video_height,
                args.video_views,
                args.acceleration,
            )
            return 0
        if args.real:
            replay_real(
                mujoco,
                model,
                waypoints,
                args.fps,
                args.speed,
                args.settle_steps,
                args.real_initial_settle_steps,
                args.acceleration,
            )
        else:
            replay(
                mujoco,
                model,
                waypoints,
                args.fps,
                args.speed,
                args.acceleration,
            )
        return 0
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

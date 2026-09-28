#!/usr/bin/env python3
"""Generate and validate the 35-DoF IGRIS-C planning URDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

try:
    import xacro
except ImportError as error:  # pragma: no cover - exercised only on missing setup
    raise SystemExit(
        "Python package 'xacro' is required (for example: pip install xacro)."
    ) from error


MOVABLE_TYPES = {"continuous", "prismatic", "revolute"}
XACRO_NAMESPACE = "http://www.ros.org/wiki/xacro"
SOURCE_PACKAGE_PREFIX = "package://igris_c_description/"

RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]
DESCRIPTION_DIR = REPOSITORY_DIR / "igris_c_description_public"
DEFAULT_SOURCE = DESCRIPTION_DIR / "urdf" / "igris_c_v2.urdf.xacro"
DEFAULT_OUTPUT = RESOURCE_DIR / "igris_c_planning.urdf"
DEFAULT_METADATA = RESOURCE_DIR / "model_metadata.json"
DEFAULT_CONSTRAINT_CONTRACT = RESOURCE_DIR / "constraint_contract.json"

XACRO_MAPPINGS = {
    "base_type": "pelvis",
    "parallel": "false",
    "end_effector": "hand",
}

VIRTUAL_JOINTS = (
    "world_to_x",
    "x_to_y",
    "y_to_z",
    "z_to_roll",
    "roll_to_pitch",
    "pitch_to_yaw",
)
LEFT_LEG_JOINTS = (
    "l_hip_pitch",
    "l_hip_roll",
    "l_hip_yaw",
    "l_knee_pitch",
    "l_ankle_pitch",
    "l_ankle_roll",
)
RIGHT_LEG_JOINTS = (
    "r_hip_pitch",
    "r_hip_roll",
    "r_hip_yaw",
    "r_knee_pitch",
    "r_ankle_pitch",
    "r_ankle_roll",
)
WAIST_JOINTS = ("waist_pitch", "waist_roll", "waist_yaw")
LEFT_ARM_JOINTS = (
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow_pitch",
    "l_wrist_yaw",
    "l_wrist_roll",
    "l_wrist_pitch",
)
RIGHT_ARM_JOINTS = (
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow_pitch",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
)

LEFT_FINGER_JOINTS = (
    "l_0_joint_thumb_proximal",
    "l_1_joint_thumb_middle",
    "l_2_joint_thumb_distal",
    "l_3_joint_index_middle",
    "l_4_joint_index_distal",
    "l_5_joint_middle_middle",
    "l_6_joint_middle_distal",
    "l_7_joint_ring_middle",
    "l_8_joint_ring_distal",
    "l_9_joint_little_middle",
    "l_10_joint_little_distal",
)
RIGHT_FINGER_JOINTS = tuple(
    "r" + name[1:] for name in LEFT_FINGER_JOINTS
)

# Fixed, symmetric box-grasp posture. These angles are baked into the fixed
# joint origins, so hand geometry and inertials remain while planning stays 35-D.
FINGER_GRASP_POSITIONS_RAD = {
    "l_0_joint_thumb_proximal": 0.65,
    "l_1_joint_thumb_middle": 0.55,
    "l_2_joint_thumb_distal": 0.45,
    "r_0_joint_thumb_proximal": -0.65,
    "r_1_joint_thumb_middle": 0.55,
    "r_2_joint_thumb_distal": 0.45,
    **{
        f"{side}_{index}_joint_{finger}_{segment}": angle
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

ACTIVE_JOINTS = (
    VIRTUAL_JOINTS
    + LEFT_LEG_JOINTS
    + RIGHT_LEG_JOINTS
    + WAIST_JOINTS
    + LEFT_ARM_JOINTS
    + RIGHT_ARM_JOINTS
)
SOURCE_MOVABLE_JOINTS = (
    WAIST_JOINTS
    + LEFT_LEG_JOINTS
    + RIGHT_LEG_JOINTS
    + LEFT_ARM_JOINTS
    + LEFT_FINGER_JOINTS
    + RIGHT_ARM_JOINTS
    + RIGHT_FINGER_JOINTS
    + ("neck_yaw", "neck_pitch")
)
JOINTS_FIXED_AT_ZERO = ("neck_yaw", "neck_pitch")

# These limits deliberately match the six-joint floating-base convention used by
# the existing G1 planning model. They are configuration bounds, not actuators.
VIRTUAL_JOINT_SPECS = (
    ("world_to_x", "world", "floating_x", "prismatic", "1 0 0", -2.0, 2.0),
    ("x_to_y", "floating_x", "floating_y", "prismatic", "0 1 0", -2.0, 2.0),
    ("y_to_z", "floating_y", "floating_z", "prismatic", "0 0 1", -2.0, 2.0),
    (
        "z_to_roll",
        "floating_z",
        "floating_roll",
        "revolute",
        "1 0 0",
        -math.pi,
        math.pi,
    ),
    (
        "roll_to_pitch",
        "floating_roll",
        "floating_pitch",
        "revolute",
        "0 1 0",
        -math.pi,
        math.pi,
    ),
    (
        "pitch_to_yaw",
        "floating_pitch",
        "floating_yaw",
        "revolute",
        "0 0 1",
        -math.pi,
        math.pi,
    ),
)

# Sole z is the bottom plane of the MuJoCo foot box:
# center z (-0.0685 m) - half-height (0.0025 m) = -0.071 m.
TASK_FRAMES = {
    "l_sole": {
        "parent": "l_foot_original",
        "xyz": "0.048 0 -0.071",
        "rpy": "0 0 0",
    },
    "r_sole": {
        "parent": "r_foot_original",
        "xyz": "0.048 0 -0.071",
        "rpy": "0 0 0",
    },
    # Move the task frame 10 cm from the palm base toward the fingers and
    # 2.5026 cm along local -x so that the shelf-task start midpoint is at
    # world x=0. The identical offset on both wrists preserves the 34 cm
    # relative pose.
    "l_grasp": {
        "parent": "l_wrist_connector",
        "xyz": "-0.025025641714385472 0 -0.124",
        "rpy": "0 0 0",
    },
    "r_grasp": {
        "parent": "r_wrist_connector",
        "xyz": "-0.025025641714385472 0 -0.124",
        "rpy": "0 0 0",
    },
}

BIMANUAL_SEPARATION_M = 0.34
BIMANUAL_TARGET_TRANSLATION_M = (0.0, -BIMANUAL_SEPARATION_M, 0.0)
BIMANUAL_TARGET_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
PAYLOAD_MASS_KG = 0.15
PAYLOAD_COLLISION_LINK = "attached_payload_collision"
PAYLOAD_COLLISION_JOINT = "attached_payload_collision_fixed"
PAYLOAD_BOX_SIZE_XYZ_M = (0.21, 0.34, 0.25)
PAYLOAD_CENTER_IN_LEFT_GRASP_XYZ_M = (0.0, -0.17, 0.0)

# The contact dimensions come from the two MuJoCo foot boxes. Their sole
# frames are at the center of each bottom face, so these are planar half sizes.
FOOT_SUPPORT_HALF_SIZE_M = (0.1, 0.035)
SUPPORT_MARGIN_M = 0.05
FOOT_TARGETS = (
    (1.0, 0.0, 0.0, 0.0, -0.0115, 0.1897, 0.0),
    (1.0, 0.0, 0.0, 0.0, -0.0115, -0.1897, 0.0),
)
RAW_SUPPORT_POLYGON_XY_M = (
    (-0.1115, -0.2247),
    (0.0885, -0.2247),
    (0.0885, 0.2247),
    (-0.1115, 0.2247),
)
MARGINED_SUPPORT_POLYGON_XY_M = (
    (-0.0615, -0.1747),
    (0.0385, -0.1747),
    (0.0385, 0.1747),
    (-0.0615, 0.1747),
)

# A simple upright configuration used only to validate the frame convention and
# the 0.34 m target before shelf-specific start/goal IK is introduced.
VALIDATED_SEED_NONZERO = {
    "y_to_z": 0.89,
    "l_shoulder_roll": 0.008221739098831353,
    "l_wrist_roll": -0.008221739098831353,
    "r_shoulder_roll": -0.008221739098831353,
    "r_wrist_roll": 0.008221739098831353,
}

# Current shelf-task endpoints. Both lock the widened sole targets, preserve the
# 34 cm bimanual relative pose, and satisfy the 5 cm CoM-margin constraint. The
# start grasp frames retain the bimanual world-axis equality while the torso
# is level and has a 60 degree yaw in world coordinates.
SHELF_START_CONFIGURATION = (
    -0.10552353584615864,
    -0.08609750690396635,
    0.740464169639126,
    0.28516429370543833,
    -0.6428576269613212,
    0.8850177269947921,
    -0.12471804842755553,
    -0.09883079456979697,
    -0.9104226811428755,
    1.375903102191281,
    -0.5781483487166762,
    -0.33289355195549614,
    0.306314903927044,
    -0.15703389853492353,
    -0.6884504441756436,
    1.2338485057439947,
    -0.6981316909999998,
    0.10812721725843127,
    -0.6484637915035528,
    -0.27035182877720376,
    -0.34906589275310745,
    -0.5229344747027915,
    0.25788166042347543,
    0.07191891427561502,
    -0.7853981705924812,
    0.13407105254393384,
    -0.23612069367877694,
    -0.23765256919321864,
    -0.6875636739628872,
    -0.09930497286644596,
    0.825735200681536,
    -0.785398150327359,
    -0.7179232211892548,
    -0.4679991912807191,
    -0.3183034825219579,
)
SHELF_GOAL_CONFIGURATION = (
    -0.029416445037105863,
    -0.001072161431754747,
    0.8819064137637489,
    0.0012100566745907103,
    -0.2835692009751181,
    0.007810785720772758,
    0.2096652266262342,
    0.1488984961291416,
    -0.018419024831001198,
    0.1722335826869323,
    -0.10025813988489611,
    -0.14886359739313687,
    0.21893830736672318,
    -0.14518545754100134,
    0.001929151696064105,
    0.16227957426971656,
    -0.09723613091210172,
    0.14598109217230776,
    -0.23607594417828323,
    0.0023139579053271997,
    0.006426657561840087,
    -1.1110108145075066,
    0.011376175452472622,
    -0.01114225799598548,
    -0.30527668891954785,
    0.010978093917583996,
    -0.008146965652282033,
    -0.10702758817255945,
    -1.1067057957702366,
    -0.007719707229552052,
    0.0013246361016998156,
    -0.30869871569431245,
    -0.006676019657646112,
    0.006309382216506431,
    -0.10784810560790056,
)
if len(SHELF_START_CONFIGURATION) != len(ACTIVE_JOINTS):
    raise ValueError("shelf start configuration does not have 35 coordinates")
if len(SHELF_GOAL_CONFIGURATION) != len(ACTIVE_JOINTS):
    raise ValueError("shelf goal configuration does not have 35 coordinates")
SHELF_START_NONZERO = dict(zip(ACTIVE_JOINTS, SHELF_START_CONFIGURATION))
SHELF_GOAL_NONZERO = dict(zip(ACTIVE_JOINTS, SHELF_GOAL_CONFIGURATION))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--constraint-contract",
        type=Path,
        default=DEFAULT_CONSTRAINT_CONTRACT,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the source and verify that committed artifacts are current",
    )
    return parser.parse_args()


def remove_children(parent: ET.Element, tags: tuple[str, ...]) -> None:
    for tag in tags:
        for child in list(parent.findall(tag)):
            parent.remove(child)


def add_link(robot: ET.Element, name: str) -> None:
    ET.SubElement(robot, "link", {"name": name})


def add_joint(
    robot: ET.Element,
    *,
    name: str,
    parent: str,
    child: str,
    joint_type: str,
    xyz: str | None = None,
    rpy: str | None = None,
    axis: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": joint_type})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    if xyz is not None or rpy is not None:
        ET.SubElement(
            joint,
            "origin",
            {"xyz": xyz or "0 0 0", "rpy": rpy or "0 0 0"},
        )
    if axis is not None:
        ET.SubElement(joint, "axis", {"xyz": axis})
    if lower is not None and upper is not None:
        ET.SubElement(
            joint,
            "limit",
            {
                "lower": format(lower, ".15g"),
                "upper": format(upper, ".15g"),
                "effort": "0",
                "velocity": "0",
            },
        )


def expand_xacro(source: Path) -> ET.Element:
    document = xacro.process_file(str(source), mappings=XACRO_MAPPINGS)
    return ET.fromstring(document.toxml())


def movable_joint_names(robot: ET.Element) -> tuple[str, ...]:
    return tuple(
        joint.get("name", "")
        for joint in robot.findall("joint")
        if joint.get("type") in MOVABLE_TYPES
    )


def total_mass(robot: ET.Element) -> float:
    return sum(
        float(mass.get("value", "nan"))
        for mass in robot.findall("./link/inertial/mass")
    )


def validate_source(robot: ET.Element) -> tuple[float, int, int]:
    if robot.tag != "robot" or robot.get("name") != "igris_c_v2":
        raise ValueError(
            f"unexpected source robot: {robot.tag} {robot.get('name')!r}"
        )
    movable = movable_joint_names(robot)
    if movable != SOURCE_MOVABLE_JOINTS:
        raise ValueError(
            "expanded source movable-joint order changed:\n"
            f"expected {SOURCE_MOVABLE_JOINTS}\nactual   {movable}"
        )
    names = [element.get("name", "") for element in robot.findall("link")]
    names += [element.get("name", "") for element in robot.findall("joint")]
    unexpected = sorted(name for name in names if "parallel" in name)
    if unexpected:
        raise ValueError(f"parallel mechanism leaked into serial model: {unexpected}")
    mass = total_mass(robot)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"invalid source total mass: {mass}")
    return (
        mass,
        len(robot.findall("./link/inertial")),
        len(robot.findall("./link/collision")),
    )


def add_floating_base(robot: ET.Element) -> None:
    existing_links = {link.get("name", "") for link in robot.findall("link")}
    existing_joints = {joint.get("name", "") for joint in robot.findall("joint")}
    virtual_links = (
        "world",
        "floating_x",
        "floating_y",
        "floating_z",
        "floating_roll",
        "floating_pitch",
        "floating_yaw",
    )
    if set(virtual_links) & existing_links or set(VIRTUAL_JOINTS) & existing_joints:
        raise ValueError("source already contains the requested floating-base chain")
    for link_name in virtual_links:
        add_link(robot, link_name)
    for name, parent, child, joint_type, axis, lower, upper in VIRTUAL_JOINT_SPECS:
        add_joint(
            robot,
            name=name,
            parent=parent,
            child=child,
            joint_type=joint_type,
            axis=axis,
            lower=lower,
            upper=upper,
        )
    add_joint(
        robot,
        name="floating_base_mount",
        parent="floating_yaw",
        child="base_link",
        joint_type="fixed",
    )


def freeze_neck(robot: ET.Element) -> None:
    joints = {joint.get("name", ""): joint for joint in robot.findall("joint")}
    for name in JOINTS_FIXED_AT_ZERO:
        joint = joints[name]
        if joint.get("type") != "revolute":
            raise ValueError(
                f"joint {name!r} changed type: expected revolute, "
                f"got {joint.get('type')!r}"
            )
        joint.set("type", "fixed")
        remove_children(
            joint,
            (
                "axis",
                "limit",
                "mimic",
                "dynamics",
                "safety_controller",
                "calibration",
            ),
        )


def rpy_matrix(rpy: tuple[float, float, float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def axis_angle_matrix(axis: tuple[float, float, float], angle: float) -> list[list[float]]:
    norm = math.sqrt(sum(component * component for component in axis))
    x, y, z = (component / norm for component in axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return [
        [cosine + x * x * one_minus_cosine,
         x * y * one_minus_cosine - z * sine,
         x * z * one_minus_cosine + y * sine],
        [y * x * one_minus_cosine + z * sine,
         cosine + y * y * one_minus_cosine,
         y * z * one_minus_cosine - x * sine],
        [z * x * one_minus_cosine - y * sine,
         z * y * one_minus_cosine + x * sine,
         cosine + z * z * one_minus_cosine],
    ]


def multiply3(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(3))
            for column in range(3)
        ]
        for row in range(3)
    ]


def matrix_rpy(rotation: list[list[float]]) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -rotation[2][0])))
    if abs(math.cos(pitch)) > 1.0e-10:
        roll = math.atan2(rotation[2][1], rotation[2][2])
        yaw = math.atan2(rotation[1][0], rotation[0][0])
    else:
        roll = math.atan2(-rotation[1][2], rotation[1][1])
        yaw = 0.0
    return roll, pitch, yaw


def freeze_fingers_at_grasp(robot: ET.Element) -> None:
    joints = {joint.get("name", ""): joint for joint in robot.findall("joint")}
    for name, angle in FINGER_GRASP_POSITIONS_RAD.items():
        joint = joints[name]
        if joint.get("type") != "revolute":
            raise ValueError(f"finger joint {name!r} is not revolute")
        limit = joint.find("limit")
        lower = float(limit.get("lower", "nan"))
        upper = float(limit.get("upper", "nan"))
        if not lower <= angle <= upper:
            raise ValueError(
                f"finger grasp angle {name}={angle} is outside [{lower}, {upper}]"
            )
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        source_rpy = tuple(float(value) for value in origin.get("rpy", "0 0 0").split())
        axis_element = joint.find("axis")
        axis = tuple(float(value) for value in axis_element.get("xyz").split())
        fixed_rotation = multiply3(
            rpy_matrix(source_rpy), axis_angle_matrix(axis, angle)
        )
        origin.set(
            "rpy",
            " ".join(format(value, ".15g") for value in matrix_rpy(fixed_rotation)),
        )
        joint.set("type", "fixed")
        remove_children(
            joint,
            (
                "axis", "limit", "mimic", "dynamics", "safety_controller",
                "calibration",
            ),
        )


def add_task_frames(robot: ET.Element) -> None:
    existing_links = {link.get("name", "") for link in robot.findall("link")}
    for frame, transform in TASK_FRAMES.items():
        if frame in existing_links:
            raise ValueError(f"source already contains task frame {frame!r}")
        add_link(robot, frame)
        add_joint(
            robot,
            name=f"{frame}_fixed",
            parent=transform["parent"],
            child=frame,
            joint_type="fixed",
            xyz=transform["xyz"],
            rpy=transform["rpy"],
        )


def add_payload_collision_model(robot: ET.Element) -> None:
    """Attach the payload geometry without duplicating its CoM mass."""
    existing_links = {link.get("name", "") for link in robot.findall("link")}
    existing_joints = {joint.get("name", "") for joint in robot.findall("joint")}
    if PAYLOAD_COLLISION_LINK in existing_links:
        raise ValueError(
            f"source already contains payload link {PAYLOAD_COLLISION_LINK!r}"
        )
    if PAYLOAD_COLLISION_JOINT in existing_joints:
        raise ValueError(
            f"source already contains payload joint {PAYLOAD_COLLISION_JOINT!r}"
        )

    payload = ET.SubElement(robot, "link", {"name": PAYLOAD_COLLISION_LINK})
    collision = ET.SubElement(
        payload,
        "collision",
        {"name": "attached_payload_box"},
    )
    ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(
        geometry,
        "box",
        {
            "size": " ".join(
                format(value, ".15g") for value in PAYLOAD_BOX_SIZE_XYZ_M
            )
        },
    )
    add_joint(
        robot,
        name=PAYLOAD_COLLISION_JOINT,
        parent="l_grasp",
        child=PAYLOAD_COLLISION_LINK,
        joint_type="fixed",
        xyz=" ".join(
            format(value, ".15g")
            for value in PAYLOAD_CENTER_IN_LEFT_GRASP_XYZ_M
        ),
        rpy="0 0 0",
    )


def rewrite_for_planning(robot: ET.Element, source: Path, output: Path) -> None:
    # CoM needs every physical inertial; visuals are unnecessary for planning.
    for link in robot.findall("link"):
        remove_children(link, ("visual",))

    description_root = source.parent.parent
    relative_root = Path(os.path.relpath(description_root, output.parent)).as_posix()
    replacement_prefix = f"{relative_root}/"
    for mesh in robot.findall(".//collision/geometry/mesh"):
        filename = mesh.get("filename", "")
        if not filename.startswith(SOURCE_PACKAGE_PREFIX):
            raise ValueError(f"unexpected collision mesh path: {filename!r}")
        mesh.set(
            "filename",
            replacement_prefix + filename[len(SOURCE_PACKAGE_PREFIX) :],
        )


def reorder_joints(robot: ET.Element) -> None:
    all_joints = list(robot.findall("joint"))
    by_name = {joint.get("name", ""): joint for joint in all_joints}
    if len(by_name) != len(all_joints):
        raise ValueError("duplicate joint name")
    for joint in all_joints:
        robot.remove(joint)
    fixed_names = [
        joint.get("name", "")
        for joint in all_joints
        if joint.get("name", "") not in ACTIVE_JOINTS
    ]
    for name in ACTIVE_JOINTS + tuple(fixed_names):
        robot.append(by_name[name])


def validate_generated(
    robot: ET.Element,
    *,
    source_mass: float,
    source_inertial_count: int,
    source_collision_count: int,
    output: Path,
) -> dict[str, object]:
    links = list(robot.findall("link"))
    joints = list(robot.findall("joint"))
    link_names = [link.get("name", "") for link in links]
    joint_names = [joint.get("name", "") for joint in joints]
    if len(set(link_names)) != len(link_names):
        raise ValueError("duplicate link name")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("duplicate joint name")

    movable = movable_joint_names(robot)
    if movable != ACTIVE_JOINTS:
        raise ValueError(
            "generated movable-joint order differs from ACTIVE_JOINTS:\n"
            f"expected {ACTIVE_JOINTS}\nactual   {movable}"
        )
    if len(ACTIVE_JOINTS) != 35:
        raise AssertionError("IGRIS planning configuration must stay at 35 DoF")

    limits: dict[str, dict[str, float]] = {}
    for joint in joints:
        name = joint.get("name", "")
        if joint.get("type") not in MOVABLE_TYPES:
            continue
        axis = joint.find("axis")
        limit = joint.find("limit")
        if axis is None or not axis.get("xyz"):
            raise ValueError(f"movable joint {name!r} has no axis")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"movable joint {name!r} has no finite limits")
        lower = float(limit.get("lower", "nan"))
        upper = float(limit.get("upper", "nan"))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"invalid limits for {name!r}: [{lower}, {upper}]")
        limits[name] = {"lower": lower, "upper": upper}

    parents: dict[str, str] = {}
    children_by_parent: dict[str, list[str]] = {name: [] for name in link_names}
    for joint in joints:
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        if parent_element is None or child_element is None:
            raise ValueError(f"joint {joint.get('name')!r} lacks parent or child")
        parent = parent_element.get("link", "")
        child = child_element.get("link", "")
        if parent not in children_by_parent or child not in children_by_parent:
            raise ValueError(
                f"joint {joint.get('name')!r} references an unknown link"
            )
        if child in parents:
            raise ValueError(f"link {child!r} has multiple parents")
        parents[child] = parent
        children_by_parent[parent].append(child)

    roots = sorted(set(link_names) - parents.keys())
    if roots != ["world"]:
        raise ValueError(f"generated model must have only root 'world', got {roots}")
    reachable: set[str] = set()
    stack = ["world"]
    while stack:
        link = stack.pop()
        if link in reachable:
            raise ValueError(f"cycle detected at link {link!r}")
        reachable.add(link)
        stack.extend(children_by_parent[link])
    disconnected = sorted(set(link_names) - reachable)
    if disconnected:
        raise ValueError(f"links disconnected from world: {disconnected}")

    generated_mass = total_mass(robot)
    if not math.isclose(generated_mass, source_mass, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"physical mass changed: source={source_mass}, generated={generated_mass}"
        )
    inertial_count = len(robot.findall("./link/inertial"))
    collision_count = len(robot.findall("./link/collision"))
    if inertial_count != source_inertial_count:
        raise ValueError("source inertials were not preserved")
    if collision_count != source_collision_count + 1:
        raise ValueError(
            "source collision geometry plus one attached payload was not preserved"
        )
    if robot.findall("./link/visual"):
        raise ValueError("planning model unexpectedly contains visual geometry")

    expected_frames = set(TASK_FRAMES)
    if not expected_frames.issubset(link_names):
        raise ValueError("one or more task frames are missing")

    payload_link = robot.find(f"./link[@name='{PAYLOAD_COLLISION_LINK}']")
    if payload_link is None:
        raise ValueError("attached payload collision link is missing")
    if payload_link.find("inertial") is not None:
        raise ValueError("payload collision link must remain massless")
    payload_collisions = payload_link.findall("collision")
    if len(payload_collisions) != 1:
        raise ValueError("payload collision link must contain exactly one geometry")
    payload_box = payload_collisions[0].find("geometry/box")
    if payload_box is None:
        raise ValueError("attached payload collision geometry is not a box")
    payload_size = tuple(
        float(value) for value in payload_box.get("size", "").split()
    )
    if payload_size != PAYLOAD_BOX_SIZE_XYZ_M:
        raise ValueError(f"attached payload size changed: {payload_size}")

    payload_joint = robot.find(f"./joint[@name='{PAYLOAD_COLLISION_JOINT}']")
    if payload_joint is None or payload_joint.get("type") != "fixed":
        raise ValueError("attached payload fixed joint is missing")
    payload_parent = payload_joint.find("parent")
    payload_child = payload_joint.find("child")
    payload_origin = payload_joint.find("origin")
    if (
        payload_parent is None
        or payload_parent.get("link") != "l_grasp"
        or payload_child is None
        or payload_child.get("link") != PAYLOAD_COLLISION_LINK
        or payload_origin is None
    ):
        raise ValueError("attached payload joint transform chain is invalid")
    payload_xyz = tuple(
        float(value) for value in payload_origin.get("xyz", "").split()
    )
    if payload_xyz != PAYLOAD_CENTER_IN_LEFT_GRASP_XYZ_M:
        raise ValueError(f"attached payload center changed: {payload_xyz}")

    for mesh in robot.findall(".//collision/geometry/mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("package://") or Path(filename).is_absolute():
            raise ValueError(f"mesh path is not relocatable: {filename!r}")
        mesh_path = (output.parent / filename).resolve()
        if not mesh_path.is_file():
            raise ValueError(f"collision mesh does not exist: {mesh_path}")

    return {
        "joint_limits": limits,
        "link_count": len(links),
        "joint_count": len(joints),
        "fixed_joint_count": sum(
            joint.get("type") == "fixed" for joint in joints
        ),
        "inertial_count": inertial_count,
        "collision_count": collision_count,
        "total_mass_kg": generated_mass,
    }


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def xacro_dependencies(source: Path) -> tuple[Path, ...]:
    dependencies: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in dependencies:
            return
        dependencies.add(resolved)
        root = ET.parse(resolved).getroot()
        include_tag = f"{{{XACRO_NAMESPACE}}}include"
        for include in root.findall(f".//{include_tag}"):
            filename = include.get("filename", "")
            if "$" in filename:
                raise ValueError(f"dynamic xacro include is not auditable: {filename!r}")
            visit((resolved.parent / filename).resolve())

    visit(source)
    return tuple(sorted(dependencies))


def serialize_robot(robot: ET.Element) -> bytes:
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    return ET.tostring(robot, encoding="utf-8", xml_declaration=True)


def build_metadata(
    *,
    source: Path,
    output: Path,
    urdf_bytes: bytes,
    validation: dict[str, object],
) -> dict[str, object]:
    source_files = {
        display_path(path): sha256_file(path) for path in xacro_dependencies(source)
    }
    reference_paths = (
        source.parent.parent / "mujoco" / "igris_c_v2.xml.xacro",
        source.parent.parent / "mujoco" / "igris_c_lowerarm_module.xml.xacro",
    )
    references = {
        display_path(path): sha256_file(path) for path in reference_paths
    }
    configuration = [
        {
            "index": index,
            "name": name,
            "group": (
                "floating_base"
                if name in VIRTUAL_JOINTS
                else "left_leg"
                if name in LEFT_LEG_JOINTS
                else "right_leg"
                if name in RIGHT_LEG_JOINTS
                else "waist"
                if name in WAIST_JOINTS
                else "left_arm"
                if name in LEFT_ARM_JOINTS
                else "right_arm"
            ),
        }
        for index, name in enumerate(ACTIVE_JOINTS)
    ]
    return {
        "schema_version": 1,
        "model": "IGRIS-C v2 planning model",
        "dof": len(ACTIVE_JOINTS),
        "root_link": "world",
        "source_xacro": display_path(source),
        "source_xacro_mappings": XACRO_MAPPINGS,
        "source_files_sha256": source_files,
        "reference_files_sha256": references,
        "generated_urdf": display_path(output),
        "generated_urdf_sha256": sha256_bytes(urdf_bytes),
        "configuration": configuration,
        "joint_limits": validation["joint_limits"],
        "fixed_at_zero": list(JOINTS_FIXED_AT_ZERO),
        "fixed_finger_grasp_positions_rad": FINGER_GRASP_POSITIONS_RAD,
        "floating_base": {
            "joint_order": list(VIRTUAL_JOINTS),
            "transform_order": "Tx * Ty * Tz * Rx * Ry * Rz",
            "actuated": False,
        },
        "task_frames": TASK_FRAMES,
        "foot_reference": {
            "collision_box_center_m": [0.048, 0.0, -0.0685],
            "collision_box_half_size_m": [0.1, 0.035, 0.0025],
            "sole_plane_z_m": -0.071,
        },
        "grasp_frame_status": "palm task frame 0.10 m forward and 0.0250256417 m along local -x from the wrist-mount frame with articulated fingers fixed in a box-grasp posture",
        "default_bimanual_separation_m": BIMANUAL_SEPARATION_M,
        "constraint_contract": display_path(DEFAULT_CONSTRAINT_CONTRACT),
        "end_effector_in_source": "hand",
        "visual_geometry_removed": True,
        "inertials_preserved": True,
        "collision_geometry_preserved": True,
        "payload_collision_model": {
            "link": PAYLOAD_COLLISION_LINK,
            "joint": PAYLOAD_COLLISION_JOINT,
            "parent_frame": "l_grasp",
            "center_in_parent_xyz_m": list(
                PAYLOAD_CENTER_IN_LEFT_GRASP_XYZ_M
            ),
            "box_size_xyz_m": list(PAYLOAD_BOX_SIZE_XYZ_M),
            "massless_collision_only": True,
        },
        "counts": {
            "links": validation["link_count"],
            "joints": validation["joint_count"],
            "movable_joints": len(ACTIVE_JOINTS),
            "fixed_joints": validation["fixed_joint_count"],
            "inertials": validation["inertial_count"],
            "collisions": validation["collision_count"],
        },
        "robot_mass_with_fixed_hands_without_payload_kg": validation["total_mass_kg"],
    }


def build_constraint_contract(urdf_bytes: bytes) -> dict[str, object]:
    robot = ET.fromstring(urdf_bytes)
    robot_mass_kg = sum(
        float(mass.get("value", "0"))
        for mass in robot.findall("./link/inertial/mass")
    )
    separation = math.sqrt(
        sum(component * component for component in BIMANUAL_TARGET_TRANSLATION_M)
    )
    if not math.isclose(
        separation,
        BIMANUAL_SEPARATION_M,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("bimanual target translation is not 0.34 m")
    equality_dimension = 12 + 6 + 2
    seed_configuration = [
        VALIDATED_SEED_NONZERO.get(name, 0.0) for name in ACTIVE_JOINTS
    ]
    start_configuration = list(SHELF_START_CONFIGURATION)
    goal_configuration = list(SHELF_GOAL_CONFIGURATION)
    return {
        "schema_version": 1,
        "task": "igris_c_bimanual_shelf_lift",
        "planning_model": DEFAULT_OUTPUT.name,
        "planning_model_sha256": sha256_bytes(urdf_bytes),
        "configuration_dimension": len(ACTIVE_JOINTS),
        "constraints": {
            "feet": {
                "kind": "two_pose_equalities",
                "frames": ["l_sole", "r_sole"],
                "target_policy": "lock each world pose from the shelf-task start configuration",
                "targets": [
                    {
                        "quaternion_wxyz": list(target[:4]),
                        "translation_xyz_m": list(target[4:]),
                    }
                    for target in FOOT_TARGETS
                ],
                "residual_dimension": 12,
            },
            "bimanual": {
                "kind": "relative_pose_equality",
                "left_frame": "l_grasp",
                "right_frame": "r_grasp",
                "relative_pose_convention": "inverse(T_world_l_grasp) * T_world_r_grasp",
                "target": {
                    "quaternion_wxyz": list(BIMANUAL_TARGET_QUATERNION_WXYZ),
                    "translation_xyz_m": list(BIMANUAL_TARGET_TRANSLATION_M),
                },
                "hand_origin_distance_m": BIMANUAL_SEPARATION_M,
                "residual_dimension": 6,
            },
            "bimanual_axis": {
                "kind": "world_axis_alignment_equality",
                "frame": "l_grasp",
                "local_axis": [1.0, 0.0, 0.0],
                "target_world_axis": [0.0, 0.0, 1.0],
                "residual": ["world_x_component", "world_y_component"],
                "right_frame_policy": "inherited through the bimanual relative-pose equality",
                "residual_dimension": 2,
            },
            "center_of_mass": {
                "kind": "support_polygon_inequality",
                "point": "robot_and_payload_center_of_mass_xy",
                "support_policy": "convex hull of both foot boxes, inset from every outer edge",
                "foot_support_half_size_xy_m": list(FOOT_SUPPORT_HALF_SIZE_M),
                "raw_support_polygon_xy_m": [
                    list(vertex) for vertex in RAW_SUPPORT_POLYGON_XY_M
                ],
                "support_margin_m": SUPPORT_MARGIN_M,
                "support_polygon_xy_m": [
                    list(vertex) for vertex in MARGINED_SUPPORT_POLYGON_XY_M
                ],
                "inactive_policy": "zero residual and zero Jacobian while inside the polygon",
                "active_residual": "xy correction vector to the closest point in the support polygon",
                "residual_dimension": 2,
                "mass_model": (
                    f"{robot_mass_kg:.3f} kg robot with fixed articulated hands "
                    f"plus {PAYLOAD_MASS_KG:.2f} kg point-mass payload at the "
                    "midpoint of l_grasp and r_grasp"
                ),
            },
        },
        "tangent_space": {
            "constraints": ["feet", "bimanual", "bimanual_axis"],
            "jacobian_shape": [equality_dimension, len(ACTIVE_JOINTS)],
            "expected_full_rank": equality_dimension,
            "tangent_dimension": len(ACTIVE_JOINTS) - equality_dimension,
        },
        "projection": {
            "constraints": [
                "feet",
                "center_of_mass",
                "bimanual",
                "bimanual_axis",
            ],
            "combined_jacobian_shape": [22, len(ACTIVE_JOINTS)],
            "single_combined_projection_step": True,
            "parallel_projection": {
                "waypoint_smoothness": True,
                "node_projection_threshold_policy": "granularity * projection_smoothness_threshold",
                "reference_repository": "/home/dam2/gh_ws/new/cpRRTC_IGRIS",
                "planner_source_policy": "do not modify planner algorithm sources",
            },
        },
        "shelf_task": {
            "start_configuration": start_configuration,
            "goal_configurations": [goal_configuration],
            "start_nonzero_configuration": SHELF_START_NONZERO,
            "goal_nonzero_configuration": SHELF_GOAL_NONZERO,
            "expected_start_total_com_xyz_m": [
                -0.058500250336,
                -0.048470585884,
                0.718590288055,
            ],
            "expected_goal_total_com_xyz_m": [
                -0.012301933953,
                -0.000381691781,
                0.848256160745,
            ],
            "temporary_grasp_interface": False,
            "grasp_interface": "palm task frames 0.10 m forward and 0.0250256417 m along local -x from the wrist-mount frames",
            "payload_mass_kg": PAYLOAD_MASS_KG,
            "payload_collision_model": {
                "link": PAYLOAD_COLLISION_LINK,
                "parent_frame": "l_grasp",
                "center_in_parent_xyz_m": list(
                    PAYLOAD_CENTER_IN_LEFT_GRASP_XYZ_M
                ),
                "box_size_xyz_m": list(PAYLOAD_BOX_SIZE_XYZ_M),
                "massless_collision_only": True,
            },
        },
        "constraint_consistent_seed": {
            "purpose": "frame and Jacobian/projection validation; not the shelf-task start pose",
            "configuration": seed_configuration,
            "nonzero_configuration": VALIDATED_SEED_NONZERO,
            "expected_world_frames": {
                "l_sole": {
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "translation_xyz_m": [-0.0115, 0.0897, 0.0],
                },
                "r_sole": {
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "translation_xyz_m": [-0.0115, -0.0897, 0.0],
                },
                "l_grasp": {
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "translation_xyz_m": [-0.075025641714, 0.17, 0.611475334644],
                },
                "r_grasp": {
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "translation_xyz_m": [-0.075025641714, -0.17, 0.611475334644],
                },
            },
            "expected_total_com_xyz_m": [
                -0.039745203365,
                0.000313213493,
                0.823799044756,
            ],
            "validated_with": "Pinocchio 4.1.0",
        },
    }


def generate(source: Path, output: Path) -> tuple[bytes, dict[str, object]]:
    robot = expand_xacro(source)
    source_mass, source_inertials, source_collisions = validate_source(robot)
    robot.set("name", "igris_c_v2_planning")
    freeze_neck(robot)
    freeze_fingers_at_grasp(robot)
    add_floating_base(robot)
    add_task_frames(robot)
    add_payload_collision_model(robot)
    rewrite_for_planning(robot, source, output)
    reorder_joints(robot)
    validation = validate_generated(
        robot,
        source_mass=source_mass,
        source_inertial_count=source_inertials,
        source_collision_count=source_collisions,
        output=output,
    )
    urdf_bytes = serialize_robot(robot)
    metadata = build_metadata(
        source=source,
        output=output,
        urdf_bytes=urdf_bytes,
        validation=validation,
    )
    return urdf_bytes, metadata


def check_file(path: Path, expected: bytes) -> None:
    if not path.is_file():
        raise ValueError(f"generated artifact is missing: {path}")
    if path.read_bytes() != expected:
        raise ValueError(
            f"generated artifact is stale: {path}\n"
            "run prepare_planning_urdf.py without --check"
        )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    metadata_path = args.metadata.resolve()
    constraint_contract_path = args.constraint_contract.resolve()
    if not source.is_file():
        raise SystemExit(f"source xacro does not exist: {source}")
    try:
        urdf_bytes, metadata = generate(source, output)
        metadata_bytes = (
            json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
        )
        constraint_contract_bytes = (
            json.dumps(
                build_constraint_contract(urdf_bytes),
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        if args.check:
            check_file(output, urdf_bytes)
            check_file(metadata_path, metadata_bytes)
            check_file(constraint_contract_path, constraint_contract_bytes)
            print(
                f"PASS: {len(ACTIVE_JOINTS)} DoF, "
                f"{metadata['counts']['links']} links, "
                "mass="
                f"{metadata['robot_mass_with_fixed_hands_without_payload_kg']:.3f} kg"
            )
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        constraint_contract_path.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(urdf_bytes)
        metadata_path.write_bytes(metadata_bytes)
        constraint_contract_path.write_bytes(constraint_contract_bytes)
        print(f"wrote {output}")
        print(f"wrote {metadata_path}")
        print(f"wrote {constraint_contract_path}")
    except (KeyError, OSError, ET.ParseError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()

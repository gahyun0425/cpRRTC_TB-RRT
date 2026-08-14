#!/usr/bin/env python3
"""Generate the collision-only 15-DoF FFW-SG2 planning URDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


ACTIVE_JOINTS = (
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

JOINTS_FIXED_AT_ZERO = (
    "head_joint1",
    "head_joint2",
    "gripper_l_joint1",
    "gripper_l_joint2",
    "gripper_l_joint3",
    "gripper_l_joint4",
    "gripper_r_joint1",
    "gripper_r_joint2",
    "gripper_r_joint3",
    "gripper_r_joint4",
    "left_wheel_steer",
    "left_wheel_drive",
    "right_wheel_steer",
    "right_wheel_drive",
    "rear_wheel_steer",
    "rear_wheel_drive",
)

ORIGINAL_ASSET_PREFIX = (
    "/home/dam2/gh_ws/tb_rrt_ws/src/"
    "cptbrrt_pkg/models/ffw_sg2/assets/"
)
PLANNING_ASSET_PREFIX = "../../ffw_lift/assets/"

RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]
DEFAULT_SOURCE = REPOSITORY_DIR / "ffw_lift" / "ffw_sg2.urdf"
DEFAULT_OUTPUT = RESOURCE_DIR / "ffw_sg2_planning.urdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def remove_children(parent: ET.Element, tag: str) -> None:
    for child in list(parent.findall(tag)):
        parent.remove(child)


def generate(source: Path, output: Path) -> None:
    tree = ET.parse(source)
    robot = tree.getroot()
    if robot.tag != "robot" or robot.get("name") != "ffw_sg2_follower":
        raise ValueError(f"unexpected source robot: {robot.tag} {robot.get('name')!r}")

    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    missing = (set(ACTIVE_JOINTS) | set(JOINTS_FIXED_AT_ZERO)) - joints.keys()
    if missing:
        raise ValueError(f"source URDF is missing joints: {sorted(missing)}")

    # Planning and collision generation do not require visual or inertial data.
    # Removing them also eliminates the external RealSense visual dependency.
    for link in robot.findall("link"):
        remove_children(link, "visual")
        remove_children(link, "inertial")

    # The fixed transform is the original joint origin evaluated at q=0.
    for joint_name in JOINTS_FIXED_AT_ZERO:
        joint = joints[joint_name]
        if joint.get("type") != "revolute":
            raise ValueError(
                f"joint {joint_name!r} changed type: expected revolute, "
                f"got {joint.get('type')!r}"
            )
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "mimic", "dynamics", "safety_controller"):
            remove_children(joint, tag)

    for mesh in robot.findall(".//collision/geometry/mesh"):
        filename = mesh.get("filename", "")
        if not filename.startswith(ORIGINAL_ASSET_PREFIX):
            raise ValueError(f"unexpected collision mesh path: {filename!r}")
        mesh.set(
            "filename",
            PLANNING_ASSET_PREFIX + filename[len(ORIGINAL_ASSET_PREFIX) :],
        )

    movable = tuple(
        joint.get("name", "")
        for joint in robot.findall("joint")
        if joint.get("type") in {"continuous", "prismatic", "revolute"}
    )
    if movable != ACTIVE_JOINTS:
        raise ValueError(
            "generated movable-joint order differs from ACTIVE_JOINTS: "
            f"{movable}"
        )

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    generate(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

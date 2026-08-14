#!/usr/bin/env python3
"""Generate the lift-plus-right-arm 8-DoF FFW-SG2 planning resources."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from prepare_collision_models import (
    BATCH_SIZE,
    build_approximation,
    load_and_validate_spheres,
    validate_containment,
    append_sphere_collision,
)


RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]

SOURCE_PLANNING_URDF = RESOURCE_DIR / "ffw_sg2_planning.urdf"
SOURCE_SPHERES = RESOURCE_DIR / "ffw_sg2_fine_spheres.json"
SOURCE_PROBLEMS = REPOSITORY_DIR / "scripts" / "ffw_sg2_problems.json"

PLANNING_OUTPUT = RESOURCE_DIR / "ffw_sg2_single_planning.urdf"
FINE_OUTPUT = RESOURCE_DIR / "ffw_sg2_single_spherized.urdf"
APPROX_OUTPUT = RESOURCE_DIR / "ffw_sg2_single_spherized_approx.urdf"
PROBLEM_OUTPUT = REPOSITORY_DIR / "scripts" / "ffw_sg2_single_problems.json"
METADATA_OUTPUT = RESOURCE_DIR / "ffw_sg2_single_collision_metadata.json"

DUAL_ACTIVE_JOINTS = (
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
LEFT_ARM_JOINTS = tuple(f"arm_l_joint{index}" for index in range(1, 8))
SINGLE_ACTIVE_JOINTS = (
    "lift_joint",
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
)
SINGLE_SOURCE_INDICES = (0, 8, 9, 10, 11, 12, 13, 14)
LOWER_LIMITS = (-0.5, -3.14, -3.14, -3.14, -2.9361, -3.14, -1.57, -1.5804)
UPPER_LIMITS = (0.0, 3.14, 0.0, 3.14, 1.0786, 3.14, 1.57, 1.8201)

# These values are validated again against the official Cricket output.
EXPECTED_JOINT_FLAG_STRIDE = 9
EXPECTED_TRANSFORM_SLOTS = 1
FINE_SPHERE_COUNT = 124
APPROX_SPHERE_COUNT = 27


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_children(parent: ET.Element, tags: tuple[str, ...]) -> None:
    for tag in tags:
        for child in list(parent.findall(tag)):
            parent.remove(child)


def active_joint_order(robot: ET.Element) -> tuple[str, ...]:
    return tuple(
        joint.get("name", "")
        for joint in robot.findall("joint")
        if joint.get("type") in {"continuous", "prismatic", "revolute"}
    )


def generate_planning_urdf() -> None:
    tree = ET.parse(SOURCE_PLANNING_URDF)
    robot = tree.getroot()
    if active_joint_order(robot) != DUAL_ACTIVE_JOINTS:
        raise ValueError("source planning URDF does not have the canonical 15-DoF order")

    joints = {joint.get("name", ""): joint for joint in robot.findall("joint")}
    for name in LEFT_ARM_JOINTS:
        joint = joints[name]
        if joint.get("type") != "revolute":
            raise ValueError(f"expected revolute left-arm joint: {name}")
        joint.set("type", "fixed")
        remove_children(
            joint,
            ("axis", "limit", "mimic", "dynamics", "safety_controller"),
        )

    robot.set("name", "ffw_sg2_single")
    if active_joint_order(robot) != SINGLE_ACTIVE_JOINTS:
        raise ValueError("single-arm URDF does not have the canonical 8-DoF order")
    ET.indent(tree, space="  ")
    tree.write(PLANNING_OUTPUT, encoding="utf-8", xml_declaration=True)


def generate_spherized_urdf(
    output: Path,
    sphere_map: dict[str, list[dict]],
) -> None:
    tree = ET.parse(PLANNING_OUTPUT)
    robot = tree.getroot()
    if active_joint_order(robot) != SINGLE_ACTIVE_JOINTS:
        raise ValueError("single-arm planning URDF active order changed")

    links = {link.get("name", ""): link for link in robot.findall("link")}
    if set(sphere_map) - links.keys():
        raise ValueError("sphere data contains links absent from single-arm URDF")

    sphere_index = 0
    for link in robot.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
        for sphere in sphere_map.get(link.get("name", ""), []):
            append_sphere_collision(link, sphere, sphere_index)
            sphere_index += 1

    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)


def project_configuration(configuration: list[float]) -> list[float]:
    if len(configuration) != len(DUAL_ACTIVE_JOINTS):
        raise ValueError("source problem configuration is not 15-DoF")
    projected = [float(configuration[index]) for index in SINGLE_SOURCE_INDICES]
    for index, (value, lower, upper) in enumerate(
        zip(projected, LOWER_LIMITS, UPPER_LIMITS)
    ):
        if value < lower or value > upper:
            raise ValueError(f"projected joint {index}={value} outside [{lower}, {upper}]")
    return projected


def generate_problem_file() -> None:
    source = json.loads(SOURCE_PROBLEMS.read_text(encoding="utf-8"))
    if tuple(source.get("joints", ())) != DUAL_ACTIVE_JOINTS:
        raise ValueError("source problem joint order is not canonical")

    result = copy.deepcopy(source)
    result["robot"] = "ffw_sg2_single"
    result["joints"] = list(SINGLE_ACTIVE_JOINTS)
    for problem_set in result["problems"].values():
        for problem in problem_set:
            problem["start"] = project_configuration(problem["start"])
            problem["goals"] = [
                project_configuration(goal) for goal in problem["goals"]
            ]
    PROBLEM_OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def generate_metadata(minimum_margin: float) -> None:
    fine_position_floats = FINE_SPHERE_COUNT * BATCH_SIZE * 3
    approximate_position_floats = APPROX_SPHERE_COUNT * BATCH_SIZE * 3
    joint_flag_ints = EXPECTED_JOINT_FLAG_STRIDE * BATCH_SIZE
    transform_floats = BATCH_SIZE * EXPECTED_TRANSFORM_SLOTS * 16
    model_shared_bytes = 4 * (
        fine_position_floats
        + approximate_position_floats
        + joint_flag_ints
        + transform_floats
    )
    metadata = {
        "schema_version": 1,
        "robot": "ffw_sg2_single",
        "active_dof": len(SINGLE_ACTIVE_JOINTS),
        "active_joint_order": list(SINGLE_ACTIVE_JOINTS),
        "fixed_left_arm_joints": list(LEFT_ARM_JOINTS),
        "fixed_left_arm_configuration": [0.0] * len(LEFT_ARM_JOINTS),
        "batch_size": BATCH_SIZE,
        "fine_model": {
            "urdf": FINE_OUTPUT.name,
            "urdf_sha256": sha256_file(FINE_OUTPUT),
            "sphere_count": FINE_SPHERE_COUNT,
        },
        "approximate_model": {
            "urdf": APPROX_OUTPUT.name,
            "urdf_sha256": sha256_file(APPROX_OUTPUT),
            "sphere_count": APPROX_SPHERE_COUNT,
            "minimum_fine_sphere_containment_margin_m": minimum_margin,
        },
        "generated_cricket_layout": {
            "joint_count_including_universe": EXPECTED_JOINT_FLAG_STRIDE,
            "joint_flag_stride": EXPECTED_JOINT_FLAG_STRIDE,
            "transform_slots": EXPECTED_TRANSFORM_SLOTS,
        },
        "memory_bytes": {
            "fine_runtime_positions": fine_position_floats * 4,
            "approximate_runtime_positions": approximate_position_floats * 4,
            "joint_flags": joint_flag_ints * 4,
            "transform_slots": transform_floats * 4,
            "model_dependent_shared_total": model_shared_bytes,
        },
        "sources": {
            "planning_urdf": PLANNING_OUTPUT.name,
            "planning_urdf_sha256": sha256_file(PLANNING_OUTPUT),
            "sphere_data": SOURCE_SPHERES.name,
            "sphere_data_sha256": sha256_file(SOURCE_SPHERES),
            "problem_file": SOURCE_PROBLEMS.name,
            "problem_file_sha256": sha256_file(SOURCE_PROBLEMS),
        },
    }
    METADATA_OUTPUT.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    generate_planning_urdf()
    _, fine_spheres = load_and_validate_spheres(SOURCE_SPHERES)
    approximate_spheres = build_approximation(fine_spheres)
    minimum_margin = validate_containment(fine_spheres, approximate_spheres)
    generate_spherized_urdf(FINE_OUTPUT, fine_spheres)
    generate_spherized_urdf(APPROX_OUTPUT, approximate_spheres)
    generate_problem_file()
    generate_metadata(minimum_margin)


if __name__ == "__main__":
    main()

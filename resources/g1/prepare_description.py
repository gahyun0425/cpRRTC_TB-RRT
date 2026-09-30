#!/usr/bin/env python3
"""Generate and verify PATACON's adapter for the official Unitree G1 model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]
DESCRIPTION_DIR = REPOSITORY_DIR / "unitree_ros"
SOURCE_DIR = DESCRIPTION_DIR / "robots" / "g1_description"
SOURCE_URDF = SOURCE_DIR / "g1_29dof.urdf"
SOURCE_MJCF = SOURCE_DIR / "g1_29dof.xml"
SCENE_PATH = RESOURCE_DIR / "g1_humanoid_shelf.xml"
METADATA_PATH = RESOURCE_DIR / "model_metadata.json"

SOURCE_REPOSITORY = "https://github.com/unitreerobotics/unitree_ros.git"
SOURCE_COMMIT = "da52948f035165aae2709d30255f5cd3e62875a0"
SOURCE_MODEL = "g1_29dof"

PLANNING_JOINTS = (
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

JOINT_LIMITS = (
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    (-2.5307, 2.8798),
    (-2.9671, 0.5236),
    (-2.7576, 2.7576),
    (-0.087267, 2.8798),
    (-0.87267, 0.5236),
    (-0.2618, 0.2618),
    (-2.618, 2.618),
    (-0.52, 0.52),
    (-0.52, 0.52),
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.972222054, 1.972222054),
    (-1.614429558, 1.614429558),
    (-1.614429558, 1.614429558),
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.972222054, 1.972222054),
    (-1.614429558, 1.614429558),
    (-1.614429558, 1.614429558),
)

GENERATED_FILES = (
    "resources/g1/g1_humanoid_shelf.xml",
    "src/robots/g1_collision.cuh",
    "src/robots/g1_kinematics.cuh",
    "src/constraints/backends/g1_constraint.cuh",
)

LOCAL_DEPENDENCY_FILES = (
    "resources/g1/g1_humanoid_shelf.xml",
    "scripts/visualize_g1.py",
    "scripts/visualize_g1_replanning.py",
    "src/robots/g1_collision.cuh",
    "src/robots/g1_kinematics.cuh",
    "src/constraints/backends/g1_constraint.cuh",
)

FORBIDDEN_EXTERNAL_REFERENCES = (
    "VAMP_ROOT",
    "gh_ws/vamp",
    "third_party/unitree_ros",
    "cptbrrt_pkg",
)

SCENE_CONTENT = b'''<mujoco model="g1_humanoid_shelf">
  <!--
    Official Unitree G1 model pinned by PATACON's repository-local
    unitree_ros submodule. Paths are resolved relative to this file.
  -->
  <include file="../../unitree_ros/robots/g1_description/g1_29dof.xml"/>

  <compiler meshdir="../../unitree_ros/robots/g1_description/meshes"/>

  <worldbody>
    <geom name="shelf_0" type="box"
          pos="0.45 0 0.4" size="0.1925 0.5 0.0075"
          contype="0" conaffinity="1" density="0"
          rgba="0.45 0.32 0.18 0.65"/>
    <geom name="shelf_1" type="box"
          pos="0.45 0 0.78" size="0.1925 0.5 0.0075"
          contype="0" conaffinity="1" density="0"
          rgba="0.45 0.32 0.18 0.65"/>
    <geom name="shelf_2" type="box"
          pos="0.45 0 1.2" size="0.1925 0.5 0.0075"
          contype="0" conaffinity="1" density="0"
          rgba="0.45 0.32 0.18 0.65"/>
    <geom name="shelf_back" type="box"
          pos="0.6 0 1.0" size="0.015 0.5 1.0"
          contype="0" conaffinity="1" density="0"
          rgba="0.45 0.32 0.18 0.65"/>
    <geom name="obstacle" type="box"
          pos="0 0.4 0.225" size="0.2 0.15 0.225"
          contype="0" conaffinity="1" density="0"
          rgba="0.85 0.20 0.15 0.65"/>
  </worldbody>
</mujoco>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed outputs without modifying them",
    )
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def description_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(DESCRIPTION_DIR), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "unitree_ros is not initialized; run "
            "git submodule update --init unitree_ros"
        ) from error
    return result.stdout.strip()


def validate_source() -> list[str]:
    if not SOURCE_URDF.is_file() or not SOURCE_MJCF.is_file():
        raise RuntimeError(
            "official G1 description is missing; run "
            "git submodule update --init unitree_ros"
        )
    commit = description_commit()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"unitree_ros is at {commit}, expected {SOURCE_COMMIT}"
        )

    urdf = ET.parse(SOURCE_URDF).getroot()
    revolute = [
        joint for joint in urdf.findall("joint")
        if joint.get("type") == "revolute"
    ]
    actual_names = tuple(joint.get("name", "") for joint in revolute)
    if actual_names != PLANNING_JOINTS:
        raise RuntimeError("official G1 URDF joint order changed")
    for joint, expected in zip(revolute, JOINT_LIMITS):
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"{joint.get('name')} has no limit")
        actual = (float(limit.get("lower", "nan")), float(limit.get("upper", "nan")))
        if any(abs(left - right) > 1.0e-9 for left, right in zip(actual, expected)):
            raise RuntimeError(
                f"{joint.get('name')} limits {actual} do not match {expected}"
            )

    mjcf = ET.parse(SOURCE_MJCF).getroot()
    named_joints = tuple(
        joint.get("name", "") for joint in mjcf.findall(".//joint")
        if joint.get("name")
    )
    if named_joints != ("floating_base_joint",) + PLANNING_JOINTS:
        raise RuntimeError("official G1 MJCF joint order changed")

    mesh_files = sorted(
        {
            f"robots/g1_description/meshes/{mesh.get('file')}"
            for mesh in mjcf.findall(".//mesh")
            if mesh.get("file")
        }
    )
    for relative_path in mesh_files:
        if not (DESCRIPTION_DIR / relative_path).is_file():
            raise RuntimeError(
                f"official G1 description is missing {relative_path}"
            )
    return mesh_files


def validate_local_dependencies() -> None:
    for relative_path in LOCAL_DEPENDENCY_FILES:
        path = REPOSITORY_DIR / relative_path
        content = path.read_text(encoding="utf-8")
        for reference in FORBIDDEN_EXTERNAL_REFERENCES:
            if reference in content:
                raise RuntimeError(
                    f"{relative_path} still references external {reference}"
                )


def metadata_content(mesh_files: list[str]) -> bytes:
    source_files = [
        "LICENSE",
        "robots/g1_description/README.md",
        "robots/g1_description/g1_29dof.urdf",
        "robots/g1_description/g1_29dof.xml",
        *mesh_files,
    ]
    metadata = {
        "generated": {
            relative_path: sha256_file(REPOSITORY_DIR / relative_path)
            for relative_path in GENERATED_FILES
        },
        "patacon_contract": {
            "base_coordinates": ["x", "y", "z", "roll", "pitch", "yaw"],
            "base_spheres": 133,
            "end_effectors": [
                "left_rubber_hand",
                "right_rubber_hand",
                "left_foot",
                "right_foot",
            ],
            "hand_spheres_per_hand": 16,
            "joint_limits_rad": [list(limit) for limit in JOINT_LIMITS],
            "joint_order": list(PLANNING_JOINTS),
            "planning_dof": 35,
            "self_collision_pairs": 6888,
            "total_spheres": 165,
        },
        "source": {
            "commit": SOURCE_COMMIT,
            "files_sha256": {
                relative_path: sha256_file(DESCRIPTION_DIR / relative_path)
                for relative_path in source_files
            },
            "model": SOURCE_MODEL,
            "repository": SOURCE_REPOSITORY,
        },
    }
    return (
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def check_equal(path: Path, expected: bytes) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing generated file: {path}")
    if path.read_bytes() != expected:
        raise RuntimeError(
            f"{path.relative_to(REPOSITORY_DIR)} is stale; run "
            "python3 resources/g1/prepare_description.py"
        )


def main() -> None:
    args = parse_args()
    mesh_files = validate_source()
    if args.check:
        check_equal(SCENE_PATH, SCENE_CONTENT)
    else:
        SCENE_PATH.write_bytes(SCENE_CONTENT)

    validate_local_dependencies()
    metadata = metadata_content(mesh_files)
    if args.check:
        check_equal(METADATA_PATH, metadata)
        print(
            "verified official Unitree G1 description adapter at "
            f"{SOURCE_COMMIT}"
        )
    else:
        METADATA_PATH.write_bytes(metadata)
        print(f"generated {SCENE_PATH.relative_to(REPOSITORY_DIR)}")
        print(f"generated {METADATA_PATH.relative_to(REPOSITORY_DIR)}")


if __name__ == "__main__":
    main()

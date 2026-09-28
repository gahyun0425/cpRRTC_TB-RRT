#!/usr/bin/env python3
"""Generate independent Pinocchio FK/Jacobian references for IGRIS-C."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin


RESOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = RESOURCE_DIR / "igris_c_planning.urdf"
DEFAULT_MODEL_METADATA = RESOURCE_DIR / "model_metadata.json"
DEFAULT_CONSTRAINT_CONTRACT = RESOURCE_DIR / "constraint_contract.json"
DEFAULT_OUTPUT = RESOURCE_DIR / "kinematics_reference.json"
TASK_FRAMES = ("l_sole", "r_sole", "l_grasp", "r_grasp")
FINITE_DIFFERENCE_STEP = 1.0e-7

NONTRIVIAL_CONFIGURATION = {
    "world_to_x": 0.13,
    "x_to_y": -0.08,
    "y_to_z": 0.94,
    "z_to_roll": 0.08,
    "roll_to_pitch": -0.06,
    "pitch_to_yaw": 0.12,
    "l_hip_pitch": -0.35,
    "l_hip_roll": 0.08,
    "l_hip_yaw": 0.05,
    "l_knee_pitch": 0.65,
    "l_ankle_pitch": -0.30,
    "l_ankle_roll": -0.04,
    "r_hip_pitch": -0.32,
    "r_hip_roll": -0.07,
    "r_hip_yaw": -0.06,
    "r_knee_pitch": 0.61,
    "r_ankle_pitch": -0.28,
    "r_ankle_roll": 0.05,
    "waist_pitch": -0.12,
    "waist_roll": 0.08,
    "waist_yaw": 0.18,
    "l_shoulder_pitch": -0.45,
    "l_shoulder_roll": 0.22,
    "l_shoulder_yaw": -0.19,
    "l_elbow_pitch": -0.76,
    "l_wrist_yaw": 0.17,
    "l_wrist_roll": -0.24,
    "l_wrist_pitch": 0.13,
    "r_shoulder_pitch": -0.51,
    "r_shoulder_roll": -0.25,
    "r_shoulder_yaw": 0.16,
    "r_elbow_pitch": -0.71,
    "r_wrist_yaw": -0.15,
    "r_wrist_roll": 0.27,
    "r_wrist_pitch": -0.11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--model-metadata", type=Path, default=DEFAULT_MODEL_METADATA)
    parser.add_argument(
        "--constraint-contract", type=Path, default=DEFAULT_CONSTRAINT_CONTRACT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_placements(
    model: pin.Model, data: pin.Data, q: np.ndarray
) -> dict[str, pin.SE3]:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return {name: data.oMf[model.getFrameId(name)].copy() for name in TASK_FRAMES}


def center_of_mass(model: pin.Model, data: pin.Data, q: np.ndarray) -> np.ndarray:
    return np.asarray(pin.centerOfMass(model, data, q)).copy()


def relative_grasp_pose(placements: dict[str, pin.SE3]) -> pin.SE3:
    return placements["l_grasp"].inverse() * placements["r_grasp"]


def matrix_values(matrix: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(matrix).reshape(-1)]


def vector_values(vector: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(vector).reshape(-1)]


def reference_for_configuration(
    model: pin.Model,
    q: np.ndarray,
    name: str,
) -> dict[str, object]:
    data = model.createData()
    placements = frame_placements(model, data, q)
    com = center_of_mass(model, data, q)
    relative = relative_grasp_pose(placements)

    frame_position_jacobians = {
        frame: np.zeros((3, model.nq)) for frame in TASK_FRAMES
    }
    frame_angular_jacobians = {
        frame: np.zeros((3, model.nq)) for frame in TASK_FRAMES
    }
    com_jacobian = np.zeros((3, model.nq))
    relative_position_jacobian = np.zeros((3, model.nq))
    relative_angular_jacobian = np.zeros((3, model.nq))

    for joint in range(model.nq):
        plus = q.copy()
        minus = q.copy()
        plus[joint] += FINITE_DIFFERENCE_STEP
        minus[joint] -= FINITE_DIFFERENCE_STEP
        plus_data = model.createData()
        minus_data = model.createData()
        plus_placements = frame_placements(model, plus_data, plus)
        minus_placements = frame_placements(model, minus_data, minus)
        denominator = 2.0 * FINITE_DIFFERENCE_STEP

        for frame in TASK_FRAMES:
            frame_position_jacobians[frame][:, joint] = (
                plus_placements[frame].translation -
                minus_placements[frame].translation
            ) / denominator
            rotation_delta = (
                plus_placements[frame].rotation @
                minus_placements[frame].rotation.T
            )
            frame_angular_jacobians[frame][:, joint] = (
                np.asarray(pin.log3(rotation_delta)) / denominator
            )

        plus_com = center_of_mass(model, plus_data, plus)
        minus_com = center_of_mass(model, minus_data, minus)
        com_jacobian[:, joint] = (plus_com - minus_com) / denominator

        plus_relative = relative_grasp_pose(plus_placements)
        minus_relative = relative_grasp_pose(minus_placements)
        relative_position_jacobian[:, joint] = (
            plus_relative.translation - minus_relative.translation
        ) / denominator
        relative_rotation_delta = (
            plus_relative.rotation @ minus_relative.rotation.T
        )
        relative_angular_jacobian[:, joint] = (
            np.asarray(pin.log3(relative_rotation_delta)) / denominator
        )

    return {
        "name": name,
        "configuration": vector_values(q),
        "frames": {
            frame: {
                "translation": vector_values(placements[frame].translation),
                "rotation_row_major": matrix_values(placements[frame].rotation),
                "position_jacobian_row_major": matrix_values(
                    frame_position_jacobians[frame]
                ),
                "world_angular_jacobian_row_major": matrix_values(
                    frame_angular_jacobians[frame]
                ),
            }
            for frame in TASK_FRAMES
        },
        "center_of_mass": {
            "translation": vector_values(com),
            "jacobian_row_major": matrix_values(com_jacobian),
        },
        "bimanual_relative_pose": {
            "convention": "inverse(T_world_l_grasp) * T_world_r_grasp",
            "translation": vector_values(relative.translation),
            "rotation_row_major": matrix_values(relative.rotation),
            "position_jacobian_row_major": matrix_values(
                relative_position_jacobian
            ),
            "left_frame_angular_jacobian_row_major": matrix_values(
                relative_angular_jacobian
            ),
        },
    }


def generate(
    urdf: Path,
    model_metadata_path: Path,
    constraint_contract_path: Path,
) -> bytes:
    metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    contract = json.loads(constraint_contract_path.read_text(encoding="utf-8"))
    urdf_hash = sha256_file(urdf)
    if metadata["generated_urdf_sha256"] != urdf_hash:
        raise ValueError("model metadata does not match the planning URDF")
    if contract["planning_model_sha256"] != urdf_hash:
        raise ValueError("constraint contract does not match the planning URDF")

    joint_names = [item["name"] for item in metadata["configuration"]]
    model = pin.buildModelFromUrdf(str(urdf))
    pinocchio_joint_names = list(model.names)[1:]
    if model.nq != 35 or model.nv != 35 or pinocchio_joint_names != joint_names:
        raise ValueError(
            "Pinocchio configuration order differs from the 35-DoF contract"
        )
    for frame in TASK_FRAMES:
        if not model.existFrame(frame):
            raise ValueError(f"Pinocchio model is missing frame {frame!r}")

    seed = np.asarray(
        contract["constraint_consistent_seed"]["configuration"], dtype=float
    )
    nontrivial = np.asarray(
        [NONTRIVIAL_CONFIGURATION[name] for name in joint_names], dtype=float
    )
    for name, q in (("constraint_consistent_seed", seed), ("nontrivial", nontrivial)):
        if q.shape != (35,):
            raise ValueError(f"{name} has invalid shape {q.shape}")
        if np.any(q < model.lowerPositionLimit) or np.any(q > model.upperPositionLimit):
            raise ValueError(f"{name} violates a joint limit")

    result = {
        "schema_version": 1,
        "reference_implementation": {
            "name": "Pinocchio",
            "version": pin.__version__,
            "jacobian_method": "central finite difference of Pinocchio FK",
            "finite_difference_step": FINITE_DIFFERENCE_STEP,
        },
        "planning_urdf": urdf.name,
        "planning_urdf_sha256": urdf_hash,
        "configuration_order": joint_names,
        "task_frame_order": list(TASK_FRAMES),
        "configurations": [
            reference_for_configuration(model, seed, "constraint_consistent_seed"),
            reference_for_configuration(model, nontrivial, "nontrivial"),
        ],
    }
    return json.dumps(result, indent=2).encode("utf-8") + b"\n"


def main() -> None:
    args = parse_args()
    urdf = args.urdf.resolve()
    model_metadata = args.model_metadata.resolve()
    constraint_contract = args.constraint_contract.resolve()
    output = args.output.resolve()
    try:
        content = generate(urdf, model_metadata, constraint_contract)
        if args.check:
            if not output.is_file() or output.read_bytes() != content:
                raise ValueError(f"generated artifact is missing or stale: {output}")
            print("PASS: Pinocchio FK/Jacobian reference is current")
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        print(f"wrote {output}")
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate deterministic CUDA FK/Jacobian model data for IGRIS-C."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]
DEFAULT_URDF = RESOURCE_DIR / "igris_c_planning.urdf"
DEFAULT_MODEL_METADATA = RESOURCE_DIR / "model_metadata.json"
DEFAULT_CONSTRAINT_CONTRACT = RESOURCE_DIR / "constraint_contract.json"
DEFAULT_OUTPUT = REPOSITORY_DIR / "src" / "robots" / "igris_c_kinematics.cuh"
DEFAULT_METADATA_OUTPUT = RESOURCE_DIR / "kinematics_metadata.json"

MOVABLE_TYPES = {"continuous", "prismatic", "revolute"}
JOINT_TYPE = {"fixed": 0, "prismatic": 1, "revolute": 2, "continuous": 2}
TASK_FRAMES = ("l_sole", "r_sole", "l_grasp", "r_grasp")


HEADER_TEMPLATE = r'''#pragma once

// Generated from @URDF_PATH@ by
// resources/igris_c/generate_kinematics_header.py.
// Planning-model SHA-256: @URDF_SHA256@

#include <cuda_runtime.h>

namespace ppln::collision {

constexpr int IGRIS_C_DIM = @DIM@;
constexpr int IGRIS_C_LINK_COUNT = @LINK_COUNT@;
constexpr int IGRIS_C_TREE_JOINT_COUNT = @TREE_JOINT_COUNT@;
constexpr int IGRIS_C_TASK_FRAME_COUNT = 4;
constexpr int IGRIS_C_L_SOLE_FRAME = 0;
constexpr int IGRIS_C_R_SOLE_FRAME = 1;
constexpr int IGRIS_C_L_GRASP_FRAME = 2;
constexpr int IGRIS_C_R_GRASP_FRAME = 3;
constexpr float IGRIS_C_TOTAL_MASS_KG = @TOTAL_MASS@;

struct IgrisCTransform {
    float rotation[9];
    float translation[3];
};

struct IgrisCJointSpec {
    int parent_link;
    int child_link;
    int configuration_index;
    int type;  // 0 fixed, 1 prismatic, 2 revolute
    float origin_translation[3];
    float origin_rotation[9];
    float axis[3];
};

__device__ __constant__ IgrisCJointSpec
igris_c_tree_joints[IGRIS_C_TREE_JOINT_COUNT] = {
@JOINT_SPECS@
};

__device__ __constant__ int
igris_c_configuration_types[IGRIS_C_DIM] = {
@CONFIGURATION_TYPES@
};

__device__ __constant__ unsigned long long
igris_c_link_ancestor_masks[IGRIS_C_LINK_COUNT] = {
@ANCESTOR_MASKS@
};

__device__ __constant__ float
igris_c_link_masses[IGRIS_C_LINK_COUNT] = {
@LINK_MASSES@
};

__device__ __constant__ float
igris_c_link_com_local[IGRIS_C_LINK_COUNT][3] = {
@LINK_COM_LOCAL@
};

__device__ __constant__ int
igris_c_task_link_indices[IGRIS_C_TASK_FRAME_COUNT] = {
@TASK_LINK_INDICES@
};

__device__ __forceinline__ void igris_c_identity3(float matrix[9]) {
    for (int index = 0; index < 9; ++index) {
        matrix[index] = index % 4 == 0 ? 1.0f : 0.0f;
    }
}

__device__ __forceinline__ void igris_c_multiply3(
    const float left[9],
    const float right[9],
    float result[9]
) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result[row * 3 + column] =
                left[row * 3] * right[column] +
                left[row * 3 + 1] * right[3 + column] +
                left[row * 3 + 2] * right[6 + column];
        }
    }
}

__device__ __forceinline__ void igris_c_multiply_at_b(
    const float left[9],
    const float right[9],
    float result[9]
) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result[row * 3 + column] =
                left[row] * right[column] +
                left[3 + row] * right[3 + column] +
                left[6 + row] * right[6 + column];
        }
    }
}

__device__ __forceinline__ void igris_c_rotate(
    const float rotation[9],
    const float vector[3],
    float result[3]
) {
    for (int row = 0; row < 3; ++row) {
        result[row] =
            rotation[row * 3] * vector[0] +
            rotation[row * 3 + 1] * vector[1] +
            rotation[row * 3 + 2] * vector[2];
    }
}

__device__ __forceinline__ void igris_c_rotate_transpose(
    const float rotation[9],
    const float vector[3],
    float result[3]
) {
    for (int column = 0; column < 3; ++column) {
        result[column] =
            rotation[column] * vector[0] +
            rotation[3 + column] * vector[1] +
            rotation[6 + column] * vector[2];
    }
}

__device__ __forceinline__ void igris_c_axis_angle_rotation(
    const float axis[3],
    float angle,
    float rotation[9]
) {
    const float x = axis[0];
    const float y = axis[1];
    const float z = axis[2];
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float one_minus_cosine = 1.0f - cosine;
    rotation[0] = cosine + x * x * one_minus_cosine;
    rotation[1] = x * y * one_minus_cosine - z * sine;
    rotation[2] = x * z * one_minus_cosine + y * sine;
    rotation[3] = y * x * one_minus_cosine + z * sine;
    rotation[4] = cosine + y * y * one_minus_cosine;
    rotation[5] = y * z * one_minus_cosine - x * sine;
    rotation[6] = z * x * one_minus_cosine - y * sine;
    rotation[7] = z * y * one_minus_cosine + x * sine;
    rotation[8] = cosine + z * z * one_minus_cosine;
}

__device__ __forceinline__ void igris_c_cross(
    const float left[3],
    const float right[3],
    float result[3]
) {
    result[0] = left[1] * right[2] - left[2] * right[1];
    result[1] = left[2] * right[0] - left[0] * right[2];
    result[2] = left[0] * right[1] - left[1] * right[0];
}

// Computes every link pose and each active joint's world origin/axis once.
// The generated joint table is topologically ordered from the world link.
__device__ __noinline__ void igris_c_forward_model(
    const float q[IGRIS_C_DIM],
    IgrisCTransform link_poses[IGRIS_C_LINK_COUNT],
    float joint_origins[IGRIS_C_DIM][3],
    float joint_axes[IGRIS_C_DIM][3]
) {
    igris_c_identity3(link_poses[0].rotation);
    for (int component = 0; component < 3; ++component) {
        link_poses[0].translation[component] = 0.0f;
    }

    for (int joint_index = 0;
         joint_index < IGRIS_C_TREE_JOINT_COUNT;
         ++joint_index) {
        const IgrisCJointSpec &joint = igris_c_tree_joints[joint_index];
        const IgrisCTransform &parent = link_poses[joint.parent_link];
        IgrisCTransform &child = link_poses[joint.child_link];

        float translated_origin[3];
        igris_c_rotate(
            parent.rotation,
            joint.origin_translation,
            translated_origin
        );
        for (int component = 0; component < 3; ++component) {
            child.translation[component] =
                parent.translation[component] + translated_origin[component];
        }

        float origin_rotation[9];
        igris_c_multiply3(
            parent.rotation,
            joint.origin_rotation,
            origin_rotation
        );

        float axis_world[3];
        igris_c_rotate(origin_rotation, joint.axis, axis_world);
        if (joint.configuration_index >= 0) {
            for (int component = 0; component < 3; ++component) {
                joint_origins[joint.configuration_index][component] =
                    child.translation[component];
                joint_axes[joint.configuration_index][component] =
                    axis_world[component];
            }
        }

        if (joint.type == 1) {
            const float displacement = q[joint.configuration_index];
            for (int component = 0; component < 3; ++component) {
                child.translation[component] +=
                    axis_world[component] * displacement;
            }
            for (int index = 0; index < 9; ++index) {
                child.rotation[index] = origin_rotation[index];
            }
        } else if (joint.type == 2) {
            float joint_rotation[9];
            igris_c_axis_angle_rotation(
                joint.axis,
                q[joint.configuration_index],
                joint_rotation
            );
            igris_c_multiply3(
                origin_rotation,
                joint_rotation,
                child.rotation
            );
        } else {
            for (int index = 0; index < 9; ++index) {
                child.rotation[index] = origin_rotation[index];
            }
        }
    }
}

__device__ __forceinline__ void igris_c_frame_kinematics_from_model(
    int link_index,
    const IgrisCTransform link_poses[IGRIS_C_LINK_COUNT],
    const float joint_origins[IGRIS_C_DIM][3],
    const float joint_axes[IGRIS_C_DIM][3],
    IgrisCTransform &pose,
    float position_jacobian[3 * IGRIS_C_DIM],
    float angular_jacobian[3 * IGRIS_C_DIM]
) {
    for (int index = 0; index < 9; ++index) {
        pose.rotation[index] = link_poses[link_index].rotation[index];
    }
    for (int component = 0; component < 3; ++component) {
        pose.translation[component] = link_poses[link_index].translation[component];
    }
    for (int index = 0; index < 3 * IGRIS_C_DIM; ++index) {
        position_jacobian[index] = 0.0f;
        angular_jacobian[index] = 0.0f;
    }

    const unsigned long long ancestors =
        igris_c_link_ancestor_masks[link_index];
    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
        if ((ancestors & (1ULL << joint)) == 0ULL) {
            continue;
        }
        if (igris_c_configuration_types[joint] == 1) {
            for (int component = 0; component < 3; ++component) {
                position_jacobian[component * IGRIS_C_DIM + joint] =
                    joint_axes[joint][component];
            }
        } else {
            float displacement[3];
            for (int component = 0; component < 3; ++component) {
                displacement[component] = pose.translation[component] -
                    joint_origins[joint][component];
            }
            float linear[3];
            igris_c_cross(joint_axes[joint], displacement, linear);
            for (int component = 0; component < 3; ++component) {
                position_jacobian[component * IGRIS_C_DIM + joint] =
                    linear[component];
                angular_jacobian[component * IGRIS_C_DIM + joint] =
                    joint_axes[joint][component];
            }
        }
    }
}

__device__ __noinline__ void igris_c_task_frame_kinematics(
    const float q[IGRIS_C_DIM],
    int task_frame,
    IgrisCTransform &pose,
    float position_jacobian[3 * IGRIS_C_DIM],
    float angular_jacobian[3 * IGRIS_C_DIM]
) {
    IgrisCTransform link_poses[IGRIS_C_LINK_COUNT];
    float joint_origins[IGRIS_C_DIM][3];
    float joint_axes[IGRIS_C_DIM][3];
    igris_c_forward_model(q, link_poses, joint_origins, joint_axes);
    igris_c_frame_kinematics_from_model(
        igris_c_task_link_indices[task_frame],
        link_poses,
        joint_origins,
        joint_axes,
        pose,
        position_jacobian,
        angular_jacobian
    );
}

__device__ __forceinline__ void igris_c_center_of_mass_from_model(
    const IgrisCTransform link_poses[IGRIS_C_LINK_COUNT],
    const float joint_origins[IGRIS_C_DIM][3],
    const float joint_axes[IGRIS_C_DIM][3],
    float center_of_mass[3],
    float jacobian[3 * IGRIS_C_DIM]
) {
    for (int component = 0; component < 3; ++component) {
        center_of_mass[component] = 0.0f;
    }
    for (int index = 0; index < 3 * IGRIS_C_DIM; ++index) {
        jacobian[index] = 0.0f;
    }

    for (int link = 0; link < IGRIS_C_LINK_COUNT; ++link) {
        const float mass = igris_c_link_masses[link];
        if (mass <= 0.0f) {
            continue;
        }
        float rotated_local_com[3];
        igris_c_rotate(
            link_poses[link].rotation,
            igris_c_link_com_local[link],
            rotated_local_com
        );
        float world_com[3];
        for (int component = 0; component < 3; ++component) {
            world_com[component] =
                link_poses[link].translation[component] +
                rotated_local_com[component];
            center_of_mass[component] += mass * world_com[component];
        }

        const unsigned long long ancestors =
            igris_c_link_ancestor_masks[link];
        for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
            if ((ancestors & (1ULL << joint)) == 0ULL) {
                continue;
            }
            float linear[3];
            if (igris_c_configuration_types[joint] == 1) {
                for (int component = 0; component < 3; ++component) {
                    linear[component] = joint_axes[joint][component];
                }
            } else {
                float displacement[3];
                for (int component = 0; component < 3; ++component) {
                    displacement[component] = world_com[component] -
                        joint_origins[joint][component];
                }
                igris_c_cross(joint_axes[joint], displacement, linear);
            }
            for (int component = 0; component < 3; ++component) {
                jacobian[component * IGRIS_C_DIM + joint] +=
                    mass * linear[component];
            }
        }
    }

    const float inverse_mass = 1.0f / IGRIS_C_TOTAL_MASS_KG;
    for (int component = 0; component < 3; ++component) {
        center_of_mass[component] *= inverse_mass;
    }
    for (int index = 0; index < 3 * IGRIS_C_DIM; ++index) {
        jacobian[index] *= inverse_mass;
    }
}

__device__ __noinline__ void igris_c_center_of_mass_kinematics(
    const float q[IGRIS_C_DIM],
    float center_of_mass[3],
    float jacobian[3 * IGRIS_C_DIM]
) {
    IgrisCTransform link_poses[IGRIS_C_LINK_COUNT];
    float joint_origins[IGRIS_C_DIM][3];
    float joint_axes[IGRIS_C_DIM][3];
    igris_c_forward_model(q, link_poses, joint_origins, joint_axes);
    igris_c_center_of_mass_from_model(
        link_poses,
        joint_origins,
        joint_axes,
        center_of_mass,
        jacobian
    );
}

// Returns T_l_grasp^-1 * T_r_grasp. The angular Jacobian is the relative
// spatial angular velocity expressed in l_grasp coordinates.
__device__ __forceinline__ void igris_c_bimanual_from_model(
    const IgrisCTransform link_poses[IGRIS_C_LINK_COUNT],
    const float joint_origins[IGRIS_C_DIM][3],
    const float joint_axes[IGRIS_C_DIM][3],
    IgrisCTransform &relative_pose,
    float position_jacobian[3 * IGRIS_C_DIM],
    float angular_jacobian[3 * IGRIS_C_DIM]
) {
    IgrisCTransform left_pose;
    IgrisCTransform right_pose;
    float left_position_jacobian[3 * IGRIS_C_DIM];
    float right_position_jacobian[3 * IGRIS_C_DIM];
    float left_angular_jacobian[3 * IGRIS_C_DIM];
    float right_angular_jacobian[3 * IGRIS_C_DIM];
    igris_c_frame_kinematics_from_model(
        igris_c_task_link_indices[IGRIS_C_L_GRASP_FRAME],
        link_poses,
        joint_origins,
        joint_axes,
        left_pose,
        left_position_jacobian,
        left_angular_jacobian
    );
    igris_c_frame_kinematics_from_model(
        igris_c_task_link_indices[IGRIS_C_R_GRASP_FRAME],
        link_poses,
        joint_origins,
        joint_axes,
        right_pose,
        right_position_jacobian,
        right_angular_jacobian
    );

    igris_c_multiply_at_b(
        left_pose.rotation,
        right_pose.rotation,
        relative_pose.rotation
    );
    float world_displacement[3];
    for (int component = 0; component < 3; ++component) {
        world_displacement[component] =
            right_pose.translation[component] -
            left_pose.translation[component];
    }
    igris_c_rotate_transpose(
        left_pose.rotation,
        world_displacement,
        relative_pose.translation
    );

    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
        float left_angular[3];
        float relative_world_linear[3];
        float relative_world_angular[3];
        for (int component = 0; component < 3; ++component) {
            left_angular[component] =
                left_angular_jacobian[component * IGRIS_C_DIM + joint];
            relative_world_linear[component] =
                right_position_jacobian[component * IGRIS_C_DIM + joint] -
                left_position_jacobian[component * IGRIS_C_DIM + joint];
            relative_world_angular[component] =
                right_angular_jacobian[component * IGRIS_C_DIM + joint] -
                left_angular_jacobian[component * IGRIS_C_DIM + joint];
        }
        float rotating_frame_term[3];
        igris_c_cross(left_angular, world_displacement, rotating_frame_term);
        for (int component = 0; component < 3; ++component) {
            relative_world_linear[component] -= rotating_frame_term[component];
        }
        float relative_local_linear[3];
        float relative_local_angular[3];
        igris_c_rotate_transpose(
            left_pose.rotation,
            relative_world_linear,
            relative_local_linear
        );
        igris_c_rotate_transpose(
            left_pose.rotation,
            relative_world_angular,
            relative_local_angular
        );
        for (int component = 0; component < 3; ++component) {
            position_jacobian[component * IGRIS_C_DIM + joint] =
                relative_local_linear[component];
            angular_jacobian[component * IGRIS_C_DIM + joint] =
                relative_local_angular[component];
        }
    }
}

__device__ __noinline__ void igris_c_bimanual_kinematics(
    const float q[IGRIS_C_DIM],
    IgrisCTransform &relative_pose,
    float position_jacobian[3 * IGRIS_C_DIM],
    float angular_jacobian[3 * IGRIS_C_DIM]
) {
    IgrisCTransform link_poses[IGRIS_C_LINK_COUNT];
    float joint_origins[IGRIS_C_DIM][3];
    float joint_axes[IGRIS_C_DIM][3];
    igris_c_forward_model(q, link_poses, joint_origins, joint_axes);
    igris_c_bimanual_from_model(
        link_poses,
        joint_origins,
        joint_axes,
        relative_pose,
        position_jacobian,
        angular_jacobian
    );
}

}  // namespace ppln::collision
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--model-metadata", type=Path, default=DEFAULT_MODEL_METADATA)
    parser.add_argument(
        "--constraint-contract", type=Path, default=DEFAULT_CONSTRAINT_CONTRACT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_vector(element: ET.Element | None, attribute: str) -> tuple[float, ...]:
    if element is None or not element.get(attribute):
        return (0.0, 0.0, 0.0)
    values = tuple(float(value) for value in element.get(attribute, "").split())
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid {attribute}: {values}")
    return values


def multiply3(
    left: tuple[float, ...], right: tuple[float, ...]
) -> tuple[float, ...]:
    return tuple(
        sum(left[row * 3 + inner] * right[inner * 3 + column] for inner in range(3))
        for row in range(3)
        for column in range(3)
    )


def axis_rotation(axis: int, angle: float) -> tuple[float, ...]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    if axis == 0:
        return (1.0, 0.0, 0.0, 0.0, cosine, -sine, 0.0, sine, cosine)
    if axis == 1:
        return (cosine, 0.0, sine, 0.0, 1.0, 0.0, -sine, 0.0, cosine)
    return (cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0)


def rpy_rotation(rpy: tuple[float, ...]) -> tuple[float, ...]:
    roll, pitch, yaw = rpy
    return multiply3(
        multiply3(axis_rotation(2, yaw), axis_rotation(1, pitch)),
        axis_rotation(0, roll),
    )


def normalize_axis(axis: tuple[float, ...], joint_name: str) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 0.0:
        raise ValueError(f"movable joint {joint_name!r} has a zero axis")
    return tuple(value / norm for value in axis)


def float_literal(value: float) -> str:
    if abs(value) < 5.0e-18:
        value = 0.0
    result = format(value, ".17g")
    if "." not in result and "e" not in result:
        result += ".0"
    return result + "f"


def row(values: tuple[float, ...] | list[float]) -> str:
    return ", ".join(float_literal(float(value)) for value in values)


def build_model(
    urdf: Path,
    model_metadata_path: Path,
    constraint_contract_path: Path,
) -> dict[str, object]:
    urdf_bytes = urdf.read_bytes()
    urdf_hash = sha256_bytes(urdf_bytes)
    metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    contract = json.loads(constraint_contract_path.read_text(encoding="utf-8"))
    if metadata["generated_urdf_sha256"] != urdf_hash:
        raise ValueError("model metadata does not match the planning URDF")
    if contract["planning_model_sha256"] != urdf_hash:
        raise ValueError("constraint contract does not match the planning URDF")

    robot = ET.fromstring(urdf_bytes)
    links = {link.get("name", ""): link for link in robot.findall("link")}
    joints = list(robot.findall("joint"))
    active_names = tuple(item["name"] for item in metadata["configuration"])
    movable_names = tuple(
        joint.get("name", "")
        for joint in joints
        if joint.get("type") in MOVABLE_TYPES
    )
    if movable_names != active_names or len(active_names) != 35:
        raise ValueError("planning URDF and metadata joint orders differ")
    configuration_index = {name: index for index, name in enumerate(active_names)}

    children: dict[str, list[ET.Element]] = {name: [] for name in links}
    child_links: set[str] = set()
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {joint.get('name')!r} has no parent/child")
        parent_name = parent.get("link", "")
        child_name = child.get("link", "")
        if parent_name not in links or child_name not in links:
            raise ValueError(f"joint {joint.get('name')!r} refers to an unknown link")
        children[parent_name].append(joint)
        if child_name in child_links:
            raise ValueError(f"link {child_name!r} has multiple parents")
        child_links.add(child_name)
    roots = set(links) - child_links
    if roots != {"world"}:
        raise ValueError(f"expected root world, got {sorted(roots)}")

    topological_links = ["world"]
    topological_joints: list[ET.Element] = []

    def visit(parent_name: str) -> None:
        for joint in children[parent_name]:
            child_element = joint.find("child")
            assert child_element is not None
            child_name = child_element.get("link", "")
            topological_joints.append(joint)
            topological_links.append(child_name)
            visit(child_name)

    visit("world")
    if len(topological_links) != len(links):
        raise ValueError("planning URDF graph is disconnected")
    link_index = {name: index for index, name in enumerate(topological_links)}

    masks = [0] * len(topological_links)
    specs: list[dict[str, object]] = []
    configuration_types = [-1] * len(active_names)
    for joint in topological_joints:
        name = joint.get("name", "")
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        assert parent_element is not None and child_element is not None
        parent_name = parent_element.get("link", "")
        child_name = child_element.get("link", "")
        joint_type_name = joint.get("type", "")
        if joint_type_name not in JOINT_TYPE:
            raise ValueError(f"unsupported joint type {joint_type_name!r}")
        q_index = configuration_index.get(name, -1)
        joint_type = JOINT_TYPE[joint_type_name]
        origin = joint.find("origin")
        translation = parse_vector(origin, "xyz")
        rotation = rpy_rotation(parse_vector(origin, "rpy"))
        axis = parse_vector(joint.find("axis"), "xyz")
        if q_index >= 0:
            axis = normalize_axis(axis, name)
            configuration_types[q_index] = joint_type
        else:
            axis = (0.0, 0.0, 0.0)
        parent_index = link_index[parent_name]
        child_index = link_index[child_name]
        masks[child_index] = masks[parent_index]
        if q_index >= 0:
            masks[child_index] |= 1 << q_index
        specs.append(
            {
                "name": name,
                "parent": parent_index,
                "child": child_index,
                "q": q_index,
                "type": joint_type,
                "translation": translation,
                "rotation": rotation,
                "axis": axis,
            }
        )
    if any(value not in (1, 2) for value in configuration_types):
        raise ValueError(f"invalid active joint types: {configuration_types}")

    masses: list[float] = []
    local_coms: list[tuple[float, ...]] = []
    for name in topological_links:
        inertial = links[name].find("inertial")
        if inertial is None:
            masses.append(0.0)
            local_coms.append((0.0, 0.0, 0.0))
            continue
        mass_element = inertial.find("mass")
        if mass_element is None:
            raise ValueError(f"link {name!r} inertial has no mass")
        mass = float(mass_element.get("value", "nan"))
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"invalid mass for link {name!r}: {mass}")
        masses.append(mass)
        local_coms.append(parse_vector(inertial.find("origin"), "xyz"))
    total_mass = sum(masses)
    expected_mass = float(
        metadata["robot_mass_with_fixed_hands_without_payload_kg"]
    )
    if not math.isclose(total_mass, expected_mass, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"mass changed: expected {expected_mass}, got {total_mass}")
    missing_frames = set(TASK_FRAMES) - link_index.keys()
    if missing_frames:
        raise ValueError(f"missing task frames: {sorted(missing_frames)}")

    return {
        "urdf_hash": urdf_hash,
        "active_names": active_names,
        "topological_links": topological_links,
        "specs": specs,
        "configuration_types": configuration_types,
        "masks": masks,
        "masses": masses,
        "local_coms": local_coms,
        "total_mass": total_mass,
        "task_link_indices": [link_index[name] for name in TASK_FRAMES],
    }


def generate_header(model: dict[str, object], urdf: Path) -> bytes:
    specs = model["specs"]
    assert isinstance(specs, list)
    joint_rows = []
    for spec in specs:
        assert isinstance(spec, dict)
        joint_rows.append(
            "    {"
            f"{spec['parent']}, {spec['child']}, {spec['q']}, {spec['type']}, "
            f"{{{row(spec['translation'])}}}, "
            f"{{{row(spec['rotation'])}}}, "
            f"{{{row(spec['axis'])}}}"
            "},"
        )
    masks = model["masks"]
    assert isinstance(masks, list)
    mask_rows = [f"    0x{mask:09x}ULL," for mask in masks]
    masses = model["masses"]
    local_coms = model["local_coms"]
    assert isinstance(masses, list) and isinstance(local_coms, list)
    replacements = {
        "@URDF_PATH@": display_path(urdf),
        "@URDF_SHA256@": str(model["urdf_hash"]),
        "@DIM@": str(len(model["active_names"])),
        "@LINK_COUNT@": str(len(model["topological_links"])),
        "@TREE_JOINT_COUNT@": str(len(specs)),
        "@TOTAL_MASS@": float_literal(float(model["total_mass"])),
        "@JOINT_SPECS@": "\n".join(joint_rows),
        "@CONFIGURATION_TYPES@": "    "
        + ", ".join(str(value) for value in model["configuration_types"]),
        "@ANCESTOR_MASKS@": "\n".join(mask_rows),
        "@LINK_MASSES@": "\n".join(
            "    " + ", ".join(float_literal(value) for value in masses[index:index + 8]) + ","
            for index in range(0, len(masses), 8)
        ),
        "@LINK_COM_LOCAL@": "\n".join(
            f"    {{{row(values)}}}," for values in local_coms
        ),
        "@TASK_LINK_INDICES@": "    "
        + ", ".join(str(value) for value in model["task_link_indices"]),
    }
    header = HEADER_TEMPLATE
    for marker, value in replacements.items():
        header = header.replace(marker, value)
    if "@" in header:
        raise ValueError("unresolved generated-header marker")
    return header.encode("utf-8")


def build_metadata(
    model: dict[str, object], urdf: Path, output: Path, header: bytes
) -> bytes:
    metadata = {
        "schema_version": 1,
        "planning_urdf": display_path(urdf),
        "planning_urdf_sha256": model["urdf_hash"],
        "generated_header": display_path(output),
        "generated_header_sha256": sha256_bytes(header),
        "configuration_dimension": len(model["active_names"]),
        "configuration_order": list(model["active_names"]),
        "link_count": len(model["topological_links"]),
        "tree_joint_count": len(model["specs"]),
        "topological_link_order": model["topological_links"],
        "task_frames": {
            name: index
            for name, index in zip(TASK_FRAMES, model["task_link_indices"])
        },
        "total_mass_kg": model["total_mass"],
        "jacobians": {
            "task_frame_position": [3, len(model["active_names"])],
            "task_frame_world_angular": [3, len(model["active_names"])],
            "center_of_mass": [3, len(model["active_names"])],
            "bimanual_relative_position": [3, len(model["active_names"])],
            "bimanual_relative_angular": [3, len(model["active_names"])],
        },
    }
    return json.dumps(metadata, indent=2).encode("utf-8") + b"\n"


def check_file(path: Path, expected: bytes) -> None:
    if not path.is_file() or path.read_bytes() != expected:
        raise ValueError(f"generated artifact is missing or stale: {path}")


def main() -> None:
    args = parse_args()
    urdf = args.urdf.resolve()
    model_metadata = args.model_metadata.resolve()
    constraint_contract = args.constraint_contract.resolve()
    output = args.output.resolve()
    metadata_output = args.metadata_output.resolve()
    try:
        model = build_model(urdf, model_metadata, constraint_contract)
        header = generate_header(model, urdf)
        metadata = build_metadata(model, urdf, output, header)
        if args.check:
            check_file(output, header)
            check_file(metadata_output, metadata)
            print(
                f"PASS: {len(model['active_names'])} DoF, "
                f"{len(model['topological_links'])} links, "
                f"{len(model['specs'])} tree joints"
            )
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(header)
        metadata_output.write_bytes(metadata)
        print(f"wrote {output}")
        print(f"wrote {metadata_output}")
    except (KeyError, OSError, ET.ParseError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the IGRIS-C CUDA sphere-collision backend from planning meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import trimesh


RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]
DEFAULT_URDF = RESOURCE_DIR / "igris_c_planning.urdf"
DEFAULT_MODEL_METADATA = RESOURCE_DIR / "model_metadata.json"
DEFAULT_KINEMATICS_METADATA = RESOURCE_DIR / "kinematics_metadata.json"
DEFAULT_CONTRACT = RESOURCE_DIR / "constraint_contract.json"
DEFAULT_PROBLEM = REPOSITORY_DIR / "scripts" / "igris_c_problems.json"
DEFAULT_OUTPUT = REPOSITORY_DIR / "src" / "robots" / "igris_c_collision.cuh"
DEFAULT_METADATA_OUTPUT = RESOURCE_DIR / "collision_metadata.json"

# Each collision mesh link-frame AABB is partitioned into cells no longer than
# this value. The AABB construction uses only deterministic min/max reductions.
# One circumscribed sphere per cell gives a deterministic conservative cover.
# The full planner kernel stores fine-sphere positions in static shared memory.
# These two extents keep IGRIS-C below the 48 KiB kernel limit while preserving
# a conservative cover. The payload's simple box can use a coarser regular grid
# than the detailed robot meshes.
CELL_EXTENT_M = 0.105
PAYLOAD_CELL_EXTENT_M = 0.125
SPHERE_INFLATION_M = 0.002
ALLOWED_OVERLAP_BUFFER_M = 0.001
KINEMATIC_EXCLUSION_DISTANCE = 2


HEADER_TEMPLATE = r'''#pragma once

// Generated from @URDF_PATH@ by
// resources/igris_c/generate_collision_header.py.
// Planning-model SHA-256: @URDF_SHA256@

#include "src/collision/environment.hh"
#include "src/robots/igris_c.cuh"
#include "src/robots/igris_c_kinematics.cuh"

namespace ppln::collision {

constexpr int IGRIS_C_COLLISION_BATCH_SIZE = 16;
constexpr int IGRIS_C_SPHERE_COUNT = @SPHERE_COUNT@;
constexpr int IGRIS_C_APPROX_SPHERE_COUNT = @APPROX_SPHERE_COUNT@;
constexpr int IGRIS_C_PAYLOAD_SPHERE_BEGIN = @PAYLOAD_SPHERE_BEGIN@;
constexpr int IGRIS_C_PAYLOAD_SPHERE_COUNT = @PAYLOAD_SPHERE_COUNT@;
constexpr int IGRIS_C_JOINT_FLAG_STRIDE = 1;
constexpr int IGRIS_C_TRANSFORM_SLOTS = 1;
constexpr int IGRIS_C_SELF_COLLISION_PAIR_COUNT = @PAIR_COUNT@;
constexpr int IGRIS_C_SELF_COLLISION_GROUP_COUNT = @GROUP_COUNT@;

struct IgrisCCollisionSphere {
    int link_index;
    float center[3];
    float radius;
};

struct IgrisCApproxCollisionSphere {
    int link_index;
    float center[3];
    float radius;
};

struct IgrisCSelfCollisionGroup {
    unsigned short first_approximate;
    unsigned short second_approximate;
    unsigned short first_representative;
    unsigned short second_representative;
    float first_radius;
    float second_radius;
    unsigned int pair_begin;
    unsigned int pair_count;
};

__device__ __constant__ IgrisCCollisionSphere
igris_c_collision_spheres[IGRIS_C_SPHERE_COUNT] = {
@SPHERES@
};

__device__ __constant__ IgrisCApproxCollisionSphere
igris_c_approx_collision_spheres[IGRIS_C_APPROX_SPHERE_COUNT] = {
@APPROX_SPHERES@
};

// The pair table is global read-only device memory because it can exceed the
// 64 KiB CUDA constant-memory budget.
__device__ const unsigned short
igris_c_self_collision_pairs[IGRIS_C_SELF_COLLISION_PAIR_COUNT][2] = {
@PAIRS@
};

// Each representative sphere encloses every fine sphere on one rigid link.
// A non-overlapping link pair therefore cannot contain a fine-sphere collision.
__device__ const IgrisCSelfCollisionGroup
igris_c_self_collision_groups[IGRIS_C_SELF_COLLISION_GROUP_COUNT] = {
@GROUPS@
};

template <>
__device__ void fk<ppln::robots::IgrisC>(
    const float *q,
    volatile float *sphere_pos,
    float *,
    const int tid
) {
    const int lane = tid % 4;
    if (lane != 0) {
        return;
    }
    const int batch = tid / 4;
    IgrisCTransform link_poses[IGRIS_C_LINK_COUNT];
    float joint_origins[IGRIS_C_DIM][3];
    float joint_axes[IGRIS_C_DIM][3];
    igris_c_forward_model(q, link_poses, joint_origins, joint_axes);

    for (int sphere = 0; sphere < IGRIS_C_SPHERE_COUNT; ++sphere) {
        const IgrisCCollisionSphere &spec = igris_c_collision_spheres[sphere];
        const IgrisCTransform &link_pose = link_poses[spec.link_index];
        float world_center[3];
        igris_c_rotate(link_pose.rotation, spec.center, world_center);
        for (int component = 0; component < 3; ++component) {
            sphere_pos[
                sphere * IGRIS_C_COLLISION_BATCH_SIZE * 3 +
                batch * 3 + component
            ] = world_center[component] + link_pose.translation[component];
        }
    }
}

template <>
__device__ void fk_approx<ppln::robots::IgrisC>(
    const float *q,
    volatile float *sphere_pos,
    float *,
    const int tid
) {
    const int lane = tid % 4;
    const int batch = tid / 4;
    if (lane != 0) {
        return;
    }
    IgrisCTransform link_poses[IGRIS_C_LINK_COUNT];
    float joint_origins[IGRIS_C_DIM][3];
    float joint_axes[IGRIS_C_DIM][3];
    igris_c_forward_model(q, link_poses, joint_origins, joint_axes);
    for (int sphere = 0; sphere < IGRIS_C_APPROX_SPHERE_COUNT; ++sphere) {
        const IgrisCApproxCollisionSphere &spec =
            igris_c_approx_collision_spheres[sphere];
        const IgrisCTransform &link_pose = link_poses[spec.link_index];
        float world_center[3];
        igris_c_rotate(link_pose.rotation, spec.center, world_center);
        for (int component = 0; component < 3; ++component) {
            sphere_pos[
                sphere * IGRIS_C_COLLISION_BATCH_SIZE * 3
                + batch * 3 + component
            ] = world_center[component] + link_pose.translation[component];
        }
    }
}

template <>
__device__ bool env_collision_check_approx<ppln::robots::IgrisC>(
    volatile float *sphere_pos,
    volatile int *joint_in_collision,
    Environment<float> *environment,
    const int tid
) {
    const int lane = tid % 4;
    const int batch = tid / 4;
    bool possible_collision = false;
    for (int sphere = lane;
         sphere < IGRIS_C_APPROX_SPHERE_COUNT;
         sphere += 4) {
        possible_collision = possible_collision || sphere_environment_in_collision(
            environment,
            sphere_pos[
                sphere * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3
            ],
            sphere_pos[
                sphere * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 1
            ],
            sphere_pos[
                sphere * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 2
            ],
            igris_c_approx_collision_spheres[sphere].radius
        );
    }
    if (possible_collision) {
        joint_in_collision[batch] = 1;
    }
    return !possible_collision;
}

template <>
__device__ bool self_collision_check_approx<ppln::robots::IgrisC>(
    volatile float *sphere_pos,
    volatile int *joint_in_collision,
    const int tid
) {
    const int lane = tid % 4;
    const int batch = tid / 4;
    bool possible_collision = false;
    for (int group_index = lane;
         group_index < IGRIS_C_SELF_COLLISION_GROUP_COUNT;
         group_index += 4) {
        const IgrisCSelfCollisionGroup &group =
            igris_c_self_collision_groups[group_index];
        const int first = group.first_approximate;
        const int second = group.second_approximate;
        possible_collision = possible_collision || (
            sphere_sphere_self_collision(
                sphere_pos[
                    first * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3
                ],
                sphere_pos[
                    first * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 1
                ],
                sphere_pos[
                    first * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 2
                ],
                igris_c_approx_collision_spheres[first].radius,
                sphere_pos[
                    second * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3
                ],
                sphere_pos[
                    second * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 1
                ],
                sphere_pos[
                    second * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 2
                ],
                igris_c_approx_collision_spheres[second].radius
            ) != 0.0f
        );
    }
    if (possible_collision) {
        joint_in_collision[batch] = 1;
    }
    return !possible_collision;
}

template <>
__device__ bool env_collision_check<ppln::robots::IgrisC>(
    volatile float *sphere_pos,
    volatile int *,
    Environment<float> *environment,
    const int tid
) {
    const int lane = tid % 4;
    const int batch = tid / 4;
    bool collision = false;
    for (int sphere = lane; sphere < IGRIS_C_SPHERE_COUNT; sphere += 4) {
        collision = collision || sphere_environment_in_collision(
            environment,
            sphere_pos[
                sphere * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3
            ],
            sphere_pos[
                sphere * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 1
            ],
            sphere_pos[
                sphere * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 2
            ],
            igris_c_collision_spheres[sphere].radius
        );
    }
    return !collision;
}

template <>
__device__ bool self_collision_check<ppln::robots::IgrisC>(
    volatile float *sphere_pos,
    volatile int *,
    const int tid
) {
    const int lane = tid % 4;
    const int batch = tid / 4;
    bool collision = false;
    for (int group_index = lane;
         group_index < IGRIS_C_SELF_COLLISION_GROUP_COUNT;
         group_index += 4) {
        const IgrisCSelfCollisionGroup &group =
            igris_c_self_collision_groups[group_index];
        const int first_representative = group.first_representative;
        const int second_representative = group.second_representative;
        const bool link_bounds_overlap = sphere_sphere_self_collision(
            sphere_pos[
                first_representative * IGRIS_C_COLLISION_BATCH_SIZE * 3
                + batch * 3
            ],
            sphere_pos[
                first_representative * IGRIS_C_COLLISION_BATCH_SIZE * 3
                + batch * 3 + 1
            ],
            sphere_pos[
                first_representative * IGRIS_C_COLLISION_BATCH_SIZE * 3
                + batch * 3 + 2
            ],
            group.first_radius,
            sphere_pos[
                second_representative * IGRIS_C_COLLISION_BATCH_SIZE * 3
                + batch * 3
            ],
            sphere_pos[
                second_representative * IGRIS_C_COLLISION_BATCH_SIZE * 3
                + batch * 3 + 1
            ],
            sphere_pos[
                second_representative * IGRIS_C_COLLISION_BATCH_SIZE * 3
                + batch * 3 + 2
            ],
            group.second_radius
        ) != 0.0f;
        if (!link_bounds_overlap) {
            continue;
        }
        const unsigned int pair_end = group.pair_begin + group.pair_count;
        for (unsigned int pair_index = group.pair_begin;
             pair_index < pair_end;
             ++pair_index) {
            const int first = igris_c_self_collision_pairs[pair_index][0];
            const int second = igris_c_self_collision_pairs[pair_index][1];
            collision = collision || (sphere_sphere_self_collision(
                sphere_pos[
                    first * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3
                ],
                sphere_pos[
                    first * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 1
                ],
                sphere_pos[
                    first * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 2
                ],
                igris_c_collision_spheres[first].radius,
                sphere_pos[
                    second * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3
                ],
                sphere_pos[
                    second * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 1
                ],
                sphere_pos[
                    second * IGRIS_C_COLLISION_BATCH_SIZE * 3 + batch * 3 + 2
                ],
                igris_c_collision_spheres[second].radius
            ) != 0.0f);
        }
    }
    return !collision;
}

}  // namespace ppln::collision
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--model-metadata", type=Path, default=DEFAULT_MODEL_METADATA)
    parser.add_argument(
        "--kinematics-metadata", type=Path, default=DEFAULT_KINEMATICS_METADATA
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--problem", type=Path, default=DEFAULT_PROBLEM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def float_literal(value: float) -> str:
    if abs(value) < 5.0e-18:
        value = 0.0
    text = format(value, ".17g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"


def parse_vector(element: ET.Element | None, attribute: str, default: str) -> np.ndarray:
    text = default if element is None else element.get(attribute, default)
    values = np.fromstring(text, sep=" ", dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid {attribute}: {text!r}")
    return values


def rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=float,
    )


def collision_spheres(
    urdf: Path,
    robot: ET.Element,
    link_indices: dict[str, int],
    payload_link: str,
) -> list[dict[str, object]]:
    spheres: list[dict[str, object]] = []
    for link in robot.findall("link"):
        link_name = link.get("name", "")
        for collision_index, collision in enumerate(link.findall("collision")):
            geometry = collision.find("geometry")
            mesh_element = None if geometry is None else geometry.find("mesh")
            box_element = None if geometry is None else geometry.find("box")
            if mesh_element is not None and box_element is None:
                mesh_path = (
                    urdf.parent / mesh_element.get("filename", "")
                ).resolve()
                mesh = trimesh.load_mesh(mesh_path, process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(
                        f"collision asset is not a triangle mesh: {mesh_path}"
                    )
                scale = parse_vector(mesh_element, "scale", "1 1 1")
                vertices = np.asarray(mesh.vertices, dtype=float) * scale
            elif box_element is not None and mesh_element is None:
                size = parse_vector(box_element, "size", "")
                if np.any(size <= 0.0):
                    raise ValueError(
                        f"collision box has non-positive size: {link_name}"
                    )
                half = 0.5 * size
                vertices = np.asarray(
                    [
                        (sx * half[0], sy * half[1], sz * half[2])
                        for sx in (-1.0, 1.0)
                        for sy in (-1.0, 1.0)
                        for sz in (-1.0, 1.0)
                    ],
                    dtype=float,
                )
            else:
                raise ValueError(
                    "IGRIS collision generator requires exactly one mesh or box "
                    f"geometry: {link_name}"
                )
            origin = collision.find("origin")
            origin_translation = parse_vector(origin, "xyz", "0 0 0")
            origin_rotation = rpy_rotation(parse_vector(origin, "rpy", "0 0 0"))
            vertices = (origin_rotation @ vertices.T).T + origin_translation
            minimum = vertices.min(axis=0)
            maximum = vertices.max(axis=0)
            extents = maximum - minimum
            cell_extent = (
                PAYLOAD_CELL_EXTENT_M
                if link_name == payload_link
                else CELL_EXTENT_M
            )
            divisions = np.maximum(1, np.ceil(extents / cell_extent).astype(int))
            cell_extents = extents / divisions
            radius = 0.5 * float(np.linalg.norm(cell_extents)) + SPHERE_INFLATION_M
            for cell_index in np.ndindex(tuple(int(value) for value in divisions)):
                center = minimum + (
                    np.asarray(cell_index, dtype=float) + 0.5
                ) * cell_extents
                spheres.append(
                    {
                        "link": link_name,
                        "link_index": link_indices[link_name],
                        "collision_index": collision_index,
                        "center": center,
                        "radius": radius,
                    }
                )
    if not spheres or len(spheres) >= 65536:
        raise ValueError(f"invalid collision sphere count: {len(spheres)}")
    return spheres


def build_tree_data(
    robot: ET.Element,
    active_joint_names: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, int]]:
    parent_joint = {
        joint.find("child").get("link", ""): joint
        for joint in robot.findall("joint")
    }
    active_index = {name: index for index, name in enumerate(active_joint_names)}
    parent_links: dict[str, str] = {}
    masks: dict[str, int] = {}

    def ancestor_mask(link: str) -> int:
        if link in masks:
            return masks[link]
        if link not in parent_joint:
            masks[link] = 0
            return 0
        joint = parent_joint[link]
        parent = joint.find("parent").get("link", "")
        parent_links[link] = parent
        result = ancestor_mask(parent)
        joint_index = active_index.get(joint.get("name", ""))
        if joint_index is not None:
            result |= 1 << joint_index
        masks[link] = result
        return result

    for link in robot.findall("link"):
        ancestor_mask(link.get("name", ""))
    return parent_links, masks


def tree_distance(first: str, second: str, parents: dict[str, str]) -> int:
    first_ancestors: dict[str, int] = {}
    current = first
    distance = 0
    while True:
        first_ancestors[current] = distance
        if current not in parents:
            break
        current = parents[current]
        distance += 1
    current = second
    distance = 0
    while current not in first_ancestors:
        current = parents[current]
        distance += 1
    return distance + first_ancestors[current]


def allowed_configurations(contract: dict, problem_root: dict) -> list[np.ndarray]:
    problem = problem_root["problems"]["igris_c_shelf_lift"][0]
    values = [
        contract["constraint_consistent_seed"]["configuration"],
        problem["start"],
        *problem["goals"],
    ]
    result = [np.asarray(configuration, dtype=float) for configuration in values]
    if any(configuration.shape != (35,) for configuration in result):
        raise ValueError("collision allowed configurations must be 35-D")
    return result


def world_spheres(
    model: pin.Model,
    spheres: list[dict[str, object]],
    configurations: list[np.ndarray],
) -> list[list[tuple[np.ndarray, float]]]:
    data = model.createData()
    result: list[list[tuple[np.ndarray, float]]] = []
    for configuration in configurations:
        pin.forwardKinematics(model, data, configuration)
        pin.updateFramePlacements(model, data)
        posed: list[tuple[np.ndarray, float]] = []
        for sphere in spheres:
            placement = data.oMf[model.getFrameId(str(sphere["link"]))]
            center = (
                np.asarray(placement.rotation) @ np.asarray(sphere["center"])
                + np.asarray(placement.translation)
            )
            posed.append((center, float(sphere["radius"])))
        result.append(posed)
    return result


def self_collision_pairs(
    spheres: list[dict[str, object]],
    parents: dict[str, str],
    masks: dict[str, int],
    allowed_world_spheres: list[list[tuple[np.ndarray, float]]],
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    pairs: list[tuple[int, int]] = []
    excluded_same_link = 0
    excluded_fixed_relative = 0
    excluded_nearby = 0
    excluded_allowed_overlap = 0
    for first in range(len(spheres)):
        for second in range(first + 1, len(spheres)):
            first_link = str(spheres[first]["link"])
            second_link = str(spheres[second]["link"])
            if first_link == second_link:
                excluded_same_link += 1
                continue
            if masks[first_link] == masks[second_link]:
                excluded_fixed_relative += 1
                continue
            if tree_distance(first_link, second_link, parents) <= KINEMATIC_EXCLUSION_DISTANCE:
                excluded_nearby += 1
                continue
            overlaps_allowed_state = any(
                np.linalg.norm(posed[first][0] - posed[second][0])
                <= posed[first][1] + posed[second][1] + ALLOWED_OVERLAP_BUFFER_M
                for posed in allowed_world_spheres
            )
            if overlaps_allowed_state:
                excluded_allowed_overlap += 1
                continue
            pairs.append((first, second))
    return pairs, {
        "same_link": excluded_same_link,
        "fixed_relative": excluded_fixed_relative,
        "nearby_kinematic_links": excluded_nearby,
        "allowed_state_proxy_overlap": excluded_allowed_overlap,
    }


def approximate_collision_spheres(
    spheres: list[dict[str, object]],
    masks: dict[str, int],
    model: pin.Model,
) -> list[dict[str, object]]:
    by_rigid_group: dict[int, list[int]] = {}
    for index, sphere in enumerate(spheres):
        by_rigid_group.setdefault(masks[str(sphere["link"])], []).append(index)
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    result: list[dict[str, object]] = []
    for indices in by_rigid_group.values():
        anchor_link = str(spheres[indices[0]]["link"])
        anchor = data.oMf[model.getFrameId(anchor_link)]
        centers = []
        for index in indices:
            sphere = spheres[index]
            placement = data.oMf[model.getFrameId(str(sphere["link"]))]
            world_center = (
                np.asarray(placement.rotation) @ np.asarray(sphere["center"])
                + np.asarray(placement.translation)
            )
            centers.append(
                np.asarray(anchor.rotation).T
                @ (world_center - np.asarray(anchor.translation))
            )
        centers = np.asarray(centers)
        radii = np.asarray([spheres[index]["radius"] for index in indices])
        minimum = np.min(centers - radii[:, None], axis=0)
        maximum = np.max(centers + radii[:, None], axis=0)
        center = 0.5 * (minimum + maximum)
        radius = max(
            float(np.linalg.norm(centers[offset] - center)) + radii[offset]
            for offset in range(len(indices))
        )
        result.append(
            {
                "links": list(
                    dict.fromkeys(str(spheres[index]["link"]) for index in indices)
                ),
                "link": anchor_link,
                "link_index": spheres[indices[0]]["link_index"],
                "center": center,
                "radius": radius,
            }
        )
    return result


def grouped_self_collision_pairs(
    spheres: list[dict[str, object]],
    pairs: list[tuple[int, int]],
    approximate_spheres: list[dict[str, object]],
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    link_spheres: dict[str, list[int]] = {}
    for sphere_index, sphere in enumerate(spheres):
        link_spheres.setdefault(str(sphere["link"]), []).append(sphere_index)

    link_bounds: dict[str, tuple[int, float]] = {}
    for link, indices in link_spheres.items():
        representative = indices[0]
        center = np.asarray(spheres[representative]["center"], dtype=float)
        radius = max(
            float(np.linalg.norm(np.asarray(spheres[index]["center"]) - center))
            + float(spheres[index]["radius"])
            for index in indices
        )
        link_bounds[link] = (representative, radius)
    approximate_index = {
        link: index
        for index, sphere in enumerate(approximate_spheres)
        for link in sphere["links"]
    }

    by_link_pair: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for first, second in pairs:
        key = (str(spheres[first]["link"]), str(spheres[second]["link"]))
        by_link_pair.setdefault(key, []).append((first, second))

    flattened: list[tuple[int, int]] = []
    groups: list[dict[str, object]] = []
    for (first_link, second_link), group_pairs in by_link_pair.items():
        first_representative, first_radius = link_bounds[first_link]
        second_representative, second_radius = link_bounds[second_link]
        groups.append(
            {
                "first_approximate": approximate_index[first_link],
                "second_approximate": approximate_index[second_link],
                "first_representative": first_representative,
                "second_representative": second_representative,
                "first_radius": first_radius,
                "second_radius": second_radius,
                "pair_begin": len(flattened),
                "pair_count": len(group_pairs),
            }
        )
        flattened.extend(group_pairs)
    if len(flattened) != len(pairs):
        raise ValueError("self-collision pair grouping lost pairs")
    return flattened, groups


def generate(args: argparse.Namespace) -> tuple[bytes, bytes]:
    urdf = args.urdf.resolve()
    urdf_bytes = urdf.read_bytes()
    urdf_hash = sha256_bytes(urdf_bytes)
    model_metadata = json.loads(args.model_metadata.read_text(encoding="utf-8"))
    kinematics_metadata = json.loads(
        args.kinematics_metadata.read_text(encoding="utf-8")
    )
    if model_metadata["generated_urdf_sha256"] != urdf_hash:
        raise ValueError("model metadata does not match collision URDF")
    if kinematics_metadata["planning_urdf_sha256"] != urdf_hash:
        raise ValueError("kinematics metadata does not match collision URDF")
    robot = ET.fromstring(urdf_bytes)
    topological_links = kinematics_metadata["topological_link_order"]
    link_indices = {name: index for index, name in enumerate(topological_links)}
    active_joint_names = tuple(
        item["name"] for item in model_metadata["configuration"]
    )
    payload_model = model_metadata["payload_collision_model"]
    payload_link = payload_model["link"]
    spheres = collision_spheres(urdf, robot, link_indices, payload_link)
    payload_sphere_indices = [
        index
        for index, sphere in enumerate(spheres)
        if sphere["link"] == payload_link
    ]
    if not payload_sphere_indices:
        raise ValueError("attached payload generated no collision spheres")
    payload_sphere_begin = payload_sphere_indices[0]
    if payload_sphere_indices != list(
        range(payload_sphere_begin, payload_sphere_begin + len(payload_sphere_indices))
    ):
        raise ValueError("attached payload spheres are not contiguous")
    parents, masks = build_tree_data(robot, active_joint_names)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    problem_root = json.loads(args.problem.read_text(encoding="utf-8"))
    configurations = allowed_configurations(contract, problem_root)
    pin_model = pin.buildModelFromUrdf(str(urdf))
    approximate_spheres = approximate_collision_spheres(
        spheres, masks, pin_model
    )
    posed = world_spheres(pin_model, spheres, configurations)
    pairs, exclusions = self_collision_pairs(spheres, parents, masks, posed)
    pairs, pair_groups = grouped_self_collision_pairs(
        spheres, pairs, approximate_spheres
    )

    sphere_rows = "\n".join(
        "    {"
        f"{sphere['link_index']}, "
        "{" + ", ".join(float_literal(float(v)) for v in sphere["center"]) + "}, "
        f"{float_literal(float(sphere['radius']))}"
        "},"
        for sphere in spheres
    )
    approximate_sphere_rows = "\n".join(
        "    {"
        f"{sphere['link_index']}, "
        "{" + ", ".join(float_literal(float(v)) for v in sphere["center"]) + "}, "
        f"{float_literal(float(sphere['radius']))}"
        "},"
        for sphere in approximate_spheres
    )
    pair_rows = "\n".join(
        "    " + ", ".join(
            f"{{{first}, {second}}}"
            for first, second in pairs[index:index + 10]
        ) + ","
        for index in range(0, len(pairs), 10)
    )
    group_rows = "\n".join(
        "    {"
        f"{group['first_approximate']}, "
        f"{group['second_approximate']}, "
        f"{group['first_representative']}, "
        f"{group['second_representative']}, "
        f"{float_literal(float(group['first_radius']))}, "
        f"{float_literal(float(group['second_radius']))}, "
        f"{group['pair_begin']}, {group['pair_count']}"
        "},"
        for group in pair_groups
    )
    replacements = {
        "@URDF_PATH@": display_path(urdf),
        "@URDF_SHA256@": urdf_hash,
        "@SPHERE_COUNT@": str(len(spheres)),
        "@APPROX_SPHERE_COUNT@": str(len(approximate_spheres)),
        "@PAYLOAD_SPHERE_BEGIN@": str(payload_sphere_begin),
        "@PAYLOAD_SPHERE_COUNT@": str(len(payload_sphere_indices)),
        "@PAIR_COUNT@": str(len(pairs)),
        "@GROUP_COUNT@": str(len(pair_groups)),
        "@SPHERES@": sphere_rows,
        "@APPROX_SPHERES@": approximate_sphere_rows,
        "@PAIRS@": pair_rows,
        "@GROUPS@": group_rows,
    }
    header = HEADER_TEMPLATE
    for marker, value in replacements.items():
        header = header.replace(marker, value)
    if "@" in header:
        raise ValueError("unresolved collision-header marker")
    header_bytes = header.encode("utf-8")
    metadata = {
        "schema_version": 1,
        "planning_urdf": display_path(urdf),
        "planning_urdf_sha256": urdf_hash,
        "generated_header": display_path(args.output),
        "generated_header_sha256": sha256_bytes(header_bytes),
        "representation": "conservative circumscribed spheres over deterministic partitioned link-frame collision-geometry AABBs",
        "cell_extent_m": CELL_EXTENT_M,
        "payload_cell_extent_m": PAYLOAD_CELL_EXTENT_M,
        "sphere_inflation_m": SPHERE_INFLATION_M,
        "sphere_count": len(spheres),
        "approximate_sphere_count": len(approximate_spheres),
        "self_collision_pair_count": len(pairs),
        "self_collision_link_pair_group_count": len(pair_groups),
        "kinematic_exclusion_distance": KINEMATIC_EXCLUSION_DISTANCE,
        "allowed_overlap_buffer_m": ALLOWED_OVERLAP_BUFFER_M,
        "allowed_overlap_configurations": ["constraint_seed", "shelf_start", "shelf_goal"],
        "excluded_pair_counts": exclusions,
        "payload_box_included": True,
        "payload_collision_model": {
            **payload_model,
            "sphere_index_begin": payload_sphere_begin,
            "sphere_count": len(payload_sphere_indices),
        },
    }
    metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8") + b"\n"
    return header_bytes, metadata_bytes


def check_file(path: Path, expected: bytes) -> None:
    if not path.is_file() or path.read_bytes() != expected:
        raise ValueError(f"generated artifact is stale: {path}")


def main() -> None:
    args = parse_args()
    header, metadata = generate(args)
    if args.check:
        check_file(args.output.resolve(), header)
        check_file(args.metadata_output.resolve(), metadata)
        print("PASS: IGRIS-C collision backend is current")
        return
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_bytes(header)
    args.metadata_output.resolve().write_bytes(metadata)
    parsed = json.loads(metadata)
    print(
        f"wrote {args.output.resolve()} "
        f"({parsed['sphere_count']} spheres, "
        f"{parsed['self_collision_pair_count']} self pairs)"
    )
    print(f"wrote {args.metadata_output.resolve()}")


if __name__ == "__main__":
    main()

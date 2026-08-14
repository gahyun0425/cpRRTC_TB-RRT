#!/usr/bin/env python3
"""Build fine and conservative one-sphere-per-link FFW-SG2 URDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


RESOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_PLANNING_URDF = RESOURCE_DIR / "ffw_sg2_planning.urdf"
DEFAULT_SPHERE_DATA = RESOURCE_DIR / "ffw_sg2_fine_spheres.json"
DEFAULT_FINE_OUTPUT = RESOURCE_DIR / "ffw_sg2_spherized.urdf"
DEFAULT_APPROX_OUTPUT = RESOURCE_DIR / "ffw_sg2_spherized_approx.urdf"
DEFAULT_METADATA_OUTPUT = RESOURCE_DIR / "collision_model_metadata.json"

BATCH_SIZE = 16
JOINT_FLAG_STRIDE = 16  # Pinocchio universe joint plus 15 active joints.
COORDINATES_PER_SPHERE = 3
FLOAT_BYTES = 4
INT_BYTES = 4
TRANSFORM_SLOTS_PER_CONFIGURATION = 2
TRANSFORM_FLOATS = 16
APPROX_PADDING_METERS = 1.0e-6

EXPECTED_SOURCE_YAML_SHA256 = (
    "915b4fb919d3c4d779956283ba945a0f44a34df27d9c1e23673706674392ea86"
)
EXPECTED_ACTIVE_JOINTS = (
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
EXPECTED_SPHERES_PER_LINK = {
    "lift_link": 9,
    "arm_base_link": 6,
    "arm_l_link2": 10,
    "arm_l_link1": 10,
    "arm_l_link3": 10,
    "arm_l_link4": 10,
    "arm_l_link5": 10,
    "arm_l_link6": 5,
    "arm_l_link7": 7,
    "gripper_l_rh_p12_rn_base": 2,
    "gripper_l_rh_p12_rn_r1": 1,
    "gripper_l_rh_p12_rn_r2": 1,
    "gripper_l_rh_p12_rn_l1": 1,
    "gripper_l_rh_p12_rn_l2": 1,
    "arm_r_link1": 3,
    "arm_r_link2": 3,
    "arm_r_link3": 3,
    "arm_r_link4": 4,
    "arm_r_link5": 2,
    "arm_r_link6": 5,
    "arm_r_link7": 7,
    "gripper_r_rh_p12_rn_base": 2,
    "gripper_r_rh_p12_rn_r1": 1,
    "gripper_r_rh_p12_rn_l1": 1,
    "head_link2": 8,
    "gripper_r_rh_p12_rn_r2": 1,
    "gripper_r_rh_p12_rn_l2": 1,
}
EXPECTED_UNMODELED_COLLISION_LINKS = {
    "base_link",
    "head_link1",
    "camera_l_link",
    "camera_r_link",
    "left_wheel_steer_link",
    "left_wheel_drive_link",
    "right_wheel_steer_link",
    "right_wheel_drive_link",
    "rear_wheel_steer_link",
    "rear_wheel_drive_link",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-urdf", type=Path, default=DEFAULT_PLANNING_URDF)
    parser.add_argument("--sphere-data", type=Path, default=DEFAULT_SPHERE_DATA)
    parser.add_argument(
        "--import-yaml",
        type=Path,
        help="Import the known 124-sphere cuRobo YAML before generating models.",
    )
    parser.add_argument("--fine-output", type=Path, default=DEFAULT_FINE_OUTPUT)
    parser.add_argument("--approx-output", type=Path, default=DEFAULT_APPROX_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_yaml(source: Path, destination: Path) -> None:
    source_hash = sha256_file(source)
    if source_hash != EXPECTED_SOURCE_YAML_SHA256:
        raise ValueError(
            "unexpected fine-sphere YAML hash: "
            f"expected {EXPECTED_SOURCE_YAML_SHA256}, got {source_hash}"
        )

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required only for --import-yaml") from error

    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    sphere_map = raw.get("collision_spheres")
    if not isinstance(sphere_map, dict):
        raise ValueError("YAML does not contain a collision_spheres mapping")

    normalized: dict[str, list[dict[str, object]]] = {}
    for link_name, spheres in sphere_map.items():
        normalized[link_name] = []
        for sphere in spheres:
            normalized[link_name].append(
                {
                    "center": [float(value) for value in sphere["center"]],
                    "radius": float(sphere["radius"]),
                }
            )

    imported = {
        "schema_version": 1,
        "source": {
            "format": "curobo_collision_spheres_yaml",
            "sha256": source_hash,
            "original_filename": source.name,
        },
        "link_order": list(normalized),
        "collision_spheres": normalized,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(imported, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_and_validate_spheres(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sphere_map = data.get("collision_spheres")
    link_order = data.get("link_order")
    if not isinstance(sphere_map, dict) or link_order != list(sphere_map):
        raise ValueError("sphere data link_order does not match collision_spheres")
    if set(sphere_map) != set(EXPECTED_SPHERES_PER_LINK):
        raise ValueError("sphere data link set differs from the validated FFW-SG2 set")

    for link_name, expected_count in EXPECTED_SPHERES_PER_LINK.items():
        spheres = sphere_map[link_name]
        if len(spheres) != expected_count:
            raise ValueError(
                f"{link_name}: expected {expected_count} spheres, got {len(spheres)}"
            )
        for index, sphere in enumerate(spheres):
            center = sphere.get("center")
            radius = sphere.get("radius")
            if (
                not isinstance(center, list)
                or len(center) != 3
                or not all(math.isfinite(float(value)) for value in center)
                or not math.isfinite(float(radius))
                or float(radius) <= 0.0
            ):
                raise ValueError(f"invalid sphere {link_name}[{index}]: {sphere}")

    sphere_count = sum(len(spheres) for spheres in sphere_map.values())
    if sphere_count != 124:
        raise ValueError(f"expected 124 fine spheres, got {sphere_count}")
    return data, sphere_map


def format_float(value: float) -> str:
    if abs(value) < 5.0e-15:
        value = 0.0
    return format(value, ".12g")


def append_sphere_collision(link: ET.Element, sphere: dict, index: int) -> None:
    collision = ET.SubElement(link, "collision", {"name": f"sphere_{index:03d}"})
    center = [float(value) for value in sphere["center"]]
    ET.SubElement(
        collision,
        "origin",
        {
            "rpy": "0 0 0",
            "xyz": " ".join(format_float(value) for value in center),
        },
    )
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(
        geometry,
        "sphere",
        {"radius": format_float(float(sphere["radius"]))},
    )


def active_joint_order(robot: ET.Element) -> tuple[str, ...]:
    return tuple(
        joint.get("name", "")
        for joint in robot.findall("joint")
        if joint.get("type") in {"continuous", "prismatic", "revolute"}
    )


def generate_spherized_urdf(
    planning_urdf: Path,
    output: Path,
    sphere_map: dict[str, list[dict]],
) -> None:
    tree = ET.parse(planning_urdf)
    robot = tree.getroot()
    if active_joint_order(robot) != EXPECTED_ACTIVE_JOINTS:
        raise ValueError("planning URDF does not have the canonical 15-DoF order")

    links = {link.get("name", ""): link for link in robot.findall("link")}
    missing_links = set(sphere_map) - links.keys()
    if missing_links:
        raise ValueError(f"sphere data refers to missing links: {sorted(missing_links)}")

    sphere_index = 0
    for link in robot.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
        for sphere in sphere_map.get(link.get("name", ""), []):
            append_sphere_collision(link, sphere, sphere_index)
            sphere_index += 1

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def conservative_link_sphere(spheres: list[dict]) -> dict[str, object]:
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for sphere in spheres:
        center = [float(value) for value in sphere["center"]]
        radius = float(sphere["radius"])
        for axis in range(3):
            mins[axis] = min(mins[axis], center[axis] - radius)
            maxs[axis] = max(maxs[axis], center[axis] + radius)

    center = [(mins[axis] + maxs[axis]) * 0.5 for axis in range(3)]
    radius = max(
        math.dist(center, [float(value) for value in sphere["center"]])
        + float(sphere["radius"])
        for sphere in spheres
    )
    return {"center": center, "radius": radius + APPROX_PADDING_METERS}


def build_approximation(
    fine_spheres: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    return {
        link_name: [conservative_link_sphere(spheres)]
        for link_name, spheres in fine_spheres.items()
    }


def validate_containment(
    fine_spheres: dict[str, list[dict]],
    approximate_spheres: dict[str, list[dict]],
) -> float:
    minimum_margin = math.inf
    for link_name, spheres in fine_spheres.items():
        approximate = approximate_spheres[link_name][0]
        approximate_center = [float(value) for value in approximate["center"]]
        approximate_radius = float(approximate["radius"])
        for sphere in spheres:
            required_radius = (
                math.dist(
                    approximate_center,
                    [float(value) for value in sphere["center"]],
                )
                + float(sphere["radius"])
            )
            margin = approximate_radius - required_radius
            if margin < -1.0e-12:
                raise ValueError(
                    f"approximate sphere does not contain {link_name}: {margin}"
                )
            minimum_margin = min(minimum_margin, margin)
    return minimum_margin


def planning_collision_links(planning_urdf: Path) -> set[str]:
    robot = ET.parse(planning_urdf).getroot()
    return {
        link.get("name", "")
        for link in robot.findall("link")
        if link.findall("collision")
    }


def build_metadata(
    planning_urdf: Path,
    sphere_data_path: Path,
    sphere_data: dict,
    fine_output: Path,
    approximate_output: Path,
    fine_spheres: dict[str, list[dict]],
    approximate_spheres: dict[str, list[dict]],
    minimum_margin: float,
) -> dict:
    link_count = len(fine_spheres)
    fine_count = sum(len(spheres) for spheres in fine_spheres.values())
    approximate_count = sum(len(spheres) for spheres in approximate_spheres.values())

    fine_position_floats = fine_count * BATCH_SIZE * COORDINATES_PER_SPHERE
    approximate_position_floats = (
        approximate_count * BATCH_SIZE * COORDINATES_PER_SPHERE
    )
    joint_flag_ints = JOINT_FLAG_STRIDE * BATCH_SIZE
    transform_floats = (
        BATCH_SIZE * TRANSFORM_SLOTS_PER_CONFIGURATION * TRANSFORM_FLOATS
    )
    model_shared_bytes = (
        (fine_position_floats + approximate_position_floats + transform_floats)
        * FLOAT_BYTES
        + joint_flag_ints * INT_BYTES
    )
    current_static_model_shared_bytes = (
        (6000 + 2500 + 512) * FLOAT_BYTES + 640 * INT_BYTES
    )

    unmodeled = planning_collision_links(planning_urdf) - set(fine_spheres)
    if unmodeled != EXPECTED_UNMODELED_COLLISION_LINKS:
        raise ValueError(
            "unexpected collision links omitted from the inherited sphere model: "
            f"{sorted(unmodeled)}"
        )

    return {
        "schema_version": 1,
        "robot": "ffw_sg2_follower",
        "active_dof": len(EXPECTED_ACTIVE_JOINTS),
        "active_joint_order": list(EXPECTED_ACTIVE_JOINTS),
        "batch_size": BATCH_SIZE,
        "sources": {
            "planning_urdf": planning_urdf.name,
            "planning_urdf_sha256": sha256_file(planning_urdf),
            "fine_sphere_data": sphere_data_path.name,
            "fine_sphere_data_sha256": sha256_file(sphere_data_path),
            "imported_yaml_sha256": sphere_data["source"]["sha256"],
        },
        "fine_model": {
            "urdf": fine_output.name,
            "urdf_sha256": sha256_file(fine_output),
            "collision_links": link_count,
            "sphere_count": fine_count,
        },
        "approximate_model": {
            "urdf": approximate_output.name,
            "urdf_sha256": sha256_file(approximate_output),
            "collision_links": link_count,
            "sphere_count": approximate_count,
            "construction": "AABB midpoint plus max(center distance + fine radius)",
            "padding_m": APPROX_PADDING_METERS,
            "minimum_fine_sphere_containment_margin_m": minimum_margin,
        },
        "unmodeled_planning_collision_links": sorted(unmodeled),
        "memory_bytes": {
            "fine_runtime_positions": fine_position_floats * FLOAT_BYTES,
            "approximate_runtime_positions": approximate_position_floats
            * FLOAT_BYTES,
            "joint_flags": joint_flag_ints * INT_BYTES,
            "two_transform_slots": transform_floats * FLOAT_BYTES,
            "model_dependent_shared_total": model_shared_bytes,
            "current_static_model_buffers_total": current_static_model_shared_bytes,
            "right_sizing_savings": current_static_model_shared_bytes
            - model_shared_bytes,
            "fine_constant_float4": fine_count * 4 * FLOAT_BYTES,
            "approximate_constant_float4": approximate_count * 4 * FLOAT_BYTES,
        },
        "required_element_counts": {
            "fine_runtime_position_floats": fine_position_floats,
            "approximate_runtime_position_floats": approximate_position_floats,
            "joint_flag_ints": joint_flag_ints,
            "transform_floats_for_two_slots": transform_floats,
        },
        "current_prrtc_static_capacities": {
            "fine_runtime_position_floats": 6000,
            "approximate_runtime_position_floats": 2500,
            "joint_flag_ints": 640,
            "transform_floats": 512,
        },
        "capacity_headroom_elements": {
            "fine_runtime_position_floats": 6000 - fine_position_floats,
            "approximate_runtime_position_floats": 2500
            - approximate_position_floats,
            "joint_flag_ints": 640 - joint_flag_ints,
            "transform_floats_for_two_slots": 512 - transform_floats,
        },
        "generated_cricket_layout": {
            "joint_count_including_universe": JOINT_FLAG_STRIDE,
            "joint_flag_stride": JOINT_FLAG_STRIDE,
            "transform_slots": TRANSFORM_SLOTS_PER_CONFIGURATION,
        },
    }


def main() -> None:
    args = parse_args()
    planning_urdf = args.planning_urdf.resolve()
    sphere_data_path = args.sphere_data.resolve()
    fine_output = args.fine_output.resolve()
    approximate_output = args.approx_output.resolve()
    metadata_output = args.metadata_output.resolve()

    if args.import_yaml is not None:
        import_yaml(args.import_yaml.resolve(), sphere_data_path)

    sphere_data, fine_spheres = load_and_validate_spheres(sphere_data_path)
    approximate_spheres = build_approximation(fine_spheres)
    minimum_margin = validate_containment(fine_spheres, approximate_spheres)

    generate_spherized_urdf(planning_urdf, fine_output, fine_spheres)
    generate_spherized_urdf(
        planning_urdf,
        approximate_output,
        approximate_spheres,
    )
    metadata = build_metadata(
        planning_urdf,
        sphere_data_path,
        sphere_data,
        fine_output,
        approximate_output,
        fine_spheres,
        approximate_spheres,
        minimum_margin,
    )
    metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

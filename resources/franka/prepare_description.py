#!/usr/bin/env python3
"""Generate PATACON robot adapters from the pinned Franka FER description."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]
DESCRIPTION_DIR = REPOSITORY_DIR / "franka_description"
SOURCE_XACRO = DESCRIPTION_DIR / "robots" / "fer" / "fer.urdf.xacro"
SINGLE_URDF = RESOURCE_DIR / "fer_single.urdf"
DUAL_URDF = RESOURCE_DIR / "fer_dual.urdf"
SINGLE_MJCF = RESOURCE_DIR / "mujoco" / "fer_single.xml"
DUAL_MJCF = RESOURCE_DIR / "mujoco" / "fer_dual.xml"
METADATA_PATH = RESOURCE_DIR / "model_metadata.json"

SOURCE_REPOSITORY = "https://github.com/frankarobotics/franka_description.git"
SOURCE_RELEASE = "2.9.0"
SOURCE_COMMIT = "7aeeddc449edf8d62b594f9e36a81da53e7796f9"
SOURCE_PACKAGE_PREFIX = "package://franka_description/"
RELATIVE_DESCRIPTION_PREFIX = "../../franka_description/"

SINGLE_MOUNT = (0.0, 0.0, 0.0)
DUAL_MOUNTS = ((0.0, 0.2, 0.6), (0.0, -0.2, 0.6))
JOINT_LIMITS = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)
HAND_TCP_OFFSET_M = 0.1034

ARM_ORIGINS = (
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-1.570796326794897, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (1.570796326794897, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (1.570796326794897, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (-1.570796326794897, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.570796326794897, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (1.570796326794897, 0.0, 0.0)),
)

ARM_INERTIALS = (
    ((-0.0172, 0.0004, 0.0745), 2.3966, (0.0090, 0.0115, 0.0085, 0.0, 0.0020, 0.0)),
    ((0.00033, -0.02204, -0.04762), 2.7907, (0.01564782655, 0.01439883526, 0.00500443991, 0.00000531236, -0.00003676721, -0.00248480843)),
    ((0.00038, -0.09211, 0.01908), 2.5420, (0.01662427728, 0.00462501261, 0.01545124526, 0.00003086077, 0.00000835940, 0.00355899830)),
    ((0.05152, 0.01696, -0.02971), 2.2513, (0.00631741992, 0.00866285493, 0.00657804051, 0.00097309955, 0.00292525834, 0.00093469224)),
    ((-0.05113, 0.05825, 0.01698), 2.2037, (0.00784585566, 0.00649543136, 0.01003079917, -0.00339591957, 0.00158704074, -0.00181477884)),
    ((-0.00005, 0.03730, -0.09280), 2.2855, (0.02297014781, 0.02095060919, 0.00430606551, -0.00000949345, -0.00002063156, 0.00382345782)),
    ((0.06572, -0.00371, 0.00153), 1.353, (0.00087964522, 0.00277796968, 0.00286701969, -0.00021487814, -0.00011911662, 0.00001274322)),
    ((0.00089, -0.00044, 0.05491), 0.35973, (0.00019541063, 0.00019210361, 0.00017936256, 0.00000165231, 0.00000148826, -0.00000131132)),
)
HAND_INERTIAL = (
    (-0.0000376, 0.0119128, 0.0207260),
    0.6544,
    (0.00186, 0.00030, 0.00174, 0.0, 0.0, -0.00002),
)
FINGER_INERTIAL = (
    (0.0, 0.0152850, 0.0219675),
    0.0291,
    (0.00000849, 0.00000853, 0.00000177, 0.0, 0.0, -0.00000106),
)

SOURCE_FILES = (
    "package.xml",
    "robots/fer/fer.urdf.xacro",
    "robots/fer/kinematics.yaml",
    "robots/fer/joint_limits.yaml",
    "robots/fer/inertials.yaml",
    "robots/fer/dynamics.yaml",
    "robots/common/franka_robot.xacro",
    "robots/common/franka_arm.xacro",
    "robots/common/utils.xacro",
    "end_effectors/common/franka_hand.xacro",
    "end_effectors/common/utils.xacro",
    "end_effectors/franka_hand/franka_hand_arguments.xacro",
    "end_effectors/franka_hand/inertials.yaml",
)
MESH_FILES = tuple(
    f"meshes/robots/fer/{kind}/link{index}.{extension}"
    for kind, extension in (("collision", "stl"), ("visual", "dae"))
    for index in range(8)
) + (
    "meshes/robot_ee/franka_hand_white/collision/hand.stl",
    "meshes/robot_ee/franka_hand_white/visual/hand.dae",
    "meshes/robot_ee/franka_hand_white/visual/finger.dae",
)


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
            "franka_description is not initialized; run "
            "git submodule update --init franka_description"
        ) from error
    return result.stdout.strip()


def validate_source() -> None:
    if not SOURCE_XACRO.is_file():
        raise RuntimeError(
            "missing official Franka description; run "
            "git submodule update --init franka_description"
        )
    commit = description_commit()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"franka_description is at {commit}, expected {SOURCE_COMMIT} "
            f"(release {SOURCE_RELEASE})"
        )
    for relative_path in SOURCE_FILES + MESH_FILES:
        path = DESCRIPTION_DIR / relative_path
        if not path.is_file():
            raise RuntimeError(f"official description is missing {relative_path}")


def expand_official_xacro() -> ET.Element:
    """Expand upstream xacro without requiring a ROS workspace install."""
    with tempfile.TemporaryDirectory(prefix="patacon_franka_ament_") as temp:
        prefix = Path(temp)
        resource_index = (
            prefix / "share" / "ament_index" / "resource_index" / "packages"
        )
        resource_index.mkdir(parents=True)
        (resource_index / "franka_description").write_text("", encoding="utf-8")
        package_link = prefix / "share" / "franka_description"
        package_link.symlink_to(DESCRIPTION_DIR, target_is_directory=True)

        previous_ament_path = os.environ.get("AMENT_PREFIX_PATH")
        os.environ["AMENT_PREFIX_PATH"] = (
            str(prefix)
            if not previous_ament_path
            else f"{prefix}{os.pathsep}{previous_ament_path}"
        )
        try:
            try:
                import xacro
            except ImportError as error:
                raise RuntimeError(
                    "Python package 'xacro' is required (pip install xacro)"
                ) from error
            document = xacro.process_file(
                str(SOURCE_XACRO),
                mappings={
                    "with_sc": "False",
                    "ee_id": "franka_hand",
                    "hand": "True",
                    "no_prefix": "false",
                    "robot_type": "fer",
                },
            )
        finally:
            if previous_ament_path is None:
                os.environ.pop("AMENT_PREFIX_PATH", None)
            else:
                os.environ["AMENT_PREFIX_PATH"] = previous_ament_path
    return ET.fromstring(document.toxml())


def prefixed_name(value: str, prefix: str) -> str:
    if value == "base":
        return f"{prefix}_base"
    if value.startswith("fer_"):
        return f"{prefix}_{value.removeprefix('fer_')}"
    return value


def normalize_arm_element(element: ET.Element, prefix: str) -> ET.Element:
    element = copy.deepcopy(element)
    for node in element.iter():
        for attribute in ("name", "link", "joint"):
            if attribute in node.attrib:
                node.attrib[attribute] = prefixed_name(
                    node.attrib[attribute], prefix
                )
        filename = node.attrib.get("filename")
        if filename and filename.startswith(SOURCE_PACKAGE_PREFIX):
            node.attrib["filename"] = (
                RELATIVE_DESCRIPTION_PREFIX
                + filename.removeprefix(SOURCE_PACKAGE_PREFIX)
            )
    return element


def add_mount(root: ET.Element, prefix: str, xyz: tuple[float, float, float]) -> None:
    joint = ET.SubElement(
        root, "joint", {"name": f"{prefix}_mount_joint", "type": "fixed"}
    )
    ET.SubElement(joint, "origin", {"rpy": "0 0 0", "xyz": " ".join(map(str, xyz))})
    ET.SubElement(joint, "parent", {"link": "world"})
    ET.SubElement(joint, "child", {"link": f"{prefix}_base"})


def build_urdf(source: ET.Element, mounts: tuple[tuple[float, float, float], ...]) -> bytes:
    model_name = "fer_single" if len(mounts) == 1 else "fer_dual"
    root = ET.Element("robot", {"name": model_name})
    ET.SubElement(root, "link", {"name": "world"})
    for arm_index, mount in enumerate(mounts):
        prefix = f"fer{arm_index}"
        add_mount(root, prefix, mount)
        for child in source:
            root.append(normalize_arm_element(child, prefix))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def find_joint(root: ET.Element, name: str) -> ET.Element:
    joint = root.find(f"./joint[@name='{name}']")
    if joint is None:
        raise RuntimeError(f"generated URDF is missing joint {name}")
    return joint


def validate_generated_urdf(content: bytes, arm_count: int) -> None:
    root = ET.fromstring(content)
    movable = [
        joint
        for joint in root.findall("joint")
        if joint.attrib.get("type") in {"revolute", "prismatic", "continuous"}
    ]
    expected_movable = arm_count * 9
    if len(movable) != expected_movable:
        raise RuntimeError(
            f"generated URDF has {len(movable)} movable joints, "
            f"expected {expected_movable}"
        )
    for arm_index in range(arm_count):
        prefix = f"fer{arm_index}"
        for joint_index, expected in enumerate(JOINT_LIMITS, start=1):
            joint = find_joint(root, f"{prefix}_joint{joint_index}")
            limit = joint.find("limit")
            if limit is None:
                raise RuntimeError(f"{joint.attrib['name']} has no limit")
            actual = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
            if actual != expected:
                raise RuntimeError(
                    f"{joint.attrib['name']} limits {actual} do not match {expected}"
                )
        tcp_joint = find_joint(root, f"{prefix}_hand_tcp_joint")
        tcp_origin = tcp_joint.find("origin")
        tcp_z = float(tcp_origin.attrib["xyz"].split()[2])
        if tcp_z != HAND_TCP_OFFSET_M:
            raise RuntimeError(
                f"official hand TCP is {tcp_z} m, expected {HAND_TCP_OFFSET_M} m"
            )


def vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.15g}" for value in values)


def add_inertial(
    body: ET.Element,
    specification: tuple[
        tuple[float, float, float],
        float,
        tuple[float, float, float, float, float, float],
    ],
) -> None:
    position, mass, full_inertia = specification
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": vector(position),
            "mass": f"{mass:.15g}",
            "fullinertia": vector(full_inertia),
        },
    )


def add_mesh_geometries(body: ET.Element, mesh_prefix: str) -> None:
    ET.SubElement(
        body,
        "geom",
        {"class": "franka_visual", "mesh": f"{mesh_prefix}_visual"},
    )
    ET.SubElement(
        body,
        "geom",
        {"class": "franka_collision", "mesh": f"{mesh_prefix}_collision"},
    )


def add_finger_collisions(body: ET.Element, right: bool) -> None:
    boxes = (
        ((0.011, 0.0075, 0.010), (0.0, 0.0185, 0.011), None),
        ((0.011, 0.0044, 0.0019), (0.0, 0.0068, 0.0022), None),
        (
            (0.00875, 0.0035, 0.01175),
            (0.0, 0.0159, 0.02835),
            (-0.5235987755982988, 0.0, 3.141592653589793)
            if right
            else (0.5235987755982988, 0.0, 0.0),
        ),
        ((0.00875, 0.0076, 0.00925), (0.0, 0.00758, 0.04525), None),
    )
    for size, position, euler in boxes:
        attributes = {
            "type": "box",
            "size": vector(size),
            "pos": vector(position),
        }
        if euler is not None:
            attributes["euler"] = vector(euler)
        ET.SubElement(
            body, "geom", {"class": "franka_visual", **attributes}
        )
        ET.SubElement(
            body, "geom", {"class": "franka_collision", **attributes}
        )


def add_arm(
    world: ET.Element,
    prefix: str,
    mount: tuple[float, float, float],
    site_name: str,
) -> None:
    link = ET.SubElement(
        world,
        "body",
        {"name": f"{prefix}_link0", "pos": vector(mount), "childclass": "franka"},
    )
    add_inertial(link, ARM_INERTIALS[0])
    add_mesh_geometries(link, "fer_link0")

    for joint_index, (origin, euler) in enumerate(ARM_ORIGINS, start=1):
        link = ET.SubElement(
            link,
            "body",
            {
                "name": f"{prefix}_link{joint_index}",
                "pos": vector(origin),
                "euler": vector(euler),
            },
        )
        lower, upper = JOINT_LIMITS[joint_index - 1]
        ET.SubElement(
            link,
            "joint",
            {
                "name": f"{prefix}_joint{joint_index}",
                "type": "hinge",
                "axis": "0 0 1",
                "range": vector((lower, upper)),
                "damping": "0.003",
                "frictionloss": "0.2",
            },
        )
        add_inertial(link, ARM_INERTIALS[joint_index])
        add_mesh_geometries(link, f"fer_link{joint_index}")

    flange = ET.SubElement(
        link, "body", {"name": f"{prefix}_link8", "pos": "0 0 0.107"}
    )
    hand = ET.SubElement(
        flange,
        "body",
        {"name": f"{prefix}_hand", "euler": "0 0 -0.7853981633974483"},
    )
    add_inertial(hand, HAND_INERTIAL)
    add_mesh_geometries(hand, "fer_hand")
    tcp = ET.SubElement(
        hand,
        "body",
        {"name": f"{prefix}_hand_tcp", "pos": f"0 0 {HAND_TCP_OFFSET_M}"},
    )
    ET.SubElement(tcp, "site", {"name": site_name, "size": "0.01"})

    for right in (False, True):
        side = "right" if right else "left"
        attributes = {
            "name": f"{prefix}_{side}finger",
            "pos": "0 0 0.0584",
        }
        if right:
            attributes["euler"] = "0 0 3.141592653589793"
        finger = ET.SubElement(hand, "body", attributes)
        add_inertial(finger, FINGER_INERTIAL)
        ET.SubElement(
            finger,
            "joint",
            {
                "name": f"{prefix}_finger_joint{2 if right else 1}",
                "class": "franka_finger",
                "type": "slide",
                "axis": "0 1 0",
                "range": "0 0.04",
            },
        )
        add_finger_collisions(finger, right)


def add_free_object(
    world: ET.Element,
    name: str,
    position: tuple[float, float, float],
    geoms: tuple[dict[str, str], ...],
) -> None:
    body = ET.SubElement(world, "body", {"name": name, "pos": vector(position)})
    ET.SubElement(body, "freejoint")
    ET.SubElement(
        body,
        "inertial",
        {"pos": "0 0 0", "mass": "0.1", "diaginertia": "0.002 0.002 0.002"},
    )
    for attributes in geoms:
        ET.SubElement(body, "geom", attributes)


def add_single_attachment(world: ET.Element) -> None:
    add_free_object(
        world,
        "object",
        (0.43, 0.0, 1.1),
        ({"name": "pot_wall_back", "type": "box", "size": "0.02 0.025 0.1", "rgba": "0.75 0.75 0.78 1"},),
    )


def add_dual_attachment(world: ET.Element) -> None:
    tray_geoms = (
        {"name": "pot_bottom", "type": "box", "size": "0.08 0.125 0.01", "pos": "0 0 0", "rgba": "0.65 0.65 0.68 1"},
        {"name": "pot_wall_front", "type": "box", "size": "0.08 0.005 0.03", "pos": "0 0.12 0.04", "rgba": "0.65 0.65 0.68 1"},
        {"name": "pot_wall_back", "type": "box", "size": "0.08 0.005 0.03", "pos": "0 -0.12 0.04", "rgba": "0.65 0.65 0.68 1"},
        {"name": "pot_wall_left", "type": "box", "size": "0.005 0.125 0.03", "pos": "-0.075 0 0.04", "rgba": "0.65 0.65 0.68 1"},
        {"name": "pot_wall_right", "type": "box", "size": "0.005 0.125 0.03", "pos": "0.075 0 0.04", "rgba": "0.65 0.65 0.68 1"},
    )
    add_free_object(world, "object", (0.45, -0.325, 0.9), tray_geoms)


def build_mjcf(dual: bool) -> bytes:
    root = ET.Element("mujoco", {"model": "patacon_fer_dual" if dual else "patacon_fer_single"})
    root.append(
        ET.Comment(
            f" Generated from Franka Robotics franka_description {SOURCE_RELEASE}; "
            "see resources/franka/prepare_description.py. "
        )
    )
    ET.SubElement(
        root,
        "compiler",
        {"angle": "radian", "autolimits": "true", "balanceinertia": "true"},
    )
    ET.SubElement(root, "option", {"timestep": "0.002"})

    asset = ET.SubElement(root, "asset")
    for link_index in range(8):
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"fer_link{link_index}_visual", "file": f"../../../franka_description/meshes/robots/fer/collision/link{link_index}.stl"},
        )
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"fer_link{link_index}_collision", "file": f"../../../franka_description/meshes/robots/fer/collision/link{link_index}.stl"},
        )
    ET.SubElement(
        asset,
        "mesh",
        {"name": "fer_hand_visual", "file": "../../../franka_description/meshes/robot_ee/franka_hand_white/collision/hand.stl"},
    )
    ET.SubElement(
        asset,
        "mesh",
        {"name": "fer_hand_collision", "file": "../../../franka_description/meshes/robot_ee/franka_hand_white/collision/hand.stl"},
    )
    ET.SubElement(
        asset,
        "texture",
        {"name": "floor_grid_tex", "type": "2d", "builtin": "checker", "width": "1024", "height": "1024", "rgb1": "0.48 0.48 0.48", "rgb2": "0.48 0.48 0.48"},
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "floor_grid_mat", "texture": "floor_grid_tex", "texrepeat": "6 6", "texuniform": "true", "reflectance": "0"},
    )

    defaults = ET.SubElement(root, "default")
    franka = ET.SubElement(defaults, "default", {"class": "franka"})
    visual = ET.SubElement(franka, "default", {"class": "franka_visual"})
    ET.SubElement(
        visual,
        "geom",
        {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "0", "rgba": "0.92 0.92 0.95 1"},
    )
    collision = ET.SubElement(franka, "default", {"class": "franka_collision"})
    ET.SubElement(
        collision,
        "geom",
        {"contype": "1", "conaffinity": "1", "group": "3", "rgba": "0.5 0.6 0.7 0.25", "friction": "1 0.005 0.0001"},
    )
    finger = ET.SubElement(franka, "default", {"class": "franka_finger"})
    ET.SubElement(finger, "joint", {"damping": "0.3", "limited": "true"})

    visual_settings = ET.SubElement(root, "visual")
    ET.SubElement(
        visual_settings,
        "headlight",
        {"ambient": "0.65 0.65 0.65", "diffuse": "0.45 0.45 0.45", "specular": "0.05 0.05 0.05"},
    )
    ET.SubElement(visual_settings, "rgba", {"haze": "0.72 0.72 0.72 1"})

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        {"name": "main_light", "pos": "0 0 5", "dir": "0 0 -1", "diffuse": "0.85 0.85 0.85", "directional": "true"},
    )
    ET.SubElement(
        world,
        "geom",
        {"name": "floor", "type": "plane", "size": "0 0 0.05", "material": "floor_grid_mat", "friction": "1 0.005 0.0001"},
    )
    if dual:
        add_dual_attachment(world)
        add_arm(world, "fer0", DUAL_MOUNTS[0], "end_effector")
        add_arm(world, "fer1", DUAL_MOUNTS[1], "end_effector1")
    else:
        add_single_attachment(world)
        add_arm(world, "fer0", SINGLE_MOUNT, "end_effector")

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def metadata(single: bytes, dual: bytes, single_mjcf: bytes, dual_mjcf: bytes) -> bytes:
    sources = {
        relative_path: sha256_file(DESCRIPTION_DIR / relative_path)
        for relative_path in SOURCE_FILES + MESH_FILES
    }
    collision_header = REPOSITORY_DIR / "src" / "robots" / "franka_fer.cuh"
    document = {
        "source": {
            "repository": SOURCE_REPOSITORY,
            "release": SOURCE_RELEASE,
            "commit": SOURCE_COMMIT,
            "robot_model": "fer",
            "end_effector": "franka_hand",
            "files_sha256": sources,
        },
        "patacon_contract": {
            "single_planning_dof": 7,
            "dual_planning_dof": 14,
            "joint_order_per_arm": [f"joint{index}" for index in range(1, 8)],
            "joint_limits_rad": [list(limit) for limit in JOINT_LIMITS],
            "single_mount_xyz_m": list(SINGLE_MOUNT),
            "dual_mounts_xyz_m": [list(mount) for mount in DUAL_MOUNTS],
            "hand_tcp_offset_m": HAND_TCP_OFFSET_M,
            "fine_spheres_per_arm": 59,
            "approximate_spheres_per_arm": 11,
        },
        "generated": {
            "resources/franka/fer_single.urdf": sha256_bytes(single),
            "resources/franka/fer_dual.urdf": sha256_bytes(dual),
            "resources/franka/mujoco/fer_single.xml": sha256_bytes(single_mjcf),
            "resources/franka/mujoco/fer_dual.xml": sha256_bytes(dual_mjcf),
            "src/robots/franka_fer.cuh": sha256_file(collision_header),
        },
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def write_or_check(path: Path, content: bytes, check: bool) -> None:
    if check:
        if not path.is_file():
            raise RuntimeError(f"missing generated file: {path.relative_to(REPOSITORY_DIR)}")
        if path.read_bytes() != content:
            raise RuntimeError(
                f"generated file is stale: {path.relative_to(REPOSITORY_DIR)}"
            )
        return
    path.write_bytes(content)


def validate_mujoco_adapters() -> None:
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo is required to validate the Franka visualization adapters"
        ) from error
    for filename, expected_joints in (("fer_single.xml", 9), ("fer_dual.xml", 18)):
        path = RESOURCE_DIR / "mujoco" / filename
        model = mujoco.MjModel.from_xml_path(str(path))
        if model.njnt != expected_joints + 1:
            # The attached task object has one free joint.
            raise RuntimeError(
                f"{filename} has {model.njnt} joints, expected {expected_joints + 1}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify committed generated files"
    )
    args = parser.parse_args()

    validate_source()
    source = expand_official_xacro()
    single = build_urdf(source, (SINGLE_MOUNT,))
    dual = build_urdf(source, DUAL_MOUNTS)
    single_mjcf = build_mjcf(False)
    dual_mjcf = build_mjcf(True)
    validate_generated_urdf(single, 1)
    validate_generated_urdf(dual, 2)
    generated_metadata = metadata(single, dual, single_mjcf, dual_mjcf)

    write_or_check(SINGLE_URDF, single, args.check)
    write_or_check(DUAL_URDF, dual, args.check)
    SINGLE_MJCF.parent.mkdir(parents=True, exist_ok=True)
    write_or_check(SINGLE_MJCF, single_mjcf, args.check)
    write_or_check(DUAL_MJCF, dual_mjcf, args.check)
    write_or_check(METADATA_PATH, generated_metadata, args.check)
    validate_mujoco_adapters()
    action = "validated" if args.check else "generated"
    print(
        f"{action} official Franka FER {SOURCE_RELEASE} planning descriptions "
        f"at {SOURCE_COMMIT[:12]}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"Franka description error: {error}", file=sys.stderr)
        raise SystemExit(1)

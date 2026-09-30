"""Compose PATACON task scenes from the official FFW-SG2 MuJoCo model."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import Iterator
import xml.etree.ElementTree as ET


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
FFW_DESCRIPTION_DIR = REPOSITORY_DIR / "ai_worker" / "ffw_description"
DEFAULT_ROBOT = (
    FFW_DESCRIPTION_DIR
    / "mujoco"
    / "ffw_sg2"
    / "ffw_sg2.xml"
)
DEFAULT_OVERLAY_DIR = REPOSITORY_DIR / "resources" / "ffw_sg2" / "mujoco"

SCENE_LIFT = "lift"
SCENE_RACK_UPPER_TO_LOWER = "rack_upper_to_lower"
SCENE_OVERLAYS = {
    SCENE_LIFT: "lift_overlay.xml",
    SCENE_RACK_UPPER_TO_LOWER: "rack_upper_to_lower_overlay.xml",
}
SCENE_OUTPUTS = {
    SCENE_LIFT: "ffw_sg2_lift.xml",
    SCENE_RACK_UPPER_TO_LOWER: "ffw_sg2_rack_upper_to_lower.xml",
}

GRIPPER_SITES = (
    "gripper_l_rh_p12_rn_base",
    "gripper_r_rh_p12_rn_base",
)


def find_named_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise ValueError(f"official MJCF is missing body {name!r}")


def configure_fixed_patacon_base(root: ET.Element) -> None:
    base = find_named_body(root, "base_link")
    base.set("pos", "0 0 0")
    for child in list(base):
        if child.tag == "freejoint" or (
            child.tag == "joint" and child.get("type") == "free"
        ):
            base.remove(child)


def add_patacon_gripper_sites(root: ET.Element) -> None:
    for name in GRIPPER_SITES:
        body = find_named_body(root, name)
        for child in list(body):
            if child.tag == "site" and child.get("name") == name:
                body.remove(child)
        # The official gripper-base body is rotated pi around X. A local pi
        # rotation around Z preserves PATACON's historical attached-object
        # site frame (pi around Y in the arm-link frame).
        ET.SubElement(
            body,
            "site",
            {
                "name": name,
                "pos": "0 0 0",
                "quat": "0 0 0 1",
                "type": "sphere",
                "size": "0.001",
                "rgba": "1 0 0 0.8" if "_l_" in name else "0 0 1 0.8",
            },
        )


def replace_option(root: ET.Element, overlay: ET.Element) -> None:
    source = overlay.find("option")
    if source is None:
        return
    target = root.find("option")
    if target is None:
        target = ET.Element("option")
        root.insert(1, target)
    target.attrib.clear()
    target.attrib.update(source.attrib)


def configure_compiler(
    root: ET.Element,
    overlay: ET.Element,
    output: Path,
    mesh_directory: Path,
    overlay_directory: Path,
) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set(
        "meshdir",
        os.path.relpath(mesh_directory.resolve(), output.parent.resolve()),
    )
    compiler.set("autolimits", "true")
    overlay_compiler = overlay.find("compiler")
    if overlay_compiler is not None and overlay_compiler.get("texturedir"):
        texture_directory = (
            overlay_directory / overlay_compiler.get("texturedir", "")
        )
        compiler.set(
            "texturedir",
            os.path.relpath(texture_directory.resolve(), output.parent.resolve()),
        )


def insert_includes(root: ET.Element, overlay: ET.Element) -> None:
    includes = [deepcopy(element) for element in overlay.findall("include")]
    if not includes:
        return
    option = root.find("option")
    insert_at = list(root).index(option) + 1 if option is not None else 1
    for include in includes:
        root.insert(insert_at, include)
        insert_at += 1


def append_children(
    root: ET.Element,
    overlay: ET.Element,
    tag: str,
) -> None:
    source = overlay.find(tag)
    if source is None:
        return
    target = root.find(tag)
    if target is None:
        target = ET.SubElement(root, tag)
    for child in source:
        target.append(deepcopy(child))


def compose_scene(
    robot: Path,
    overlay_path: Path,
    output: Path,
    mesh_directory: Path = FFW_DESCRIPTION_DIR,
) -> None:
    robot = robot.resolve()
    overlay_path = overlay_path.resolve()
    output = output.resolve()
    if not robot.is_file():
        raise FileNotFoundError(f"official FFW-SG2 model not found: {robot}")
    if not overlay_path.is_file():
        raise FileNotFoundError(f"PATACON MuJoCo overlay not found: {overlay_path}")
    if not mesh_directory.is_dir():
        raise FileNotFoundError(
            f"FFW-SG2 description directory not found: {mesh_directory}"
        )

    tree = ET.parse(robot)
    root = tree.getroot()
    overlay = ET.parse(overlay_path).getroot()
    if root.tag != "mujoco" or overlay.tag != "mujoco":
        raise ValueError("robot and overlay inputs must have mujoco roots")

    root.set("model", overlay.get("model", output.stem))
    configure_compiler(
        root,
        overlay,
        output,
        mesh_directory,
        overlay_path.parent,
    )
    replace_option(root, overlay)
    insert_includes(root, overlay)
    append_children(root, overlay, "asset")
    append_children(root, overlay, "worldbody")
    append_children(root, overlay, "contact")
    configure_fixed_patacon_base(root)
    add_patacon_gripper_sites(root)

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def compose_named_scene(
    scene: str,
    output: Path,
    robot: Path = DEFAULT_ROBOT,
    overlay_directory: Path = DEFAULT_OVERLAY_DIR,
) -> None:
    try:
        overlay_name = SCENE_OVERLAYS[scene]
    except KeyError as error:
        choices = ", ".join(sorted(SCENE_OVERLAYS))
        raise ValueError(f"unknown FFW-SG2 scene {scene!r}; expected {choices}") from error
    compose_scene(robot, overlay_directory / overlay_name, output)


@contextmanager
def temporary_scene(
    scene: str,
    robot: Path = DEFAULT_ROBOT,
    overlay_directory: Path = DEFAULT_OVERLAY_DIR,
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="patacon_ffw_sg2_") as directory:
        output = Path(directory) / SCENE_OUTPUTS[scene]
        compose_named_scene(scene, output, robot, overlay_directory)
        yield output

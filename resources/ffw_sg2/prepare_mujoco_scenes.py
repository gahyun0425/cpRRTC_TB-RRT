#!/usr/bin/env python3
"""Compose PATACON task adapters around ROBOTIS' official FFW-SG2 MJCF."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]
DEFAULT_ROBOT = (
    REPOSITORY_DIR
    / "ai_worker"
    / "ffw_description"
    / "mujoco"
    / "ffw_sg2"
    / "ffw_sg2.xml"
)
DEFAULT_OVERLAY_DIR = RESOURCE_DIR / "mujoco"
DEFAULT_OUTPUT_DIR = REPOSITORY_DIR / "ffw_lift"

SCENES = {
    "lift_overlay.xml": "ffw_sg2_lift.xml",
    "rack_upper_to_lower_overlay.xml": "ffw_sg2_rack_upper_to_lower.xml",
}

GRIPPER_SITES = (
    "gripper_l_rh_p12_rn_base",
    "gripper_r_rh_p12_rn_base",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


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
        # The official gripper-base body is rotated pi around X.  A local pi
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


def configure_compiler(root: ET.Element, overlay: ET.Element) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("meshdir", "../ai_worker/ffw_description")
    compiler.set("autolimits", "true")
    overlay_compiler = overlay.find("compiler")
    if overlay_compiler is not None and overlay_compiler.get("texturedir"):
        compiler.set("texturedir", overlay_compiler.get("texturedir", ""))


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


def compose(robot: Path, overlay_path: Path, output: Path) -> None:
    tree = ET.parse(robot)
    root = tree.getroot()
    overlay = ET.parse(overlay_path).getroot()
    if root.tag != "mujoco" or overlay.tag != "mujoco":
        raise ValueError("robot and overlay inputs must have mujoco roots")

    root.set("model", overlay.get("model", output.stem))
    configure_compiler(root, overlay)
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


def main() -> None:
    args = parse_args()
    robot = args.robot.resolve()
    overlay_dir = args.overlay_dir.resolve()
    output_dir = args.output_dir.resolve()
    for overlay_name, output_name in SCENES.items():
        compose(robot, overlay_dir / overlay_name, output_dir / output_name)


if __name__ == "__main__":
    main()

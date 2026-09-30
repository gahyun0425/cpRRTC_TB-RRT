#!/usr/bin/env python3
"""Export PATACON task scenes composed from the official FFW-SG2 MJCF."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]
SCRIPTS_DIR = REPOSITORY_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ffw_sg2_mujoco_scene import (  # noqa: E402
    DEFAULT_OVERLAY_DIR,
    DEFAULT_ROBOT,
    SCENE_OUTPUTS,
    compose_named_scene,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = args.robot.resolve()
    overlay_dir = args.overlay_dir.resolve()
    output_dir = args.output_dir.resolve()
    for scene, output_name in SCENE_OUTPUTS.items():
        compose_named_scene(scene, output_dir / output_name, robot, overlay_dir)


if __name__ == "__main__":
    main()

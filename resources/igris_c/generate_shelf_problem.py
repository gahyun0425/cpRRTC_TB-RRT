#!/usr/bin/env python3
"""Generate the committed IGRIS-C shelf-task planner input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]
DEFAULT_CONTRACT = RESOURCE_DIR / "constraint_contract.json"
DEFAULT_OUTPUT = REPOSITORY_DIR / "scripts" / "igris_c_problems.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def generate(contract: dict[str, object]) -> bytes:
    constraints = contract["constraints"]
    shelf_task = contract["shelf_task"]
    feet_targets = [
        target["quaternion_wxyz"] + target["translation_xyz_m"]
        for target in constraints["feet"]["targets"]
    ]
    support_polygon = [
        component
        for vertex in constraints["center_of_mass"]["support_polygon_xy_m"]
        for component in vertex
    ]
    bimanual = constraints["bimanual"]["target"]
    result = {
        "schema_version": 1,
        "robot": "igris_c",
        "problems": {
            "igris_c_shelf_lift": [
                {
                    "valid": True,
                    "start": shelf_task["start_configuration"],
                    "goals": shelf_task["goal_configurations"],
                    "sphere": [],
                    "cylinder": [],
                    "box": [
                        {
                            "name": "shelf_0",
                            "position": [0.65, 0.0, 0.45],
                            "orientation_euler_xyz": [0.0, 0.0, 0.0],
                            "half_extents": [0.1925, 0.5, 0.0075],
                        },
                        {
                            "name": "shelf_1",
                            "position": [0.65, 0.0, 0.84],
                            "orientation_euler_xyz": [0.0, 0.0, 0.0],
                            "half_extents": [0.1925, 0.5, 0.0075],
                        },
                        {
                            "name": "shelf_2",
                            "position": [0.65, 0.0, 1.25],
                            "orientation_euler_xyz": [0.0, 0.0, 0.0],
                            "half_extents": [0.1925, 0.5, 0.0075],
                        },
                        {
                            "name": "shelf_back",
                            "position": [0.8, 0.0, 1.0],
                            "orientation_euler_xyz": [0.0, 0.0, 0.0],
                            "half_extents": [0.015, 0.5, 1.0],
                        },
                    ],
                    "constraints": {
                        "feet": {"target": feet_targets},
                        "com": {
                            "support_margin_m": constraints["center_of_mass"][
                                "support_margin_m"
                            ],
                            "payload_mass_kg": shelf_task["payload_mass_kg"],
                            "raw_support_polygon": constraints[
                                "center_of_mass"
                            ]["raw_support_polygon_xy_m"],
                            "support_polygon": support_polygon,
                        },
                        "bimanual": {
                            "target": bimanual["quaternion_wxyz"]
                            + bimanual["translation_xyz_m"]
                        },
                        "bimanual_axis": constraints["bimanual_axis"],
                        "tolerance_squared": 1.0e-6,
                    },
                    "task": {
                        "description": "Lift a box with both palm grasp frames and place it on the shelf",
                        "temporary_box_half_extents_m": [0.105, 0.17, 0.125],
                        "payload_mass_kg": shelf_task["payload_mass_kg"],
                    },
                }
            ]
        },
    }
    return json.dumps(result, indent=2).encode("utf-8") + b"\n"


def main() -> None:
    args = parse_args()
    content = generate(json.loads(args.contract.read_text(encoding="utf-8")))
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != content:
            raise SystemExit(f"ERROR: generated artifact is stale: {args.output}")
        print("PASS: IGRIS-C shelf problem is current")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

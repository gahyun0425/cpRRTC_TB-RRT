#!/usr/bin/env python3
"""Run a compact evaluate_mbm case and validate its versioned JSON output."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--planner-schema", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="franka_single")
    parser.add_argument("--run-name", default="result_contract")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        from jsonschema import Draft202012Validator, RefResolver
    except ImportError as error:
        raise RuntimeError(
            "jsonschema is required; run "
            "`python3 -m pip install -r requirements-test.txt`"
        ) from error

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.unlink(missing_ok=True)
    command = [
        str(arguments.evaluator),
        arguments.robot,
        arguments.run_name,
        "--max-problems",
        "1",
        "--save-json",
        str(arguments.output),
        "--no-print-path",
    ]
    completed = subprocess.run(
        command,
        cwd=arguments.repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=arguments.timeout,
        check=False,
    )
    sys.stdout.write(completed.stdout)
    if completed.returncode != 0:
        raise AssertionError(f"evaluate_mbm exited with {completed.returncode}")

    with arguments.schema.open("r", encoding="utf-8") as stream:
        schema = json.load(stream)
    with arguments.planner_schema.open("r", encoding="utf-8") as stream:
        planner_schema = json.load(stream)
    with arguments.output.open("r", encoding="utf-8") as stream:
        result = json.load(stream)

    Draft202012Validator.check_schema(schema)
    resolver = RefResolver(
        base_uri=arguments.schema.resolve().as_uri(),
        referrer=schema,
        store={
            planner_schema["$id"]: planner_schema,
            arguments.planner_schema.resolve().as_uri(): planner_schema,
        },
    )
    errors = list(Draft202012Validator(schema, resolver=resolver).iter_errors(result))
    if errors:
        details = "\n".join(f"  - {error.message}" for error in errors[:10])
        raise AssertionError(f"problem-set schema validation failed:\n{details}")

    if result["format"] != "PATACON_problem_set_results_v1":
        raise AssertionError("problem-set format changed")
    if result["robot"] != arguments.robot:
        raise AssertionError("problem-set robot changed")
    if result["problems"] != 1 or result["runs"] != 1:
        raise AssertionError("compact problem-set case count changed")
    if result["solved_runs"] != 1 or not result["results"][0]["solved"]:
        raise AssertionError("compact problem-set regression was not solved")
    cost = result["results"][0]["cost"]
    if not isinstance(cost, (int, float)) or not math.isfinite(cost):
        raise AssertionError("compact problem-set result cost is invalid")

    print("problem-set result contract passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError, OSError, ValueError) as error:
        print(f"problem-set regression failed: {error}", file=sys.stderr)
        raise SystemExit(1)

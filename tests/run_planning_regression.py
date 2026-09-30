#!/usr/bin/env python3
"""Run one deterministic planning case and validate its result contract."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--problem-index", type=int, default=1)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--aorrtc-time", type=float)
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def reject_nonstandard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_nonstandard_number)


def validate_schema(instance: Any, schema: Any, schema_path: Path) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:
        raise RuntimeError(
            "jsonschema is required; run "
            "`python3 -m pip install -r requirements-test.txt`"
        ) from error

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return

    lines = [f"{schema_path}: result does not satisfy the frozen schema"]
    for error in errors[:10]:
        location = "/".join(str(part) for part in error.absolute_path)
        lines.append(f"  {location or '<root>'}: {error.message}")
        if not error.absolute_path and error.context:
            for detail in error.context[:4]:
                nested = "/".join(
                    str(part) for part in detail.absolute_path
                )
                lines.append(
                    f"    {nested or '<root>'}: {detail.message}"
                )
    raise AssertionError("\n".join(lines))


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def configuration_is_valid(configuration: Any, dimension: int) -> bool:
    return (
        isinstance(configuration, list)
        and len(configuration) == dimension
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in configuration
        )
    )


def max_abs_difference(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def validate_single_result(
    result: dict[str, Any],
    *,
    robot: str,
    dimension: int,
    seed: int,
    expected_format: str,
) -> None:
    ensure(result["format"] == expected_format, "unexpected result format")
    ensure(result["robot"] == robot, "result robot does not match the case")
    ensure(result["dimension"] == dimension, "compiled dimension changed")
    ensure(result["seed"] == seed, "planner seed changed")
    ensure(result["settings"]["random_seed"] == seed, "settings seed changed")
    ensure(result["solved"] is True, "regression problem was not solved")
    ensure(
        isinstance(result["cost"], (int, float))
        and math.isfinite(result["cost"])
        and result["cost"] >= 0,
        "solved result has an invalid cost",
    )
    ensure(
        len(result["joint_names"]) == dimension,
        "joint_names length differs from dimension",
    )
    ensure(
        configuration_is_valid(result["start"], dimension),
        "start configuration has an invalid dimension or value",
    )
    ensure(result["goals"], "result contains no goal")
    ensure(
        all(configuration_is_valid(goal, dimension) for goal in result["goals"]),
        "goal configuration has an invalid dimension or value",
    )

    for key in ("path", "path_start_to_goal"):
        ensure(result[key], f"{key} is empty for a solved result")
        ensure(
            all(configuration_is_valid(q, dimension) for q in result[key]),
            f"{key} contains an invalid configuration",
        )

    ordered_path = result["path_start_to_goal"]
    ensure(
        result["path_waypoint_count"] == len(result["path"]),
        "path_waypoint_count does not match path",
    )
    ensure(
        result["path_edge_count"] == max(0, len(result["path"]) - 1),
        "path_edge_count does not match path",
    )
    ensure(
        len(ordered_path) == len(result["path"]),
        "ordered and raw paths have different lengths",
    )
    tolerance = 5.0e-3
    ensure(
        max_abs_difference(ordered_path[0], result["start"]) <= tolerance,
        "path_start_to_goal does not begin at start",
    )
    ensure(
        min(
            max_abs_difference(ordered_path[-1], goal)
            for goal in result["goals"]
        )
        <= tolerance,
        "path_start_to_goal does not end at a goal",
    )


def main() -> int:
    arguments = parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.unlink(missing_ok=True)

    command = [
        str(arguments.planner),
        arguments.robot,
        arguments.problem,
        str(arguments.problem_index),
        "--seed",
        str(arguments.seed),
        "--runs",
        str(arguments.runs),
        "--save-json",
        str(arguments.output),
        "--no-print-path",
    ]
    expected_format = "PATACON_result_v1"
    if arguments.aorrtc_time is not None:
        command.extend(["--aorrtc", "--time", str(arguments.aorrtc_time)])
        expected_format = "AORRTC_result_v1"

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
    ensure(completed.returncode == 0, f"planner exited with {completed.returncode}")
    ensure(arguments.output.is_file(), "planner did not create result JSON")

    result = load_json(arguments.output)
    schema = load_json(arguments.schema)
    validate_schema(result, schema, arguments.schema)

    if arguments.runs == 1:
        validate_single_result(
            result,
            robot=arguments.robot,
            dimension=arguments.dimension,
            seed=arguments.seed,
            expected_format=expected_format,
        )
    else:
        wrapper_format = expected_format.replace("_result_v1", "_run_results_v1")
        ensure(result["format"] == wrapper_format, "unexpected run bundle format")
        ensure(result["runs"] == arguments.runs, "run count changed")
        ensure(len(result["results"]) == arguments.runs, "missing run result")
        for run_index, run_result in enumerate(result["results"], start=1):
            validate_single_result(
                run_result,
                robot=arguments.robot,
                dimension=arguments.dimension,
                seed=arguments.seed + run_index - 1,
                expected_format=expected_format,
            )
            ensure(run_result["run_idx"] == run_index, "run_idx changed")
            ensure(
                run_result["run_count"] == arguments.runs,
                "nested run_count changed",
            )

    print(
        f"planning regression passed: {arguments.robot} / "
        f"{arguments.problem} #{arguments.problem_index}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError, ValueError) as error:
        print(f"planning regression failed: {error}", file=sys.stderr)
        raise SystemExit(1)

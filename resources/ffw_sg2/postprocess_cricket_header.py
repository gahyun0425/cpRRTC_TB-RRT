#!/usr/bin/env python3
"""Validate and namespace Cricket's generated FFW-SG2 CUDA header."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]
DEFAULT_OUTPUT = REPOSITORY_DIR / "src" / "robots" / "ffw_sg2.cuh"

CRICKET_COMMIT = "98582c35d81c6ed0d8c4badb7fdf78327523524c"
EXPECTED_RAW_SHA256 = (
    "1d07485ad5763a55bcbc5a3baaf5ffc81a74cdb130a58dc098448b582bd31950"
)
EXPECTED_FINE_SPHERES = 124
EXPECTED_APPROX_SPHERES = 27
EXPECTED_JOINTS = 16  # Pinocchio universe joint plus 15 active joints.
EXPECTED_TRANSFORM_SLOTS = 2
EXPECTED_RAW_STRIDE_OCCURRENCES = 5
EXPECTED_UNSIGNED_SENTINEL_OCCURRENCES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def macro_value(source: str, name: str) -> int:
    match = re.search(rf"^#define {re.escape(name)} (\d+)$", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing generated macro {name}")
    return int(match.group(1))


def integer_array(source: str, name: str) -> list[int]:
    match = re.search(
        rf"int {re.escape(name)}\[(\d+)\] = \{{(.*?)\}};",
        source,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"missing generated array {name}")
    declared_size = int(match.group(1))
    values = [int(value) for value in re.findall(r"-?\d+", match.group(2))]
    if len(values) != declared_size:
        raise ValueError(
            f"{name}: declared {declared_size} entries but parsed {len(values)}"
        )
    return values


def validate_raw_header(source: str) -> None:
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_hash != EXPECTED_RAW_SHA256:
        raise ValueError(
            "unexpected raw Cricket header hash: "
            f"expected {EXPECTED_RAW_SHA256}, got {source_hash}"
        )

    expected_macros = {
        "FFWSG2_SPHERE_COUNT": EXPECTED_FINE_SPHERES,
        "FFWSG2_APPROX_SPHERE_COUNT": EXPECTED_APPROX_SPHERES,
        "FFWSG2_JOINT_COUNT": EXPECTED_JOINTS,
        "FFWSG2_APPROX_JOINT_COUNT": EXPECTED_JOINTS,
    }
    for name, expected in expected_macros.items():
        actual = macro_value(source, name)
        if actual != expected:
            raise ValueError(f"{name}: expected {expected}, got {actual}")

    for name in ("ffwsg2_T_memory_idx", "ffwsg2_approx_T_memory_idx"):
        slots = max(integer_array(source, name)) + 1
        if slots != EXPECTED_TRANSFORM_SLOTS:
            raise ValueError(
                f"{name}: expected {EXPECTED_TRANSFORM_SLOTS} slots, got {slots}"
            )

    for name in ("ffwsg2_sphere_to_joint", "ffwsg2_approx_sphere_to_joint"):
        stride = max(integer_array(source, name)) + 1
        if stride != EXPECTED_JOINTS:
            raise ValueError(f"{name}: expected flag stride {EXPECTED_JOINTS}, got {stride}")

    stride_occurrences = len(re.findall(r"20\s*\*\s*batch_ind", source))
    if stride_occurrences != EXPECTED_RAW_STRIDE_OCCURRENCES:
        raise ValueError(
            "Cricket template stride pattern changed: expected "
            f"{EXPECTED_RAW_STRIDE_OCCURRENCES}, got {stride_occurrences}"
        )


def deduplicate_generated_macros(source: str) -> str:
    generated_macros = {
        "FFW_SG2_FIXED",
        "FFW_SG2_X_PRISM",
        "FFW_SG2_Y_PRISM",
        "FFW_SG2_Z_PRISM",
        "FFW_SG2_X_ROT",
        "FFW_SG2_Y_ROT",
        "FFW_SG2_Z_ROT",
        "FFW_SG2_BATCH_SIZE",
    }
    seen: set[str] = set()
    result: list[str] = []
    for line in source.splitlines():
        match = re.match(r"#define (\S+)", line)
        if match is not None and match.group(1) in generated_macros:
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
        result.append(line)
    return "\n".join(result) + "\n"


def postprocess(source: str) -> str:
    validate_raw_header(source)

    source = source.replace("Ffwsg2", "FfwSg2")
    source = source.replace("FFWSG2", "FFW_SG2")
    source = source.replace("ffwsg2", "ffw_sg2")

    token_replacements = {
        "BATCH_SIZE": "FFW_SG2_BATCH_SIZE",
        "FIXED": "FFW_SG2_FIXED",
        "X_PRISM": "FFW_SG2_X_PRISM",
        "Y_PRISM": "FFW_SG2_Y_PRISM",
        "Z_PRISM": "FFW_SG2_Z_PRISM",
        "X_ROT": "FFW_SG2_X_ROT",
        "Y_ROT": "FFW_SG2_Y_ROT",
        "Z_ROT": "FFW_SG2_Z_ROT",
    }
    for old, new in token_replacements.items():
        source = re.sub(rf"\b{old}\b", new, source)

    source, replacements = re.subn(
        r"20\s*\*\s*batch_ind",
        "FFW_SG2_JOINT_FLAG_STRIDE * batch_ind",
        source,
    )
    if replacements != EXPECTED_RAW_STRIDE_OCCURRENCES:
        raise ValueError(f"replaced {replacements} hard-coded stride expressions")

    source, sentinel_replacements = re.subn(
        r"\b18446744073709551615\b",
        "-1",
        source,
    )
    if sentinel_replacements != EXPECTED_UNSIGNED_SENTINEL_OCCURRENCES:
        raise ValueError(
            "Cricket sentinel pattern changed: expected "
            f"{EXPECTED_UNSIGNED_SENTINEL_OCCURRENCES}, got {sentinel_replacements}"
        )

    source = deduplicate_generated_macros(source)
    marker = "#define FFW_SG2_BATCH_SIZE 16\n"
    if source.count(marker) != 1:
        raise ValueError("expected one FFW_SG2_BATCH_SIZE definition")
    source = source.replace(
        marker,
        marker
        + f"#define FFW_SG2_JOINT_FLAG_STRIDE {EXPECTED_JOINTS}\n"
        + f"#define FFW_SG2_TRANSFORM_SLOTS {EXPECTED_TRANSFORM_SLOTS}\n",
        1,
    )

    if "20*batch_ind" in source or "Ffwsg2" in source or "FFWSG2" in source:
        raise ValueError("postprocessed header still contains an obsolete token")

    provenance = (
        "#pragma once\n\n"
        "// Generated by CoMMALab/cricket gpu-cc-early-exit at commit\n"
        f"// {CRICKET_COMMIT}, then validated and namespaced by\n"
        "// resources/ffw_sg2/postprocess_cricket_header.py.\n\n"
    )
    return provenance + source.lstrip()


def main() -> None:
    args = parse_args()
    source = args.input.resolve().read_text(encoding="utf-8")
    result = postprocess(source)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()

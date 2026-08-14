#!/usr/bin/env python3
"""Validate and namespace Cricket's generated 8-DoF FFW-SG2 CUDA header."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


RESOURCE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = RESOURCE_DIR.parents[1]
DEFAULT_OUTPUT = REPOSITORY_DIR / "src" / "robots" / "ffw_sg2_single.cuh"

CRICKET_COMMIT = "98582c35d81c6ed0d8c4badb7fdf78327523524c"
EXPECTED_RAW_SHA256 = (
    "96be55d0823bd237fe887e966c57216e36de9a29c078a36e84dfe0b845233244"
)
EXPECTED_FINE_SPHERES = 124
EXPECTED_APPROX_SPHERES = 27
EXPECTED_JOINTS = 9  # Pinocchio universe joint plus 8 active joints.
EXPECTED_TRANSFORM_SLOTS = 1
EXPECTED_RAW_STRIDE_OCCURRENCES = 5
EXPECTED_UNSIGNED_SENTINEL_OCCURRENCES = 2

RAW_TYPE = "Ffwsg2single"
TYPE = "FfwSg2Single"
RAW_MACRO_PREFIX = "FFWSG2SINGLE"
MACRO_PREFIX = "FFW_SG2_SINGLE"
RAW_SYMBOL_PREFIX = "ffwsg2single"
SYMBOL_PREFIX = "ffw_sg2_single"


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
        f"{RAW_MACRO_PREFIX}_SPHERE_COUNT": EXPECTED_FINE_SPHERES,
        f"{RAW_MACRO_PREFIX}_APPROX_SPHERE_COUNT": EXPECTED_APPROX_SPHERES,
        f"{RAW_MACRO_PREFIX}_JOINT_COUNT": EXPECTED_JOINTS,
        f"{RAW_MACRO_PREFIX}_APPROX_JOINT_COUNT": EXPECTED_JOINTS,
    }
    for name, expected in expected_macros.items():
        actual = macro_value(source, name)
        if actual != expected:
            raise ValueError(f"{name}: expected {expected}, got {actual}")

    for suffix in ("T_memory_idx", "approx_T_memory_idx"):
        name = f"{RAW_SYMBOL_PREFIX}_{suffix}"
        slots = max(integer_array(source, name)) + 1
        if slots != EXPECTED_TRANSFORM_SLOTS:
            raise ValueError(
                f"{name}: expected {EXPECTED_TRANSFORM_SLOTS} slots, got {slots}"
            )

    for suffix in ("sphere_to_joint", "approx_sphere_to_joint"):
        name = f"{RAW_SYMBOL_PREFIX}_{suffix}"
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
        f"{MACRO_PREFIX}_FIXED",
        f"{MACRO_PREFIX}_X_PRISM",
        f"{MACRO_PREFIX}_Y_PRISM",
        f"{MACRO_PREFIX}_Z_PRISM",
        f"{MACRO_PREFIX}_X_ROT",
        f"{MACRO_PREFIX}_Y_ROT",
        f"{MACRO_PREFIX}_Z_ROT",
        f"{MACRO_PREFIX}_BATCH_SIZE",
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

    source = source.replace(RAW_TYPE, TYPE)
    source = source.replace(RAW_MACRO_PREFIX, MACRO_PREFIX)
    source = source.replace(RAW_SYMBOL_PREFIX, SYMBOL_PREFIX)

    token_replacements = {
        "BATCH_SIZE": f"{MACRO_PREFIX}_BATCH_SIZE",
        "FIXED": f"{MACRO_PREFIX}_FIXED",
        "X_PRISM": f"{MACRO_PREFIX}_X_PRISM",
        "Y_PRISM": f"{MACRO_PREFIX}_Y_PRISM",
        "Z_PRISM": f"{MACRO_PREFIX}_Z_PRISM",
        "X_ROT": f"{MACRO_PREFIX}_X_ROT",
        "Y_ROT": f"{MACRO_PREFIX}_Y_ROT",
        "Z_ROT": f"{MACRO_PREFIX}_Z_ROT",
    }
    for old, new in token_replacements.items():
        source = re.sub(rf"\b{old}\b", new, source)

    source, replacements = re.subn(
        r"20\s*\*\s*batch_ind",
        f"{MACRO_PREFIX}_JOINT_FLAG_STRIDE * batch_ind",
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
    marker = f"#define {MACRO_PREFIX}_BATCH_SIZE 16\n"
    if source.count(marker) != 1:
        raise ValueError(f"expected one {MACRO_PREFIX}_BATCH_SIZE definition")
    source = source.replace(
        marker,
        marker
        + f"#define {MACRO_PREFIX}_JOINT_FLAG_STRIDE {EXPECTED_JOINTS}\n"
        + f"#define {MACRO_PREFIX}_TRANSFORM_SLOTS {EXPECTED_TRANSFORM_SLOTS}\n",
        1,
    )

    obsolete = (RAW_TYPE, RAW_MACRO_PREFIX, RAW_SYMBOL_PREFIX, "20*batch_ind")
    if any(token in source for token in obsolete):
        raise ValueError("postprocessed header still contains an obsolete token")

    provenance = (
        "#pragma once\n\n"
        "// Generated by CoMMALab/cricket gpu-cc-early-exit at commit\n"
        f"// {CRICKET_COMMIT}, then validated and namespaced by\n"
        "// resources/ffw_sg2/postprocess_cricket_single_header.py.\n\n"
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

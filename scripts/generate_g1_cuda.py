#!/usr/bin/env python3
"""Generate scalar CUDA G1 collision and analytic constraint code from VAMP."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def function_body(source: str, signature: str) -> str:
    signature_index = source.index(signature)
    body_start = source.index("{", signature_index)
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start + 1:index]
    raise ValueError(f"unterminated function: {signature}")


def scalar_sphere_fk(source: str) -> tuple[str, list[float]]:
    body = function_body(source, "inline static void sphere_fk")
    body = re.sub(
        r"\s*std::array<FloatVector<rake, 1>, 32> v;\s*"
        r"std::array<FloatVector<rake, 1>, 532> y;\s*",
        "\n",
        body,
        count=1,
    )
    body = body[:body.index("for (auto i = 0U; i < 133;")].rstrip()
    body = re.sub(r"\bcos\(", "cosf(", body)
    body = re.sub(r"\bsin\(", "sinf(", body)
    body = re.sub(r"^ {12}", "    ", body, flags=re.MULTILINE)

    radii: list[float | None] = [None] * 133
    for sphere_index, value in re.findall(
        r"y\[(\d+)\]\s*=\s*([-+0-9.eE]+);", body
    ):
        scalar_index = int(sphere_index)
        if scalar_index % 4 == 3:
            radii[scalar_index // 4] = float(value)
    if any(value is None for value in radii):
        missing = [index for index, value in enumerate(radii) if value is None]
        raise ValueError(f"missing G1 radii: {missing}")
    return body, [float(value) for value in radii]


def scalar_eef_fk(source: str) -> str:
    body = function_body(source, "inline static auto eefk")
    body = re.sub(
        r"\s*std::array<float, 68> v;\s*std::array<float, 48> y;\s*",
        "\n",
        body,
        count=1,
    )
    body = body[:body.index("return to_isometries<4>")].rstrip()
    body = re.sub(r"\bcos\(", "cosf(", body)
    body = re.sub(r"\bsin\(", "sinf(", body)
    return re.sub(r"^ {12}", "    ", body, flags=re.MULTILINE)


def scalar_com_fk(source: str) -> str:
    body = function_body(source, "inline static auto compute_com")
    body = re.sub(
        r"\s*std::array<FloatVector<rake, 1>, 221> v;\s*"
        r"std::array<FloatVector<rake, 1>, 108> y;\s*",
        "\n",
        body,
        count=1,
    )
    last_assignment = body.index("y[2] =")
    body = body[:body.index(";", last_assignment) + 1].rstrip()
    body = re.sub(r"\bcos\(", "cosf(", body)
    body = re.sub(r"\bsin\(", "sinf(", body)
    return re.sub(r"^ {12}", "    ", body, flags=re.MULTILINE)


def scalar_analytic_body(source: str, signature: str) -> str:
    """Convert one of VAMP's generated SIMD analytic functions to scalar CUDA."""
    body = function_body(source, signature)
    body = re.sub(
        r"\s*std::array<FloatVector<rake, 1>, \d+> v;\s*"
        r"(?:std::array<FloatVector<rake, 1>, \d+>|FloatVector<rake, \d+>) y;\s*",
        "\n",
        body,
        count=1,
    )
    for function in ("cos", "sin", "sqrt", "acos"):
        body = re.sub(rf"\b{function}\(", f"{function}f(", body)
    body = body.replace("for (size_t i", "for (int i")
    body = re.sub(r"^ {12}", "    ", body, flags=re.MULTILINE)
    return "\n".join(line.rstrip() for line in body.splitlines())


ANALYTIC_ASSIGNMENT_RE = re.compile(
    r"(?ms)^[ \t]*([vy]\[\d+\])\s*=\s*(.*?);"
)


def analytic_dependency_slice(
    body: str,
    output_loop: str,
    outputs: set[str],
    label: str,
) -> str:
    """Backward-slice straight-line generated arithmetic from selected outputs."""
    if output_loop not in body:
        raise ValueError(f"missing {label} analytic output loop")
    body = body[:body.rfind(output_loop)]

    assignments = list(ANALYTIC_ASSIGNMENT_RE.finditer(body))
    live = set(outputs)
    kept: list[str] = []
    for assignment in reversed(assignments):
        lhs = assignment.group(1)
        if lhs not in live:
            continue
        live.remove(lhs)
        live.update(re.findall(r"[vy]\[\d+\]", assignment.group(2)))
        kept.append(assignment.group(0).strip())

    if live:
        raise ValueError(f"unresolved {label} dependencies: {sorted(live)}")
    return "\n".join(reversed(kept))


def indexed_array_indices(body: str, array: str) -> set[int]:
    return {int(index) for index in re.findall(rf"{array}\[(\d+)\]", body)}


def remap_indexed_array(body: str, array: str, mapping: dict[int, int]) -> str:
    pattern = re.compile(rf"{array}\[(\d+)\]")

    def replace(match: re.Match[str]) -> str:
        old_index = int(match.group(1))
        if old_index not in mapping:
            raise ValueError(f"missing {array} index mapping for {old_index}")
        return f"{array}[{mapping[old_index]}]"

    return pattern.sub(replace, body)


def scalar_feet_position_body(source: str) -> tuple[str, int]:
    """Extract both feet's position error and 6x18 nonzero Jacobian."""
    body = scalar_analytic_body(source, "inline static auto tsr_error")
    source_rows = (12, 13, 14, 18, 19, 20)
    outputs = {
        *(f"y[{row * 35 + joint}]" for row in source_rows for joint in range(18)),
        *(f"y[{840 + row}]" for row in source_rows),
    }
    sliced = analytic_dependency_slice(
        body,
        "for (int i = 0; i < 864; i++)",
        outputs,
        "G1 feet position equality",
    )

    used_v = sorted(indexed_array_indices(sliced, "v"))
    sliced = remap_indexed_array(
        sliced,
        "v",
        {old_index: new_index for new_index, old_index in enumerate(used_v)},
    )

    input_mapping = {index: index for index in range(18)}
    input_mapping.update({index: 18 + index - 87 for index in range(87, 101)})
    input_mapping.update({index: 32 + index - 113 for index in range(113, 127)})
    sliced = remap_indexed_array(sliced, "x", input_mapping)

    output_mapping = {
        row * 35 + joint: compact_row * 18 + joint
        for compact_row, row in enumerate(source_rows)
        for joint in range(18)
    }
    output_mapping.update({
        840 + row: 108 + compact_row
        for compact_row, row in enumerate(source_rows)
    })
    sliced = remap_indexed_array(sliced, "y", output_mapping)

    if indexed_array_indices(sliced, "v") != set(range(len(used_v))):
        raise ValueError("invalid compact G1 feet temporary layout")
    if indexed_array_indices(sliced, "x") != set(range(46)):
        raise ValueError("invalid compact G1 feet input layout")
    if indexed_array_indices(sliced, "y") != set(range(114)):
        raise ValueError("invalid compact G1 feet output layout")
    return sliced, len(used_v)


def scalar_bimanual_position_body(source: str) -> tuple[str, int]:
    """Keep only the position-error/Jacobian dependency slice of VAMP's 6D TSR."""
    body = scalar_analytic_body(source, "inline static auto tsr_bimanual_error")
    active_joints = [*range(6), *range(18, 35)]
    outputs = {
        *(
            f"y[{row * 35 + joint}]"
            for row in range(3)
            for joint in active_joints
        ),
        *(f"y[{index}]" for index in range(210, 213)),
    }
    sliced = analytic_dependency_slice(
        body,
        "for (int i = 0; i < 216; i++)",
        outputs,
        "G1 bimanual position",
    )

    input_mapping = {
        old_index: new_index
        for new_index, old_index in enumerate(active_joints)
    }
    input_mapping.update({index: 23 + index - 35 for index in range(35, 42)})
    sliced = remap_indexed_array(sliced, "x", input_mapping)

    output_mapping = {
        row * 35 + joint: row * 23 + compact_joint
        for row in range(3)
        for compact_joint, joint in enumerate(active_joints)
    }
    output_mapping.update({index: 69 + index - 210 for index in range(210, 213)})
    sliced = remap_indexed_array(sliced, "y", output_mapping)

    used_v = sorted(indexed_array_indices(sliced, "v"))
    sliced = remap_indexed_array(
        sliced,
        "v",
        {old_index: new_index for new_index, old_index in enumerate(used_v)},
    )
    if indexed_array_indices(sliced, "v") != set(range(len(used_v))):
        raise ValueError("invalid compact G1 bimanual temporary layout")
    if indexed_array_indices(sliced, "x") != set(range(30)):
        raise ValueError("invalid G1 bimanual position input layout")
    if indexed_array_indices(sliced, "y") != set(range(72)):
        raise ValueError("invalid G1 bimanual position output layout")
    return sliced, len(used_v)


def self_collision_pairs(source: str) -> list[tuple[int, int]]:
    start = source.index("inline static auto fkcc_debug")
    end = source.index("inline static bool fkcc", start)
    pairs = [
        (int(first), int(second))
        for first, second in re.findall(
            r"output\.second\.emplace_back\((\d+), (\d+)\)",
            source[start:end],
        )
    ]
    if len(pairs) != 6888:
        raise ValueError(f"expected 6888 G1 self-collision pairs, got {len(pairs)}")
    return pairs


def generate(source: str) -> str:
    fk_body, radii = scalar_sphere_fk(source)
    pairs = self_collision_pairs(source)
    radius_rows = "\n".join(
        "    " + ", ".join(f"{value:.15g}f" for value in radii[index:index + 8]) + ","
        for index in range(0, len(radii), 8)
    )
    pair_rows = "\n".join(
        "    " + ", ".join(f"{{{a}, {b}}}" for a, b in pairs[index:index + 12]) + ","
        for index in range(0, len(pairs), 12)
    )

    return f'''#pragma once

// Generated from VAMP's g1_unitree.hh by scripts/generate_g1_cuda.py.
// The generated arithmetic is the exact scalar form of VAMP sphere_fk.

#include "src/collision/environment.hh"

namespace ppln::collision {{

constexpr int G1_DIM = 35;
constexpr int G1_SPHERE_COUNT = 133;
constexpr int G1_BATCH_SIZE = 16;
constexpr int G1_APPROX_SPHERE_COUNT = 1;
constexpr int G1_JOINT_FLAG_STRIDE = 1;
constexpr int G1_TRANSFORM_SLOTS = 1;
constexpr int G1_SELF_COLLISION_PAIR_COUNT = {len(pairs)};

__device__ __constant__ float g1_sphere_radii[G1_SPHERE_COUNT] = {{
{radius_rows}
}};

// Global read-only device memory is used because the complete pair table is
// too large to share CUDA constant memory with the other generated robots.
__device__ const unsigned char
g1_self_collision_pairs[G1_SELF_COLLISION_PAIR_COUNT][2] = {{
{pair_rows}
}};

__device__ __noinline__ void g1_sphere_fk_values(
    const float *x,
    float *y
) {{
    float v[32];
{fk_body}
}}

__device__ __forceinline__ void g1_sphere_fk(
    const float q[G1_DIM],
    float sphere_xyzr[G1_SPHERE_COUNT][4]
) {{
    float values[G1_SPHERE_COUNT * 4];
    g1_sphere_fk_values(q, values);
    for (int sphere = 0; sphere < G1_SPHERE_COUNT; ++sphere) {{
        for (int component = 0; component < 4; ++component) {{
            sphere_xyzr[sphere][component] = values[sphere * 4 + component];
        }}
    }}
}}

__device__ __forceinline__ bool g1_collision_free(
    const float q[G1_DIM],
    Environment<float> *environment
) {{
    float values[G1_SPHERE_COUNT * 4];
    g1_sphere_fk_values(q, values);

    for (int sphere = 0; sphere < G1_SPHERE_COUNT; ++sphere) {{
        if (sphere_environment_in_collision(
                environment,
                values[sphere * 4 + 0],
                values[sphere * 4 + 1],
                values[sphere * 4 + 2],
                values[sphere * 4 + 3])) {{
            return false;
        }}
    }}

    for (int pair_index = 0; pair_index < G1_SELF_COLLISION_PAIR_COUNT; ++pair_index) {{
        const int first = g1_self_collision_pairs[pair_index][0];
        const int second = g1_self_collision_pairs[pair_index][1];
        if (sphere_sphere_self_collision(
                values[first * 4 + 0], values[first * 4 + 1],
                values[first * 4 + 2], values[first * 4 + 3],
                values[second * 4 + 0], values[second * 4 + 1],
                values[second * 4 + 2], values[second * 4 + 3])) {{
            return false;
        }}
    }}
    return true;
}}

template <>
__device__ void fk<ppln::robots::G1>(
    const float *q,
    volatile float *sphere_pos,
    float *,
    const int tid
) {{
    const int lane = tid % 4;
    if (lane != 0) {{
        return;
    }}
    const int batch = tid / 4;
    float values[G1_SPHERE_COUNT * 4];
    g1_sphere_fk_values(q, values);
    for (int sphere = 0; sphere < G1_SPHERE_COUNT; ++sphere) {{
        for (int component = 0; component < 3; ++component) {{
            sphere_pos[sphere * G1_BATCH_SIZE * 3 + batch * 3 + component] =
                values[sphere * 4 + component];
        }}
    }}
}}

template <>
__device__ void fk_approx<ppln::robots::G1>(
    const float *q,
    volatile float *sphere_pos,
    float *,
    const int tid
) {{
    const int lane = tid % 4;
    const int batch = tid / 4;
    if (lane < 3) {{
        sphere_pos[batch * 3 + lane] = q[lane];
    }}
}}

template <>
__device__ bool env_collision_check_approx<ppln::robots::G1>(
    volatile float *sphere_pos,
    volatile int *joint_in_collision,
    Environment<float> *environment,
    const int tid
) {{
    const int lane = tid % 4;
    const int batch = tid / 4;
    if (lane != 0) {{
        return true;
    }}
    const bool collision = sphere_environment_in_collision(
        environment,
        sphere_pos[batch * 3 + 0],
        sphere_pos[batch * 3 + 1],
        sphere_pos[batch * 3 + 2],
        2.0f
    );
    if (collision) {{
        joint_in_collision[batch] = 1;
    }}
    return !collision;
}}

template <>
__device__ bool self_collision_check_approx<ppln::robots::G1>(
    volatile float *,
    volatile int *joint_in_collision,
    const int tid
) {{
    const int batch = tid / 4;
    if (tid % 4 == 0) {{
        joint_in_collision[batch] = 1;
    }}
    // Force the exact G1 pair test; this conservative broad phase cannot
    // produce a false negative.
    return false;
}}

template <>
__device__ bool env_collision_check<ppln::robots::G1>(
    volatile float *sphere_pos,
    volatile int *,
    Environment<float> *environment,
    const int tid
) {{
    const int lane = tid % 4;
    const int batch = tid / 4;
    bool collision = false;
    for (int sphere = lane; sphere < G1_SPHERE_COUNT; sphere += 4) {{
        collision |= sphere_environment_in_collision(
            environment,
            sphere_pos[sphere * G1_BATCH_SIZE * 3 + batch * 3 + 0],
            sphere_pos[sphere * G1_BATCH_SIZE * 3 + batch * 3 + 1],
            sphere_pos[sphere * G1_BATCH_SIZE * 3 + batch * 3 + 2],
            g1_sphere_radii[sphere]
        );
    }}
    return !collision;
}}

template <>
__device__ bool self_collision_check<ppln::robots::G1>(
    volatile float *sphere_pos,
    volatile int *,
    const int tid
) {{
    const int lane = tid % 4;
    const int batch = tid / 4;
    bool collision = false;
    for (int pair_index = lane; pair_index < G1_SELF_COLLISION_PAIR_COUNT; pair_index += 4) {{
        const int first = g1_self_collision_pairs[pair_index][0];
        const int second = g1_self_collision_pairs[pair_index][1];
        collision = collision || (sphere_sphere_self_collision(
            sphere_pos[first * G1_BATCH_SIZE * 3 + batch * 3 + 0],
            sphere_pos[first * G1_BATCH_SIZE * 3 + batch * 3 + 1],
            sphere_pos[first * G1_BATCH_SIZE * 3 + batch * 3 + 2],
            g1_sphere_radii[first],
            sphere_pos[second * G1_BATCH_SIZE * 3 + batch * 3 + 0],
            sphere_pos[second * G1_BATCH_SIZE * 3 + batch * 3 + 1],
            sphere_pos[second * G1_BATCH_SIZE * 3 + batch * 3 + 2],
            g1_sphere_radii[second]
        ) != 0.0f);
    }}
    return !collision;
}}

}}  // namespace ppln::collision
'''


def generate_kinematics(source: str) -> str:
    eef_body = scalar_eef_fk(source)
    com_position_body = scalar_com_fk(source)
    feet_body, feet_v_count = scalar_feet_position_body(source)
    com_body = scalar_analytic_body(source, "inline static auto compute_com")
    bimanual_body, bimanual_v_count = scalar_bimanual_position_body(source)
    return f'''#pragma once

// Generated from VAMP's g1_unitree.hh by scripts/generate_g1_cuda.py.
// FK, residuals, and Jacobians below preserve VAMP's generated scalar arithmetic.

namespace ppln::collision {{

__device__ __noinline__ void g1_end_effector_fk(
    const float *x,
    float *output
) {{
    float v[68];
    float y[48];
{eef_body}
    for (int index = 0; index < 48; ++index) {{
        output[index] = y[index];
    }}
}}

__device__ __noinline__ void g1_center_of_mass(
    const float *x,
    float *output
) {{
    float v[221];
    float y[3];
{com_position_body}
    output[0] = y[0];
    output[1] = y[1];
    output[2] = y[2];
}}

// Input layout is base/leg q[18], then left/right foot reference[7], target[7].
// Output is the nonzero position J[6][18], followed by position error[6].
// Stable SO(3) orientation residuals/Jacobians are assembled separately.
__device__ __noinline__ void g1_feet_position_error_analytic(
    const float *x,
    float *out
) {{
    float v[{feet_v_count}];
    float y[114];
{feet_body}
    for (int index = 0; index < 114; ++index) {{
        out[index] = y[index];
    }}
}}

// Output is CoM[3], followed by its row-major 3x35 analytic Jacobian.
__device__ __noinline__ void g1_com_kinematics_analytic(
    const float *x,
    float *out
) {{
    float v[221];
    float y[108];
{com_body}
}}

// Input layout is base/upper-body q[23], then target relative pose[7].
// Output is the nonzero position J[3][23], followed by position error[3].
__device__ __noinline__ void g1_bimanual_position_error_analytic(
    const float *x,
    float *out
) {{
    float v[{bimanual_v_count}];
    float y[72];
{bimanual_body}
    for (int index = 0; index < 72; ++index) {{
        out[index] = y[index];
    }}
}}

}}  // namespace ppln::collision
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vamp-header",
        type=Path,
        default=Path.home() / "gh_ws/vamp/src/impl/vamp/robots/g1_unitree.hh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/robots/g1_collision.cuh"),
    )
    parser.add_argument(
        "--kinematics-output",
        type=Path,
        default=Path("src/robots/g1_kinematics.cuh"),
    )
    arguments = parser.parse_args()
    output = generate(arguments.vamp_header.read_text())
    arguments.output.write_text(output)
    kinematics_output = generate_kinematics(arguments.vamp_header.read_text())
    arguments.kinematics_output.write_text(kinematics_output)
    print(f"generated {arguments.output}")
    print(f"generated {arguments.kinematics_output}")


if __name__ == "__main__":
    main()

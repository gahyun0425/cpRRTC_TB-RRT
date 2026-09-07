#pragma once

#include "src/planning/Robots.hh"
#include "src/robots/ffw_sg2_mobility.cuh"
#include "src/robots/ffw_sg2_constraint.cuh"

namespace ppln::collision {

#define FFW_SG2_MOBILITY_DIM 18
#define FFW_SG2_MOBILITY_RESIDUAL_DIM 8
#define FFW_SG2_MOBILITY_TANGENT_DIM 10
#define FFW_SG2_MOBILITY_TANGENT_BASIS_SIZE \
    (FFW_SG2_MOBILITY_DIM * FFW_SG2_MOBILITY_TANGENT_DIM)

__device__ __forceinline__ int ffw_sg2_mobility_constraint_dim() {
    return FFW_SG2_MOBILITY_RESIDUAL_DIM;
}

__device__ __forceinline__ int ffw_sg2_mobility_tangent_dim() {
    return FFW_SG2_MOBILITY_TANGENT_DIM;
}

__device__ __forceinline__ void ffw_sg2_mobility_constraint_residual(
    const float *q,
    float h[FFW_SG2_MOBILITY_RESIDUAL_DIM]
) {
    float h_dual_arm[FFW_SG2_MAX_RESIDUAL_DIM];
    ffw_sg2_constraint_residual(
        q + FFW_SG2_MOBILITY_BASE_DOF,
        true,
        h_dual_arm
    );
    for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
        h[row] = h_dual_arm[row];
    }
}

__device__ __forceinline__ bool ffw_sg2_mobility_constraint_jacobian(
    const float *q,
    float h[FFW_SG2_MOBILITY_RESIDUAL_DIM],
    float J[FFW_SG2_MOBILITY_RESIDUAL_DIM * FFW_SG2_MOBILITY_DIM]
) {
    float h_dual_arm[FFW_SG2_MAX_RESIDUAL_DIM];
    float J_dual_arm[FFW_SG2_MAX_RESIDUAL_DIM * FFW_SG2_DIM];
    if (
        !ffw_sg2_constraint_jacobian(
            q + FFW_SG2_MOBILITY_BASE_DOF,
            true,
            h_dual_arm,
            J_dual_arm
        )
    ) {
        return false;
    }

    for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
        h[row] = h_dual_arm[row];
        for (int col = 0; col < FFW_SG2_MOBILITY_DIM; ++col) {
            J[row * FFW_SG2_MOBILITY_DIM + col] = 0.0f;
        }
        for (int col = 0; col < FFW_SG2_DIM; ++col) {
            J[
                row * FFW_SG2_MOBILITY_DIM +
                FFW_SG2_MOBILITY_BASE_DOF +
                col
            ] = J_dual_arm[row * FFW_SG2_DIM + col];
        }
    }

    return true;
}

__device__ __forceinline__ float ffw_sg2_mobility_residual_norm(
    const float h[FFW_SG2_MOBILITY_RESIDUAL_DIM]
) {
    float n2 = 0.0f;
    for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
        n2 += h[row] * h[row];
    }
    return sqrtf(n2);
}

__device__ __forceinline__ void ffw_sg2_mobility_clamp(
    float q[FFW_SG2_MOBILITY_DIM]
) {
    for (int col = 0; col < FFW_SG2_MOBILITY_DIM; ++col) {
        const float lo = ppln::robots::FfwSg2Mobility::get_s_a(col);
        const float hi = lo + ppln::robots::FfwSg2Mobility::get_s_m(col);
        q[col] = fminf(fmaxf(q[col], lo), hi);
    }
}

__device__ __forceinline__ bool ffw_sg2_mobility_solve_residual_system(
    float A_in[
        FFW_SG2_MOBILITY_RESIDUAL_DIM * FFW_SG2_MOBILITY_RESIDUAL_DIM
    ],
    const float b_in[FFW_SG2_MOBILITY_RESIDUAL_DIM],
    float x[FFW_SG2_MOBILITY_RESIDUAL_DIM]
) {
    float aug[FFW_SG2_MOBILITY_RESIDUAL_DIM][
        FFW_SG2_MOBILITY_RESIDUAL_DIM + 1
    ];

    for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
        for (int col = 0; col < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++col) {
            aug[row][col] =
                A_in[row * FFW_SG2_MOBILITY_RESIDUAL_DIM + col];
        }
        aug[row][FFW_SG2_MOBILITY_RESIDUAL_DIM] = b_in[row];
    }

    for (int col = 0; col < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++col) {
        int pivot = col;
        float best = fabsf(aug[col][col]);
        for (int row = col + 1; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
            const float value = fabsf(aug[row][col]);
            if (value > best) {
                best = value;
                pivot = row;
            }
        }
        if (best < 1.0e-9f) {
            return false;
        }
        if (pivot != col) {
            for (
                int k = col;
                k <= FFW_SG2_MOBILITY_RESIDUAL_DIM;
                ++k
            ) {
                const float tmp = aug[col][k];
                aug[col][k] = aug[pivot][k];
                aug[pivot][k] = tmp;
            }
        }

        const float inv_pivot = 1.0f / aug[col][col];
        for (int k = col; k <= FFW_SG2_MOBILITY_RESIDUAL_DIM; ++k) {
            aug[col][k] *= inv_pivot;
        }

        for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
            if (row == col) {
                continue;
            }
            const float factor = aug[row][col];
            for (int k = col; k <= FFW_SG2_MOBILITY_RESIDUAL_DIM; ++k) {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }

    for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
        x[row] = aug[row][FFW_SG2_MOBILITY_RESIDUAL_DIM];
    }
    return true;
}

__device__ __forceinline__ bool ffw_sg2_mobility_tangent_basis_from_jacobian(
    const float J[FFW_SG2_MOBILITY_RESIDUAL_DIM * FFW_SG2_MOBILITY_DIM],
    float basis[FFW_SG2_MOBILITY_TANGENT_BASIS_SIZE]
) {
    float A[FFW_SG2_MOBILITY_RESIDUAL_DIM][FFW_SG2_MOBILITY_DIM];
    for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
        for (int col = 0; col < FFW_SG2_MOBILITY_DIM; ++col) {
            A[row][col] = J[row * FFW_SG2_MOBILITY_DIM + col];
        }
    }

    int pivot_cols[FFW_SG2_MOBILITY_RESIDUAL_DIM];
    int rank = 0;
    for (
        int col = 0;
        col < FFW_SG2_MOBILITY_DIM && rank < FFW_SG2_MOBILITY_RESIDUAL_DIM;
        ++col
    ) {
        int pivot = rank;
        float best = fabsf(A[rank][col]);
        for (int row = rank + 1; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
            const float value = fabsf(A[row][col]);
            if (value > best) {
                best = value;
                pivot = row;
            }
        }
        if (best < 1.0e-5f) {
            continue;
        }
        if (pivot != rank) {
            for (int c = 0; c < FFW_SG2_MOBILITY_DIM; ++c) {
                const float tmp = A[rank][c];
                A[rank][c] = A[pivot][c];
                A[pivot][c] = tmp;
            }
        }

        const float inv_pivot = 1.0f / A[rank][col];
        for (int c = col; c < FFW_SG2_MOBILITY_DIM; ++c) {
            A[rank][c] *= inv_pivot;
        }
        for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
            if (row == rank) {
                continue;
            }
            const float factor = A[row][col];
            for (int c = col; c < FFW_SG2_MOBILITY_DIM; ++c) {
                A[row][c] -= factor * A[rank][c];
            }
        }

        pivot_cols[rank] = col;
        ++rank;
    }

    bool is_pivot[FFW_SG2_MOBILITY_DIM];
    for (int col = 0; col < FFW_SG2_MOBILITY_DIM; ++col) {
        is_pivot[col] = false;
    }
    for (int row = 0; row < rank; ++row) {
        is_pivot[pivot_cols[row]] = true;
    }
    for (int index = 0; index < FFW_SG2_MOBILITY_TANGENT_BASIS_SIZE; ++index) {
        basis[index] = 0.0f;
    }

    int basis_col = 0;
    for (
        int free_col = 0;
        free_col < FFW_SG2_MOBILITY_DIM &&
        basis_col < FFW_SG2_MOBILITY_TANGENT_DIM;
        ++free_col
    ) {
        if (is_pivot[free_col]) {
            continue;
        }

        float v[FFW_SG2_MOBILITY_DIM];
        for (int index = 0; index < FFW_SG2_MOBILITY_DIM; ++index) {
            v[index] = 0.0f;
        }
        v[free_col] = 1.0f;

        for (int row = 0; row < rank; ++row) {
            v[pivot_cols[row]] = -A[row][free_col];
        }

        for (int previous = 0; previous < basis_col; ++previous) {
            float dot = 0.0f;
            for (int index = 0; index < FFW_SG2_MOBILITY_DIM; ++index) {
                dot +=
                    v[index] *
                    basis[index * FFW_SG2_MOBILITY_TANGENT_DIM + previous];
            }
            for (int index = 0; index < FFW_SG2_MOBILITY_DIM; ++index) {
                v[index] -=
                    dot *
                    basis[index * FFW_SG2_MOBILITY_TANGENT_DIM + previous];
            }
        }

        float norm2 = 0.0f;
        for (int index = 0; index < FFW_SG2_MOBILITY_DIM; ++index) {
            norm2 += v[index] * v[index];
        }
        if (norm2 < 1.0e-10f) {
            continue;
        }

        const float inv_norm = rsqrtf(norm2);
        for (int index = 0; index < FFW_SG2_MOBILITY_DIM; ++index) {
            basis[
                index * FFW_SG2_MOBILITY_TANGENT_DIM + basis_col
            ] = v[index] * inv_norm;
        }
        ++basis_col;
    }

    return
        rank == FFW_SG2_MOBILITY_RESIDUAL_DIM &&
        basis_col == FFW_SG2_MOBILITY_TANGENT_DIM;
}

__device__ __forceinline__ bool ffw_sg2_mobility_tangent_basis(
    const float *q,
    float basis[FFW_SG2_MOBILITY_TANGENT_BASIS_SIZE]
) {
    float h[FFW_SG2_MOBILITY_RESIDUAL_DIM];
    float J[FFW_SG2_MOBILITY_RESIDUAL_DIM * FFW_SG2_MOBILITY_DIM];
    if (!ffw_sg2_mobility_constraint_jacobian(q, h, J)) {
        return false;
    }
    return ffw_sg2_mobility_tangent_basis_from_jacobian(J, basis);
}

__device__ __forceinline__ bool ffw_sg2_mobility_task_correction(
    const float q[FFW_SG2_MOBILITY_DIM],
    float damping,
    float max_step,
    float correction[FFW_SG2_MOBILITY_DIM],
    float &task_error_norm
) {
    float h[FFW_SG2_MOBILITY_RESIDUAL_DIM];
    float J[FFW_SG2_MOBILITY_RESIDUAL_DIM * FFW_SG2_MOBILITY_DIM];
    if (!ffw_sg2_mobility_constraint_jacobian(q, h, J)) {
        return false;
    }

    task_error_norm = ffw_sg2_mobility_residual_norm(h);
    for (int col = 0; col < FFW_SG2_MOBILITY_DIM; ++col) {
        correction[col] = 0.0f;
    }
    if (task_error_norm <= 1.0e-12f) {
        return true;
    }

    float A[
        FFW_SG2_MOBILITY_RESIDUAL_DIM * FFW_SG2_MOBILITY_RESIDUAL_DIM
    ];
    for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
        for (int col = 0; col < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++col) {
            float value = 0.0f;
            for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                value +=
                    J[row * FFW_SG2_MOBILITY_DIM + joint] *
                    J[col * FFW_SG2_MOBILITY_DIM + joint];
            }
            A[row * FFW_SG2_MOBILITY_RESIDUAL_DIM + col] =
                value + (row == col ? damping : 0.0f);
        }
    }

    float y[FFW_SG2_MOBILITY_RESIDUAL_DIM];
    if (!ffw_sg2_mobility_solve_residual_system(A, h, y)) {
        return false;
    }

    float step_norm2 = 0.0f;
    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
        float value = 0.0f;
        for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
            value += J[row * FFW_SG2_MOBILITY_DIM + joint] * y[row];
        }
        correction[joint] = value;
        step_norm2 += value * value;
    }

    const float step_norm = sqrtf(step_norm2);
    if (max_step > 0.0f && step_norm > max_step) {
        const float scale = max_step / step_norm;
        for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
            correction[joint] *= scale;
        }
    }
    return true;
}

__device__ __forceinline__ bool ffw_sg2_mobility_project_config(
    float q[FFW_SG2_MOBILITY_DIM]
) {
    constexpr int max_iters = 15;
    constexpr float tol = 1.0e-3f;
    constexpr float damping = 1.0e-4f;
    constexpr float max_step = 0.2f;

    ffw_sg2_mobility_clamp(q);
    for (int iter = 0; iter < max_iters; ++iter) {
        float h[FFW_SG2_MOBILITY_RESIDUAL_DIM];
        ffw_sg2_mobility_constraint_residual(q, h);
        if (ffw_sg2_mobility_residual_norm(h) < tol) {
            return true;
        }

        float correction[FFW_SG2_MOBILITY_DIM];
        float task_error_norm = 1.0e30f;
        if (
            !ffw_sg2_mobility_task_correction(
                q,
                damping,
                max_step,
                correction,
                task_error_norm
            )
        ) {
            return false;
        }
        for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
            q[joint] -= correction[joint];
        }
        ffw_sg2_mobility_clamp(q);
    }

    float h[FFW_SG2_MOBILITY_RESIDUAL_DIM];
    ffw_sg2_mobility_constraint_residual(q, h);
    return ffw_sg2_mobility_residual_norm(h) < tol;
}

__device__ __forceinline__ bool ffw_sg2_mobility_project_motion(
    volatile float *motion_segment,
    volatile float *motion_segment_next,
    int granularity,
    volatile unsigned char *projection_valid,
    volatile int *projection_prog,
    volatile unsigned int *projection_success,
    int max_iters,
    float alpha,
    float damping,
    float task_tolerance,
    float smoothness_threshold,
    float smoothness_weight,
    bool use_smoothness,
    float max_step,
    int tid,
    bool return_when_success = true
) {
    const int waypoint = tid / 4 + 1;
    const int lane = tid % 4;

    if (tid == 0) {
        projection_prog[0] = 0;
        projection_success[0] = 0;
        projection_valid[0] = 1;
    }
    if (tid < FFW_SG2_MOBILITY_DIM) {
        motion_segment_next[tid] = motion_segment[tid];
    }
    if (waypoint <= granularity && lane == 0) {
        projection_valid[waypoint] = 0;
    }
    __syncthreads();

    for (int iter = 0; iter < max_iters; ++iter) {
        if (projection_success[0] == 0 && waypoint <= granularity) {
            const int current_prog = projection_prog[0];
            if (waypoint > current_prog) {
                if (lane == 0) {
                    float q[FFW_SG2_MOBILITY_DIM];
                    float correction[FFW_SG2_MOBILITY_DIM];
                    float task_error_norm = 1.0e30f;
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        q[joint] =
                            motion_segment[
                                waypoint * FFW_SG2_MOBILITY_DIM + joint
                            ];
                    }

                    const bool correction_ok =
                        ffw_sg2_mobility_task_correction(
                            q,
                            damping,
                            max_step,
                            correction,
                            task_error_norm
                        );

                    float diff[FFW_SG2_MOBILITY_DIM];
                    float smooth_dist2 = 0.0f;
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        diff[joint] =
                            q[joint] -
                            motion_segment[
                                (waypoint - 1) *
                                FFW_SG2_MOBILITY_DIM +
                                joint
                            ];
                        smooth_dist2 += diff[joint] * diff[joint];
                    }
                    const float smooth_dist = sqrtf(smooth_dist2);
                    const float smooth_error = use_smoothness
                        ? fmaxf(0.0f, smooth_dist - smoothness_threshold)
                        : 0.0f;
                    const float inv_smooth_dist = smooth_dist > 1.0e-8f
                        ? 1.0f / smooth_dist
                        : 0.0f;

                    float combined[FFW_SG2_MOBILITY_DIM];
                    float combined_norm2 = 0.0f;
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        const float grad_smooth =
                            diff[joint] * inv_smooth_dist * smooth_error;
                        combined[joint] =
                            alpha *
                            (
                                correction[joint] +
                                smoothness_weight * grad_smooth
                            );
                        combined_norm2 += combined[joint] * combined[joint];
                    }

                    const float combined_norm = sqrtf(combined_norm2);
                    const float combined_scale =
                        max_step > 0.0f && combined_norm > max_step
                            ? max_step / combined_norm
                            : 1.0f;
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        motion_segment_next[
                            waypoint * FFW_SG2_MOBILITY_DIM + joint
                        ] = q[joint] - combined_scale * combined[joint];
                    }

                    float q_next[FFW_SG2_MOBILITY_DIM];
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        q_next[joint] =
                            motion_segment_next[
                                waypoint * FFW_SG2_MOBILITY_DIM + joint
                            ];
                    }
                    ffw_sg2_mobility_clamp(q_next);
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        motion_segment_next[
                            waypoint * FFW_SG2_MOBILITY_DIM + joint
                        ] = q_next[joint];
                    }

                    projection_valid[waypoint] =
                        correction_ok &&
                        task_error_norm < task_tolerance &&
                        (
                            !use_smoothness ||
                            smooth_dist <= smoothness_threshold
                        );
                }
            } else {
                if (lane == 0) {
                    projection_valid[waypoint] = 1;
                }
                for (
                    int joint = lane;
                    joint < FFW_SG2_MOBILITY_DIM;
                    joint += 4
                ) {
                    motion_segment_next[
                        waypoint * FFW_SG2_MOBILITY_DIM + joint
                    ] = motion_segment[
                        waypoint * FFW_SG2_MOBILITY_DIM + joint
                    ];
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            int prog = projection_prog[0];
            while (
                prog + 1 <= granularity &&
                projection_valid[prog + 1] != 0
            ) {
                ++prog;
            }
            projection_prog[0] = prog;
            if (prog == granularity) {
                projection_success[0] = 1;
            }
        }
        __syncthreads();

        if (projection_success[0] != 0 && return_when_success) {
            return true;
        }

        if (waypoint <= granularity && waypoint > projection_prog[0]) {
            for (
                int joint = lane;
                joint < FFW_SG2_MOBILITY_DIM;
                joint += 4
            ) {
                motion_segment[
                    waypoint * FFW_SG2_MOBILITY_DIM + joint
                ] = motion_segment_next[
                    waypoint * FFW_SG2_MOBILITY_DIM + joint
                ];
            }
        }
        __syncthreads();
    }

    return projection_success[0] != 0;
}

} // namespace ppln::collision

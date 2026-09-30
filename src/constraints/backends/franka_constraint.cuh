#pragma once

// Compiled Franka constraint backend.

#include <type_traits>

#include "src/planning/FrankaConstraintParameters.hh"
#include "src/planning/JointLimits.cuh"
#include "src/planning/Robots.hh"
#include "src/robots/franka_kinematics.cuh"

namespace ppln::collision {

constexpr int FRANKA_WORLD_YAW_CONSTRAINT_DIM = 2;
constexpr int FRANKA_POSE_CONSTRAINT_DIM = 6;
constexpr int FRANKA_MAX_CONSTRAINT_DIM = 8;
constexpr int FRANKA_SINGLE_MAX_TANGENT_DIM = 7;
constexpr int FRANKA_SINGLE_TANGENT_BASIS_SIZE = 7 * 7;
constexpr int FRANKA_DUAL_MAX_TANGENT_DIM = 8;
constexpr int FRANKA_DUAL_TANGENT_BASIS_SIZE = 14 * 8;

__device__ __forceinline__ void franka_quaternion_to_matrix(
    const float quaternion[4],
    float rotation[9]
) {
    const float inverse_norm = rsqrtf(fmaxf(
        quaternion[0] * quaternion[0] +
        quaternion[1] * quaternion[1] +
        quaternion[2] * quaternion[2] +
        quaternion[3] * quaternion[3],
        1.0e-20f
    ));
    const float w = quaternion[0] * inverse_norm;
    const float x = quaternion[1] * inverse_norm;
    const float y = quaternion[2] * inverse_norm;
    const float z = quaternion[3] * inverse_norm;
    rotation[0] = 1.0f - 2.0f * (y * y + z * z);
    rotation[1] = 2.0f * (x * y - z * w);
    rotation[2] = 2.0f * (x * z + y * w);
    rotation[3] = 2.0f * (x * y + z * w);
    rotation[4] = 1.0f - 2.0f * (x * x + z * z);
    rotation[5] = 2.0f * (y * z - x * w);
    rotation[6] = 2.0f * (x * z - y * w);
    rotation[7] = 2.0f * (y * z + x * w);
    rotation[8] = 1.0f - 2.0f * (x * x + y * y);
}

__device__ __forceinline__ void franka_multiply3(
    const float left[9],
    const float right[9],
    float result[9]
) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result[row * 3 + column] = 0.0f;
            for (int inner = 0; inner < 3; ++inner) {
                result[row * 3 + column] +=
                    left[row * 3 + inner] * right[inner * 3 + column];
            }
        }
    }
}

__device__ __forceinline__ void franka_so3_log(
    const float rotation[9],
    float omega[3]
) {
    float cosine = 0.5f * (
        rotation[0] + rotation[4] + rotation[8] - 1.0f
    );
    cosine = fminf(fmaxf(cosine, -1.0f), 1.0f);
    const float vee[3] = {
        0.5f * (rotation[7] - rotation[5]),
        0.5f * (rotation[2] - rotation[6]),
        0.5f * (rotation[3] - rotation[1])
    };
    const float sine = sqrtf(
        vee[0] * vee[0] + vee[1] * vee[1] + vee[2] * vee[2]
    );
    const float angle = atan2f(sine, cosine);
    if (sine < 1.0e-8f && angle < 1.0e-8f) {
        omega[0] = omega[1] = omega[2] = 0.0f;
        return;
    }
    if (fabsf(angle - 3.14159265358979323846f) < 1.0e-4f) {
        float axis[3] = {
            sqrtf(fmaxf(0.5f * (rotation[0] + 1.0f), 0.0f)),
            sqrtf(fmaxf(0.5f * (rotation[4] + 1.0f), 0.0f)),
            sqrtf(fmaxf(0.5f * (rotation[8] + 1.0f), 0.0f))
        };
        if (rotation[7] - rotation[5] < 0.0f) axis[0] = -axis[0];
        if (rotation[2] - rotation[6] < 0.0f) axis[1] = -axis[1];
        if (rotation[3] - rotation[1] < 0.0f) axis[2] = -axis[2];
        const float norm = sqrtf(
            axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]
        );
        const float scale = angle / fmaxf(norm, 1.0e-8f);
        for (int component = 0; component < 3; ++component) {
            omega[component] = axis[component] * scale;
        }
        return;
    }
    const float scale = angle / fmaxf(sine, 1.0e-12f);
    for (int component = 0; component < 3; ++component) {
        omega[component] = vee[component] * scale;
    }
}

__device__ __forceinline__ void franka_so3_right_jacobian_inverse(
    const float omega[3],
    float inverse[9]
) {
    const float angle_squared =
        omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    const float angle = sqrtf(angle_squared);
    const float skew[9] = {
        0.0f, -omega[2], omega[1],
        omega[2], 0.0f, -omega[0],
        -omega[1], omega[0], 0.0f
    };
    float skew_squared[9];
    franka_multiply3(skew, skew, skew_squared);
    float coefficient;
    if (angle < 1.0e-4f) {
        coefficient = 1.0f / 12.0f + angle_squared / 720.0f;
    } else {
        const float half_angle = 0.5f * angle;
        coefficient = (
            1.0f - half_angle * cosf(half_angle) /
                fmaxf(sinf(half_angle), 1.0e-12f)
        ) / angle_squared;
    }
    for (int index = 0; index < 9; ++index) {
        const float identity = index % 4 == 0 ? 1.0f : 0.0f;
        inverse[index] = identity + 0.5f * skew[index] +
            coefficient * skew_squared[index];
    }
}

template <int Dimension>
__device__ __forceinline__ void franka_arm_pose_jacobian(
    const float q[7],
    const float base[3],
    int joint_offset,
    FrankaTransform &pose,
    float position_jacobian[3 * Dimension],
    float angular_jacobian[3 * Dimension]
) {
    float origins[7][3];
    float axes[7][3];
    franka_arm_kinematics(q, base, pose, origins, axes);
    for (int index = 0; index < 3 * Dimension; ++index) {
        position_jacobian[index] = 0.0f;
        angular_jacobian[index] = 0.0f;
    }
    for (int local_joint = 0; local_joint < 7; ++local_joint) {
        const int joint = joint_offset + local_joint;
        float lever[3] = {
            pose.translation[0] - origins[local_joint][0],
            pose.translation[1] - origins[local_joint][1],
            pose.translation[2] - origins[local_joint][2]
        };
        float linear[3];
        franka_cross3(axes[local_joint], lever, linear);
        for (int component = 0; component < 3; ++component) {
            position_jacobian[component * Dimension + joint] =
                linear[component];
            angular_jacobian[component * Dimension + joint] =
                axes[local_joint][component];
        }
    }
}

__device__ __forceinline__ void franka_world_yaw_residual_and_jacobian(
    const FrankaTransform &pose,
    const float *angular_jacobian,
    int dimension,
    const constraints::FrankaConstraintParameters &parameters,
    float residual[FRANKA_WORLD_YAW_CONSTRAINT_DIM],
    float *jacobian
) {
    // This is R(q) * R_start^T * world_Z.  Setting its world X/Y
    // components to zero leaves the connected solution set
    // R(q) = Rz(yaw) * R_start, so yaw is the only free orientation DOF.
    const float axis_world[3] = {
        pose.rotation[0] * parameters.world_yaw_axis_local[0] +
            pose.rotation[1] * parameters.world_yaw_axis_local[1] +
            pose.rotation[2] * parameters.world_yaw_axis_local[2],
        pose.rotation[3] * parameters.world_yaw_axis_local[0] +
            pose.rotation[4] * parameters.world_yaw_axis_local[1] +
            pose.rotation[5] * parameters.world_yaw_axis_local[2],
        pose.rotation[6] * parameters.world_yaw_axis_local[0] +
            pose.rotation[7] * parameters.world_yaw_axis_local[1] +
            pose.rotation[8] * parameters.world_yaw_axis_local[2]
    };
    for (int row = 0; row < FRANKA_WORLD_YAW_CONSTRAINT_DIM; ++row) {
        residual[row] = axis_world[row];
        if (jacobian == nullptr) continue;
        for (int joint = 0; joint < dimension; ++joint) {
            const float angular[3] = {
                angular_jacobian[joint],
                angular_jacobian[dimension + joint],
                angular_jacobian[2 * dimension + joint]
            };
            float axis_derivative[3];
            franka_cross3(angular, axis_world, axis_derivative);
            jacobian[row * dimension + joint] = axis_derivative[row];
        }
    }
}

__device__ __forceinline__ void franka_single_constraint_residual_and_jacobian(
    const float q[7],
    const constraints::FrankaConstraintParameters &parameters,
    float residual[FRANKA_WORLD_YAW_CONSTRAINT_DIM],
    float *jacobian
) {
    const float base[3] = {0.0f, 0.0f, 0.0f};
    FrankaTransform pose{};
    float position_jacobian[3 * 7];
    float angular_jacobian[3 * 7];
    franka_arm_pose_jacobian<7>(
        q, base, 0, pose, position_jacobian, angular_jacobian
    );
    franka_world_yaw_residual_and_jacobian(
        pose, angular_jacobian, 7, parameters, residual, jacobian
    );
}

inline __device__ __noinline__ void franka_dual_constraint_residual_and_jacobian(
    const float q[14],
    const constraints::FrankaConstraintParameters &parameters,
    bool axis_enabled,
    float residual[FRANKA_MAX_CONSTRAINT_DIM],
    float *jacobian
) {
    const float left_base[3] = {0.0f, 0.2f, 0.6f};
    const float right_base[3] = {0.0f, -0.2f, 0.6f};
    FrankaTransform left{};
    FrankaTransform right{};
    float left_position_jacobian[3 * 14];
    float left_angular_jacobian[3 * 14];
    float right_position_jacobian[3 * 14];
    float right_angular_jacobian[3 * 14];
    franka_arm_pose_jacobian<14>(
        q, left_base, 0, left,
        left_position_jacobian, left_angular_jacobian
    );
    franka_arm_pose_jacobian<14>(
        q + 7, right_base, 7, right,
        right_position_jacobian, right_angular_jacobian
    );

    float world_delta[3] = {
        right.translation[0] - left.translation[0],
        right.translation[1] - left.translation[1],
        right.translation[2] - left.translation[2]
    };
    float relative_position[3];
    franka_rotation_transpose_vector(
        left.rotation, world_delta, relative_position
    );
    for (int component = 0; component < 3; ++component) {
        residual[component] = relative_position[component] -
            parameters.relative_pose_target[4 + component];
    }

    float relative_rotation[9];
    franka_matrix_transpose_multiply(
        left.rotation, right.rotation, relative_rotation
    );
    float target_rotation[9];
    franka_quaternion_to_matrix(parameters.relative_pose_target, target_rotation);
    float orientation_error_rotation[9];
    franka_matrix_transpose_multiply(
        target_rotation, relative_rotation, orientation_error_rotation
    );
    float orientation_residual[3];
    franka_so3_log(orientation_error_rotation, orientation_residual);
    for (int component = 0; component < 3; ++component) {
        residual[3 + component] = orientation_residual[component];
    }

    if (jacobian != nullptr) {
        float inverse_right_jacobian[9];
        franka_so3_right_jacobian_inverse(
            orientation_residual, inverse_right_jacobian
        );
        for (int joint = 0; joint < 14; ++joint) {
            float left_angular[3] = {
                left_angular_jacobian[joint],
                left_angular_jacobian[14 + joint],
                left_angular_jacobian[28 + joint]
            };
            float delta_derivative[3] = {
                right_position_jacobian[joint] -
                    left_position_jacobian[joint],
                right_position_jacobian[14 + joint] -
                    left_position_jacobian[14 + joint],
                right_position_jacobian[28 + joint] -
                    left_position_jacobian[28 + joint]
            };
            float frame_rotation_term[3];
            franka_cross3(left_angular, world_delta, frame_rotation_term);
            for (int component = 0; component < 3; ++component) {
                delta_derivative[component] -= frame_rotation_term[component];
            }
            float relative_position_derivative[3];
            franka_rotation_transpose_vector(
                left.rotation,
                delta_derivative,
                relative_position_derivative
            );
            for (int component = 0; component < 3; ++component) {
                jacobian[component * 14 + joint] =
                    relative_position_derivative[component];
            }

            float relative_spatial_angular_world[3] = {
                right_angular_jacobian[joint] - left_angular[0],
                right_angular_jacobian[14 + joint] - left_angular[1],
                right_angular_jacobian[28 + joint] - left_angular[2]
            };
            float relative_spatial_angular_left[3];
            franka_rotation_transpose_vector(
                left.rotation,
                relative_spatial_angular_world,
                relative_spatial_angular_left
            );
            float body_angular[3];
            franka_rotation_transpose_vector(
                relative_rotation,
                relative_spatial_angular_left,
                body_angular
            );
            for (int row = 0; row < 3; ++row) {
                jacobian[(3 + row) * 14 + joint] =
                    inverse_right_jacobian[row * 3] * body_angular[0] +
                    inverse_right_jacobian[row * 3 + 1] * body_angular[1] +
                    inverse_right_jacobian[row * 3 + 2] * body_angular[2];
            }
        }
    }

    if (axis_enabled) {
        franka_world_yaw_residual_and_jacobian(
            left,
            left_angular_jacobian,
            14,
            parameters,
            residual + FRANKA_POSE_CONSTRAINT_DIM,
            jacobian == nullptr
                ? nullptr
                : jacobian + FRANKA_POSE_CONSTRAINT_DIM * 14
        );
    }
}

template <typename Robot>
__device__ __forceinline__ int franka_constraint_dim(bool axis_enabled) {
    if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
        return axis_enabled ? FRANKA_WORLD_YAW_CONSTRAINT_DIM : 0;
    }
    return FRANKA_POSE_CONSTRAINT_DIM +
        (axis_enabled ? FRANKA_WORLD_YAW_CONSTRAINT_DIM : 0);
}

template <typename Robot>
__device__ __forceinline__ void franka_constraint_residual_and_jacobian(
    const float *q,
    const constraints::FrankaConstraintParameters &parameters,
    bool axis_enabled,
    float residual[FRANKA_MAX_CONSTRAINT_DIM],
    float *jacobian
) {
    if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
        if (axis_enabled) {
            franka_single_constraint_residual_and_jacobian(
                q, parameters, residual, jacobian
            );
        }
    } else {
        franka_dual_constraint_residual_and_jacobian(
            q, parameters, axis_enabled, residual, jacobian
        );
    }
}

template <typename Robot>
__device__ __forceinline__ float franka_constraint_error_squared(
    const float *q,
    const constraints::FrankaConstraintParameters &parameters,
    bool axis_enabled
) {
    float residual[FRANKA_MAX_CONSTRAINT_DIM]{};
    franka_constraint_residual_and_jacobian<Robot>(
        q, parameters, axis_enabled, residual, nullptr
    );
    float squared = 0.0f;
    const int rows = franka_constraint_dim<Robot>(axis_enabled);
    for (int row = 0; row < rows; ++row) {
        squared += residual[row] * residual[row];
    }
    return squared;
}

template <typename Robot>
__device__ __forceinline__ float franka_constraint_error_norm(
    const float *q,
    const constraints::FrankaConstraintParameters &parameters,
    bool axis_enabled
) {
    return sqrtf(franka_constraint_error_squared<Robot>(
        q, parameters, axis_enabled
    ));
}

template <int Dimension, int MaximumTangentDimension>
__device__ __forceinline__ bool franka_tangent_basis_from_jacobian(
    const float *jacobian,
    int rows,
    float *basis
) {
    float reduced[FRANKA_MAX_CONSTRAINT_DIM][14]{};
    for (int row = 0; row < rows; ++row) {
        for (int joint = 0; joint < Dimension; ++joint) {
            reduced[row][joint] = jacobian[row * Dimension + joint];
        }
    }
    int pivot_columns[FRANKA_MAX_CONSTRAINT_DIM]{};
    int rank = 0;
    for (int column = 0;
         column < Dimension && rank < rows;
         ++column) {
        int pivot = rank;
        float best = fabsf(reduced[rank][column]);
        for (int row = rank + 1; row < rows; ++row) {
            const float candidate = fabsf(reduced[row][column]);
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }
        if (best < 1.0e-5f) continue;
        if (pivot != rank) {
            for (int joint = 0; joint < Dimension; ++joint) {
                const float temporary = reduced[rank][joint];
                reduced[rank][joint] = reduced[pivot][joint];
                reduced[pivot][joint] = temporary;
            }
        }
        const float inverse_pivot = 1.0f / reduced[rank][column];
        for (int joint = column; joint < Dimension; ++joint) {
            reduced[rank][joint] *= inverse_pivot;
        }
        for (int row = 0; row < rows; ++row) {
            if (row == rank) continue;
            const float factor = reduced[row][column];
            for (int joint = column; joint < Dimension; ++joint) {
                reduced[row][joint] -= factor * reduced[rank][joint];
            }
        }
        pivot_columns[rank++] = column;
    }

    bool is_pivot[14]{};
    for (int row = 0; row < rank; ++row) {
        is_pivot[pivot_columns[row]] = true;
    }
    for (int index = 0;
         index < Dimension * MaximumTangentDimension;
         ++index) {
        basis[index] = 0.0f;
    }
    const int expected_columns = Dimension - rows;
    int basis_column = 0;
    for (int free_column = 0;
         free_column < Dimension && basis_column < expected_columns;
         ++free_column) {
        if (is_pivot[free_column]) continue;
        float vector[14]{};
        vector[free_column] = 1.0f;
        for (int row = 0; row < rank; ++row) {
            vector[pivot_columns[row]] = -reduced[row][free_column];
        }
        for (int previous = 0; previous < basis_column; ++previous) {
            float dot = 0.0f;
            for (int joint = 0; joint < Dimension; ++joint) {
                dot += vector[joint] *
                    basis[joint * MaximumTangentDimension + previous];
            }
            for (int joint = 0; joint < Dimension; ++joint) {
                vector[joint] -= dot *
                    basis[joint * MaximumTangentDimension + previous];
            }
        }
        float squared = 0.0f;
        for (int joint = 0; joint < Dimension; ++joint) {
            squared += vector[joint] * vector[joint];
        }
        if (squared < 1.0e-10f) continue;
        const float inverse_norm = rsqrtf(squared);
        for (int joint = 0; joint < Dimension; ++joint) {
            basis[joint * MaximumTangentDimension + basis_column] =
                vector[joint] * inverse_norm;
        }
        ++basis_column;
    }
    return rank == rows && basis_column == expected_columns;
}

__device__ __forceinline__ bool franka_single_tangent_basis(
    const float q[7],
    const constraints::FrankaConstraintParameters &parameters,
    bool axis_enabled,
    float basis[FRANKA_SINGLE_TANGENT_BASIS_SIZE]
) {
    float residual[FRANKA_MAX_CONSTRAINT_DIM]{};
    float jacobian[FRANKA_MAX_CONSTRAINT_DIM * 7]{};
    franka_constraint_residual_and_jacobian<robots::FrankaSingle>(
        q, parameters, axis_enabled, residual, jacobian
    );
    return franka_tangent_basis_from_jacobian<
        7, FRANKA_SINGLE_MAX_TANGENT_DIM
    >(jacobian, franka_constraint_dim<robots::FrankaSingle>(axis_enabled), basis);
}

__device__ __forceinline__ bool franka_dual_tangent_basis(
    const float q[14],
    const constraints::FrankaConstraintParameters &parameters,
    bool axis_enabled,
    float basis[FRANKA_DUAL_TANGENT_BASIS_SIZE]
) {
    float residual[FRANKA_MAX_CONSTRAINT_DIM]{};
    float jacobian[FRANKA_MAX_CONSTRAINT_DIM * 14]{};
    franka_constraint_residual_and_jacobian<robots::Franka>(
        q, parameters, axis_enabled, residual, jacobian
    );
    return franka_tangent_basis_from_jacobian<
        14, FRANKA_DUAL_MAX_TANGENT_DIM
    >(jacobian, franka_constraint_dim<robots::Franka>(axis_enabled), basis);
}

template <typename Robot>
__device__ __forceinline__ void franka_clamp_configuration(float *q) {
    for (int joint = 0; joint < Robot::dimension; ++joint) {
        const float lower = Robot::get_s_a(joint);
        const float upper = lower + Robot::get_s_m(joint);
        q[joint] = fminf(upper, fmaxf(lower, q[joint]));
    }
}

template <typename Robot>
__device__ __forceinline__ bool franka_task_correction(
    const float *q,
    const constraints::FrankaConstraintParameters &parameters,
    bool axis_enabled,
    float damping,
    float maximum_step,
    float *correction,
    float &task_error_norm
) {
    constexpr int dimension = Robot::dimension;
    const int rows = franka_constraint_dim<Robot>(axis_enabled);
    for (int joint = 0; joint < dimension; ++joint) correction[joint] = 0.0f;
    if (rows == 0) {
        task_error_norm = 0.0f;
        return true;
    }
    float residual[FRANKA_MAX_CONSTRAINT_DIM]{};
    float jacobian[FRANKA_MAX_CONSTRAINT_DIM * dimension]{};
    franka_constraint_residual_and_jacobian<Robot>(
        q, parameters, axis_enabled, residual, jacobian
    );
    float error_squared = 0.0f;
    for (int row = 0; row < rows; ++row) {
        error_squared += residual[row] * residual[row];
    }
    task_error_norm = sqrtf(error_squared);

    float system[FRANKA_MAX_CONSTRAINT_DIM][FRANKA_MAX_CONSTRAINT_DIM + 1]{};
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < rows; ++column) {
            float value = row == column ? damping : 0.0f;
            for (int joint = 0; joint < dimension; ++joint) {
                value += jacobian[row * dimension + joint] *
                    jacobian[column * dimension + joint];
            }
            system[row][column] = value;
        }
        system[row][rows] = residual[row];
    }
    for (int pivot = 0; pivot < rows; ++pivot) {
        int best_row = pivot;
        float best = fabsf(system[pivot][pivot]);
        for (int row = pivot + 1; row < rows; ++row) {
            const float candidate = fabsf(system[row][pivot]);
            if (candidate > best) {
                best = candidate;
                best_row = row;
            }
        }
        if (best < 1.0e-12f) return false;
        if (best_row != pivot) {
            for (int column = pivot; column <= rows; ++column) {
                const float temporary = system[pivot][column];
                system[pivot][column] = system[best_row][column];
                system[best_row][column] = temporary;
            }
        }
        const float inverse_pivot = 1.0f / system[pivot][pivot];
        for (int column = pivot; column <= rows; ++column) {
            system[pivot][column] *= inverse_pivot;
        }
        for (int row = 0; row < rows; ++row) {
            if (row == pivot) continue;
            const float factor = system[row][pivot];
            for (int column = pivot; column <= rows; ++column) {
                system[row][column] -= factor * system[pivot][column];
            }
        }
    }

    float correction_squared = 0.0f;
    for (int joint = 0; joint < dimension; ++joint) {
        for (int row = 0; row < rows; ++row) {
            correction[joint] += jacobian[row * dimension + joint] *
                system[row][rows];
        }
        correction_squared += correction[joint] * correction[joint];
    }
    const float correction_norm = sqrtf(correction_squared);
    if (maximum_step > 0.0f && correction_norm > maximum_step) {
        const float scale = maximum_step / correction_norm;
        for (int joint = 0; joint < dimension; ++joint) {
            correction[joint] *= scale;
        }
    }
    return true;
}

template <typename Robot>
__device__ __forceinline__ bool franka_project_motion_impl(
    volatile float *motion_segment,
    volatile float *motion_segment_next,
    int granularity,
    const constraints::FrankaConstraintParameters &parameters,
    bool axis_enabled,
    volatile unsigned char *projection_valid,
    volatile int *projection_progress,
    volatile unsigned int *projection_success,
    int max_iterations,
    float alpha,
    float damping,
    float task_tolerance,
    float smoothness_threshold,
    float smoothness_weight,
    bool use_smoothness,
    float maximum_step,
    int tid,
    bool return_when_success = true
) {
    constexpr int dimension = Robot::dimension;
    const int waypoint = tid / 4 + 1;
    const int lane = tid % 4;
    if (tid == 0) {
        projection_progress[0] = 0;
        projection_success[0] = 0;
        projection_valid[0] = 1;
    }
    if (tid < dimension) motion_segment_next[tid] = motion_segment[tid];
    if (waypoint <= granularity && lane == 0) {
        projection_valid[waypoint] = 0;
    }
    __syncthreads();

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        if (projection_success[0] == 0 && waypoint <= granularity) {
            const int progress = projection_progress[0];
            if (waypoint > progress) {
                if (lane == 0) {
                    float q[dimension];
                    float correction[dimension]{};
                    for (int joint = 0; joint < dimension; ++joint) {
                        q[joint] = motion_segment[
                            waypoint * dimension + joint
                        ];
                    }
                    float task_error = 1.0e30f;
                    const bool correction_ok = franka_task_correction<Robot>(
                        q,
                        parameters,
                        axis_enabled,
                        damping,
                        maximum_step,
                        correction,
                        task_error
                    );
                    float difference[dimension];
                    float distance_squared = 0.0f;
                    for (int joint = 0; joint < dimension; ++joint) {
                        difference[joint] = q[joint] - motion_segment[
                            (waypoint - 1) * dimension + joint
                        ];
                        distance_squared += difference[joint] * difference[joint];
                    }
                    const float distance = sqrtf(distance_squared);
                    const float smoothness_error = correction_ok && use_smoothness
                        ? fmaxf(0.0f, distance - smoothness_threshold)
                        : 0.0f;
                    const float inverse_distance = distance > 1.0e-8f
                        ? 1.0f / distance
                        : 0.0f;
                    float combined[dimension];
                    float combined_squared = 0.0f;
                    for (int joint = 0; joint < dimension; ++joint) {
                        const float smoothness_gradient =
                            difference[joint] * inverse_distance * smoothness_error;
                        combined[joint] = correction_ok
                            ? alpha * (
                                correction[joint] +
                                smoothness_weight * smoothness_gradient
                            )
                            : 0.0f;
                        combined_squared += combined[joint] * combined[joint];
                    }
                    const float combined_norm = sqrtf(combined_squared);
                    const float combined_scale =
                        maximum_step > 0.0f && combined_norm > maximum_step
                            ? maximum_step / combined_norm
                            : 1.0f;
                    for (int joint = 0; joint < dimension; ++joint) {
                        q[joint] -= combined_scale * combined[joint];
                    }
                    franka_clamp_configuration<Robot>(q);
                    for (int joint = 0; joint < dimension; ++joint) {
                        motion_segment_next[
                            waypoint * dimension + joint
                        ] = q[joint];
                    }
                    projection_valid[waypoint] = correction_ok &&
                        task_error < task_tolerance &&
                        (!use_smoothness || distance <= smoothness_threshold);
                }
            } else {
                if (lane == 0) projection_valid[waypoint] = 1;
                for (int joint = lane; joint < dimension; joint += 4) {
                    motion_segment_next[waypoint * dimension + joint] =
                        motion_segment[waypoint * dimension + joint];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            int progress = projection_progress[0];
            while (progress + 1 <= granularity &&
                   projection_valid[progress + 1] != 0) {
                ++progress;
            }
            projection_progress[0] = progress;
            if (progress == granularity) projection_success[0] = 1;
        }
        __syncthreads();
        if (projection_success[0] != 0 && return_when_success) {
            break;
        }
        if (waypoint <= granularity && waypoint > projection_progress[0]) {
            for (int joint = lane; joint < dimension; joint += 4) {
                motion_segment[waypoint * dimension + joint] =
                    motion_segment_next[waypoint * dimension + joint];
            }
        }
        __syncthreads();
    }
    if (projection_success[0] == 0) return false;
    if (tid == 0) {
        projection_valid[0] =
            planning::configuration_within_joint_limits<Robot>(
                &motion_segment[granularity * dimension]
            );
    }
    __syncthreads();
    return projection_valid[0] != 0;
}

__device__ __forceinline__ bool franka_single_project_motion(
    volatile float *motion_segment,
    volatile float *motion_segment_next,
    int granularity,
    const constraints::FrankaConstraintParameters &parameters,
    bool axis_enabled,
    volatile unsigned char *projection_valid,
    volatile int *projection_progress,
    volatile unsigned int *projection_success,
    int max_iterations,
    float alpha,
    float damping,
    float task_tolerance,
    float smoothness_threshold,
    float smoothness_weight,
    bool use_smoothness,
    float maximum_step,
    int tid,
    bool return_when_success = true
) {
    return franka_project_motion_impl<robots::FrankaSingle>(
        motion_segment, motion_segment_next, granularity, parameters,
        axis_enabled, projection_valid, projection_progress,
        projection_success, max_iterations, alpha, damping,
        task_tolerance, smoothness_threshold, smoothness_weight,
        use_smoothness, maximum_step, tid, return_when_success
    );
}

__device__ __forceinline__ bool franka_dual_project_motion(
    volatile float *motion_segment,
    volatile float *motion_segment_next,
    int granularity,
    const constraints::FrankaConstraintParameters &parameters,
    bool axis_enabled,
    volatile unsigned char *projection_valid,
    volatile int *projection_progress,
    volatile unsigned int *projection_success,
    int max_iterations,
    float alpha,
    float damping,
    float task_tolerance,
    float smoothness_threshold,
    float smoothness_weight,
    bool use_smoothness,
    float maximum_step,
    int tid,
    bool return_when_success = true
) {
    return franka_project_motion_impl<robots::Franka>(
        motion_segment, motion_segment_next, granularity, parameters,
        axis_enabled, projection_valid, projection_progress,
        projection_success, max_iterations, alpha, damping,
        task_tolerance, smoothness_threshold, smoothness_weight,
        use_smoothness, maximum_step, tid, return_when_success
    );
}

}  // namespace ppln::collision

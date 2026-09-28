#pragma once

#include "src/planning/IgrisCConstraintParameters.hh"
#include "src/planning/JointLimits.cuh"
#include "src/robots/igris_c.cuh"
#include "src/robots/igris_c_kinematics.cuh"

namespace ppln::collision {

constexpr int IGRIS_C_FOOT_POSE_DIM = 6;
constexpr int IGRIS_C_FEET_CONSTRAINT_DIM = 12;
constexpr int IGRIS_C_COM_CONSTRAINT_DIM = 2;
constexpr int IGRIS_C_BIMANUAL_CONSTRAINT_DIM = 6;
constexpr int IGRIS_C_BIMANUAL_AXIS_CONSTRAINT_DIM = 2;
constexpr int IGRIS_C_CONSTRAINT_DIM =
    IGRIS_C_FEET_CONSTRAINT_DIM + IGRIS_C_COM_CONSTRAINT_DIM +
    IGRIS_C_BIMANUAL_CONSTRAINT_DIM +
    IGRIS_C_BIMANUAL_AXIS_CONSTRAINT_DIM;
constexpr int IGRIS_C_EQUALITY_CONSTRAINT_DIM =
    IGRIS_C_FEET_CONSTRAINT_DIM + IGRIS_C_BIMANUAL_CONSTRAINT_DIM +
    IGRIS_C_BIMANUAL_AXIS_CONSTRAINT_DIM;
constexpr int IGRIS_C_TANGENT_DIM = IGRIS_C_DIM - IGRIS_C_EQUALITY_CONSTRAINT_DIM;
constexpr int IGRIS_C_TANGENT_BASIS_SIZE = IGRIS_C_DIM * IGRIS_C_TANGENT_DIM;

__device__ __forceinline__ void igris_c_quaternion_to_matrix(
    const float pose[7],
    float rotation[9]
) {
    const float inverse_norm = 1.0f / fmaxf(sqrtf(
        pose[0] * pose[0] + pose[1] * pose[1] +
        pose[2] * pose[2] + pose[3] * pose[3]
    ), 1.0e-12f);
    const float w = pose[0] * inverse_norm;
    const float x = pose[1] * inverse_norm;
    const float y = pose[2] * inverse_norm;
    const float z = pose[3] * inverse_norm;
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

__device__ __forceinline__ void igris_c_so3_log(
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
        omega[0] = 0.0f;
        omega[1] = 0.0f;
        omega[2] = 0.0f;
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

__device__ __forceinline__ void igris_c_so3_right_jacobian_inverse(
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
    igris_c_multiply3(skew, skew, skew_squared);
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

// residual = Log(R_target^T R_actual). The incoming angular Jacobian is a
// spatial Jacobian in the coordinate system containing R_actual.
__device__ __forceinline__ void igris_c_orientation_residual_and_jacobian(
    const float actual_rotation[9],
    const float target_pose[7],
    const float spatial_angular_jacobian[3 * IGRIS_C_DIM],
    float residual[3],
    float jacobian[3 * IGRIS_C_DIM]
) {
    float target_rotation[9];
    float error_rotation[9];
    igris_c_quaternion_to_matrix(target_pose, target_rotation);
    igris_c_multiply_at_b(target_rotation, actual_rotation, error_rotation);
    igris_c_so3_log(error_rotation, residual);

    float inverse_right_jacobian[9];
    igris_c_so3_right_jacobian_inverse(residual, inverse_right_jacobian);
    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
        float body_axis[3] = {0.0f, 0.0f, 0.0f};
        for (int row = 0; row < 3; ++row) {
            for (int axis = 0; axis < 3; ++axis) {
                body_axis[row] += actual_rotation[axis * 3 + row] *
                    spatial_angular_jacobian[axis * IGRIS_C_DIM + joint];
            }
        }
        for (int row = 0; row < 3; ++row) {
            jacobian[row * IGRIS_C_DIM + joint] =
                inverse_right_jacobian[row * 3] * body_axis[0] +
                inverse_right_jacobian[row * 3 + 1] * body_axis[1] +
                inverse_right_jacobian[row * 3 + 2] * body_axis[2];
        }
    }
}

// One 22x35 system: foot-pose equality[12], CoM inequality[2], bimanual
// relative-pose equality[6], and bimanual-axis equality[2]. CoM rows are
// exactly zero in the feasible set.
__device__ __noinline__ void igris_c_constraint_residual_and_jacobian(
    const float q[IGRIS_C_DIM],
    const constraints::IgrisCConstraintParameters &parameters,
    float residual[IGRIS_C_CONSTRAINT_DIM],
    float *jacobian
) {
    IgrisCTransform link_poses[IGRIS_C_LINK_COUNT];
    float joint_origins[IGRIS_C_DIM][3];
    float joint_axes[IGRIS_C_DIM][3];
    igris_c_forward_model(q, link_poses, joint_origins, joint_axes);

    for (int foot = 0; foot < 2; ++foot) {
        IgrisCTransform pose;
        float position_jacobian[3 * IGRIS_C_DIM];
        float angular_jacobian[3 * IGRIS_C_DIM];
        igris_c_frame_kinematics_from_model(
            igris_c_task_link_indices[foot],
            link_poses,
            joint_origins,
            joint_axes,
            pose,
            position_jacobian,
            angular_jacobian
        );
        for (int component = 0; component < 3; ++component) {
            const int row = foot * IGRIS_C_FOOT_POSE_DIM + component;
            residual[row] = pose.translation[component] -
                parameters.feet_target[foot][4 + component];
            if (jacobian != nullptr) {
                for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                    jacobian[row * IGRIS_C_DIM + joint] =
                        position_jacobian[component * IGRIS_C_DIM + joint];
                }
            }
        }
        float orientation_residual[3];
        float orientation_jacobian[3 * IGRIS_C_DIM];
        igris_c_orientation_residual_and_jacobian(
            pose.rotation,
            parameters.feet_target[foot],
            angular_jacobian,
            orientation_residual,
            orientation_jacobian
        );
        for (int component = 0; component < 3; ++component) {
            const int row = foot * IGRIS_C_FOOT_POSE_DIM + 3 + component;
            residual[row] = orientation_residual[component];
            if (jacobian != nullptr) {
                for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                    jacobian[row * IGRIS_C_DIM + joint] =
                        orientation_jacobian[component * IGRIS_C_DIM + joint];
                }
            }
        }
    }

    float center_of_mass[3];
    float center_of_mass_jacobian[3 * IGRIS_C_DIM];
    igris_c_center_of_mass_from_model(
        link_poses,
        joint_origins,
        joint_axes,
        center_of_mass,
        center_of_mass_jacobian
    );
    const float payload_mass = fmaxf(parameters.payload_mass_kg, 0.0f);
    if (payload_mass > 0.0f) {
        IgrisCTransform grasp_poses[2];
        float grasp_position_jacobians[2][3 * IGRIS_C_DIM];
        float unused_angular_jacobian[3 * IGRIS_C_DIM];
        for (int grasp = 0; grasp < 2; ++grasp) {
            igris_c_frame_kinematics_from_model(
                igris_c_task_link_indices[IGRIS_C_L_GRASP_FRAME + grasp],
                link_poses,
                joint_origins,
                joint_axes,
                grasp_poses[grasp],
                grasp_position_jacobians[grasp],
                unused_angular_jacobian
            );
        }
        const float total_mass = IGRIS_C_TOTAL_MASS_KG + payload_mass;
        for (int component = 0; component < 3; ++component) {
            const float payload_center = 0.5f * (
                grasp_poses[0].translation[component] +
                grasp_poses[1].translation[component]
            );
            center_of_mass[component] = (
                IGRIS_C_TOTAL_MASS_KG * center_of_mass[component] +
                payload_mass * payload_center
            ) / total_mass;
        }
        for (int index = 0; index < 3 * IGRIS_C_DIM; ++index) {
            const float payload_jacobian = 0.5f * (
                grasp_position_jacobians[0][index] +
                grasp_position_jacobians[1][index]
            );
            center_of_mass_jacobian[index] = (
                IGRIS_C_TOTAL_MASS_KG * center_of_mass_jacobian[index] +
                payload_mass * payload_jacobian
            ) / total_mass;
        }
    }
    const int com_row = IGRIS_C_FEET_CONSTRAINT_DIM;
    residual[com_row] = 0.0f;
    residual[com_row + 1] = 0.0f;
    float error_wrt_com[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int edge = 0; edge < 4; ++edge) {
        const int next = (edge + 1) % 4;
        const float x0 = parameters.support_polygon[2 * edge];
        const float y0 = parameters.support_polygon[2 * edge + 1];
        const float x1 = parameters.support_polygon[2 * next];
        const float y1 = parameters.support_polygon[2 * next + 1];
        const float normal_x = y1 - y0;
        const float normal_y = x0 - x1;
        const float inverse_length = rsqrtf(
            normal_x * normal_x + normal_y * normal_y
        );
        const float unit_x = normal_x * inverse_length;
        const float unit_y = normal_y * inverse_length;
        const float signed_distance =
            unit_x * (center_of_mass[0] - x0) +
            unit_y * (center_of_mass[1] - y0);
        if (signed_distance > 0.0f) {
            residual[com_row] += signed_distance * unit_x;
            residual[com_row + 1] += signed_distance * unit_y;
            error_wrt_com[0] += unit_x * unit_x;
            error_wrt_com[1] += unit_x * unit_y;
            error_wrt_com[2] += unit_y * unit_x;
            error_wrt_com[3] += unit_y * unit_y;
        }
    }
    if (jacobian != nullptr) {
        for (int output_axis = 0; output_axis < 2; ++output_axis) {
            for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                jacobian[(com_row + output_axis) * IGRIS_C_DIM + joint] =
                    error_wrt_com[output_axis * 2] *
                        center_of_mass_jacobian[joint] +
                    error_wrt_com[output_axis * 2 + 1] *
                        center_of_mass_jacobian[IGRIS_C_DIM + joint];
            }
        }
    }

    IgrisCTransform relative_pose;
    float relative_position_jacobian[3 * IGRIS_C_DIM];
    float relative_angular_jacobian[3 * IGRIS_C_DIM];
    igris_c_bimanual_from_model(
        link_poses,
        joint_origins,
        joint_axes,
        relative_pose,
        relative_position_jacobian,
        relative_angular_jacobian
    );
    const int bimanual_row =
        IGRIS_C_FEET_CONSTRAINT_DIM + IGRIS_C_COM_CONSTRAINT_DIM;
    for (int component = 0; component < 3; ++component) {
        residual[bimanual_row + component] =
            relative_pose.translation[component] -
            parameters.bimanual_target[4 + component];
        if (jacobian != nullptr) {
            for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                jacobian[(bimanual_row + component) * IGRIS_C_DIM + joint] =
                    relative_position_jacobian[
                        component * IGRIS_C_DIM + joint
                    ];
            }
        }
    }
    float bimanual_orientation_residual[3];
    float bimanual_orientation_jacobian[3 * IGRIS_C_DIM];
    igris_c_orientation_residual_and_jacobian(
        relative_pose.rotation,
        parameters.bimanual_target,
        relative_angular_jacobian,
        bimanual_orientation_residual,
        bimanual_orientation_jacobian
    );
    for (int component = 0; component < 3; ++component) {
        residual[bimanual_row + 3 + component] =
            bimanual_orientation_residual[component];
        if (jacobian != nullptr) {
            for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                jacobian[(bimanual_row + 3 + component) * IGRIS_C_DIM + joint] =
                    bimanual_orientation_jacobian[
                        component * IGRIS_C_DIM + joint
                    ];
            }
        }
    }

    // Keep the grasp-frame local +X axis parallel to world +Z. With the
    // IGRIS palm-frame convention this leaves the fingertip direction
    // (local -Z) horizontal. The relative-pose equality transfers the same
    // orientation constraint to the right grasp frame. Two independent
    // residuals are sufficient because the axis is unit length.
    IgrisCTransform left_grasp_pose;
    float unused_position_jacobian[3 * IGRIS_C_DIM];
    float left_grasp_angular_jacobian[3 * IGRIS_C_DIM];
    igris_c_frame_kinematics_from_model(
        igris_c_task_link_indices[IGRIS_C_L_GRASP_FRAME],
        link_poses,
        joint_origins,
        joint_axes,
        left_grasp_pose,
        unused_position_jacobian,
        left_grasp_angular_jacobian
    );
    const float grasp_x_axis[3] = {
        left_grasp_pose.rotation[0],
        left_grasp_pose.rotation[3],
        left_grasp_pose.rotation[6]
    };
    const int axis_row = bimanual_row + IGRIS_C_BIMANUAL_CONSTRAINT_DIM;
    residual[axis_row] = grasp_x_axis[0];
    residual[axis_row + 1] = grasp_x_axis[1];
    if (jacobian != nullptr) {
        for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
            const float angular_x =
                left_grasp_angular_jacobian[joint];
            const float angular_y =
                left_grasp_angular_jacobian[IGRIS_C_DIM + joint];
            const float angular_z =
                left_grasp_angular_jacobian[2 * IGRIS_C_DIM + joint];
            jacobian[axis_row * IGRIS_C_DIM + joint] =
                angular_y * grasp_x_axis[2] -
                angular_z * grasp_x_axis[1];
            jacobian[(axis_row + 1) * IGRIS_C_DIM + joint] =
                angular_z * grasp_x_axis[0] -
                angular_x * grasp_x_axis[2];
        }
    }
}

__device__ __forceinline__ void igris_c_constraint_residual(
    const float q[IGRIS_C_DIM],
    const constraints::IgrisCConstraintParameters &parameters,
    float residual[IGRIS_C_CONSTRAINT_DIM]
) {
    igris_c_constraint_residual_and_jacobian(q, parameters, residual, nullptr);
}

__device__ __forceinline__ void igris_c_equality_residual_and_jacobian(
    const float q[IGRIS_C_DIM],
    const constraints::IgrisCConstraintParameters &parameters,
    float residual[IGRIS_C_EQUALITY_CONSTRAINT_DIM],
    float *jacobian
) {
    float combined_residual[IGRIS_C_CONSTRAINT_DIM];
    float combined_jacobian[IGRIS_C_CONSTRAINT_DIM * IGRIS_C_DIM];
    igris_c_constraint_residual_and_jacobian(
        q,
        parameters,
        combined_residual,
        jacobian == nullptr ? nullptr : combined_jacobian
    );
    for (int row = 0; row < IGRIS_C_FEET_CONSTRAINT_DIM; ++row) {
        residual[row] = combined_residual[row];
        if (jacobian != nullptr) {
            for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                jacobian[row * IGRIS_C_DIM + joint] =
                    combined_jacobian[row * IGRIS_C_DIM + joint];
            }
        }
    }
    for (int local_row = 0;
         local_row < IGRIS_C_BIMANUAL_CONSTRAINT_DIM;
         ++local_row) {
        const int equality_row = IGRIS_C_FEET_CONSTRAINT_DIM + local_row;
        const int combined_row = IGRIS_C_FEET_CONSTRAINT_DIM +
            IGRIS_C_COM_CONSTRAINT_DIM + local_row;
        residual[equality_row] = combined_residual[combined_row];
        if (jacobian != nullptr) {
            for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                jacobian[equality_row * IGRIS_C_DIM + joint] =
                    combined_jacobian[combined_row * IGRIS_C_DIM + joint];
            }
        }
    }
    for (int local_row = 0;
         local_row < IGRIS_C_BIMANUAL_AXIS_CONSTRAINT_DIM;
         ++local_row) {
        const int equality_row = IGRIS_C_FEET_CONSTRAINT_DIM +
            IGRIS_C_BIMANUAL_CONSTRAINT_DIM + local_row;
        const int combined_row = IGRIS_C_FEET_CONSTRAINT_DIM +
            IGRIS_C_COM_CONSTRAINT_DIM +
            IGRIS_C_BIMANUAL_CONSTRAINT_DIM + local_row;
        residual[equality_row] = combined_residual[combined_row];
        if (jacobian != nullptr) {
            for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                jacobian[equality_row * IGRIS_C_DIM + joint] =
                    combined_jacobian[combined_row * IGRIS_C_DIM + joint];
            }
        }
    }
}

__device__ __forceinline__ float igris_c_equality_residual_norm(
    const float q[IGRIS_C_DIM],
    const constraints::IgrisCConstraintParameters &parameters
) {
    float residual[IGRIS_C_EQUALITY_CONSTRAINT_DIM];
    igris_c_equality_residual_and_jacobian(q, parameters, residual, nullptr);
    float squared = 0.0f;
    for (float value : residual) {
        squared += value * value;
    }
    return sqrtf(squared);
}

__device__ __forceinline__ bool igris_c_tangent_basis_from_jacobian(
    const float jacobian[IGRIS_C_EQUALITY_CONSTRAINT_DIM * IGRIS_C_DIM],
    float basis[IGRIS_C_TANGENT_BASIS_SIZE]
) {
    float reduced[IGRIS_C_EQUALITY_CONSTRAINT_DIM][IGRIS_C_DIM];
    for (int row = 0; row < IGRIS_C_EQUALITY_CONSTRAINT_DIM; ++row) {
        for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
            reduced[row][joint] = jacobian[row * IGRIS_C_DIM + joint];
        }
    }
    int pivot_columns[IGRIS_C_EQUALITY_CONSTRAINT_DIM];
    int rank = 0;
    for (int column = 0;
         column < IGRIS_C_DIM && rank < IGRIS_C_EQUALITY_CONSTRAINT_DIM;
         ++column) {
        int pivot = rank;
        float best = fabsf(reduced[rank][column]);
        for (int row = rank + 1;
             row < IGRIS_C_EQUALITY_CONSTRAINT_DIM;
             ++row) {
            const float value = fabsf(reduced[row][column]);
            if (value > best) {
                best = value;
                pivot = row;
            }
        }
        if (best < 1.0e-5f) {
            continue;
        }
        if (pivot != rank) {
            for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                const float temporary = reduced[rank][joint];
                reduced[rank][joint] = reduced[pivot][joint];
                reduced[pivot][joint] = temporary;
            }
        }
        const float inverse_pivot = 1.0f / reduced[rank][column];
        for (int joint = column; joint < IGRIS_C_DIM; ++joint) {
            reduced[rank][joint] *= inverse_pivot;
        }
        for (int row = 0; row < IGRIS_C_EQUALITY_CONSTRAINT_DIM; ++row) {
            if (row == rank) continue;
            const float factor = reduced[row][column];
            for (int joint = column; joint < IGRIS_C_DIM; ++joint) {
                reduced[row][joint] -= factor * reduced[rank][joint];
            }
        }
        pivot_columns[rank++] = column;
    }

    bool is_pivot[IGRIS_C_DIM];
    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) is_pivot[joint] = false;
    for (int row = 0; row < rank; ++row) is_pivot[pivot_columns[row]] = true;
    for (int index = 0; index < IGRIS_C_TANGENT_BASIS_SIZE; ++index) {
        basis[index] = 0.0f;
    }
    int basis_column = 0;
    for (int free_column = 0;
         free_column < IGRIS_C_DIM && basis_column < IGRIS_C_TANGENT_DIM;
         ++free_column) {
        if (is_pivot[free_column]) continue;
        float vector[IGRIS_C_DIM]{};
        vector[free_column] = 1.0f;
        for (int row = 0; row < rank; ++row) {
            vector[pivot_columns[row]] = -reduced[row][free_column];
        }
        for (int previous = 0; previous < basis_column; ++previous) {
            float dot = 0.0f;
            for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                dot += vector[joint] *
                    basis[joint * IGRIS_C_TANGENT_DIM + previous];
            }
            for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                vector[joint] -= dot *
                    basis[joint * IGRIS_C_TANGENT_DIM + previous];
            }
        }
        float squared = 0.0f;
        for (float value : vector) squared += value * value;
        if (squared < 1.0e-10f) continue;
        const float inverse_norm = rsqrtf(squared);
        for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
            basis[joint * IGRIS_C_TANGENT_DIM + basis_column] =
                vector[joint] * inverse_norm;
        }
        ++basis_column;
    }
    return rank == IGRIS_C_EQUALITY_CONSTRAINT_DIM &&
        basis_column == IGRIS_C_TANGENT_DIM;
}

__device__ __forceinline__ bool igris_c_tangent_basis(
    const float q[IGRIS_C_DIM],
    const constraints::IgrisCConstraintParameters &parameters,
    float basis[IGRIS_C_TANGENT_BASIS_SIZE]
) {
    float residual[IGRIS_C_EQUALITY_CONSTRAINT_DIM];
    float jacobian[IGRIS_C_EQUALITY_CONSTRAINT_DIM * IGRIS_C_DIM];
    igris_c_equality_residual_and_jacobian(
        q, parameters, residual, jacobian
    );
    return igris_c_tangent_basis_from_jacobian(jacobian, basis);
}

__device__ __forceinline__ float igris_c_constraint_error_squared(
    const float q[IGRIS_C_DIM],
    const constraints::IgrisCConstraintParameters &parameters
) {
    float residual[IGRIS_C_CONSTRAINT_DIM];
    igris_c_constraint_residual(q, parameters, residual);
    float result = 0.0f;
    for (float value : residual) result += value * value;
    return result;
}

__device__ __forceinline__ void igris_c_clamp_configuration(
    float q[IGRIS_C_DIM]
) {
    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
        const float lower = robots::IgrisC::get_s_a(joint);
        const float upper = lower + robots::IgrisC::get_s_m(joint);
        q[joint] = fminf(upper, fmaxf(lower, q[joint]));
    }
}

__device__ __forceinline__ bool igris_c_task_correction(
    const float q[IGRIS_C_DIM],
    const constraints::IgrisCConstraintParameters &parameters,
    float damping,
    float maximum_step,
    float correction[IGRIS_C_DIM],
    float &task_error_norm
) {
    float residual[IGRIS_C_CONSTRAINT_DIM];
    float jacobian[IGRIS_C_CONSTRAINT_DIM * IGRIS_C_DIM];
    igris_c_constraint_residual_and_jacobian(q, parameters, residual, jacobian);
    float squared = 0.0f;
    for (float value : residual) squared += value * value;
    task_error_norm = sqrtf(squared);

    float system[IGRIS_C_CONSTRAINT_DIM][IGRIS_C_CONSTRAINT_DIM + 1];
    for (int row = 0; row < IGRIS_C_CONSTRAINT_DIM; ++row) {
        for (int column = 0; column < IGRIS_C_CONSTRAINT_DIM; ++column) {
            float value = row == column ? damping : 0.0f;
            for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                value += jacobian[row * IGRIS_C_DIM + joint] *
                    jacobian[column * IGRIS_C_DIM + joint];
            }
            system[row][column] = value;
        }
        system[row][IGRIS_C_CONSTRAINT_DIM] = residual[row];
    }
    for (int pivot = 0; pivot < IGRIS_C_CONSTRAINT_DIM; ++pivot) {
        int best_row = pivot;
        float best_value = fabsf(system[pivot][pivot]);
        for (int row = pivot + 1; row < IGRIS_C_CONSTRAINT_DIM; ++row) {
            const float value = fabsf(system[row][pivot]);
            if (value > best_value) {
                best_value = value;
                best_row = row;
            }
        }
        if (best_value < 1.0e-12f) return false;
        if (best_row != pivot) {
            for (int column = pivot;
                 column <= IGRIS_C_CONSTRAINT_DIM;
                 ++column) {
                const float temporary = system[pivot][column];
                system[pivot][column] = system[best_row][column];
                system[best_row][column] = temporary;
            }
        }
        const float inverse_pivot = 1.0f / system[pivot][pivot];
        for (int column = pivot;
             column <= IGRIS_C_CONSTRAINT_DIM;
             ++column) {
            system[pivot][column] *= inverse_pivot;
        }
        for (int row = 0; row < IGRIS_C_CONSTRAINT_DIM; ++row) {
            if (row == pivot) continue;
            const float factor = system[row][pivot];
            for (int column = pivot;
                 column <= IGRIS_C_CONSTRAINT_DIM;
                 ++column) {
                system[row][column] -= factor * system[pivot][column];
            }
        }
    }
    float correction_squared = 0.0f;
    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
        float value = 0.0f;
        for (int row = 0; row < IGRIS_C_CONSTRAINT_DIM; ++row) {
            value += jacobian[row * IGRIS_C_DIM + joint] *
                system[row][IGRIS_C_CONSTRAINT_DIM];
        }
        correction[joint] = value;
        correction_squared += value * value;
    }
    const float correction_norm = sqrtf(correction_squared);
    if (maximum_step > 0.0f && correction_norm > maximum_step) {
        const float scale = maximum_step / correction_norm;
        for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
            correction[joint] *= scale;
        }
    }
    return true;
}

__device__ __noinline__ bool igris_c_project_configuration(
    float q[IGRIS_C_DIM],
    const constraints::IgrisCConstraintParameters &parameters,
    int max_iterations,
    float alpha,
    float damping,
    float maximum_step
) {
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        const float error = igris_c_constraint_error_squared(q, parameters);
        if (error <= parameters.tolerance_squared) return true;
        float correction[IGRIS_C_DIM];
        float task_error = 0.0f;
        if (!igris_c_task_correction(
                q, parameters, damping, maximum_step, correction, task_error
            )) {
            return false;
        }
        float scale = alpha;
        bool accepted = false;
        for (int line_search = 0; line_search < 8; ++line_search) {
            float candidate[IGRIS_C_DIM];
            for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                candidate[joint] = q[joint] - scale * correction[joint];
            }
            igris_c_clamp_configuration(candidate);
            if (igris_c_constraint_error_squared(candidate, parameters) < error) {
                for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                    q[joint] = candidate[joint];
                }
                accepted = true;
                break;
            }
            scale *= 0.5f;
        }
        if (!accepted) return false;
    }
    return igris_c_constraint_error_squared(q, parameters) <=
        parameters.tolerance_squared;
}

// ParallelProject kernel body used by the robot adapter. `smoothness_threshold`
// is explicit so node anchors can use the cpRRTC_TB-RRT policy
// granularity * projection_smoothness_threshold without changing this robot code.
__device__ __forceinline__ bool igris_c_project_motion(
    volatile float *motion_segment,
    volatile float *motion_segment_next,
    int granularity,
    const constraints::IgrisCConstraintParameters &parameters,
    volatile unsigned char *projection_valid,
    volatile int *projection_progress,
    volatile unsigned int *projection_success,
    int max_iterations,
    float alpha,
    float beta,
    float gamma,
    float damping,
    float task_tolerance,
    float smoothness_threshold,
    float smoothness_weight,
    bool use_smoothness,
    float maximum_step,
    int tid,
    bool return_when_success = true
) {
    (void)beta;
    (void)gamma;
    const int waypoint = tid / 4 + 1;
    const int lane = tid % 4;
    if (tid == 0) {
        projection_progress[0] = 0;
        projection_success[0] = 0;
        projection_valid[0] = 1;
    }
    if (tid < IGRIS_C_DIM) motion_segment_next[tid] = motion_segment[tid];
    if (waypoint <= granularity && lane == 0) projection_valid[waypoint] = 0;
    __syncthreads();

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        if (projection_success[0] == 0 && waypoint <= granularity) {
            const int progress = projection_progress[0];
            if (waypoint > progress) {
                if (lane == 0) {
                    float q[IGRIS_C_DIM];
                    float correction[IGRIS_C_DIM]{};
                    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                        q[joint] = motion_segment[
                            waypoint * IGRIS_C_DIM + joint
                        ];
                    }
                    float task_error = 1.0e30f;
                    const bool correction_ok = igris_c_task_correction(
                        q,
                        parameters,
                        damping,
                        maximum_step,
                        correction,
                        task_error
                    );
                    float difference[IGRIS_C_DIM];
                    float distance_squared = 0.0f;
                    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                        difference[joint] = q[joint] - motion_segment[
                            (waypoint - 1) * IGRIS_C_DIM + joint
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
                    float combined[IGRIS_C_DIM];
                    float combined_squared = 0.0f;
                    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
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
                    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                        q[joint] -= combined_scale * combined[joint];
                    }
                    igris_c_clamp_configuration(q);
                    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
                        motion_segment_next[
                            waypoint * IGRIS_C_DIM + joint
                        ] = q[joint];
                    }
                    projection_valid[waypoint] = correction_ok &&
                        task_error < task_tolerance &&
                        (!use_smoothness || distance <= smoothness_threshold);
                }
            } else {
                if (lane == 0) projection_valid[waypoint] = 1;
                for (int joint = lane; joint < IGRIS_C_DIM; joint += 4) {
                    motion_segment_next[waypoint * IGRIS_C_DIM + joint] =
                        motion_segment[waypoint * IGRIS_C_DIM + joint];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            int progress = projection_progress[0];
            while (
                progress + 1 <= granularity &&
                projection_valid[progress + 1] != 0
            ) {
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
            for (int joint = lane; joint < IGRIS_C_DIM; joint += 4) {
                motion_segment[waypoint * IGRIS_C_DIM + joint] =
                    motion_segment_next[waypoint * IGRIS_C_DIM + joint];
            }
        }
        __syncthreads();
    }
    if (projection_success[0] == 0) return false;
    if (tid == 0) {
        projection_valid[0] =
            planning::configuration_within_joint_limits<robots::IgrisC>(
                &motion_segment[granularity * IGRIS_C_DIM]
            );
    }
    __syncthreads();
    return projection_valid[0] != 0;
}

}  // namespace ppln::collision

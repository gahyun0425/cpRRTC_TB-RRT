#pragma once

#include "src/planning/G1ConstraintParameters.hh"
#include "src/planning/Robots.hh"
#include "src/robots/g1_kinematics.cuh"

namespace ppln::collision {

constexpr int G1_JOINT_DIM = 35;

constexpr int G1_FEET_JOINT_DIM = 18;
constexpr int G1_FEET_POSITION_DIM = 3;
constexpr int G1_FEET_POSE_DIM = 6;
constexpr int G1_FEET_CONSTRAINT_DIM = 12;
constexpr int G1_COM_CONSTRAINT_DIM = 2;
constexpr int G1_BIMANUAL_JOINT_DIM = 23;
constexpr int G1_BIMANUAL_POSITION_DIM = 3;
constexpr int G1_BIMANUAL_ORIENTATION_DIM = 3;
constexpr int G1_BIMANUAL_CONSTRAINT_DIM =
    G1_BIMANUAL_POSITION_DIM + G1_BIMANUAL_ORIENTATION_DIM;

constexpr int G1_CONSTRAINT_DIM = G1_FEET_CONSTRAINT_DIM +
    G1_COM_CONSTRAINT_DIM + G1_BIMANUAL_CONSTRAINT_DIM;
constexpr int G1_EQUALITY_CONSTRAINT_DIM =
    G1_FEET_CONSTRAINT_DIM + G1_BIMANUAL_CONSTRAINT_DIM;
constexpr int G1_TANGENT_DIM = G1_JOINT_DIM - G1_EQUALITY_CONSTRAINT_DIM;
constexpr int G1_TANGENT_BASIS_SIZE = G1_JOINT_DIM * G1_TANGENT_DIM;

__device__ __forceinline__ void g1_identity3(float matrix[9]) {
    for (int index = 0; index < 9; ++index) {
        matrix[index] = index % 4 == 0 ? 1.0f : 0.0f;
    }
}

__device__ __forceinline__ void g1_mul3(
    const float left[9],
    const float right[9],
    float result[9]
) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result[row * 3 + column] =
                left[row * 3] * right[column] +
                left[row * 3 + 1] * right[3 + column] +
                left[row * 3 + 2] * right[6 + column];
        }
    }
}

__device__ __forceinline__ void g1_mul3_at_b(
    const float left[9],
    const float right[9],
    float result[9]
) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result[row * 3 + column] =
                left[row] * right[column] +
                left[3 + row] * right[3 + column] +
                left[6 + row] * right[6 + column];
        }
    }
}

__device__ __forceinline__ void g1_apply_axis_rotation(
    float rotation[9],
    int axis,
    float angle
) {
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    float step[9];
    g1_identity3(step);
    if (axis == 0) {
        step[4] = cosine;
        step[5] = -sine;
        step[7] = sine;
        step[8] = cosine;
    } else if (axis == 1) {
        step[0] = cosine;
        step[2] = sine;
        step[6] = -sine;
        step[8] = cosine;
    } else {
        step[0] = cosine;
        step[1] = -sine;
        step[3] = sine;
        step[4] = cosine;
    }
    float result[9];
    g1_mul3(rotation, step, result);
    for (int index = 0; index < 9; ++index) {
        rotation[index] = result[index];
    }
}

__device__ __forceinline__ void g1_apply_rpy_rotation(
    float rotation[9],
    float roll,
    float pitch,
    float yaw
) {
    // URDF origin rotations use Rz(yaw) * Ry(pitch) * Rx(roll).
    g1_apply_axis_rotation(rotation, 2, yaw);
    g1_apply_axis_rotation(rotation, 1, pitch);
    g1_apply_axis_rotation(rotation, 0, roll);
}

__device__ __forceinline__ void g1_record_world_axis(
    const float rotation[9],
    int axis,
    int joint,
    float jacobian[3 * G1_FEET_JOINT_DIM]
) {
    for (int component = 0; component < 3; ++component) {
        jacobian[component * G1_FEET_JOINT_DIM + joint] =
            rotation[component * 3 + axis];
    }
}

__device__ __forceinline__ void g1_foot_rotation_and_world_jacobian(
    const float q[G1_JOINT_DIM],
    int foot,
    float rotation[9],
    float jacobian[3 * G1_FEET_JOINT_DIM]
) {
    g1_identity3(rotation);
    for (int index = 0; index < 3 * G1_FEET_JOINT_DIM; ++index) {
        jacobian[index] = 0.0f;
    }

    g1_record_world_axis(rotation, 0, 3, jacobian);
    g1_apply_axis_rotation(rotation, 0, q[3]);
    g1_record_world_axis(rotation, 1, 4, jacobian);
    g1_apply_axis_rotation(rotation, 1, q[4]);
    g1_record_world_axis(rotation, 2, 5, jacobian);
    g1_apply_axis_rotation(rotation, 2, q[5]);

    const int leg = foot == 0 ? 6 : 12;
    g1_record_world_axis(rotation, 1, leg, jacobian);
    g1_apply_axis_rotation(rotation, 1, q[leg]);

    // Fixed hip-roll frame rotation from the G1 URDF.
    g1_apply_axis_rotation(rotation, 1, -0.1749f);
    g1_record_world_axis(rotation, 0, leg + 1, jacobian);
    g1_apply_axis_rotation(rotation, 0, q[leg + 1]);

    g1_record_world_axis(rotation, 2, leg + 2, jacobian);
    g1_apply_axis_rotation(rotation, 2, q[leg + 2]);

    // Fixed knee frame rotation cancels the hip frame offset at zero pose.
    g1_apply_axis_rotation(rotation, 1, 0.1749f);
    g1_record_world_axis(rotation, 1, leg + 3, jacobian);
    g1_apply_axis_rotation(rotation, 1, q[leg + 3]);

    g1_record_world_axis(rotation, 1, leg + 4, jacobian);
    g1_apply_axis_rotation(rotation, 1, q[leg + 4]);
    g1_record_world_axis(rotation, 0, leg + 5, jacobian);
    g1_apply_axis_rotation(rotation, 0, q[leg + 5]);
}

__device__ __forceinline__ void g1_record_world_axis_full(
    const float rotation[9],
    int axis,
    int joint,
    float jacobian[3 * G1_JOINT_DIM]
) {
    for (int component = 0; component < 3; ++component) {
        jacobian[component * G1_JOINT_DIM + joint] =
            rotation[component * 3 + axis];
    }
}

__device__ __forceinline__ void g1_hand_rotation_and_world_jacobian(
    const float q[G1_JOINT_DIM],
    int hand,
    float rotation[9],
    float jacobian[3 * G1_JOINT_DIM]
) {
    g1_identity3(rotation);
    for (int index = 0; index < 3 * G1_JOINT_DIM; ++index) {
        jacobian[index] = 0.0f;
    }

    g1_record_world_axis_full(rotation, 0, 3, jacobian);
    g1_apply_axis_rotation(rotation, 0, q[3]);
    g1_record_world_axis_full(rotation, 1, 4, jacobian);
    g1_apply_axis_rotation(rotation, 1, q[4]);
    g1_record_world_axis_full(rotation, 2, 5, jacobian);
    g1_apply_axis_rotation(rotation, 2, q[5]);

    g1_record_world_axis_full(rotation, 2, 18, jacobian);
    g1_apply_axis_rotation(rotation, 2, q[18]);
    g1_record_world_axis_full(rotation, 0, 19, jacobian);
    g1_apply_axis_rotation(rotation, 0, q[19]);
    g1_record_world_axis_full(rotation, 1, 20, jacobian);
    g1_apply_axis_rotation(rotation, 1, q[20]);

    const bool left = hand == 0;
    const int arm = left ? 21 : 28;
    g1_apply_rpy_rotation(
        rotation,
        left ? 0.27931f : -0.27931f,
        5.4949e-05f,
        left ? -0.00019159f : 0.00019159f
    );
    g1_record_world_axis_full(rotation, 1, arm, jacobian);
    g1_apply_axis_rotation(rotation, 1, q[arm]);

    g1_apply_rpy_rotation(
        rotation,
        left ? -0.27925f : 0.27925f,
        0.0f,
        0.0f
    );
    g1_record_world_axis_full(rotation, 0, arm + 1, jacobian);
    g1_apply_axis_rotation(rotation, 0, q[arm + 1]);

    g1_record_world_axis_full(rotation, 2, arm + 2, jacobian);
    g1_apply_axis_rotation(rotation, 2, q[arm + 2]);
    g1_record_world_axis_full(rotation, 1, arm + 3, jacobian);
    g1_apply_axis_rotation(rotation, 1, q[arm + 3]);
    g1_record_world_axis_full(rotation, 0, arm + 4, jacobian);
    g1_apply_axis_rotation(rotation, 0, q[arm + 4]);
    g1_record_world_axis_full(rotation, 1, arm + 5, jacobian);
    g1_apply_axis_rotation(rotation, 1, q[arm + 5]);
    g1_record_world_axis_full(rotation, 2, arm + 6, jacobian);
    g1_apply_axis_rotation(rotation, 2, q[arm + 6]);
}

__device__ __forceinline__ void g1_quaternion_to_matrix(
    const float quaternion[7],
    float rotation[9]
) {
    const float inverse_norm = 1.0f / fmaxf(sqrtf(
        quaternion[0] * quaternion[0] +
        quaternion[1] * quaternion[1] +
        quaternion[2] * quaternion[2] +
        quaternion[3] * quaternion[3]
    ), 1.0e-12f);
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

// Same stable atan2-based SO(3) logarithm used by FFW SG2.
__device__ __forceinline__ void g1_so3_log(
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
        omega[0] = axis[0] * scale;
        omega[1] = axis[1] * scale;
        omega[2] = axis[2] * scale;
        return;
    }
    const float scale = angle / fmaxf(sine, 1.0e-12f);
    omega[0] = vee[0] * scale;
    omega[1] = vee[1] * scale;
    omega[2] = vee[2] * scale;
}

__device__ __forceinline__ void g1_so3_right_jacobian_inverse(
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
    g1_mul3(skew, skew, skew_squared);
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

__device__ __forceinline__ void g1_foot_orientation_residual_and_jacobian(
    const float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters,
    int foot,
    float residual[3],
    float jacobian[3 * G1_FEET_JOINT_DIM]
) {
    float actual_rotation[9];
    float world_jacobian[3 * G1_FEET_JOINT_DIM];
    g1_foot_rotation_and_world_jacobian(
        q,
        foot,
        actual_rotation,
        world_jacobian
    );

    float reference_rotation[9];
    float target_rotation[9];
    float desired_rotation[9];
    g1_quaternion_to_matrix(parameters.feet_reference[foot], reference_rotation);
    g1_quaternion_to_matrix(parameters.feet_target[foot], target_rotation);
    g1_mul3(target_rotation, reference_rotation, desired_rotation);

    float error_rotation[9];
    g1_mul3_at_b(desired_rotation, actual_rotation, error_rotation);
    g1_so3_log(error_rotation, residual);

    float inverse_right_jacobian[9];
    g1_so3_right_jacobian_inverse(residual, inverse_right_jacobian);
    for (int joint = 0; joint < G1_FEET_JOINT_DIM; ++joint) {
        float body_axis[3] = {0.0f, 0.0f, 0.0f};
        for (int row = 0; row < 3; ++row) {
            for (int axis = 0; axis < 3; ++axis) {
                body_axis[row] += actual_rotation[axis * 3 + row] *
                    world_jacobian[axis * G1_FEET_JOINT_DIM + joint];
            }
        }
        for (int row = 0; row < 3; ++row) {
            jacobian[row * G1_FEET_JOINT_DIM + joint] =
                inverse_right_jacobian[row * 3] * body_axis[0] +
                inverse_right_jacobian[row * 3 + 1] * body_axis[1] +
                inverse_right_jacobian[row * 3 + 2] * body_axis[2];
        }
    }
}

__device__ __forceinline__ void g1_bimanual_orientation_residual_and_jacobian(
    const float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters,
    float residual[G1_BIMANUAL_ORIENTATION_DIM],
    float jacobian[G1_BIMANUAL_ORIENTATION_DIM * G1_JOINT_DIM]
) {
    float left_rotation[9];
    float right_rotation[9];
    float left_world_jacobian[3 * G1_JOINT_DIM];
    float right_world_jacobian[3 * G1_JOINT_DIM];
    g1_hand_rotation_and_world_jacobian(
        q,
        0,
        left_rotation,
        left_world_jacobian
    );
    g1_hand_rotation_and_world_jacobian(
        q,
        1,
        right_rotation,
        right_world_jacobian
    );

    // Target and actual values are R_left^T * R_right: the right hand
    // orientation expressed in the left hand frame.
    float actual_relative_rotation[9];
    float target_relative_rotation[9];
    float error_rotation[9];
    g1_mul3_at_b(left_rotation, right_rotation, actual_relative_rotation);
    g1_quaternion_to_matrix(
        parameters.bimanual_target,
        target_relative_rotation
    );
    g1_mul3_at_b(
        target_relative_rotation,
        actual_relative_rotation,
        error_rotation
    );
    g1_so3_log(error_rotation, residual);

    float inverse_right_jacobian[9];
    g1_so3_right_jacobian_inverse(residual, inverse_right_jacobian);
    for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
        float relative_body_axis[3] = {0.0f, 0.0f, 0.0f};
        for (int row = 0; row < 3; ++row) {
            for (int world_axis = 0; world_axis < 3; ++world_axis) {
                relative_body_axis[row] +=
                    right_rotation[world_axis * 3 + row] * (
                        right_world_jacobian[
                            world_axis * G1_JOINT_DIM + joint
                        ] -
                        left_world_jacobian[
                            world_axis * G1_JOINT_DIM + joint
                        ]
                    );
            }
        }
        for (int row = 0; row < G1_BIMANUAL_ORIENTATION_DIM; ++row) {
            jacobian[row * G1_JOINT_DIM + joint] =
                inverse_right_jacobian[row * 3] * relative_body_axis[0] +
                inverse_right_jacobian[row * 3 + 1] * relative_body_axis[1] +
                inverse_right_jacobian[row * 3 + 2] * relative_body_axis[2];
        }
    }
}

// feet analytic 함수에 넣을 입력 데이터를 한 줄짜리 배열로 포장하는 함수
__device__ __forceinline__ void g1_fill_feet_input(
    const float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters,
    float input[46]
) {
    for (int joint = 0; joint < G1_FEET_JOINT_DIM; ++joint) {
        input[joint] = q[joint];
    }
    for (int foot = 0; foot < 2; ++foot) {
        const int input_offset = G1_FEET_JOINT_DIM + foot * 14;
        for (int component = 0; component < 7; ++component) {
            input[input_offset + component] = parameters.feet_reference[foot][component];
            input[input_offset + 7 + component] = parameters.feet_target[foot][component];
        }
    }
}

__device__ __forceinline__ void g1_fill_bimanual_input(
    const float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters,
    float input[30]
) {
    for (int joint = 0; joint < 6; ++joint) {
        input[joint] = q[joint];
    }
    for (int joint = 18; joint < G1_JOINT_DIM; ++joint) {
        input[6 + joint - 18] = q[joint];
    }
    for (int component = 0; component < 7; ++component) {
        input[G1_BIMANUAL_JOINT_DIM + component] = parameters.bimanual_target[component];
    }
}

// Build one combined 20x35 system: feet pose equality[12], CoM inequality[2],
// and bimanual relative-pose equality[6].
__device__ __noinline__ void g1_constraint_residual_and_jacobian(
    const float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters,
    float residual[G1_CONSTRAINT_DIM],
    float *jacobian
) {
    float feet_input[46];
    constexpr int feet_position_constraint_dim = 2 * G1_FEET_POSITION_DIM;
    float feet_output[
        feet_position_constraint_dim * (G1_FEET_JOINT_DIM + 1)
    ];
    g1_fill_feet_input(q, parameters, feet_input);
    g1_feet_position_error_analytic(feet_input, feet_output);

    for (int foot = 0; foot < 2; ++foot) {
        for (int component = 0; component < G1_FEET_POSITION_DIM; ++component) {
            const int compact_row = foot * G1_FEET_POSITION_DIM + component;
            const int row = foot * G1_FEET_POSE_DIM + component;
            residual[row] = feet_output[
                feet_position_constraint_dim * G1_FEET_JOINT_DIM +
                compact_row
            ];
            if (jacobian != nullptr) {
                for (int joint = 0; joint < G1_FEET_JOINT_DIM; ++joint) {
                    jacobian[row * G1_JOINT_DIM + joint] = feet_output[
                        compact_row * G1_FEET_JOINT_DIM + joint
                    ];
                }
                for (int joint = G1_FEET_JOINT_DIM; joint < G1_JOINT_DIM; ++joint) {
                    jacobian[row * G1_JOINT_DIM + joint] = 0.0f;
                }
            }
        }

        float orientation_residual[3];
        float orientation_jacobian[3 * G1_FEET_JOINT_DIM];
        g1_foot_orientation_residual_and_jacobian(
            q,
            parameters,
            foot,
            orientation_residual,
            orientation_jacobian
        );
        for (int component = 0; component < 3; ++component) {
            const int row = foot * G1_FEET_POSE_DIM +
                G1_FEET_POSITION_DIM + component;
            residual[row] = orientation_residual[component];
            if (jacobian != nullptr) {
                for (int joint = 0; joint < G1_FEET_JOINT_DIM; ++joint) {
                    jacobian[row * G1_JOINT_DIM + joint] =
                        orientation_jacobian[
                            component * G1_FEET_JOINT_DIM + joint
                        ];
                }
                for (int joint = G1_FEET_JOINT_DIM; joint < G1_JOINT_DIM; ++joint) {
                    jacobian[row * G1_JOINT_DIM + joint] = 0.0f;
                }
            }
        }
    }

    float com_output[108];
    g1_com_kinematics_analytic(q, com_output);
    residual[G1_FEET_CONSTRAINT_DIM] = 0.0f;
    residual[G1_FEET_CONSTRAINT_DIM + 1] = 0.0f;
    float com_error_jacobian[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Scalar form of VAMP G1Unitree::com_constraint_error.
    for (int edge = 0; edge < 4; ++edge) {
        const int next = (edge + 1) % 4;
        const float x0 = parameters.support_polygon[2 * edge];
        const float y0 = parameters.support_polygon[2 * edge + 1];
        const float x1 = parameters.support_polygon[2 * next];
        const float y1 = parameters.support_polygon[2 * next + 1];
        const float normal_x = y1 - y0;
        const float normal_y = x0 - x1;
        const float length = sqrtf(normal_x * normal_x + normal_y * normal_y);
        const float unit_x = normal_x / length;
        const float unit_y = normal_y / length;
        const float signed_distance =unit_x * (com_output[0] - x0) + unit_y * (com_output[1] - y0);
        const float active = signed_distance > 0.0f ? 1.0f : 0.0f;

        residual[G1_FEET_CONSTRAINT_DIM] += signed_distance * unit_x * active;
        residual[G1_FEET_CONSTRAINT_DIM + 1] += signed_distance * unit_y * active;
        com_error_jacobian[0] += unit_x * unit_x * active;
        com_error_jacobian[1] += unit_x * unit_y * active;
        com_error_jacobian[3] += unit_y * unit_x * active;
        com_error_jacobian[4] += unit_y * unit_y * active;
    }

    if (jacobian != nullptr) {
        for (int output_axis = 0; output_axis < 2; ++output_axis) {
            const int row = G1_FEET_CONSTRAINT_DIM + output_axis;
            for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                float value = 0.0f;
                for (int com_axis = 0; com_axis < 3; ++com_axis) {
                    value += com_error_jacobian[output_axis * 3 + com_axis] * com_output[3 + com_axis * G1_JOINT_DIM + joint];
                }
                jacobian[row * G1_JOINT_DIM + joint] = value;
            }
        }
    }

    float bimanual_input[30];
    float bimanual_output[G1_BIMANUAL_POSITION_DIM * (G1_BIMANUAL_JOINT_DIM + 1)];
    g1_fill_bimanual_input(q, parameters, bimanual_input);
    g1_bimanual_position_error_analytic(bimanual_input, bimanual_output);
    for (int component = 0; component < G1_BIMANUAL_POSITION_DIM; ++component) {
        const int row = G1_FEET_CONSTRAINT_DIM + G1_COM_CONSTRAINT_DIM + component;
        residual[row] = bimanual_output[G1_BIMANUAL_POSITION_DIM * G1_BIMANUAL_JOINT_DIM + component];
        if (jacobian != nullptr) {
            for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                jacobian[row * G1_JOINT_DIM + joint] = 0.0f;
            }
            for (int compact_joint = 0; compact_joint < G1_BIMANUAL_JOINT_DIM; ++compact_joint) {
                const int joint = compact_joint < 6? compact_joint: compact_joint + 12;
                jacobian[row * G1_JOINT_DIM + joint] = bimanual_output[component * G1_BIMANUAL_JOINT_DIM + compact_joint];
            }
        }
    }

    float bimanual_orientation_residual[G1_BIMANUAL_ORIENTATION_DIM];
    float bimanual_orientation_jacobian[
        G1_BIMANUAL_ORIENTATION_DIM * G1_JOINT_DIM
    ];
    g1_bimanual_orientation_residual_and_jacobian(
        q,
        parameters,
        bimanual_orientation_residual,
        bimanual_orientation_jacobian
    );
    for (int component = 0;
         component < G1_BIMANUAL_ORIENTATION_DIM;
         ++component) {
        const int row = G1_FEET_CONSTRAINT_DIM + G1_COM_CONSTRAINT_DIM +
            G1_BIMANUAL_POSITION_DIM + component;
        residual[row] = bimanual_orientation_residual[component];
        if (jacobian != nullptr) {
            for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                jacobian[row * G1_JOINT_DIM + joint] =
                    bimanual_orientation_jacobian[
                        component * G1_JOINT_DIM + joint
                    ];
            }
        }
    }
}

__device__ __forceinline__ void g1_constraint_residual(const float q[G1_JOINT_DIM], const constraints::G1ConstraintParameters &parameters, float residual[G1_CONSTRAINT_DIM]) {
    g1_constraint_residual_and_jacobian(q, parameters, residual, nullptr);
}

__device__ __forceinline__ void g1_equality_residual_and_jacobian(
    const float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters,
    float residual[G1_EQUALITY_CONSTRAINT_DIM],
    float *jacobian
) {
    float feet_input[46];
    constexpr int feet_position_constraint_dim =
        2 * G1_FEET_POSITION_DIM;
    float feet_output[
        feet_position_constraint_dim * (G1_FEET_JOINT_DIM + 1)
    ];
    g1_fill_feet_input(q, parameters, feet_input);
    g1_feet_position_error_analytic(feet_input, feet_output);

    for (int foot = 0; foot < 2; ++foot) {
        for (int component = 0;
             component < G1_FEET_POSITION_DIM;
             ++component) {
            const int compact_row =
                foot * G1_FEET_POSITION_DIM + component;
            const int row = foot * G1_FEET_POSE_DIM + component;
            residual[row] = feet_output[
                feet_position_constraint_dim * G1_FEET_JOINT_DIM +
                compact_row
            ];
            if (jacobian != nullptr) {
                for (int joint = 0;
                     joint < G1_FEET_JOINT_DIM;
                     ++joint) {
                    jacobian[row * G1_JOINT_DIM + joint] =
                        feet_output[
                            compact_row * G1_FEET_JOINT_DIM + joint
                        ];
                }
                for (int joint = G1_FEET_JOINT_DIM;
                     joint < G1_JOINT_DIM;
                     ++joint) {
                    jacobian[row * G1_JOINT_DIM + joint] = 0.0f;
                }
            }
        }

        float orientation_residual[3];
        float orientation_jacobian[3 * G1_FEET_JOINT_DIM];
        g1_foot_orientation_residual_and_jacobian(
            q,
            parameters,
            foot,
            orientation_residual,
            orientation_jacobian
        );
        for (int component = 0; component < 3; ++component) {
            const int row = foot * G1_FEET_POSE_DIM +
                G1_FEET_POSITION_DIM + component;
            residual[row] = orientation_residual[component];
            if (jacobian != nullptr) {
                for (int joint = 0;
                     joint < G1_FEET_JOINT_DIM;
                     ++joint) {
                    jacobian[row * G1_JOINT_DIM + joint] =
                        orientation_jacobian[
                            component * G1_FEET_JOINT_DIM + joint
                        ];
                }
                for (int joint = G1_FEET_JOINT_DIM;
                     joint < G1_JOINT_DIM;
                     ++joint) {
                    jacobian[row * G1_JOINT_DIM + joint] = 0.0f;
                }
            }
        }
    }

    float bimanual_input[30];
    float bimanual_output[
        G1_BIMANUAL_POSITION_DIM * (G1_BIMANUAL_JOINT_DIM + 1)
    ];
    g1_fill_bimanual_input(q, parameters, bimanual_input);
    g1_bimanual_position_error_analytic(
        bimanual_input,
        bimanual_output
    );
    for (int component = 0;
         component < G1_BIMANUAL_POSITION_DIM;
         ++component) {
        const int equality_row = G1_FEET_CONSTRAINT_DIM + component;
        residual[equality_row] = bimanual_output[
            G1_BIMANUAL_POSITION_DIM * G1_BIMANUAL_JOINT_DIM +
            component
        ];
        if (jacobian != nullptr) {
            for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                jacobian[equality_row * G1_JOINT_DIM + joint] = 0.0f;
            }
            for (int compact_joint = 0;
                 compact_joint < G1_BIMANUAL_JOINT_DIM;
                 ++compact_joint) {
                const int joint = compact_joint < 6
                    ? compact_joint
                    : compact_joint + 12;
                jacobian[equality_row * G1_JOINT_DIM + joint] =
                    bimanual_output[
                        component * G1_BIMANUAL_JOINT_DIM +
                        compact_joint
                    ];
            }
        }
    }

    float bimanual_orientation_residual[G1_BIMANUAL_ORIENTATION_DIM];
    float bimanual_orientation_jacobian[
        G1_BIMANUAL_ORIENTATION_DIM * G1_JOINT_DIM
    ];
    g1_bimanual_orientation_residual_and_jacobian(
        q,
        parameters,
        bimanual_orientation_residual,
        bimanual_orientation_jacobian
    );
    for (int component = 0;
         component < G1_BIMANUAL_ORIENTATION_DIM;
         ++component) {
        const int equality_row = G1_FEET_CONSTRAINT_DIM +
            G1_BIMANUAL_POSITION_DIM + component;
        residual[equality_row] = bimanual_orientation_residual[component];
        if (jacobian != nullptr) {
            for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                jacobian[equality_row * G1_JOINT_DIM + joint] =
                    bimanual_orientation_jacobian[
                        component * G1_JOINT_DIM + joint
                    ];
            }
        }
    }
}

__device__ __forceinline__ void g1_equality_residual(
    const float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters,
    float residual[G1_EQUALITY_CONSTRAINT_DIM]
) {
    g1_equality_residual_and_jacobian(q, parameters, residual, nullptr);
}

__device__ __forceinline__ float g1_equality_residual_norm(
    const float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters
) {
    float residual[G1_EQUALITY_CONSTRAINT_DIM];
    g1_equality_residual(q, parameters, residual);
    float norm_squared = 0.0f;
    for (int row = 0; row < G1_EQUALITY_CONSTRAINT_DIM; ++row) {
        norm_squared += residual[row] * residual[row];
    }
    return sqrtf(norm_squared);
}

__device__ __forceinline__ bool g1_tangent_basis_from_jacobian(
    const float jacobian[G1_EQUALITY_CONSTRAINT_DIM * G1_JOINT_DIM],
    float basis[G1_TANGENT_BASIS_SIZE]
) {
    float reduced[G1_EQUALITY_CONSTRAINT_DIM][G1_JOINT_DIM];
    for (int row = 0; row < G1_EQUALITY_CONSTRAINT_DIM; ++row) {
        for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
            reduced[row][joint] =
                jacobian[row * G1_JOINT_DIM + joint];
        }
    }

    int pivot_columns[G1_EQUALITY_CONSTRAINT_DIM];
    int rank = 0;
    for (int column = 0;
         column < G1_JOINT_DIM && rank < G1_EQUALITY_CONSTRAINT_DIM;
         ++column) {
        int pivot = rank;
        float best = fabsf(reduced[rank][column]);
        for (int row = rank + 1;
             row < G1_EQUALITY_CONSTRAINT_DIM;
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
            for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                const float temporary = reduced[rank][joint];
                reduced[rank][joint] = reduced[pivot][joint];
                reduced[pivot][joint] = temporary;
            }
        }

        const float inverse_pivot = 1.0f / reduced[rank][column];
        for (int joint = column; joint < G1_JOINT_DIM; ++joint) {
            reduced[rank][joint] *= inverse_pivot;
        }
        for (int row = 0; row < G1_EQUALITY_CONSTRAINT_DIM; ++row) {
            if (row == rank) {
                continue;
            }
            const float factor = reduced[row][column];
            for (int joint = column; joint < G1_JOINT_DIM; ++joint) {
                reduced[row][joint] -= factor * reduced[rank][joint];
            }
        }
        pivot_columns[rank] = column;
        ++rank;
    }

    bool is_pivot[G1_JOINT_DIM];
    for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
        is_pivot[joint] = false;
    }
    for (int row = 0; row < rank; ++row) {
        is_pivot[pivot_columns[row]] = true;
    }
    for (int index = 0; index < G1_TANGENT_BASIS_SIZE; ++index) {
        basis[index] = 0.0f;
    }

    int basis_column = 0;
    for (int free_column = 0;
         free_column < G1_JOINT_DIM && basis_column < G1_TANGENT_DIM;
         ++free_column) {
        if (is_pivot[free_column]) {
            continue;
        }

        float vector[G1_JOINT_DIM];
        for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
            vector[joint] = 0.0f;
        }
        vector[free_column] = 1.0f;
        for (int row = 0; row < rank; ++row) {
            vector[pivot_columns[row]] = -reduced[row][free_column];
        }

        for (int previous = 0; previous < basis_column; ++previous) {
            float dot = 0.0f;
            for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                dot += vector[joint] *
                    basis[joint * G1_TANGENT_DIM + previous];
            }
            for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                vector[joint] -= dot *
                    basis[joint * G1_TANGENT_DIM + previous];
            }
        }

        float norm_squared = 0.0f;
        for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
            norm_squared += vector[joint] * vector[joint];
        }
        if (norm_squared < 1.0e-10f) {
            continue;
        }
        const float inverse_norm = rsqrtf(norm_squared);
        for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
            basis[joint * G1_TANGENT_DIM + basis_column] =
                vector[joint] * inverse_norm;
        }
        ++basis_column;
    }

    return rank == G1_EQUALITY_CONSTRAINT_DIM &&
        basis_column == G1_TANGENT_DIM;
}

__device__ __forceinline__ bool g1_tangent_basis(
    const float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters,
    float basis[G1_TANGENT_BASIS_SIZE]
) {
    float residual[G1_EQUALITY_CONSTRAINT_DIM];
    float jacobian[G1_EQUALITY_CONSTRAINT_DIM * G1_JOINT_DIM];
    g1_equality_residual_and_jacobian(
        q,
        parameters,
        residual,
        jacobian
    );
    return g1_tangent_basis_from_jacobian(jacobian, basis);
}

__device__ __forceinline__ float g1_constraint_error_squared(
    const float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters
) {
    float residual[G1_CONSTRAINT_DIM];
    g1_constraint_residual(q, parameters, residual);
    float result = 0.0f;
    for (int row = 0; row < G1_CONSTRAINT_DIM; ++row) {
        result += residual[row] * residual[row];
    }
    return result;
}

__device__ __forceinline__ void g1_clamp_configuration(
    float q[G1_JOINT_DIM]
) {
    for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
        const float lower = robots::G1::get_s_a(joint);
        const float upper = lower + robots::G1::get_s_m(joint);
        q[joint] = fminf(upper, fmaxf(lower, q[joint]));
    }
}

// Compute one damped-least-squares task correction. This is the G1
// counterpart of ffw_sg2_task_correction used by cpRRTC ParallelProject.
__device__ __forceinline__ bool g1_task_correction(
    const float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters,
    float damping,
    float maximum_step,
    float correction[G1_JOINT_DIM],
    float &task_error_norm
) {
    float residual[G1_CONSTRAINT_DIM];
    float jacobian[G1_CONSTRAINT_DIM * G1_JOINT_DIM];
    g1_constraint_residual_and_jacobian(q, parameters, residual, jacobian);

    float error_squared = 0.0f;
    for (int row = 0; row < G1_CONSTRAINT_DIM; ++row) {
        error_squared += residual[row] * residual[row];
    }
    task_error_norm = sqrtf(error_squared);

    float system[G1_CONSTRAINT_DIM][G1_CONSTRAINT_DIM + 1];
    for (int row = 0; row < G1_CONSTRAINT_DIM; ++row) {
        for (int column = 0; column < G1_CONSTRAINT_DIM; ++column) {
            float value = row == column ? damping : 0.0f;
            for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                value += jacobian[row * G1_JOINT_DIM + joint] *
                    jacobian[column * G1_JOINT_DIM + joint];
            }
            system[row][column] = value;
        }
        system[row][G1_CONSTRAINT_DIM] = residual[row];
    }

    for (int pivot = 0; pivot < G1_CONSTRAINT_DIM; ++pivot) {
        int best_row = pivot;
        float best_value = fabsf(system[pivot][pivot]);
        for (int row = pivot + 1; row < G1_CONSTRAINT_DIM; ++row) {
            const float value = fabsf(system[row][pivot]);
            if (value > best_value) {
                best_value = value;
                best_row = row;
            }
        }
        if (best_value < 1.0e-12f) {
            return false;
        }
        if (best_row != pivot) {
            for (int column = pivot; column <= G1_CONSTRAINT_DIM; ++column) {
                const float temporary = system[pivot][column];
                system[pivot][column] = system[best_row][column];
                system[best_row][column] = temporary;
            }
        }
        const float inverse_pivot = 1.0f / system[pivot][pivot];
        for (int column = pivot; column <= G1_CONSTRAINT_DIM; ++column) {
            system[pivot][column] *= inverse_pivot;
        }
        for (int row = 0; row < G1_CONSTRAINT_DIM; ++row) {
            if (row == pivot) {
                continue;
            }
            const float factor = system[row][pivot];
            for (int column = pivot; column <= G1_CONSTRAINT_DIM; ++column) {
                system[row][column] -= factor * system[pivot][column];
            }
        }
    }

    float correction_norm_squared = 0.0f;
    for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
        float value = 0.0f;
        for (int row = 0; row < G1_CONSTRAINT_DIM; ++row) {
            value += jacobian[row * G1_JOINT_DIM + joint] *
                system[row][G1_CONSTRAINT_DIM];
        }
        correction[joint] = value;
        correction_norm_squared += value * value;
    }

    const float correction_norm = sqrtf(correction_norm_squared);
    if (maximum_step > 0.0f && correction_norm > maximum_step) {
        const float scale = maximum_step / correction_norm;
        for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
            correction[joint] *= scale;
        }
    }
    return true;
}

__device__ __noinline__ bool g1_project_configuration(
    float q[G1_JOINT_DIM],
    const constraints::G1ConstraintParameters &parameters,
    int max_iterations,
    float alpha,
    float damping,
    float maximum_step
) {
    float residual[G1_CONSTRAINT_DIM];
    float jacobian[G1_CONSTRAINT_DIM * G1_JOINT_DIM];

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        g1_constraint_residual_and_jacobian(q, parameters, residual, jacobian);
        float error_squared = 0.0f;
        for (int row = 0; row < G1_CONSTRAINT_DIM; ++row) {
            error_squared += residual[row] * residual[row];
        }
        if (error_squared <= parameters.tolerance_squared) {
            return true;
        }

        // One combined inner-LM system for all three constraints.
        // step = J^T * (J * J^T + lambda I)^-1 * residual.
        float system[G1_CONSTRAINT_DIM][G1_CONSTRAINT_DIM + 1];
        for (int row = 0; row < G1_CONSTRAINT_DIM; ++row) {
            for (int column = 0; column < G1_CONSTRAINT_DIM; ++column) {
                float value = row == column ? damping : 0.0f;
                for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                    value += jacobian[row * G1_JOINT_DIM + joint] * jacobian[column * G1_JOINT_DIM + joint];
                }
                system[row][column] = value;
            }
            system[row][G1_CONSTRAINT_DIM] = residual[row];
        }

        for (int pivot = 0; pivot < G1_CONSTRAINT_DIM; ++pivot) {
            int best_row = pivot;
            float best_value = fabsf(system[pivot][pivot]);
            for (int row = pivot + 1; row < G1_CONSTRAINT_DIM; ++row) {
                const float value = fabsf(system[row][pivot]);
                if (value > best_value) {
                    best_value = value;
                    best_row = row;
                }
            }
            if (best_value < 1.0e-12f) {
                return false;
            }
            if (best_row != pivot) {
                for (int column = pivot; column <= G1_CONSTRAINT_DIM; ++column) {
                    const float temporary = system[pivot][column];
                    system[pivot][column] = system[best_row][column];
                    system[best_row][column] = temporary;
                }
            }
            const float inverse_pivot = 1.0f / system[pivot][pivot];
            for (int column = pivot; column <= G1_CONSTRAINT_DIM; ++column) {
                system[pivot][column] *= inverse_pivot;
            }
            for (int row = 0; row < G1_CONSTRAINT_DIM; ++row) {
                if (row == pivot) {
                    continue;
                }
                const float factor = system[row][pivot];
                for (int column = pivot; column <= G1_CONSTRAINT_DIM; ++column) {
                    system[row][column] -= factor * system[pivot][column];
                }
            }
        }

        float step[G1_JOINT_DIM];
        float step_norm_squared = 0.0f;
        for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
            float value = 0.0f;
            for (int row = 0; row < G1_CONSTRAINT_DIM; ++row) {
                value += jacobian[row * G1_JOINT_DIM + joint] *
                    system[row][G1_CONSTRAINT_DIM];
            }
            step[joint] = alpha * value;
            step_norm_squared += step[joint] * step[joint];
        }

        const float step_norm = sqrtf(step_norm_squared);
        float scale = maximum_step > 0.0f && step_norm > maximum_step
            ? maximum_step / step_norm
            : 1.0f;
        bool accepted = false;
        for (int line_search = 0; line_search < 8; ++line_search) {
            float candidate[G1_JOINT_DIM];
            for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                candidate[joint] = q[joint] - scale * step[joint];
            }
            g1_clamp_configuration(candidate);
            const float candidate_error =
                g1_constraint_error_squared(candidate, parameters);
            if (candidate_error < error_squared) {
                for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                    q[joint] = candidate[joint];
                }
                accepted = true;
                break;
            }
            scale *= 0.5f;
        }
        if (!accepted) {
            return false;
        }
    }

    return g1_constraint_error_squared(q, parameters) <= parameters.tolerance_squared;
}

__device__ __forceinline__ bool g1_project_motion(
    volatile float *motion_segment,
    volatile float *motion_segment_next,
    int granularity,
    const constraints::G1ConstraintParameters &parameters,
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
    int tid
) {
    const int waypoint = tid / 4 + 1;
    const int lane = tid % 4;

    if (tid == 0) {
        projection_progress[0] = 0;
        projection_success[0] = 0;
        projection_valid[0] = 1;
    }
    if (tid < G1_JOINT_DIM) {
        motion_segment_next[tid] = motion_segment[tid];
    }
    if (waypoint <= granularity && lane == 0) {
        projection_valid[waypoint] = 0;
    }
    __syncthreads();

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        if (waypoint <= granularity) {
            const int current_progress = projection_progress[0];
            if (waypoint > current_progress) {
                if (lane == 0) {
                    float q[G1_JOINT_DIM];
                    float correction[G1_JOINT_DIM];
                    float task_error_norm = 1.0e30f;
                    for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                        q[joint] = motion_segment[waypoint * G1_JOINT_DIM + joint];
                    }

                    const bool correction_ok = g1_task_correction(
                        q,
                        parameters,
                        damping,
                        maximum_step,
                        correction,
                        task_error_norm
                    );

                    float difference[G1_JOINT_DIM];
                    float smoothness_distance_squared = 0.0f;
                    for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                        difference[joint] = q[joint] - motion_segment[(waypoint - 1) * G1_JOINT_DIM + joint];
                        smoothness_distance_squared += difference[joint] * difference[joint];
                    }
                    const float smoothness_distance = sqrtf(smoothness_distance_squared);
                    const float smoothness_error = use_smoothness? fmaxf(0.0f, smoothness_distance - smoothness_threshold): 0.0f;

                    const float inv_smoothness_distance = smoothness_distance > 1.0e-8f ? 1.0f / smoothness_distance : 0.0f;

                    float combined[G1_JOINT_DIM];
                    float combined_norm_squared = 0.0f;
                    for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                        // const float smoothness_gradient =
                        //     difference[joint] * smoothness_error;
                        const float smoothness_gradient = difference[joint] * inv_smoothness_distance * smoothness_error;
                        combined[joint] = alpha * (correction[joint] + smoothness_weight * smoothness_gradient);
                        // combined[joint] =
                        //     alpha * correction[joint]
                        //     + beta * smoothness_gradient;
                        combined_norm_squared += combined[joint] * combined[joint];
                    }

                    const float combined_norm = sqrtf(combined_norm_squared);
                    const float combined_scale =
                        maximum_step > 0.0f && combined_norm > maximum_step
                            ? maximum_step / combined_norm
                            : 1.0f;
                    float q_next[G1_JOINT_DIM];
                    for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                        q_next[joint] =
                            q[joint] - combined_scale * combined[joint];
                    }
                    g1_clamp_configuration(q_next);
                    for (int joint = 0; joint < G1_JOINT_DIM; ++joint) {
                        motion_segment_next[
                            waypoint * G1_JOINT_DIM + joint
                        ] = q_next[joint];
                    }

                    projection_valid[waypoint] =
                        correction_ok &&
                        task_error_norm < task_tolerance &&
                        (
                            !use_smoothness ||
                            smoothness_distance <= smoothness_threshold
                        );
                }
            } else {
                if (lane == 0) {
                    projection_valid[waypoint] = 1;
                }
                for (int joint = lane;
                     joint < G1_JOINT_DIM;
                     joint += 4) {
                    motion_segment_next[
                        waypoint * G1_JOINT_DIM + joint
                    ] = motion_segment[
                        waypoint * G1_JOINT_DIM + joint
                    ];
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
            if (progress == granularity) {
                projection_success[0] = 1;
            }
        }
        __syncthreads();

        if (projection_success[0] != 0) {
            return true;
        }

        if (
            waypoint <= granularity &&
            waypoint > projection_progress[0]
        ) {
            for (int joint = lane;
                 joint < G1_JOINT_DIM;
                 joint += 4) {
                motion_segment[waypoint * G1_JOINT_DIM + joint] =
                    motion_segment_next[
                        waypoint * G1_JOINT_DIM + joint
                    ];
            }
        }
        __syncthreads();
    }

    return false;
}

}  // namespace ppln::collision

#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace ppln::collision {

constexpr float FRANKA_FER_HAND_TCP_OFFSET_M = 0.1034f;

struct FrankaTransform {
    float rotation[9];  // row major
    float translation[3];
};

__host__ __device__ __forceinline__ FrankaTransform franka_identity() {
    FrankaTransform transform{};
    transform.rotation[0] = 1.0f;
    transform.rotation[4] = 1.0f;
    transform.rotation[8] = 1.0f;
    return transform;
}

__host__ __device__ __forceinline__ FrankaTransform franka_compose(
    const FrankaTransform &left,
    const FrankaTransform &right
) {
    FrankaTransform result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            for (int inner = 0; inner < 3; ++inner) {
                result.rotation[row * 3 + column] +=
                    left.rotation[row * 3 + inner] *
                    right.rotation[inner * 3 + column];
            }
        }
        result.translation[row] = left.translation[row];
        for (int inner = 0; inner < 3; ++inner) {
            result.translation[row] += left.rotation[row * 3 + inner] *
                right.translation[inner];
        }
    }
    return result;
}

__host__ __device__ __forceinline__ FrankaTransform franka_z_rotation(
    float angle
) {
    FrankaTransform transform = franka_identity();
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    transform.rotation[0] = cosine;
    transform.rotation[1] = -sine;
    transform.rotation[3] = sine;
    transform.rotation[4] = cosine;
    return transform;
}

__host__ __device__ __forceinline__ FrankaTransform franka_joint_origin(
    int joint
) {
    FrankaTransform transform = franka_identity();
    float sine = 0.0f;
    switch (joint) {
        case 0:
            transform.translation[2] = 0.333f;
            break;
        case 1:
            sine = -1.0f;
            break;
        case 2:
            transform.translation[1] = -0.316f;
            sine = 1.0f;
            break;
        case 3:
            transform.translation[0] = 0.0825f;
            sine = 1.0f;
            break;
        case 4:
            transform.translation[0] = -0.0825f;
            transform.translation[1] = 0.384f;
            sine = -1.0f;
            break;
        case 5:
            sine = 1.0f;
            break;
        default:
            transform.translation[0] = 0.088f;
            sine = 1.0f;
            break;
    }
    if (sine != 0.0f) {
        // Rx(+/- pi/2)
        transform.rotation[4] = 0.0f;
        transform.rotation[5] = -sine;
        transform.rotation[7] = sine;
        transform.rotation[8] = 0.0f;
    }
    return transform;
}

__host__ __device__ __forceinline__ void franka_arm_kinematics(
    const float q[7],
    const float base_translation[3],
    FrankaTransform &end_effector,
    float joint_origins[7][3] = nullptr,
    float joint_axes[7][3] = nullptr
) {
    FrankaTransform current = franka_identity();
    for (int component = 0; component < 3; ++component) {
        current.translation[component] = base_translation[component];
    }
    for (int joint = 0; joint < 7; ++joint) {
        current = franka_compose(current, franka_joint_origin(joint));
        if (joint_origins != nullptr) {
            for (int component = 0; component < 3; ++component) {
                joint_origins[joint][component] =
                    current.translation[component];
            }
        }
        if (joint_axes != nullptr) {
            joint_axes[joint][0] = current.rotation[2];
            joint_axes[joint][1] = current.rotation[5];
            joint_axes[joint][2] = current.rotation[8];
        }
        current = franka_compose(current, franka_z_rotation(q[joint]));
    }

    FrankaTransform flange = franka_identity();
    flange.translation[2] = 0.107f;
    current = franka_compose(current, flange);
    current = franka_compose(
        current, franka_z_rotation(-0.7853981633974483f)
    );
    FrankaTransform ee_offset = franka_identity();
    ee_offset.translation[2] = FRANKA_FER_HAND_TCP_OFFSET_M;
    end_effector = franka_compose(current, ee_offset);
}

__host__ __device__ __forceinline__ void franka_matrix_transpose_multiply(
    const float left[9],
    const float right[9],
    float result[9]
) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result[row * 3 + column] = 0.0f;
            for (int inner = 0; inner < 3; ++inner) {
                result[row * 3 + column] +=
                    left[inner * 3 + row] * right[inner * 3 + column];
            }
        }
    }
}

__host__ __device__ __forceinline__ void franka_rotation_transpose_vector(
    const float rotation[9],
    const float vector[3],
    float result[3]
) {
    for (int row = 0; row < 3; ++row) {
        result[row] = rotation[row] * vector[0] +
            rotation[3 + row] * vector[1] +
            rotation[6 + row] * vector[2];
    }
}

__host__ __device__ __forceinline__ void franka_cross3(
    const float left[3],
    const float right[3],
    float result[3]
) {
    result[0] = left[1] * right[2] - left[2] * right[1];
    result[1] = left[2] * right[0] - left[0] * right[2];
    result[2] = left[0] * right[1] - left[1] * right[0];
}

}  // namespace ppln::collision

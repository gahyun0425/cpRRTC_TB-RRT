#pragma once

// Franka constraint parameters are materialized at the constraint boundary;
// the planner consumes only the resulting trivially-copyable parameter block.

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "src/planning/FrankaConstraintParameters.hh"
#include "src/planning/Robots.hh"
#include "src/robots/franka_kinematics.cuh"

inline void franka_rotation_to_quaternion_wxyz(
    const float rotation[9],
    float quaternion[4]
) {
    const float trace = rotation[0] + rotation[4] + rotation[8];
    if (trace > 0.0f) {
        const float scale = 2.0f * std::sqrt(trace + 1.0f);
        quaternion[0] = 0.25f * scale;
        quaternion[1] = (rotation[7] - rotation[5]) / scale;
        quaternion[2] = (rotation[2] - rotation[6]) / scale;
        quaternion[3] = (rotation[3] - rotation[1]) / scale;
    } else if (rotation[0] > rotation[4] && rotation[0] > rotation[8]) {
        const float scale = 2.0f * std::sqrt(
            std::max(0.0f, 1.0f + rotation[0] - rotation[4] - rotation[8])
        );
        quaternion[0] = (rotation[7] - rotation[5]) / scale;
        quaternion[1] = 0.25f * scale;
        quaternion[2] = (rotation[1] + rotation[3]) / scale;
        quaternion[3] = (rotation[2] + rotation[6]) / scale;
    } else if (rotation[4] > rotation[8]) {
        const float scale = 2.0f * std::sqrt(
            std::max(0.0f, 1.0f + rotation[4] - rotation[0] - rotation[8])
        );
        quaternion[0] = (rotation[2] - rotation[6]) / scale;
        quaternion[1] = (rotation[1] + rotation[3]) / scale;
        quaternion[2] = 0.25f * scale;
        quaternion[3] = (rotation[5] + rotation[7]) / scale;
    } else {
        const float scale = 2.0f * std::sqrt(
            std::max(0.0f, 1.0f + rotation[8] - rotation[0] - rotation[4])
        );
        quaternion[0] = (rotation[3] - rotation[1]) / scale;
        quaternion[1] = (rotation[2] + rotation[6]) / scale;
        quaternion[2] = (rotation[5] + rotation[7]) / scale;
        quaternion[3] = 0.25f * scale;
    }
    float norm = 0.0f;
    for (int component = 0; component < 4; ++component) {
        norm += quaternion[component] * quaternion[component];
    }
    norm = std::sqrt(norm);
    if (!(norm > 1.0e-8f)) {
        throw std::runtime_error("invalid Franka reference orientation");
    }
    for (int component = 0; component < 4; ++component) {
        quaternion[component] /= norm;
    }
    if (quaternion[0] < 0.0f) {
        for (int component = 0; component < 4; ++component) {
            quaternion[component] = -quaternion[component];
        }
    }
}

template <typename Robot>
inline ppln::constraints::FrankaConstraintParameters
franka_constraint_parameters_from_start(
    const typename Robot::Configuration &start
) {
    static_assert(
        std::is_same_v<Robot, ppln::robots::FrankaSingle> ||
        std::is_same_v<Robot, ppln::robots::Franka>,
        "Franka constraint parameters require a Franka robot"
    );
    ppln::constraints::FrankaConstraintParameters parameters{};
    const float single_base[3] = {0.0f, 0.0f, 0.0f};
    const float left_base[3] = {0.0f, 0.2f, 0.6f};
    const float right_base[3] = {0.0f, -0.2f, 0.6f};
    ppln::collision::FrankaTransform left{};
    ppln::collision::franka_arm_kinematics(
        start.data(),
        std::is_same_v<Robot, ppln::robots::Franka>
            ? left_base : single_base,
        left
    );

    // R_start^T * world_Z.  At a valid configuration, mapping this fixed
    // EE-frame vector through R(q) must produce world_Z again.  The connected
    // orientation manifold is R(q) = Rz(yaw) * R_start.
    parameters.world_yaw_axis_local[0] = left.rotation[6];
    parameters.world_yaw_axis_local[1] = left.rotation[7];
    parameters.world_yaw_axis_local[2] = left.rotation[8];

    if constexpr (std::is_same_v<Robot, ppln::robots::Franka>) {
        ppln::collision::FrankaTransform right{};
        ppln::collision::franka_arm_kinematics(
            start.data() + 7, right_base, right
        );
        float relative_rotation[9];
        ppln::collision::franka_matrix_transpose_multiply(
            left.rotation, right.rotation, relative_rotation
        );
        franka_rotation_to_quaternion_wxyz(
            relative_rotation, parameters.relative_pose_target
        );
        float world_delta[3] = {
            right.translation[0] - left.translation[0],
            right.translation[1] - left.translation[1],
            right.translation[2] - left.translation[2]
        };
        ppln::collision::franka_rotation_transpose_vector(
            left.rotation,
            world_delta,
            parameters.relative_pose_target + 4
        );
    }
    return parameters;
}

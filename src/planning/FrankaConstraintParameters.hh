#pragma once

#include <type_traits>

namespace ppln::constraints {

struct FrankaConstraintParameters {
    // inverse(T_world_left_ee) * T_world_right_ee, stored as
    // quaternion_wxyz followed by translation_xyz_m.
    float relative_pose_target[7] = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    // World +Z (the world-frame yaw axis) expressed in the end-effector
    // frame at the start configuration.  Requiring the current EE to map
    // this vector back to world +Z constrains roll/pitch while leaving only
    // rotation about world +Z free.
    float world_yaw_axis_local[3] = {0.0f, 0.0f, 1.0f};

    float tolerance_squared = 1.0e-6f;
};

static_assert(
    std::is_trivially_copyable_v<FrankaConstraintParameters>,
    "Franka parameters must be copied to CUDA memory"
);

}  // namespace ppln::constraints

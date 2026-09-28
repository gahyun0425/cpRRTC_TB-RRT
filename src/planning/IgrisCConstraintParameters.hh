#pragma once

#include <type_traits>

namespace ppln::constraints {

struct IgrisCConstraintParameters {
    // quaternion_wxyz followed by translation_xyz_m
    float feet_target[2][7]{};

    // Four counter-clockwise XY vertices. The committed task uses the convex
    // hull of both foot boxes inset by 0.05 m.
    float support_polygon[8]{};
    float support_margin_m = 0.05f;

    // The temporary box center is the midpoint of l_grasp and r_grasp.
    float payload_mass_kg = 0.15f;

    // inverse(T_world_l_grasp) * T_world_r_grasp
    float bimanual_target[7]{};

    float tolerance_squared = 1.0e-6f;
};

static_assert(
    std::is_trivially_copyable_v<IgrisCConstraintParameters>,
    "IGRIS-C parameters must be copied to CUDA memory"
);

}  // namespace ppln::constraints

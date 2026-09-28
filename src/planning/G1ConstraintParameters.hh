#pragma once

#include <type_traits>

namespace ppln::constraints {

constexpr int G1_ATTACHED_OBJECT_MAX_SPHERES = 64;
constexpr int G1_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_SPHERES = 64;
constexpr float G1_DEFAULT_SUPPORT_MARGIN_M = 0.05f;

struct G1AttachedObjectCollisionSpec
{
    bool enabled = false;
    int sphere_count = 0;

    // Payload-center origin expressed in the left-hand end-effector frame.
    float left_hand_center_offset[3]{};
    // Sphere center relative to the payload center, followed by radius.
    float spheres[G1_ATTACHED_OBJECT_MAX_SPHERES][4]{};

    int ignored_robot_sphere_count = 0;
    int ignored_robot_spheres[
        G1_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_SPHERES
    ]{};
};

struct EmptyConstraintParameters
{
};

struct G1ConstraintParameters
{
    float feet_reference[2][7]{};
    float feet_target[2][7]{};

    float support_polygon[8]{};
    float support_margin_m = G1_DEFAULT_SUPPORT_MARGIN_M;

    // Payload CoM uses the attached-object center in the left-hand frame.
    // Zero mass preserves the robot-only CoM behavior of older problem files.
    float payload_mass_kg = 0.0f;
    G1AttachedObjectCollisionSpec attached_object_collision{};

    float bimanual_target[7]{};

    float tolerance_squared = 1.0e-6f;
};

static_assert(
    std::is_trivially_copyable_v<G1ConstraintParameters>,
    "G1 parameters must be copied to CUDA constant memory"
);

}

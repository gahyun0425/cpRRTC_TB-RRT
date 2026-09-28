#pragma once

// The imported MuJoCo models use the same Panda link kinematics as the
// repository's existing Panda collision backend.  Keep that proven sphere
// model and add the fixed dual-arm mount transforms plus the source models'
// end-effector-attached payloads here.

#include <type_traits>

#include "src/robots/franka_kinematics.cuh"

namespace ppln::collision {

constexpr int FRANKA_ARM_FINE_SPHERES = 59;
constexpr int FRANKA_ARM_APPROX_SPHERES = 11;
constexpr int FRANKA_COLLISION_BATCH = 16;
constexpr int FRANKA_SINGLE_ATTACHED_SPHERES = 5;
constexpr int FRANKA_DUAL_ATTACHED_SPHERES = 44;
constexpr float FRANKA_ATTACHED_SPHERE_RADIUS = 0.02f;

// T_start_EE_object for the object body in franka_single.xml.  The source
// implementation computes this transform when it attaches the MuJoCo object
// at the default start configuration.
__device__ __constant__ float franka_single_attached_rotation[9] = {
    5.65184217e-06f, 4.04983458e-06f, -1.0f,
    7.46365351e-06f, 1.0f, 4.04987676e-06f,
    1.0f, -7.46367640e-06f, 5.65181194e-06f
};
__device__ __constant__ float franka_single_attached_translation[3] = {
    -0.0299951539f, -3.27119653e-06f, -0.0200040056f
};

// T_start_left_EE_object for the five-box tray in franka_panda.xml.  The
// payload is parented to the left EE; the dual relative-pose equality keeps
// the right grasp fixed with respect to it.
__device__ __constant__ float franka_dual_attached_rotation[9] = {
    1.0f, -1.12881255e-05f, 4.01251945e-06f,
    -1.12881331e-05f, -1.0f, 1.89331703e-06f,
    4.01249807e-06f, -1.89336233e-06f, -1.0f
};
__device__ __constant__ float franka_dual_attached_translation[3] = {
    -7.26699543e-08f, 0.124998270f, 0.0499992695f
};

template <typename Robot>
__device__ __forceinline__ int franka_attached_sphere_count() {
    static_assert(
        std::is_same_v<Robot, robots::FrankaSingle> ||
        std::is_same_v<Robot, robots::Franka>,
        "attached payload is only defined for imported Franka models"
    );
    if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
        return FRANKA_SINGLE_ATTACHED_SPHERES;
    }
    return FRANKA_DUAL_ATTACHED_SPHERES;
}

template <typename Robot>
__device__ __forceinline__ void franka_attached_object_relative_pose(
    FrankaTransform &relative
) {
    const float *rotation;
    const float *translation;
    if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
        rotation = franka_single_attached_rotation;
        translation = franka_single_attached_translation;
    } else {
        rotation = franka_dual_attached_rotation;
        translation = franka_dual_attached_translation;
    }
    for (int index = 0; index < 9; ++index) {
        relative.rotation[index] = rotation[index];
    }
    for (int component = 0; component < 3; ++component) {
        relative.translation[component] = translation[component];
    }
}

template <typename Robot>
__device__ __forceinline__ void franka_attached_sphere_object_center(
    int sphere,
    float center[3]
) {
    if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
        center[0] = 0.0f;
        center[1] = 0.0f;
        center[2] = -0.08f + 0.04f * sphere;
        return;
    }

    // Exact 0.02 m-radius grid fill used by the source attachment code for
    // the tray's bottom, front/back walls, and left/right walls.
    if (sphere < 24) {
        center[0] = -0.06f + 0.04f * (sphere / 6);
        center[1] = -0.10f + 0.04f * (sphere % 6);
        center[2] = 0.0f;
    } else if (sphere < 28) {
        center[0] = -0.06f + 0.04f * (sphere - 24);
        center[1] = 0.12f;
        center[2] = 0.04f;
    } else if (sphere < 32) {
        center[0] = -0.06f + 0.04f * (sphere - 28);
        center[1] = -0.12f;
        center[2] = 0.04f;
    } else if (sphere < 38) {
        center[0] = -0.075f;
        center[1] = -0.10f + 0.04f * (sphere - 32);
        center[2] = 0.04f;
    } else {
        center[0] = 0.075f;
        center[1] = -0.10f + 0.04f * (sphere - 38);
        center[2] = 0.04f;
    }
}

template <typename Robot>
__device__ __noinline__ bool franka_attached_object_env_collision_check(
    const float *q,
    Environment<float> *environment,
    int tid
) {
    const float base[3] = {
        0.0f,
        std::is_same_v<Robot, robots::Franka> ? 0.2f : 0.0f,
        std::is_same_v<Robot, robots::Franka> ? 0.6f : 0.0f
    };
    FrankaTransform end_effector{};
    FrankaTransform relative{};
    franka_arm_kinematics(q, base, end_effector);
    franka_attached_object_relative_pose<Robot>(relative);
    const FrankaTransform object_pose = franka_compose(
        end_effector, relative
    );

    bool collision_found = false;
    const int lane = tid % 4;
    for (int sphere = lane;
         sphere < franka_attached_sphere_count<Robot>();
         sphere += 4) {
        float local[3];
        franka_attached_sphere_object_center<Robot>(sphere, local);
        float world[3];
        for (int row = 0; row < 3; ++row) {
            world[row] = object_pose.translation[row] +
                object_pose.rotation[row * 3] * local[0] +
                object_pose.rotation[row * 3 + 1] * local[1] +
                object_pose.rotation[row * 3 + 2] * local[2];
        }
        if (sphere_environment_in_collision(
                environment,
                world[0], world[1], world[2],
                FRANKA_ATTACHED_SPHERE_RADIUS
            )) {
            collision_found = true;
        }
    }
    return !collision_found;
}

__device__ __forceinline__ bool franka_fine_env_check_and_mark(
    volatile float *sphere_positions,
    volatile int *joint_in_collision,
    Environment<float> *environment,
    int tid
) {
    const int lane = tid % 4;
    const int batch = tid / 4;
    bool collision_found = false;
    for (int sphere = lane;
         sphere < FRANKA_ARM_FINE_SPHERES;
         sphere += 4) {
        const int offset =
            sphere * FRANKA_COLLISION_BATCH * 3 + batch * 3;
        if (sphere_environment_in_collision(
                environment,
                sphere_positions[offset],
                sphere_positions[offset + 1],
                sphere_positions[offset + 2],
                panda_spheres_array[sphere].w
            )) {
            collision_found = true;
            atomicAdd(
                (int *)&joint_in_collision[
                    20 * batch + panda_sphere_to_joint[sphere]
                ],
                1
            );
        }
    }
    return !collision_found;
}

__device__ __forceinline__ bool franka_fine_arm_self_check_and_mark(
    volatile float *sphere_positions,
    volatile int *joint_in_collision,
    int tid
) {
    const int lane = tid % 4;
    const int batch = tid / 4;
    bool collision_found = false;
    for (int range = lane; range < PANDA_SELF_CC_RANGE_COUNT; range += 4) {
        const int first = panda_self_cc_ranges[range][0];
        const int first_offset =
            first * FRANKA_COLLISION_BATCH * 3 + batch * 3;
        for (int second = panda_self_cc_ranges[range][1];
             second <= panda_self_cc_ranges[range][2];
             ++second) {
            const int second_offset =
                second * FRANKA_COLLISION_BATCH * 3 + batch * 3;
            if (sphere_sphere_self_collision(
                    sphere_positions[first_offset],
                    sphere_positions[first_offset + 1],
                    sphere_positions[first_offset + 2],
                    panda_spheres_array[first].w,
                    sphere_positions[second_offset],
                    sphere_positions[second_offset + 1],
                    sphere_positions[second_offset + 2],
                    panda_spheres_array[second].w
                )) {
                collision_found = true;
                atomicAdd(
                    (int *)&joint_in_collision[
                        20 * batch + panda_sphere_to_joint[first]
                    ],
                    1
                );
            }
        }
    }
    return !collision_found;
}

template <int SphereCount>
__device__ __forceinline__ void franka_translate_arm_spheres(
    volatile float *sphere_positions,
    float *transforms,
    float tx,
    float ty,
    float tz,
    int tid
) {
    const int coordinate = tid % 4;
    const int batch = tid / 4;
    if (coordinate < 3) {
        const float translation = coordinate == 0 ? tx :
            (coordinate == 1 ? ty : tz);
        for (int sphere = 0; sphere < SphereCount; ++sphere) {
            sphere_positions[
                sphere * FRANKA_COLLISION_BATCH * 3 + batch * 3 + coordinate
            ] += translation;
        }
    } else {
        float *transform = transforms + batch * 16;
        transform[12] += tx;
        transform[13] += ty;
        transform[14] += tz;
    }
}

template <int SphereCount, bool Approximate>
__device__ __forceinline__ bool franka_cross_arm_self_collision(
    volatile float *sphere_positions,
    volatile int *joint_in_collision,
    int tid,
    bool mark_links
) {
    const int lane = tid % 4;
    const int batch = tid / 4;
    const int arm_offset = SphereCount * FRANKA_COLLISION_BATCH * 3;
    bool collision_found = false;
    for (int pair = lane; pair < SphereCount * SphereCount; pair += 4) {
        const int left = pair / SphereCount;
        const int right = pair - left * SphereCount;
        const int left_offset =
            left * FRANKA_COLLISION_BATCH * 3 + batch * 3;
        const int right_offset = arm_offset +
            right * FRANKA_COLLISION_BATCH * 3 + batch * 3;
        const float left_radius = Approximate
            ? panda_approx_spheres_array[left].w
            : panda_spheres_array[left].w;
        const float right_radius = Approximate
            ? panda_approx_spheres_array[right].w
            : panda_spheres_array[right].w;
        if (sphere_sphere_self_collision(
                sphere_positions[left_offset],
                sphere_positions[left_offset + 1],
                sphere_positions[left_offset + 2],
                left_radius,
                sphere_positions[right_offset],
                sphere_positions[right_offset + 1],
                sphere_positions[right_offset + 2],
                right_radius
            )) {
            collision_found = true;
            if (mark_links) {
                const int left_joint = Approximate
                    ? panda_approx_sphere_to_joint[left]
                    : panda_sphere_to_joint[left];
                const int right_joint = Approximate
                    ? panda_approx_sphere_to_joint[right]
                    : panda_sphere_to_joint[right];
                atomicAdd(
                    (int *)&joint_in_collision[20 * batch + left_joint], 1
                );
                atomicAdd(
                    (int *)&joint_in_collision[20 * batch + right_joint], 1
                );
            }
        }
    }
    return !collision_found;
}

template <>
__device__ __forceinline__ void fk<ppln::robots::FrankaSingle>(
    const float *q,
    volatile float *sphere_positions,
    float *transforms,
    int tid
) {
    fk<ppln::robots::PandaCollisionModel>(q, sphere_positions, transforms, tid);
}

template <>
__device__ __forceinline__ void fk_approx<ppln::robots::FrankaSingle>(
    const float *q,
    volatile float *sphere_positions,
    float *transforms,
    int tid
) {
    fk_approx<ppln::robots::PandaCollisionModel>(q, sphere_positions, transforms, tid);
}

template <>
__device__ __forceinline__ bool self_collision_check<ppln::robots::FrankaSingle>(
    volatile float *sphere_positions,
    volatile int *joint_in_collision,
    int tid
) {
    return self_collision_check<ppln::robots::PandaCollisionModel>(
        sphere_positions, joint_in_collision, tid
    );
}

template <>
__device__ __forceinline__ bool env_collision_check<ppln::robots::FrankaSingle>(
    volatile float *sphere_positions,
    volatile int *joint_in_collision,
    Environment<float> *environment,
    int tid
) {
    return env_collision_check<ppln::robots::PandaCollisionModel>(
        sphere_positions, joint_in_collision, environment, tid
    );
}

template <>
__device__ __forceinline__ bool self_collision_check_approx<ppln::robots::FrankaSingle>(
    volatile float *sphere_positions,
    volatile int *joint_in_collision,
    int tid
) {
    return self_collision_check_approx<ppln::robots::PandaCollisionModel>(
        sphere_positions, joint_in_collision, tid
    );
}

template <>
__device__ __forceinline__ bool env_collision_check_approx<ppln::robots::FrankaSingle>(
    volatile float *sphere_positions,
    volatile int *joint_in_collision,
    Environment<float> *environment,
    int tid
) {
    return env_collision_check_approx<ppln::robots::PandaCollisionModel>(
        sphere_positions, joint_in_collision, environment, tid
    );
}

template <>
__device__ __noinline__ void fk<ppln::robots::Franka>(
    const float *q,
    volatile float *sphere_positions,
    float *transforms,
    int tid
) {
    constexpr int sphere_stride =
        FRANKA_ARM_FINE_SPHERES * FRANKA_COLLISION_BATCH * 3;
    constexpr int transform_stride = FRANKA_COLLISION_BATCH * 16;
    fk<ppln::robots::PandaCollisionModel>(q, sphere_positions, transforms, tid);
    franka_translate_arm_spheres<FRANKA_ARM_FINE_SPHERES>(
        sphere_positions, transforms, 0.0f, 0.2f, 0.6f, tid
    );
    fk<ppln::robots::PandaCollisionModel>(
        q + 7,
        sphere_positions + sphere_stride,
        transforms + transform_stride,
        tid
    );
    franka_translate_arm_spheres<FRANKA_ARM_FINE_SPHERES>(
        sphere_positions + sphere_stride,
        transforms + transform_stride,
        0.0f, -0.2f, 0.6f, tid
    );
}

template <>
__device__ __noinline__ void fk_approx<ppln::robots::Franka>(
    const float *q,
    volatile float *sphere_positions,
    float *transforms,
    int tid
) {
    constexpr int sphere_stride =
        FRANKA_ARM_APPROX_SPHERES * FRANKA_COLLISION_BATCH * 3;
    constexpr int transform_stride = FRANKA_COLLISION_BATCH * 16;
    fk_approx<ppln::robots::PandaCollisionModel>(q, sphere_positions, transforms, tid);
    franka_translate_arm_spheres<FRANKA_ARM_APPROX_SPHERES>(
        sphere_positions, transforms, 0.0f, 0.2f, 0.6f, tid
    );
    fk_approx<ppln::robots::PandaCollisionModel>(
        q + 7,
        sphere_positions + sphere_stride,
        transforms + transform_stride,
        tid
    );
    franka_translate_arm_spheres<FRANKA_ARM_APPROX_SPHERES>(
        sphere_positions + sphere_stride,
        transforms + transform_stride,
        0.0f, -0.2f, 0.6f, tid
    );
}

template <>
__device__ __noinline__ bool env_collision_check<ppln::robots::Franka>(
    volatile float *sphere_positions,
    volatile int *joint_in_collision,
    Environment<float> *environment,
    int tid
) {
    constexpr int stride =
        FRANKA_ARM_FINE_SPHERES * FRANKA_COLLISION_BATCH * 3;
    const bool left_ok = env_collision_check<ppln::robots::PandaCollisionModel>(
        sphere_positions, joint_in_collision, environment, tid
    );
    const bool right_ok = env_collision_check<ppln::robots::PandaCollisionModel>(
        sphere_positions + stride, joint_in_collision, environment, tid
    );
    return left_ok && right_ok;
}

template <>
__device__ __noinline__ bool env_collision_check_approx<ppln::robots::Franka>(
    volatile float *sphere_positions,
    volatile int *joint_in_collision,
    Environment<float> *environment,
    int tid
) {
    constexpr int stride =
        FRANKA_ARM_APPROX_SPHERES * FRANKA_COLLISION_BATCH * 3;
    const bool left_ok = env_collision_check_approx<ppln::robots::PandaCollisionModel>(
        sphere_positions, joint_in_collision, environment, tid
    );
    const bool right_ok = env_collision_check_approx<ppln::robots::PandaCollisionModel>(
        sphere_positions + stride, joint_in_collision, environment, tid
    );
    return left_ok && right_ok;
}

template <>
__device__ __noinline__ bool self_collision_check<ppln::robots::Franka>(
    volatile float *sphere_positions,
    volatile int *joint_in_collision,
    int tid
) {
    constexpr int stride =
        FRANKA_ARM_FINE_SPHERES * FRANKA_COLLISION_BATCH * 3;
    const bool left_ok = self_collision_check<ppln::robots::PandaCollisionModel>(
        sphere_positions, joint_in_collision, tid
    );
    const bool right_ok = self_collision_check<ppln::robots::PandaCollisionModel>(
        sphere_positions + stride, joint_in_collision, tid
    );
    const bool cross_ok =
        franka_cross_arm_self_collision<FRANKA_ARM_FINE_SPHERES, false>(
            sphere_positions, joint_in_collision, tid, false
        );
    return left_ok && right_ok && cross_ok;
}

template <>
__device__ __noinline__ bool self_collision_check_approx<ppln::robots::Franka>(
    volatile float *sphere_positions,
    volatile int *joint_in_collision,
    int tid
) {
    constexpr int stride =
        FRANKA_ARM_APPROX_SPHERES * FRANKA_COLLISION_BATCH * 3;
    const bool left_ok = self_collision_check_approx<ppln::robots::PandaCollisionModel>(
        sphere_positions, joint_in_collision, tid
    );
    const bool right_ok = self_collision_check_approx<ppln::robots::PandaCollisionModel>(
        sphere_positions + stride, joint_in_collision, tid
    );
    const bool cross_ok =
        franka_cross_arm_self_collision<FRANKA_ARM_APPROX_SPHERES, true>(
            sphere_positions, joint_in_collision, tid, true
        );
    return left_ok && right_ok && cross_ok;
}

}  // namespace ppln::collision

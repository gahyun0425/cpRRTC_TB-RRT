#pragma once

#include "src/planning/G1ConstraintParameters.hh"
#include "src/robots/g1_collision.cuh"
#include "src/robots/g1_kinematics.cuh"

namespace ppln::collision {

__device__ __constant__ constraints::G1AttachedObjectCollisionSpec
    g1_attached_object_collision;

__device__ __forceinline__ int g1_attached_object_sphere_count() {
    const int count = g1_attached_object_collision.sphere_count;
    return count < constraints::G1_ATTACHED_OBJECT_MAX_SPHERES
        ? count
        : constraints::G1_ATTACHED_OBJECT_MAX_SPHERES;
}

__device__ __forceinline__ bool
g1_attached_object_ignores_robot_sphere(int robot_sphere) {
    // The carried box intentionally overlaps both grasping rubber hands.
    // Keep the new hand proxies active against the world, but omit them only
    // from attached-object-vs-robot checks.
    if (robot_sphere >= G1_BASE_SPHERE_COUNT) {
        return true;
    }
    const int raw_count =
        g1_attached_object_collision.ignored_robot_sphere_count;
    const int count = raw_count <
        constraints::G1_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_SPHERES
        ? raw_count
        : constraints::G1_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_SPHERES;
    for (int index = 0; index < count; ++index) {
        if (g1_attached_object_collision.ignored_robot_spheres[index] ==
            robot_sphere) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ void g1_attached_object_sphere_world_from_transform(
    const float transforms[48],
    int sphere_index,
    float world_sphere[4]
) {
    const float *local_sphere =
        g1_attached_object_collision.spheres[sphere_index];
    const float local_x =
        g1_attached_object_collision.left_hand_center_offset[0] +
        local_sphere[0];
    const float local_y =
        g1_attached_object_collision.left_hand_center_offset[1] +
        local_sphere[1];
    const float local_z =
        g1_attached_object_collision.left_hand_center_offset[2] +
        local_sphere[2];
    world_sphere[0] = transforms[0] +
        transforms[3] * local_x +
        transforms[6] * local_y +
        transforms[9] * local_z;
    world_sphere[1] = transforms[1] +
        transforms[4] * local_x +
        transforms[7] * local_y +
        transforms[10] * local_z;
    world_sphere[2] = transforms[2] +
        transforms[5] * local_x +
        transforms[8] * local_y +
        transforms[11] * local_z;
    world_sphere[3] = local_sphere[3];
}

__device__ __forceinline__ void g1_attached_object_sphere_world(
    const float q[G1_DIM],
    int sphere_index,
    float world_sphere[4]
) {
    // g1_end_effector_fk stores [translation, column-major rotation] for
    // each end effector. The first transform is the left rubber hand.
    float transforms[48];
    g1_end_effector_fk(q, transforms);
    g1_attached_object_sphere_world_from_transform(
        transforms, sphere_index, world_sphere
    );
}

__device__ __forceinline__ bool
g1_attached_object_environment_collision_check(
    const float q[G1_DIM],
    Environment<float> *environment,
    int tid
) {
    if (!g1_attached_object_collision.enabled) {
        return true;
    }
    bool collision_free = true;
    const int lane = tid % 4;
    const int sphere_count = g1_attached_object_sphere_count();
    float transforms[48];
    g1_end_effector_fk(q, transforms);
    for (int sphere = lane; sphere < sphere_count; sphere += 4) {
        float world_sphere[4];
        g1_attached_object_sphere_world_from_transform(
            transforms, sphere, world_sphere
        );
        collision_free = collision_free && !sphere_environment_in_collision(
            environment,
            world_sphere[0],
            world_sphere[1],
            world_sphere[2],
            world_sphere[3]
        );
    }
    return collision_free;
}

__device__ __forceinline__ bool g1_attached_object_robot_collision_check(
    const float q[G1_DIM],
    volatile float *sphere_pos,
    int tid
) {
    if (!g1_attached_object_collision.enabled) {
        return true;
    }
    bool collision_free = true;
    const int lane = tid % 4;
    const int batch = tid / 4;
    const int object_sphere_count = g1_attached_object_sphere_count();
    float transforms[48];
    g1_end_effector_fk(q, transforms);
    for (int object_sphere = 0;
         object_sphere < object_sphere_count;
         ++object_sphere) {
        float world_sphere[4];
        g1_attached_object_sphere_world_from_transform(
            transforms, object_sphere, world_sphere
        );
        for (int robot_sphere = lane;
             robot_sphere < G1_SPHERE_COUNT;
             robot_sphere += 4) {
            if (g1_attached_object_ignores_robot_sphere(robot_sphere)) {
                continue;
            }
            const int offset = robot_sphere * G1_BATCH_SIZE * 3 + batch * 3;
            collision_free = collision_free && !sphere_sphere_self_collision(
                world_sphere[0],
                world_sphere[1],
                world_sphere[2],
                world_sphere[3],
                sphere_pos[offset],
                sphere_pos[offset + 1],
                sphere_pos[offset + 2],
                g1_sphere_radii[robot_sphere]
            );
        }
    }
    return collision_free;
}

// The G1 approximate model is one conservative base sphere, so object-vs-
// robot checks are deferred to the detailed phase. Environment checks stay
// active in both phases to avoid missing a payload-only obstacle collision.
__device__ __forceinline__ bool
g1_attached_object_collision_check_approx(
    const float q[G1_DIM],
    Environment<float> *environment,
    int tid
) {
    return g1_attached_object_environment_collision_check(q, environment, tid);
}

__device__ __forceinline__ bool g1_attached_object_collision_check(
    const float q[G1_DIM],
    volatile float *sphere_pos,
    Environment<float> *environment,
    int tid
) {
    return g1_attached_object_environment_collision_check(
        q, environment, tid
    ) && g1_attached_object_robot_collision_check(q, sphere_pos, tid);
}

__device__ __forceinline__ bool g1_attached_object_collision_free(
    const float q[G1_DIM],
    const float robot_spheres[G1_SPHERE_COUNT * 4],
    Environment<float> *environment
) {
    if (!g1_attached_object_collision.enabled) {
        return true;
    }
    const int object_sphere_count = g1_attached_object_sphere_count();
    float transforms[48];
    g1_end_effector_fk(q, transforms);
    for (int object_sphere = 0;
         object_sphere < object_sphere_count;
         ++object_sphere) {
        float world_sphere[4];
        g1_attached_object_sphere_world_from_transform(
            transforms, object_sphere, world_sphere
        );
        if (sphere_environment_in_collision(
                environment,
                world_sphere[0],
                world_sphere[1],
                world_sphere[2],
                world_sphere[3])) {
            return false;
        }
        for (int robot_sphere = 0;
             robot_sphere < G1_SPHERE_COUNT;
             ++robot_sphere) {
            if (g1_attached_object_ignores_robot_sphere(robot_sphere)) {
                continue;
            }
            if (sphere_sphere_self_collision(
                    world_sphere[0],
                    world_sphere[1],
                    world_sphere[2],
                    world_sphere[3],
                    robot_spheres[robot_sphere * 4],
                    robot_spheres[robot_sphere * 4 + 1],
                    robot_spheres[robot_sphere * 4 + 2],
                    robot_spheres[robot_sphere * 4 + 3])) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace ppln::collision

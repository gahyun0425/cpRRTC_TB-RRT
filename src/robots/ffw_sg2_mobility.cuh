#pragma once

#include "src/planning/Robots.hh"
#include "src/planning/utils.cuh"
#include "src/robots/ffw_sg2.cuh"
#include "src/robots/ffw_sg2_constraint.cuh"

namespace ppln::collision {

#define FFW_SG2_MOBILITY_BASE_DOF 3
#define FFW_SG2_MOBILITY_BATCH_SIZE FFW_SG2_BATCH_SIZE
#define FFW_SG2_MOBILITY_BASE_SPHERE_COUNT 1
#define FFW_SG2_MOBILITY_APPROX_BASE_SPHERE_COUNT 1
#define FFW_SG2_MOBILITY_SPHERE_COUNT (FFW_SG2_SPHERE_COUNT + FFW_SG2_MOBILITY_BASE_SPHERE_COUNT)
#define FFW_SG2_MOBILITY_APPROX_SPHERE_COUNT (FFW_SG2_APPROX_SPHERE_COUNT + FFW_SG2_MOBILITY_APPROX_BASE_SPHERE_COUNT)
#define FFW_SG2_MOBILITY_JOINT_FLAG_STRIDE FFW_SG2_JOINT_FLAG_STRIDE
#define FFW_SG2_MOBILITY_TRANSFORM_SLOTS FFW_SG2_TRANSFORM_SLOTS

__device__ __constant__ float4 ffw_sg2_mobility_base_spheres_array[
    FFW_SG2_MOBILITY_BASE_SPHERE_COUNT
] = {
    {0.0f, 0.0f, 0.24f, 0.307f},
};

__device__ __forceinline__ void ffw_sg2_mobility_apply_base_pose(
    float base_x,
    float base_y,
    float base_yaw,
    float local_x,
    float local_y,
    float local_z,
    float &world_x,
    float &world_y,
    float &world_z
) {
    const float c = __cosf(base_yaw);
    const float s = __sinf(base_yaw);
    world_x = base_x + c * local_x - s * local_y;
    world_y = base_y + s * local_x + c * local_y;
    world_z = local_z;
}

__device__ __forceinline__ void ffw_sg2_mobility_transform_spheres(
    volatile float *sphere_pos,
    int sphere_count,
    int batch_ind,
    float base_x,
    float base_y,
    float base_yaw
) {
    for (int sphere = 0; sphere < sphere_count; ++sphere) {
        const int offset =
            sphere * FFW_SG2_MOBILITY_BATCH_SIZE * 3 + batch_ind * 3;
        const float local_x = sphere_pos[offset + 0];
        const float local_y = sphere_pos[offset + 1];
        const float local_z = sphere_pos[offset + 2];
        float world_x, world_y, world_z;
        ffw_sg2_mobility_apply_base_pose(
            base_x,
            base_y,
            base_yaw,
            local_x,
            local_y,
            local_z,
            world_x,
            world_y,
            world_z
        );
        sphere_pos[offset + 0] = world_x;
        sphere_pos[offset + 1] = world_y;
        sphere_pos[offset + 2] = world_z;
    }
}

__device__ __forceinline__ void ffw_sg2_mobility_write_base_spheres(
    volatile float *sphere_pos,
    int first_sphere_index,
    int batch_ind,
    float base_x,
    float base_y,
    float base_yaw
) {
    for (int sphere = 0; sphere < FFW_SG2_MOBILITY_BASE_SPHERE_COUNT; ++sphere) {
        const float4 local = ffw_sg2_mobility_base_spheres_array[sphere];
        float world_x, world_y, world_z;
        ffw_sg2_mobility_apply_base_pose(
            base_x,
            base_y,
            base_yaw,
            local.x,
            local.y,
            local.z,
            world_x,
            world_y,
            world_z
        );
        const int offset =
            (first_sphere_index + sphere) *
            FFW_SG2_MOBILITY_BATCH_SIZE * 3 +
            batch_ind * 3;
        sphere_pos[offset + 0] = world_x;
        sphere_pos[offset + 1] = world_y;
        sphere_pos[offset + 2] = world_z;
    }
}

template <>
__device__ void fk_approx<ppln::robots::FfwSg2Mobility>(
    const float *q,
    volatile float *sphere_pos_approx,
    float *T,
    const int tid
) {
    fk_approx<ppln::robots::FfwSg2>(
        q + FFW_SG2_MOBILITY_BASE_DOF,
        sphere_pos_approx,
        T,
        tid
    );
    __syncthreads();

    const int lane = tid % 4;
    const int batch_ind = tid / 4;
    if (batch_ind < FFW_SG2_MOBILITY_BATCH_SIZE && lane == 0) {
        ffw_sg2_mobility_transform_spheres(
            sphere_pos_approx,
            FFW_SG2_APPROX_SPHERE_COUNT,
            batch_ind,
            q[0],
            q[1],
            q[2]
        );
        ffw_sg2_mobility_write_base_spheres(
            sphere_pos_approx,
            FFW_SG2_APPROX_SPHERE_COUNT,
            batch_ind,
            q[0],
            q[1],
            q[2]
        );
    }
    __syncthreads();
}

template <>
__device__ void fk<ppln::robots::FfwSg2Mobility>(
    const float *q,
    volatile float *sphere_pos,
    float *T,
    const int tid
) {
    fk<ppln::robots::FfwSg2>(
        q + FFW_SG2_MOBILITY_BASE_DOF,
        sphere_pos,
        T,
        tid
    );
    __syncthreads();

    const int lane = tid % 4;
    const int batch_ind = tid / 4;
    if (batch_ind < FFW_SG2_MOBILITY_BATCH_SIZE && lane == 0) {
        ffw_sg2_mobility_transform_spheres(
            sphere_pos,
            FFW_SG2_SPHERE_COUNT,
            batch_ind,
            q[0],
            q[1],
            q[2]
        );
        ffw_sg2_mobility_write_base_spheres(
            sphere_pos,
            FFW_SG2_SPHERE_COUNT,
            batch_ind,
            q[0],
            q[1],
            q[2]
        );
    }
    __syncthreads();
}

__device__ __forceinline__ bool ffw_sg2_mobility_base_env_collision(
    volatile float *sphere_pos,
    int first_sphere_index,
    volatile int *joint_in_collision,
    ppln::collision::Environment<float> *env,
    const int tid,
    volatile unsigned int *motion_cc_flag = nullptr
) {
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool collision_free = true;

    for (
        int sphere = thread_ind;
        sphere < FFW_SG2_MOBILITY_BASE_SPHERE_COUNT;
        sphere += 4
    ) {
        if (motion_cc_should_stop(motion_cc_flag)) {
            return false;
        }

        const int sphere_index = first_sphere_index + sphere;
        if (
            sphere_environment_in_collision(
                env,
                sphere_pos[
                    sphere_index * FFW_SG2_MOBILITY_BATCH_SIZE * 3 +
                    batch_ind * 3 + 0
                ],
                sphere_pos[
                    sphere_index * FFW_SG2_MOBILITY_BATCH_SIZE * 3 +
                    batch_ind * 3 + 1
                ],
                sphere_pos[
                    sphere_index * FFW_SG2_MOBILITY_BATCH_SIZE * 3 +
                    batch_ind * 3 + 2
                ],
                ffw_sg2_mobility_base_spheres_array[sphere].w
            )
        ) {
            atomicAdd(
                (int *)&joint_in_collision[
                    FFW_SG2_MOBILITY_JOINT_FLAG_STRIDE * batch_ind
                ],
                1
            );
            motion_cc_report_collision(motion_cc_flag);
            collision_free = false;
        }
    }

    return collision_free;
}

template <>
__device__ bool env_collision_check_approx<ppln::robots::FfwSg2Mobility>(
    volatile float *sphere_pos_approx,
    volatile int *joint_in_collision,
    ppln::collision::Environment<float> *env,
    const int tid
) {
    const bool arm_collision_free =
        env_collision_check_approx<ppln::robots::FfwSg2>(
            sphere_pos_approx,
            joint_in_collision,
            env,
            tid
        );
    const bool base_collision_free = ffw_sg2_mobility_base_env_collision(
        sphere_pos_approx,
        FFW_SG2_APPROX_SPHERE_COUNT,
        joint_in_collision,
        env,
        tid
    );
    return arm_collision_free && base_collision_free;
}

template <>
__device__ bool env_collision_check<ppln::robots::FfwSg2Mobility>(
    volatile float *sphere_pos,
    volatile int *joint_in_collision,
    ppln::collision::Environment<float> *env,
    const int tid
) {
    const bool arm_collision_free =
        env_collision_check<ppln::robots::FfwSg2>(
            sphere_pos,
            joint_in_collision,
            env,
            tid
        );
    const bool base_collision_free = ffw_sg2_mobility_base_env_collision(
        sphere_pos,
        FFW_SG2_SPHERE_COUNT,
        joint_in_collision,
        env,
        tid
    );
    return arm_collision_free && base_collision_free;
}

__device__ __forceinline__ bool ffw_sg2_mobility_env_collision_check_early(
    volatile float *sphere_pos,
    volatile int *joint_in_collision,
    ppln::collision::Environment<float> *env,
    const int tid,
    volatile unsigned int *motion_cc_flag
) {
    if (
        !ffw_sg2_env_collision_check_early(
            sphere_pos,
            joint_in_collision,
            env,
            tid,
            motion_cc_flag
        )
    ) {
        return false;
    }

    return ffw_sg2_mobility_base_env_collision(
        sphere_pos,
        FFW_SG2_SPHERE_COUNT,
        joint_in_collision,
        env,
        tid,
        motion_cc_flag
    );
}

template <>
__device__ bool self_collision_check_approx<ppln::robots::FfwSg2Mobility>(
    volatile float *sphere_pos_approx,
    volatile int *joint_in_collision,
    const int tid
) {
    return self_collision_check_approx<ppln::robots::FfwSg2>(
        sphere_pos_approx,
        joint_in_collision,
        tid
    );
}

template <>
__device__ bool self_collision_check<ppln::robots::FfwSg2Mobility>(
    volatile float *sphere_pos,
    volatile int *joint_in_collision,
    const int tid
) {
    return self_collision_check<ppln::robots::FfwSg2>(
        sphere_pos,
        joint_in_collision,
        tid
    );
}

__device__ __forceinline__ bool ffw_sg2_mobility_self_collision_check_early(
    volatile float *sphere_pos,
    volatile int *joint_in_collision,
    const int tid,
    volatile unsigned int *motion_cc_flag
) {
    return ffw_sg2_self_collision_check_early(
        sphere_pos,
        joint_in_collision,
        tid,
        motion_cc_flag
    );
}

} // namespace ppln::collision

#pragma once

#include "src/planning/Robots.hh"
#include "src/planning/PATACON_settings.hh"
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

__device__ __constant__ FfwSg2AttachedObjectCollisionSpec
    ffw_sg2_mobility_attached_object_collision;

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

__device__ __forceinline__ void ffw_sg2_mobility_rotate_base_vector(
    float base_yaw,
    float local_x,
    float local_y,
    float local_z,
    float out[3]
) {
    const float c = __cosf(base_yaw);
    const float s = __sinf(base_yaw);
    out[0] = c * local_x - s * local_y;
    out[1] = s * local_x + c * local_y;
    out[2] = local_z;
}

__device__ __forceinline__ float ffw_sg2_mobility_dot3(
    const float a[3],
    const float b[3]
) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ void ffw_sg2_mobility_cross3(
    const float a[3],
    const float b[3],
    float out[3]
) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ bool ffw_sg2_mobility_normalize3(float v[3]) {
    const float len_sq = ffw_sg2_mobility_dot3(v, v);
    if (len_sq <= 1.0e-18f) {
        return false;
    }
    const float inv_len = 1.0f / sqrtf(len_sq);
    v[0] *= inv_len;
    v[1] *= inv_len;
    v[2] *= inv_len;
    return true;
}

__device__ __forceinline__ void ffw_sg2_mobility_matrix_column_world(
    float base_yaw,
    const float R[9],
    int column,
    float out[3]
) {
    ffw_sg2_mobility_rotate_base_vector(
        base_yaw,
        R[column],
        R[3 + column],
        R[6 + column],
        out
    );
}

__device__ __forceinline__ void ffw_sg2_mobility_attached_object_frame(
    const float *q,
    float origin[3],
    float x_axis[3],
    float y_axis[3],
    float z_axis[3]
) {
    const FfwSg2AttachedObjectCollisionSpec &spec =
        ffw_sg2_mobility_attached_object_collision;
    const float *arm_q = q + FFW_SG2_MOBILITY_BASE_DOF;

    float left_local[3], right_local[3];
    float left_R[9], right_R[9];
    ffw_sg2_fk_left(arm_q, left_local, left_R);
    ffw_sg2_fk_right(arm_q, right_local, right_R);

    float left_world[3], right_world[3];
    ffw_sg2_mobility_apply_base_pose(
        q[0],
        q[1],
        q[2],
        left_local[0],
        left_local[1],
        left_local[2],
        left_world[0],
        left_world[1],
        left_world[2]
    );
    ffw_sg2_mobility_apply_base_pose(
        q[0],
        q[1],
        q[2],
        right_local[0],
        right_local[1],
        right_local[2],
        right_world[0],
        right_world[1],
        right_world[2]
    );

    const float center[3] = {
        0.5f * (left_world[0] + right_world[0]),
        0.5f * (left_world[1] + right_world[1]),
        0.5f * (left_world[2] + right_world[2])
    };

    y_axis[0] = left_world[0] - right_world[0];
    y_axis[1] = left_world[1] - right_world[1];
    y_axis[2] = left_world[2] - right_world[2];
    if (!ffw_sg2_mobility_normalize3(y_axis)) {
        x_axis[0] = 1.0f;
        x_axis[1] = 0.0f;
        x_axis[2] = 0.0f;
        y_axis[0] = 0.0f;
        y_axis[1] = 1.0f;
        y_axis[2] = 0.0f;
        z_axis[0] = 0.0f;
        z_axis[1] = 0.0f;
        z_axis[2] = 1.0f;
        origin[0] = center[0] + spec.world_offset[0];
        origin[1] = center[1] + spec.world_offset[1];
        origin[2] = center[2] + spec.world_offset[2];
        return;
    }

    float left_z[3], right_z[3];
    ffw_sg2_mobility_matrix_column_world(q[2], left_R, 2, left_z);
    ffw_sg2_mobility_matrix_column_world(q[2], right_R, 2, right_z);
    x_axis[0] = left_z[0] + right_z[0];
    x_axis[1] = left_z[1] + right_z[1];
    x_axis[2] = left_z[2] + right_z[2];
    if (!ffw_sg2_mobility_normalize3(x_axis)) {
        x_axis[0] = 1.0f;
        x_axis[1] = 0.0f;
        x_axis[2] = 0.0f;
    }

    const float projection = ffw_sg2_mobility_dot3(x_axis, y_axis);
    x_axis[0] -= projection * y_axis[0];
    x_axis[1] -= projection * y_axis[1];
    x_axis[2] -= projection * y_axis[2];
    if (!ffw_sg2_mobility_normalize3(x_axis)) {
        const float world_z[3] = {0.0f, 0.0f, 1.0f};
        ffw_sg2_mobility_cross3(y_axis, world_z, x_axis);
    }
    if (!ffw_sg2_mobility_normalize3(x_axis)) {
        x_axis[0] = 1.0f;
        x_axis[1] = 0.0f;
        x_axis[2] = 0.0f;
    }

    ffw_sg2_mobility_cross3(x_axis, y_axis, z_axis);
    if (!ffw_sg2_mobility_normalize3(z_axis)) {
        x_axis[0] = 1.0f;
        x_axis[1] = 0.0f;
        x_axis[2] = 0.0f;
        y_axis[0] = 0.0f;
        y_axis[1] = 1.0f;
        y_axis[2] = 0.0f;
        z_axis[0] = 0.0f;
        z_axis[1] = 0.0f;
        z_axis[2] = 1.0f;
        origin[0] = center[0] + spec.world_offset[0];
        origin[1] = center[1] + spec.world_offset[1];
        origin[2] = center[2] + spec.world_offset[2];
        return;
    }

    float left_y[3], right_y[3], z_hint[3];
    ffw_sg2_mobility_matrix_column_world(q[2], left_R, 1, left_y);
    ffw_sg2_mobility_matrix_column_world(q[2], right_R, 1, right_y);
    z_hint[0] = left_y[0] - right_y[0];
    z_hint[1] = left_y[1] - right_y[1];
    z_hint[2] = left_y[2] - right_y[2];
    if (
        ffw_sg2_mobility_normalize3(z_hint) &&
        ffw_sg2_mobility_dot3(z_axis, z_hint) < 0.0f
    ) {
        x_axis[0] = -x_axis[0];
        x_axis[1] = -x_axis[1];
        x_axis[2] = -x_axis[2];
        z_axis[0] = -z_axis[0];
        z_axis[1] = -z_axis[1];
        z_axis[2] = -z_axis[2];
    }

    origin[0] =
        center[0] +
        x_axis[0] * spec.world_offset[0] +
        y_axis[0] * spec.world_offset[1] +
        z_axis[0] * spec.world_offset[2];
    origin[1] =
        center[1] +
        x_axis[1] * spec.world_offset[0] +
        y_axis[1] * spec.world_offset[1] +
        z_axis[1] * spec.world_offset[2];
    origin[2] =
        center[2] +
        x_axis[2] * spec.world_offset[0] +
        y_axis[2] * spec.world_offset[1] +
        z_axis[2] * spec.world_offset[2];
}

__device__ __forceinline__ void
ffw_sg2_mobility_attached_object_sphere_world(
    const float origin[3],
    const float x_axis[3],
    const float y_axis[3],
    const float z_axis[3],
    int sphere_index,
    float &world_x,
    float &world_y,
    float &world_z,
    float &radius
) {
    const FfwSg2AttachedObjectCollisionSpec &spec =
        ffw_sg2_mobility_attached_object_collision;
    const float local_x = spec.spheres[sphere_index][0];
    const float local_y = spec.spheres[sphere_index][1];
    const float local_z = spec.spheres[sphere_index][2];
    radius = spec.spheres[sphere_index][3];
    world_x =
        origin[0] +
        x_axis[0] * local_x +
        y_axis[0] * local_y +
        z_axis[0] * local_z;
    world_y =
        origin[1] +
        x_axis[1] * local_x +
        y_axis[1] * local_y +
        z_axis[1] * local_z;
    world_z =
        origin[2] +
        x_axis[2] * local_x +
        y_axis[2] * local_y +
        z_axis[2] * local_z;
}

__device__ __forceinline__ bool
ffw_sg2_mobility_attached_object_ignores_robot_sphere(
    int robot_sphere,
    bool approximate
) {
    const FfwSg2AttachedObjectCollisionSpec &spec =
        ffw_sg2_mobility_attached_object_collision;
    const int raw_count =
        approximate ?
            spec.ignored_robot_approx_sphere_count :
            spec.ignored_robot_sphere_count;
    const int max_count =
        approximate ?
            FFW_SG2_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_APPROX_SPHERES :
            FFW_SG2_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_SPHERES;
    const int count = raw_count < max_count ? raw_count : max_count;

    for (int i = 0; i < count; ++i) {
        const int ignored =
            approximate ?
                spec.ignored_robot_approx_spheres[i] :
                spec.ignored_robot_spheres[i];
        if (ignored == robot_sphere) {
            return true;
        }
    }

    return false;
}

__device__ __forceinline__ float
ffw_sg2_mobility_robot_sphere_radius(int sphere_index, bool approximate) {
    if (approximate) {
        if (sphere_index < FFW_SG2_APPROX_SPHERE_COUNT) {
            return ffw_sg2_approx_spheres_array[sphere_index].w;
        }
        return ffw_sg2_mobility_base_spheres_array[
            sphere_index - FFW_SG2_APPROX_SPHERE_COUNT
        ].w;
    }

    if (sphere_index < FFW_SG2_SPHERE_COUNT) {
        return ffw_sg2_spheres_array[sphere_index].w;
    }
    return ffw_sg2_mobility_base_spheres_array[
        sphere_index - FFW_SG2_SPHERE_COUNT
    ].w;
}

__device__ __forceinline__ bool
ffw_sg2_mobility_attached_object_env_collision_check(
    const float *q,
    ppln::collision::Environment<float> *env,
    const int tid,
    volatile unsigned int *motion_cc_flag = nullptr
) {
    const FfwSg2AttachedObjectCollisionSpec &spec =
        ffw_sg2_mobility_attached_object_collision;
    if (!spec.enabled || spec.sphere_count <= 0) {
        return true;
    }

    const int thread_ind = tid % 4;
    const int object_sphere_count =
        spec.sphere_count < FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES ?
            spec.sphere_count :
            FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES;

    float origin[3], x_axis[3], y_axis[3], z_axis[3];
    ffw_sg2_mobility_attached_object_frame(
        q,
        origin,
        x_axis,
        y_axis,
        z_axis
    );

    bool collision_free = true;
    for (
        int object_sphere = thread_ind;
        object_sphere < object_sphere_count;
        object_sphere += 4
    ) {
        if (motion_cc_should_stop(motion_cc_flag)) {
            return false;
        }

        float object_x, object_y, object_z, object_r;
        ffw_sg2_mobility_attached_object_sphere_world(
            origin,
            x_axis,
            y_axis,
            z_axis,
            object_sphere,
            object_x,
            object_y,
            object_z,
            object_r
        );
        if (object_r <= 0.0f) {
            continue;
        }

        if (
            sphere_environment_in_collision(
                env,
                object_x,
                object_y,
                object_z,
                object_r
            )
        ) {
            motion_cc_report_collision(motion_cc_flag);
            collision_free = false;
        }
    }

    return collision_free;
}

__device__ __forceinline__ bool
ffw_sg2_mobility_attached_object_robot_collision_check(
    const float *q,
    volatile float *sphere_pos,
    bool approximate,
    const int tid,
    volatile unsigned int *motion_cc_flag = nullptr
) {
    const FfwSg2AttachedObjectCollisionSpec &spec =
        ffw_sg2_mobility_attached_object_collision;
    if (!spec.enabled || spec.sphere_count <= 0) {
        return true;
    }

    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    const int object_sphere_count =
        spec.sphere_count < FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES ?
            spec.sphere_count :
            FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES;
    const int robot_sphere_count =
        approximate ?
            FFW_SG2_MOBILITY_APPROX_SPHERE_COUNT :
            FFW_SG2_MOBILITY_SPHERE_COUNT;

    float origin[3], x_axis[3], y_axis[3], z_axis[3];
    ffw_sg2_mobility_attached_object_frame(
        q,
        origin,
        x_axis,
        y_axis,
        z_axis
    );

    bool collision_free = true;
    for (int object_sphere = 0; object_sphere < object_sphere_count; ++object_sphere) {
        if (motion_cc_should_stop(motion_cc_flag)) {
            return false;
        }

        float object_x, object_y, object_z, object_r;
        ffw_sg2_mobility_attached_object_sphere_world(
            origin,
            x_axis,
            y_axis,
            z_axis,
            object_sphere,
            object_x,
            object_y,
            object_z,
            object_r
        );
        if (object_r <= 0.0f) {
            continue;
        }

        for (
            int robot_sphere = thread_ind;
            robot_sphere < robot_sphere_count;
            robot_sphere += 4
        ) {
            if (motion_cc_should_stop(motion_cc_flag)) {
                return false;
            }
            if (
                ffw_sg2_mobility_attached_object_ignores_robot_sphere(
                    robot_sphere,
                    approximate
                )
            ) {
                continue;
            }

            const int offset =
                robot_sphere * FFW_SG2_MOBILITY_BATCH_SIZE * 3 +
                batch_ind * 3;
            if (
                sphere_sphere_self_collision(
                    object_x,
                    object_y,
                    object_z,
                    object_r,
                    sphere_pos[offset + 0],
                    sphere_pos[offset + 1],
                    sphere_pos[offset + 2],
                    ffw_sg2_mobility_robot_sphere_radius(
                        robot_sphere,
                        approximate
                    )
                )
            ) {
                motion_cc_report_collision(motion_cc_flag);
                collision_free = false;
            }
        }
    }

    return collision_free;
}

__device__ __forceinline__ bool
ffw_sg2_mobility_attached_object_collision_check_approx(
    const float *q,
    volatile float *sphere_pos_approx,
    ppln::collision::Environment<float> *env,
    const int tid,
    volatile unsigned int *motion_cc_flag = nullptr
) {
    const bool env_collision_free =
        ffw_sg2_mobility_attached_object_env_collision_check(
            q,
            env,
            tid,
            motion_cc_flag
        );
    const bool robot_collision_free =
        ffw_sg2_mobility_attached_object_robot_collision_check(
            q,
            sphere_pos_approx,
            true,
            tid,
            motion_cc_flag
        );
    return env_collision_free && robot_collision_free;
}

__device__ __forceinline__ bool
ffw_sg2_mobility_attached_object_collision_check(
    const float *q,
    volatile float *sphere_pos,
    ppln::collision::Environment<float> *env,
    const int tid,
    volatile unsigned int *motion_cc_flag = nullptr
) {
    const bool env_collision_free =
        ffw_sg2_mobility_attached_object_env_collision_check(
            q,
            env,
            tid,
            motion_cc_flag
        );
    const bool robot_collision_free =
        ffw_sg2_mobility_attached_object_robot_collision_check(
            q,
            sphere_pos,
            false,
            tid,
            motion_cc_flag
        );
    return env_collision_free && robot_collision_free;
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
inline __device__ void fk_approx<ppln::robots::FfwSg2Mobility>(
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
inline __device__ void fk<ppln::robots::FfwSg2Mobility>(
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
inline __device__ bool env_collision_check_approx<ppln::robots::FfwSg2Mobility>(
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
inline __device__ bool env_collision_check<ppln::robots::FfwSg2Mobility>(
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
inline __device__ bool self_collision_check_approx<ppln::robots::FfwSg2Mobility>(
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
inline __device__ bool self_collision_check<ppln::robots::FfwSg2Mobility>(
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

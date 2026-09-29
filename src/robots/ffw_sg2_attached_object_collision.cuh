#pragma once

#include "src/planning/PATACON_settings.hh"
#include "src/planning/utils.cuh"
#include "src/robots/ffw_sg2.cuh"
#include "src/robots/ffw_sg2_constraint.cuh"
#include "src/robots/ffw_sg2_mobility.cuh"

namespace ppln::collision {

__device__ __forceinline__ float ffw_sg2_attached_object_dot3(
    const float a[3],
    const float b[3]
) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ void ffw_sg2_attached_object_cross3(
    const float a[3],
    const float b[3],
    float out[3]
) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ bool ffw_sg2_attached_object_normalize3(
    float value[3]
) {
    const float norm_squared = ffw_sg2_attached_object_dot3(value, value);
    if (norm_squared <= 1.0e-18f) {
        return false;
    }
    const float inverse_norm = rsqrtf(norm_squared);
    value[0] *= inverse_norm;
    value[1] *= inverse_norm;
    value[2] *= inverse_norm;
    return true;
}

__device__ __forceinline__ void ffw_sg2_attached_object_matrix_column(
    const float matrix[9],
    int column,
    float out[3]
) {
    out[0] = matrix[column];
    out[1] = matrix[3 + column];
    out[2] = matrix[6 + column];
}

// Object coordinates follow the frame used by visualize_ffw_sg2.py:
// origin at the gripper midpoint, +Y from right to left, and +X from the
// average gripper Z direction.  world_offset is expressed in this frame.
__device__ __forceinline__ void ffw_sg2_attached_object_frame(
    const float *q,
    float origin[3],
    float x_axis[3],
    float y_axis[3],
    float z_axis[3]
) {
    const FfwSg2AttachedObjectCollisionSpec &spec =
        ffw_sg2_mobility_attached_object_collision;

    float left_position[3], right_position[3];
    float left_rotation[9], right_rotation[9];
    ffw_sg2_fk_left(q, left_position, left_rotation);
    ffw_sg2_fk_right(q, right_position, right_rotation);

    const float center[3] = {
        0.5f * (left_position[0] + right_position[0]),
        0.5f * (left_position[1] + right_position[1]),
        0.5f * (left_position[2] + right_position[2]),
    };

    y_axis[0] = left_position[0] - right_position[0];
    y_axis[1] = left_position[1] - right_position[1];
    y_axis[2] = left_position[2] - right_position[2];
    if (!ffw_sg2_attached_object_normalize3(y_axis)) {
        x_axis[0] = 1.0f;
        x_axis[1] = 0.0f;
        x_axis[2] = 0.0f;
        y_axis[0] = 0.0f;
        y_axis[1] = 1.0f;
        y_axis[2] = 0.0f;
        z_axis[0] = 0.0f;
        z_axis[1] = 0.0f;
        z_axis[2] = 1.0f;
    } else {
        float left_z[3], right_z[3];
        ffw_sg2_attached_object_matrix_column(left_rotation, 2, left_z);
        ffw_sg2_attached_object_matrix_column(right_rotation, 2, right_z);
        x_axis[0] = left_z[0] + right_z[0];
        x_axis[1] = left_z[1] + right_z[1];
        x_axis[2] = left_z[2] + right_z[2];
        if (!ffw_sg2_attached_object_normalize3(x_axis)) {
            x_axis[0] = 1.0f;
            x_axis[1] = 0.0f;
            x_axis[2] = 0.0f;
        }

        const float projection =
            ffw_sg2_attached_object_dot3(x_axis, y_axis);
        x_axis[0] -= projection * y_axis[0];
        x_axis[1] -= projection * y_axis[1];
        x_axis[2] -= projection * y_axis[2];
        if (!ffw_sg2_attached_object_normalize3(x_axis)) {
            const float world_z[3] = {0.0f, 0.0f, 1.0f};
            ffw_sg2_attached_object_cross3(y_axis, world_z, x_axis);
        }
        if (!ffw_sg2_attached_object_normalize3(x_axis)) {
            x_axis[0] = 1.0f;
            x_axis[1] = 0.0f;
            x_axis[2] = 0.0f;
        }

        ffw_sg2_attached_object_cross3(x_axis, y_axis, z_axis);
        if (!ffw_sg2_attached_object_normalize3(z_axis)) {
            x_axis[0] = 1.0f;
            x_axis[1] = 0.0f;
            x_axis[2] = 0.0f;
            y_axis[0] = 0.0f;
            y_axis[1] = 1.0f;
            y_axis[2] = 0.0f;
            z_axis[0] = 0.0f;
            z_axis[1] = 0.0f;
            z_axis[2] = 1.0f;
        } else {
            float left_y[3], right_y[3], z_hint[3];
            ffw_sg2_attached_object_matrix_column(left_rotation, 1, left_y);
            ffw_sg2_attached_object_matrix_column(right_rotation, 1, right_y);
            // ffw_sg2_fk_* uses the constraint gripper frame.  Its Y axis is
            // the negative of the MuJoCo gripper-site Y axis used by the
            // visualizer, so compensate that sign when disambiguating Z.
            z_hint[0] = left_y[0] - right_y[0];
            z_hint[1] = left_y[1] - right_y[1];
            z_hint[2] = left_y[2] - right_y[2];
            if (
                ffw_sg2_attached_object_normalize3(z_hint) &&
                ffw_sg2_attached_object_dot3(z_axis, z_hint) < 0.0f
            ) {
                x_axis[0] = -x_axis[0];
                x_axis[1] = -x_axis[1];
                x_axis[2] = -x_axis[2];
                z_axis[0] = -z_axis[0];
                z_axis[1] = -z_axis[1];
                z_axis[2] = -z_axis[2];
            }
        }
    }

    origin[0] = center[0] +
        x_axis[0] * spec.world_offset[0] +
        y_axis[0] * spec.world_offset[1] +
        z_axis[0] * spec.world_offset[2];
    origin[1] = center[1] +
        x_axis[1] * spec.world_offset[0] +
        y_axis[1] * spec.world_offset[1] +
        z_axis[1] * spec.world_offset[2];
    origin[2] = center[2] +
        x_axis[2] * spec.world_offset[0] +
        y_axis[2] * spec.world_offset[1] +
        z_axis[2] * spec.world_offset[2];
}

__device__ __forceinline__ void ffw_sg2_attached_object_sphere_world(
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
    const float *sphere =
        ffw_sg2_mobility_attached_object_collision.spheres[sphere_index];
    world_x = origin[0] +
        x_axis[0] * sphere[0] +
        y_axis[0] * sphere[1] +
        z_axis[0] * sphere[2];
    world_y = origin[1] +
        x_axis[1] * sphere[0] +
        y_axis[1] * sphere[1] +
        z_axis[1] * sphere[2];
    world_z = origin[2] +
        x_axis[2] * sphere[0] +
        y_axis[2] * sphere[1] +
        z_axis[2] * sphere[2];
    radius = sphere[3];
}

__device__ __forceinline__ bool
ffw_sg2_attached_object_ignores_robot_sphere(
    int robot_sphere,
    bool approximate
) {
    const FfwSg2AttachedObjectCollisionSpec &spec =
        ffw_sg2_mobility_attached_object_collision;
    const int raw_count = approximate ?
        spec.ignored_robot_approx_sphere_count :
        spec.ignored_robot_sphere_count;
    const int maximum_count = approximate ?
        FFW_SG2_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_APPROX_SPHERES :
        FFW_SG2_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_SPHERES;
    const int count = raw_count < maximum_count ? raw_count : maximum_count;
    for (int index = 0; index < count; ++index) {
        const int ignored = approximate ?
            spec.ignored_robot_approx_spheres[index] :
            spec.ignored_robot_spheres[index];
        if (ignored == robot_sphere) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ float ffw_sg2_robot_sphere_radius(
    int sphere_index,
    bool approximate
) {
    return approximate ?
        ffw_sg2_approx_spheres_array[sphere_index].w :
        ffw_sg2_spheres_array[sphere_index].w;
}

__device__ __forceinline__ bool
ffw_sg2_attached_object_environment_collision_check(
    const float *q,
    ppln::collision::Environment<float> *environment,
    int tid,
    volatile unsigned int *motion_cc_flag = nullptr
) {
    const FfwSg2AttachedObjectCollisionSpec &spec =
        ffw_sg2_mobility_attached_object_collision;
    if (!spec.enabled || spec.sphere_count <= 0) {
        return true;
    }

    const int lane = tid % 4;
    const int sphere_count =
        spec.sphere_count < FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES ?
            spec.sphere_count :
            FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES;
    float origin[3], x_axis[3], y_axis[3], z_axis[3];
    ffw_sg2_attached_object_frame(q, origin, x_axis, y_axis, z_axis);

    bool collision_free = true;
    for (int sphere = lane; sphere < sphere_count; sphere += 4) {
        if (motion_cc_should_stop(motion_cc_flag)) {
            return false;
        }
        float world_x, world_y, world_z, radius;
        ffw_sg2_attached_object_sphere_world(
            origin,
            x_axis,
            y_axis,
            z_axis,
            sphere,
            world_x,
            world_y,
            world_z,
            radius
        );
        if (
            radius > 0.0f &&
            sphere_environment_in_collision(
                environment,
                world_x,
                world_y,
                world_z,
                radius
            )
        ) {
            motion_cc_report_collision(motion_cc_flag);
            collision_free = false;
        }
    }
    return collision_free;
}

__device__ __forceinline__ bool
ffw_sg2_attached_object_robot_collision_check(
    const float *q,
    volatile float *robot_sphere_positions,
    bool approximate,
    int tid,
    volatile unsigned int *motion_cc_flag = nullptr
) {
    const FfwSg2AttachedObjectCollisionSpec &spec =
        ffw_sg2_mobility_attached_object_collision;
    if (!spec.enabled || spec.sphere_count <= 0) {
        return true;
    }

    const int lane = tid % 4;
    const int batch = tid / 4;
    const int object_sphere_count =
        spec.sphere_count < FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES ?
            spec.sphere_count :
            FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES;
    const int robot_sphere_count = approximate ?
        FFW_SG2_APPROX_SPHERE_COUNT : FFW_SG2_SPHERE_COUNT;
    float origin[3], x_axis[3], y_axis[3], z_axis[3];
    ffw_sg2_attached_object_frame(q, origin, x_axis, y_axis, z_axis);

    bool collision_free = true;
    for (int object_sphere = 0;
         object_sphere < object_sphere_count;
         ++object_sphere) {
        if (motion_cc_should_stop(motion_cc_flag)) {
            return false;
        }
        float object_x, object_y, object_z, object_radius;
        ffw_sg2_attached_object_sphere_world(
            origin,
            x_axis,
            y_axis,
            z_axis,
            object_sphere,
            object_x,
            object_y,
            object_z,
            object_radius
        );
        if (object_radius <= 0.0f) {
            continue;
        }

        for (int robot_sphere = lane;
             robot_sphere < robot_sphere_count;
             robot_sphere += 4) {
            if (motion_cc_should_stop(motion_cc_flag)) {
                return false;
            }
            if (ffw_sg2_attached_object_ignores_robot_sphere(
                    robot_sphere,
                    approximate)) {
                continue;
            }
            const int offset =
                robot_sphere * FFW_SG2_BATCH_SIZE * 3 + batch * 3;
            if (sphere_sphere_self_collision(
                    object_x,
                    object_y,
                    object_z,
                    object_radius,
                    robot_sphere_positions[offset + 0],
                    robot_sphere_positions[offset + 1],
                    robot_sphere_positions[offset + 2],
                    ffw_sg2_robot_sphere_radius(robot_sphere, approximate))) {
                motion_cc_report_collision(motion_cc_flag);
                collision_free = false;
            }
        }
    }
    return collision_free;
}

__device__ __forceinline__ bool
ffw_sg2_attached_object_collision_check_approx(
    const float *q,
    volatile float *robot_sphere_positions,
    ppln::collision::Environment<float> *environment,
    int tid,
    volatile unsigned int *motion_cc_flag = nullptr
) {
    const bool environment_free =
        ffw_sg2_attached_object_environment_collision_check(
            q,
            environment,
            tid,
            motion_cc_flag
        );
    const bool robot_free = ffw_sg2_attached_object_robot_collision_check(
        q,
        robot_sphere_positions,
        true,
        tid,
        motion_cc_flag
    );
    return environment_free && robot_free;
}

__device__ __forceinline__ bool ffw_sg2_attached_object_collision_check(
    const float *q,
    volatile float *robot_sphere_positions,
    ppln::collision::Environment<float> *environment,
    int tid,
    volatile unsigned int *motion_cc_flag = nullptr
) {
    const bool environment_free =
        ffw_sg2_attached_object_environment_collision_check(
            q,
            environment,
            tid,
            motion_cc_flag
        );
    const bool robot_free = ffw_sg2_attached_object_robot_collision_check(
        q,
        robot_sphere_positions,
        false,
        tid,
        motion_cc_flag
    );
    return environment_free && robot_free;
}

} // namespace ppln::collision

#pragma once

#include "src/robots/ffw_sg2_mobility_constraint.cuh"

namespace ppln::collision {

#define FFW_SG2_MOBILITY_COM_EQUALITY_DIM FFW_SG2_MOBILITY_RESIDUAL_DIM
#define FFW_SG2_MOBILITY_COM_INEQUALITY_DIM 3
#define FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM \
    (FFW_SG2_MOBILITY_COM_EQUALITY_DIM + FFW_SG2_MOBILITY_COM_INEQUALITY_DIM)
#define FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS 8
#define FFW_SG2_MOBILITY_ATTACHED_OBJECT_FRAME_OFFSET_X 0.1f
#define FFW_SG2_MOBILITY_ATTACHED_OBJECT_FRAME_OFFSET_Y 0.0f
#define FFW_SG2_MOBILITY_ATTACHED_OBJECT_FRAME_OFFSET_Z 0.0f

__device__ __forceinline__ void ffw_sg2_mobility_com_transform_point(
    const float T[16],
    float local_x,
    float local_y,
    float local_z,
    float out[3]
) {
    out[0] =
        T[0] * local_x +
        T[1] * local_y +
        T[2] * local_z +
        T[3];
    out[1] =
        T[4] * local_x +
        T[5] * local_y +
        T[6] * local_z +
        T[7];
    out[2] =
        T[8] * local_x +
        T[9] * local_y +
        T[10] * local_z +
        T[11];
}

__device__ __forceinline__ void ffw_sg2_mobility_com_accumulate_link(
    const float T[16],
    float mass,
    float local_x,
    float local_y,
    float local_z,
    float weighted[3],
    float &total_mass
) {
    float point[3];
    ffw_sg2_mobility_com_transform_point(
        T,
        local_x,
        local_y,
        local_z,
        point
    );
    weighted[0] += mass * point[0];
    weighted[1] += mass * point[1];
    weighted[2] += mass * point[2];
    total_mass += mass;
}

__device__ __forceinline__ void ffw_sg2_mobility_com_attached_object_offset_base(
    const float left_p[3],
    const float right_p[3],
    const float left_R[9],
    const float right_R[9],
    float offset[3]
) {
    const FfwSg2AttachedObjectCollisionSpec &spec =
        ffw_sg2_mobility_attached_object_collision;
    const float frame_offset[3] = {
        spec.enabled ?
            spec.world_offset[0] :
            FFW_SG2_MOBILITY_ATTACHED_OBJECT_FRAME_OFFSET_X,
        spec.enabled ?
            spec.world_offset[1] :
            FFW_SG2_MOBILITY_ATTACHED_OBJECT_FRAME_OFFSET_Y,
        spec.enabled ?
            spec.world_offset[2] :
            FFW_SG2_MOBILITY_ATTACHED_OBJECT_FRAME_OFFSET_Z
    };

    float y_axis[3] = {
        left_p[0] - right_p[0],
        left_p[1] - right_p[1],
        left_p[2] - right_p[2]
    };
    if (!ffw_sg2_mobility_normalize3(y_axis)) {
        offset[0] = frame_offset[0];
        offset[1] = frame_offset[1];
        offset[2] = frame_offset[2];
        return;
    }

    float x_axis[3] = {
        left_R[2] + right_R[2],
        left_R[5] + right_R[5],
        left_R[8] + right_R[8]
    };
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
        const float base_z[3] = {0.0f, 0.0f, 1.0f};
        ffw_sg2_mobility_cross3(y_axis, base_z, x_axis);
    }
    if (!ffw_sg2_mobility_normalize3(x_axis)) {
        x_axis[0] = 1.0f;
        x_axis[1] = 0.0f;
        x_axis[2] = 0.0f;
    }

    float z_axis[3];
    ffw_sg2_mobility_cross3(x_axis, y_axis, z_axis);
    if (!ffw_sg2_mobility_normalize3(z_axis)) {
        offset[0] = frame_offset[0];
        offset[1] = frame_offset[1];
        offset[2] = frame_offset[2];
        return;
    }

    float z_hint[3] = {
        left_R[1] - right_R[1],
        left_R[4] - right_R[4],
        left_R[7] - right_R[7]
    };
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

    offset[0] =
        x_axis[0] * frame_offset[0] +
        y_axis[0] * frame_offset[1] +
        z_axis[0] * frame_offset[2];
    offset[1] =
        x_axis[1] * frame_offset[0] +
        y_axis[1] * frame_offset[1] +
        z_axis[1] * frame_offset[2];
    offset[2] =
        x_axis[2] * frame_offset[0] +
        y_axis[2] * frame_offset[1] +
        z_axis[2] * frame_offset[2];
}

__device__ __forceinline__ void ffw_sg2_mobility_com_apply_fixed_rpy(
    float T[16],
    float x,
    float y,
    float z,
    float roll,
    float pitch,
    float yaw
) {
    float step_tf[16];
    ffw_sg2_identity4(step_tf);

    const float cr = __cosf(roll);
    const float sr = __sinf(roll);
    const float cp = __cosf(pitch);
    const float sp = __sinf(pitch);
    const float cy = __cosf(yaw);
    const float sy = __sinf(yaw);

    step_tf[0] = cy * cp;
    step_tf[1] = cy * sp * sr - sy * cr;
    step_tf[2] = cy * sp * cr + sy * sr;
    step_tf[4] = sy * cp;
    step_tf[5] = sy * sp * sr + cy * cr;
    step_tf[6] = sy * sp * cr - cy * sr;
    step_tf[8] = -sp;
    step_tf[9] = cp * sr;
    step_tf[10] = cp * cr;

    step_tf[3] = x;
    step_tf[7] = y;
    step_tf[11] = z;

    ffw_sg2_apply_transform(T, step_tf);
}

__device__ __forceinline__ void ffw_sg2_mobility_com_accumulate_gripper(
    const float link7_T[16],
    float weighted[3],
    float &total_mass
) {
    float gripper_T[16];
    ffw_sg2_copy4(link7_T, gripper_T);
    ffw_sg2_apply_gripper_fixed(gripper_T);

    ffw_sg2_mobility_com_accumulate_link(
        gripper_T,
        0.236f,
        0.0f,
        0.0f,
        0.032f,
        weighted,
        total_mass
    );

    float r1_T[16];
    ffw_sg2_copy4(gripper_T, r1_T);
    ffw_sg2_apply_translation(r1_T, 0.0f, 0.008f, 0.048f);
    ffw_sg2_mobility_com_accumulate_link(
        r1_T,
        0.068f,
        0.0f,
        0.034f,
        0.004f,
        weighted,
        total_mass
    );

    float r2_T[16];
    ffw_sg2_copy4(r1_T, r2_T);
    ffw_sg2_apply_translation(r2_T, 0.0f, 0.0493634f, 0.0285f);
    ffw_sg2_mobility_com_accumulate_link(
        r2_T,
        0.022f,
        0.0f,
        0.006f,
        0.011f,
        weighted,
        total_mass
    );

    float l1_T[16];
    ffw_sg2_copy4(gripper_T, l1_T);
    ffw_sg2_apply_translation(l1_T, 0.0f, -0.008f, 0.048f);
    ffw_sg2_mobility_com_accumulate_link(
        l1_T,
        0.068f,
        0.0f,
        -0.034f,
        0.004f,
        weighted,
        total_mass
    );

    float l2_T[16];
    ffw_sg2_copy4(l1_T, l2_T);
    ffw_sg2_apply_translation(l2_T, 0.0f, -0.0493634f, 0.0285f);
    ffw_sg2_mobility_com_accumulate_link(
        l2_T,
        0.022f,
        0.0f,
        -0.006f,
        0.011f,
        weighted,
        total_mass
    );
}

__device__ __forceinline__ void ffw_sg2_mobility_com_accumulate_wheel(
    float origin_x,
    float origin_y,
    const float steer_com[3],
    const float drive_com[3],
    float weighted[3],
    float &total_mass
) {
    float steer_T[16];
    ffw_sg2_identity4(steer_T);
    ffw_sg2_apply_translation(steer_T, origin_x, origin_y, 0.27555f);
    ffw_sg2_mobility_com_accumulate_link(
        steer_T,
        0.7550951f,
        steer_com[0],
        steer_com[1],
        steer_com[2],
        weighted,
        total_mass
    );

    float drive_T[16];
    ffw_sg2_copy4(steer_T, drive_T);
    ffw_sg2_apply_translation(drive_T, 0.0f, 0.0f, -0.18905f);
    ffw_sg2_mobility_com_accumulate_link(
        drive_T,
        3.7954605f,
        drive_com[0],
        drive_com[1],
        drive_com[2],
        weighted,
        total_mass
    );
}

__device__ __forceinline__ void ffw_sg2_mobility_com_robot_com_base(
    const float q[FFW_SG2_MOBILITY_DIM],
    float com[3],
    float &robot_mass
) {
    const float *arm_q = q + FFW_SG2_MOBILITY_BASE_DOF;

    float weighted[3] = {0.0f, 0.0f, 0.0f};
    float total_mass = 0.0f;

    float identity_T[16];
    ffw_sg2_identity4(identity_T);
    ffw_sg2_mobility_com_accumulate_link(
        identity_T,
        35.989f,
        -0.045747964f,
        0.0065810588f,
        0.19431842f,
        weighted,
        total_mass
    );
    ffw_sg2_mobility_com_accumulate_link(
        identity_T,
        17.9f,
        -0.12869878f,
        -0.000016045808f,
        0.89456017f,
        weighted,
        total_mass
    );

    const float left_steer_com[3] = {
        0.0098027693f,
        -0.054712409f,
        -0.13765338f
    };
    const float right_steer_com[3] = {
        0.0098027286f,
        0.054712409f,
        -0.13765334f
    };
    const float drive_left_com[3] = {
        0.0f,
        -0.0014996017f,
        0.0f
    };
    const float drive_right_com[3] = {
        0.0f,
        0.0014996017f,
        0.0f
    };

    ffw_sg2_mobility_com_accumulate_wheel(
        0.1371f,
        0.2554f,
        left_steer_com,
        drive_left_com,
        weighted,
        total_mass
    );
    ffw_sg2_mobility_com_accumulate_wheel(
        0.1371f,
        -0.2554f,
        right_steer_com,
        drive_right_com,
        weighted,
        total_mass
    );
    ffw_sg2_mobility_com_accumulate_wheel(
        -0.2899f,
        0.0f,
        right_steer_com,
        drive_right_com,
        weighted,
        total_mass
    );

    float arm_base_T[16];
    ffw_sg2_identity4(arm_base_T);
    ffw_sg2_apply_prismatic_z(
        arm_base_T,
        -0.0199f,
        0.0f,
        1.4316f,
        arm_q[0]
    );
    ffw_sg2_mobility_com_accumulate_link(
        arm_base_T,
        6.1939559f,
        -0.014922878f,
        0.0027315225f,
        -0.043425264f,
        weighted,
        total_mass
    );

    float head1_T[16];
    ffw_sg2_copy4(arm_base_T, head1_T);
    ffw_sg2_apply_revolute_y(head1_T, 0.0494878f, 0.0f, 0.102123f, 0.0f);
    ffw_sg2_mobility_com_accumulate_link(
        head1_T,
        0.12351802f,
        0.025339656f,
        0.0f,
        0.047745146f,
        weighted,
        total_mass
    );

    float head2_T[16];
    ffw_sg2_copy4(head1_T, head2_T);
    ffw_sg2_apply_revolute_z(head2_T, 0.0365694f, 0.0f, 0.0563797f, 0.0f);
    ffw_sg2_mobility_com_accumulate_link(
        head2_T,
        0.3254213f,
        0.0030525634f,
        0.0011804757f,
        -0.015932804f,
        weighted,
        total_mass
    );

    float left_T[16];
    ffw_sg2_copy4(arm_base_T, left_T);
    ffw_sg2_apply_revolute_y(left_T, 0.0f, 0.1045f, 0.0f, arm_q[1]);
    ffw_sg2_mobility_com_accumulate_link(
        left_T,
        2.0013322f,
        0.014210373f,
        0.11553718f,
        0.0001120644f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_x(left_T, 0.0f, 0.123f, 0.0f, arm_q[2]);
    ffw_sg2_mobility_com_accumulate_link(
        left_T,
        2.128261f,
        0.0091098413f,
        -0.00010530165f,
        -0.14386407f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_z(left_T, 0.0f, 0.0f, -0.165f, arm_q[3]);
    ffw_sg2_mobility_com_accumulate_link(
        left_T,
        1.6847551f,
        0.029875373f,
        0.013913949f,
        -0.1210139f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_y(
        left_T,
        0.041004f,
        0.0f,
        -0.135f,
        arm_q[4]
    );
    ffw_sg2_mobility_com_accumulate_link(
        left_T,
        1.5082144f,
        -0.038538653f,
        0.0098072286f,
        -0.13131228f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_z(
        left_T,
        -0.041004f,
        0.0f,
        -0.1489f,
        arm_q[5]
    );
    ffw_sg2_mobility_com_accumulate_link(
        left_T,
        1.3917831f,
        -0.000031226161f,
        0.016819227f,
        -0.098738156f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_y(left_T, 0.0f, 0.0f, -0.1041f, arm_q[6]);
    ffw_sg2_mobility_com_accumulate_link(
        left_T,
        0.65770741f,
        0.0022438917f,
        0.018764105f,
        -0.064805576f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_x(left_T, 0.0f, 0.0f, -0.0885f, arm_q[7]);
    ffw_sg2_mobility_com_accumulate_link(
        left_T,
        0.1668563f,
        0.025506593f,
        0.0f,
        -0.055395331f,
        weighted,
        total_mass
    );

    float camera_left_T[16];
    ffw_sg2_copy4(left_T, camera_left_T);
    ffw_sg2_mobility_com_apply_fixed_rpy(
        camera_left_T,
        0.108236f,
        -0.021f,
        -0.062552f,
        -1.57079632679f,
        1.66678943569f,
        0.0f
    );
    ffw_sg2_apply_translation(camera_left_T, 0.01085f, 0.009f, 0.021f);
    ffw_sg2_mobility_com_accumulate_link(
        camera_left_T,
        0.072f,
        0.0f,
        0.0f,
        0.0f,
        weighted,
        total_mass
    );
    ffw_sg2_mobility_com_accumulate_gripper(
        left_T,
        weighted,
        total_mass
    );

    float right_T[16];
    ffw_sg2_copy4(arm_base_T, right_T);
    ffw_sg2_apply_revolute_y(right_T, 0.0f, -0.1045f, 0.0f, arm_q[8]);
    ffw_sg2_mobility_com_accumulate_link(
        right_T,
        2.0013322f,
        0.014210373f,
        -0.11553718f,
        -0.0001120644f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_x(right_T, 0.0f, -0.123f, 0.0f, arm_q[9]);
    ffw_sg2_mobility_com_accumulate_link(
        right_T,
        2.128261f,
        0.0091098413f,
        -0.00010530165f,
        -0.14386407f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_z(right_T, 0.0f, 0.0f, -0.165f, arm_q[10]);
    ffw_sg2_mobility_com_accumulate_link(
        right_T,
        1.6847551f,
        0.029845765f,
        -0.013922959f,
        -0.12104351f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_y(
        right_T,
        0.041004f,
        0.0f,
        -0.135f,
        arm_q[11]
    );
    ffw_sg2_mobility_com_accumulate_link(
        right_T,
        1.4942511f,
        -0.038677468f,
        -0.0093080099f,
        -0.13229713f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_z(
        right_T,
        -0.041004f,
        0.0f,
        -0.1489f,
        arm_q[12]
    );
    ffw_sg2_mobility_com_accumulate_link(
        right_T,
        1.3917831f,
        0.0008387686f,
        -0.016798499f,
        -0.098738156f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_y(right_T, 0.0f, 0.0f, -0.1041f, arm_q[13]);
    ffw_sg2_mobility_com_accumulate_link(
        right_T,
        0.65770741f,
        0.0032674912f,
        -0.01861259f,
        -0.064792616f,
        weighted,
        total_mass
    );
    ffw_sg2_apply_revolute_x(right_T, 0.0f, 0.0f, -0.0885f, arm_q[14]);
    ffw_sg2_mobility_com_accumulate_link(
        right_T,
        0.1668563f,
        0.025468362f,
        0.0013960012f,
        -0.055395331f,
        weighted,
        total_mass
    );

    float camera_right_T[16];
    ffw_sg2_copy4(right_T, camera_right_T);
    ffw_sg2_mobility_com_apply_fixed_rpy(
        camera_right_T,
        0.108236f,
        -0.021f,
        -0.062552f,
        -1.57079632679f,
        1.66678943569f,
        0.0f
    );
    ffw_sg2_apply_translation(camera_right_T, 0.01085f, 0.009f, 0.021f);
    ffw_sg2_mobility_com_accumulate_link(
        camera_right_T,
        0.072f,
        0.0f,
        0.0f,
        0.0f,
        weighted,
        total_mass
    );
    ffw_sg2_mobility_com_accumulate_gripper(
        right_T,
        weighted,
        total_mass
    );

    robot_mass = total_mass;
    const float inv_mass = 1.0f / total_mass;
    com[0] = weighted[0] * inv_mass;
    com[1] = weighted[1] * inv_mass;
    com[2] = weighted[2] * inv_mass;
}

__device__ __forceinline__ void ffw_sg2_mobility_com_object_proxy_base(
    const float q[FFW_SG2_MOBILITY_DIM],
    float object_com[3]
) {
    const float *arm_q = q + FFW_SG2_MOBILITY_BASE_DOF;

    float left_p[3], left_R[9];
    float right_p[3], right_R[9];
    ffw_sg2_fk_left(arm_q, left_p, left_R);
    ffw_sg2_fk_right(arm_q, right_p, right_R);

    float object_offset[3];
    ffw_sg2_mobility_com_attached_object_offset_base(
        left_p,
        right_p,
        left_R,
        right_R,
        object_offset
    );

    object_com[0] = 0.5f * (left_p[0] + right_p[0]) + object_offset[0];
    object_com[1] = 0.5f * (left_p[1] + right_p[1]) + object_offset[1];
    object_com[2] = 0.5f * (left_p[2] + right_p[2]) + object_offset[2];
}

__device__ __forceinline__ void ffw_sg2_mobility_com_total_com_base(
    const float q[FFW_SG2_MOBILITY_DIM],
    float object_mass_kg,
    float com[3]
) {
    float robot_com[3];
    float robot_mass = 0.0f;
    ffw_sg2_mobility_com_robot_com_base(q, robot_com, robot_mass);

    if (object_mass_kg <= 0.0f) {
        com[0] = robot_com[0];
        com[1] = robot_com[1];
        com[2] = robot_com[2];
        return;
    }

    float object_com[3];
    ffw_sg2_mobility_com_object_proxy_base(q, object_com);

    const float total_mass = robot_mass + object_mass_kg;
    const float inv_mass = 1.0f / total_mass;
    com[0] =
        (robot_mass * robot_com[0] + object_mass_kg * object_com[0]) *
        inv_mass;
    com[1] =
        (robot_mass * robot_com[1] + object_mass_kg * object_com[1]) *
        inv_mass;
    com[2] =
        (robot_mass * robot_com[2] + object_mass_kg * object_com[2]) *
        inv_mass;
}

__device__ __forceinline__ void ffw_sg2_mobility_com_zero_jacobian(
    float J[3 * FFW_SG2_MOBILITY_DIM]
) {
    for (int index = 0; index < 3 * FFW_SG2_MOBILITY_DIM; ++index) {
        J[index] = 0.0f;
    }
}

__device__ __forceinline__ void ffw_sg2_mobility_com_add_active_joint(
    float active_joint_T[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS][16],
    int active_joint_col[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS],
    int active_joint_axis[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS],
    int active_joint_prismatic[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS],
    int &active_count,
    const float T_joint[16],
    int col,
    int axis,
    int prismatic
) {
    if (active_count >= FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS) {
        return;
    }

    const int slot = active_count;
    ffw_sg2_copy4(T_joint, active_joint_T[slot]);
    active_joint_col[slot] = col;
    active_joint_axis[slot] = axis;
    active_joint_prismatic[slot] = prismatic;
    ++active_count;
}

__device__ __forceinline__ void ffw_sg2_mobility_com_accumulate_link_with_jacobian(
    const float T[16],
    float mass,
    float local_x,
    float local_y,
    float local_z,
    const float active_joint_T[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS][16],
    const int active_joint_col[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS],
    const int active_joint_axis[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS],
    const int active_joint_prismatic[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS],
    int active_count,
    float weighted[3],
    float weighted_jacobian[3 * FFW_SG2_MOBILITY_DIM],
    float &total_mass
) {
    float point[3];
    ffw_sg2_mobility_com_transform_point(
        T,
        local_x,
        local_y,
        local_z,
        point
    );

    weighted[0] += mass * point[0];
    weighted[1] += mass * point[1];
    weighted[2] += mass * point[2];
    total_mass += mass;

    for (int joint = 0; joint < active_count; ++joint) {
        const int col = active_joint_col[joint];
        const int axis = active_joint_axis[joint];
        const float *T_joint = active_joint_T[joint];

        float derivative[3];
        derivative[0] = T_joint[0 * 4 + axis];
        derivative[1] = T_joint[1 * 4 + axis];
        derivative[2] = T_joint[2 * 4 + axis];

        if (active_joint_prismatic[joint] == 0) {
            const float p_joint[3] = {
                T_joint[3],
                T_joint[7],
                T_joint[11]
            };
            const float r[3] = {
                point[0] - p_joint[0],
                point[1] - p_joint[1],
                point[2] - p_joint[2]
            };
            const float axis_world[3] = {
                derivative[0],
                derivative[1],
                derivative[2]
            };
            ffw_sg2_cross(axis_world, r, derivative);
        }

        weighted_jacobian[0 * FFW_SG2_MOBILITY_DIM + col] +=
            mass * derivative[0];
        weighted_jacobian[1 * FFW_SG2_MOBILITY_DIM + col] +=
            mass * derivative[1];
        weighted_jacobian[2 * FFW_SG2_MOBILITY_DIM + col] +=
            mass * derivative[2];
    }
}

__device__ __forceinline__ void ffw_sg2_mobility_com_accumulate_gripper_with_jacobian(
    const float link7_T[16],
    const float active_joint_T[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS][16],
    const int active_joint_col[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS],
    const int active_joint_axis[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS],
    const int active_joint_prismatic[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS],
    int active_count,
    float weighted[3],
    float weighted_jacobian[3 * FFW_SG2_MOBILITY_DIM],
    float &total_mass
) {
    float gripper_T[16];
    ffw_sg2_copy4(link7_T, gripper_T);
    ffw_sg2_apply_gripper_fixed(gripper_T);

    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        gripper_T,
        0.236f,
        0.0f,
        0.0f,
        0.032f,
        active_joint_T,
        active_joint_col,
        active_joint_axis,
        active_joint_prismatic,
        active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    float r1_T[16];
    ffw_sg2_copy4(gripper_T, r1_T);
    ffw_sg2_apply_translation(r1_T, 0.0f, 0.008f, 0.048f);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        r1_T,
        0.068f,
        0.0f,
        0.034f,
        0.004f,
        active_joint_T,
        active_joint_col,
        active_joint_axis,
        active_joint_prismatic,
        active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    float r2_T[16];
    ffw_sg2_copy4(r1_T, r2_T);
    ffw_sg2_apply_translation(r2_T, 0.0f, 0.0493634f, 0.0285f);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        r2_T,
        0.022f,
        0.0f,
        0.006f,
        0.011f,
        active_joint_T,
        active_joint_col,
        active_joint_axis,
        active_joint_prismatic,
        active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    float l1_T[16];
    ffw_sg2_copy4(gripper_T, l1_T);
    ffw_sg2_apply_translation(l1_T, 0.0f, -0.008f, 0.048f);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        l1_T,
        0.068f,
        0.0f,
        -0.034f,
        0.004f,
        active_joint_T,
        active_joint_col,
        active_joint_axis,
        active_joint_prismatic,
        active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    float l2_T[16];
    ffw_sg2_copy4(l1_T, l2_T);
    ffw_sg2_apply_translation(l2_T, 0.0f, -0.0493634f, 0.0285f);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        l2_T,
        0.022f,
        0.0f,
        -0.006f,
        0.011f,
        active_joint_T,
        active_joint_col,
        active_joint_axis,
        active_joint_prismatic,
        active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );
}

__device__ __forceinline__ void ffw_sg2_mobility_com_robot_com_jacobian_base(
    const float q[FFW_SG2_MOBILITY_DIM],
    float J_robot_com[3 * FFW_SG2_MOBILITY_DIM],
    float &robot_mass
) {
    const float *arm_q = q + FFW_SG2_MOBILITY_BASE_DOF;

    ffw_sg2_mobility_com_zero_jacobian(J_robot_com);

    float weighted[3] = {0.0f, 0.0f, 0.0f};
    float weighted_jacobian[3 * FFW_SG2_MOBILITY_DIM];
    ffw_sg2_mobility_com_zero_jacobian(weighted_jacobian);
    float total_mass = 0.0f;

    float no_active_T[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS][16];
    int no_active_col[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    int no_active_axis[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    int no_active_prismatic[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    const int no_active_count = 0;

    float identity_T[16];
    ffw_sg2_identity4(identity_T);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        identity_T,
        35.989f,
        -0.045747964f,
        0.0065810588f,
        0.19431842f,
        no_active_T,
        no_active_col,
        no_active_axis,
        no_active_prismatic,
        no_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        identity_T,
        17.9f,
        -0.12869878f,
        -0.000016045808f,
        0.89456017f,
        no_active_T,
        no_active_col,
        no_active_axis,
        no_active_prismatic,
        no_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    const float left_steer_com[3] = {
        0.0098027693f,
        -0.054712409f,
        -0.13765338f
    };
    const float right_steer_com[3] = {
        0.0098027286f,
        0.054712409f,
        -0.13765334f
    };
    const float drive_left_com[3] = {
        0.0f,
        -0.0014996017f,
        0.0f
    };
    const float drive_right_com[3] = {
        0.0f,
        0.0014996017f,
        0.0f
    };

    ffw_sg2_mobility_com_accumulate_wheel(
        0.1371f,
        0.2554f,
        left_steer_com,
        drive_left_com,
        weighted,
        total_mass
    );
    ffw_sg2_mobility_com_accumulate_wheel(
        0.1371f,
        -0.2554f,
        right_steer_com,
        drive_right_com,
        weighted,
        total_mass
    );
    ffw_sg2_mobility_com_accumulate_wheel(
        -0.2899f,
        0.0f,
        right_steer_com,
        drive_right_com,
        weighted,
        total_mass
    );

    float lift_joint_T[16];
    ffw_sg2_identity4(lift_joint_T);

    float base_active_T[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS][16];
    int base_active_col[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    int base_active_axis[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    int base_active_prismatic[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    int base_active_count = 0;
    ffw_sg2_mobility_com_add_active_joint(
        base_active_T,
        base_active_col,
        base_active_axis,
        base_active_prismatic,
        base_active_count,
        lift_joint_T,
        FFW_SG2_MOBILITY_BASE_DOF,
        2,
        1
    );

    float arm_base_T[16];
    ffw_sg2_identity4(arm_base_T);
    ffw_sg2_apply_prismatic_z(
        arm_base_T,
        -0.0199f,
        0.0f,
        1.4316f,
        arm_q[0]
    );
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        arm_base_T,
        6.1939559f,
        -0.014922878f,
        0.0027315225f,
        -0.043425264f,
        base_active_T,
        base_active_col,
        base_active_axis,
        base_active_prismatic,
        base_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    float head1_T[16];
    ffw_sg2_copy4(arm_base_T, head1_T);
    ffw_sg2_apply_revolute_y(head1_T, 0.0494878f, 0.0f, 0.102123f, 0.0f);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        head1_T,
        0.12351802f,
        0.025339656f,
        0.0f,
        0.047745146f,
        base_active_T,
        base_active_col,
        base_active_axis,
        base_active_prismatic,
        base_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    float head2_T[16];
    ffw_sg2_copy4(head1_T, head2_T);
    ffw_sg2_apply_revolute_z(head2_T, 0.0365694f, 0.0f, 0.0563797f, 0.0f);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        head2_T,
        0.3254213f,
        0.0030525634f,
        0.0011804757f,
        -0.015932804f,
        base_active_T,
        base_active_col,
        base_active_axis,
        base_active_prismatic,
        base_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    float left_active_T[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS][16];
    int left_active_col[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    int left_active_axis[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    int left_active_prismatic[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    int left_active_count = 0;
    ffw_sg2_mobility_com_add_active_joint(
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        lift_joint_T,
        FFW_SG2_MOBILITY_BASE_DOF,
        2,
        1
    );

    float left_T[16];
    float joint_T[16];
    ffw_sg2_copy4(arm_base_T, left_T);
    ffw_sg2_copy4(left_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.0f, 0.1045f, 0.0f);
    ffw_sg2_mobility_com_add_active_joint(
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 1,
        1,
        0
    );
    ffw_sg2_apply_revolute_y(left_T, 0.0f, 0.1045f, 0.0f, arm_q[1]);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        left_T,
        2.0013322f,
        0.014210373f,
        0.11553718f,
        0.0001120644f,
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(left_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.0f, 0.123f, 0.0f);
    ffw_sg2_mobility_com_add_active_joint(
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 2,
        0,
        0
    );
    ffw_sg2_apply_revolute_x(left_T, 0.0f, 0.123f, 0.0f, arm_q[2]);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        left_T,
        2.128261f,
        0.0091098413f,
        -0.00010530165f,
        -0.14386407f,
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(left_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.0f, 0.0f, -0.165f);
    ffw_sg2_mobility_com_add_active_joint(
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 3,
        2,
        0
    );
    ffw_sg2_apply_revolute_z(left_T, 0.0f, 0.0f, -0.165f, arm_q[3]);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        left_T,
        1.6847551f,
        0.029875373f,
        0.013913949f,
        -0.1210139f,
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(left_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.041004f, 0.0f, -0.135f);
    ffw_sg2_mobility_com_add_active_joint(
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 4,
        1,
        0
    );
    ffw_sg2_apply_revolute_y(
        left_T,
        0.041004f,
        0.0f,
        -0.135f,
        arm_q[4]
    );
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        left_T,
        1.5082144f,
        -0.038538653f,
        0.0098072286f,
        -0.13131228f,
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(left_T, joint_T);
    ffw_sg2_apply_translation(joint_T, -0.041004f, 0.0f, -0.1489f);
    ffw_sg2_mobility_com_add_active_joint(
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 5,
        2,
        0
    );
    ffw_sg2_apply_revolute_z(
        left_T,
        -0.041004f,
        0.0f,
        -0.1489f,
        arm_q[5]
    );
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        left_T,
        1.3917831f,
        -0.000031226161f,
        0.016819227f,
        -0.098738156f,
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(left_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.0f, 0.0f, -0.1041f);
    ffw_sg2_mobility_com_add_active_joint(
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 6,
        1,
        0
    );
    ffw_sg2_apply_revolute_y(left_T, 0.0f, 0.0f, -0.1041f, arm_q[6]);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        left_T,
        0.65770741f,
        0.0022438917f,
        0.018764105f,
        -0.064805576f,
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(left_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.0f, 0.0f, -0.0885f);
    ffw_sg2_mobility_com_add_active_joint(
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 7,
        0,
        0
    );
    ffw_sg2_apply_revolute_x(left_T, 0.0f, 0.0f, -0.0885f, arm_q[7]);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        left_T,
        0.1668563f,
        0.025506593f,
        0.0f,
        -0.055395331f,
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    float camera_left_T[16];
    ffw_sg2_copy4(left_T, camera_left_T);
    ffw_sg2_mobility_com_apply_fixed_rpy(
        camera_left_T,
        0.108236f,
        -0.021f,
        -0.062552f,
        -1.57079632679f,
        1.66678943569f,
        0.0f
    );
    ffw_sg2_apply_translation(camera_left_T, 0.01085f, 0.009f, 0.021f);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        camera_left_T,
        0.072f,
        0.0f,
        0.0f,
        0.0f,
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );
    ffw_sg2_mobility_com_accumulate_gripper_with_jacobian(
        left_T,
        left_active_T,
        left_active_col,
        left_active_axis,
        left_active_prismatic,
        left_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    float right_active_T[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS][16];
    int right_active_col[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    int right_active_axis[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    int right_active_prismatic[FFW_SG2_MOBILITY_COM_MAX_ACTIVE_JOINTS];
    int right_active_count = 0;
    ffw_sg2_mobility_com_add_active_joint(
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        lift_joint_T,
        FFW_SG2_MOBILITY_BASE_DOF,
        2,
        1
    );

    float right_T[16];
    ffw_sg2_copy4(arm_base_T, right_T);
    ffw_sg2_copy4(right_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.0f, -0.1045f, 0.0f);
    ffw_sg2_mobility_com_add_active_joint(
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 8,
        1,
        0
    );
    ffw_sg2_apply_revolute_y(right_T, 0.0f, -0.1045f, 0.0f, arm_q[8]);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        right_T,
        2.0013322f,
        0.014210373f,
        -0.11553718f,
        -0.0001120644f,
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(right_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.0f, -0.123f, 0.0f);
    ffw_sg2_mobility_com_add_active_joint(
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 9,
        0,
        0
    );
    ffw_sg2_apply_revolute_x(right_T, 0.0f, -0.123f, 0.0f, arm_q[9]);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        right_T,
        2.128261f,
        0.0091098413f,
        -0.00010530165f,
        -0.14386407f,
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(right_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.0f, 0.0f, -0.165f);
    ffw_sg2_mobility_com_add_active_joint(
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 10,
        2,
        0
    );
    ffw_sg2_apply_revolute_z(right_T, 0.0f, 0.0f, -0.165f, arm_q[10]);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        right_T,
        1.6847551f,
        0.029845765f,
        -0.013922959f,
        -0.12104351f,
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(right_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.041004f, 0.0f, -0.135f);
    ffw_sg2_mobility_com_add_active_joint(
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 11,
        1,
        0
    );
    ffw_sg2_apply_revolute_y(
        right_T,
        0.041004f,
        0.0f,
        -0.135f,
        arm_q[11]
    );
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        right_T,
        1.4942511f,
        -0.038677468f,
        -0.0093080099f,
        -0.13229713f,
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(right_T, joint_T);
    ffw_sg2_apply_translation(joint_T, -0.041004f, 0.0f, -0.1489f);
    ffw_sg2_mobility_com_add_active_joint(
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 12,
        2,
        0
    );
    ffw_sg2_apply_revolute_z(
        right_T,
        -0.041004f,
        0.0f,
        -0.1489f,
        arm_q[12]
    );
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        right_T,
        1.3917831f,
        0.0008387686f,
        -0.016798499f,
        -0.098738156f,
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(right_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.0f, 0.0f, -0.1041f);
    ffw_sg2_mobility_com_add_active_joint(
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 13,
        1,
        0
    );
    ffw_sg2_apply_revolute_y(right_T, 0.0f, 0.0f, -0.1041f, arm_q[13]);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        right_T,
        0.65770741f,
        0.0032674912f,
        -0.01861259f,
        -0.064792616f,
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    ffw_sg2_copy4(right_T, joint_T);
    ffw_sg2_apply_translation(joint_T, 0.0f, 0.0f, -0.0885f);
    ffw_sg2_mobility_com_add_active_joint(
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        joint_T,
        FFW_SG2_MOBILITY_BASE_DOF + 14,
        0,
        0
    );
    ffw_sg2_apply_revolute_x(right_T, 0.0f, 0.0f, -0.0885f, arm_q[14]);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        right_T,
        0.1668563f,
        0.025468362f,
        0.0013960012f,
        -0.055395331f,
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    float camera_right_T[16];
    ffw_sg2_copy4(right_T, camera_right_T);
    ffw_sg2_mobility_com_apply_fixed_rpy(
        camera_right_T,
        0.108236f,
        -0.021f,
        -0.062552f,
        -1.57079632679f,
        1.66678943569f,
        0.0f
    );
    ffw_sg2_apply_translation(camera_right_T, 0.01085f, 0.009f, 0.021f);
    ffw_sg2_mobility_com_accumulate_link_with_jacobian(
        camera_right_T,
        0.072f,
        0.0f,
        0.0f,
        0.0f,
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );
    ffw_sg2_mobility_com_accumulate_gripper_with_jacobian(
        right_T,
        right_active_T,
        right_active_col,
        right_active_axis,
        right_active_prismatic,
        right_active_count,
        weighted,
        weighted_jacobian,
        total_mass
    );

    robot_mass = total_mass;
    if (total_mass <= 0.0f) {
        return;
    }

    const float inv_mass = 1.0f / total_mass;
    for (int index = 0; index < 3 * FFW_SG2_MOBILITY_DIM; ++index) {
        J_robot_com[index] = weighted_jacobian[index] * inv_mass;
    }
}

__device__ __forceinline__ void ffw_sg2_mobility_com_object_proxy_jacobian_base(
    const float q[FFW_SG2_MOBILITY_DIM],
    float J_object_com[3 * FFW_SG2_MOBILITY_DIM]
) {
    ffw_sg2_mobility_com_zero_jacobian(J_object_com);

    constexpr float eps = 1.0e-4f;
    for (int col = FFW_SG2_MOBILITY_BASE_DOF; col < FFW_SG2_MOBILITY_DIM; ++col) {
        float q_plus[FFW_SG2_MOBILITY_DIM];
        float q_minus[FFW_SG2_MOBILITY_DIM];
        for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
            q_plus[joint] = q[joint];
            q_minus[joint] = q[joint];
        }
        q_plus[col] += eps;
        q_minus[col] -= eps;

        float object_plus[3], object_minus[3];
        ffw_sg2_mobility_com_object_proxy_base(q_plus, object_plus);
        ffw_sg2_mobility_com_object_proxy_base(q_minus, object_minus);

        const float inv_step = 0.5f / eps;
        J_object_com[0 * FFW_SG2_MOBILITY_DIM + col] =
            (object_plus[0] - object_minus[0]) * inv_step;
        J_object_com[1 * FFW_SG2_MOBILITY_DIM + col] =
            (object_plus[1] - object_minus[1]) * inv_step;
        J_object_com[2 * FFW_SG2_MOBILITY_DIM + col] =
            (object_plus[2] - object_minus[2]) * inv_step;
    }
}

__device__ __forceinline__ void ffw_sg2_mobility_com_jacobian_base(
    const float q[FFW_SG2_MOBILITY_DIM],
    float object_mass_kg,
    float J_com[3 * FFW_SG2_MOBILITY_DIM]
) {
    float J_robot_com[3 * FFW_SG2_MOBILITY_DIM];
    float robot_mass = 0.0f;
    ffw_sg2_mobility_com_robot_com_jacobian_base(
        q,
        J_robot_com,
        robot_mass
    );

    if (object_mass_kg <= 0.0f || robot_mass <= 0.0f) {
        for (int index = 0; index < 3 * FFW_SG2_MOBILITY_DIM; ++index) {
            J_com[index] = J_robot_com[index];
        }
        return;
    }

    float J_object_com[3 * FFW_SG2_MOBILITY_DIM];
    ffw_sg2_mobility_com_object_proxy_jacobian_base(q, J_object_com);

    const float total_mass = robot_mass + object_mass_kg;
    const float inv_mass = 1.0f / total_mass;
    for (int index = 0; index < 3 * FFW_SG2_MOBILITY_DIM; ++index) {
        J_com[index] =
            (robot_mass * J_robot_com[index] +
             object_mass_kg * J_object_com[index]) *
            inv_mass;
    }
}

__device__ __forceinline__ void ffw_sg2_mobility_com_support_vertex(
    int index,
    float &x,
    float &y
) {
    if (index == 0) {
        x = 0.1371f;
        y = 0.2554f;
    } else if (index == 1) {
        x = -0.2899f;
        y = 0.0f;
    } else {
        x = 0.1371f;
        y = -0.2554f;
    }
}

__device__ __forceinline__ void ffw_sg2_mobility_com_support_edge(
    int edge,
    float &point_x,
    float &point_y,
    float &normal_x,
    float &normal_y
) {
    float next_x, next_y;
    ffw_sg2_mobility_com_support_vertex(edge, point_x, point_y);
    ffw_sg2_mobility_com_support_vertex((edge + 1) % 3, next_x, next_y);

    const float edge_x = next_x - point_x;
    const float edge_y = next_y - point_y;
    const float inv_len = rsqrtf(edge_x * edge_x + edge_y * edge_y);

    normal_x = -edge_y * inv_len;
    normal_y = edge_x * inv_len;
}

__device__ __forceinline__ float ffw_sg2_mobility_com_signed_distance(
    const float com[3],
    int edge
) {
    float point_x, point_y, normal_x, normal_y;
    ffw_sg2_mobility_com_support_edge(
        edge,
        point_x,
        point_y,
        normal_x,
        normal_y
    );
    return
        normal_x * (com[0] - point_x) +
        normal_y * (com[1] - point_y);
}

__device__ __forceinline__ void ffw_sg2_mobility_com_inequality_residual(
    const float q[FFW_SG2_MOBILITY_DIM],
    float support_margin_m,
    float object_mass_kg,
    float h[FFW_SG2_MOBILITY_COM_INEQUALITY_DIM]
) {
    float com[3];
    ffw_sg2_mobility_com_total_com_base(q, object_mass_kg, com);

    for (int edge = 0; edge < FFW_SG2_MOBILITY_COM_INEQUALITY_DIM; ++edge) {
        const float signed_distance =
            ffw_sg2_mobility_com_signed_distance(com, edge);
        h[edge] = fmaxf(0.0f, support_margin_m - signed_distance);
    }
}

__device__ __forceinline__ void ffw_sg2_mobility_com_constraint_residual(
    const float q[FFW_SG2_MOBILITY_DIM],
    float support_margin_m,
    float object_mass_kg,
    float h[FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM]
) {
    float h_equality[FFW_SG2_MOBILITY_RESIDUAL_DIM];
    ffw_sg2_mobility_constraint_residual(q, h_equality);
    for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
        h[row] = h_equality[row];
    }

    float h_com[FFW_SG2_MOBILITY_COM_INEQUALITY_DIM];
    ffw_sg2_mobility_com_inequality_residual(
        q,
        support_margin_m,
        object_mass_kg,
        h_com
    );
    for (int row = 0; row < FFW_SG2_MOBILITY_COM_INEQUALITY_DIM; ++row) {
        h[FFW_SG2_MOBILITY_RESIDUAL_DIM + row] = h_com[row];
    }
}

__device__ __forceinline__ float ffw_sg2_mobility_com_residual_norm(
    const float h[FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM]
) {
    float norm2 = 0.0f;
    for (int row = 0; row < FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM; ++row) {
        norm2 += h[row] * h[row];
    }
    return sqrtf(norm2);
}

__device__ __forceinline__ bool ffw_sg2_mobility_com_active_jacobian(
    const float q[FFW_SG2_MOBILITY_DIM],
    float support_margin_m,
    float object_mass_kg,
    float h_active[FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM],
    float J_active[
        FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM * FFW_SG2_MOBILITY_DIM
    ],
    int &active_dim
) {
    float h_equality[FFW_SG2_MOBILITY_RESIDUAL_DIM];
    float J_equality[FFW_SG2_MOBILITY_RESIDUAL_DIM * FFW_SG2_MOBILITY_DIM];
    if (!ffw_sg2_mobility_constraint_jacobian(q, h_equality, J_equality)) {
        return false;
    }

    for (
        int index = 0;
        index < FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM * FFW_SG2_MOBILITY_DIM;
        ++index
    ) {
        J_active[index] = 0.0f;
    }
    active_dim = 0;
    for (int row = 0; row < FFW_SG2_MOBILITY_RESIDUAL_DIM; ++row) {
        h_active[active_dim] = h_equality[row];
        for (int col = 0; col < FFW_SG2_MOBILITY_DIM; ++col) {
            J_active[active_dim * FFW_SG2_MOBILITY_DIM + col] =
                J_equality[row * FFW_SG2_MOBILITY_DIM + col];
        }
        ++active_dim;
    }

    float com[3];
    float J_com[3 * FFW_SG2_MOBILITY_DIM];
    ffw_sg2_mobility_com_total_com_base(q, object_mass_kg, com);
    ffw_sg2_mobility_com_jacobian_base(q, object_mass_kg, J_com);

    for (int edge = 0; edge < FFW_SG2_MOBILITY_COM_INEQUALITY_DIM; ++edge) {
        float point_x, point_y, normal_x, normal_y;
        ffw_sg2_mobility_com_support_edge(
            edge,
            point_x,
            point_y,
            normal_x,
            normal_y
        );
        const float signed_distance =
            normal_x * (com[0] - point_x) +
            normal_y * (com[1] - point_y);
        const float violation =
            fmaxf(0.0f, support_margin_m - signed_distance);
        if (violation <= 0.0f) {
            continue;
        }

        h_active[active_dim] = violation;
        for (int col = 0; col < FFW_SG2_MOBILITY_DIM; ++col) {
            J_active[active_dim * FFW_SG2_MOBILITY_DIM + col] =
                -normal_x * J_com[0 * FFW_SG2_MOBILITY_DIM + col] -
                normal_y * J_com[1 * FFW_SG2_MOBILITY_DIM + col];
        }
        ++active_dim;
    }

    return true;
}

__device__ __forceinline__ bool ffw_sg2_mobility_com_solve_residual_system(
    float A_in[
        FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM *
        FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM
    ],
    const float b_in[FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM],
    float x[FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM],
    int active_dim
) {
    float aug[FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM][
        FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM + 1
    ];

    for (int row = 0; row < active_dim; ++row) {
        for (int col = 0; col < active_dim; ++col) {
            aug[row][col] =
                A_in[row * FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM + col];
        }
        aug[row][active_dim] = b_in[row];
    }

    for (int col = 0; col < active_dim; ++col) {
        int pivot = col;
        float best = fabsf(aug[col][col]);
        for (int row = col + 1; row < active_dim; ++row) {
            const float value = fabsf(aug[row][col]);
            if (value > best) {
                best = value;
                pivot = row;
            }
        }
        if (best < 1.0e-9f) {
            return false;
        }
        if (pivot != col) {
            for (int k = col; k <= active_dim; ++k) {
                const float tmp = aug[col][k];
                aug[col][k] = aug[pivot][k];
                aug[pivot][k] = tmp;
            }
        }

        const float inv_pivot = 1.0f / aug[col][col];
        for (int k = col; k <= active_dim; ++k) {
            aug[col][k] *= inv_pivot;
        }

        for (int row = 0; row < active_dim; ++row) {
            if (row == col) {
                continue;
            }
            const float factor = aug[row][col];
            for (int k = col; k <= active_dim; ++k) {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }

    for (int row = 0; row < active_dim; ++row) {
        x[row] = aug[row][active_dim];
    }
    return true;
}

__device__ __forceinline__ bool ffw_sg2_mobility_com_task_correction(
    const float q[FFW_SG2_MOBILITY_DIM],
    float support_margin_m,
    float object_mass_kg,
    float damping,
    float max_step,
    float correction[FFW_SG2_MOBILITY_DIM],
    float &task_error_norm
) {
    float h_full[FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM];
    ffw_sg2_mobility_com_constraint_residual(
        q,
        support_margin_m,
        object_mass_kg,
        h_full
    );
    task_error_norm = ffw_sg2_mobility_com_residual_norm(h_full);

    for (int col = 0; col < FFW_SG2_MOBILITY_DIM; ++col) {
        correction[col] = 0.0f;
    }
    if (task_error_norm <= 1.0e-12f) {
        return true;
    }

    float h[FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM];
    float J[
        FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM * FFW_SG2_MOBILITY_DIM
    ];
    int active_dim = 0;
    if (
        !ffw_sg2_mobility_com_active_jacobian(
            q,
            support_margin_m,
            object_mass_kg,
            h,
            J,
            active_dim
        )
    ) {
        return false;
    }

    float A[
        FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM *
        FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM
    ];
    for (int row = 0; row < active_dim; ++row) {
        for (int col = 0; col < active_dim; ++col) {
            float value = 0.0f;
            for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                value +=
                    J[row * FFW_SG2_MOBILITY_DIM + joint] *
                    J[col * FFW_SG2_MOBILITY_DIM + joint];
            }
            A[row * FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM + col] =
                value + (row == col ? damping : 0.0f);
        }
    }

    float y[FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM];
    if (!ffw_sg2_mobility_com_solve_residual_system(A, h, y, active_dim)) {
        return false;
    }

    float step_norm2 = 0.0f;
    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
        float value = 0.0f;
        for (int row = 0; row < active_dim; ++row) {
            value += J[row * FFW_SG2_MOBILITY_DIM + joint] * y[row];
        }
        correction[joint] = value;
        step_norm2 += value * value;
    }

    const float step_norm = sqrtf(step_norm2);
    if (max_step > 0.0f && step_norm > max_step) {
        const float scale = max_step / step_norm;
        for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
            correction[joint] *= scale;
        }
    }
    return true;
}

__device__ __forceinline__ bool ffw_sg2_mobility_com_project_config(
    float q[FFW_SG2_MOBILITY_DIM],
    float support_margin_m,
    float object_mass_kg
) {
    constexpr int max_iters = 15;
    constexpr float tol = 1.0e-3f;
    constexpr float damping = 1.0e-4f;
    constexpr float max_step = 0.2f;

    ffw_sg2_mobility_clamp(q);
    for (int iter = 0; iter < max_iters; ++iter) {
        float h[FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM];
        ffw_sg2_mobility_com_constraint_residual(
            q,
            support_margin_m,
            object_mass_kg,
            h
        );
        if (ffw_sg2_mobility_com_residual_norm(h) < tol) {
            return true;
        }

        float correction[FFW_SG2_MOBILITY_DIM];
        float task_error_norm = 1.0e30f;
        if (
            !ffw_sg2_mobility_com_task_correction(
                q,
                support_margin_m,
                object_mass_kg,
                damping,
                max_step,
                correction,
                task_error_norm
            )
        ) {
            return false;
        }
        for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
            q[joint] -= correction[joint];
        }
        ffw_sg2_mobility_clamp(q);
    }

    float h[FFW_SG2_MOBILITY_COM_CONSTRAINT_DIM];
    ffw_sg2_mobility_com_constraint_residual(
        q,
        support_margin_m,
        object_mass_kg,
        h
    );
    return ffw_sg2_mobility_com_residual_norm(h) < tol;
}

__device__ __forceinline__ bool ffw_sg2_mobility_com_project_motion(
    volatile float *motion_segment,
    volatile float *motion_segment_next,
    int granularity,
    float support_margin_m,
    float object_mass_kg,
    volatile unsigned char *projection_valid,
    volatile int *projection_prog,
    volatile unsigned int *projection_success,
    int max_iters,
    float alpha,
    float damping,
    float task_tolerance,
    float smoothness_threshold,
    float smoothness_weight,
    bool use_smoothness,
    float max_step,
    int tid,
    bool return_when_success = true
) {
    const int waypoint = tid / 4 + 1;
    const int lane = tid % 4;

    if (tid == 0) {
        projection_prog[0] = 0;
        projection_success[0] = 0;
        projection_valid[0] = 1;
    }
    if (tid < FFW_SG2_MOBILITY_DIM) {
        motion_segment_next[tid] = motion_segment[tid];
    }
    if (waypoint <= granularity && lane == 0) {
        projection_valid[waypoint] = 0;
    }
    __syncthreads();

    for (int iter = 0; iter < max_iters; ++iter) {
        if (projection_success[0] == 0 && waypoint <= granularity) {
            const int current_prog = projection_prog[0];
            if (waypoint > current_prog) {
                if (lane == 0) {
                    float q[FFW_SG2_MOBILITY_DIM];
                    float correction[FFW_SG2_MOBILITY_DIM];
                    float task_error_norm = 1.0e30f;
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        q[joint] =
                            motion_segment[
                                waypoint * FFW_SG2_MOBILITY_DIM + joint
                            ];
                    }

                    const bool correction_ok =
                        ffw_sg2_mobility_com_task_correction(
                            q,
                            support_margin_m,
                            object_mass_kg,
                            damping,
                            max_step,
                            correction,
                            task_error_norm
                        );

                    float diff[FFW_SG2_MOBILITY_DIM];
                    float smooth_dist2 = 0.0f;
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        diff[joint] =
                            q[joint] -
                            motion_segment[
                                (waypoint - 1) *
                                FFW_SG2_MOBILITY_DIM +
                                joint
                            ];
                        smooth_dist2 += diff[joint] * diff[joint];
                    }
                    const float smooth_dist = sqrtf(smooth_dist2);
                    const float smooth_error = use_smoothness
                        ? fmaxf(0.0f, smooth_dist - smoothness_threshold)
                        : 0.0f;
                    const float inv_smooth_dist = smooth_dist > 1.0e-8f
                        ? 1.0f / smooth_dist
                        : 0.0f;

                    float combined[FFW_SG2_MOBILITY_DIM];
                    float combined_norm2 = 0.0f;
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        const float grad_smooth =
                            diff[joint] * inv_smooth_dist * smooth_error;
                        combined[joint] =
                            alpha *
                            (
                                correction[joint] +
                                smoothness_weight * grad_smooth
                            );
                        combined_norm2 += combined[joint] * combined[joint];
                    }

                    const float combined_norm = sqrtf(combined_norm2);
                    const float combined_scale =
                        max_step > 0.0f && combined_norm > max_step
                            ? max_step / combined_norm
                            : 1.0f;
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        motion_segment_next[
                            waypoint * FFW_SG2_MOBILITY_DIM + joint
                        ] = q[joint] - combined_scale * combined[joint];
                    }

                    float q_next[FFW_SG2_MOBILITY_DIM];
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        q_next[joint] =
                            motion_segment_next[
                                waypoint * FFW_SG2_MOBILITY_DIM + joint
                            ];
                    }
                    ffw_sg2_mobility_clamp(q_next);
                    for (int joint = 0; joint < FFW_SG2_MOBILITY_DIM; ++joint) {
                        motion_segment_next[
                            waypoint * FFW_SG2_MOBILITY_DIM + joint
                        ] = q_next[joint];
                    }

                    projection_valid[waypoint] =
                        correction_ok &&
                        task_error_norm < task_tolerance &&
                        (
                            !use_smoothness ||
                            smooth_dist <= smoothness_threshold
                        );
                }
            } else {
                if (lane == 0) {
                    projection_valid[waypoint] = 1;
                }
                for (
                    int joint = lane;
                    joint < FFW_SG2_MOBILITY_DIM;
                    joint += 4
                ) {
                    motion_segment_next[
                        waypoint * FFW_SG2_MOBILITY_DIM + joint
                    ] = motion_segment[
                        waypoint * FFW_SG2_MOBILITY_DIM + joint
                    ];
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            int prog = projection_prog[0];
            while (
                prog + 1 <= granularity &&
                projection_valid[prog + 1] != 0
            ) {
                ++prog;
            }
            projection_prog[0] = prog;
            if (prog == granularity) {
                projection_success[0] = 1;
            }
        }
        __syncthreads();

        if (projection_success[0] != 0 && return_when_success) {
            return true;
        }

        if (waypoint <= granularity && waypoint > projection_prog[0]) {
            for (
                int joint = lane;
                joint < FFW_SG2_MOBILITY_DIM;
                joint += 4
            ) {
                motion_segment[
                    waypoint * FFW_SG2_MOBILITY_DIM + joint
                ] = motion_segment_next[
                    waypoint * FFW_SG2_MOBILITY_DIM + joint
                ];
            }
        }
        __syncthreads();
    }

    return projection_success[0] != 0;
}

} // namespace ppln::collision

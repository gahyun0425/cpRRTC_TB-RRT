#pragma once

#include "src/planning/Robots.hh"

namespace ppln::collision {

#define FFW_SG2_DIM 15
#define FFW_SG2_RESIDUAL_DIM 6
#define FFW_SG2_MAX_RESIDUAL_DIM 8

// reference constraint + 수학 helper

    __device__ __constant__ float ffw_sg2_p_rel_ref[3] = {
        -3.99994876e-01f,
        1.59923112e-07f,
        -5.10448455e-06f
    };

    __device__ __constant__ float ffw_sg2_q_rel_ref[4] = {
        -1.40688437e-06f,
        -1.04612950e-05f,
        -2.64479476e-06f,
        1.00000000e+00f
    };

    __device__ __forceinline__ void ffw_sg2_identity4(float T[16]) {
        for (int i = 0; i < 16; i++) {
            T[i] = 0.0f;
        }
        T[0] = 1.0f;
        T[5] = 1.0f;
        T[10] = 1.0f;
        T[15] = 1.0f;
    }

    __device__ __forceinline__ void ffw_sg2_mul4(const float A[16], const float B[16], float C[16]) {
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float v = 0.0f;
                for (int k = 0; k < 4; k++) {
                    v += A[r * 4 + k] * B[k * 4 + c];
                }
                C[r * 4 + c] = v;
            }
        }
    }

    __device__ __forceinline__ void ffw_sg2_apply_transform(float T[16], const float step_tf[16]) {
        float C[16];
        ffw_sg2_mul4(T, step_tf, C);
        for (int i = 0; i < 16; i++) {
            T[i] = C[i];
        }
    }

    __device__ __forceinline__ void ffw_sg2_copy4(const float src[16], float dst[16]) {
        for (int i = 0; i < 16; i++) {
            dst[i] = src[i];
        }
    }

    __device__ __forceinline__ void ffw_sg2_apply_translation(float T[16], float x, float y, float z) {
        float step_tf[16];
        ffw_sg2_identity4(step_tf);
        step_tf[3] = x;
        step_tf[7] = y;
        step_tf[11] = z;
        ffw_sg2_apply_transform(T, step_tf);
    }

    __device__ __forceinline__ void ffw_sg2_apply_prismatic_z(float T[16], float x, float y, float z, float q) {
        float step_tf[16];
        ffw_sg2_identity4(step_tf);
        step_tf[3] = x;
        step_tf[7] = y;
        step_tf[11] = z + q;
        ffw_sg2_apply_transform(T, step_tf);
    }

    __device__ __forceinline__ void ffw_sg2_apply_revolute_x(float T[16], float x, float y, float z, float q) {
        float step_tf[16];
        ffw_sg2_identity4(step_tf);
        const float c = __cosf(q);
        const float s = __sinf(q);
        step_tf[5] = c;
        step_tf[6] = -s;
        step_tf[9] = s;
        step_tf[10] = c;
        step_tf[3] = x;
        step_tf[7] = y;
        step_tf[11] = z;
        ffw_sg2_apply_transform(T, step_tf);
    }

    __device__ __forceinline__ void ffw_sg2_apply_revolute_y(float T[16], float x, float y, float z, float q) {
        float step_tf[16];
        ffw_sg2_identity4(step_tf);
        const float c = __cosf(q);
        const float s = __sinf(q);
        step_tf[0] = c;
        step_tf[2] = s;
        step_tf[8] = -s;
        step_tf[10] = c;
        step_tf[3] = x;
        step_tf[7] = y;
        step_tf[11] = z;
        ffw_sg2_apply_transform(T, step_tf);
    }

    __device__ __forceinline__ void ffw_sg2_apply_revolute_z(float T[16], float x, float y, float z, float q) {
        float step_tf[16];
        ffw_sg2_identity4(step_tf);
        const float c = __cosf(q);
        const float s = __sinf(q);
        step_tf[0] = c;
        step_tf[1] = -s;
        step_tf[4] = s;
        step_tf[5] = c;
        step_tf[3] = x;
        step_tf[7] = y;
        step_tf[11] = z;
        ffw_sg2_apply_transform(T, step_tf);
    }

    __device__ __forceinline__ void ffw_sg2_apply_gripper_fixed(float T[16]) {
        float step_tf[16];
        ffw_sg2_identity4(step_tf);
        step_tf[5] = -1.0f;
        step_tf[10] = -1.0f;
        step_tf[11] = -0.078f;
        ffw_sg2_apply_transform(T, step_tf);
    }

    __device__ __forceinline__ void ffw_sg2_extract_pose(const float T[16], float p[3], float R[9]) {
        p[0] = T[3];
        p[1] = T[7];
        p[2] = T[11];
        R[0] = T[0];
        R[1] = T[1];
        R[2] = T[2];
        R[3] = T[4];
        R[4] = T[5];
        R[5] = T[6];
        R[6] = T[8];
        R[7] = T[9];
        R[8] = T[10];
    }

    __device__ __forceinline__ bool ffw_sg2_solve6(float A_in[36], const float b_in[6], float x[6]);
    __device__ __forceinline__ void ffw_sg2_se3_log(const float R[9], const float p[3], float xi[6]);
    __device__ __forceinline__ bool ffw_sg2_relative_pose_jacobian(const float *q, float h[6], float J[6 * 15]);
    __device__ __forceinline__ bool ffw_sg2_tangent_basis_from_jacobian(const float J[FFW_SG2_MAX_RESIDUAL_DIM * 15], bool rigid_orientation, float basis[15 * 9]);

    __device__ __forceinline__ void ffw_sg2_cross(const float a[3], const float b[3], float out[3]) {
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
    }

    __device__ __forceinline__ void ffw_sg2_skew(const float v[3], float S[9]) {
        S[0] = 0.0f;
        S[1] = -v[2];
        S[2] = v[1];
        S[3] = v[2];
        S[4] = 0.0f;
        S[5] = -v[0];
        S[6] = -v[1];
        S[7] = v[0];
        S[8] = 0.0f;
    }

    __device__ __forceinline__ void ffw_sg2_mat3_vec(const float R[9], const float v[3], float out[3]) {
        out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
        out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
        out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
    }

    __device__ __forceinline__ void ffw_sg2_mat3_transpose_vec(const float R[9], const float v[3], float out[3]) {
        out[0] = R[0] * v[0] + R[3] * v[1] + R[6] * v[2];
        out[1] = R[1] * v[0] + R[4] * v[1] + R[7] * v[2];
        out[2] = R[2] * v[0] + R[5] * v[1] + R[8] * v[2];
    }

    __device__ __forceinline__ void ffw_sg2_mat3_mul(const float A[9], const float B[9], float C[9]) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                C[r * 3 + c] =
                    A[r * 3 + 0] * B[0 * 3 + c] +
                    A[r * 3 + 1] * B[1 * 3 + c] +
                    A[r * 3 + 2] * B[2 * 3 + c];
            }
        }
    }

    __device__ __forceinline__ void ffw_sg2_mat3_mul_at_b(const float A[9], const float B[9], float C[9]) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                C[r * 3 + c] =
                    A[0 * 3 + r] * B[0 * 3 + c] +
                    A[1 * 3 + r] * B[1 * 3 + c] +
                    A[2 * 3 + r] * B[2 * 3 + c];
            }
        }
    }

    __device__ __forceinline__ void ffw_sg2_quat_to_matrix(const float q[4], float R[9]) {
        const float w = q[0];
        const float x = q[1];
        const float y = q[2];
        const float z = q[3];

        R[0] = 1.0f - 2.0f * (y * y + z * z);
        R[1] = 2.0f * (x * y - z * w);
        R[2] = 2.0f * (x * z + y * w);
        R[3] = 2.0f * (x * y + z * w);
        R[4] = 1.0f - 2.0f * (x * x + z * z);
        R[5] = 2.0f * (y * z - x * w);
        R[6] = 2.0f * (x * z - y * w);
        R[7] = 2.0f * (y * z + x * w);
        R[8] = 1.0f - 2.0f * (x * x + y * y);
    }

// Analytic FK
    __device__ __forceinline__ void ffw_sg2_fk_left(const float *q, float p[3], float R[9]) {
        float T[16];
        ffw_sg2_identity4(T);
        ffw_sg2_apply_prismatic_z(T, -0.0199f, 0.0f, 1.4316f, q[0]);
        ffw_sg2_apply_revolute_y(T, 0.0f, 0.1045f, 0.0f, q[1]);
        ffw_sg2_apply_revolute_x(T, 0.0f, 0.123f, 0.0f, q[2]);
        ffw_sg2_apply_revolute_z(T, 0.0f, 0.0f, -0.165f, q[3]);
        ffw_sg2_apply_revolute_y(T, 0.041004f, 0.0f, -0.135f, q[4]);
        ffw_sg2_apply_revolute_z(T, -0.041004f, 0.0f, -0.1489f, q[5]);
        ffw_sg2_apply_revolute_y(T, 0.0f, 0.0f, -0.1041f, q[6]);
        ffw_sg2_apply_revolute_x(T, 0.0f, 0.0f, -0.0885f, q[7]);
        ffw_sg2_apply_gripper_fixed(T);
        ffw_sg2_extract_pose(T, p, R);
    }

    __device__ __forceinline__ void ffw_sg2_fk_right(const float *q, float p[3], float R[9]) {
        float T[16];
        ffw_sg2_identity4(T);
        ffw_sg2_apply_prismatic_z(T, -0.0199f, 0.0f, 1.4316f, q[0]);
        ffw_sg2_apply_revolute_y(T, 0.0f, -0.1045f, 0.0f, q[8]);
        ffw_sg2_apply_revolute_x(T, 0.0f, -0.123f, 0.0f, q[9]);
        ffw_sg2_apply_revolute_z(T, 0.0f, 0.0f, -0.165f, q[10]);
        ffw_sg2_apply_revolute_y(T, 0.041004f, 0.0f, -0.135f, q[11]);
        ffw_sg2_apply_revolute_z(T, -0.041004f, 0.0f, -0.1489f, q[12]);
        ffw_sg2_apply_revolute_y(T, 0.0f, 0.0f, -0.1041f, q[13]);
        ffw_sg2_apply_revolute_x(T, 0.0f, 0.0f, -0.0885f, q[14]);
        ffw_sg2_apply_gripper_fixed(T);
        ffw_sg2_extract_pose(T, p, R);
    }

    __device__ __forceinline__ void ffw_sg2_zero_jacobian(float J[6 * 15]) {
        for (int i = 0; i < 6 * 15; i++) {
            J[i] = 0.0f;
        }
    }

    __device__ __forceinline__ void ffw_sg2_record_revolute_column(
        const float T_joint[16],
        int col,
        int axis,
        const float p_end[3],
        float J[6 * 15]
    ) {
        float w[3];
        w[0] = T_joint[0 * 4 + axis];
        w[1] = T_joint[1 * 4 + axis];
        w[2] = T_joint[2 * 4 + axis];

        const float p_joint[3] = {T_joint[3], T_joint[7], T_joint[11]};
        const float r[3] = {
            p_end[0] - p_joint[0],
            p_end[1] - p_joint[1],
            p_end[2] - p_joint[2]
        };

        float v[3];
        ffw_sg2_cross(w, r, v);

        J[0 * 15 + col] = v[0];
        J[1 * 15 + col] = v[1];
        J[2 * 15 + col] = v[2];
        J[3 * 15 + col] = w[0];
        J[4 * 15 + col] = w[1];
        J[5 * 15 + col] = w[2];
    }

    __device__ __forceinline__ void ffw_sg2_record_prismatic_column(
        const float T_joint[16],
        int col,
        int axis,
        float J[6 * 15]
    ) {
        J[0 * 15 + col] = T_joint[0 * 4 + axis];
        J[1 * 15 + col] = T_joint[1 * 4 + axis];
        J[2 * 15 + col] = T_joint[2 * 4 + axis];
        J[3 * 15 + col] = 0.0f;
        J[4 * 15 + col] = 0.0f;
        J[5 * 15 + col] = 0.0f;
    }

    __device__ __forceinline__ void ffw_sg2_fk_jacobian_left(
        const float *q,
        float p[3],
        float R[9],
        float Jw[6 * 15]
    ) {
        ffw_sg2_zero_jacobian(Jw);

        float T[16];
        float T_joint[8][16];
        ffw_sg2_identity4(T);

        ffw_sg2_copy4(T, T_joint[0]);
        ffw_sg2_apply_prismatic_z(T, -0.0199f, 0.0f, 1.4316f, q[0]);

        ffw_sg2_copy4(T, T_joint[1]);
        ffw_sg2_apply_translation(T_joint[1], 0.0f, 0.1045f, 0.0f);
        ffw_sg2_apply_revolute_y(T, 0.0f, 0.1045f, 0.0f, q[1]);

        ffw_sg2_copy4(T, T_joint[2]);
        ffw_sg2_apply_translation(T_joint[2], 0.0f, 0.123f, 0.0f);
        ffw_sg2_apply_revolute_x(T, 0.0f, 0.123f, 0.0f, q[2]);

        ffw_sg2_copy4(T, T_joint[3]);
        ffw_sg2_apply_translation(T_joint[3], 0.0f, 0.0f, -0.165f);
        ffw_sg2_apply_revolute_z(T, 0.0f, 0.0f, -0.165f, q[3]);

        ffw_sg2_copy4(T, T_joint[4]);
        ffw_sg2_apply_translation(T_joint[4], 0.041004f, 0.0f, -0.135f);
        ffw_sg2_apply_revolute_y(T, 0.041004f, 0.0f, -0.135f, q[4]);

        ffw_sg2_copy4(T, T_joint[5]);
        ffw_sg2_apply_translation(T_joint[5], -0.041004f, 0.0f, -0.1489f);
        ffw_sg2_apply_revolute_z(T, -0.041004f, 0.0f, -0.1489f, q[5]);

        ffw_sg2_copy4(T, T_joint[6]);
        ffw_sg2_apply_translation(T_joint[6], 0.0f, 0.0f, -0.1041f);
        ffw_sg2_apply_revolute_y(T, 0.0f, 0.0f, -0.1041f, q[6]);

        ffw_sg2_copy4(T, T_joint[7]);
        ffw_sg2_apply_translation(T_joint[7], 0.0f, 0.0f, -0.0885f);
        ffw_sg2_apply_revolute_x(T, 0.0f, 0.0f, -0.0885f, q[7]);

        ffw_sg2_apply_gripper_fixed(T);
        ffw_sg2_extract_pose(T, p, R);

        ffw_sg2_record_prismatic_column(T_joint[0], 0, 2, Jw);
        ffw_sg2_record_revolute_column(T_joint[1], 1, 1, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[2], 2, 0, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[3], 3, 2, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[4], 4, 1, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[5], 5, 2, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[6], 6, 1, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[7], 7, 0, p, Jw);
    }

    __device__ __forceinline__ void ffw_sg2_fk_jacobian_right(
        const float *q,
        float p[3],
        float R[9],
        float Jw[6 * 15]
    ) {
        ffw_sg2_zero_jacobian(Jw);

        float T[16];
        float T_joint[8][16];
        ffw_sg2_identity4(T);

        ffw_sg2_copy4(T, T_joint[0]);
        ffw_sg2_apply_prismatic_z(T, -0.0199f, 0.0f, 1.4316f, q[0]);

        ffw_sg2_copy4(T, T_joint[1]);
        ffw_sg2_apply_translation(T_joint[1], 0.0f, -0.1045f, 0.0f);
        ffw_sg2_apply_revolute_y(T, 0.0f, -0.1045f, 0.0f, q[8]);

        ffw_sg2_copy4(T, T_joint[2]);
        ffw_sg2_apply_translation(T_joint[2], 0.0f, -0.123f, 0.0f);
        ffw_sg2_apply_revolute_x(T, 0.0f, -0.123f, 0.0f, q[9]);

        ffw_sg2_copy4(T, T_joint[3]);
        ffw_sg2_apply_translation(T_joint[3], 0.0f, 0.0f, -0.165f);
        ffw_sg2_apply_revolute_z(T, 0.0f, 0.0f, -0.165f, q[10]);

        ffw_sg2_copy4(T, T_joint[4]);
        ffw_sg2_apply_translation(T_joint[4], 0.041004f, 0.0f, -0.135f);
        ffw_sg2_apply_revolute_y(T, 0.041004f, 0.0f, -0.135f, q[11]);

        ffw_sg2_copy4(T, T_joint[5]);
        ffw_sg2_apply_translation(T_joint[5], -0.041004f, 0.0f, -0.1489f);
        ffw_sg2_apply_revolute_z(T, -0.041004f, 0.0f, -0.1489f, q[12]);

        ffw_sg2_copy4(T, T_joint[6]);
        ffw_sg2_apply_translation(T_joint[6], 0.0f, 0.0f, -0.1041f);
        ffw_sg2_apply_revolute_y(T, 0.0f, 0.0f, -0.1041f, q[13]);

        ffw_sg2_copy4(T, T_joint[7]);
        ffw_sg2_apply_translation(T_joint[7], 0.0f, 0.0f, -0.0885f);
        ffw_sg2_apply_revolute_x(T, 0.0f, 0.0f, -0.0885f, q[14]);

        ffw_sg2_apply_gripper_fixed(T);
        ffw_sg2_extract_pose(T, p, R);

        ffw_sg2_record_prismatic_column(T_joint[0], 0, 2, Jw);
        ffw_sg2_record_revolute_column(T_joint[1], 8, 1, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[2], 9, 0, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[3], 10, 2, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[4], 11, 1, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[5], 12, 2, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[6], 13, 1, p, Jw);
        ffw_sg2_record_revolute_column(T_joint[7], 14, 0, p, Jw);
    }

    // relative-pose contraint
    __device__ __forceinline__ void ffw_sg2_relative_pose_residual(const float *q, float h[6]) {
        float p_l[3], p_r[3], R_l[9], R_r[9];
        ffw_sg2_fk_left(q, p_l, R_l);
        ffw_sg2_fk_right(q, p_r, R_r);

        const float dp[3] = {
            p_r[0] - p_l[0],
            p_r[1] - p_l[1],
            p_r[2] - p_l[2]
        };

        float p_g[3];
        ffw_sg2_mat3_transpose_vec(R_l, dp, p_g);

        float R_g[9];
        ffw_sg2_mat3_mul_at_b(R_l, R_r, R_g);

        const float q_ref[4] = {
            ffw_sg2_q_rel_ref[0],
            ffw_sg2_q_rel_ref[1],
            ffw_sg2_q_rel_ref[2],
            ffw_sg2_q_rel_ref[3]
        };
        float R_des[9];
        ffw_sg2_quat_to_matrix(q_ref, R_des);

        const float p_delta[3] = {
            p_g[0] - ffw_sg2_p_rel_ref[0],
            p_g[1] - ffw_sg2_p_rel_ref[1],
            p_g[2] - ffw_sg2_p_rel_ref[2]
        };

        float p_err[3];
        ffw_sg2_mat3_transpose_vec(R_des, p_delta, p_err);

        float R_err[9];
        ffw_sg2_mat3_mul_at_b(R_des, R_g, R_err);

        ffw_sg2_se3_log(R_err, p_err, h);
    }

    __device__ __forceinline__ float ffw_sg2_residual_norm(const float h[6]) {
        float n2 = 0.0f;
        for (int i = 0; i < 6; i++) {
            n2 += h[i] * h[i];
        }
        return sqrtf(n2);
    }

    __device__ __forceinline__ void ffw_sg2_clamp(float q[15]) {
        for (int i = 0; i < 15; i++) {
            const float lo = ppln::robots::FfwSg2::get_s_a(i);
            const float hi = lo + ppln::robots::FfwSg2::get_s_m(i);
            q[i] = fminf(fmaxf(q[i], lo), hi);
        }
    }

    __device__ __forceinline__ bool ffw_sg2_solve6(float A_in[36], const float b_in[6], float x[6]) {
        float aug[6][7];
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                aug[r][c] = A_in[r * 6 + c];
            }
            aug[r][6] = b_in[r];
        }

        for (int c = 0; c < 6; c++) {
            int pivot = c;
            float best = fabsf(aug[c][c]);
            for (int r = c + 1; r < 6; r++) {
                const float v = fabsf(aug[r][c]);
                if (v > best) {
                    best = v;
                    pivot = r;
                }
            }
            if (best < 1.0e-9f) {
                return false;
            }
            if (pivot != c) {
                for (int k = c; k < 7; k++) {
                    const float tmp = aug[c][k];
                    aug[c][k] = aug[pivot][k];
                    aug[pivot][k] = tmp;
                }
            }
            const float inv_pivot = 1.0f / aug[c][c];
            for (int k = c; k < 7; k++) {
                aug[c][k] *= inv_pivot;
            }
            for (int r = 0; r < 6; r++) {
                if (r == c) {
                    continue;
                }
                const float factor = aug[r][c];
                for (int k = c; k < 7; k++) {
                    aug[r][k] -= factor * aug[c][k];
                }
            }
        }

        for (int i = 0; i < 6; i++) {
            x[i] = aug[i][6];
        }
        return true;
    }

    __device__ __forceinline__ void ffw_sg2_so3_log(const float R[9], float omega[3]) {
        float cos_theta = 0.5f * (R[0] + R[4] + R[8] - 1.0f);
        cos_theta = fminf(fmaxf(cos_theta, -1.0f), 1.0f);

        const float vx = 0.5f * (R[7] - R[5]);
        const float vy = 0.5f * (R[2] - R[6]);
        const float vz = 0.5f * (R[3] - R[1]);
        const float sin_theta = sqrtf(vx * vx + vy * vy + vz * vz);
        const float theta = atan2f(sin_theta, cos_theta);

        if (sin_theta < 1.0e-8f && theta < 1.0e-8f) {
            omega[0] = 0.0f;
            omega[1] = 0.0f;
            omega[2] = 0.0f;
            return;
        }

        if (fabsf(theta - 3.14159265358979323846f) < 1.0e-4f) {
            float axis[3] = {
                sqrtf(fmaxf(0.5f * (R[0] + 1.0f), 0.0f)),
                sqrtf(fmaxf(0.5f * (R[4] + 1.0f), 0.0f)),
                sqrtf(fmaxf(0.5f * (R[8] + 1.0f), 0.0f))
            };
            if (R[7] - R[5] < 0.0f) {
                axis[0] = -axis[0];
            }
            if (R[2] - R[6] < 0.0f) {
                axis[1] = -axis[1];
            }
            if (R[3] - R[1] < 0.0f) {
                axis[2] = -axis[2];
            }
            const float n = sqrtf(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
            const float inv_n = n > 1.0e-8f ? 1.0f / n : 1.0f;
            omega[0] = axis[0] * inv_n * theta;
            omega[1] = axis[1] * inv_n * theta;
            omega[2] = axis[2] * inv_n * theta;
            return;
        }

        const float scale = theta / fmaxf(sin_theta, 1.0e-12f);
        omega[0] = vx * scale;
        omega[1] = vy * scale;
        omega[2] = vz * scale;
    }

    __device__ __forceinline__ void ffw_sg2_se3_log(const float R[9], const float p[3], float xi[6]) {
        float omega[3];
        ffw_sg2_so3_log(R, omega);

        const float theta = sqrtf(omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]);

        float W[9];
        ffw_sg2_skew(omega, W);

        float W2[9];
        ffw_sg2_mat3_mul(W, W, W2);

        float V[9];
        if (theta < 1.0e-6f) {
            for (int i = 0; i < 9; i++) {
                const float I = (i == 0 || i == 4 || i == 8) ? 1.0f : 0.0f;
                V[i] = I + 0.5f * W[i] + (1.0f / 6.0f) * W2[i];
            }
        } else {
            const float th2 = theta * theta;
            const float A = (1.0f - cosf(theta)) / th2;
            const float B = (theta - sinf(theta)) / (theta * th2);
            for (int i = 0; i < 9; i++) {
                const float I = (i == 0 || i == 4 || i == 8) ? 1.0f : 0.0f;
                V[i] = I + A * W[i] + B * W2[i];
            }
        }

        float A6[36] = {0.0f};
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                A6[r * 6 + c] = V[r * 3 + c];
            }
        }
        A6[3 * 6 + 3] = 1.0f;
        A6[4 * 6 + 4] = 1.0f;
        A6[5 * 6 + 5] = 1.0f;

        const float b[6] = {p[0], p[1], p[2], 0.0f, 0.0f, 0.0f};
        float rho[6];
        if (!ffw_sg2_solve6(A6, b, rho)) {
            rho[0] = p[0];
            rho[1] = p[1];
            rho[2] = p[2];
        }

        xi[0] = rho[0];
        xi[1] = rho[1];
        xi[2] = rho[2];
        xi[3] = omega[0];
        xi[4] = omega[1];
        xi[5] = omega[2];
    }

    __device__ __forceinline__ void ffw_sg2_world_to_body_jacobian(
        const float R[9],
        const float Jw[6 * 15],
        float Jb[6 * 15]
    ) {
        for (int j = 0; j < 15; j++) {
            const float v[3] = {
                Jw[0 * 15 + j],
                Jw[1 * 15 + j],
                Jw[2 * 15 + j]
            };
            const float w[3] = {
                Jw[3 * 15 + j],
                Jw[4 * 15 + j],
                Jw[5 * 15 + j]
            };
            float vb[3], wb[3];
            ffw_sg2_mat3_transpose_vec(R, v, vb);
            ffw_sg2_mat3_transpose_vec(R, w, wb);

            Jb[0 * 15 + j] = vb[0];
            Jb[1 * 15 + j] = vb[1];
            Jb[2 * 15 + j] = vb[2];
            Jb[3 * 15 + j] = wb[0];
            Jb[4 * 15 + j] = wb[1];
            Jb[5 * 15 + j] = wb[2];
        }
    }

    __device__ __forceinline__ void ffw_sg2_adjoint_ginv_times_jacobian(
        const float R_g[9],
        const float p_g[3],
        const float J_in[6 * 15],
        float J_out[6 * 15]
    ) {
        float p_inv[3];
        ffw_sg2_mat3_transpose_vec(R_g, p_g, p_inv);
        p_inv[0] = -p_inv[0];
        p_inv[1] = -p_inv[1];
        p_inv[2] = -p_inv[2];

        for (int j = 0; j < 15; j++) {
            const float v[3] = {
                J_in[0 * 15 + j],
                J_in[1 * 15 + j],
                J_in[2 * 15 + j]
            };
            const float w[3] = {
                J_in[3 * 15 + j],
                J_in[4 * 15 + j],
                J_in[5 * 15 + j]
            };

            float Rv[3], Rw[3], pxRw[3];
            ffw_sg2_mat3_transpose_vec(R_g, v, Rv);
            ffw_sg2_mat3_transpose_vec(R_g, w, Rw);
            ffw_sg2_cross(p_inv, Rw, pxRw);

            J_out[0 * 15 + j] = Rv[0] + pxRw[0];
            J_out[1 * 15 + j] = Rv[1] + pxRw[1];
            J_out[2 * 15 + j] = Rv[2] + pxRw[2];
            J_out[3 * 15 + j] = Rw[0];
            J_out[4 * 15 + j] = Rw[1];
            J_out[5 * 15 + j] = Rw[2];
        }
    }

    __device__ __forceinline__ void ffw_sg2_mat6_mul(const float A[36], const float B[36], float C[36]) {
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                float v = 0.0f;
                for (int k = 0; k < 6; k++) {
                    v += A[r * 6 + k] * B[k * 6 + c];
                }
                C[r * 6 + c] = v;
            }
        }
    }

    __device__ __forceinline__ void ffw_sg2_build_ad_matrix(const float xi[6], float ad[36]) {
        for (int i = 0; i < 36; i++) {
            ad[i] = 0.0f;
        }

        const float v[3] = {xi[0], xi[1], xi[2]};
        const float w[3] = {xi[3], xi[4], xi[5]};
        float V[9], W[9];
        ffw_sg2_skew(v, V);
        ffw_sg2_skew(w, W);

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                ad[r * 6 + c] = W[r * 3 + c];
                ad[r * 6 + (c + 3)] = V[r * 3 + c];
                ad[(r + 3) * 6 + (c + 3)] = W[r * 3 + c];
            }
        }
    }

    __device__ __forceinline__ void ffw_sg2_build_jr(const float xi[6], float Jr[36]) {
        float ad[36];
        ffw_sg2_build_ad_matrix(xi, ad);

        float term[36];
        for (int i = 0; i < 36; i++) {
            Jr[i] = 0.0f;
            term[i] = (i / 6 == i % 6) ? 1.0f : 0.0f;
        }

        float factorial = 1.0f;
        float sign = 1.0f;
        for (int n = 0; n <= 20; n++) {
            if (n > 0) {
                factorial *= (float)(n + 1);
            }
            const float coeff = sign / factorial;
            for (int i = 0; i < 36; i++) {
                Jr[i] += coeff * term[i];
            }

            float next[36];
            ffw_sg2_mat6_mul(term, ad, next);
            for (int i = 0; i < 36; i++) {
                term[i] = next[i];
            }
            sign = -sign;
        }
    }

    __device__ __forceinline__ bool ffw_sg2_apply_jr_inv(
        const float xi[6],
        const float J_rel[6 * 15],
        float J[6 * 15]
    ) {
        float Jr[36];
        ffw_sg2_build_jr(xi, Jr);

        for (int col = 0; col < 15; col++) {
            float b[6];
            for (int r = 0; r < 6; r++) {
                b[r] = J_rel[r * 15 + col];
            }

            float x[6];
            if (!ffw_sg2_solve6(Jr, b, x)) {
                return false;
            }

            for (int r = 0; r < 6; r++) {
                J[r * 15 + col] = x[r];
            }
        }

        return true;
    }

    __device__ __forceinline__ bool ffw_sg2_relative_pose_jacobian(
        const float *q,
        float h[6],
        float J[6 * 15]
    ) {
        float p_l[3], p_r[3], R_l[9], R_r[9];
        float Jw_l[6 * 15], Jw_r[6 * 15];
        ffw_sg2_fk_jacobian_left(q, p_l, R_l, Jw_l);
        ffw_sg2_fk_jacobian_right(q, p_r, R_r, Jw_r);

        float Jb_l[6 * 15], Jb_r[6 * 15];
        ffw_sg2_world_to_body_jacobian(R_l, Jw_l, Jb_l);
        ffw_sg2_world_to_body_jacobian(R_r, Jw_r, Jb_r);

        const float dp[3] = {
            p_r[0] - p_l[0],
            p_r[1] - p_l[1],
            p_r[2] - p_l[2]
        };

        float p_g[3];
        ffw_sg2_mat3_transpose_vec(R_l, dp, p_g);

        float R_g[9];
        ffw_sg2_mat3_mul_at_b(R_l, R_r, R_g);

        const float q_ref[4] = {
            ffw_sg2_q_rel_ref[0],
            ffw_sg2_q_rel_ref[1],
            ffw_sg2_q_rel_ref[2],
            ffw_sg2_q_rel_ref[3]
        };
        float R_des[9];
        ffw_sg2_quat_to_matrix(q_ref, R_des);

        const float p_delta[3] = {
            p_g[0] - ffw_sg2_p_rel_ref[0],
            p_g[1] - ffw_sg2_p_rel_ref[1],
            p_g[2] - ffw_sg2_p_rel_ref[2]
        };

        float p_err[3];
        ffw_sg2_mat3_transpose_vec(R_des, p_delta, p_err);

        float R_err[9];
        ffw_sg2_mat3_mul_at_b(R_des, R_g, R_err);

        ffw_sg2_se3_log(R_err, p_err, h);

        float Ad_ginv_Jb_l[6 * 15];
        ffw_sg2_adjoint_ginv_times_jacobian(R_g, p_g, Jb_l, Ad_ginv_Jb_l);

        float J_rel[6 * 15];
        for (int i = 0; i < 6 * 15; i++) {
            J_rel[i] = Jb_r[i] - Ad_ginv_Jb_l[i];
        }

        return ffw_sg2_apply_jr_inv(h, J_rel, J);
    }

    __device__ __forceinline__ int ffw_sg2_constraint_dim(bool rigid_orientation) {
        return rigid_orientation ? 8 : 6;
    }

    __device__ __forceinline__ int ffw_sg2_tangent_dim(bool rigid_orientation) {
        return 15 - ffw_sg2_constraint_dim(rigid_orientation);
    }

    __device__ __forceinline__ void ffw_sg2_axis_residual_and_jacobian(
        const float *q,
        float h[FFW_SG2_MAX_RESIDUAL_DIM],
        float J[FFW_SG2_MAX_RESIDUAL_DIM * 15]
    ) {
        float p_l[3], R_l[9], Jw_l[6 * 15];
        ffw_sg2_fk_jacobian_left(q, p_l, R_l, Jw_l);

        const int axis_col = 1;
        const float axis[3] = {
            R_l[0 * 3 + axis_col],
            R_l[1 * 3 + axis_col],
            R_l[2 * 3 + axis_col]
        };
        h[6] = axis[0];
        h[7] = axis[1];

        for (int col = 0; col < 15; col++) {
            const float wx = Jw_l[3 * 15 + col];
            const float wy = Jw_l[4 * 15 + col];
            const float wz = Jw_l[5 * 15 + col];
            const float dax = wy * axis[2] - wz * axis[1];
            const float day = wz * axis[0] - wx * axis[2];
            J[6 * 15 + col] = dax;
            J[7 * 15 + col] = day;
        }
    }

    __device__ __forceinline__ bool ffw_sg2_constraint_jacobian(
        const float *q,
        bool rigid_orientation,
        float h[FFW_SG2_MAX_RESIDUAL_DIM],
        float J[FFW_SG2_MAX_RESIDUAL_DIM * 15]
    ) {
        if (!ffw_sg2_relative_pose_jacobian(q, h, J)) {
            return false;
        }
        if (rigid_orientation) {
            ffw_sg2_axis_residual_and_jacobian(q, h, J);
        }
        return true;
    }

    __device__ __forceinline__ void ffw_sg2_constraint_residual(
        const float *q,
        bool rigid_orientation,
        float h[FFW_SG2_MAX_RESIDUAL_DIM]
    ) {
        ffw_sg2_relative_pose_residual(q, h);
        if (rigid_orientation) {
            float p_l[3], R_l[9];
            ffw_sg2_fk_left(q, p_l, R_l);
            const int axis_col = 1;
            h[6] = R_l[0 * 3 + axis_col];
            h[7] = R_l[1 * 3 + axis_col];
        }
    }

    __device__ __forceinline__ float ffw_sg2_residual_norm(
        const float h[FFW_SG2_MAX_RESIDUAL_DIM],
        int residual_dim
    ) {
        float n2 = 0.0f;
        for (int i = 0; i < residual_dim; i++) {
            n2 += h[i] * h[i];
        }
        return sqrtf(n2);
    }

    __device__ __forceinline__ bool ffw_sg2_solve_residual_system(
        float A_in[FFW_SG2_MAX_RESIDUAL_DIM * FFW_SG2_MAX_RESIDUAL_DIM],
        const float b_in[FFW_SG2_MAX_RESIDUAL_DIM],
        float x[FFW_SG2_MAX_RESIDUAL_DIM],
        int residual_dim
    ) {
        float aug[FFW_SG2_MAX_RESIDUAL_DIM][FFW_SG2_MAX_RESIDUAL_DIM + 1];
        for (int r = 0; r < residual_dim; r++) {
            for (int c = 0; c < residual_dim; c++) {
                aug[r][c] = A_in[r * FFW_SG2_MAX_RESIDUAL_DIM + c];
            }
            aug[r][residual_dim] = b_in[r];
        }

        for (int c = 0; c < residual_dim; c++) {
            int pivot = c;
            float best = fabsf(aug[c][c]);
            for (int r = c + 1; r < residual_dim; r++) {
                const float v = fabsf(aug[r][c]);
                if (v > best) {
                    best = v;
                    pivot = r;
                }
            }
            if (best < 1.0e-9f) {
                return false;
            }
            if (pivot != c) {
                for (int k = c; k <= residual_dim; k++) {
                    const float tmp = aug[c][k];
                    aug[c][k] = aug[pivot][k];
                    aug[pivot][k] = tmp;
                }
            }
            const float inv_pivot = 1.0f / aug[c][c];
            for (int k = c; k <= residual_dim; k++) {
                aug[c][k] *= inv_pivot;
            }
            for (int r = 0; r < residual_dim; r++) {
                if (r == c) {
                    continue;
                }
                const float factor = aug[r][c];
                for (int k = c; k <= residual_dim; k++) {
                    aug[r][k] -= factor * aug[c][k];
                }
            }
        }

        for (int i = 0; i < residual_dim; i++) {
            x[i] = aug[i][residual_dim];
        }
        return true;
    }

    __device__ __forceinline__ bool ffw_sg2_tangent_basis_from_jacobian(
        const float J[FFW_SG2_MAX_RESIDUAL_DIM * 15],
        bool rigid_orientation,
        float basis[15 * 9]
    ) {
        const int residual_dim = ffw_sg2_constraint_dim(rigid_orientation);

        const int active_tangent_dim = ffw_sg2_tangent_dim(rigid_orientation);

        // 1. 전달받은 Jacobian J를 계산용 행렬 A로 복사
        float A[FFW_SG2_MAX_RESIDUAL_DIM][15];

        for (int r = 0; r < residual_dim; r++) {
            for (int c = 0; c < 15; c++) {
                A[r][c] = J[r * 15 + c];
            }
        }

        // 2. Gaussian elimination으로 pivot column 찾기
        int pivot_cols[FFW_SG2_MAX_RESIDUAL_DIM];
        int rank = 0;

        for (int col = 0; col < 15 && rank < residual_dim; col++) {
            int pivot = rank;
            float best = fabsf(A[rank][col]);

            for (int r = rank + 1; r < residual_dim;  r++) {
                const float v = fabsf(A[r][col]);

                if (v > best) {
                    best = v;
                    pivot = r;
                }
            }

            // 이 column은 pivot으로 사용하기 너무 작음
            if (best < 1.0e-5f) {
                continue;
            }

            // 가장 좋은 pivot row를 현재 rank 위치로 이동
            if (pivot != rank) {
                for (int c = 0; c < 15; c++) {
                    const float tmp = A[rank][c];
                    A[rank][c] = A[pivot][c];
                    A[pivot][c] = tmp;
                }
            }

            // pivot 값을 1로 만듦
            const float inv_pivot = 1.0f / A[rank][col];

            for (int c = col; c < 15; c++) {
                A[rank][c] *= inv_pivot;
            }

            // 다른 row에서 현재 pivot column 제거
            for (int r = 0; r < residual_dim; r++) {

                if (r == rank) {
                    continue;
                }

                const float factor =A[r][col];

                for (int c = col; c < 15; c++) {

                    A[r][c] -= factor * A[rank][c];
                }
            }

            pivot_cols[rank] = col;
            rank++;
        }

        // 3. 어떤 joint dimension이 pivot인지 표시
        bool is_pivot[15];

        for (int c = 0; c < 15; c++) {
            is_pivot[c] = false;
        }

        for (int r = 0; r < rank; r++) {
            is_pivot[pivot_cols[r]] = true;
        }

        // 4. Tangent basis 저장 공간 초기화
        for (int i = 0; i < 15 * 9; i++) {
            basis[i] = 0.0f;
        }

        // 5. free variable을 이용해서 null-space basis 생성
        int basis_col = 0;

        for (int free_col = 0; free_col < 15 && basis_col < active_tangent_dim; free_col++) {

            if (is_pivot[free_col]) {
                continue;
            }

            float v[15];

            for (int i = 0; i < 15; i++) {
                v[i] = 0.0f;
            }

            // 현재 free variable을 1로 둠
            v[free_col] = 1.0f;

            // pivot variable 계산
            for (int r = 0; r < rank; r++) {
                v[pivot_cols[r]] =
                    -A[r][free_col];
            }

            // 6. Gram-Schmidt orthogonalization
            for (int prev = 0; prev < basis_col; prev++) {
                float dot = 0.0f;

                for (int i = 0; i < 15; i++) {
                    dot += v[i] * basis[i * 9 + prev];
                }

                for (int i = 0; i < 15; i++) {
                    v[i] -= dot * basis[i * 9 + prev];
                }
            }

            // 7. basis vector normalize
            float norm2 = 0.0f;

            for (int i = 0; i < 15; i++) {
                norm2 += v[i] * v[i];
            }

            if (norm2 < 1.0e-10f) {
                continue;
            }

            const float inv_norm = rsqrtf(norm2);

            for (int i = 0; i < 15; i++) {
                basis[i * 9 + basis_col] = v[i] * inv_norm;
            }

            basis_col++;
        }

        // Jacobian이 기대한 rank를 가지고 있고
        // 필요한 tangent basis 개수도 만들어졌는지 검사
        return
            rank == residual_dim &&
            basis_col == active_tangent_dim;
    }
    __device__ __forceinline__ bool ffw_sg2_tangent_basis(
        const float *q,
        bool rigid_orientation,
        float basis[15 * 9]
    ) {
        float h[FFW_SG2_MAX_RESIDUAL_DIM];
        float J[FFW_SG2_MAX_RESIDUAL_DIM * 15];

        for (int i = 0;
            i < FFW_SG2_MAX_RESIDUAL_DIM * 15;
            i++) {
            J[i] = 0.0f;
        }

        // 현재 tree node q에서 Jacobian 새로 1회 계산
        if (!ffw_sg2_constraint_jacobian(
                q,
                rigid_orientation,
                h,
                J
            )) {
            return false;
        }

        // 이미 구현한 J -> null-space basis 함수 사용
        return ffw_sg2_tangent_basis_from_jacobian(
            J,
            rigid_orientation,
            basis
        );
    }

    __device__ __forceinline__ void ffw_sg2_copy_constraint_jacobian(
        const float src[FFW_SG2_MAX_RESIDUAL_DIM * 15],
        float *dst
    ) {
        if (dst == nullptr) {
            return;
        }

        for (int i = 0; i < FFW_SG2_MAX_RESIDUAL_DIM * 15; i++) {
            dst[i] = src[i];
        }
    }

    // DLS projection
    __device__ __forceinline__ bool ffw_sg2_task_correction(
        const float q[15],
        bool rigid_orientation,
        float damping,
        float max_step,
        float correction[15],
        float &task_error_norm
    ) {
        const int residual_dim = ffw_sg2_constraint_dim(rigid_orientation);
        float h[FFW_SG2_MAX_RESIDUAL_DIM];
        float J[FFW_SG2_MAX_RESIDUAL_DIM * 15];
        for (int i = 0; i < FFW_SG2_MAX_RESIDUAL_DIM * 15; i++) {
            J[i] = 0.0f;
        }
        if (!ffw_sg2_constraint_jacobian(q, rigid_orientation, h, J)) {
            return false;
        }

        task_error_norm = ffw_sg2_residual_norm(h, residual_dim);
        for (int j = 0; j < 15; j++) {
            correction[j] = 0.0f;
        }
        if (task_error_norm <= 1.0e-12f) {
            return true;
        }

        float A[FFW_SG2_MAX_RESIDUAL_DIM * FFW_SG2_MAX_RESIDUAL_DIM];
        for (int r = 0; r < residual_dim; r++) {
            for (int c = 0; c < residual_dim; c++) {
                float v = 0.0f;
                for (int j = 0; j < 15; j++) {
                    v += J[r * 15 + j] * J[c * 15 + j];
                }
                A[r * FFW_SG2_MAX_RESIDUAL_DIM + c] =
                    v + (r == c ? damping : 0.0f);
            }
        }

        float y[FFW_SG2_MAX_RESIDUAL_DIM];
        if (!ffw_sg2_solve_residual_system(A, h, y, residual_dim)) {
            return false;
        }

        float step_norm2 = 0.0f;
        for (int j = 0; j < 15; j++) {
            float v = 0.0f;
            for (int r = 0; r < residual_dim; r++) {
                v += J[r * 15 + j] * y[r];
            }
            correction[j] = v;
            step_norm2 += v * v;
        }

        const float step_norm = sqrtf(step_norm2);
        if (max_step > 0.0f && step_norm > max_step) {
            const float scale = max_step / step_norm;
            for (int j = 0; j < 15; j++) {
                correction[j] *= scale;
            }
        }
        return true;
    }

    // project_config
    __device__ __forceinline__ bool ffw_sg2_project_config(
        float q[15],
        bool rigid_orientation,
        float *projected_jacobian = nullptr
    ) {
        constexpr int max_iters = 15;
        constexpr float tol = 1.0e-3f;
        constexpr float damping = 1.0e-4f;
        constexpr float max_step = 0.2f;
        const int residual_dim = ffw_sg2_constraint_dim(rigid_orientation);

        ffw_sg2_clamp(q);
        float h[FFW_SG2_MAX_RESIDUAL_DIM];
        for (int iter = 0; iter < max_iters; iter++) {
            float J[FFW_SG2_MAX_RESIDUAL_DIM * 15];
            for (int i = 0; i < FFW_SG2_MAX_RESIDUAL_DIM * 15; i++) {
                J[i] = 0.0f;
            }
            if (!ffw_sg2_constraint_jacobian(q, rigid_orientation, h, J)) {
                return false;
            }
            if (ffw_sg2_residual_norm(h, residual_dim) < tol) {
                ffw_sg2_copy_constraint_jacobian(J, projected_jacobian);
                return true;
            }

            float A[FFW_SG2_MAX_RESIDUAL_DIM * FFW_SG2_MAX_RESIDUAL_DIM];
            for (int r = 0; r < residual_dim; r++) {
                for (int c = 0; c < residual_dim; c++) {
                    float v = 0.0f;
                    for (int j = 0; j < 15; j++) {
                        v += J[r * 15 + j] * J[c * 15 + j];
                    }
                    A[r * FFW_SG2_MAX_RESIDUAL_DIM + c] = v + (r == c ? damping : 0.0f);
                }
            }

            float y[FFW_SG2_MAX_RESIDUAL_DIM];
            if (!ffw_sg2_solve_residual_system(A, h, y, residual_dim)) {
                return false;
            }

            float dq[15];
            float step_norm2 = 0.0f;
            for (int j = 0; j < 15; j++) {
                float v = 0.0f;
                for (int r = 0; r < residual_dim; r++) {
                    v += J[r * 15 + j] * y[r];
                }
                dq[j] = v;
                step_norm2 += v * v;
            }
            const float step_norm = sqrtf(step_norm2);
            const float step_scale = step_norm > max_step ? max_step / step_norm : 1.0f;
            for (int j = 0; j < 15; j++) {
                q[j] -= step_scale * dq[j];
            }
            ffw_sg2_clamp(q);
        }

        float J[FFW_SG2_MAX_RESIDUAL_DIM * 15];
        for (int i = 0; i < FFW_SG2_MAX_RESIDUAL_DIM * 15; i++) {
            J[i] = 0.0f;
        }
        if (!ffw_sg2_constraint_jacobian(q, rigid_orientation, h, J)) {
            return false;
        }
        const bool ok = ffw_sg2_residual_norm(h, residual_dim) < tol;
        if (ok) {
            ffw_sg2_copy_constraint_jacobian(J, projected_jacobian);
        }
        return ok;
    }

    // ffw-sg2 project motion
    __device__ __forceinline__ bool ffw_sg2_project_motion(
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        int granularity,
        bool rigid_orientation,
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
        int tid
    ) {
        // Four CUDA threads form one waypoint group.  The group mapping is
        // waypoint = tid / 4 + 1 and lane = tid % 4.  To preserve the exact
        // none_NVRTC cpRRTC projection algorithm, lane 0 executes the complete
        // DLS + smoothness update while all four lanes cooperatively move the
        // waypoint vector and later share FK/collision work.
        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        if (tid == 0) {
            projection_prog[0] = 0;
            projection_success[0] = 0;
            projection_valid[0] = 1;
        }
        if (tid < 15) {
            motion_segment_next[tid] = motion_segment[tid];
        }
        if (waypoint <= granularity && lane == 0) {
            projection_valid[waypoint] = 0;
        }
        __syncthreads();

        for (int iter = 0; iter < max_iters; iter++) {
            if (waypoint <= granularity) {
                const int current_prog = projection_prog[0];
                if (waypoint > current_prog) {
                    if (lane == 0) {
                        float q[15];
                        float correction[15];
                        float task_error_norm = 1.0e30f;
                        for (int j = 0; j < 15; j++) {
                            q[j] = motion_segment[waypoint * 15 + j];
                        }

                        const bool correction_ok = ffw_sg2_task_correction(
                                q,
                                rigid_orientation,
                                damping,
                                max_step,
                                correction,
                                task_error_norm                            
                            );

                        float diff[15];
                        float smooth_dist2 = 0.0f;
                        for (int j = 0; j < 15; j++) {
                            diff[j] =
                                q[j] -
                                motion_segment[(waypoint - 1) * 15 + j];
                            smooth_dist2 += diff[j] * diff[j];
                        }
                        const float smooth_dist = sqrtf(smooth_dist2);
                        const float smooth_error = use_smoothness
                            ? fmaxf(
                                0.0f,
                                smooth_dist - smoothness_threshold
                            )
                            : 0.0f;

                        const float inv_smooth_dist =
                            smooth_dist > 1.0e-8f
                                ? 1.0f / smooth_dist
                                : 0.0f;

                        float combined_norm2 = 0.0f;
                        float combined[15];
                        for (int j = 0; j < 15; j++) {
                            const float grad_smooth =
                                diff[j] * inv_smooth_dist * smooth_error;
                            combined[j] = alpha * (
                                correction[j] +
                                smoothness_weight * grad_smooth
                            );
                            combined_norm2 += combined[j] * combined[j];
                        }

                        const float combined_norm = sqrtf(combined_norm2);
                        const float combined_scale =
                            max_step > 0.0f && combined_norm > max_step
                                ? max_step / combined_norm
                                : 1.0f;
                        for (int j = 0; j < 15; j++) {
                            motion_segment_next[waypoint * 15 + j] =
                                q[j] - combined_scale * combined[j];
                        }

                        float q_next[15];
                        for (int j = 0; j < 15; j++) {
                            q_next[j] =
                                motion_segment_next[waypoint * 15 + j];
                        }
                        ffw_sg2_clamp(q_next);
                        for (int j = 0; j < 15; j++) {
                            motion_segment_next[waypoint * 15 + j] =
                                q_next[j];
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
                    for (int j = lane; j < 15; j += 4) {
                        motion_segment_next[waypoint * 15 + j] =
                            motion_segment[waypoint * 15 + j];
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
                    prog++;
                }
                projection_prog[0] = prog;
                if (prog == granularity) {
                    projection_success[0] = 1;
                }
            }
            __syncthreads();

            if (projection_success[0] != 0) {
                return true;
            }

            if (
                waypoint <= granularity &&
                waypoint > projection_prog[0]
            ) {
                for (int j = lane; j < 15; j += 4) {
                    motion_segment[waypoint * 15 + j] =
                        motion_segment_next[waypoint * 15 + j];
                }
            }
            __syncthreads();
        }

        return false;
    }

} // namespace ppln::collision
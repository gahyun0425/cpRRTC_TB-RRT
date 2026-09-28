#pragma once

#include <array>
#include <cstddef>
#include <iostream>
#include <string_view>

#include <cuda_runtime.h>

namespace ppln::robots {

// Descriptor for the 35-coordinate IGRIS-C planning model. The articulated
// hand geometry is present, while its finger joints are fixed outside q.
struct IgrisC {
    static constexpr auto name = "igris_c";
    static constexpr int dimension = 35;
    static constexpr int resolution = 16;
    static constexpr int n_spheres = 190;
    static constexpr int n_approx_spheres = 30;
    static constexpr int n_eef = 4;
    using Configuration = std::array<float, dimension>;

    static constexpr std::array<std::string_view, dimension> joint_names = {
        "world_to_x", "x_to_y", "y_to_z", "z_to_roll",
        "roll_to_pitch", "pitch_to_yaw", "l_hip_pitch", "l_hip_roll",
        "l_hip_yaw", "l_knee_pitch", "l_ankle_pitch", "l_ankle_roll",
        "r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee_pitch",
        "r_ankle_pitch", "r_ankle_roll", "waist_pitch", "waist_roll",
        "waist_yaw", "l_shoulder_pitch", "l_shoulder_roll",
        "l_shoulder_yaw", "l_elbow_pitch", "l_wrist_yaw",
        "l_wrist_roll", "l_wrist_pitch", "r_shoulder_pitch",
        "r_shoulder_roll", "r_shoulder_yaw", "r_elbow_pitch",
        "r_wrist_yaw", "r_wrist_roll", "r_wrist_pitch"
    };

    static constexpr std::array<std::string_view, n_eef> end_effectors = {
        "l_grasp", "r_grasp", "l_sole", "r_sole"
    };

    __host__ __device__ static constexpr float get_s_m(int index) {
        constexpr float values[dimension] = {
            4.0f, 4.0f, 4.0f,
            6.28318530717958f, 6.28318530717958f, 6.28318530717958f,
            2.583087293f, 2.687807048f, 3.141592654f, 2.28638132f,
            1.396263402f, 0.6981317f,
            2.583087293f, 2.687807048f, 3.141592654f, 2.28638132f,
            1.396263402f, 0.6981317f,
            1.160643953f, 0.62831853f, 3.141592654f,
            4.188790205f, 3.316125579f, 3.141592654f, 2.094395102f,
            3.141592654f, 2.094395102f, 1.396263402f,
            4.188790205f, 3.316125579f, 3.141592654f, 2.094395102f,
            3.141592654f, 2.094395102f, 1.396263402f
        };
        return values[index];
    }

    __host__ __device__ static constexpr float get_s_a(int index) {
        constexpr float values[dimension] = {
            -2.0f, -2.0f, -2.0f,
            -3.14159265358979f, -3.14159265358979f, -3.14159265358979f,
            -2.059488517f, -0.331612558f, -1.570796327f, 0.0f,
            -0.698131701f, -0.34906585f,
            -2.059488517f, -2.35619449f, -1.570796327f, 0.0f,
            -0.698131701f, -0.34906585f,
            -0.872664626f, -0.314159265f, -1.570796327f,
            -3.141592654f, -0.174532925f, -1.570796327f,
            -2.094395102f, -1.570796327f, -1.221730476f, -0.698131701f,
            -3.141592654f, -3.141592654f, -1.570796327f,
            -2.094395102f, -1.570796327f, -0.872664626f, -0.698131701f
        };
        return values[index];
    }

    __host__ __device__ static constexpr float get_d_m(int index) {
        return 1.0f / get_s_m(index);
    }

    template<std::size_t Index = 0>
    __device__ __forceinline__ static void scale_cfg_impl(float *q) {
        if constexpr (Index < dimension) {
            q[Index] = q[Index] * get_s_m(Index) + get_s_a(Index);
            scale_cfg_impl<Index + 1>(q);
        }
    }

    __device__ __forceinline__ static void scale_cfg(float *q) {
        scale_cfg_impl(q);
    }

    template<std::size_t Index = 0>
    __device__ __forceinline__ static void descale_cfg_impl(float *q) {
        if constexpr (Index < dimension) {
            q[Index] = (q[Index] - get_s_a(Index)) * get_d_m(Index);
            descale_cfg_impl<Index + 1>(q);
        }
    }

    __device__ __forceinline__ static void descale_cfg(float *q) {
        descale_cfg_impl(q);
    }

    inline static void print_robot_config(Configuration &q) {
        for (float value : q) {
            std::cout << value << ' ';
        }
        std::cout << '\n';
    }
};

}  // namespace ppln::robots

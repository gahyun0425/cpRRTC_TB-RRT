#pragma once

#include <array>
#include <iostream>
#include <string_view>
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace ppln::robots {
    struct Panda
    {
        static constexpr auto name = "panda";
        static constexpr auto dimension = 7;
        using Configuration = std::array<float, dimension>;

        // necessary to generate the scale_cfg function at compile time
        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                5.9342f,
                3.6652f,
                5.9342f,
                3.2289f,
                5.9342f,
                3.9095999999999997f,
                5.9342f
            };
            return values[i];
        }
        
        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                -2.9671f,
                -1.8326f,
                -2.9671f,
                -3.1416f,
                -2.9671f,
                -0.0873f,
                -2.9671f
            };
            return values[i];
        }

        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };
        
        // template metaprogramming to generate the scale_cfg function
        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };

    struct Fetch
    {
        static constexpr auto name = "fetch";
        static constexpr auto dimension = 8;
        using Configuration = std::array<float, dimension>;

        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                0.38615f,
                3.2112f,
                2.739f,
                6.28318f,
                4.502f,
                6.28318f,
                4.32f,
                6.28318f
            };
            return values[i];
        }
        
        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                0.0f,
                -1.6056f,
                -1.221f,
                -3.14159f,
                -2.251f,
                -3.14159f,
                -2.16f,
                -3.14159f
            };
            return values[i];
        }
        
        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };
        
        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };

    struct Baxter
    {
        static constexpr auto name = "baxter";
        static constexpr auto dimension = 14;
        using Configuration = std::array<float, dimension>;

        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                3.40335987756,
                3.194,
                6.10835987756,
                2.6679999999999997,
                6.118,
                3.66479632679,
                6.118,
                3.40335987756,
                3.194,
                6.10835987756,
                2.6679999999999997,
                6.118,
                3.66479632679,
                6.118
            };
            return values[i];
        }
        
        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                -1.70167993878,
                -2.147,
                -3.05417993878,
                -0.05,
                -3.059,
                -1.57079632679,
                -3.059,
                -1.70167993878,
                -2.147,
                -3.05417993878,
                -0.05,
                -3.059,
                -1.57079632679,
                -3.059
            };
            return values[i];
        }

        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };
        
        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };

    struct FfwSg2
    {
        static constexpr auto name = "ffw_sg2";
        static constexpr auto dimension = 15;
        using Configuration = std::array<float, dimension>;

        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                0.5f,
                6.28f,
                3.14f,
                6.28f,
                4.0147f,
                6.28f,
                3.14f,
                3.4005f,
                6.28f,
                3.14f,
                6.28f,
                4.0147f,
                6.28f,
                3.14f,
                3.4005f
            };
            return values[i];
        }

        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                -0.5f,
                -3.14f,
                0.0f,
                -3.14f,
                -2.9361f,
                -3.14f,
                -1.57f,
                -1.8201f,
                -3.14f,
                -3.14f,
                -3.14f,
                -2.9361f,
                -3.14f,
                -1.57f,
                -1.5804f
            };
            return values[i];
        }

        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };

        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };

    struct FfwSg2Single
    {
        static constexpr auto name = "ffw_sg2_single";
        static constexpr auto dimension = 8;
        using Configuration = std::array<float, dimension>;

        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                0.5f,
                6.28f,
                3.14f,
                6.28f,
                4.0147f,
                6.28f,
                3.14f,
                3.4005f
            };
            return values[i];
        }

        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                -0.5f,
                -3.14f,
                -3.14f,
                -3.14f,
                -2.9361f,
                -3.14f,
                -1.57f,
                -1.5804f
            };
            return values[i];
        }

        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };

        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };

    struct Sphere
    {
        static constexpr auto name = "sphere";
        static constexpr auto dimension = 3;
        static constexpr float radius = 1.0;
        using Configuration = std::array<float, dimension>;

        // necessary to generate the scale_cfg function at compile time
        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                10.0f,
                10.0f,
                10.0f
            };
            return values[i];
        }
        
        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                -5.0f,
                -5.0f,
                -5.0f
            };
            return values[i];
        }

        inline static void print_robot_config(Configuration &cfg) {
            for (int i = 0; i < dimension; i++) {
                std::cout << cfg[i] << ' ';
            }
            std::cout << '\n';
        };

        // template metaprogramming to generate the scale_cfg function
        template<size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }
    };
    struct G1
    {
        static constexpr auto name = "g1_unitree";
        static constexpr int dimension = 35;
        static constexpr int n_spheres = 133;
        static constexpr float min_radius = 0.012152000330388546f;
        static constexpr float max_radius = 0.12820099294185638f;
        static constexpr int resolution = 16;
        static constexpr int n_eef = 4;
        static constexpr int num_bounding_spheres = 31;
        static constexpr int num_closed_link_chains = 0;
        using Configuration = std::array<float, dimension>;

        static constexpr std::array<std::string_view, dimension> joint_names = {
            "world_to_x",
            "x_to_y",
            "y_to_z",
            "z_to_roll",
            "roll_to_pitch",
            "pitch_to_yaw",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint"
        };
        static constexpr std::array<std::string_view, n_eef> end_effectors = {
            "left_rubber_hand",
            "right_rubber_hand",
            "left_foot",
            "right_foot"
        };

        __device__ static constexpr float get_s_m(int i)
        {
            constexpr float values[dimension] = {
                4.0f,
                4.0f,
                4.0f,
                6.283180236816406f,
                6.283180236816406f,
                6.283180236816406f,
                5.4105000495910645f,
                3.4907000064849854f,
                5.515200138092041f,
                2.967067003250122f,
                1.3962700366973877f,
                0.5235999822616577f,
                5.4105000495910645f,
                3.4907000064849854f,
                5.515200138092041f,
                2.967067003250122f,
                1.3962700366973877f,
                0.5235999822616577f,
                5.236000061035156f,
                1.0399999618530273f,
                1.0399999618530273f,
                5.7596001625061035f,
                3.8396999835968018f,
                5.236000061035156f,
                3.1415998935699463f,
                3.944444179534912f,
                3.2288591861724854f,
                3.2288591861724854f,
                5.7596001625061035f,
                3.8396999835968018f,
                5.236000061035156f,
                3.1415998935699463f,
                3.944444179534912f,
                3.2288591861724854f,
                3.2288591861724854f
            };
            return values[i];
        }

        __device__ static constexpr float get_s_a(int i)
        {
            constexpr float values[dimension] = {
                -2.0f,
                -2.0f,
                -2.0f,
                -3.141590118408203f,
                -3.141590118408203f,
                -3.141590118408203f,
                -2.5306999683380127f,
                -0.5235999822616577f,
                -2.7576000690460205f,
                -0.08726699650287628f,
                -0.8726699948310852f,
                -0.26179999113082886f,
                -2.5306999683380127f,
                -2.967099905014038f,
                -2.7576000690460205f,
                -0.08726699650287628f,
                -0.8726699948310852f,
                -0.26179999113082886f,
                -2.618000030517578f,
                -0.5199999809265137f,
                -0.5199999809265137f,
                -3.089200019836426f,
                -1.5881999731063843f,
                -2.618000030517578f,
                -1.0471999645233154f,
                -1.972222089767456f,
                -1.6144295930862427f,
                -1.6144295930862427f,
                -3.089200019836426f,
                -2.251499891281128f,
                -2.618000030517578f,
                -1.0471999645233154f,
                -1.972222089767456f,
                -1.6144295930862427f,
                -1.6144295930862427f
            };
            return values[i];
        }

        __device__ static constexpr float get_d_m(int i)
        {
            constexpr float values[dimension] = {
                0.25f,
                0.25f,
                0.25f,
                0.15915507078170776f,
                0.15915507078170776f,
                0.15915507078170776f,
                0.1848258078098297f,
                0.2864754796028137f,
                0.18131709098815918f,
                0.3370331823825836f,
                0.7161938548088074f,
                1.9098548889160156f,
                0.1848258078098297f,
                0.2864754796028137f,
                0.18131709098815918f,
                0.3370331823825836f,
                0.7161938548088074f,
                1.9098548889160156f,
                0.19098548591136932f,
                0.9615384340286255f,
                0.9615384340286255f,
                0.17362317442893982f,
                0.26043701171875f,
                0.19098548591136932f,
                0.31830912828445435f,
                0.2535211443901062f,
                0.3097069263458252f,
                0.3097069263458252f,
                0.17362317442893982f,
                0.26043701171875f,
                0.19098548591136932f,
                0.31830912828445435f,
                0.2535211443901062f,
                0.3097069263458252f,
                0.3097069263458252f
            };
            return values[i];
        }

        template<std::size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = q[I] * get_s_m(I) + get_s_a(I);
                scale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void scale_cfg(float *q)
        {
            scale_cfg_impl(q);
        }

        template<std::size_t I = 0>
        __device__ __forceinline__ static void descale_cfg_impl(float *q)
        {
            if constexpr (I < dimension) {
                q[I] = (q[I] - get_s_a(I)) * get_d_m(I);
                descale_cfg_impl<I + 1>(q);
            }
        }

        __device__ __forceinline__ static void descale_cfg(float *q)
        {
            descale_cfg_impl(q);
        }

        inline static void print_robot_config(Configuration &q)
        {
            for (float value : q) {
                std::cout << value << ' ';
            }
            std::cout << '\n';
        }
    };
}

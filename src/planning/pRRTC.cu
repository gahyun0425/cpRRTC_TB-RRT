#include "Planners.hh"
#include "Robots.hh"
#include "RobotCollisionTraits.hh"
#include "utils.cuh"
#include "pRRTC_settings.hh"
#include "src/collision/environment.hh"
#include "src/robots/panda.cuh"
#include "src/robots/fetch.cuh"
#include "src/robots/baxter.cuh"
#include "src/robots/ffw_sg2.cuh"
#include "src/robots/ffw_sg2_mobility.cuh"
#include "src/robots/ffw_sg2_single.cuh"
#include "src/robots/ffw_sg2_constraint.cuh"
#include "src/robots/ffw_sg2_mobility_constraint.cuh"
#include "src/robots/ffw_sg2_mobility_com_constraint.cuh"
#include "src/robots/g1_collision.cuh"
#include "src/robots/g1_constraint.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <type_traits>



/*
Parallelized RRTC: Each block works to add a config to the tree (either start or goal depending on balance)
*/


namespace pRRTC {
    using namespace ppln;

    static_assert(robots::CollisionTraits<robots::FfwSg2>::batch_size== FFW_SG2_BATCH_SIZE,
        "FFW-SG2 batch size differs from the generated Cricket code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2>::fine_sphere_count== FFW_SG2_SPHERE_COUNT,
        "FFW-SG2 fine sphere count differs from the generated Cricket code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2>::approximate_sphere_count== FFW_SG2_APPROX_SPHERE_COUNT,
        "FFW-SG2 approximate sphere count differs from the generated Cricket code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2>::joint_flag_stride== FFW_SG2_JOINT_FLAG_STRIDE,
        "FFW-SG2 joint flag stride differs from the generated Cricket code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2>::transform_slots== FFW_SG2_TRANSFORM_SLOTS,
        "FFW-SG2 transform slot count differs from the generated Cricket code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2Mobility>::batch_size== FFW_SG2_MOBILITY_BATCH_SIZE,
        "FFW-SG2 mobility batch size differs from the wrapper collision code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2Mobility>::fine_sphere_count== FFW_SG2_MOBILITY_SPHERE_COUNT,
        "FFW-SG2 mobility fine sphere count differs from the wrapper collision code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2Mobility>::approximate_sphere_count== FFW_SG2_MOBILITY_APPROX_SPHERE_COUNT,
        "FFW-SG2 mobility approximate sphere count differs from the wrapper collision code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2Mobility>::joint_flag_stride== FFW_SG2_MOBILITY_JOINT_FLAG_STRIDE,
        "FFW-SG2 mobility joint flag stride differs from the wrapper collision code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2Mobility>::transform_slots== FFW_SG2_MOBILITY_TRANSFORM_SLOTS,
        "FFW-SG2 mobility transform slot count differs from the wrapper collision code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2Single>::batch_size== FFW_SG2_SINGLE_BATCH_SIZE,
        "FFW-SG2 single batch size differs from the generated Cricket code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2Single>::fine_sphere_count== FFW_SG2_SINGLE_SPHERE_COUNT,
        "FFW-SG2 single fine sphere count differs from the generated Cricket code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2Single>::approximate_sphere_count== FFW_SG2_SINGLE_APPROX_SPHERE_COUNT,
        "FFW-SG2 single approximate sphere count differs from the generated Cricket code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2Single>::joint_flag_stride== FFW_SG2_SINGLE_JOINT_FLAG_STRIDE,
        "FFW-SG2 single joint flag stride differs from the generated Cricket code"
    );
    static_assert(robots::CollisionTraits<robots::FfwSg2Single>::transform_slots== FFW_SG2_SINGLE_TRANSFORM_SLOTS,
        "FFW-SG2 single transform slot count differs from the generated Cricket code"
    );
    static_assert(
        robots::CollisionTraits<robots::G1>::batch_size == collision::G1_BATCH_SIZE,
        "G1 batch size differs from the generated collision code"
    );
    static_assert(
        robots::CollisionTraits<robots::G1>::fine_sphere_count == collision::G1_SPHERE_COUNT,
        "G1 sphere count differs from the generated collision code"
    );
    static_assert(
        robots::CollisionTraits<robots::G1>::approximate_sphere_count == collision::G1_APPROX_SPHERE_COUNT,
        "G1 approximate sphere count differs from the generated collision code"
    );
    static_assert(
        robots::CollisionTraits<robots::G1>::joint_flag_stride == collision::G1_JOINT_FLAG_STRIDE,
        "G1 joint flag stride differs from the generated collision code"
    );
    static_assert(
        robots::CollisionTraits<robots::G1>::transform_slots == collision::G1_TRANSFORM_SLOTS,
        "G1 transform slot count differs from the generated collision code"
    );
    __device__ volatile int solved = 0;
    __device__ volatile int atomic_free_index[2]; // separate for tree_a and tree_b
    __device__ volatile int nodes_size[2];
    __device__ volatile int completed_nodes[2]; // track completed nodes for each tree
    constexpr int MAX_PATH_NODES = 5000;
    constexpr int MAX_PATH_STORAGE =
        MAX_PATH_NODES * ppln::robots::G1::dimension;
    __device__ float path[2][MAX_PATH_STORAGE]; // solution path segments for tree_a, and tree_b
    __device__ int path_size[2] = {0, 0};
    __device__ float cost = 0.0;
    __device__ int reached_goal_idx = 0;
    __device__ int connection_tree_id = -1;
    __device__ int connection_node_idx = -1;
    __device__ int connection_other_tree_id = -1;
    __device__ int connection_other_node_idx = -1;
    __device__ int solved_iters = 0; // value of iters in the block that solves the problem
    __constant__ pRRTC_settings d_settings;

    constexpr int MAX_GRANULARITY = 16;
    constexpr int CONCON_COLLISION_THREADS_PER_EDGE = 4 * MAX_GRANULARITY;
    constexpr int MAX_PARALLEL_CONCON_EDGES = 5;
    constexpr int MAX_CONCON_NODE_ANCHORS =
        MAX_PARALLEL_CONCON_EDGES + 1;
    constexpr int MAX_THREADS_PER_BLOCK =
        CONCON_COLLISION_THREADS_PER_EDGE * MAX_PARALLEL_CONCON_EDGES;

    // cpRRTC projected motion shared buffer용
    constexpr int MAX_ROBOT_DIM = ppln::robots::G1::dimension;
    constexpr int CONCON_MOTION_SEGMENT_STRIDE =
        (MAX_GRANULARITY + 1) * MAX_ROBOT_DIM;
    constexpr int CONCON_PROJECTION_STATE_STRIDE = MAX_GRANULARITY + 1;
    constexpr int MAX_CONCON_PROJECTION_STATES =
        MAX_PARALLEL_CONCON_EDGES * CONCON_PROJECTION_STATE_STRIDE;
    constexpr int FFW_SG2_TANGENT_DIM = 9; // 기본 constraint에 따른 tangent dim 15 - 6 = 9. 최대로 필요한 tangent 차원
    constexpr int MAX_TANGENT_DIM = ppln::collision::G1_TANGENT_DIM;

    constexpr int FFW_SG2_TANGENT_BASIS_SIZE = ppln::robots::FfwSg2::dimension * FFW_SG2_TANGENT_DIM;
    constexpr int FFW_SG2_MOBILITY_TANGENT_BASIS_STORAGE_SIZE =
        ppln::robots::FfwSg2Mobility::dimension *
        FFW_SG2_MOBILITY_TANGENT_DIM;
    constexpr int BLOCK_SIZE = 64; // RNG와 Halton 상태 초기화 커널의 thread block 크기. halton 수열의 random성을 위해 RNG 사용
    constexpr float UNWRITTEN_VAL = -9999.0f; // 미작성 configuration 메모리 표기 sentinel 값. 유효성 판단을 위한 flag로 사용

    template <typename Robot>
    struct TangentSpaceTraits {
        static constexpr bool enabled = false;
        static constexpr int max_tangent_dim = 1;
        static constexpr int basis_size = 1;
    };

    template <>
    struct TangentSpaceTraits<robots::FfwSg2> {
        static constexpr bool enabled = true;
        static constexpr int max_tangent_dim = FFW_SG2_TANGENT_DIM;
        static constexpr int basis_size = FFW_SG2_TANGENT_BASIS_SIZE;
    };

    template <>
    struct TangentSpaceTraits<robots::FfwSg2Mobility> {
        static constexpr bool enabled = true;
        static constexpr int max_tangent_dim = FFW_SG2_MOBILITY_TANGENT_DIM;
        static constexpr int basis_size =
            FFW_SG2_MOBILITY_TANGENT_BASIS_STORAGE_SIZE;
    };

    template <>
    struct TangentSpaceTraits<robots::G1> {
        static constexpr bool enabled = true;
        static constexpr int max_tangent_dim = collision::G1_TANGENT_DIM;
        static constexpr int basis_size = collision::G1_TANGENT_BASIS_SIZE;
    };

    template <typename Robot>
    __device__ __forceinline__ int cprrtc_active_tangent_dim() {
        if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
            return d_settings.rigid_orientation ? 7 : FFW_SG2_TANGENT_DIM;
        } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
            return FFW_SG2_MOBILITY_TANGENT_DIM;
        } else if constexpr (std::is_same_v<Robot, robots::G1>) {
            return collision::G1_TANGENT_DIM;
        }
        return 0;
    }


    // 일반 로봇은 모든 관절 가중치가 1
    template <typename Robot>
    __device__ __forceinline__ float cprrtc_joint_distance_weight(
        int joint_index
    ) {
        return 1.0f;
    }


    // FFW-SG2의 0번 좌표는 lift_joint
    template <>
    __device__ __forceinline__ float
    cprrtc_joint_distance_weight<robots::FfwSg2>(
        int joint_index
    ) {
        return joint_index == 0
            ? d_settings.lift_distance_weight
            : 1.0f;
    }


    template <>
    __device__ __forceinline__ float
    cprrtc_joint_distance_weight<robots::FfwSg2Mobility>(
        int joint_index
    ) {
        return joint_index == 3
            ? d_settings.lift_distance_weight
            : 1.0f;
    }


    // Single-arm 모델도 0번 좌표가 lift_joint
    template <>
    __device__ __forceinline__ float
    cprrtc_joint_distance_weight<robots::FfwSg2Single>(
        int joint_index
    ) {
        return joint_index == 0
            ? d_settings.lift_distance_weight
            : 1.0f;
    }


    template <typename Robot>
    __device__ __forceinline__ float cprrtc_sq_config_distance(
        const float* q_a,
        const float* q_b
    ) {
        float result = 0.0f;

        #pragma unroll
        for (int i = 0; i < Robot::dimension; i++) {
            const float weight =
                cprrtc_joint_distance_weight<Robot>(i);

            const float weighted_diff =
                weight * (q_a[i] - q_b[i]);

            result += weighted_diff * weighted_diff;
        }

        return result;
    }


    template <typename Robot>
    __device__ __forceinline__ float cprrtc_config_distance(
        const float* q_a,
        const float* q_b
    ) {
        return sqrtf(
            cprrtc_sq_config_distance<Robot>(q_a, q_b)
        );
    }

    template<typename Robot>
    struct HaltonState {
        float b[Robot::dimension];   // bases
        float n[Robot::dimension];   // numerators
        float d[Robot::dimension];   // denominators
    };

    // Halton에 사용할 prime base 순서를 RNG로 섞는 부분
    void __device__ shuffle_array(float *array, int n, curandState &state) {
        for (int i = n - 1; i > 0; i--) {
            int j = curand(&state) % (i + 1);
            float temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    // 각 CUDA block이 사용할 Halton 수열의 초기 상태 한 번 설정
    template<typename Robot>
    __device__ void halton_initialize(HaltonState<Robot>& state, size_t skip_iterations, curandState& rng_state, int idx) {
        constexpr float prime_table[] = {
            3.f, 5.f, 7.f, 11.f, 13.f, 17.f, 19.f,
            23.f, 29.f, 31.f, 37.f, 41.f, 43.f, 47.f,
            53.f, 59.f, 61.f, 67.f, 71.f, 73.f, 79.f,
            83.f, 89.f, 97.f, 101.f, 103.f, 107.f,
            109.f, 113.f, 127.f, 131.f, 137.f, 139.f,
            149.f, 151.f
        };
        constexpr int shuffle_count =
            std::is_same_v<Robot, robots::G1>
                ? Robot::dimension
                : 16;
        constexpr int prime_count =
            Robot::dimension > shuffle_count
                ? Robot::dimension
                : shuffle_count;
        static_assert(
            prime_count <= sizeof(prime_table) / sizeof(prime_table[0]),
            "Robot dimension exceeds available Halton prime table"
        );

        float primes[prime_count];
        for (size_t i = 0; i < prime_count; i++) {
            primes[i] = prime_table[i];
        }
        if (idx != 0) {
            shuffle_array(primes, shuffle_count, rng_state);
        }
        for (size_t i = 0; i < Robot::dimension; i++) {
            state.b[i] = primes[i];
            state.n[i] = 0.0f;
            state.d[i] = 1.0f;
        }
        
        // Skip iterations if requested
        volatile float temp_result[Robot::dimension];
        for (size_t i = 0; i < skip_iterations; i++) {
            halton_next(state, (float *)temp_result);
        }
    }

    // 초기화된 상태를 이용해 다음 Halton sample 하나 생성
    template<typename Robot>
    __device__ void halton_next(HaltonState<Robot>& state, float* result) {
        for (size_t i = 0; i < Robot::dimension; i++) {
            float xf = state.d[i] - state.n[i];
            bool x_eq_1 = (xf == 1.0f);
            
            if (x_eq_1) {
                // x == 1 case
                state.d[i] = floorf(state.d[i] * state.b[i]);
                state.n[i] = 1.0f;
            } else {
                // x != 1 case
                float y = floorf(state.d[i] / state.b[i]);
                
                // Continue dividing by b until we find the right digit position
                while (xf <= y) {
                    y = floorf(y / state.b[i]);
                }
                
                state.n[i] = floorf((state.b[i] + 1.0f) * y) - xf;
            }
            
            result[i] = state.n[i] / state.d[i];
        }
    }

    // RNG state를 GPU thread들이 병렬로 초기화 (global)
    __global__ void init_rng(curandState* states, unsigned long seed, int num_rng_states) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_rng_states) return;
        curand_init(seed + idx, idx, 0, &states[idx]);
    }

    // Halton state를 GPU thread들이 병렬로 초기화 (global)
    template <typename Robot>
    __global__ void init_halton(HaltonState<Robot>* states, curandState* cr_states) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= d_settings.num_new_configs) return;
        // int skip = (curand_uniform(&cr_states[idx]) * 50000.0f);
        int skip = 0;
        if (idx == 0) skip = 0;
        halton_initialize(states[idx], skip, cr_states[idx], idx);
    }

    __device__ inline void print_config(volatile float *config, int dim) {
        for (int i = 0; i < dim; i++) {
            printf("%f ,", config[i]);
        }
        printf("\n");
    }

    inline void setup_environment_on_device(ppln::collision::Environment<float> *&d_env, const ppln::collision::Environment<float> &h_env) {
        // allocate the environment struct
        cudaMalloc(&d_env, sizeof(ppln::collision::Environment<float>));
        // Initialize struct to zeros first
        cudaMemset(d_env, 0, sizeof(ppln::collision::Environment<float>));

        // Handle each primitive type separately
        if (h_env.num_spheres > 0) {
            // Allocate and copy spheres array
            ppln::collision::Sphere<float> *d_spheres;
            cudaMalloc(&d_spheres, sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres);
            cudaMemcpy(d_spheres, h_env.spheres, sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres, cudaMemcpyHostToDevice);
            // Update the struct fields directly
            cudaMemcpy(&(d_env->spheres), &d_spheres, sizeof(ppln::collision::Sphere<float>*), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_spheres), &h_env.num_spheres, sizeof(unsigned int), cudaMemcpyHostToDevice);
        }

        if (h_env.num_capsules > 0) {
            ppln::collision::Capsule<float> *d_capsules;
            cudaMalloc(&d_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules);
            cudaMemcpy(d_capsules, h_env.capsules,sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules,cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->capsules), &d_capsules, sizeof(ppln::collision::Capsule<float>*),cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_capsules), &h_env.num_capsules, sizeof(unsigned int),cudaMemcpyHostToDevice);
        }

        // Repeat for each primitive type...
        if (h_env.num_z_aligned_capsules > 0) {
            ppln::collision::Capsule<float> *d_z_capsules;
            cudaMalloc(&d_z_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules);
            cudaMemcpy(d_z_capsules, h_env.z_aligned_capsules,sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules,cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->z_aligned_capsules), &d_z_capsules, sizeof(ppln::collision::Capsule<float>*),cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_capsules), &h_env.num_z_aligned_capsules, sizeof(unsigned int),cudaMemcpyHostToDevice);
        }

        if (h_env.num_cylinders > 0) {
            ppln::collision::Cylinder<float> *d_cylinders;
            cudaMalloc(&d_cylinders, sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders);
            cudaMemcpy(d_cylinders, h_env.cylinders,sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders,cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->cylinders), &d_cylinders, sizeof(ppln::collision::Cylinder<float>*),cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cylinders), &h_env.num_cylinders, sizeof(unsigned int),cudaMemcpyHostToDevice);
        }

        if (h_env.num_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_cuboids;
            cudaMalloc(&d_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids);
            cudaMemcpy(d_cuboids, h_env.cuboids,sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids,cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->cuboids), &d_cuboids, sizeof(ppln::collision::Cuboid<float>*),cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cuboids), &h_env.num_cuboids, sizeof(unsigned int),cudaMemcpyHostToDevice);
        }

        if (h_env.num_z_aligned_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_z_cuboids;
            cudaMalloc(&d_z_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids);
            cudaMemcpy(d_z_cuboids, h_env.z_aligned_cuboids,sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids,cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->z_aligned_cuboids), &d_z_cuboids, sizeof(ppln::collision::Cuboid<float>*),cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_cuboids), &h_env.num_z_aligned_cuboids, sizeof(unsigned int),cudaMemcpyHostToDevice);
        }
    }


    inline void cleanup_environment_on_device(ppln::collision::Environment<float> *d_env, const ppln::collision::Environment<float> &h_env) {
        // Get the pointers from device struct before freeing
        ppln::collision::Sphere<float> *d_spheres = nullptr;
        ppln::collision::Capsule<float> *d_capsules = nullptr;
        ppln::collision::Capsule<float> *d_z_capsules = nullptr;
        ppln::collision::Cylinder<float> *d_cylinders = nullptr;
        ppln::collision::Cuboid<float> *d_cuboids = nullptr;
        ppln::collision::Cuboid<float> *d_z_cuboids = nullptr;

        // Copy each pointer from device memory
        if (h_env.num_spheres > 0) {
            cudaMemcpy(&d_spheres, &(d_env->spheres), sizeof(ppln::collision::Sphere<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_spheres);
        }
        
        if (h_env.num_capsules > 0) {
            cudaMemcpy(&d_capsules, &(d_env->capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_capsules);
        }
        
        if (h_env.num_z_aligned_capsules > 0) {
            cudaMemcpy(&d_z_capsules, &(d_env->z_aligned_capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_capsules);
        }
        
        if (h_env.num_cylinders > 0) {
            cudaMemcpy(&d_cylinders, &(d_env->cylinders), sizeof(ppln::collision::Cylinder<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cylinders);
        }
        
        if (h_env.num_cuboids > 0) {
            cudaMemcpy(&d_cuboids, &(d_env->cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cuboids);
        }
        
        if (h_env.num_z_aligned_cuboids > 0) {
            cudaMemcpy(&d_z_cuboids, &(d_env->z_aligned_cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_cuboids);
        }

        // Finally free the environment struct itself
        cudaFree(d_env);
    }

    __global__ void reset_device_variables_kernel() {
        solved = 0;
        atomic_free_index[0] = 0;
        atomic_free_index[1] = 0;
        nodes_size[0] = 0;
        nodes_size[1] = 0;
        completed_nodes[0] = 0;
        completed_nodes[1] = 0;
        path_size[0] = 0;
        path_size[1] = 0;
        cost = 0.0f;
        reached_goal_idx = 0;
        connection_tree_id = -1;
        connection_node_idx = -1;
        connection_other_tree_id = -1;
        connection_other_node_idx = -1;
    }

    void reset_device_variables() {
        reset_device_variables_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
    }

    __device__ __forceinline__ void reset_to_unwritten_state(volatile float *buffer, int size, int tid) {
        if (tid == 0) {
            for (int i = 0; i < size; i++) {
                buffer[i] = UNWRITTEN_VAL;
            }
        }
        __syncthreads();
    }

    // cpRRTC motion generation / projection wrapper
    // Generic robot: straight-line motion만 생성하고 projection은 하지 않는다.
    // FfwSg2: straight-line motion 생성 후 analytic-Jacobian ParallelProject 수행.

    template <typename Robot>
    __device__ __forceinline__ bool cprrtc_project_motion(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid    
    ) {
        static constexpr auto dim = Robot::dimension;

        static_assert(
            dim <= MAX_ROBOT_DIM,
            "Robot dimension exceeds cpRRTC motion segment buffer"
        );

        // q0 = 시작 configuration
        if (tid < dim) {
            motion_segment[tid] = q_start[tid];
        }

        // waypoint 하나당 CUDA thread 4개 사용
        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        if (waypoint <= d_settings.granularity) {
            for (int j = lane; j < dim; j += 4) {
                motion_segment[waypoint * dim + j] = q_start[j] + static_cast<float>(waypoint) * q_step[j];
            }
        }

        __syncthreads();

        // 일반 robot은 constraint projection 없음
        return true;
    }

    // ffw-sg2 specializatoin
    template <>
    __device__ __forceinline__ bool cprrtc_project_motion<ppln::robots::FfwSg2>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid    
    ) {
        static constexpr auto dim =ppln::robots::FfwSg2::dimension;

        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        // ξ[0] = q_start
        if (tid < dim) {
            motion_segment[tid] = q_start[tid];
        }

        // ξ[1] ... ξ[granularity] 생성
        if (waypoint <= d_settings.granularity) {
            for (int j = lane; j < dim; j += 4) {
                motion_segment[waypoint * dim + j] =q_start[j] +static_cast<float>(waypoint) * q_step[j];
            }
        }

        __syncthreads();

        // FFW SG2만 실제 analytic-Jacobian projection 수행
        return ppln::collision::ffw_sg2_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            d_settings.rigid_orientation,

            projection_valid,
            projection_prog,
            projection_success,

            d_settings.projection_max_iters,
            d_settings.projection_alpha,
            d_settings.projection_damping,
            d_settings.projection_task_tolerance,

            d_settings.projection_smoothness_threshold,
            d_settings.projection_smoothness_weight,
            d_settings.projection_smoothness,

            d_settings.projection_max_step,

            tid
        );
    }

    template <>
    __device__ __forceinline__ bool cprrtc_project_motion<ppln::robots::FfwSg2Mobility>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        static constexpr auto dim = ppln::robots::FfwSg2Mobility::dimension;

        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        if (tid < dim) {
            motion_segment[tid] = q_start[tid];
        }

        if (waypoint <= d_settings.granularity) {
            for (int j = lane; j < dim; j += 4) {
                motion_segment[waypoint * dim + j] =
                    q_start[j] + static_cast<float>(waypoint) * q_step[j];
            }
        }

        __syncthreads();

        if (d_settings.ffw_sg2_enable_com_constraint) {
            return ppln::collision::ffw_sg2_mobility_com_project_motion(
                motion_segment,
                motion_segment_next,
                d_settings.granularity,
                d_settings.ffw_sg2_support_margin_m,
                d_settings.ffw_sg2_object_mass_kg,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                d_settings.projection_smoothness_threshold,
                d_settings.projection_smoothness_weight,
                d_settings.projection_smoothness,
                d_settings.projection_max_step,
                tid
            );
        }

        return ppln::collision::ffw_sg2_mobility_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            projection_valid,
            projection_prog,
            projection_success,
            d_settings.projection_max_iters,
            d_settings.projection_alpha,
            d_settings.projection_damping,
            d_settings.projection_task_tolerance,
            d_settings.projection_smoothness_threshold,
            d_settings.projection_smoothness_weight,
            d_settings.projection_smoothness,
            d_settings.projection_max_step,
            tid
        );
    }

    template <>
    __device__ __forceinline__ bool cprrtc_project_motion<ppln::robots::G1>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        static constexpr int dim = ppln::robots::G1::dimension;
        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        if (tid < dim) {
            motion_segment[tid] = q_start[tid];
        }
        if (waypoint <= d_settings.granularity) {
            for (int joint = lane; joint < dim; joint += 4) {
                motion_segment[waypoint * dim + joint] =
                    q_start[joint] + static_cast<float>(waypoint) * q_step[joint];
            }
        }
        __syncthreads();

        return ppln::collision::g1_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            d_settings.g1_constraints,
            projection_valid,
            projection_prog,
            projection_success,
            d_settings.projection_max_iters,
            d_settings.projection_alpha,
            d_settings.beta,
            d_settings.gamma,
            d_settings.projection_damping,
            d_settings.projection_task_tolerance,
            d_settings.projection_smoothness_threshold,
            d_settings.projection_smoothness_weight,
            d_settings.projection_smoothness,
            d_settings.projection_max_step,
            tid
        );
    }

    template <typename Robot>
    __device__ __forceinline__ float cprrtc_config_distance_from_volatile(
        volatile const float *q_a,
        const float *q_b
    ) {
        float result = 0.0f;

        #pragma unroll
        for (int joint = 0; joint < Robot::dimension; joint++) {
            const float weight =
                cprrtc_joint_distance_weight<Robot>(joint);
            const float weighted_diff =
                weight * (q_a[joint] - q_b[joint]);
            result += weighted_diff * weighted_diff;
        }

        return sqrtf(result);
    }

    template <typename Robot>
    __device__ __noinline__ bool cprrtc_project_prebuilt_motion(
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        int waypoint_count,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid,
        bool use_smoothness,
        bool return_when_success = true
    ) {
        if (waypoint_count <= 0) {
            if (tid == 0) {
                projection_prog[0] = 0;
                projection_success[0] = 1;
                projection_valid[0] = 1;
            }
            __syncthreads();
            return true;
        }

        if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
            return ppln::collision::ffw_sg2_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                d_settings.rigid_orientation,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                d_settings.projection_smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
            if (d_settings.ffw_sg2_enable_com_constraint) {
                return ppln::collision::ffw_sg2_mobility_com_project_motion(
                    motion_segment,
                    motion_segment_next,
                    waypoint_count,
                    d_settings.ffw_sg2_support_margin_m,
                    d_settings.ffw_sg2_object_mass_kg,
                    projection_valid,
                    projection_prog,
                    projection_success,
                    d_settings.projection_max_iters,
                    d_settings.projection_alpha,
                    d_settings.projection_damping,
                    d_settings.projection_task_tolerance,
                    d_settings.projection_smoothness_threshold,
                    d_settings.projection_smoothness_weight,
                    use_smoothness,
                    d_settings.projection_max_step,
                    tid,
                    return_when_success
                );
            }

            return ppln::collision::ffw_sg2_mobility_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                d_settings.projection_smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else if constexpr (std::is_same_v<Robot, robots::G1>) {
            return ppln::collision::g1_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                d_settings.g1_constraints,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.beta,
                d_settings.gamma,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                d_settings.projection_smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else {
            if (tid == 0) {
                projection_prog[0] = waypoint_count;
                projection_success[0] = 1;
                projection_valid[0] = 1;
            }
            __syncthreads();
            return true;
        }
    }

    template <typename Robot>
    __device__ __noinline__ bool
    cprrtc_project_concon_node_anchors(
        volatile const float *q_start,
        const float *node_nominal_targets,
        int edge_count,
        volatile float *node_motion,
        volatile float *node_motion_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        static constexpr auto dim = Robot::dimension;

        static_assert(
            dim <= MAX_ROBOT_DIM,
            "Robot dimension exceeds cpRRTC node motion buffer"
        );

        if (edge_count <= 0) {
            if (tid == 0) {
                projection_prog[0] = 0;
                projection_success[0] = 0;
                projection_valid[0] = 1;
            }
            __syncthreads();
            return false;
        }

        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        if (tid < dim) {
            node_motion[tid] = q_start[tid];
        }

        if (waypoint <= edge_count) {
            for (int joint = lane; joint < dim; joint += 4) {
                node_motion[waypoint * dim + joint] =
                    node_nominal_targets[
                        (waypoint - 1) * MAX_ROBOT_DIM + joint
                    ];
            }
        }
        __syncthreads();

        return cprrtc_project_prebuilt_motion<Robot>(
            node_motion,
            node_motion_next,
            edge_count,
            projection_valid,
            projection_prog,
            projection_success,
            tid,
            false
        );
    }

    template <typename Robot>
    __device__ __noinline__ void
    cprrtc_project_concon_edge_segments_from_node_anchors(
        int edge_count,
        volatile const float *node_anchors,
        volatile float *edge_motion_segments,
        volatile float *edge_motion_segment_next,
        volatile unsigned char *edge_projection_valid,
        volatile int *edge_projection_prog,
        volatile unsigned int *edge_projection_success,
        volatile int *first_projection_failure_edge,
        int tid
    ) {
        static constexpr auto dim = Robot::dimension;
        const int edge_slot = tid / CONCON_COLLISION_THREADS_PER_EDGE;
        const int edge_tid =
            tid - edge_slot * CONCON_COLLISION_THREADS_PER_EDGE;

        if (edge_count <= 0) {
            if (tid == 0) {
                first_projection_failure_edge[0] = 0;
            }
            __syncthreads();
            return;
        }

        const int source_edge_slot =
            edge_slot < edge_count ? edge_slot : edge_count - 1;
        volatile float *edge_motion =
            &edge_motion_segments[edge_slot * CONCON_MOTION_SEGMENT_STRIDE];
        volatile float *edge_motion_next =
            &edge_motion_segment_next[
                edge_slot * CONCON_MOTION_SEGMENT_STRIDE
            ];
        volatile unsigned char *edge_valid =
            &edge_projection_valid[
                edge_slot * CONCON_PROJECTION_STATE_STRIDE
            ];
        volatile int *edge_prog = &edge_projection_prog[edge_slot];
        volatile unsigned int *edge_success =
            &edge_projection_success[edge_slot];

        const int edge_segment_values =
            (d_settings.granularity + 1) * dim;
        for (
            int value = edge_tid;
            value < edge_segment_values;
            value += CONCON_COLLISION_THREADS_PER_EDGE
        ) {
            const int local_waypoint = value / dim;
            const int joint = value - local_waypoint * dim;
            const float alpha =
                static_cast<float>(local_waypoint) /
                static_cast<float>(d_settings.granularity);
            const float q0 =
                node_anchors[source_edge_slot * dim + joint];
            const float q1 =
                node_anchors[(source_edge_slot + 1) * dim + joint];

            edge_motion[local_waypoint * dim + joint] =
                q0 + alpha * (q1 - q0);
        }
        __syncthreads();

        const int internal_waypoint_count =
            d_settings.granularity > 0 ? d_settings.granularity - 1 : 0;
        cprrtc_project_prebuilt_motion<Robot>(
            edge_motion,
            edge_motion_next,
            internal_waypoint_count,
            edge_valid,
            edge_prog,
            edge_success,
            edge_tid,
            d_settings.projection_smoothness,
            false
        );
        __syncthreads();

        if (edge_tid < dim) {
            edge_motion[edge_tid] =
                node_anchors[source_edge_slot * dim + edge_tid];
            edge_motion[
                d_settings.granularity * dim + edge_tid
            ] =
                node_anchors[
                    (source_edge_slot + 1) * dim + edge_tid
                ];
        }
        __syncthreads();

        if (tid == 0) {
            first_projection_failure_edge[0] = edge_count;
        }
        __syncthreads();

        if (
            edge_slot < edge_count &&
            edge_tid == 0 &&
            edge_success[0] == 0u
        ) {
            atomicMin(
                (int *)first_projection_failure_edge,
                edge_slot
            );
        }
        __syncthreads();
    }

    template <typename Robot>
    __device__ __forceinline__ bool cprrtc_store_tangent_basis(
        const float *q,
        float *tree_tangent_bases,
        int node_idx
    ) {
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            if (tree_tangent_bases == nullptr) {
                return false;
            }

            constexpr int basis_size = TangentSpaceTraits<Robot>::basis_size;
            float basis[basis_size];

            // 현재 node q에서 Jacobian을 새로 계산하고 tangent basis까지 생성
            bool basis_ok = false;
            if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
                basis_ok = ppln::collision::ffw_sg2_tangent_basis(
                    q,
                    d_settings.rigid_orientation,
                    basis
                );
            } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
                basis_ok = ppln::collision::ffw_sg2_mobility_tangent_basis(
                    q,
                    basis
                );
            } else if constexpr (std::is_same_v<Robot, robots::G1>) {
                basis_ok = ppln::collision::g1_tangent_basis(
                    q,
                    d_settings.g1_constraints,
                    basis
                );
            }
            if (!basis_ok) {
                return false;
            }

            float *dst = &tree_tangent_bases[node_idx * basis_size];

            for (int i = 0; i < basis_size; i++) {
                dst[i] = basis[i];
            }
        }

        return true;
    }

    template <typename Robot>
    __device__ __forceinline__ void cprrtc_sample_tangent_config(
        float *tree_nodes,
        float *ts_bases,
        int ts_root_node_idx,
        int selected_ts_id,
        float *ts_coeff,
        float alpha_fraction,
        float *ts_tangent_dir,
        float *sdata,
        float *sampled_config,
        int tid
    )
    {
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            static constexpr auto dim = Robot::dimension;
            static constexpr int basis_stride =
                TangentSpaceTraits<Robot>::max_tangent_dim;
            static constexpr int basis_size =
                TangentSpaceTraits<Robot>::basis_size;
            const int active_tangent_dim = cprrtc_active_tangent_dim<Robot>();

            // 선택된 Tangent Space의 root configuration
            const float *base_q =&tree_nodes[ts_root_node_idx * dim];

            // 선택된 Tangent Space의 tangent basis
            const float *basis = &ts_bases[selected_ts_id * basis_size];
            float alpha_limit = FLT_MAX;

            if (tid < dim) {
                // tangent basis들의 linear combination dir = B * c. B는 선택된 TS의 tangent basis 행렬. c는 각 basis vector를 얼마나 섞을지 나타내는 계수 벡터
                float dir = 0.0f;

                for (int k = 0; k < active_tangent_dim; k++) {
                    dir += basis[tid * basis_stride + k] * ts_coeff[k];
                }

                ts_tangent_dir[tid] = dir;

                // joint limit 안에서 최대 이동 가능 거리 계산
                const float lo = Robot::get_s_a(tid); // 해당 차원의 최솟값
                const float hi = lo + Robot::get_s_m(tid); // 해당 차원의 max값

                if (dir > 1.0e-8f) {
                    alpha_limit =(hi - base_q[tid])/ dir;
                }
                else if (dir < -1.0e-8f) {
                    alpha_limit =(lo - base_q[tid])/ dir;
                }

                alpha_limit =fmaxf(alpha_limit,0.0f);
            }

            // 각 joint가 허용하는 alpha
            sdata[tid] =tid < dim? alpha_limit: FLT_MAX;

            __syncthreads();

            // 모든 joint 중 가장 작은 alpha_limit 찾기
            for (unsigned int s =blockDim.x / 2; s > 0; s >>= 1) {
                const float lhs =sdata[tid];
                float rhs =FLT_MAX;

                if (tid < s) {
                    rhs =sdata[tid + s];
                }

                __syncthreads();

                if (tid < s) {
                    sdata[tid] =fminf(lhs,rhs);
                }

                __syncthreads();
            }

            // 실제 이동거리
            const float alpha =alpha_fraction *sdata[0];

            // 최종 q_rand
            // q_rand = q_TS_root + alpha * tangent_direction
            if (tid < dim) {

                sampled_config[tid] =base_q[tid]+alpha *ts_tangent_dir[tid];
            }

            __syncthreads();
        }
    }

    template <typename Robot>
    __device__ __forceinline__ float cprrtc_constraint_error_norm(const float *q)
    {
        if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
            float h[FFW_SG2_MAX_RESIDUAL_DIM];

            // 현재 configuration q의 constraint residual h(q) 계산
            ppln::collision::ffw_sg2_constraint_residual(q,d_settings.rigid_orientation,h);

            // 현재 constraint의 residual dimension
            const int residual_dim =ppln::collision::ffw_sg2_constraint_dim(d_settings.rigid_orientation);

            // EM = ||h(q)||
            return ppln::collision::ffw_sg2_residual_norm(h,residual_dim);
        } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
            float h[FFW_SG2_MOBILITY_RESIDUAL_DIM];
            ppln::collision::ffw_sg2_mobility_constraint_residual(q, h);
            return ppln::collision::ffw_sg2_mobility_residual_norm(h);
        } else if constexpr (std::is_same_v<Robot, robots::G1>) {
            return ppln::collision::g1_equality_residual_norm(
                q,
                d_settings.g1_constraints
            );
        }

        return 0.0f;
    }

    template <typename Robot>
    __global__ void init_root_ts_banks(
        float **nodes,
        int **ts_root_node_idx,
        float **ts_bases,
        int **ts_ready,
        int **node_ts_id,
        float **node_ts_q,
        int **ts_node_count,
        int **ts_lane_head,
        int **node_next_in_ts,
        int start_count,
        int goal_count
    )
    {
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            const int global_idx = blockIdx.x;

            // 지금 처리하고 있는 것이 start tree인지 goal tree인지 결정
            const int tree =(global_idx < start_count) ? 0 : 1;

            // 해당 tree 안에서의 TS index
            const int ts_idx =(tree == 0)? global_idx: global_idx - start_count;

            // 해당 tree의 initial node 개수
            const int count =(tree == 0)? start_count: goal_count;

            if (threadIdx.x == 0 &&ts_idx < count) {
                // 처음에는 initial node 하나가 TS 하나의 root
                const int node_idx = ts_idx;

                // 이 TS가 어느 tree node에서 만들어졌는지 저장
                ts_root_node_idx[tree][ts_idx] =node_idx;

                // root q에서 tangent basis 계산
                const bool basis_ok =cprrtc_store_tangent_basis<Robot>(&nodes[tree][node_idx * Robot::dimension],ts_bases[tree],ts_idx);

                if (basis_ok) {

                    // 이 node는 방금 만든 TS에 소속
                    node_ts_id[tree][node_idx] =ts_idx;

                    // root에서는 nominal TS q == 실제 tree q
                    for (int j = 0; j < Robot::dimension; j++) {
                        node_ts_q[tree][node_idx * Robot::dimension + j] =nodes[tree][node_idx * Robot::dimension + j];
                    }

                    // 최초 root node를 이 TS의 thread 0 목록에 등록
                    ts_node_count[tree][ts_idx] = 1;
                    ts_lane_head[tree][ts_idx * MAX_THREADS_PER_BLOCK] =node_idx;
                    node_next_in_ts[tree][node_idx] = -1;
                }

                // 위 정보들이 global memory에 기록된 후 ready를 켜기 위해 사용
                __threadfence();

                if (basis_ok) {
                    ts_ready[tree][ts_idx] = 1;
                }
            }
        }
    }

    template <typename Robot>
    __device__ __forceinline__ float cprrtc_shared_config_distance(
        volatile const float *q_a,
        volatile const float *q_b,
        float *sdata,
        int tid
    )
    {
        static constexpr auto dim = Robot::dimension;
        float value = 0.0f;

        // 각 thread가 joint dimension 하나의 거리 제곱을 계산
        if (tid < dim) {
            const float weight =
                cprrtc_joint_distance_weight<Robot>(tid);

            const float weighted_diff =
                weight * (q_a[tid] - q_b[tid]);

            value = weighted_diff * weighted_diff;
        }

        // dim보다 큰 thread는 0
        sdata[tid] = value;

        __syncthreads();

        // block reduction
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < static_cast<int>(s)) {
                sdata[tid] += sdata[tid + s];
            }

            __syncthreads();
        }

        return sqrtf(sdata[0]);
    }

    template <typename Robot>
    __device__ __forceinline__ bool cprrtc_detailed_env_collision_check(
        volatile float *sphere_pos,
        volatile int *link_CC,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        // 다른 robot은 기존 방식
        return ppln::collision::env_collision_check<Robot>(
            sphere_pos,
            link_CC,
            env,
            tid
        );
    }


    template <>
    __device__ __forceinline__ bool cprrtc_detailed_env_collision_check<ppln::robots::FfwSg2>(
        volatile float *sphere_pos,
        volatile int *link_CC,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_env_collision_check_early(
                sphere_pos,
                link_CC,
                env,
                tid,
                motion_cc_flag
            );
    }

    template <>
    __device__ __forceinline__ bool cprrtc_detailed_env_collision_check<ppln::robots::FfwSg2Mobility>(
        volatile float *sphere_pos,
        volatile int *link_CC,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_mobility_env_collision_check_early(
                sphere_pos,
                link_CC,
                env,
                tid,
                motion_cc_flag
            );
    }

    template <typename Robot>
    __device__ __forceinline__ bool cprrtc_detailed_self_collision_check(
        volatile float *sphere_pos,
        volatile int *link_CC,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::self_collision_check<Robot>(
                sphere_pos,
                link_CC,
                tid
            );
    }


    template <>
    __device__ __forceinline__ bool cprrtc_detailed_self_collision_check<ppln::robots::FfwSg2>(
        volatile float *sphere_pos,
        volatile int *link_CC,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_self_collision_check_early(
                sphere_pos,
                link_CC,
                tid,
                motion_cc_flag
            );
    }

    template <>
    __device__ __forceinline__ bool cprrtc_detailed_self_collision_check<ppln::robots::FfwSg2Mobility>(
        volatile float *sphere_pos,
        volatile int *link_CC,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_mobility_self_collision_check_early(
                sphere_pos,
                link_CC,
                tid,
                motion_cc_flag
            );
    }

    template <typename Robot>
    __device__ __forceinline__ bool cprrtc_attached_object_collision_check_approx(
        const float *q,
        volatile float *sphere_pos_approx,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return true;
    }

    template <>
    __device__ __forceinline__ bool
    cprrtc_attached_object_collision_check_approx<ppln::robots::FfwSg2Mobility>(
        const float *q,
        volatile float *sphere_pos_approx,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_mobility_attached_object_collision_check_approx(
                q,
                sphere_pos_approx,
                env,
                tid,
                motion_cc_flag
            );
    }

    template <typename Robot>
    __device__ __forceinline__ bool cprrtc_attached_object_collision_check(
        const float *q,
        volatile float *sphere_pos,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return true;
    }

    template <>
    __device__ __forceinline__ bool
    cprrtc_attached_object_collision_check<ppln::robots::FfwSg2Mobility>(
        const float *q,
        volatile float *sphere_pos,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_mobility_attached_object_collision_check(
                q,
                sphere_pos,
                env,
                tid,
                motion_cc_flag
            );
    }

    template <typename Robot>
    __device__ __forceinline__ void
    cprrtc_check_projected_edges_collision_parallel(
        int edge_count,
        volatile float *edge_motion_segments,
        volatile float *sphere_pos_scratch,
        volatile float *sphere_pos_approx_scratch,
        volatile int *link_cc_scratch,
        float *transform_scratch,
        ppln::collision::Environment<float> *env,
        volatile unsigned int *edge_cc_result,
        bool *edge_run_detailed_env_check,
        bool *edge_run_self_collision_check,
        bool *edge_run_detailed_self_check,
        bool *any_detailed_env_check,
        bool *any_detailed_self_check,
        volatile int *first_collision_edge,
        int tid
    ) {
        using Collision = robots::CollisionTraits<Robot>;
        static constexpr int dim = Robot::dimension;
        static constexpr int fine_scratch_stride =
            Collision::fine_sphere_count * Collision::batch_size * 3;
        static constexpr int approx_scratch_stride =
            Collision::approximate_sphere_count * Collision::batch_size * 3;
        static constexpr int link_scratch_stride =
            Collision::joint_flag_stride * Collision::batch_size;
        static constexpr int transform_scratch_stride =
            Collision::batch_size * Collision::transform_slots * 16;

        if (edge_count <= 0) {
            return;
        }

        const int edge_slot = tid / CONCON_COLLISION_THREADS_PER_EDGE;
        const int edge_tid = tid - edge_slot * CONCON_COLLISION_THREADS_PER_EDGE;
        const int safe_edge_slot =
            edge_slot < edge_count ? edge_slot : edge_count - 1;
        const int waypoint = edge_tid / 4 + 1;

        volatile float *edge_motion =
            &edge_motion_segments[
                safe_edge_slot * CONCON_MOTION_SEGMENT_STRIDE
            ];
        volatile float *edge_sphere_pos =
            &sphere_pos_scratch[edge_slot * fine_scratch_stride];
        volatile float *edge_sphere_pos_approx =
            &sphere_pos_approx_scratch[edge_slot * approx_scratch_stride];
        volatile int *edge_link_cc =
            &link_cc_scratch[edge_slot * link_scratch_stride];
        float *edge_transform =
            &transform_scratch[edge_slot * transform_scratch_stride];

        float interp_cfg[dim];
        #pragma unroll
        for (int joint = 0; joint < dim; joint++) {
            interp_cfg[joint] = edge_motion[waypoint * dim + joint];
        }

        if (tid < MAX_PARALLEL_CONCON_EDGES) {
            edge_cc_result[tid] = 0u;
            edge_run_detailed_env_check[tid] = false;
            edge_run_self_collision_check[tid] = false;
            edge_run_detailed_self_check[tid] = false;
        }
        if (tid == 0) {
            first_collision_edge[0] = edge_count;
            any_detailed_env_check[0] = false;
            any_detailed_self_check[0] = false;
        }
        __syncthreads();

        for (int r = edge_tid; r < link_scratch_stride;
             r += CONCON_COLLISION_THREADS_PER_EDGE) {
            edge_link_cc[r] = 0;
        }
        __syncthreads();

        ppln::collision::fk_approx<Robot>(
            interp_cfg,
            edge_sphere_pos_approx,
            edge_transform,
            edge_tid
        );
        __syncthreads();

        if (edge_slot < edge_count && edge_slot < first_collision_edge[0]) {
            const bool env_collision_approx =
                not ppln::collision::env_collision_check_approx<Robot>(
                    edge_sphere_pos_approx,
                    edge_link_cc,
                    env,
                    edge_tid
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                env_collision_approx ? 1u : 0u
            );

            const bool attached_object_collision_approx =
                not cprrtc_attached_object_collision_check_approx<Robot>(
                    interp_cfg,
                    edge_sphere_pos_approx,
                    env,
                    edge_tid,
                    &edge_cc_result[edge_slot]
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                attached_object_collision_approx ? 1u : 0u
            );
        }
        __syncthreads();

        if (tid < edge_count) {
            edge_run_detailed_env_check[tid] = edge_cc_result[tid] != 0u;
            if (edge_run_detailed_env_check[tid]) {
                edge_cc_result[tid] = 0u;
            }
        }
        __syncthreads();

        if (tid == 0) {
            bool run_any = false;
            for (int edge = 0; edge < edge_count; edge++) {
                run_any = run_any || edge_run_detailed_env_check[edge];
            }
            any_detailed_env_check[0] = run_any;
        }
        __syncthreads();

        if (any_detailed_env_check[0]) {
            ppln::collision::fk<Robot>(
                interp_cfg,
                edge_sphere_pos,
                edge_transform,
                edge_tid
            );
        }
        __syncthreads();

        if (
            edge_slot < edge_count &&
            edge_slot < first_collision_edge[0] &&
            edge_run_detailed_env_check[edge_slot]
        ) {
            const bool env_collision =
                not cprrtc_detailed_env_collision_check<Robot>(
                    edge_sphere_pos,
                    edge_link_cc,
                    env,
                    edge_tid,
                    &edge_cc_result[edge_slot]
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                env_collision ? 1u : 0u
            );

            const bool attached_object_collision =
                not cprrtc_attached_object_collision_check<Robot>(
                    interp_cfg,
                    edge_sphere_pos,
                    env,
                    edge_tid,
                    &edge_cc_result[edge_slot]
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                attached_object_collision ? 1u : 0u
            );
        }
        __syncthreads();

        if (
            tid < edge_count &&
            edge_run_detailed_env_check[tid] &&
            edge_cc_result[tid] != 0u
        ) {
            atomicMin((int *)&first_collision_edge[0], tid);
        }
        __syncthreads();

        for (int r = edge_tid; r < link_scratch_stride;
             r += CONCON_COLLISION_THREADS_PER_EDGE) {
            edge_link_cc[r] = 0;
        }
        __syncthreads();

        if (tid < edge_count) {
            edge_run_self_collision_check[tid] =
                edge_cc_result[tid] == 0u && tid < first_collision_edge[0];
        }
        __syncthreads();

        if (
            edge_slot < edge_count &&
            edge_slot < first_collision_edge[0] &&
            edge_run_self_collision_check[edge_slot]
        ) {
            const bool self_collision_approx =
                not ppln::collision::self_collision_check_approx<Robot>(
                    edge_sphere_pos_approx,
                    edge_link_cc,
                    edge_tid
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                self_collision_approx ? 1u : 0u
            );
        }
        __syncthreads();

        if (tid < edge_count) {
            edge_run_detailed_self_check[tid] =
                edge_run_self_collision_check[tid] &&
                edge_cc_result[tid] != 0u;
            if (edge_run_detailed_self_check[tid]) {
                edge_cc_result[tid] = 0u;
            }
        }
        __syncthreads();

        if (tid == 0) {
            bool run_any = false;
            for (int edge = 0; edge < edge_count; edge++) {
                run_any = run_any || edge_run_detailed_self_check[edge];
            }
            any_detailed_self_check[0] = run_any;
        }
        __syncthreads();

        if (any_detailed_self_check[0]) {
            ppln::collision::fk<Robot>(
                interp_cfg,
                edge_sphere_pos,
                edge_transform,
                edge_tid
            );
        }
        __syncthreads();

        if (
            edge_slot < edge_count &&
            edge_slot < first_collision_edge[0] &&
            edge_run_detailed_self_check[edge_slot]
        ) {
            const bool self_collision =
                not cprrtc_detailed_self_collision_check<Robot>(
                    edge_sphere_pos,
                    edge_link_cc,
                    edge_tid,
                    &edge_cc_result[edge_slot]
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                self_collision ? 1u : 0u
            );
        }
        __syncthreads();

        if (
            tid < edge_count &&
            edge_run_detailed_self_check[tid] &&
            edge_cc_result[tid] != 0u
        ) {
            atomicMin((int *)&first_collision_edge[0], tid);
        }
        __syncthreads();
    }

    __device__ __forceinline__
    int cprrtc_reserve_slot(volatile int *counter,int capacity){
        int current =atomicAdd((int *)counter,0);

        while (current < capacity) {
            const int observed =atomicCAS((int *)counter,current,current + 1);

            if (observed == current) {
                return current;
            }

            current = observed;
        }

        return -1;
    }

    __device__ __forceinline__
    void cprrtc_register_node_in_ts(
        int node_idx,
        int ts_id,
        int *ts_node_count,
        int *ts_lane_head,
        int *node_next_in_ts
    ) {
        if (ts_id < 0) {
            return;
        }

        // TS 안에서 등록된 순서에 따라 64개 thread 목록에 고르게 배정
        const int ordinal =atomicAdd(&ts_node_count[ts_id],1);
        const int lane =ordinal % blockDim.x;
        const int head_slot =ts_id * MAX_THREADS_PER_BLOCK + lane;

        int old_head =atomicAdd(&ts_lane_head[head_slot],0);

        while (true) {
            // next를 먼저 기록한 뒤 새 head를 공개한다.
            node_next_in_ts[node_idx] =old_head;
            __threadfence();

            const int observed =atomicCAS(
                &ts_lane_head[head_slot],
                old_head,
                node_idx
            );

            if (observed == old_head) {
                break;
            }

            old_head =observed;
        }
    }

    template <typename Robot>
    __device__ __forceinline__ float cprrtc_project_target_direction_to_tangent(
        const float *q_current,
        const float *q_target,
        const float *basis,
        float *ts_coeff,
        float *projected_dir,
        float *sdata,
        int tid
    ) {
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            static constexpr int dim = Robot::dimension;
            static constexpr int basis_stride =
                TangentSpaceTraits<Robot>::max_tangent_dim;

            const int active_tangent_dim = cprrtc_active_tangent_dim<Robot>();

            // 1. q_current -> q_target 방향을 Tangent basis 좌표계의 coefficient로 변환
            // ts_coeff = B^T * (q_target - q_current)
            if (tid < active_tangent_dim) {
                float coeff = 0.0f;

                for (int j = 0; j < dim; j++) {
                    const float target_vector =q_target[j] - q_current[j];
                    coeff += basis[j * basis_stride + tid] * target_vector;
                }

                ts_coeff[tid] = coeff;
            }

            __syncthreads();

            // 2. coefficient를 다시 joint-space 방향으로 변환
            // projected_dir = B * ts_coeff = B * B^T * (q_target - q_current)
            float projected_component = 0.0f;

            if (tid < dim) {

                for (int k = 0; k < active_tangent_dim; k++) {
                    projected_component += basis[tid * basis_stride + k] * ts_coeff[k];
                }

                projected_dir[tid] =projected_component;
            }

            // 3. projected direction의 norm 계산 준비
            if (tid < dim) {
                const float weight =
                    cprrtc_joint_distance_weight<Robot>(tid);

                const float weighted_component =
                    weight * projected_component;

                sdata[tid] =
                    weighted_component * weighted_component;
            }
            else {
                sdata[tid] = 0.0f;
            }

            __syncthreads();

            // block reduction
            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] +=sdata[tid + s];
                }

                __syncthreads();
            }

            const float projected_norm =sqrtf(sdata[0]);

            // 4. unit direction으로 normalize
            if (tid < dim) {

                if (projected_norm > 1.0e-8f) {
                    projected_dir[tid] /=projected_norm;
                }
                else {
                    projected_dir[tid] = 0.0f;
                }
            }

            __syncthreads();

            return projected_norm;
        }

        return 0.0f;
    }
        
    template <typename Robot, bool TraceTrees>
    __global__ void
    // __launch_bounds__(128, 8)
    rrtc(
        float **nodes,
        int **parents,
        int **node_ready,
        // Tangent Space membership
        int **node_ts_id,
        float **node_ts_q,

        // Tangent Space Bank
        int *ts_count,
        int **ts_root_node_idx,
        float **ts_bases,
        int **ts_ready,
        int **ts_node_count,
        int **ts_lane_head,
        int **node_next_in_ts,
        float **radii,
        HaltonState<Robot> *halton_states,
        curandState *rng_states,
        ppln::collision::Environment<float> *env,
        volatile float *concon_sphere_pos_scratch,
        volatile float *concon_sphere_pos_approx_scratch,
        volatile int *concon_link_cc_scratch,
        float *concon_transform_scratch
    )
    {
        static constexpr auto dim = Robot::dimension;
        using Collision = robots::CollisionTraits<Robot>;
        const int tid = threadIdx.x;
        const int bid = blockIdx.x; // 0 ... NUM_NEW_CONFIGS
        __shared__ int t_tree_id; // this tree
        __shared__ int o_tree_id; // the other tree
        __shared__ float config[dim];
        __shared__ float sdata[MAX_THREADS_PER_BLOCK];
        __shared__ int sindex[MAX_THREADS_PER_BLOCK];
        __shared__ float *t_nodes;
        __shared__ float *o_nodes;
        __shared__ int *t_parents;
        __shared__ int *o_parents;
        // 현재 선택된 tree의 Tangent Space 정보
        __shared__ int *t_node_ts_id;
        __shared__ float *t_node_ts_q;
        __shared__ int *t_ts_root_node_idx;
        __shared__ float *t_ts_bases;
        __shared__ int *t_ts_ready;
        __shared__ int *t_ts_node_count;
        __shared__ int *t_ts_lane_head;
        __shared__ int *t_node_next_in_ts;
        __shared__ int t_ts_count;
        // 이번 EXTEND iteration에서 선택한 Tangent Space
        __shared__ int selected_ts_id;
        __shared__ int selected_ts_root_idx;
        __shared__ int *t_node_ready; // 현재 TREE의 node 배열
        __shared__ int *o_node_ready;
        __shared__ int t_tree_size; // 현재 tree에 index가 할당된 node 수
        __shared__ float ts_coeff[MAX_TANGENT_DIM]; // Tangent basis들을 어떤 비율로 조합할지
        __shared__ float ts_alpha_fraction; // Tangent 방향으로 얼마나 이동할지
        __shared__ float ts_tangent_dir[MAX_ROBOT_DIM]; // 최종 15차원 Tangent 방향
        __shared__ float scale;
        // 실제 constraint manifold 위의 tree node
        __shared__ float *nearest_node;
        // 위 node에 대응하는 Tangent Space 위 nominal q
        __shared__ float *nearest_ts_node;
        // q_rand와 nearest_ts_node 사이 거리
        __shared__ float q_rand_dist;
        // q_rand - q_near_TS의 normalized direction
        __shared__ float extend_dir[dim];
        // ConCon 후보 검사에 임시로 사용할 nominal configuration
        __shared__ float concon_probe[dim];
        // 이번 EXTEND에서 생성 가능한 ConCon candidate 개수
        __shared__ int concon_count;
        // EM threshold를 만나서 종료했는지
        __shared__ bool concon_em_stop;
        // 실제 projection + collision까지 성공한 ConCon node 수
        __shared__ int concon_valid_count;
        // EM boundary에서 새로 생성할 TS 번호
        __shared__ int new_ts_id;
        // 새 Tangent Space basis 생성 성공 여부
        __shared__ bool new_ts_basis_ok;
        // 다음 새 node가 연결될 실제 tree parent
        __shared__ int concon_parent_idx;
        // 실제로 검사할 edge 개수
        // FFW-SG2에서는 concon_count, 다른 robot에서는 기존처럼 1
        __shared__ int extend_edge_count;
        __shared__ int index;
        __shared__ bool should_skip;
        // cpRRTC CONNECT state
        // target 도달 여부
        __shared__ bool connection_reached_shared;
        __shared__ unsigned int n_extensions;
        // CONNECT 시작 시 상대 tree에서 한 번 선택하고 끝까지 유지할 target
        __shared__ int connect_target_idx;
        __shared__ float *connect_target_node;

        // 새 TB-RRT CONNECT 상태
        __shared__ bool connect_failed;
        __shared__ bool connect_reached;
        // cpRRTC parallel projection shared memory
        __align__(16) __shared__ volatile float motion_segment[
            MAX_CONCON_NODE_ANCHORS * MAX_ROBOT_DIM
        ];
        __align__(16) __shared__ volatile float motion_segment_next[
            MAX_PARALLEL_CONCON_EDGES * CONCON_MOTION_SEGMENT_STRIDE
        ];
        __shared__ volatile unsigned char motion_projection_valid[
            MAX_CONCON_PROJECTION_STATES
        ];
        __shared__ volatile int motion_projection_prog[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ volatile unsigned int motion_projection_success[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ volatile int concon_first_projection_failure_edge[1];
        __align__(16) __shared__ volatile float concon_motion_segments[
            MAX_PARALLEL_CONCON_EDGES * CONCON_MOTION_SEGMENT_STRIDE
        ];
        __shared__ float concon_nominal_targets[
            MAX_PARALLEL_CONCON_EDGES * MAX_ROBOT_DIM
        ];
        __shared__ volatile unsigned int concon_edge_cc_result[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool concon_run_detailed_env_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool concon_run_self_collision_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool concon_run_detailed_self_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool concon_any_detailed_env_check;
        __shared__ bool concon_any_detailed_self_check;
        __shared__ volatile int concon_first_collision_edge[1];
        __shared__ int concon_projected_edge_count;
        int iter = 0;

        while (true) {
            if (tid == 0) {
                // printf("iter: %d\n", iter);
                // printf("tree size: %d\n", atomic_free_index[0]);
                iter++;
                if (iter > d_settings.max_iters) {
                    atomicCAS((int *)&solved, 0, -1);
                }

                // tree 선택. 더 작은 tree 선택
                if (d_settings.balance == 0 || iter == 1) {
                    t_tree_id = (bid < (d_settings.num_new_configs / 2))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1 && abs(atomic_free_index[0]-atomic_free_index[1]) < 1.5 * d_settings.num_new_configs) { // dynamic balance
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0]+atomic_free_index[1]);
                    float balance_factor = 1 - ratio;
                    t_tree_id = (bid < (d_settings.num_new_configs * balance_factor))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1) {
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0] + atomic_free_index[1]);
                    if (ratio < d_settings.tree_ratio) t_tree_id = 0;
                    else t_tree_id = 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 2) { // vamp balance
                    float ratio = abs(atomic_free_index[t_tree_id] - atomic_free_index[o_tree_id]) / (float) atomic_free_index[t_tree_id]; // |현재 tree 크기 - 반대 tree 크기| / 현재 tree 크기 (두 tree의 상대적인 크기 차이)
                    if (ratio < d_settings.tree_ratio) // tree size가 비슷한 경우에는 번갈아 확장. tree size 차이가 많이 날 경우에는 작은 tree 계속 확장
                    {
                        t_tree_id = 1 - t_tree_id;
                        o_tree_id = 1 - t_tree_id;
                    }
                }

                t_nodes = nodes[t_tree_id];
                o_nodes = nodes[o_tree_id];
                t_parents = parents[t_tree_id];
                o_parents = parents[o_tree_id];
                if constexpr (TangentSpaceTraits<Robot>::enabled) {
                    // 현재 확장할 tree에 속한 node들의 TS 정보
                    t_node_ts_id =node_ts_id[t_tree_id];
                    t_node_ts_q =node_ts_q[t_tree_id];
                    // 현재 확장할 tree의 TSBank
                    t_ts_root_node_idx =ts_root_node_idx[t_tree_id];
                    t_ts_bases =ts_bases[t_tree_id];
                    t_ts_ready =ts_ready[t_tree_id];
                    t_ts_node_count =ts_node_count[t_tree_id];
                    t_ts_lane_head =ts_lane_head[t_tree_id];
                    t_node_next_in_ts =node_next_in_ts[t_tree_id];
                    // 현재 tree에 존재하는 Tangent Space 개수
                    t_ts_count =ts_count[t_tree_id];
                }
                t_node_ready =node_ready[t_tree_id];
                o_node_ready = node_ready[o_tree_id];
                t_tree_size = min((int)atomic_free_index[t_tree_id], d_settings.max_samples);

                // FFW-SG2 → Tangent Space sampling. q_rand 생성에서 사용할 파라미터 생성
                if constexpr (TangentSpaceTraits<Robot>::enabled) {
                    float halton_sample[dim];
                    halton_next(halton_states[bid],halton_sample);

                    // 새 TSBank에서 이번 iteration이 사용할 TS 하나 선택
                    selected_ts_id = -1;
                    selected_ts_root_idx = -1;

                    if (t_ts_count > 0) { // TS 존재 확인
                        // 탐색을 시작할 TS 번호 결정
                        const int start_ts =(bid +iter * d_settings.num_new_configs)% t_ts_count; // TS를 랜덤 선택하는 것이 아닌 여러 block의 시작점을 분산시키는 계산

                        // 혹시 아직 생성 중인 TS가 있을 수 있으므로 ready인 TS를 찾는다.
                        for (int attempt = 0; attempt < t_ts_count; attempt++) {
                            const int candidate_ts =(start_ts + attempt)% t_ts_count;

                            // 완전히 생성된 TS만 사용
                            if (t_ts_ready[candidate_ts] != 0) {
                                selected_ts_id =candidate_ts;
                                selected_ts_root_idx =t_ts_root_node_idx[candidate_ts];

                                break;
                            }
                        }
                    }

                    // 2. 현재 constraint의 tangent dimension
                    const int active_tangent_dim = cprrtc_active_tangent_dim<Robot>();

                    // 3. Tangent Space 안에서 random direction 생성
                    float coeff_norm2 = 0.0f;

                    for (int k = 0; k < active_tangent_dim; k++) {
                        // Halton [0,1] → coefficient [-1,1]
                        const float coeff =2.0f *halton_sample[k]- 1.0f;
                        ts_coeff[k] =coeff;
                        coeff_norm2 +=coeff * coeff;
                    }

                    // 방향 크기를 1로 normalize
                    const float inv_coeff_norm =1.0f /fmaxf(sqrtf(coeff_norm2),1.0e-8f);

                    for (int k = 0; k < active_tangent_dim; k++) {
                        ts_coeff[k] *=inv_coeff_norm;
                    }

                    // 4. 그 방향으로 얼마나 갈지
                    ts_alpha_fraction =halton_sample[active_tangent_dim];
                }

                // 다른 Robot은 기존 ambient sampling 그대로
                else {
                    halton_next(halton_states[bid],(float *)config);
                    Robot::scale_cfg((float *)config);
                }

            }

            __syncthreads();

            if constexpr (TraceTrees) {
                if (tid == 0) {
                    should_skip = (solved != 0);
                }
                __syncthreads();
                if (should_skip) {
                    return;
                }
            }
            else {
                if (tid == 0) {
                    should_skip = (solved != 0);
                }
                __syncthreads();
                if (should_skip) {
                    return;
                }
            }

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                // 사용할 수 있는 Tangent Space를 찾지 못했으면 이번 EXTEND iteration을 버린다.
                if (selected_ts_id < 0) {
                    continue;
                }
            }

            // q_rand 생성 생성
            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                cprrtc_sample_tangent_config<Robot>(
                    t_nodes, 
                    t_ts_bases, // node별 basis가 아니라 TSBank
                    selected_ts_root_idx, // 선택한 TS root
                    selected_ts_id, // 선택한 TS ID
                    ts_coeff,
                    ts_alpha_fraction,
                    ts_tangent_dir,
                    sdata,
                    (float *)config,
                    tid
                );
            }

            // parallelized nearest neighbor search
            float local_min_dist = FLT_MAX;
            int local_near_idx = 0;
            float dist;

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                // selected TS에서 이 thread에 미리 배정된 node 목록만 검사한다.
                int node_idx =t_ts_lane_head[selected_ts_id * MAX_THREADS_PER_BLOCK + tid]; // 선택된 TS에서 현재 thread tid가 담당하는 첫 번째 노드 번호를 가져와라

                while (node_idx >= 0) { // thread 하나에서 진행하는 내용
                    if (t_node_ready[node_idx] != 0) { // 사용할 수 있는 node인지 확인
                        // 현재 node와 q_rand 사이의 거리 계산 (거리 제곱 반환)
                        const float candidate_dist =
                            cprrtc_sq_config_distance<Robot>(
                                (float *)&t_node_ts_q[node_idx * dim],
                                (float *)config
                            );
                        // 지금까지 본 노드 중 가장 가까우면 기록
                        if (candidate_dist < local_min_dist) {
                            local_min_dist =candidate_dist;
                            local_near_idx =node_idx;
                        }
                    }

                    node_idx =t_node_next_in_ts[node_idx]; // 다음 노드 검사
                }
            }
            else {
                // 다른 robot은 기존처럼 tree 전체 node를 thread들이 나눠 검사한다.
                const int size =t_tree_size;

                for (int i =tid; i < size; i +=blockDim.x) {
                    if (t_node_ready[i] == 0) {
                        continue;
                    }

                    const float candidate_dist =
                        cprrtc_sq_config_distance<Robot>(
                            (float *)&t_nodes[i * dim],
                            (float *)config
                        );

                    if (candidate_dist < local_min_dist) {
                        local_min_dist =candidate_dist;
                        local_near_idx =i;
                    }
                }
            }
            sdata[tid] = local_min_dist; // block 내 최솟값
            sindex[tid] = local_near_idx; // 그 최솟값의 원래 인덱스
            __syncthreads();

            // sdata 최솟값과 해당 index를 병렬로 찾는 reduction 코드. thread들이 각각 구한 가장 작은 node들 중 가장 작은 node를 구하는 과정 (q_near 선택 과정)
            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    if (sdata[tid + s] < sdata[tid]) {
                        sdata[tid]  = sdata[tid + s];
                        sindex[tid] = sindex[tid + s];
                    }
                }
                __syncthreads();
            }

            // NN 결과를 바탕으로 거리, 노드 포인터 및 확장 가능 여부 설정
            if (tid == 0) {
                // 같은 TS에서 NN 후보를 하나도 찾지 못했는지 확인
                const bool no_nn_candidate =(sdata[0] == FLT_MAX);

                if (no_nn_candidate) {
                    q_rand_dist = 0.0f;
                    // 뒤의 q_steer/projection을 수행하지 않도록 함
                    should_skip = true;
                }
                else {
                    // q_rand와 nominal q_near_TS 사이 거리
                    q_rand_dist =sqrtf(sdata[0]);

                    // 실제 tree node 가져오기 (q_near)
                    nearest_node =&t_nodes[sindex[0] * dim];

                    if constexpr (TangentSpaceTraits<Robot>::enabled) {
                        // 같은 node의 Tangent Space상 nominal 위치
                        nearest_ts_node =&t_node_ts_q[sindex[0] * dim];
                    }

                    // 확장 방향을 정규화할 수 있는지 확인
                    const bool zero_direction =q_rand_dist <= 1.0e-8f; // 방향 벡터가 0인지 확인

                    // 기존 single q_steer를 사용하는 다른 로봇에서만 scale 계산
                    if constexpr (!TangentSpaceTraits<Robot>::enabled) {
                            if (!zero_direction) {
                                scale = min(1.0f, d_settings.range / q_rand_dist);
                            }
                            else {
                                scale = 0.0f;
                            }
                        }
                    // 기존 Dynamic Domain 설정 그대로 사용
                    const bool outside_dynamic_domain =d_settings.dynamic_domain&&radii[t_tree_id][sindex[0]]<q_rand_dist;
                    should_skip =zero_direction||outside_dynamic_domain;
                }
            }
            __syncthreads();

            if (should_skip) {
                continue;
            }
            __syncthreads();

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                if (tid < dim) {
                    // q_rand 자체를 목표점으로 사용하지 않는다. q_rand - q_near_TS에서 방향만 얻는다.
                    extend_dir[tid] =(config[tid]-nearest_ts_node[tid])/q_rand_dist;
                }
            }

            // 시작 전 상태 초기화
            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                if (tid == 0) {
                    concon_count = 0;
                    concon_em_stop = false;
                }
            }

            __syncthreads();

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                for (int step = 1; step <= d_settings.max_concon_nodes; step++) {
                    // Tangent Space 위의 nominal candidate 생성
                    // q_step = q_near_TS + step * range * extend_dir
                    if (tid < dim) {
                        concon_probe[tid] =nearest_ts_node[tid]+((float)step*d_settings.range*extend_dir[tid]);
                    }
                    __syncthreads();

                    // EM 계산은 thread 0 하나만 수행
                    if (tid == 0) {
                        const float em_error =cprrtc_constraint_error_norm<Robot>(concon_probe); // constraint residual 검사

                        // 먼저 현재 candidate를 포함한다.
                        concon_count = step;

                        // 그 다음 threshold 검사
                        // 즉 threshold를 처음 초과한 node도 포함된다.
                        if (em_error >d_settings.em_threshold
                        ) {
                            concon_em_stop = true;
                        }
                    }

                    __syncthreads();

                    // EM threshold를 넘으면 block 전체가 loop 종료
                    if (concon_em_stop) {
                        break;
                    }
                }
            }

            __syncthreads();

            // ConCon 실제 edge validation 준비
            if (tid == 0) {
                // 아직 실제로 성공한 candidate 없음
                concon_valid_count = 0;
                // 첫 edge의 실제 parent는 NN node
                concon_parent_idx = sindex[0];

                if constexpr (TangentSpaceTraits<Robot>::enabled) {
                    // FFW-SG2는 앞에서 EM으로 구한 candidate 전부 검사
                    extend_edge_count = concon_count;
                }
                else {
                    // 다른 robot은 기존 EXTEND 구조 유지
                    extend_edge_count = 1;
                }
            }

            // config를 "현재 실제 edge 시작점"으로 바꾼다.
            if (tid < dim) {
                if constexpr (TangentSpaceTraits<Robot>::enabled) {
                    // 첫 edge 시작점 = 실제 projected q_near
                    config[tid] =nearest_node[tid];
                }
                else {
                    // 다른 robot은 기존 q_steer를 concon_probe에 잠시 저장 (concon_probe = 확장 시작 전에 공유 상태를 초기화하는 부분)
                    concon_probe[tid] =nearest_node[tid]+(config[tid]-nearest_node[tid])*scale;

                    // 실제 edge 시작점
                    config[tid] =nearest_node[tid];
                }
            }
            __syncthreads();

            if (tid == 0) {
                concon_projected_edge_count = 0;
            }
            __syncthreads();

            for (
                int edge_step = 1;
                edge_step <= extend_edge_count;
                edge_step++
            ) {
                // prefix motion의 nominal endpoint 설정
                if (tid < dim) {
                    if constexpr (TangentSpaceTraits<Robot>::enabled) {
                        // selected Tangent Space 위 nominal target
                        // q_k =q_near_TS + k * range * extend_dir
                        concon_probe[tid] =nearest_ts_node[tid]+((float)edge_step*d_settings.range*extend_dir[tid]);
                    }
                    concon_nominal_targets[
                        (edge_step - 1) * MAX_ROBOT_DIM + tid
                    ] = concon_probe[tid];
                }
                __syncthreads();
            }

            const bool projection_good =
                cprrtc_project_concon_node_anchors<Robot>(
                    config,
                    concon_nominal_targets,
                    extend_edge_count,
                    motion_segment,
                    motion_segment_next,
                    motion_projection_valid,
                    motion_projection_prog,
                    motion_projection_success,
                    tid
                );
            __syncthreads();

            if (tid == 0) {
                int valid_edges =
                    projection_good
                        ? extend_edge_count
                        : motion_projection_prog[0];
                if (valid_edges > extend_edge_count) {
                    valid_edges = extend_edge_count;
                }
                if (valid_edges < 0) {
                    valid_edges = 0;
                }
                concon_projected_edge_count = valid_edges;
            }
            __syncthreads();

            cprrtc_project_concon_edge_segments_from_node_anchors<Robot>(
                concon_projected_edge_count,
                motion_segment,
                concon_motion_segments,
                motion_segment_next,
                motion_projection_valid,
                motion_projection_prog,
                motion_projection_success,
                concon_first_projection_failure_edge,
                tid
            );
            __syncthreads();

            if (tid == 0) {
                if (
                    concon_first_projection_failure_edge[0] <
                    concon_projected_edge_count
                ) {
                    concon_projected_edge_count =
                        concon_first_projection_failure_edge[0];
                }
            }
            __syncthreads();

            cprrtc_check_projected_edges_collision_parallel<Robot>(
                concon_projected_edge_count,
                concon_motion_segments,
                &concon_sphere_pos_scratch[
                    bid * d_settings.max_concon_nodes *
                    Collision::fine_sphere_count * Collision::batch_size * 3
                ],
                &concon_sphere_pos_approx_scratch[
                    bid * d_settings.max_concon_nodes *
                    Collision::approximate_sphere_count *
                    Collision::batch_size * 3
                ],
                &concon_link_cc_scratch[
                    bid * d_settings.max_concon_nodes *
                    Collision::joint_flag_stride * Collision::batch_size
                ],
                &concon_transform_scratch[
                    bid * d_settings.max_concon_nodes *
                    Collision::batch_size * Collision::transform_slots * 16
                ],
                env,
                concon_edge_cc_result,
                concon_run_detailed_env_check,
                concon_run_self_collision_check,
                concon_run_detailed_self_check,
                &concon_any_detailed_env_check,
                &concon_any_detailed_self_check,
                concon_first_collision_edge,
                tid
            );
            __syncthreads();

            if (tid < dim) {
                config[tid] = nearest_node[tid];
            }
            if (tid == 0) {
                concon_parent_idx = sindex[0];
            }
            __syncthreads();

            for (
                int edge_step = 1;
                edge_step <= concon_projected_edge_count;
                edge_step++
            ) {
                if (edge_step - 1 >= concon_first_collision_edge[0]) {
                    break;
                }

                float stored_edge_endpoint = 0.0f;
                if (tid < dim) {
                    concon_probe[tid] =
                        concon_nominal_targets[
                            (edge_step - 1) * MAX_ROBOT_DIM + tid
                        ];
                    stored_edge_endpoint =
                        concon_motion_segments[
                            (edge_step - 1) * CONCON_MOTION_SEGMENT_STRIDE +
                            d_settings.granularity * dim + tid
                        ];
                }
                __syncthreads();

                    // grow tree
                    if (tid == 0) {
                        index = cprrtc_reserve_slot(
                            &atomic_free_index[t_tree_id],
                            d_settings.max_samples
                        );

                        if (index < 0) {
                            atomicCAS((int *)&solved, 0, -1);
                        }
                    }
                    __syncthreads();

                    // thread 0이 slot 확보에 실패했다면 block 전체가 tree memory에 접근하기 전에 종료
                    if (index < 0) {
                        return;
                    }

                    // index가 정상이라는 것이 확정된 뒤에만 tree metadata를 기록
                    if (tid == 0) {
                        t_parents[index] =concon_parent_idx;

                        if (d_settings.dynamic_domain) {
                            radii[t_tree_id][index] =FLT_MAX;

                            volatile float *radius_ptr =&radii[t_tree_id][concon_parent_idx];
                            float old_radius;
                            float new_radius;
                            int expected;
                            int desired;

                            do {old_radius =*radius_ptr;
                                if (old_radius == FLT_MAX) {
                                    break;
                                }

                                new_radius = old_radius*(1+d_settings.dd_alpha);
                                expected =__float_as_int(old_radius);
                                desired =__float_as_int(new_radius);

                            } while (atomicCAS((int *)radius_ptr,expected,desired)!= expected);
                        }
                    }
                    __syncthreads();

                    if (tid < dim) {
                        // 실제 tree node에는 projection 결과 저장
                        config[tid] =stored_edge_endpoint;
                        t_nodes[index * dim + tid] =config[tid];
                    }
                    __syncthreads();

                    if constexpr (TangentSpaceTraits<Robot>::enabled) {
                        // 이 node가 EM threshold를 넘어서 만들어진 마지막 ConCon node인지 확인
                        const bool is_em_boundary_node =concon_em_stop&&(edge_step == concon_count);

                        if (tid == 0) {
                            // 기본값
                            new_ts_id = -1;
                            new_ts_basis_ok = true;

                            // Case 1: 일반 ConCon node
                            // 기존 selected TS에 그대로 편입
                            if (!is_em_boundary_node) {
                                t_node_ts_id[index] =selected_ts_id;
                            }

                            // Case 2: EM boundary node
                            // 여기서 새로운 Tangent Space 생성
                            else {
                                // 새 TS 번호 하나 확보
                                new_ts_id =cprrtc_reserve_slot(&ts_count[t_tree_id],d_settings.max_tangent_spaces);

                                // TSBank 공간 부족
                                if (new_ts_id < 0) {
                                    new_ts_basis_ok = false;
                                    atomicCAS((int *)&solved,0,-1);
                                }
                                else {
                                    // 아직 다른 block이 이 TS를 사용하면 안 됨
                                    t_ts_ready[new_ts_id] = 0;

                                    // 새 TS의 root는 방금 projection된 실제 tree node
                                    t_ts_root_node_idx[new_ts_id] =index;

                                    // projected actual q에서 Jacobian 계산
                                    // → null space basis 생성
                                    // → TSBank에 저장
                                    new_ts_basis_ok =cprrtc_store_tangent_basis<Robot>(&t_nodes[index * dim],t_ts_bases,new_ts_id);

                                    if (new_ts_basis_ok) {
                                        // 이 node는 기존 TS가 아니라 새 TS의 root가 된다.
                                        t_node_ts_id[index] =new_ts_id;
                                    }
                                    else {
                                        t_node_ts_id[index] =-1;
                                        atomicCAS((int *)&solved,0,-1);
                                    }
                                }
                            }
                        }
                        __syncthreads();

                        // EM boundary인데 새 TS를 만들지 못했다면 이 node를 tree에 ready 상태로 공개하면 안 된다.
                        if (is_em_boundary_node&&(new_ts_id < 0||!new_ts_basis_ok)) {
                            return;
                        }

                        if (tid < dim) {
                            // 일반 node
                            // 기존 Tangent Space상의 nominal 위치를 저장
                            if (!is_em_boundary_node) {
                                t_node_ts_q[index * dim + tid] =concon_probe[tid];
                            }
                            // 새 TS root
                            // 새 Tangent Space는 projected actual q에서 시작하므로 nominal q == actual q
                            else if (new_ts_basis_ok) {
                                t_node_ts_q[index * dim + tid] =config[tid];
                            }
                        }   
                        __syncthreads();
                    }

                    // 모든 node / TS metadata가 global memory에
                    // 기록될 때까지 보장
                    __threadfence();
                    __syncthreads();

                    if constexpr (TraceTrees) {
                        if (tid == 0) {
                            should_skip = false;
                            for (int joint = 0; joint < dim; joint++) {
                                const float value = t_nodes[index * dim + joint];
                                if (!isfinite(value) || value == UNWRITTEN_VAL) {
                                    should_skip = true;
                                    break;
                                }
                            }
                            if (should_skip) {
                                atomicCAS((int *)&solved, 0, -1);
                            }
                        }
                        __syncthreads();

                        if (should_skip) {
                            return;
                        }
                    }

                    if constexpr (TangentSpaceTraits<Robot>::enabled) {
                        if (tid == 0) {
                            const int assigned_ts_id =t_node_ts_id[index];

                            cprrtc_register_node_in_ts(index,assigned_ts_id,t_ts_node_count,t_ts_lane_head,t_node_next_in_ts);
                        }
                    }
                    __syncthreads();

                    if (tid == 0) {
                        // 먼저 tree node 공개
                        node_ready[t_tree_id][index] = 1;
                        if constexpr (TraceTrees) {
                            __threadfence();
                            atomicAdd((int *)&completed_nodes[t_tree_id],1);
                        }
                        else {
                            atomicAdd((int *)&completed_nodes[t_tree_id],1);
                            __threadfence();
                        }

                        if constexpr (TangentSpaceTraits<Robot>::enabled) {
                            const bool is_em_boundary_node =concon_em_stop&&(edge_step == concon_count);

                            // 새 TS는 모든 데이터가 준비된 가장 마지막에 ready = 1로 공개
                            if (is_em_boundary_node&&new_ts_basis_ok&&new_ts_id >= 0) {
                                t_ts_ready[new_ts_id] = 1;
                            }
                        }
                    }
                    __syncthreads();

                    // 방금 성공한 node가 다음 edge의 parent가 된다.
                    if (tid == 0) {
                        concon_parent_idx =index;
                        concon_valid_count++;
                    }
                    __syncthreads();
            } // edge 검사 완료

            // ConCon validation 전체가 끝난 뒤 CONNECT 여부 결정
            if (concon_valid_count > 0 && concon_valid_count == extend_edge_count) { // concon edge가 모두 성공했는지 확인. (성공한 edge가 최소 하나 이상 && 계획했던 모든 edge가 성공)
                // connect
                local_min_dist = FLT_MAX;
                local_near_idx = 0;
                int size = min((int)atomic_free_index[o_tree_id], d_settings.max_samples); // 빈데편 tree 크기 가져오기

                // 반대편 tree node를 thread들이 나눠서 NN 검사
                for (unsigned int i = tid; i < size; i += blockDim.x) { 
                    if (o_node_ready[i] == 0) {
                        continue;
                    }
                    dist = cprrtc_sq_config_distance<Robot>(
                        &o_nodes[i * dim],
                        config
                    );
                    if (dist < local_min_dist) { // 현재 thread가 찾은 최근접 노드 갱신
                        local_min_dist = dist;
                        local_near_idx = i;
                    }
                }
                // thread 별 결과를 shared memory에 저장
                sdata[tid] = local_min_dist;
                sindex[tid] = local_near_idx;
                __syncthreads();
                
                // 모든 thread의 결과 중 최솟값 선택
                for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) {
                        if (sdata[tid + s] < sdata[tid]) {
                            sdata[tid]  = sdata[tid + s];
                            sindex[tid] = sindex[tid + s];
                        }
                    }
                    __syncthreads();
                }
                 
                // 반대편 tree에서 찾은 최근접 노드를 CONNECT 목표로 고정하고, 그 노드까지 몇 번 확장해야 하는지 계산하는 초기화 과정
                if (tid == 0) {
                    // CONNECT 시작 시 상대 tree의 target을 딱 한 번 결정
                    connect_target_idx = sindex[0];
                    connect_target_node =&o_nodes[connect_target_idx * dim];

                    // 새 TB-RRT CONNECT 상태 초기화
                    connect_failed = false;
                    const float connect_total_distance =sqrtf(sdata[0]);
                    connect_reached =connect_total_distance<= d_settings.connect_reached_tolerance; // 현재 위치가 목표 노드에 충분히 가까운지 확인

                    n_extensions =static_cast<unsigned int>(ceilf(connect_total_distance/ d_settings.range)); // 목표 노드까지 range 간격으로 이동하려면 몇 번 확장해야 하는지 계산

                    // 계산 결과가 0이더라도 1로 보정
                    if (n_extensions < 1u) {n_extensions = 1u;}

                    connection_reached_shared =connect_reached;
                }
                __syncthreads();

                int connect_chunk_count = 0;

                while (connect_chunk_count< d_settings.max_connect_concon_chunks) { // 상대 tree의 target 향해 계속 확장
                    if constexpr (TraceTrees) {
                        if (tid == 0) {
                            should_skip = (solved != 0);
                        }
                        __syncthreads();
                        if (should_skip) {
                            return;
                        }
                    }
                    else {
                        if (tid == 0) {
                            should_skip = (solved != 0);
                        }
                        __syncthreads();
                        if (should_skip) {
                            return;
                        }
                    }

                    const float chunk_start_target_distance =cprrtc_shared_config_distance<Robot>(
                            config,
                            connect_target_node,
                            sdata,
                            tid
                        );

                    if (tid == 0) {
                        connect_reached =chunk_start_target_distance<= d_settings.connect_reached_tolerance;
                    }
                    __syncthreads();

                    if (connect_reached) {
                        break;
                    }

                    float connect_tangent_dist = 0.0f;

                    if constexpr (TangentSpaceTraits<Robot>::enabled) {
                        // [NEW CONNECT] 현재 node가 속한 Tangent Space 확인
                        if (tid == 0) {
                            // 현재 CONNECT 시작 node가 속한 Tangent Space
                            selected_ts_id =t_node_ts_id[index];

                            // 유효한 Tangent Space인지 확인
                            if (selected_ts_id < 0||t_ts_ready[selected_ts_id] == 0) {
                                connect_failed = true;
                            }
                            else {
                                // 현재 실제 manifold상의 tree node
                                nearest_node =&t_nodes[index * dim];

                                // 같은 node의 Tangent Space 위 nominal configuration
                                nearest_ts_node =&t_node_ts_q[index * dim];
                            }
                        }
                        __syncthreads();

                        // [NEW CONNECT] 현재 TS의 tangent basis
                        const float *connect_basis = nullptr;

                        if (!connect_failed) {
                            connect_basis = &t_ts_bases[
                                selected_ts_id * TangentSpaceTraits<Robot>::basis_size
                            ];
                        }

                        __syncthreads();

                        if (!connect_failed) {
                            connect_tangent_dist = cprrtc_project_target_direction_to_tangent<Robot>(
                                    config,
                                    connect_target_node,
                                    connect_basis,
                                    ts_coeff,
                                    extend_dir,
                                    sdata,
                                    tid
                                );
                        }
                        __syncthreads();

                        if (tid == 0) {
                            if (!connect_failed&&connect_tangent_dist <= 1.0e-8f) {
                                connect_failed = true;
                            }
                        }
                        __syncthreads();

                        if (tid == 0) {
                            concon_count = 0;
                            concon_em_stop = false;
                        }
                        __syncthreads();

                        if (!connect_failed) {
                            for (int step = 1; step <= d_settings.max_concon_nodes; step++) {
                                // 원래라면 step * range 만큼 이동 하지만 target까지 tangent distance를 넘지 않도록 제한
                                const float raw_step_distance =static_cast<float>(step)* d_settings.range;
                                const float connect_step_distance =fminf(raw_step_distance,connect_tangent_dist);

                                // 이번 candidate가 target까지의 마지막 candidate인지
                                const bool connect_target_step =raw_step_distance>= connect_tangent_dist;

                                // 현재 TS 위 nominal candidate
                                if (tid < dim) {
                                    concon_probe[tid] =nearest_ts_node[tid]+connect_step_distance* extend_dir[tid];
                                }
                                __syncthreads();

                                // 기존과 동일하게 EM 검사
                                if (tid == 0) {
                                    const float em_error =cprrtc_constraint_error_norm<Robot>(concon_probe);
                                    concon_count = step;
                                    if (em_error >d_settings.em_threshold) {
                                        concon_em_stop = true;
                                    }
                                }
                                __syncthreads();

                                // 1. EM threshold를 넘었거나
                                // 2. target까지 필요한 tangent distance에 도달했으면
                                // 더 이상 candidate를 만들지 않음
                                if (concon_em_stop||connect_target_step
                                ) {
                                    break;
                                }
                            }
                        }
                        __syncthreads();
                    }
                    else {
                        if (tid == 0) {
                            nearest_node =&t_nodes[index * dim];
                            concon_count = 1;
                            concon_em_stop = false;
                        }
                        __syncthreads();
                    }

                    // 이번 ConCon chunk의 실제 edge validation 준비
                    if (tid == 0) {
                        concon_valid_count = 0;

                        // 첫 CONNECT edge의 parent는 EXTEND에서 마지막으로 추가된 실제 node
                        concon_parent_idx = index;
                    }

                    // config를 현재 실제 projected configuration으로 복구
                    if (tid < dim) {
                        config[tid] = nearest_node[tid];
                    }
                    __syncthreads();

                    if (tid == 0) {
                        concon_projected_edge_count = 0;
                    }
                    __syncthreads();

                    if constexpr (TraceTrees) {
                        if (tid == 0) {
                            should_skip = (solved != 0);
                        }
                        __syncthreads();
                        if (should_skip) {
                            return;
                        }
                    }
                    else {
                        if (tid == 0) {
                            should_skip = (solved != 0);
                        }
                        __syncthreads();
                        if (should_skip) {
                            return;
                        }
                    }

                    // 이번 chunk의 nominal endpoint를 먼저 만들고 prefix motion 하나로 projection한다.
                    for (
                        int edge_step = 1;
                        edge_step <= concon_count;
                        edge_step++
                    ) {
                        if (tid < dim) {
                            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                                // 이번 edge의 Tangent Space 위 nominal target 생성
                                // target까지 필요한 tangent distance보다
                                // 멀리 가지 않도록 마지막 edge 길이를 줄인다.
                                const float raw_edge_distance =
                                    static_cast<float>(edge_step) * d_settings.range;
                                const float connect_edge_distance =
                                    fminf(raw_edge_distance, connect_tangent_dist);

                                concon_probe[tid] =
                                    nearest_ts_node[tid]
                                    + connect_edge_distance * extend_dir[tid];
                            }
                            else {
                                const float step_scale = fminf(
                                    1.0f,
                                    d_settings.range
                                        / fmaxf(chunk_start_target_distance, 1.0e-8f)
                                );
                                concon_probe[tid] =
                                    config[tid]
                                    + (connect_target_node[tid] - config[tid])
                                        * step_scale;
                            }
                            // 실제 projected 현재 node
                            //          ↓
                            // 이번 TS nominal target
                            // 사이를 granularity만큼 나눈다.
                            concon_nominal_targets[
                                (edge_step - 1) * MAX_ROBOT_DIM + tid
                            ] = concon_probe[tid];
                        }
                        __syncthreads();
                    }

                    const bool extension_projection_good =
                        cprrtc_project_concon_node_anchors<Robot>(
                                config,
                                concon_nominal_targets,
                                concon_count,
                                motion_segment,
                                motion_segment_next,
                                motion_projection_valid,
                                motion_projection_prog,
                                motion_projection_success,
                                tid
                            );
                    __syncthreads();

                    if (tid == 0) {
                        int valid_edges =
                            extension_projection_good
                                ? concon_count
                                : motion_projection_prog[0];
                        if (valid_edges > concon_count) {
                            valid_edges = concon_count;
                        }
                        if (valid_edges < 0) {
                            valid_edges = 0;
                        }

                        for (int edge_step = 1; edge_step <= valid_edges; edge_step++) {
                            const float distance_before =
                                cprrtc_config_distance_from_volatile<Robot>(
                                    &motion_segment[
                                        (edge_step - 1) * dim
                                    ],
                                    connect_target_node
                                );
                            const float distance_after =
                                cprrtc_config_distance_from_volatile<Robot>(
                                    &motion_segment[
                                        edge_step * dim
                                    ],
                                    connect_target_node
                                );

                            if (
                                distance_after >=
                                distance_before -
                                    d_settings.connect_progress_epsilon
                            ) {
                                valid_edges = edge_step - 1;
                                break;
                            }
                        }

                        concon_projected_edge_count = valid_edges;
                    }
                    __syncthreads();

                    cprrtc_project_concon_edge_segments_from_node_anchors<Robot>(
                        concon_projected_edge_count,
                        motion_segment,
                        concon_motion_segments,
                        motion_segment_next,
                        motion_projection_valid,
                        motion_projection_prog,
                        motion_projection_success,
                        concon_first_projection_failure_edge,
                        tid
                    );
                    __syncthreads();

                    if (tid == 0) {
                        if (
                            concon_first_projection_failure_edge[0] <
                            concon_projected_edge_count
                        ) {
                            concon_projected_edge_count =
                                concon_first_projection_failure_edge[0];
                        }
                    }
                    __syncthreads();

                    cprrtc_check_projected_edges_collision_parallel<Robot>(
                        concon_projected_edge_count,
                        concon_motion_segments,
                        &concon_sphere_pos_scratch[
                            bid * d_settings.max_concon_nodes *
                            Collision::fine_sphere_count *
                            Collision::batch_size * 3
                        ],
                        &concon_sphere_pos_approx_scratch[
                            bid * d_settings.max_concon_nodes *
                            Collision::approximate_sphere_count *
                            Collision::batch_size * 3
                        ],
                        &concon_link_cc_scratch[
                            bid * d_settings.max_concon_nodes *
                            Collision::joint_flag_stride *
                            Collision::batch_size
                        ],
                        &concon_transform_scratch[
                            bid * d_settings.max_concon_nodes *
                            Collision::batch_size *
                            Collision::transform_slots * 16
                        ],
                        env,
                        concon_edge_cc_result,
                        concon_run_detailed_env_check,
                        concon_run_self_collision_check,
                        concon_run_detailed_self_check,
                        &concon_any_detailed_env_check,
                        &concon_any_detailed_self_check,
                        concon_first_collision_edge,
                        tid
                    );
                    __syncthreads();

                    if (tid < dim) {
                        config[tid] = nearest_node[tid];
                    }
                    if (tid == 0) {
                        concon_parent_idx = index;
                    }
                    __syncthreads();

                    // collision-free로 확인된 edge만 앞에서부터 tree에 추가한다.
                    for (
                        int edge_step = 1;
                        edge_step <= concon_projected_edge_count;
                        edge_step++
                    ) {
                        if (edge_step - 1 >= concon_first_collision_edge[0]) {
                            break;
                        }

                        float connect_projected_endpoint = 0.0f;
                        if (tid < dim) {
                            concon_probe[tid] =
                                concon_nominal_targets[
                                    (edge_step - 1) * MAX_ROBOT_DIM + tid
                                ];
                            connect_projected_endpoint =
                                concon_motion_segments[
                                    (edge_step - 1) *
                                        CONCON_MOTION_SEGMENT_STRIDE +
                                    d_settings.granularity * dim + tid
                                ];
                        }
                        __syncthreads();

                        // CONNECT node slot 확보
                        if (tid == 0) {
                            index = cprrtc_reserve_slot(
                                &atomic_free_index[t_tree_id],
                                d_settings.max_samples
                            );

                            if (index < 0) {
                                atomicCAS((int *)&solved, 0, -1);
                            }
                        }
                        __syncthreads();

                        if (index < 0) {
                            return;
                        }
                        __syncthreads();

                        if (index < 0) {
                            return;
                        }

                        // CONNECT node metadata
                        if (tid == 0) {
                            t_parents[index] =concon_parent_idx;
                            radii[t_tree_id][index] =FLT_MAX;
                        }
                        __syncthreads();

                        // projected endpoint 저장
                        if (tid < dim) {
                            config[tid] =connect_projected_endpoint;
                            t_nodes[index * dim + tid] =config[tid];
                        }
                        __syncthreads();

                        // CONNECT node의 Tangent Space 정보 저장
                        if constexpr (TangentSpaceTraits<Robot>::enabled) {
                            // EM threshold를 처음 넘은 마지막 node인가?
                            const bool is_connect_em_boundary_node =concon_em_stop&&(edge_step == concon_count);

                            if (tid == 0) {
                                new_ts_id = -1;
                                new_ts_basis_ok = true;

                                // 일반 CONNECT node → 현재 TS에 그대로 포함
                                if (!is_connect_em_boundary_node) {
                                    t_node_ts_id[index] = selected_ts_id;
                                }

                                // EM boundary node
                                // → 실제 projected node에서 새 TS 생성
                                else {
                                    new_ts_id =cprrtc_reserve_slot(
                                        &ts_count[t_tree_id],
                                        d_settings.max_tangent_spaces
                                    );

                                    if (new_ts_id < 0) {
                                        new_ts_basis_ok = false;

                                        atomicCAS((int *)&solved,0,-1);
                                    }
                                    else {
                                        // 아직 다른 block이 사용하면 안 됨
                                        t_ts_ready[new_ts_id] = 0;

                                        // 현재 projected node가 새 TS root
                                        t_ts_root_node_idx[new_ts_id] = index;

                                        // 실제 projected configuration에서
                                        // Jacobian/null-space basis 생성
                                        new_ts_basis_ok =cprrtc_store_tangent_basis<Robot>(
                                                &t_nodes[index * dim],
                                                t_ts_bases,
                                                new_ts_id
                                            );

                                        if (new_ts_basis_ok) {
                                            t_node_ts_id[index] =new_ts_id;
                                        }
                                        else {
                                            t_node_ts_id[index] =-1;

                                            atomicCAS((int *)&solved,0,-1);
                                        }
                                    }
                                }
                            }
                            __syncthreads();


                            if (is_connect_em_boundary_node&&(new_ts_id < 0||!new_ts_basis_ok)) {
                                return;
                            }

                            // node의 TS nominal configuration 저장
                            if (tid < dim) {
                                if (!is_connect_em_boundary_node) {
                                    t_node_ts_q[index * dim + tid] =concon_probe[tid];
                                }

                                // 새 TS root에서는 nominal q == actual projected q
                                else if (new_ts_basis_ok) {
                                    t_node_ts_q[index * dim + tid] =config[tid];
                                }
                            }
                            __syncthreads();
                        }
                        __threadfence();
                        __syncthreads();

                        if constexpr (TraceTrees) {
                            if (tid == 0) {
                                should_skip = false;
                                for (int joint = 0; joint < dim; joint++) {
                                    const float value = t_nodes[index * dim + joint];
                                    if (!isfinite(value) || value == UNWRITTEN_VAL) {
                                        should_skip = true;
                                        break;
                                    }
                                }
                                if (should_skip) {
                                    atomicCAS((int *)&solved, 0, -1);
                                }
                            }
                            __syncthreads();

                            if (should_skip) {
                                return;
                            }
                        }

                        if constexpr (TangentSpaceTraits<Robot>::enabled) {
                            if (tid == 0) {
                                const int assigned_ts_id =t_node_ts_id[index];

                                cprrtc_register_node_in_ts(index,assigned_ts_id,t_ts_node_count,t_ts_lane_head,t_node_next_in_ts);
                            }
                        }
                        __syncthreads();

                        if (tid == 0) {
                            // 먼저 tree node 공개
                            t_node_ready[index] = 1;
                            if constexpr (TraceTrees) {
                                __threadfence();
                                atomicAdd((int *)&completed_nodes[t_tree_id],1);
                            }
                            else {
                                atomicAdd((int *)&completed_nodes[t_tree_id],1);
                                __threadfence();
                            }

                            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                                const bool is_connect_em_boundary_node =concon_em_stop&&(edge_step == concon_count);

                                // 모든 TS 정보가 저장된 후 마지막으로 ready
                                if (is_connect_em_boundary_node&&new_ts_basis_ok&&new_ts_id >= 0) {
                                    t_ts_ready[new_ts_id] = 1;
                                }
                            }
                        }
                        __syncthreads();

                        if (tid == 0) {
                            // 다음 edge의 parent는 방금 성공한 실제 projected node
                            concon_parent_idx = index;
                            concon_valid_count++;
                        }
                        __syncthreads();

                        const float current_target_distance = cprrtc_shared_config_distance<Robot>(config,connect_target_node,sdata,tid);

                        if (tid == 0) {
                            connect_reached =current_target_distance<=d_settings.connect_reached_tolerance;
                        }
                        __syncthreads();

                        // target에 도달했다면 뒤의 nominal ConCon candidate는 처리하지 않는다.
                        if (connect_reached) {
                            break;
                        }
                        __syncthreads();
                    }

                    // 이번 chunk의 validation 결과 확인
                    if (tid == 0) {
                        // target에 도달하지 않았는데 candidate를 끝까지 검증하지 못했다면 projection 또는 collision 실패
                        if (!connect_reached&&concon_valid_count != concon_count) {
                            connect_failed = true;
                        }
                    }
                    __syncthreads();

                    if (connect_failed) {
                        break;
                    }

                    if (connect_reached) {
                        break;
                    }

                    // 이번 chunk는 정상적으로 끝났지만
                    // 아직 target에는 도달하지 않음
                    connect_chunk_count++;
                }

                const float final_connection_distance =
                    cprrtc_shared_config_distance<Robot>(
                        config,
                        connect_target_node,
                        sdata,
                        tid
                    );

                if (tid == 0) {
                    connect_reached =final_connection_distance<= d_settings.connect_reached_tolerance;
                }
                __syncthreads();
                    
                // CONNECT 성공 시 양쪽 트리의 parent를 역추적하여 최종 경로 복원
                if (!connect_failed&&connect_reached) { // connected
                    if (tid == 0 && atomicCAS((int *)&solved, 0, 1) == 0) { // block 0번 thread만 최종 경로 복원 수행
                        cost = final_connection_distance;

                        if constexpr (TraceTrees) {
                            connection_tree_id = t_tree_id;
                            connection_node_idx = index;
                            connection_other_tree_id = o_tree_id;
                            connection_other_node_idx = connect_target_idx;
                        }
                        // trace back to the start and goal.
                        int current = index;
                        int parent;
                        int t_path_size = 0;
                        int o_path_size = 0;
                        while (true) {
                            if (current < 0 ||
                                current >= d_settings.max_samples ||
                                t_path_size >= MAX_PATH_NODES) {
                                break;
                            }

                            parent = t_parents[current];

                            if (parent == current) {
                                break;
                            }

                            if (parent < 0 || parent >= d_settings.max_samples) {
                                break;
                            }

                            cost += cprrtc_config_distance<Robot>(
                                (float *)&t_nodes[current * dim],
                                (float *)&t_nodes[parent * dim]
                            );

                            for (int i = 0; i < dim; i++) {
                                path[t_tree_id][t_path_size * dim + i] =
                                    t_nodes[current * dim + i];
                            }

                            t_path_size++;
                            current = parent;
                        }
                        if (t_tree_id == 1) reached_goal_idx = current; // 현재 tree가 goal tree라면, 역추적이 끝난 현재 current가 goal tree의 root (여러 goal을 지원하는 경우 어떤 goal root에 도달했는지 기록)
                        current = connect_target_idx; // 반대편 tree의 CONNECT 목표 노드에서 역추적 시작
                        while(true) { // 반대편 tree의 root에 도달할 때까지 부모 node 따라감
                            if (current < 0 ||
                                current >= d_settings.max_samples ||
                                o_path_size >= MAX_PATH_NODES) {
                                break;
                            }

                            parent = o_parents[current];

                            if (parent == current) {
                                break;
                            }

                            if (parent < 0 || parent >= d_settings.max_samples) {
                                break;
                            }

                            cost += cprrtc_config_distance<Robot>(
                                &o_nodes[current * dim],
                                &o_nodes[parent * dim]
                            );

                            for (int i = 0; i < dim; i++) {
                                path[o_tree_id][o_path_size * dim + i] =
                                    o_nodes[current * dim + i];
                            }

                            o_path_size++;
                            current = parent;
                        }
                        if (t_tree_id == 0) reached_goal_idx = current;
                        path_size[t_tree_id] = t_path_size;
                        path_size[o_tree_id] = o_path_size;
                        solved_iters = iter;
                    }
                    __syncthreads();
                }
            } 
            else if (d_settings.dynamic_domain && tid == 0) {      
                // printf("no config added\n");
                volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                float old_radius, new_radius;
                int expected, desired;
                do {
                    old_radius = *radius_ptr;
                    if (old_radius == FLT_MAX) {
                        new_radius = d_settings.dd_radius;
                    } else {
                        new_radius = fmaxf(old_radius * (1.f - d_settings.dd_alpha), d_settings.dd_min_radius);
                    }
                    expected = __float_as_int(old_radius);
                    desired = __float_as_int(new_radius);
                } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
            }
        __syncthreads();

        if constexpr (TraceTrees) {
            if (tid == 0) {
                should_skip = (solved != 0);
            }
            __syncthreads();
            if (should_skip) return;
        }
        else {
            if (tid == 0) {
                should_skip = (solved != 0);
            }
            __syncthreads();
            if (should_skip) {
                return;
            }
        }
        }
    }




    template <typename Robot>
    void copy_tree_trace_to_result(
        PlannerResult<Robot> &res,
        float *nodes[2],
        int *parents[2],
        int *node_ready[2],
        const int current_samples[2]
    ) {
        static constexpr auto dim = Robot::dimension;
        for (int tree = 0; tree < 2; tree++) {
            const int tree_size = current_samples[tree];
            res.tree_nodes[tree].resize(tree_size);
            res.tree_parents[tree].resize(tree_size);
            res.tree_node_ready[tree].resize(tree_size);
            if (tree_size == 0) {
                continue;
            }

            std::vector<float> host_nodes(
                static_cast<std::size_t>(tree_size) * dim
            );
            cudaMemcpy(host_nodes.data(), nodes[tree], sizeof(float) * host_nodes.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(res.tree_parents[tree].data(),parents[tree],sizeof(int) * tree_size,cudaMemcpyDeviceToHost);
            cudaMemcpy(res.tree_node_ready[tree].data(),node_ready[tree],sizeof(int) * tree_size,cudaMemcpyDeviceToHost);

            for (int index = 0; index < tree_size; index++) {
                std::copy_n(
                    host_nodes.data() + static_cast<std::size_t>(index) * dim,
                    dim,
                    res.tree_nodes[tree][index].begin()
                );
            }
        }
    }

    template <typename Robot>
    std::vector<int> trace_parent_chain(
        const PlannerResult<Robot> &res,
        int tree,
        int node_index
    ) {
        std::vector<int> chain;
        if (tree < 0 || tree >= 2) {
            return chain;
        }
        const auto &parents = res.tree_parents[tree];
        const auto &ready = res.tree_node_ready[tree];
        int current = node_index;
        for (int guard = 0; guard < static_cast<int>(parents.size()); guard++) {
            if (current < 0 || current >= static_cast<int>(parents.size())) {
                break;
            }
            if (current >= static_cast<int>(ready.size()) || ready[current] == 0) {
                break;
            }
            chain.push_back(current);
            const int parent = parents[current];
            if (parent == current) {
                break;
            }
            current = parent;
        }
        return chain;
    }

    template <typename Robot>
    void fill_solution_trace(PlannerResult<Robot> &res) {
        if (!res.solved || res.connection_tree_id < 0
            || res.connection_other_tree_id < 0) {
            return;
        }

        const int start_connection_index = res.connection_tree_id == 0
            ? res.connection_node_idx
            : res.connection_other_node_idx;
        const int goal_connection_index = res.connection_tree_id == 1
            ? res.connection_node_idx
            : res.connection_other_node_idx;

        auto start_chain = trace_parent_chain(res, 0, start_connection_index);
        std::reverse(start_chain.begin(), start_chain.end());
        for (int index : start_chain) {
            res.solution_trace.push_back({0, index});
        }
        for (int index : trace_parent_chain(res, 1, goal_connection_index)) {
            res.solution_trace.push_back({1, index});
        }
    }

    template <typename Robot>
    PlannerResult<Robot> solve(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &h_environment,
        pRRTC_settings &settings
    ) 
    {
        auto start_time = std::chrono::steady_clock::now();
        static constexpr auto dim = Robot::dimension;
        using Collision = robots::CollisionTraits<Robot>;
        if (settings.granularity != Collision::batch_size) {
            throw std::invalid_argument(
                "pRRTC granularity must match the selected robot's collision batch size"
            );
        }
        if (settings.granularity > MAX_GRANULARITY) {
            throw std::invalid_argument(
                "pRRTC prefix ConCon projection supports granularity up to 16"
            );
        }
        if (settings.max_concon_nodes <= 0) {
            throw std::invalid_argument(
                "pRRTC max_concon_nodes must be positive"
            );
        }
        if (settings.max_concon_nodes > MAX_PARALLEL_CONCON_EDGES) {
            throw std::invalid_argument(
                "pRRTC parallel ConCon collision check supports up to 5 edges"
            );
        }
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            if (settings.max_tangent_spaces <= 0) {
                throw std::invalid_argument(
                    "max_tangent_spaces must be positive"
                );
            }

            if (goals.size() >static_cast<std::size_t>(settings.max_tangent_spaces)) {
                throw std::invalid_argument(
                    "number of goals exceeds max_tangent_spaces"
                );
            }
        }
        std::size_t start_index = 0;
        PlannerResult<Robot> res;

        // copy data to GPU
        cudaMemcpyToSymbol(d_settings, &settings, sizeof(settings));
        if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
            cudaMemcpyToSymbol(
                ppln::collision::ffw_sg2_mobility_attached_object_collision,
                &settings.ffw_sg2_attached_object_collision,
                sizeof(settings.ffw_sg2_attached_object_collision)
            );
        }
        int num_goals = goals.size();
        float *nodes[2];
        int *parents[2];
        int *node_ready[2] = {nullptr, nullptr};
        float *radii[2];
        float **d_nodes;
        int **d_parents;
        int **d_node_ready = nullptr;
        float **d_radii;
        // Tangent Space membership for tree nodes
        int *node_ts_id[2] = {nullptr, nullptr};
        float *node_ts_q[2] = {nullptr, nullptr};

        int **d_node_ts_id = nullptr;
        float **d_node_ts_q = nullptr;

        // Tangent Space Bank
        int *ts_count = nullptr;

        int *ts_root_node_idx[2] = {nullptr, nullptr};
        float *ts_bases[2] = {nullptr, nullptr};
        int *ts_ready[2] = {nullptr, nullptr};

        int **d_ts_root_node_idx = nullptr;
        float **d_ts_bases = nullptr;
        int **d_ts_ready = nullptr;
        // TS별로 node를 64개 thread 목록에 나눠 저장하는 역방향 index
        int *ts_node_count[2] = {nullptr, nullptr};
        int *ts_lane_head[2] = {nullptr, nullptr};
        int *node_next_in_ts[2] = {nullptr, nullptr};

        int **d_ts_node_count = nullptr;
        int **d_ts_lane_head = nullptr;
        int **d_node_next_in_ts = nullptr;
        float *concon_sphere_pos_scratch = nullptr;
        float *concon_sphere_pos_approx_scratch = nullptr;
        int *concon_link_cc_scratch = nullptr;
        float *concon_transform_scratch = nullptr;
        cudaMalloc(&d_nodes, 2 * sizeof(float*));
        cudaMalloc(&d_parents, 2 * sizeof(int*));
        cudaMalloc(&d_radii, 2 * sizeof(float*));
        cudaMalloc(&d_node_ready, 2 * sizeof(int*));
        cudaMalloc(&d_node_ts_id, 2 * sizeof(int*));
        cudaMalloc(&d_node_ts_q, 2 * sizeof(float*));
        cudaMalloc(&ts_count, 2 * sizeof(int));
        cudaMalloc(&d_ts_root_node_idx,2 * sizeof(int*));
        cudaMalloc(&d_ts_bases,2 * sizeof(float*));
        cudaMalloc(&d_ts_ready,2 * sizeof(int*));
        cudaMalloc(&d_ts_node_count,2 * sizeof(int*));
        cudaMalloc(&d_ts_lane_head,2 * sizeof(int*));
        cudaMalloc(&d_node_next_in_ts,2 * sizeof(int*));
        const std::size_t config_size = dim * sizeof(float);

        for (int i = 0; i < 2; i++) {
            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                const std::size_t basis_bytes =
                    static_cast<std::size_t>(settings.max_tangent_spaces) *
                    TangentSpaceTraits<Robot>::basis_size * sizeof(float);
                const std::size_t ts_count_bytes =static_cast<std::size_t>(settings.max_tangent_spaces)*sizeof(int);
                const std::size_t ts_lane_head_bytes =static_cast<std::size_t>(settings.max_tangent_spaces)*MAX_THREADS_PER_BLOCK*sizeof(int);
                const std::size_t node_next_bytes =static_cast<std::size_t>(settings.max_samples)*sizeof(int);

                // TSBank의 tangent basis 저장 공간
                cudaMalloc(&ts_bases[i],basis_bytes);
                cudaMemset(ts_bases[i],0,basis_bytes);

                cudaMalloc(&ts_node_count[i],ts_count_bytes);
                cudaMemset(ts_node_count[i],0,ts_count_bytes);

                cudaMalloc(&ts_lane_head[i],ts_lane_head_bytes);
                cudaMemset(ts_lane_head[i],0xff,ts_lane_head_bytes);

                cudaMalloc(&node_next_in_ts[i],node_next_bytes);
                cudaMemset(node_next_in_ts[i],0xff,node_next_bytes);
            }
            cudaMalloc(&nodes[i], settings.max_samples * config_size);
            cudaMalloc(&parents[i], settings.max_samples * sizeof(int));
            cudaMalloc(&radii[i], settings.max_samples * sizeof(float));
            cudaMalloc(&node_ready[i], settings.max_samples * sizeof(int));
            cudaMemset(node_ready[i], 0, settings.max_samples * sizeof(int));
            // 이 tree의 각 node가 어느 TS에 속하는지
            cudaMalloc(&node_ts_id[i],settings.max_samples * sizeof(int));
            // 처음에는 어떤 TS에도 속하지 않음: -1
            cudaMemset(node_ts_id[i],0xff,settings.max_samples * sizeof(int));
            // 각 tree node의 Tangent Space 상 nominal q
            cudaMalloc(&node_ts_q[i],settings.max_samples * config_size);
            // TS마다 root node index 저장
            cudaMalloc(&ts_root_node_idx[i],settings.max_tangent_spaces * sizeof(int));
            // TS가 완전히 생성되었는지
            cudaMalloc(&ts_ready[i],settings.max_tangent_spaces * sizeof(int));
            // 처음에는 모든 TS가 아직 생성되지 않음
            cudaMemset(ts_ready[i],0,settings.max_tangent_spaces * sizeof(int));
        }
        cudaMemcpy(d_nodes, nodes, 2 * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_parents, parents, 2 * sizeof(int*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_node_ts_id,node_ts_id,2 * sizeof(int*),cudaMemcpyHostToDevice);
        cudaMemcpy(d_node_ts_q,node_ts_q,2 * sizeof(float*),cudaMemcpyHostToDevice);
        cudaMemcpy(d_ts_root_node_idx,ts_root_node_idx,2 * sizeof(int*),cudaMemcpyHostToDevice);
        cudaMemcpy(d_ts_bases,ts_bases,2 * sizeof(float*),cudaMemcpyHostToDevice);
        cudaMemcpy(d_ts_ready,ts_ready,2 * sizeof(int*),cudaMemcpyHostToDevice);
        cudaMemcpy(d_ts_node_count,ts_node_count,2 * sizeof(int*),cudaMemcpyHostToDevice);
        cudaMemcpy(d_ts_lane_head,ts_lane_head,2 * sizeof(int*),cudaMemcpyHostToDevice);
        cudaMemcpy(d_node_next_in_ts,node_next_in_ts,2 * sizeof(int*),cudaMemcpyHostToDevice);
        int h_ts_count[2] = {0, 0};
        cudaMemcpy(d_radii,radii,2 * sizeof(float*),cudaMemcpyHostToDevice);
        cudaMemcpy(ts_count,h_ts_count,2 * sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_node_ready, node_ready, 2 * sizeof(int*), cudaMemcpyHostToDevice);

        // set nodes to unitialized
        std::vector<float> nodes_init(settings.max_samples * dim, UNWRITTEN_VAL);
        cudaMemcpy((void *)nodes[0], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)nodes[1], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);
        cudaMemcpy(node_ts_q[0],nodes_init.data(),config_size * settings.max_samples,cudaMemcpyHostToDevice);
        cudaMemcpy(node_ts_q[1],nodes_init.data(),config_size * settings.max_samples,cudaMemcpyHostToDevice);
            
        // initialize radii
        std::vector<float> radii_init(num_goals, FLT_MAX);
        cudaMemcpy((void *)radii[0], radii_init.data(), sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)radii[1], radii_init.data(), sizeof(float) * num_goals, cudaMemcpyHostToDevice);
        
        // create a curandState for each thread
        curandState *rng_states;
        int num_rng_states = settings.num_new_configs * dim;
        cudaMalloc(&rng_states, num_rng_states * sizeof(curandState));
        int numBlocks = (num_rng_states + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_rng<<<numBlocks, BLOCK_SIZE>>>(rng_states, 1, num_rng_states);

        HaltonState<Robot> *halton_states;
        cudaMalloc(&halton_states, settings.num_new_configs * sizeof(HaltonState<Robot>));
        int numBlocks1 = (settings.num_new_configs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_halton<Robot><<<numBlocks1, BLOCK_SIZE>>>(halton_states, rng_states);

        // free index for next available position in tree_a and tree_b
        int h_free_index[2] = {1, num_goals};
        cudaMemcpyToSymbol(atomic_free_index, &h_free_index, sizeof(int) * 2);
        cudaMemcpyToSymbol(nodes_size, &h_free_index, sizeof(int) * 2);
        
        // initialize completed_nodes counter
        int h_completed_nodes[2] = {1, num_goals}; // start and goals are already written
        cudaMemcpyToSymbol(completed_nodes, &h_completed_nodes, sizeof(int) * 2);
        
        // allocate for obstacles
        ppln::collision::Environment<float> *env;
        setup_environment_on_device(env, h_environment);
        cudaCheckError(cudaGetLastError());

        const std::size_t concon_scratch_slots =
            static_cast<std::size_t>(settings.num_new_configs) *
            static_cast<std::size_t>(settings.max_concon_nodes);
        cudaMalloc(
            &concon_sphere_pos_scratch,
            concon_scratch_slots * Collision::fine_sphere_count *
                Collision::batch_size * 3 * sizeof(float)
        );
        cudaMalloc(
            &concon_sphere_pos_approx_scratch,
            concon_scratch_slots * Collision::approximate_sphere_count *
                Collision::batch_size * 3 * sizeof(float)
        );
        cudaMalloc(
            &concon_link_cc_scratch,
            concon_scratch_slots * Collision::joint_flag_stride *
                Collision::batch_size * sizeof(int)
        );
        cudaMalloc(
            &concon_transform_scratch,
            concon_scratch_slots * Collision::batch_size *
                Collision::transform_slots * 16 * sizeof(float)
        );
        cudaCheckError(cudaGetLastError());
        
        // Setup pinned memory for signaling
        int *h_solved;
        int current_samples[2];
        int h_solved_iters = -1;
        cudaMallocHost(&h_solved, sizeof(int));
        *h_solved = -1;

        
        auto copy_start_time = std::chrono::steady_clock::now();
        // add start to tree_a and goals to tree_b
        cudaMemcpy((void *)nodes[0], start.data(), config_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)parents[0], &start_index, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)nodes[1], goals.data(), config_size * num_goals, cudaMemcpyHostToDevice);
        std::vector<int> parents_b_init(num_goals);
        iota(parents_b_init.begin(), parents_b_init.end(), 0); // consecutive integers from 0 ... num_goals - 1
        cudaMemcpy((void *)parents[1], parents_b_init.data(), sizeof(int) * num_goals, cudaMemcpyHostToDevice);
        const int start_ready = 1;
        std::vector<int> goals_ready(num_goals, 1);
        cudaMemcpy(node_ready[0],&start_ready,sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(node_ready[1],goals_ready.data(),sizeof(int) * num_goals,cudaMemcpyHostToDevice);
        // root tangent basis 초기화
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            // Initial Tangent Space Bank
            // start tree: q_start 하나 → TS 하나
            // goal tree: 각 initial goal → TS 하나
            init_root_ts_banks<Robot><<<1 + num_goals, 1>>>(
                d_nodes,
                d_ts_root_node_idx,
                d_ts_bases,
                d_ts_ready,
                d_node_ts_id,
                d_node_ts_q,
                d_ts_node_count,
                d_ts_lane_head,
                d_node_next_in_ts,
                1,
                num_goals
            );

            cudaDeviceSynchronize();
            cudaCheckError(cudaGetLastError());

            int h_initial_ts_count[2] = {1,num_goals};

            cudaMemcpy(ts_count,h_initial_ts_count,2 * sizeof(int),cudaMemcpyHostToDevice);

            cudaCheckError(cudaGetLastError());
        }
        res.copy_ns = get_elapsed_nanoseconds(copy_start_time);

        auto kernel_start_time = std::chrono::steady_clock::now();
        const int concon_threads_per_block =
            CONCON_COLLISION_THREADS_PER_EDGE * settings.max_concon_nodes;
        if (settings.trace_trees) {
            rrtc<Robot, true><<<settings.num_new_configs, concon_threads_per_block>>> (
                d_nodes,
                d_parents,
                d_node_ready,
                // 새 Tree → TS 정보
                d_node_ts_id,
                d_node_ts_q,

                // 새 TSBank
                ts_count,
                d_ts_root_node_idx,
                d_ts_bases,
                d_ts_ready,
                d_ts_node_count,
                d_ts_lane_head,
                d_node_next_in_ts,
                d_radii,
                halton_states,
                rng_states,
                env,
                concon_sphere_pos_scratch,
                concon_sphere_pos_approx_scratch,
                concon_link_cc_scratch,
                concon_transform_scratch
            );
        } else {
            rrtc<Robot, false><<<settings.num_new_configs, concon_threads_per_block>>> (
                d_nodes,
                d_parents,
                d_node_ready,
                // 새 Tree → TS 정보
                d_node_ts_id,
                d_node_ts_q,

                // 새 TSBank
                ts_count,
                d_ts_root_node_idx,
                d_ts_bases,
                d_ts_ready,
                d_ts_node_count,
                d_ts_lane_head,
                d_node_next_in_ts,
                d_radii,
                halton_states,
                rng_states,
                env,
                concon_sphere_pos_scratch,
                concon_sphere_pos_approx_scratch,
                concon_link_cc_scratch,
                concon_transform_scratch
            );
        }

        // kernel launch 자체가 정상인지
        cudaCheckError(cudaGetLastError());

        // kernel 실행 중 illegal memory access가 있었는지
        cudaCheckError(cudaDeviceSynchronize());

        res.kernel_ns = get_elapsed_nanoseconds(kernel_start_time);

        // get data from device
        copy_start_time = std::chrono::steady_clock::now();
        cudaMemcpyFromSymbol(current_samples, atomic_free_index, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(h_solved, solved, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&h_solved_iters, solved_iters, sizeof(int), 0, cudaMemcpyDeviceToHost);
        for (int tree = 0; tree < 2; tree++) {
            current_samples[tree] = std::clamp(
                current_samples[tree],
                0,
                settings.max_samples
            );
        }
        res.copy_ns += get_elapsed_nanoseconds(copy_start_time);

        cudaCheckError(cudaGetLastError());

        // add data to result struct
        if (*h_solved!=1) *h_solved=0;
        res.start_tree_size = current_samples[0];
        res.goal_tree_size = current_samples[1];
        if (*h_solved) {
            int h_path_size[2];
            std::vector<float> h_paths[2];
            float h_cost;
            int h_reached_goal_idx;
            cudaMemcpyFromSymbol(h_path_size, path_size, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
            for (int tree = 0; tree < 2; ++tree) {
                if (h_path_size[tree] < 0
                    || h_path_size[tree] > MAX_PATH_NODES) {
                    throw std::runtime_error(
                        "pRRTC device path size is outside its valid range"
                    );
                }
                h_paths[tree].resize(
                    static_cast<std::size_t>(h_path_size[tree]) * dim
                );
                if (!h_paths[tree].empty()) {
                    cudaMemcpyFromSymbol(
                        h_paths[tree].data(),
                        path,
                        sizeof(float) * h_paths[tree].size(),
                        sizeof(float) * static_cast<std::size_t>(tree)
                            * MAX_PATH_STORAGE,
                        cudaMemcpyDeviceToHost
                    );
                }
            }
            cudaMemcpyFromSymbol(&h_cost, cost, sizeof(float), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&h_reached_goal_idx, reached_goal_idx, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaCheckError(cudaGetLastError());
            res.path.emplace_back(goals[h_reached_goal_idx]);
            typename Robot::Configuration config;
            for (int i = h_path_size[1] - 1; i >= 0; i--) {
                std::copy_n(h_paths[1].data() + i * dim, dim, config.begin());
                res.path.emplace_back(config);
            }
            for (int i = 0; i < h_path_size[0]; i++) {
                std::copy_n(h_paths[0].data() + i * dim, dim, config.begin());
                res.path.emplace_back(config);
            }
            res.path.emplace_back(start);
            res.cost = configuration_space_path_arclength<Robot>(res.path);
            res.path_length = (h_path_size[0] + h_path_size[1]);
        }
        res.solved = (*h_solved) != 0;
        res.iters = h_solved_iters;
        if (settings.trace_trees) {
            copy_start_time = std::chrono::steady_clock::now();
            copy_tree_trace_to_result(
                res,
                nodes,
                parents,
                node_ready,
                current_samples
            );
            if (res.solved) {
                cudaMemcpyFromSymbol(
                    &res.connection_tree_id,
                    connection_tree_id,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );
                cudaMemcpyFromSymbol(
                    &res.connection_node_idx,
                    connection_node_idx,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );
                cudaMemcpyFromSymbol(
                    &res.connection_other_tree_id,
                    connection_other_tree_id,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );
                cudaMemcpyFromSymbol(
                    &res.connection_other_node_idx,
                    connection_other_node_idx,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );
                fill_solution_trace(res);
            }
            res.copy_ns += get_elapsed_nanoseconds(copy_start_time);
            cudaCheckError(cudaGetLastError());
        }
        
        cleanup_environment_on_device(env, h_environment);
        reset_device_variables();
        cudaFree((void *)nodes[0]);
        cudaFree((void *)nodes[1]);
        cudaFree((void *)parents[0]);
        cudaFree((void *)parents[1]);

        for (int i = 0; i < 2; i++) {

        if (ts_root_node_idx[i] != nullptr) {
            cudaFree(ts_root_node_idx[i]);
        }

        if (ts_ready[i] != nullptr) {
            cudaFree(ts_ready[i]);
        }

        if (ts_bases[i] != nullptr) {
            cudaFree(ts_bases[i]);
        }

        if (node_ts_id[i] != nullptr) {
            cudaFree(node_ts_id[i]);
        }

        if (node_ts_q[i] != nullptr) {
            cudaFree(node_ts_q[i]);
        }

        if (ts_node_count[i] != nullptr) {
            cudaFree(ts_node_count[i]);
        }

        if (ts_lane_head[i] != nullptr) {
            cudaFree(ts_lane_head[i]);
        }

        if (node_next_in_ts[i] != nullptr) {
            cudaFree(node_next_in_ts[i]);
        }
    }
        cudaFree((void *)node_ready[0]);
        cudaFree((void *)node_ready[1]);
        cudaFree((void *)radii[0]);
        cudaFree((void *)radii[1]);
        cudaFree(rng_states);
        cudaFree(halton_states);
        cudaFree(d_nodes);
        cudaFree(d_parents);
        cudaFree(d_node_ready);
        cudaFree(d_radii);
        cudaFree(d_node_ts_id);
        cudaFree(d_node_ts_q);

        cudaFree(ts_count);

        cudaFree(d_ts_root_node_idx);
        cudaFree(d_ts_bases);
        cudaFree(d_ts_ready);
        cudaFree(d_ts_node_count);
        cudaFree(d_ts_lane_head);
        cudaFree(d_node_next_in_ts);
        cudaFree(concon_sphere_pos_scratch);
        cudaFree(concon_sphere_pos_approx_scratch);
        cudaFree(concon_link_cc_scratch);
        cudaFree(concon_transform_scratch);
        cudaFreeHost(h_solved);
        cudaCheckError(cudaGetLastError());
        res.wall_ns = get_elapsed_nanoseconds(start_time);
        cudaDeviceReset();
        return res;
    }

    //template PlannerResult<typename ppln::robots::Sphere> solve<ppln::robots::Sphere>(std::array<float, 3>&, std::vector<std::array<float, 3>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::Panda> solve<ppln::robots::Panda>(std::array<float, 7>&, std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::Fetch> solve<ppln::robots::Fetch>(std::array<float, 8>&, std::vector<std::array<float, 8>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::Baxter> solve<ppln::robots::Baxter>(std::array<float, 14>&, std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::FfwSg2> solve<ppln::robots::FfwSg2>(std::array<float, 15>&, std::vector<std::array<float, 15>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::FfwSg2Mobility> solve<ppln::robots::FfwSg2Mobility>(std::array<float, 18>&, std::vector<std::array<float, 18>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::FfwSg2Single> solve<ppln::robots::FfwSg2Single>(std::array<float, 8>&, std::vector<std::array<float, 8>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::G1> solve<ppln::robots::G1>(std::array<float, 35>&, std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, pRRTC_settings&);

}

#include "Planners.hh"
#include "Robots.hh"
#include "RobotCollisionTraits.hh"
#include "JointLimits.cuh"
#include "utils.cuh"
#include "PATACON_settings.hh"
#include "src/collision/environment.hh"
#include "src/robots/panda.cuh"
#include "src/robots/franka_collision.cuh"
#include "src/robots/franka_constraint.cuh"
#include "src/robots/ffw_sg2.cuh"
#include "src/robots/ffw_sg2_attached_object_collision.cuh"
#include "src/robots/ffw_sg2_mobility.cuh"
#include "src/robots/ffw_sg2_constraint.cuh"
#include "src/robots/ffw_sg2_mobility_constraint.cuh"
#include "src/robots/ffw_sg2_mobility_com_constraint.cuh"
#include "src/robots/g1_collision.cuh"
#include "src/robots/g1_attached_object_collision.cuh"
#include "src/robots/g1_constraint.cuh"
#include "src/robots/igris_c_collision.cuh"
#include "src/robots/igris_c_constraint.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>



/*
PATACON: Each block works to add a config to the tree (either start or goal depending on balance)
*/


namespace PATACON {
    using namespace ppln;

    __global__ void project_g1_configuration_kernel(
        float *configuration,
        constraints::G1ConstraintParameters parameters,
        bool rigid_orientation,
        int max_iterations,
        float alpha,
        float damping,
        float maximum_step,
        bool *success
    ) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            *success = collision::g1_project_configuration(
                configuration,
                parameters,
                rigid_orientation,
                max_iterations,
                alpha,
                damping,
                maximum_step
            );
        }
    }

    namespace {
        bool cuda_device_reset_enabled = true;
        bool persistent_workspace_enabled = false;
        double time_limit_seconds = 0.0;
        bool time_limit_counts_kernel_only = false;
        float *persistent_g1_projection_configuration = nullptr;
        bool *persistent_g1_projection_success = nullptr;

        void release_g1_projection_workspace() noexcept {
            if (persistent_g1_projection_success != nullptr) {
                cudaFree(persistent_g1_projection_success);
                persistent_g1_projection_success = nullptr;
            }
            if (persistent_g1_projection_configuration != nullptr) {
                cudaFree(persistent_g1_projection_configuration);
                persistent_g1_projection_configuration = nullptr;
            }
        }
    }

    void set_cuda_device_reset_enabled(const bool enabled) {
        cuda_device_reset_enabled = enabled;
    }

    void set_persistent_workspace_enabled(const bool enabled) {
        persistent_workspace_enabled = enabled;
    }

    void set_time_limit_seconds(const double seconds) {
        if (!std::isfinite(seconds) || seconds < 0.0) {
            throw std::invalid_argument(
                "PATACON time limit must be finite and nonnegative"
            );
        }
        time_limit_seconds = seconds;
    }

    bool project_g1_configuration(
        robots::G1::Configuration &configuration,
        const PATACON_settings &settings
    ) {
        float *device_configuration = nullptr;
        bool *device_success = nullptr;
        if (persistent_workspace_enabled) {
            if (persistent_g1_projection_configuration == nullptr) {
                cudaCheckError(cudaMalloc(
                    &persistent_g1_projection_configuration,
                    sizeof(float) * configuration.size()
                ));
                cudaCheckError(cudaMalloc(
                    &persistent_g1_projection_success,
                    sizeof(bool)
                ));
            }
            device_configuration = persistent_g1_projection_configuration;
            device_success = persistent_g1_projection_success;
        } else {
            cudaCheckError(cudaMalloc(
                &device_configuration,
                sizeof(float) * configuration.size()
            ));
            cudaCheckError(cudaMalloc(&device_success, sizeof(bool)));
        }
        cudaCheckError(cudaMemcpy(
            device_configuration,
            configuration.data(),
            sizeof(float) * configuration.size(),
            cudaMemcpyHostToDevice
        ));

        project_g1_configuration_kernel<<<1, 1>>>(
            device_configuration,
            settings.g1_constraints,
            settings.rigid_orientation,
            settings.projection_max_iters,
            settings.projection_alpha,
            settings.projection_damping,
            settings.projection_max_step,
            device_success
        );
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize());

        bool success = false;
        cudaCheckError(cudaMemcpy(
            &success,
            device_success,
            sizeof(bool),
            cudaMemcpyDeviceToHost
        ));
        if (success) {
            cudaCheckError(cudaMemcpy(
                configuration.data(),
                device_configuration,
                sizeof(float) * configuration.size(),
                cudaMemcpyDeviceToHost
            ));
        }
        if (!persistent_workspace_enabled) {
            cudaCheckError(cudaFree(device_success));
            cudaCheckError(cudaFree(device_configuration));
        }
        return success;
    }

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
    static_assert(
        robots::CollisionTraits<robots::IgrisC>::batch_size ==
            collision::IGRIS_C_COLLISION_BATCH_SIZE &&
        robots::CollisionTraits<robots::IgrisC>::fine_sphere_count ==
            collision::IGRIS_C_SPHERE_COUNT &&
        robots::CollisionTraits<robots::IgrisC>::approximate_sphere_count ==
            collision::IGRIS_C_APPROX_SPHERE_COUNT &&
        robots::CollisionTraits<robots::IgrisC>::joint_flag_stride ==
            collision::IGRIS_C_JOINT_FLAG_STRIDE &&
        robots::CollisionTraits<robots::IgrisC>::transform_slots ==
            collision::IGRIS_C_TRANSFORM_SLOTS,
        "IGRIS-C collision traits differ from the generated collision code"
    );
    __device__ volatile int solved = 0;
    __device__ volatile int atomic_free_index[2]; // separate for tree_a and tree_b
    __device__ volatile int nodes_size[2];
    __device__ volatile int completed_nodes[2]; // track completed nodes for each tree
    // Ready entries are request generations, not booleans. Persistent
    // replanning can therefore invalidate old nodes without a full memset.
    __device__ int current_search_generation = 1;
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
    __constant__ PATACON_settings d_settings;
    __constant__ unsigned long long patacon_time_limit_ns;

    __device__ __forceinline__ unsigned long long global_timer_ns() {
        unsigned long long value;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    }

    enum DiagnosticCounter : int {
        DIAG_EXTEND_ATTEMPTS = 0,
        DIAG_EXTEND_BACKTRACKING_FLIPS,
        DIAG_EXTEND_EM_STOPS,
        DIAG_EXTEND_ANCHOR_PROJECTION_STOPS,
        DIAG_EXTEND_EDGE_PROJECTION_STOPS,
        DIAG_EXTEND_COLLISION_STOPS,
        DIAG_EXTEND_FULL_SUCCESSES,
        DIAG_CONNECT_ATTEMPTS,
        DIAG_CONNECT_CHUNKS,
        DIAG_CONNECT_INVALID_TANGENT_SPACES,
        DIAG_CONNECT_TANGENT_DIRECTION_STOPS,
        DIAG_CONNECT_EM_STOPS,
        DIAG_CONNECT_ANCHOR_PROJECTION_STOPS,
        DIAG_CONNECT_EDGE_PROJECTION_STOPS,
        DIAG_CONNECT_PROGRESS_STOPS,
        DIAG_CONNECT_COLLISION_STOPS,
        DIAG_CONNECT_SUCCESSES,
        DIAG_CONNECT_FAILURES,
        DIAGNOSTIC_COUNTER_COUNT
    };
    __device__ unsigned long long diagnostic_counters[
        DIAGNOSTIC_COUNTER_COUNT
    ];

    __device__ __forceinline__ void diagnostic_increment(
        DiagnosticCounter counter
    ) {
        if (d_settings.collect_diagnostics) {
            atomicAdd(&diagnostic_counters[static_cast<int>(counter)], 1ULL);
        }
    }

    constexpr int MAX_GRANULARITY = 16;
    constexpr int CONCON_COLLISION_THREADS_PER_EDGE = 4 * MAX_GRANULARITY;
    constexpr int MAX_PARALLEL_CONCON_EDGES = 5;
    constexpr int MAX_CONCON_NODE_ANCHORS =
        MAX_PARALLEL_CONCON_EDGES + 1;
    constexpr int MAX_THREADS_PER_BLOCK =
        CONCON_COLLISION_THREADS_PER_EDGE * MAX_PARALLEL_CONCON_EDGES;

    // PATACON projected motion shared buffer용
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

    template <>
    struct TangentSpaceTraits<robots::IgrisC> {
        static constexpr bool enabled = true;
        static constexpr int max_tangent_dim = collision::IGRIS_C_TANGENT_DIM;
        static constexpr int basis_size = collision::IGRIS_C_TANGENT_BASIS_SIZE;
    };

    template <>
    struct TangentSpaceTraits<robots::FrankaSingle> {
        static constexpr bool enabled = true;
        static constexpr int max_tangent_dim =
            collision::FRANKA_SINGLE_MAX_TANGENT_DIM;
        static constexpr int basis_size =
            collision::FRANKA_SINGLE_TANGENT_BASIS_SIZE;
    };

    template <>
    struct TangentSpaceTraits<robots::Franka> {
        static constexpr bool enabled = true;
        static constexpr int max_tangent_dim =
            collision::FRANKA_DUAL_MAX_TANGENT_DIM;
        static constexpr int basis_size =
            collision::FRANKA_DUAL_TANGENT_BASIS_SIZE;
    };

    template <typename Robot>
    __device__ __forceinline__ int patacon_active_tangent_dim() {
        if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
            return d_settings.rigid_orientation ? 7 : FFW_SG2_TANGENT_DIM;
        } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
            return FFW_SG2_MOBILITY_TANGENT_DIM;
        } else if constexpr (std::is_same_v<Robot, robots::G1>) {
            return collision::g1_tangent_dim(d_settings.rigid_orientation);
        } else if constexpr (std::is_same_v<Robot, robots::IgrisC>) {
            return collision::IGRIS_C_TANGENT_DIM;
        } else if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
            return d_settings.rigid_orientation ? 5 : 7;
        } else if constexpr (std::is_same_v<Robot, robots::Franka>) {
            return d_settings.rigid_orientation ? 6 : 8;
        }
        return 0;
    }


    // 일반 로봇은 모든 관절 가중치가 1
    template <typename Robot>
    __device__ __forceinline__ float patacon_joint_distance_weight(
        int joint_index
    ) {
        return 1.0f;
    }


    // FFW-SG2의 0번 좌표는 lift_joint
    template <>
    __device__ __forceinline__ float
    patacon_joint_distance_weight<robots::FfwSg2>(
        int joint_index
    ) {
        return joint_index == 0
            ? d_settings.lift_distance_weight
            : 1.0f;
    }


    template <>
    __device__ __forceinline__ float
    patacon_joint_distance_weight<robots::FfwSg2Mobility>(
        int joint_index
    ) {
        return joint_index == 3
            ? d_settings.lift_distance_weight
            : 1.0f;
    }

    template <typename Robot>
    __device__ __forceinline__ float patacon_sq_config_distance(
        const float* q_a,
        const float* q_b
    ) {
        float result = 0.0f;

        #pragma unroll
        for (int i = 0; i < Robot::dimension; i++) {
            const float weight =
                patacon_joint_distance_weight<Robot>(i);

            const float weighted_diff =
                weight * (q_a[i] - q_b[i]);

            result += weighted_diff * weighted_diff;
        }

        return result;
    }


    template <typename Robot>
    __device__ __forceinline__ float patacon_config_distance(
        const float* q_a,
        const float* q_b
    ) {
        return sqrtf(
            patacon_sq_config_distance<Robot>(q_a, q_b)
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
            (std::is_same_v<Robot, robots::G1> ||
             std::is_same_v<Robot, robots::IgrisC>)
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
    __global__ void init_rng(
        curandState* states,
        unsigned long long seed,
        int num_rng_states
    ) {
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

    // Keep fixed world primitives resident across replans. update() transfers
    // only arrays whose host contents changed (normally the moving sphere).
    class PersistentDeviceEnvironment {
    public:
        ~PersistentDeviceEnvironment() {
            release();
        }

        ppln::collision::Environment<float> *update(
            const ppln::collision::Environment<float> &host
        ) {
            if (!matches(host)) {
                release();
                allocate(host);
            } else {
                update_buffer(device_spheres_, host.spheres, host.num_spheres,
                              cached_spheres_);
                update_buffer(device_capsules_, host.capsules,
                              host.num_capsules, cached_capsules_);
                update_buffer(device_z_aligned_capsules_,
                              host.z_aligned_capsules,
                              host.num_z_aligned_capsules,
                              cached_z_aligned_capsules_);
                update_buffer(device_cylinders_, host.cylinders,
                              host.num_cylinders, cached_cylinders_);
                update_buffer(device_cuboids_, host.cuboids,
                              host.num_cuboids, cached_cuboids_);
                update_buffer(device_z_aligned_cuboids_,
                              host.z_aligned_cuboids,
                              host.num_z_aligned_cuboids,
                              cached_z_aligned_cuboids_);
            }
            return device_environment_;
        }

    private:
        static bool same_device_primitive(
            const ppln::collision::Sphere<float> &left,
            const ppln::collision::Sphere<float> &right
        ) {
            return left.min_distance == right.min_distance
                && left.x == right.x && left.y == right.y
                && left.z == right.z && left.r == right.r;
        }

        static bool same_device_primitive(
            const ppln::collision::Cylinder<float> &left,
            const ppln::collision::Cylinder<float> &right
        ) {
            return left.min_distance == right.min_distance
                && left.x1 == right.x1 && left.y1 == right.y1
                && left.z1 == right.z1 && left.xv == right.xv
                && left.yv == right.yv && left.zv == right.zv
                && left.r == right.r && left.rdv == right.rdv;
        }

        static bool same_device_primitive(
            const ppln::collision::Cuboid<float> &left,
            const ppln::collision::Cuboid<float> &right
        ) {
            return left.min_distance == right.min_distance
                && left.x == right.x && left.y == right.y
                && left.z == right.z
                && left.axis_1_x == right.axis_1_x
                && left.axis_1_y == right.axis_1_y
                && left.axis_1_z == right.axis_1_z
                && left.axis_2_x == right.axis_2_x
                && left.axis_2_y == right.axis_2_y
                && left.axis_2_z == right.axis_2_z
                && left.axis_3_x == right.axis_3_x
                && left.axis_3_y == right.axis_3_y
                && left.axis_3_z == right.axis_3_z
                && left.axis_1_r == right.axis_1_r
                && left.axis_2_r == right.axis_2_r
                && left.axis_3_r == right.axis_3_r;
        }

        template <typename Primitive>
        static void update_buffer(
            Primitive *device,
            const Primitive *host,
            unsigned int count,
            std::vector<Primitive> &cached
        ) {
            const std::size_t bytes =
                static_cast<std::size_t>(count) * sizeof(Primitive);
            if (cached.size() != count) {
                if (bytes > 0) {
                    cudaCheckError(cudaMemcpy(
                        device, host, bytes, cudaMemcpyHostToDevice
                    ));
                }
                cached.assign(host, host + count);
                return;
            }

            // Primitive names live only on the host. Compare the numeric
            // collision payload and upload only elements that changed, so a
            // moving sphere does not re-upload static world geometry.
            for (unsigned int index = 0; index < count; ++index) {
                if (same_device_primitive(cached[index], host[index])) {
                    continue;
                }
                cudaCheckError(cudaMemcpy(
                    device + index,
                    host + index,
                    sizeof(Primitive),
                    cudaMemcpyHostToDevice
                ));
                cached[index] = host[index];
            }
        }

        template <typename Primitive>
        static void allocate_buffer(
            Primitive *&device,
            const Primitive *host,
            unsigned int count,
            std::vector<Primitive> &cached
        ) {
            if (count == 0) {
                device = nullptr;
                cached.clear();
                return;
            }
            cudaCheckError(cudaMalloc(
                &device,
                static_cast<std::size_t>(count) * sizeof(Primitive)
            ));
            cached.clear();
            update_buffer(device, host, count, cached);
        }

        template <typename Primitive>
        static void free_buffer(Primitive *&device) noexcept {
            if (device != nullptr) {
                cudaFree(device);
                device = nullptr;
            }
        }

        bool matches(
            const ppln::collision::Environment<float> &host
        ) const {
            return device_environment_ != nullptr
                && num_spheres_ == host.num_spheres
                && num_capsules_ == host.num_capsules
                && num_z_aligned_capsules_ == host.num_z_aligned_capsules
                && num_cylinders_ == host.num_cylinders
                && num_cuboids_ == host.num_cuboids
                && num_z_aligned_cuboids_ == host.num_z_aligned_cuboids;
        }

        void allocate(const ppln::collision::Environment<float> &host) {
            num_spheres_ = host.num_spheres;
            num_capsules_ = host.num_capsules;
            num_z_aligned_capsules_ = host.num_z_aligned_capsules;
            num_cylinders_ = host.num_cylinders;
            num_cuboids_ = host.num_cuboids;
            num_z_aligned_cuboids_ = host.num_z_aligned_cuboids;

            cudaCheckError(cudaMalloc(
                &device_environment_,
                sizeof(ppln::collision::Environment<float>)
            ));
            cudaCheckError(cudaMemset(
                device_environment_, 0,
                sizeof(ppln::collision::Environment<float>)
            ));
            allocate_buffer(device_spheres_, host.spheres, host.num_spheres,
                            cached_spheres_);
            allocate_buffer(device_capsules_, host.capsules,
                            host.num_capsules, cached_capsules_);
            allocate_buffer(device_z_aligned_capsules_,
                            host.z_aligned_capsules,
                            host.num_z_aligned_capsules,
                            cached_z_aligned_capsules_);
            allocate_buffer(device_cylinders_, host.cylinders,
                            host.num_cylinders, cached_cylinders_);
            allocate_buffer(device_cuboids_, host.cuboids,
                            host.num_cuboids, cached_cuboids_);
            allocate_buffer(device_z_aligned_cuboids_,
                            host.z_aligned_cuboids,
                            host.num_z_aligned_cuboids,
                            cached_z_aligned_cuboids_);

#define INSTALL_ENV_FIELD(field, device_value, count_field, count_value)    \
            cudaCheckError(cudaMemcpy(                                      \
                &(device_environment_->field), &device_value,               \
                sizeof(device_value), cudaMemcpyHostToDevice                \
            ));                                                              \
            cudaCheckError(cudaMemcpy(                                      \
                &(device_environment_->count_field), &count_value,           \
                sizeof(count_value), cudaMemcpyHostToDevice                  \
            ))
            INSTALL_ENV_FIELD(spheres, device_spheres_, num_spheres,
                              num_spheres_);
            INSTALL_ENV_FIELD(capsules, device_capsules_, num_capsules,
                              num_capsules_);
            INSTALL_ENV_FIELD(z_aligned_capsules,
                              device_z_aligned_capsules_,
                              num_z_aligned_capsules,
                              num_z_aligned_capsules_);
            INSTALL_ENV_FIELD(cylinders, device_cylinders_, num_cylinders,
                              num_cylinders_);
            INSTALL_ENV_FIELD(cuboids, device_cuboids_, num_cuboids,
                              num_cuboids_);
            INSTALL_ENV_FIELD(z_aligned_cuboids,
                              device_z_aligned_cuboids_,
                              num_z_aligned_cuboids,
                              num_z_aligned_cuboids_);
#undef INSTALL_ENV_FIELD
        }

        void release() noexcept {
            free_buffer(device_spheres_);
            free_buffer(device_capsules_);
            free_buffer(device_z_aligned_capsules_);
            free_buffer(device_cylinders_);
            free_buffer(device_cuboids_);
            free_buffer(device_z_aligned_cuboids_);
            if (device_environment_ != nullptr) {
                cudaFree(device_environment_);
                device_environment_ = nullptr;
            }
            num_spheres_ = 0;
            num_capsules_ = 0;
            num_z_aligned_capsules_ = 0;
            num_cylinders_ = 0;
            num_cuboids_ = 0;
            num_z_aligned_cuboids_ = 0;
        }

        ppln::collision::Environment<float> *device_environment_ = nullptr;
        ppln::collision::Sphere<float> *device_spheres_ = nullptr;
        ppln::collision::Capsule<float> *device_capsules_ = nullptr;
        ppln::collision::Capsule<float> *device_z_aligned_capsules_ = nullptr;
        ppln::collision::Cylinder<float> *device_cylinders_ = nullptr;
        ppln::collision::Cuboid<float> *device_cuboids_ = nullptr;
        ppln::collision::Cuboid<float> *device_z_aligned_cuboids_ = nullptr;
        unsigned int num_spheres_ = 0;
        unsigned int num_capsules_ = 0;
        unsigned int num_z_aligned_capsules_ = 0;
        unsigned int num_cylinders_ = 0;
        unsigned int num_cuboids_ = 0;
        unsigned int num_z_aligned_cuboids_ = 0;
        std::vector<ppln::collision::Sphere<float>> cached_spheres_;
        std::vector<ppln::collision::Capsule<float>> cached_capsules_;
        std::vector<ppln::collision::Capsule<float>> cached_z_aligned_capsules_;
        std::vector<ppln::collision::Cylinder<float>> cached_cylinders_;
        std::vector<ppln::collision::Cuboid<float>> cached_cuboids_;
        std::vector<ppln::collision::Cuboid<float>> cached_z_aligned_cuboids_;
    };

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

    // PATACON motion generation / projection wrapper
    // Generic robot: straight-line motion만 생성하고 projection은 하지 않는다.
    // FfwSg2: straight-line motion 생성 후 analytic-Jacobian ParallelProject 수행.

    template <typename Robot>
    __device__ __forceinline__ bool patacon_project_motion(
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
            "Robot dimension exceeds PATACON motion segment buffer"
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
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::FfwSg2>(
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
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::FfwSg2Mobility>(
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
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::G1>(
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
            d_settings.rigid_orientation,
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

    template <>
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::IgrisC>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        static constexpr int dim = ppln::robots::IgrisC::dimension;
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

        return ppln::collision::igris_c_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            d_settings.igris_c_constraints,
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

    template <>
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::FrankaSingle>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        constexpr int dim = ppln::robots::FrankaSingle::dimension;
        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;
        if (tid < dim) motion_segment[tid] = q_start[tid];
        if (waypoint <= d_settings.granularity) {
            for (int joint = lane; joint < dim; joint += 4) {
                motion_segment[waypoint * dim + joint] =
                    q_start[joint] + static_cast<float>(waypoint) * q_step[joint];
            }
        }
        __syncthreads();
        return ppln::collision::franka_single_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            d_settings.franka_constraints,
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
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::Franka>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        constexpr int dim = ppln::robots::Franka::dimension;
        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;
        if (tid < dim) motion_segment[tid] = q_start[tid];
        if (waypoint <= d_settings.granularity) {
            for (int joint = lane; joint < dim; joint += 4) {
                motion_segment[waypoint * dim + joint] =
                    q_start[joint] + static_cast<float>(waypoint) * q_step[joint];
            }
        }
        __syncthreads();
        return ppln::collision::franka_dual_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            d_settings.franka_constraints,
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

    template <typename Robot>
    __device__ __forceinline__ float patacon_config_distance_from_volatile(
        volatile const float *q_a,
        const float *q_b
    ) {
        float result = 0.0f;

        #pragma unroll
        for (int joint = 0; joint < Robot::dimension; joint++) {
            const float weight =
                patacon_joint_distance_weight<Robot>(joint);
            const float weighted_diff =
                weight * (q_a[joint] - q_b[joint]);
            result += weighted_diff * weighted_diff;
        }

        return sqrtf(result);
    }

    template <typename Robot>
    __device__ __noinline__ bool patacon_project_prebuilt_motion(
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        int waypoint_count,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid,
        bool use_smoothness,
        float smoothness_threshold,
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
                smoothness_threshold,
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
                    smoothness_threshold,
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
                smoothness_threshold,
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
                d_settings.rigid_orientation,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.beta,
                d_settings.gamma,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else if constexpr (std::is_same_v<Robot, robots::IgrisC>) {
            return ppln::collision::igris_c_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                d_settings.igris_c_constraints,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.beta,
                d_settings.gamma,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
            return ppln::collision::franka_single_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                d_settings.franka_constraints,
                d_settings.rigid_orientation,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else if constexpr (std::is_same_v<Robot, robots::Franka>) {
            return ppln::collision::franka_dual_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                d_settings.franka_constraints,
                d_settings.rigid_orientation,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                smoothness_threshold,
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
    patacon_project_concon_node_anchors(
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
            "Robot dimension exceeds PATACON node motion buffer"
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

        const float node_smoothness_threshold =
            static_cast<float>(d_settings.granularity) *
            d_settings.projection_smoothness_threshold;
        return patacon_project_prebuilt_motion<Robot>(
            node_motion,
            node_motion_next,
            edge_count,
            projection_valid,
            projection_prog,
            projection_success,
            tid,
            d_settings.projection_smoothness,
            node_smoothness_threshold
        );
    }

    template <typename Robot>
    __device__ __noinline__ void
    patacon_project_concon_edge_segments_from_node_anchors(
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

        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;
        const int total_waypoint_count =
            edge_count * d_settings.granularity;

        // Keep the ConCon prefix continuous while including each projected
        // node (waypoints granularity, 2 * granularity, ...) in smoothing.
        volatile float *prefix_motion = edge_motion_segment_next;
        volatile float *prefix_motion_next = edge_motion_segments;

        if (tid < dim) {
            prefix_motion[tid] = node_anchors[tid];
        }
        if (waypoint <= total_waypoint_count) {
            const int prefix_edge_slot =
                (waypoint - 1) / d_settings.granularity;
            const int local_waypoint =
                (waypoint - 1) % d_settings.granularity + 1;
            const float alpha =
                static_cast<float>(local_waypoint) /
                static_cast<float>(d_settings.granularity);

            for (int joint = lane; joint < dim; joint += 4) {
                const float q0 =
                    node_anchors[prefix_edge_slot * dim + joint];
                const float q1 =
                    node_anchors[(prefix_edge_slot + 1) * dim + joint];
                prefix_motion[waypoint * dim + joint] =
                    q0 + alpha * (q1 - q0);
            }
        }
        __syncthreads();

        const bool projection_good = patacon_project_prebuilt_motion<Robot>(
            prefix_motion,
            prefix_motion_next,
            total_waypoint_count,
            edge_projection_valid,
            edge_projection_prog,
            edge_projection_success,
            tid,
            d_settings.projection_smoothness,
            d_settings.projection_smoothness_threshold,
            false
        );
        __syncthreads();

        if (tid == 0) {
            int completed_waypoints = projection_good
                ? total_waypoint_count
                : edge_projection_prog[0];
            if (completed_waypoints < 0) {
                completed_waypoints = 0;
            }
            if (completed_waypoints > total_waypoint_count) {
                completed_waypoints = total_waypoint_count;
            }
            first_projection_failure_edge[0] =
                completed_waypoints / d_settings.granularity;
        }
        __syncthreads();

        // Collision checking keeps one source duplicate per edge. Repack the
        // continuous projected prefix without changing the thread layout.
        const int source_edge_slot =
            edge_slot < edge_count ? edge_slot : edge_count - 1;
        volatile float *edge_motion =
            &edge_motion_segments[edge_slot * CONCON_MOTION_SEGMENT_STRIDE];
        const int edge_segment_values =
            (d_settings.granularity + 1) * dim;
        for (
            int value = edge_tid;
            value < edge_segment_values;
            value += CONCON_COLLISION_THREADS_PER_EDGE
        ) {
            const int local_waypoint = value / dim;
            const int joint = value - local_waypoint * dim;
            const int prefix_waypoint =
                source_edge_slot * d_settings.granularity + local_waypoint;
            edge_motion[local_waypoint * dim + joint] =
                prefix_motion[prefix_waypoint * dim + joint];
        }
        __syncthreads();

        // A projection backend may report success even when an already-valid
        // waypoint has drifted just outside a joint boundary.  Reject the
        // first affected edge before collision checking or tree insertion.
        // Checking the repacked segment also covers every interpolated
        // waypoint, rather than only the endpoint stored in the tree.
        if (edge_slot < edge_count) {
            for (
                int value = edge_tid;
                value < edge_segment_values;
                value += CONCON_COLLISION_THREADS_PER_EDGE
            ) {
                const int joint = value % dim;
                if (!planning::joint_value_within_limits<Robot>(
                        edge_motion[value],
                        joint
                    )) {
                    atomicMin(
                        (int *)&first_projection_failure_edge[0],
                        edge_slot
                    );
                }
            }
        }
        __syncthreads();
    }

    template <typename Robot>
    __device__ __forceinline__ bool patacon_store_tangent_basis(
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
                    d_settings.rigid_orientation,
                    basis
                );
            } else if constexpr (std::is_same_v<Robot, robots::IgrisC>) {
                basis_ok = ppln::collision::igris_c_tangent_basis(
                    q,
                    d_settings.igris_c_constraints,
                    basis
                );
            } else if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
                basis_ok = ppln::collision::franka_single_tangent_basis(
                    q,
                    d_settings.franka_constraints,
                    d_settings.rigid_orientation,
                    basis
                );
            } else if constexpr (std::is_same_v<Robot, robots::Franka>) {
                basis_ok = ppln::collision::franka_dual_tangent_basis(
                    q,
                    d_settings.franka_constraints,
                    d_settings.rigid_orientation,
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
    __device__ __forceinline__ void patacon_sample_tangent_config(
        float *tree_nodes,
        float *ts_bases,
        const int *ts_root_node_indices,
        const int *ts_parent_ids,
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
            const int active_tangent_dim = patacon_active_tangent_dim<Robot>();

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
            }

            __syncthreads();

            // TB-RRT Section 3.5.1: a non-root TS samples only in the
            // half-space pointing away from its parent TS.  Since the
            // sample direction already lies in the new tangent space,
            // dot(dir, root - parent_root) has the same sign as the dot
            // product with the explicitly projected forward direction.
            if (tid == 0) {
                bool flip_direction = false;

                if (d_settings.prevent_ts_backtracking) {
                    const int parent_ts_id = ts_parent_ids[selected_ts_id];

                    if (parent_ts_id >= 0) {
                        const int parent_root_idx =
                            ts_root_node_indices[parent_ts_id];
                        const float *parent_q =
                            &tree_nodes[parent_root_idx * dim];
                        float direction_dot = 0.0f;

                        for (int joint = 0; joint < dim; joint++) {
                            direction_dot +=
                                ts_tangent_dir[joint] *
                                (base_q[joint] - parent_q[joint]);
                        }
                        flip_direction = direction_dot < 0.0f;
                    }
                }

                sdata[0] = flip_direction ? 1.0f : 0.0f;
                if (flip_direction) {
                    diagnostic_increment(DIAG_EXTEND_BACKTRACKING_FLIPS);
                }
            }

            __syncthreads();

            if (tid < dim) {
                float dir = ts_tangent_dir[tid];

                if (sdata[0] != 0.0f) {
                    dir = -dir;
                    ts_tangent_dir[tid] = dir;
                }

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
    __device__ __forceinline__ float patacon_constraint_error_norm(const float *q)
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
                d_settings.g1_constraints,
                d_settings.rigid_orientation
            );
        } else if constexpr (std::is_same_v<Robot, robots::IgrisC>) {
            return ppln::collision::igris_c_equality_residual_norm(
                q,
                d_settings.igris_c_constraints
            );
        } else if constexpr (
            std::is_same_v<Robot, robots::FrankaSingle> ||
            std::is_same_v<Robot, robots::Franka>
        ) {
            return ppln::collision::franka_constraint_error_norm<Robot>(
                q,
                d_settings.franka_constraints,
                d_settings.rigid_orientation
            );
        }

        return 0.0f;
    }

    template <typename Robot, int LaneStride = MAX_THREADS_PER_BLOCK>
    __global__ void init_root_ts_banks(
        float **nodes,
        int **ts_root_node_idx,
        int **ts_parent_id,
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
                // start/goal root TS는 부모 TS가 없다.
                ts_parent_id[tree][ts_idx] = -1;

                // root q에서 tangent basis 계산
                const bool basis_ok =patacon_store_tangent_basis<Robot>(&nodes[tree][node_idx * Robot::dimension],ts_bases[tree],ts_idx);

                if (basis_ok) {

                    // 이 node는 방금 만든 TS에 소속
                    node_ts_id[tree][node_idx] =ts_idx;

                    // root에서는 nominal TS q == 실제 tree q
                    for (int j = 0; j < Robot::dimension; j++) {
                        node_ts_q[tree][node_idx * Robot::dimension + j] =nodes[tree][node_idx * Robot::dimension + j];
                    }

                    // This slot may contain a linked list from an older
                    // request.  Initialize only the TS slot being published;
                    // request startup never has to clear the whole bank.
                    ts_node_count[tree][ts_idx] = 0;
                    for (int lane = 0; lane < MAX_THREADS_PER_BLOCK; ++lane) {
                        ts_lane_head[tree][
                            ts_idx * MAX_THREADS_PER_BLOCK + lane
                        ] = -1;
                    }

                    // 최초 root node를 이 TS의 thread 0 목록에 등록
                    ts_node_count[tree][ts_idx] = 1;
                    ts_lane_head[tree][ts_idx * LaneStride] =node_idx;
                    node_next_in_ts[tree][node_idx] = -1;
                }

                // 위 정보들이 global memory에 기록된 후 ready를 켜기 위해 사용
                __threadfence();

                if (basis_ok) {
                    ts_ready[tree][ts_idx] = current_search_generation;
                }
            }
        }
    }

    template <typename Robot>
    __device__ __forceinline__ float patacon_shared_config_distance(
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
                patacon_joint_distance_weight<Robot>(tid);

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
    __device__ __forceinline__ bool patacon_detailed_env_collision_check(
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
    __device__ __forceinline__ bool patacon_detailed_env_collision_check<ppln::robots::FfwSg2>(
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
    __device__ __forceinline__ bool patacon_detailed_env_collision_check<ppln::robots::FfwSg2Mobility>(
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
    __device__ __forceinline__ bool patacon_detailed_self_collision_check(
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
    __device__ __forceinline__ bool patacon_detailed_self_collision_check<ppln::robots::FfwSg2>(
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
    __device__ __forceinline__ bool patacon_detailed_self_collision_check<ppln::robots::FfwSg2Mobility>(
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
    __device__ __forceinline__ bool patacon_attached_object_collision_check_approx(
        const float *q,
        volatile float *sphere_pos_approx,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        if constexpr (
            std::is_same_v<Robot, ppln::robots::FrankaSingle> ||
            std::is_same_v<Robot, ppln::robots::Franka>
        ) {
            return ppln::collision::franka_attached_object_env_collision_check<Robot>(
                q, env, tid
            );
        }
        if constexpr (std::is_same_v<Robot, ppln::robots::FfwSg2>) {
            return ppln::collision::ffw_sg2_attached_object_collision_check_approx(
                q,
                sphere_pos_approx,
                env,
                tid,
                motion_cc_flag
            );
        }
        if constexpr (std::is_same_v<Robot, ppln::robots::G1>) {
            return ppln::collision::g1_attached_object_collision_check_approx(
                q, env, tid
            );
        }
        return true;
    }

    template <>
    __device__ __forceinline__ bool
    patacon_attached_object_collision_check_approx<ppln::robots::FfwSg2Mobility>(
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
    __device__ __forceinline__ bool patacon_attached_object_collision_check(
        const float *q,
        volatile float *sphere_pos,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        if constexpr (
            std::is_same_v<Robot, ppln::robots::FrankaSingle> ||
            std::is_same_v<Robot, ppln::robots::Franka>
        ) {
            return ppln::collision::franka_attached_object_env_collision_check<Robot>(
                q, env, tid
            );
        }
        if constexpr (std::is_same_v<Robot, ppln::robots::FfwSg2>) {
            return ppln::collision::ffw_sg2_attached_object_collision_check(
                q,
                sphere_pos,
                env,
                tid,
                motion_cc_flag
            );
        }
        if constexpr (std::is_same_v<Robot, ppln::robots::G1>) {
            return ppln::collision::g1_attached_object_collision_check(
                q, sphere_pos, env, tid
            );
        }
        return true;
    }

    template <>
    __device__ __forceinline__ bool
    patacon_attached_object_collision_check<ppln::robots::FfwSg2Mobility>(
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
    patacon_check_projected_edges_collision_parallel(
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
                not patacon_attached_object_collision_check_approx<Robot>(
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
                not patacon_detailed_env_collision_check<Robot>(
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
                not patacon_attached_object_collision_check<Robot>(
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
                not patacon_detailed_self_collision_check<Robot>(
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

    template <typename Robot>
    __global__ void patacon_validate_visualization_shortcut_edge(
        const float *node_anchors,
        volatile float *edge_motion_segments,
        volatile float *edge_motion_segment_next,
        volatile float *sphere_pos_scratch,
        volatile float *sphere_pos_approx_scratch,
        volatile int *link_cc_scratch,
        float *transform_scratch,
        ppln::collision::Environment<float> *env,
        int *edge_is_valid,
        float *edge_maximum_projection_delta
    ) {
        using Collision = robots::CollisionTraits<Robot>;
        static constexpr int fine_scratch_stride =
            Collision::fine_sphere_count * Collision::batch_size * 3;
        static constexpr int approx_scratch_stride =
            Collision::approximate_sphere_count * Collision::batch_size * 3;
        static constexpr int link_scratch_stride =
            Collision::joint_flag_stride * Collision::batch_size;
        static constexpr int transform_scratch_stride =
            Collision::batch_size * Collision::transform_slots * 16;
        const std::size_t batch_edge = blockIdx.x;
        node_anchors += batch_edge * 2 * Robot::dimension;
        edge_motion_segments +=
            batch_edge * CONCON_MOTION_SEGMENT_STRIDE;
        edge_motion_segment_next +=
            batch_edge * CONCON_MOTION_SEGMENT_STRIDE;
        sphere_pos_scratch += batch_edge * fine_scratch_stride;
        sphere_pos_approx_scratch += batch_edge * approx_scratch_stride;
        link_cc_scratch += batch_edge * link_scratch_stride;
        transform_scratch += batch_edge * transform_scratch_stride;
        edge_is_valid += batch_edge;
        if (edge_maximum_projection_delta != nullptr) {
            edge_maximum_projection_delta += batch_edge;
        }

        const int tid = threadIdx.x;
        __shared__ volatile unsigned char projection_valid[
            CONCON_PROJECTION_STATE_STRIDE
        ];
        __shared__ volatile int projection_progress[1];
        __shared__ volatile unsigned int projection_success[1];
        __shared__ volatile int first_projection_failure_edge[1];
        __shared__ volatile unsigned int edge_cc_result[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_detailed_env_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_self_collision_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_detailed_self_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool any_detailed_env_check;
        __shared__ bool any_detailed_self_check;
        __shared__ volatile int first_collision_edge[1];
        __shared__ int joint_limits_good;

        if (tid == 0) {
            edge_is_valid[0] = 0;
            if (edge_maximum_projection_delta != nullptr) {
                edge_maximum_projection_delta[0] = 0.0f;
            }
            joint_limits_good = 1;
            first_collision_edge[0] = 0;
        }
        __syncthreads();

        patacon_project_concon_edge_segments_from_node_anchors<Robot>(
            1,
            node_anchors,
            edge_motion_segments,
            edge_motion_segment_next,
            projection_valid,
            projection_progress,
            projection_success,
            first_projection_failure_edge,
            tid
        );
        __syncthreads();

        if (
            edge_maximum_projection_delta != nullptr &&
            first_projection_failure_edge[0] >= 1 &&
            tid <= d_settings.granularity
        ) {
            const float ratio = static_cast<float>(tid) /
                static_cast<float>(d_settings.granularity);
            float thread_maximum = 0.0f;
            for (int joint = 0; joint < Robot::dimension; ++joint) {
                const float nominal = node_anchors[joint] + ratio *
                    (node_anchors[Robot::dimension + joint] -
                     node_anchors[joint]);
                thread_maximum = fmaxf(
                    thread_maximum,
                    fabsf(
                        edge_motion_segments[
                            tid * Robot::dimension + joint
                        ] - nominal
                    )
                );
            }
            atomicMax(
                reinterpret_cast<unsigned int *>(
                    edge_maximum_projection_delta
                ),
                __float_as_uint(thread_maximum)
            );
        }
        __syncthreads();

        if (
            first_projection_failure_edge[0] >= 1 &&
            tid <= d_settings.granularity &&
            !planning::configuration_within_joint_limits<Robot>(
                &edge_motion_segments[tid * Robot::dimension]
            )
        ) {
            atomicExch(&joint_limits_good, 0);
        }
        __syncthreads();

        if (
            first_projection_failure_edge[0] >= 1 &&
            joint_limits_good != 0
        ) {
            patacon_check_projected_edges_collision_parallel<Robot>(
                1,
                edge_motion_segments,
                sphere_pos_scratch,
                sphere_pos_approx_scratch,
                link_cc_scratch,
                transform_scratch,
                env,
                edge_cc_result,
                run_detailed_env_check,
                run_self_collision_check,
                run_detailed_self_check,
                &any_detailed_env_check,
                &any_detailed_self_check,
                first_collision_edge,
                tid
            );
        }
        __syncthreads();

        if (tid == 0) {
            edge_is_valid[0] =
                first_projection_failure_edge[0] >= 1 &&
                joint_limits_good != 0 &&
                first_collision_edge[0] >= 1;
        }
    }

    template <typename Robot>
    __global__ void patacon_validate_visualization_nominal_edge(
        const float *node_anchors,
        volatile float *edge_motion_segments,
        volatile float *sphere_pos_scratch,
        volatile float *sphere_pos_approx_scratch,
        volatile int *link_cc_scratch,
        float *transform_scratch,
        ppln::collision::Environment<float> *env,
        int *edge_is_valid
    ) {
        using Collision = robots::CollisionTraits<Robot>;
        static constexpr int fine_scratch_stride =
            Collision::fine_sphere_count * Collision::batch_size * 3;
        static constexpr int approx_scratch_stride =
            Collision::approximate_sphere_count * Collision::batch_size * 3;
        static constexpr int link_scratch_stride =
            Collision::joint_flag_stride * Collision::batch_size;
        static constexpr int transform_scratch_stride =
            Collision::batch_size * Collision::transform_slots * 16;
        const std::size_t batch_edge = blockIdx.x;
        node_anchors += batch_edge * 2 * Robot::dimension;
        edge_motion_segments +=
            batch_edge * CONCON_MOTION_SEGMENT_STRIDE;
        sphere_pos_scratch += batch_edge * fine_scratch_stride;
        sphere_pos_approx_scratch += batch_edge * approx_scratch_stride;
        link_cc_scratch += batch_edge * link_scratch_stride;
        transform_scratch += batch_edge * transform_scratch_stride;
        edge_is_valid += batch_edge;

        const int tid = threadIdx.x;
        __shared__ volatile unsigned int edge_cc_result[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_detailed_env_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_self_collision_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_detailed_self_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool any_detailed_env_check;
        __shared__ bool any_detailed_self_check;
        __shared__ volatile int first_collision_edge[1];
        __shared__ int joint_limits_good;

        if (tid == 0) {
            edge_is_valid[0] = 0;
            joint_limits_good = 1;
            first_collision_edge[0] = 0;
        }
        if (tid <= d_settings.granularity) {
            const float ratio = static_cast<float>(tid) /
                static_cast<float>(d_settings.granularity);
            for (int joint = 0; joint < Robot::dimension; ++joint) {
                edge_motion_segments[tid * Robot::dimension + joint] =
                    node_anchors[joint] + ratio *
                    (node_anchors[Robot::dimension + joint] -
                     node_anchors[joint]);
            }
        }
        __syncthreads();

        if (
            tid <= d_settings.granularity &&
            !planning::configuration_within_joint_limits<Robot>(
                &edge_motion_segments[tid * Robot::dimension]
            )
        ) {
            atomicExch(&joint_limits_good, 0);
        }
        __syncthreads();

        if (joint_limits_good != 0) {
            patacon_check_projected_edges_collision_parallel<Robot>(
                1,
                edge_motion_segments,
                sphere_pos_scratch,
                sphere_pos_approx_scratch,
                link_cc_scratch,
                transform_scratch,
                env,
                edge_cc_result,
                run_detailed_env_check,
                run_self_collision_check,
                run_detailed_self_check,
                &any_detailed_env_check,
                &any_detailed_self_check,
                first_collision_edge,
                tid
            );
        }
        __syncthreads();

        if (tid == 0) {
            edge_is_valid[0] =
                joint_limits_good != 0 && first_collision_edge[0] >= 1;
        }
    }

    __device__ __forceinline__
    int patacon_reserve_slot(volatile int *counter,int capacity){
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

    template <int LaneStride = MAX_THREADS_PER_BLOCK>
    __device__ __forceinline__
    void patacon_register_node_in_ts(
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
        const int head_slot =ts_id * LaneStride + lane;

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
    __device__ __forceinline__ float patacon_project_target_direction_to_tangent(
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

            const int active_tangent_dim = patacon_active_tangent_dim<Robot>();

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
                    patacon_joint_distance_weight<Robot>(tid);

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
    patacon(
        float **nodes,
        int **parents,
        int **node_ready,
        // Tangent Space membership
        int **node_ts_id,
        float **node_ts_q,

        // Tangent Space Bank
        int *ts_count,
        int **ts_root_node_idx,
        int **ts_parent_id,
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
        const unsigned long long block_start_time_ns = global_timer_ns();
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
        __shared__ int *t_ts_parent_id;
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
        // PATACON CONNECT state
        // target 도달 여부
        __shared__ bool connection_reached_shared;
        __shared__ unsigned int n_extensions;
        // CONNECT 시작 시 상대 tree에서 한 번 선택하고 끝까지 유지할 target
        __shared__ int connect_target_idx;
        __shared__ float *connect_target_node;

        // 새 TB-RRT CONNECT 상태
        __shared__ bool connect_failed;
        __shared__ bool connect_reached;
        // PATACON parallel projection shared memory
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
                const bool time_limit_reached =
                    patacon_time_limit_ns > 0 &&
                    global_timer_ns() - block_start_time_ns >=
                        patacon_time_limit_ns;
                if (iter > d_settings.max_iters || time_limit_reached) {
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
                    t_ts_parent_id =ts_parent_id[t_tree_id];
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
                            if (t_ts_ready[candidate_ts]
                                == current_search_generation) {
                                selected_ts_id =candidate_ts;
                                selected_ts_root_idx =t_ts_root_node_idx[candidate_ts];

                                break;
                            }
                        }
                    }

                    // 2. 현재 constraint의 tangent dimension
                    const int active_tangent_dim = patacon_active_tangent_dim<Robot>();

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
                    ts_alpha_fraction = active_tangent_dim < dim
                        ? halton_sample[active_tangent_dim]
                        : curand_uniform(&rng_states[bid]);
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
                patacon_sample_tangent_config<Robot>(
                    t_nodes, 
                    t_ts_bases, // node별 basis가 아니라 TSBank
                    t_ts_root_node_idx,
                    t_ts_parent_id,
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
                    if (t_node_ready[node_idx]
                        == current_search_generation) { // 이번 요청의 node인지 확인
                        // 현재 node와 q_rand 사이의 거리 계산 (거리 제곱 반환)
                        const float candidate_dist =
                            patacon_sq_config_distance<Robot>(
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
                    if (t_node_ready[i] != current_search_generation) {
                        continue;
                    }

                    const float candidate_dist =
                        patacon_sq_config_distance<Robot>(
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

            if (tid == 0) {
                diagnostic_increment(DIAG_EXTEND_ATTEMPTS);
            }

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
                        const float em_error =patacon_constraint_error_norm<Robot>(concon_probe); // constraint residual 검사

                        // 먼저 현재 candidate를 포함한다.
                        concon_count = step;

                        // 그 다음 threshold 검사
                        // 즉 threshold를 처음 초과한 node도 포함된다.
                        if (em_error >d_settings.em_threshold
                        ) {
                            concon_em_stop = true;
                            diagnostic_increment(DIAG_EXTEND_EM_STOPS);
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
                patacon_project_concon_node_anchors<Robot>(
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
                if (valid_edges < extend_edge_count) {
                    diagnostic_increment(
                        DIAG_EXTEND_ANCHOR_PROJECTION_STOPS
                    );
                }
                concon_projected_edge_count = valid_edges;
            }
            __syncthreads();

            patacon_project_concon_edge_segments_from_node_anchors<Robot>(
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
                    diagnostic_increment(
                        DIAG_EXTEND_EDGE_PROJECTION_STOPS
                    );
                    concon_projected_edge_count =
                        concon_first_projection_failure_edge[0];
                }
            }
            __syncthreads();

            patacon_check_projected_edges_collision_parallel<Robot>(
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

            if (
                tid == 0 &&
                concon_first_collision_edge[0] < concon_projected_edge_count
            ) {
                diagnostic_increment(DIAG_EXTEND_COLLISION_STOPS);
            }

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
                        index = patacon_reserve_slot(
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
                                new_ts_id =patacon_reserve_slot(&ts_count[t_tree_id],d_settings.max_tangent_spaces);

                                // TSBank 공간 부족
                                if (new_ts_id < 0) {
                                    new_ts_basis_ok = false;
                                    atomicCAS((int *)&solved,0,-1);
                                }
                                else {
                                    // 아직 다른 block이 이 TS를 사용하면 안 됨
                                    t_ts_ready[new_ts_id] = 0;

                                    // Reused persistent TS slots can contain
                                    // linked-list metadata from an older
                                    // generation.  Clear just this new slot.
                                    t_ts_node_count[new_ts_id] = 0;
                                    for (
                                        int lane = 0;
                                        lane < MAX_THREADS_PER_BLOCK;
                                        ++lane
                                    ) {
                                        t_ts_lane_head[
                                            new_ts_id * MAX_THREADS_PER_BLOCK
                                                + lane
                                        ] = -1;
                                    }

                                    // 새 TS의 root는 방금 projection된 실제 tree node
                                    t_ts_root_node_idx[new_ts_id] =index;

                                    // projected actual q에서 Jacobian 계산
                                    // → null space basis 생성
                                    // → TSBank에 저장
                                    new_ts_basis_ok =patacon_store_tangent_basis<Robot>(&t_nodes[index * dim],t_ts_bases,new_ts_id);

                                    if (new_ts_basis_ok) {
                                        // 논문 3.5.1의 forward half-space를 구성할
                                        // 수 있도록 새 TS의 부모 chart를 기록한다.
                                        t_ts_parent_id[new_ts_id] =selected_ts_id;
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

                            patacon_register_node_in_ts(index,assigned_ts_id,t_ts_node_count,t_ts_lane_head,t_node_next_in_ts);
                        }
                    }
                    __syncthreads();

                    if (tid == 0) {
                        // 먼저 tree node 공개
                        node_ready[t_tree_id][index] =
                            current_search_generation;
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
                                t_ts_ready[new_ts_id] =
                                    current_search_generation;
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
            if (
                tid == 0 &&
                concon_valid_count > 0 &&
                concon_valid_count == extend_edge_count
            ) {
                diagnostic_increment(DIAG_EXTEND_FULL_SUCCESSES);
            }
            // A non-empty valid prefix is enough to attempt CONNECT from the
            // last EXTEND node that was actually inserted into the tree.
            if (concon_valid_count > 0) {
                // connect
                local_min_dist = FLT_MAX;
                local_near_idx = 0;
                int size = min((int)atomic_free_index[o_tree_id], d_settings.max_samples); // 빈데편 tree 크기 가져오기

                // 반대편 tree node를 thread들이 나눠서 NN 검사
                for (unsigned int i = tid; i < size; i += blockDim.x) { 
                    if (o_node_ready[i] != current_search_generation) {
                        continue;
                    }
                    dist = patacon_sq_config_distance<Robot>(
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
                    diagnostic_increment(DIAG_CONNECT_ATTEMPTS);
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

                    const float chunk_start_target_distance =patacon_shared_config_distance<Robot>(
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

                    if (tid == 0) {
                        diagnostic_increment(DIAG_CONNECT_CHUNKS);
                    }

                    float connect_tangent_dist = 0.0f;

                    if constexpr (TangentSpaceTraits<Robot>::enabled) {
                        // [NEW CONNECT] 현재 node가 속한 Tangent Space 확인
                        if (tid == 0) {
                            // 현재 CONNECT 시작 node가 속한 Tangent Space
                            selected_ts_id =t_node_ts_id[index];

                            // 유효한 Tangent Space인지 확인
                            if (selected_ts_id < 0
                                || t_ts_ready[selected_ts_id]
                                    != current_search_generation) {
                                connect_failed = true;
                                diagnostic_increment(
                                    DIAG_CONNECT_INVALID_TANGENT_SPACES
                                );
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
                            connect_tangent_dist = patacon_project_target_direction_to_tangent<Robot>(
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
                                diagnostic_increment(
                                    DIAG_CONNECT_TANGENT_DIRECTION_STOPS
                                );
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
                                    const float em_error =patacon_constraint_error_norm<Robot>(concon_probe);
                                    concon_count = step;
                                    if (em_error >d_settings.em_threshold) {
                                        concon_em_stop = true;
                                        diagnostic_increment(
                                            DIAG_CONNECT_EM_STOPS
                                        );
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
                        patacon_project_concon_node_anchors<Robot>(
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
                        if (valid_edges < concon_count) {
                            diagnostic_increment(
                                DIAG_CONNECT_ANCHOR_PROJECTION_STOPS
                            );
                        }

                        for (int edge_step = 1; edge_step <= valid_edges; edge_step++) {
                            const float distance_before =
                                patacon_config_distance_from_volatile<Robot>(
                                    &motion_segment[
                                        (edge_step - 1) * dim
                                    ],
                                    connect_target_node
                                );
                            const float distance_after =
                                patacon_config_distance_from_volatile<Robot>(
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
                                diagnostic_increment(
                                    DIAG_CONNECT_PROGRESS_STOPS
                                );
                                valid_edges = edge_step - 1;
                                break;
                            }
                        }

                        concon_projected_edge_count = valid_edges;
                    }
                    __syncthreads();

                    patacon_project_concon_edge_segments_from_node_anchors<Robot>(
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
                            diagnostic_increment(
                                DIAG_CONNECT_EDGE_PROJECTION_STOPS
                            );
                            concon_projected_edge_count =
                                concon_first_projection_failure_edge[0];
                        }

                        // Endpoint nodes can move during edge smoothing, so
                        // recheck CONNECT progress using the final prefix.
                        for (
                            int edge_step = 1;
                            edge_step <= concon_projected_edge_count;
                            edge_step++
                        ) {
                            volatile float *projected_edge =
                                &concon_motion_segments[
                                    (edge_step - 1) *
                                    CONCON_MOTION_SEGMENT_STRIDE
                                ];
                            const float distance_before =
                                patacon_config_distance_from_volatile<Robot>(
                                    projected_edge,
                                    connect_target_node
                                );
                            const float distance_after =
                                patacon_config_distance_from_volatile<Robot>(
                                    &projected_edge[
                                        d_settings.granularity * dim
                                    ],
                                    connect_target_node
                                );

                            if (
                                distance_after >=
                                distance_before -
                                    d_settings.connect_progress_epsilon
                            ) {
                                diagnostic_increment(
                                    DIAG_CONNECT_PROGRESS_STOPS
                                );
                                concon_projected_edge_count = edge_step - 1;
                                break;
                            }
                        }
                    }
                    __syncthreads();

                    patacon_check_projected_edges_collision_parallel<Robot>(
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

                    if (
                        tid == 0 &&
                        concon_first_collision_edge[0] <
                            concon_projected_edge_count
                    ) {
                        diagnostic_increment(DIAG_CONNECT_COLLISION_STOPS);
                    }

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
                            index = patacon_reserve_slot(
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
                                    new_ts_id =patacon_reserve_slot(
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

                                        // Initialize only the newly reserved
                                        // TS slot for this generation.
                                        t_ts_node_count[new_ts_id] = 0;
                                        for (
                                            int lane = 0;
                                            lane < MAX_THREADS_PER_BLOCK;
                                            ++lane
                                        ) {
                                            t_ts_lane_head[
                                                new_ts_id
                                                    * MAX_THREADS_PER_BLOCK
                                                    + lane
                                            ] = -1;
                                        }

                                        // 현재 projected node가 새 TS root
                                        t_ts_root_node_idx[new_ts_id] = index;

                                        // 실제 projected configuration에서
                                        // Jacobian/null-space basis 생성
                                        new_ts_basis_ok =patacon_store_tangent_basis<Robot>(
                                                &t_nodes[index * dim],
                                                t_ts_bases,
                                                new_ts_id
                                            );

                                        if (new_ts_basis_ok) {
                                            t_ts_parent_id[new_ts_id] =selected_ts_id;
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

                                patacon_register_node_in_ts(index,assigned_ts_id,t_ts_node_count,t_ts_lane_head,t_node_next_in_ts);
                            }
                        }
                        __syncthreads();

                        if (tid == 0) {
                            // 먼저 tree node 공개
                            t_node_ready[index] = current_search_generation;
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
                                    t_ts_ready[new_ts_id] =
                                        current_search_generation;
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

                        const float current_target_distance = patacon_shared_config_distance<Robot>(config,connect_target_node,sdata,tid);

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
                    patacon_shared_config_distance<Robot>(
                        config,
                        connect_target_node,
                        sdata,
                        tid
                    );

                if (tid == 0) {
                    connect_reached =final_connection_distance<= d_settings.connect_reached_tolerance;
                    if (!connect_failed && connect_reached) {
                        diagnostic_increment(DIAG_CONNECT_SUCCESSES);
                    }
                    else {
                        diagnostic_increment(DIAG_CONNECT_FAILURES);
                    }
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

                            cost += patacon_config_distance<Robot>(
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

                            cost += patacon_config_distance<Robot>(
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
        const int current_samples[2],
        int search_generation
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
            for (int &ready : res.tree_node_ready[tree]) {
                ready = ready == search_generation ? 1 : 0;
            }

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

    inline void visualization_shortcut_cuda_check(
        cudaError_t status,
        const char *operation
    ) {
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("visualization path shortcut CUDA failure during ") +
                operation + ": " + cudaGetErrorString(status)
            );
        }
    }

    template <typename Robot>
    float visualization_shortcut_distance(
        const typename Robot::Configuration &a,
        const typename Robot::Configuration &b,
        const PATACON_settings &settings,
        bool use_planner_weights
    ) {
        double squared_distance = 0.0;
        for (int joint = 0; joint < Robot::dimension; ++joint) {
            float weight = 1.0f;
            if (use_planner_weights) {
                if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
                    if (joint == 0) {
                        weight = settings.lift_distance_weight;
                    }
                } else if constexpr (
                    std::is_same_v<Robot, robots::FfwSg2Mobility>
                ) {
                    if (joint == 3) {
                        weight = settings.lift_distance_weight;
                    }
                }
            }
            const double difference =
                static_cast<double>(weight) *
                (static_cast<double>(a[joint]) - b[joint]);
            squared_distance += difference * difference;
        }
        return static_cast<float>(std::sqrt(squared_distance));
    }

    inline float visualization_shortcut_maximum_chunk_length(
        const PATACON_settings &settings
    ) {
        float maximum_chunk_length = settings.range;
        if (
            settings.projection_smoothness &&
            settings.projection_smoothness_threshold > 0.0f
        ) {
            maximum_chunk_length = std::min(
                maximum_chunk_length,
                0.9f * static_cast<float>(settings.granularity) *
                    settings.projection_smoothness_threshold
            );
        }
        return maximum_chunk_length;
    }

    template <typename Robot>
    class VisualizationShortcutWorkspace {
    public:
        using Collision = robots::CollisionTraits<Robot>;

        explicit VisualizationShortcutWorkspace(
            const ppln::collision::Environment<float> &host_environment,
            std::size_t edge_capacity = 1
        ) :
            host_environment_(host_environment),
            edge_capacity_(edge_capacity)
        {
            if (edge_capacity_ == 0) {
                throw std::invalid_argument(
                    "visualization validation workspace capacity must be positive"
                );
            }
        }

        VisualizationShortcutWorkspace(
            const VisualizationShortcutWorkspace &
        ) = delete;
        VisualizationShortcutWorkspace &operator=(
            const VisualizationShortcutWorkspace &
        ) = delete;

        ~VisualizationShortcutWorkspace() {
            if (device_environment != nullptr) {
                cleanup_environment_on_device(
                    device_environment,
                    host_environment_
                );
            }
            cudaFree(device_node_anchors);
            cudaFree(device_edge_motion_segments);
            cudaFree(device_edge_motion_segment_next);
            cudaFree(device_sphere_pos_scratch);
            cudaFree(device_sphere_pos_approx_scratch);
            cudaFree(device_link_cc_scratch);
            cudaFree(device_transform_scratch);
            cudaFree(device_edge_is_valid);
            cudaFree(device_nominal_edge_is_valid);
            cudaFree(device_maximum_projection_delta);
        }

        void allocate() {
            setup_environment_on_device(
                device_environment,
                host_environment_
            );
            visualization_shortcut_cuda_check(
                cudaGetLastError(),
                "environment setup"
            );
            visualization_shortcut_cuda_check(
                cudaMalloc(
                    &device_node_anchors,
                    edge_capacity_ * 2 * Robot::dimension * sizeof(float)
                ),
                "node-anchor allocation"
            );
            visualization_shortcut_cuda_check(
                cudaMalloc(
                    &device_edge_motion_segments,
                    edge_capacity_ * CONCON_MOTION_SEGMENT_STRIDE *
                        sizeof(float)
                ),
                "projected-motion allocation"
            );
            visualization_shortcut_cuda_check(
                cudaMalloc(
                    &device_edge_motion_segment_next,
                    edge_capacity_ * CONCON_MOTION_SEGMENT_STRIDE *
                        sizeof(float)
                ),
                "projection scratch allocation"
            );
            visualization_shortcut_cuda_check(
                cudaMalloc(
                    &device_sphere_pos_scratch,
                    edge_capacity_ * Collision::fine_sphere_count *
                        Collision::batch_size * 3 * sizeof(float)
                ),
                "fine collision scratch allocation"
            );
            visualization_shortcut_cuda_check(
                cudaMalloc(
                    &device_sphere_pos_approx_scratch,
                    edge_capacity_ * Collision::approximate_sphere_count *
                        Collision::batch_size * 3 * sizeof(float)
                ),
                "approximate collision scratch allocation"
            );
            visualization_shortcut_cuda_check(
                cudaMalloc(
                    &device_link_cc_scratch,
                    edge_capacity_ * Collision::joint_flag_stride *
                        Collision::batch_size * sizeof(int)
                ),
                "collision flag allocation"
            );
            visualization_shortcut_cuda_check(
                cudaMalloc(
                    &device_transform_scratch,
                    edge_capacity_ * Collision::batch_size *
                        Collision::transform_slots * 16 * sizeof(float)
                ),
                "collision transform allocation"
            );
            visualization_shortcut_cuda_check(
                cudaMalloc(
                    &device_edge_is_valid,
                    edge_capacity_ * sizeof(int)
                ),
                "shortcut result allocation"
            );
            visualization_shortcut_cuda_check(
                cudaMalloc(
                    &device_nominal_edge_is_valid,
                    edge_capacity_ * sizeof(int)
                ),
                "nominal path-validation result allocation"
            );
            visualization_shortcut_cuda_check(
                cudaMalloc(
                    &device_maximum_projection_delta,
                    edge_capacity_ * sizeof(float)
                ),
                "projection-delta result allocation"
            );
        }

        ppln::collision::Environment<float> *device_environment = nullptr;
        float *device_node_anchors = nullptr;
        float *device_edge_motion_segments = nullptr;
        float *device_edge_motion_segment_next = nullptr;
        float *device_sphere_pos_scratch = nullptr;
        float *device_sphere_pos_approx_scratch = nullptr;
        int *device_link_cc_scratch = nullptr;
        float *device_transform_scratch = nullptr;
        int *device_edge_is_valid = nullptr;
        int *device_nominal_edge_is_valid = nullptr;
        float *device_maximum_projection_delta = nullptr;

    private:
        const ppln::collision::Environment<float> &host_environment_;
        const std::size_t edge_capacity_;
    };

    template <typename Robot>
    bool validate_visualization_shortcut_edge(
        const typename Robot::Configuration &source,
        const typename Robot::Configuration &target,
        const PATACON_settings &settings,
        VisualizationShortcutWorkspace<Robot> &workspace,
        std::vector<typename Robot::Configuration> &projected_edge
    ) {
        std::array<float, 2 * Robot::dimension> node_anchors{};
        std::copy(source.begin(), source.end(), node_anchors.begin());
        std::copy(
            target.begin(),
            target.end(),
            node_anchors.begin() + Robot::dimension
        );
        visualization_shortcut_cuda_check(
            cudaMemcpy(
                workspace.device_node_anchors,
                node_anchors.data(),
                node_anchors.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "node-anchor upload"
        );

        patacon_validate_visualization_shortcut_edge<Robot>
            <<<1, CONCON_COLLISION_THREADS_PER_EDGE>>>(
                workspace.device_node_anchors,
                workspace.device_edge_motion_segments,
                workspace.device_edge_motion_segment_next,
                workspace.device_sphere_pos_scratch,
                workspace.device_sphere_pos_approx_scratch,
                workspace.device_link_cc_scratch,
                workspace.device_transform_scratch,
                workspace.device_environment,
                workspace.device_edge_is_valid,
                nullptr
            );
        visualization_shortcut_cuda_check(
            cudaGetLastError(),
            "shortcut validation kernel launch"
        );
        visualization_shortcut_cuda_check(
            cudaDeviceSynchronize(),
            "shortcut validation kernel execution"
        );

        int edge_is_valid = 0;
        visualization_shortcut_cuda_check(
            cudaMemcpy(
                &edge_is_valid,
                workspace.device_edge_is_valid,
                sizeof(edge_is_valid),
                cudaMemcpyDeviceToHost
            ),
            "shortcut validation result download"
        );
        if (edge_is_valid == 0) {
            return false;
        }

        const std::size_t waypoint_count =
            static_cast<std::size_t>(settings.granularity) + 1;
        std::vector<float> flat_path(
            waypoint_count * static_cast<std::size_t>(Robot::dimension)
        );
        visualization_shortcut_cuda_check(
            cudaMemcpy(
                flat_path.data(),
                workspace.device_edge_motion_segments,
                flat_path.size() * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "projected shortcut download"
        );

        projected_edge.resize(waypoint_count);
        for (std::size_t waypoint = 0; waypoint < waypoint_count; ++waypoint) {
            std::copy_n(
                flat_path.data() + waypoint * Robot::dimension,
                Robot::dimension,
                projected_edge[waypoint].begin()
            );
        }
        projected_edge.front() = source;
        return true;
    }

    template <typename Robot>
    bool validate_visualization_nominal_edge(
        const typename Robot::Configuration &source,
        const typename Robot::Configuration &target,
        VisualizationShortcutWorkspace<Robot> &workspace
    ) {
        std::array<float, 2 * Robot::dimension> node_anchors{};
        std::copy(source.begin(), source.end(), node_anchors.begin());
        std::copy(
            target.begin(),
            target.end(),
            node_anchors.begin() + Robot::dimension
        );
        visualization_shortcut_cuda_check(
            cudaMemcpy(
                workspace.device_node_anchors,
                node_anchors.data(),
                node_anchors.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "nominal path-validation anchor upload"
        );

        patacon_validate_visualization_nominal_edge<Robot>
            <<<1, CONCON_COLLISION_THREADS_PER_EDGE>>>(
                workspace.device_node_anchors,
                workspace.device_edge_motion_segments,
                workspace.device_sphere_pos_scratch,
                workspace.device_sphere_pos_approx_scratch,
                workspace.device_link_cc_scratch,
                workspace.device_transform_scratch,
                workspace.device_environment,
                workspace.device_edge_is_valid
            );
        visualization_shortcut_cuda_check(
            cudaGetLastError(),
            "nominal path-validation kernel launch"
        );
        visualization_shortcut_cuda_check(
            cudaDeviceSynchronize(),
            "nominal path-validation kernel execution"
        );

        int edge_is_valid = 0;
        visualization_shortcut_cuda_check(
            cudaMemcpy(
                &edge_is_valid,
                workspace.device_edge_is_valid,
                sizeof(edge_is_valid),
                cudaMemcpyDeviceToHost
            ),
            "nominal path-validation result download"
        );
        return edge_is_valid != 0;
    }

    template <typename Robot>
    bool build_visualization_shortcut(
        const typename Robot::Configuration &source,
        const typename Robot::Configuration &target,
        const PATACON_settings &settings,
        VisualizationShortcutWorkspace<Robot> &workspace,
        std::vector<typename Robot::Configuration> &shortcut
    ) {
        const float euclidean_distance =
            visualization_shortcut_distance<Robot>(
                source,
                target,
                settings,
                false
            );
        const float planner_distance =
            visualization_shortcut_distance<Robot>(
                source,
                target,
                settings,
                true
            );
        const float maximum_chunk_length =
            visualization_shortcut_maximum_chunk_length(settings);
        if (maximum_chunk_length <= 0.0f) {
            return false;
        }

        const float connection_distance = std::max(
            euclidean_distance,
            planner_distance
        );
        const int chunk_count = std::max(
            1,
            static_cast<int>(std::ceil(
                connection_distance / maximum_chunk_length
            ))
        );
        if (chunk_count > settings.max_connect_concon_chunks) {
            return false;
        }

        shortcut.clear();
        shortcut.push_back(source);
        typename Robot::Configuration current = source;
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const int remaining_chunks = chunk_count - chunk;
            typename Robot::Configuration nominal_target{};
            for (int joint = 0; joint < Robot::dimension; ++joint) {
                nominal_target[joint] = current[joint] +
                    (target[joint] - current[joint]) /
                    static_cast<float>(remaining_chunks);
            }

            std::vector<typename Robot::Configuration> projected_edge;
            if (!validate_visualization_shortcut_edge<Robot>(
                    current,
                    nominal_target,
                    settings,
                    workspace,
                    projected_edge
                )) {
                return false;
            }

            const float distance_before =
                visualization_shortcut_distance<Robot>(
                    current,
                    target,
                    settings,
                    true
                );
            const float distance_after =
                visualization_shortcut_distance<Robot>(
                    projected_edge.back(),
                    target,
                    settings,
                    true
                );
            if (
                chunk + 1 < chunk_count &&
                distance_after >= distance_before -
                    settings.connect_progress_epsilon
            ) {
                return false;
            }

            shortcut.insert(
                shortcut.end(),
                std::next(projected_edge.begin()),
                projected_edge.end()
            );
            current = shortcut.back();
        }

        constexpr float endpoint_tolerance = 1.0e-4f;
        if (
            visualization_shortcut_distance<Robot>(
                shortcut.back(),
                target,
                settings,
                false
            ) > endpoint_tolerance
        ) {
            return false;
        }
        shortcut.back() = target;
        return true;
    }

    template <typename Robot>
    PathSimplificationResult<Robot> simplify_path_for_visualization(
        const std::vector<typename Robot::Configuration> &path,
        ppln::collision::Environment<float> &h_environment,
        PATACON_settings &settings
    ) {
        using Configuration = typename Robot::Configuration;
        using Collision = robots::CollisionTraits<Robot>;
        PathSimplificationResult<Robot> result;
        result.path = path;
        result.original_cost = configuration_space_path_arclength<Robot>(path);
        result.simplified_cost = result.original_cost;
        if (path.size() < 3) {
            return result;
        }
        if (settings.granularity != Collision::batch_size) {
            throw std::invalid_argument(
                "path shortcut granularity must match the robot collision batch size"
            );
        }
        if (
            settings.granularity <= 0 ||
            settings.granularity > MAX_GRANULARITY
        ) {
            throw std::invalid_argument(
                "path shortcut supports granularity in [1, 16]"
            );
        }
        if (settings.max_connect_concon_chunks <= 0) {
            throw std::invalid_argument(
                "path shortcut requires max_connect_concon_chunks > 0"
            );
        }

        visualization_shortcut_cuda_check(
            cudaMemcpyToSymbol(d_settings, &settings, sizeof(settings)),
            "planner setting upload"
        );
        if constexpr (std::is_same_v<Robot, robots::G1>) {
            visualization_shortcut_cuda_check(
                cudaMemcpyToSymbol(
                    ppln::collision::g1_attached_object_collision,
                    &settings.g1_constraints.attached_object_collision,
                    sizeof(settings.g1_constraints.attached_object_collision)
                ),
                "G1 attached-object setting upload"
            );
        }
        if constexpr (
            std::is_same_v<Robot, robots::FfwSg2> ||
            std::is_same_v<Robot, robots::FfwSg2Mobility>
        ) {
            visualization_shortcut_cuda_check(
                cudaMemcpyToSymbol(
                    ppln::collision::ffw_sg2_mobility_attached_object_collision,
                    &settings.ffw_sg2_attached_object_collision,
                    sizeof(settings.ffw_sg2_attached_object_collision)
                ),
                "FFW-SG2 attached-object setting upload"
            );
        }

        VisualizationShortcutWorkspace<Robot> workspace(h_environment);
        workspace.allocate();

        std::vector<float> prefix_cost(path.size(), 0.0f);
        for (std::size_t index = 1; index < path.size(); ++index) {
            prefix_cost[index] = prefix_cost[index - 1] +
                visualization_shortcut_distance<Robot>(
                    path[index - 1],
                    path[index],
                    settings,
                    false
                );
        }

        std::vector<Configuration> simplified_path;
        simplified_path.reserve(path.size());
        simplified_path.push_back(path.front());
        std::size_t source_index = 0;
        std::size_t considered_candidates = 0;
        const std::size_t maximum_attempts = std::clamp<std::size_t>(
            path.size() * 4,
            32,
            256
        );
        const std::size_t maximum_candidate_checks =
            std::clamp<std::size_t>(path.size() * 16, 128, 4096);
        constexpr float minimum_improvement = 1.0e-5f;

        while (source_index + 1 < path.size()) {
            if (
                result.attempted_shortcuts >= maximum_attempts ||
                considered_candidates >= maximum_candidate_checks
            ) {
                simplified_path.insert(
                    simplified_path.end(),
                    std::next(path.begin(), source_index + 1),
                    path.end()
                );
                break;
            }

            bool accepted = false;
            for (
                std::size_t target_index = path.size() - 1;
                target_index > source_index + 1;
                --target_index
            ) {
                ++considered_candidates;
                if (considered_candidates > maximum_candidate_checks) {
                    break;
                }
                const float replaced_cost =
                    prefix_cost[target_index] - prefix_cost[source_index];
                const float direct_lower_bound =
                    visualization_shortcut_distance<Robot>(
                        path[source_index],
                        path[target_index],
                        settings,
                        false
                    );
                if (
                    direct_lower_bound + minimum_improvement >= replaced_cost
                ) {
                    continue;
                }

                const float planner_distance =
                    visualization_shortcut_distance<Robot>(
                        path[source_index],
                        path[target_index],
                        settings,
                        true
                    );
                const float maximum_chunk_length =
                    visualization_shortcut_maximum_chunk_length(settings);
                if (
                    maximum_chunk_length <= 0.0f ||
                    std::max(direct_lower_bound, planner_distance) >
                        maximum_chunk_length *
                        static_cast<float>(
                            settings.max_connect_concon_chunks
                        )
                ) {
                    continue;
                }

                ++result.attempted_shortcuts;
                std::vector<Configuration> shortcut;
                if (
                    build_visualization_shortcut<Robot>(
                        path[source_index],
                        path[target_index],
                        settings,
                        workspace,
                        shortcut
                    )
                ) {
                    const float shortcut_cost =
                        configuration_space_path_arclength<Robot>(shortcut);
                    if (
                        shortcut_cost + minimum_improvement < replaced_cost
                    ) {
                        simplified_path.insert(
                            simplified_path.end(),
                            std::next(shortcut.begin()),
                            shortcut.end()
                        );
                        source_index = target_index;
                        ++result.accepted_shortcuts;
                        accepted = true;
                        break;
                    }
                }
                if (result.attempted_shortcuts >= maximum_attempts) {
                    break;
                }
            }

            if (!accepted) {
                simplified_path.push_back(path[source_index + 1]);
                ++source_index;
            }
        }

        const float simplified_cost =
            configuration_space_path_arclength<Robot>(simplified_path);
        if (
            simplified_path.front() == path.front() &&
            simplified_path.back() == path.back() &&
            simplified_cost <= result.original_cost + minimum_improvement
        ) {
            result.path = std::move(simplified_path);
            result.simplified_cost = simplified_cost;
        } else {
            result.accepted_shortcuts = 0;
        }
        return result;
    }

    template <typename Robot>
    PathValidationResult validate_path_for_visualization(
        const std::vector<typename Robot::Configuration> &path,
        ppln::collision::Environment<float> &h_environment,
        PATACON_settings &settings,
        float projection_tolerance
    ) {
        using Collision = robots::CollisionTraits<Robot>;
        PathValidationResult result;
        result.failed_edge = path.size() > 1 ? 0 : path.size();
        if (path.size() < 2) {
            return result;
        }
        if (!std::isfinite(projection_tolerance) || projection_tolerance < 0.0f) {
            throw std::invalid_argument(
                "path validation projection tolerance must be finite and nonnegative"
            );
        }
        if (settings.granularity != Collision::batch_size) {
            throw std::invalid_argument(
                "path validation granularity must match the robot collision batch size"
            );
        }
        if (
            settings.granularity <= 0 ||
            settings.granularity > MAX_GRANULARITY
        ) {
            throw std::invalid_argument(
                "path validation supports granularity in [1, 16]"
            );
        }

        visualization_shortcut_cuda_check(
            cudaMemcpyToSymbol(d_settings, &settings, sizeof(settings)),
            "path validation setting upload"
        );
        if constexpr (std::is_same_v<Robot, robots::G1>) {
            visualization_shortcut_cuda_check(
                cudaMemcpyToSymbol(
                    ppln::collision::g1_attached_object_collision,
                    &settings.g1_constraints.attached_object_collision,
                    sizeof(settings.g1_constraints.attached_object_collision)
                ),
                "G1 path validation attached-object setting upload"
            );
        }
        if constexpr (
            std::is_same_v<Robot, robots::FfwSg2> ||
            std::is_same_v<Robot, robots::FfwSg2Mobility>
        ) {
            visualization_shortcut_cuda_check(
                cudaMemcpyToSymbol(
                    ppln::collision::ffw_sg2_mobility_attached_object_collision,
                    &settings.ffw_sg2_attached_object_collision,
                    sizeof(settings.ffw_sg2_attached_object_collision)
                ),
                "FFW-SG2 path validation attached-object setting upload"
            );
        }

        const std::size_t edge_count = path.size() - 1;
        if (
            edge_count > static_cast<std::size_t>(
                std::numeric_limits<unsigned int>::max()
            )
        ) {
            throw std::length_error(
                "path validation edge count exceeds the CUDA grid limit"
            );
        }

        VisualizationShortcutWorkspace<Robot> workspace(
            h_environment,
            edge_count
        );
        workspace.allocate();

        const std::size_t anchor_stride =
            2 * static_cast<std::size_t>(Robot::dimension);
        std::vector<float> node_anchors(edge_count * anchor_stride);
        for (std::size_t edge = 0; edge < edge_count; ++edge) {
            float *anchors =
                node_anchors.data() + edge * anchor_stride;
            std::copy(
                path[edge].begin(),
                path[edge].end(),
                anchors
            );
            std::copy(
                path[edge + 1].begin(),
                path[edge + 1].end(),
                anchors + Robot::dimension
            );
        }
        visualization_shortcut_cuda_check(
            cudaMemcpy(
                workspace.device_node_anchors,
                node_anchors.data(),
                node_anchors.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "batched path-validation anchor upload"
        );

        const unsigned int cuda_edge_count =
            static_cast<unsigned int>(edge_count);
        patacon_validate_visualization_shortcut_edge<Robot>
            <<<cuda_edge_count, CONCON_COLLISION_THREADS_PER_EDGE>>>(
                workspace.device_node_anchors,
                workspace.device_edge_motion_segments,
                workspace.device_edge_motion_segment_next,
                workspace.device_sphere_pos_scratch,
                workspace.device_sphere_pos_approx_scratch,
                workspace.device_link_cc_scratch,
                workspace.device_transform_scratch,
                workspace.device_environment,
                workspace.device_edge_is_valid,
                workspace.device_maximum_projection_delta
            );
        visualization_shortcut_cuda_check(
            cudaGetLastError(),
            "batched projected path-validation kernel launch"
        );

        patacon_validate_visualization_nominal_edge<Robot>
            <<<cuda_edge_count, CONCON_COLLISION_THREADS_PER_EDGE>>>(
                workspace.device_node_anchors,
                workspace.device_edge_motion_segments,
                workspace.device_sphere_pos_scratch,
                workspace.device_sphere_pos_approx_scratch,
                workspace.device_link_cc_scratch,
                workspace.device_transform_scratch,
                workspace.device_environment,
                workspace.device_nominal_edge_is_valid
            );
        visualization_shortcut_cuda_check(
            cudaGetLastError(),
            "batched nominal path-validation kernel launch"
        );
        visualization_shortcut_cuda_check(
            cudaDeviceSynchronize(),
            "batched path-validation kernel execution"
        );

        std::vector<int> projected_edge_is_valid(edge_count);
        std::vector<int> nominal_edge_is_valid(edge_count);
        std::vector<float> maximum_projection_delta(edge_count);
        visualization_shortcut_cuda_check(
            cudaMemcpy(
                projected_edge_is_valid.data(),
                workspace.device_edge_is_valid,
                edge_count * sizeof(int),
                cudaMemcpyDeviceToHost
            ),
            "batched projected path-validation result download"
        );
        visualization_shortcut_cuda_check(
            cudaMemcpy(
                nominal_edge_is_valid.data(),
                workspace.device_nominal_edge_is_valid,
                edge_count * sizeof(int),
                cudaMemcpyDeviceToHost
            ),
            "batched nominal path-validation result download"
        );
        visualization_shortcut_cuda_check(
            cudaMemcpy(
                maximum_projection_delta.data(),
                workspace.device_maximum_projection_delta,
                edge_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "batched projection-delta result download"
        );

        for (std::size_t edge = 0; edge < edge_count; ++edge) {
            ++result.checked_edges;
            if (projected_edge_is_valid[edge] == 0) {
                result.failed_edge = edge;
                return result;
            }
            result.maximum_projection_delta = std::max(
                result.maximum_projection_delta,
                maximum_projection_delta[edge]
            );
            if (result.maximum_projection_delta > projection_tolerance) {
                result.failed_edge = edge;
                return result;
            }
            if (nominal_edge_is_valid[edge] == 0) {
                result.failed_edge = edge;
                return result;
            }
        }

        result.valid = true;
        result.failed_edge = path.size() - 1;
        return result;
    }

    template <typename Robot>
    class SolveWorkspace {
    public:
        static constexpr int dim = Robot::dimension;
        using Collision = robots::CollisionTraits<Robot>;

        explicit SolveWorkspace(const PATACON_settings &settings)
            : max_samples(settings.max_samples),
              max_tangent_spaces(settings.max_tangent_spaces),
              num_new_configs(settings.num_new_configs),
              max_concon_nodes(settings.max_concon_nodes) {
            allocate();
        }

        SolveWorkspace(const SolveWorkspace &) = delete;
        SolveWorkspace &operator=(const SolveWorkspace &) = delete;

        ~SolveWorkspace() {
            release();
        }

        bool matches(const PATACON_settings &settings) const {
            return max_samples == settings.max_samples
                && max_tangent_spaces == settings.max_tangent_spaces
                && num_new_configs == settings.num_new_configs
                && max_concon_nodes == settings.max_concon_nodes;
        }

        int begin_request() {
            if (generation_ == std::numeric_limits<int>::max()) {
                const std::size_t node_bytes =
                    static_cast<std::size_t>(max_samples) * sizeof(int);
                const std::size_t ts_bytes =
                    static_cast<std::size_t>(max_tangent_spaces)
                    * sizeof(int);
                for (int tree = 0; tree < 2; ++tree) {
                    cudaCheckError(cudaMemset(node_ready[tree], 0, node_bytes));
                    cudaCheckError(cudaMemset(ts_ready[tree], 0, ts_bytes));
                }
                generation_ = 1;
            } else {
                ++generation_;
            }

            return generation_;
        }

        void set_random_seed(unsigned long long seed) {
            current_random_seed = seed;
        }

        float *nodes[2] = {nullptr, nullptr};
        int *parents[2] = {nullptr, nullptr};
        int *node_ready[2] = {nullptr, nullptr};
        float *radii[2] = {nullptr, nullptr};
        float **d_nodes = nullptr;
        int **d_parents = nullptr;
        int **d_node_ready = nullptr;
        float **d_radii = nullptr;
        int *node_ts_id[2] = {nullptr, nullptr};
        float *node_ts_q[2] = {nullptr, nullptr};
        int **d_node_ts_id = nullptr;
        float **d_node_ts_q = nullptr;
        int *ts_count = nullptr;
        int *ts_root_node_idx[2] = {nullptr, nullptr};
        int *ts_parent_id[2] = {nullptr, nullptr};
        float *ts_bases[2] = {nullptr, nullptr};
        int *ts_ready[2] = {nullptr, nullptr};
        int **d_ts_root_node_idx = nullptr;
        int **d_ts_parent_id = nullptr;
        float **d_ts_bases = nullptr;
        int **d_ts_ready = nullptr;
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
        curandState *rng_states = nullptr;
        HaltonState<Robot> *halton_states = nullptr;
        int *h_solved = nullptr;
        PersistentDeviceEnvironment environment;

    private:
        void allocate() {
            const std::size_t config_bytes =
                static_cast<std::size_t>(max_samples) * dim * sizeof(float);

            cudaMalloc(&d_nodes, 2 * sizeof(float *));
            cudaMalloc(&d_parents, 2 * sizeof(int *));
            cudaMalloc(&d_radii, 2 * sizeof(float *));
            cudaMalloc(&d_node_ready, 2 * sizeof(int *));
            cudaMalloc(&d_node_ts_id, 2 * sizeof(int *));
            cudaMalloc(&d_node_ts_q, 2 * sizeof(float *));
            cudaMalloc(&ts_count, 2 * sizeof(int));
            cudaMalloc(&d_ts_root_node_idx, 2 * sizeof(int *));
            cudaMalloc(&d_ts_parent_id, 2 * sizeof(int *));
            cudaMalloc(&d_ts_bases, 2 * sizeof(float *));
            cudaMalloc(&d_ts_ready, 2 * sizeof(int *));
            cudaMalloc(&d_ts_node_count, 2 * sizeof(int *));
            cudaMalloc(&d_ts_lane_head, 2 * sizeof(int *));
            cudaMalloc(&d_node_next_in_ts, 2 * sizeof(int *));

            for (int tree = 0; tree < 2; ++tree) {
                if constexpr (TangentSpaceTraits<Robot>::enabled) {
                    cudaMalloc(
                        &ts_bases[tree],
                        static_cast<std::size_t>(max_tangent_spaces)
                            * TangentSpaceTraits<Robot>::basis_size
                            * sizeof(float)
                    );
                    cudaMalloc(
                        &ts_parent_id[tree],
                        static_cast<std::size_t>(max_tangent_spaces)
                            * sizeof(int)
                    );
                    cudaMalloc(
                        &ts_node_count[tree],
                        static_cast<std::size_t>(max_tangent_spaces)
                            * sizeof(int)
                    );
                    cudaMemset(
                        ts_node_count[tree], 0,
                        static_cast<std::size_t>(max_tangent_spaces)
                            * sizeof(int)
                    );
                    cudaMalloc(
                        &ts_lane_head[tree],
                        static_cast<std::size_t>(max_tangent_spaces)
                            * MAX_THREADS_PER_BLOCK * sizeof(int)
                    );
                    cudaMemset(
                        ts_lane_head[tree], 0xff,
                        static_cast<std::size_t>(max_tangent_spaces)
                            * MAX_THREADS_PER_BLOCK * sizeof(int)
                    );
                    cudaMalloc(
                        &node_next_in_ts[tree],
                        static_cast<std::size_t>(max_samples) * sizeof(int)
                    );
                }
                cudaMalloc(&nodes[tree], config_bytes);
                cudaMalloc(
                    &parents[tree],
                    static_cast<std::size_t>(max_samples) * sizeof(int)
                );
                cudaMalloc(
                    &radii[tree],
                    static_cast<std::size_t>(max_samples) * sizeof(float)
                );
                cudaMalloc(
                    &node_ready[tree],
                    static_cast<std::size_t>(max_samples) * sizeof(int)
                );
                cudaMemset(
                    node_ready[tree], 0,
                    static_cast<std::size_t>(max_samples) * sizeof(int)
                );
                cudaMalloc(
                    &node_ts_id[tree],
                    static_cast<std::size_t>(max_samples) * sizeof(int)
                );
                cudaMalloc(&node_ts_q[tree], config_bytes);
                cudaMalloc(
                    &ts_root_node_idx[tree],
                    static_cast<std::size_t>(max_tangent_spaces)
                        * sizeof(int)
                );
                cudaMalloc(
                    &ts_ready[tree],
                    static_cast<std::size_t>(max_tangent_spaces)
                        * sizeof(int)
                );
                cudaMemset(
                    ts_ready[tree], 0,
                    static_cast<std::size_t>(max_tangent_spaces) * sizeof(int)
                );
            }

            cudaMemcpy(d_nodes, nodes, 2 * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_parents, parents, 2 * sizeof(int *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_radii, radii, 2 * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_node_ready, node_ready, 2 * sizeof(int *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_node_ts_id, node_ts_id, 2 * sizeof(int *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_node_ts_q, node_ts_q, 2 * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_root_node_idx, ts_root_node_idx, 2 * sizeof(int *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_parent_id, ts_parent_id, 2 * sizeof(int *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_bases, ts_bases, 2 * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_ready, ts_ready, 2 * sizeof(int *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_node_count, ts_node_count, 2 * sizeof(int *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_lane_head, ts_lane_head, 2 * sizeof(int *), cudaMemcpyHostToDevice);
            cudaMemcpy(d_node_next_in_ts, node_next_in_ts, 2 * sizeof(int *), cudaMemcpyHostToDevice);

            const int rng_state_count = num_new_configs * dim;
            cudaMalloc(
                &rng_states,
                static_cast<std::size_t>(rng_state_count) * sizeof(curandState)
            );
            cudaMalloc(
                &halton_states,
                static_cast<std::size_t>(num_new_configs)
                    * sizeof(HaltonState<Robot>)
            );

            const std::size_t scratch_slots =
                static_cast<std::size_t>(num_new_configs)
                * static_cast<std::size_t>(max_concon_nodes);
            cudaMalloc(
                &concon_sphere_pos_scratch,
                scratch_slots * Collision::fine_sphere_count
                    * Collision::batch_size * 3 * sizeof(float)
            );
            cudaMalloc(
                &concon_sphere_pos_approx_scratch,
                scratch_slots * Collision::approximate_sphere_count
                    * Collision::batch_size * 3 * sizeof(float)
            );
            cudaMalloc(
                &concon_link_cc_scratch,
                scratch_slots * Collision::joint_flag_stride
                    * Collision::batch_size * sizeof(int)
            );
            cudaMalloc(
                &concon_transform_scratch,
                scratch_slots * Collision::batch_size
                    * Collision::transform_slots * 16 * sizeof(float)
            );
            cudaMallocHost(&h_solved, sizeof(int));
            cudaCheckError(cudaGetLastError());
        }

        void release() noexcept {
            auto free_device = [](auto *&pointer) {
                if (pointer != nullptr) {
                    cudaFree(pointer);
                    pointer = nullptr;
                }
            };
            for (int tree = 0; tree < 2; ++tree) {
                free_device(nodes[tree]);
                free_device(parents[tree]);
                free_device(node_ready[tree]);
                free_device(radii[tree]);
                free_device(node_ts_id[tree]);
                free_device(node_ts_q[tree]);
                free_device(ts_root_node_idx[tree]);
                free_device(ts_parent_id[tree]);
                free_device(ts_bases[tree]);
                free_device(ts_ready[tree]);
                free_device(ts_node_count[tree]);
                free_device(ts_lane_head[tree]);
                free_device(node_next_in_ts[tree]);
            }
            free_device(d_nodes);
            free_device(d_parents);
            free_device(d_node_ready);
            free_device(d_radii);
            free_device(d_node_ts_id);
            free_device(d_node_ts_q);
            free_device(ts_count);
            free_device(d_ts_root_node_idx);
            free_device(d_ts_parent_id);
            free_device(d_ts_bases);
            free_device(d_ts_ready);
            free_device(d_ts_node_count);
            free_device(d_ts_lane_head);
            free_device(d_node_next_in_ts);
            free_device(rng_states);
            free_device(halton_states);
            free_device(concon_sphere_pos_scratch);
            free_device(concon_sphere_pos_approx_scratch);
            free_device(concon_link_cc_scratch);
            free_device(concon_transform_scratch);
            if (h_solved != nullptr) {
                cudaFreeHost(h_solved);
                h_solved = nullptr;
            }
        }

        int max_samples;
        int max_tangent_spaces;
        int num_new_configs;
        int max_concon_nodes;
        unsigned long long current_random_seed = 0;
        int generation_ = 0;
    };

    std::unique_ptr<SolveWorkspace<robots::G1>> &persistent_g1_workspace() {
        static std::unique_ptr<SolveWorkspace<robots::G1>> workspace;
        return workspace;
    }

    void release_persistent_workspace() {
        persistent_g1_workspace().reset();
        release_g1_projection_workspace();
    }

    template <typename Robot>
    PlannerResult<Robot> solve(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &h_environment,
        PATACON_settings &settings
    ) 
    {
        auto start_time = std::chrono::steady_clock::now();
        static constexpr auto dim = Robot::dimension;
        using Collision = robots::CollisionTraits<Robot>;
        if (settings.granularity != Collision::batch_size) {
            throw std::invalid_argument(
                "PATACON granularity must match the selected robot's collision batch size"
            );
        }
        if (settings.granularity > MAX_GRANULARITY) {
            throw std::invalid_argument(
                "PATACON prefix ConCon projection supports granularity up to 16"
            );
        }
        if (settings.max_concon_nodes <= 0) {
            throw std::invalid_argument(
                "PATACON max_concon_nodes must be positive"
            );
        }
        if (settings.max_concon_nodes > MAX_PARALLEL_CONCON_EDGES) {
            throw std::invalid_argument(
                "PATACON parallel ConCon collision check supports up to 5 edges"
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
        const unsigned long long zero_diagnostics[
            DIAGNOSTIC_COUNTER_COUNT
        ] = {};
        cudaMemcpyToSymbol(
            diagnostic_counters,
            zero_diagnostics,
            sizeof(zero_diagnostics)
        );
        if constexpr (std::is_same_v<Robot, robots::G1>) {
            cudaMemcpyToSymbol(
                ppln::collision::g1_attached_object_collision,
                &settings.g1_constraints.attached_object_collision,
                sizeof(settings.g1_constraints.attached_object_collision)
            );
        }
        if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
            cudaMemcpyToSymbol(
                ppln::collision::ffw_sg2_mobility_attached_object_collision,
                &settings.ffw_sg2_attached_object_collision,
                sizeof(settings.ffw_sg2_attached_object_collision)
            );
        }
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
        int *ts_parent_id[2] = {nullptr, nullptr};
        float *ts_bases[2] = {nullptr, nullptr};
        int *ts_ready[2] = {nullptr, nullptr};

        int **d_ts_root_node_idx = nullptr;
        int **d_ts_parent_id = nullptr;
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
        SolveWorkspace<Robot> *reusable_workspace = nullptr;
        if constexpr (std::is_same_v<Robot, robots::G1>) {
            if (persistent_workspace_enabled) {
                auto &cached_workspace = persistent_g1_workspace();
                if (cached_workspace == nullptr
                    || !cached_workspace->matches(settings)) {
                    cached_workspace =
                        std::make_unique<SolveWorkspace<robots::G1>>(settings);
                }
                reusable_workspace = cached_workspace.get();
            }
        }

        if (reusable_workspace != nullptr) {
            for (int tree = 0; tree < 2; ++tree) {
                nodes[tree] = reusable_workspace->nodes[tree];
                parents[tree] = reusable_workspace->parents[tree];
                node_ready[tree] = reusable_workspace->node_ready[tree];
                radii[tree] = reusable_workspace->radii[tree];
                node_ts_id[tree] = reusable_workspace->node_ts_id[tree];
                node_ts_q[tree] = reusable_workspace->node_ts_q[tree];
                ts_root_node_idx[tree] =
                    reusable_workspace->ts_root_node_idx[tree];
                ts_parent_id[tree] =
                    reusable_workspace->ts_parent_id[tree];
                ts_bases[tree] = reusable_workspace->ts_bases[tree];
                ts_ready[tree] = reusable_workspace->ts_ready[tree];
                ts_node_count[tree] =
                    reusable_workspace->ts_node_count[tree];
                ts_lane_head[tree] = reusable_workspace->ts_lane_head[tree];
                node_next_in_ts[tree] =
                    reusable_workspace->node_next_in_ts[tree];
            }
            d_nodes = reusable_workspace->d_nodes;
            d_parents = reusable_workspace->d_parents;
            d_node_ready = reusable_workspace->d_node_ready;
            d_radii = reusable_workspace->d_radii;
            d_node_ts_id = reusable_workspace->d_node_ts_id;
            d_node_ts_q = reusable_workspace->d_node_ts_q;
            ts_count = reusable_workspace->ts_count;
            d_ts_root_node_idx = reusable_workspace->d_ts_root_node_idx;
            d_ts_parent_id = reusable_workspace->d_ts_parent_id;
            d_ts_bases = reusable_workspace->d_ts_bases;
            d_ts_ready = reusable_workspace->d_ts_ready;
            d_ts_node_count = reusable_workspace->d_ts_node_count;
            d_ts_lane_head = reusable_workspace->d_ts_lane_head;
            d_node_next_in_ts = reusable_workspace->d_node_next_in_ts;
        } else {
            cudaMalloc(&d_nodes, 2 * sizeof(float*));
            cudaMalloc(&d_parents, 2 * sizeof(int*));
            cudaMalloc(&d_radii, 2 * sizeof(float*));
            cudaMalloc(&d_node_ready, 2 * sizeof(int*));
            cudaMalloc(&d_node_ts_id, 2 * sizeof(int*));
            cudaMalloc(&d_node_ts_q, 2 * sizeof(float*));
            cudaMalloc(&ts_count, 2 * sizeof(int));
            cudaMalloc(&d_ts_root_node_idx,2 * sizeof(int*));
            cudaMalloc(&d_ts_parent_id,2 * sizeof(int*));
            cudaMalloc(&d_ts_bases,2 * sizeof(float*));
            cudaMalloc(&d_ts_ready,2 * sizeof(int*));
            cudaMalloc(&d_ts_node_count,2 * sizeof(int*));
            cudaMalloc(&d_ts_lane_head,2 * sizeof(int*));
            cudaMalloc(&d_node_next_in_ts,2 * sizeof(int*));
        }
        const int search_generation = reusable_workspace != nullptr
            ? reusable_workspace->begin_request()
            : 1;
        cudaCheckError(cudaMemcpyToSymbol(
            current_search_generation,
            &search_generation,
            sizeof(search_generation)
        ));
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
                if (reusable_workspace == nullptr) {
                    cudaMalloc(&ts_bases[i],basis_bytes);
                }
                if (reusable_workspace == nullptr) {
                    cudaMemset(ts_bases[i],0,basis_bytes);
                }

                // TS root chart의 부모 chart. Root TS는 -1을 사용한다.
                if (reusable_workspace == nullptr) {
                    cudaMalloc(&ts_parent_id[i],ts_count_bytes);
                }
                if (reusable_workspace == nullptr) {
                    cudaMemset(ts_parent_id[i],0xff,ts_count_bytes);
                }

                if (reusable_workspace == nullptr) {
                    cudaMalloc(&ts_node_count[i],ts_count_bytes);
                }
                if (reusable_workspace == nullptr) {
                    cudaMemset(ts_node_count[i],0,ts_count_bytes);
                }

                if (reusable_workspace == nullptr) {
                    cudaMalloc(&ts_lane_head[i],ts_lane_head_bytes);
                }
                if (reusable_workspace == nullptr) {
                    cudaMemset(ts_lane_head[i],0xff,ts_lane_head_bytes);
                }

                if (reusable_workspace == nullptr) {
                    cudaMalloc(&node_next_in_ts[i],node_next_bytes);
                }
                if (reusable_workspace == nullptr) {
                    cudaMemset(node_next_in_ts[i],0xff,node_next_bytes);
                }
            }
            if (reusable_workspace == nullptr) {
                cudaMalloc(&nodes[i], settings.max_samples * config_size);
                cudaMalloc(&parents[i], settings.max_samples * sizeof(int));
                cudaMalloc(&radii[i], settings.max_samples * sizeof(float));
                cudaMalloc(&node_ready[i], settings.max_samples * sizeof(int));
            }
            if (reusable_workspace == nullptr) {
                cudaMemset(
                    node_ready[i], 0,
                    settings.max_samples * sizeof(int)
                );
            }
            // 이 tree의 각 node가 어느 TS에 속하는지
            if (reusable_workspace == nullptr) {
                cudaMalloc(&node_ts_id[i],settings.max_samples * sizeof(int));
            }
            // 처음에는 어떤 TS에도 속하지 않음: -1
            if (reusable_workspace == nullptr) {
                cudaMemset(
                    node_ts_id[i], 0xff,
                    settings.max_samples * sizeof(int)
                );
            }
            // 각 tree node의 Tangent Space 상 nominal q
            if (reusable_workspace == nullptr) {
                cudaMalloc(&node_ts_q[i],settings.max_samples * config_size);
            }
            // TS마다 root node index 저장
            if (reusable_workspace == nullptr) {
                cudaMalloc(&ts_root_node_idx[i],settings.max_tangent_spaces * sizeof(int));
            }
            // TS가 완전히 생성되었는지
            if (reusable_workspace == nullptr) {
                cudaMalloc(&ts_ready[i],settings.max_tangent_spaces * sizeof(int));
            }
            // 처음에는 모든 TS가 아직 생성되지 않음
            if (reusable_workspace == nullptr) {
                cudaMemset(
                    ts_ready[i], 0,
                    settings.max_tangent_spaces * sizeof(int)
                );
            }
        }
        if (reusable_workspace == nullptr) {
            cudaMemcpy(d_nodes, nodes, 2 * sizeof(float*), cudaMemcpyHostToDevice);
            cudaMemcpy(d_parents, parents, 2 * sizeof(int*), cudaMemcpyHostToDevice);
            cudaMemcpy(d_node_ts_id,node_ts_id,2 * sizeof(int*),cudaMemcpyHostToDevice);
            cudaMemcpy(d_node_ts_q,node_ts_q,2 * sizeof(float*),cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_root_node_idx,ts_root_node_idx,2 * sizeof(int*),cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_parent_id,ts_parent_id,2 * sizeof(int*),cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_bases,ts_bases,2 * sizeof(float*),cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_ready,ts_ready,2 * sizeof(int*),cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_node_count,ts_node_count,2 * sizeof(int*),cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts_lane_head,ts_lane_head,2 * sizeof(int*),cudaMemcpyHostToDevice);
            cudaMemcpy(d_node_next_in_ts,node_next_in_ts,2 * sizeof(int*),cudaMemcpyHostToDevice);
        }
        int h_ts_count[2] = {0, 0};
        cudaMemcpy(d_radii,radii,2 * sizeof(float*),cudaMemcpyHostToDevice);
        cudaMemcpy(ts_count,h_ts_count,2 * sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_node_ready, node_ready, 2 * sizeof(int*), cudaMemcpyHostToDevice);

        // set nodes to unitialized
        if (reusable_workspace == nullptr) {
            std::vector<float> nodes_init(
                settings.max_samples * dim, UNWRITTEN_VAL
            );
            cudaMemcpy((void *)nodes[0], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);
            cudaMemcpy((void *)nodes[1], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);
            cudaMemcpy(node_ts_q[0],nodes_init.data(),config_size * settings.max_samples,cudaMemcpyHostToDevice);
            cudaMemcpy(node_ts_q[1],nodes_init.data(),config_size * settings.max_samples,cudaMemcpyHostToDevice);
        }
            
        // initialize radii
        std::vector<float> radii_init(num_goals, FLT_MAX);
        cudaMemcpy((void *)radii[0], radii_init.data(), sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)radii[1], radii_init.data(), sizeof(float) * num_goals, cudaMemcpyHostToDevice);
        
        // create a curandState for each thread
        curandState *rng_states = reusable_workspace != nullptr
            ? reusable_workspace->rng_states
            : nullptr;
        int num_rng_states = settings.num_new_configs * dim;
        if (reusable_workspace == nullptr) {
            cudaMalloc(&rng_states, num_rng_states * sizeof(curandState));
        }
        int numBlocks = (num_rng_states + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_rng<<<numBlocks, BLOCK_SIZE>>>(
            rng_states,
            settings.random_seed,
            num_rng_states
        );

        HaltonState<Robot> *halton_states = reusable_workspace != nullptr
            ? reusable_workspace->halton_states
            : nullptr;
        if (reusable_workspace == nullptr) {
            cudaMalloc(&halton_states, settings.num_new_configs * sizeof(HaltonState<Robot>));
        }
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
        if (reusable_workspace != nullptr) {
            env = reusable_workspace->environment.update(h_environment);
        } else {
            setup_environment_on_device(env, h_environment);
        }
        cudaCheckError(cudaGetLastError());

        const std::size_t concon_scratch_slots =
            static_cast<std::size_t>(settings.num_new_configs) *
            static_cast<std::size_t>(settings.max_concon_nodes);
        if (reusable_workspace != nullptr) {
            concon_sphere_pos_scratch =
                reusable_workspace->concon_sphere_pos_scratch;
            concon_sphere_pos_approx_scratch =
                reusable_workspace->concon_sphere_pos_approx_scratch;
            concon_link_cc_scratch =
                reusable_workspace->concon_link_cc_scratch;
            concon_transform_scratch =
                reusable_workspace->concon_transform_scratch;
        } else {
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
        }
        cudaCheckError(cudaGetLastError());
        
        // Setup pinned memory for signaling
        int *h_solved = reusable_workspace != nullptr
            ? reusable_workspace->h_solved
            : nullptr;
        int current_samples[2];
        int h_solved_iters = -1;
        if (reusable_workspace == nullptr) {
            cudaMallocHost(&h_solved, sizeof(int));
        }
        *h_solved = -1;

        
        auto copy_start_time = std::chrono::steady_clock::now();
        // add start to tree_a and goals to tree_b
        cudaMemcpy((void *)nodes[0], start.data(), config_size, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)parents[0], &start_index, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)nodes[1], goals.data(), config_size * num_goals, cudaMemcpyHostToDevice);
        std::vector<int> parents_b_init(num_goals);
        iota(parents_b_init.begin(), parents_b_init.end(), 0); // consecutive integers from 0 ... num_goals - 1
        cudaMemcpy((void *)parents[1], parents_b_init.data(), sizeof(int) * num_goals, cudaMemcpyHostToDevice);
        const int start_ready = search_generation;
        std::vector<int> goals_ready(num_goals, search_generation);
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
                d_ts_parent_id,
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

        unsigned long long kernel_time_limit_ns = 0;
        if (time_limit_seconds > 0.0) {
            const auto total_time_limit_ns = static_cast<std::uint64_t>(
                std::llround(time_limit_seconds * 1.0e9)
            );
            if (time_limit_counts_kernel_only) {
                kernel_time_limit_ns = total_time_limit_ns;
            }
            else {
                const auto elapsed_before_kernel_ns =
                    static_cast<std::uint64_t>(
                        get_elapsed_nanoseconds(start_time)
                    );
                kernel_time_limit_ns =
                    elapsed_before_kernel_ns < total_time_limit_ns
                        ? total_time_limit_ns - elapsed_before_kernel_ns
                        : 1;
            }
        }
        cudaMemcpyToSymbol(
            patacon_time_limit_ns,
            &kernel_time_limit_ns,
            sizeof(kernel_time_limit_ns)
        );

        auto kernel_start_time = std::chrono::steady_clock::now();
        const int concon_threads_per_block =
            CONCON_COLLISION_THREADS_PER_EDGE * settings.max_concon_nodes;
        if (settings.trace_trees) {
            patacon<Robot, true><<<settings.num_new_configs, concon_threads_per_block>>> (
                d_nodes,
                d_parents,
                d_node_ready,
                // 새 Tree → TS 정보
                d_node_ts_id,
                d_node_ts_q,

                // 새 TSBank
                ts_count,
                d_ts_root_node_idx,
                d_ts_parent_id,
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
            patacon<Robot, false><<<settings.num_new_configs, concon_threads_per_block>>> (
                d_nodes,
                d_parents,
                d_node_ready,
                // 새 Tree → TS 정보
                d_node_ts_id,
                d_node_ts_q,

                // 새 TSBank
                ts_count,
                d_ts_root_node_idx,
                d_ts_parent_id,
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
        if (settings.collect_diagnostics) {
            unsigned long long host_diagnostics[
                DIAGNOSTIC_COUNTER_COUNT
            ] = {};
            cudaMemcpy(
                res.diagnostics.tangent_space_count.data(),
                ts_count,
                2 * sizeof(int),
                cudaMemcpyDeviceToHost
            );
            cudaMemcpyFromSymbol(
                host_diagnostics,
                diagnostic_counters,
                sizeof(host_diagnostics),
                0,
                cudaMemcpyDeviceToHost
            );
            res.diagnostics.extend_attempts =
                host_diagnostics[DIAG_EXTEND_ATTEMPTS];
            res.diagnostics.extend_backtracking_flips =
                host_diagnostics[DIAG_EXTEND_BACKTRACKING_FLIPS];
            res.diagnostics.extend_em_stops =
                host_diagnostics[DIAG_EXTEND_EM_STOPS];
            res.diagnostics.extend_anchor_projection_stops =
                host_diagnostics[DIAG_EXTEND_ANCHOR_PROJECTION_STOPS];
            res.diagnostics.extend_edge_projection_stops =
                host_diagnostics[DIAG_EXTEND_EDGE_PROJECTION_STOPS];
            res.diagnostics.extend_collision_stops =
                host_diagnostics[DIAG_EXTEND_COLLISION_STOPS];
            res.diagnostics.extend_full_successes =
                host_diagnostics[DIAG_EXTEND_FULL_SUCCESSES];
            res.diagnostics.connect_attempts =
                host_diagnostics[DIAG_CONNECT_ATTEMPTS];
            res.diagnostics.connect_chunks =
                host_diagnostics[DIAG_CONNECT_CHUNKS];
            res.diagnostics.connect_invalid_tangent_spaces =
                host_diagnostics[DIAG_CONNECT_INVALID_TANGENT_SPACES];
            res.diagnostics.connect_tangent_direction_stops =
                host_diagnostics[DIAG_CONNECT_TANGENT_DIRECTION_STOPS];
            res.diagnostics.connect_em_stops =
                host_diagnostics[DIAG_CONNECT_EM_STOPS];
            res.diagnostics.connect_anchor_projection_stops =
                host_diagnostics[DIAG_CONNECT_ANCHOR_PROJECTION_STOPS];
            res.diagnostics.connect_edge_projection_stops =
                host_diagnostics[DIAG_CONNECT_EDGE_PROJECTION_STOPS];
            res.diagnostics.connect_progress_stops =
                host_diagnostics[DIAG_CONNECT_PROGRESS_STOPS];
            res.diagnostics.connect_collision_stops =
                host_diagnostics[DIAG_CONNECT_COLLISION_STOPS];
            res.diagnostics.connect_successes =
                host_diagnostics[DIAG_CONNECT_SUCCESSES];
            res.diagnostics.connect_failures =
                host_diagnostics[DIAG_CONNECT_FAILURES];
        }
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
                        "PATACON device path size is outside its valid range"
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
                current_samples,
                search_generation
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
        
        if (reusable_workspace == nullptr) {
            cleanup_environment_on_device(env, h_environment);
        }
        reset_device_variables();
        if (reusable_workspace == nullptr) {
        cudaFree((void *)nodes[0]);
        cudaFree((void *)nodes[1]);
        cudaFree((void *)parents[0]);
        cudaFree((void *)parents[1]);

        for (int i = 0; i < 2; i++) {

        if (ts_root_node_idx[i] != nullptr) {
            cudaFree(ts_root_node_idx[i]);
        }

        if (ts_parent_id[i] != nullptr) {
            cudaFree(ts_parent_id[i]);
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
        cudaFree(d_ts_parent_id);
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
        }
        cudaCheckError(cudaGetLastError());
        res.wall_ns = get_elapsed_nanoseconds(start_time);
        if (cuda_device_reset_enabled) {
            cudaDeviceReset();
        }
        return res;
    }

    //template PlannerResult<typename ppln::robots::Sphere> solve<ppln::robots::Sphere>(std::array<float, 3>&, std::vector<std::array<float, 3>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::FrankaSingle> solve<ppln::robots::FrankaSingle>(std::array<float, 7>&, std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::Franka> solve<ppln::robots::Franka>(std::array<float, 14>&, std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::FfwSg2> solve<ppln::robots::FfwSg2>(std::array<float, 15>&, std::vector<std::array<float, 15>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::FfwSg2Mobility> solve<ppln::robots::FfwSg2Mobility>(std::array<float, 18>&, std::vector<std::array<float, 18>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::G1> solve<ppln::robots::G1>(std::array<float, 35>&, std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::IgrisC> solve<ppln::robots::IgrisC>(std::array<float, 35>&, std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&);


    template PathSimplificationResult<ppln::robots::FrankaSingle> simplify_path_for_visualization<ppln::robots::FrankaSingle>(const std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PathSimplificationResult<ppln::robots::Franka> simplify_path_for_visualization<ppln::robots::Franka>(const std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PathSimplificationResult<ppln::robots::FfwSg2> simplify_path_for_visualization<ppln::robots::FfwSg2>(const std::vector<std::array<float, 15>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PathSimplificationResult<ppln::robots::FfwSg2Mobility> simplify_path_for_visualization<ppln::robots::FfwSg2Mobility>(const std::vector<std::array<float, 18>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PathSimplificationResult<ppln::robots::G1> simplify_path_for_visualization<ppln::robots::G1>(const std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PathSimplificationResult<ppln::robots::IgrisC> simplify_path_for_visualization<ppln::robots::IgrisC>(const std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&);

    template PathValidationResult validate_path_for_visualization<ppln::robots::FrankaSingle>(const std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);
    template PathValidationResult validate_path_for_visualization<ppln::robots::Franka>(const std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);
    template PathValidationResult validate_path_for_visualization<ppln::robots::FfwSg2>(const std::vector<std::array<float, 15>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);
    template PathValidationResult validate_path_for_visualization<ppln::robots::FfwSg2Mobility>(const std::vector<std::array<float, 18>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);
    template PathValidationResult validate_path_for_visualization<ppln::robots::G1>(const std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);
    template PathValidationResult validate_path_for_visualization<ppln::robots::IgrisC>(const std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);

}

// AORRTC is part of the same CUDA translation unit so the initial PATACON
// search and the bounded optimization phase can share one compiled backend.
#include "AORRTCOptimization.cuh"

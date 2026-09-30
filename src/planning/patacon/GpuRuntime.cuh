// Internal CUDA device state, diagnostics, distance, and sampling implementation.
// Included inside namespace PATACON by each planner CUDA translation unit.

#ifndef PATACON_SKIP_G1_PROJECTION_RUNTIME
    __global__ void project_g1_configuration_kernel(
        float *configuration,
        constraints::G1ConstraintParameters parameters,
        bool axis,
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
                axis,
                max_iterations,
                alpha,
                damping,
                maximum_step
            );
        }
    }

    namespace {
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

    bool project_g1_configuration(
        robots::G1::Configuration &configuration,
        const PATACON_settings &settings
    ) {
        const bool use_persistent_workspace =
            runtime_control_state().persistent_workspace_enabled;
        float *device_configuration = nullptr;
        bool *device_success = nullptr;
        if (use_persistent_workspace) {
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
            settings.axis,
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
        if (!use_persistent_workspace) {
            cudaCheckError(cudaFree(device_success));
            cudaCheckError(cudaFree(device_configuration));
        }
        return success;
    }
#endif

    static_assert(
        robots::CollisionTraits<robots::FrankaSingle>::batch_size ==
            collision::FRANKA_FER_BATCH_SIZE &&
        robots::CollisionTraits<robots::FrankaSingle>::fine_sphere_count ==
            collision::FRANKA_FER_SPHERE_COUNT &&
        robots::CollisionTraits<robots::FrankaSingle>::approximate_sphere_count ==
            collision::FRANKA_FER_APPROX_SPHERE_COUNT &&
        robots::CollisionTraits<robots::FrankaSingle>::joint_flag_stride ==
            collision::FRANKA_COLLISION_JOINT_FLAG_STRIDE &&
        robots::CollisionTraits<robots::FrankaSingle>::transform_slots == 1 &&
        robots::CollisionTraits<robots::Franka>::fine_sphere_count ==
            2 * collision::FRANKA_FER_SPHERE_COUNT &&
        robots::CollisionTraits<robots::Franka>::approximate_sphere_count ==
            2 * collision::FRANKA_FER_APPROX_SPHERE_COUNT &&
        robots::CollisionTraits<robots::Franka>::joint_flag_stride ==
            collision::FRANKA_COLLISION_JOINT_FLAG_STRIDE &&
        robots::CollisionTraits<robots::Franka>::transform_slots == 2,
        "Franka collision traits differ from the FER sphere model"
    );
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
    constexpr int MAX_GRANULARITY = 16;

#ifdef PATACON_RUNTIME_STORAGE_EXTERN
    extern __device__ volatile int solved;
    extern __device__ volatile int atomic_free_index[2];
    extern __device__ volatile int nodes_size[2];
    extern __device__ volatile int completed_nodes[2];
#else
    __device__ volatile int solved = 0;
    __device__ volatile int atomic_free_index[2]; // separate for tree_a and tree_b
    __device__ volatile int nodes_size[2];
    __device__ volatile int completed_nodes[2]; // track completed nodes for each tree
#endif
    // Ready entries are request generations, not booleans. Persistent
    // replanning can therefore invalidate old nodes without a full memset.
#ifdef PATACON_RUNTIME_STORAGE_EXTERN
    extern __device__ int current_search_generation;
#else
    __device__ int current_search_generation = 1;
#endif
    constexpr int MAX_PATH_NODES = 5000;
    constexpr int MAX_PATH_STORAGE =
        MAX_PATH_NODES * ppln::robots::G1::dimension;
#ifdef PATACON_RUNTIME_STORAGE_EXTERN
    extern __device__ float path[2][MAX_PATH_STORAGE];
    extern __device__ int path_size[2];
    extern __device__ float cost;
    extern __device__ int reached_goal_idx;
    extern __device__ int connection_tree_id;
    extern __device__ int connection_node_idx;
    extern __device__ int connection_other_tree_id;
    extern __device__ int connection_other_node_idx;
    extern __device__ int solved_iters;
    extern __constant__ PATACON_settings d_settings;
    extern __constant__ unsigned long long patacon_time_limit_ns;
#else
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
#endif

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
#ifdef PATACON_RUNTIME_STORAGE_EXTERN
    extern __device__ unsigned long long diagnostic_counters[
        DIAGNOSTIC_COUNTER_COUNT
    ];
#else
    __device__ unsigned long long diagnostic_counters[
        DIAGNOSTIC_COUNTER_COUNT
    ];
#endif

    __device__ __forceinline__ void diagnostic_increment(
        DiagnosticCounter counter
    ) {
        if (d_settings.collect_diagnostics) {
            atomicAdd(&diagnostic_counters[static_cast<int>(counter)], 1ULL);
        }
    }

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
            return d_settings.axis ? 7 : FFW_SG2_TANGENT_DIM;
        } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
            return FFW_SG2_MOBILITY_TANGENT_DIM;
        } else if constexpr (std::is_same_v<Robot, robots::G1>) {
            return collision::g1_tangent_dim(d_settings.axis);
        } else if constexpr (std::is_same_v<Robot, robots::IgrisC>) {
            return collision::IGRIS_C_TANGENT_DIM;
        } else if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
            return d_settings.axis ? 5 : 7;
        } else if constexpr (std::is_same_v<Robot, robots::Franka>) {
            return d_settings.axis ? 6 : 8;
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
    static void __device__ shuffle_array(
        float *array,
        int n,
        curandState &state
    ) {
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
    static __global__ void init_rng(
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

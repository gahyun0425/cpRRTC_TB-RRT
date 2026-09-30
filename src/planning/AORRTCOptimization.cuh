#pragma once

#include "AORRTC.hh"
#include <climits>


/*
PATACON: Each block works to add a config to the tree (either start or goal depending on balance)
*/


namespace AORRTC {
    using namespace ppln;

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

    // AORRTC has independent state so the legacy first-solution planner keeps
    // its original signaling and path buffers.
    __device__ volatile int aorrtc_stop_requested = 0;
    __device__ volatile int aorrtc_solution_found = 0;
    // Set only for the current fresh-tree search.  The host clears this when
    // Algorithm 1 restarts RRT-Connect with a tighter c_max.
    __device__ volatile int aorrtc_search_solution_found = 0;
    __device__ int aorrtc_best_lock = 0;
    __device__ int aorrtc_solution_updates = 0;
    __device__ int aorrtc_best_iters = 0;
    __device__ float aorrtc_best_cost = FLT_MAX;
    __device__ float aorrtc_initial_cost = 0.0f;

    struct AORRTCDeviceSolutionUpdate {
        int update_index;
        int source_tree_id;
        int source_node_idx;
        int target_tree_id;
        int target_node_idx;
        int iteration;
        float cost;
    };

    __device__ AORRTCDeviceSolutionUpdate *aorrtc_update_records = nullptr;
    __device__ int aorrtc_update_capacity = 0;
    __device__ int aorrtc_update_overflow = 0;
    __constant__ AORRTC_settings d_settings;

    constexpr int MAX_GRANULARITY = 16;
    constexpr int MAX_THREADS_PER_BLOCK = 4*MAX_GRANULARITY;

    // PATACON projected motion shared buffer용
    constexpr int MAX_ROBOT_DIM = ppln::robots::G1::dimension;
    constexpr int FFW_SG2_TANGENT_DIM = 9; // 기본 constraint에 따른 tangent dim 15 - 6 = 9. 최대로 필요한 tangent 차원
    constexpr int MAX_TANGENT_DIM = ppln::collision::G1_TANGENT_DIM;

    constexpr int FFW_SG2_TANGENT_BASIS_SIZE = ppln::robots::FfwSg2::dimension * FFW_SG2_TANGENT_DIM;
    constexpr int FFW_SG2_MOBILITY_TANGENT_BASIS_STORAGE_SIZE =
        ppln::robots::FfwSg2Mobility::dimension *
        FFW_SG2_MOBILITY_TANGENT_DIM;
    constexpr int BLOCK_SIZE = 64; // RNG와 Halton 상태 초기화 커널의 thread block 크기. halton 수열의 random성을 위해 RNG 사용
    constexpr float UNWRITTEN_VAL = -9999.0f; // 미작성 configuration 메모리 표기 sentinel 값. 유효성 판단을 위한 flag로 사용


    // Shared initial-search primitives. AORRTC keeps only its cost-aware
    // state, parent resampling, and bounded-search implementation here.
    using PATACON::TangentSpaceTraits;
    using PATACON::HaltonState;
    using PATACON::patacon_active_tangent_dim;
    using PATACON::patacon_joint_distance_weight;
    using PATACON::patacon_sq_config_distance;
    using PATACON::patacon_config_distance;
    using PATACON::init_rng;
    using PATACON::init_halton;
    using PATACON::setup_environment_on_device;
    using PATACON::cleanup_environment_on_device;
    using PATACON::reset_to_unwritten_state;
    using PATACON::patacon_project_motion;
    using PATACON::patacon_store_tangent_basis;
    using PATACON::patacon_sample_tangent_config;
    using PATACON::patacon_constraint_error_norm;
    using PATACON::init_root_ts_banks;
    using PATACON::patacon_shared_config_distance;
    using PATACON::patacon_detailed_env_collision_check;
    using PATACON::patacon_detailed_self_collision_check;
    using PATACON::patacon_attached_object_collision_check_approx;
    using PATACON::patacon_attached_object_collision_check;
    using PATACON::patacon_reserve_slot;
    using PATACON::patacon_register_node_in_ts;
    using PATACON::patacon_project_target_direction_to_tangent;

    template <typename Robot>
    __device__ __forceinline__ float aorrtc_root_distance_lower_bound(
        int tree_id,
        float **nodes,
        int num_goals,
        const float *configuration
    ) {
        if (tree_id == 0) {
            return patacon_config_distance<Robot>(
                nodes[0],
                configuration
            );
        }

        float minimum = FLT_MAX;
        for (int goal_index = 0; goal_index < num_goals; goal_index++) {
            minimum = fminf(
                minimum,
                patacon_config_distance<Robot>(
                    &nodes[1][goal_index * Robot::dimension],
                    configuration
                )
            );
        }
        return minimum;
    }


    __device__ __forceinline__ float aorrtc_read_best_cost() {
        return atomicAdd(&aorrtc_best_cost, 0.0f);
    }


    template <typename Robot, bool TRACE_SOLUTION_UPDATES>
    __device__ __forceinline__ void aorrtc_try_store_solution(
        int current_tree_id,
        int other_tree_id,
        int current_node_index,
        int other_node_index,
        float *current_costs,
        float *other_costs,
        float bridge_cost,
        int iteration
    ) {
        // Algorithm 1 returns the first feasible path for the current bounded
        // RRT-Connect search, then the host restarts with the new c_max.
        if (aorrtc_search_solution_found != 0) {
            return;
        }

        const float candidate_cost =
            current_costs[current_node_index]
            + other_costs[other_node_index]
            + bridge_cost;

        if (candidate_cost + d_settings.cost_improvement_epsilon
            >= aorrtc_read_best_cost()) {
            return;
        }

        while (atomicCAS(&aorrtc_best_lock, 0, 1) != 0) {
        }

        if (aorrtc_search_solution_found == 0
            && candidate_cost + d_settings.cost_improvement_epsilon
                < aorrtc_best_cost) {
            const int update_index = aorrtc_solution_updates;
            if (update_index == 0) {
                aorrtc_initial_cost = candidate_cost;
            }
            connection_tree_id = current_tree_id;
            connection_node_idx = current_node_index;
            connection_other_tree_id = other_tree_id;
            connection_other_node_idx = other_node_index;
            aorrtc_best_iters = iteration;
            aorrtc_best_cost = candidate_cost;
            if constexpr (TRACE_SOLUTION_UPDATES) {
                if (aorrtc_update_records != nullptr
                    && update_index < aorrtc_update_capacity) {
                    AORRTCDeviceSolutionUpdate &record =
                        aorrtc_update_records[update_index];
                    record.update_index = update_index;
                    record.source_tree_id = current_tree_id;
                    record.source_node_idx = current_node_index;
                    record.target_tree_id = other_tree_id;
                    record.target_node_idx = other_node_index;
                    record.iteration = iteration;
                    record.cost = candidate_cost;
                }
                else {
                    aorrtc_update_overflow = 1;
                }
            }
            aorrtc_solution_updates = update_index + 1;
            __threadfence();
            aorrtc_solution_found = 1;
            aorrtc_search_solution_found = 1;
            // A fresh AORRTC search returns its first feasible solution.
            // Ask the remaining blocks in this launch to exit at their next
            // block-uniform stop check; the host then restarts with tighter c_max.
            aorrtc_stop_requested = 1;
        }

        atomicExch(&aorrtc_best_lock, 0);
    }

    __global__ void reset_device_variables_kernel() {
        solved = 0;
        solved_iters = 0;
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

        aorrtc_stop_requested = 0;
        aorrtc_solution_found = 0;
        aorrtc_search_solution_found = 0;
        aorrtc_best_lock = 0;
        aorrtc_solution_updates = 0;
        aorrtc_best_iters = 0;
        aorrtc_best_cost = FLT_MAX;
        aorrtc_initial_cost = 0.0f;
        aorrtc_update_records = nullptr;
        aorrtc_update_capacity = 0;
        aorrtc_update_overflow = 0;
    }

    void reset_device_variables() {
        reset_device_variables_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
    }

    // Conservative validator used only by Algorithm 1 Lines 27-32 when a
    // newly projected x_new is reconsidered with a lower-cost parent.  The
    // endpoint is kept fixed; the interpolated edge must already satisfy the
    // equality constraint and collision checks.  This avoids silently changing
    // x_new during parent resampling.
    template <typename Robot>
    __device__ __forceinline__ bool aorrtc_validate_fixed_edge(
        const float *q_start,
        const float *q_target,
        volatile unsigned char *waypoint_valid,
        volatile float *sphere_pos,
        volatile float *sphere_pos_approx,
        volatile int *link_CC,
        float *T,
        ppln::collision::Environment<float> *env,
        volatile unsigned int *local_cc_result,
        int tid
    ) {
        static constexpr int dim = Robot::dimension;
        using Collision = robots::CollisionTraits<Robot>;

        const int waypoint = tid / 4 + 1;
        float interp_cfg[dim];
        const float alpha = static_cast<float>(waypoint)
            / static_cast<float>(d_settings.granularity);

        #pragma unroll
        for (int joint = 0; joint < dim; joint++) {
            interp_cfg[joint] = q_start[joint]
                + alpha * (q_target[joint] - q_start[joint]);
        }

        if ((tid & 3) == 0) {
            bool valid = true;
            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                const float residual =
                    patacon_constraint_error_norm<Robot>(interp_cfg);
                valid = isfinite(residual)
                    && residual <= d_settings.projection_task_tolerance;
            }
            waypoint_valid[waypoint] = valid ? 1u : 0u;
        }
        __syncthreads();

        if (tid == 0) {
            local_cc_result[0] = 0;
            for (int wp = 1; wp <= d_settings.granularity; wp++) {
                if (waypoint_valid[wp] == 0u) {
                    local_cc_result[0] = 1u;
                    break;
                }
            }
        }
        __syncthreads();

        if (local_cc_result[0] != 0u) {
            return false;
        }

        for (int r = tid;
             r < Collision::joint_flag_stride * Collision::batch_size;
             r += blockDim.x) {
            link_CC[r] = 0;
        }
        __syncthreads();

        int detailed_FK = 0;
        ppln::collision::fk_approx<Robot>(
            interp_cfg,
            sphere_pos_approx,
            T,
            tid
        );
        __syncthreads();

        const bool env_collision_approx =
            not ppln::collision::env_collision_check_approx<Robot>(
                sphere_pos_approx,
                link_CC,
                env,
                tid
            );
        atomicOr(
            (unsigned int *)&local_cc_result[0],
            env_collision_approx ? 1u : 0u
        );

        const bool attached_object_collision_approx =
            not patacon_attached_object_collision_check_approx<Robot>(
                interp_cfg,
                sphere_pos_approx,
                env,
                tid,
                local_cc_result
            );
        atomicOr(
            (unsigned int *)&local_cc_result[0],
            attached_object_collision_approx ? 1u : 0u
        );
        __syncthreads();

        if (local_cc_result[0] == 1u) {
            if (tid == 0) {
                local_cc_result[0] = 0u;
            }
            __syncthreads();

            ppln::collision::fk<Robot>(interp_cfg, sphere_pos, T, tid);
            detailed_FK = 1;
            __syncthreads();

            const bool env_collision =
                not patacon_detailed_env_collision_check<Robot>(
                    sphere_pos,
                    link_CC,
                    env,
                    tid,
                    local_cc_result
                );
            atomicOr(
                (unsigned int *)&local_cc_result[0],
                env_collision ? 1u : 0u
            );

            const bool attached_object_collision =
                not patacon_attached_object_collision_check<Robot>(
                    interp_cfg,
                    sphere_pos,
                    env,
                    tid,
                    local_cc_result
                );
            atomicOr(
                (unsigned int *)&local_cc_result[0],
                attached_object_collision ? 1u : 0u
            );
            __syncthreads();
        }

        for (int r = tid;
             r < Collision::joint_flag_stride * Collision::batch_size;
             r += blockDim.x) {
            link_CC[r] = 0;
        }
        __syncthreads();

        if (local_cc_result[0] == 0u) {
            const bool self_collision_approx =
                not ppln::collision::self_collision_check_approx<Robot>(
                    sphere_pos_approx,
                    link_CC,
                    tid
                );
            atomicOr(
                (unsigned int *)&local_cc_result[0],
                self_collision_approx ? 1u : 0u
            );
            __syncthreads();

            if (local_cc_result[0] == 1u) {
                if (tid == 0) {
                    local_cc_result[0] = 0u;
                }
                __syncthreads();

                if (detailed_FK == 0) {
                    ppln::collision::fk<Robot>(
                        interp_cfg,
                        sphere_pos,
                        T,
                        tid
                    );
                    detailed_FK = 1;
                    __syncthreads();
                }

                const bool self_collision =
                    not patacon_detailed_self_collision_check<Robot>(
                        sphere_pos,
                        link_CC,
                        tid,
                        local_cc_result
                    );
                atomicOr(
                    (unsigned int *)&local_cc_result[0],
                    self_collision ? 1u : 0u
                );
                __syncthreads();
            }
        }

        return local_cc_result[0] == 0u;
    }


    // Algorithm 1 Lines 27-32.  Starting from the already validated parent,
    // repeatedly sample a lower cost bound and look for a lower-cost parent.
    // A replacement parent is accepted only after validating the exact fixed
    // edge to x_new.  Tangent-bundle planners keep the replacement in the same
    // tangent-space lane set so the current planner's TS semantics are kept.
    template <typename Robot>
    __device__ __forceinline__ int aorrtc_resample_parent(
        int tree_id,
        float **all_nodes,
        float *tree_nodes,
        float *tree_costs,
        int *tree_ready,
        int *tree_ts_lane_head,
        int *tree_next_in_ts,
        int selected_ts_id,
        int tree_size,
        int initial_parent,
        const float *x_new,
        int num_goals,
        curandState *rng_states,
        int bid,
        float *sdata,
        int *sindex,
        int *shared_parent,
        float *shared_parent_cost,
        float *shared_c_rand,
        bool *shared_done,
        volatile unsigned char *waypoint_valid,
        volatile float *sphere_pos,
        volatile float *sphere_pos_approx,
        volatile int *link_CC,
        float *T,
        ppln::collision::Environment<float> *env,
        volatile unsigned int *local_cc_result,
        int tid
    ) {
        if (tid == 0) {
            *shared_parent = initial_parent;
            *shared_parent_cost =
                tree_costs[initial_parent]
                + patacon_config_distance<Robot>(
                    &tree_nodes[initial_parent * Robot::dimension],
                    x_new
                );
            *shared_done = false;
        }
        __syncthreads();

        for (int attempt = 0;
             attempt < d_settings.aorrtc_max_parent_resamples;
             attempt++) {
            if (tid == 0) {
                const float lower =
                    aorrtc_root_distance_lower_bound<Robot>(
                        tree_id,
                        all_nodes,
                        num_goals,
                        x_new
                    );

                if (!isfinite(lower)
                    || lower + d_settings.cost_improvement_epsilon
                        >= *shared_parent_cost) {
                    *shared_done = true;
                }
                else {
                    const float unit = fminf(
                        curand_uniform(&rng_states[bid]),
                        1.0f - FLT_EPSILON
                    );
                    *shared_c_rand = lower
                        + unit * (*shared_parent_cost - lower);
                    *shared_done = false;
                }
            }
            __syncthreads();

            if (*shared_done) {
                break;
            }

            float local_score = FLT_MAX;
            int local_index = -1;

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                int node_idx = tree_ts_lane_head[
                    selected_ts_id * MAX_THREADS_PER_BLOCK + tid
                ];
                while (node_idx >= 0) {
                    if (tree_ready[node_idx] != 0) {
                        const float distance =
                            patacon_config_distance<Robot>(
                                &tree_nodes[node_idx * Robot::dimension],
                                x_new
                            );
                        const float candidate_cost =
                            tree_costs[node_idx] + distance;
                        if (candidate_cost
                            + d_settings.cost_improvement_epsilon
                            < *shared_c_rand) {
                            const float cost_difference =
                                *shared_c_rand - tree_costs[node_idx];

                            const float score =
                                d_settings.aorrtc_config_weight
                                    * (distance * distance)
                                + d_settings.aorrtc_cost_weight
                                    * (cost_difference * cost_difference);
                            if (score < local_score) {
                                local_score = score;
                                local_index = node_idx;
                            }
                        }
                    }
                    node_idx = tree_next_in_ts[node_idx];
                }
            }
            else {
                for (int node_idx = tid;
                     node_idx < tree_size;
                     node_idx += blockDim.x) {
                    if (tree_ready[node_idx] == 0) {
                        continue;
                    }
                    const float distance =
                        patacon_config_distance<Robot>(
                            &tree_nodes[node_idx * Robot::dimension],
                            x_new
                        );
                    const float candidate_cost =
                        tree_costs[node_idx] + distance;
                    if (candidate_cost
                        + d_settings.cost_improvement_epsilon
                        < *shared_c_rand) {
                        const float cost_difference =
                            *shared_c_rand - tree_costs[node_idx];

                        const float score =
                            d_settings.aorrtc_config_weight
                                * (distance * distance)
                            + d_settings.aorrtc_cost_weight
                                * (cost_difference * cost_difference);
                        if (score < local_score) {
                            local_score = score;
                            local_index = node_idx;
                        }
                    }
                }
            }

            sdata[tid] = local_score;
            sindex[tid] = local_index;
            __syncthreads();

            for (unsigned int stride = blockDim.x / 2;
                 stride > 0;
                 stride >>= 1) {
                if (tid < static_cast<int>(stride)
                    && sdata[tid + stride] < sdata[tid]) {
                    sdata[tid] = sdata[tid + stride];
                    sindex[tid] = sindex[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                if (sindex[0] < 0 || sindex[0] == *shared_parent) {
                    *shared_done = true;
                }
            }
            __syncthreads();

            if (*shared_done) {
                break;
            }

            const int candidate_parent = sindex[0];
            const bool valid = aorrtc_validate_fixed_edge<Robot>(
                &tree_nodes[candidate_parent * Robot::dimension],
                x_new,
                waypoint_valid,
                sphere_pos,
                sphere_pos_approx,
                link_CC,
                T,
                env,
                local_cc_result,
                tid
            );
            __syncthreads();

            if (tid == 0) {
                if (!valid) {
                    // Algorithm 1 stops and keeps x_p, the previous valid parent.
                    *shared_done = true;
                }
                else {
                    *shared_parent = candidate_parent;
                    *shared_parent_cost =
                        tree_costs[candidate_parent]
                        + patacon_config_distance<Robot>(
                            &tree_nodes[
                                candidate_parent * Robot::dimension
                            ],
                            x_new
                        );
                }
            }
            __syncthreads();

            if (*shared_done) {
                break;
            }
        }

        return *shared_parent;
    }


    template <typename Robot, bool TraceTrees, bool AORRTC>
    __global__ void
    // __launch_bounds__(128, 8)
    patacon(
        float **nodes,
        int **parents,
        float **node_costs,
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
        HaltonState<Robot> *halton_states,
        curandState *rng_states,
        int *block_tree_ids,
        ppln::collision::Environment<float> *env,
        int num_goals,
        int round_index,
        bool persistent_initial_search
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
        __shared__ volatile unsigned int local_cc_result[1];
        __shared__ float *t_nodes;
        __shared__ float *o_nodes;
        __shared__ int *t_parents;
        __shared__ int *o_parents;
        __shared__ float *t_node_costs;
        __shared__ float *o_node_costs;
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
        __shared__ float delta[dim];
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
        // AORRTC Algorithm 1 Lines 27-32 parent-resampling state.
        __shared__ float aorrtc_new_config[MAX_ROBOT_DIM];
        __shared__ int aorrtc_resample_parent_idx;
        __shared__ float aorrtc_resample_parent_cost;
        __shared__ float aorrtc_resample_crand;
        __shared__ bool aorrtc_resample_done;
        // 실제로 검사할 edge 개수
        // FFW-SG2에서는 concon_count, 다른 robot에서는 기존처럼 1
        __shared__ int extend_edge_count;
        __shared__ int index;
        __shared__ bool should_skip;
        __shared__ bool aorrtc_bound_active;
        __shared__ bool aorrtc_sample_valid;
        __shared__ float aorrtc_sample_cost;
        __shared__ float aorrtc_best_cost_snapshot;
        // PATACON CONNECT state
        // projection 전 target까지 거리
        __shared__ float connect_distance_before;
        // projection 후 target까지 거리
        __shared__ float connect_distance_after;
        // projection 후 실제로 target에 가까워졌는지
        __shared__ bool connect_made_progress;
        // target 도달 여부
        __shared__ bool connection_reached_shared;
        // Make collision-check branch decisions block-uniform before thread 0
        // resets the shared collision accumulator.
        __shared__ bool run_detailed_env_check;
        __shared__ bool run_self_collision_check;
        __shared__ bool run_detailed_self_check;
        __shared__ unsigned int n_extensions;
        // CONNECT 시작 시 상대 tree에서 한 번 선택하고 끝까지 유지할 target
        __shared__ int connect_target_idx;
        __shared__ float *connect_target_node;

        // 새 TB-RRT CONNECT 상태
        __shared__ bool connect_failed;
        __shared__ bool connect_reached;
        // PATACON parallel projection shared memory
        __align__(16) __shared__ volatile float motion_segment[(MAX_GRANULARITY + 1) * MAX_ROBOT_DIM];
        __align__(16) __shared__ volatile float motion_segment_next[(MAX_GRANULARITY + 1) * MAX_ROBOT_DIM];
        __shared__ volatile unsigned char motion_projection_valid[MAX_GRANULARITY + 1];
        __shared__ volatile int motion_projection_prog[1];
        __shared__ volatile unsigned int motion_projection_success[1];
        __align__(16) __shared__ volatile float sphere_pos[Collision::fine_sphere_count * Collision::batch_size * 3];
        __align__(16) __shared__ volatile float sphere_pos_approx[Collision::approximate_sphere_count * Collision::batch_size * 3];
        __align__(16) __shared__ volatile int link_CC[Collision::joint_flag_stride * Collision::batch_size];
        __align__(16) __shared__ float T[Collision::batch_size * Collision::transform_slots * 16];

        int iter = AORRTC ? round_index - 1 : 0;

        while (true) {
            if (tid == 0) {
                // printf("iter: %d\n", iter);
                // printf("tree size: %d\n", atomic_free_index[0]);
                iter++;
                if constexpr (!AORRTC) {
                    if (iter > d_settings.max_iters) {
                        atomicCAS((int *)&solved, 0, -1);
                    }
                }

                // Tree selection. During the first persistent search, use
                // the same in-kernel balance logic as PATACON. After the first
                // solution, each AORRTC launch performs one iteration, so the
                // previous per-block tree choice is preserved in block_tree_ids.
                if constexpr (AORRTC) {
                    if (persistent_initial_search) {
                        if (d_settings.balance == 0 || iter == 1) {
                            t_tree_id =
                                (bid < (d_settings.num_new_configs / 2)) ? 0 : 1;
                            o_tree_id = 1 - t_tree_id;
                        }
                        else if (d_settings.balance == 1
                            && abs(atomic_free_index[0] - atomic_free_index[1])
                                < 1.5f * d_settings.num_new_configs) {
                            const float ratio = atomic_free_index[0]
                                / static_cast<float>(
                                    atomic_free_index[0]
                                    + atomic_free_index[1]
                                );
                            const float balance_factor = 1.0f - ratio;
                            t_tree_id =
                                bid < static_cast<int>(
                                    d_settings.num_new_configs
                                    * balance_factor
                                )
                                ? 0 : 1;
                            o_tree_id = 1 - t_tree_id;
                        }
                        else if (d_settings.balance == 1) {
                            const float ratio = atomic_free_index[0]
                                / static_cast<float>(
                                    atomic_free_index[0]
                                    + atomic_free_index[1]
                                );
                            t_tree_id =
                                ratio < d_settings.tree_ratio ? 0 : 1;
                            o_tree_id = 1 - t_tree_id;
                        }
                        else if (d_settings.balance == 2) {
                            const float ratio =
                                abs(atomic_free_index[t_tree_id]
                                    - atomic_free_index[o_tree_id])
                                / static_cast<float>(
                                    atomic_free_index[t_tree_id]
                                );
                            if (ratio < d_settings.tree_ratio) {
                                t_tree_id = 1 - t_tree_id;
                                o_tree_id = 1 - t_tree_id;
                            }
                        }
                    }
                    else {
                        if (d_settings.balance == 0 || round_index == 1) {
                            t_tree_id =
                                (bid < (d_settings.num_new_configs / 2)) ? 0 : 1;
                            o_tree_id = 1 - t_tree_id;
                        }
                        else {
                            t_tree_id = block_tree_ids[bid];
                            if (t_tree_id != 0 && t_tree_id != 1) {
                                t_tree_id =
                                    (bid < (d_settings.num_new_configs / 2))
                                    ? 0 : 1;
                            }
                            o_tree_id = 1 - t_tree_id;

                            if (d_settings.balance == 1
                                && abs(atomic_free_index[0] - atomic_free_index[1])
                                    < 1.5f * d_settings.num_new_configs) {
                                const float ratio = atomic_free_index[0]
                                    / static_cast<float>(
                                        atomic_free_index[0]
                                        + atomic_free_index[1]
                                    );
                                const float balance_factor = 1.0f - ratio;
                                t_tree_id =
                                    bid < static_cast<int>(
                                        d_settings.num_new_configs
                                        * balance_factor
                                    )
                                    ? 0 : 1;
                                o_tree_id = 1 - t_tree_id;
                            }
                            else if (d_settings.balance == 1) {
                                const float ratio = atomic_free_index[0]
                                    / static_cast<float>(
                                        atomic_free_index[0]
                                        + atomic_free_index[1]
                                    );
                                t_tree_id =
                                    ratio < d_settings.tree_ratio ? 0 : 1;
                                o_tree_id = 1 - t_tree_id;
                            }
                            else if (d_settings.balance == 2) {
                                const float ratio =
                                    abs(atomic_free_index[t_tree_id]
                                        - atomic_free_index[o_tree_id])
                                    / static_cast<float>(
                                        atomic_free_index[t_tree_id]
                                    );
                                if (ratio < d_settings.tree_ratio) {
                                    t_tree_id = 1 - t_tree_id;
                                    o_tree_id = 1 - t_tree_id;
                                }
                            }
                        }

                        block_tree_ids[bid] = t_tree_id;
                    }
                }
                else {
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
                }

                t_nodes = nodes[t_tree_id];
                o_nodes = nodes[o_tree_id];
                t_parents = parents[t_tree_id];
                o_parents = parents[o_tree_id];
                if constexpr (AORRTC) {
                    t_node_costs = node_costs[t_tree_id];
                    o_node_costs = node_costs[o_tree_id];
                }
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
                } else {
                    t_ts_lane_head = nullptr;
                    t_node_next_in_ts = nullptr;
                    selected_ts_id = -1;
                }
                t_node_ready =node_ready[t_tree_id];
                o_node_ready = node_ready[o_tree_id];
                t_tree_size = atomic_free_index[t_tree_id];

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
                                const int candidate_root_idx =
                                    t_ts_root_node_idx[candidate_ts];

                                if (candidate_root_idx < 0
                                    || candidate_root_idx >= t_tree_size
                                    || t_node_ready[candidate_root_idx] == 0) {
                                    continue;
                                }

                                selected_ts_id =candidate_ts;
                                selected_ts_root_idx =candidate_root_idx;

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

                local_cc_result[0] = 0;
            }

            __syncthreads();

            if constexpr (TraceTrees) {
                if (tid == 0) {
                    if constexpr (AORRTC) {
                        should_skip = aorrtc_stop_requested != 0;
                    }
                    else {
                        should_skip = (solved != 0);
                    }
                }
                __syncthreads();
                if (should_skip) {
                    return;
                }
            }
            else {
                if (tid == 0) {
                    if constexpr (AORRTC) {
                        should_skip = (aorrtc_stop_requested != 0);
                    }
                    else {
                        should_skip = (solved != 0);
                    }
                }
                __syncthreads();
                if (should_skip) {
                    return;
                }
            }

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                // 사용할 수 있는 Tangent Space를 찾지 못했으면 이번 EXTEND iteration을 버린다.
                if (selected_ts_id < 0) {
                    if constexpr (AORRTC) {
                        if (persistent_initial_search) {
                            continue;
                        }
                        return;
                    }
                    else {
                        continue;
                    }
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

            if constexpr (AORRTC) {
                __syncthreads();
                if (tid == 0) {
                    aorrtc_best_cost_snapshot = aorrtc_read_best_cost();
                    aorrtc_bound_active =
                        aorrtc_solution_found != 0
                        && aorrtc_best_cost_snapshot < FLT_MAX;
                    aorrtc_sample_valid = true;
                    aorrtc_sample_cost = FLT_MAX;

                    if (aorrtc_bound_active) {
                        const float lower_from =
                            aorrtc_root_distance_lower_bound<Robot>(
                                t_tree_id,
                                nodes,
                                num_goals,
                                (float *)config
                            );
                        const float lower_to =
                            aorrtc_root_distance_lower_bound<Robot>(
                                o_tree_id,
                                nodes,
                                num_goals,
                                (float *)config
                            );
                        const float available_cost =
                            aorrtc_best_cost_snapshot - lower_to - lower_from;

                        if (available_cost <= d_settings.cost_improvement_epsilon) {
                            aorrtc_sample_valid = false;
                        }
                        else {
                            const float unit_sample = fminf(
                                curand_uniform(&rng_states[bid]),
                                1.0f - FLT_EPSILON
                            );
                            aorrtc_sample_cost =
                                lower_from + unit_sample * available_cost;
                        }
                    }
                }
                __syncthreads();

                if (!aorrtc_sample_valid) {
                    return;
                }
            }

            // reset link_CC every iteration
            for (int r = tid; r < Collision::joint_flag_stride * Collision::batch_size; r += blockDim.x) {
                link_CC[r]=0;
            }

            __syncthreads();

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
                            patacon_sq_config_distance<Robot>(
                                (float *)&t_node_ts_q[node_idx * dim],
                                (float *)config
                            );
                        float candidate_score = candidate_dist;
                        bool candidate_allowed = true;
                        if constexpr (AORRTC) {
                            if (aorrtc_bound_active) {
                                const float actual_distance =
                                    patacon_config_distance<Robot>(
                                        &t_nodes[node_idx * dim],
                                        (float *)config
                                    );
                                candidate_allowed =
                                    t_node_costs[node_idx] + actual_distance
                                    < aorrtc_sample_cost;
                                const float cost_difference =
                                    aorrtc_sample_cost - t_node_costs[node_idx];

                                candidate_score =
                                    d_settings.aorrtc_config_weight
                                        * candidate_dist
                                    + d_settings.aorrtc_cost_weight
                                        * (cost_difference * cost_difference);
                            }
                        }
                        // 지금까지 본 노드 중 가장 가까우면 기록
                        if (candidate_allowed && candidate_score < local_min_dist) {
                            local_min_dist =candidate_score;
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
                        patacon_sq_config_distance<Robot>(
                            (float *)&t_nodes[i * dim],
                            (float *)config
                        );
                    float candidate_score = candidate_dist;
                    bool candidate_allowed = true;
                    if constexpr (AORRTC) {
                        if (aorrtc_bound_active) {
                            const float configuration_distance =
                                sqrtf(candidate_dist);
                            candidate_allowed =
                                t_node_costs[i] + configuration_distance
                                < aorrtc_sample_cost;
                            const float cost_difference =
                                aorrtc_sample_cost - t_node_costs[i];

                            candidate_score =
                                d_settings.aorrtc_config_weight
                                    * candidate_dist
                                + d_settings.aorrtc_cost_weight
                                    * (cost_difference * cost_difference);
                        }
                    }

                    if (candidate_allowed && candidate_score < local_min_dist) {
                        local_min_dist =candidate_score;
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
                    // In bounded AORRTC search, sdata contains augmented-state
                    // score rather than squared configuration distance.
                    if constexpr (AORRTC) {
                        if (aorrtc_bound_active) {
                            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                                q_rand_dist = patacon_config_distance<Robot>(
                                    &t_node_ts_q[sindex[0] * dim],
                                    (float *)config
                                );
                            }
                            else {
                                q_rand_dist = patacon_config_distance<Robot>(
                                    &t_nodes[sindex[0] * dim],
                                    (float *)config
                                );
                            }
                        }
                        else {
                            q_rand_dist = sqrtf(sdata[0]);
                        }
                    }
                    else {
                        q_rand_dist =sqrtf(sdata[0]);
                    }

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
                    should_skip = zero_direction;
                }
            }
            __syncthreads();

            if (should_skip) {
                if constexpr (AORRTC) {
                    if (persistent_initial_search) {
                        continue;
                    }
                    return;
                }
                else {
                    continue;
                }
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
                        const float em_error =patacon_constraint_error_norm<Robot>(concon_probe); // constraint residual 검사

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

            const int waypoint = tid / 4 + 1;
            float interp_cfg[dim];
            
            for (int edge_step = 1; edge_step <= extend_edge_count; edge_step++) {
                // 이번 edge의 target 설정
                if (tid < dim) {
                    if constexpr (TangentSpaceTraits<Robot>::enabled) {
                        // selected Tangent Space 위 nominal target
                        // q_k =q_near_TS + k * range * extend_dir
                        concon_probe[tid] =nearest_ts_node[tid]+((float)edge_step*d_settings.range*extend_dir[tid]);
                    }
                    // config = 현재 실제 projected node
                    // concon_probe = 이번 nominal target
                    // 이 둘 사이를 granularity만큼 interpolation
                    delta[tid] =(concon_probe[tid]-config[tid])/(float)d_settings.granularity;
                }
                __syncthreads();

                // PATACON EXTEND
                // 1. q_near -> q_steer straight-line motion 생성
                // 2. FFW SG2는 analytic-Jacobian ParallelProject 수행
                // 3. projected waypoint들에 대해 기존 collision check 수행
                const bool projection_good = patacon_project_motion<Robot>(
                    config,
                    delta,
                    motion_segment,
                    motion_segment_next,
                    motion_projection_valid,
                    motion_projection_prog,
                    motion_projection_success,
                    tid
                );
                __syncthreads();

                // projection 후 마지막 waypoint가 실제로 tree에 저장할 endpoint
                // ξ_projected = [q_near, q'_1, ..., q'_N]
                // q'_N을 저장한다.
                float stored_edge_endpoint = 0.0f;

                if (tid < dim) {
                    stored_edge_endpoint =motion_segment[d_settings.granularity * dim + tid];
                }

                // waypoint ↔ thread mapping
                // waypoint 1 : tid  0,1,2,3
                // waypoint 2 : tid  4,5,6,7
                // ...
                // Projection에서는 lane 0만 analytic Jacobian 계산.
                // 이제 collision 단계에서는 다시 4개 thread가 모두 사용된다.
                bool motion_collision_free = false;

                // projection이 성공한 motion만 collision check
                if (projection_good) {
                    // 각 4-thread group이 자신이 담당하는 projected waypoint를 읽는다.
                    for (int i = 0; i < dim; i++) {
                        interp_cfg[i] =motion_segment[waypoint * dim + i];
                    }

                    // 새로운 motion collision check 시작
                    if (tid == 0) {
                        local_cc_result[0] = 0;
                    }
                    __syncthreads();

                    for (int r = tid; r < Collision::joint_flag_stride * Collision::batch_size; r += blockDim.x) {
                        link_CC[r] = 0;
                    }
                    __syncthreads();

                    // 기존 approximate FK / environment collision check
                    int detailed_FK = 0;

                    ppln::collision::fk_approx<Robot>(interp_cfg,sphere_pos_approx,T,tid);

                    __syncthreads();

                    // environment와 근사 collisoin 검사
                    bool config_in_collision2_approx = not ppln::collision::env_collision_check_approx<Robot>(sphere_pos_approx,link_CC,env,tid);

                    // 여러 thread의 충돌 검사 결과를 하나의 공유 결과로 합치는 코드
                    atomicOr((unsigned int *)&local_cc_result[0],config_in_collision2_approx ? 1u : 0u);

                    bool attached_object_collision_approx = not patacon_attached_object_collision_check_approx<Robot>(
                            interp_cfg,
                            sphere_pos_approx,
                            env,
                            tid,
                            local_cc_result
                        );

                    atomicOr((unsigned int *)&local_cc_result[0],attached_object_collision_approx ? 1u : 0u);

                    __syncthreads();

                    if (tid == 0) {
                        run_detailed_env_check = local_cc_result[0] == 1;
                    }
                    __syncthreads();

                    // approximate env collision 가능성이 있으면 detailed env collision
                    if (run_detailed_env_check) { // 근사 충돌 검사 결과 확인
                        if (tid == 0) {
                            local_cc_result[0] = 0;
                        }
                        __syncthreads();

                        ppln::collision::fk<Robot>(interp_cfg,sphere_pos,T,tid); // 정밀 FK 계산

                        detailed_FK = 1; // 정밀 FK 수행 여부. flag
                        __syncthreads();

                        // 정밀 충돌 검사
                        bool config_in_collision2 = not patacon_detailed_env_collision_check<Robot>(
                                sphere_pos,
                                link_CC,
                                env,
                                tid,
                                local_cc_result
                            );

                        // 각 thread 검사 합치기
                        atomicOr((unsigned int *)&local_cc_result[0],config_in_collision2 ? 1u : 0u);

                        bool attached_object_collision = not patacon_attached_object_collision_check<Robot>(
                                interp_cfg,
                                sphere_pos,
                                env,
                                tid,
                                local_cc_result
                            );

                        atomicOr((unsigned int *)&local_cc_result[0],attached_object_collision ? 1u : 0u);

                        __syncthreads();
                    }

                    // self collision용 link flag 초기화
                    for (int r = tid; r < Collision::joint_flag_stride * Collision::batch_size; r += blockDim.x) {
                        link_CC[r] = 0;
                    }

                    __syncthreads();

                    if (tid == 0) {
                        run_self_collision_check = local_cc_result[0] == 0;
                    }
                    __syncthreads();

                    // environment가 collision-free일 때 self collision 검사
                    if (run_self_collision_check) {
                        bool config_in_collision_approx = not ppln::collision::self_collision_check_approx<Robot>(
                                sphere_pos_approx,
                                link_CC,
                                tid
                            );

                        atomicOr((unsigned int *)&local_cc_result[0],config_in_collision_approx ? 1u : 0u);

                        __syncthreads();

                        if (tid == 0) {
                            run_detailed_self_check = local_cc_result[0] == 1;
                        }
                        __syncthreads();

                        // approximate self collision 가능성이 있으면 detailed 검사
                        if (run_detailed_self_check) {
                            if (tid == 0) {
                                local_cc_result[0] = 0;
                            }

                            __syncthreads();

                            if (detailed_FK == 0) {
                                ppln::collision::fk<Robot>(interp_cfg,sphere_pos,T,tid);

                                detailed_FK = 1;

                                __syncthreads();
                            }

                            bool config_in_collision =not patacon_detailed_self_collision_check<Robot>(
                                    sphere_pos,
                                    link_CC,
                                    tid,
                                    local_cc_result
                                ); 

                            atomicOr((unsigned int *)&local_cc_result[0],config_in_collision ? 1u : 0u);

                            __syncthreads();
                        }
                    }

                    motion_collision_free =(local_cc_result[0] == 0);
                }
                __syncthreads();


                // projection도 성공하고 collision도 없어야 edge 성공
                bool edge_good = projection_good && motion_collision_free;

                __syncthreads();

                // 현재 ConCon edge가 실패하면 이후 edge는 검사하지 않는다.
                if (!edge_good) {
                    break;
                }

                if (edge_good) {
                    if constexpr (AORRTC) {
                        // The paper/VAMP implementation obtains the initial
                        // solution with the underlying RRT-Connect planner.
                        // Cost-bound parent resampling therefore starts only
                        // after the first solution established c_max.
                        if (aorrtc_bound_active && tid < dim) {
                            aorrtc_new_config[tid] = stored_edge_endpoint;
                        }
                        __syncthreads();

                        if (aorrtc_bound_active) {
                            const int resample_tree_size = atomicAdd(
                                (int *)&atomic_free_index[t_tree_id],
                                0
                            );
                            const int resampled_parent =
                                aorrtc_resample_parent<Robot>(
                                t_tree_id,
                                nodes,
                                t_nodes,
                                t_node_costs,
                                t_node_ready,
                                t_ts_lane_head,
                                t_node_next_in_ts,
                                selected_ts_id,
                                resample_tree_size,
                                concon_parent_idx,
                                aorrtc_new_config,
                                num_goals,
                                rng_states,
                                bid,
                                sdata,
                                sindex,
                                &aorrtc_resample_parent_idx,
                                &aorrtc_resample_parent_cost,
                                &aorrtc_resample_crand,
                                &aorrtc_resample_done,
                                motion_projection_valid,
                                sphere_pos,
                                sphere_pos_approx,
                                link_CC,
                                T,
                                env,
                                local_cc_result,
                                tid
                            );
                            if (tid == 0) {
                                concon_parent_idx = resampled_parent;
                            }
                            __syncthreads();
                        }
                    }

                    // grow tree
                    if (tid == 0) {
                        if constexpr (AORRTC) {
                            index = patacon_reserve_slot(
                                &atomic_free_index[t_tree_id],
                                d_settings.max_samples
                            );
                        }
                        else {
                            index =atomicAdd((int *)&atomic_free_index[t_tree_id],1);
                        }

                        if (index < 0 || index >= d_settings.max_samples) {
                            if constexpr (AORRTC) {
                                atomicExch((int *)&aorrtc_stop_requested, 1);
                            }
                            else {
                                atomicCAS((int *)&solved,0,-1);
                            }

                            index = -1;
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
                    }
                    __syncthreads();

                    if (tid < dim) {
                        // 실제 tree node에는 projection 결과 저장
                        config[tid] =stored_edge_endpoint;
                        t_nodes[index * dim + tid] =config[tid];
                    }
                    __syncthreads();

                    if constexpr (AORRTC) {
                        if (tid == 0) {
                            t_node_costs[index] =
                                t_node_costs[concon_parent_idx]
                                + patacon_config_distance<Robot>(
                                    &t_nodes[concon_parent_idx * dim],
                                    &t_nodes[index * dim]
                                );
                        }
                        __syncthreads();
                    }

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
                                    if constexpr (AORRTC) {
                                        atomicExch((int *)&aorrtc_stop_requested, 1);
                                    }
                                    else {
                                        atomicCAS((int *)&solved,0,-1);
                                    }
                                }
                                else {
                                    // 아직 다른 block이 이 TS를 사용하면 안 됨
                                    t_ts_ready[new_ts_id] = 0;

                                    // 새 TS의 root는 방금 projection된 실제 tree node
                                    t_ts_root_node_idx[new_ts_id] =index;

                                    // projected actual q에서 Jacobian 계산
                                    // → null space basis 생성
                                    // → TSBank에 저장
                                    new_ts_basis_ok =patacon_store_tangent_basis<Robot>(&t_nodes[index * dim],t_ts_bases,new_ts_id);

                                    if (new_ts_basis_ok) {
                                        // Record the chart ancestry used by
                                        // PATACON's forward half-space rule.
                                        t_ts_parent_id[new_ts_id] =selected_ts_id;
                                        // 이 node는 기존 TS가 아니라 새 TS의 root가 된다.
                                        t_node_ts_id[index] =new_ts_id;
                                    }
                                    else {
                                        t_node_ts_id[index] =-1;
                                        if constexpr (AORRTC) {
                                            atomicExch((int *)&aorrtc_stop_requested, 1);
                                        }
                                        else {
                                            atomicCAS((int *)&solved,0,-1);
                                        }
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
                                if constexpr (AORRTC) {
                                    atomicExch(
                                        (int *)&aorrtc_stop_requested,
                                        1
                                    );
                                }
                                else {
                                    atomicCAS((int *)&solved, 0, -1);
                                }
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

                            patacon_register_node_in_ts<MAX_THREADS_PER_BLOCK>(index,assigned_ts_id,t_ts_node_count,t_ts_lane_head,t_node_next_in_ts);
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
                }
            } // edge 검사 완료

            // ConCon validation 전체가 끝난 뒤 CONNECT 여부 결정
            // A non-empty valid prefix is enough to attempt CONNECT from the
            // last EXTEND node that was actually inserted into the tree.
            if (concon_valid_count > 0) {
                // connect
                local_min_dist = FLT_MAX;
                local_near_idx = 0;
                int size = atomic_free_index[o_tree_id]; // 빈데편 tree 크기 가져오기

                // 반대편 tree node를 thread들이 나눠서 NN 검사
                for (unsigned int i = tid; i < size; i += blockDim.x) { 
                    if (o_node_ready[i] == 0) {
                        continue;
                    }
                    dist = patacon_sq_config_distance<Robot>(
                        &o_nodes[i * dim],
                        config
                    );
                    float candidate_score = dist;
                    bool candidate_allowed = true;
                    if constexpr (AORRTC) {
                        if (aorrtc_solution_found != 0) {
                            const float configuration_distance = sqrtf(dist);
                            const float current_best = aorrtc_read_best_cost();
                            const float total_lower_bound =
                                t_node_costs[index]
                                + o_node_costs[i]
                                + configuration_distance;
                            candidate_allowed =
                                total_lower_bound
                                + d_settings.cost_improvement_epsilon
                                < current_best;
                            const float remaining_cost =
                                current_best - t_node_costs[index];
                            const float cost_difference =
                                remaining_cost - o_node_costs[i];

                            candidate_score =
                                d_settings.aorrtc_config_weight
                                    * dist
                                + d_settings.aorrtc_cost_weight
                                    * (cost_difference * cost_difference);
                        }
                    }
                    if (candidate_allowed && candidate_score < local_min_dist) { // 현재 thread가 찾은 최근접 노드 갱신
                        local_min_dist = candidate_score;
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

                if constexpr (AORRTC) {
                    if (sdata[0] == FLT_MAX) {
                        return;
                    }
                }
                 
                // 반대편 tree에서 찾은 최근접 노드를 CONNECT 목표로 고정하고, 그 노드까지 몇 번 확장해야 하는지 계산하는 초기화 과정
                if (tid == 0) {
                    // CONNECT 시작 시 상대 tree의 target을 딱 한 번 결정
                    connect_target_idx = sindex[0];
                    connect_target_node =&o_nodes[connect_target_idx * dim];

                    // 새 TB-RRT CONNECT 상태 초기화
                    connect_failed = false;
                    float connect_total_distance = sqrtf(sdata[0]);
                    if constexpr (AORRTC) {
                        if (aorrtc_solution_found != 0) {
                            connect_total_distance =
                                patacon_config_distance<Robot>(
                                    config,
                                    connect_target_node
                                );
                        }
                    }
                    connect_reached =connect_total_distance<= d_settings.connect_reached_tolerance; // 현재 위치가 목표 노드에 충분히 가까운지 확인

                    n_extensions =static_cast<unsigned int>(ceilf(connect_total_distance/ d_settings.range)); // 목표 노드까지 range 간격으로 이동하려면 몇 번 확장해야 하는지 계산

                    // 계산 결과가 0이더라도 1로 보정
                    if (n_extensions < 1u) {n_extensions = 1u;}

                    local_cc_result[0] = 0;

                    connection_reached_shared =connect_reached;
                }
                __syncthreads();

                int connect_chunk_count = 0;

                while (connect_chunk_count< d_settings.max_connect_concon_chunks) { // 상대 tree의 target 향해 계속 확장
                    if constexpr (TraceTrees) {
                        if (tid == 0) {
                            if constexpr (AORRTC) {
                                should_skip = (aorrtc_stop_requested != 0);
                            }
                            else {
                                should_skip = (solved != 0);
                            }
                        }
                        __syncthreads();
                        if (should_skip) {
                            return;
                        }
                    }
                    else {
                        if (tid == 0) {
                            if constexpr (AORRTC) {
                                should_skip = (aorrtc_stop_requested != 0);
                            }
                            else {
                                should_skip = (solved != 0);
                            }
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
                        if constexpr (AORRTC) {
                            if (aorrtc_solution_found != 0) {
                                const float total_lower_bound =
                                    t_node_costs[index]
                                    + o_node_costs[connect_target_idx]
                                    + chunk_start_target_distance;
                                if (total_lower_bound
                                    + d_settings.cost_improvement_epsilon
                                    >= aorrtc_read_best_cost()) {
                                    connect_failed = true;
                                }
                            }
                        }
                    }
                    __syncthreads();

                    if (connect_failed) {
                        break;
                    }

                    if (connect_reached) {
                        break;
                    }

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

                    float connect_tangent_dist = 0.0f;

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

                    // 이번 ConCon chunk의 실제 edge validation 준비
                    if (tid == 0) {
                        concon_valid_count = 0;

                        // 첫 CONNECT edge의 parent는 EXTEND에서 마지막으로 추가된 실제 node
                        concon_parent_idx = index;
                        local_cc_result[0] = 0;
                    }

                    // config를 현재 실제 projected configuration으로 복구
                    if (tid < dim) {
                        config[tid] = nearest_node[tid];
                    }
                    __syncthreads();

                    // 이번 chunk에서 생성한 ConCon candidate들을 앞에서부터 하나씩 검증
                    for (int edge_step = 1; edge_step <= concon_count; edge_step++) {

                        if constexpr (TraceTrees) {
                            if (tid == 0) {
                                if constexpr (AORRTC) {
                                    should_skip = (aorrtc_stop_requested != 0);
                                }
                                else {
                                    should_skip = (solved != 0);
                                }
                            }
                            __syncthreads();
                            if (should_skip) {
                                return;
                            }
                        }
                        else {
                            if (tid == 0) {
                                if constexpr (AORRTC) {
                                    should_skip = (aorrtc_stop_requested != 0);
                                }
                                else {
                                    should_skip = (solved != 0);
                                }
                            }
                            __syncthreads();
                            if (should_skip) {
                                return;
                            }
                        }

                        // 이번 edge의 Tangent Space 위 nominal target 생성
                        // target까지 필요한 tangent distance보다
                        // 멀리 가지 않도록 마지막 edge 길이를 줄인다.
                        const float raw_edge_distance =static_cast<float>(edge_step)* d_settings.range;
                        const float connect_edge_distance =fminf(raw_edge_distance,connect_tangent_dist);

                        if (tid < dim) {
                            concon_probe[tid] =nearest_ts_node[tid]+connect_edge_distance* extend_dir[tid];
                            // 실제 projected 현재 node
                            //          ↓
                            // 이번 TS nominal target
                            // 사이를 granularity만큼 나눈다.
                            delta[tid] =(concon_probe[tid]- config[tid])/static_cast<float>(d_settings.granularity);
                        }
                        __syncthreads();

                        // 4. ParallelProject
                        const bool extension_projection_good = patacon_project_motion<Robot>(
                                config,
                                delta,
                                motion_segment,
                                motion_segment_next,
                                motion_projection_valid,
                                motion_projection_prog,
                                motion_projection_success,
                                tid
                            );
                        __syncthreads();

                        // 5. projected endpoint
                        float connect_projected_endpoint = 0.0f;

                        if (tid < dim) {
                            connect_projected_endpoint =motion_segment[d_settings.granularity* dim+ tid];
                        }
                        __syncthreads();

                        const float distance_before_value = patacon_shared_config_distance<Robot>(
                            config,
                            connect_target_node,
                            sdata,
                            tid
                        );

                    const float distance_after_value = patacon_shared_config_distance<Robot>(
                            &motion_segment[d_settings.granularity * dim],
                            connect_target_node,
                            sdata,
                            tid
                        );

                    if (tid == 0) {
                        connect_distance_before = distance_before_value;
                        connect_distance_after = distance_after_value;
                        connect_made_progress = extension_projection_good && connect_distance_after < connect_distance_before - d_settings.connect_progress_epsilon;
                    }
                    __syncthreads();

                    if (!extension_projection_good || !connect_made_progress) {
                        break;
                    }

                        bool extension_collision_free = false;
                        if (extension_projection_good) {
                            // 7. projected waypoint 가져오기
                            for (int i = 0; i < dim; i++) {
                                interp_cfg[i] =
                                    motion_segment[waypoint * dim + i];
                            }
                            __syncthreads();

                            // 8. projected motion에 대해 기존 4-thread/waypoint collision check
                            // 새로운 CONNECT segment 검사 시작
                            if (tid == 0) {
                                local_cc_result[0] = 0;
                            }
                            __syncthreads();


                            // link collision flag 초기화
                            for (int r = tid; r < Collision::joint_flag_stride * Collision::batch_size; r += blockDim.x
                            ) {
                                link_CC[r] = 0;
                            }
                            __syncthreads();


                            int detailed_FK = 0;

                            // approximate FK + environment CC
                            ppln::collision::fk_approx<Robot>(interp_cfg,sphere_pos_approx,T,tid);

                            __syncthreads();

                            bool config_in_collision2_approx =not ppln::collision::env_collision_check_approx<Robot>(sphere_pos_approx,link_CC,env,tid);

                            atomicOr((unsigned int *)&local_cc_result[0],config_in_collision2_approx ? 1u : 0u);

                            bool attached_object_collision_approx = not patacon_attached_object_collision_check_approx<Robot>(
                                    interp_cfg,
                                    sphere_pos_approx,
                                    env,
                                    tid,
                                    local_cc_result
                                );

                            atomicOr((unsigned int *)&local_cc_result[0],attached_object_collision_approx ? 1u : 0u);

                            __syncthreads();


                            if (tid == 0) {
                                run_detailed_env_check = local_cc_result[0] == 1;
                            }
                            __syncthreads();

                            // approximate env에서 걸렸으면 detailed env 검사
                            if (run_detailed_env_check) {
                                if (tid == 0) {
                                    local_cc_result[0] = 0;
                                }
                                __syncthreads();

                                ppln::collision::fk<Robot>(interp_cfg,sphere_pos,T,tid);

                                detailed_FK = 1;

                                __syncthreads();

                                bool config_in_collision2 = not patacon_detailed_env_collision_check<Robot>(sphere_pos,link_CC,env,tid,local_cc_result);

                                atomicOr((unsigned int *)&local_cc_result[0],config_in_collision2 ? 1u : 0u);

                                bool attached_object_collision = not patacon_attached_object_collision_check<Robot>(
                                        interp_cfg,
                                        sphere_pos,
                                        env,
                                        tid,
                                        local_cc_result
                                    );

                                atomicOr((unsigned int *)&local_cc_result[0],attached_object_collision ? 1u : 0u);

                                __syncthreads();
                            }

                            // self collision용 flag 초기화
                            for (int r = tid; r < Collision::joint_flag_stride * Collision::batch_size; r += blockDim.x) {
                                link_CC[r] = 0;
                            }
                            __syncthreads();

                            if (tid == 0) {
                                run_self_collision_check = local_cc_result[0] == 0;
                            }
                            __syncthreads();

                            // environment collision-free이면 self collision
                            if (run_self_collision_check) {
                                bool config_in_collision_approx =not ppln::collision::self_collision_check_approx<Robot>(sphere_pos_approx,link_CC,tid);

                                atomicOr((unsigned int *)&local_cc_result[0],config_in_collision_approx ? 1u : 0u);

                                __syncthreads();

                                if (tid == 0) {
                                    run_detailed_self_check =
                                        local_cc_result[0] == 1;
                                }
                                __syncthreads();

                                if (run_detailed_self_check) {
                                    if (tid == 0) {
                                        local_cc_result[0] = 0;
                                    }
                                    __syncthreads();


                                    if (detailed_FK == 0) {
                                        ppln::collision::fk<Robot>(interp_cfg,sphere_pos,T,tid);

                                        detailed_FK = 1;

                                        __syncthreads();
                                    }


                                    bool config_in_collision =not patacon_detailed_self_collision_check<Robot>(sphere_pos,link_CC,tid,local_cc_result);

                                    atomicOr((unsigned int *)&local_cc_result[0],config_in_collision ? 1u : 0u);

                                    __syncthreads();
                                }
                            }

                            extension_collision_free = (local_cc_result[0] == 0);
                        }
                        bool ext_edge_good = extension_projection_good && connect_made_progress && extension_collision_free;

                        __syncthreads();

                        if (!ext_edge_good) {
                            break;
                        }

                        if constexpr (AORRTC) {
                            if (aorrtc_bound_active && tid < dim) {
                                aorrtc_new_config[tid] =
                                    connect_projected_endpoint;
                            }
                            __syncthreads();

                            if (aorrtc_bound_active) {
                                const int resample_tree_size = atomicAdd(
                                    (int *)&atomic_free_index[t_tree_id],
                                    0
                                );
                                const int resampled_parent =
                                    aorrtc_resample_parent<Robot>(
                                    t_tree_id,
                                    nodes,
                                    t_nodes,
                                    t_node_costs,
                                    t_node_ready,
                                    t_ts_lane_head,
                                    t_node_next_in_ts,
                                    selected_ts_id,
                                    resample_tree_size,
                                    concon_parent_idx,
                                    aorrtc_new_config,
                                    num_goals,
                                    rng_states,
                                    bid,
                                    sdata,
                                    sindex,
                                    &aorrtc_resample_parent_idx,
                                    &aorrtc_resample_parent_cost,
                                    &aorrtc_resample_crand,
                                    &aorrtc_resample_done,
                                    motion_projection_valid,
                                    sphere_pos,
                                    sphere_pos_approx,
                                    link_CC,
                                    T,
                                    env,
                                    local_cc_result,
                                    tid
                                );
                                if (tid == 0) {
                                    concon_parent_idx = resampled_parent;
                                }
                                __syncthreads();
                            }
                        }

                        // CONNECT node slot 확보
                        if (tid == 0) {
                            if constexpr (AORRTC) {
                                index = patacon_reserve_slot(
                                    &atomic_free_index[t_tree_id],
                                    d_settings.max_samples
                                );
                            }
                            else {
                                index =atomicAdd((int *)&atomic_free_index[t_tree_id],1);
                            }

                            if (index < 0 || index >= d_settings.max_samples) {
                                if constexpr (AORRTC) {
                                    atomicExch((int *)&aorrtc_stop_requested, 1);
                                }
                                else {
                                    atomicCAS((int *)&solved,0,-1);
                                }

                                index = -1;
                            }
                        }
                        __syncthreads();

                        if (index < 0) {
                            return;
                        }

                        // CONNECT node metadata
                        if (tid == 0) {
                            t_parents[index] =concon_parent_idx;
                        }
                        __syncthreads();

                        // projected endpoint 저장
                        if (tid < dim) {
                            config[tid] =connect_projected_endpoint;
                            t_nodes[index * dim + tid] =config[tid];
                        }
                        __syncthreads();

                        if constexpr (AORRTC) {
                            if (tid == 0) {
                                t_node_costs[index] =
                                    t_node_costs[concon_parent_idx]
                                    + patacon_config_distance<Robot>(
                                        &t_nodes[concon_parent_idx * dim],
                                        &t_nodes[index * dim]
                                    );
                            }
                            __syncthreads();
                        }

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
                                        if constexpr (AORRTC) {
                                            atomicExch((int *)&aorrtc_stop_requested, 1);
                                        }
                                        else {
                                            atomicCAS((int *)&solved,0,-1);
                                        }
                                    }
                                    else {
                                        // 아직 다른 block이 사용하면 안 됨
                                        t_ts_ready[new_ts_id] = 0;

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
                                            if constexpr (AORRTC) {
                                                atomicExch((int *)&aorrtc_stop_requested, 1);
                                            }
                                            else {
                                                atomicCAS((int *)&solved,0,-1);
                                            }
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
                                    if constexpr (AORRTC) {
                                        atomicExch(
                                            (int *)&aorrtc_stop_requested,
                                            1
                                        );
                                    }
                                    else {
                                        atomicCAS((int *)&solved, 0, -1);
                                    }
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

                                patacon_register_node_in_ts<MAX_THREADS_PER_BLOCK>(index,assigned_ts_id,t_ts_node_count,t_ts_lane_head,t_node_next_in_ts);
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
                }
                __syncthreads();
                    
                // CONNECT 성공 시 양쪽 트리의 parent를 역추적하여 최종 경로 복원
                if (!connect_failed&&connect_reached) { // connected
                    if constexpr (AORRTC) {
                        if (tid == 0) {
                            aorrtc_try_store_solution<Robot, false>(
                                t_tree_id,
                                o_tree_id,
                                index,
                                connect_target_idx,
                                t_node_costs,
                                o_node_costs,
                                final_connection_distance,
                                iter
                            );
                        }
                    }
                    else {
                        if (tid == 0 && atomicCAS((int *)&solved, 0, 1) == 0) { // block 0번 thread만 최종 경로 복원 수행
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
                            while (t_parents[current] != current) { // 현재 노드가 현재 tree의 root가 아닐 때까지 부모를 따라감
                                parent = t_parents[current];
                                cost += patacon_config_distance<Robot>(
                                    (float *)&t_nodes[current * dim],
                                    (float *)&t_nodes[parent * dim]
                                );
                                for (int i = 0; i < dim; i++) path[t_tree_id][t_path_size * dim + i] = t_nodes[current * dim + i];
                                t_path_size++;
                                current = parent;
                                
                            }
                            if (t_tree_id == 1) reached_goal_idx = current; // 현재 tree가 goal tree라면, 역추적이 끝난 현재 current가 goal tree의 root (여러 goal을 지원하는 경우 어떤 goal root에 도달했는지 기록)
                            current = connect_target_idx; // 반대편 tree의 CONNECT 목표 노드에서 역추적 시작
                            while(o_parents[current] != current) { // 반대편 tree의 root에 도달할 때까지 부모 node 따라감
                                parent = o_parents[current];
                                cost += patacon_config_distance<Robot>(
                                    (float *)&t_nodes[current * dim],
                                    (float *)&t_nodes[parent * dim]
                                );
                                for (int i = 0; i < dim; i++) path[o_tree_id][o_path_size * dim + i] = o_nodes[current * dim + i];
                                o_path_size++;
                                current = parent;
                            }
                            if (t_tree_id == 0) reached_goal_idx = current;
                            path_size[t_tree_id] = t_path_size;
                            path_size[o_tree_id] = o_path_size;
                            solved_iters = iter;
                        }
                    }
                    __syncthreads();
                }
            }
        __syncthreads();

        if constexpr (AORRTC) {
            // Before the first solution, behave like PATACON: keep the CUDA
            // kernel alive and execute the next RRT-Connect iteration here.
            // A solution or fatal capacity/TS condition sets stop_requested.
            if (aorrtc_stop_requested != 0) {
                return;
            }
            if (!persistent_initial_search) {
                return;
            }
        }
        else if constexpr (TraceTrees) {
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





    inline std::vector<int> aorrtc_parent_chain(
        const std::vector<int> &parents,
        int connection_index
    ) {
        if (connection_index < 0
            || connection_index >= static_cast<int>(parents.size())) {
            throw std::runtime_error(
                "AORRTC connection index is outside its tree"
            );
        }

        std::vector<int> chain;
        int current = connection_index;
        for (int guard = 0; guard < static_cast<int>(parents.size()); guard++) {
            chain.push_back(current);
            const int parent = parents[current];
            if (parent == current) {
                return chain;
            }
            if (parent < 0 || parent >= static_cast<int>(parents.size())) {
                throw std::runtime_error(
                    "AORRTC parent index is outside its tree"
                );
            }
            current = parent;
        }

        throw std::runtime_error("AORRTC parent chain contains a cycle");
    }


    template <typename Robot>
    void reconstruct_aorrtc_path(
        AORRTCResult<Robot> &res,
        float *nodes[2],
        int *parents[2],
        const int current_samples[2]
    ) {
        static constexpr int dim = Robot::dimension;
        std::array<std::vector<int>, 2> host_parents;
        for (int tree = 0; tree < 2; tree++) {
            host_parents[tree].resize(current_samples[tree]);
            cudaMemcpy(
                host_parents[tree].data(),
                parents[tree],
                sizeof(int) * current_samples[tree],
                cudaMemcpyDeviceToHost
            );
        }

        const int start_connection_index = res.connection_tree_id == 0
            ? res.connection_node_idx
            : res.connection_other_node_idx;
        const int goal_connection_index = res.connection_tree_id == 1
            ? res.connection_node_idx
            : res.connection_other_node_idx;

        auto start_chain = aorrtc_parent_chain(
            host_parents[0],
            start_connection_index
        );
        auto goal_chain = aorrtc_parent_chain(
            host_parents[1],
            goal_connection_index
        );
        std::reverse(goal_chain.begin(), goal_chain.end());

        auto append_configuration = [&](int tree, int node_index) {
            typename Robot::Configuration configuration;
            cudaMemcpy(
                configuration.data(),
                &nodes[tree][node_index * dim],
                sizeof(float) * dim,
                cudaMemcpyDeviceToHost
            );
            res.path.push_back(configuration);
        };

        res.path.clear();
        res.path.reserve(goal_chain.size() + start_chain.size());
        for (int node_index : goal_chain) {
            append_configuration(1, node_index);
        }
        for (int node_index : start_chain) {
            append_configuration(0, node_index);
        }
        res.path_length = static_cast<int>(
            goal_chain.size() + start_chain.size() - 2
        );
        cudaCheckError(cudaGetLastError());
    }

    template <typename Robot>
    float aorrtc_bound_path_cost(
        const std::vector<typename Robot::Configuration> &path,
        const AORRTC_settings &settings
    ) {
        double total = 0.0;
        for (std::size_t state = 1; state < path.size(); state++) {
            double squared_distance = 0.0;
            for (int joint = 0; joint < Robot::dimension; joint++) {
                float weight = 1.0f;
                if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
                    if (joint == 0) {
                        weight = settings.lift_distance_weight;
                    }
                }
                else if constexpr (
                    std::is_same_v<Robot, robots::FfwSg2Mobility>
                ) {
                    if (joint == 3) {
                        weight = settings.lift_distance_weight;
                    }
                }
                const double difference = static_cast<double>(weight)
                    * (static_cast<double>(path[state][joint])
                        - static_cast<double>(path[state - 1][joint]));
                squared_distance += difference * difference;
            }
            total += std::sqrt(squared_distance);
        }
        return static_cast<float>(total);
    }

    template <typename Robot>
    AORRTCResult<Robot> solve(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &h_environment,
        AORRTC_settings &settings
    ) {
        const auto solve_start = std::chrono::steady_clock::now();
        static constexpr int dim = Robot::dimension;
        using Collision = robots::CollisionTraits<Robot>;

        if (!settings.aorrtc) {
            throw std::invalid_argument(
                "AORRTC::solve requires --aorrtc"
            );
        }
        if (!std::isfinite(settings.time_limit_sec)
            || settings.time_limit_sec <= 0.0) {
            throw std::invalid_argument(
                "AORRTC time_limit_sec must be finite and greater than zero"
            );
        }
        if (settings.aorrtc_config_weight <= 0.0f
            || settings.aorrtc_cost_weight <= 0.0f) {
            throw std::invalid_argument(
                "AORRTC distance weights must be positive"
            );
        }
        if (settings.cost_improvement_epsilon < 0.0f) {
            throw std::invalid_argument(
                "AORRTC cost_improvement_epsilon must not be negative"
            );
        }
        if (settings.aorrtc_max_parent_resamples <= 0) {
            throw std::invalid_argument(
                "AORRTC max parent resamples must be positive"
            );
        }
        if (settings.granularity != Collision::batch_size) {
            throw std::invalid_argument(
                "PATACON granularity must match the selected robot's collision batch size"
            );
        }
        if (goals.empty()) {
            throw std::invalid_argument("AORRTC requires at least one goal");
        }
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            if (settings.max_tangent_spaces <= 0) {
                throw std::invalid_argument(
                    "max_tangent_spaces must be positive"
                );
            }
            if (goals.size()
                > static_cast<std::size_t>(settings.max_tangent_spaces)) {
                throw std::invalid_argument(
                    "number of goals exceeds max_tangent_spaces"
                );
            }
        }

        // AORRTC starts from the exact same first-feasible-path search as the
        // default planner. Keep the total --time budget across both phases.
        const PATACON::RuntimeControlState saved_runtime_control =
            PATACON::runtime_control_state();
        const bool saved_cuda_device_reset_enabled =
            saved_runtime_control.cuda_device_reset_enabled;

        PATACON::set_runtime_control_state(PATACON::RuntimeControlState{
            false,
            false,
            settings.time_limit_sec,
            true
        });

        PlannerResult<Robot> initial_result;
        try {
            initial_result = PATACON::solve<Robot>(
                start,
                goals,
                h_environment,
                static_cast<PATACON_settings &>(settings)
            );
        }
        catch (...) {
            PATACON::set_runtime_control_state(saved_runtime_control);
            throw;
        }

        PATACON::set_runtime_control_state(saved_runtime_control);

        AORRTCResult<Robot> res;
        static_cast<PlannerResult<Robot> &>(res) = std::move(initial_result);
        const std::size_t initial_found_ns = res.kernel_ns + res.copy_ns;
        const std::size_t planning_budget_ns = static_cast<std::size_t>(
            std::llround(settings.time_limit_sec * 1.0e9)
        );

        if (!res.solved) {
            res.planning_ns = initial_found_ns;
            res.wall_ns = get_elapsed_nanoseconds(solve_start);
            if (saved_cuda_device_reset_enabled) {
                cudaDeviceReset();
            }
            return res;
        }

        res.initial_cost = res.cost;
        const float initial_bound_cost = aorrtc_bound_path_cost<Robot>(
            res.path,
            settings
        );
        res.initial_solution_ns = initial_found_ns;
        res.best_solution_ns = initial_found_ns;
        res.initial_kernel_ns = res.kernel_ns;
        res.solution_updates = 1;

        AORRTCSolutionUpdate<Robot> initial_update;
        initial_update.update_index = 0;
        initial_update.source_tree_id = res.connection_tree_id;
        initial_update.source_node_idx = res.connection_node_idx;
        initial_update.target_tree_id = res.connection_other_tree_id;
        initial_update.target_node_idx = res.connection_other_node_idx;
        initial_update.iteration = res.iters;
        initial_update.cost = res.cost;
        initial_update.found_ns = initial_found_ns;
        initial_update.path_start_to_goal.assign(
            res.path.rbegin(),
            res.path.rend()
        );
        if (settings.trace_trees) {
            initial_update.solution_trace = res.solution_trace;
        }
        res.solution_history.push_back(std::move(initial_update));

        if (initial_found_ns >= planning_budget_ns) {
            res.planning_ns = initial_found_ns;
            res.wall_ns = get_elapsed_nanoseconds(solve_start);
            if (saved_cuda_device_reset_enabled) {
                cudaDeviceReset();
            }
            return res;
        }

        const int num_goals = static_cast<int>(goals.size());
        const std::size_t config_size = dim * sizeof(float);
        const std::size_t node_count =
            static_cast<std::size_t>(settings.max_samples);

        // AORRTC owns independent device symbols in this translation unit.
        reset_device_variables();
        cudaMemcpyToSymbol(d_settings, &settings, sizeof(settings));
        const PATACON_settings common_settings = settings;
        cudaMemcpyToSymbol(
            PATACON::d_settings,
            &common_settings,
            sizeof(common_settings)
        );
        const int initial_solution_found = 1;
        const int initial_solution_updates = 1;
        cudaMemcpyToSymbol(
            aorrtc_solution_found,
            &initial_solution_found,
            sizeof(initial_solution_found)
        );
        cudaMemcpyToSymbol(
            aorrtc_solution_updates,
            &initial_solution_updates,
            sizeof(initial_solution_updates)
        );
        cudaMemcpyToSymbol(
            aorrtc_best_cost,
            &initial_bound_cost,
            sizeof(initial_bound_cost)
        );
        cudaMemcpyToSymbol(
            aorrtc_initial_cost,
            &res.initial_cost,
            sizeof(res.initial_cost)
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

        float *nodes[2] = {nullptr, nullptr};
        int *parents[2] = {nullptr, nullptr};
        float *node_costs[2] = {nullptr, nullptr};
        int *node_ready[2] = {nullptr, nullptr};
        int *node_ts_id[2] = {nullptr, nullptr};
        float *node_ts_q[2] = {nullptr, nullptr};
        int *ts_root_node_idx[2] = {nullptr, nullptr};
        int *ts_parent_id[2] = {nullptr, nullptr};
        float *ts_bases[2] = {nullptr, nullptr};
        int *ts_ready[2] = {nullptr, nullptr};
        int *ts_node_count[2] = {nullptr, nullptr};
        int *ts_lane_head[2] = {nullptr, nullptr};
        int *node_next_in_ts[2] = {nullptr, nullptr};

        float **d_nodes = nullptr;
        int **d_parents = nullptr;
        float **d_node_costs = nullptr;
        int **d_node_ready = nullptr;
        int **d_node_ts_id = nullptr;
        float **d_node_ts_q = nullptr;
        int **d_ts_root_node_idx = nullptr;
        int **d_ts_parent_id = nullptr;
        float **d_ts_bases = nullptr;
        int **d_ts_ready = nullptr;
        int **d_ts_node_count = nullptr;
        int **d_ts_lane_head = nullptr;
        int **d_node_next_in_ts = nullptr;
        int *ts_count = nullptr;

        cudaMalloc(&d_nodes, 2 * sizeof(float *));
        cudaMalloc(&d_parents, 2 * sizeof(int *));
        cudaMalloc(&d_node_costs, 2 * sizeof(float *));
        cudaMalloc(&d_node_ready, 2 * sizeof(int *));
        cudaMalloc(&d_node_ts_id, 2 * sizeof(int *));
        cudaMalloc(&d_node_ts_q, 2 * sizeof(float *));
        cudaMalloc(&d_ts_root_node_idx, 2 * sizeof(int *));
        cudaMalloc(&d_ts_parent_id, 2 * sizeof(int *));
        cudaMalloc(&d_ts_bases, 2 * sizeof(float *));
        cudaMalloc(&d_ts_ready, 2 * sizeof(int *));
        cudaMalloc(&d_ts_node_count, 2 * sizeof(int *));
        cudaMalloc(&d_ts_lane_head, 2 * sizeof(int *));
        cudaMalloc(&d_node_next_in_ts, 2 * sizeof(int *));
        cudaMalloc(&ts_count, 2 * sizeof(int));

        for (int tree = 0; tree < 2; tree++) {
            cudaMalloc(&nodes[tree], node_count * config_size);
            cudaMalloc(&parents[tree], node_count * sizeof(int));
            cudaMalloc(&node_costs[tree], node_count * sizeof(float));
            cudaMalloc(&node_ready[tree], node_count * sizeof(int));
            cudaMalloc(&node_ts_id[tree], node_count * sizeof(int));
            cudaMalloc(&node_ts_q[tree], node_count * config_size);
            cudaMalloc(
                &ts_root_node_idx[tree],
                static_cast<std::size_t>(settings.max_tangent_spaces)
                    * sizeof(int)
            );
            cudaMalloc(
                &ts_ready[tree],
                static_cast<std::size_t>(settings.max_tangent_spaces)
                    * sizeof(int)
            );

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                const std::size_t basis_bytes =
                    static_cast<std::size_t>(settings.max_tangent_spaces)
                    * TangentSpaceTraits<Robot>::basis_size
                    * sizeof(float);
                const std::size_t ts_count_bytes =
                    static_cast<std::size_t>(settings.max_tangent_spaces)
                    * sizeof(int);
                const std::size_t lane_bytes =
                    static_cast<std::size_t>(settings.max_tangent_spaces)
                    * MAX_THREADS_PER_BLOCK
                    * sizeof(int);

                cudaMalloc(&ts_bases[tree], basis_bytes);
                cudaMalloc(&ts_parent_id[tree], ts_count_bytes);
                cudaMemset(ts_parent_id[tree], 0xff, ts_count_bytes);
                cudaMalloc(&ts_node_count[tree], ts_count_bytes);
                cudaMalloc(&ts_lane_head[tree], lane_bytes);
                cudaMalloc(
                    &node_next_in_ts[tree],
                    node_count * sizeof(int)
                );
            }
        }

        cudaMemcpy(d_nodes, nodes, 2 * sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(d_parents, parents, 2 * sizeof(int *), cudaMemcpyHostToDevice);
        cudaMemcpy(
            d_node_costs,
            node_costs,
            2 * sizeof(float *),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            d_node_ready,
            node_ready,
            2 * sizeof(int *),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            d_node_ts_id,
            node_ts_id,
            2 * sizeof(int *),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            d_node_ts_q,
            node_ts_q,
            2 * sizeof(float *),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            d_ts_root_node_idx,
            ts_root_node_idx,
            2 * sizeof(int *),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            d_ts_parent_id,
            ts_parent_id,
            2 * sizeof(int *),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            d_ts_bases,
            ts_bases,
            2 * sizeof(float *),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            d_ts_ready,
            ts_ready,
            2 * sizeof(int *),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            d_ts_node_count,
            ts_node_count,
            2 * sizeof(int *),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            d_ts_lane_head,
            ts_lane_head,
            2 * sizeof(int *),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            d_node_next_in_ts,
            node_next_in_ts,
            2 * sizeof(int *),
            cudaMemcpyHostToDevice
        );

        // One-time sentinel initialization. Fresh-tree restarts use ready flags
        // as the visibility barrier, so the large configuration arrays do not
        // need to be cleared on every restart.
        std::vector<float> nodes_init(node_count * dim, UNWRITTEN_VAL);
        std::vector<float> cost_init(node_count, FLT_MAX);
        for (int tree = 0; tree < 2; tree++) {
            cudaMemcpy(
                nodes[tree],
                nodes_init.data(),
                node_count * config_size,
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                node_ts_q[tree],
                nodes_init.data(),
                node_count * config_size,
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                node_costs[tree],
                cost_init.data(),
                node_count * sizeof(float),
                cudaMemcpyHostToDevice
            );
        }

        curandState *rng_states = nullptr;
        const int num_rng_states = settings.num_new_configs * dim;
        cudaMalloc(
            &rng_states,
            static_cast<std::size_t>(num_rng_states)
                * sizeof(curandState)
        );
        const int rng_blocks =
            (num_rng_states + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_rng<<<rng_blocks, BLOCK_SIZE>>>(
            rng_states,
            settings.random_seed,
            num_rng_states
        );

        HaltonState<Robot> *halton_states = nullptr;
        cudaMalloc(
            &halton_states,
            static_cast<std::size_t>(settings.num_new_configs)
                * sizeof(HaltonState<Robot>)
        );
        const int halton_blocks =
            (settings.num_new_configs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_halton<Robot><<<halton_blocks, BLOCK_SIZE>>>(
            halton_states,
            rng_states
        );

        // AORRTC launches one planner iteration per kernel launch so the host
        // can enforce --time. Preserve each block's previous tree choice here
        // to keep the legacy balance=0/1/2 semantics across launches.
        int *block_tree_ids = nullptr;
        cudaMalloc(
            &block_tree_ids,
            static_cast<std::size_t>(settings.num_new_configs) * sizeof(int)
        );
        cudaMemset(
            block_tree_ids,
            0xff,
            static_cast<std::size_t>(settings.num_new_configs) * sizeof(int)
        );

        cudaDeviceSynchronize();
        cudaCheckError(cudaGetLastError());

        ppln::collision::Environment<float> *env = nullptr;
        setup_environment_on_device(env, h_environment);
        cudaCheckError(cudaGetLastError());

        const int start_parent = 0;
        std::vector<int> goal_parents(num_goals);
        std::iota(goal_parents.begin(), goal_parents.end(), 0);
        std::vector<int> goal_ready(num_goals, 1);
        std::vector<float> goal_costs(num_goals, 0.0f);
        const float start_cost = 0.0f;
        const int start_ready = 1;

        auto initialize_fresh_search = [&]() {
            const auto copy_start = std::chrono::steady_clock::now();

            cudaMemset(
                block_tree_ids,
                0xff,
                static_cast<std::size_t>(settings.num_new_configs)
                    * sizeof(int)
            );

            for (int tree = 0; tree < 2; tree++) {
                cudaMemset(
                    node_ready[tree],
                    0,
                    node_count * sizeof(int)
                );
                cudaMemset(
                    node_ts_id[tree],
                    0xff,
                    node_count * sizeof(int)
                );
                cudaMemset(
                    ts_ready[tree],
                    0,
                    static_cast<std::size_t>(settings.max_tangent_spaces)
                        * sizeof(int)
                );

                if constexpr (TangentSpaceTraits<Robot>::enabled) {
                    cudaMemset(
                        ts_node_count[tree],
                        0,
                        static_cast<std::size_t>(settings.max_tangent_spaces)
                            * sizeof(int)
                    );
                    cudaMemset(
                        ts_lane_head[tree],
                        0xff,
                        static_cast<std::size_t>(settings.max_tangent_spaces)
                            * MAX_THREADS_PER_BLOCK
                            * sizeof(int)
                    );
                    cudaMemset(
                        node_next_in_ts[tree],
                        0xff,
                        node_count * sizeof(int)
                    );
                }
            }

            const int free_index[2] = {1, num_goals};
            cudaMemcpyToSymbol(
                atomic_free_index,
                free_index,
                sizeof(free_index)
            );
            cudaMemcpyToSymbol(nodes_size, free_index, sizeof(free_index));
            cudaMemcpyToSymbol(
                completed_nodes,
                free_index,
                sizeof(free_index)
            );

            int search_solution = 0;
            int stop_requested = 0;
            cudaMemcpyToSymbol(
                aorrtc_search_solution_found,
                &search_solution,
                sizeof(int)
            );
            cudaMemcpyToSymbol(
                aorrtc_stop_requested,
                &stop_requested,
                sizeof(int)
            );

            cudaMemcpy(
                nodes[0],
                start.data(),
                config_size,
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                parents[0],
                &start_parent,
                sizeof(int),
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                node_costs[0],
                &start_cost,
                sizeof(float),
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                node_ready[0],
                &start_ready,
                sizeof(int),
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                nodes[1],
                goals.data(),
                static_cast<std::size_t>(num_goals) * config_size,
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                parents[1],
                goal_parents.data(),
                static_cast<std::size_t>(num_goals) * sizeof(int),
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                node_costs[1],
                goal_costs.data(),
                static_cast<std::size_t>(num_goals) * sizeof(float),
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                node_ready[1],
                goal_ready.data(),
                static_cast<std::size_t>(num_goals) * sizeof(int),
                cudaMemcpyHostToDevice
            );
            int zero_ts_count[2] = {0, 0};
            cudaMemcpy(
                ts_count,
                zero_ts_count,
                sizeof(zero_ts_count),
                cudaMemcpyHostToDevice
            );

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                init_root_ts_banks<Robot, MAX_THREADS_PER_BLOCK>
                    <<<1 + num_goals, 1>>>(
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

                const int initial_ts_count[2] = {1, num_goals};
                cudaMemcpy(
                    ts_count,
                    initial_ts_count,
                    sizeof(initial_ts_count),
                    cudaMemcpyHostToDevice
                );
            }

            res.copy_ns += get_elapsed_nanoseconds(copy_start);
            cudaCheckError(cudaGetLastError());
        };

        const auto optimization_start = std::chrono::steady_clock::now();
        const auto planning_deadline = optimization_start
            + std::chrono::nanoseconds(
                planning_budget_ns - initial_found_ns
            );

        initialize_fresh_search();
        res.search_restarts = 1;

        int total_rounds = 0;
        int search_round = 0;

        constexpr bool persistent_initial_search = false;

        while (total_rounds < settings.max_iters
               && std::chrono::steady_clock::now() < planning_deadline) {
            total_rounds++;
            search_round++;

            const auto kernel_start = std::chrono::steady_clock::now();
            if (settings.trace_trees) {
                patacon<Robot, true, true>
                    <<<settings.num_new_configs,
                       4 * settings.granularity>>>(
                        d_nodes,
                        d_parents,
                        d_node_costs,
                        d_node_ready,
                        d_node_ts_id,
                        d_node_ts_q,
                        ts_count,
                        d_ts_root_node_idx,
                        d_ts_parent_id,
                        d_ts_bases,
                        d_ts_ready,
                        d_ts_node_count,
                        d_ts_lane_head,
                        d_node_next_in_ts,
                        halton_states,
                        rng_states,
                        block_tree_ids,
                        env,
                        num_goals,
                        search_round,
                        persistent_initial_search
                    );
            }
            else {
                patacon<Robot, false, true>
                    <<<settings.num_new_configs,
                       4 * settings.granularity>>>(
                        d_nodes,
                        d_parents,
                        d_node_costs,
                        d_node_ready,
                        d_node_ts_id,
                        d_node_ts_q,
                        ts_count,
                        d_ts_root_node_idx,
                        d_ts_parent_id,
                        d_ts_bases,
                        d_ts_ready,
                        d_ts_node_count,
                        d_ts_lane_head,
                        d_node_next_in_ts,
                        halton_states,
                        rng_states,
                        block_tree_ids,
                        env,
                        num_goals,
                        search_round,
                        persistent_initial_search
                    );
            }
            cudaDeviceSynchronize();
            res.kernel_ns += get_elapsed_nanoseconds(kernel_start);
            cudaCheckError(cudaGetLastError());

            const auto state_copy_start = std::chrono::steady_clock::now();
            int search_solution = 0;
            int stop_requested = 0;
            cudaMemcpyFromSymbol(
                &search_solution,
                aorrtc_search_solution_found,
                sizeof(int),
                0,
                cudaMemcpyDeviceToHost
            );
            cudaMemcpyFromSymbol(
                &stop_requested,
                aorrtc_stop_requested,
                sizeof(int),
                0,
                cudaMemcpyDeviceToHost
            );
            res.copy_ns += get_elapsed_nanoseconds(state_copy_start);

            if (search_solution != 0) {
                int current_samples[2] = {0, 0};
                float best_cost = FLT_MAX;
                int source_tree = -1;
                int source_node = -1;
                int target_tree = -1;
                int target_node = -1;

                const auto snapshot_start = std::chrono::steady_clock::now();
                cudaMemcpyFromSymbol(
                    current_samples,
                    atomic_free_index,
                    sizeof(current_samples),
                    0,
                    cudaMemcpyDeviceToHost
                );
                for (int tree = 0; tree < 2; tree++) {
                    current_samples[tree] = std::clamp(
                        current_samples[tree],
                        0,
                        settings.max_samples
                    );
                }
                cudaMemcpyFromSymbol(
                    &best_cost,
                    aorrtc_best_cost,
                    sizeof(float),
                    0,
                    cudaMemcpyDeviceToHost
                );
                cudaMemcpyFromSymbol(
                    &source_tree,
                    connection_tree_id,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );
                cudaMemcpyFromSymbol(
                    &source_node,
                    connection_node_idx,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );
                cudaMemcpyFromSymbol(
                    &target_tree,
                    connection_other_tree_id,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );
                cudaMemcpyFromSymbol(
                    &target_node,
                    connection_other_node_idx,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );

                AORRTCResult<Robot> snapshot;
                snapshot.solved = true;
                snapshot.start_tree_size = current_samples[0];
                snapshot.goal_tree_size = current_samples[1];
                snapshot.connection_tree_id = source_tree;
                snapshot.connection_node_idx = source_node;
                snapshot.connection_other_tree_id = target_tree;
                snapshot.connection_other_node_idx = target_node;
                reconstruct_aorrtc_path(
                    snapshot,
                    nodes,
                    parents,
                    current_samples
                );

                const float evaluated_cost =
                    configuration_space_path_arclength<Robot>(
                        snapshot.path
                    );

                snapshot.cost = evaluated_cost;

                if (settings.trace_trees) {
                    PATACON::copy_tree_trace_to_result<Robot>(
                        static_cast<PlannerResult<Robot> &>(snapshot),
                        nodes,
                        parents,
                        node_ready,
                        current_samples,
                        1
                    );
                    PATACON::fill_solution_trace<Robot>(
                        static_cast<PlannerResult<Robot> &>(snapshot)
                    );
                }

                const std::size_t found_ns =
                    initial_found_ns
                    + get_elapsed_nanoseconds(optimization_start);

                res.solved = true;
                res.cost = evaluated_cost;
                res.path = snapshot.path;
                res.path_length = snapshot.path_length;
                res.start_tree_size = snapshot.start_tree_size;
                res.goal_tree_size = snapshot.goal_tree_size;
                res.connection_tree_id = snapshot.connection_tree_id;
                res.connection_node_idx = snapshot.connection_node_idx;
                res.connection_other_tree_id =
                    snapshot.connection_other_tree_id;
                res.connection_other_node_idx =
                    snapshot.connection_other_node_idx;
                if (settings.trace_trees) {
                    res.tree_nodes = std::move(snapshot.tree_nodes);
                    res.tree_parents = std::move(snapshot.tree_parents);
                    res.tree_node_ready =
                        std::move(snapshot.tree_node_ready);
                    res.solution_trace =
                        std::move(snapshot.solution_trace);
                }

                res.best_solution_ns = found_ns;

                AORRTCSolutionUpdate<Robot> update;
                update.update_index = res.solution_updates;
                update.source_tree_id = source_tree;
                update.source_node_idx = source_node;
                update.target_tree_id = target_tree;
                update.target_node_idx = target_node;
                update.iteration = total_rounds;
                update.cost = evaluated_cost;
                update.found_ns = found_ns;
                update.path_start_to_goal.assign(
                    res.path.rbegin(),
                    res.path.rend()
                );
                if (settings.trace_trees) {
                    update.solution_trace = res.solution_trace;
                }
                constexpr std::size_t kMaxSolutionHistory = 1024;
                if (res.solution_history.size() < kMaxSolutionHistory) {
                    res.solution_history.push_back(std::move(update));
                }
                else {
                    res.solution_history_overflow = true;
                }
                res.solution_updates++;
                res.copy_ns += get_elapsed_nanoseconds(snapshot_start);

                // Algorithm 1 Lines 5-8: the newly found cost is already stored
                // in aorrtc_best_cost.  Discard both trees and start a fresh
                // bounded search.  Device allocations are reused, but no tree
                // node or TS membership is reused.
                if (std::chrono::steady_clock::now() < planning_deadline) {
                    initialize_fresh_search();
                    search_round = 0;
                    res.search_restarts++;
                }
                continue;
            }

            // Capacity exhaustion is not an AORRTC termination condition.  A
            // fresh tree with the same c_max is started while time remains.
            if (stop_requested != 0
                && std::chrono::steady_clock::now() < planning_deadline) {
                initialize_fresh_search();
                search_round = 0;
                res.search_restarts++;
            }
        }

        res.iters += total_rounds;

        // planning과 wall 모두 cleanup 직전에 종료
        res.planning_ns = initial_found_ns
            + get_elapsed_nanoseconds(optimization_start);

        cleanup_environment_on_device(env, h_environment);
        reset_device_variables();

        for (int tree = 0; tree < 2; tree++) {
            cudaFree(nodes[tree]);
            cudaFree(parents[tree]);
            cudaFree(node_costs[tree]);
            cudaFree(node_ready[tree]);
            cudaFree(node_ts_id[tree]);
            cudaFree(node_ts_q[tree]);
            cudaFree(ts_root_node_idx[tree]);
            if (ts_parent_id[tree] != nullptr) {
                cudaFree(ts_parent_id[tree]);
            }
            cudaFree(ts_ready[tree]);
            if (ts_bases[tree] != nullptr) {
                cudaFree(ts_bases[tree]);
            }
            if (ts_node_count[tree] != nullptr) {
                cudaFree(ts_node_count[tree]);
            }
            if (ts_lane_head[tree] != nullptr) {
                cudaFree(ts_lane_head[tree]);
            }
            if (node_next_in_ts[tree] != nullptr) {
                cudaFree(node_next_in_ts[tree]);
            }
        }

        cudaFree(rng_states);
        cudaFree(halton_states);
        cudaFree(block_tree_ids);
        cudaFree(d_nodes);
        cudaFree(d_parents);
        cudaFree(d_node_costs);
        cudaFree(d_node_ready);
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
        cudaCheckError(cudaGetLastError());

        res.wall_ns = get_elapsed_nanoseconds(solve_start);
        if (saved_cuda_device_reset_enabled) {
            cudaDeviceReset();
        }
        return res;
    }

    template AORRTCResult<ppln::robots::FrankaSingle> solve<ppln::robots::FrankaSingle>(std::array<float, 7>&, std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, AORRTC_settings&);
    template AORRTCResult<ppln::robots::Franka> solve<ppln::robots::Franka>(std::array<float, 14>&, std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, AORRTC_settings&);
    template AORRTCResult<ppln::robots::FfwSg2> solve<ppln::robots::FfwSg2>(std::array<float, 15>&, std::vector<std::array<float, 15>>&, ppln::collision::Environment<float>&, AORRTC_settings&);
    template AORRTCResult<ppln::robots::FfwSg2Mobility> solve<ppln::robots::FfwSg2Mobility>(std::array<float, 18>&, std::vector<std::array<float, 18>>&, ppln::collision::Environment<float>&, AORRTC_settings&);
    template AORRTCResult<ppln::robots::G1> solve<ppln::robots::G1>(std::array<float, 35>&, std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, AORRTC_settings&);
    template AORRTCResult<ppln::robots::IgrisC> solve<ppln::robots::IgrisC>(std::array<float, 35>&, std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, AORRTC_settings&);

}

namespace PATACON {
    template <typename Robot>
    AORRTCResult<Robot> solve(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &environment,
        AORRTC_settings &settings
    ) {
        if (settings.aorrtc) {
            return AORRTC::solve<Robot>(
                start,
                goals,
                environment,
                settings
            );
        }

        AORRTCResult<Robot> result;
        static_cast<PlannerResult<Robot> &>(result) = PATACON::solve<Robot>(
            start,
            goals,
            environment,
            static_cast<PATACON_settings &>(settings)
        );
        return result;
    }

    template AORRTCResult<ppln::robots::FrankaSingle> solve<ppln::robots::FrankaSingle>(std::array<float, 7>&, std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, AORRTC_settings&);
    template AORRTCResult<ppln::robots::Franka> solve<ppln::robots::Franka>(std::array<float, 14>&, std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, AORRTC_settings&);
    template AORRTCResult<ppln::robots::FfwSg2> solve<ppln::robots::FfwSg2>(std::array<float, 15>&, std::vector<std::array<float, 15>>&, ppln::collision::Environment<float>&, AORRTC_settings&);
    template AORRTCResult<ppln::robots::FfwSg2Mobility> solve<ppln::robots::FfwSg2Mobility>(std::array<float, 18>&, std::vector<std::array<float, 18>>&, ppln::collision::Environment<float>&, AORRTC_settings&);
    template AORRTCResult<ppln::robots::G1> solve<ppln::robots::G1>(std::array<float, 35>&, std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, AORRTC_settings&);
    template AORRTCResult<ppln::robots::IgrisC> solve<ppln::robots::IgrisC>(std::array<float, 35>&, std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, AORRTC_settings&);
}

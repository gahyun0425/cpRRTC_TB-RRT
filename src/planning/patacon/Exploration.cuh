// Exploration setup used by Algorithm 1 before calling EXTEND.
// Included by PATACON.cu inside namespace PATACON.

    template <typename Robot>
    __device__ __forceinline__ void
    PataconBlockState<Robot>::sample_q_rand(
        const PataconSearchContext<Robot> &search,
        int bid,
        int tid
    ) {
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            patacon_sample_tangent_config<Robot>(
                t_nodes,
                t_ts_bases,
                t_ts_root_node_idx,
                t_ts_parent_id,
                selected_ts_root_idx,
                selected_ts_id,
                ts_coeff,
                ts_alpha_fraction,
                ts_tangent_dir,
                sdata,
                (float *)config,
                tid
            );
        }
        else {
            if (tid == 0) {
                halton_next(
                    search.halton_states[bid],
                    (float *)config
                );
                Robot::scale_cfg((float *)config);
            }
            __syncthreads();
        }
    }

    template <typename Robot>
    __device__ __forceinline__ bool
    PataconBlockState<Robot>::select_q_near(
        const PataconSearchContext<Robot> &search,
        int tid
    ) {
        float local_min_dist = FLT_MAX;
        int local_near_idx = 0;

        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            int node_idx =
                t_ts_lane_head[
                    selected_ts_id * MAX_THREADS_PER_BLOCK + tid
                ];

            while (node_idx >= 0) {
                if (
                    t_node_ready[node_idx] ==
                    current_search_generation
                ) {
                    const float candidate_dist =
                        patacon_sq_config_distance<Robot>(
                            (float *)&t_node_ts_q[node_idx * dim],
                            (float *)config
                        );
                    if (candidate_dist < local_min_dist) {
                        local_min_dist = candidate_dist;
                        local_near_idx = node_idx;
                    }
                }

                node_idx = t_node_next_in_ts[node_idx];
            }
        }
        else {
            for (
                int node_idx = tid;
                node_idx < t_tree_size;
                node_idx += blockDim.x
            ) {
                if (
                    t_node_ready[node_idx] !=
                    current_search_generation
                ) {
                    continue;
                }

                const float candidate_dist =
                    patacon_sq_config_distance<Robot>(
                        (float *)&t_nodes[node_idx * dim],
                        (float *)config
                    );
                if (candidate_dist < local_min_dist) {
                    local_min_dist = candidate_dist;
                    local_near_idx = node_idx;
                }
            }
        }

        sdata[tid] = local_min_dist;
        sindex[tid] = local_near_idx;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (
                tid < stride &&
                sdata[tid + stride] < sdata[tid]
            ) {
                sdata[tid] = sdata[tid + stride];
                sindex[tid] = sindex[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const bool no_nn_candidate = sdata[0] == FLT_MAX;

            if (no_nn_candidate) {
                q_rand_dist = 0.0f;
                should_skip = true;
            }
            else {
                q_rand_dist = sqrtf(sdata[0]);
                nearest_node = &t_nodes[sindex[0] * dim];

                if constexpr (TangentSpaceTraits<Robot>::enabled) {
                    nearest_ts_node =
                        &t_node_ts_q[sindex[0] * dim];
                }

                const bool zero_direction =
                    q_rand_dist <= 1.0e-8f;

                if constexpr (!TangentSpaceTraits<Robot>::enabled) {
                    scale = zero_direction
                        ? 0.0f
                        : min(1.0f, d_settings.range / q_rand_dist);
                }

                should_skip = zero_direction;
            }
        }
        __syncthreads();

        if (should_skip) {
            return false;
        }
        __syncthreads();
        return true;
    }

    template <typename Robot>
    __device__ __forceinline__ void
    PataconBlockState<Robot>::compute_v_ext(int tid) {
        if (tid == 0) {
            diagnostic_increment(DIAG_EXTEND_ATTEMPTS);
        }

        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            if (tid < dim) {
                extend_dir[tid] =
                    (config[tid] - nearest_ts_node[tid]) /
                    q_rand_dist;
            }
        }
    }

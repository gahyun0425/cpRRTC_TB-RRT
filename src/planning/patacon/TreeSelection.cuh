// Existing tree balancing and tangent-space selection, extracted verbatim
// from the original PATACON kernel.

    template <typename Robot>
    __device__ __forceinline__ void
    PataconBlockState<Robot>::check_iteration_limit(
        int iteration,
        unsigned long long block_start_time_ns
    ) {
        const bool time_limit_reached =
            patacon_time_limit_ns > 0 &&
            global_timer_ns() - block_start_time_ns >= patacon_time_limit_ns;
        if (iteration > d_settings.max_iters || time_limit_reached) {
            atomicCAS((int *)&solved, 0, -1);
        }
    }

    template <typename Robot>
    __device__ __forceinline__ void PataconBlockState<Robot>::select_tree(
        const PataconSearchContext<Robot> &search,
        int block_id,
        int iteration
    ) {
        // tree 선택. 더 작은 tree 선택
        if (d_settings.balance == 0 || iteration == 1) {
            t_tree_id =
                (block_id < (d_settings.num_new_configs / 2)) ? 0 : 1;
            o_tree_id = 1 - t_tree_id;
        }
        else if (
            d_settings.balance == 1 &&
            abs(atomic_free_index[0] - atomic_free_index[1]) <
                1.5 * d_settings.num_new_configs
        ) {
            float ratio = atomic_free_index[0] /
                (float)(atomic_free_index[0] + atomic_free_index[1]);
            float balance_factor = 1 - ratio;
            t_tree_id =
                (block_id < (d_settings.num_new_configs * balance_factor))
                    ? 0
                    : 1;
            o_tree_id = 1 - t_tree_id;
        }
        else if (d_settings.balance == 1) {
            float ratio = atomic_free_index[0] /
                (float)(atomic_free_index[0] + atomic_free_index[1]);
            if (ratio < d_settings.tree_ratio) {
                t_tree_id = 0;
            }
            else {
                t_tree_id = 1;
            }
            o_tree_id = 1 - t_tree_id;
        }
        else if (d_settings.balance == 2) {
            float ratio = abs(
                atomic_free_index[t_tree_id] -
                atomic_free_index[o_tree_id]
            ) / (float)atomic_free_index[t_tree_id];
            if (ratio < d_settings.tree_ratio) {
                t_tree_id = 1 - t_tree_id;
                o_tree_id = 1 - t_tree_id;
            }
        }

        t_nodes = search.nodes[t_tree_id];
        o_nodes = search.nodes[o_tree_id];
        t_parents = search.parents[t_tree_id];
        o_parents = search.parents[o_tree_id];
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            t_node_ts_id = search.node_ts_id[t_tree_id];
            t_node_ts_q = search.node_ts_q[t_tree_id];
            t_ts_root_node_idx = search.ts_root_node_idx[t_tree_id];
            t_ts_parent_id = search.ts_parent_id[t_tree_id];
            t_ts_bases = search.ts_bases[t_tree_id];
            t_ts_ready = search.ts_ready[t_tree_id];
            t_ts_node_count = search.ts_node_count[t_tree_id];
            t_ts_lane_head = search.ts_lane_head[t_tree_id];
            t_node_next_in_ts = search.node_next_in_ts[t_tree_id];
            t_ts_count = search.ts_count[t_tree_id];
        }
        t_node_ready = search.node_ready[t_tree_id];
        o_node_ready = search.node_ready[o_tree_id];
        t_tree_size = min(
            (int)atomic_free_index[t_tree_id],
            d_settings.max_samples
        );
    }

    template <typename Robot>
    __device__ __forceinline__ void
    PataconBlockState<Robot>::select_tangent_space(
        const PataconSearchContext<Robot> &search,
        int block_id,
        int iteration
    ) {
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            float halton_sample[dim];
            halton_next(search.halton_states[block_id], halton_sample);

            selected_ts_id = -1;
            selected_ts_root_idx = -1;

            if (t_ts_count > 0) {
                const int start_ts =
                    (block_id + iteration * d_settings.num_new_configs) %
                    t_ts_count;

                for (int attempt = 0; attempt < t_ts_count; attempt++) {
                    const int candidate_ts =
                        (start_ts + attempt) % t_ts_count;

                    if (
                        t_ts_ready[candidate_ts] ==
                        current_search_generation
                    ) {
                        selected_ts_id = candidate_ts;
                        selected_ts_root_idx =
                            t_ts_root_node_idx[candidate_ts];
                        break;
                    }
                }
            }

            const int active_tangent_dim =
                patacon_active_tangent_dim<Robot>();
            float coeff_norm2 = 0.0f;

            for (int k = 0; k < active_tangent_dim; k++) {
                const float coeff = 2.0f * halton_sample[k] - 1.0f;
                ts_coeff[k] = coeff;
                coeff_norm2 += coeff * coeff;
            }

            const float inv_coeff_norm =
                1.0f / fmaxf(sqrtf(coeff_norm2), 1.0e-8f);

            for (int k = 0; k < active_tangent_dim; k++) {
                ts_coeff[k] *= inv_coeff_norm;
            }

            ts_alpha_fraction = active_tangent_dim < dim
                ? halton_sample[active_tangent_dim]
                : curand_uniform(&search.rng_states[block_id]);
        }
    }

    template <typename Robot>
    template <bool TraceTrees>
    __device__ __forceinline__ bool
    PataconBlockState<Robot>::should_terminate() {
        if constexpr (TraceTrees) {
            if (threadIdx.x == 0) {
                should_skip = (solved != 0);
            }
            __syncthreads();
            if (should_skip) {
                return true;
            }
        }
        else {
            if (threadIdx.x == 0) {
                should_skip = (solved != 0);
            }
            __syncthreads();
            if (should_skip) {
                return true;
            }
        }
        return false;
    }

    template <typename Robot>
    __device__ __forceinline__ bool
    PataconBlockState<Robot>::has_selected_tangent_space() const {
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            return selected_ts_id >= 0;
        }
        return true;
    }

// PATACON CONNECT flow from Algorithm 1:
// choose a fixed target, repeatedly EXTEND toward it, then check connection.

    template <typename Robot>
    __device__ __forceinline__ void
    PataconBlockState<Robot>::select_connect_target(int tid) {
        float local_min_dist = FLT_MAX;
        int local_near_idx = 0;
        const int size = min(
            (int)atomic_free_index[o_tree_id],
            d_settings.max_samples
        );

        // Select the closest node in the opposite tree once. It remains the
        // fixed target for every EXTEND call in this CONNECT attempt.
        for (unsigned int i = tid; i < size; i += blockDim.x) {
            if (o_node_ready[i] != current_search_generation) {
                continue;
            }

            const float candidate_dist =
                patacon_sq_config_distance<Robot>(
                    &o_nodes[i * dim],
                    config
                );
            if (candidate_dist < local_min_dist) {
                local_min_dist = candidate_dist;
                local_near_idx = i;
            }
        }

        sdata[tid] = local_min_dist;
        sindex[tid] = local_near_idx;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sindex[tid] = sindex[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            connect_target_idx = sindex[0];
            connect_target_node = &o_nodes[connect_target_idx * dim];
            connect_failed = false;
            diagnostic_increment(DIAG_CONNECT_ATTEMPTS);
        }
        __syncthreads();
    }

    template <typename Robot>
    template <bool TraceTrees>
    __device__ __forceinline__ PataconConnectResult
    PataconBlockState<Robot>::connect_trees(
        const PataconSearchContext<Robot> &search,
        int bid,
        int tid
    ) {
        for (
            int attempt = 0;
            attempt < d_settings.max_connect_concon_chunks;
            attempt++
        ) {
            const PataconExtendResult extension =
                extend<PataconExtendMode::ConnectTarget, TraceTrees>(
                    search,
                    bid,
                    tid
                );

            if (extension.status == PataconExtendStatus::Terminate) {
                return {PataconConnectStatus::Terminate, 0.0f};
            }
            if (extension.status == PataconExtendStatus::Trapped) {
                break;
            }

            const PataconConnectResult connection =
                check_connection(search, bid, tid);
            if (connection.connected()) {
                return connection;
            }
        }

        const float final_connection_distance =
            patacon_shared_config_distance<Robot>(
                config,
                connect_target_node,
                sdata,
                tid
            );

        if (tid == 0) {
            diagnostic_increment(DIAG_CONNECT_FAILURES);
        }
        __syncthreads();

        return {
            PataconConnectStatus::NotConnected,
            final_connection_distance
        };
    }

// Device-side solution claiming and parent-chain extraction.

    struct PataconDevicePathTrace {
        int root_index = -1;
        int node_count = 0;
    };

    template <typename Robot>
    __device__ __forceinline__ PataconDevicePathTrace
    trace_device_parent_chain(
        const float *tree_nodes,
        const int *tree_parents,
        int tree_id,
        int connection_index,
        float *solution_cost
    ) {
        static constexpr int dim = Robot::dimension;
        int current = connection_index;
        int path_node_count = 0;

        while (true) {
            if (
                current < 0 ||
                current >= d_settings.max_samples ||
                path_node_count >= MAX_PATH_NODES
            ) {
                break;
            }

            const int parent = tree_parents[current];
            if (parent == current) {
                break;
            }
            if (parent < 0 || parent >= d_settings.max_samples) {
                break;
            }

            *solution_cost += patacon_config_distance<Robot>(
                &tree_nodes[current * dim],
                &tree_nodes[parent * dim]
            );

            for (int joint = 0; joint < dim; joint++) {
                path[tree_id][path_node_count * dim + joint] =
                    tree_nodes[current * dim + joint];
            }

            path_node_count++;
            current = parent;
        }

        return {current, path_node_count};
    }

    template <typename Robot>
    template <bool TraceTrees>
    __device__ __forceinline__ void
    PataconBlockState<Robot>::derive_solution_path(
        const PataconSearchContext<Robot> &,
        float final_connection_distance,
        int iteration,
        int tid
    ) {
        if (tid == 0 && atomicCAS((int *)&solved, 0, 1) == 0) {
            cost = final_connection_distance;

            if constexpr (TraceTrees) {
                connection_tree_id = t_tree_id;
                connection_node_idx = index;
                connection_other_tree_id = o_tree_id;
                connection_other_node_idx = connect_target_idx;
            }

            const PataconDevicePathTrace active_tree_path =
                trace_device_parent_chain<Robot>(
                    t_nodes,
                    t_parents,
                    t_tree_id,
                    index,
                    &cost
                );

            if (t_tree_id == 1) {
                reached_goal_idx = active_tree_path.root_index;
            }

            const PataconDevicePathTrace other_tree_path =
                trace_device_parent_chain<Robot>(
                    o_nodes,
                    o_parents,
                    o_tree_id,
                    connect_target_idx,
                    &cost
                );

            if (t_tree_id == 0) {
                reached_goal_idx = other_tree_path.root_index;
            }

            path_size[t_tree_id] = active_tree_path.node_count;
            path_size[o_tree_id] = other_tree_path.node_count;
            solved_iters = iteration;
        }
        __syncthreads();
    }

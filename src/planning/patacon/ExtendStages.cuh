// Low-level stages used by the single EXTEND operation in Extend.cuh.
// The stage boundaries are structural only: CUDA synchronization and the
// original Exploration/ConnectTarget calculations remain in the same order.

    template <typename Robot>
    template <PataconExtendMode Mode, bool TraceTrees>
    __device__ __forceinline__ PataconExtendPreparation
    PataconBlockState<Robot>::prepare_extension(int tid) {
        if constexpr (Mode == PataconExtendMode::Exploration) {
            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                if (tid == 0) {
                    concon_count = 0;
                    concon_em_stop = false;
                }
            }
            __syncthreads();

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                for (
                    int step = 1;
                    step <= d_settings.max_concon_nodes;
                    step++
                ) {
                    if (tid < dim) {
                        concon_probe[tid] =
                            nearest_ts_node[tid]
                            + static_cast<float>(step)
                                * d_settings.range
                                * extend_dir[tid];
                    }
                    __syncthreads();

                    if (tid == 0) {
                        const float em_error =
                            patacon_constraint_error_norm<Robot>(
                                concon_probe
                            );
                        concon_count = step;

                        if (em_error > d_settings.em_threshold) {
                            concon_em_stop = true;
                            diagnostic_increment(DIAG_EXTEND_EM_STOPS);
                        }
                    }
                    __syncthreads();

                    if (concon_em_stop) {
                        break;
                    }
                }
            }
            __syncthreads();

            if (tid == 0) {
                concon_valid_count = 0;
                concon_parent_idx = sindex[0];

                if constexpr (TangentSpaceTraits<Robot>::enabled) {
                    extend_edge_count = concon_count;
                }
                else {
                    extend_edge_count = 1;
                }
            }

            if (tid < dim) {
                if constexpr (TangentSpaceTraits<Robot>::enabled) {
                    config[tid] = nearest_node[tid];
                }
                else {
                    concon_probe[tid] =
                        nearest_node[tid]
                        + (config[tid] - nearest_node[tid]) * scale;
                    config[tid] = nearest_node[tid];
                }
            }
            __syncthreads();

            if (tid == 0) {
                concon_projected_edge_count = 0;
            }
            __syncthreads();

            return {PataconExtendStatus::Advanced, 0.0f, 0.0f};
        }
        else {
            if constexpr (TraceTrees) {
                if (tid == 0) {
                    should_skip = (solved != 0);
                }
                __syncthreads();
                if (should_skip) {
                    return {PataconExtendStatus::Terminate, 0.0f, 0.0f};
                }
            }
            else {
                if (tid == 0) {
                    should_skip = (solved != 0);
                }
                __syncthreads();
                if (should_skip) {
                    return {PataconExtendStatus::Terminate, 0.0f, 0.0f};
                }
            }

            const float target_distance =
                patacon_shared_config_distance<Robot>(
                    config,
                    connect_target_node,
                    sdata,
                    tid
                );

            if (tid == 0) {
                diagnostic_increment(DIAG_CONNECT_CHUNKS);
            }

            float tangent_distance = 0.0f;

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                if (tid == 0) {
                    selected_ts_id = t_node_ts_id[index];

                    if (
                        selected_ts_id < 0 ||
                        t_ts_ready[selected_ts_id] !=
                            current_search_generation
                    ) {
                        connect_failed = true;
                        diagnostic_increment(
                            DIAG_CONNECT_INVALID_TANGENT_SPACES
                        );
                    }
                    else {
                        nearest_node = &t_nodes[index * dim];
                        nearest_ts_node = &t_node_ts_q[index * dim];
                    }
                }
                __syncthreads();

                const float *connect_basis = nullptr;
                if (!connect_failed) {
                    connect_basis = &t_ts_bases[
                        selected_ts_id *
                        TangentSpaceTraits<Robot>::basis_size
                    ];
                }
                __syncthreads();

                if (!connect_failed) {
                    tangent_distance =
                        patacon_project_target_direction_to_tangent<Robot>(
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
                    if (!connect_failed && tangent_distance <= 1.0e-8f) {
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
                    for (
                        int step = 1;
                        step <= d_settings.max_concon_nodes;
                        step++
                    ) {
                        const float raw_step_distance =
                            static_cast<float>(step) * d_settings.range;
                        const float connect_step_distance =
                            fminf(raw_step_distance, tangent_distance);
                        const bool connect_target_step =
                            raw_step_distance >= tangent_distance;

                        if (tid < dim) {
                            concon_probe[tid] =
                                nearest_ts_node[tid]
                                + connect_step_distance * extend_dir[tid];
                        }
                        __syncthreads();

                        if (tid == 0) {
                            const float em_error =
                                patacon_constraint_error_norm<Robot>(
                                    concon_probe
                                );
                            concon_count = step;
                            if (em_error > d_settings.em_threshold) {
                                concon_em_stop = true;
                                diagnostic_increment(
                                    DIAG_CONNECT_EM_STOPS
                                );
                            }
                        }
                        __syncthreads();

                        if (concon_em_stop || connect_target_step) {
                            break;
                        }
                    }
                }
                __syncthreads();
            }
            else {
                if (tid == 0) {
                    nearest_node = &t_nodes[index * dim];
                    concon_count = 1;
                    concon_em_stop = false;
                }
                __syncthreads();
            }

            if (tid == 0) {
                concon_valid_count = 0;
                concon_parent_idx = index;
            }

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
                    return {
                        PataconExtendStatus::Terminate,
                        target_distance,
                        tangent_distance
                    };
                }
            }
            else {
                if (tid == 0) {
                    should_skip = (solved != 0);
                }
                __syncthreads();
                if (should_skip) {
                    return {
                        PataconExtendStatus::Terminate,
                        target_distance,
                        tangent_distance
                    };
                }
            }

            return {
                PataconExtendStatus::Advanced,
                target_distance,
                tangent_distance
            };
        }
    }

    template <typename Robot>
    template <PataconExtendMode Mode>
    __device__ __forceinline__ void
    PataconBlockState<Robot>::project_extension_candidates(
        const PataconExtendPreparation &preparation,
        int tid
    ) {
        if constexpr (Mode == PataconExtendMode::Exploration) {
            for (
                int edge_step = 1;
                edge_step <= extend_edge_count;
                edge_step++
            ) {
                if (tid < dim) {
                    if constexpr (TangentSpaceTraits<Robot>::enabled) {
                        concon_probe[tid] =
                            nearest_ts_node[tid]
                            + static_cast<float>(edge_step)
                                * d_settings.range
                                * extend_dir[tid];
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
        }
        else {
            for (
                int edge_step = 1;
                edge_step <= concon_count;
                edge_step++
            ) {
                if (tid < dim) {
                    if constexpr (TangentSpaceTraits<Robot>::enabled) {
                        const float raw_edge_distance =
                            static_cast<float>(edge_step) *
                            d_settings.range;
                        const float connect_edge_distance =
                            fminf(
                                raw_edge_distance,
                                preparation.tangent_distance
                            );

                        concon_probe[tid] =
                            nearest_ts_node[tid]
                            + connect_edge_distance * extend_dir[tid];
                    }
                    else {
                        const float step_scale = fminf(
                            1.0f,
                            d_settings.range /
                                fmaxf(preparation.target_distance, 1.0e-8f)
                        );
                        concon_probe[tid] =
                            config[tid]
                            + (connect_target_node[tid] - config[tid])
                                * step_scale;
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
                    projection_good
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

                for (
                    int edge_step = 1;
                    edge_step <= valid_edges;
                    edge_step++
                ) {
                    const float distance_before =
                        patacon_config_distance_from_volatile<Robot>(
                            &motion_segment[(edge_step - 1) * dim],
                            connect_target_node
                        );
                    const float distance_after =
                        patacon_config_distance_from_volatile<Robot>(
                            &motion_segment[edge_step * dim],
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
                            &projected_edge[d_settings.granularity * dim],
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
        }
    }

    template <typename Robot>
    template <PataconExtendMode Mode>
    __device__ __forceinline__ void
    PataconBlockState<Robot>::validate_extension_edges(
        const PataconSearchContext<Robot> &search,
        int bid,
        int tid
    ) {
        using Collision = robots::CollisionTraits<Robot>;

        patacon_check_projected_edges_collision_parallel<Robot>(
            concon_projected_edge_count,
            concon_motion_segments,
            &search.concon_sphere_pos_scratch[
                bid * d_settings.max_concon_nodes *
                Collision::fine_sphere_count * Collision::batch_size * 3
            ],
            &search.concon_sphere_pos_approx_scratch[
                bid * d_settings.max_concon_nodes *
                Collision::approximate_sphere_count *
                Collision::batch_size * 3
            ],
            &search.concon_link_cc_scratch[
                bid * d_settings.max_concon_nodes *
                Collision::joint_flag_stride * Collision::batch_size
            ],
            &search.concon_transform_scratch[
                bid * d_settings.max_concon_nodes *
                Collision::batch_size * Collision::transform_slots * 16
            ],
            search.environment,
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
            if constexpr (Mode == PataconExtendMode::Exploration) {
                diagnostic_increment(DIAG_EXTEND_COLLISION_STOPS);
            }
            else {
                diagnostic_increment(DIAG_CONNECT_COLLISION_STOPS);
            }
        }
    }

    template <typename Robot>
    template <PataconExtendMode Mode, bool TraceTrees>
    __device__ __forceinline__ PataconExtendResult
    PataconBlockState<Robot>::insert_extension_nodes(
        const PataconSearchContext<Robot> &search,
        int tid
    ) {
        if (tid < dim) {
            config[tid] = nearest_node[tid];
        }
        if (tid == 0) {
            if constexpr (Mode == PataconExtendMode::Exploration) {
                concon_parent_idx = sindex[0];
            }
            else {
                concon_parent_idx = index;
            }
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
                return {PataconExtendStatus::Terminate};
            }

            if constexpr (Mode == PataconExtendMode::ConnectTarget) {
                __syncthreads();
                if (index < 0) {
                    return {PataconExtendStatus::Terminate};
                }
            }

            if (tid == 0) {
                t_parents[index] = concon_parent_idx;
            }
            __syncthreads();

            if (tid < dim) {
                config[tid] = stored_edge_endpoint;
                t_nodes[index * dim + tid] = config[tid];
            }
            __syncthreads();

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                const bool is_em_boundary_node =
                    concon_em_stop && edge_step == concon_count;

                if (tid == 0) {
                    new_ts_id = -1;
                    new_ts_basis_ok = true;

                    if (!is_em_boundary_node) {
                        t_node_ts_id[index] = selected_ts_id;
                    }
                    else {
                        new_ts_id = patacon_reserve_slot(
                            &search.ts_count[t_tree_id],
                            d_settings.max_tangent_spaces
                        );

                        if (new_ts_id < 0) {
                            new_ts_basis_ok = false;
                            atomicCAS((int *)&solved, 0, -1);
                        }
                        else {
                            t_ts_ready[new_ts_id] = 0;
                            t_ts_node_count[new_ts_id] = 0;
                            for (
                                int lane = 0;
                                lane < MAX_THREADS_PER_BLOCK;
                                ++lane
                            ) {
                                t_ts_lane_head[
                                    new_ts_id * MAX_THREADS_PER_BLOCK + lane
                                ] = -1;
                            }

                            t_ts_root_node_idx[new_ts_id] = index;
                            new_ts_basis_ok =
                                patacon_store_tangent_basis<Robot>(
                                    &t_nodes[index * dim],
                                    t_ts_bases,
                                    new_ts_id
                                );

                            if (new_ts_basis_ok) {
                                t_ts_parent_id[new_ts_id] = selected_ts_id;
                                t_node_ts_id[index] = new_ts_id;
                            }
                            else {
                                t_node_ts_id[index] = -1;
                                atomicCAS((int *)&solved, 0, -1);
                            }
                        }
                    }
                }
                __syncthreads();

                if (
                    is_em_boundary_node &&
                    (new_ts_id < 0 || !new_ts_basis_ok)
                ) {
                    return {PataconExtendStatus::Terminate};
                }

                if (tid < dim) {
                    if (!is_em_boundary_node) {
                        t_node_ts_q[index * dim + tid] = concon_probe[tid];
                    }
                    else if (new_ts_basis_ok) {
                        t_node_ts_q[index * dim + tid] = config[tid];
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
                    return {PataconExtendStatus::Terminate};
                }
            }

            if constexpr (TangentSpaceTraits<Robot>::enabled) {
                if (tid == 0) {
                    const int assigned_ts_id = t_node_ts_id[index];
                    patacon_register_node_in_ts(
                        index,
                        assigned_ts_id,
                        t_ts_node_count,
                        t_ts_lane_head,
                        t_node_next_in_ts
                    );
                }
            }
            __syncthreads();

            if (tid == 0) {
                if constexpr (Mode == PataconExtendMode::Exploration) {
                    search.node_ready[t_tree_id][index] =
                        current_search_generation;
                }
                else {
                    t_node_ready[index] = current_search_generation;
                }

                if constexpr (TraceTrees) {
                    __threadfence();
                    atomicAdd((int *)&completed_nodes[t_tree_id], 1);
                }
                else {
                    atomicAdd((int *)&completed_nodes[t_tree_id], 1);
                    __threadfence();
                }

                if constexpr (TangentSpaceTraits<Robot>::enabled) {
                    const bool is_em_boundary_node =
                        concon_em_stop && edge_step == concon_count;

                    if (
                        is_em_boundary_node &&
                        new_ts_basis_ok &&
                        new_ts_id >= 0
                    ) {
                        t_ts_ready[new_ts_id] =
                            current_search_generation;
                    }
                }
            }
            __syncthreads();

            if (tid == 0) {
                concon_parent_idx = index;
                concon_valid_count++;
            }
            __syncthreads();
        }

        if constexpr (Mode == PataconExtendMode::Exploration) {
            return {
                concon_valid_count > 0
                    ? PataconExtendStatus::Advanced
                    : PataconExtendStatus::Trapped
            };
        }
        else {
            if (connect_failed) {
                return {PataconExtendStatus::Trapped};
            }
            return {
                concon_valid_count > 0
                    ? PataconExtendStatus::Advanced
                    : PataconExtendStatus::Trapped
            };
        }
    }

// Host-side implementation stages used by the pseudocode-shaped solve flow.
// Included by Solve.cuh after the PATACON kernel definition is available.

    template <typename Robot>
    class PataconSolveSession {
    public:
        static constexpr int dim = Robot::dimension;
        using Collision = robots::CollisionTraits<Robot>;
        using Configuration = typename Robot::Configuration;

        PataconSolveSession(
            Configuration &start_configuration,
            std::vector<Configuration> &goal_configurations,
            ppln::collision::Environment<float> &host_environment,
            PATACON_settings &solve_settings
        )
            : start(start_configuration),
              goals(goal_configurations),
              h_environment(host_environment),
              settings(solve_settings),
              start_time(std::chrono::steady_clock::now()),
              num_goals(static_cast<int>(goal_configurations.size())),
              config_size(dim * sizeof(float)) {}

        void validate_request() const {
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
                if (
                    goals.size() > static_cast<std::size_t>(
                        settings.max_tangent_spaces
                    )
                ) {
                    throw std::invalid_argument(
                        "number of goals exceeds max_tangent_spaces"
                    );
                }
            }
        }

        void publish_request_settings() {
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
        }

        void prepare_workspace() {
            select_reusable_workspace();
            acquire_pointer_tables();

            search_generation = reusable_workspace != nullptr
                ? reusable_workspace->begin_request()
                : 1;
            cudaCheckError(cudaMemcpyToSymbol(
                current_search_generation,
                &search_generation,
                sizeof(search_generation)
            ));

            allocate_tree_storage();
            publish_tree_pointer_tables();
        }

        void initialize_request_state() {
            const int host_ts_count[2] = {0, 0};
            cudaMemcpy(
                ts_count,
                host_ts_count,
                2 * sizeof(int),
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                d_node_ready,
                node_ready,
                2 * sizeof(int *),
                cudaMemcpyHostToDevice
            );

            if (reusable_workspace == nullptr) {
                std::vector<float> nodes_init(
                    static_cast<std::size_t>(settings.max_samples) * dim,
                    UNWRITTEN_VAL
                );
                cudaMemcpy(
                    nodes[0],
                    nodes_init.data(),
                    config_size * settings.max_samples,
                    cudaMemcpyHostToDevice
                );
                cudaMemcpy(
                    nodes[1],
                    nodes_init.data(),
                    config_size * settings.max_samples,
                    cudaMemcpyHostToDevice
                );
                cudaMemcpy(
                    node_ts_q[0],
                    nodes_init.data(),
                    config_size * settings.max_samples,
                    cudaMemcpyHostToDevice
                );
                cudaMemcpy(
                    node_ts_q[1],
                    nodes_init.data(),
                    config_size * settings.max_samples,
                    cudaMemcpyHostToDevice
                );
            }

            initialize_samplers();
            initialize_tree_counters();
            prepare_environment();
            prepare_collision_scratch();
            prepare_host_signal();
        }

        void upload_problem_roots() {
            const auto copy_start_time = std::chrono::steady_clock::now();

            cudaMemcpy(
                nodes[0],
                start.data(),
                config_size,
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                parents[0],
                &start_index,
                sizeof(int),
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                nodes[1],
                goals.data(),
                config_size * num_goals,
                cudaMemcpyHostToDevice
            );

            std::vector<int> goal_parents(num_goals);
            std::iota(goal_parents.begin(), goal_parents.end(), 0);
            cudaMemcpy(
                parents[1],
                goal_parents.data(),
                sizeof(int) * num_goals,
                cudaMemcpyHostToDevice
            );

            const int start_ready = search_generation;
            std::vector<int> goals_ready(num_goals, search_generation);
            cudaMemcpy(
                node_ready[0],
                &start_ready,
                sizeof(int),
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                node_ready[1],
                goals_ready.data(),
                sizeof(int) * num_goals,
                cudaMemcpyHostToDevice
            );

            initialize_root_tangent_spaces();
            result.copy_ns = get_elapsed_nanoseconds(copy_start_time);
        }

        void launch_search() {
            publish_kernel_time_limit();

            const auto kernel_start_time = std::chrono::steady_clock::now();
            const int concon_threads_per_block =
                CONCON_COLLISION_THREADS_PER_EDGE * settings.max_concon_nodes;
            const PataconSearchContext<Robot> search_context{
                d_nodes,
                d_parents,
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
                environment,
                concon_sphere_pos_scratch,
                concon_sphere_pos_approx_scratch,
                concon_link_cc_scratch,
                concon_transform_scratch
            };

            if (settings.trace_trees) {
                patacon<Robot, true>
                    <<<settings.num_new_configs, concon_threads_per_block>>>(
                        search_context
                    );
            }
            else {
                patacon<Robot, false>
                    <<<settings.num_new_configs, concon_threads_per_block>>>(
                        search_context
                    );
            }

            cudaCheckError(cudaGetLastError());
            cudaCheckError(cudaDeviceSynchronize());
            result.kernel_ns = get_elapsed_nanoseconds(kernel_start_time);
        }

        void collect_result() {
            auto copy_start_time = std::chrono::steady_clock::now();
            cudaMemcpyFromSymbol(
                current_samples,
                atomic_free_index,
                sizeof(int) * 2,
                0,
                cudaMemcpyDeviceToHost
            );
            cudaMemcpyFromSymbol(
                h_solved,
                solved,
                sizeof(int),
                0,
                cudaMemcpyDeviceToHost
            );
            cudaMemcpyFromSymbol(
                &h_solved_iters,
                solved_iters,
                sizeof(int),
                0,
                cudaMemcpyDeviceToHost
            );

            if (settings.collect_diagnostics) {
                collect_diagnostics();
            }

            for (int tree = 0; tree < 2; tree++) {
                current_samples[tree] = std::clamp(
                    current_samples[tree],
                    0,
                    settings.max_samples
                );
            }
            result.copy_ns += get_elapsed_nanoseconds(copy_start_time);
            cudaCheckError(cudaGetLastError());

            if (*h_solved != 1) {
                *h_solved = 0;
            }
            result.start_tree_size = current_samples[0];
            result.goal_tree_size = current_samples[1];
            if (*h_solved) {
                assemble_solution_path_from_device<Robot>(
                    result,
                    start,
                    goals
                );
            }
            result.solved = (*h_solved) != 0;
            result.iters = h_solved_iters;

            if (settings.trace_trees) {
                collect_tree_trace();
            }
        }

        void release_request_resources() {
            if (reusable_workspace == nullptr) {
                cleanup_environment_on_device(environment, h_environment);
            }
            reset_device_variables();

            if (reusable_workspace == nullptr) {
                release_request_local_storage();
            }

            cudaCheckError(cudaGetLastError());
            result.wall_ns = get_elapsed_nanoseconds(start_time);
            if (runtime_control_state().cuda_device_reset_enabled) {
                cudaDeviceReset();
            }
        }

        PlannerResult<Robot> take_result() {
            return std::move(result);
        }

    private:
        void select_reusable_workspace() {
            if constexpr (std::is_same_v<Robot, robots::G1>) {
                if (runtime_control_state().persistent_workspace_enabled) {
                    auto &cached_workspace = persistent_g1_workspace();
                    if (
                        cached_workspace == nullptr ||
                        !cached_workspace->matches(settings)
                    ) {
                        cached_workspace =
                            std::make_unique<SolveWorkspace<robots::G1>>(
                                settings
                            );
                    }
                    reusable_workspace = cached_workspace.get();
                }
            }
        }

        void acquire_pointer_tables() {
            if (reusable_workspace != nullptr) {
                for (int tree = 0; tree < 2; ++tree) {
                    nodes[tree] = reusable_workspace->nodes[tree];
                    parents[tree] = reusable_workspace->parents[tree];
                    node_ready[tree] = reusable_workspace->node_ready[tree];
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
                    ts_lane_head[tree] =
                        reusable_workspace->ts_lane_head[tree];
                    node_next_in_ts[tree] =
                        reusable_workspace->node_next_in_ts[tree];
                }
                d_nodes = reusable_workspace->d_nodes;
                d_parents = reusable_workspace->d_parents;
                d_node_ready = reusable_workspace->d_node_ready;
                d_node_ts_id = reusable_workspace->d_node_ts_id;
                d_node_ts_q = reusable_workspace->d_node_ts_q;
                ts_count = reusable_workspace->ts_count;
                d_ts_root_node_idx =
                    reusable_workspace->d_ts_root_node_idx;
                d_ts_parent_id = reusable_workspace->d_ts_parent_id;
                d_ts_bases = reusable_workspace->d_ts_bases;
                d_ts_ready = reusable_workspace->d_ts_ready;
                d_ts_node_count = reusable_workspace->d_ts_node_count;
                d_ts_lane_head = reusable_workspace->d_ts_lane_head;
                d_node_next_in_ts =
                    reusable_workspace->d_node_next_in_ts;
                return;
            }

            cudaMalloc(&d_nodes, 2 * sizeof(float *));
            cudaMalloc(&d_parents, 2 * sizeof(int *));
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
        }

        void allocate_tree_storage() {
            for (int tree = 0; tree < 2; tree++) {
                if constexpr (TangentSpaceTraits<Robot>::enabled) {
                    allocate_tangent_space_storage(tree);
                }

                if (reusable_workspace == nullptr) {
                    cudaMalloc(
                        &nodes[tree],
                        settings.max_samples * config_size
                    );
                    cudaMalloc(
                        &parents[tree],
                        settings.max_samples * sizeof(int)
                    );
                    cudaMalloc(
                        &node_ready[tree],
                        settings.max_samples * sizeof(int)
                    );
                    cudaMemset(
                        node_ready[tree],
                        0,
                        settings.max_samples * sizeof(int)
                    );
                    cudaMalloc(
                        &node_ts_id[tree],
                        settings.max_samples * sizeof(int)
                    );
                    cudaMemset(
                        node_ts_id[tree],
                        0xff,
                        settings.max_samples * sizeof(int)
                    );
                    cudaMalloc(
                        &node_ts_q[tree],
                        settings.max_samples * config_size
                    );
                    cudaMalloc(
                        &ts_root_node_idx[tree],
                        settings.max_tangent_spaces * sizeof(int)
                    );
                    cudaMalloc(
                        &ts_ready[tree],
                        settings.max_tangent_spaces * sizeof(int)
                    );
                    cudaMemset(
                        ts_ready[tree],
                        0,
                        settings.max_tangent_spaces * sizeof(int)
                    );
                }
            }
        }

        void allocate_tangent_space_storage(int tree) {
            if (reusable_workspace != nullptr) {
                return;
            }

            const std::size_t basis_bytes =
                static_cast<std::size_t>(settings.max_tangent_spaces) *
                TangentSpaceTraits<Robot>::basis_size * sizeof(float);
            const std::size_t ts_count_bytes =
                static_cast<std::size_t>(settings.max_tangent_spaces) *
                sizeof(int);
            const std::size_t ts_lane_head_bytes =
                static_cast<std::size_t>(settings.max_tangent_spaces) *
                MAX_THREADS_PER_BLOCK * sizeof(int);
            const std::size_t node_next_bytes =
                static_cast<std::size_t>(settings.max_samples) * sizeof(int);

            cudaMalloc(&ts_bases[tree], basis_bytes);
            cudaMemset(ts_bases[tree], 0, basis_bytes);
            cudaMalloc(&ts_parent_id[tree], ts_count_bytes);
            cudaMemset(ts_parent_id[tree], 0xff, ts_count_bytes);
            cudaMalloc(&ts_node_count[tree], ts_count_bytes);
            cudaMemset(ts_node_count[tree], 0, ts_count_bytes);
            cudaMalloc(&ts_lane_head[tree], ts_lane_head_bytes);
            cudaMemset(ts_lane_head[tree], 0xff, ts_lane_head_bytes);
            cudaMalloc(&node_next_in_ts[tree], node_next_bytes);
            cudaMemset(node_next_in_ts[tree], 0xff, node_next_bytes);
        }

        void publish_tree_pointer_tables() {
            if (reusable_workspace != nullptr) {
                return;
            }

            cudaMemcpy(
                d_nodes,
                nodes,
                2 * sizeof(float *),
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                d_parents,
                parents,
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
        }

        void initialize_samplers() {
            rng_states = reusable_workspace != nullptr
                ? reusable_workspace->rng_states
                : nullptr;
            const int rng_state_count = settings.num_new_configs * dim;
            if (reusable_workspace == nullptr) {
                cudaMalloc(
                    &rng_states,
                    rng_state_count * sizeof(curandState)
                );
            }
            const int rng_block_count =
                (rng_state_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            init_rng<<<rng_block_count, BLOCK_SIZE>>>(
                rng_states,
                settings.random_seed,
                rng_state_count
            );

            halton_states = reusable_workspace != nullptr
                ? reusable_workspace->halton_states
                : nullptr;
            if (reusable_workspace == nullptr) {
                cudaMalloc(
                    &halton_states,
                    settings.num_new_configs * sizeof(HaltonState<Robot>)
                );
            }
            const int halton_block_count =
                (settings.num_new_configs + BLOCK_SIZE - 1) / BLOCK_SIZE;
            init_halton<Robot><<<halton_block_count, BLOCK_SIZE>>>(
                halton_states,
                rng_states
            );
        }

        void initialize_tree_counters() {
            const int free_index[2] = {1, num_goals};
            cudaMemcpyToSymbol(
                atomic_free_index,
                free_index,
                sizeof(int) * 2
            );
            cudaMemcpyToSymbol(nodes_size, free_index, sizeof(int) * 2);

            const int completed[2] = {1, num_goals};
            cudaMemcpyToSymbol(
                completed_nodes,
                completed,
                sizeof(int) * 2
            );
        }

        void prepare_environment() {
            if (reusable_workspace != nullptr) {
                environment =
                    reusable_workspace->environment.update(h_environment);
            }
            else {
                setup_environment_on_device(environment, h_environment);
            }
            cudaCheckError(cudaGetLastError());
        }

        void prepare_collision_scratch() {
            const std::size_t scratch_slots =
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
            }
            else {
                cudaMalloc(
                    &concon_sphere_pos_scratch,
                    scratch_slots * Collision::fine_sphere_count *
                        Collision::batch_size * 3 * sizeof(float)
                );
                cudaMalloc(
                    &concon_sphere_pos_approx_scratch,
                    scratch_slots * Collision::approximate_sphere_count *
                        Collision::batch_size * 3 * sizeof(float)
                );
                cudaMalloc(
                    &concon_link_cc_scratch,
                    scratch_slots * Collision::joint_flag_stride *
                        Collision::batch_size * sizeof(int)
                );
                cudaMalloc(
                    &concon_transform_scratch,
                    scratch_slots * Collision::batch_size *
                        Collision::transform_slots * 16 * sizeof(float)
                );
            }
            cudaCheckError(cudaGetLastError());
        }

        void prepare_host_signal() {
            h_solved = reusable_workspace != nullptr
                ? reusable_workspace->h_solved
                : nullptr;
            if (reusable_workspace == nullptr) {
                cudaMallocHost(&h_solved, sizeof(int));
            }
            *h_solved = -1;
        }

        void initialize_root_tangent_spaces() {
            if constexpr (TangentSpaceTraits<Robot>::enabled) {
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

                const int initial_ts_count[2] = {1, num_goals};
                cudaMemcpy(
                    ts_count,
                    initial_ts_count,
                    2 * sizeof(int),
                    cudaMemcpyHostToDevice
                );
                cudaCheckError(cudaGetLastError());
            }
        }

        void publish_kernel_time_limit() {
            const RuntimeControlState runtime = runtime_control_state();
            unsigned long long kernel_time_limit_ns = 0;
            if (runtime.time_limit_seconds > 0.0) {
                const auto total_time_limit_ns =
                    static_cast<std::uint64_t>(
                        std::llround(runtime.time_limit_seconds * 1.0e9)
                    );
                if (runtime.time_limit_counts_kernel_only) {
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
        }

        void collect_diagnostics() {
            unsigned long long host_diagnostics[
                DIAGNOSTIC_COUNTER_COUNT
            ] = {};
            cudaMemcpy(
                result.diagnostics.tangent_space_count.data(),
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

            result.diagnostics.extend_attempts =
                host_diagnostics[DIAG_EXTEND_ATTEMPTS];
            result.diagnostics.extend_backtracking_flips =
                host_diagnostics[DIAG_EXTEND_BACKTRACKING_FLIPS];
            result.diagnostics.extend_em_stops =
                host_diagnostics[DIAG_EXTEND_EM_STOPS];
            result.diagnostics.extend_anchor_projection_stops =
                host_diagnostics[DIAG_EXTEND_ANCHOR_PROJECTION_STOPS];
            result.diagnostics.extend_edge_projection_stops =
                host_diagnostics[DIAG_EXTEND_EDGE_PROJECTION_STOPS];
            result.diagnostics.extend_collision_stops =
                host_diagnostics[DIAG_EXTEND_COLLISION_STOPS];
            result.diagnostics.extend_full_successes =
                host_diagnostics[DIAG_EXTEND_FULL_SUCCESSES];
            result.diagnostics.connect_attempts =
                host_diagnostics[DIAG_CONNECT_ATTEMPTS];
            result.diagnostics.connect_chunks =
                host_diagnostics[DIAG_CONNECT_CHUNKS];
            result.diagnostics.connect_invalid_tangent_spaces =
                host_diagnostics[DIAG_CONNECT_INVALID_TANGENT_SPACES];
            result.diagnostics.connect_tangent_direction_stops =
                host_diagnostics[DIAG_CONNECT_TANGENT_DIRECTION_STOPS];
            result.diagnostics.connect_em_stops =
                host_diagnostics[DIAG_CONNECT_EM_STOPS];
            result.diagnostics.connect_anchor_projection_stops =
                host_diagnostics[DIAG_CONNECT_ANCHOR_PROJECTION_STOPS];
            result.diagnostics.connect_edge_projection_stops =
                host_diagnostics[DIAG_CONNECT_EDGE_PROJECTION_STOPS];
            result.diagnostics.connect_progress_stops =
                host_diagnostics[DIAG_CONNECT_PROGRESS_STOPS];
            result.diagnostics.connect_collision_stops =
                host_diagnostics[DIAG_CONNECT_COLLISION_STOPS];
            result.diagnostics.connect_successes =
                host_diagnostics[DIAG_CONNECT_SUCCESSES];
            result.diagnostics.connect_failures =
                host_diagnostics[DIAG_CONNECT_FAILURES];
        }

        void collect_tree_trace() {
            const auto copy_start_time = std::chrono::steady_clock::now();
            copy_tree_trace_to_result(
                result,
                nodes,
                parents,
                node_ready,
                current_samples,
                search_generation
            );

            if (result.solved) {
                cudaMemcpyFromSymbol(
                    &result.connection_tree_id,
                    connection_tree_id,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );
                cudaMemcpyFromSymbol(
                    &result.connection_node_idx,
                    connection_node_idx,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );
                cudaMemcpyFromSymbol(
                    &result.connection_other_tree_id,
                    connection_other_tree_id,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );
                cudaMemcpyFromSymbol(
                    &result.connection_other_node_idx,
                    connection_other_node_idx,
                    sizeof(int),
                    0,
                    cudaMemcpyDeviceToHost
                );
                fill_solution_trace(result);
            }

            result.copy_ns += get_elapsed_nanoseconds(copy_start_time);
            cudaCheckError(cudaGetLastError());
        }

        void release_request_local_storage() {
            cudaFree(nodes[0]);
            cudaFree(nodes[1]);
            cudaFree(parents[0]);
            cudaFree(parents[1]);

            for (int tree = 0; tree < 2; tree++) {
                if (ts_root_node_idx[tree] != nullptr) {
                    cudaFree(ts_root_node_idx[tree]);
                }
                if (ts_parent_id[tree] != nullptr) {
                    cudaFree(ts_parent_id[tree]);
                }
                if (ts_ready[tree] != nullptr) {
                    cudaFree(ts_ready[tree]);
                }
                if (ts_bases[tree] != nullptr) {
                    cudaFree(ts_bases[tree]);
                }
                if (node_ts_id[tree] != nullptr) {
                    cudaFree(node_ts_id[tree]);
                }
                if (node_ts_q[tree] != nullptr) {
                    cudaFree(node_ts_q[tree]);
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

            cudaFree(node_ready[0]);
            cudaFree(node_ready[1]);
            cudaFree(rng_states);
            cudaFree(halton_states);
            cudaFree(d_nodes);
            cudaFree(d_parents);
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
            cudaFree(concon_sphere_pos_scratch);
            cudaFree(concon_sphere_pos_approx_scratch);
            cudaFree(concon_link_cc_scratch);
            cudaFree(concon_transform_scratch);
            cudaFreeHost(h_solved);
        }

        Configuration &start;
        std::vector<Configuration> &goals;
        ppln::collision::Environment<float> &h_environment;
        PATACON_settings &settings;
        std::chrono::steady_clock::time_point start_time;
        std::size_t start_index = 0;
        PlannerResult<Robot> result;
        int num_goals;
        std::size_t config_size;

        float *nodes[2] = {nullptr, nullptr};
        int *parents[2] = {nullptr, nullptr};
        int *node_ready[2] = {nullptr, nullptr};
        float **d_nodes = nullptr;
        int **d_parents = nullptr;
        int **d_node_ready = nullptr;
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
        SolveWorkspace<Robot> *reusable_workspace = nullptr;
        int search_generation = 1;
        curandState *rng_states = nullptr;
        HaltonState<Robot> *halton_states = nullptr;
        ppln::collision::Environment<float> *environment = nullptr;
        int *h_solved = nullptr;
        int current_samples[2] = {0, 0};
        int h_solved_iters = -1;
    };

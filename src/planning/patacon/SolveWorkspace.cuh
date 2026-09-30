// Internal reusable solve-workspace implementation.
// Included by PATACON.cu inside namespace PATACON.

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

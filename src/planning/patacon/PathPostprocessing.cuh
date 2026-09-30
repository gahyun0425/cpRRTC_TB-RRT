// Internal solution tracing, visualization simplification, and validation implementation.
// Included by PATACON.cu inside namespace PATACON.

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


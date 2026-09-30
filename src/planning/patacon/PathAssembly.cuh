// Host-side download and assembly of the two device path segments.

    template <typename Robot>
    void assemble_solution_path_from_device(
        PlannerResult<Robot> &result,
        const typename Robot::Configuration &start,
        const std::vector<typename Robot::Configuration> &goals
    ) {
        static constexpr int dim = Robot::dimension;
        int device_path_size[2];
        std::vector<float> device_paths[2];
        float device_cost;
        int reached_goal_index;

        cudaMemcpyFromSymbol(
            device_path_size,
            path_size,
            sizeof(int) * 2,
            0,
            cudaMemcpyDeviceToHost
        );

        for (int tree = 0; tree < 2; ++tree) {
            if (
                device_path_size[tree] < 0 ||
                device_path_size[tree] > MAX_PATH_NODES
            ) {
                throw std::runtime_error(
                    "PATACON device path size is outside its valid range"
                );
            }

            device_paths[tree].resize(
                static_cast<std::size_t>(device_path_size[tree]) * dim
            );

            if (!device_paths[tree].empty()) {
                cudaMemcpyFromSymbol(
                    device_paths[tree].data(),
                    path,
                    sizeof(float) * device_paths[tree].size(),
                    sizeof(float) * static_cast<std::size_t>(tree) *
                        MAX_PATH_STORAGE,
                    cudaMemcpyDeviceToHost
                );
            }
        }

        cudaMemcpyFromSymbol(
            &device_cost,
            cost,
            sizeof(float),
            0,
            cudaMemcpyDeviceToHost
        );
        cudaMemcpyFromSymbol(
            &reached_goal_index,
            reached_goal_idx,
            sizeof(int),
            0,
            cudaMemcpyDeviceToHost
        );
        cudaCheckError(cudaGetLastError());

        result.path.emplace_back(goals[reached_goal_index]);

        typename Robot::Configuration config;
        for (int i = device_path_size[1] - 1; i >= 0; i--) {
            std::copy_n(
                device_paths[1].data() + i * dim,
                dim,
                config.begin()
            );
            result.path.emplace_back(config);
        }

        for (int i = 0; i < device_path_size[0]; i++) {
            std::copy_n(
                device_paths[0].data() + i * dim,
                dim,
                config.begin()
            );
            result.path.emplace_back(config);
        }

        result.path.emplace_back(start);
        result.cost = configuration_space_path_arclength<Robot>(result.path);
        result.path_length = device_path_size[0] + device_path_size[1];

        // Preserve the device cost download used by the original result path.
        (void)device_cost;
    }

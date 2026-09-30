#pragma once

#include <vector>
#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include <cmath>

#include "Robots.hh"
#include "src/collision/environment.hh"
#include "PATACON_settings.hh"
#include "PlannerResult.hh"

template <typename Robot>
inline float l2dist(typename Robot::Configuration &a, typename Robot::Configuration &b)
{
    float res = 0;
    float diff;
    for (int i = 0; i < a.size(); i++) {
        diff = a[i] - b[i];
        res += diff * diff;
    }
    return sqrt(res);
}

template <typename Robot>
inline void print_cfg_ptr(float *config) {
    for (int i = 0; i < Robot::dimension; i++) {
        std::cout << config[i] << " ";
    }
    std::cout << "\n";
}

template <typename Robot>
inline void print_cfg(typename Robot::Configuration &config) {
    for (int i = 0; i < Robot::dimension; i++) {
        std::cout << config[i] << " ";
    }
    std::cout << "\n";
}

template <typename Robot>
inline void print_cfg_to_ss(typename Robot::Configuration &config, std::stringstream &out) {
    for (int i = 0; i < Robot::dimension; i++) {
        out << config[i] << " ";
    }
    out << "\\n";
}

template <typename Robot>
inline float configuration_space_path_arclength(
    const std::vector<typename Robot::Configuration>& path
) {
    if (path.size() < 2) {
        return 0.0f;
    }

    double total_length = 0.0;

    for (std::size_t k = 1; k < path.size(); ++k) {
        double squared_distance = 0.0;

        for (int j = 0; j < Robot::dimension; ++j) {
            const double dq =
                static_cast<double>(path[k][j])
                - static_cast<double>(path[k - 1][j]);

            squared_distance += dq * dq;
        }

        total_length += std::sqrt(squared_distance);
    }

    return static_cast<float>(total_length);
}



inline std::size_t get_elapsed_nanoseconds(const std::chrono::time_point<std::chrono::steady_clock> &start)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
}

/* This file handles the declarations of each solve function so that they may be called from .cpp files. Implementation is in .cu files.*/
namespace RRT {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment);
}

namespace pRRT {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment);
}


namespace nRRT {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment);
}

namespace PATACON {
    struct RuntimeControlState {
        bool cuda_device_reset_enabled = true;
        bool persistent_workspace_enabled = false;
        double time_limit_seconds = 0.0;
        bool time_limit_counts_kernel_only = false;
    };

    RuntimeControlState runtime_control_state();
    void set_runtime_control_state(const RuntimeControlState &state);
    void set_cuda_device_reset_enabled(bool enabled);
    void set_persistent_workspace_enabled(bool enabled);
    void release_persistent_workspace();
    void set_time_limit_seconds(double seconds);
    bool project_g1_configuration(
        ppln::robots::G1::Configuration &configuration,
        const PATACON_settings &settings
    );

    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment, PATACON_settings &settings);

    template <typename Robot>
    PathSimplificationResult<Robot> simplify_path_for_visualization(
        const std::vector<typename Robot::Configuration> &path,
        ppln::collision::Environment<float> &environment,
        PATACON_settings &settings
    );

    template <typename Robot>
    PathValidationResult validate_path_for_visualization(
        const std::vector<typename Robot::Configuration> &path,
        ppln::collision::Environment<float> &environment,
        PATACON_settings &settings,
        float projection_tolerance = 1.0e-5f
    );
}

namespace nPATACON {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment, PATACON_settings &settings);
}

namespace pwPATACON {
    template <typename Robot>
    PlannerResult<Robot> solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals, ppln::collision::Environment<float> &environment, PATACON_settings &settings);
}

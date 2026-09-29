#pragma once

#include "Planners.hh"

struct AORRTC_settings : PATACON_settings {
    // Opt-in flag used by single_mbm.
    bool aorrtc = false;

    // Total anytime search budget in seconds. Default: 5 s.
    double time_limit_sec = 5.0;

    // Algorithm 1, Line 46: augmented nearest-neighbour score weights.
    float aorrtc_config_weight = 1.0f;
    float aorrtc_cost_weight = 1.0f;

    // Strict improvement tolerance used for c_max comparisons.
    float cost_improvement_epsilon = 1.0e-6f;

    // Algorithm 1, Lines 27-32 can theoretically repeat until the same parent
    // or an invalid edge is encountered. The GPU implementation uses this
    // finite safety cap; every accepted replacement strictly lowers c_new.
    int aorrtc_max_parent_resamples = 1000;
};

template <typename Robot>
struct AORRTCSolutionUpdate {
    int update_index = -1;
    int source_tree_id = -1;
    int source_node_idx = -1;
    int target_tree_id = -1;
    int target_node_idx = -1;
    int iteration = -1;
    float cost = 0.0f;
    std::size_t found_ns = 0;
    std::vector<typename Robot::Configuration> path_start_to_goal;
    std::vector<std::array<int, 2>> solution_trace;
};

template <typename Robot>
struct AORRTCResult : PlannerResult<Robot> {
    float initial_cost = 0.0f;
    std::size_t planning_ns = 0;
    std::size_t initial_solution_ns = 0;
    std::size_t best_solution_ns = 0;
    std::size_t initial_kernel_ns = 0;
    int solution_updates = 0;
    int search_restarts = 0;
    std::vector<AORRTCSolutionUpdate<Robot>> solution_history;
    bool solution_history_overflow = false;
};

namespace AORRTC {
    template <typename Robot>
    AORRTCResult<Robot> solve(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &environment,
        AORRTC_settings &settings
    );
}

namespace PATACON {
    template <typename Robot>
    AORRTCResult<Robot> solve(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &environment,
        AORRTC_settings &settings
    );
}

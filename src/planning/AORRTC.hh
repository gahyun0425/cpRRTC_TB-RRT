#pragma once

#include "Planners.hh"

struct AORRTC_settings : pRRTC_settings {
    // Kept in the CLI settings object so result/JSON code can select the
    // AORRTC implementation without changing pRRTC_settings.
    bool aorrtc = false;

    // Budget for AORRTC tree expansion, excluding one-time GPU setup/cleanup.
    double time_limit_sec = 5.0;

    // Augmented nearest-neighbour score weights.
    float aorrtc_config_weight = 1.0f;
    float aorrtc_cost_weight = 1.0f;

    // A candidate must improve the current best by more than this tolerance.
    float cost_improvement_epsilon = 1.0e-6f;
};

template <typename Robot>
struct AORRTCResult : PlannerResult<Robot> {
    float initial_cost = 0.0f;
    std::size_t initial_solution_ns = 0;
    std::size_t best_solution_ns = 0;
    int solution_updates = 0;
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

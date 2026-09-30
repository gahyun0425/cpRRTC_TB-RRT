#pragma once

#include <nlohmann/json.hpp>

#include <type_traits>
#include <vector>

#include "src/planning/Robots.hh"
#include "src/planning/PATACON_settings.hh"
#include "FrankaConstraintConfig.hh"
#include "G1ConstraintConfig.hh"
#include "IgrisCConstraintConfig.hh"

namespace ppln::constraints {

template <typename Robot>
struct PreparedConstraintQuery {
    typename Robot::Configuration start{};
    std::vector<typename Robot::Configuration> goals;
};

template <typename Robot>
inline PreparedConstraintQuery<Robot> prepare_constraint_query(
    const nlohmann::json &problem,
    PATACON_settings &settings
) {
    PreparedConstraintQuery<Robot> query{
        problem.at("start").template get<typename Robot::Configuration>(),
        problem.at("goals").template get<
            std::vector<typename Robot::Configuration>
        >()
    };

    if constexpr (std::is_same_v<Robot, robots::G1>) {
        query.start = g1_start_from_problem(
            problem, settings.axis
        ).template get<typename Robot::Configuration>();
        query.goals = g1_goals_from_problem(
            problem, settings.axis
        ).template get<std::vector<typename Robot::Configuration>>();
        settings.g1_constraints = g1_constraint_parameters_from_problem(problem);
    } else if constexpr (std::is_same_v<Robot, robots::IgrisC>) {
        settings.igris_c_constraints =
            igris_c_constraint_parameters_from_problem(problem);
    } else if constexpr (
        std::is_same_v<Robot, robots::FrankaSingle> ||
        std::is_same_v<Robot, robots::Franka>
    ) {
        settings.franka_constraints =
            franka_constraint_parameters_from_start<Robot>(query.start);
    }
    return query;
}

template <typename Robot>
inline void apply_constraint_backend_defaults(PATACON_settings &settings) {
    if constexpr (std::is_same_v<Robot, robots::G1>) {
        settings.granularity = robots::G1::resolution;
    } else if constexpr (std::is_same_v<Robot, robots::IgrisC>) {
        settings.granularity = robots::IgrisC::resolution;
    }
}

}  // namespace ppln::constraints

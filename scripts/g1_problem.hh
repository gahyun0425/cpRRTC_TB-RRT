#pragma once

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>

#include "src/planning/G1ConstraintParameters.hh"

inline ppln::constraints::G1ConstraintParameters g1_constraint_parameters_from_problem(
    const nlohmann::json &problem
) {
    if (!problem.contains("constraints")) {
        throw std::invalid_argument("G1 problem is missing constraints");
    }
    const auto &constraints = problem.at("constraints");
    const auto &feet = constraints.at("feet");
    const auto &center_of_mass = constraints.at("com");
    const auto &bimanual = constraints.at("bimanual");

    ppln::constraints::G1ConstraintParameters parameters{};
    for (int foot = 0; foot < 2; ++foot) {
        for (int component = 0; component < 7; ++component) {
            parameters.feet_reference[foot][component] =
                feet.at("reference").at(foot).at(component).get<float>();
            parameters.feet_target[foot][component] =
                feet.at("target").at(foot).at(component).get<float>();
        }
    }
    for (int component = 0; component < 8; ++component) {
        parameters.support_polygon[component] =
            center_of_mass.at("support_polygon").at(component).get<float>();
    }
    for (int component = 0; component < 7; ++component) {
        parameters.bimanual_target[component] =
            bimanual.at("target").at(component).get<float>();
    }
    parameters.tolerance_squared =
        constraints.at("tolerance_squared").get<float>();
    return parameters;
}

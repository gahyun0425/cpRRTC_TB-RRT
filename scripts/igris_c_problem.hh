#pragma once

#include <nlohmann/json.hpp>

#include <stdexcept>

#include "src/planning/IgrisCConstraintParameters.hh"
#include "src/robots/igris_c.cuh"

inline ppln::constraints::IgrisCConstraintParameters
igris_c_constraint_parameters_from_problem(const nlohmann::json &problem) {
    if (!problem.contains("constraints")) {
        throw std::invalid_argument("IGRIS-C problem is missing constraints");
    }
    const auto &constraints = problem.at("constraints");
    const auto &feet = constraints.at("feet").at("target");
    const auto &center_of_mass = constraints.at("com");
    const auto &bimanual = constraints.at("bimanual").at("target");
    const auto &bimanual_axis = constraints.at("bimanual_axis");
    if (bimanual_axis.at("local_axis") !=
            nlohmann::json::array({1.0, 0.0, 0.0}) ||
        bimanual_axis.at("target_world_axis") !=
            nlohmann::json::array({0.0, 0.0, 1.0})) {
        throw std::invalid_argument(
            "IGRIS-C bimanual axis must align l_grasp local +X with world +Z"
        );
    }

    ppln::constraints::IgrisCConstraintParameters parameters{};
    for (int foot = 0; foot < 2; ++foot) {
        for (int component = 0; component < 7; ++component) {
            parameters.feet_target[foot][component] =
                feet.at(foot).at(component).get<float>();
        }
    }
    for (int component = 0; component < 8; ++component) {
        parameters.support_polygon[component] =
            center_of_mass.at("support_polygon").at(component).get<float>();
    }
    parameters.support_margin_m =
        center_of_mass.at("support_margin_m").get<float>();
    parameters.payload_mass_kg =
        center_of_mass.at("payload_mass_kg").get<float>();
    for (int component = 0; component < 7; ++component) {
        parameters.bimanual_target[component] =
            bimanual.at(component).get<float>();
    }
    parameters.tolerance_squared =
        constraints.at("tolerance_squared").get<float>();
    return parameters;
}

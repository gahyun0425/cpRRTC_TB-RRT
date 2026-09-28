#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "src/planning/G1ConstraintParameters.hh"

inline const nlohmann::json &g1_start_from_problem(
    const nlohmann::json &problem,
    bool rigid_orientation
) {
    if (!rigid_orientation) {
        return problem.at("start");
    }
    if (!problem.contains("rigid_orientation_endpoints")) {
        throw std::invalid_argument(
            "G1 problem is missing rigid_orientation_endpoints"
        );
    }
    return problem.at("rigid_orientation_endpoints").at("start");
}

inline const nlohmann::json &g1_goals_from_problem(
    const nlohmann::json &problem,
    bool /*rigid_orientation*/
) {
    // Use the axis-feasible goal in both modes so rigid_orientation changes
    // only the active constraint set, not the planning query's goal.
    if (!problem.contains("rigid_orientation_endpoints")) {
        throw std::invalid_argument(
            "G1 problem is missing rigid_orientation_endpoints"
        );
    }
    return problem.at("rigid_orientation_endpoints").at("goals");
}

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
    if (center_of_mass.contains("support_margin_m")) {
        parameters.support_margin_m =
            center_of_mass.at("support_margin_m").get<float>();
    }
    if (!std::isfinite(parameters.support_margin_m) ||
        parameters.support_margin_m < 0.0f) {
        throw std::invalid_argument(
            "G1 constraints.com.support_margin_m must be finite and non-negative"
        );
    }
    for (int component = 0; component < 7; ++component) {
        parameters.bimanual_target[component] =
            bimanual.at("target").at(component).get<float>();
    }
    if (center_of_mass.contains("payload")) {
        const auto &payload = center_of_mass.at("payload");
        parameters.payload_mass_kg =
            payload.at("mass_kg").get<float>();
        if (!std::isfinite(parameters.payload_mass_kg) ||
            parameters.payload_mass_kg < 0.0f) {
            throw std::invalid_argument(
                "G1 constraints.com.payload.mass_kg must be finite and non-negative"
            );
        }

        auto &collision = parameters.attached_object_collision;
        collision.enabled = parameters.payload_mass_kg > 0.0f;
        std::array<float, 3> hand_midpoint_offset{};
        if (payload.contains("hand_midpoint_offset")) {
            const auto &offset = payload.at("hand_midpoint_offset");
            if (!offset.is_array() || offset.size() != 3) {
                throw std::invalid_argument(
                    "G1 payload hand_midpoint_offset must contain 3 values"
                );
            }
            for (int axis = 0; axis < 3; ++axis) {
                hand_midpoint_offset[axis] = offset.at(axis).get<float>();
                if (!std::isfinite(hand_midpoint_offset[axis])) {
                    throw std::invalid_argument(
                        "G1 payload hand_midpoint_offset must be finite"
                    );
                }
            }
        }
        if (!payload.contains("half_extents")) {
            throw std::invalid_argument(
                "G1 constraints.com.payload requires half_extents"
            );
        }
        std::array<float, 3> half_extents{};
        for (int axis = 0; axis < 3; ++axis) {
            half_extents[axis] =
                payload.at("half_extents").at(axis).get<float>();
            if (!std::isfinite(half_extents[axis]) ||
                half_extents[axis] <= 0.0f) {
                throw std::invalid_argument(
                    "G1 payload half_extents must be finite and positive"
                );
            }
        }

        std::array<int, 3> spheres_per_axis{};
        constexpr float target_cell_half_extent = 0.02f;
        for (int axis = 0; axis < 3; ++axis) {
            spheres_per_axis[axis] = static_cast<int>(std::ceil(
                half_extents[axis] / target_cell_half_extent
            ));
        }
        if (payload.contains("collision_spheres_per_axis")) {
            const auto &counts = payload.at("collision_spheres_per_axis");
            if (!counts.is_array() || counts.size() != 3) {
                throw std::invalid_argument(
                    "G1 payload collision_spheres_per_axis must contain 3 integers"
                );
            }
            for (int axis = 0; axis < 3; ++axis) {
                spheres_per_axis[axis] = counts.at(axis).get<int>();
                if (spheres_per_axis[axis] <= 0) {
                    throw std::invalid_argument(
                        "G1 payload collision sphere counts must be positive"
                    );
                }
            }
        }

        const int sphere_count = spheres_per_axis[0] *
            spheres_per_axis[1] * spheres_per_axis[2];
        if (sphere_count >
            ppln::constraints::G1_ATTACHED_OBJECT_MAX_SPHERES) {
            throw std::invalid_argument(
                "G1 payload collision sphere count exceeds capacity"
            );
        }

        const std::array<float, 3> cell_half_extents = {
            half_extents[0] / spheres_per_axis[0],
            half_extents[1] / spheres_per_axis[1],
            half_extents[2] / spheres_per_axis[2],
        };
        float sphere_radius = std::sqrt(
            cell_half_extents[0] * cell_half_extents[0] +
            cell_half_extents[1] * cell_half_extents[1] +
            cell_half_extents[2] * cell_half_extents[2]
        );
        if (payload.contains("collision_sphere_radius")) {
            sphere_radius =
                payload.at("collision_sphere_radius").get<float>();
            if (!std::isfinite(sphere_radius) || sphere_radius <= 0.0f) {
                throw std::invalid_argument(
                    "G1 payload collision_sphere_radius must be finite and positive"
                );
            }
        }
        for (int z = 0; z < spheres_per_axis[2]; ++z) {
            for (int y = 0; y < spheres_per_axis[1]; ++y) {
                for (int x = 0; x < spheres_per_axis[0]; ++x) {
                    float *sphere = collision.spheres[collision.sphere_count++];
                    sphere[0] = -half_extents[0] +
                        (2.0f * x + 1.0f) * cell_half_extents[0];
                    sphere[1] = -half_extents[1] +
                        (2.0f * y + 1.0f) * cell_half_extents[1];
                    sphere[2] = -half_extents[2] +
                        (2.0f * z + 1.0f) * cell_half_extents[2];
                    sphere[3] = sphere_radius;
                }
            }
        }

        // The bimanual target translation is the right-hand origin in the
        // left-hand frame. Place the payload relative to that midpoint.
        for (int axis = 0; axis < 3; ++axis) {
            collision.left_hand_center_offset[axis] =
                0.5f * parameters.bimanual_target[4 + axis] +
                hand_midpoint_offset[axis];
        }

        if (payload.contains("collision_ignored_robot_spheres")) {
            const auto &ignored =
                payload.at("collision_ignored_robot_spheres");
            if (!ignored.is_array() || ignored.size() >
                ppln::constraints::G1_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_SPHERES) {
                throw std::invalid_argument(
                    "G1 payload ignored robot sphere list is invalid"
                );
            }
            for (const auto &entry : ignored) {
                const int sphere = entry.get<int>();
                if (sphere < 0 || sphere >= 133) {
                    throw std::invalid_argument(
                        "G1 payload ignored robot sphere index is out of range"
                    );
                }
                collision.ignored_robot_spheres[
                    collision.ignored_robot_sphere_count++
                ] = sphere;
            }
        }
    }
    parameters.tolerance_squared =
        constraints.at("tolerance_squared").get<float>();
    return parameters;
}

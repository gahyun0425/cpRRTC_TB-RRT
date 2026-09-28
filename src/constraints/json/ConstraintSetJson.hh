#pragma once

#include "AxisConstraintJson.hh"
#include "BimanualPoseConstraintJson.hh"
#include "ComConstraintJson.hh"
#include "FootPoseConstraintJson.hh"

namespace ppln::constraints::json_io {

inline Json normalize_constraint_set(const Json &constraints) {
    // Existing problem files already use the backend parameter layout. Keep
    // that layout fully backward compatible.
    if (constraints.is_object() && !constraints.contains("items")) {
        return constraints;
    }

    const Json *items = nullptr;
    Json output = Json::object();
    if (constraints.is_array()) {
        items = &constraints;
    } else if (constraints.is_object() && constraints.contains("items")) {
        items = &constraints.at("items");
        if (constraints.contains("tolerance_squared")) {
            output["tolerance_squared"] = constraints.at("tolerance_squared");
        }
    } else {
        throw std::invalid_argument(
            "constraints must be a legacy object, an array, or an object with items"
        );
    }
    if (!items->is_array()) {
        throw std::invalid_argument("constraints.items must be an array");
    }

    for (const auto &entry : *items) {
        const std::string type = constraint_type(entry);
        if (is_foot_pose_constraint(type)) {
            append_foot_pose_constraint(output, entry);
        } else if (is_bimanual_pose_constraint(type)) {
            append_bimanual_pose_constraint(output, entry);
        } else if (is_axis_constraint(type)) {
            append_axis_constraint(output, entry);
        } else if (is_com_constraint(type)) {
            append_com_constraint(output, entry);
        } else {
            throw std::invalid_argument("unsupported constraint type: " + type);
        }
    }
    validate_foot_pose_constraints(output);
    if (!output.contains("tolerance_squared")) {
        output["tolerance_squared"] = 1.0e-6;
    }
    return output;
}

}  // namespace ppln::constraints::json_io

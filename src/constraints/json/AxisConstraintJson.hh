#pragma once

#include "ConstraintJsonCommon.hh"

namespace ppln::constraints::json_io {

inline bool is_axis_constraint(const std::string &type) {
    return type == "axis_alignment" || type == "bimanual_axis";
}

inline void append_axis_constraint(Json &output, const Json &entry) {
    if (output.contains("bimanual_axis")) {
        throw std::invalid_argument("only one axis-alignment constraint is supported");
    }
    const auto &local_axis = entry.at("local_axis");
    const auto &target_world_axis = entry.at("target_world_axis");
    require_number_array(local_axis, 3, "axis_alignment.local_axis");
    require_number_array(
        target_world_axis, 3, "axis_alignment.target_world_axis"
    );
    output["bimanual_axis"] = {
        {"local_axis", local_axis},
        {"target_world_axis", target_world_axis}
    };
}

}  // namespace ppln::constraints::json_io

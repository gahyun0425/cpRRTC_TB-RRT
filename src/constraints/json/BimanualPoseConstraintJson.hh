#pragma once

#include "ConstraintJsonCommon.hh"

namespace ppln::constraints::json_io {

inline bool is_bimanual_pose_constraint(const std::string &type) {
    return type == "relative_pose" || type == "bimanual_pose";
}

inline void append_bimanual_pose_constraint(Json &output, const Json &entry) {
    if (output.contains("bimanual")) {
        throw std::invalid_argument("only one bimanual pose constraint is supported");
    }
    output["bimanual"] = {
        {"target", pose7_from_json(entry.at("target"), "bimanual_pose.target")}
    };
}

}  // namespace ppln::constraints::json_io

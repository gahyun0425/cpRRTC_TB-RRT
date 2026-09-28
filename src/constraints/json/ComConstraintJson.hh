#pragma once

#include "ConstraintJsonCommon.hh"

namespace ppln::constraints::json_io {

inline bool is_com_constraint(const std::string &type) {
    return type == "com_support" || type == "center_of_mass";
}

inline void append_com_constraint(Json &output, const Json &entry) {
    if (output.contains("com")) {
        throw std::invalid_argument("only one center-of-mass constraint is supported");
    }
    Json center_of_mass = entry;
    center_of_mass.erase("type");
    center_of_mass.erase("support_frames");
    output["com"] = std::move(center_of_mass);
}

}  // namespace ppln::constraints::json_io

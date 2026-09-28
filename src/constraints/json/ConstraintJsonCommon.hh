#pragma once

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>

namespace ppln::constraints::json_io {

using Json = nlohmann::json;

inline void require_number_array(
    const Json &value,
    const std::size_t expected_size,
    const std::string &context
) {
    if (!value.is_array() || value.size() != expected_size) {
        throw std::invalid_argument(
            context + " must contain exactly "
            + std::to_string(expected_size) + " numbers"
        );
    }
    for (const auto &component : value) {
        if (!component.is_number()) {
            throw std::invalid_argument(context + " must contain only numbers");
        }
    }
}

inline Json pose7_from_json(const Json &value, const std::string &context) {
    if (value.is_array()) {
        require_number_array(value, 7, context);
        return value;
    }
    if (!value.is_object()) {
        throw std::invalid_argument(
            context + " must be a 7-value pose or a pose object"
        );
    }

    const auto &quaternion = value.at("quaternion_wxyz");
    const auto &position = value.at("position");
    require_number_array(quaternion, 4, context + ".quaternion_wxyz");
    require_number_array(position, 3, context + ".position");

    Json pose = Json::array();
    for (const auto &component : quaternion) {
        pose.push_back(component);
    }
    for (const auto &component : position) {
        pose.push_back(component);
    }
    return pose;
}

inline std::string constraint_type(const Json &entry) {
    if (!entry.is_object()) {
        throw std::invalid_argument("each constraint entry must be an object");
    }
    if (!entry.contains("type") || !entry.at("type").is_string()) {
        throw std::invalid_argument("each constraint entry requires a string type");
    }
    return entry.at("type").get<std::string>();
}

}  // namespace ppln::constraints::json_io

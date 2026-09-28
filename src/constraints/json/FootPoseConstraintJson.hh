#pragma once

#include "ConstraintJsonCommon.hh"

namespace ppln::constraints::json_io {

inline bool is_foot_pose_constraint(const std::string &type) {
    return type == "fixed_frame_pose" || type == "fixed_pose" ||
        type == "foot_pose";
}

inline int foot_index_from_frame(const std::string &frame) {
    if (frame == "left_foot") {
        return 0;
    }
    if (frame == "right_foot") {
        return 1;
    }
    throw std::invalid_argument(
        "the compiled fixed-pose backend currently supports the semantic "
        "frames left_foot and right_foot; received " + frame
    );
}

inline void append_foot_pose_constraint(Json &output, const Json &entry) {
    if (!entry.contains("frame") || !entry.at("frame").is_string()) {
        throw std::invalid_argument("fixed_frame_pose requires a string frame");
    }
    const std::string frame = entry.at("frame").get<std::string>();
    const int foot = foot_index_from_frame(frame);
    const Json target = pose7_from_json(
        entry.at("target"),
        "fixed_frame_pose(" + frame + ").target"
    );
    const Json reference = entry.contains("reference")
        ? pose7_from_json(
            entry.at("reference"),
            "fixed_frame_pose(" + frame + ").reference"
        )
        : Json::array({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

    if (!output.contains("feet")) {
        output["feet"] = {
            {"reference", Json::array({nullptr, nullptr})},
            {"target", Json::array({nullptr, nullptr})}
        };
    }
    if (!output.at("feet").at("target").at(foot).is_null()) {
        throw std::invalid_argument(
            "duplicate fixed_frame_pose for semantic frame " + frame
        );
    }
    output["feet"]["reference"][foot] = reference;
    output["feet"]["target"][foot] = target;
}

inline void validate_foot_pose_constraints(const Json &output) {
    if (!output.contains("feet")) {
        return;
    }
    for (int foot = 0; foot < 2; ++foot) {
        if (output.at("feet").at("target").at(foot).is_null()) {
            throw std::invalid_argument(
                std::string("fixed-frame constraints require both feet; missing ")
                + (foot == 0 ? "left_foot" : "right_foot")
            );
        }
    }
}

}  // namespace ppln::constraints::json_io

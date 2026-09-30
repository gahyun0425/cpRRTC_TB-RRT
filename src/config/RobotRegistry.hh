#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ppln::config {

struct RobotDescriptor {
    std::string_view name;
    int dimension;
    std::string_view default_problem_file;
    bool evaluate_visualization_supported;
};

inline constexpr std::array<RobotDescriptor, 6> robot_descriptors = {{
    {
        "franka_single",
        7,
        "scripts/franka_single_problems.json",
        true,
    },
    {
        "franka",
        14,
        "scripts/franka_problems.json",
        true,
    },
    {
        "ffw_sg2",
        15,
        "scripts/ffw_sg2_problems.json",
        true,
    },
    {
        "ffw_sg2_mobility",
        18,
        "scripts/ffw_sg2_mobility_problems.json",
        true,
    },
    {
        "g1",
        35,
        "scripts/g1_problems.json",
        true,
    },
    {
        "igris_c",
        35,
        "scripts/igris_c_problems.json",
        false,
    },
}};

inline constexpr const RobotDescriptor *find_robot_descriptor(
    const std::string_view name
) noexcept {
    for (const auto &descriptor : robot_descriptors) {
        if (descriptor.name == name) {
            return &descriptor;
        }
    }
    return nullptr;
}

inline const RobotDescriptor &require_robot_descriptor(
    const std::string_view name
) {
    const auto *descriptor = find_robot_descriptor(name);
    if (descriptor == nullptr) {
        throw std::invalid_argument(
            "unsupported robot model: " + std::string(name)
        );
    }
    return *descriptor;
}

inline constexpr bool is_supported_robot(const std::string_view name) noexcept {
    return find_robot_descriptor(name) != nullptr;
}

inline constexpr int compiled_robot_dimension(
    const std::string_view name
) noexcept {
    const auto *descriptor = find_robot_descriptor(name);
    return descriptor == nullptr ? -1 : descriptor->dimension;
}

inline std::string default_problem_file(const std::string_view name) {
    return std::string(require_robot_descriptor(name).default_problem_file);
}

inline std::vector<std::string> result_joint_names(
    const std::string_view robot_name,
    const int dimension
) {
    if (robot_name == "ffw_sg2_mobility") {
        return {
            "base_x", "base_y", "base_yaw", "lift_joint",
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
            "arm_l_joint4", "arm_l_joint5", "arm_l_joint6",
            "arm_l_joint7", "arm_r_joint1", "arm_r_joint2",
            "arm_r_joint3", "arm_r_joint4", "arm_r_joint5",
            "arm_r_joint6", "arm_r_joint7",
        };
    }
    if (robot_name == "ffw_sg2") {
        return {
            "lift_joint",
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
            "arm_l_joint4", "arm_l_joint5", "arm_l_joint6",
            "arm_l_joint7", "arm_r_joint1", "arm_r_joint2",
            "arm_r_joint3", "arm_r_joint4", "arm_r_joint5",
            "arm_r_joint6", "arm_r_joint7",
        };
    }

    std::vector<std::string> names;
    names.reserve(dimension);
    for (int index = 0; index < dimension; ++index) {
        names.push_back("q" + std::to_string(index));
    }
    return names;
}

}  // namespace ppln::config

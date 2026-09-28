#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "src/planning/pRRTC_settings.hh"

namespace ffw_sg2_attached_object_collision {

using json = nlohmann::json;

constexpr int kFfwSg2FixedFineSphereCount = 124;
constexpr int kFfwSg2FixedApproxSphereCount = 27;
constexpr int kFfwSg2MobilityFineSphereCount = 125;
constexpr int kFfwSg2MobilityApproxSphereCount = 28;

struct LinkSphereRange {
    const char *name;
    int fine_begin;
    int fine_end;
    int approx_index;
};

inline constexpr LinkSphereRange kLinkSphereRanges[] = {
    {"lift_link", 0, 8, 0},
    {"arm_base_link", 9, 14, 1},
    {"head_link2", 116, 123, 26},
    {"arm_l_link1", 15, 24, 2},
    {"arm_l_link2", 25, 34, 3},
    {"arm_l_link3", 35, 44, 4},
    {"arm_l_link4", 45, 54, 5},
    {"arm_l_link5", 55, 64, 6},
    {"arm_l_link6", 65, 69, 7},
    {"arm_l_link7", 70, 76, 8},
    {"arm_r_link1", 83, 85, 14},
    {"arm_r_link2", 86, 88, 15},
    {"arm_r_link3", 89, 91, 16},
    {"arm_r_link4", 92, 95, 17},
    {"arm_r_link5", 96, 97, 18},
    {"arm_r_link6", 98, 102, 19},
    {"arm_r_link7", 103, 109, 20},
    {"gripper_l_rh_p12_rn_base", 77, 78, 9},
    {"gripper_l_rh_p12_rn_r1", 79, 79, 10},
    {"gripper_l_rh_p12_rn_r2", 80, 80, 11},
    {"gripper_l_rh_p12_rn_l1", 81, 81, 12},
    {"gripper_l_rh_p12_rn_l2", 82, 82, 13},
    {"gripper_r_rh_p12_rn_base", 110, 111, 21},
    {"gripper_r_rh_p12_rn_r1", 112, 112, 22},
    {"gripper_r_rh_p12_rn_r2", 113, 113, 23},
    {"gripper_r_rh_p12_rn_l1", 114, 114, 24},
    {"gripper_r_rh_p12_rn_l2", 115, 115, 25},
    {"base_link", 124, 124, 27},
    {"ffw_sg2_mobility_base", 124, 124, 27},
};

inline constexpr const char *kDefaultContactLinks[] = {
    "gripper_l_rh_p12_rn_base",
    "gripper_l_rh_p12_rn_r1",
    "gripper_l_rh_p12_rn_r2",
    "gripper_l_rh_p12_rn_l1",
    "gripper_l_rh_p12_rn_l2",
    "gripper_r_rh_p12_rn_base",
    "gripper_r_rh_p12_rn_r1",
    "gripper_r_rh_p12_rn_r2",
    "gripper_r_rh_p12_rn_l1",
    "gripper_r_rh_p12_rn_l2",
};

inline float parse_float(const json &value, const std::string &field_name) {
    if (!value.is_number()) {
        throw std::invalid_argument(field_name + " must be a number");
    }
    return static_cast<float>(value.get<double>());
}

inline int parse_positive_int(
    const json &value,
    const std::string &field_name
) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument(field_name + " must be an integer");
    }
    const int parsed = value.get<int>();
    if (parsed <= 0) {
        throw std::invalid_argument(field_name + " must be > 0");
    }
    return parsed;
}

inline std::array<float, 3> parse_vec3(
    const json &value,
    const std::string &field_name
) {
    if (!value.is_array() || value.size() != 3) {
        throw std::invalid_argument(field_name + " must be [x, y, z]");
    }
    return {
        parse_float(value[0], field_name + "[0]"),
        parse_float(value[1], field_name + "[1]"),
        parse_float(value[2], field_name + "[2]"),
    };
}

inline std::array<int, 3> parse_positive_int_vec3(
    const json &value,
    const std::string &field_name
) {
    if (!value.is_array() || value.size() != 3) {
        throw std::invalid_argument(field_name + " must be [x, y, z]");
    }
    return {
        parse_positive_int(value[0], field_name + "[0]"),
        parse_positive_int(value[1], field_name + "[1]"),
        parse_positive_int(value[2], field_name + "[2]"),
    };
}

inline void require_positive_vec3(
    const std::array<float, 3> &value,
    const std::string &field_name
) {
    if (value[0] <= 0.0f || value[1] <= 0.0f || value[2] <= 0.0f) {
        throw std::invalid_argument(field_name + " entries must be > 0");
    }
}

inline void add_unique_int(
    int *values,
    int &count,
    int capacity,
    int value,
    const std::string &field_name
) {
    for (int i = 0; i < count; i++) {
        if (values[i] == value) {
            return;
        }
    }
    if (count >= capacity) {
        throw std::invalid_argument(field_name + " exceeds configured capacity");
    }
    values[count++] = value;
}

inline void append_sphere(
    FfwSg2AttachedObjectCollisionSpec &spec,
    const std::array<float, 3> &center,
    float radius,
    const std::string &field_name
) {
    if (radius <= 0.0f) {
        throw std::invalid_argument(field_name + " radius must be > 0");
    }
    if (spec.sphere_count >= FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES) {
        throw std::invalid_argument(
            field_name + " exceeds configured sphere capacity"
        );
    }

    spec.spheres[spec.sphere_count][0] = center[0];
    spec.spheres[spec.sphere_count][1] = center[1];
    spec.spheres[spec.sphere_count][2] = center[2];
    spec.spheres[spec.sphere_count][3] = radius;
    spec.sphere_count++;
}

inline void add_ignored_fine_sphere(
    FfwSg2AttachedObjectCollisionSpec &spec,
    int sphere,
    int robot_sphere_count
) {
    if (sphere < 0 || sphere >= robot_sphere_count) {
        throw std::invalid_argument(
            "ignored fine robot sphere index is out of range"
        );
    }
    add_unique_int(
        spec.ignored_robot_spheres,
        spec.ignored_robot_sphere_count,
        FFW_SG2_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_SPHERES,
        sphere,
        "ignored_robot_spheres"
    );
}

inline void add_ignored_approx_sphere(
    FfwSg2AttachedObjectCollisionSpec &spec,
    int sphere,
    int robot_sphere_count
) {
    if (sphere < 0 || sphere >= robot_sphere_count) {
        throw std::invalid_argument(
            "ignored approximate robot sphere index is out of range"
        );
    }
    add_unique_int(
        spec.ignored_robot_approx_spheres,
        spec.ignored_robot_approx_sphere_count,
        FFW_SG2_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_APPROX_SPHERES,
        sphere,
        "ignored_robot_approx_spheres"
    );
}

inline const LinkSphereRange *find_link_range(const std::string &link_name) {
    for (const auto &range : kLinkSphereRanges) {
        if (link_name == range.name) {
            return &range;
        }
    }
    return nullptr;
}

inline void add_contact_link(
    FfwSg2AttachedObjectCollisionSpec &spec,
    const std::string &link_name,
    int fine_sphere_count,
    int approximate_sphere_count
) {
    const LinkSphereRange *range = find_link_range(link_name);
    if (range == nullptr) {
        throw std::invalid_argument(
            "unknown FFW-SG2 contact link: " + link_name
        );
    }
    for (int sphere = range->fine_begin; sphere <= range->fine_end; sphere++) {
        add_ignored_fine_sphere(spec, sphere, fine_sphere_count);
    }
    add_ignored_approx_sphere(
        spec,
        range->approx_index,
        approximate_sphere_count
    );
}

inline void add_default_contact_links(
    FfwSg2AttachedObjectCollisionSpec &spec,
    int fine_sphere_count,
    int approximate_sphere_count
) {
    for (const char *link_name : kDefaultContactLinks) {
        add_contact_link(
            spec,
            link_name,
            fine_sphere_count,
            approximate_sphere_count
        );
    }
}

inline void parse_contact_links(
    const json &object,
    FfwSg2AttachedObjectCollisionSpec &spec,
    int fine_sphere_count,
    int approximate_sphere_count
) {
    if (!object.contains("contact_links")) {
        add_default_contact_links(
            spec,
            fine_sphere_count,
            approximate_sphere_count
        );
        return;
    }

    const auto &links = object["contact_links"];
    if (!links.is_array()) {
        throw std::invalid_argument("contact_links must be an array");
    }
    for (const auto &entry : links) {
        if (!entry.is_string()) {
            throw std::invalid_argument("contact_links entries must be strings");
        }
        add_contact_link(
            spec,
            entry.get<std::string>(),
            fine_sphere_count,
            approximate_sphere_count
        );
    }
}

inline void parse_manual_ignored_spheres(
    const json &object,
    FfwSg2AttachedObjectCollisionSpec &spec,
    int fine_sphere_count,
    int approximate_sphere_count
) {
    for (const char *key : {"ignored_robot_spheres", "ignore_robot_spheres"}) {
        if (!object.contains(key)) {
            continue;
        }
        const auto &values = object[key];
        if (!values.is_array()) {
            throw std::invalid_argument(std::string(key) + " must be an array");
        }
        for (const auto &entry : values) {
            if (!entry.is_number_integer()) {
                throw std::invalid_argument(
                    std::string(key) + " entries must be integers"
                );
            }
            add_ignored_fine_sphere(
                spec,
                entry.get<int>(),
                fine_sphere_count
            );
        }
    }

    for (
        const char *key :
        {"ignored_robot_approx_spheres", "ignore_robot_approx_spheres"}
    ) {
        if (!object.contains(key)) {
            continue;
        }
        const auto &values = object[key];
        if (!values.is_array()) {
            throw std::invalid_argument(std::string(key) + " must be an array");
        }
        for (const auto &entry : values) {
            if (!entry.is_number_integer()) {
                throw std::invalid_argument(
                    std::string(key) + " entries must be integers"
                );
            }
            add_ignored_approx_sphere(
                spec,
                entry.get<int>(),
                approximate_sphere_count
            );
        }
    }
}

inline void parse_sphere(
    const json &entry,
    FfwSg2AttachedObjectCollisionSpec &spec
) {
    std::array<float, 3> center{};
    float radius = 0.0f;

    if (entry.is_array()) {
        if (entry.size() != 4) {
            throw std::invalid_argument(
                "attached object sphere array must be [x, y, z, radius]"
            );
        }
        center = {
            parse_float(entry[0], "attached_object_collision.spheres[][0]"),
            parse_float(entry[1], "attached_object_collision.spheres[][1]"),
            parse_float(entry[2], "attached_object_collision.spheres[][2]"),
        };
        radius = parse_float(
            entry[3],
            "attached_object_collision.spheres[][3]"
        );
    } else if (entry.is_object()) {
        const char *center_key =
            entry.contains("center") ? "center" : "position";
        if (!entry.contains(center_key) || !entry.contains("radius")) {
            throw std::invalid_argument(
                "attached object sphere object needs center/position and radius"
            );
        }
        center = parse_vec3(entry[center_key], "attached object sphere center");
        radius = parse_float(entry["radius"], "attached object sphere radius");
    } else {
        throw std::invalid_argument(
            "attached object spheres entries must be arrays or objects"
        );
    }

    append_sphere(spec, center, radius, "attached object sphere");
}

inline void parse_box_sphere_grid(
    const json &object,
    FfwSg2AttachedObjectCollisionSpec &spec
) {
    if (!object.contains("box_sphere_grid")) {
        return;
    }

    const auto &grid = object["box_sphere_grid"];
    if (!grid.is_object()) {
        throw std::invalid_argument(
            "attached_object_collision.box_sphere_grid must be an object"
        );
    }
    if (!grid.contains("half_extents")) {
        throw std::invalid_argument(
            "attached_object_collision.box_sphere_grid.half_extents is required"
        );
    }
    if (!grid.contains("counts")) {
        throw std::invalid_argument(
            "attached_object_collision.box_sphere_grid.counts is required"
        );
    }

    const std::array<float, 3> center =
        grid.contains("center") ?
            parse_vec3(
                grid["center"],
                "attached_object_collision.box_sphere_grid.center"
            ) :
            std::array<float, 3>{0.0f, 0.0f, 0.0f};
    const std::array<float, 3> half_extents = parse_vec3(
        grid["half_extents"],
        "attached_object_collision.box_sphere_grid.half_extents"
    );
    require_positive_vec3(
        half_extents,
        "attached_object_collision.box_sphere_grid.half_extents"
    );
    const std::array<int, 3> counts = parse_positive_int_vec3(
        grid["counts"],
        "attached_object_collision.box_sphere_grid.counts"
    );
    const float radius_padding =
        grid.contains("radius_padding") ?
            parse_float(
                grid["radius_padding"],
                "attached_object_collision.box_sphere_grid.radius_padding"
            ) :
            0.0f;
    if (radius_padding < 0.0f) {
        throw std::invalid_argument(
            "attached_object_collision.box_sphere_grid.radius_padding must be >= 0"
        );
    }

    const std::size_t generated_count =
        static_cast<std::size_t>(counts[0]) *
        static_cast<std::size_t>(counts[1]) *
        static_cast<std::size_t>(counts[2]);
    const std::size_t remaining_capacity =
        FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES -
        static_cast<std::size_t>(spec.sphere_count);
    if (generated_count > remaining_capacity) {
        throw std::invalid_argument(
            "attached_object_collision.box_sphere_grid exceeds configured sphere capacity"
        );
    }

    const float cell_x = half_extents[0] / counts[0];
    const float cell_y = half_extents[1] / counts[1];
    const float cell_z = half_extents[2] / counts[2];
    const float radius =
        std::sqrt(cell_x * cell_x + cell_y * cell_y + cell_z * cell_z) +
        radius_padding;

    for (int z = 0; z < counts[2]; z++) {
        for (int y = 0; y < counts[1]; y++) {
            for (int x = 0; x < counts[0]; x++) {
                const std::array<float, 3> sphere_center = {
                    center[0] - half_extents[0] + (2.0f * x + 1.0f) * cell_x,
                    center[1] - half_extents[1] + (2.0f * y + 1.0f) * cell_y,
                    center[2] - half_extents[2] + (2.0f * z + 1.0f) * cell_z,
                };
                append_sphere(
                    spec,
                    sphere_center,
                    radius,
                    "attached_object_collision.box_sphere_grid"
                );
            }
        }
    }
}

inline void apply_from_problem(
    const json &problem,
    pRRTC_settings &settings,
    int fine_sphere_count = kFfwSg2MobilityFineSphereCount,
    int approximate_sphere_count = kFfwSg2MobilityApproxSphereCount
) {
    auto &spec = settings.ffw_sg2_attached_object_collision;
    spec = {};

    if (!problem.contains("attached_object_collision")) {
        return;
    }

    const auto &object = problem["attached_object_collision"];
    if (!object.is_object()) {
        throw std::invalid_argument(
            "attached_object_collision must be an object"
        );
    }

    spec.enabled = object.value("enabled", true);
    if (!spec.enabled) {
        return;
    }

    spec.world_offset[0] = 0.1f;
    spec.world_offset[1] = 0.0f;
    spec.world_offset[2] = 0.0f;
    if (object.contains("world_offset")) {
        const auto offset = parse_vec3(
            object["world_offset"],
            "attached_object_collision.world_offset"
        );
        spec.world_offset[0] = offset[0];
        spec.world_offset[1] = offset[1];
        spec.world_offset[2] = offset[2];
    }

    if (object.contains("spheres")) {
        if (!object["spheres"].is_array()) {
            throw std::invalid_argument(
                "attached_object_collision.spheres must be an array"
            );
        }
        const auto &spheres = object["spheres"];
        if (
            spheres.size() >
            static_cast<std::size_t>(FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES)
        ) {
            throw std::invalid_argument(
                "attached_object_collision.spheres exceeds configured capacity"
            );
        }

        for (const auto &sphere : spheres) {
            parse_sphere(sphere, spec);
        }
    }

    parse_box_sphere_grid(object, spec);

    if (spec.sphere_count <= 0) {
        throw std::invalid_argument(
            "attached_object_collision needs spheres or box_sphere_grid"
        );
    }

    parse_contact_links(
        object,
        spec,
        fine_sphere_count,
        approximate_sphere_count
    );
    parse_manual_ignored_spheres(
        object,
        spec,
        fine_sphere_count,
        approximate_sphere_count
    );
}

} // namespace ffw_sg2_attached_object_collision

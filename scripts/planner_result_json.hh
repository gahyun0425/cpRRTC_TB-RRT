#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "src/collision/environment.hh"
#include "src/planning/AORRTC.hh"
#include "src/planning/Planners.hh"
#include "src/planning/pRRTC_settings.hh"

namespace planner_result_json {

using json = nlohmann::json;

inline std::vector<std::string> joint_names_for_robot(
    const std::string &robot_name,
    int dimension
) {
    if (robot_name == "ffw_sg2") {
        return {
            "lift_joint",
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
            "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
            "arm_r_joint5", "arm_r_joint6", "arm_r_joint7",
        };
    }
    if (robot_name == "ffw_sg2_single") {
        return {
            "lift_joint",
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
            "arm_r_joint5", "arm_r_joint6", "arm_r_joint7",
        };
    }

    std::vector<std::string> names;
    names.reserve(dimension);
    for (int index = 0; index < dimension; index++) {
        names.push_back("q" + std::to_string(index));
    }
    return names;
}

template <typename Robot>
json config_to_json(const typename Robot::Configuration &config) {
    json output = json::array();
    for (float value : config) {
        output.push_back(value);
    }
    return output;
}

template <typename Robot>
json path_to_json(const std::vector<typename Robot::Configuration> &path) {
    json output = json::array();
    for (const auto &configuration : path) {
        output.push_back(config_to_json<Robot>(configuration));
    }
    return output;
}

template <typename Robot>
float max_abs_diff(
    const typename Robot::Configuration &left,
    const typename Robot::Configuration &right
) {
    float difference = 0.0f;
    for (int index = 0; index < Robot::dimension; index++) {
        difference = std::max(
            difference,
            std::fabs(left[index] - right[index])
        );
    }
    return difference;
}

template <typename Robot>
float nearest_goal_diff(
    const typename Robot::Configuration &configuration,
    const std::vector<typename Robot::Configuration> &goals
) {
    float difference = std::numeric_limits<float>::infinity();
    for (const auto &goal : goals) {
        difference = std::min(
            difference,
            max_abs_diff<Robot>(configuration, goal)
        );
    }
    return difference;
}

template <typename Robot>
std::string infer_path_orientation(
    const PlannerResult<Robot> &result,
    const typename Robot::Configuration &start,
    const std::vector<typename Robot::Configuration> &goals
) {
    if (result.path.empty() || goals.empty()) {
        return "unknown";
    }
    constexpr float tolerance = 5.0e-3f;
    const bool first_is_start =
        max_abs_diff<Robot>(result.path.front(), start) <= tolerance;
    const bool last_is_start =
        max_abs_diff<Robot>(result.path.back(), start) <= tolerance;
    const bool first_is_goal =
        nearest_goal_diff<Robot>(result.path.front(), goals) <= tolerance;
    const bool last_is_goal =
        nearest_goal_diff<Robot>(result.path.back(), goals) <= tolerance;
    if (first_is_start && last_is_goal) {
        return "start_to_goal";
    }
    if (first_is_goal && last_is_start) {
        return "goal_to_start";
    }
    return "unknown";
}

template <typename Robot>
json start_to_goal_path_json(
    const PlannerResult<Robot> &result,
    const std::string &orientation
) {
    if (orientation == "goal_to_start") {
        json output = json::array();
        for (auto iterator = result.path.rbegin();
             iterator != result.path.rend();
             ++iterator) {
            output.push_back(config_to_json<Robot>(*iterator));
        }
        return output;
    }
    return path_to_json<Robot>(result.path);
}

inline std::string tree_name(int tree_id) {
    return tree_id == 0 ? "start" : "goal";
}

template <typename Robot>
int solution_order_for_node(
    const PlannerResult<Robot> &result,
    int tree_id,
    int node_index
) {
    for (int order = 0;
         order < static_cast<int>(result.solution_trace.size());
         order++) {
        if (result.solution_trace[order][0] == tree_id
            && result.solution_trace[order][1] == node_index) {
            return order;
        }
    }
    return -1;
}

template <typename Robot>
json tree_trace_to_json(const PlannerResult<Robot> &result) {
    json trees = json::array();
    for (int tree = 0; tree < 2; tree++) {
        json nodes = json::array();
        for (int index = 0;
             index < static_cast<int>(result.tree_nodes[tree].size());
             index++) {
            const bool ready =
                index < static_cast<int>(result.tree_node_ready[tree].size())
                && result.tree_node_ready[tree][index] != 0;
            if (!ready) {
                continue;
            }
            const int parent_index =
                index < static_cast<int>(result.tree_parents[tree].size())
                ? result.tree_parents[tree][index]
                : -1;
            const int solution_order = solution_order_for_node(
                result,
                tree,
                index
            );
            nodes.push_back({
                {"idx", index},
                {"parent_idx", parent_index},
                {"ready", true},
                {"solution", solution_order >= 0},
                {"solution_order", solution_order},
                {"q", config_to_json<Robot>(result.tree_nodes[tree][index])},
            });
        }
        trees.push_back({
            {"tree_id", tree},
            {"name", tree_name(tree)},
            {"allocated_size", result.tree_nodes[tree].size()},
            {"ready_size", nodes.size()},
            {"nodes", nodes},
        });
    }

    json solution_order = json::array();
    for (int order = 0;
         order < static_cast<int>(result.solution_trace.size());
         order++) {
        const int tree_id = result.solution_trace[order][0];
        solution_order.push_back({
            {"order", order},
            {"tree_id", tree_id},
            {"tree", tree_name(tree_id)},
            {"idx", result.solution_trace[order][1]},
        });
    }

    json connection = nullptr;
    if (result.connection_tree_id >= 0
        && result.connection_other_tree_id >= 0) {
        connection = {
            {"source_tree_id", result.connection_tree_id},
            {"source_tree", tree_name(result.connection_tree_id)},
            {"source_idx", result.connection_node_idx},
            {"target_tree_id", result.connection_other_tree_id},
            {"target_tree", tree_name(result.connection_other_tree_id)},
            {"target_idx", result.connection_other_node_idx},
        };
    }

    return {
        {"trees", trees},
        {"connection", connection},
        {"solution_order", solution_order},
    };
}

template <typename Robot>
json solution_history_to_json(const AORRTCResult<Robot> &result) {
    json output = json::array();
    for (std::size_t index = 0; index < result.solution_history.size(); index++) {
        const auto &update = result.solution_history[index];
        json solution_order = json::array();
        for (int order = 0;
             order < static_cast<int>(update.solution_trace.size());
             order++) {
            const int tree_id = update.solution_trace[order][0];
            solution_order.push_back({
                {"order", order},
                {"tree_id", tree_id},
                {"tree", tree_name(tree_id)},
                {"idx", update.solution_trace[order][1]},
            });
        }

        output.push_back({
            {"update_index", update.update_index},
            {"candidate_idx", update.update_index},
            {"iteration", update.iteration},
            {"iter", update.iteration},
            {"cost", update.cost},
            {"accepted", true},
            {"final", index + 1 == result.solution_history.size()},
            {"connection", {
                {"source_tree_id", update.source_tree_id},
                {"source_tree", tree_name(update.source_tree_id)},
                {"source_idx", update.source_node_idx},
                {"target_tree_id", update.target_tree_id},
                {"target_tree", tree_name(update.target_tree_id)},
                {"target_idx", update.target_node_idx},
            }},
            {"solution_order", solution_order},
            {"path_waypoint_count", update.path_start_to_goal.size()},
            {"path_edge_count", update.path_start_to_goal.empty()
                ? 0
                : update.path_start_to_goal.size() - 1},
            {"path_start_to_goal", path_to_json<Robot>(
                update.path_start_to_goal
            )},
        });
    }
    return output;
}

inline json settings_to_json(const pRRTC_settings &settings) {
    json output = {
        {"max_samples", settings.max_samples},
        {"max_tangent_spaces", settings.max_tangent_spaces},
        {"max_iters", settings.max_iters},
        {"num_new_configs", settings.num_new_configs},
        {"granularity", settings.granularity},
        {"range", settings.range},
        {"lift_distance_weight", settings.lift_distance_weight},
        {"balance", settings.balance},
        {"tree_ratio", settings.tree_ratio},
        {"dynamic_domain", settings.dynamic_domain},
        {"trace_trees", settings.trace_trees},
        {"dd_alpha", settings.dd_alpha},
        {"dd_radius", settings.dd_radius},
        {"dd_min_radius", settings.dd_min_radius},
    };
    return output;
}

inline json settings_to_json(const AORRTC_settings &settings) {
    json output = settings_to_json(
        static_cast<const pRRTC_settings &>(settings)
    );
    if (settings.aorrtc) {
        output["aorrtc"] = true;
        output["time_limit_sec"] = settings.time_limit_sec;
        output["aorrtc_config_weight"] = settings.aorrtc_config_weight;
        output["aorrtc_cost_weight"] = settings.aorrtc_cost_weight;
        output["cost_improvement_epsilon"] =
            settings.cost_improvement_epsilon;
    }
    return output;
}

inline json environment_summary_json(
    const ppln::collision::Environment<float> &environment
) {
    return {
        {"num_spheres", environment.num_spheres},
        {"num_capsules", environment.num_capsules},
        {"num_cuboids", environment.num_cuboids},
    };
}

template <typename Robot>
json result_to_json(
    const AORRTCResult<Robot> &result,
    const AORRTC_settings &settings,
    const ppln::collision::Environment<float> &environment,
    const typename Robot::Configuration &start,
    const std::vector<typename Robot::Configuration> &goals,
    const std::string &robot_name,
    const std::string &problem_name,
    int problem_index
) {
    const std::string orientation = infer_path_orientation(
        result,
        start,
        goals
    );
    json goal_json = json::array();
    for (const auto &goal : goals) {
        goal_json.push_back(config_to_json<Robot>(goal));
    }

    json payload = {
        {"format", "pRRTC_result_v1"},
        {"planner", settings.aorrtc ? "pRRTC-AORRTC" : "pRRTC"},
        {"robot", robot_name},
        {"problem_name", problem_name},
        {"problem_idx", problem_index},
        {"dimension", Robot::dimension},
        {"joint_names", joint_names_for_robot(robot_name, Robot::dimension)},
        {"solved", result.solved},
        {"cost", result.cost},
        {"path_length", result.path_length},
        {"path_waypoint_count", result.path.size()},
        {"path_edge_count", result.path.empty() ? 0 : result.path.size() - 1},
        {"start_tree_size", result.start_tree_size},
        {"goal_tree_size", result.goal_tree_size},
        {"iters", result.iters},
        {"wall_ns", result.wall_ns},
        {"kernel_ns", result.kernel_ns},
        {"copy_ns", result.copy_ns},
        {"path_orientation", orientation},
        {"path", path_to_json<Robot>(result.path)},
        {"path_start_to_goal", start_to_goal_path_json<Robot>(result, orientation)},
        {"start", config_to_json<Robot>(start)},
        {"goals", goal_json},
        {"settings", settings_to_json(settings)},
        {"environment", environment_summary_json(environment)},
    };
    if (settings.aorrtc) {
        payload["initial_cost"] = result.initial_cost;
        payload["initial_solution_ns"] = result.initial_solution_ns;
        payload["best_solution_ns"] = result.best_solution_ns;
        payload["solution_updates"] = result.solution_updates;
        if (settings.trace_trees) {
            payload["solution_history"] =
                solution_history_to_json<Robot>(result);
            payload["solution_history_overflow"] =
                result.solution_history_overflow;
        }
    }
    if (!result.tree_nodes[0].empty() || !result.tree_nodes[1].empty()) {
        payload["tree_trace"] = tree_trace_to_json<Robot>(result);
    }
    return payload;
}

inline void write_json_file(const json &payload, const std::string &path) {
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("failed to create planner result JSON: " + path);
    }
    output << payload.dump(2) << '\n';
}

}  // namespace planner_result_json

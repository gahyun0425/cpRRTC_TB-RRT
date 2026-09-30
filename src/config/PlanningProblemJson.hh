#pragma once

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/constraints/json/ConstraintSetJson.hh"
#include "src/config/RobotRegistry.hh"

namespace ppln::config {

using Json = nlohmann::json;

struct PlanningProblem {
    std::string robot_name;
    std::string problem_name;
    int problem_index = 1;
    Json data;
};

// Compatibility name retained while frontends migrate to the common model.
using SelectedPlanningProblem = PlanningProblem;

inline Json read_json_file(const std::string &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open problem file: " + path);
    }
    try {
        return Json::parse(input);
    } catch (const std::exception &error) {
        throw std::runtime_error(
            "failed to parse problem file " + path + ": " + error.what()
        );
    }
}

inline std::string robot_name_from_json(
    const Json &document,
    const std::string &fallback = {}
) {
    if (!document.contains("robot")) {
        if (!fallback.empty()) return fallback;
        throw std::invalid_argument(
            "a standalone planning JSON requires robot.model"
        );
    }
    const auto &robot = document.at("robot");
    std::string name;
    if (robot.is_string()) {
        name = robot.get<std::string>();
    } else if (robot.is_object()) {
        if (robot.contains("model")) {
            name = robot.at("model").get<std::string>();
        } else if (robot.contains("name")) {
            name = robot.at("name").get<std::string>();
        } else {
            throw std::invalid_argument("robot requires model");
        }
    } else {
        throw std::invalid_argument("robot must be a model name or an object");
    }
    if (!fallback.empty() && fallback != name) {
        throw std::invalid_argument(
            "command-line robot " + fallback
            + " does not match JSON robot " + name
        );
    }
    return name;
}

inline void validate_declared_dimension(
    const Json &document,
    const std::string &robot_name
) {
    int declared_dimension = -1;
    if (document.contains("robot_dim")) {
        declared_dimension = document.at("robot_dim").get<int>();
    }
    if (document.contains("robot") && document.at("robot").is_object()) {
        const auto &robot = document.at("robot");
        if (robot.contains("dimension")) {
            const int nested_dimension = robot.at("dimension").get<int>();
            if (declared_dimension >= 0 && declared_dimension != nested_dimension) {
                throw std::invalid_argument(
                    "robot_dim and robot.dimension do not match"
                );
            }
            declared_dimension = nested_dimension;
        }
    }

    const int compiled_dimension = compiled_robot_dimension(robot_name);
    if (compiled_dimension < 0) {
        throw std::invalid_argument("unsupported robot model: " + robot_name);
    }
    if (declared_dimension >= 0 && declared_dimension != compiled_dimension) {
        throw std::invalid_argument(
            "JSON dimension " + std::to_string(declared_dimension)
            + " does not match compiled " + robot_name + " dimension "
            + std::to_string(compiled_dimension)
        );
    }

    const Json *joints = nullptr;
    if (document.contains("joints")) {
        joints = &document.at("joints");
    } else if (document.contains("robot") &&
        document.at("robot").is_object() &&
        document.at("robot").contains("joints")) {
        joints = &document.at("robot").at("joints");
    }
    if (joints != nullptr &&
        (!joints->is_array() ||
         joints->size() != static_cast<std::size_t>(compiled_dimension))) {
        throw std::invalid_argument(
            "robot joints must contain exactly "
            + std::to_string(compiled_dimension) + " entries"
        );
    }
}

inline void copy_object_members(
    Json &destination,
    const Json &source,
    const std::string &context
) {
    if (!source.is_object()) {
        throw std::invalid_argument(context + " must be an object");
    }
    for (auto iterator = source.begin(); iterator != source.end(); ++iterator) {
        destination[iterator.key()] = iterator.value();
    }
}

inline Json normalize_problem(
    Json problem,
    const std::string &robot_name = {}
) {
    if (problem.contains("query")) {
        copy_object_members(problem, problem.at("query"), "query");
        problem.erase("query");
    }
    if (problem.contains("world")) {
        const Json world = problem.at("world");
        if (!world.is_object()) {
            throw std::invalid_argument("world must be an object");
        }
        for (const char *shape : {"sphere", "cylinder", "box"}) {
            problem[shape] = world.value(shape, Json::array());
        }
        problem.erase("world");
    }
    for (const char *shape : {"sphere", "cylinder", "box"}) {
        if (!problem.contains(shape)) {
            problem[shape] = Json::array();
        }
    }
    if (problem.contains("constraints")) {
        problem["constraints"] =
            constraints::json_io::normalize_constraint_set(
                problem.at("constraints")
            );
    }
    if (!problem.contains("valid")) {
        problem["valid"] = true;
    }
    // The existing G1 backend obtains its goal set through this field in both
    // orientation modes. Unified files may state the query only once.
    if (robot_name == "g1" &&
        !problem.contains("axis_endpoints") &&
        problem.contains("start") && problem.contains("goals")) {
        problem["axis_endpoints"] = {
            {"start", problem.at("start")},
            {"goals", problem.at("goals")}
        };
    }
    return problem;
}

inline void validate_configuration(
    const Json &configuration,
    const int dimension,
    const std::string &context
) {
    if (!configuration.is_array() ||
        configuration.size() != static_cast<std::size_t>(dimension)) {
        throw std::invalid_argument(
            context + " must contain exactly "
            + std::to_string(dimension) + " joint values"
        );
    }
    for (const auto &value : configuration) {
        if (!value.is_number()) {
            throw std::invalid_argument(context + " must contain only numbers");
        }
    }
}

inline void validate_query(Json &problem, const std::string &robot_name) {
    const int dimension = compiled_robot_dimension(robot_name);
    if (!problem.contains("start")) {
        throw std::invalid_argument("planning query is missing start");
    }
    if (!problem.contains("goals") || !problem.at("goals").is_array() ||
        problem.at("goals").empty()) {
        throw std::invalid_argument("planning query requires at least one goal");
    }
    validate_configuration(problem.at("start"), dimension, "query.start");
    for (std::size_t index = 0; index < problem.at("goals").size(); ++index) {
        validate_configuration(
            problem.at("goals").at(index),
            dimension,
            "query.goals[" + std::to_string(index) + "]"
        );
    }

    if (problem.contains("axis_endpoints")) {
        auto &endpoints = problem.at("axis_endpoints");
        validate_configuration(
            endpoints.at("start"), dimension,
            "axis_endpoints.start"
        );
        if (!endpoints.at("goals").is_array() || endpoints.at("goals").empty()) {
            throw std::invalid_argument(
                "axis_endpoints requires at least one goal"
            );
        }
        for (std::size_t index = 0; index < endpoints.at("goals").size(); ++index) {
            validate_configuration(
                endpoints.at("goals").at(index), dimension,
                "axis_endpoints.goals["
                    + std::to_string(index) + "]"
            );
        }
    }
}

inline SelectedPlanningProblem select_problem(
    const Json &document,
    const std::string &requested_robot = {},
    const std::string &requested_name = {},
    const int requested_index = 1
) {
    const std::string robot_name = robot_name_from_json(
        document, requested_robot
    );
    validate_declared_dimension(document, robot_name);

    SelectedPlanningProblem selected;
    selected.robot_name = robot_name;

    if (document.contains("problems")) {
        const auto &problems = document.at("problems");
        if (!problems.is_object() || problems.empty()) {
            throw std::invalid_argument("problems must be a non-empty object");
        }

        if (!requested_name.empty()) {
            selected.problem_name = requested_name;
            selected.problem_index = requested_index;
        } else if (document.contains("selected_problem")) {
            const auto &selection = document.at("selected_problem");
            if (selection.is_string()) {
                selected.problem_name = selection.get<std::string>();
                selected.problem_index = 1;
            } else {
                selected.problem_name = selection.at("name").get<std::string>();
                selected.problem_index = selection.value("index", 1);
            }
        } else if (problems.size() == 1) {
            selected.problem_name = problems.begin().key();
            selected.problem_index = 1;
        } else {
            throw std::invalid_argument(
                "standalone collection JSON requires selected_problem when it "
                "contains more than one problem group"
            );
        }

        if (!problems.contains(selected.problem_name) ||
            !problems.at(selected.problem_name).is_array() ||
            selected.problem_index < 1 ||
            selected.problem_index > static_cast<int>(
                problems.at(selected.problem_name).size()
            )) {
            throw std::invalid_argument(
                "unknown problem or problem index: " + selected.problem_name
                + " " + std::to_string(selected.problem_index)
            );
        }
        selected.data = problems.at(selected.problem_name).at(
            selected.problem_index - 1
        );
        if (document.contains("planner") &&
            !selected.data.contains("planner")) {
            selected.data["planner"] = document.at("planner");
        }
    } else {
        if (!requested_name.empty() || requested_index != 1) {
            throw std::invalid_argument(
                "a direct planning JSON contains exactly one query"
            );
        }
        selected.problem_name = document.value("name", "default");
        selected.problem_index = 1;
        selected.data = document;
        selected.data.erase("schema_version");
        selected.data.erase("robot");
        selected.data.erase("robot_dim");
        selected.data.erase("name");
        selected.data.erase("selected_problem");
    }

    selected.data = normalize_problem(
        std::move(selected.data), selected.robot_name
    );
    if (selected.data.value("valid", true)) {
        validate_query(selected.data, selected.robot_name);
    }
    return selected;
}

inline SelectedPlanningProblem load_selected_problem(
    const std::string &path,
    const std::string &requested_robot = {},
    const std::string &requested_name = {},
    const int requested_index = 1
) {
    return select_problem(
        read_json_file(path),
        requested_robot,
        requested_name,
        requested_index
    );
}

}  // namespace ppln::config

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>
#include <iomanip>
#include <limits>
#include <optional>

#include <cuda_runtime.h>

#include "src/collision/environment.hh"
#include "src/planning/Planners.hh"
#include "src/planning/AORRTC.hh"
#include "src/planning/PATACON_settings.hh"
#include "src/planning/RobotDispatch.hh"
#include "src/config/PlanningProblemJson.hh"
#include "src/config/PrimitiveEnvironmentJson.hh"
#include "src/constraints/RobotConstraintAdapter.hh"
#include "scripts/ffw_sg2_attached_object_collision.hh"
#include "src/io/PlannerResultJson.hh"

using json = nlohmann::json;
using namespace ppln::collision;

struct TraceExportOptions {
    bool requested = false;
    std::string trace_mode = "auto";
    std::string path_key = "path_start_to_goal";
    std::string graphml_path;
    std::string html_path;
    std::string html_trace_mode = "path";
    int html_max_tree_nodes = 6000;
    std::string patacon_root;
};

struct G1ReplanningOptions {
    bool enabled = false;
    bool server = false;
    double time_limit_sec = 5.0;
    std::string planner_executable;
};

constexpr float FFW_SG2_DEFAULT_OBJECT_MASS_KG = 25.0f;
constexpr float FFW_SG2_DEFAULT_SUPPORT_MARGIN_M = 0.07f;



std::string shell_quote(const std::string &value) {
    std::string output = "'";
    for (char character : value) {
        if (character == '\'') {
            output += "'\\''";
        } else {
            output += character;
        }
    }
    output += "'";
    return output;
}

std::string filename_component(std::string value) {
    for (char &character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (!std::isalnum(byte) && character != '-' && character != '_') {
            character = '_';
        }
    }
    return value;
}

void plot_patacon_run_ecdf(
    const json &runs,
    const std::string &robot_name,
    const std::string &problem_name,
    int problem_index,
    int run_count
) {
    if (runs.empty()) {
        std::cout << "PATACON ECDF plot skipped: no run history.\n";
        return;
    }

    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto input_path = std::filesystem::temp_directory_path()
        / ("patacon_run_ecdf_" + std::to_string(timestamp) + ".json");
    const auto output_path = std::filesystem::absolute(
        std::filesystem::path("logs")
        / (
            "patacon_" + filename_component(robot_name)
            + "_" + filename_component(problem_name)
            + "_" + std::to_string(problem_index)
            + "_" + std::to_string(run_count) + "runs_ecdf.png"
        )
    );
    std::filesystem::create_directories(output_path.parent_path());

    std::ofstream input(input_path);
    if (!input) {
        throw std::runtime_error(
            "failed to create temporary PATACON ECDF JSON"
        );
    }
    input << json{
        {"format", "PATACON_run_ecdf_v1"},
        {"planner", "PATACON"},
        {"robot", robot_name},
        {"problem_name", problem_name},
        {"problem_idx", problem_index},
        {"runs", run_count},
        {"results", runs},
    }.dump(2) << '\n';
    input.close();

    const auto script_path = std::filesystem::absolute(
        "scripts/plot_patacon_ecdf.py"
    );
    const std::string title =
        "PATACON - " + robot_name + " / " + problem_name
        + " #" + std::to_string(problem_index);
    const std::string command =
        "python3 "
        + shell_quote(script_path.string())
        + " "
        + shell_quote(input_path.string())
        + " --output "
        + shell_quote(output_path.string())
        + " --title "
        + shell_quote(title);

    std::cout << "plotting PATACON run-time ECDF...\n";
    std::cout.flush();
    std::cerr.flush();
    const int status = std::system(command.c_str());
    std::error_code remove_error;
    std::filesystem::remove(input_path, remove_error);

    if (status != 0) {
        throw std::runtime_error(
            "PATACON ECDF plotting script exited with an error"
        );
    }
    std::cout << "patacon_ecdf_plot: " << output_path.string() << "\n";
}

void plot_aorrtc_convergence(
    const json &runs,
    const std::string &robot_name,
    const std::string &problem_name,
    int problem_index,
    int run_count
) {
    if (runs.empty()) {
        std::cout << "AORRTC plot skipped: no run history.\n";
        return;
    }
    const bool any_solved = std::any_of(
        runs.begin(),
        runs.end(),
        [](const json &run) {
            return run.value("solved", false);
        }
    );
    if (!any_solved) {
        std::cout << "AORRTC plot skipped: no solved runs.\n";
        return;
    }

    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto input_path = std::filesystem::temp_directory_path()
        / ("aorrtc_convergence_" + std::to_string(timestamp) + ".json");
    const auto output_path = std::filesystem::absolute(
        std::filesystem::path("logs")
        / (
            "aorrtc_" + filename_component(robot_name)
            + "_" + filename_component(problem_name)
            + "_" + std::to_string(problem_index)
            + "_" + std::to_string(run_count) + "runs.png"
        )
    );
    std::filesystem::create_directories(output_path.parent_path());

    std::ofstream input(input_path);
    if (!input) {
        throw std::runtime_error(
            "failed to create temporary AORRTC plot JSON"
        );
    }
    input << json{
        {"format", "AORRTC_plot_runs_v1"},
        {"planner", "AORRTC"},
        {"robot", robot_name},
        {"problem_name", problem_name},
        {"problem_idx", problem_index},
        {"runs", run_count},
        {"results", runs},
    }.dump(2) << '\n';
    input.close();

    const auto script_path = std::filesystem::absolute(
        "scripts/plot_aorrtc.py"
    );
    const std::string title = "G1 whole body";
    const std::string command =
        "python3 "
        + shell_quote(script_path.string())
        + " "
        + shell_quote(input_path.string())
        + " --output "
        + shell_quote(output_path.string())
        + " --title "
        + shell_quote(title);

    std::cout << "plotting AORRTC convergence...\n";
    std::cout.flush();
    std::cerr.flush();
    const int status = std::system(command.c_str());
    std::error_code remove_error;
    std::filesystem::remove(input_path, remove_error);

    if (status != 0) {
        throw std::runtime_error(
            "AORRTC plotting script exited with an error"
        );
    }
    std::cout << "aorrtc_plot: " << output_path.string() << "\n";
}


std::string default_trace_result_json_path(
    const TraceExportOptions &options,
    const std::string &robot_name,
    const std::string &problem_name,
    int problem_index
) {
    if (!options.graphml_path.empty()) {
        const std::filesystem::path graphml(options.graphml_path);
        return (
            graphml.parent_path()
            / (graphml.stem().string() + "_result.json")
        ).string();
    }
    if (!options.html_path.empty()) {
        const std::filesystem::path html(options.html_path);
        return (
            html.parent_path()
            / (html.stem().string() + "_result.json")
        ).string();
    }
    return (
        std::filesystem::path("traces")
        / (robot_name + "_" + problem_name + "_"
           + std::to_string(problem_index) + "_result.json")
    ).string();
}


int export_trace_files(
    const std::string &result_json_path,
    const TraceExportOptions &options
) {
    if (!options.requested) {
        return 0;
    }
    std::ostringstream command;
    command
        << "python3 scripts/patacon_path_trace.py "
        << shell_quote(result_json_path)
        << " --trace-mode " << shell_quote(options.trace_mode)
        << " --path-key " << shell_quote(options.path_key)
        << " --html-trace-mode " << shell_quote(options.html_trace_mode)
        << " --html-max-tree-nodes " << options.html_max_tree_nodes;
    if (!options.graphml_path.empty()) {
        command << " --graphml " << shell_quote(options.graphml_path);
    }
    if (!options.html_path.empty()) {
        command << " --html " << shell_quote(options.html_path);
    }
    if (!options.patacon_root.empty()) {
        command << " --patacon-root " << shell_quote(options.patacon_root);
    }

    std::cout << "running_trace_export: " << command.str() << "\n";
    std::cout.flush();
    return std::system(command.str().c_str()) == 0 ? 0 : 1;
}


std::size_t measure_planner_warmup_ns() {
    const auto warmup_start = std::chrono::steady_clock::now();
    const cudaError_t free_status = cudaFree(nullptr);
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (free_status != cudaSuccess || sync_status != cudaSuccess) {
        std::cerr << "CUDA warmup failed: "
                  << cudaGetErrorString(free_status) << ", "
                  << cudaGetErrorString(sync_status) << "\n";
        return 0;
    }
    return get_elapsed_nanoseconds(warmup_start);
}


template <typename Robot>
json build_validated_quintic_hermite_geometry(
    const std::vector<typename Robot::Configuration> &path,
    const typename Robot::Configuration &start,
    Environment<float> &environment,
    PATACON_settings &settings
) {
    if (path.size() < 2) {
        throw std::runtime_error(
            "quintic Hermite interpolation requires at least two waypoints"
        );
    }
    auto squared_distance = [](const auto &left, const auto &right) {
        double squared = 0.0;
        for (std::size_t index = 0; index < left.size(); ++index) {
            const double difference =
                static_cast<double>(left[index]) - right[index];
            squared += difference * difference;
        }
        return squared;
    };

    json start_to_goal_waypoints = json::array();
    const bool forward = squared_distance(path.front(), start) <=
        squared_distance(path.back(), start);
    if (forward) {
        for (const auto &configuration : path) {
            start_to_goal_waypoints.push_back(configuration);
        }
    } else {
        for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
            start_to_goal_waypoints.push_back(*iterator);
        }
    }

    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto temporary_directory = std::filesystem::temp_directory_path();
    const auto input_path = temporary_directory /
        ("patacon_quintic_input_" + std::to_string(timestamp) + ".json");
    const auto output_path = temporary_directory /
        ("patacon_quintic_output_" + std::to_string(timestamp) + ".json");
    {
        std::ofstream input(input_path);
        if (!input) {
            throw std::runtime_error(
                "failed to create temporary quintic Hermite input"
            );
        }
        input << json{{"waypoints", start_to_goal_waypoints}}.dump() << '\n';
    }

    const auto script_path = std::filesystem::absolute(
        "scripts/trajectory_pipeline.py"
    );
    const std::array<double, 10> derivative_scales = {
        1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001,
        0.0001, 0.00001, 0.000001
    };
    PathValidationResult last_validation;
    bool generated_candidate = false;
    for (double derivative_scale : derivative_scales) {
        const std::string command =
            "python3 " + shell_quote(script_path.string()) +
            " --input " + shell_quote(input_path.string()) +
            " --output " + shell_quote(output_path.string()) +
            " --derivative-scale " + std::to_string(derivative_scale) +
            " --validation-samples-per-segment " +
            std::to_string(settings.granularity);
        std::cout.flush();
        std::cerr.flush();
        const int status = std::system(command.c_str());
        if (status != 0) {
            std::error_code remove_error;
            std::filesystem::remove(input_path, remove_error);
            std::filesystem::remove(output_path, remove_error);
            throw std::runtime_error(
                "quintic Hermite interpolation script exited with an error"
            );
        }

        json output;
        {
            std::ifstream stream(output_path);
            if (!stream) {
                throw std::runtime_error(
                    "quintic Hermite interpolation did not produce output"
                );
            }
            stream >> output;
        }
        generated_candidate = true;
        const auto validation_samples = output.at("validation_samples")
            .template get<std::vector<typename Robot::Configuration>>();
        last_validation = PATACON::validate_path_for_visualization<Robot>(
            validation_samples,
            environment,
            settings,
            2.0f * settings.projection_task_tolerance
        );
        if (last_validation.valid) {
            json geometry = std::move(output.at("geometric_path"));
            geometry["cuda_revalidated"] = true;
            geometry["validation_sample_count"] = validation_samples.size();
            geometry["validation_checked_edges"] =
                last_validation.checked_edges;
            geometry["validation_maximum_projection_delta"] =
                last_validation.maximum_projection_delta;
            geometry["validation_projection_tolerance"] =
                2.0f * settings.projection_task_tolerance;
            geometry["nominal_spline_collision_revalidated"] = true;
            std::error_code remove_error;
            std::filesystem::remove(input_path, remove_error);
            std::filesystem::remove(output_path, remove_error);
            std::cout
                << "visualization_quintic_hermite: degree=5 basis=bernstein"
                << " derivative_scale=" << derivative_scale
                << " validation_samples=" << validation_samples.size()
                << " validation_edges=" << last_validation.checked_edges
                << " max_projection_delta="
                << last_validation.maximum_projection_delta
                << " projection_tolerance="
                << 2.0f * settings.projection_task_tolerance
                << "\n";
            return geometry;
        }
        std::cout
            << "visualization_quintic_hermite_retry: derivative_scale="
            << derivative_scale
            << " failed_edge=" << last_validation.failed_edge
            << " max_projection_delta="
            << last_validation.maximum_projection_delta
            << "\n";
    }

    std::error_code remove_error;
    std::filesystem::remove(input_path, remove_error);
    std::filesystem::remove(output_path, remove_error);
    if (!generated_candidate) {
        throw std::runtime_error("failed to generate a quintic Hermite spline");
    }
    throw std::runtime_error(
        "quintic Hermite spline failed full CUDA revalidation at edge " +
        std::to_string(last_validation.failed_edge)
    );
}


template <typename Robot>
void visualize_ffw_sg2_path(
    const PlannerResult<Robot> &result,
    const typename Robot::Configuration &start,
    const std::vector<std::string> &joint_names,
    const json &problem,
    const json &geometric_path,
    std::array<float, 3> attached_object_frame_offset = {0.1f, 0.0f, 0.0f},
    bool use_ctrl = false
) {
    if (result.path.size() < 2) {
        throw std::runtime_error("cannot visualize an unsolved or empty path");
    }

    auto squared_distance = [](const auto &a, const auto &b) {
        float distance = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const float difference = a[i] - b[i];
            distance += difference * difference;
        }
        return distance;
    };

    json trajectory;
    trajectory["joint_names"] = joint_names;
    trajectory["waypoints"] = json::array();
    trajectory["start"] = start;
    trajectory["environment"] = {
        {"sphere", problem.value("sphere", json::array())},
        {"cylinder", problem.value("cylinder", json::array())},
        {"box", problem.value("box", json::array())},
    };
    trajectory["path_smoothing"] = !geometric_path.is_null();
    if (!geometric_path.is_null()) {
        trajectory["geometric_path"] = geometric_path;
    }
    trajectory["attached_object_frame_offset"] = {
        attached_object_frame_offset[0],
        attached_object_frame_offset[1],
        attached_object_frame_offset[2],
    };

    const bool path_is_start_to_goal =
        squared_distance(result.path.front(), start)
        <= squared_distance(result.path.back(), start);
    if (path_is_start_to_goal) {
        for (const auto &configuration : result.path) {
            trajectory["waypoints"].push_back(configuration);
        }
    } else {
        for (auto iterator = result.path.rbegin(); iterator != result.path.rend(); ++iterator) {
            trajectory["waypoints"].push_back(*iterator);
        }
    }

    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto trajectory_path = std::filesystem::temp_directory_path()
        / ("patacon_" + std::string(Robot::name) + "_trajectory_"
            + std::to_string(timestamp) + ".json");
    {
        std::ofstream trajectory_file(trajectory_path);
        if (!trajectory_file) {
            throw std::runtime_error("failed to create temporary visualization trajectory");
        }
        trajectory_file << trajectory.dump(2) << '\n';
    }

    const auto visualizer_path = std::filesystem::absolute(
        "scripts/visualize_ffw_sg2.py"
    );
    const bool mobility_model =
        !joint_names.empty() && joint_names.front() == "base_x";
    const std::string scene =
        mobility_model ? "rack_upper_to_lower" : "lift";
    const std::string command =
        "python3 \"" + visualizer_path.string() + "\""
        + " --scene " + scene
        + " --trajectory \"" + trajectory_path.string() + "\""
        + (use_ctrl ? " --input-mode ctrl" : " --input-mode qpos");

    std::cout.flush();
    std::cerr.flush();
    const int status = std::system(command.c_str());
    std::error_code remove_error;
    std::filesystem::remove(trajectory_path, remove_error);
    if (status != 0) {
        throw std::runtime_error("MuJoCo visualizer exited with an error");
    }
}

template <typename Robot>
void visualize_franka_path(
    const PlannerResult<Robot> &result,
    const typename Robot::Configuration &start,
    const std::vector<std::string> &joint_names,
    const json &problem,
    const json &geometric_path
) {
    if (result.path.size() < 2) {
        throw std::runtime_error("cannot visualize an unsolved Franka path");
    }
    auto squared_distance = [](const auto &left, const auto &right) {
        float squared = 0.0f;
        for (std::size_t index = 0; index < left.size(); ++index) {
            const float difference = left[index] - right[index];
            squared += difference * difference;
        }
        return squared;
    };
    json trajectory;
    trajectory["joint_names"] = joint_names;
    trajectory["start"] = start;
    trajectory["waypoints"] = json::array();
    trajectory["environment"] = {
        {"sphere", problem.value("sphere", json::array())},
        {"cylinder", problem.value("cylinder", json::array())},
        {"box", problem.value("box", json::array())},
    };
    trajectory["path_smoothing"] = !geometric_path.is_null();
    if (!geometric_path.is_null()) {
        trajectory["geometric_path"] = geometric_path;
    }
    const bool forward = squared_distance(result.path.front(), start) <=
        squared_distance(result.path.back(), start);
    if (forward) {
        for (const auto &configuration : result.path) {
            trajectory["waypoints"].push_back(configuration);
        }
    } else {
        for (auto iterator = result.path.rbegin();
             iterator != result.path.rend(); ++iterator) {
            trajectory["waypoints"].push_back(*iterator);
        }
    }
    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto trajectory_path = std::filesystem::temp_directory_path() /
        ("patacon_" + std::string(Robot::name) + "_trajectory_" +
         std::to_string(timestamp) + ".json");
    {
        std::ofstream output(trajectory_path);
        if (!output) {
            throw std::runtime_error(
                "failed to create temporary Franka trajectory"
            );
        }
        output << trajectory.dump(2) << '\n';
    }
    const auto visualizer = std::filesystem::absolute(
        "scripts/visualize_franka.py"
    );
    const auto model = std::filesystem::absolute(
        std::is_same_v<Robot, robots::FrankaSingle>
            ? "resources/franka/mujoco/fer_single.xml"
            : "resources/franka/mujoco/fer_dual.xml"
    );
    std::string command =
        "python3 " + shell_quote(visualizer.string()) +
        " --model " + shell_quote(model.string()) +
        " --trajectory " + shell_quote(trajectory_path.string());
    const char *validate_only = std::getenv("PATACON_MUJOCO_VALIDATE_ONLY");
    if (validate_only != nullptr && std::string(validate_only) == "1") {
        command += " --validate-only";
    }
    std::cout.flush();
    std::cerr.flush();
    const int status = std::system(command.c_str());
    std::error_code remove_error;
    std::filesystem::remove(trajectory_path, remove_error);
    if (status != 0) {
        throw std::runtime_error("Franka MuJoCo visualizer exited with an error");
    }
}

template <typename Robot>
void visualize_ffw_sg2_mobility_ctrl_path(
    const PlannerResult<Robot> &result,
    const typename Robot::Configuration &start,
    const std::vector<std::string> &joint_names,
    const json &problem,
    std::array<float, 3> attached_object_frame_offset,
    float object_mass_kg,
    float support_margin_m,
    const json &geometric_path
) {
    if (result.path.size() < 2) {
        throw std::runtime_error("cannot visualize an unsolved or empty path");
    }

    auto squared_distance = [](const auto &a, const auto &b) {
        float distance = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const float difference = a[i] - b[i];
            distance += difference * difference;
        }
        return distance;
    };

    json trajectory;
    trajectory["joint_names"] = joint_names;
    trajectory["waypoints"] = json::array();
    trajectory["start"] = start;
    trajectory["environment"] = {
        {"sphere", problem.value("sphere", json::array())},
        {"cylinder", problem.value("cylinder", json::array())},
        {"box", problem.value("box", json::array())},
    };
    trajectory["path_smoothing"] = !geometric_path.is_null();
    if (!geometric_path.is_null()) {
        trajectory["geometric_path"] = geometric_path;
    }
    trajectory["attached_object_frame_offset"] = {
        attached_object_frame_offset[0],
        attached_object_frame_offset[1],
        attached_object_frame_offset[2],
    };

    const bool path_is_start_to_goal =
        squared_distance(result.path.front(), start)
        <= squared_distance(result.path.back(), start);
    if (path_is_start_to_goal) {
        for (const auto &configuration : result.path) {
            trajectory["waypoints"].push_back(configuration);
        }
    } else {
        for (auto iterator = result.path.rbegin(); iterator != result.path.rend(); ++iterator) {
            trajectory["waypoints"].push_back(*iterator);
        }
    }

    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto trajectory_path = std::filesystem::temp_directory_path()
        / ("patacon_" + std::string(Robot::name) + "_trajectory_"
            + std::to_string(timestamp) + ".json");
    {
        std::ofstream trajectory_file(trajectory_path);
        if (!trajectory_file) {
            throw std::runtime_error("failed to create temporary visualization trajectory");
        }
        trajectory_file << trajectory.dump(2) << '\n';
    }

    const auto visualizer_path = std::filesystem::absolute(
        "scripts/sim_ffw_sg2_rack_upper_to_lower.py"
    );
    std::string command =
        "python3 " + shell_quote(visualizer_path.string())
        + " --trajectory " + shell_quote(trajectory_path.string())
        + " --attach-payload"
        + " --object-mass " + std::to_string(object_mass_kg)
        + " --support-margin " + std::to_string(support_margin_m)
        + " --payload-offset "
        + std::to_string(attached_object_frame_offset[0])
        + " "
        + std::to_string(attached_object_frame_offset[1])
        + " "
        + std::to_string(attached_object_frame_offset[2]);
    std::cout.flush();
    std::cerr.flush();
    const int status = std::system(command.c_str());
    std::error_code remove_error;
    std::filesystem::remove(trajectory_path, remove_error);
    if (status != 0) {
        throw std::runtime_error("MuJoCo mobility ctrl visualizer exited with an error");
    }
}


void visualize_g1_path(
    const PlannerResult<robots::G1> &result,
    const robots::G1::Configuration &start,
    const json &problem,
    const json &geometric_path,
    const G1ReplanningOptions &replanning,
    const AORRTC_settings &settings,
    double planning_time_sec
) {
    if (result.path.size() < 2) {
        throw std::runtime_error("cannot visualize an unsolved or empty G1 path");
    }

    auto squared_distance = [](const auto &a, const auto &b) {
        float distance = 0.0f;
        for (std::size_t index = 0; index < a.size(); ++index) {
            const float difference = a[index] - b[index];
            distance += difference * difference;
        }
        return distance;
    };

    json trajectory;
    trajectory["start"] = start;
    trajectory["waypoints"] = json::array();
    trajectory["planning_time_sec"] = planning_time_sec;
    trajectory["path_smoothing"] = !geometric_path.is_null();
    if (!geometric_path.is_null()) {
        trajectory["geometric_path"] = geometric_path;
    }
    trajectory["environment"] = {
        {"sphere", problem.value("sphere", json::array())},
        {"cylinder", problem.value("cylinder", json::array())},
        {"box", problem.value("box", json::array())}
    };
    const auto &constraints = problem.at("constraints");
    trajectory["constraints"] = constraints;
    const auto &goals = g1_goals_from_problem(
        problem,
        settings.axis
    );
    if (!goals.is_array() || goals.empty()) {
        throw std::runtime_error("G1 visualization problem has no goal");
    }
    trajectory["goal"] = goals.at(0);
    trajectory["replanning"] = {
        {"enabled", replanning.enabled},
        {"planner_executable", replanning.planner_executable},
        {"base_seed", settings.random_seed},
        {"aorrtc", settings.aorrtc},
        {
            "time_limit_sec",
            settings.aorrtc
                ? settings.time_limit_sec
                : replanning.time_limit_sec
        },
        {"projection_smoothness", settings.projection_smoothness},
        {"axis", settings.axis},
        {"mouse_obstacle_radius_m", 0.040},
        {"mouse_obstacle_collision_radius_m", 0.040},
        {"mouse_obstacle_initial_position", {0.400, 0.200, 0.800}}
    };
    const auto &center_of_mass = constraints.at("com");
    if (center_of_mass.contains("payload")) {
        trajectory["payload"] = center_of_mass.at("payload");

        // The bimanual translation is the right-hand frame origin expressed
        // in the left-hand frame.  The Unitree/MuJoCo rubber-hand endpoint is at
        // [0.0415, 0.003, 0] in the left wrist-yaw body.
        const auto &target = constraints.at("bimanual").at("target");
        const auto midpoint_offset = trajectory["payload"].value(
            "hand_midpoint_offset",
            std::vector<double>{0.0, 0.0, 0.0}
        );
        trajectory["payload"]["left_hand_center_offset"] = {
            0.0415 + 0.5 * target.at(4).get<double>() + midpoint_offset.at(0),
            0.003 + 0.5 * target.at(5).get<double>() + midpoint_offset.at(1),
            0.5 * target.at(6).get<double>() + midpoint_offset.at(2)
        };
    }

    const bool path_is_start_to_goal =
        squared_distance(result.path.front(), start)
        <= squared_distance(result.path.back(), start);
    if (path_is_start_to_goal) {
        for (const auto &configuration : result.path) {
            trajectory["waypoints"].push_back(configuration);
        }
    } else {
        for (auto iterator = result.path.rbegin(); iterator != result.path.rend(); ++iterator) {
            trajectory["waypoints"].push_back(*iterator);
        }
    }

    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto trajectory_path = std::filesystem::temp_directory_path()
        / ("patacon_g1_trajectory_" + std::to_string(timestamp) + ".json");
    {
        std::ofstream trajectory_file(trajectory_path);
        if (!trajectory_file) {
            throw std::runtime_error("failed to create temporary G1 trajectory");
        }
        trajectory_file << trajectory.dump(2) << '\n';
    }

    const auto visualizer_path = std::filesystem::absolute(
        replanning.enabled
            ? "scripts/visualize_g1_replanning.py"
            : "scripts/visualize_g1.py"
    );
    std::string command =
        "python3 " + shell_quote(visualizer_path.string())
        + " --trajectory " + shell_quote(trajectory_path.string())
        + " --control-mode ctrl";
    if (replanning.enabled) {
        command += " --replanning";
    }

    std::cout.flush();
    std::cerr.flush();
    const int status = std::system(command.c_str());
    std::error_code remove_error;
    std::filesystem::remove(trajectory_path, remove_error);
    if (status != 0) {
        throw std::runtime_error("G1 MuJoCo visualizer exited with an error");
    }
}


void visualize_igris_c_path(
    const PlannerResult<robots::IgrisC> &result,
    const robots::IgrisC::Configuration &start,
    const json &problem,
    float object_mass_kg,
    const json &geometric_path
) {
    if (result.path.size() < 2) {
        throw std::runtime_error(
            "cannot visualize an unsolved or empty IGRIS-C path"
        );
    }

    auto squared_distance = [](const auto &a, const auto &b) {
        float distance = 0.0f;
        for (std::size_t index = 0; index < a.size(); ++index) {
            const float difference = a[index] - b[index];
            distance += difference * difference;
        }
        return distance;
    };

    json trajectory;
    trajectory["start"] = start;
    trajectory["waypoints"] = json::array();
    trajectory["path_smoothing"] = !geometric_path.is_null();
    if (!geometric_path.is_null()) {
        trajectory["geometric_path"] = geometric_path;
    }
    trajectory["environment"] = {
        {"sphere", problem.value("sphere", json::array())},
        {"cylinder", problem.value("cylinder", json::array())},
        {"box", problem.value("box", json::array())}
    };
    trajectory["task"] = problem.value("task", json::object());
    trajectory["constraints"] = problem.value(
        "constraints",
        json::object()
    );

    const bool path_is_start_to_goal =
        squared_distance(result.path.front(), start)
        <= squared_distance(result.path.back(), start);
    if (path_is_start_to_goal) {
        for (const auto &configuration : result.path) {
            trajectory["waypoints"].push_back(configuration);
        }
    } else {
        for (
            auto iterator = result.path.rbegin();
            iterator != result.path.rend();
            ++iterator
        ) {
            trajectory["waypoints"].push_back(*iterator);
        }
    }

    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto trajectory_path = std::filesystem::temp_directory_path()
        / ("patacon_igris_c_trajectory_" + std::to_string(timestamp) + ".json");
    {
        std::ofstream trajectory_file(trajectory_path);
        if (!trajectory_file) {
            throw std::runtime_error(
                "failed to create temporary IGRIS-C trajectory"
            );
        }
        trajectory_file << trajectory.dump(2) << '\n';
    }

    const auto visualizer_path = std::filesystem::absolute(
        "scripts/visualize_igris_c.py"
    );
    std::string command =
        "python3 " + shell_quote(visualizer_path.string())
        + " --trajectory " + shell_quote(trajectory_path.string())
        + " --object-mass " + std::to_string(object_mass_kg);

    std::cout.flush();
    std::cerr.flush();
    const int status = std::system(command.c_str());
    std::error_code remove_error;
    std::filesystem::remove(trajectory_path, remove_error);
    if (status != 0) {
        throw std::runtime_error("IGRIS-C MuJoCo visualizer exited with an error");
    }
}


template <typename Robot>
int run_planner(
    json &data,
    Environment<float> &env,
    AORRTC_settings &settings,
    bool visualize,
    bool path_smoothing,
    bool print_path,
     bool plot,
    const std::string &robot_name,
    const std::string &problem_name,
    int problem_index,
    const std::string &save_json_path,
    const TraceExportOptions &trace_options,
    int runs,
    const G1ReplanningOptions &g1_replanning = G1ReplanningOptions{},
    bool perform_warmup = true,
    json *result_payload_out = nullptr
) {
    using Configuration = typename Robot::Configuration;
    if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
        ffw_sg2_attached_object_collision::apply_from_problem(
            data,
            settings,
            ffw_sg2_attached_object_collision::kFfwSg2FixedFineSphereCount,
            ffw_sg2_attached_object_collision::kFfwSg2FixedApproxSphereCount
        );
    } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
        ffw_sg2_attached_object_collision::apply_from_problem(
            data,
            settings
        );
    } else {
        settings.ffw_sg2_attached_object_collision = {};
        if (data.contains("attached_object_collision")) {
            throw std::invalid_argument(
                "attached_object_collision is supported only for ffw_sg2 and ffw_sg2_mobility"
            );
        }
    }

    ppln::constraints::apply_constraint_backend_defaults<Robot>(settings);
    auto prepared_query =
        ppln::constraints::prepare_constraint_query<Robot>(data, settings);
    Configuration start = prepared_query.start;
    std::vector<Configuration> goals = std::move(prepared_query.goals);
    if constexpr (std::is_same_v<Robot, robots::G1>) {
        if (g1_replanning.enabled) {
            if (!PATACON::project_g1_configuration(start, settings)) {
                throw std::runtime_error(
                    "G1 replanning start constraint projection failed"
                );
            }
            for (auto &goal : goals) {
                if (!PATACON::project_g1_configuration(goal, settings)) {
                    throw std::runtime_error(
                        "G1 replanning goal constraint projection failed"
                    );
                }
            }
            data["start"] = start;
            data["goals"] = goals;
            data["axis_endpoints"]["start"] = start;
            data["axis_endpoints"]["goals"] = goals;
        }
    }
    json saved_results = json::array();
    int solved_count = 0;
    std::vector<double> times_sec;
    std::vector<int> path_lengths;
    std::vector<float> costs;
    json patacon_plot_runs = json::array();
    json aorrtc_plot_runs = json::array();
    PlannerResult<Robot> visualization_result;
    json visualization_geometric_path;
    double visualization_planning_time_sec = 0.0;
    const unsigned long long base_seed = settings.random_seed;

    for (int run_index = 1; run_index <= runs; run_index++) {
        settings.random_seed = base_seed
            + static_cast<unsigned long long>(run_index - 1);
        if (runs > 1) {
            std::cout << "run: " << run_index << "\n";
        }
        std::cout << "seed: " << settings.random_seed << "\n";

        std::size_t warmup_ns = 0;
        if (run_index == 1 && perform_warmup) {
            warmup_ns = measure_planner_warmup_ns();
        }
        AORRTCResult<Robot> result = PATACON::solve<Robot>(
            start,
            goals,
            env,
            settings
        );

        if (print_path) {
            for (auto& cfg : result.path) {
                print_cfg<Robot>(cfg);
            }
        }
        if (not result.solved) {
            std::cout << "failed!" << std::endl;
        } else {
            solved_count++;
            path_lengths.push_back(result.path_length);
            costs.push_back(result.cost);
        }

        double planning_sec = 0.0;

        if (settings.aorrtc) {
            // 실제 AORRTC planning 시작부터 cleanup 직전까지
            planning_sec =
                static_cast<double>(result.planning_ns) / 1.0e9;
        } else {
            // 기존 PATACON 동작은 그대로 유지
            planning_sec =
                static_cast<double>(result.kernel_ns) / 1.0e9;
        }

        const double warmup_sec = static_cast<double>(warmup_ns) / 1.0e9;

        if (result.solved) {
            times_sec.push_back(planning_sec);
        }
        std::cout << "cost: " << result.cost << "\n";
        if (runs > 1 && run_index == 1) {
            std::cout << "warmup_s: " << warmup_sec << "\n";
        }
        std::cout << "planning_s: " << planning_sec << "\n";
        if (settings.collect_diagnostics) {
            std::cout << "diagnostics: "
                      << planner_result_json::diagnostics_to_json(
                             result.diagnostics
                         ).dump()
                      << "\n";
        }
        // std::cout << "time (us): " << result.kernel_ns/1000.0f << "\n";
        // std::cout << "time (s): " << static_cast<double>(result.kernel_ns) / 1.0e9 << "\n";
        if (settings.aorrtc) {
            std::cout << "aorrtc_initial_cost: " << result.initial_cost << "\n";
            std::cout << "aorrtc_solution_updates: "
                      << result.solution_updates << "\n";
            std::cout << "aorrtc_search_restarts: "
                      << result.search_restarts << "\n";
            std::cout << "aorrtc_initial_solution_sec: "
                      << static_cast<double>(result.initial_solution_ns) / 1.0e9
                      << "\n";
            std::cout << "aorrtc_best_solution_sec: "
                      << static_cast<double>(result.best_solution_ns) / 1.0e9
                      << "\n";
        }

        if (!save_json_path.empty() || result_payload_out != nullptr) {
            auto payload = planner_result_json::result_to_json<Robot>(
                result,
                settings,
                env,
                start,
                goals,
                robot_name,
                problem_name,
                problem_index
            );
            payload["run_idx"] = run_index;
            payload["run_count"] = runs;
            saved_results.push_back(payload);
        }

        if (plot && settings.aorrtc) {
            json solution_history = json::array();
            for (const auto &update : result.solution_history) {
                solution_history.push_back({
                    {
                        "found_sec",
                        static_cast<double>(update.found_ns) / 1.0e9
                    },
                    {"cost", update.cost},
                });
            }
            aorrtc_plot_runs.push_back({
                {"run_idx", run_index},
                {"solved", result.solved},
                {"cost", result.cost},
                {"planning_sec", planning_sec},
                {"initial_cost", result.initial_cost},
                {
                    "initial_solution_sec",
                    static_cast<double>(result.initial_solution_ns) / 1.0e9
                },
                {
                    "best_solution_sec",
                    static_cast<double>(result.best_solution_ns) / 1.0e9
                },
                {"solution_updates", result.solution_updates},
                {
                    "solution_history_overflow",
                    result.solution_history_overflow
                },
                {"solution_history", solution_history},
            });
        } else if (plot) {
            patacon_plot_runs.push_back({
                {"run_idx", run_index},
                {"seed", settings.random_seed},
                {"solved", result.solved},
                {"planning_sec", planning_sec},
                {"kernel_ns", result.kernel_ns},
            });
        }

        if (visualize && run_index == runs) {
            visualization_planning_time_sec = planning_sec;
            constexpr bool visualization_supported =
                std::is_same_v<Robot, robots::FfwSg2> ||
                std::is_same_v<Robot, robots::FfwSg2Mobility> ||
                std::is_same_v<Robot, robots::G1> ||
                std::is_same_v<Robot, robots::IgrisC> ||
                std::is_same_v<Robot, robots::FrankaSingle> ||
                std::is_same_v<Robot, robots::Franka>;
            if constexpr (visualization_supported) {
                if (!path_smoothing) {
                    std::cout
                        << "visualization_path_smoothing: disabled "
                        << "(raw planner path; no shortcut, quintic spline, "
                        << "or TOPP-RA)\n";
                } else if (result.solved && result.path.size() >= 3) {
                    try {
                        const std::size_t original_waypoint_count =
                            result.path.size();
                        auto simplification =
                            PATACON::simplify_path_for_visualization<Robot>(
                                result.path,
                                env,
                                settings
                            );
                        result.path = std::move(simplification.path);
                        std::cout
                            << "visualization_path_simplification: "
                            << "attempted="
                            << simplification.attempted_shortcuts
                            << " accepted="
                            << simplification.accepted_shortcuts
                            << " cost="
                            << simplification.original_cost
                            << "->"
                            << simplification.simplified_cost
                            << " waypoints="
                            << original_waypoint_count
                            << "->"
                            << result.path.size()
                            << "\n";
                    } catch (const std::exception &error) {
                        std::cerr
                            << "warning: visualization path simplification "
                            << "failed; using the original planner path: "
                            << error.what()
                            << "\n";
                    }
                }
                if (
                    path_smoothing &&
                    result.solved &&
                    result.path.size() >= 2
                ) {
                    try {
                        visualization_geometric_path =
                            build_validated_quintic_hermite_geometry<Robot>(
                                result.path,
                                start,
                                env,
                                settings
                            );
                    } catch (const std::exception &error) {
                        visualization_geometric_path = json();
                        std::cerr
                            << "warning: visualization quintic Hermite "
                            << "smoothing failed CUDA revalidation; using "
                            << "the validated planner polyline without "
                            << "smoothing or TOPP-RA: "
                            << error.what()
                            << "\n";
                    }
                }
            }
            visualization_result = std::move(result);
        }
    }

    if (plot && settings.aorrtc) {
        plot_aorrtc_convergence(
            aorrtc_plot_runs,
            robot_name,
            problem_name,
            problem_index,
            runs
        );
    } else if (plot) {
        plot_patacon_run_ecdf(
            patacon_plot_runs,
            robot_name,
            problem_name,
            problem_index,
            runs
        );
    }

    if (runs > 1) {
        std::cout << "solved_runs: " << solved_count << "/" << runs << "\n";
    }

    if (runs > 1 && !times_sec.empty()) {
        const auto [minimum, maximum] = std::minmax_element(
            times_sec.begin(),
            times_sec.end()
        );
        double sum = 0.0;
        for (double elapsed_sec : times_sec) {
            sum += elapsed_sec;
        }
        const double average = sum / static_cast<double>(times_sec.size());
        double squared_deviation_sum = 0.0;
        for (double elapsed_sec : times_sec) {
            const double difference = elapsed_sec - average;
            squared_deviation_sum += difference * difference;
        }
        const double standard_deviation = std::sqrt(
            squared_deviation_sum / static_cast<double>(times_sec.size())
        );

        std::cout << "planning_s_avg: " << average << "\n";
        std::cout << "planning_s_min: " << *minimum << "\n";
        std::cout << "planning_s_max: " << *maximum << "\n";
        std::cout << "planning_s_std: " << standard_deviation << "\n";
    }

    if (runs > 1 && !path_lengths.empty()) {
        const auto [path_length_minimum, path_length_maximum] =
            std::minmax_element(path_lengths.begin(), path_lengths.end());
        double path_length_sum = 0.0;
        for (int path_length : path_lengths) {
            path_length_sum += path_length;
        }

        const auto [cost_minimum, cost_maximum] =
            std::minmax_element(costs.begin(), costs.end());
        double cost_sum = 0.0;
        for (float cost : costs) {
            cost_sum += cost;
        }

        std::cout << "path_length_avg: "
                  << path_length_sum / static_cast<double>(path_lengths.size())
                  << "\n";
        std::cout << "path_length_min: " << *path_length_minimum << "\n";
        std::cout << "path_length_max: " << *path_length_maximum << "\n";
        std::cout << "cost_avg: "
                  << cost_sum / static_cast<double>(costs.size()) << "\n";
        std::cout << "cost_min: " << *cost_minimum << "\n";
        std::cout << "cost_max: " << *cost_maximum << "\n";
    }

    if (!save_json_path.empty()) {
        if (runs == 1) {
            planner_result_json::write_json_file(saved_results[0], save_json_path);
        } else {
            const json payload = {
                {
                    "format",
                    settings.aorrtc
                        ? "AORRTC_run_results_v1"
                        : "PATACON_run_results_v1"
                },
                {"planner", settings.aorrtc ? "AORRTC" : "PATACON"},
                {"robot", robot_name},
                {"problem_name", problem_name},
                {"problem_idx", problem_index},
                {"runs", runs},
                {"solved_runs", solved_count},
                {"results", saved_results},
            };
            planner_result_json::write_json_file(payload, save_json_path);
        }
        std::cout << "saved_json: " << save_json_path << "\n";
    }
    if (result_payload_out != nullptr) {
        if (runs == 1) {
            *result_payload_out = saved_results.at(0);
        } else {
            *result_payload_out = {
                {
                    "format",
                    settings.aorrtc
                        ? "AORRTC_run_results_v1"
                        : "PATACON_run_results_v1"
                },
                {"planner", settings.aorrtc ? "AORRTC" : "PATACON"},
                {"robot", robot_name},
                {"problem_name", problem_name},
                {"problem_idx", problem_index},
                {"runs", runs},
                {"solved_runs", solved_count},
                {"results", saved_results},
            };
        }
    }
    if (trace_options.requested) {
        if (save_json_path.empty()) {
            throw std::runtime_error("trace export requires a saved result JSON");
        }
        if (export_trace_files(save_json_path, trace_options) != 0) {
            throw std::runtime_error("trace exporter exited with an error");
        }
    }
    if (visualize) {
        if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
            const auto &attached_object =
                settings.ffw_sg2_attached_object_collision;
            const std::array<float, 3> attached_object_frame_offset = {
                attached_object.enabled ? attached_object.world_offset[0] : 0.1f,
                attached_object.enabled ? attached_object.world_offset[1] : 0.0f,
                attached_object.enabled ? attached_object.world_offset[2] : 0.0f,
            };
            visualize_ffw_sg2_path(visualization_result, start, {
                "lift_joint",
                "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
                "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
                "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
                "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"
            }, data, visualization_geometric_path, attached_object_frame_offset);
        } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
            const auto &attached_object =
                settings.ffw_sg2_attached_object_collision;
            const std::array<float, 3> attached_object_frame_offset = {
                attached_object.enabled ? attached_object.world_offset[0] : 0.1f,
                attached_object.enabled ? attached_object.world_offset[1] : 0.0f,
                attached_object.enabled ? attached_object.world_offset[2] : 0.0f,
            };
            const std::vector<std::string> joint_names = {
                "base_x", "base_y", "base_yaw",
                "lift_joint",
                "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
                "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
                "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
                "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"
            };
            const float object_mass_kg =
                settings.ffw_sg2_object_mass_kg > 0.0f
                    ? settings.ffw_sg2_object_mass_kg
                    : 3.0f;
            const float support_margin_m =
                settings.ffw_sg2_support_margin_m > 0.0f
                    ? settings.ffw_sg2_support_margin_m
                    : 0.05f;
            visualize_ffw_sg2_mobility_ctrl_path(
                visualization_result,
                start,
                joint_names,
                data,
                attached_object_frame_offset,
                object_mass_kg,
                support_margin_m,
                visualization_geometric_path
            );
        } else if constexpr (std::is_same_v<Robot, robots::G1>) {
            visualize_g1_path(
                visualization_result,
                start,
                data,
                visualization_geometric_path,
                g1_replanning,
                settings,
                visualization_planning_time_sec
            );
        } else if constexpr (std::is_same_v<Robot, robots::IgrisC>) {
            float object_mass_kg = 0.15f;
            if (data.contains("task")
                && data["task"].contains("payload_mass_kg")) {
                object_mass_kg = data["task"]["payload_mass_kg"].get<float>();
            }
            if (data.contains("constraints")
                && data["constraints"].contains("com")
                && data["constraints"]["com"].contains("payload_mass_kg")) {
                object_mass_kg = data["constraints"]["com"]
                    ["payload_mass_kg"].get<float>();
            }
            if (settings.ffw_sg2_object_mass_kg > 0.0f) {
                object_mass_kg = settings.ffw_sg2_object_mass_kg;
            }
            visualize_igris_c_path(
                visualization_result,
                start,
                data,
                object_mass_kg,
                visualization_geometric_path
            );
        } else if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
            visualize_franka_path(visualization_result, start, {
                "fer0_joint1", "fer0_joint2", "fer0_joint3",
                "fer0_joint4", "fer0_joint5", "fer0_joint6",
                "fer0_joint7"
            }, data, visualization_geometric_path);
        } else if constexpr (std::is_same_v<Robot, robots::Franka>) {
            visualize_franka_path(visualization_result, start, {
                "fer0_joint1", "fer0_joint2", "fer0_joint3",
                "fer0_joint4", "fer0_joint5", "fer0_joint6",
                "fer0_joint7", "fer1_joint1", "fer1_joint2",
                "fer1_joint3", "fer1_joint4", "fer1_joint5",
                "fer1_joint6", "fer1_joint7"
            }, data, visualization_geometric_path);
        } else {
            throw std::runtime_error(
                "--visualize supports only ffw_sg2, ffw_sg2_mobility, "
                "g1, igris_c, franka_single, and franka"
            );
        }
    }
    return 0;
}

int run_g1_replan_server(
    const json &problem_template,
    AORRTC_settings &settings,
    const G1ReplanningOptions &replanning
) {
    const G1ReplanningOptions no_nested_replanning;
    const TraceExportOptions no_trace;

    PATACON::set_cuda_device_reset_enabled(false);
    PATACON::set_persistent_workspace_enabled(!settings.aorrtc);
    PATACON::set_time_limit_seconds(
        settings.aorrtc ? 0.0 : replanning.time_limit_sec
    );
    const std::size_t warmup_ns = measure_planner_warmup_ns();
    std::cout
        << "g1_replan_server: PATACON ready, warmup_s="
        << static_cast<double>(warmup_ns) / 1.0e9
        << ", time_limit_s="
        << (settings.aorrtc
            ? settings.time_limit_sec
            : replanning.time_limit_sec)
        << "\n";
    std::cout.flush();

    std::string request_line;
    int request_index = 0;
    const unsigned long long replanning_base_seed = settings.random_seed;
    while (std::getline(std::cin, request_line)) {
        if (request_line.empty()) {
            continue;
        }
        ++request_index;
        // Keep the initial plan on the requested base seed, then give every
        // replanning request a fresh deterministic seed: base+1, base+2, ...
        settings.random_seed = replanning_base_seed
            + static_cast<unsigned long long>(request_index);
        int response_request_index = request_index;
        json response;
        try {
            const json request = json::parse(request_line);
            response_request_index = request.value(
                "request_index", request_index
            );

            json problem = problem_template;
            robots::G1::Configuration projected_start =
                request.at("start").get<robots::G1::Configuration>();
            settings.g1_constraints =
                g1_constraint_parameters_from_problem(problem);
            if (!PATACON::project_g1_configuration(
                    projected_start,
                    settings
                )) {
                throw std::runtime_error(
                    "G1 replanning start projection failed"
                );
            }

            problem["start"] = projected_start;
            problem["goals"] = json::array({request.at("goal")});
            problem["axis_endpoints"]["start"] =
                projected_start;
            problem["axis_endpoints"]["goals"] =
                json::array({request.at("goal")});
            if (request.contains("environment")) {
                const auto &request_environment = request.at("environment");
                problem["sphere"] = request_environment.at("sphere");
                problem["cylinder"] = request_environment.at("cylinder");
                problem["box"] = request_environment.at("box");
            }

            auto environment = ppln::config::environment_from_problem_json(
                problem,
                "g1_continuous_replan"
            );
            const int status = run_planner<robots::G1>(
                problem,
                environment,
                settings,
                false,
                false,
                false,
                false,
                "g1",
                "g1_continuous_replan",
                1,
                "",
                no_trace,
                1,
                no_nested_replanning,
                false,
                &response
            );
            if (status != 0) {
                response = {
                    {"solved", false},
                    {"server_error", "planner returned nonzero status"}
                };
            }
        } catch (const std::exception &error) {
            response = {
                {"solved", false},
                {"server_error", error.what()}
            };
        }

        response["request_index"] = response_request_index;
        response["seed"] = settings.random_seed;
        std::cout << "G1_REPLAN_RESULT " << response.dump() << "\n";
        std::cout.flush();
        std::cerr.flush();
    }
    PATACON::release_persistent_workspace();
    PATACON::set_persistent_workspace_enabled(false);
    return 0;
}

int main(int argc, char* argv[]) {
    std::string robot_name;
    std::string name;
    int problem_idx = 1;
    bool visualize = false;
    bool path_smoothing = true;
    bool plot = false;
    bool trace_trees = false;
    bool axis = false;
    bool projection_smoothness = true;
    bool print_path = true;
    bool collect_diagnostics = false;
    bool validate_config_only = false;
    bool aorrtc = false;
    bool enable_com_constraint = false;
    bool object_mass_option_provided = false;
    bool support_margin_option_provided = false;
    bool time_option_provided = false;
    bool range_option_provided = false;
    bool axis_option_provided = false;
    float object_mass_kg = 0.0f;
    float support_margin_m = 0.0f;
    float planner_range = 0.4f;
    double time_limit_sec = 5.0;
    unsigned long long random_seed = 1ULL;
    int runs = 1;
    int max_concon_nodes = 4;
    std::string problem_file_path;
    std::string save_json_path;
    TraceExportOptions trace_options;
    G1ReplanningOptions g1_replanning;
    std::optional<ppln::config::PlanningProblem> selected_problem;
    const bool standalone_config_mode = argc >= 2 &&
        std::string(argv[1]) == "--config";
    int option_start = 4;

    if ((!standalone_config_mode && argc < 4) ||
        (standalone_config_mode && argc < 3)) {
        std::cout
            << "Usage: ./single_mbm <robot_name> <problem_name> <problem_idx> "
            << "[--visualize] [--replanning] "
            << "[--save-json PATH] [--run N|--runs N] "
            << "[--problem-file PATH] "
            << "[--seed N] [--range VALUE] "
            << "[--aorrtc] [--time SECONDS] [--plot] "
            << "[--com] [--object-mass-kg KG] [--support-margin M] "
            << "[--axis] "
            << "[--validate-config] "
            << "[--diagnostics] "
            << "[--no-path-smoothing] "
            << "[--no-waypoint-smoothing] "
            << "[--max-concon-nodes N] "
            << "[--trace-mode auto|path|tree] "
            << "[--html-trace-mode path|tree] "
            << "[--no-print-path] "
            << "[--html-max-tree-nodes N] [--graphml PATH] [--html PATH] "
            << "[--patacon-root PATH]\n"
            << "       ./single_mbm --config <planning.json> [options]\n";
        return 1;
    }
    try {
        if (standalone_config_mode) {
            problem_file_path = argv[2];
            selected_problem = ppln::config::load_selected_problem(
                problem_file_path
            );
            robot_name = selected_problem->robot_name;
            name = selected_problem->problem_name;
            problem_idx = selected_problem->problem_index;
            option_start = 3;
        } else {
            robot_name = argv[1];
            name = argv[2];
            problem_idx = std::stoi(argv[3]);
        }
    } catch (const std::exception &error) {
        std::cerr << "single_mbm config error: " << error.what() << "\n";
        return 1;
    }
    auto parse_positive_double = [](const std::string &option, const std::string &value) {
        std::size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
        if (consumed != value.size()
            || !std::isfinite(parsed)
            || parsed <= 0.0) {
            throw std::invalid_argument(
                option + " must be a finite number greater than 0"
            );
        }
        return parsed;
    };
    auto parse_nonnegative_double = [](const std::string &option, const std::string &value) {
        std::size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
        if (consumed != value.size()
            || !std::isfinite(parsed)
            || parsed < 0.0) {
            throw std::invalid_argument(
                option + " must be a finite number greater than or equal to 0"
            );
        }
        return parsed;
    };
    auto parse_seed = [](const std::string &value) {
        if (value.empty() || value.front() == '-') {
            throw std::invalid_argument(
                "--seed must be a nonnegative integer"
            );
        }
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument(
                "--seed must be a nonnegative integer"
            );
        }
        return parsed;
    };
    try {
        for (int index = option_start; index < argc; index++) {
            const std::string argument = argv[index];
            if (argument == "--visualize") {
                visualize = true;
            } else if (argument == "--replanning") {
                g1_replanning.enabled = true;
            } else if (argument == "--replan-server") {
                g1_replanning.server = true;
            } else if (argument == "--no-path-smoothing") {
                path_smoothing = false;
            } else if (argument == "--save-json" && index + 1 < argc) {
                save_json_path = argv[++index];
            } else if (argument == "--problem-file" && index + 1 < argc) {
                if (standalone_config_mode) {
                    throw std::invalid_argument(
                        "--problem-file cannot be combined with --config"
                    );
                }
                problem_file_path = argv[++index];
            } else if ((argument == "--run" || argument == "--runs") && index + 1 < argc) {
                runs = std::max(1, std::stoi(argv[++index]));
            } else if (argument == "--seed" && index + 1 < argc) {
                random_seed = parse_seed(argv[++index]);
            } else if (argument == "--range" && index + 1 < argc) {
                planner_range = static_cast<float>(
                    parse_positive_double(argument, argv[++index])
                );
                range_option_provided = true;
            } else if (
                (
                    argument == "--max-concon-nodes" ||
                    argument == "--concon-nodes"
                ) &&
                index + 1 < argc
            ) {
                max_concon_nodes = std::stoi(argv[++index]);
                if (max_concon_nodes <= 0) {
                    throw std::invalid_argument(
                        "--max-concon-nodes must be positive"
                    );
                }
            } else if (argument == "--plot") {
                plot = true;
            } else if (argument == "--aorrtc") {
                aorrtc = true;
            } else if (argument == "--com") {
                enable_com_constraint = true;
            } else if (
                (
                    argument == "--object-mass-kg" ||
                    argument == "--object-mass"
                ) &&
                index + 1 < argc
            ) {
                const std::string value = argv[++index];
                std::size_t consumed = 0;
                const double parsed_mass = std::stod(value, &consumed);
                if (
                    consumed != value.size() ||
                    !std::isfinite(parsed_mass) ||
                    parsed_mass < 0.0
                ) {
                    throw std::invalid_argument(
                        "--object-mass-kg must be a finite number >= 0"
                    );
                }
                object_mass_kg = static_cast<float>(parsed_mass);
                object_mass_option_provided = true;
            } else if (
                (
                    argument == "--support-margin" ||
                    argument == "--support-margin-m"
                ) &&
                index + 1 < argc
            ) {
                support_margin_m = static_cast<float>(
                    parse_nonnegative_double(argument, argv[++index])
                );
                support_margin_option_provided = true;
            } else if (argument == "--time" && index + 1 < argc) {
                const std::string value = argv[++index];
                std::size_t consumed = 0;
                time_limit_sec = std::stod(value, &consumed);
                time_option_provided = true;
                if (consumed != value.size()
                    || !std::isfinite(time_limit_sec)
                    || time_limit_sec <= 0.0) {
                    throw std::invalid_argument(
                        "--time must be a finite number greater than 0"
                    );
                }
            } else if (argument == "--trace-trees") {
                trace_trees = true;
            } else if (argument == "--trace-mode" && index + 1 < argc) {
                trace_options.trace_mode = argv[++index];
                trace_options.requested = true;
                if (trace_options.trace_mode != "auto"
                    && trace_options.trace_mode != "path"
                    && trace_options.trace_mode != "tree") {
                    throw std::invalid_argument(
                        "--trace-mode must be one of: auto, path, tree"
                    );
                }
            } else if (argument == "--html-trace-mode" && index + 1 < argc) {
                trace_options.html_trace_mode = argv[++index];
                trace_options.requested = true;
                if (trace_options.html_trace_mode != "path"
                    && trace_options.html_trace_mode != "tree") {
                    throw std::invalid_argument(
                        "--html-trace-mode must be one of: path, tree"
                    );
                }
            } else if (argument == "--html-max-tree-nodes" && index + 1 < argc) {
                trace_options.html_max_tree_nodes = std::max(
                    0,
                    std::stoi(argv[++index])
                );
                trace_options.requested = true;
            } else if (argument == "--graphml" && index + 1 < argc) {
                trace_options.graphml_path = argv[++index];
                trace_options.requested = true;
            } else if (argument == "--html" && index + 1 < argc) {
                trace_options.html_path = argv[++index];
                trace_options.requested = true;
            } else if (argument == "--path-key" && index + 1 < argc) {
                trace_options.path_key = argv[++index];
                trace_options.requested = true;
            } else if (argument == "--patacon-root" && index + 1 < argc) {
                trace_options.patacon_root = argv[++index];
                trace_options.requested = true;
            }else if (argument == "--axis") {
                axis = true;
                axis_option_provided = true;
            }
            else if (argument == "--diagnostics") {
                collect_diagnostics = true;
            }
            else if (argument == "--validate-config") {
                validate_config_only = true;
            }
            else if (argument == "--no-waypoint-smoothing") {
                projection_smoothness = false;
            }
            else if (argument == "--no-print-path") {
                print_path = false;
            }
            else {
                throw std::invalid_argument("unknown or incomplete option: " + argument);
            }
        }
        if (time_option_provided && !aorrtc) {
            throw std::invalid_argument("--time requires --aorrtc");
        }
        if (g1_replanning.enabled) {
            if (robot_name != "g1") {
                throw std::invalid_argument(
                    "--replanning is supported only for G1"
                );
            }
            if (runs != 1) {
                throw std::invalid_argument(
                    "--replanning requires one planner run"
                );
            }
            visualize = true;
            path_smoothing = false;
            g1_replanning.planner_executable =
                std::filesystem::absolute(argv[0]).string();
        }
        if (g1_replanning.server
            && (robot_name != "g1"
                || visualize
                || g1_replanning.enabled
                || runs != 1
                || plot
                || trace_trees
                || !save_json_path.empty())) {
            throw std::invalid_argument(
                "--replan-server is an internal G1 non-visual mode"
            );
        }
        if (collect_diagnostics && aorrtc) {
            throw std::invalid_argument(
                "--diagnostics currently supports PATACON/TB-RRT only"
            );
        }
        if (
            random_seed > std::numeric_limits<unsigned long long>::max()
                - static_cast<unsigned long long>(runs - 1)
        ) {
            throw std::invalid_argument(
                "--seed plus the run count exceeds the supported seed range"
            );
        }
        if (!path_smoothing && !visualize) {
            throw std::invalid_argument(
                "--no-path-smoothing requires --visualize"
            );
        }
        if (
            enable_com_constraint &&
            robot_name != "ffw_sg2_mobility"
        ) {
            throw std::invalid_argument(
                "--com is supported only for ffw_sg2_mobility"
            );
        }
        if (
            object_mass_option_provided &&
            robot_name != "ffw_sg2_mobility" &&
            robot_name != "igris_c"
        ) {
            throw std::invalid_argument(
                "--object-mass-kg is supported only for "
                "ffw_sg2_mobility and igris_c"
            );
        }
        if (
            support_margin_option_provided &&
            robot_name != "ffw_sg2_mobility"
        ) {
            throw std::invalid_argument(
                "--support-margin is supported only for ffw_sg2_mobility"
            );
        }
        if (object_mass_option_provided && object_mass_kg <= 0.0f) {
            throw std::invalid_argument(
                "--object-mass-kg must be greater than 0"
            );
        }

    } catch (const std::exception &error) {
        std::cerr << "single_mbm option error: " << error.what() << "\n";
        return 1;
    }

    if (trace_options.trace_mode == "tree"
        || trace_options.html_trace_mode == "tree"
        || trace_trees) {
        trace_trees = true;
    }
    if (trace_options.requested && save_json_path.empty()) {
        save_json_path = default_trace_result_json_path(
            trace_options,
            robot_name,
            name,
            problem_idx
        );
    }
    if (!ppln::config::is_supported_robot(robot_name)) {
        std::cerr << "Unsupported robot type: " << robot_name << "\n";
        return 1;
    }
    const std::string path = problem_file_path.empty()
        ? ppln::config::default_problem_file(robot_name)
        : problem_file_path;
    try {
        if (!selected_problem.has_value()) {
            selected_problem = ppln::config::load_selected_problem(
                path, robot_name, name, problem_idx
            );
        }
    } catch (const std::exception &error) {
        std::cerr << "single_mbm config error: " << error.what() << "\n";
        return 1;
    }
    json data = selected_problem->data;
    if (data.contains("planner")) {
        const auto &planner = data.at("planner");
        if (!planner.is_object()) {
            std::cerr << "single_mbm config error: planner must be an object\n";
            return 1;
        }
        if (!range_option_provided && planner.contains("range")) {
            try {
                planner_range = static_cast<float>(
                    parse_positive_double(
                        "planner.range",
                        planner.at("range").dump()
                    )
                );
            } catch (const std::exception &error) {
                std::cerr << "single_mbm config error: "
                          << error.what() << "\n";
                return 1;
            }
        }
        if (!axis_option_provided &&
            planner.contains("axis")) {
            try {
                axis =
                    planner.at("axis").get<bool>();
            } catch (const std::exception &) {
                std::cerr << "single_mbm config error: "
                          << "planner.axis must be boolean\n";
                return 1;
            }
        }
    }
    if (!axis_option_provided && robot_name == "g1" &&
        data.contains("constraints") &&
        data.at("constraints").contains("bimanual_axis")) {
        axis = true;
    }
    if (validate_config_only) {
        std::cout << "config_valid: " << path << "\n"
                  << "robot: " << robot_name << "\n"
                  << "dimension: "
                  << ppln::config::compiled_robot_dimension(robot_name) << "\n"
                  << "problem: " << name << "\n"
                  << "problem_index: " << problem_idx << "\n";
        return 0;
    }
    if (not data["valid"]) {
        return -1;
    }
    if (g1_replanning.enabled && robot_name == "g1") {
        if (!data.contains("replanning_endpoints")) {
            throw std::invalid_argument(
                "G1 --replanning requires replanning_endpoints"
            );
        }
        const json replanning_endpoints = data.at("replanning_endpoints");
        data["start"] = replanning_endpoints.at("start");
        data["goals"] = replanning_endpoints.at("goals");
        data["axis_endpoints"]["start"] =
            replanning_endpoints.at("start");
        data["axis_endpoints"]["goals"] =
            replanning_endpoints.at("goals");

        // In replanning mode the moving sphere is the only world obstacle.
        // The MuJoCo floor remains part of the robot model.
        data["sphere"] = json::array();
        data["cylinder"] = json::array();
        data["box"] = json::array();
        data["sphere"].push_back(
            {
                {"name", "mouse_dynamic_obstacle"},
                {"position", json::array({0.400, 0.200, 0.800})},
                {"radius", 0.040}
            }
        );
    }
    auto env = ppln::config::environment_from_problem_json(data, name);
    AORRTC_settings settings;
    settings.num_new_configs = 512; //usually:512
    settings.max_iters = 100000000;
    settings.random_seed = random_seed;
    settings.aorrtc = aorrtc;
    settings.time_limit_sec = time_limit_sec;
    settings.granularity = 16;
    settings.range = planner_range;
    settings.lift_distance_weight = 1.0f;
    settings.ffw_sg2_enable_com_constraint = enable_com_constraint;
    settings.axis = axis;
    settings.projection_smoothness = projection_smoothness;
    settings.balance = 2;
    settings.tree_ratio = 1.0;
    settings.trace_trees = trace_trees;
    settings.collect_diagnostics = collect_diagnostics;
    // Always use the TB-RRT forward-half-space rule for PATACON and AORRTC.
    settings.prevent_ts_backtracking = true;
    settings.em_threshold = 0.1f;
    settings.max_concon_nodes = max_concon_nodes;
    settings.max_connect_concon_chunks = 16;

    if (data.contains("constraints") && data["constraints"].contains("com")) {
        const auto &com_constraints = data["constraints"]["com"];
        if (com_constraints.value("enabled", false)) {
            settings.ffw_sg2_enable_com_constraint = true;
        }
        if (com_constraints.contains("support_margin_m")) {
            settings.ffw_sg2_support_margin_m =
                static_cast<float>(
                    com_constraints["support_margin_m"].get<double>()
                );
        }
        if (com_constraints.contains("object_mass_kg")) {
            settings.ffw_sg2_object_mass_kg =
                static_cast<float>(
                    com_constraints["object_mass_kg"].get<double>()
                );
        }
    }
    if (object_mass_option_provided) {
        settings.ffw_sg2_object_mass_kg = object_mass_kg;
    } else if (
        enable_com_constraint &&
        robot_name == "ffw_sg2_mobility" &&
        settings.ffw_sg2_object_mass_kg <= 0.0f
    ) {
        settings.ffw_sg2_object_mass_kg = FFW_SG2_DEFAULT_OBJECT_MASS_KG;
    }
    if (support_margin_option_provided) {
        settings.ffw_sg2_support_margin_m = support_margin_m;
    } else if (
        enable_com_constraint &&
        robot_name == "ffw_sg2_mobility" &&
        settings.ffw_sg2_support_margin_m <= 0.0f
    ) {
        settings.ffw_sg2_support_margin_m = FFW_SG2_DEFAULT_SUPPORT_MARGIN_M;
    }

    try {
        if (g1_replanning.server) {
            return run_g1_replan_server(data, settings, g1_replanning);
        }
        return ppln::planning::dispatch_robot(
            robot_name,
            [&](auto robot_tag) {
                using Robot = typename decltype(robot_tag)::type;
                return run_planner<Robot>(
                    data,
                    env,
                    settings,
                    visualize,
                    path_smoothing,
                    print_path,
                    plot,
                    robot_name,
                    name,
                    problem_idx,
                    save_json_path,
                    trace_options,
                    runs,
                    g1_replanning
                );
            }
        );
    } catch (const std::exception &error) {
        std::cerr << "single_mbm error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

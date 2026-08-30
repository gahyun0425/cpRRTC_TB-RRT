#include <nlohmann/json.hpp>
#include <algorithm>
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

#include <cuda_runtime.h>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/planning/Planners.hh"
#include "src/planning/AORRTC.hh"
#include "src/planning/pRRTC_settings.hh"
#include "scripts/g1_problem.hh"
#include "scripts/planner_result_json.hh"

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

template <typename Robot>
void plot_aorrtc_cost_history(
    const AORRTCResult<Robot> &result,
    const std::string &robot_name,
    const std::string &problem_name,
    int problem_index,
    int run_index
) {
    // AORRTC가 solution을 하나도 찾지 못한 경우
    if (result.solution_history.empty()) {
        std::cout
            << "AORRTC plot skipped: no solution history.\n";
        return;
    }

    // 임시 CSV 파일 이름을 겹치지 않게 생성
    const auto timestamp =
        std::chrono::steady_clock::now()
            .time_since_epoch()
            .count();

    const auto csv_path =
        std::filesystem::temp_directory_path()
        / (
            "aorrtc_cost_history_"
            + std::to_string(timestamp)
            + ".csv"
        );

    // CSV 파일 생성
    std::ofstream csv(csv_path);

    if (!csv) {
        throw std::runtime_error(
            "failed to create temporary AORRTC plot CSV"
        );
    }

    csv << "time_sec,cost\n";
    csv << std::setprecision(12);

    for (const auto &update : result.solution_history) {
        const double time_sec =
            static_cast<double>(update.found_ns) / 1.0e9;

        csv
            << time_sec
            << ","
            << update.cost
            << "\n";
    }

    csv.close();

    // Python plotting script의 절대 경로
    const auto script_path =
        std::filesystem::absolute(
            "scripts/plot_aorrtc.py"
        );

    const std::string title =
        "AORRTC Cost Convergence - "
        + robot_name
        + " / "
        + problem_name
        + " #"
        + std::to_string(problem_index)
        + " / run "
        + std::to_string(run_index);

    // python3 scripts/plot_aorrtc.py <csv> --title "..."
    const std::string command =
        "python3 "
        + shell_quote(script_path.string())
        + " "
        + shell_quote(csv_path.string())
        + " --title "
        + shell_quote(title);

    std::cout
        << "plotting AORRTC cost history...\n";

    std::cout.flush();
    std::cerr.flush();

    const int status =
        std::system(command.c_str());

    // Python이 끝났으면 임시 CSV 삭제
    std::error_code remove_error;
    std::filesystem::remove(
        csv_path,
        remove_error
    );

    if (status != 0) {
        throw std::runtime_error(
            "AORRTC plotting script exited with an error"
        );
    }
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
        << "python3 scripts/prrtc_path_trace.py "
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


Environment<float> problem_dict_to_env(const json& problem, const std::string& name) {
    Environment<float> env{};

    std::vector<Sphere<float>> spheres;
    std::vector<Capsule<float>> capsules;
    std::vector<Cuboid<float>> cuboids;
    // Fill spheres
    for (const auto& obj : problem["sphere"]) {
        const json& position = obj["position"];
        Sphere<float> sphere(position[0], position[1], position[2], obj["radius"]);
        sphere.name = obj["name"];
        spheres.push_back(sphere);
    }
    // Handle cylinders based on name
    if (name == "box") {
        for (const auto& obj : problem["cylinder"]) {
            const json& position = obj["position"];
            const json& orientation = obj["orientation_euler_xyz"];
            const float radius = obj["radius"];
            const std::array<float, 3> dims = {radius, radius, radius/2.0f};
            auto cuboid = factory::cuboid::array(
                position, orientation,
                dims
            );
            cuboid.name = obj["name"];
            cuboids.push_back(cuboid);
        }
    } else {
        for (const auto& obj : problem["cylinder"]) {
            const json& position = obj["position"];
            const json& orientation = obj["orientation_euler_xyz"];
            const float radius = obj["radius"];
            const float length = obj["length"];
            auto cylinder = factory::cylinder::center::array(
                position, orientation,
                radius, length
            );
            cylinder.name = obj["name"];
            capsules.push_back(cylinder);
        }
    }

    // Fill boxes
    for (const auto& obj : problem["box"]) {
        const json& position = obj["position"];
        const json& orientation = obj["orientation_euler_xyz"];
        const json& half_extents = obj["half_extents"];
        auto cuboid = factory::cuboid::array(
            position, orientation, half_extents
        );
        cuboid.name = obj["name"];
        cuboids.push_back(cuboid);
    }

    // Allocate memory on the heap for the arrays
    if (!spheres.empty()) {
        env.spheres = new Sphere<float>[spheres.size()];
        std::copy(spheres.begin(), spheres.end(), env.spheres);
        env.num_spheres = spheres.size();
    }

    if (!capsules.empty()) {
        env.capsules = new Capsule<float>[capsules.size()];
        std::copy(capsules.begin(), capsules.end(), env.capsules);
        env.num_capsules = capsules.size();
    }

    if (!cuboids.empty()) {
        env.cuboids = new Cuboid<float>[cuboids.size()];
        std::copy(cuboids.begin(), cuboids.end(), env.cuboids);
        env.num_cuboids = cuboids.size();
    }

    return env;
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
void visualize_ffw_sg2_path(
    const PlannerResult<Robot> &result,
    const typename Robot::Configuration &start,
    const std::vector<std::string> &joint_names
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
        / ("prrtc_" + std::string(Robot::name) + "_trajectory_"
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
    const auto model_path = std::filesystem::absolute(
        mobility_model
            ? "ffw_lift/ffw_sg2_rack_upper_to_lower.xml"
            : "ffw_lift/ffw_sg2_lift.xml"
    );
    const std::string command =
        "python3 \"" + visualizer_path.string() + "\""
        + " --model \"" + model_path.string() + "\""
        + " --trajectory \"" + trajectory_path.string() + "\"";

    std::cout.flush();
    std::cerr.flush();
    const int status = std::system(command.c_str());
    std::error_code remove_error;
    std::filesystem::remove(trajectory_path, remove_error);
    if (status != 0) {
        throw std::runtime_error("MuJoCo visualizer exited with an error");
    }
}


void visualize_g1_path(
    const PlannerResult<robots::G1> &result,
    const robots::G1::Configuration &start,
    const json &problem
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
    trajectory["environment"] = {
        {"sphere", problem.value("sphere", json::array())},
        {"cylinder", problem.value("cylinder", json::array())},
        {"box", problem.value("box", json::array())}
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
        / ("prrtc_g1_trajectory_" + std::to_string(timestamp) + ".json");
    {
        std::ofstream trajectory_file(trajectory_path);
        if (!trajectory_file) {
            throw std::runtime_error("failed to create temporary G1 trajectory");
        }
        trajectory_file << trajectory.dump(2) << '\n';
    }

    const auto visualizer_path = std::filesystem::absolute(
        "scripts/visualize_g1.py"
    );
    const std::string command =
        "python3 " + shell_quote(visualizer_path.string())
        + " --trajectory " + shell_quote(trajectory_path.string());

    std::cout.flush();
    std::cerr.flush();
    const int status = std::system(command.c_str());
    std::error_code remove_error;
    std::filesystem::remove(trajectory_path, remove_error);
    if (status != 0) {
        throw std::runtime_error("G1 MuJoCo visualizer exited with an error");
    }
}


template <typename Robot>
int run_planner(
    json &data,
    Environment<float> &env,
    AORRTC_settings &settings,
    bool visualize,
    bool print_path,
     bool plot,
    const std::string &robot_name,
    const std::string &problem_name,
    int problem_index,
    const std::string &save_json_path,
    const TraceExportOptions &trace_options,
    int runs
) {
    using Configuration = typename Robot::Configuration;
    Configuration start = data["start"];
    std::vector<Configuration> goals = data["goals"];
    json saved_results = json::array();
    int solved_count = 0;
    std::vector<double> times_sec;
    std::vector<int> path_lengths;
    std::vector<float> costs;
    PlannerResult<Robot> visualization_result;

    for (int run_index = 1; run_index <= runs; run_index++) {
        if (runs > 1) {
            std::cout << "run: " << run_index << "\n";
        }

        const std::size_t warmup_ns = measure_planner_warmup_ns();
        AORRTCResult<Robot> result;
        if (settings.aorrtc) {
            result = AORRTC::solve<Robot>(start, goals, env, settings);
        } else {
            static_cast<PlannerResult<Robot> &>(result) =
                pRRTC::solve<Robot>(start, goals, env, settings);
        }
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
            // 기존 pRRTC 동작은 그대로 유지
            planning_sec =
                static_cast<double>(result.kernel_ns) / 1.0e9;
        }

        const double warmup_sec = static_cast<double>(warmup_ns) / 1.0e9;

        if (result.solved) {
            times_sec.push_back(planning_sec);
        }
        std::cout << "cost: " << result.cost << "\n";
        if (runs > 1) {
            std::cout << "warmup_s: " << warmup_sec << "\n";
        }
        std::cout << "planning_s: " << planning_sec << "\n";
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

        if (!save_json_path.empty()) {
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

        if (plot&& settings.aorrtc && run_index == runs) {

            plot_aorrtc_cost_history(
                result,
                robot_name,
                problem_name,
                problem_index,
                run_index
            );
        }

        if (visualize && run_index == runs) {
            visualization_result = std::move(result);
        }
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
                {"format", "pRRTC_run_results_v1"},
                {"planner", "pRRTC"},
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
            visualize_ffw_sg2_path(visualization_result, start, {
                "lift_joint",
                "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
                "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
                "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
                "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"
            });
        } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
            visualize_ffw_sg2_path(visualization_result, start, {
                "base_x", "base_y", "base_yaw",
                "lift_joint",
                "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
                "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
                "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
                "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"
            });
        } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Single>) {
            visualize_ffw_sg2_path(visualization_result, start, {
                "lift_joint",
                "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
                "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"
            });
        } else if constexpr (std::is_same_v<Robot, robots::G1>) {
            visualize_g1_path(visualization_result, start, data);
        } else {
            throw std::runtime_error(
                "--visualize supports only ffw_sg2, ffw_sg2_mobility, ffw_sg2_single, and g1"
            );
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    std::string robot_name = "panda";
    std::string name = "cage";
    int problem_idx = 1;
    bool visualize = false;
    bool plot = false;
    bool trace_trees = false;
    bool rigid_orientation = false;
    bool projection_smoothness = true;
    bool print_path = true;
    bool aorrtc = false;
    bool enable_com_constraint = false;
    bool object_mass_option_provided = false;
    bool time_option_provided = false;
    float object_mass_kg = 0.0f;
    double time_limit_sec = 5.0;
    int runs = 1;
    std::string save_json_path;
    TraceExportOptions trace_options;

    if (argc < 4) {
        std::cout
            << "Usage: ./single_mbm <robot_name> <problem_name> <problem_idx> "
            << "[--visualize] [--save-json PATH] [--run N|--runs N] "
            << "[--aorrtc] [--time SECONDS] [--plot] "
            << "[--com] [--object-mass-kg KG] "
            << "[--rigid-orientation] "
            << "[--no-waypoint-smoothing] "
            << "[--trace-mode auto|path|tree] "
            << "[--html-trace-mode path|tree] "
            << "[--no-print-path] "
            << "[--html-max-tree-nodes N] [--graphml PATH] [--html PATH] "
            << "[--patacon-root PATH]\n";
        return 1;
    }
    robot_name = argv[1];
    name = argv[2];
    try {
        problem_idx = std::stoi(argv[3]);
        for (int index = 4; index < argc; index++) {
            const std::string argument = argv[index];
            if (argument == "--visualize") {
                visualize = true;
            } else if (argument == "--save-json" && index + 1 < argc) {
                save_json_path = argv[++index];
            } else if ((argument == "--run" || argument == "--runs") && index + 1 < argc) {
                runs = std::max(1, std::stoi(argv[++index]));
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
            }else if (
                argument == "--rigid-orientation" ||
                argument == "--rigid-orientiation"
            ) {
                rigid_orientation = true;
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

        if (plot && !aorrtc) {
            throw std::invalid_argument(
                "--plot requires --aorrtc"
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
            robot_name != "ffw_sg2_mobility"
        ) {
            throw std::invalid_argument(
                "--object-mass-kg is supported only for ffw_sg2_mobility"
            );
        }
        if (object_mass_option_provided && !enable_com_constraint) {
            throw std::invalid_argument(
                "--object-mass-kg requires --com"
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
    if (visualize
        && robot_name != "ffw_sg2"
        && robot_name != "ffw_sg2_mobility"
        && robot_name != "ffw_sg2_single"
        && robot_name != "g1") {
        std::cerr << "--visualize supports only ffw_sg2, ffw_sg2_mobility, ffw_sg2_single, and g1\n";
        return 1;
    }
    std::string path = "scripts/" + robot_name + "_problems.json";
    std::ifstream f(path);
    if (!f) {
        std::cerr << "Failed to open problem file: " << path << "\n";
        return 1;
    }
    json all_data;
    try {
        all_data = json::parse(f);
    } catch (const std::exception &error) {
        std::cerr << "Failed to parse problem file: " << error.what() << "\n";
        return 1;
    }
    if (!all_data.contains("problems")
        || !all_data["problems"].contains(name)
        || problem_idx < 1
        || problem_idx > static_cast<int>(all_data["problems"][name].size())) {
        std::cerr << "Unknown problem or problem index: " << name
                  << " " << problem_idx << "\n";
        return 1;
    }
    json data = all_data["problems"][name][problem_idx - 1];
    if (not data["valid"]) {
        return -1;
    }
    auto env = problem_dict_to_env(data, name);
    AORRTC_settings settings;
    settings.num_new_configs = 512; //usually:512
    settings.max_iters = 100000000;
    settings.aorrtc = aorrtc;
    settings.time_limit_sec = time_limit_sec;
    settings.granularity = 16;
    settings.range = 0.4;
    settings.lift_distance_weight = 1.0f;
    settings.ffw_sg2_enable_com_constraint = enable_com_constraint;
    settings.rigid_orientation = rigid_orientation;
    settings.projection_smoothness = projection_smoothness;
    settings.balance = 2;
    settings.tree_ratio = 1.0;
    settings.dynamic_domain = false;
    settings.trace_trees = trace_trees;
    settings.dd_radius = 4.0;
    settings.dd_min_radius = 1.0;
    settings.dd_alpha = 0.0001;
    settings.em_threshold = 0.1f;
    settings.max_concon_nodes = 4;
    settings.max_connect_concon_chunks = 16;

    if (data.contains("constraints") && data["constraints"].contains("com")) {
        const auto &com_constraints = data["constraints"]["com"];
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
    }

    try {
        if (robot_name == "g1") {
            settings.granularity = robots::G1::resolution;
            settings.g1_constraints = g1_constraint_parameters_from_problem(data);
        }
        if (robot_name == "fetch") {
            return run_planner<robots::Fetch>(data, env, settings, visualize, print_path, plot,
                robot_name, name, problem_idx, save_json_path, trace_options, runs);
        } else if (robot_name == "panda") {
            return run_planner<robots::Panda>(data, env, settings, visualize, print_path, plot,
                robot_name, name, problem_idx, save_json_path, trace_options, runs);
        } else if (robot_name == "baxter") {
            return run_planner<robots::Baxter>(data, env, settings, visualize, print_path, plot,
                robot_name, name, problem_idx, save_json_path, trace_options, runs);
        } else if (robot_name == "ffw_sg2") {
            return run_planner<robots::FfwSg2>(data, env, settings, visualize, print_path, plot,
                robot_name, name, problem_idx, save_json_path, trace_options, runs);
        } else if (robot_name == "ffw_sg2_mobility") {
            return run_planner<robots::FfwSg2Mobility>(data, env, settings, visualize, print_path, plot,
                robot_name, name, problem_idx, save_json_path, trace_options, runs);
        } else if (robot_name == "ffw_sg2_single") {
            return run_planner<robots::FfwSg2Single>(data, env, settings, visualize, print_path, plot,
                robot_name, name, problem_idx, save_json_path, trace_options, runs);
        } else if (robot_name == "g1") {
            return run_planner<robots::G1>(data, env, settings, visualize, print_path, plot,
                robot_name, name, problem_idx, save_json_path, trace_options, runs);
        } else {
            std::cerr << "Unsupported robot type: " << robot_name << "\n";
            return 1;
        }
    } catch (const std::exception &error) {
        std::cerr << "single_mbm error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}

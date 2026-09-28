#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/planning/AORRTC.hh"
#include "src/planning/Planners.hh"
#include "src/planning/pRRTC_settings.hh"
#include "src/config/PlanningProblemJson.hh"
#include "src/constraints/RobotConstraintAdapter.hh"
#include "scripts/ffw_sg2_attached_object_collision.hh"
#include "scripts/planner_result_json.hh"

using json = nlohmann::json;
using namespace ppln::collision;

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

void plot_aorrtc_problem_set(
    const std::filesystem::path &input_path,
    const std::string &robot_name,
    const std::string &run_name,
    int problem_count
) {
    const auto output_path = std::filesystem::absolute(
        std::filesystem::path("logs")
        / (
            "aorrtc_" + filename_component(robot_name)
            + "_" + filename_component(run_name)
            + "_" + std::to_string(problem_count)
            + "problems_convergence.png"
        )
    );
    std::filesystem::create_directories(output_path.parent_path());

    const auto script_path = std::filesystem::absolute(
        "scripts/plot_aorrtc.py"
    );
    const std::string title =
        "AORRTC - " + robot_name + " / " + run_name
        + " (" + std::to_string(problem_count) + " problems)";
    const std::string command =
        "python3 "
        + shell_quote(script_path.string())
        + " "
        + shell_quote(std::filesystem::absolute(input_path).string())
        + " --output "
        + shell_quote(output_path.string())
        + " --title "
        + shell_quote(title);

    std::cout << "plotting AORRTC problem-set convergence...\n";
    std::cout.flush();
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error(
            "AORRTC problem-set plotting script exited with an error"
        );
    }
    std::cout << "aorrtc_plot: " << output_path.string() << "\n";
}

template <typename Robot>
json ordered_trajectory_json(
    const PlannerResult<Robot> &result,
    const typename Robot::Configuration &start,
    const std::string &label
) {
    if (result.path.size() < 2) {
        throw std::runtime_error("cannot visualize an unsolved or empty path");
    }
    auto squared_distance = [](const auto &left, const auto &right) {
        float squared = 0.0f;
        for (std::size_t index = 0; index < left.size(); ++index) {
            const float difference = left[index] - right[index];
            squared += difference * difference;
        }
        return squared;
    };
    json trajectory = {
        {"label", label},
        {"start", start},
        {"waypoints", json::array()},
    };
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
    return trajectory;
}

json g1_trajectory_json(
    const PlannerResult<robots::G1> &result,
    const robots::G1::Configuration &start,
    const json &problem,
    double planning_time_sec,
    const std::string &label
) {
    json trajectory = ordered_trajectory_json<robots::G1>(
        result, start, label
    );
    trajectory["planning_time_sec"] = planning_time_sec;
    trajectory["environment"] = {
        {"sphere", problem.value("sphere", json::array())},
        {"cylinder", problem.value("cylinder", json::array())},
        {"box", problem.value("box", json::array())}
    };
    const auto &constraints = problem.at("constraints");
    const auto &center_of_mass = constraints.at("com");
    if (center_of_mass.contains("payload")) {
        trajectory["payload"] = center_of_mass.at("payload");

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

    return trajectory;
}

void visualize_g1_paths(const json &trajectories) {
    if (trajectories.empty()) {
        throw std::runtime_error("cannot visualize an empty G1 path set");
    }
    const json bundle = {{"trajectories", trajectories}};
    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto trajectory_path = std::filesystem::temp_directory_path()
        / ("prrtc_g1_trajectories_" + std::to_string(timestamp) + ".json");
    {
        std::ofstream trajectory_file(trajectory_path);
        if (!trajectory_file) {
            throw std::runtime_error(
                "failed to create temporary G1 trajectory"
            );
        }
        trajectory_file << bundle.dump(2) << '\n';
    }

    const auto visualizer_path = std::filesystem::absolute(
        "scripts/visualize_g1.py"
    );
    const std::string command =
        "python3 " + shell_quote(visualizer_path.string())
        + " --trajectory " + shell_quote(trajectory_path.string())
        + " --control-mode qpos";

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
void visualize_franka_paths(
    const json &trajectories,
    const std::vector<std::string> &joint_names
) {
    if (trajectories.empty()) {
        throw std::runtime_error("cannot visualize an empty Franka path set");
    }
    const json bundle = {
        {"joint_names", joint_names},
        {"trajectories", trajectories},
    };
    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto trajectory_path = std::filesystem::temp_directory_path()
        / ("prrtc_" + std::string(Robot::name) + "_trajectories_"
           + std::to_string(timestamp) + ".json");
    {
        std::ofstream trajectory_file(trajectory_path);
        if (!trajectory_file) {
            throw std::runtime_error(
                "failed to create temporary Franka trajectory"
            );
        }
        trajectory_file << bundle.dump(2) << '\n';
    }

    const auto visualizer_path = std::filesystem::absolute(
        "scripts/visualize_franka.py"
    );
    const auto model_path = std::filesystem::absolute(
        std::is_same_v<Robot, robots::FrankaSingle>
            ? "resources/franka/franka_sim/franka_single.xml"
            : "resources/franka/franka_sim/franka_panda.xml"
    );
    std::string command =
        "python3 " + shell_quote(visualizer_path.string())
        + " --model " + shell_quote(model_path.string())
        + " --trajectory " + shell_quote(trajectory_path.string());
    const char *validate_only = std::getenv("PRRTC_MUJOCO_VALIDATE_ONLY");
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
json ffw_sg2_trajectory_json(
    const PlannerResult<Robot> &result,
    const typename Robot::Configuration &start,
    const AORRTC_settings &settings,
    const std::string &label
) {
    json trajectory = ordered_trajectory_json<Robot>(result, start, label);
    const auto &attached = settings.ffw_sg2_attached_object_collision;
    trajectory["attached_object_frame_offset"] = {
        attached.enabled ? attached.world_offset[0] : 0.1f,
        attached.enabled ? attached.world_offset[1] : 0.0f,
        attached.enabled ? attached.world_offset[2] : 0.0f,
    };
    return trajectory;
}

template <typename Robot>
void visualize_ffw_sg2_paths(
    const json &trajectories,
    const std::vector<std::string> &joint_names,
    bool use_ctrl
) {
    if (trajectories.empty()) {
        throw std::runtime_error("cannot visualize an empty FFW-SG2 path set");
    }
    const json bundle = {
        {"joint_names", joint_names},
        {"trajectories", trajectories},
    };
    const auto timestamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto trajectory_path = std::filesystem::temp_directory_path() /
        ("prrtc_" + std::string(Robot::name) + "_trajectories_" +
         std::to_string(timestamp) + ".json");
    {
        std::ofstream output(trajectory_path);
        if (!output) {
            throw std::runtime_error(
                "failed to create temporary FFW-SG2 trajectory bundle"
            );
        }
        output << bundle.dump(2) << '\n';
    }
    const auto visualizer = std::filesystem::absolute(
        "scripts/visualize_ffw_sg2.py"
    );
    const bool mobility_model =
        !joint_names.empty() && joint_names.front() == "base_x";
    const auto model = std::filesystem::absolute(
        mobility_model
            ? "ffw_lift/ffw_sg2_rack_upper_to_lower.xml"
            : "ffw_lift/ffw_sg2_lift.xml"
    );
    const std::string command =
        "python3 " + shell_quote(visualizer.string()) +
        " --model " + shell_quote(model.string()) +
        " --trajectory " + shell_quote(trajectory_path.string()) +
        " --input-mode " + (use_ctrl ? "ctrl" : "qpos");
    std::cout.flush();
    std::cerr.flush();
    const int status = std::system(command.c_str());
    std::error_code remove_error;
    std::filesystem::remove(trajectory_path, remove_error);
    if (status != 0) {
        throw std::runtime_error("FFW-SG2 MuJoCo visualizer exited with an error");
    }
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

std::size_t warm_up_planner() {
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

void print_csv_header(std::ofstream &outfile) {
    outfile << "problem_name,problem_idx,solved,cost,path_length,start_tree_size,goal_tree_size,iters,wall_ns,kernel_ns,";
    outfile << "copy_ns,num_new_configs,granularity,range,balance,tree_ratio,dynamic_domain,dd_alpha,dd_radius,dd_min_radius\n";
}

template<typename Robot>
void print_planner_result_to_file(PlannerResult<Robot> &result, pRRTC_settings &settings, std::string problem_name, int problem_idx, std::ofstream &outfile) {
    outfile << problem_name << ", ";
    outfile << problem_idx << ", ";
    outfile << result.solved << ", ";
    outfile << result.cost << ", ";
    outfile << result.path_length << ", ";
    outfile << result.start_tree_size << ", ";
    outfile << result.goal_tree_size << ", ";
    outfile << result.iters << ", ";
    outfile << result.wall_ns << ", ";
    outfile << result.kernel_ns << ", ";
    outfile << result.copy_ns << ", ";
    outfile << settings.num_new_configs << ", ";
    outfile << settings.granularity << ", ";
    outfile << settings.range << ", ";
    outfile << settings.balance << ", ";
    outfile << settings.tree_ratio << ", ";
    outfile << settings.dynamic_domain << ", ";
    outfile << settings.dd_alpha << ", ";
    outfile << settings.dd_radius << ", ";
    outfile << settings.dd_min_radius;
    outfile << "\n";
}


template <typename Robot>
void run_planning(
    const json &problems,
    AORRTC_settings &settings,
    std::string run_name,
    std::string robot_name,
    int runs,
    int max_problems,
    bool print_path,
    bool visualize,
    bool plot,
    const std::string &save_json_path,
    const std::string &problem_file_path
) {
    using Configuration = typename Robot::Configuration;
    std::filesystem::create_directories("test_output");
    std::ofstream outfile("test_output/"+robot_name+"_"+run_name+".csv");
    if (!outfile) {
        throw std::runtime_error("failed to create benchmark output CSV");
    }
    print_csv_header(outfile);
    int failed = 0;
    int solved_count = 0;
    int total_runs = 0;
    int processed_problems = 0;
    std::vector<double> times_sec;
    std::vector<int> path_lengths;
    std::vector<float> costs;
    json saved_results = json::array();
    bool has_visualization_result = false;
    json visualization_trajectories = json::array();
    const unsigned long long base_seed = settings.random_seed;
    for (auto& [name, pset] : problems.items()) {
        if (max_problems > 0 && processed_problems >= max_problems) {
            break;
        }
        std::cout << name << "\n";        
        for (int i = 0; i < pset.size(); i++) {
            if (max_problems > 0 && processed_problems >= max_problems) {
                break;
            }
            std::cout << "idx: " << i + 1 << "\n";
            json data = ppln::config::normalize_problem(pset[i], robot_name);
            if (not data["valid"]) {
                continue;
            }
            ppln::config::validate_query(data, robot_name);
            processed_problems++;
            AORRTC_settings problem_settings = settings;
            auto env = problem_dict_to_env(data, name);
            ppln::constraints::apply_constraint_backend_defaults<Robot>(
                problem_settings
            );
            auto prepared_query =
                ppln::constraints::prepare_constraint_query<Robot>(
                    data, problem_settings
                );
            Configuration start = prepared_query.start;
            std::vector<Configuration> goals =
                std::move(prepared_query.goals);
            if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
                ffw_sg2_attached_object_collision::apply_from_problem(
                    data,
                    problem_settings,
                    ffw_sg2_attached_object_collision::kFfwSg2FixedFineSphereCount,
                    ffw_sg2_attached_object_collision::kFfwSg2FixedApproxSphereCount
                );
            } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
                ffw_sg2_attached_object_collision::apply_from_problem(
                    data,
                    problem_settings
                );
                if (data.contains("constraints")
                    && data["constraints"].contains("com")) {
                    const auto &com = data["constraints"]["com"];
                    if (com.value("enabled", false)) {
                        problem_settings.ffw_sg2_enable_com_constraint = true;
                    }
                    if (com.contains("support_margin_m")) {
                        problem_settings.ffw_sg2_support_margin_m =
                            com["support_margin_m"].get<float>();
                    }
                    if (com.contains("object_mass_kg")) {
                        problem_settings.ffw_sg2_object_mass_kg =
                            com["object_mass_kg"].get<float>();
                    }
                }
            } else {
                problem_settings.ffw_sg2_attached_object_collision = {};
                if (data.contains("attached_object_collision")) {
                    throw std::invalid_argument(
                        "attached_object_collision is supported only for ffw_sg2 and ffw_sg2_mobility"
                    );
                }
            }

            for (int run_index = 1; run_index <= runs; run_index++) {
                total_runs++;
                problem_settings.random_seed = base_seed
                    + static_cast<unsigned long long>(run_index - 1);
                if (runs > 1) {
                    std::cout << "run: " << run_index << "\n";
                }
                std::cout << "seed: " << problem_settings.random_seed << "\n";

                if (run_index == 1) {
                    warm_up_planner();
                }
                AORRTCResult<Robot> result;
                if (problem_settings.aorrtc) {
                    result = AORRTC::solve<Robot>(
                        start, goals, env, problem_settings
                    );
                } else {
                    static_cast<PlannerResult<Robot> &>(result) =
                        pRRTC::solve<Robot>(
                            start, goals, env, problem_settings
                        );
                }
                if (print_path) {
                    for (auto& cfg: result.path) {
                        print_cfg<Robot>(cfg);
                    }
                }
                const double planning_sec = problem_settings.aorrtc
                    ? static_cast<double>(result.planning_ns) / 1.0e9
                    : static_cast<double>(result.kernel_ns) / 1.0e9;
                std::cout << "kernel_ns: " << result.kernel_ns << "\n";
                std::cout << "planning_s: " << planning_sec << "\n";
                if (not result.solved) {
                    failed ++;
                    std::cout << "failed " << name << std::endl;
                } else {
                    solved_count++;
                    times_sec.push_back(planning_sec);
                    path_lengths.push_back(result.path_length);
                    costs.push_back(result.cost);
                }
                std::cout << "cost: " << result.cost << "\n";
                if (problem_settings.aorrtc) {
                    std::cout << "aorrtc_initial_cost: "
                              << result.initial_cost << "\n";
                    std::cout << "aorrtc_solution_updates: "
                              << result.solution_updates << "\n";
                    std::cout << "aorrtc_search_restarts: "
                              << result.search_restarts << "\n";
                    std::cout << "aorrtc_initial_solution_sec: "
                              << static_cast<double>(
                                     result.initial_solution_ns
                                 ) / 1.0e9
                              << "\n";
                    std::cout << "aorrtc_best_solution_sec: "
                              << static_cast<double>(
                                     result.best_solution_ns
                                 ) / 1.0e9
                              << "\n";
                }

                json saved_result = {
                    {"run_idx", total_runs},
                    {"problem_name", name},
                    {"problem_idx", i + 1},
                    {"pair_id", data.value("pair_id", i + 1)},
                    {"trial_idx", run_index},
                    {"seed", problem_settings.random_seed},
                    {"solved", result.solved},
                    {"kernel_ns", result.kernel_ns},
                    {"wall_ns", result.wall_ns},
                    {"copy_ns", result.copy_ns},
                    {"cost", result.cost},
                    {"path_length", result.path_length},
                    {"start_tree_size", result.start_tree_size},
                    {"goal_tree_size", result.goal_tree_size},
                    {"iters", result.iters},
                    {"planning_sec", planning_sec},
                };
                if (problem_settings.collect_diagnostics) {
                    saved_result["diagnostics"] =
                        planner_result_json::diagnostics_to_json(
                            result.diagnostics
                        );
                }
                if (problem_settings.aorrtc) {
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
                    saved_result["initial_cost"] = result.initial_cost;
                    saved_result["initial_solution_sec"] =
                        static_cast<double>(result.initial_solution_ns) / 1.0e9;
                    saved_result["best_solution_sec"] =
                        static_cast<double>(result.best_solution_ns) / 1.0e9;
                    saved_result["solution_updates"] = result.solution_updates;
                    saved_result["search_restarts"] = result.search_restarts;
                    saved_result["solution_history_overflow"] =
                        result.solution_history_overflow;
                    saved_result["solution_history"] =
                        std::move(solution_history);
                }
                saved_results.push_back(std::move(saved_result));

                print_planner_result_to_file(
                    result, problem_settings, name, i + 1, outfile
                );
                if (visualize && result.solved) {
                    const std::string label =
                        name + " pair " + std::to_string(i + 1) +
                        ", run " + std::to_string(run_index);
                    const auto &planner_result =
                        static_cast<const PlannerResult<Robot> &>(result);
                    if constexpr (
                        std::is_same_v<Robot, robots::FrankaSingle> ||
                        std::is_same_v<Robot, robots::Franka>
                    ) {
                        visualization_trajectories.push_back(
                            ordered_trajectory_json<Robot>(
                                planner_result, start, label
                            )
                        );
                    } else if constexpr (std::is_same_v<Robot, robots::G1>) {
                        visualization_trajectories.push_back(
                            g1_trajectory_json(
                                planner_result,
                                start,
                                data,
                                planning_sec,
                                label
                            )
                        );
                    } else if constexpr (
                        std::is_same_v<Robot, robots::FfwSg2> ||
                        std::is_same_v<Robot, robots::FfwSg2Mobility>
                    ) {
                        visualization_trajectories.push_back(
                            ffw_sg2_trajectory_json<Robot>(
                                planner_result,
                                start,
                                problem_settings,
                                label
                            )
                        );
                    }
                    has_visualization_result = true;
                }
            }
        }
    }

    std::filesystem::path result_json_path;
    if (!save_json_path.empty()) {
        result_json_path = save_json_path;
    } else if (plot) {
        result_json_path = std::filesystem::path("logs") / (
            "aorrtc_" + filename_component(robot_name)
            + "_" + filename_component(run_name)
            + "_results.json"
        );
    }
    if (!result_json_path.empty()) {
        const std::filesystem::path output_path(result_json_path);
        if (!output_path.parent_path().empty()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        const auto temporary_path = output_path.string() + ".tmp";
        std::ofstream output(temporary_path);
        if (!output) {
            throw std::runtime_error(
                "failed to create benchmark JSON: " + temporary_path
            );
        }
        const std::string problem_set_name = problems.size() == 1
            ? problems.begin().key()
            : "multiple";
        output << json{
            {
                "format",
                settings.aorrtc
                    ? "AORRTC_problem_set_results_v1"
                    : "pRRTC_problem_set_results_v1"
            },
            {"planner", settings.aorrtc ? "AORRTC" : "pRRTC"},
            {"robot", robot_name},
            {"run_name", run_name},
            {"problem_name", problem_set_name},
            {"problem_file", problem_file_path},
            {"problems", processed_problems},
            {"runs_per_problem", runs},
            {"runs", total_runs},
            {"solved_runs", solved_count},
            {
                "timing_scope",
                settings.aorrtc
                    ? "AORRTC end-to-end planning"
                    : "CUDA planning kernel"
            },
            {
                "time_field",
                settings.aorrtc ? "planning_sec" : "kernel_ns"
            },
            {"time_unit", settings.aorrtc ? "s" : "ns"},
            {"rigid_orientation", settings.rigid_orientation},
            {"settings", planner_result_json::settings_to_json(settings)},
            {"results", std::move(saved_results)},
        }.dump(2) << '\n';
        output.close();
        std::filesystem::rename(temporary_path, output_path);
        std::cout << "saved_json: " << output_path.string() << "\n";
    }

    if (plot) {
        if (solved_count == 0) {
            std::cout << "AORRTC plot skipped: no solved problems.\n";
        } else {
            plot_aorrtc_problem_set(
                result_json_path,
                robot_name,
                run_name,
                processed_problems
            );
        }
    }

    if (total_runs > 1) {
        std::cout << "solved_runs: " << solved_count << "/" << total_runs << "\n";
    }
    if (total_runs > 1 && !times_sec.empty()) {
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
    if (total_runs > 1 && !path_lengths.empty()) {
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
    if (visualize) {
        if (!has_visualization_result) {
            throw std::runtime_error(
                "cannot visualize because no evaluated problem was solved"
            );
        }
        if constexpr (std::is_same_v<Robot, robots::G1>) {
            visualize_g1_paths(visualization_trajectories);
        } else if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
            visualize_franka_paths<Robot>(visualization_trajectories, {
                "panda0_joint1", "panda0_joint2", "panda0_joint3",
                "panda0_joint4", "panda0_joint5", "panda0_joint6",
                "panda0_joint7"
            });
        } else if constexpr (std::is_same_v<Robot, robots::Franka>) {
            visualize_franka_paths<Robot>(visualization_trajectories, {
                "panda0_joint1", "panda0_joint2", "panda0_joint3",
                "panda0_joint4", "panda0_joint5", "panda0_joint6",
                "panda0_joint7", "panda1_joint1", "panda1_joint2",
                "panda1_joint3", "panda1_joint4", "panda1_joint5",
                "panda1_joint6", "panda1_joint7"
            });
        } else if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
            visualize_ffw_sg2_paths<Robot>(visualization_trajectories, {
                "lift_joint",
                "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
                "arm_l_joint4", "arm_l_joint5", "arm_l_joint6",
                "arm_l_joint7",
                "arm_r_joint1", "arm_r_joint2", "arm_r_joint3",
                "arm_r_joint4", "arm_r_joint5", "arm_r_joint6",
                "arm_r_joint7"
            }, true);
        } else if constexpr (
            std::is_same_v<Robot, robots::FfwSg2Mobility>
        ) {
            visualize_ffw_sg2_paths<Robot>(visualization_trajectories, {
                "base_x", "base_y", "base_yaw", "lift_joint",
                "arm_l_joint1", "arm_l_joint2", "arm_l_joint3",
                "arm_l_joint4", "arm_l_joint5", "arm_l_joint6",
                "arm_l_joint7",
                "arm_r_joint1", "arm_r_joint2", "arm_r_joint3",
                "arm_r_joint4", "arm_r_joint5", "arm_r_joint6",
                "arm_r_joint7"
            }, false);
        }
    }
}

int main(int argc, char* argv[]) {
    std::string robot_name;
    std::string run_name;
    std::string problem_file_path;
    std::string save_json_path;
    int runs = 1;
    int max_problems = 0;
    bool print_path = true;
    bool visualize = false;
    bool plot = false;
    bool time_option_provided = false;
    AORRTC_settings settings;
    settings.num_new_configs = 512;
    settings.max_iters = 100000000;
    settings.random_seed = 1ULL;
    settings.granularity = 16;
    settings.range = 0.3f;
    settings.lift_distance_weight = 1.0f;
    settings.projection_smoothness = true;
    settings.balance = 2;
    settings.tree_ratio = 1.0;
    settings.dynamic_domain = false;
    settings.dd_radius = 4.0;
    settings.dd_min_radius = 1.0;
    settings.dd_alpha = 0.0001;
    settings.em_threshold = 0.1f;
    settings.max_concon_nodes = 4;
    settings.max_connect_concon_chunks = 16;
    settings.prevent_ts_backtracking = true;
    settings.collect_diagnostics = false;
    

    if (argc >= 3) {
        robot_name = argv[1];
        run_name = argv[2];
        try {
            for (int index = 3; index < argc; index++) {
                const std::string argument = argv[index];
                if ((argument == "--run" || argument == "--runs")
                    && index + 1 < argc) {
                    runs = std::stoi(argv[++index]);
                    if (runs <= 0) {
                        throw std::invalid_argument(
                            "--runs must be a positive integer"
                        );
                    }
                } else if (argument == "--max-problems"
                    && index + 1 < argc) {
                    max_problems = std::stoi(argv[++index]);
                    if (max_problems <= 0) {
                        throw std::invalid_argument(
                            "--max-problems must be a positive integer"
                        );
                    }
                } else if (argument == "--problem-file" && index + 1 < argc) {
                    problem_file_path = argv[++index];
                } else if (argument == "--save-json" && index + 1 < argc) {
                    save_json_path = argv[++index];
                } else if (argument == "--range" && index + 1 < argc) {
                    const std::string value = argv[++index];
                    std::size_t consumed = 0;
                    settings.range = std::stof(value, &consumed);
                    if (consumed != value.size()
                        || !std::isfinite(settings.range)
                        || settings.range <= 0.0f) {
                        throw std::invalid_argument(
                            "--range must be a finite number greater than 0"
                        );
                    }
                } else if (argument == "--aorrtc") {
                    settings.aorrtc = true;
                } else if (argument == "--time" && index + 1 < argc) {
                    const std::string value = argv[++index];
                    std::size_t consumed = 0;
                    settings.time_limit_sec = std::stod(value, &consumed);
                    time_option_provided = true;
                    if (consumed != value.size()
                        || !std::isfinite(settings.time_limit_sec)
                        || settings.time_limit_sec <= 0.0) {
                        throw std::invalid_argument(
                            "--time must be a finite number greater than 0"
                        );
                    }
                } else if (argument == "--plot") {
                    plot = true;
                } else if (argument == "--visualize") {
                    visualize = true;
                } else if (argument == "--no-print-path") {
                    print_path = false;
                } else if (argument == "--rigid-orientation") {
                    settings.rigid_orientation = true;
                } else if (argument == "--com") {
                    settings.ffw_sg2_enable_com_constraint = true;
                } else {
                    throw std::invalid_argument(
                        "unknown or incomplete option: " + argument
                    );
                }
            }
            if (time_option_provided && !settings.aorrtc) {
                throw std::invalid_argument("--time requires --aorrtc");
            }
            if (plot && !settings.aorrtc) {
                throw std::invalid_argument("--plot requires --aorrtc");
            }
        } catch (const std::exception &error) {
            std::cerr << "evaluate_mbm option error: "
                      << error.what() << "\n";
            return 1;
        }
        if (settings.ffw_sg2_enable_com_constraint
            && robot_name != "ffw_sg2_mobility") {
            std::cerr << "--com is supported only for ffw_sg2_mobility\n";
            return 1;
        }
        if (visualize
            && robot_name != "g1"
            && robot_name != "franka_single"
            && robot_name != "franka"
            && robot_name != "ffw_sg2"
            && robot_name != "ffw_sg2_mobility") {
            std::cerr
                << "--visualize is currently supported only for g1, "
                << "franka_single, franka, ffw_sg2, and "
                << "ffw_sg2_mobility\n";
            return 1;
        }
        if (runs > 1
            && settings.random_seed
                > std::numeric_limits<unsigned long long>::max()
                    - static_cast<unsigned long long>(runs - 1)) {
            std::cerr << "--runs exceeds the supported seed range\n";
            return 1;
        }
    }
    else {
        std::cout << "Usage: evaluate_mbm <robot_name> <run_name> "
                  << "[--problem-file PATH] [--save-json PATH] "
                  << "[--run N|--runs N] "
                  << "[--range VALUE] "
                  << "[--max-problems N] [--aorrtc] [--time SECONDS] "
                  << "[--plot] [--visualize] "
                  << "[--no-print-path] [--rigid-orientation] [--com]\n";
        return -1;
    }

    const bool robot_supported =
        robot_name == "franka_single" ||
        robot_name == "franka" ||
        robot_name == "ffw_sg2" ||
        robot_name == "ffw_sg2_mobility" ||
        robot_name == "g1" ||
        robot_name == "igris_c";
    if (!robot_supported) {
        std::cerr << "Unsupported robot type: " << robot_name << "\n";
        return 1;
    }

    const std::string path = problem_file_path.empty()
        ? "scripts/" + robot_name + "_problems.json"
        : problem_file_path;
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
    try {
        (void)ppln::config::robot_name_from_json(all_data, robot_name);
        ppln::config::validate_declared_dimension(all_data, robot_name);
    } catch (const std::exception &error) {
        std::cerr << "Problem file robot metadata error: "
                  << error.what() << "\n";
        return 1;
    }
    if (!all_data.contains("problems") || !all_data["problems"].is_object()) {
        std::cerr << "Problem file is missing a problems object: " << path << "\n";
        return 1;
    }
    json problems = all_data["problems"];
    if (robot_name == "g1") {
        settings.granularity = robots::G1::resolution;
        settings.projection_max_iters = 60;
        settings.max_concon_nodes = 4;
    } else if (robot_name == "igris_c") {
        settings.granularity = robots::IgrisC::resolution;
        settings.projection_max_iters = 60;
        settings.max_concon_nodes = 4;
    }
    if (robot_name == "ffw_sg2") {
        run_planning<robots::FfwSg2>(problems, settings, run_name, robot_name, runs, max_problems, print_path, visualize, plot, save_json_path, path);
    } else if (robot_name == "ffw_sg2_mobility") {
        run_planning<robots::FfwSg2Mobility>(problems, settings, run_name, robot_name, runs, max_problems, print_path, visualize, plot, save_json_path, path);
    } else if (robot_name == "g1") {
        run_planning<robots::G1>(problems, settings, run_name, robot_name, runs, max_problems, print_path, visualize, plot, save_json_path, path);
    } else if (robot_name == "igris_c") {
        run_planning<robots::IgrisC>(problems, settings, run_name, robot_name, runs, max_problems, print_path, visualize, plot, save_json_path, path);
    } else if (robot_name == "franka_single") {
        run_planning<robots::FrankaSingle>(problems, settings, run_name, robot_name, runs, max_problems, print_path, visualize, plot, save_json_path, path);
    } else if (robot_name == "franka") {
        run_planning<robots::Franka>(problems, settings, run_name, robot_name, runs, max_problems, print_path, visualize, plot, save_json_path, path);
    } else {
        std::cerr << "Unsupported robot type: " << robot_name << "\n";
        return 1;
    }
}

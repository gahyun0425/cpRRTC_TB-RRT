#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "scripts/ffw_sg2_attached_object_collision.hh"
#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/planning/Planners.hh"
#include "src/planning/utils.cuh"
#include "src/robots/ffw_sg2.cuh"
#include "src/robots/ffw_sg2_attached_object_collision.cuh"
#include "src/robots/ffw_sg2_constraint.cuh"

using json = nlohmann::json;

namespace {

constexpr int kDimension = 15;
constexpr int kThreads = 64;
constexpr int kDefaultCount = 100;
constexpr int kCandidateBatchSize = 4096;
constexpr int kDefaultMaximumCandidates = 200000;
constexpr std::uint64_t kDefaultSeed = 20260916ULL;
constexpr float kDuplicateDistanceSquared = 1.0e-8f;

using Configuration = std::array<float, kDimension>;

struct Region {
    float minimum[3];
    float maximum[3];
};

struct CandidateResult {
    float configuration[kDimension]{};
    float object_position[3]{};
    float constraint_residual = std::numeric_limits<float>::infinity();
    int projection_valid = 0;
    int collision_free = 0;
};

struct Options {
    std::filesystem::path template_path = "scripts/ffw_sg2_problems.json";
    std::filesystem::path output_path =
        "scripts/ffw_sg2_random_pairs_100.json";
    std::string source_name = "tray_lift";
    int source_index = 1;
    std::string problem_name = "tray_lift_random_pairs";
    int count = kDefaultCount;
    int maximum_candidates = kDefaultMaximumCandidates;
    std::uint64_t seed = kDefaultSeed;
    bool axis = false;
    float start_sigma = 0.12f;
    float goal_sigma = 0.16f;
    Region start_region{{0.53f, -0.05f, 1.04f}, {0.61f, 0.05f, 1.12f}};
    Region goal_region{{0.75f, -0.03f, 1.47f}, {0.77f, 0.03f, 1.56f}};
};

struct DeviceEnvironment {
    ppln::collision::Environment<float> *environment = nullptr;
    ppln::collision::Sphere<float> *spheres = nullptr;
    ppln::collision::Capsule<float> *capsules = nullptr;
    ppln::collision::Cuboid<float> *cuboids = nullptr;
};

void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

int parse_positive_int(const std::string &option, const char *value) {
    std::size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (consumed != std::string(value).size() || parsed <= 0) {
        throw std::invalid_argument(option + " must be a positive integer");
    }
    return parsed;
}

float parse_positive_float(const std::string &option, const char *value) {
    std::size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (
        consumed != std::string(value).size() ||
        !std::isfinite(parsed) || parsed <= 0.0f
    ) {
        throw std::invalid_argument(option + " must be a positive finite number");
    }
    return parsed;
}

std::uint64_t parse_seed(const char *value) {
    const std::string text(value);
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument("--seed must be a nonnegative integer");
    }
    std::size_t consumed = 0;
    const auto parsed = std::stoull(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument("--seed must be a nonnegative integer");
    }
    return parsed;
}

Region parse_region(int argc, char **argv, int &index, const std::string &option) {
    if (index + 6 >= argc) {
        throw std::invalid_argument(
            option + " requires xmin xmax ymin ymax zmin zmax"
        );
    }
    std::array<float, 6> values{};
    for (float &value : values) {
        std::size_t consumed = 0;
        const std::string text(argv[++index]);
        value = std::stof(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value)) {
            throw std::invalid_argument(option + " values must be finite numbers");
        }
    }
    Region region{
        {values[0], values[2], values[4]},
        {values[1], values[3], values[5]},
    };
    for (int axis = 0; axis < 3; ++axis) {
        if (region.minimum[axis] > region.maximum[axis]) {
            throw std::invalid_argument(option + " minimum exceeds maximum");
        }
    }
    return region;
}

void print_usage() {
    std::cout
        << "Usage: generate_ffw_sg2_random_pairs [options]\n"
        << "  --template PATH\n"
        << "  --output PATH\n"
        << "  --source-name NAME --source-index N\n"
        << "  --problem-name NAME --count N --seed N\n"
        << "  --axis\n"
        << "  --start-sigma RAD --goal-sigma RAD\n"
        << "  --start-region xmin xmax ymin ymax zmin zmax\n"
        << "  --goal-region xmin xmax ymin ymax zmin zmax\n"
        << "  --max-candidates N\n";
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        auto require_value = [&]() -> const char * {
            if (index + 1 >= argc) {
                throw std::invalid_argument(argument + " requires a value");
            }
            return argv[++index];
        };
        if (argument == "--template") {
            options.template_path = require_value();
        } else if (argument == "--output") {
            options.output_path = require_value();
        } else if (argument == "--source-name") {
            options.source_name = require_value();
        } else if (argument == "--source-index") {
            options.source_index = parse_positive_int(argument, require_value());
        } else if (argument == "--problem-name") {
            options.problem_name = require_value();
        } else if (argument == "--count") {
            options.count = parse_positive_int(argument, require_value());
        } else if (argument == "--seed") {
            options.seed = parse_seed(require_value());
        } else if (argument == "--axis") {
            options.axis = true;
        } else if (argument == "--start-sigma") {
            options.start_sigma = parse_positive_float(argument, require_value());
        } else if (argument == "--goal-sigma") {
            options.goal_sigma = parse_positive_float(argument, require_value());
        } else if (argument == "--start-region") {
            options.start_region = parse_region(argc, argv, index, argument);
        } else if (argument == "--goal-region") {
            options.goal_region = parse_region(argc, argv, index, argument);
        } else if (argument == "--max-candidates") {
            options.maximum_candidates =
                parse_positive_int(argument, require_value());
        } else if (argument == "--help" || argument == "-h") {
            print_usage();
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return options;
}

template <typename Value>
std::array<float, 3> json_vec3(const Value &value, const std::string &field) {
    if (!value.is_array() || value.size() != 3) {
        throw std::invalid_argument(field + " must contain 3 values");
    }
    std::array<float, 3> result{};
    for (int axis = 0; axis < 3; ++axis) {
        result[axis] = value.at(axis).template get<float>();
        if (!std::isfinite(result[axis])) {
            throw std::invalid_argument(field + " contains a non-finite value");
        }
    }
    return result;
}

Configuration json_configuration(const json &value, const std::string &field) {
    if (!value.is_array() || value.size() != kDimension) {
        throw std::invalid_argument(field + " must contain 15 values");
    }
    Configuration configuration{};
    for (int joint = 0; joint < kDimension; ++joint) {
        configuration[joint] = value.at(joint).get<float>();
        if (!std::isfinite(configuration[joint])) {
            throw std::invalid_argument(field + " contains a non-finite value");
        }
    }
    return configuration;
}

DeviceEnvironment make_device_environment(const json &problem) {
    using namespace ppln::collision;
    std::vector<Sphere<float>> spheres;
    std::vector<Capsule<float>> capsules;
    std::vector<Cuboid<float>> cuboids;

    for (const auto &entry : problem.value("sphere", json::array())) {
        const auto position = json_vec3(entry.at("position"), "sphere.position");
        spheres.emplace_back(
            position[0], position[1], position[2], entry.at("radius").get<float>()
        );
    }
    for (const auto &entry : problem.value("cylinder", json::array())) {
        const auto position = json_vec3(entry.at("position"), "cylinder.position");
        const auto orientation = json_vec3(
            entry.at("orientation_euler_xyz"), "cylinder.orientation_euler_xyz"
        );
        capsules.push_back(factory::cylinder::center::array(
            position,
            orientation,
            entry.at("radius").get<float>(),
            entry.at("length").get<float>()
        ));
    }
    for (const auto &entry : problem.value("box", json::array())) {
        cuboids.push_back(factory::cuboid::array(
            json_vec3(entry.at("position"), "box.position"),
            json_vec3(entry.at("orientation_euler_xyz"), "box.orientation_euler_xyz"),
            json_vec3(entry.at("half_extents"), "box.half_extents")
        ));
    }

    DeviceEnvironment device;
    check_cuda(
        cudaMalloc(&device.environment, sizeof(Environment<float>)),
        "cudaMalloc environment"
    );
    check_cuda(
        cudaMemset(device.environment, 0, sizeof(Environment<float>)),
        "cudaMemset environment"
    );

    if (!spheres.empty()) {
        check_cuda(
            cudaMalloc(&device.spheres, spheres.size() * sizeof(Sphere<float>)),
            "cudaMalloc spheres"
        );
        check_cuda(
            cudaMemcpy(
                device.spheres,
                spheres.data(),
                spheres.size() * sizeof(Sphere<float>),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy spheres"
        );
        const unsigned int count = static_cast<unsigned int>(spheres.size());
        check_cuda(
            cudaMemcpy(
                &(device.environment->spheres),
                &device.spheres,
                sizeof(device.spheres),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy environment.spheres"
        );
        check_cuda(
            cudaMemcpy(
                &(device.environment->num_spheres),
                &count,
                sizeof(count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy environment.num_spheres"
        );
    }
    if (!capsules.empty()) {
        check_cuda(
            cudaMalloc(
                &device.capsules, capsules.size() * sizeof(Capsule<float>)
            ),
            "cudaMalloc capsules"
        );
        check_cuda(
            cudaMemcpy(
                device.capsules,
                capsules.data(),
                capsules.size() * sizeof(Capsule<float>),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy capsules"
        );
        const unsigned int count = static_cast<unsigned int>(capsules.size());
        check_cuda(
            cudaMemcpy(
                &(device.environment->capsules),
                &device.capsules,
                sizeof(device.capsules),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy environment.capsules"
        );
        check_cuda(
            cudaMemcpy(
                &(device.environment->num_capsules),
                &count,
                sizeof(count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy environment.num_capsules"
        );
    }
    if (!cuboids.empty()) {
        check_cuda(
            cudaMalloc(&device.cuboids, cuboids.size() * sizeof(Cuboid<float>)),
            "cudaMalloc cuboids"
        );
        check_cuda(
            cudaMemcpy(
                device.cuboids,
                cuboids.data(),
                cuboids.size() * sizeof(Cuboid<float>),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy cuboids"
        );
        const unsigned int count = static_cast<unsigned int>(cuboids.size());
        check_cuda(
            cudaMemcpy(
                &(device.environment->cuboids),
                &device.cuboids,
                sizeof(device.cuboids),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy environment.cuboids"
        );
        check_cuda(
            cudaMemcpy(
                &(device.environment->num_cuboids),
                &count,
                sizeof(count),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy environment.num_cuboids"
        );
    }
    return device;
}

void destroy_device_environment(DeviceEnvironment &device) {
    cudaFree(device.spheres);
    cudaFree(device.capsules);
    cudaFree(device.cuboids);
    cudaFree(device.environment);
    device = {};
}

__device__ bool scalar_robot_collision_free(
    volatile float *sphere_positions,
    ppln::collision::Environment<float> *environment
) {
    using namespace ppln::collision;
    for (int sphere = 0; sphere < FFW_SG2_SPHERE_COUNT; ++sphere) {
        const int offset = sphere * FFW_SG2_BATCH_SIZE * 3;
        if (sphere_environment_in_collision(
                environment,
                sphere_positions[offset + 0],
                sphere_positions[offset + 1],
                sphere_positions[offset + 2],
                ffw_sg2_spheres_array[sphere].w)) {
            return false;
        }
    }
    for (int range = 0; range < FFW_SG2_SELF_CC_RANGE_COUNT; ++range) {
        const int first = ffw_sg2_self_cc_ranges[range][0];
        const int first_offset = first * FFW_SG2_BATCH_SIZE * 3;
        for (
            int second = ffw_sg2_self_cc_ranges[range][1];
            second <= ffw_sg2_self_cc_ranges[range][2];
            ++second
        ) {
            const int second_offset = second * FFW_SG2_BATCH_SIZE * 3;
            if (sphere_sphere_self_collision(
                    sphere_positions[first_offset + 0],
                    sphere_positions[first_offset + 1],
                    sphere_positions[first_offset + 2],
                    ffw_sg2_spheres_array[first].w,
                    sphere_positions[second_offset + 0],
                    sphere_positions[second_offset + 1],
                    sphere_positions[second_offset + 2],
                    ffw_sg2_spheres_array[second].w)) {
                return false;
            }
        }
    }
    return true;
}

__device__ bool scalar_attached_object_collision_free(
    const float *configuration,
    volatile float *robot_sphere_positions,
    ppln::collision::Environment<float> *environment,
    float object_position[3]
) {
    using namespace ppln::collision;
    float x_axis[3], y_axis[3], z_axis[3];
    ffw_sg2_attached_object_frame(
        configuration,
        object_position,
        x_axis,
        y_axis,
        z_axis
    );
    const auto &spec = ffw_sg2_mobility_attached_object_collision;
    if (!spec.enabled) {
        return true;
    }
    const int object_sphere_count = min(
        spec.sphere_count,
        FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES
    );
    for (int object_sphere = 0;
         object_sphere < object_sphere_count;
         ++object_sphere) {
        float object_x, object_y, object_z, object_radius;
        ffw_sg2_attached_object_sphere_world(
            object_position,
            x_axis,
            y_axis,
            z_axis,
            object_sphere,
            object_x,
            object_y,
            object_z,
            object_radius
        );
        if (object_radius <= 0.0f) {
            continue;
        }
        if (sphere_environment_in_collision(
                environment,
                object_x,
                object_y,
                object_z,
                object_radius)) {
            return false;
        }
        for (int robot_sphere = 0;
             robot_sphere < FFW_SG2_SPHERE_COUNT;
             ++robot_sphere) {
            if (ffw_sg2_attached_object_ignores_robot_sphere(
                    robot_sphere, false)) {
                continue;
            }
            const int offset = robot_sphere * FFW_SG2_BATCH_SIZE * 3;
            if (sphere_sphere_self_collision(
                    object_x,
                    object_y,
                    object_z,
                    object_radius,
                    robot_sphere_positions[offset + 0],
                    robot_sphere_positions[offset + 1],
                    robot_sphere_positions[offset + 2],
                    ffw_sg2_spheres_array[robot_sphere].w)) {
                return false;
            }
        }
    }
    return true;
}

template <bool kAxis>
__global__ void project_and_validate_candidates(
    const float *input,
    CandidateResult *results,
    ppln::collision::Environment<float> *environment,
    int count
) {
    const int candidate = static_cast<int>(blockIdx.x);
    if (candidate >= count) {
        return;
    }
    __shared__ float configuration[kDimension];
    __shared__ volatile float sphere_positions[
        FFW_SG2_SPHERE_COUNT * FFW_SG2_BATCH_SIZE * 3
    ];
    __shared__ float transforms[
        FFW_SG2_BATCH_SIZE * FFW_SG2_TRANSFORM_SLOTS * 16
    ];
    __shared__ int projection_valid;

    if (threadIdx.x == 0) {
        for (int joint = 0; joint < kDimension; ++joint) {
            configuration[joint] = input[candidate * kDimension + joint];
        }
        projection_valid = ppln::collision::ffw_sg2_project_config(
            configuration,
            kAxis
        ) ? 1 : 0;
    }
    __syncthreads();
    if (projection_valid == 0) {
        if (threadIdx.x == 0) {
            results[candidate].projection_valid = 0;
        }
        return;
    }

    ppln::device_utils::fk<ppln::robots::FfwSg2>(
        configuration,
        sphere_positions,
        transforms,
        static_cast<int>(threadIdx.x)
    );
    __syncthreads();

    if (threadIdx.x == 0) {
        CandidateResult &result = results[candidate];
        result.projection_valid = 1;
        for (int joint = 0; joint < kDimension; ++joint) {
            result.configuration[joint] = configuration[joint];
        }
        float residual[FFW_SG2_MAX_RESIDUAL_DIM]{};
        ppln::collision::ffw_sg2_constraint_residual(
            configuration,
            kAxis,
            residual
        );
        result.constraint_residual =
            ppln::collision::ffw_sg2_residual_norm(
                residual,
                kAxis ? 8 : 6
            );
        const bool robot_free = scalar_robot_collision_free(
            sphere_positions,
            environment
        );
        const bool object_free = scalar_attached_object_collision_free(
            configuration,
            sphere_positions,
            environment,
            result.object_position
        );
        result.collision_free = robot_free && object_free ? 1 : 0;
    }
}

bool inside_region(const float position[3], const Region &region) {
    for (int axis = 0; axis < 3; ++axis) {
        if (
            position[axis] < region.minimum[axis] ||
            position[axis] > region.maximum[axis]
        ) {
            return false;
        }
    }
    return true;
}

float squared_distance(
    const float left[kDimension],
    const float right[kDimension]
) {
    float distance = 0.0f;
    for (int joint = 0; joint < kDimension; ++joint) {
        const float difference = left[joint] - right[joint];
        distance += difference * difference;
    }
    return distance;
}

std::vector<CandidateResult> sample_pool(
    const Configuration &reference,
    float joint_sigma,
    const Region &region,
    int requested_count,
    int maximum_candidates,
    std::mt19937_64 &rng,
    ppln::collision::Environment<float> *device_environment,
    bool axis,
    const std::string &label
) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<CandidateResult> accepted;
    std::vector<float> host_input;
    std::vector<CandidateResult> host_results;
    float *device_input = nullptr;
    CandidateResult *device_results = nullptr;
    check_cuda(
        cudaMalloc(
            &device_input,
            kCandidateBatchSize * kDimension * sizeof(float)
        ),
        "cudaMalloc candidate input"
    );
    check_cuda(
        cudaMalloc(
            &device_results,
            kCandidateBatchSize * sizeof(CandidateResult)
        ),
        "cudaMalloc candidate results"
    );

    int attempted = 0;
    while (
        static_cast<int>(accepted.size()) < requested_count &&
        attempted < maximum_candidates
    ) {
        const int batch_count = std::min(
            kCandidateBatchSize,
            maximum_candidates - attempted
        );
        host_input.resize(batch_count * kDimension);
        for (int candidate = 0; candidate < batch_count; ++candidate) {
            for (int joint = 0; joint < kDimension; ++joint) {
                const float sigma = joint == 0 ? 0.25f * joint_sigma : joint_sigma;
                host_input[candidate * kDimension + joint] =
                    reference[joint] + sigma * normal(rng);
            }
        }
        check_cuda(
            cudaMemcpy(
                device_input,
                host_input.data(),
                host_input.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy candidate input"
        );
        check_cuda(
            cudaMemset(
                device_results,
                0,
                batch_count * sizeof(CandidateResult)
            ),
            "cudaMemset candidate results"
        );
        if (axis) {
            project_and_validate_candidates<true><<<batch_count, kThreads>>>(
                device_input,
                device_results,
                device_environment,
                batch_count
            );
        } else {
            project_and_validate_candidates<false><<<batch_count, kThreads>>>(
                device_input,
                device_results,
                device_environment,
                batch_count
            );
        }
        check_cuda(cudaGetLastError(), "project_and_validate_candidates launch");
        check_cuda(cudaDeviceSynchronize(), "project_and_validate_candidates sync");
        host_results.resize(batch_count);
        check_cuda(
            cudaMemcpy(
                host_results.data(),
                device_results,
                batch_count * sizeof(CandidateResult),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy candidate results"
        );
        attempted += batch_count;

        for (const auto &candidate : host_results) {
            if (
                candidate.projection_valid == 0 ||
                candidate.collision_free == 0 ||
                !std::isfinite(candidate.constraint_residual) ||
                candidate.constraint_residual >= 1.0e-3f ||
                !inside_region(candidate.object_position, region)
            ) {
                continue;
            }
            const bool duplicate = std::any_of(
                accepted.begin(),
                accepted.end(),
                [&](const CandidateResult &other) {
                    return squared_distance(
                        candidate.configuration,
                        other.configuration
                    ) < kDuplicateDistanceSquared;
                }
            );
            if (!duplicate) {
                accepted.push_back(candidate);
                if (static_cast<int>(accepted.size()) >= requested_count) {
                    break;
                }
            }
        }
        std::cout << label << ": accepted " << accepted.size()
                  << "/" << requested_count << " after " << attempted
                  << " candidates\n";
    }

    cudaFree(device_input);
    cudaFree(device_results);
    if (static_cast<int>(accepted.size()) < requested_count) {
        throw std::runtime_error(
            "could not generate enough " + label + " states; accepted " +
            std::to_string(accepted.size()) + " of " +
            std::to_string(requested_count)
        );
    }
    return accepted;
}

json region_json(const Region &region) {
    return {
        {"minimum", {
            region.minimum[0], region.minimum[1], region.minimum[2]
        }},
        {"maximum", {
            region.maximum[0], region.maximum[1], region.maximum[2]
        }},
    };
}

json candidate_position_json(const CandidateResult &candidate) {
    return {
        candidate.object_position[0],
        candidate.object_position[1],
        candidate.object_position[2],
    };
}

json generate_output(
    const Options &options,
    const json &source_problem,
    const std::vector<CandidateResult> &starts,
    const std::vector<CandidateResult> &goals
) {
    json generated_problems = json::array();
    for (int index = 0; index < options.count; ++index) {
        json problem = source_problem;
        problem["start"] = std::vector<float>(
            starts[index].configuration,
            starts[index].configuration + kDimension
        );
        problem["goals"] = json::array({std::vector<float>(
            goals[index].configuration,
            goals[index].configuration + kDimension
        )});
        problem["pair_id"] = index + 1;
        problem["sampling"] = {
            {"base_seed", options.seed},
            {"start_object_position", candidate_position_json(starts[index])},
            {"goal_object_position", candidate_position_json(goals[index])},
            {"start_constraint_residual", starts[index].constraint_residual},
            {"goal_constraint_residual", goals[index].constraint_residual},
            {"endpoint_collision_free", true},
        };
        if (options.axis) {
            problem["sampling"]["axis"] = true;
        }
        generated_problems.push_back(std::move(problem));
    }

    json output = {
        {"format", "ffw_sg2_random_start_goal_pairs_v1"},
        {"generator", {
            {"executable", "build/generate_ffw_sg2_random_pairs"},
            {"seed", options.seed},
            {"count", options.count},
            {"sampling", "Gaussian joint perturbation followed by closed-chain projection and rejection"},
            {"start_joint_sigma", options.start_sigma},
            {"goal_joint_sigma", options.goal_sigma},
            {"source_problem", {
                {"file", options.template_path.string()},
                {"name", options.source_name},
                {"index", options.source_index},
            }},
            {"start_object_task_region_m", region_json(options.start_region)},
            {"goal_object_task_region_m", region_json(options.goal_region)},
            {"validation", {
                {"joint_limits", true},
                {"dual_arm_relative_pose_constraint", true},
                {"robot_self_collision", true},
                {"robot_environment_collision", true},
                {"attached_object_robot_collision", true},
                {"attached_object_environment_collision", true},
            }},
        }},
        {"problems", {{options.problem_name, std::move(generated_problems)}}},
    };
    if (options.axis) {
        output["generator"]["axis"] = true;
        output["generator"]["constraint_dimension"] = 8;
        output["generator"]["validation"]["left_gripper_axis_constraint"] = true;
    }
    return output;
}

void write_json(const std::filesystem::path &path, const json &document) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary);
        if (!output) {
            throw std::runtime_error("failed to create output: " + temporary);
        }
        output << std::setprecision(9) << document.dump(2) << '\n';
    }
    std::filesystem::rename(temporary, path);
}

} // namespace

int main(int argc, char **argv) {
    DeviceEnvironment device_environment;
    try {
        const Options options = parse_options(argc, argv);
        std::ifstream input(options.template_path);
        if (!input) {
            throw std::runtime_error(
                "failed to open template: " + options.template_path.string()
            );
        }
        const json document = json::parse(input);
        const auto &problem_set =
            document.at("problems").at(options.source_name);
        if (
            options.source_index < 1 ||
            options.source_index > static_cast<int>(problem_set.size())
        ) {
            throw std::invalid_argument("source problem index is out of range");
        }
        const json source_problem = problem_set.at(options.source_index - 1);
        const Configuration reference_start = json_configuration(
            source_problem.at("start"), "start"
        );
        const Configuration reference_goal = json_configuration(
            source_problem.at("goals").at(0), "goals[0]"
        );

        PATACON_settings settings;
        ffw_sg2_attached_object_collision::apply_from_problem(
            source_problem,
            settings,
            ffw_sg2_attached_object_collision::kFfwSg2FixedFineSphereCount,
            ffw_sg2_attached_object_collision::kFfwSg2FixedApproxSphereCount
        );
        check_cuda(
            cudaMemcpyToSymbol(
                ppln::collision::ffw_sg2_mobility_attached_object_collision,
                &settings.ffw_sg2_attached_object_collision,
                sizeof(settings.ffw_sg2_attached_object_collision)
            ),
            "cudaMemcpyToSymbol attached object"
        );
        device_environment = make_device_environment(source_problem);

        std::mt19937_64 rng(options.seed);
        const auto starts = sample_pool(
            reference_start,
            options.start_sigma,
            options.start_region,
            options.count,
            options.maximum_candidates,
            rng,
            device_environment.environment,
            options.axis,
            "start"
        );
        const auto goals = sample_pool(
            reference_goal,
            options.goal_sigma,
            options.goal_region,
            options.count,
            options.maximum_candidates,
            rng,
            device_environment.environment,
            options.axis,
            "goal"
        );
        const json output = generate_output(
            options,
            source_problem,
            starts,
            goals
        );
        write_json(options.output_path, output);
        destroy_device_environment(device_environment);

        std::cout << "generated " << options.count
                  << " collision-free FFW-SG2 start-goal pairs: "
                  << options.output_path << "\n";
        return 0;
    } catch (const std::exception &error) {
        destroy_device_environment(device_environment);
        std::cerr << "generate_ffw_sg2_random_pairs: "
                  << error.what() << "\n";
        return 1;
    }
}

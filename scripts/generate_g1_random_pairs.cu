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

#include "scripts/g1_problem.hh"
#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/planning/utils.cuh"
#include "src/robots/g1_attached_object_collision.cuh"
#include "src/robots/g1_collision.cuh"
#include "src/robots/g1_constraint.cuh"

using json = nlohmann::json;

namespace {

constexpr int kDimension = ppln::collision::G1_DIM;
constexpr int kDefaultCount = 100;
constexpr int kCandidateBatchSize = 1024;
constexpr int kDefaultMaximumCandidates = 200000;
constexpr std::uint64_t kDefaultSeed = 20260918ULL;
constexpr std::uint64_t kAxisSeed = 20260919ULL;
constexpr float kDuplicateDistanceSquared = 1.0e-8f;

using Configuration = std::array<float, kDimension>;

struct Region {
    float minimum[3]{};
    float maximum[3]{};
};

struct CandidateResult {
    float configuration[kDimension]{};
    float object_position[3]{};
    float constraint_error_squared = std::numeric_limits<float>::infinity();
    float equality_residual = std::numeric_limits<float>::infinity();
    float com_residual = std::numeric_limits<float>::infinity();
    float axis_residual = 0.0f;
    int projection_valid = 0;
    int collision_free = 0;
};

struct Options {
    std::filesystem::path template_path = "scripts/g1_problems.json";
    std::filesystem::path output_path =
        "scripts/g1_random_pairs_no_axis_100.json";
    std::string source_name = "humanoid_shelf";
    int source_index = 1;
    std::string problem_name =
        "humanoid_shelf_random_pairs_no_axis";
    int count = kDefaultCount;
    int maximum_candidates = kDefaultMaximumCandidates;
    std::uint64_t seed = kDefaultSeed;
    float start_sigma = 0.08f;
    float goal_sigma = 0.10f;
    bool axis = false;
    bool output_path_provided = false;
    bool problem_name_provided = false;
    bool seed_provided = false;
    bool start_region_provided = false;
    bool goal_region_provided = false;
    Region start_region{};
    Region goal_region{};
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
    const std::string text(value);
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size() || parsed <= 0) {
        throw std::invalid_argument(option + " must be a positive integer");
    }
    return parsed;
}

float parse_positive_float(const std::string &option, const char *value) {
    std::size_t consumed = 0;
    const std::string text(value);
    const float parsed = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(parsed) || parsed <= 0.0f) {
        throw std::invalid_argument(
            option + " must be a positive finite number"
        );
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

Region parse_region(
    int argc,
    char **argv,
    int &index,
    const std::string &option
) {
    if (index + 6 >= argc) {
        throw std::invalid_argument(
            option + " requires xmin xmax ymin ymax zmin zmax"
        );
    }
    std::array<float, 6> values{};
    for (float &value : values) {
        const std::string text(argv[++index]);
        std::size_t consumed = 0;
        value = std::stof(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value)) {
            throw std::invalid_argument(
                option + " values must be finite numbers"
            );
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
        << "Usage: generate_g1_random_pairs [options]\n"
        << "  --template PATH --output PATH\n"
        << "  --source-name NAME --source-index N\n"
        << "  --problem-name NAME --count N --seed N\n"
        << "  --start-sigma RAD --goal-sigma RAD\n"
        << "  --axis\n"
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
            options.output_path_provided = true;
        } else if (argument == "--source-name") {
            options.source_name = require_value();
        } else if (argument == "--source-index") {
            options.source_index = parse_positive_int(
                argument, require_value()
            );
        } else if (argument == "--problem-name") {
            options.problem_name = require_value();
            options.problem_name_provided = true;
        } else if (argument == "--count") {
            options.count = parse_positive_int(argument, require_value());
        } else if (argument == "--seed") {
            options.seed = parse_seed(require_value());
            options.seed_provided = true;
        } else if (argument == "--start-sigma") {
            options.start_sigma = parse_positive_float(
                argument, require_value()
            );
        } else if (argument == "--goal-sigma") {
            options.goal_sigma = parse_positive_float(
                argument, require_value()
            );
        } else if (argument == "--axis") {
            options.axis = true;
        } else if (argument == "--start-region") {
            options.start_region = parse_region(
                argc, argv, index, argument
            );
            options.start_region_provided = true;
        } else if (argument == "--goal-region") {
            options.goal_region = parse_region(
                argc, argv, index, argument
            );
            options.goal_region_provided = true;
        } else if (argument == "--max-candidates") {
            options.maximum_candidates = parse_positive_int(
                argument, require_value()
            );
        } else if (argument == "--help" || argument == "-h") {
            print_usage();
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.axis) {
        if (!options.output_path_provided) {
            options.output_path =
                "scripts/g1_random_pairs_axis_100.json";
        }
        if (!options.problem_name_provided) {
            options.problem_name =
                "humanoid_shelf_random_pairs_axis";
        }
        if (!options.seed_provided) {
            options.seed = kAxisSeed;
        }
    }
    return options;
}

template <typename Value>
std::array<float, 3> json_vec3(
    const Value &value,
    const std::string &field
) {
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

Configuration json_configuration(
    const json &value,
    const std::string &field
) {
    if (!value.is_array() || value.size() != kDimension) {
        throw std::invalid_argument(field + " must contain 35 values");
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
            position[0],
            position[1],
            position[2],
            entry.at("radius").get<float>()
        );
    }
    for (const auto &entry : problem.value("cylinder", json::array())) {
        const auto position = json_vec3(
            entry.at("position"), "cylinder.position"
        );
        const auto orientation = json_vec3(
            entry.at("orientation_euler_xyz"),
            "cylinder.orientation_euler_xyz"
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
            json_vec3(
                entry.at("orientation_euler_xyz"),
                "box.orientation_euler_xyz"
            ),
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

__device__ void g1_payload_center(
    const float configuration[kDimension],
    float position[3]
) {
    float transforms[48];
    ppln::collision::g1_end_effector_fk(configuration, transforms);
    const auto &spec = ppln::collision::g1_attached_object_collision;
    for (int axis = 0; axis < 3; ++axis) {
        position[axis] = transforms[axis]
            + transforms[3 + axis] * spec.left_hand_center_offset[0]
            + transforms[6 + axis] * spec.left_hand_center_offset[1]
            + transforms[9 + axis] * spec.left_hand_center_offset[2];
    }
}

__global__ void project_and_validate_candidates(
    const float *input,
    CandidateResult *results,
    ppln::collision::Environment<float> *environment,
    ppln::constraints::G1ConstraintParameters parameters,
    bool axis,
    int count
) {
    const int candidate = static_cast<int>(blockIdx.x);
    if (candidate >= count || threadIdx.x != 0) {
        return;
    }

    CandidateResult &result = results[candidate];
    float configuration[kDimension];
    for (int joint = 0; joint < kDimension; ++joint) {
        configuration[joint] = input[candidate * kDimension + joint];
    }
    result.projection_valid = ppln::collision::g1_project_configuration(
        configuration,
        parameters,
        axis,
        100,
        0.6f,
        1.0e-4f,
        0.12f
    ) ? 1 : 0;
    if (result.projection_valid == 0) {
        return;
    }

    for (int joint = 0; joint < kDimension; ++joint) {
        result.configuration[joint] = configuration[joint];
    }
    float residual[ppln::collision::G1_CONSTRAINT_DIM]{};
    ppln::collision::g1_constraint_residual(
        configuration, parameters, axis, residual
    );
    result.constraint_error_squared = 0.0f;
    const int constraint_dimension =
        ppln::collision::g1_constraint_dim(axis);
    for (int row = 0; row < constraint_dimension; ++row) {
        result.constraint_error_squared += residual[row] * residual[row];
    }
    result.equality_residual = 0.0f;
    for (int row = 0; row < ppln::collision::G1_FEET_CONSTRAINT_DIM; ++row) {
        result.equality_residual += residual[row] * residual[row];
    }
    for (int row = ppln::collision::G1_FEET_CONSTRAINT_DIM +
             ppln::collision::G1_COM_CONSTRAINT_DIM;
         row < ppln::collision::G1_BASE_CONSTRAINT_DIM;
         ++row) {
        result.equality_residual += residual[row] * residual[row];
    }
    if (axis) {
        for (int row = ppln::collision::G1_BASE_CONSTRAINT_DIM;
             row < ppln::collision::G1_CONSTRAINT_DIM;
             ++row) {
            result.equality_residual += residual[row] * residual[row];
            result.axis_residual += residual[row] * residual[row];
        }
        result.axis_residual = sqrtf(result.axis_residual);
    }
    result.equality_residual = sqrtf(result.equality_residual);
    result.com_residual = sqrtf(
        residual[ppln::collision::G1_FEET_CONSTRAINT_DIM] *
            residual[ppln::collision::G1_FEET_CONSTRAINT_DIM]
        + residual[ppln::collision::G1_FEET_CONSTRAINT_DIM + 1] *
            residual[ppln::collision::G1_FEET_CONSTRAINT_DIM + 1]
    );
    g1_payload_center(configuration, result.object_position);

    float robot_spheres[ppln::collision::G1_SPHERE_COUNT][4];
    ppln::collision::g1_sphere_fk(configuration, robot_spheres);
    result.collision_free =
        ppln::collision::g1_collision_free(configuration, environment) &&
        ppln::collision::g1_attached_object_collision_free(
            configuration, &robot_spheres[0][0], environment
        ) ? 1 : 0;
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

CandidateResult validate_reference(
    const Configuration &reference,
    ppln::collision::Environment<float> *device_environment,
    const ppln::constraints::G1ConstraintParameters &parameters,
    bool axis
) {
    float *device_input = nullptr;
    CandidateResult *device_result = nullptr;
    check_cuda(
        cudaMalloc(&device_input, sizeof(reference)),
        "cudaMalloc reference input"
    );
    check_cuda(
        cudaMalloc(&device_result, sizeof(CandidateResult)),
        "cudaMalloc reference result"
    );
    check_cuda(
        cudaMemcpy(
            device_input,
            reference.data(),
            sizeof(reference),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy reference input"
    );
    check_cuda(
        cudaMemset(device_result, 0, sizeof(CandidateResult)),
        "cudaMemset reference result"
    );
    project_and_validate_candidates<<<1, 1>>>(
        device_input,
        device_result,
        device_environment,
        parameters,
        axis,
        1
    );
    check_cuda(cudaGetLastError(), "reference validation launch");
    check_cuda(cudaDeviceSynchronize(), "reference validation sync");
    CandidateResult result{};
    check_cuda(
        cudaMemcpy(
            &result,
            device_result,
            sizeof(result),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy reference result"
    );
    cudaFree(device_result);
    cudaFree(device_input);
    if (result.projection_valid == 0 || result.collision_free == 0) {
        throw std::runtime_error(
            "source endpoint is not projection-valid and collision-free"
        );
    }
    return result;
}

Region centered_region(
    const float center[3],
    const std::array<float, 3> &half_width
) {
    Region result{};
    for (int axis = 0; axis < 3; ++axis) {
        result.minimum[axis] = center[axis] - half_width[axis];
        result.maximum[axis] = center[axis] + half_width[axis];
    }
    return result;
}

std::vector<CandidateResult> sample_pool(
    const Configuration &reference,
    float joint_sigma,
    const Region &region,
    int requested_count,
    int maximum_candidates,
    std::mt19937_64 &rng,
    ppln::collision::Environment<float> *device_environment,
    const ppln::constraints::G1ConstraintParameters &parameters,
    bool axis,
    const std::string &label
) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<CandidateResult> accepted;
    std::vector<float> host_input;
    std::vector<CandidateResult> host_results;
    float *device_input = nullptr;
    CandidateResult *device_results = nullptr;
    int projected_count = 0;
    int collision_free_count = 0;
    int residual_valid_count = 0;
    int region_valid_count = 0;
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
                float sigma = joint_sigma;
                if (joint < 3) {
                    sigma = 0.25f * joint_sigma;
                } else if (joint < 6) {
                    sigma = 0.4f * joint_sigma;
                } else if (joint < 18) {
                    sigma = 0.8f * joint_sigma;
                }
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
        project_and_validate_candidates<<<batch_count, 1>>>(
            device_input,
            device_results,
            device_environment,
            parameters,
            axis,
            batch_count
        );
        check_cuda(cudaGetLastError(), "candidate validation launch");
        check_cuda(cudaDeviceSynchronize(), "candidate validation sync");
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
            if (candidate.projection_valid != 0) {
                ++projected_count;
            }
            if (
                candidate.projection_valid != 0 &&
                candidate.collision_free != 0
            ) {
                ++collision_free_count;
            }
            const bool residual_valid =
                candidate.projection_valid != 0 &&
                candidate.collision_free != 0 &&
                std::isfinite(candidate.constraint_error_squared) &&
                candidate.constraint_error_squared <=
                    parameters.tolerance_squared;
            if (residual_valid) {
                ++residual_valid_count;
                if (inside_region(candidate.object_position, region)) {
                    ++region_valid_count;
                }
            }
            if (!residual_valid ||
                !inside_region(candidate.object_position, region)) {
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
                  << " candidates"
                  << " (projected=" << projected_count
                  << ", collision_free=" << collision_free_count
                  << ", residual_valid=" << residual_valid_count
                  << ", region_valid=" << region_valid_count << ")\n";
    }

    cudaFree(device_results);
    cudaFree(device_input);
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

json position_json(const CandidateResult &candidate) {
    return {
        candidate.object_position[0],
        candidate.object_position[1],
        candidate.object_position[2],
    };
}

std::vector<float> configuration_vector(const CandidateResult &candidate) {
    return std::vector<float>(
        candidate.configuration,
        candidate.configuration + kDimension
    );
}

json generate_output(
    const Options &options,
    const json &source_problem,
    const Region &start_region,
    const Region &goal_region,
    const std::vector<CandidateResult> &starts,
    const std::vector<CandidateResult> &goals
) {
    json generated_problems = json::array();
    for (int index = 0; index < options.count; ++index) {
        json problem = source_problem;
        const auto start = configuration_vector(starts[index]);
        const auto goal = configuration_vector(goals[index]);
        problem["start"] = start;
        problem["goals"] = json::array({goal});
        // g1_goals_from_problem intentionally reads this goal in both modes.
        // Keep it synchronized so the generated axis-disabled query is evaluated.
        problem["axis_endpoints"]["start"] = start;
        problem["axis_endpoints"]["goals"] =
            json::array({goal});
        problem["pair_id"] = index + 1;
        problem["sampling"] = {
            {"base_seed", options.seed},
            {"axis", options.axis},
            {"start_object_position", position_json(starts[index])},
            {"goal_object_position", position_json(goals[index])},
            {"start_constraint_error_squared",
                starts[index].constraint_error_squared},
            {"goal_constraint_error_squared",
                goals[index].constraint_error_squared},
            {"start_equality_residual", starts[index].equality_residual},
            {"goal_equality_residual", goals[index].equality_residual},
            {"start_com_residual", starts[index].com_residual},
            {"goal_com_residual", goals[index].com_residual},
            {"start_axis_residual", starts[index].axis_residual},
            {"goal_axis_residual", goals[index].axis_residual},
            {"endpoint_collision_free", true},
        };
        generated_problems.push_back(std::move(problem));
    }

    return {
        {"format", "g1_random_start_goal_pairs_v1"},
        {"generator", {
            {"executable", "build/generate_g1_random_pairs"},
            {"seed", options.seed},
            {"count", options.count},
            {"sampling", options.axis
                ? "Gaussian joint perturbation followed by G1 base-plus-axis-constraint projection and rejection"
                : "Gaussian joint perturbation followed by G1 base-constraint projection and rejection"},
            {"start_joint_sigma_rad", options.start_sigma},
            {"goal_joint_sigma_rad", options.goal_sigma},
            {"axis", options.axis},
            {"constraint_dimension",
                ppln::collision::G1_BASE_CONSTRAINT_DIM +
                (options.axis
                    ? ppln::collision::G1_AXIS_CONSTRAINT_DIM
                    : 0)},
            {"equality_constraint_dimension",
                ppln::collision::G1_FEET_CONSTRAINT_DIM +
                ppln::collision::G1_BIMANUAL_CONSTRAINT_DIM +
                (options.axis
                    ? ppln::collision::G1_AXIS_CONSTRAINT_DIM
                    : 0)},
            {"tangent_dimension", kDimension -
                ppln::collision::G1_FEET_CONSTRAINT_DIM -
                ppln::collision::G1_BIMANUAL_CONSTRAINT_DIM -
                (options.axis
                    ? ppln::collision::G1_AXIS_CONSTRAINT_DIM
                    : 0)},
            {"source_problem", {
                {"file", options.template_path.string()},
                {"name", options.source_name},
                {"index", options.source_index},
            }},
            {"start_object_task_region_m", region_json(start_region)},
            {"goal_object_task_region_m", region_json(goal_region)},
            {"validation", {
                {"joint_limits", true},
                {"feet_pose_constraint", true},
                {"center_of_mass_support_constraint", true},
                {"bimanual_relative_pose_constraint", true},
                {"axis_constraint",
                    options.axis},
                {"robot_self_collision", true},
                {"robot_environment_collision", true},
                {"attached_object_robot_collision", true},
                {"attached_object_environment_collision", true},
            }},
        }},
        {"problems", {{options.problem_name, std::move(generated_problems)}}},
    };
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

}  // namespace

int main(int argc, char **argv) {
    DeviceEnvironment device_environment;
    try {
        Options options = parse_options(argc, argv);
        std::ifstream input(options.template_path);
        if (!input) {
            throw std::runtime_error(
                "failed to open template: " + options.template_path.string()
            );
        }
        const json document = json::parse(input);
        const auto &problem_set = document.at("problems").at(
            options.source_name
        );
        if (
            options.source_index < 1 ||
            options.source_index > static_cast<int>(problem_set.size())
        ) {
            throw std::invalid_argument("source problem index is out of range");
        }
        const json source_problem = problem_set.at(options.source_index - 1);
        const Configuration reference_start = json_configuration(
            g1_start_from_problem(
                source_problem, options.axis
            ),
            "start"
        );
        const Configuration reference_goal = json_configuration(
            g1_goals_from_problem(
                source_problem, options.axis
            ).at(0),
            "goals[0]"
        );
        const auto parameters = g1_constraint_parameters_from_problem(
            source_problem
        );
        check_cuda(
            cudaMemcpyToSymbol(
                ppln::collision::g1_attached_object_collision,
                &parameters.attached_object_collision,
                sizeof(parameters.attached_object_collision)
            ),
            "cudaMemcpyToSymbol attached object"
        );
        device_environment = make_device_environment(source_problem);

        const CandidateResult validated_start = validate_reference(
            reference_start,
            device_environment.environment,
            parameters,
            options.axis
        );
        const CandidateResult validated_goal = validate_reference(
            reference_goal,
            device_environment.environment,
            parameters,
            options.axis
        );
        if (!options.start_region_provided) {
            options.start_region = centered_region(
                validated_start.object_position, {0.06f, 0.08f, 0.06f}
            );
        }
        if (!options.goal_region_provided) {
            options.goal_region = centered_region(
                validated_goal.object_position, {0.05f, 0.08f, 0.06f}
            );
        }
        std::cout << "reference start object center: "
                  << validated_start.object_position[0] << ' '
                  << validated_start.object_position[1] << ' '
                  << validated_start.object_position[2] << '\n'
                  << "reference goal object center: "
                  << validated_goal.object_position[0] << ' '
                  << validated_goal.object_position[1] << ' '
                  << validated_goal.object_position[2] << '\n';

        std::mt19937_64 rng(options.seed);
        const auto starts = sample_pool(
            reference_start,
            options.start_sigma,
            options.start_region,
            options.count,
            options.maximum_candidates,
            rng,
            device_environment.environment,
            parameters,
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
            parameters,
            options.axis,
            "goal"
        );
        const json output = generate_output(
            options,
            source_problem,
            options.start_region,
            options.goal_region,
            starts,
            goals
        );
        write_json(options.output_path, output);
        destroy_device_environment(device_environment);

        std::cout << "generated " << options.count
                  << " collision-free G1 "
                  << (options.axis
                      ? "axis"
                      : "no-axis")
                  << " start-goal pairs: "
                  << options.output_path << '\n';
        return 0;
    } catch (const std::exception &error) {
        destroy_device_environment(device_environment);
        std::cerr << "generate_g1_random_pairs: " << error.what() << '\n';
        return 1;
    }
}

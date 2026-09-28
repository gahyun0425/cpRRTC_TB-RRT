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

#include "scripts/franka_problem.hh"
#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/planning/RobotCollisionTraits.hh"
#include "src/planning/utils.cuh"
#include "src/robots/panda.cuh"
#include "src/robots/franka_collision.cuh"
#include "src/robots/franka_constraint.cuh"

using json = nlohmann::json;

namespace {

using Robot = ppln::robots::FrankaSingle;
using Configuration = Robot::Configuration;
using ConstraintParameters =
    ppln::constraints::FrankaConstraintParameters;

constexpr int kDimension = Robot::dimension;
constexpr int kThreads = 32;
constexpr int kDefaultCount = 100;
constexpr int kCandidateBatchSize = 512;
constexpr int kDefaultMaximumCandidates = 200000;
constexpr std::uint64_t kDefaultSeed = 20260920ULL;
constexpr std::uint64_t kRigidOrientationSeed = 20260921ULL;
constexpr float kDuplicateDistanceSquared = 1.0e-8f;
constexpr float kMinimumPairDistanceSquared = 0.04f;
constexpr float kProjectionTolerance = 1.0e-3f;
constexpr int kProjectionIterations = 100;
constexpr float kProjectionAlpha = 0.8f;
constexpr float kProjectionDamping = 1.0e-4f;
constexpr float kProjectionMaximumStep = 0.2f;

constexpr std::array<float, kDimension> kJointLower = {
    -2.8973f, -1.7628f, -2.8973f, -3.0718f,
    -2.8973f, -0.0175f, -2.8973f,
};
constexpr std::array<float, kDimension> kJointUpper = {
    2.8973f, 1.7628f, 2.8973f, -0.0698f,
    2.8973f, 3.7525f, 2.8973f,
};

struct Region {
    float minimum[3]{};
    float maximum[3]{};
};

struct CandidateInput {
    float start[kDimension]{};
    float goal[kDimension]{};
    ConstraintParameters parameters{};
};

struct CandidateResult {
    float start[kDimension]{};
    float goal[kDimension]{};
    float start_payload_position[3]{};
    float goal_payload_position[3]{};
    float start_axis_residual = std::numeric_limits<float>::infinity();
    float goal_axis_residual = std::numeric_limits<float>::infinity();
    int projection_valid = 0;
    int start_collision_free = 0;
    int goal_collision_free = 0;
};

struct Options {
    std::filesystem::path template_path =
        "scripts/franka_single_problems.json";
    std::filesystem::path output_path =
        "scripts/franka_single_random_pairs_no_rigid_orientation_100.json";
    std::string source_name = "demo";
    int source_index = 1;
    std::string problem_name =
        "franka_single_random_pairs_no_rigid_orientation";
    int count = kDefaultCount;
    int maximum_candidates = kDefaultMaximumCandidates;
    std::uint64_t seed = kDefaultSeed;
    float start_sigma = 0.12f;
    float goal_sigma = 0.16f;
    bool rigid_orientation = false;
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
    const std::string text(value);
    std::size_t consumed = 0;
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size() || parsed <= 0) {
        throw std::invalid_argument(option + " must be a positive integer");
    }
    return parsed;
}

float parse_positive_float(const std::string &option, const char *value) {
    const std::string text(value);
    std::size_t consumed = 0;
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
        << "Usage: generate_franka_single_random_pairs [options]\n"
        << "  --template PATH --output PATH\n"
        << "  --source-name NAME --source-index N\n"
        << "  --problem-name NAME --count N --seed N\n"
        << "  --start-sigma RAD --goal-sigma RAD\n"
        << "  --rigid-orientation\n"
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
        } else if (argument == "--rigid-orientation") {
            options.rigid_orientation = true;
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
    if (options.rigid_orientation) {
        if (!options.output_path_provided) {
            options.output_path =
                "scripts/franka_single_random_pairs_rigid_orientation_100.json";
        }
        if (!options.problem_name_provided) {
            options.problem_name =
                "franka_single_random_pairs_rigid_orientation";
        }
        if (!options.seed_provided) {
            options.seed = kRigidOrientationSeed;
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
        throw std::invalid_argument(field + " must contain 7 values");
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
            position[0], position[1], position[2],
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

__device__ bool project_configuration(
    float configuration[kDimension],
    const ConstraintParameters &parameters,
    bool rigid_orientation
) {
    ppln::collision::franka_clamp_configuration<Robot>(configuration);
    for (int iteration = 0; iteration < kProjectionIterations; ++iteration) {
        float correction[kDimension]{};
        float error = 1.0e30f;
        if (!ppln::collision::franka_task_correction<Robot>(
                configuration,
                parameters,
                rigid_orientation,
                kProjectionDamping,
                kProjectionMaximumStep,
                correction,
                error
            )) {
            return false;
        }
        if (error < kProjectionTolerance) {
            return true;
        }
        for (int joint = 0; joint < kDimension; ++joint) {
            configuration[joint] -= kProjectionAlpha * correction[joint];
        }
        ppln::collision::franka_clamp_configuration<Robot>(configuration);
    }
    return ppln::collision::franka_constraint_error_norm<Robot>(
        configuration, parameters, rigid_orientation
    ) < kProjectionTolerance;
}

__device__ void payload_position(
    const float configuration[kDimension],
    float position[3]
) {
    const float base[3] = {0.0f, 0.0f, 0.0f};
    ppln::collision::FrankaTransform end_effector{};
    ppln::collision::FrankaTransform relative{};
    ppln::collision::franka_arm_kinematics(
        configuration, base, end_effector
    );
    ppln::collision::franka_attached_object_relative_pose<Robot>(relative);
    const auto object_pose = ppln::collision::franka_compose(
        end_effector, relative
    );
    for (int axis = 0; axis < 3; ++axis) {
        position[axis] = object_pose.translation[axis];
    }
}

template <int FineSphereCount, int TransformSlots>
__device__ void validate_collision(
    const float configuration[kDimension],
    ppln::collision::Environment<float> *environment,
    volatile float *fine,
    float *transforms,
    volatile int *flags,
    int *collision_free
) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int index = tid;
         index < ppln::robots::CollisionTraits<Robot>::joint_flag_stride * 16;
         index += blockDim.x) {
        flags[index] = 1;
    }
    if (tid == 0) {
        *collision_free = 1;
    }
    __syncthreads();
    ppln::collision::fk<Robot>(configuration, fine, transforms, tid);
    __syncthreads();
    if (!ppln::collision::env_collision_check<Robot>(
            fine, flags, environment, tid
        )) {
        atomicExch(collision_free, 0);
    }
    if (!ppln::collision::self_collision_check<Robot>(fine, flags, tid)) {
        atomicExch(collision_free, 0);
    }
    if (!ppln::collision::franka_attached_object_env_collision_check<Robot>(
            configuration, environment, tid
        )) {
        atomicExch(collision_free, 0);
    }
    __syncthreads();
}

__global__ void project_and_validate_candidates(
    const CandidateInput *inputs,
    CandidateResult *results,
    ppln::collision::Environment<float> *environment,
    bool rigid_orientation,
    int count
) {
    using Traits = ppln::robots::CollisionTraits<Robot>;
    const int candidate = static_cast<int>(blockIdx.x);
    if (candidate >= count) return;

    __shared__ float configuration[kDimension];
    __shared__ float fine[Traits::fine_sphere_count * 16 * 3];
    __shared__ float transforms[Traits::transform_slots * 16 * 16];
    __shared__ int flags[Traits::joint_flag_stride * 16];
    __shared__ int collision_free;
    __shared__ int projection_valid;

    const int tid = static_cast<int>(threadIdx.x);
    CandidateResult &result = results[candidate];
    const CandidateInput &input = inputs[candidate];

    if (tid < kDimension) {
        result.start[tid] = input.start[tid];
        result.goal[tid] = input.goal[tid];
    }
    __syncthreads();
    if (tid == 0) {
        projection_valid = project_configuration(
            result.goal, input.parameters, rigid_orientation
        ) ? 1 : 0;
        result.projection_valid = projection_valid;
        result.start_axis_residual =
            ppln::collision::franka_constraint_error_norm<Robot>(
                result.start, input.parameters, rigid_orientation
            );
        result.goal_axis_residual = projection_valid != 0
            ? ppln::collision::franka_constraint_error_norm<Robot>(
                result.goal, input.parameters, rigid_orientation
            )
            : 1.0e30f;
        payload_position(result.start, result.start_payload_position);
        if (projection_valid != 0) {
            payload_position(result.goal, result.goal_payload_position);
        }
    }
    __syncthreads();
    if (projection_valid == 0) return;

    if (tid < kDimension) configuration[tid] = result.start[tid];
    __syncthreads();
    validate_collision<Traits::fine_sphere_count, Traits::transform_slots>(
        configuration, environment, fine, transforms, flags, &collision_free
    );
    if (tid == 0) result.start_collision_free = collision_free;
    __syncthreads();

    if (tid < kDimension) configuration[tid] = result.goal[tid];
    __syncthreads();
    validate_collision<Traits::fine_sphere_count, Traits::transform_slots>(
        configuration, environment, fine, transforms, flags, &collision_free
    );
    if (tid == 0) result.goal_collision_free = collision_free;
}

Configuration clamped_configuration(
    const Configuration &reference,
    float sigma,
    std::normal_distribution<float> &normal,
    std::mt19937_64 &rng
) {
    Configuration result{};
    for (int joint = 0; joint < kDimension; ++joint) {
        result[joint] = std::clamp(
            reference[joint] + sigma * normal(rng),
            kJointLower[joint],
            kJointUpper[joint]
        );
    }
    return result;
}

CandidateInput make_candidate_input(
    const Configuration &reference_start,
    const Configuration &reference_goal,
    const Options &options,
    std::normal_distribution<float> &normal,
    std::mt19937_64 &rng
) {
    const Configuration start = clamped_configuration(
        reference_start, options.start_sigma, normal, rng
    );
    const Configuration goal = clamped_configuration(
        reference_goal, options.goal_sigma, normal, rng
    );
    CandidateInput input{};
    std::copy(start.begin(), start.end(), input.start);
    std::copy(goal.begin(), goal.end(), input.goal);
    input.parameters =
        franka_constraint_parameters_from_start<Robot>(start);
    return input;
}

std::vector<CandidateResult> run_candidate_batch(
    const std::vector<CandidateInput> &inputs,
    ppln::collision::Environment<float> *environment,
    bool rigid_orientation,
    CandidateInput *device_inputs,
    CandidateResult *device_results
) {
    check_cuda(
        cudaMemcpy(
            device_inputs,
            inputs.data(),
            inputs.size() * sizeof(CandidateInput),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy candidate inputs"
    );
    check_cuda(
        cudaMemset(
            device_results,
            0,
            inputs.size() * sizeof(CandidateResult)
        ),
        "cudaMemset candidate results"
    );
    project_and_validate_candidates<<<inputs.size(), kThreads>>>(
        device_inputs,
        device_results,
        environment,
        rigid_orientation,
        static_cast<int>(inputs.size())
    );
    check_cuda(cudaGetLastError(), "candidate validation launch");
    check_cuda(cudaDeviceSynchronize(), "candidate validation sync");
    std::vector<CandidateResult> results(inputs.size());
    check_cuda(
        cudaMemcpy(
            results.data(),
            device_results,
            results.size() * sizeof(CandidateResult),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy candidate results"
    );
    return results;
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
    float result = 0.0f;
    for (int joint = 0; joint < kDimension; ++joint) {
        const float difference = left[joint] - right[joint];
        result += difference * difference;
    }
    return result;
}

bool duplicate_pair(
    const CandidateResult &candidate,
    const std::vector<CandidateResult> &accepted
) {
    return std::any_of(
        accepted.begin(),
        accepted.end(),
        [&](const CandidateResult &other) {
            return squared_distance(candidate.start, other.start) <
                    kDuplicateDistanceSquared &&
                squared_distance(candidate.goal, other.goal) <
                    kDuplicateDistanceSquared;
        }
    );
}

Region centered_region(
    const float center[3],
    const std::array<float, 3> &half_width
) {
    Region region{};
    for (int axis = 0; axis < 3; ++axis) {
        region.minimum[axis] = center[axis] - half_width[axis];
        region.maximum[axis] = center[axis] + half_width[axis];
    }
    return region;
}

std::vector<CandidateResult> sample_pairs(
    const Configuration &reference_start,
    const Configuration &reference_goal,
    const Options &options,
    ppln::collision::Environment<float> *environment
) {
    CandidateInput *device_inputs = nullptr;
    CandidateResult *device_results = nullptr;
    check_cuda(
        cudaMalloc(
            &device_inputs,
            kCandidateBatchSize * sizeof(CandidateInput)
        ),
        "cudaMalloc candidate inputs"
    );
    check_cuda(
        cudaMalloc(
            &device_results,
            kCandidateBatchSize * sizeof(CandidateResult)
        ),
        "cudaMalloc candidate results"
    );

    std::mt19937_64 rng(options.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<CandidateResult> accepted;
    int attempted = 0;
    int projected = 0;
    int collision_free = 0;
    int residual_valid = 0;
    int region_valid = 0;
    while (
        static_cast<int>(accepted.size()) < options.count &&
        attempted < options.maximum_candidates
    ) {
        const int batch_count = std::min(
            kCandidateBatchSize,
            options.maximum_candidates - attempted
        );
        std::vector<CandidateInput> inputs;
        inputs.reserve(batch_count);
        for (int candidate = 0; candidate < batch_count; ++candidate) {
            inputs.push_back(make_candidate_input(
                reference_start,
                reference_goal,
                options,
                normal,
                rng
            ));
        }
        const auto results = run_candidate_batch(
            inputs,
            environment,
            options.rigid_orientation,
            device_inputs,
            device_results
        );
        attempted += batch_count;
        for (const auto &candidate : results) {
            if (candidate.projection_valid != 0) ++projected;
            const bool collisions_ok = candidate.projection_valid != 0 &&
                candidate.start_collision_free != 0 &&
                candidate.goal_collision_free != 0;
            if (collisions_ok) ++collision_free;
            const bool residuals_ok = collisions_ok &&
                std::isfinite(candidate.start_axis_residual) &&
                std::isfinite(candidate.goal_axis_residual) &&
                candidate.start_axis_residual < kProjectionTolerance &&
                candidate.goal_axis_residual < kProjectionTolerance;
            if (residuals_ok) ++residual_valid;
            const bool regions_ok = residuals_ok &&
                inside_region(
                    candidate.start_payload_position,
                    options.start_region
                ) &&
                inside_region(
                    candidate.goal_payload_position,
                    options.goal_region
                );
            if (regions_ok) ++region_valid;
            if (!regions_ok ||
                squared_distance(candidate.start, candidate.goal) <
                    kMinimumPairDistanceSquared ||
                duplicate_pair(candidate, accepted)) {
                continue;
            }
            accepted.push_back(candidate);
            if (static_cast<int>(accepted.size()) >= options.count) break;
        }
        std::cout << "pairs: accepted " << accepted.size()
                  << "/" << options.count
                  << " after " << attempted << " candidates"
                  << " (projected=" << projected
                  << ", collision_free=" << collision_free
                  << ", residual_valid=" << residual_valid
                  << ", region_valid=" << region_valid << ")\n";
    }
    cudaFree(device_results);
    cudaFree(device_inputs);
    if (static_cast<int>(accepted.size()) < options.count) {
        throw std::runtime_error(
            "could not generate enough pairs; accepted " +
            std::to_string(accepted.size()) + " of " +
            std::to_string(options.count)
        );
    }
    return accepted;
}

CandidateResult validate_reference_pair(
    const Configuration &start,
    const Configuration &goal,
    bool rigid_orientation,
    ppln::collision::Environment<float> *environment
) {
    CandidateInput input{};
    std::copy(start.begin(), start.end(), input.start);
    std::copy(goal.begin(), goal.end(), input.goal);
    input.parameters =
        franka_constraint_parameters_from_start<Robot>(start);
    CandidateInput *device_input = nullptr;
    CandidateResult *device_result = nullptr;
    check_cuda(cudaMalloc(&device_input, sizeof(input)), "cudaMalloc reference");
    check_cuda(
        cudaMalloc(&device_result, sizeof(CandidateResult)),
        "cudaMalloc reference result"
    );
    const auto results = run_candidate_batch(
        {input},
        environment,
        rigid_orientation,
        device_input,
        device_result
    );
    cudaFree(device_result);
    cudaFree(device_input);
    const auto &result = results.front();
    if (
        result.projection_valid == 0 ||
        result.start_collision_free == 0 ||
        result.goal_collision_free == 0 ||
        result.start_axis_residual >= kProjectionTolerance ||
        result.goal_axis_residual >= kProjectionTolerance
    ) {
        throw std::runtime_error(
            "source endpoints are not projection-valid and collision-free"
        );
    }
    return result;
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

json position_json(const float position[3]) {
    return {position[0], position[1], position[2]};
}

std::vector<float> configuration_vector(
    const float configuration[kDimension]
) {
    return std::vector<float>(configuration, configuration + kDimension);
}

json generate_output(
    const Options &options,
    const json &source_problem,
    const std::vector<CandidateResult> &pairs
) {
    json generated_problems = json::array();
    for (int index = 0; index < options.count; ++index) {
        json problem = source_problem;
        problem["start"] = configuration_vector(pairs[index].start);
        problem["goals"] = json::array({
            configuration_vector(pairs[index].goal)
        });
        problem["pair_id"] = index + 1;
        problem["sampling"] = {
            {"base_seed", options.seed},
            {"rigid_orientation", options.rigid_orientation},
            {"start_payload_position_m",
                position_json(pairs[index].start_payload_position)},
            {"goal_payload_position_m",
                position_json(pairs[index].goal_payload_position)},
            {"start_axis_residual", pairs[index].start_axis_residual},
            {"goal_axis_residual", pairs[index].goal_axis_residual},
            {"endpoint_collision_free", true},
        };
        generated_problems.push_back(std::move(problem));
    }

    return {
        {"format", "franka_single_random_start_goal_pairs_v1"},
        {"generator", {
            {"executable", "build/generate_franka_single_random_pairs"},
            {"seed", options.seed},
            {"count", options.count},
            {"sampling", options.rigid_orientation
                ? "Gaussian joint perturbation followed by Franka single world-yaw-axis projection and rejection"
                : "Gaussian joint perturbation and rejection"},
            {"start_joint_sigma_rad", options.start_sigma},
            {"goal_joint_sigma_rad", options.goal_sigma},
            {"rigid_orientation", options.rigid_orientation},
            {"constraint_dimension", options.rigid_orientation
                ? ppln::collision::FRANKA_WORLD_YAW_CONSTRAINT_DIM
                : 0},
            {"tangent_dimension", options.rigid_orientation ? 5 : 7},
            {"source_problem", {
                {"file", options.template_path.string()},
                {"name", options.source_name},
                {"index", options.source_index},
            }},
            {"start_payload_task_region_m", region_json(options.start_region)},
            {"goal_payload_task_region_m", region_json(options.goal_region)},
            {"projection", {
                {"iterations", kProjectionIterations},
                {"alpha", kProjectionAlpha},
                {"damping", kProjectionDamping},
                {"maximum_step", kProjectionMaximumStep},
                {"tolerance", kProjectionTolerance},
            }},
            {"validation", {
                {"joint_limits", true},
                {"rigid_orientation_world_yaw_axis_constraint",
                    options.rigid_orientation},
                {"robot_self_collision", true},
                {"robot_environment_collision", true},
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
            source_problem.at("start"), "start"
        );
        const Configuration reference_goal = json_configuration(
            source_problem.at("goals").at(0), "goals[0]"
        );
        device_environment = make_device_environment(source_problem);
        const CandidateResult reference = validate_reference_pair(
            reference_start,
            reference_goal,
            options.rigid_orientation,
            device_environment.environment
        );
        if (!options.start_region_provided) {
            options.start_region = centered_region(
                reference.start_payload_position, {0.08f, 0.08f, 0.06f}
            );
        }
        if (!options.goal_region_provided) {
            options.goal_region = centered_region(
                reference.goal_payload_position, {0.08f, 0.08f, 0.06f}
            );
        }
        std::cout << "reference start payload center: "
                  << reference.start_payload_position[0] << ' '
                  << reference.start_payload_position[1] << ' '
                  << reference.start_payload_position[2] << '\n'
                  << "reference goal payload center: "
                  << reference.goal_payload_position[0] << ' '
                  << reference.goal_payload_position[1] << ' '
                  << reference.goal_payload_position[2] << '\n';

        const auto pairs = sample_pairs(
            reference_start,
            reference_goal,
            options,
            device_environment.environment
        );
        const json output = generate_output(options, source_problem, pairs);
        write_json(options.output_path, output);
        destroy_device_environment(device_environment);
        std::cout << "generated " << options.count
                  << " collision-free Franka single "
                  << (options.rigid_orientation
                      ? "rigid-orientation"
                      : "non-rigid-orientation")
                  << " start-goal pairs: "
                  << options.output_path << '\n';
        return 0;
    } catch (const std::exception &error) {
        destroy_device_environment(device_environment);
        std::cerr << "generate_franka_single_random_pairs: "
                  << error.what() << '\n';
        return 1;
    }
}

#include <nlohmann/json.hpp>

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/collision/factory.hh"
#include "src/planning/RobotCollisionTraits.hh"
#include "src/planning/utils.cuh"
#include "src/robots/franka_fer.cuh"
#include "src/robots/franka_collision.cuh"

using json = nlohmann::json;

template <typename Robot>
__global__ void validate_franka_collision_kernel(
    const float *configuration,
    ppln::collision::Environment<float> *environment,
    int *results
) {
    using Traits = ppln::robots::CollisionTraits<Robot>;
    __shared__ float fine[
        Traits::fine_sphere_count * Traits::batch_size * 3
    ];
    __shared__ float approximate[
        Traits::approximate_sphere_count * Traits::batch_size * 3
    ];
    __shared__ float transforms[
        Traits::transform_slots * Traits::batch_size * 16
    ];
    __shared__ int flags[Traits::joint_flag_stride * Traits::batch_size];
    const int tid = threadIdx.x;
    for (int index = tid;
         index < Traits::joint_flag_stride * Traits::batch_size;
         index += blockDim.x) {
        flags[index] = 0;
    }
    if (tid < 5) results[tid] = 1;
    __syncthreads();
    ppln::collision::fk_approx<Robot>(
        configuration, approximate, transforms, tid
    );
    __syncthreads();
    if (!ppln::collision::env_collision_check_approx<Robot>(
            approximate, flags, environment, tid)) {
        atomicExch(&results[0], 0);
    }
    if (!ppln::collision::self_collision_check_approx<Robot>(
            approximate, flags, tid)) {
        atomicExch(&results[1], 0);
    }
    __syncthreads();
    for (int index = tid;
         index < Traits::joint_flag_stride * Traits::batch_size;
         index += blockDim.x) {
        flags[index] = 1;
    }
    ppln::collision::fk<Robot>(configuration, fine, transforms, tid);
    __syncthreads();
    if (!ppln::collision::env_collision_check<Robot>(
            fine, flags, environment, tid)) {
        atomicExch(&results[2], 0);
    }
    if (!ppln::collision::self_collision_check<Robot>(fine, flags, tid)) {
        atomicExch(&results[3], 0);
    }
    if (!ppln::collision::franka_attached_object_env_collision_check<Robot>(
            configuration, environment, tid)) {
        atomicExch(&results[4], 0);
    }
}

template <typename Robot>
std::array<int, 5> check_configuration(
    const typename Robot::Configuration &configuration,
    ppln::collision::Environment<float> *environment
) {
    float *device_configuration = nullptr;
    int *device_results = nullptr;
    cudaMalloc(&device_configuration, sizeof(configuration));
    cudaMalloc(&device_results, 5 * sizeof(int));
    cudaMemcpy(device_configuration, configuration.data(), sizeof(configuration),
               cudaMemcpyHostToDevice);
    validate_franka_collision_kernel<Robot><<<1, 64>>>(
        device_configuration, environment, device_results
    );
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(sync));
    }
    std::array<int, 5> results{};
    cudaMemcpy(results.data(), device_results, sizeof(results),
               cudaMemcpyDeviceToHost);
    cudaFree(device_results);
    cudaFree(device_configuration);
    return results;
}

template <typename Robot>
int validate_problem(const json &problem) {
    const auto start = problem.at("start").get<typename Robot::Configuration>();
    const auto goal = problem.at("goals").at(0).get<typename Robot::Configuration>();
    std::vector<ppln::collision::Cuboid<float>> cuboids;
    for (const auto &box : problem.at("box")) {
        cuboids.push_back(ppln::collision::factory::cuboid::array(
            box.at("position"),
            box.at("orientation_euler_xyz"),
            box.at("half_extents")
        ));
    }
    ppln::collision::Environment<float> *environment = nullptr;
    ppln::collision::Cuboid<float> *device_cuboids = nullptr;
    cudaMalloc(&environment, sizeof(*environment));
    cudaMemset(environment, 0, sizeof(*environment));
    if (!cuboids.empty()) {
        cudaMalloc(&device_cuboids, cuboids.size() * sizeof(cuboids.front()));
        cudaMemcpy(device_cuboids, cuboids.data(),
                   cuboids.size() * sizeof(cuboids.front()),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(&(environment->cuboids), &device_cuboids,
                   sizeof(device_cuboids), cudaMemcpyHostToDevice);
        const unsigned int count = static_cast<unsigned int>(cuboids.size());
        cudaMemcpy(&(environment->num_cuboids), &count, sizeof(count),
                   cudaMemcpyHostToDevice);
    }
    const auto start_results = check_configuration<Robot>(start, environment);
    const auto goal_results = check_configuration<Robot>(goal, environment);

    // Place a tiny synthetic obstacle at one known start-pose payload sphere
    // center.  This regression check proves that the attached-object path is
    // active independently of the task world's endpoint clearances.
    const std::array<float, 3> probe_center =
        std::is_same_v<Robot, ppln::robots::FrankaSingle>
        ? std::array<float, 3>{0.43f, 0.0f, 1.10f}
        : std::array<float, 3>{0.39f, -0.425f, 0.90f};
    const auto probe = ppln::collision::factory::cuboid::array(
        probe_center,
        std::array<float, 3>{0.0f, 0.0f, 0.0f},
        std::array<float, 3>{0.001f, 0.001f, 0.001f}
    );
    ppln::collision::Environment<float> *probe_environment = nullptr;
    ppln::collision::Cuboid<float> *device_probe = nullptr;
    cudaMalloc(&probe_environment, sizeof(*probe_environment));
    cudaMemset(probe_environment, 0, sizeof(*probe_environment));
    cudaMalloc(&device_probe, sizeof(probe));
    cudaMemcpy(
        device_probe, &probe, sizeof(probe), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        &(probe_environment->cuboids), &device_probe,
        sizeof(device_probe), cudaMemcpyHostToDevice
    );
    const unsigned int probe_count = 1;
    cudaMemcpy(
        &(probe_environment->num_cuboids), &probe_count,
        sizeof(probe_count), cudaMemcpyHostToDevice
    );
    const auto probe_results = check_configuration<Robot>(
        start, probe_environment
    );
    cudaFree(device_probe);
    cudaFree(probe_environment);

    cudaFree(device_cuboids);
    cudaFree(environment);
    auto print = [](const char *label, const std::array<int, 5> &result) {
        std::cout << label
                  << " approx_env=" << result[0]
                  << " approx_self=" << result[1]
                  << " fine_env=" << result[2]
                  << " fine_self=" << result[3]
                  << " attached_env=" << result[4] << '\n';
    };
    print("start", start_results);
    print("goal", goal_results);
    std::cout << "payload_probe detected=" << (probe_results[4] == 0)
              << '\n';
    return start_results[2] && start_results[3] &&
        start_results[4] &&
        goal_results[2] && goal_results[3] && goal_results[4] &&
        probe_results[4] == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
    const std::string robot = argc > 1 ? argv[1] : "franka_single";
    const std::string path = argc > 2
        ? argv[2]
        : "scripts/" + robot + "_problems.json";
    std::ifstream input(path);
    if (!input) {
        std::cerr << "failed to open " << path << '\n';
        return 1;
    }
    const auto root = json::parse(input);
    const auto &problem = root.at("problems").at("demo").at(0);
    if (robot == "franka_single") {
        return validate_problem<ppln::robots::FrankaSingle>(problem);
    }
    if (robot == "franka") {
        return validate_problem<ppln::robots::Franka>(problem);
    }
    std::cerr << "robot must be franka_single or franka\n";
    return 1;
}

#include <nlohmann/json.hpp>

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/collision/factory.hh"
#include "src/planning/utils.cuh"
#include "src/robots/g1_collision.cuh"

using json = nlohmann::json;

__global__ void validate_g1_configuration(
    const float *configuration,
    ppln::collision::Environment<float> *environment,
    bool *collision_free,
    float *first_sphere
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float spheres[ppln::collision::G1_SPHERE_COUNT][4];
    ppln::collision::g1_sphere_fk(configuration, spheres);
    for (int component = 0; component < 4; ++component) {
        first_sphere[component] = spheres[0][component];
    }
    *collision_free = ppln::collision::g1_collision_free(
        configuration,
        environment
    );
}

bool check_configuration(
    const std::array<float, ppln::collision::G1_DIM> &configuration,
    ppln::collision::Environment<float> *device_environment,
    std::array<float, 4> &first_sphere
) {
    float *device_configuration = nullptr;
    float *device_first_sphere = nullptr;
    bool *device_result = nullptr;
    cudaMalloc(&device_configuration, sizeof(configuration));
    cudaMalloc(&device_first_sphere, sizeof(first_sphere));
    cudaMalloc(&device_result, sizeof(bool));
    cudaMemcpy(
        device_configuration,
        configuration.data(),
        sizeof(configuration),
        cudaMemcpyHostToDevice
    );

    validate_g1_configuration<<<1, 1>>>(
        device_configuration,
        device_environment,
        device_result,
        device_first_sphere
    );
    cudaDeviceSynchronize();

    bool result = false;
    cudaMemcpy(&result, device_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        first_sphere.data(),
        device_first_sphere,
        sizeof(first_sphere),
        cudaMemcpyDeviceToHost
    );
    const cudaError_t error = cudaGetLastError();
    cudaFree(device_configuration);
    cudaFree(device_first_sphere);
    cudaFree(device_result);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    return result;
}

int main(int argc, char **argv) {
    const char *problem_path = argc > 1 ? argv[1] : "scripts/g1_problems.json";
    std::ifstream input(problem_path);
    if (!input) {
        std::cerr << "failed to open " << problem_path << "\n";
        return 1;
    }
    const json problems = json::parse(input);
    const json &problem = problems.at("problems").at("humanoid_shelf").at(0);
    const auto start = problem.at("start").get<std::array<float, ppln::collision::G1_DIM>>();
    const auto goal = problem.at("goals").at(0).get<std::array<float, ppln::collision::G1_DIM>>();

    ppln::collision::Environment<float> *device_environment = nullptr;
    cudaMalloc(&device_environment, sizeof(ppln::collision::Environment<float>));
    cudaMemset(device_environment, 0, sizeof(ppln::collision::Environment<float>));

    std::array<float, 4> start_sphere{};
    std::array<float, 4> goal_sphere{};
    const bool start_free = check_configuration(start, device_environment, start_sphere);
    const bool goal_free = check_configuration(goal, device_environment, goal_sphere);

    std::vector<ppln::collision::Cuboid<float>> problem_cuboids;
    for (const auto &box : problem.at("box")) {
        problem_cuboids.push_back(ppln::collision::factory::cuboid::array(
            box.at("position"),
            box.at("orientation_euler_xyz"),
            box.at("half_extents")
        ));
    }
    ppln::collision::Cuboid<float> *device_cuboids = nullptr;
    if (!problem_cuboids.empty()) {
        cudaMalloc(
            &device_cuboids,
            problem_cuboids.size() * sizeof(ppln::collision::Cuboid<float>)
        );
        cudaMemcpy(
            device_cuboids,
            problem_cuboids.data(),
            problem_cuboids.size() * sizeof(ppln::collision::Cuboid<float>),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            &(device_environment->cuboids),
            &device_cuboids,
            sizeof(device_cuboids),
            cudaMemcpyHostToDevice
        );
        const unsigned int cuboid_count = problem_cuboids.size();
        cudaMemcpy(
            &(device_environment->num_cuboids),
            &cuboid_count,
            sizeof(cuboid_count),
            cudaMemcpyHostToDevice
        );
    }
    std::array<float, 4> ignored_problem_sphere{};
    const bool start_problem_free = check_configuration(
        start,
        device_environment,
        ignored_problem_sphere
    );
    const bool goal_problem_free = check_configuration(
        goal,
        device_environment,
        ignored_problem_sphere
    );
    const unsigned int zero_count = 0;
    cudaMemcpy(
        &(device_environment->num_cuboids),
        &zero_count,
        sizeof(zero_count),
        cudaMemcpyHostToDevice
    );
    cudaFree(device_cuboids);

    ppln::collision::Sphere<float> obstacle(
        start_sphere[0], start_sphere[1], start_sphere[2], 0.01f
    );
    ppln::collision::Sphere<float> *device_obstacle = nullptr;
    cudaMalloc(&device_obstacle, sizeof(obstacle));
    cudaMemcpy(device_obstacle, &obstacle, sizeof(obstacle), cudaMemcpyHostToDevice);
    cudaMemcpy(
        &(device_environment->spheres),
        &device_obstacle,
        sizeof(device_obstacle),
        cudaMemcpyHostToDevice
    );
    const unsigned int obstacle_count = 1;
    cudaMemcpy(
        &(device_environment->num_spheres),
        &obstacle_count,
        sizeof(obstacle_count),
        cudaMemcpyHostToDevice
    );

    std::array<float, 4> ignored_sphere{};
    const bool blocked_start_free = check_configuration(
        start,
        device_environment,
        ignored_sphere
    );

    cudaFree(device_obstacle);
    cudaFree(device_environment);

    std::cout << "G1 collision validation\n"
              << "start empty environment: " << (start_free ? "free" : "collision") << "\n"
              << "goal empty environment: " << (goal_free ? "free" : "collision") << "\n"
              << "start problem environment: "
              << (start_problem_free ? "free" : "collision") << "\n"
              << "goal problem environment: "
              << (goal_problem_free ? "free" : "collision") << "\n"
              << "start with sphere obstacle: "
              << (blocked_start_free ? "free" : "collision") << "\n";

    const bool passed = start_free && goal_free && start_problem_free &&
        goal_problem_free && !blocked_start_free;
    std::cout << (passed ? "PASS" : "FAIL") << "\n";
    return passed ? 0 : 1;
}

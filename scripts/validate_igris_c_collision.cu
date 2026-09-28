#include <nlohmann/json.hpp>

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/collision/factory.hh"
#include "src/planning/utils.cuh"
#include "src/robots/igris_c_collision.cuh"

using json = nlohmann::json;

__global__ void validate_igris_c_configuration(
    const float *configuration,
    ppln::collision::Environment<float> *environment,
    bool *collision_free,
    bool *approximation_requests_detailed,
    float *selected_sphere,
    const int selected_sphere_index
) {
    __shared__ float sphere_positions[
        ppln::collision::IGRIS_C_SPHERE_COUNT *
        ppln::collision::IGRIS_C_COLLISION_BATCH_SIZE * 3
    ];
    __shared__ float approximate_sphere_positions[
        ppln::collision::IGRIS_C_APPROX_SPHERE_COUNT *
        ppln::collision::IGRIS_C_COLLISION_BATCH_SIZE * 3
    ];
    __shared__ int result;
    __shared__ int detailed;
    __shared__ int approximate_flags[1];
    const int tid = threadIdx.x;
    if (tid == 0) {
        result = 1;
        detailed = 0;
        approximate_flags[0] = 0;
    }
    __syncthreads();
    ppln::collision::fk_approx<ppln::robots::IgrisC>(
        configuration, approximate_sphere_positions, nullptr, tid
    );
    __syncthreads();
    if (!ppln::collision::env_collision_check_approx<ppln::robots::IgrisC>(
            approximate_sphere_positions,
            approximate_flags,
            environment,
            tid)) {
        atomicExch(&detailed, 1);
    }
    if (!ppln::collision::self_collision_check_approx<ppln::robots::IgrisC>(
            approximate_sphere_positions, approximate_flags, tid)) {
        atomicExch(&detailed, 1);
    }
    __syncthreads();
    ppln::collision::fk<ppln::robots::IgrisC>(
        configuration, sphere_positions, nullptr, tid
    );
    __syncthreads();
    if (!ppln::collision::env_collision_check<ppln::robots::IgrisC>(
            sphere_positions, nullptr, environment, tid)) {
        atomicExch(&result, 0);
    }
    if (!ppln::collision::self_collision_check<ppln::robots::IgrisC>(
            sphere_positions, nullptr, tid)) {
        atomicExch(&result, 0);
    }
    __syncthreads();
    if (tid == 0) {
        *collision_free = result != 0;
        *approximation_requests_detailed = detailed != 0;
        const int offset =
            selected_sphere_index *
            ppln::collision::IGRIS_C_COLLISION_BATCH_SIZE * 3;
        for (int component = 0; component < 3; ++component) {
            selected_sphere[component] = sphere_positions[offset + component];
        }
        selected_sphere[3] = ppln::collision::igris_c_collision_spheres[
            selected_sphere_index
        ].radius;
    }
}

bool check_configuration(
    const std::array<float, ppln::collision::IGRIS_C_DIM> &configuration,
    ppln::collision::Environment<float> *device_environment,
    std::array<float, 4> &selected_sphere,
    bool &approximation_requests_detailed,
    const int selected_sphere_index = 0
) {
    float *device_configuration = nullptr;
    float *device_selected_sphere = nullptr;
    bool *device_result = nullptr;
    bool *device_detailed = nullptr;
    cudaMalloc(&device_configuration, sizeof(configuration));
    cudaMalloc(&device_selected_sphere, sizeof(selected_sphere));
    cudaMalloc(&device_result, sizeof(bool));
    cudaMalloc(&device_detailed, sizeof(bool));
    cudaMemcpy(
        device_configuration,
        configuration.data(),
        sizeof(configuration),
        cudaMemcpyHostToDevice
    );
    validate_igris_c_configuration<<<1, 4>>>(
        device_configuration,
        device_environment,
        device_result,
        device_detailed,
        device_selected_sphere,
        selected_sphere_index
    );
    cudaDeviceSynchronize();
    bool result = false;
    cudaMemcpy(&result, device_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        &approximation_requests_detailed,
        device_detailed,
        sizeof(bool),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        selected_sphere.data(),
        device_selected_sphere,
        sizeof(selected_sphere),
        cudaMemcpyDeviceToHost
    );
    const cudaError_t error = cudaGetLastError();
    cudaFree(device_configuration);
    cudaFree(device_selected_sphere);
    cudaFree(device_result);
    cudaFree(device_detailed);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    return result;
}

int main(int argc, char **argv) {
    const char *problem_path =
        argc > 1 ? argv[1] : "scripts/igris_c_problems.json";
    std::ifstream input(problem_path);
    if (!input) {
        std::cerr << "failed to open " << problem_path << "\n";
        return 1;
    }
    const json root = json::parse(input);
    const json &problem =
        root.at("problems").at("igris_c_shelf_lift").at(0);
    const auto start = problem.at("start").get<
        std::array<float, ppln::collision::IGRIS_C_DIM>
    >();
    const auto goal = problem.at("goals").at(0).get<
        std::array<float, ppln::collision::IGRIS_C_DIM>
    >();

    ppln::collision::Environment<float> *device_environment = nullptr;
    cudaMalloc(&device_environment, sizeof(*device_environment));
    cudaMemset(device_environment, 0, sizeof(*device_environment));

    std::array<float, 4> start_payload_sphere{};
    std::array<float, 4> goal_sphere{};
    bool start_empty_detailed = false;
    bool goal_empty_detailed = false;
    const bool start_empty_free = check_configuration(
        start,
        device_environment,
        start_payload_sphere,
        start_empty_detailed,
        ppln::collision::IGRIS_C_PAYLOAD_SPHERE_BEGIN
    );
    const bool goal_empty_free = check_configuration(
        goal, device_environment, goal_sphere, goal_empty_detailed
    );

    std::vector<ppln::collision::Cuboid<float>> cuboids;
    for (const auto &box : problem.at("box")) {
        cuboids.push_back(ppln::collision::factory::cuboid::array(
            box.at("position"),
            box.at("orientation_euler_xyz"),
            box.at("half_extents")
        ));
    }
    ppln::collision::Cuboid<float> *device_cuboids = nullptr;
    cudaMalloc(&device_cuboids, cuboids.size() * sizeof(cuboids.front()));
    cudaMemcpy(
        device_cuboids,
        cuboids.data(),
        cuboids.size() * sizeof(cuboids.front()),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        &(device_environment->cuboids),
        &device_cuboids,
        sizeof(device_cuboids),
        cudaMemcpyHostToDevice
    );
    const unsigned int cuboid_count = cuboids.size();
    cudaMemcpy(
        &(device_environment->num_cuboids),
        &cuboid_count,
        sizeof(cuboid_count),
        cudaMemcpyHostToDevice
    );
    std::array<float, 4> ignored_sphere{};
    bool start_problem_detailed = false;
    bool goal_problem_detailed = false;
    const bool start_problem_free = check_configuration(
        start, device_environment, ignored_sphere, start_problem_detailed
    );
    const bool goal_problem_free = check_configuration(
        goal, device_environment, ignored_sphere, goal_problem_detailed
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
        start_payload_sphere[0],
        start_payload_sphere[1],
        start_payload_sphere[2],
        0.01f
    );
    ppln::collision::Sphere<float> *device_obstacle = nullptr;
    cudaMalloc(&device_obstacle, sizeof(obstacle));
    cudaMemcpy(
        device_obstacle, &obstacle, sizeof(obstacle), cudaMemcpyHostToDevice
    );
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
    bool blocked_start_detailed = false;
    const bool blocked_start_free = check_configuration(
        start, device_environment, ignored_sphere, blocked_start_detailed
    );

    cudaFree(device_obstacle);
    cudaFree(device_environment);

    std::cout
        << "IGRIS-C collision validation\n"
        << "fine spheres: " << ppln::collision::IGRIS_C_SPHERE_COUNT << "\n"
        << "approximate spheres: "
        << ppln::collision::IGRIS_C_APPROX_SPHERE_COUNT << "\n"
        << "self pairs: "
        << ppln::collision::IGRIS_C_SELF_COLLISION_PAIR_COUNT << "\n"
        << "start empty environment: "
        << (start_empty_free ? "free" : "collision") << "\n"
        << "goal empty environment: "
        << (goal_empty_free ? "free" : "collision") << "\n"
        << "start shelf environment: "
        << (start_problem_free ? "free" : "collision") << "\n"
        << "goal shelf environment: "
        << (goal_problem_free ? "free" : "collision") << "\n"
        << "payload spheres: "
        << ppln::collision::IGRIS_C_PAYLOAD_SPHERE_COUNT << "\n"
        << "start with payload-overlapping sphere obstacle: "
        << (blocked_start_free ? "free" : "collision") << "\n";
    const bool passed = start_empty_free && goal_empty_free &&
        start_problem_free && goal_problem_free && !blocked_start_free &&
        blocked_start_detailed;
    std::cout << (passed ? "PASS" : "FAIL") << "\n";
    return passed ? 0 : 1;
}

#include <nlohmann/json.hpp>

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/collision/factory.hh"
#include "src/planning/utils.cuh"
#include "src/robots/g1_collision.cuh"
#include "src/robots/g1_attached_object_collision.cuh"
#include "scripts/g1_problem.hh"

using json = nlohmann::json;

__global__ void validate_g1_configuration(
    const float *configuration,
    ppln::collision::Environment<float> *environment,
    bool *collision_free,
    float *selected_spheres
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float spheres[ppln::collision::G1_SPHERE_COUNT][4];
    ppln::collision::g1_sphere_fk(configuration, spheres);
    for (int component = 0; component < 4; ++component) {
        selected_spheres[component] = spheres[0][component];
        selected_spheres[4 + component] =
            spheres[ppln::collision::G1_BASE_SPHERE_COUNT][component];
        selected_spheres[8 + component] = spheres[
            ppln::collision::G1_BASE_SPHERE_COUNT +
            ppln::collision::G1_HAND_SPHERES_PER_HAND
        ][component];
    }
    *collision_free = ppln::collision::g1_collision_free(
        configuration,
        environment
    ) && ppln::collision::g1_attached_object_collision_free(
        configuration,
        &spheres[0][0],
        environment
    );
}

__global__ void diagnose_g1_payload_contacts(
    const float *configuration,
    unsigned char *robot_contacts,
    float *first_payload_sphere
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float robot_spheres[ppln::collision::G1_SPHERE_COUNT][4];
    ppln::collision::g1_sphere_fk(configuration, robot_spheres);
    const int payload_sphere_count =
        ppln::collision::g1_attached_object_sphere_count();
    float transforms[48];
    ppln::collision::g1_end_effector_fk(configuration, transforms);
    for (int robot_sphere = 0;
         robot_sphere < ppln::collision::G1_SPHERE_COUNT;
         ++robot_sphere) {
        robot_contacts[robot_sphere] = 0;
        for (int payload_sphere = 0;
             payload_sphere < payload_sphere_count;
             ++payload_sphere) {
            float world_sphere[4];
            ppln::collision::g1_attached_object_sphere_world_from_transform(
                transforms, payload_sphere, world_sphere
            );
            if (payload_sphere == 0) {
                for (int component = 0; component < 4; ++component) {
                    first_payload_sphere[component] = world_sphere[component];
                }
            }
            if (ppln::collision::sphere_sphere_self_collision(
                    world_sphere[0],
                    world_sphere[1],
                    world_sphere[2],
                    world_sphere[3],
                    robot_spheres[robot_sphere][0],
                    robot_spheres[robot_sphere][1],
                    robot_spheres[robot_sphere][2],
                    robot_spheres[robot_sphere][3])) {
                robot_contacts[robot_sphere] = 1;
                break;
            }
        }
    }
}

std::vector<int> payload_robot_contacts(
    const std::array<float, ppln::collision::G1_DIM> &configuration,
    std::array<float, 4> &first_payload_sphere
) {
    float *device_configuration = nullptr;
    float *device_first_payload_sphere = nullptr;
    unsigned char *device_contacts = nullptr;
    cudaMalloc(&device_configuration, sizeof(configuration));
    cudaMalloc(&device_first_payload_sphere, sizeof(first_payload_sphere));
    cudaMalloc(
        &device_contacts,
        ppln::collision::G1_SPHERE_COUNT * sizeof(unsigned char)
    );
    cudaMemcpy(
        device_configuration,
        configuration.data(),
        sizeof(configuration),
        cudaMemcpyHostToDevice
    );
    diagnose_g1_payload_contacts<<<1, 1>>>(
        device_configuration,
        device_contacts,
        device_first_payload_sphere
    );
    cudaDeviceSynchronize();

    std::array<unsigned char, ppln::collision::G1_SPHERE_COUNT> contacts{};
    cudaMemcpy(
        contacts.data(),
        device_contacts,
        contacts.size() * sizeof(unsigned char),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        first_payload_sphere.data(),
        device_first_payload_sphere,
        sizeof(first_payload_sphere),
        cudaMemcpyDeviceToHost
    );
    cudaFree(device_configuration);
    cudaFree(device_first_payload_sphere);
    cudaFree(device_contacts);

    std::vector<int> indices;
    for (int sphere = 0;
         sphere < ppln::collision::G1_SPHERE_COUNT;
         ++sphere) {
        if (contacts[sphere] != 0) {
            indices.push_back(sphere);
        }
    }
    return indices;
}

bool check_configuration(
    const std::array<float, ppln::collision::G1_DIM> &configuration,
    ppln::collision::Environment<float> *device_environment,
    std::array<float, 12> &selected_spheres
) {
    float *device_configuration = nullptr;
    float *device_selected_spheres = nullptr;
    bool *device_result = nullptr;
    cudaMalloc(&device_configuration, sizeof(configuration));
    cudaMalloc(&device_selected_spheres, sizeof(selected_spheres));
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
        device_selected_spheres
    );
    cudaDeviceSynchronize();

    bool result = false;
    cudaMemcpy(&result, device_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        selected_spheres.data(),
        device_selected_spheres,
        sizeof(selected_spheres),
        cudaMemcpyDeviceToHost
    );
    const cudaError_t error = cudaGetLastError();
    cudaFree(device_configuration);
    cudaFree(device_selected_spheres);
    cudaFree(device_result);
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
    return result;
}

int main(int argc, char **argv) {
    const char *problem_path = argc > 1 ? argv[1] : "scripts/g1_problems.json";
    const bool axis =
        argc > 2 && std::string(argv[2]) == "--axis";
    std::ifstream input(problem_path);
    if (!input) {
        std::cerr << "failed to open " << problem_path << "\n";
        return 1;
    }
    const json problems = json::parse(input);
    const json &problem = problems.at("problems").at("humanoid_shelf").at(0);
    const auto start = g1_start_from_problem(problem, axis)
        .get<std::array<float, ppln::collision::G1_DIM>>();
    const auto goal = g1_goals_from_problem(problem, axis)
        .at(0)
        .get<std::array<float, ppln::collision::G1_DIM>>();
    const auto parameters = g1_constraint_parameters_from_problem(problem);
    cudaMemcpyToSymbol(
        ppln::collision::g1_attached_object_collision,
        &parameters.attached_object_collision,
        sizeof(parameters.attached_object_collision)
    );

    std::array<float, 4> first_payload_sphere{};
    const auto start_payload_contacts = payload_robot_contacts(
        start, first_payload_sphere
    );
    std::array<float, 4> ignored_goal_payload_sphere{};
    const auto goal_payload_contacts = payload_robot_contacts(
        goal, ignored_goal_payload_sphere
    );

    ppln::collision::Environment<float> *device_environment = nullptr;
    cudaMalloc(&device_environment, sizeof(ppln::collision::Environment<float>));
    cudaMemset(device_environment, 0, sizeof(ppln::collision::Environment<float>));

    std::array<float, 12> start_sphere{};
    std::array<float, 12> goal_sphere{};
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
    std::array<float, 12> ignored_problem_sphere{};
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

    std::array<float, 12> ignored_sphere{};
    const bool blocked_start_free = check_configuration(
        start,
        device_environment,
        ignored_sphere
    );

    ppln::collision::Sphere<float> hand_obstacle(
        start_sphere[4], start_sphere[5], start_sphere[6], 0.005f
    );
    cudaMemcpy(
        device_obstacle,
        &hand_obstacle,
        sizeof(hand_obstacle),
        cudaMemcpyHostToDevice
    );
    const bool left_hand_blocked_start_free = check_configuration(
        start,
        device_environment,
        ignored_sphere
    );

    ppln::collision::Sphere<float> right_hand_obstacle(
        start_sphere[8], start_sphere[9], start_sphere[10], 0.005f
    );
    cudaMemcpy(
        device_obstacle,
        &right_hand_obstacle,
        sizeof(right_hand_obstacle),
        cudaMemcpyHostToDevice
    );
    const bool right_hand_blocked_start_free = check_configuration(
        start,
        device_environment,
        ignored_sphere
    );

    ppln::collision::Sphere<float> payload_obstacle(
        first_payload_sphere[0],
        first_payload_sphere[1],
        first_payload_sphere[2],
        0.005f
    );
    cudaMemcpy(
        device_obstacle,
        &payload_obstacle,
        sizeof(payload_obstacle),
        cudaMemcpyHostToDevice
    );
    const bool payload_blocked_start_free = check_configuration(
        start,
        device_environment,
        ignored_sphere
    );

    cudaFree(device_obstacle);
    cudaFree(device_environment);

    std::cout << "G1 collision validation (axis: "
              << (axis ? "on" : "off") << ")\n"
              << "start empty environment: " << (start_free ? "free" : "collision") << "\n"
              << "goal empty environment: " << (goal_free ? "free" : "collision") << "\n"
              << "start problem environment: "
              << (start_problem_free ? "free" : "collision") << "\n"
              << "goal problem environment: "
              << (goal_problem_free ? "free" : "collision") << "\n"
              << "start with sphere obstacle: "
              << (blocked_start_free ? "free" : "collision") << "\n"
              << "hand spheres: "
              << ppln::collision::G1_HAND_SPHERE_COUNT << "\n"
              << "start with left-hand obstacle: "
              << (left_hand_blocked_start_free ? "free" : "collision") << "\n"
              << "start with right-hand obstacle: "
              << (right_hand_blocked_start_free ? "free" : "collision") << "\n"
              << "payload sphere count: "
              << parameters.attached_object_collision.sphere_count << "\n"
              << "start payload/robot contacts:";
    for (const int sphere : start_payload_contacts) {
        std::cout << ' ' << sphere;
    }
    std::cout << "\ngoal payload/robot contacts:";
    for (const int sphere : goal_payload_contacts) {
        std::cout << ' ' << sphere;
    }
    std::cout << "\nstart with payload obstacle: "
              << (payload_blocked_start_free ? "free" : "collision") << "\n";

    const bool passed = start_free && goal_free && start_problem_free &&
        goal_problem_free && !blocked_start_free &&
        !left_hand_blocked_start_free &&
        !right_hand_blocked_start_free &&
        !payload_blocked_start_free;
    std::cout << (passed ? "PASS" : "FAIL") << "\n";
    return passed ? 0 : 1;
}

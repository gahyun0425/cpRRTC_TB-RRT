#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"

namespace ppln::config {

inline collision::Environment<float> environment_from_problem_json(
    const nlohmann::json &problem,
    const std::string &problem_name
) {
    collision::Environment<float> environment{};
    std::vector<collision::Sphere<float>> spheres;
    std::vector<collision::Capsule<float>> capsules;
    std::vector<collision::Cuboid<float>> cuboids;

    for (const auto &object : problem.at("sphere")) {
        const auto &position = object.at("position");
        collision::Sphere<float> sphere(
            position.at(0),
            position.at(1),
            position.at(2),
            object.at("radius")
        );
        sphere.name = object.at("name");
        spheres.push_back(sphere);
    }

    // Preserve the legacy MotionBenchMaker "box" interpretation where a
    // cylinder entry represents a cuboid obstacle.
    if (problem_name == "box") {
        for (const auto &object : problem.at("cylinder")) {
            const auto &position = object.at("position");
            const auto &orientation = object.at("orientation_euler_xyz");
            const float radius = object.at("radius");
            const std::array<float, 3> dimensions = {
                radius,
                radius,
                radius / 2.0f,
            };
            auto cuboid = collision::factory::cuboid::array(
                position,
                orientation,
                dimensions
            );
            cuboid.name = object.at("name");
            cuboids.push_back(cuboid);
        }
    } else {
        for (const auto &object : problem.at("cylinder")) {
            const auto &position = object.at("position");
            const auto &orientation = object.at("orientation_euler_xyz");
            auto capsule = collision::factory::cylinder::center::array(
                position,
                orientation,
                object.at("radius"),
                object.at("length")
            );
            capsule.name = object.at("name");
            capsules.push_back(capsule);
        }
    }

    for (const auto &object : problem.at("box")) {
        auto cuboid = collision::factory::cuboid::array(
            object.at("position"),
            object.at("orientation_euler_xyz"),
            object.at("half_extents")
        );
        cuboid.name = object.at("name");
        cuboids.push_back(cuboid);
    }

    if (!spheres.empty()) {
        environment.spheres = new collision::Sphere<float>[spheres.size()];
        std::copy(spheres.begin(), spheres.end(), environment.spheres);
        environment.num_spheres = spheres.size();
    }
    if (!capsules.empty()) {
        environment.capsules =
            new collision::Capsule<float>[capsules.size()];
        std::copy(capsules.begin(), capsules.end(), environment.capsules);
        environment.num_capsules = capsules.size();
    }
    if (!cuboids.empty()) {
        environment.cuboids = new collision::Cuboid<float>[cuboids.size()];
        std::copy(cuboids.begin(), cuboids.end(), environment.cuboids);
        environment.num_cuboids = cuboids.size();
    }

    return environment;
}

}  // namespace ppln::config

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/robots/igris_c_kinematics.cuh"

namespace {

constexpr int kDimension = ppln::collision::IGRIS_C_DIM;
constexpr int kTaskFrameCount = ppln::collision::IGRIS_C_TASK_FRAME_COUNT;
constexpr int kJacobianSize = 3 * kDimension;

struct KinematicsOutput {
    float frame_translation[kTaskFrameCount][3];
    float frame_rotation[kTaskFrameCount][9];
    float frame_position_jacobian[kTaskFrameCount][kJacobianSize];
    float frame_angular_jacobian[kTaskFrameCount][kJacobianSize];
    float center_of_mass[3];
    float center_of_mass_jacobian[kJacobianSize];
    float bimanual_translation[3];
    float bimanual_rotation[9];
    float bimanual_position_jacobian[kJacobianSize];
    float bimanual_angular_jacobian[kJacobianSize];
};

__global__ void evaluate_kinematics(
    const float *configuration,
    KinematicsOutput *output
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    ppln::collision::IgrisCTransform
        link_poses[ppln::collision::IGRIS_C_LINK_COUNT];
    float joint_origins[kDimension][3];
    float joint_axes[kDimension][3];
    ppln::collision::igris_c_forward_model(
        configuration,
        link_poses,
        joint_origins,
        joint_axes
    );

    for (int frame = 0; frame < kTaskFrameCount; ++frame) {
        ppln::collision::IgrisCTransform pose;
        ppln::collision::igris_c_frame_kinematics_from_model(
            ppln::collision::igris_c_task_link_indices[frame],
            link_poses,
            joint_origins,
            joint_axes,
            pose,
            output->frame_position_jacobian[frame],
            output->frame_angular_jacobian[frame]
        );
        for (int component = 0; component < 3; ++component) {
            output->frame_translation[frame][component] =
                pose.translation[component];
        }
        for (int index = 0; index < 9; ++index) {
            output->frame_rotation[frame][index] = pose.rotation[index];
        }
    }

    ppln::collision::igris_c_center_of_mass_from_model(
        link_poses,
        joint_origins,
        joint_axes,
        output->center_of_mass,
        output->center_of_mass_jacobian
    );

    ppln::collision::IgrisCTransform bimanual_pose;
    ppln::collision::igris_c_bimanual_from_model(
        link_poses,
        joint_origins,
        joint_axes,
        bimanual_pose,
        output->bimanual_position_jacobian,
        output->bimanual_angular_jacobian
    );
    for (int component = 0; component < 3; ++component) {
        output->bimanual_translation[component] =
            bimanual_pose.translation[component];
    }
    for (int index = 0; index < 9; ++index) {
        output->bimanual_rotation[index] = bimanual_pose.rotation[index];
    }
}

void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

float maximum_error(
    const float *actual,
    const nlohmann::json &expected_json,
    int count
) {
    const std::vector<double> expected =
        expected_json.get<std::vector<double>>();
    if (static_cast<int>(expected.size()) != count) {
        throw std::runtime_error("reference array has an unexpected size");
    }
    float result = 0.0f;
    for (int index = 0; index < count; ++index) {
        result = std::max(
            result,
            std::abs(actual[index] - static_cast<float>(expected[index]))
        );
    }
    return result;
}

struct ErrorSummary {
    float frame_translation = 0.0f;
    float frame_rotation = 0.0f;
    float frame_position_jacobian = 0.0f;
    float frame_angular_jacobian = 0.0f;
    float center_of_mass = 0.0f;
    float center_of_mass_jacobian = 0.0f;
    float bimanual_translation = 0.0f;
    float bimanual_rotation = 0.0f;
    float bimanual_position_jacobian = 0.0f;
    float bimanual_angular_jacobian = 0.0f;
};

void accumulate_errors(
    const KinematicsOutput &actual,
    const nlohmann::json &reference,
    const std::vector<std::string> &frame_order,
    ErrorSummary &errors
) {
    for (int frame = 0; frame < kTaskFrameCount; ++frame) {
        const auto &expected = reference.at("frames").at(frame_order.at(frame));
        errors.frame_translation = std::max(
            errors.frame_translation,
            maximum_error(
                actual.frame_translation[frame], expected.at("translation"), 3
            )
        );
        errors.frame_rotation = std::max(
            errors.frame_rotation,
            maximum_error(
                actual.frame_rotation[frame], expected.at("rotation_row_major"), 9
            )
        );
        errors.frame_position_jacobian = std::max(
            errors.frame_position_jacobian,
            maximum_error(
                actual.frame_position_jacobian[frame],
                expected.at("position_jacobian_row_major"),
                kJacobianSize
            )
        );
        errors.frame_angular_jacobian = std::max(
            errors.frame_angular_jacobian,
            maximum_error(
                actual.frame_angular_jacobian[frame],
                expected.at("world_angular_jacobian_row_major"),
                kJacobianSize
            )
        );
    }

    const auto &expected_com = reference.at("center_of_mass");
    errors.center_of_mass = std::max(
        errors.center_of_mass,
        maximum_error(actual.center_of_mass, expected_com.at("translation"), 3)
    );
    errors.center_of_mass_jacobian = std::max(
        errors.center_of_mass_jacobian,
        maximum_error(
            actual.center_of_mass_jacobian,
            expected_com.at("jacobian_row_major"),
            kJacobianSize
        )
    );

    const auto &expected_bimanual = reference.at("bimanual_relative_pose");
    errors.bimanual_translation = std::max(
        errors.bimanual_translation,
        maximum_error(
            actual.bimanual_translation,
            expected_bimanual.at("translation"),
            3
        )
    );
    errors.bimanual_rotation = std::max(
        errors.bimanual_rotation,
        maximum_error(
            actual.bimanual_rotation,
            expected_bimanual.at("rotation_row_major"),
            9
        )
    );
    errors.bimanual_position_jacobian = std::max(
        errors.bimanual_position_jacobian,
        maximum_error(
            actual.bimanual_position_jacobian,
            expected_bimanual.at("position_jacobian_row_major"),
            kJacobianSize
        )
    );
    errors.bimanual_angular_jacobian = std::max(
        errors.bimanual_angular_jacobian,
        maximum_error(
            actual.bimanual_angular_jacobian,
            expected_bimanual.at("left_frame_angular_jacobian_row_major"),
            kJacobianSize
        )
    );
}

void print_error(const char *name, float value) {
    std::cout << name << ": " << value << '\n';
}

}  // namespace

int main(int argc, char **argv) {
    const char *reference_path = argc > 1
        ? argv[1]
        : "resources/igris_c/kinematics_reference.json";
    try {
        std::ifstream input(reference_path);
        if (!input) {
            throw std::runtime_error(
                std::string("failed to open reference: ") + reference_path
            );
        }
        const nlohmann::json reference = nlohmann::json::parse(input);
        const auto frame_order =
            reference.at("task_frame_order").get<std::vector<std::string>>();
        if (frame_order.size() != kTaskFrameCount) {
            throw std::runtime_error("unexpected task-frame count");
        }

        float *device_configuration = nullptr;
        KinematicsOutput *device_output = nullptr;
        check_cuda(
            cudaMalloc(&device_configuration, kDimension * sizeof(float)),
            "cudaMalloc configuration"
        );
        check_cuda(
            cudaMalloc(&device_output, sizeof(KinematicsOutput)),
            "cudaMalloc output"
        );

        ErrorSummary errors;
        for (const auto &configuration_reference :
             reference.at("configurations")) {
            const auto configuration =
                configuration_reference.at("configuration")
                    .get<std::array<float, kDimension>>();
            check_cuda(
                cudaMemcpy(
                    device_configuration,
                    configuration.data(),
                    sizeof(configuration),
                    cudaMemcpyHostToDevice
                ),
                "copy configuration"
            );
            evaluate_kinematics<<<1, 1>>>(
                device_configuration,
                device_output
            );
            check_cuda(cudaGetLastError(), "launch evaluate_kinematics");
            check_cuda(cudaDeviceSynchronize(), "evaluate_kinematics");

            KinematicsOutput output{};
            check_cuda(
                cudaMemcpy(
                    &output,
                    device_output,
                    sizeof(output),
                    cudaMemcpyDeviceToHost
                ),
                "copy kinematics output"
            );
            accumulate_errors(
                output,
                configuration_reference,
                frame_order,
                errors
            );
        }
        cudaFree(device_output);
        cudaFree(device_configuration);

        std::cout << "IGRIS-C CUDA kinematics validation\n";
        print_error("frame translation max error", errors.frame_translation);
        print_error("frame rotation max error", errors.frame_rotation);
        print_error(
            "frame position Jacobian max error",
            errors.frame_position_jacobian
        );
        print_error(
            "frame angular Jacobian max error",
            errors.frame_angular_jacobian
        );
        print_error("CoM max error", errors.center_of_mass);
        print_error("CoM Jacobian max error", errors.center_of_mass_jacobian);
        print_error(
            "bimanual translation max error",
            errors.bimanual_translation
        );
        print_error("bimanual rotation max error", errors.bimanual_rotation);
        print_error(
            "bimanual position Jacobian max error",
            errors.bimanual_position_jacobian
        );
        print_error(
            "bimanual angular Jacobian max error",
            errors.bimanual_angular_jacobian
        );

        constexpr float kFkTolerance = 5.0e-6f;
        constexpr float kJacobianTolerance = 2.0e-3f;
        const bool passed =
            errors.frame_translation < kFkTolerance &&
            errors.frame_rotation < kFkTolerance &&
            errors.center_of_mass < kFkTolerance &&
            errors.bimanual_translation < kFkTolerance &&
            errors.bimanual_rotation < kFkTolerance &&
            errors.frame_position_jacobian < kJacobianTolerance &&
            errors.frame_angular_jacobian < kJacobianTolerance &&
            errors.center_of_mass_jacobian < kJacobianTolerance &&
            errors.bimanual_position_jacobian < kJacobianTolerance &&
            errors.bimanual_angular_jacobian < kJacobianTolerance;
        std::cout << (passed ? "PASS" : "FAIL") << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}

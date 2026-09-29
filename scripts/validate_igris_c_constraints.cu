#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "scripts/igris_c_problem.hh"
#include "src/robots/igris_c_constraint.cuh"

namespace {

constexpr int kDimension = ppln::collision::IGRIS_C_DIM;
constexpr int kConstraintDimension = ppln::collision::IGRIS_C_CONSTRAINT_DIM;
constexpr int kEqualityDimension =
    ppln::collision::IGRIS_C_EQUALITY_CONSTRAINT_DIM;
constexpr int kTangentDimension = ppln::collision::IGRIS_C_TANGENT_DIM;

struct ConstraintValidationResult {
    float start_error;
    float goal_error;
    float perturbed_error_before;
    float perturbed_error_after;
    float maximum_active_jacobian_error;
    int active_jacobian_rows;
    int projected;
    int tangent_basis_valid;
    float maximum_tangent_nullspace_error;
    float maximum_tangent_orthonormality_error;
    float start_residual[kConstraintDimension];
    float goal_residual[kConstraintDimension];
};

struct MotionProjectionResult {
    int success;
    int progress;
    float distance_before;
    float distance_after;
    float final_constraint_error;
};

__global__ void validate_constraints_kernel(
    const float *start,
    const float *goal,
    ppln::constraints::IgrisCConstraintParameters parameters,
    ConstraintValidationResult *result
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    result->start_error = ppln::collision::igris_c_constraint_error_squared(
        start, parameters
    );
    result->goal_error = ppln::collision::igris_c_constraint_error_squared(
        goal, parameters
    );
    ppln::collision::igris_c_constraint_residual(
        start, parameters, result->start_residual
    );
    ppln::collision::igris_c_constraint_residual(
        goal, parameters, result->goal_residual
    );

    float equality_residual[kEqualityDimension];
    float equality_jacobian[kEqualityDimension * kDimension];
    float tangent_basis[ppln::collision::IGRIS_C_TANGENT_BASIS_SIZE];
    ppln::collision::igris_c_equality_residual_and_jacobian(
        start, parameters, equality_residual, equality_jacobian
    );
    result->tangent_basis_valid =
        ppln::collision::igris_c_tangent_basis_from_jacobian(
            equality_jacobian, tangent_basis
        );
    result->maximum_tangent_nullspace_error = 0.0f;
    for (int row = 0; row < kEqualityDimension; ++row) {
        for (int column = 0; column < kTangentDimension; ++column) {
            float value = 0.0f;
            for (int joint = 0; joint < kDimension; ++joint) {
                value += equality_jacobian[row * kDimension + joint] *
                    tangent_basis[joint * kTangentDimension + column];
            }
            result->maximum_tangent_nullspace_error = fmaxf(
                result->maximum_tangent_nullspace_error, fabsf(value)
            );
        }
    }
    result->maximum_tangent_orthonormality_error = 0.0f;
    for (int left = 0; left < kTangentDimension; ++left) {
        for (int right = 0; right < kTangentDimension; ++right) {
            float value = 0.0f;
            for (int joint = 0; joint < kDimension; ++joint) {
                value += tangent_basis[joint * kTangentDimension + left] *
                    tangent_basis[joint * kTangentDimension + right];
            }
            const float expected = left == right ? 1.0f : 0.0f;
            result->maximum_tangent_orthonormality_error = fmaxf(
                result->maximum_tangent_orthonormality_error,
                fabsf(value - expected)
            );
        }
    }

    float perturbed[kDimension];
    for (int joint = 0; joint < kDimension; ++joint) {
        perturbed[joint] = start[joint];
    }
    // Activates foot, CoM, bimanual relative-pose, and axis rows.
    perturbed[0] += 0.08f;
    perturbed[3] += 0.04f;
    perturbed[21] += 0.10f;
    perturbed[23] -= 0.07f;

    result->perturbed_error_before =
        ppln::collision::igris_c_constraint_error_squared(
            perturbed, parameters
        );
    float analytic_residual[kConstraintDimension];
    float analytic_jacobian[kConstraintDimension * kDimension];
    ppln::collision::igris_c_constraint_residual_and_jacobian(
        perturbed, parameters, analytic_residual, analytic_jacobian
    );
    result->maximum_active_jacobian_error = 0.0f;
    result->active_jacobian_rows = 0;
    constexpr float difference_step = 1.0e-4f;
    for (int row = 0; row < kConstraintDimension; ++row) {
        if (fabsf(analytic_residual[row]) <= 1.0e-5f) continue;
        ++result->active_jacobian_rows;
        for (int joint = 0; joint < kDimension; ++joint) {
            float plus[kDimension];
            float minus[kDimension];
            for (int index = 0; index < kDimension; ++index) {
                plus[index] = perturbed[index];
                minus[index] = perturbed[index];
            }
            plus[joint] += difference_step;
            minus[joint] -= difference_step;
            float plus_residual[kConstraintDimension];
            float minus_residual[kConstraintDimension];
            ppln::collision::igris_c_constraint_residual(
                plus, parameters, plus_residual
            );
            ppln::collision::igris_c_constraint_residual(
                minus, parameters, minus_residual
            );
            const float numerical =
                (plus_residual[row] - minus_residual[row]) /
                (2.0f * difference_step);
            result->maximum_active_jacobian_error = fmaxf(
                result->maximum_active_jacobian_error,
                fabsf(numerical - analytic_jacobian[row * kDimension + joint])
            );
        }
    }
    result->projected = ppln::collision::igris_c_project_configuration(
        perturbed, parameters, 80, 0.75f, 1.0e-4f, 0.10f
    );
    result->perturbed_error_after =
        ppln::collision::igris_c_constraint_error_squared(
            perturbed, parameters
        );
}

__global__ void validate_node_smoothness_kernel(
    const float *start,
    const float *goal,
    ppln::constraints::IgrisCConstraintParameters parameters,
    float *motion,
    float *motion_next,
    unsigned char *valid,
    int *progress,
    unsigned int *success,
    MotionProjectionResult *result
) {
    const int tid = threadIdx.x;
    if (tid < kDimension) {
        motion[tid] = start[tid];
        motion[kDimension + tid] = goal[tid];
    }
    __syncthreads();
    if (tid == 0) {
        float squared = 0.0f;
        for (int joint = 0; joint < kDimension; ++joint) {
            const float difference = goal[joint] - start[joint];
            squared += difference * difference;
        }
        result->distance_before = sqrtf(squared);
    }
    __syncthreads();

    // Exact PATACON_TB-RRT node-anchor policy: for granularity=16 and base
    // threshold=0.03, the adjacent-node threshold is 16 * 0.03 = 0.48.
    const bool projected = ppln::collision::igris_c_project_motion(
        motion,
        motion_next,
        1,
        parameters,
        valid,
        progress,
        success,
        100,
        1.0f,
        8.0f,
        1.0f,
        1.0e-4f,
        1.0e-3f,
        16.0f * 0.03f,
        1.0f,
        true,
        0.20f,
        tid
    );
    __syncthreads();
    if (tid == 0) {
        result->success = projected ? 1 : 0;
        result->progress = progress[0];
        float squared = 0.0f;
        for (int joint = 0; joint < kDimension; ++joint) {
            const float difference = motion[kDimension + joint] - motion[joint];
            squared += difference * difference;
        }
        result->distance_after = sqrtf(squared);
        result->final_constraint_error =
            ppln::collision::igris_c_constraint_error_squared(
                &motion[kDimension], parameters
            );
    }
}

void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

}  // namespace

int main(int argc, char **argv) {
    const char *problem_path = argc > 1
        ? argv[1]
        : "scripts/igris_c_problems.json";
    try {
        std::ifstream input(problem_path);
        if (!input) {
            throw std::runtime_error(
                std::string("failed to open problem: ") + problem_path
            );
        }
        const auto root = nlohmann::json::parse(input);
        const auto &problem = root.at("problems").at("igris_c_shelf_lift").at(0);
        const auto parameters = igris_c_constraint_parameters_from_problem(problem);
        const auto start =
            problem.at("start").get<std::array<float, kDimension>>();
        const auto goal =
            problem.at("goals").at(0).get<std::array<float, kDimension>>();

        float *device_start = nullptr;
        float *device_goal = nullptr;
        ConstraintValidationResult *device_constraint_result = nullptr;
        MotionProjectionResult *device_motion_result = nullptr;
        float *device_motion = nullptr;
        float *device_motion_next = nullptr;
        unsigned char *device_valid = nullptr;
        int *device_progress = nullptr;
        unsigned int *device_success = nullptr;
        check_cuda(cudaMalloc(&device_start, sizeof(start)), "cudaMalloc start");
        check_cuda(cudaMalloc(&device_goal, sizeof(goal)), "cudaMalloc goal");
        check_cuda(cudaMalloc(
            &device_constraint_result, sizeof(ConstraintValidationResult)
        ), "cudaMalloc constraint result");
        check_cuda(cudaMalloc(
            &device_motion_result, sizeof(MotionProjectionResult)
        ), "cudaMalloc motion result");
        check_cuda(cudaMalloc(
            &device_motion, 2 * kDimension * sizeof(float)
        ), "cudaMalloc motion");
        check_cuda(cudaMalloc(
            &device_motion_next, 2 * kDimension * sizeof(float)
        ), "cudaMalloc motion next");
        check_cuda(cudaMalloc(&device_valid, 2), "cudaMalloc valid");
        check_cuda(cudaMalloc(&device_progress, sizeof(int)), "cudaMalloc progress");
        check_cuda(cudaMalloc(
            &device_success, sizeof(unsigned int)
        ), "cudaMalloc success");
        check_cuda(cudaMemcpy(
            device_start, start.data(), sizeof(start), cudaMemcpyHostToDevice
        ), "copy start");
        check_cuda(cudaMemcpy(
            device_goal, goal.data(), sizeof(goal), cudaMemcpyHostToDevice
        ), "copy goal");

        validate_constraints_kernel<<<1, 1>>>(
            device_start, device_goal, parameters, device_constraint_result
        );
        check_cuda(cudaDeviceSynchronize(), "constraint validation kernel");
        validate_node_smoothness_kernel<<<1, 64>>>(
            device_start,
            device_goal,
            parameters,
            device_motion,
            device_motion_next,
            device_valid,
            device_progress,
            device_success,
            device_motion_result
        );
        check_cuda(cudaDeviceSynchronize(), "motion projection kernel");

        ConstraintValidationResult constraint_result{};
        MotionProjectionResult motion_result{};
        check_cuda(cudaMemcpy(
            &constraint_result,
            device_constraint_result,
            sizeof(constraint_result),
            cudaMemcpyDeviceToHost
        ), "copy constraint result");
        check_cuda(cudaMemcpy(
            &motion_result,
            device_motion_result,
            sizeof(motion_result),
            cudaMemcpyDeviceToHost
        ), "copy motion result");

        cudaFree(device_success);
        cudaFree(device_progress);
        cudaFree(device_valid);
        cudaFree(device_motion_next);
        cudaFree(device_motion);
        cudaFree(device_motion_result);
        cudaFree(device_constraint_result);
        cudaFree(device_goal);
        cudaFree(device_start);

        std::cout
            << "IGRIS-C constraint validation\n"
            << "start error squared: " << constraint_result.start_error << '\n'
            << "goal error squared: " << constraint_result.goal_error << '\n'
            << "active Jacobian rows: "
            << constraint_result.active_jacobian_rows << '\n'
            << "maximum analytic Jacobian error: "
            << constraint_result.maximum_active_jacobian_error << '\n'
            << "tangent basis: "
            << (constraint_result.tangent_basis_valid ? "valid" : "invalid")
            << '\n'
            << "maximum tangent nullspace error: "
            << constraint_result.maximum_tangent_nullspace_error << '\n'
            << "maximum tangent orthonormality error: "
            << constraint_result.maximum_tangent_orthonormality_error << '\n'
            << "configuration projection: "
            << (constraint_result.projected ? "success" : "failure") << '\n'
            << "perturbed error before/after: "
            << constraint_result.perturbed_error_before << " / "
            << constraint_result.perturbed_error_after << '\n'
            << "node waypoint smoothness projection: "
            << (motion_result.success ? "success" : "failure") << '\n'
            << "node distance before/after/threshold: "
            << motion_result.distance_before << " / "
            << motion_result.distance_after << " / 0.48\n"
            << "node final constraint error: "
            << motion_result.final_constraint_error << '\n';

        const bool pass =
            constraint_result.start_error <= 1.0e-9f &&
            constraint_result.goal_error <= 1.0e-9f &&
            constraint_result.maximum_active_jacobian_error <= 3.0e-3f &&
            constraint_result.tangent_basis_valid != 0 &&
            constraint_result.maximum_tangent_nullspace_error <= 1.0e-4f &&
            constraint_result.maximum_tangent_orthonormality_error <= 1.0e-4f &&
            constraint_result.projected != 0 &&
            constraint_result.perturbed_error_after <= parameters.tolerance_squared &&
            motion_result.success != 0 &&
            motion_result.distance_after <= 0.4801f &&
            motion_result.final_constraint_error <= parameters.tolerance_squared;
        std::cout << (pass ? "PASS" : "FAIL") << '\n';
        return pass ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "IGRIS-C constraint validation error: "
                  << error.what() << '\n';
        return 1;
    }
}

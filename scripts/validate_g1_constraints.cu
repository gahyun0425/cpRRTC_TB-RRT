#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "scripts/g1_problem.hh"
#include "src/robots/g1_constraint.cuh"

namespace {

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
    float start_residual[ppln::collision::G1_CONSTRAINT_DIM];
    float goal_residual[ppln::collision::G1_CONSTRAINT_DIM];
};

__global__ void validate_constraints_kernel(
    const float *start,
    const float *goal,
    ppln::constraints::G1ConstraintParameters parameters,
    ConstraintValidationResult *result
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    result->start_error = ppln::collision::g1_constraint_error_squared(
        start,
        parameters
    );
    result->goal_error = ppln::collision::g1_constraint_error_squared(
        goal,
        parameters
    );
    ppln::collision::g1_constraint_residual(start, parameters, result->start_residual);
    ppln::collision::g1_constraint_residual(goal, parameters, result->goal_residual);

    float equality_residual[ppln::collision::G1_EQUALITY_CONSTRAINT_DIM];
    float equality_jacobian[
        ppln::collision::G1_EQUALITY_CONSTRAINT_DIM *
        ppln::collision::G1_JOINT_DIM
    ];
    float tangent_basis[ppln::collision::G1_TANGENT_BASIS_SIZE];
    ppln::collision::g1_equality_residual_and_jacobian(
        start,
        parameters,
        equality_residual,
        equality_jacobian
    );
    result->tangent_basis_valid = ppln::collision::g1_tangent_basis_from_jacobian(
        equality_jacobian,
        tangent_basis
    );
    result->maximum_tangent_nullspace_error = 0.0f;
    for (int row = 0;
         row < ppln::collision::G1_EQUALITY_CONSTRAINT_DIM;
         ++row) {
        for (int column = 0;
             column < ppln::collision::G1_TANGENT_DIM;
             ++column) {
            float value = 0.0f;
            for (int joint = 0;
                 joint < ppln::collision::G1_JOINT_DIM;
                 ++joint) {
                value += equality_jacobian[
                    row * ppln::collision::G1_JOINT_DIM + joint
                ] * tangent_basis[
                    joint * ppln::collision::G1_TANGENT_DIM + column
                ];
            }
            result->maximum_tangent_nullspace_error = fmaxf(
                result->maximum_tangent_nullspace_error,
                fabsf(value)
            );
        }
    }
    result->maximum_tangent_orthonormality_error = 0.0f;
    for (int left = 0;
         left < ppln::collision::G1_TANGENT_DIM;
         ++left) {
        for (int right = 0;
             right < ppln::collision::G1_TANGENT_DIM;
             ++right) {
            float value = 0.0f;
            for (int joint = 0;
                 joint < ppln::collision::G1_JOINT_DIM;
                 ++joint) {
                value += tangent_basis[
                    joint * ppln::collision::G1_TANGENT_DIM + left
                ] * tangent_basis[
                    joint * ppln::collision::G1_TANGENT_DIM + right
                ];
            }
            const float expected = left == right ? 1.0f : 0.0f;
            result->maximum_tangent_orthonormality_error = fmaxf(
                result->maximum_tangent_orthonormality_error,
                fabsf(value - expected)
            );
        }
    }

    float perturbed[35];
    for (int joint = 0; joint < 35; ++joint) {
        perturbed[joint] = start[joint];
    }
    perturbed[2] += 0.03f;
    perturbed[21] += 0.10f;
    result->perturbed_error_before =
        ppln::collision::g1_constraint_error_squared(perturbed, parameters);

    float analytic_residual[ppln::collision::G1_CONSTRAINT_DIM];
    float analytic_jacobian[
        ppln::collision::G1_CONSTRAINT_DIM * ppln::collision::G1_JOINT_DIM
    ];
    ppln::collision::g1_constraint_residual_and_jacobian(
        perturbed,
        parameters,
        analytic_residual,
        analytic_jacobian
    );
    result->maximum_active_jacobian_error = 0.0f;
    result->active_jacobian_rows = 0;
    constexpr float difference_step = 1.0e-4f;
    for (int row = 0; row < ppln::collision::G1_CONSTRAINT_DIM; ++row) {
        if (fabsf(analytic_residual[row]) <= 1.0e-4f) {
            continue;
        }
        ++result->active_jacobian_rows;
        for (int joint = 0; joint < ppln::collision::G1_JOINT_DIM; ++joint) {
            float plus[35];
            float minus[35];
            for (int index = 0; index < 35; ++index) {
                plus[index] = perturbed[index];
                minus[index] = perturbed[index];
            }
            plus[joint] += difference_step;
            minus[joint] -= difference_step;
            float plus_residual[ppln::collision::G1_CONSTRAINT_DIM];
            float minus_residual[ppln::collision::G1_CONSTRAINT_DIM];
            ppln::collision::g1_constraint_residual(
                plus,
                parameters,
                plus_residual
            );
            ppln::collision::g1_constraint_residual(
                minus,
                parameters,
                minus_residual
            );
            const float numerical =
                (plus_residual[row] - minus_residual[row]) /
                (2.0f * difference_step);
            const float error = fabsf(
                numerical - analytic_jacobian[row * 35 + joint]
            );
            result->maximum_active_jacobian_error = fmaxf(
                result->maximum_active_jacobian_error,
                error
            );
        }
    }
    result->projected = ppln::collision::g1_project_configuration(
        perturbed,
        parameters,
        50,
        0.5f,
        1.0e-4f,
        0.10f
    );
    result->perturbed_error_after =
        ppln::collision::g1_constraint_error_squared(perturbed, parameters);
}

void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

}  // namespace

int main() {
    std::ifstream input("scripts/g1_problems.json");
    if (!input) {
        std::cerr << "failed to open scripts/g1_problems.json\n";
        return 1;
    }
    nlohmann::json problems;
    input >> problems;
    const auto &problem = problems.at("problems").at("humanoid_shelf").at(0);
    const auto parameters = g1_constraint_parameters_from_problem(problem);
    const std::array<float, 35> start = problem.at("start").get<std::array<float, 35>>();
    const std::array<float, 35> goal = problem.at("goals").at(0).get<std::array<float, 35>>();

    float *device_start = nullptr;
    float *device_goal = nullptr;
    ConstraintValidationResult *device_result = nullptr;
    check_cuda(cudaMalloc(&device_start, sizeof(start)), "cudaMalloc start");
    check_cuda(cudaMalloc(&device_goal, sizeof(goal)), "cudaMalloc goal");
    check_cuda(cudaMalloc(&device_result, sizeof(ConstraintValidationResult)), "cudaMalloc result");
    check_cuda(cudaMemcpy(device_start, start.data(), sizeof(start), cudaMemcpyHostToDevice), "copy start");
    check_cuda(cudaMemcpy(device_goal, goal.data(), sizeof(goal), cudaMemcpyHostToDevice), "copy goal");

    validate_constraints_kernel<<<1, 1>>>(
        device_start,
        device_goal,
        parameters,
        device_result
    );
    check_cuda(cudaDeviceSynchronize(), "constraint validation kernel");

    ConstraintValidationResult result{};
    check_cuda(cudaMemcpy(
        &result,
        device_result,
        sizeof(result),
        cudaMemcpyDeviceToHost
    ), "copy result");
    cudaFree(device_result);
    cudaFree(device_goal);
    cudaFree(device_start);

    std::cout << "G1 constraint validation\n"
              << "start error squared: " << result.start_error << "\n"
              << "goal error squared: " << result.goal_error << "\n"
              << "perturbed before: " << result.perturbed_error_before << "\n"
              << "perturbed after: " << result.perturbed_error_after << "\n"
              << "active Jacobian rows: " << result.active_jacobian_rows << "\n"
              << "maximum analytic Jacobian error: "
              << result.maximum_active_jacobian_error << "\n"
              << "tangent basis: "
              << (result.tangent_basis_valid ? "valid" : "invalid") << "\n"
              << "maximum tangent nullspace error: "
              << result.maximum_tangent_nullspace_error << "\n"
              << "maximum tangent orthonormality error: "
              << result.maximum_tangent_orthonormality_error << "\n"
              << "projection result: " << (result.projected ? "success" : "failure") << "\n";
    std::cout << "start residual:";
    for (float value : result.start_residual) {
        std::cout << ' ' << value;
    }
    std::cout << "\ngoal residual:";
    for (float value : result.goal_residual) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';

    const bool endpoints_finite =
        std::isfinite(result.start_error) && std::isfinite(result.goal_error);
    const bool endpoints_satisfy_constraint =
        result.start_error <= parameters.tolerance_squared &&
        result.goal_error <= parameters.tolerance_squared;
    const bool projection_reduces_error =
        result.perturbed_error_after < result.perturbed_error_before;
    const bool analytic_jacobian_matches =
        result.active_jacobian_rows > 0 &&
        result.maximum_active_jacobian_error < 5.0e-3f;
    const bool tangent_basis_matches =
        result.tangent_basis_valid &&
        result.maximum_tangent_nullspace_error < 5.0e-3f &&
        result.maximum_tangent_orthonormality_error < 5.0e-3f;
    if (
        !endpoints_finite ||
        !endpoints_satisfy_constraint ||
        !projection_reduces_error ||
        !analytic_jacobian_matches ||
        !tangent_basis_matches ||
        !result.projected
    ) {
        std::cout << "FAIL\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}

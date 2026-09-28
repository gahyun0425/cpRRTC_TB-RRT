#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "scripts/franka_problem.hh"
#include "src/robots/franka_constraint.cuh"

struct FrankaValidationResult {
    float reference_error;
    float perturbed_error;
    float finite_difference_error;
    float null_space_error;
    float pure_world_yaw_error;
    float world_roll_error;
    int basis_ok;
    int residual_dimension;
    int tangent_dimension;
};

struct FrankaMotionValidationResult {
    int success;
    int progress;
    float goal_residual;
    float maximum_residual;
    float maximum_segment_length;
};

template <typename Robot>
__device__ void validate_franka_configuration(
    const float *reference,
    const ppln::constraints::FrankaConstraintParameters &parameters,
    bool axis_enabled,
    FrankaValidationResult &result
) {
    constexpr int dimension = Robot::dimension;
    constexpr int maximum_tangent_dimension =
        std::is_same_v<Robot, ppln::robots::FrankaSingle>
            ? ppln::collision::FRANKA_SINGLE_MAX_TANGENT_DIM
            : ppln::collision::FRANKA_DUAL_MAX_TANGENT_DIM;
    float q[dimension];
    for (int joint = 0; joint < dimension; ++joint) {
        const float direction = (joint % 3 == 0) ? 1.0f :
            ((joint % 3 == 1) ? -0.7f : 0.4f);
        q[joint] = reference[joint] + 0.025f * direction;
    }
    ppln::collision::franka_clamp_configuration<Robot>(q);
    result.reference_error = ppln::collision::franka_constraint_error_norm<Robot>(
        reference, parameters, axis_enabled
    );
    result.perturbed_error = ppln::collision::franka_constraint_error_norm<Robot>(
        q, parameters, axis_enabled
    );

    const float base[3] = {
        0.0f,
        std::is_same_v<Robot, ppln::robots::Franka> ? 0.2f : 0.0f,
        std::is_same_v<Robot, ppln::robots::Franka> ? 0.6f : 0.0f
    };
    ppln::collision::FrankaTransform reference_pose{};
    ppln::collision::franka_arm_kinematics(reference, base, reference_pose);
    constexpr float test_angle = 0.25f;
    const float cosine = cosf(test_angle);
    const float sine = sinf(test_angle);
    const float world_yaw_rotation[9] = {
        cosine, -sine, 0.0f,
        sine, cosine, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    const float world_roll_rotation[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, cosine, -sine,
        0.0f, sine, cosine
    };
    ppln::collision::FrankaTransform yaw_pose = reference_pose;
    ppln::collision::FrankaTransform roll_pose = reference_pose;
    ppln::collision::franka_multiply3(
        world_yaw_rotation, reference_pose.rotation, yaw_pose.rotation
    );
    ppln::collision::franka_multiply3(
        world_roll_rotation, reference_pose.rotation, roll_pose.rotation
    );
    float yaw_residual[ppln::collision::FRANKA_WORLD_YAW_CONSTRAINT_DIM]{};
    float roll_residual[ppln::collision::FRANKA_WORLD_YAW_CONSTRAINT_DIM]{};
    ppln::collision::franka_world_yaw_residual_and_jacobian(
        yaw_pose, nullptr, dimension, parameters, yaw_residual, nullptr
    );
    ppln::collision::franka_world_yaw_residual_and_jacobian(
        roll_pose, nullptr, dimension, parameters, roll_residual, nullptr
    );
    result.pure_world_yaw_error = sqrtf(
        yaw_residual[0] * yaw_residual[0] +
        yaw_residual[1] * yaw_residual[1]
    );
    result.world_roll_error = sqrtf(
        roll_residual[0] * roll_residual[0] +
        roll_residual[1] * roll_residual[1]
    );
    const int rows = ppln::collision::franka_constraint_dim<Robot>(axis_enabled);
    result.residual_dimension = rows;
    result.tangent_dimension = dimension - rows;

    float residual[ppln::collision::FRANKA_MAX_CONSTRAINT_DIM]{};
    float jacobian[ppln::collision::FRANKA_MAX_CONSTRAINT_DIM * dimension]{};
    ppln::collision::franka_constraint_residual_and_jacobian<Robot>(
        q, parameters, axis_enabled, residual, jacobian
    );
    result.finite_difference_error = 0.0f;
    constexpr float epsilon = 1.0e-3f;
    for (int joint = 0; joint < dimension; ++joint) {
        float plus[dimension];
        float minus[dimension];
        for (int index = 0; index < dimension; ++index) {
            plus[index] = q[index];
            minus[index] = q[index];
        }
        plus[joint] += epsilon;
        minus[joint] -= epsilon;
        float plus_residual[ppln::collision::FRANKA_MAX_CONSTRAINT_DIM]{};
        float minus_residual[ppln::collision::FRANKA_MAX_CONSTRAINT_DIM]{};
        ppln::collision::franka_constraint_residual_and_jacobian<Robot>(
            plus, parameters, axis_enabled, plus_residual, nullptr
        );
        ppln::collision::franka_constraint_residual_and_jacobian<Robot>(
            minus, parameters, axis_enabled, minus_residual, nullptr
        );
        for (int row = 0; row < rows; ++row) {
            const float numerical =
                (plus_residual[row] - minus_residual[row]) / (2.0f * epsilon);
            result.finite_difference_error = fmaxf(
                result.finite_difference_error,
                fabsf(numerical - jacobian[row * dimension + joint])
            );
        }
    }

    float basis[dimension * maximum_tangent_dimension]{};
    if constexpr (std::is_same_v<Robot, ppln::robots::FrankaSingle>) {
        result.basis_ok = ppln::collision::franka_single_tangent_basis(
            q, parameters, axis_enabled, basis
        );
    } else {
        result.basis_ok = ppln::collision::franka_dual_tangent_basis(
            q, parameters, axis_enabled, basis
        );
    }
    result.null_space_error = 0.0f;
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < dimension - rows; ++column) {
            float value = 0.0f;
            for (int joint = 0; joint < dimension; ++joint) {
                value += jacobian[row * dimension + joint] *
                    basis[joint * maximum_tangent_dimension + column];
            }
            result.null_space_error = fmaxf(
                result.null_space_error, fabsf(value)
            );
        }
    }
}

__global__ void validate_franka_single_kernel(
    const float *reference,
    ppln::constraints::FrankaConstraintParameters parameters,
    FrankaValidationResult *result
) {
    if (threadIdx.x == 0) {
        validate_franka_configuration<ppln::robots::FrankaSingle>(
            reference, parameters, true, result[0]
        );
    }
}

__global__ void validate_franka_dual_kernel(
    const float *reference,
    ppln::constraints::FrankaConstraintParameters parameters,
    FrankaValidationResult *result
) {
    if (threadIdx.x == 0) {
        validate_franka_configuration<ppln::robots::Franka>(
            reference, parameters, false, result[0]
        );
        validate_franka_configuration<ppln::robots::Franka>(
            reference, parameters, true, result[1]
        );
    }
}

template <typename Robot>
__global__ void validate_franka_motion_kernel(
    const float *start,
    const float *motion_goal,
    const float *constraint_goal,
    ppln::constraints::FrankaConstraintParameters parameters,
    bool axis_enabled,
    FrankaMotionValidationResult *result
) {
    constexpr int dimension = Robot::dimension;
    constexpr int granularity = 16;
    __shared__ float motion[(granularity + 1) * dimension];
    __shared__ float next[(granularity + 1) * dimension];
    __shared__ unsigned char valid[granularity + 1];
    __shared__ int progress;
    __shared__ unsigned int success;
    const int tid = threadIdx.x;
    const int waypoint = tid / 4 + 1;
    const int lane = tid % 4;
    if (tid < dimension) motion[tid] = start[tid];
    if (waypoint <= granularity) {
        for (int joint = lane; joint < dimension; joint += 4) {
            const float alpha = static_cast<float>(waypoint) / granularity;
            motion[waypoint * dimension + joint] =
                start[joint] + alpha * (motion_goal[joint] - start[joint]);
        }
    }
    __syncthreads();
    bool projected;
    if constexpr (std::is_same_v<Robot, ppln::robots::FrankaSingle>) {
        projected = ppln::collision::franka_single_project_motion(
            motion, next, granularity, parameters, axis_enabled,
            valid, &progress, &success, 60, 1.0f, 1.0e-4f, 1.0e-3f,
            0.03f, 1.0f, true, 0.2f, tid
        );
    } else {
        projected = ppln::collision::franka_dual_project_motion(
            motion, next, granularity, parameters, axis_enabled,
            valid, &progress, &success, 60, 1.0f, 1.0e-4f, 1.0e-3f,
            0.03f, 1.0f, true, 0.2f, tid
        );
    }
    __syncthreads();
    if (tid == 0) {
        result->success = projected ? 1 : 0;
        result->progress = progress;
        result->goal_residual =
            ppln::collision::franka_constraint_error_norm<Robot>(
                constraint_goal,
                parameters,
                axis_enabled
            );
        result->maximum_residual = 0.0f;
        result->maximum_segment_length = 0.0f;
        for (int point = 0; point <= granularity; ++point) {
            result->maximum_residual = fmaxf(
                result->maximum_residual,
                ppln::collision::franka_constraint_error_norm<Robot>(
                    motion + point * dimension,
                    parameters,
                    axis_enabled
                )
            );
            if (point > 0) {
                float squared = 0.0f;
                for (int joint = 0; joint < dimension; ++joint) {
                    const float difference =
                        motion[point * dimension + joint] -
                        motion[(point - 1) * dimension + joint];
                    squared += difference * difference;
                }
                result->maximum_segment_length = fmaxf(
                    result->maximum_segment_length, sqrtf(squared)
                );
            }
        }
    }
}

void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

int main() {
    using ppln::robots::Franka;
    using ppln::robots::FrankaSingle;
    FrankaSingle::Configuration single_start = {
        1.0182909966f, -0.2768633664f, -0.6489737630f,
        -0.9901697636f, -0.5397163630f, 2.3291680813f,
        -2.1444029808f
    };
    Franka::Configuration dual_start = {
        -0.4095856547f, 0.1583261788f, -0.3371204436f,
        -1.8272832632f, 0.0567844920f, 1.9762527943f,
        0.0201946385f, -0.3320198357f, -0.1133638993f,
        -0.1694837511f, -2.1494860649f, -0.0213693399f,
        2.0376393795f, -2.8470065594f
    };
    FrankaSingle::Configuration single_goal = {
        -0.0644896129f, -0.7252732337f, -0.0440191526f,
        -2.5226767883f, -0.4139727683f, 3.3857426814f,
        -1.9825247066f
    };
    Franka::Configuration dual_goal = {
        -0.2719763470f, 0.0813388918f, -0.3234448348f,
        -1.9176506937f, 0.0283352346f, 1.9946428706f,
        0.0416340353f, -0.0633963900f, -0.0305047526f,
        -0.2481479211f, -2.0496113952f, -0.0083176168f,
        2.0200251789f, -2.8017054184f
    };
    const auto single_parameters =
        franka_constraint_parameters_from_start<FrankaSingle>(single_start);
    const auto dual_parameters =
        franka_constraint_parameters_from_start<Franka>(dual_start);

    FrankaSingle::Configuration single_motion_goal = single_goal;
    float single_distance_squared = 0.0f;
    for (int joint = 0; joint < FrankaSingle::dimension; ++joint) {
        const float difference = single_goal[joint] - single_start[joint];
        single_distance_squared += difference * difference;
    }
    const float single_distance = std::sqrt(single_distance_squared);
    const float single_edge_scale = std::min(1.0f, 0.35f / single_distance);
    for (int joint = 0; joint < FrankaSingle::dimension; ++joint) {
        single_motion_goal[joint] = single_start[joint] + single_edge_scale *
            (single_goal[joint] - single_start[joint]);
    }

    float *device_single = nullptr;
    float *device_dual = nullptr;
    FrankaValidationResult *device_result = nullptr;
    FrankaMotionValidationResult *device_motion_result = nullptr;
    float *device_single_goal = nullptr;
    float *device_single_motion_goal = nullptr;
    float *device_dual_goal = nullptr;
    check_cuda(cudaMalloc(&device_single, sizeof(single_start)), "cudaMalloc single");
    check_cuda(cudaMalloc(&device_dual, sizeof(dual_start)), "cudaMalloc dual");
    check_cuda(cudaMalloc(&device_result, 2 * sizeof(FrankaValidationResult)),
               "cudaMalloc result");
    check_cuda(cudaMalloc(&device_motion_result, sizeof(FrankaMotionValidationResult)),
               "cudaMalloc motion result");
    check_cuda(cudaMalloc(&device_single_goal, sizeof(single_goal)),
               "cudaMalloc single goal");
    check_cuda(cudaMalloc(&device_single_motion_goal, sizeof(single_motion_goal)),
               "cudaMalloc single motion goal");
    check_cuda(cudaMalloc(&device_dual_goal, sizeof(dual_goal)),
               "cudaMalloc dual goal");
    check_cuda(cudaMemcpy(device_single, single_start.data(), sizeof(single_start),
                          cudaMemcpyHostToDevice), "copy single");
    check_cuda(cudaMemcpy(device_dual, dual_start.data(), sizeof(dual_start),
                          cudaMemcpyHostToDevice), "copy dual");
    check_cuda(cudaMemcpy(device_single_goal, single_goal.data(), sizeof(single_goal),
                          cudaMemcpyHostToDevice), "copy single goal");
    check_cuda(cudaMemcpy(device_single_motion_goal, single_motion_goal.data(),
                          sizeof(single_motion_goal), cudaMemcpyHostToDevice),
               "copy single motion goal");
    check_cuda(cudaMemcpy(device_dual_goal, dual_goal.data(), sizeof(dual_goal),
                          cudaMemcpyHostToDevice), "copy dual goal");

    FrankaValidationResult results[3]{};
    validate_franka_single_kernel<<<1, 1>>>(
        device_single, single_parameters, device_result
    );
    check_cuda(cudaDeviceSynchronize(), "single validation kernel");
    check_cuda(cudaMemcpy(&results[0], device_result, sizeof(results[0]),
                          cudaMemcpyDeviceToHost), "copy single result");
    validate_franka_dual_kernel<<<1, 1>>>(
        device_dual, dual_parameters, device_result
    );
    check_cuda(cudaDeviceSynchronize(), "dual validation kernel");
    check_cuda(cudaMemcpy(&results[1], device_result, 2 * sizeof(results[0]),
                          cudaMemcpyDeviceToHost), "copy dual result");

    FrankaMotionValidationResult motion_results[2]{};
    validate_franka_motion_kernel<FrankaSingle><<<1, 64>>>(
        device_single, device_single_motion_goal, device_single_goal,
        single_parameters, true,
        device_motion_result
    );
    check_cuda(cudaDeviceSynchronize(), "single motion validation kernel");
    check_cuda(cudaMemcpy(&motion_results[0], device_motion_result,
                          sizeof(motion_results[0]), cudaMemcpyDeviceToHost),
               "copy single motion result");
    validate_franka_motion_kernel<Franka><<<1, 64>>>(
        device_dual, device_dual_goal, device_dual_goal,
        dual_parameters, true,
        device_motion_result
    );
    check_cuda(cudaDeviceSynchronize(), "dual motion validation kernel");
    check_cuda(cudaMemcpy(&motion_results[1], device_motion_result,
                          sizeof(motion_results[1]), cudaMemcpyDeviceToHost),
               "copy dual motion result");

    cudaFree(device_dual_goal);
    cudaFree(device_single_motion_goal);
    cudaFree(device_single_goal);
    cudaFree(device_motion_result);
    cudaFree(device_result);
    cudaFree(device_dual);
    cudaFree(device_single);

    const char *labels[] = {
        "single-world-yaw", "dual-pose", "dual-pose-world-yaw"
    };
    bool success = true;
    for (int index = 0; index < 3; ++index) {
        const auto &result = results[index];
        std::cout << labels[index]
                  << " residual_dim=" << result.residual_dimension
                  << " tangent_dim=" << result.tangent_dimension
                  << " reference_error=" << result.reference_error
                  << " perturbed_error=" << result.perturbed_error
                  << " fd_error=" << result.finite_difference_error
                  << " null_error=" << result.null_space_error
                  << " pure_world_yaw_error=" << result.pure_world_yaw_error
                  << " world_roll_error=" << result.world_roll_error
                  << " basis_ok=" << result.basis_ok << '\n';
        success = success && result.basis_ok != 0
            && result.reference_error < 2.0e-5f
            && result.perturbed_error > 1.0e-5f
            && result.finite_difference_error < 2.0e-3f
            && result.null_space_error < 2.0e-4f
            && result.pure_world_yaw_error < 2.0e-5f
            && result.world_roll_error > 0.1f;
    }
    for (int index = 0; index < 2; ++index) {
        const auto &result = motion_results[index];
        std::cout << (index == 0 ? "single-motion" : "dual-motion")
                  << " success=" << result.success
                  << " progress=" << result.progress
                  << " goal_residual=" << result.goal_residual
                  << " max_residual=" << result.maximum_residual
                  << " max_segment=" << result.maximum_segment_length
                  << '\n';
        success = success && result.success != 0 && result.progress == 16
            && result.goal_residual < 2.0e-5f
            && result.maximum_residual < 1.1e-3f
            && result.maximum_segment_length < 0.031f;
    }
    return success ? 0 : 1;
}

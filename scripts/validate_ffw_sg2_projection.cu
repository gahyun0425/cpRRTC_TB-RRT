#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "src/planning/Robots.hh"
#include "src/planning/utils.cuh"
#include "src/robots/ffw_sg2.cuh"
#include "src/robots/ffw_sg2_constraint.cuh"

namespace {
constexpr int kDim = 15;
constexpr int kResidualDim = 6;
constexpr int kGranularity = 16;
constexpr float kTol = 1.0e-3f;

using Config = std::array<float, kDim>;

constexpr Config kStart = {
    0.0f,
    -0.02857562154531479f,
    0.0031426718924194574f,
    -0.09660343080759048f,
    -1.624609351158142f,
    1.5687326192855835f,
    -0.09664595872163773f,
    0.0823364332318306f,
    -0.18385747075080872f,
    -0.41972222924232483f,
    0.6199638247489929f,
    -1.6583945751190186f,
    -2.021341562271118f,
    -0.59946209192276f,
    0.0061826640740036964f,
};

constexpr Config kGoal = {
    -0.0450727977f,
    -1.37054718f,
    0.00346742105f,
    -0.138467208f,
    -0.888707638f,
    1.66112828f,
    -0.105139829f,
    0.683456421f,
    -1.39126396f,
    -0.0917326584f,
    0.348489195f,
    -0.895964622f,
    -1.86537063f,
    -0.209840178f,
    -0.669045746f,
};

void check_cuda(cudaError_t status, const char *label) {
    if (status != cudaSuccess) {
        std::cerr << label << ": " << cudaGetErrorString(status) << "\n";
        std::exit(2);
    }
}

Config lerp(const Config &a, const Config &b, float t) {
    Config out{};
    for (int i = 0; i < kDim; i++) {
        out[i] = a[i] + t * (b[i] - a[i]);
    }
    return out;
}

void clamp_to_limits(Config &q) {
    constexpr float lower[kDim] = {
        -0.5f, -3.14f, 0.0f, -3.14f, -2.9361f, -3.14f, -1.57f, -1.8201f,
        -3.14f, -3.14f, -3.14f, -2.9361f, -3.14f, -1.57f, -1.5804f,
    };
    constexpr float upper[kDim] = {
        0.0f, 3.14f, 3.14f, 3.14f, 1.0786f, 3.14f, 1.57f, 1.5804f,
        3.14f, 0.0f, 3.14f, 1.0786f, 3.14f, 1.57f, 1.8201f,
    };
    for (int i = 0; i < kDim; i++) {
        q[i] = std::min(std::max(q[i], lower[i]), upper[i]);
    }
}

std::vector<Config> make_config_cases() {
    std::vector<Config> cases;
    cases.push_back(kStart);
    cases.push_back(kGoal);
    for (int i = 1; i < 10; i++) {
        Config q = lerp(kStart, kGoal, static_cast<float>(i) / 10.0f);
        q[1] += 0.012f * static_cast<float>((i % 3) - 1);
        q[8] -= 0.010f * static_cast<float>((i % 4) - 1);
        q[4] += 0.006f * static_cast<float>(i % 2 == 0 ? 1 : -1);
        clamp_to_limits(q);
        cases.push_back(q);
    }
    return cases;
}

std::vector<Config> make_motion_pairs_flat() {
    std::vector<Config> pairs;
    pairs.push_back(kStart);
    pairs.push_back(kGoal);

    Config middle = lerp(kStart, kGoal, 0.5f);
    middle[1] += 0.02f;
    middle[8] -= 0.015f;
    clamp_to_limits(middle);
    pairs.push_back(kStart);
    pairs.push_back(middle);

    pairs.push_back(middle);
    pairs.push_back(kGoal);
    return pairs;
}
}

__global__ void validate_config_projection_kernel(
    const float *q_in,
    float *q_out,
    float *initial_norms,
    float *final_norms,
    unsigned int *success,
    int count
) {
    const int idx = blockIdx.x;
    if (idx >= count || threadIdx.x != 0) {
        return;
    }

    float q[kDim];
    for (int i = 0; i < kDim; i++) {
        q[i] = q_in[idx * kDim + i];
    }

    float h[kResidualDim];
    ppln::collision::ffw_sg2_relative_pose_residual(q, h);
    initial_norms[idx] = ppln::collision::ffw_sg2_residual_norm(h);

    const bool ok = ppln::collision::ffw_sg2_project_config(q, false);
    ppln::collision::ffw_sg2_relative_pose_residual(q, h);
    final_norms[idx] = ppln::collision::ffw_sg2_residual_norm(h);
    success[idx] = ok ? 1u : 0u;
    for (int i = 0; i < kDim; i++) {
        q_out[idx * kDim + i] = q[i];
    }
}

__global__ void validate_motion_projection_kernel(
    const float *pairs,
    float *max_final_norms,
    unsigned int *success,
    int pair_count
) {
    constexpr int kStrideDim = 16;
    const int idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (idx >= pair_count) {
        return;
    }

    __shared__ volatile float motion_segment[
        (kGranularity + 1) * kStrideDim
    ];
    __shared__ volatile float motion_segment_next[
        (kGranularity + 1) * kStrideDim
    ];
    __shared__ volatile unsigned char projection_valid[kGranularity + 1];
    __shared__ volatile int projection_prog[1];
    __shared__ volatile unsigned int projection_success[1];

    const float *q0 = pairs + (idx * 2) * kDim;
    const float *q1 = q0 + kDim;
    const int waypoint = tid / 4 + 1;
    const int lane = tid % 4;

    if (tid < kDim) {
        motion_segment[tid] = q0[tid];
    }
    if (waypoint <= kGranularity) {
        for (int j = lane; j < kDim; j += 4) {
            const float step =
                (q1[j] - q0[j]) / static_cast<float>(kGranularity);
            motion_segment[waypoint * kDim + j] =
                q0[j] + static_cast<float>(waypoint) * step;
        }
    }
    __syncthreads();

    const bool ok = ppln::collision::ffw_sg2_project_motion(
        motion_segment,
        motion_segment_next,
        kGranularity,
        false,
        projection_valid,
        projection_prog,
        projection_success,
        20,
        1.0f,
        1.0e-4f,
        kTol,
        0.03f,
        1.0f,
        true,
        0.20f,
        tid
    );

    if (tid == 0) {
        float max_norm = 0.0f;
        for (int w = 1; w <= kGranularity; w++) {
            float q[kDim];
            for (int j = 0; j < kDim; j++) {
                q[j] = motion_segment[w * kDim + j];
            }
            float h[kResidualDim];
            ppln::collision::ffw_sg2_relative_pose_residual(q, h);
            max_norm = fmaxf(
                max_norm,
                ppln::collision::ffw_sg2_residual_norm(h)
            );
        }
        max_final_norms[idx] = max_norm;
        success[idx] = ok ? 1u : 0u;
    }
}

int main() {
    const auto config_cases = make_config_cases();
    const int config_count = static_cast<int>(config_cases.size());
    std::vector<float> h_q_in(config_count * kDim);
    for (int i = 0; i < config_count; i++) {
        std::copy(config_cases[i].begin(), config_cases[i].end(), h_q_in.begin() + i * kDim);
    }

    float *d_q_in = nullptr;
    float *d_q_out = nullptr;
    float *d_initial_norms = nullptr;
    float *d_final_norms = nullptr;
    unsigned int *d_config_success = nullptr;

    check_cuda(cudaMalloc(&d_q_in, h_q_in.size() * sizeof(float)), "cudaMalloc q_in");
    check_cuda(cudaMalloc(&d_q_out, h_q_in.size() * sizeof(float)), "cudaMalloc q_out");
    check_cuda(cudaMalloc(&d_initial_norms, config_count * sizeof(float)), "cudaMalloc initial_norms");
    check_cuda(cudaMalloc(&d_final_norms, config_count * sizeof(float)), "cudaMalloc final_norms");
    check_cuda(cudaMalloc(&d_config_success, config_count * sizeof(unsigned int)), "cudaMalloc config_success");
    check_cuda(cudaMemcpy(d_q_in, h_q_in.data(), h_q_in.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy q_in");

    validate_config_projection_kernel<<<config_count, 1>>>(
        d_q_in,
        d_q_out,
        d_initial_norms,
        d_final_norms,
        d_config_success,
        config_count
    );
    check_cuda(cudaGetLastError(), "validate_config_projection_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "validate_config_projection_kernel sync");

    std::vector<float> h_initial_norms(config_count);
    std::vector<float> h_final_norms(config_count);
    std::vector<unsigned int> h_config_success(config_count);
    check_cuda(cudaMemcpy(h_initial_norms.data(), d_initial_norms, config_count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy initial_norms");
    check_cuda(cudaMemcpy(h_final_norms.data(), d_final_norms, config_count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy final_norms");
    check_cuda(cudaMemcpy(h_config_success.data(), d_config_success, config_count * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy config_success");

    const auto motion_pairs = make_motion_pairs_flat();
    const int pair_count = static_cast<int>(motion_pairs.size() / 2);
    std::vector<float> h_pairs(motion_pairs.size() * kDim);
    for (int i = 0; i < static_cast<int>(motion_pairs.size()); i++) {
        std::copy(motion_pairs[i].begin(), motion_pairs[i].end(), h_pairs.begin() + i * kDim);
    }

    float *d_pairs = nullptr;
    float *d_motion_max_norms = nullptr;
    unsigned int *d_motion_success = nullptr;
    check_cuda(cudaMalloc(&d_pairs, h_pairs.size() * sizeof(float)), "cudaMalloc pairs");
    check_cuda(cudaMalloc(&d_motion_max_norms, pair_count * sizeof(float)), "cudaMalloc motion norms");
    check_cuda(cudaMalloc(&d_motion_success, pair_count * sizeof(unsigned int)), "cudaMalloc motion success");
    check_cuda(cudaMemcpy(d_pairs, h_pairs.data(), h_pairs.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy pairs");

    validate_motion_projection_kernel<<<pair_count, 4 * kGranularity>>>(
        d_pairs,
        d_motion_max_norms,
        d_motion_success,
        pair_count
    );
    check_cuda(cudaGetLastError(), "validate_motion_projection_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "validate_motion_projection_kernel sync");

    std::vector<float> h_motion_max_norms(pair_count);
    std::vector<unsigned int> h_motion_success(pair_count);
    check_cuda(cudaMemcpy(h_motion_max_norms.data(), d_motion_max_norms, pair_count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy motion norms");
    check_cuda(cudaMemcpy(h_motion_success.data(), d_motion_success, pair_count * sizeof(unsigned int), cudaMemcpyDeviceToHost), "cudaMemcpy motion success");

    int config_success_count = 0;
    float config_initial_max = 0.0f;
    float config_final_max = 0.0f;
    for (int i = 0; i < config_count; i++) {
        config_success_count += h_config_success[i] != 0 ? 1 : 0;
        config_initial_max = std::max(config_initial_max, h_initial_norms[i]);
        config_final_max = std::max(config_final_max, h_final_norms[i]);
    }

    int motion_success_count = 0;
    float motion_final_max = 0.0f;
    for (int i = 0; i < pair_count; i++) {
        motion_success_count += h_motion_success[i] != 0 ? 1 : 0;
        motion_final_max = std::max(motion_final_max, h_motion_max_norms[i]);
    }

    std::cout << "FFW_SG2 CUDA projection validation\n";
    std::cout << "config cases: " << config_success_count << "/" << config_count << " succeeded\n";
    std::cout << "config initial residual max: " << config_initial_max << "\n";
    std::cout << "config final residual max: " << config_final_max << "\n";
    std::cout << "motion segments: " << motion_success_count << "/" << pair_count << " succeeded\n";
    std::cout << "motion final residual max: " << motion_final_max << "\n";

    const bool pass =
        config_success_count == config_count &&
        motion_success_count == pair_count &&
        config_final_max <= kTol &&
        motion_final_max <= kTol;
    std::cout << (pass ? "PASS" : "FAIL") << "\n";

    cudaFree(d_q_in);
    cudaFree(d_q_out);
    cudaFree(d_initial_norms);
    cudaFree(d_final_norms);
    cudaFree(d_config_success);
    cudaFree(d_pairs);
    cudaFree(d_motion_max_norms);
    cudaFree(d_motion_success);

    return pass ? 0 : 1;
}

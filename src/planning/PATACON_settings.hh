#pragma once

#include "G1ConstraintParameters.hh"
#include "IgrisCConstraintParameters.hh"
#include "FrankaConstraintParameters.hh"

constexpr int FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES = 256;
constexpr int FFW_SG2_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_SPHERES = 96;
constexpr int FFW_SG2_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_APPROX_SPHERES = 32;

struct FfwSg2AttachedObjectCollisionSpec {
    bool enabled = false;
    int sphere_count = 0;
    float world_offset[3] = {0.0f, 0.0f, 0.0f};
    float spheres[FFW_SG2_ATTACHED_OBJECT_MAX_SPHERES][4] = {};
    int ignored_robot_sphere_count = 0;
    int ignored_robot_spheres[
        FFW_SG2_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_SPHERES
    ] = {};
    int ignored_robot_approx_sphere_count = 0;
    int ignored_robot_approx_spheres[
        FFW_SG2_ATTACHED_OBJECT_MAX_IGNORED_ROBOT_APPROX_SPHERES
    ] = {};
};

struct PATACON_settings {
    int max_samples = 1000000;
    // Tree 하나가 만들 수 있는 최대 Tangent Space 개수
    int max_tangent_spaces = 10000;
    int max_iters = 1000000;
    int num_new_configs = 512;
    int granularity = 16;
    float range = 0.3;
    // Base seed for the per-block CURAND and shuffled Halton states.
    unsigned long long random_seed = 1ULL;
    
    // lift_joint 이동 거리 가중치
    float lift_distance_weight = 1.0f;

    bool ffw_sg2_enable_com_constraint = false;
    float ffw_sg2_support_margin_m = 0.0f;
    float ffw_sg2_object_mass_kg = 0.0f;
    FfwSg2AttachedObjectCollisionSpec ffw_sg2_attached_object_collision{};

    int balance = 1;
    float tree_ratio = 1.0;

    bool trace_trees = false;
    // Opt-in device counters for diagnosing TB-RRT expansion behavior.
    bool collect_diagnostics = false;

    // PATACON projection
    bool axis = false;

    int projection_max_iters = 60;
    float projection_alpha = 1.0f;
    float projection_damping = 1.0e-4f;
    float projection_task_tolerance = 1.0e-3f;

    float projection_smoothness_threshold = 0.03f;
    float projection_smoothness_weight = 1.0f;
    float beta = 8.0f;
    float gamma = 1.0f;
    bool projection_smoothness = true;

    float projection_max_step = 0.20f;
    
    // Tangent-Bundle / ConCon EXTEND
    float em_threshold = 0.1f;
    int max_concon_nodes = 4;
    // Section 3.5.1 of TB-RRT: keep random EXTEND samples in the
    // forward half-space of each non-root tangent space.
    bool prevent_ts_backtracking = false;

    // CONNECT 동안 허용할 최대 Tangent-Space / ConCon 반복 수
    int max_connect_concon_chunks = 16;

    // projection 후 target에 실제로 가까워졌다고 인정할 최소 거리 감소량
    float connect_progress_epsilon = 1.0e-6f;

    // opposite-tree target과 이 거리 이내이면 connected로 판단
    float connect_reached_tolerance = 1.0e-3f;

    ppln::constraints::G1ConstraintParameters g1_constraints{};
    ppln::constraints::IgrisCConstraintParameters igris_c_constraints{};
    ppln::constraints::FrankaConstraintParameters franka_constraints{};

    // single_mbm.cpp is compiled with Clang while its CUDA translation unit
    // is hosted by GCC. End the base class on its 8-byte alignment boundary
    // so Clang cannot reuse tail padding for AORRTC_settings' derived fields.
    unsigned long long abi_layout_guard = 0ULL;
};

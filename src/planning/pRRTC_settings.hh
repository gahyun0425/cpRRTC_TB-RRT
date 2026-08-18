#pragma once

#include "G1ConstraintParameters.hh"

struct pRRTC_settings {
    int max_samples = 1000000;
    // Tree 하나가 만들 수 있는 최대 Tangent Space 개수
    int max_tangent_spaces = 10000;
    int max_iters = 1000000;
    int num_new_configs = 600;
    int granularity = 16;
    float range = 0.5;
    
    // lift_joint 이동 거리 가중치
    float lift_distance_weight = 1.0f;

    int balance = 1;
    float tree_ratio = 1.0;

    bool dynamic_domain = true;
    bool trace_trees = false;

    // cpRRTC projection
    bool rigid_orientation = false;

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
    int max_concon_nodes = 16;

    // CONNECT 동안 허용할 최대 Tangent-Space / ConCon 반복 수
    int max_connect_concon_chunks = 16;

    float dd_alpha = 0.0001;
    float dd_radius = 4.0;
    float dd_min_radius = 1.0;

    // projection 후 target에 실제로 가까워졌다고 인정할 최소 거리 감소량
    float connect_progress_epsilon = 1.0e-6f;

    // opposite-tree target과 이 거리 이내이면 connected로 판단
    float connect_reached_tolerance = 1.0e-3f;

    ppln::constraints::G1ConstraintParameters g1_constraints{};
};

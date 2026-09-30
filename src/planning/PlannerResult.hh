#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

struct PlannerDiagnostics {
    std::array<int, 2> tangent_space_count = {0, 0};
    std::uint64_t extend_attempts = 0;
    std::uint64_t extend_backtracking_flips = 0;
    std::uint64_t extend_em_stops = 0;
    std::uint64_t extend_anchor_projection_stops = 0;
    std::uint64_t extend_edge_projection_stops = 0;
    std::uint64_t extend_collision_stops = 0;
    std::uint64_t extend_full_successes = 0;
    std::uint64_t connect_attempts = 0;
    std::uint64_t connect_chunks = 0;
    std::uint64_t connect_invalid_tangent_spaces = 0;
    std::uint64_t connect_tangent_direction_stops = 0;
    std::uint64_t connect_em_stops = 0;
    std::uint64_t connect_anchor_projection_stops = 0;
    std::uint64_t connect_edge_projection_stops = 0;
    std::uint64_t connect_progress_stops = 0;
    std::uint64_t connect_collision_stops = 0;
    std::uint64_t connect_successes = 0;
    std::uint64_t connect_failures = 0;
};

template <typename Robot>
struct PlannerResult {
    bool solved = false;
    std::vector<typename Robot::Configuration> path;
    int start_tree_size = 0;
    int goal_tree_size = 0;
    int path_length = 0;
    int iters = 0;
    float cost = 0.0;
    std::size_t wall_ns = 0;
    std::size_t kernel_ns = 0;
    std::size_t copy_ns = 0;
    PlannerDiagnostics diagnostics;
    std::array<std::vector<typename Robot::Configuration>, 2> tree_nodes;
    std::array<std::vector<int>, 2> tree_parents;
    std::array<std::vector<int>, 2> tree_node_ready;
    int connection_tree_id = -1;
    int connection_node_idx = -1;
    int connection_other_tree_id = -1;
    int connection_other_node_idx = -1;
    std::vector<std::array<int, 2>> solution_trace;
};

template <typename Robot>
struct PathSimplificationResult {
    std::vector<typename Robot::Configuration> path;
    std::size_t attempted_shortcuts = 0;
    std::size_t accepted_shortcuts = 0;
    float original_cost = 0.0f;
    float simplified_cost = 0.0f;
};

struct PathValidationResult {
    bool valid = false;
    std::size_t checked_edges = 0;
    std::size_t failed_edge = 0;
    float maximum_projection_delta = 0.0f;
};

// Internal robot projection, tangent-basis, and constrained-sampling implementation.
// Included by PATACON.cu inside namespace PATACON.

    // PATACON motion generation / projection wrapper
    // Generic robot: straight-line motion만 생성하고 projection은 하지 않는다.
    // FfwSg2: straight-line motion 생성 후 analytic-Jacobian ParallelProject 수행.

    template <typename Robot>
    __device__ __forceinline__ bool patacon_project_motion(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid    
    ) {
        static constexpr auto dim = Robot::dimension;

        static_assert(
            dim <= MAX_ROBOT_DIM,
            "Robot dimension exceeds PATACON motion segment buffer"
        );

        // q0 = 시작 configuration
        if (tid < dim) {
            motion_segment[tid] = q_start[tid];
        }

        // waypoint 하나당 CUDA thread 4개 사용
        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        if (waypoint <= d_settings.granularity) {
            for (int j = lane; j < dim; j += 4) {
                motion_segment[waypoint * dim + j] = q_start[j] + static_cast<float>(waypoint) * q_step[j];
            }
        }

        __syncthreads();

        // 일반 robot은 constraint projection 없음
        return true;
    }

    // ffw-sg2 specializatoin
    template <>
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::FfwSg2>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid    
    ) {
        static constexpr auto dim =ppln::robots::FfwSg2::dimension;

        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        // ξ[0] = q_start
        if (tid < dim) {
            motion_segment[tid] = q_start[tid];
        }

        // ξ[1] ... ξ[granularity] 생성
        if (waypoint <= d_settings.granularity) {
            for (int j = lane; j < dim; j += 4) {
                motion_segment[waypoint * dim + j] =q_start[j] +static_cast<float>(waypoint) * q_step[j];
            }
        }

        __syncthreads();

        // FFW SG2만 실제 analytic-Jacobian projection 수행
        return ppln::collision::ffw_sg2_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            d_settings.axis,

            projection_valid,
            projection_prog,
            projection_success,

            d_settings.projection_max_iters,
            d_settings.projection_alpha,
            d_settings.projection_damping,
            d_settings.projection_task_tolerance,

            d_settings.projection_smoothness_threshold,
            d_settings.projection_smoothness_weight,
            d_settings.projection_smoothness,

            d_settings.projection_max_step,

            tid
        );
    }

    template <>
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::FfwSg2Mobility>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        static constexpr auto dim = ppln::robots::FfwSg2Mobility::dimension;

        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        if (tid < dim) {
            motion_segment[tid] = q_start[tid];
        }

        if (waypoint <= d_settings.granularity) {
            for (int j = lane; j < dim; j += 4) {
                motion_segment[waypoint * dim + j] =
                    q_start[j] + static_cast<float>(waypoint) * q_step[j];
            }
        }

        __syncthreads();

        if (d_settings.ffw_sg2_enable_com_constraint) {
            return ppln::collision::ffw_sg2_mobility_com_project_motion(
                motion_segment,
                motion_segment_next,
                d_settings.granularity,
                d_settings.ffw_sg2_support_margin_m,
                d_settings.ffw_sg2_object_mass_kg,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                d_settings.projection_smoothness_threshold,
                d_settings.projection_smoothness_weight,
                d_settings.projection_smoothness,
                d_settings.projection_max_step,
                tid
            );
        }

        return ppln::collision::ffw_sg2_mobility_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            projection_valid,
            projection_prog,
            projection_success,
            d_settings.projection_max_iters,
            d_settings.projection_alpha,
            d_settings.projection_damping,
            d_settings.projection_task_tolerance,
            d_settings.projection_smoothness_threshold,
            d_settings.projection_smoothness_weight,
            d_settings.projection_smoothness,
            d_settings.projection_max_step,
            tid
        );
    }

    template <>
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::G1>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        static constexpr int dim = ppln::robots::G1::dimension;
        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        if (tid < dim) {
            motion_segment[tid] = q_start[tid];
        }
        if (waypoint <= d_settings.granularity) {
            for (int joint = lane; joint < dim; joint += 4) {
                motion_segment[waypoint * dim + joint] =
                    q_start[joint] + static_cast<float>(waypoint) * q_step[joint];
            }
        }
        __syncthreads();

        return ppln::collision::g1_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            d_settings.g1_constraints,
            d_settings.axis,
            projection_valid,
            projection_prog,
            projection_success,
            d_settings.projection_max_iters,
            d_settings.projection_alpha,
            d_settings.beta,
            d_settings.gamma,
            d_settings.projection_damping,
            d_settings.projection_task_tolerance,
            d_settings.projection_smoothness_threshold,
            d_settings.projection_smoothness_weight,
            d_settings.projection_smoothness,
            d_settings.projection_max_step,
            tid
        );
    }

    template <>
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::IgrisC>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        static constexpr int dim = ppln::robots::IgrisC::dimension;
        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        if (tid < dim) {
            motion_segment[tid] = q_start[tid];
        }
        if (waypoint <= d_settings.granularity) {
            for (int joint = lane; joint < dim; joint += 4) {
                motion_segment[waypoint * dim + joint] =
                    q_start[joint] + static_cast<float>(waypoint) * q_step[joint];
            }
        }
        __syncthreads();

        return ppln::collision::igris_c_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            d_settings.igris_c_constraints,
            projection_valid,
            projection_prog,
            projection_success,
            d_settings.projection_max_iters,
            d_settings.projection_alpha,
            d_settings.beta,
            d_settings.gamma,
            d_settings.projection_damping,
            d_settings.projection_task_tolerance,
            d_settings.projection_smoothness_threshold,
            d_settings.projection_smoothness_weight,
            d_settings.projection_smoothness,
            d_settings.projection_max_step,
            tid
        );
    }

    template <>
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::FrankaSingle>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        constexpr int dim = ppln::robots::FrankaSingle::dimension;
        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;
        if (tid < dim) motion_segment[tid] = q_start[tid];
        if (waypoint <= d_settings.granularity) {
            for (int joint = lane; joint < dim; joint += 4) {
                motion_segment[waypoint * dim + joint] =
                    q_start[joint] + static_cast<float>(waypoint) * q_step[joint];
            }
        }
        __syncthreads();
        return ppln::collision::franka_single_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            d_settings.franka_constraints,
            d_settings.axis,
            projection_valid,
            projection_prog,
            projection_success,
            d_settings.projection_max_iters,
            d_settings.projection_alpha,
            d_settings.projection_damping,
            d_settings.projection_task_tolerance,
            d_settings.projection_smoothness_threshold,
            d_settings.projection_smoothness_weight,
            d_settings.projection_smoothness,
            d_settings.projection_max_step,
            tid
        );
    }

    template <>
    __device__ __forceinline__ bool patacon_project_motion<ppln::robots::Franka>(
        volatile const float *q_start,
        volatile const float *q_step,
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        constexpr int dim = ppln::robots::Franka::dimension;
        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;
        if (tid < dim) motion_segment[tid] = q_start[tid];
        if (waypoint <= d_settings.granularity) {
            for (int joint = lane; joint < dim; joint += 4) {
                motion_segment[waypoint * dim + joint] =
                    q_start[joint] + static_cast<float>(waypoint) * q_step[joint];
            }
        }
        __syncthreads();
        return ppln::collision::franka_dual_project_motion(
            motion_segment,
            motion_segment_next,
            d_settings.granularity,
            d_settings.franka_constraints,
            d_settings.axis,
            projection_valid,
            projection_prog,
            projection_success,
            d_settings.projection_max_iters,
            d_settings.projection_alpha,
            d_settings.projection_damping,
            d_settings.projection_task_tolerance,
            d_settings.projection_smoothness_threshold,
            d_settings.projection_smoothness_weight,
            d_settings.projection_smoothness,
            d_settings.projection_max_step,
            tid
        );
    }

    template <typename Robot>
    __device__ __forceinline__ float patacon_config_distance_from_volatile(
        volatile const float *q_a,
        const float *q_b
    ) {
        float result = 0.0f;

        #pragma unroll
        for (int joint = 0; joint < Robot::dimension; joint++) {
            const float weight =
                patacon_joint_distance_weight<Robot>(joint);
            const float weighted_diff =
                weight * (q_a[joint] - q_b[joint]);
            result += weighted_diff * weighted_diff;
        }

        return sqrtf(result);
    }

    template <typename Robot>
    __device__ __noinline__ bool patacon_project_prebuilt_motion(
        volatile float *motion_segment,
        volatile float *motion_segment_next,
        int waypoint_count,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid,
        bool use_smoothness,
        float smoothness_threshold,
        bool return_when_success = true
    ) {
        if (waypoint_count <= 0) {
            if (tid == 0) {
                projection_prog[0] = 0;
                projection_success[0] = 1;
                projection_valid[0] = 1;
            }
            __syncthreads();
            return true;
        }

        if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
            return ppln::collision::ffw_sg2_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                d_settings.axis,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
            if (d_settings.ffw_sg2_enable_com_constraint) {
                return ppln::collision::ffw_sg2_mobility_com_project_motion(
                    motion_segment,
                    motion_segment_next,
                    waypoint_count,
                    d_settings.ffw_sg2_support_margin_m,
                    d_settings.ffw_sg2_object_mass_kg,
                    projection_valid,
                    projection_prog,
                    projection_success,
                    d_settings.projection_max_iters,
                    d_settings.projection_alpha,
                    d_settings.projection_damping,
                    d_settings.projection_task_tolerance,
                    smoothness_threshold,
                    d_settings.projection_smoothness_weight,
                    use_smoothness,
                    d_settings.projection_max_step,
                    tid,
                    return_when_success
                );
            }

            return ppln::collision::ffw_sg2_mobility_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else if constexpr (std::is_same_v<Robot, robots::G1>) {
            return ppln::collision::g1_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                d_settings.g1_constraints,
                d_settings.axis,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.beta,
                d_settings.gamma,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else if constexpr (std::is_same_v<Robot, robots::IgrisC>) {
            return ppln::collision::igris_c_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                d_settings.igris_c_constraints,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.beta,
                d_settings.gamma,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
            return ppln::collision::franka_single_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                d_settings.franka_constraints,
                d_settings.axis,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else if constexpr (std::is_same_v<Robot, robots::Franka>) {
            return ppln::collision::franka_dual_project_motion(
                motion_segment,
                motion_segment_next,
                waypoint_count,
                d_settings.franka_constraints,
                d_settings.axis,
                projection_valid,
                projection_prog,
                projection_success,
                d_settings.projection_max_iters,
                d_settings.projection_alpha,
                d_settings.projection_damping,
                d_settings.projection_task_tolerance,
                smoothness_threshold,
                d_settings.projection_smoothness_weight,
                use_smoothness,
                d_settings.projection_max_step,
                tid,
                return_when_success
            );
        } else {
            if (tid == 0) {
                projection_prog[0] = waypoint_count;
                projection_success[0] = 1;
                projection_valid[0] = 1;
            }
            __syncthreads();
            return true;
        }
    }

    template <typename Robot>
    __device__ __noinline__ bool
    patacon_project_concon_node_anchors(
        volatile const float *q_start,
        const float *node_nominal_targets,
        int edge_count,
        volatile float *node_motion,
        volatile float *node_motion_next,
        volatile unsigned char *projection_valid,
        volatile int *projection_prog,
        volatile unsigned int *projection_success,
        int tid
    ) {
        static constexpr auto dim = Robot::dimension;

        static_assert(
            dim <= MAX_ROBOT_DIM,
            "Robot dimension exceeds PATACON node motion buffer"
        );

        if (edge_count <= 0) {
            if (tid == 0) {
                projection_prog[0] = 0;
                projection_success[0] = 0;
                projection_valid[0] = 1;
            }
            __syncthreads();
            return false;
        }

        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;

        if (tid < dim) {
            node_motion[tid] = q_start[tid];
        }

        if (waypoint <= edge_count) {
            for (int joint = lane; joint < dim; joint += 4) {
                node_motion[waypoint * dim + joint] =
                    node_nominal_targets[
                        (waypoint - 1) * MAX_ROBOT_DIM + joint
                    ];
            }
        }
        __syncthreads();

        const float node_smoothness_threshold =
            static_cast<float>(d_settings.granularity) *
            d_settings.projection_smoothness_threshold;
        return patacon_project_prebuilt_motion<Robot>(
            node_motion,
            node_motion_next,
            edge_count,
            projection_valid,
            projection_prog,
            projection_success,
            tid,
            d_settings.projection_smoothness,
            node_smoothness_threshold
        );
    }

    template <typename Robot>
    __device__ __noinline__ void
    patacon_project_concon_edge_segments_from_node_anchors(
        int edge_count,
        volatile const float *node_anchors,
        volatile float *edge_motion_segments,
        volatile float *edge_motion_segment_next,
        volatile unsigned char *edge_projection_valid,
        volatile int *edge_projection_prog,
        volatile unsigned int *edge_projection_success,
        volatile int *first_projection_failure_edge,
        int tid
    ) {
        static constexpr auto dim = Robot::dimension;
        const int edge_slot = tid / CONCON_COLLISION_THREADS_PER_EDGE;
        const int edge_tid =
            tid - edge_slot * CONCON_COLLISION_THREADS_PER_EDGE;

        if (edge_count <= 0) {
            if (tid == 0) {
                first_projection_failure_edge[0] = 0;
            }
            __syncthreads();
            return;
        }

        const int waypoint = tid / 4 + 1;
        const int lane = tid % 4;
        const int total_waypoint_count =
            edge_count * d_settings.granularity;

        // Keep the ConCon prefix continuous while including each projected
        // node (waypoints granularity, 2 * granularity, ...) in smoothing.
        volatile float *prefix_motion = edge_motion_segment_next;
        volatile float *prefix_motion_next = edge_motion_segments;

        if (tid < dim) {
            prefix_motion[tid] = node_anchors[tid];
        }
        if (waypoint <= total_waypoint_count) {
            const int prefix_edge_slot =
                (waypoint - 1) / d_settings.granularity;
            const int local_waypoint =
                (waypoint - 1) % d_settings.granularity + 1;
            const float alpha =
                static_cast<float>(local_waypoint) /
                static_cast<float>(d_settings.granularity);

            for (int joint = lane; joint < dim; joint += 4) {
                const float q0 =
                    node_anchors[prefix_edge_slot * dim + joint];
                const float q1 =
                    node_anchors[(prefix_edge_slot + 1) * dim + joint];
                prefix_motion[waypoint * dim + joint] =
                    q0 + alpha * (q1 - q0);
            }
        }
        __syncthreads();

        const bool projection_good = patacon_project_prebuilt_motion<Robot>(
            prefix_motion,
            prefix_motion_next,
            total_waypoint_count,
            edge_projection_valid,
            edge_projection_prog,
            edge_projection_success,
            tid,
            d_settings.projection_smoothness,
            d_settings.projection_smoothness_threshold,
            false
        );
        __syncthreads();

        if (tid == 0) {
            int completed_waypoints = projection_good
                ? total_waypoint_count
                : edge_projection_prog[0];
            if (completed_waypoints < 0) {
                completed_waypoints = 0;
            }
            if (completed_waypoints > total_waypoint_count) {
                completed_waypoints = total_waypoint_count;
            }
            first_projection_failure_edge[0] =
                completed_waypoints / d_settings.granularity;
        }
        __syncthreads();

        // Collision checking keeps one source duplicate per edge. Repack the
        // continuous projected prefix without changing the thread layout.
        const int source_edge_slot =
            edge_slot < edge_count ? edge_slot : edge_count - 1;
        volatile float *edge_motion =
            &edge_motion_segments[edge_slot * CONCON_MOTION_SEGMENT_STRIDE];
        const int edge_segment_values =
            (d_settings.granularity + 1) * dim;
        for (
            int value = edge_tid;
            value < edge_segment_values;
            value += CONCON_COLLISION_THREADS_PER_EDGE
        ) {
            const int local_waypoint = value / dim;
            const int joint = value - local_waypoint * dim;
            const int prefix_waypoint =
                source_edge_slot * d_settings.granularity + local_waypoint;
            edge_motion[local_waypoint * dim + joint] =
                prefix_motion[prefix_waypoint * dim + joint];
        }
        __syncthreads();

        // A projection backend may report success even when an already-valid
        // waypoint has drifted just outside a joint boundary.  Reject the
        // first affected edge before collision checking or tree insertion.
        // Checking the repacked segment also covers every interpolated
        // waypoint, rather than only the endpoint stored in the tree.
        if (edge_slot < edge_count) {
            for (
                int value = edge_tid;
                value < edge_segment_values;
                value += CONCON_COLLISION_THREADS_PER_EDGE
            ) {
                const int joint = value % dim;
                if (!planning::joint_value_within_limits<Robot>(
                        edge_motion[value],
                        joint
                    )) {
                    atomicMin(
                        (int *)&first_projection_failure_edge[0],
                        edge_slot
                    );
                }
            }
        }
        __syncthreads();
    }

    template <typename Robot>
    __device__ __forceinline__ bool patacon_store_tangent_basis(
        const float *q,
        float *tree_tangent_bases,
        int node_idx
    ) {
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            if (tree_tangent_bases == nullptr) {
                return false;
            }

            constexpr int basis_size = TangentSpaceTraits<Robot>::basis_size;
            float basis[basis_size];

            // 현재 node q에서 Jacobian을 새로 계산하고 tangent basis까지 생성
            bool basis_ok = false;
            if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
                basis_ok = ppln::collision::ffw_sg2_tangent_basis(
                    q,
                    d_settings.axis,
                    basis
                );
            } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
                basis_ok = ppln::collision::ffw_sg2_mobility_tangent_basis(
                    q,
                    basis
                );
            } else if constexpr (std::is_same_v<Robot, robots::G1>) {
                basis_ok = ppln::collision::g1_tangent_basis(
                    q,
                    d_settings.g1_constraints,
                    d_settings.axis,
                    basis
                );
            } else if constexpr (std::is_same_v<Robot, robots::IgrisC>) {
                basis_ok = ppln::collision::igris_c_tangent_basis(
                    q,
                    d_settings.igris_c_constraints,
                    basis
                );
            } else if constexpr (std::is_same_v<Robot, robots::FrankaSingle>) {
                basis_ok = ppln::collision::franka_single_tangent_basis(
                    q,
                    d_settings.franka_constraints,
                    d_settings.axis,
                    basis
                );
            } else if constexpr (std::is_same_v<Robot, robots::Franka>) {
                basis_ok = ppln::collision::franka_dual_tangent_basis(
                    q,
                    d_settings.franka_constraints,
                    d_settings.axis,
                    basis
                );
            }
            if (!basis_ok) {
                return false;
            }

            float *dst = &tree_tangent_bases[node_idx * basis_size];

            for (int i = 0; i < basis_size; i++) {
                dst[i] = basis[i];
            }
        }

        return true;
    }

    template <typename Robot>
    __device__ __forceinline__ void patacon_sample_tangent_config(
        float *tree_nodes,
        float *ts_bases,
        const int *ts_root_node_indices,
        const int *ts_parent_ids,
        int ts_root_node_idx,
        int selected_ts_id,
        float *ts_coeff,
        float alpha_fraction,
        float *ts_tangent_dir,
        float *sdata,
        float *sampled_config,
        int tid
    )
    {
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            static constexpr auto dim = Robot::dimension;
            static constexpr int basis_stride = TangentSpaceTraits<Robot>::max_tangent_dim;
            static constexpr int basis_size = TangentSpaceTraits<Robot>::basis_size;
            const int active_tangent_dim = patacon_active_tangent_dim<Robot>();

            // 선택된 Tangent Space의 root configuration
            const float *base_q =&tree_nodes[ts_root_node_idx * dim];

            // 선택된 Tangent Space의 tangent basis
            const float *basis = &ts_bases[selected_ts_id * basis_size];
            float alpha_limit = FLT_MAX;

            if (tid < dim) {
                // tangent basis들의 linear combination dir = B * c. B는 선택된 TS의 tangent basis 행렬. c는 각 basis vector를 얼마나 섞을지 나타내는 계수 벡터
                float dir = 0.0f;

                for (int k = 0; k < active_tangent_dim; k++) {
                    dir += basis[tid * basis_stride + k] * ts_coeff[k];
                }

                ts_tangent_dir[tid] = dir;
            }

            __syncthreads();

            // TB-RRT Section 3.5.1: a non-root TS samples only in the
            // half-space pointing away from its parent TS.  Since the
            // sample direction already lies in the new tangent space,
            // dot(dir, root - parent_root) has the same sign as the dot
            // product with the explicitly projected forward direction.
            if (tid == 0) {
                bool flip_direction = false;

                if (d_settings.prevent_ts_backtracking) {
                    const int parent_ts_id = ts_parent_ids[selected_ts_id];

                    if (parent_ts_id >= 0) {
                        const int parent_root_idx =
                            ts_root_node_indices[parent_ts_id];
                        const float *parent_q =
                            &tree_nodes[parent_root_idx * dim];
                        float direction_dot = 0.0f;

                        for (int joint = 0; joint < dim; joint++) {
                            direction_dot +=
                                ts_tangent_dir[joint] *
                                (base_q[joint] - parent_q[joint]);
                        }
                        flip_direction = direction_dot < 0.0f;
                    }
                }

                sdata[0] = flip_direction ? 1.0f : 0.0f;
                if (flip_direction) {
                    diagnostic_increment(DIAG_EXTEND_BACKTRACKING_FLIPS);
                }
            }

            __syncthreads();

            if (tid < dim) {
                float dir = ts_tangent_dir[tid];

                if (sdata[0] != 0.0f) {
                    dir = -dir;
                    ts_tangent_dir[tid] = dir;
                }

                // joint limit 안에서 최대 이동 가능 거리 계산
                const float lo = Robot::get_s_a(tid); // 해당 차원의 최솟값
                const float hi = lo + Robot::get_s_m(tid); // 해당 차원의 max값

                if (dir > 1.0e-8f) {
                    alpha_limit =(hi - base_q[tid])/ dir;
                }
                else if (dir < -1.0e-8f) {
                    alpha_limit =(lo - base_q[tid])/ dir;
                }

                alpha_limit =fmaxf(alpha_limit,0.0f);
            }

            // 각 joint가 허용하는 alpha
            sdata[tid] =tid < dim? alpha_limit: FLT_MAX;

            __syncthreads();

            // 모든 joint 중 가장 작은 alpha_limit 찾기
            for (unsigned int s =blockDim.x / 2; s > 0; s >>= 1) {
                const float lhs =sdata[tid];
                float rhs =FLT_MAX;

                if (tid < s) {
                    rhs =sdata[tid + s];
                }

                __syncthreads();

                if (tid < s) {
                    sdata[tid] =fminf(lhs,rhs);
                }

                __syncthreads();
            }

            // 실제 이동거리
            const float alpha =alpha_fraction *sdata[0];

            // 최종 q_rand
            // q_rand = q_TS_root + alpha * tangent_direction
            if (tid < dim) {

                sampled_config[tid] =base_q[tid]+alpha *ts_tangent_dir[tid];
            }

            __syncthreads();
        }
    }

    template <typename Robot>
    __device__ __forceinline__ float patacon_constraint_error_norm(const float *q)
    {
        if constexpr (std::is_same_v<Robot, robots::FfwSg2>) {
            float h[FFW_SG2_MAX_RESIDUAL_DIM];

            // 현재 configuration q의 constraint residual h(q) 계산
            ppln::collision::ffw_sg2_constraint_residual(q,d_settings.axis,h);

            // 현재 constraint의 residual dimension
            const int residual_dim =ppln::collision::ffw_sg2_constraint_dim(d_settings.axis);

            // EM = ||h(q)||
            return ppln::collision::ffw_sg2_residual_norm(h,residual_dim);
        } else if constexpr (std::is_same_v<Robot, robots::FfwSg2Mobility>) {
            float h[FFW_SG2_MOBILITY_RESIDUAL_DIM];
            ppln::collision::ffw_sg2_mobility_constraint_residual(q, h);
            return ppln::collision::ffw_sg2_mobility_residual_norm(h);
        } else if constexpr (std::is_same_v<Robot, robots::G1>) {
            return ppln::collision::g1_equality_residual_norm(
                q,
                d_settings.g1_constraints,
                d_settings.axis
            );
        } else if constexpr (std::is_same_v<Robot, robots::IgrisC>) {
            return ppln::collision::igris_c_equality_residual_norm(
                q,
                d_settings.igris_c_constraints
            );
        } else if constexpr (
            std::is_same_v<Robot, robots::FrankaSingle> ||
            std::is_same_v<Robot, robots::Franka>
        ) {
            return ppln::collision::franka_constraint_error_norm<Robot>(
                q,
                d_settings.franka_constraints,
                d_settings.axis
            );
        }

        return 0.0f;
    }


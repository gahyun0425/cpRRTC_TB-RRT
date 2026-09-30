// Internal tangent-space initialization and collision-validation implementation.
// Included by PATACON.cu inside namespace PATACON.

    template <typename Robot, int LaneStride = MAX_THREADS_PER_BLOCK>
    __global__ void init_root_ts_banks(
        float **nodes,
        int **ts_root_node_idx,
        int **ts_parent_id,
        float **ts_bases,
        int **ts_ready,
        int **node_ts_id,
        float **node_ts_q,
        int **ts_node_count,
        int **ts_lane_head,
        int **node_next_in_ts,
        int start_count,
        int goal_count
    )
    {
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            const int global_idx = blockIdx.x;

            // 지금 처리하고 있는 것이 start tree인지 goal tree인지 결정
            const int tree =(global_idx < start_count) ? 0 : 1;

            // 해당 tree 안에서의 TS index
            const int ts_idx =(tree == 0)? global_idx: global_idx - start_count;

            // 해당 tree의 initial node 개수
            const int count =(tree == 0)? start_count: goal_count;

            if (threadIdx.x == 0 &&ts_idx < count) {
                // 처음에는 initial node 하나가 TS 하나의 root
                const int node_idx = ts_idx;

                // 이 TS가 어느 tree node에서 만들어졌는지 저장
                ts_root_node_idx[tree][ts_idx] =node_idx;
                // start/goal root TS는 부모 TS가 없다.
                ts_parent_id[tree][ts_idx] = -1;

                // root q에서 tangent basis 계산
                const bool basis_ok =patacon_store_tangent_basis<Robot>(&nodes[tree][node_idx * Robot::dimension],ts_bases[tree],ts_idx);

                if (basis_ok) {

                    // 이 node는 방금 만든 TS에 소속
                    node_ts_id[tree][node_idx] =ts_idx;

                    // root에서는 nominal TS q == 실제 tree q
                    for (int j = 0; j < Robot::dimension; j++) {
                        node_ts_q[tree][node_idx * Robot::dimension + j] =nodes[tree][node_idx * Robot::dimension + j];
                    }

                    // This slot may contain a linked list from an older
                    // request.  Initialize only the TS slot being published;
                    // request startup never has to clear the whole bank.
                    ts_node_count[tree][ts_idx] = 0;
                    for (int lane = 0; lane < MAX_THREADS_PER_BLOCK; ++lane) {
                        ts_lane_head[tree][
                            ts_idx * MAX_THREADS_PER_BLOCK + lane
                        ] = -1;
                    }

                    // 최초 root node를 이 TS의 thread 0 목록에 등록
                    ts_node_count[tree][ts_idx] = 1;
                    ts_lane_head[tree][ts_idx * LaneStride] =node_idx;
                    node_next_in_ts[tree][node_idx] = -1;
                }

                // 위 정보들이 global memory에 기록된 후 ready를 켜기 위해 사용
                __threadfence();

                if (basis_ok) {
                    ts_ready[tree][ts_idx] = current_search_generation;
                }
            }
        }
    }

    template <typename Robot>
    __device__ __forceinline__ float patacon_shared_config_distance(
        volatile const float *q_a,
        volatile const float *q_b,
        float *sdata,
        int tid
    )
    {
        static constexpr auto dim = Robot::dimension;
        float value = 0.0f;

        // 각 thread가 joint dimension 하나의 거리 제곱을 계산
        if (tid < dim) {
            const float weight =
                patacon_joint_distance_weight<Robot>(tid);

            const float weighted_diff =
                weight * (q_a[tid] - q_b[tid]);

            value = weighted_diff * weighted_diff;
        }

        // dim보다 큰 thread는 0
        sdata[tid] = value;

        __syncthreads();

        // block reduction
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < static_cast<int>(s)) {
                sdata[tid] += sdata[tid + s];
            }

            __syncthreads();
        }

        return sqrtf(sdata[0]);
    }

    template <typename Robot>
    __device__ __forceinline__ bool patacon_detailed_env_collision_check(
        volatile float *sphere_pos,
        volatile int *link_CC,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        // 다른 robot은 기존 방식
        return ppln::collision::env_collision_check<Robot>(
            sphere_pos,
            link_CC,
            env,
            tid
        );
    }


    template <>
    __device__ __forceinline__ bool patacon_detailed_env_collision_check<ppln::robots::FfwSg2>(
        volatile float *sphere_pos,
        volatile int *link_CC,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_env_collision_check_early(
                sphere_pos,
                link_CC,
                env,
                tid,
                motion_cc_flag
            );
    }

    template <>
    __device__ __forceinline__ bool patacon_detailed_env_collision_check<ppln::robots::FfwSg2Mobility>(
        volatile float *sphere_pos,
        volatile int *link_CC,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_mobility_env_collision_check_early(
                sphere_pos,
                link_CC,
                env,
                tid,
                motion_cc_flag
            );
    }

    template <typename Robot>
    __device__ __forceinline__ bool patacon_detailed_self_collision_check(
        volatile float *sphere_pos,
        volatile int *link_CC,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::self_collision_check<Robot>(
                sphere_pos,
                link_CC,
                tid
            );
    }


    template <>
    __device__ __forceinline__ bool patacon_detailed_self_collision_check<ppln::robots::FfwSg2>(
        volatile float *sphere_pos,
        volatile int *link_CC,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_self_collision_check_early(
                sphere_pos,
                link_CC,
                tid,
                motion_cc_flag
            );
    }

    template <>
    __device__ __forceinline__ bool patacon_detailed_self_collision_check<ppln::robots::FfwSg2Mobility>(
        volatile float *sphere_pos,
        volatile int *link_CC,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_mobility_self_collision_check_early(
                sphere_pos,
                link_CC,
                tid,
                motion_cc_flag
            );
    }

    template <typename Robot>
    __device__ __forceinline__ bool patacon_attached_object_collision_check_approx(
        const float *q,
        volatile float *sphere_pos_approx,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        if constexpr (
            std::is_same_v<Robot, ppln::robots::FrankaSingle> ||
            std::is_same_v<Robot, ppln::robots::Franka>
        ) {
            return ppln::collision::franka_attached_object_env_collision_check<Robot>(
                q, env, tid
            );
        }
        if constexpr (std::is_same_v<Robot, ppln::robots::FfwSg2>) {
            return ppln::collision::ffw_sg2_attached_object_collision_check_approx(
                q,
                sphere_pos_approx,
                env,
                tid,
                motion_cc_flag
            );
        }
        if constexpr (std::is_same_v<Robot, ppln::robots::G1>) {
            return ppln::collision::g1_attached_object_collision_check_approx(
                q, env, tid
            );
        }
        return true;
    }

    template <>
    __device__ __forceinline__ bool
    patacon_attached_object_collision_check_approx<ppln::robots::FfwSg2Mobility>(
        const float *q,
        volatile float *sphere_pos_approx,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_mobility_attached_object_collision_check_approx(
                q,
                sphere_pos_approx,
                env,
                tid,
                motion_cc_flag
            );
    }

    template <typename Robot>
    __device__ __forceinline__ bool patacon_attached_object_collision_check(
        const float *q,
        volatile float *sphere_pos,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        if constexpr (
            std::is_same_v<Robot, ppln::robots::FrankaSingle> ||
            std::is_same_v<Robot, ppln::robots::Franka>
        ) {
            return ppln::collision::franka_attached_object_env_collision_check<Robot>(
                q, env, tid
            );
        }
        if constexpr (std::is_same_v<Robot, ppln::robots::FfwSg2>) {
            return ppln::collision::ffw_sg2_attached_object_collision_check(
                q,
                sphere_pos,
                env,
                tid,
                motion_cc_flag
            );
        }
        if constexpr (std::is_same_v<Robot, ppln::robots::G1>) {
            return ppln::collision::g1_attached_object_collision_check(
                q, sphere_pos, env, tid
            );
        }
        return true;
    }

    template <>
    __device__ __forceinline__ bool
    patacon_attached_object_collision_check<ppln::robots::FfwSg2Mobility>(
        const float *q,
        volatile float *sphere_pos,
        ppln::collision::Environment<float> *env,
        int tid,
        volatile unsigned int *motion_cc_flag
    ) {
        return
            ppln::collision::ffw_sg2_mobility_attached_object_collision_check(
                q,
                sphere_pos,
                env,
                tid,
                motion_cc_flag
            );
    }

    template <typename Robot>
    __device__ __forceinline__ void
    patacon_check_projected_edges_collision_parallel(
        int edge_count,
        volatile float *edge_motion_segments,
        volatile float *sphere_pos_scratch,
        volatile float *sphere_pos_approx_scratch,
        volatile int *link_cc_scratch,
        float *transform_scratch,
        ppln::collision::Environment<float> *env,
        volatile unsigned int *edge_cc_result,
        bool *edge_run_detailed_env_check,
        bool *edge_run_self_collision_check,
        bool *edge_run_detailed_self_check,
        bool *any_detailed_env_check,
        bool *any_detailed_self_check,
        volatile int *first_collision_edge,
        int tid
    ) {
        using Collision = robots::CollisionTraits<Robot>;
        static constexpr int dim = Robot::dimension;
        static constexpr int fine_scratch_stride =
            Collision::fine_sphere_count * Collision::batch_size * 3;
        static constexpr int approx_scratch_stride =
            Collision::approximate_sphere_count * Collision::batch_size * 3;
        static constexpr int link_scratch_stride =
            Collision::joint_flag_stride * Collision::batch_size;
        static constexpr int transform_scratch_stride =
            Collision::batch_size * Collision::transform_slots * 16;

        if (edge_count <= 0) {
            return;
        }

        const int edge_slot = tid / CONCON_COLLISION_THREADS_PER_EDGE;
        const int edge_tid = tid - edge_slot * CONCON_COLLISION_THREADS_PER_EDGE;
        const int safe_edge_slot =
            edge_slot < edge_count ? edge_slot : edge_count - 1;
        const int waypoint = edge_tid / 4 + 1;

        volatile float *edge_motion =
            &edge_motion_segments[
                safe_edge_slot * CONCON_MOTION_SEGMENT_STRIDE
            ];
        volatile float *edge_sphere_pos =
            &sphere_pos_scratch[edge_slot * fine_scratch_stride];
        volatile float *edge_sphere_pos_approx =
            &sphere_pos_approx_scratch[edge_slot * approx_scratch_stride];
        volatile int *edge_link_cc =
            &link_cc_scratch[edge_slot * link_scratch_stride];
        float *edge_transform =
            &transform_scratch[edge_slot * transform_scratch_stride];

        float interp_cfg[dim];
        #pragma unroll
        for (int joint = 0; joint < dim; joint++) {
            interp_cfg[joint] = edge_motion[waypoint * dim + joint];
        }

        if (tid < MAX_PARALLEL_CONCON_EDGES) {
            edge_cc_result[tid] = 0u;
            edge_run_detailed_env_check[tid] = false;
            edge_run_self_collision_check[tid] = false;
            edge_run_detailed_self_check[tid] = false;
        }
        if (tid == 0) {
            first_collision_edge[0] = edge_count;
            any_detailed_env_check[0] = false;
            any_detailed_self_check[0] = false;
        }
        __syncthreads();

        for (int r = edge_tid; r < link_scratch_stride;
             r += CONCON_COLLISION_THREADS_PER_EDGE) {
            edge_link_cc[r] = 0;
        }
        __syncthreads();

        ppln::collision::fk_approx<Robot>(
            interp_cfg,
            edge_sphere_pos_approx,
            edge_transform,
            edge_tid
        );
        __syncthreads();

        if (edge_slot < edge_count && edge_slot < first_collision_edge[0]) {
            const bool env_collision_approx =
                not ppln::collision::env_collision_check_approx<Robot>(
                    edge_sphere_pos_approx,
                    edge_link_cc,
                    env,
                    edge_tid
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                env_collision_approx ? 1u : 0u
            );

            const bool attached_object_collision_approx =
                not patacon_attached_object_collision_check_approx<Robot>(
                    interp_cfg,
                    edge_sphere_pos_approx,
                    env,
                    edge_tid,
                    &edge_cc_result[edge_slot]
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                attached_object_collision_approx ? 1u : 0u
            );
        }
        __syncthreads();

        if (tid < edge_count) {
            edge_run_detailed_env_check[tid] = edge_cc_result[tid] != 0u;
            if (edge_run_detailed_env_check[tid]) {
                edge_cc_result[tid] = 0u;
            }
        }
        __syncthreads();

        if (tid == 0) {
            bool run_any = false;
            for (int edge = 0; edge < edge_count; edge++) {
                run_any = run_any || edge_run_detailed_env_check[edge];
            }
            any_detailed_env_check[0] = run_any;
        }
        __syncthreads();

        if (any_detailed_env_check[0]) {
            ppln::collision::fk<Robot>(
                interp_cfg,
                edge_sphere_pos,
                edge_transform,
                edge_tid
            );
        }
        __syncthreads();

        if (
            edge_slot < edge_count &&
            edge_slot < first_collision_edge[0] &&
            edge_run_detailed_env_check[edge_slot]
        ) {
            const bool env_collision =
                not patacon_detailed_env_collision_check<Robot>(
                    edge_sphere_pos,
                    edge_link_cc,
                    env,
                    edge_tid,
                    &edge_cc_result[edge_slot]
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                env_collision ? 1u : 0u
            );

            const bool attached_object_collision =
                not patacon_attached_object_collision_check<Robot>(
                    interp_cfg,
                    edge_sphere_pos,
                    env,
                    edge_tid,
                    &edge_cc_result[edge_slot]
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                attached_object_collision ? 1u : 0u
            );
        }
        __syncthreads();

        if (
            tid < edge_count &&
            edge_run_detailed_env_check[tid] &&
            edge_cc_result[tid] != 0u
        ) {
            atomicMin((int *)&first_collision_edge[0], tid);
        }
        __syncthreads();

        for (int r = edge_tid; r < link_scratch_stride;
             r += CONCON_COLLISION_THREADS_PER_EDGE) {
            edge_link_cc[r] = 0;
        }
        __syncthreads();

        if (tid < edge_count) {
            edge_run_self_collision_check[tid] =
                edge_cc_result[tid] == 0u && tid < first_collision_edge[0];
        }
        __syncthreads();

        if (
            edge_slot < edge_count &&
            edge_slot < first_collision_edge[0] &&
            edge_run_self_collision_check[edge_slot]
        ) {
            const bool self_collision_approx =
                not ppln::collision::self_collision_check_approx<Robot>(
                    edge_sphere_pos_approx,
                    edge_link_cc,
                    edge_tid
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                self_collision_approx ? 1u : 0u
            );
        }
        __syncthreads();

        if (tid < edge_count) {
            edge_run_detailed_self_check[tid] =
                edge_run_self_collision_check[tid] &&
                edge_cc_result[tid] != 0u;
            if (edge_run_detailed_self_check[tid]) {
                edge_cc_result[tid] = 0u;
            }
        }
        __syncthreads();

        if (tid == 0) {
            bool run_any = false;
            for (int edge = 0; edge < edge_count; edge++) {
                run_any = run_any || edge_run_detailed_self_check[edge];
            }
            any_detailed_self_check[0] = run_any;
        }
        __syncthreads();

        if (any_detailed_self_check[0]) {
            ppln::collision::fk<Robot>(
                interp_cfg,
                edge_sphere_pos,
                edge_transform,
                edge_tid
            );
        }
        __syncthreads();

        if (
            edge_slot < edge_count &&
            edge_slot < first_collision_edge[0] &&
            edge_run_detailed_self_check[edge_slot]
        ) {
            const bool self_collision =
                not patacon_detailed_self_collision_check<Robot>(
                    edge_sphere_pos,
                    edge_link_cc,
                    edge_tid,
                    &edge_cc_result[edge_slot]
                );
            atomicOr(
                (unsigned int *)&edge_cc_result[edge_slot],
                self_collision ? 1u : 0u
            );
        }
        __syncthreads();

        if (
            tid < edge_count &&
            edge_run_detailed_self_check[tid] &&
            edge_cc_result[tid] != 0u
        ) {
            atomicMin((int *)&first_collision_edge[0], tid);
        }
        __syncthreads();
    }

    template <typename Robot>
    __global__ void patacon_validate_visualization_shortcut_edge(
        const float *node_anchors,
        volatile float *edge_motion_segments,
        volatile float *edge_motion_segment_next,
        volatile float *sphere_pos_scratch,
        volatile float *sphere_pos_approx_scratch,
        volatile int *link_cc_scratch,
        float *transform_scratch,
        ppln::collision::Environment<float> *env,
        int *edge_is_valid,
        float *edge_maximum_projection_delta
    ) {
        using Collision = robots::CollisionTraits<Robot>;
        static constexpr int fine_scratch_stride =
            Collision::fine_sphere_count * Collision::batch_size * 3;
        static constexpr int approx_scratch_stride =
            Collision::approximate_sphere_count * Collision::batch_size * 3;
        static constexpr int link_scratch_stride =
            Collision::joint_flag_stride * Collision::batch_size;
        static constexpr int transform_scratch_stride =
            Collision::batch_size * Collision::transform_slots * 16;
        const std::size_t batch_edge = blockIdx.x;
        node_anchors += batch_edge * 2 * Robot::dimension;
        edge_motion_segments +=
            batch_edge * CONCON_MOTION_SEGMENT_STRIDE;
        edge_motion_segment_next +=
            batch_edge * CONCON_MOTION_SEGMENT_STRIDE;
        sphere_pos_scratch += batch_edge * fine_scratch_stride;
        sphere_pos_approx_scratch += batch_edge * approx_scratch_stride;
        link_cc_scratch += batch_edge * link_scratch_stride;
        transform_scratch += batch_edge * transform_scratch_stride;
        edge_is_valid += batch_edge;
        if (edge_maximum_projection_delta != nullptr) {
            edge_maximum_projection_delta += batch_edge;
        }

        const int tid = threadIdx.x;
        __shared__ volatile unsigned char projection_valid[
            CONCON_PROJECTION_STATE_STRIDE
        ];
        __shared__ volatile int projection_progress[1];
        __shared__ volatile unsigned int projection_success[1];
        __shared__ volatile int first_projection_failure_edge[1];
        __shared__ volatile unsigned int edge_cc_result[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_detailed_env_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_self_collision_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_detailed_self_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool any_detailed_env_check;
        __shared__ bool any_detailed_self_check;
        __shared__ volatile int first_collision_edge[1];
        __shared__ int joint_limits_good;

        if (tid == 0) {
            edge_is_valid[0] = 0;
            if (edge_maximum_projection_delta != nullptr) {
                edge_maximum_projection_delta[0] = 0.0f;
            }
            joint_limits_good = 1;
            first_collision_edge[0] = 0;
        }
        __syncthreads();

        patacon_project_concon_edge_segments_from_node_anchors<Robot>(
            1,
            node_anchors,
            edge_motion_segments,
            edge_motion_segment_next,
            projection_valid,
            projection_progress,
            projection_success,
            first_projection_failure_edge,
            tid
        );
        __syncthreads();

        if (
            edge_maximum_projection_delta != nullptr &&
            first_projection_failure_edge[0] >= 1 &&
            tid <= d_settings.granularity
        ) {
            const float ratio = static_cast<float>(tid) /
                static_cast<float>(d_settings.granularity);
            float thread_maximum = 0.0f;
            for (int joint = 0; joint < Robot::dimension; ++joint) {
                const float nominal = node_anchors[joint] + ratio *
                    (node_anchors[Robot::dimension + joint] -
                     node_anchors[joint]);
                thread_maximum = fmaxf(
                    thread_maximum,
                    fabsf(
                        edge_motion_segments[
                            tid * Robot::dimension + joint
                        ] - nominal
                    )
                );
            }
            atomicMax(
                reinterpret_cast<unsigned int *>(
                    edge_maximum_projection_delta
                ),
                __float_as_uint(thread_maximum)
            );
        }
        __syncthreads();

        if (
            first_projection_failure_edge[0] >= 1 &&
            tid <= d_settings.granularity &&
            !planning::configuration_within_joint_limits<Robot>(
                &edge_motion_segments[tid * Robot::dimension]
            )
        ) {
            atomicExch(&joint_limits_good, 0);
        }
        __syncthreads();

        if (
            first_projection_failure_edge[0] >= 1 &&
            joint_limits_good != 0
        ) {
            patacon_check_projected_edges_collision_parallel<Robot>(
                1,
                edge_motion_segments,
                sphere_pos_scratch,
                sphere_pos_approx_scratch,
                link_cc_scratch,
                transform_scratch,
                env,
                edge_cc_result,
                run_detailed_env_check,
                run_self_collision_check,
                run_detailed_self_check,
                &any_detailed_env_check,
                &any_detailed_self_check,
                first_collision_edge,
                tid
            );
        }
        __syncthreads();

        if (tid == 0) {
            edge_is_valid[0] =
                first_projection_failure_edge[0] >= 1 &&
                joint_limits_good != 0 &&
                first_collision_edge[0] >= 1;
        }
    }

    template <typename Robot>
    __global__ void patacon_validate_visualization_nominal_edge(
        const float *node_anchors,
        volatile float *edge_motion_segments,
        volatile float *sphere_pos_scratch,
        volatile float *sphere_pos_approx_scratch,
        volatile int *link_cc_scratch,
        float *transform_scratch,
        ppln::collision::Environment<float> *env,
        int *edge_is_valid
    ) {
        using Collision = robots::CollisionTraits<Robot>;
        static constexpr int fine_scratch_stride =
            Collision::fine_sphere_count * Collision::batch_size * 3;
        static constexpr int approx_scratch_stride =
            Collision::approximate_sphere_count * Collision::batch_size * 3;
        static constexpr int link_scratch_stride =
            Collision::joint_flag_stride * Collision::batch_size;
        static constexpr int transform_scratch_stride =
            Collision::batch_size * Collision::transform_slots * 16;
        const std::size_t batch_edge = blockIdx.x;
        node_anchors += batch_edge * 2 * Robot::dimension;
        edge_motion_segments +=
            batch_edge * CONCON_MOTION_SEGMENT_STRIDE;
        sphere_pos_scratch += batch_edge * fine_scratch_stride;
        sphere_pos_approx_scratch += batch_edge * approx_scratch_stride;
        link_cc_scratch += batch_edge * link_scratch_stride;
        transform_scratch += batch_edge * transform_scratch_stride;
        edge_is_valid += batch_edge;

        const int tid = threadIdx.x;
        __shared__ volatile unsigned int edge_cc_result[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_detailed_env_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_self_collision_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool run_detailed_self_check[
            MAX_PARALLEL_CONCON_EDGES
        ];
        __shared__ bool any_detailed_env_check;
        __shared__ bool any_detailed_self_check;
        __shared__ volatile int first_collision_edge[1];
        __shared__ int joint_limits_good;

        if (tid == 0) {
            edge_is_valid[0] = 0;
            joint_limits_good = 1;
            first_collision_edge[0] = 0;
        }
        if (tid <= d_settings.granularity) {
            const float ratio = static_cast<float>(tid) /
                static_cast<float>(d_settings.granularity);
            for (int joint = 0; joint < Robot::dimension; ++joint) {
                edge_motion_segments[tid * Robot::dimension + joint] =
                    node_anchors[joint] + ratio *
                    (node_anchors[Robot::dimension + joint] -
                     node_anchors[joint]);
            }
        }
        __syncthreads();

        if (
            tid <= d_settings.granularity &&
            !planning::configuration_within_joint_limits<Robot>(
                &edge_motion_segments[tid * Robot::dimension]
            )
        ) {
            atomicExch(&joint_limits_good, 0);
        }
        __syncthreads();

        if (joint_limits_good != 0) {
            patacon_check_projected_edges_collision_parallel<Robot>(
                1,
                edge_motion_segments,
                sphere_pos_scratch,
                sphere_pos_approx_scratch,
                link_cc_scratch,
                transform_scratch,
                env,
                edge_cc_result,
                run_detailed_env_check,
                run_self_collision_check,
                run_detailed_self_check,
                &any_detailed_env_check,
                &any_detailed_self_check,
                first_collision_edge,
                tid
            );
        }
        __syncthreads();

        if (tid == 0) {
            edge_is_valid[0] =
                joint_limits_good != 0 && first_collision_edge[0] >= 1;
        }
    }

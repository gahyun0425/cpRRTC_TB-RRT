// Internal tree bookkeeping helpers used by EXTEND and CONNECT.
// Included by PATACON.cu inside namespace PATACON.

    __device__ __forceinline__
    int patacon_reserve_slot(volatile int *counter,int capacity){
        int current =atomicAdd((int *)counter,0);

        while (current < capacity) {
            const int observed =atomicCAS((int *)counter,current,current + 1);

            if (observed == current) {
                return current;
            }

            current = observed;
        }

        return -1;
    }

    template <int LaneStride = MAX_THREADS_PER_BLOCK>
    __device__ __forceinline__
    void patacon_register_node_in_ts(
        int node_idx,
        int ts_id,
        int *ts_node_count,
        int *ts_lane_head,
        int *node_next_in_ts
    ) {
        if (ts_id < 0) {
            return;
        }

        // TS 안에서 등록된 순서에 따라 64개 thread 목록에 고르게 배정
        const int ordinal =atomicAdd(&ts_node_count[ts_id],1);
        const int lane =ordinal % blockDim.x;
        const int head_slot =ts_id * LaneStride + lane;

        int old_head =atomicAdd(&ts_lane_head[head_slot],0);

        while (true) {
            // next를 먼저 기록한 뒤 새 head를 공개한다.
            node_next_in_ts[node_idx] =old_head;
            __threadfence();

            const int observed =atomicCAS(
                &ts_lane_head[head_slot],
                old_head,
                node_idx
            );

            if (observed == old_head) {
                break;
            }

            old_head =observed;
        }
    }

    template <typename Robot>
    __device__ __forceinline__ float patacon_project_target_direction_to_tangent(
        const float *q_current,
        const float *q_target,
        const float *basis,
        float *ts_coeff,
        float *projected_dir,
        float *sdata,
        int tid
    ) {
        if constexpr (TangentSpaceTraits<Robot>::enabled) {
            static constexpr int dim = Robot::dimension;
            static constexpr int basis_stride =
                TangentSpaceTraits<Robot>::max_tangent_dim;

            const int active_tangent_dim = patacon_active_tangent_dim<Robot>();

            // 1. q_current -> q_target 방향을 Tangent basis 좌표계의 coefficient로 변환
            // ts_coeff = B^T * (q_target - q_current)
            if (tid < active_tangent_dim) {
                float coeff = 0.0f;

                for (int j = 0; j < dim; j++) {
                    const float target_vector =q_target[j] - q_current[j];
                    coeff += basis[j * basis_stride + tid] * target_vector;
                }

                ts_coeff[tid] = coeff;
            }

            __syncthreads();

            // 2. coefficient를 다시 joint-space 방향으로 변환
            // projected_dir = B * ts_coeff = B * B^T * (q_target - q_current)
            float projected_component = 0.0f;

            if (tid < dim) {

                for (int k = 0; k < active_tangent_dim; k++) {
                    projected_component += basis[tid * basis_stride + k] * ts_coeff[k];
                }

                projected_dir[tid] =projected_component;
            }

            // 3. projected direction의 norm 계산 준비
            if (tid < dim) {
                const float weight =
                    patacon_joint_distance_weight<Robot>(tid);

                const float weighted_component =
                    weight * projected_component;

                sdata[tid] =
                    weighted_component * weighted_component;
            }
            else {
                sdata[tid] = 0.0f;
            }

            __syncthreads();

            // block reduction
            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] +=sdata[tid + s];
                }

                __syncthreads();
            }

            const float projected_norm =sqrtf(sdata[0]);

            // 4. unit direction으로 normalize
            if (tid < dim) {

                if (projected_norm > 1.0e-8f) {
                    projected_dir[tid] /=projected_norm;
                }
                else {
                    projected_dir[tid] = 0.0f;
                }
            }

            __syncthreads();

            return projected_norm;
        }

        return 0.0f;
    }
        

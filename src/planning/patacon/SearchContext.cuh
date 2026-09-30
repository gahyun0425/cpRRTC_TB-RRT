// Internal PATACON search data and block-local state.
// Included by PATACON.cu inside namespace PATACON.

    enum class PataconExtendStatus {
        Trapped,
        Advanced,
        Terminate
    };

    enum class PataconExtendMode {
        Exploration,
        ConnectTarget
    };

    struct PataconExtendResult {
        PataconExtendStatus status = PataconExtendStatus::Trapped;

        __device__ __forceinline__ bool advanced() const {
            return status == PataconExtendStatus::Advanced;
        }
    };

    struct PataconExtendPreparation {
        PataconExtendStatus status = PataconExtendStatus::Advanced;
        float target_distance = 0.0f;
        float tangent_distance = 0.0f;
    };

    enum class PataconConnectStatus {
        NotConnected,
        Connected,
        Terminate
    };

    struct PataconConnectResult {
        PataconConnectStatus status = PataconConnectStatus::NotConnected;
        float final_distance = 0.0f;

        __device__ __forceinline__ bool connected() const {
            return status == PataconConnectStatus::Connected;
        }
    };

    template <typename Robot>
    struct PataconSearchContext {
        float **nodes;
        int **parents;
        int **node_ready;
        int **node_ts_id;
        float **node_ts_q;
        int *ts_count;
        int **ts_root_node_idx;
        int **ts_parent_id;
        float **ts_bases;
        int **ts_ready;
        int **ts_node_count;
        int **ts_lane_head;
        int **node_next_in_ts;
        HaltonState<Robot> *halton_states;
        curandState *rng_states;
        ppln::collision::Environment<float> *environment;
        volatile float *concon_sphere_pos_scratch;
        volatile float *concon_sphere_pos_approx_scratch;
        volatile int *concon_link_cc_scratch;
        float *concon_transform_scratch;
    };

    template <typename Robot>
    struct PataconBlockState {
        static constexpr int dim = Robot::dimension;

        int t_tree_id;
        int o_tree_id;
        float config[dim];
        float sdata[MAX_THREADS_PER_BLOCK];
        int sindex[MAX_THREADS_PER_BLOCK];
        float *t_nodes;
        float *o_nodes;
        int *t_parents;
        int *o_parents;
        int *t_node_ts_id;
        float *t_node_ts_q;
        int *t_ts_root_node_idx;
        int *t_ts_parent_id;
        float *t_ts_bases;
        int *t_ts_ready;
        int *t_ts_node_count;
        int *t_ts_lane_head;
        int *t_node_next_in_ts;
        int t_ts_count;
        int selected_ts_id;
        int selected_ts_root_idx;
        int *t_node_ready;
        int *o_node_ready;
        int t_tree_size;
        float ts_coeff[MAX_TANGENT_DIM];
        float ts_alpha_fraction;
        float ts_tangent_dir[MAX_ROBOT_DIM];
        float scale;
        float *nearest_node;
        float *nearest_ts_node;
        float q_rand_dist;
        float extend_dir[dim];
        float concon_probe[dim];
        int concon_count;
        bool concon_em_stop;
        int concon_valid_count;
        int new_ts_id;
        bool new_ts_basis_ok;
        int concon_parent_idx;
        int extend_edge_count;
        int index;
        bool should_skip;
        int connect_target_idx;
        float *connect_target_node;
        bool connect_failed;
        alignas(16) volatile float motion_segment[
            MAX_CONCON_NODE_ANCHORS * MAX_ROBOT_DIM
        ];
        alignas(16) volatile float motion_segment_next[
            MAX_PARALLEL_CONCON_EDGES * CONCON_MOTION_SEGMENT_STRIDE
        ];
        volatile unsigned char motion_projection_valid[
            MAX_CONCON_PROJECTION_STATES
        ];
        volatile int motion_projection_prog[MAX_PARALLEL_CONCON_EDGES];
        volatile unsigned int motion_projection_success[
            MAX_PARALLEL_CONCON_EDGES
        ];
        volatile int concon_first_projection_failure_edge[1];
        alignas(16) volatile float concon_motion_segments[
            MAX_PARALLEL_CONCON_EDGES * CONCON_MOTION_SEGMENT_STRIDE
        ];
        float concon_nominal_targets[
            MAX_PARALLEL_CONCON_EDGES * MAX_ROBOT_DIM
        ];
        volatile unsigned int concon_edge_cc_result[
            MAX_PARALLEL_CONCON_EDGES
        ];
        bool concon_run_detailed_env_check[MAX_PARALLEL_CONCON_EDGES];
        bool concon_run_self_collision_check[MAX_PARALLEL_CONCON_EDGES];
        bool concon_run_detailed_self_check[MAX_PARALLEL_CONCON_EDGES];
        bool concon_any_detailed_env_check;
        bool concon_any_detailed_self_check;
        volatile int concon_first_collision_edge[1];
        int concon_projected_edge_count;

        __device__ __forceinline__ void check_iteration_limit(
            int iteration,
            unsigned long long block_start_time_ns
        );

        __device__ __forceinline__ void select_tree(
            const PataconSearchContext<Robot> &search,
            int block_id,
            int iteration
        );

        __device__ __forceinline__ void select_tangent_space(
            const PataconSearchContext<Robot> &search,
            int block_id,
            int iteration
        );

        template <bool TraceTrees>
        __device__ __forceinline__ bool should_terminate();

        __device__ __forceinline__ bool has_selected_tangent_space() const;

        __device__ __forceinline__ void sample_q_rand(
            const PataconSearchContext<Robot> &search,
            int block_id,
            int thread_id
        );

        __device__ __forceinline__ bool select_q_near(
            const PataconSearchContext<Robot> &search,
            int thread_id
        );

        __device__ __forceinline__ void compute_v_ext(int thread_id);

        template <PataconExtendMode Mode, bool TraceTrees>
        __device__ __forceinline__ PataconExtendPreparation
        prepare_extension(int thread_id);

        template <PataconExtendMode Mode>
        __device__ __forceinline__ void project_extension_candidates(
            const PataconExtendPreparation &preparation,
            int thread_id
        );

        template <PataconExtendMode Mode>
        __device__ __forceinline__ void validate_extension_edges(
            const PataconSearchContext<Robot> &search,
            int block_id,
            int thread_id
        );

        template <PataconExtendMode Mode, bool TraceTrees>
        __device__ __forceinline__ PataconExtendResult
        insert_extension_nodes(
            const PataconSearchContext<Robot> &search,
            int thread_id
        );

        template <PataconExtendMode Mode, bool TraceTrees>
        __device__ __forceinline__ PataconExtendResult extend(
            const PataconSearchContext<Robot> &search,
            int block_id,
            int thread_id
        );

        __device__ __forceinline__ void select_connect_target(int thread_id);

        __device__ __forceinline__ PataconConnectResult check_connection(
            const PataconSearchContext<Robot> &search,
            int block_id,
            int thread_id
        );

        template <bool TraceTrees>
        __device__ __forceinline__ PataconConnectResult connect_trees(
            const PataconSearchContext<Robot> &search,
            int block_id,
            int thread_id
        );

        template <bool TraceTrees>
        __device__ __forceinline__ void derive_solution_path(
            const PataconSearchContext<Robot> &search,
            float final_connection_distance,
            int iteration,
            int thread_id
        );

    };

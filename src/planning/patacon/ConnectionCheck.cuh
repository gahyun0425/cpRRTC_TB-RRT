// CHECK_CONNECTION keeps the original PATACON/cpRRTC connection rule:
// the two trees are connected when their selected nodes are within tolerance.

    template <typename Robot>
    __device__ __forceinline__ PataconConnectResult
    PataconBlockState<Robot>::check_connection(
        const PataconSearchContext<Robot> &,
        int,
        int tid
    ) {
        const float connection_distance =
            patacon_shared_config_distance<Robot>(
                config,
                connect_target_node,
                sdata,
                tid
            );
        const bool connected =
            connection_distance <= d_settings.connect_reached_tolerance;

        if (tid == 0 && connected) {
            diagnostic_increment(DIAG_CONNECT_SUCCESSES);
        }
        __syncthreads();

        return {
            connected
                ? PataconConnectStatus::Connected
                : PataconConnectStatus::NotConnected,
            connection_distance
        };
    }

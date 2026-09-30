// The single EXTEND operation from Algorithm 2. Both exploration and CONNECT
// call this same flow; Mode only selects their existing stage calculations.

    template <typename Robot>
    template <PataconExtendMode Mode, bool TraceTrees>
    __device__ __forceinline__ PataconExtendResult
    PataconBlockState<Robot>::extend(
        const PataconSearchContext<Robot> &search,
        int bid,
        int tid
    ) {
        const PataconExtendPreparation preparation =
            prepare_extension<Mode, TraceTrees>(tid);

        if (preparation.status != PataconExtendStatus::Advanced) {
            return {preparation.status};
        }

        project_extension_candidates<Mode>(preparation, tid);
        validate_extension_edges<Mode>(search, bid, tid);

        return insert_extension_nodes<Mode, TraceTrees>(search, tid);
    }

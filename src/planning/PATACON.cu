#include "Planners.hh"
#include "Robots.hh"
#include "RobotCollisionTraits.hh"
#include "JointLimits.cuh"
#include "utils.cuh"
#include "PATACON_settings.hh"
#include "src/collision/environment.hh"
#include "src/robots/franka_fer.cuh"
#include "src/robots/franka_collision.cuh"
#include "src/robots/franka_constraint.cuh"
#include "src/robots/ffw_sg2.cuh"
#include "src/robots/ffw_sg2_attached_object_collision.cuh"
#include "src/robots/ffw_sg2_mobility.cuh"
#include "src/robots/ffw_sg2_constraint.cuh"
#include "src/robots/ffw_sg2_mobility_constraint.cuh"
#include "src/robots/ffw_sg2_mobility_com_constraint.cuh"
#include "src/robots/g1_collision.cuh"
#include "src/robots/g1_attached_object_collision.cuh"
#include "src/robots/g1_constraint.cuh"
#include "src/robots/igris_c_collision.cuh"
#include "src/robots/igris_c_constraint.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace PATACON {
    using namespace ppln;

#include "patacon/GpuRuntime.cuh"
#include "patacon/DeviceEnvironment.cuh"
#include "patacon/Projection.cuh"
#include "patacon/CollisionValidation.cuh"
#include "patacon/SearchContext.cuh"
#include "patacon/TreeOperations.cuh"
#include "patacon/TreeSelection.cuh"
#include "patacon/Exploration.cuh"
#include "patacon/ExtendStages.cuh"
#include "patacon/Extend.cuh"
#include "patacon/ConnectionCheck.cuh"
#include "patacon/Connect.cuh"
#include "patacon/PathExtraction.cuh"
#include "patacon/PathAssembly.cuh"
#include "patacon/PathPostprocessing.cuh"
#include "patacon/SolveWorkspace.cuh"

    // PATACON Algorithm 1 search flow. Each operation keeps its CUDA details
    // behind a pseudocode-shaped block-level function call.
    template <typename Robot, bool TraceTrees>
    __global__ void patacon(PataconSearchContext<Robot> search) {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const unsigned long long block_start_time_ns = global_timer_ns();
        __shared__ PataconBlockState<Robot> block;
        int iteration = 0;

        while (true) {
            if (tid == 0) {
                ++iteration;
                block.check_iteration_limit(iteration, block_start_time_ns);
                block.select_tree(search, bid, iteration);
                block.select_tangent_space(search, bid, iteration);
            }
            __syncthreads();

            if (block.template should_terminate<TraceTrees>()) {
                return;
            }

            if (!block.has_selected_tangent_space()) {
                continue;
            }

            block.sample_q_rand(search, bid, tid);

            if (!block.select_q_near(search, tid)) {
                continue;
            }

            block.compute_v_ext(tid);

            const PataconExtendResult extension = block.template extend<PataconExtendMode::Exploration, TraceTrees>(search, bid, tid);

            if (extension.status == PataconExtendStatus::Terminate) {
                return;
            }
            if (extension.advanced()) {
                block.select_connect_target(tid);

                PataconConnectResult connection = block.check_connection(search, bid, tid);

                if (!connection.connected()) {
                    connection = block.template connect_trees<TraceTrees>(search, bid, tid);
                }

                if (connection.status == PataconConnectStatus::Terminate) {
                    return;
                }

                if (connection.connected()) {
                    block.template derive_solution_path<TraceTrees>(search, connection.final_distance,iteration, tid);
                }
            }

            __syncthreads();

            if (block.template should_terminate<TraceTrees>()) {
                return;
            }
        }
    }

    // Top-level PATACON solve flow is kept in this translation-unit entry file.
    void release_persistent_workspace() {
        persistent_g1_workspace().reset();
        release_g1_projection_workspace();
    }

#include "patacon/Solve.cuh"

    template <typename Robot>
    PlannerResult<Robot> solve(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &environment,
        PATACON_settings &settings
    ) {
        return solve_backend<Robot>(start, goals, environment, settings);
    }

#include "patacon/ExplicitInstantiations.cuh"
}

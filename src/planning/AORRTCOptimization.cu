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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// AORRTC reuses PATACON's header-defined device algorithms with translation-
// unit-local CUDA state. Public host runtime controls are owned by
// RuntimeControl.cpp.
#define PATACON_SKIP_G1_PROJECTION_RUNTIME
namespace PATACON {
    using namespace ppln;

#include "patacon/GpuRuntime.cuh"
#include "patacon/DeviceEnvironment.cuh"
#include "patacon/Projection.cuh"
#include "patacon/CollisionValidation.cuh"
#include "patacon/TreeOperations.cuh"
#include "patacon/PathPostprocessing.cuh"
}
#undef PATACON_SKIP_G1_PROJECTION_RUNTIME

#include "AORRTCOptimization.cuh"

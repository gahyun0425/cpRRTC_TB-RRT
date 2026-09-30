#include "src/planning/Planners.hh"

#include <cmath>
#include <stdexcept>

namespace PATACON {
namespace {
    RuntimeControlState runtime_control;

    void validate_runtime_control_state(const RuntimeControlState &state) {
        if (
            !std::isfinite(state.time_limit_seconds) ||
            state.time_limit_seconds < 0.0
        ) {
            throw std::invalid_argument(
                "PATACON time limit must be finite and nonnegative"
            );
        }
    }
}

RuntimeControlState runtime_control_state() {
    return runtime_control;
}

void set_runtime_control_state(const RuntimeControlState &state) {
    validate_runtime_control_state(state);
    runtime_control = state;
}

void set_cuda_device_reset_enabled(const bool enabled) {
    runtime_control.cuda_device_reset_enabled = enabled;
}

void set_persistent_workspace_enabled(const bool enabled) {
    runtime_control.persistent_workspace_enabled = enabled;
}

void set_time_limit_seconds(const double seconds) {
    RuntimeControlState next = runtime_control;
    next.time_limit_seconds = seconds;
    set_runtime_control_state(next);
}

}

#pragma once

namespace ppln::planning {

template <typename Robot>
__device__ __forceinline__ bool joint_value_within_limits(
    float value,
    int joint
) {
    return
        isfinite(value) &&
        value >= Robot::get_s_a(joint) &&
        value <= Robot::get_s_a(joint) + Robot::get_s_m(joint);
}

template <typename Robot>
__device__ __forceinline__ bool configuration_within_joint_limits(
    const float *configuration
) {
    for (int joint = 0; joint < Robot::dimension; ++joint) {
        if (!joint_value_within_limits<Robot>(configuration[joint], joint)) {
            return false;
        }
    }
    return true;
}

template <typename Robot>
__device__ __forceinline__ bool configuration_within_joint_limits(
    const volatile float *configuration
) {
    for (int joint = 0; joint < Robot::dimension; ++joint) {
        if (!joint_value_within_limits<Robot>(configuration[joint], joint)) {
            return false;
        }
    }
    return true;
}

} // namespace ppln::planning

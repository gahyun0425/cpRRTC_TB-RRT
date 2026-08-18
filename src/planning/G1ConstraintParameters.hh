#pragma once

#include <type_traits>

namespace ppln::constraints {

struct EmptyConstraintParameters
{
};

struct G1ConstraintParameters
{
    float feet_reference[2][7]{};
    float feet_target[2][7]{};

    float support_polygon[8]{};

    float bimanual_target[7]{};

    float tolerance_squared = 1.0e-6f;
};

static_assert(
    std::is_trivially_copyable_v<G1ConstraintParameters>,
    "G1 parameters must be copied to CUDA constant memory"
);

}

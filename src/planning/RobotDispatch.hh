#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "src/config/RobotRegistry.hh"
#include "src/planning/Robots.hh"
#include "src/robots/igris_c.cuh"

namespace ppln::planning {

static_assert(
    config::compiled_robot_dimension("franka_single") ==
        robots::FrankaSingle::dimension &&
    config::compiled_robot_dimension("franka") ==
        robots::Franka::dimension &&
    config::compiled_robot_dimension("ffw_sg2") ==
        robots::FfwSg2::dimension &&
    config::compiled_robot_dimension("ffw_sg2_mobility") ==
        robots::FfwSg2Mobility::dimension &&
    config::compiled_robot_dimension("g1") == robots::G1::dimension &&
    config::compiled_robot_dimension("igris_c") == robots::IgrisC::dimension,
    "RobotRegistry dimensions must match the compiled robot backends"
);

template <typename Robot>
struct RobotTag {
    using type = Robot;
};

template <typename Visitor>
decltype(auto) dispatch_robot(
    const std::string_view name,
    Visitor &&visitor
) {
    if (name == "franka_single") {
        return std::forward<Visitor>(visitor)(RobotTag<robots::FrankaSingle>{});
    }
    if (name == "franka") {
        return std::forward<Visitor>(visitor)(RobotTag<robots::Franka>{});
    }
    if (name == "ffw_sg2") {
        return std::forward<Visitor>(visitor)(RobotTag<robots::FfwSg2>{});
    }
    if (name == "ffw_sg2_mobility") {
        return std::forward<Visitor>(visitor)(
            RobotTag<robots::FfwSg2Mobility>{}
        );
    }
    if (name == "g1") {
        return std::forward<Visitor>(visitor)(RobotTag<robots::G1>{});
    }
    if (name == "igris_c") {
        return std::forward<Visitor>(visitor)(RobotTag<robots::IgrisC>{});
    }
    throw std::invalid_argument(
        "unsupported robot model: " + std::string(name)
    );
}

}  // namespace ppln::planning

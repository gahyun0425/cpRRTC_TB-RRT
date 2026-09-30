#include <nlohmann/json.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/config/PlanningProblemJson.hh"

using json = nlohmann::json;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

json make_pose(double x, double y, double z) {
    return {
        {"quaternion_wxyz", {1.0, 0.0, 0.0, 0.0}},
        {"position", {x, y, z}}
    };
}

json make_unified_problem() {
    const std::vector<double> configuration(35, 0.0);
    return {
        {"schema_version", 1},
        {"robot", {{"model", "igris_c"}, {"dimension", 35}}},
        {"name", "codec_test"},
        {"planner", {{"range", 0.25}, {"axis", true}}},
        {"query", {
            {"start", configuration},
            {"goals", json::array({configuration})}
        }},
        {"world", {
            {"sphere", json::array()},
            {"box", json::array()}
        }},
        {"constraints", {
            {"tolerance_squared", 1.0e-6},
            {"items", json::array({
                {
                    {"type", "fixed_frame_pose"},
                    {"frame", "left_foot"},
                    {"target", make_pose(0.0, 0.1, 0.0)}
                },
                {
                    {"type", "fixed_frame_pose"},
                    {"frame", "right_foot"},
                    {"target", make_pose(0.0, -0.1, 0.0)}
                },
                {
                    {"type", "relative_pose"},
                    {"frame_a", "left_hand"},
                    {"frame_b", "right_hand"},
                    {"target", make_pose(0.0, -0.3, 0.0)}
                },
                {
                    {"type", "axis_alignment"},
                    {"frame", "left_hand"},
                    {"local_axis", {1.0, 0.0, 0.0}},
                    {"target_world_axis", {0.0, 0.0, 1.0}}
                },
                {
                    {"type", "com_support"},
                    {"support_frames", {"left_foot", "right_foot"}},
                    {"support_polygon", {
                        -0.1, -0.1, -0.1, 0.1,
                        0.1, 0.1, 0.1, -0.1
                    }},
                    {"support_margin_m", 0.02},
                    {"payload_mass_kg", 0.0}
                }
            })}
        }}
    };
}

}  // namespace

int main() {
    try {
        const auto selected = ppln::config::select_problem(
            make_unified_problem()
        );
        require(selected.robot_name == "igris_c", "robot model was not loaded");
        require(selected.problem_name == "codec_test", "problem name was not loaded");
        require(selected.data.at("start").size() == 35, "start was not normalized");
        require(selected.data.at("goals").at(0).size() == 35, "goal was not normalized");
        require(selected.data.at("planner").at("range") == 0.25, "planner settings were not retained");
        require(selected.data.at("cylinder").empty(), "missing world arrays were not added");

        const auto &constraints = selected.data.at("constraints");
        require(constraints.at("feet").at("target").size() == 2, "feet were not normalized");
        require(constraints.at("bimanual").at("target").size() == 7, "bimanual pose was not normalized");
        require(constraints.at("bimanual_axis").at("local_axis").size() == 3, "axis was not normalized");
        require(constraints.at("com").at("support_polygon").size() == 8, "CoM support was not normalized");

        json invalid = make_unified_problem();
        invalid["robot"]["dimension"] = 34;
        bool rejected = false;
        try {
            (void)ppln::config::select_problem(invalid);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "invalid robot dimension was accepted");
    } catch (const std::exception &error) {
        std::cerr << "problem config validation failed: "
                  << error.what() << "\n";
        return 1;
    }

    std::cout << "problem config validation passed\n";
    return 0;
}

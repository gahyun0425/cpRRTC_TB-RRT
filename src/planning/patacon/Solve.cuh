// Pseudocode-shaped host-side PATACON solve orchestration.
// Included by PATACON.cu inside namespace PATACON.

#include "SolveStages.cuh"

    template <typename Robot>
    PlannerResult<Robot> solve_backend(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &environment,
        PATACON_settings &settings
    ) {
        PataconSolveSession<Robot> session(
            start,
            goals,
            environment,
            settings
        );

        session.validate_request();
        session.publish_request_settings();
        session.prepare_workspace();
        session.initialize_request_state();
        session.upload_problem_roots();
        session.launch_search();
        session.collect_result();
        session.release_request_resources();

        return session.take_result();
    }

// Explicit instantiations for the compiled robot backends.

    //template PlannerResult<typename ppln::robots::Sphere> solve<ppln::robots::Sphere>(std::array<float, 3>&, std::vector<std::array<float, 3>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::FrankaSingle> solve<ppln::robots::FrankaSingle>(std::array<float, 7>&, std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::Franka> solve<ppln::robots::Franka>(std::array<float, 14>&, std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::FfwSg2> solve<ppln::robots::FfwSg2>(std::array<float, 15>&, std::vector<std::array<float, 15>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::FfwSg2Mobility> solve<ppln::robots::FfwSg2Mobility>(std::array<float, 18>&, std::vector<std::array<float, 18>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::G1> solve<ppln::robots::G1>(std::array<float, 35>&, std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PlannerResult<typename ppln::robots::IgrisC> solve<ppln::robots::IgrisC>(std::array<float, 35>&, std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&);


    template PathSimplificationResult<ppln::robots::FrankaSingle> simplify_path_for_visualization<ppln::robots::FrankaSingle>(const std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PathSimplificationResult<ppln::robots::Franka> simplify_path_for_visualization<ppln::robots::Franka>(const std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PathSimplificationResult<ppln::robots::FfwSg2> simplify_path_for_visualization<ppln::robots::FfwSg2>(const std::vector<std::array<float, 15>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PathSimplificationResult<ppln::robots::FfwSg2Mobility> simplify_path_for_visualization<ppln::robots::FfwSg2Mobility>(const std::vector<std::array<float, 18>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PathSimplificationResult<ppln::robots::G1> simplify_path_for_visualization<ppln::robots::G1>(const std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&);
    template PathSimplificationResult<ppln::robots::IgrisC> simplify_path_for_visualization<ppln::robots::IgrisC>(const std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&);

    template PathValidationResult validate_path_for_visualization<ppln::robots::FrankaSingle>(const std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);
    template PathValidationResult validate_path_for_visualization<ppln::robots::Franka>(const std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);
    template PathValidationResult validate_path_for_visualization<ppln::robots::FfwSg2>(const std::vector<std::array<float, 15>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);
    template PathValidationResult validate_path_for_visualization<ppln::robots::FfwSg2Mobility>(const std::vector<std::array<float, 18>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);
    template PathValidationResult validate_path_for_visualization<ppln::robots::G1>(const std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);
    template PathValidationResult validate_path_for_visualization<ppln::robots::IgrisC>(const std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, PATACON_settings&, float);

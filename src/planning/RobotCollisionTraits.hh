#pragma once

#include "Robots.hh"
#include "src/robots/igris_c.cuh"

namespace ppln::robots {

    template<typename Robot>
    struct CollisionTraits;

    template<>
    struct CollisionTraits<Panda> {
        static constexpr int batch_size = 16;
        static constexpr int fine_sphere_count = 59;
        static constexpr int approximate_sphere_count = 11;
        static constexpr int joint_flag_stride = 20;
        static constexpr int transform_slots = 1;
    };

    template<>
    struct CollisionTraits<FrankaSingle> {
        static constexpr int batch_size = 16;
        static constexpr int fine_sphere_count = 59;
        static constexpr int approximate_sphere_count = 11;
        static constexpr int joint_flag_stride = 20;
        static constexpr int transform_slots = 1;
    };

    template<>
    struct CollisionTraits<Franka> {
        static constexpr int batch_size = 16;
        static constexpr int fine_sphere_count = 118;
        static constexpr int approximate_sphere_count = 22;
        // Both identical arms share the legacy Panda link-flag slots.  A flag
        // activates the corresponding link on both arms during refinement.
        static constexpr int joint_flag_stride = 20;
        static constexpr int transform_slots = 2;
    };

    template<>
    struct CollisionTraits<Fetch> {
        static constexpr int batch_size = 16;
        static constexpr int fine_sphere_count = 111;
        static constexpr int approximate_sphere_count = 15;
        static constexpr int joint_flag_stride = 20;
        static constexpr int transform_slots = 1;
    };

    template<>
    struct CollisionTraits<Baxter> {
        static constexpr int batch_size = 16;
        static constexpr int fine_sphere_count = 75;
        static constexpr int approximate_sphere_count = 33;
        static constexpr int joint_flag_stride = 20;
        static constexpr int transform_slots = 2;
    };

    template<>
    struct CollisionTraits<FfwSg2> {
        static constexpr int batch_size = 16;
        static constexpr int fine_sphere_count = 124;
        static constexpr int approximate_sphere_count = 27;
        static constexpr int joint_flag_stride = 16;
        static constexpr int transform_slots = 2;
    };

    template<>
    struct CollisionTraits<FfwSg2Mobility> {
        static constexpr int batch_size = 16;
        static constexpr int fine_sphere_count = 125;
        static constexpr int approximate_sphere_count = 28;
        static constexpr int joint_flag_stride = 16;
        static constexpr int transform_slots = 2;
    };

    template<>
    struct CollisionTraits<FfwSg2Single> {
        static constexpr int batch_size = 16;
        static constexpr int fine_sphere_count = 124;
        static constexpr int approximate_sphere_count = 27;
        static constexpr int joint_flag_stride = 9;
        static constexpr int transform_slots = 1;
    };

    template<>
    struct CollisionTraits<G1> {
        static constexpr int batch_size = G1::resolution;
        static constexpr int fine_sphere_count = G1::n_spheres;
        static constexpr int approximate_sphere_count = 1;
        static constexpr int joint_flag_stride = 1;
        static constexpr int transform_slots = 1;
    };

    template<>
    struct CollisionTraits<IgrisC> {
        static constexpr int batch_size = IgrisC::resolution;
        static constexpr int fine_sphere_count = IgrisC::n_spheres;
        static constexpr int approximate_sphere_count =
            IgrisC::n_approx_spheres;
        static constexpr int joint_flag_stride = 1;
        static constexpr int transform_slots = 1;
    };

}

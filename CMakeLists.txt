cmake_minimum_required(VERSION 3.16...3.22)

set(PATACON_BUILD_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build_patacon")
if(
    CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
    AND NOT CMAKE_BINARY_DIR STREQUAL PATACON_BUILD_DIRECTORY
)
    message(FATAL_ERROR
        "PATACON uses a single build directory. Configure with "
        "`cmake --preset patacon` (binary directory: build_patacon)."
    )
endif()

find_program(CLANGPP "clang++")
set(CMAKE_CXX_COMPILER ${CLANGPP} CACHE STRING "C++ compiler" FORCE)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)


project (
    batched_cc
    VERSION 0.0
    LANGUAGES CXX CUDA
)

option(
    PATACON_CUDA_SPLIT_COMPILE
    "Parallelize optimization inside large CUDA translation units"
    ON
)
# Each planner translation unit contains the header-defined robot device code
# it needs. Keep whole-program CUDA compilation per object; enabling RDC would
# merge those private robot definitions and recreate duplicate device symbols.
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)
# set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)


# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo")  # enable line info in ncu ui


### Unified PATACON CUDA backend

# Keep the standard PATACON search and opt-in AORRTC optimization in separate
# self-contained CUDA translation units. Both remain in the same static
# backend library, while header-defined robot device code stays local to each
# object instead of being merged by nvlink.
add_library(patacon_planners STATIC
    src/planning/patacon/RuntimeControl.cpp
    src/planning/PATACON.cu
    src/planning/AORRTCOptimization.cu
)

set_target_properties(patacon_planners PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

target_include_directories(patacon_planners
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(patacon_planners
    PUBLIC
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
)

target_compile_options(patacon_planners PRIVATE
    -Wno-c++11-narrowing
)

if(PATACON_CUDA_SPLIT_COMPILE)
    target_compile_options(patacon_planners PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>
    )
endif()


### motion benchmark script
add_executable(evaluate_mbm
    scripts/evaluate_mbm.cpp
)

target_include_directories(evaluate_mbm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


target_link_libraries(evaluate_mbm
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
    patacon_planners
)

target_compile_options(evaluate_mbm PRIVATE
    -Wno-c++11-narrowing
)

### single_mbm
add_executable(single_mbm
    scripts/single_mbm.cpp
)

target_include_directories(single_mbm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(single_mbm
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
    patacon_planners
)

target_compile_options(single_mbm PRIVATE
    -Wno-c++11-narrowing
)

### Unified robot/problem JSON and constraint-codec validation

add_executable(validate_problem_config
    scripts/validate_problem_config.cpp
)

target_include_directories(validate_problem_config
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(validate_problem_config
    PRIVATE
    nlohmann_json::nlohmann_json
)

set_target_properties(evaluate_mbm single_mbm PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

### FFW SG2 CUDA projection validation

add_executable(validate_ffw_sg2_projection
    scripts/validate_ffw_sg2_projection.cu
)

target_include_directories(validate_ffw_sg2_projection
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(validate_ffw_sg2_projection
    PRIVATE
    Eigen3::Eigen
)

target_compile_options(validate_ffw_sg2_projection PRIVATE
    -Wno-c++11-narrowing
)

### Reproducible FFW-SG2 constrained start-goal pair generation

add_executable(generate_ffw_sg2_random_pairs
    scripts/generate_ffw_sg2_random_pairs.cu
)

target_include_directories(generate_ffw_sg2_random_pairs
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(generate_ffw_sg2_random_pairs
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
)

target_compile_options(generate_ffw_sg2_random_pairs PRIVATE
    -Wno-c++11-narrowing
)

### Reproducible FFW-SG2 mobility CoM-plus-axis start-goal generation

add_executable(generate_ffw_sg2_mobility_random_pairs
    scripts/generate_ffw_sg2_mobility_random_pairs.cu
)

target_include_directories(generate_ffw_sg2_mobility_random_pairs
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(generate_ffw_sg2_mobility_random_pairs
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
)

target_compile_options(generate_ffw_sg2_mobility_random_pairs PRIVATE
    -Wno-c++11-narrowing
)

### Reproducible G1 no-axis start-goal generation

add_executable(generate_g1_random_pairs
    scripts/generate_g1_random_pairs.cu
)

target_include_directories(generate_g1_random_pairs
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(generate_g1_random_pairs
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
)

target_compile_options(generate_g1_random_pairs PRIVATE
    -Wno-c++11-narrowing
)

### Reproducible Franka-single start-goal generation

add_executable(generate_franka_single_random_pairs
    scripts/generate_franka_single_random_pairs.cu
)

target_include_directories(generate_franka_single_random_pairs
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(generate_franka_single_random_pairs
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
)

target_compile_options(generate_franka_single_random_pairs PRIVATE
    -Wno-c++11-narrowing
)

### Reproducible Franka-dual start-goal generation

add_executable(generate_franka_dual_random_pairs
    scripts/generate_franka_dual_random_pairs.cu
)

target_include_directories(generate_franka_dual_random_pairs
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(generate_franka_dual_random_pairs
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
)

target_compile_options(generate_franka_dual_random_pairs PRIVATE
    -Wno-c++11-narrowing
)

### G1 scalar FK / self and environment collision validation

add_executable(validate_g1_collision
    scripts/validate_g1_collision.cu
)

target_include_directories(validate_g1_collision
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(validate_g1_collision
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
)

target_compile_options(validate_g1_collision PRIVATE
    -Wno-c++11-narrowing
)

### G1 feet / center-of-mass / bimanual constraint validation

add_executable(validate_g1_constraints
    scripts/validate_g1_constraints.cu
)

target_include_directories(validate_g1_constraints
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(validate_g1_constraints
    PRIVATE
    nlohmann_json::nlohmann_json
    CUDA::cudart
)

target_compile_options(validate_g1_constraints PRIVATE
    -Wno-c++11-narrowing
)

### IGRIS-C generated FK / analytic Jacobian validation

add_executable(validate_igris_c_kinematics
    scripts/validate_igris_c_kinematics.cu
)

target_include_directories(validate_igris_c_kinematics
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(validate_igris_c_kinematics
    PRIVATE
    nlohmann_json::nlohmann_json
    CUDA::cudart
)

target_compile_options(validate_igris_c_kinematics PRIVATE
    -Wno-c++11-narrowing
)

### IGRIS-C generated collision-sphere validation

add_executable(validate_igris_c_collision
    scripts/validate_igris_c_collision.cu
)

target_include_directories(validate_igris_c_collision
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(validate_igris_c_collision
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
)

target_compile_options(validate_igris_c_collision PRIVATE
    -Wno-c++11-narrowing
)

### IGRIS-C feet / center-of-mass / bimanual constraint validation

add_executable(validate_igris_c_constraints
    scripts/validate_igris_c_constraints.cu
)

target_include_directories(validate_igris_c_constraints
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(validate_igris_c_constraints
    PRIVATE
    nlohmann_json::nlohmann_json
    CUDA::cudart
)

target_compile_options(validate_igris_c_constraints PRIVATE
    -Wno-c++11-narrowing
)

### Franka single-world-yaw / dual-pose-world-yaw constraint validation

add_executable(validate_franka_constraints
    scripts/validate_franka_constraints.cu
)

target_include_directories(validate_franka_constraints
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(validate_franka_constraints
    PRIVATE
    CUDA::cudart
)

target_compile_options(validate_franka_constraints PRIVATE
    -Wno-c++11-narrowing
)

### Franka imported sphere-collision validation

add_executable(validate_franka_collision
    scripts/validate_franka_collision.cu
)

target_include_directories(validate_franka_collision
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(validate_franka_collision
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
)

target_compile_options(validate_franka_collision PRIVATE
    -Wno-c++11-narrowing
)


### Pre-refactor behavior contracts

include(CTest)

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(PATACON_TEST_RESULT_DIRECTORY
        "${CMAKE_BINARY_DIR}/test-results"
    )
    set(PATACON_RESULT_SCHEMA
        "${CMAKE_CURRENT_SOURCE_DIR}/schemas/planner-result-v1.schema.json"
    )
    set(PATACON_PROBLEM_SET_SCHEMA
        "${CMAKE_CURRENT_SOURCE_DIR}/schemas/problem-set-results-v1.schema.json"
    )

    function(add_patacon_planning_regression
        test_name robot problem problem_index dimension
    )
        add_test(
            NAME "planning.${test_name}"
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_planning_regression.py"
                --planner "$<TARGET_FILE:single_mbm>"
                --schema "${PATACON_RESULT_SCHEMA}"
                --repository "${CMAKE_CURRENT_SOURCE_DIR}"
                --output
                    "${PATACON_TEST_RESULT_DIRECTORY}/${test_name}.json"
                --robot "${robot}"
                --problem "${problem}"
                --problem-index "${problem_index}"
                --dimension "${dimension}"
                --seed 1
        )
        set_tests_properties("planning.${test_name}" PROPERTIES
            LABELS "planning;schema;${robot}"
            RESOURCE_LOCK patacon_gpu
            TIMEOUT 120
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endfunction()

    add_patacon_planning_regression(
        franka_single franka_single demo 1 7
    )
    add_patacon_planning_regression(
        franka_dual franka demo 1 14
    )
    add_patacon_planning_regression(
        ffw_sg2 ffw_sg2 tray_lift 1 15
    )
    add_patacon_planning_regression(
        ffw_sg2_mobility ffw_sg2_mobility tray_lift 1 18
    )
    add_patacon_planning_regression(
        g1 g1 humanoid_shelf 1 35
    )
    add_patacon_planning_regression(
        igris_c igris_c igris_c_shelf_lift 1 35
    )

    add_test(
        NAME contract.aorrtc_result_v1
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_planning_regression.py"
            --planner "$<TARGET_FILE:single_mbm>"
            --schema "${PATACON_RESULT_SCHEMA}"
            --repository "${CMAKE_CURRENT_SOURCE_DIR}"
            --output
                "${PATACON_TEST_RESULT_DIRECTORY}/aorrtc_result_v1.json"
            --robot franka_single
            --problem demo
            --problem-index 1
            --dimension 7
            --seed 1
            --aorrtc-time 0.05
    )
    set_tests_properties(contract.aorrtc_result_v1 PROPERTIES
        LABELS "planning;schema;aorrtc"
        RESOURCE_LOCK patacon_gpu
        TIMEOUT 120
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME contract.repeated_result_v1
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_planning_regression.py"
            --planner "$<TARGET_FILE:single_mbm>"
            --schema "${PATACON_RESULT_SCHEMA}"
            --repository "${CMAKE_CURRENT_SOURCE_DIR}"
            --output
                "${PATACON_TEST_RESULT_DIRECTORY}/repeated_result_v1.json"
            --robot franka_single
            --problem demo
            --problem-index 1
            --dimension 7
            --seed 1
            --runs 2
    )
    set_tests_properties(contract.repeated_result_v1 PROPERTIES
        LABELS "planning;schema;repeated"
        RESOURCE_LOCK patacon_gpu
        TIMEOUT 120
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME contract.problem_set_result_v1
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_problem_set_regression.py"
            --evaluator "$<TARGET_FILE:evaluate_mbm>"
            --schema "${PATACON_PROBLEM_SET_SCHEMA}"
            --planner-schema "${PATACON_RESULT_SCHEMA}"
            --repository "${CMAKE_CURRENT_SOURCE_DIR}"
            --output
                "${PATACON_TEST_RESULT_DIRECTORY}/problem_set_result_v1.json"
            --robot franka_single
    )
    set_tests_properties(contract.problem_set_result_v1 PROPERTIES
        LABELS "planning;schema;problem-set"
        RESOURCE_LOCK patacon_gpu
        TIMEOUT 120
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME validation.problem_config
        COMMAND "$<TARGET_FILE:validate_problem_config>"
    )
    set_tests_properties(validation.problem_config PROPERTIES
        LABELS "validation;config"
        TIMEOUT 30
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    function(add_patacon_cuda_validation test_name target_name)
        add_test(
            NAME "validation.${test_name}"
            COMMAND "$<TARGET_FILE:${target_name}>" ${ARGN}
        )
        set_tests_properties("validation.${test_name}" PROPERTIES
            LABELS "validation;${test_name}"
            RESOURCE_LOCK patacon_gpu
            TIMEOUT 120
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endfunction()

    add_patacon_cuda_validation(
        franka_single.constraints validate_franka_constraints
    )
    add_patacon_cuda_validation(
        franka_single.collision validate_franka_collision franka_single
    )
    add_patacon_cuda_validation(
        franka_dual.collision validate_franka_collision franka
    )
    add_patacon_cuda_validation(
        g1.constraints validate_g1_constraints
    )
    add_patacon_cuda_validation(
        g1.constraints_axis validate_g1_constraints --axis
    )
    add_patacon_cuda_validation(
        g1.collision validate_g1_collision
    )
    add_patacon_cuda_validation(
        g1.collision_axis validate_g1_collision
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/g1_problems.json" --axis
    )
    add_patacon_cuda_validation(
        igris_c.kinematics validate_igris_c_kinematics
    )
    add_patacon_cuda_validation(
        igris_c.constraints validate_igris_c_constraints
    )
    add_patacon_cuda_validation(
        igris_c.collision validate_igris_c_collision
    )

    # This executable already fails on the current pre-refactor code
    # (motion segments: 1/3). Keep it visible without making a known failure
    # look like a passing regression test.
    add_patacon_cuda_validation(
        ffw_sg2.projection_known_failure validate_ffw_sg2_projection
    )
    set_tests_properties(
        validation.ffw_sg2.projection_known_failure
        PROPERTIES
            DISABLED TRUE
            LABELS "validation;ffw_sg2;known-failure"
    )
endif()

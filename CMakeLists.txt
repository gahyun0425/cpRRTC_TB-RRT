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
option(
    PATACON_CUDA_SEPARABLE_COMPILATION
    "Build the shared planner backend with relocatable device code"
    ON
)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

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

# Compile the initial PATACON search and the opt-in AORRTC optimization phase
# as one CUDA translation unit. The implementation is split into .cuh files
# for navigation without introducing cross-translation-unit device calls.
add_library(patacon_planners STATIC
    src/planning/PATACON.cu
)

set_target_properties(patacon_planners PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${PATACON_CUDA_SEPARABLE_COMPILATION}
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
    CUDA_SEPARABLE_COMPILATION ${PATACON_CUDA_SEPARABLE_COMPILATION}
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

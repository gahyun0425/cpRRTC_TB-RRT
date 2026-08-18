cmake_minimum_required(VERSION 3.16...3.22)

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

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)


# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo")  # enable line info in ncu ui


### motion benchmark script
add_executable(evaluate_mbm
    scripts/evaluate_mbm.cpp
    src/planning/pRRTC.cu
)

target_include_directories(evaluate_mbm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    #${CMAKE_CURRENT_SOURCE_DIR}/extern/vamp/src/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/nigh-src/src
    ${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/pdqsort-src
)


target_link_libraries(evaluate_mbm
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
)

target_compile_options(evaluate_mbm PRIVATE
    -Wno-c++11-narrowing
)

### single_mbm
add_executable(single_mbm
    scripts/single_mbm.cpp
    src/planning/AORRTC.cu
)

target_include_directories(single_mbm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    #${CMAKE_CURRENT_SOURCE_DIR}/extern/vamp/src/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/nigh-src/src
    ${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/pdqsort-src
)

target_link_libraries(single_mbm
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    CUDA::cudart
)

target_compile_options(single_mbm PRIVATE
    -Wno-c++11-narrowing
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

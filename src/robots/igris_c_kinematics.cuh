#pragma once

// Generated from resources/igris_c/igris_c_planning.urdf by
// resources/igris_c/generate_kinematics_header.py.
// Planning-model SHA-256: e0aa46ee1368e1bfa39b9c912597e4899273a901b9fa4b38cd166405e605feaf

#include <cuda_runtime.h>

namespace ppln::collision {

constexpr int IGRIS_C_DIM = 35;
constexpr int IGRIS_C_LINK_COUNT = 69;
constexpr int IGRIS_C_TREE_JOINT_COUNT = 68;
constexpr int IGRIS_C_TASK_FRAME_COUNT = 4;
constexpr int IGRIS_C_L_SOLE_FRAME = 0;
constexpr int IGRIS_C_R_SOLE_FRAME = 1;
constexpr int IGRIS_C_L_GRASP_FRAME = 2;
constexpr int IGRIS_C_R_GRASP_FRAME = 3;
constexpr float IGRIS_C_TOTAL_MASS_KG = 58.311999999999998f;

struct IgrisCTransform {
    float rotation[9];
    float translation[3];
};

struct IgrisCJointSpec {
    int parent_link;
    int child_link;
    int configuration_index;
    int type;  // 0 fixed, 1 prismatic, 2 revolute
    float origin_translation[3];
    float origin_rotation[9];
    float axis[3];
};

__device__ __constant__ IgrisCJointSpec
igris_c_tree_joints[IGRIS_C_TREE_JOINT_COUNT] = {
    {0, 1, 0, 1, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    {1, 2, 1, 1, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {2, 3, 2, 1, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {3, 4, 3, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    {4, 5, 4, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {5, 6, 5, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {6, 7, -1, 0, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}},
    {7, 8, -1, 0, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}},
    {8, 9, 6, 2, {-0.0115f, 0.089700000000000002f, -0.14000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {9, 10, 7, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    {10, 11, 8, 2, {0.002f, 0.0f, -0.075999999999999998f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {11, 12, 9, 2, {-0.055f, 0.0f, -0.26300000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {12, 13, 10, 2, {0.0050000000000000001f, 0.0f, -0.34000000000000002f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {13, 14, 11, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    {14, 15, -1, 0, {0.048000000000000001f, 0.0f, -0.070999999999999994f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}},
    {8, 16, 12, 2, {-0.0115f, -0.089700000000000002f, -0.14000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {16, 17, 13, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    {17, 18, 14, 2, {0.002f, 0.0f, -0.075999999999999998f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {18, 19, 15, 2, {-0.055f, 0.0f, -0.26300000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {19, 20, 16, 2, {0.0050000000000000001f, 0.0f, -0.34000000000000002f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {20, 21, 17, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    {21, 22, -1, 0, {0.048000000000000001f, 0.0f, -0.070999999999999994f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}},
    {8, 23, 18, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, -1.0f, 0.0f}},
    {23, 24, 19, 2, {0.0f, 0.0f, 0.040000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}},
    {24, 25, 20, 2, {-0.050000000000000003f, 0.0f, 0.025999999999999999f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}},
    {25, 26, 21, 2, {0.0f, 0.17100000000000001f, 0.26600000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {26, 27, 22, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    {27, 28, 23, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {28, 29, 24, 2, {0.01f, 0.0f, -0.19700000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {29, 30, 25, 2, {-0.01f, 0.0f, -0.050000000000000003f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {30, 31, 26, 2, {0.0f, -0.0050000000000000001f, -0.23949999999999999f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    {31, 32, 27, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {32, 33, -1, 0, {0.0f, 0.0109f, -0.024f}, {1.0f, 0.0f, 0.0f, 0.0f, -0.9999987317275395f, -0.0015926529164868282f, 0.0f, 0.0015926529164868282f, -0.9999987317275395f}, {0.0f, 0.0f, 0.0f}},
    {33, 34, -1, 0, {0.025999999999999999f, 0.0132f, 0.029899999999999999f}, {0.79608379854905575f, -0.60518640573603977f, -6.9388939039072284e-17f, 0.6028835307682785f, 0.79305451462175192f, 0.08715494917921647f, -0.052744990435877166f, -0.069382643004940597f, 0.9961947675196694f}, {0.0f, 0.0f, 0.0f}},
    {34, 35, -1, 0, {0.023f, 0.0035999999999999999f, 0.014999999999999999f}, {-4.2520510328806802e-07f, 0.78274537181963388f, 0.62234209474754498f, -0.99999999999990963f, -3.3282732667285469e-07f, -2.6462303467766642e-07f, 0.0f, -0.62234209474760127f, 0.78274537181970461f}, {0.0f, 0.0f, 0.0f}},
    {35, 36, -1, 0, {0.0f, -0.001f, 0.040000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.90044710235267689f, -0.43496553411123023f, 0.0f, 0.43496553411123023f, 0.90044710235267689f}, {0.0f, 0.0f, 0.0f}},
    {33, 37, -1, 0, {0.030499999999999999f, -0.002f, 0.10100000000000001f}, {-1.0f, 3.4970820617141042e-10f, -2.1440792433333394e-10f, -4.1020310515713352e-10f, -0.85252452205950568f, 0.52268722893065922f, 0.0f, 0.52268722893065922f, 0.85252452205950568f}, {0.0f, 0.0f, 0.0f}},
    {37, 38, -1, 0, {0.0f, -0.001f, 0.040000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.93937271284737889f, -0.34289780745545134f, 0.0f, 0.34289780745545134f, 0.93937271284737889f}, {0.0f, 0.0f, 0.0f}},
    {33, 39, -1, 0, {0.0085000000000000006f, -0.002f, 0.10100000000000001f}, {-1.0f, 3.4970820617141042e-10f, -2.1440792433333394e-10f, -4.1020310515713352e-10f, -0.85252452205950568f, 0.52268722893065922f, 0.0f, 0.52268722893065922f, 0.85252452205950568f}, {0.0f, 0.0f, 0.0f}},
    {39, 40, -1, 0, {0.0f, -0.001f, 0.040000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.93937271284737889f, -0.34289780745545134f, 0.0f, 0.34289780745545134f, 0.93937271284737889f}, {0.0f, 0.0f, 0.0f}},
    {33, 41, -1, 0, {-0.0135f, -0.002f, 0.10100000000000001f}, {-1.0f, 3.4970820617141042e-10f, -2.1440792433333394e-10f, -4.1020310515713352e-10f, -0.85252452205950568f, 0.52268722893065922f, 0.0f, 0.52268722893065922f, 0.85252452205950568f}, {0.0f, 0.0f, 0.0f}},
    {41, 42, -1, 0, {0.0f, -0.001f, 0.040000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.93937271284737889f, -0.34289780745545134f, 0.0f, 0.34289780745545134f, 0.93937271284737889f}, {0.0f, 0.0f, 0.0f}},
    {33, 43, -1, 0, {-0.035499999999999997f, -0.002f, 0.10100000000000001f}, {-1.0f, 3.4970820617141042e-10f, -2.1440792433333394e-10f, -4.1020310515713352e-10f, -0.85252452205950568f, 0.52268722893065922f, 0.0f, 0.52268722893065922f, 0.85252452205950568f}, {0.0f, 0.0f, 0.0f}},
    {43, 44, -1, 0, {0.0f, -0.001f, 0.040000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.93937271284737889f, -0.34289780745545134f, 0.0f, 0.34289780745545134f, 0.93937271284737889f}, {0.0f, 0.0f, 0.0f}},
    {32, 45, -1, 0, {-0.025025641714385472f, 0.0f, -0.124f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}},
    {45, 46, -1, 0, {0.0f, -0.17000000000000001f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}},
    {25, 47, 28, 2, {0.0f, -0.17100000000000001f, 0.26600000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {47, 48, 29, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    {48, 49, 30, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {49, 50, 31, 2, {0.01f, 0.0f, -0.19700000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {50, 51, 32, 2, {-0.01f, 0.0f, -0.050000000000000003f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
    {51, 52, 33, 2, {0.0f, 0.0050000000000000001f, -0.23949999999999999f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
    {52, 53, 34, 2, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
    {53, 54, -1, 0, {0.0f, -0.0109f, -0.024f}, {1.0f, 0.0f, 0.0f, 0.0f, -0.9999987317275395f, -0.0015926529164868282f, 0.0f, 0.0015926529164868282f, -0.9999987317275395f}, {0.0f, 0.0f, 0.0f}},
    {54, 55, -1, 0, {0.025999999999999999f, -0.0132f, 0.029899999999999999f}, {0.79608379854905575f, 0.60518640573603977f, -6.9388939039072284e-17f, -0.6028835307682785f, 0.79305451462175192f, -0.08715494917921647f, -0.052744990435877166f, 0.069382643004940597f, 0.9961947675196694f}, {0.0f, 0.0f, 0.0f}},
    {55, 56, -1, 0, {0.023f, -0.0035999999999999999f, 0.014999999999999999f}, {-4.2520510328806802e-07f, 0.78274537181963388f, 0.62234209474754498f, -0.99999999999990963f, -3.3282732667285469e-07f, -2.6462303467766642e-07f, 0.0f, -0.62234209474760127f, 0.78274537181970461f}, {0.0f, 0.0f, 0.0f}},
    {56, 57, -1, 0, {0.0f, -0.001f, 0.040000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.90044710235267689f, -0.43496553411123023f, 0.0f, 0.43496553411123023f, 0.90044710235267689f}, {0.0f, 0.0f, 0.0f}},
    {54, 58, -1, 0, {0.030499999999999999f, 0.002f, 0.10100000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.85252452205950568f, -0.52268722893065922f, 0.0f, 0.52268722893065922f, 0.85252452205950568f}, {0.0f, 0.0f, 0.0f}},
    {58, 59, -1, 0, {0.0f, -0.001f, 0.040000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.93937271284737889f, -0.34289780745545134f, 0.0f, 0.34289780745545134f, 0.93937271284737889f}, {0.0f, 0.0f, 0.0f}},
    {54, 60, -1, 0, {0.0085000000000000006f, 0.002f, 0.10100000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.85252452205950568f, -0.52268722893065922f, 0.0f, 0.52268722893065922f, 0.85252452205950568f}, {0.0f, 0.0f, 0.0f}},
    {60, 61, -1, 0, {0.0f, -0.001f, 0.040000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.93937271284737889f, -0.34289780745545134f, 0.0f, 0.34289780745545134f, 0.93937271284737889f}, {0.0f, 0.0f, 0.0f}},
    {54, 62, -1, 0, {-0.0135f, 0.002f, 0.10100000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.85252452205950568f, -0.52268722893065922f, 0.0f, 0.52268722893065922f, 0.85252452205950568f}, {0.0f, 0.0f, 0.0f}},
    {62, 63, -1, 0, {0.0f, -0.001f, 0.040000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.93937271284737889f, -0.34289780745545134f, 0.0f, 0.34289780745545134f, 0.93937271284737889f}, {0.0f, 0.0f, 0.0f}},
    {54, 64, -1, 0, {-0.035499999999999997f, 0.002f, 0.10100000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.85252452205950568f, -0.52268722893065922f, 0.0f, 0.52268722893065922f, 0.85252452205950568f}, {0.0f, 0.0f, 0.0f}},
    {64, 65, -1, 0, {0.0f, -0.001f, 0.040000000000000001f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.93937271284737889f, -0.34289780745545134f, 0.0f, 0.34289780745545134f, 0.93937271284737889f}, {0.0f, 0.0f, 0.0f}},
    {53, 66, -1, 0, {-0.025025641714385472f, 0.0f, -0.124f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}},
    {25, 67, -1, 0, {0.0f, 0.0f, 0.35699999999999998f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}},
    {67, 68, -1, 0, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}},
};

__device__ __constant__ int
igris_c_configuration_types[IGRIS_C_DIM] = {
    1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
};

__device__ __constant__ unsigned long long
igris_c_link_ancestor_masks[IGRIS_C_LINK_COUNT] = {
    0x000000000ULL,
    0x000000001ULL,
    0x000000003ULL,
    0x000000007ULL,
    0x00000000fULL,
    0x00000001fULL,
    0x00000003fULL,
    0x00000003fULL,
    0x00000003fULL,
    0x00000007fULL,
    0x0000000ffULL,
    0x0000001ffULL,
    0x0000003ffULL,
    0x0000007ffULL,
    0x000000fffULL,
    0x000000fffULL,
    0x00000103fULL,
    0x00000303fULL,
    0x00000703fULL,
    0x00000f03fULL,
    0x00001f03fULL,
    0x00003f03fULL,
    0x00003f03fULL,
    0x00004003fULL,
    0x0000c003fULL,
    0x0001c003fULL,
    0x0003c003fULL,
    0x0007c003fULL,
    0x000fc003fULL,
    0x001fc003fULL,
    0x003fc003fULL,
    0x007fc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x00ffc003fULL,
    0x0101c003fULL,
    0x0301c003fULL,
    0x0701c003fULL,
    0x0f01c003fULL,
    0x1f01c003fULL,
    0x3f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x7f01c003fULL,
    0x0001c003fULL,
    0x0001c003fULL,
};

__device__ __constant__ float
igris_c_link_masses[IGRIS_C_LINK_COUNT] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    8.1240000000000006f, 1.8939999999999999f, 0.57299999999999995f, 4.8129999999999997f, 3.2120000000000002f, 0.084000000000000005f, 0.57899999999999996f, 0.0f,
    1.8939999999999999f, 0.57299999999999995f, 4.8129999999999997f, 3.2120000000000002f, 0.084000000000000005f, 0.57899999999999996f, 0.0f, 0.13600000000000001f,
    0.20799999999999999f, 12.974f, 1.105f, 0.253f, 2.1440000000000001f, 0.13700000000000001f, 2.4279999999999999f, 0.012f,
    0.032000000000000001f, 0.20000000000000001f, 0.01f, 0.0080000000000000002f, 0.0060000000000000001f, 0.0080000000000000002f, 0.0060000000000000001f, 0.0080000000000000002f,
    0.0060000000000000001f, 0.0080000000000000002f, 0.0060000000000000001f, 0.0080000000000000002f, 0.0060000000000000001f, 0.0f, 0.0f, 1.105f,
    0.253f, 2.1440000000000001f, 0.13700000000000001f, 2.4279999999999999f, 0.012f, 0.032000000000000001f, 0.20000000000000001f, 0.01f,
    0.0080000000000000002f, 0.0060000000000000001f, 0.0080000000000000002f, 0.0060000000000000001f, 0.0080000000000000002f, 0.0060000000000000001f, 0.0080000000000000002f, 0.0060000000000000001f,
    0.0080000000000000002f, 0.0060000000000000001f, 0.0f, 0.31f, 1.468f,
};

__device__ __constant__ float
igris_c_link_com_local[IGRIS_C_LINK_COUNT][3] = {
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {-0.049599999999999998f, 0.0f, -0.051630000000000002f},
    {0.0097300000000000008f, -0.0010200000000000001f, 2.0000000000000002e-05f},
    {-0.014970000000000001f, 0.0057000000000000002f, -0.035639999999999998f},
    {-0.01286f, -0.0038800000000000002f, -0.10563f},
    {0.01044f, 0.00096000000000000002f, -0.1232f},
    {0.0f, 0.0f, 0.0f},
    {0.021669999999999998f, 0.0f, -0.051929999999999997f},
    {0.0f, 0.0f, 0.0f},
    {0.0097300000000000008f, 0.0010200000000000001f, 2.0000000000000002e-05f},
    {-0.014970000000000001f, -0.0057000000000000002f, -0.035639999999999998f},
    {-0.01286f, 0.0038800000000000002f, -0.10563f},
    {0.01044f, -0.00096000000000000002f, -0.1232f},
    {0.0f, 0.0f, 0.0f},
    {0.021669999999999998f, 0.0f, -0.051929999999999997f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, -0.0082900000000000005f},
    {0.02597f, 0.0f, -0.0092700000000000005f},
    {0.00175f, 0.0011199999999999999f, 0.15154000000000001f},
    {-0.0045799999999999999f, 0.00011f, 0.044990000000000002f},
    {0.01754f, 0.025139999999999999f, 0.0f},
    {-0.0016900000000000001f, -0.0063899999999999998f, -0.10638f},
    {0.0054400000000000004f, 0.0038f, -0.038379999999999997f},
    {0.010240000000000001f, -0.0036900000000000001f, -0.11953f},
    {0.0f, 0.0060400000000000002f, 0.00031f},
    {0.0f, 0.012030000000000001f, -0.02699f},
    {0.0091000000000000004f, 0.0037100000000000002f, 0.078390000000000001f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {-0.0045799999999999999f, -0.00011f, 0.044990000000000002f},
    {0.01754f, -0.025139999999999999f, 0.0f},
    {-0.0016900000000000001f, 0.0063899999999999998f, -0.10638f},
    {0.0054400000000000004f, -0.0038f, -0.038379999999999997f},
    {0.010240000000000001f, 0.0036900000000000001f, -0.11953f},
    {0.0f, -0.0060400000000000002f, 0.00031f},
    {0.0f, -0.012030000000000001f, -0.02699f},
    {0.0091000000000000004f, -0.0037100000000000002f, 0.078390000000000001f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f},
    {-5.0000000000000002e-05f, -0.0012999999999999999f, -0.00264f},
    {0.028500000000000001f, 0.0028999999999999998f, 0.12862000000000001f},
};

__device__ __constant__ int
igris_c_task_link_indices[IGRIS_C_TASK_FRAME_COUNT] = {
    15, 22, 45, 66
};

__device__ __forceinline__ void igris_c_identity3(float matrix[9]) {
    for (int index = 0; index < 9; ++index) {
        matrix[index] = index % 4 == 0 ? 1.0f : 0.0f;
    }
}

__device__ __forceinline__ void igris_c_multiply3(
    const float left[9],
    const float right[9],
    float result[9]
) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result[row * 3 + column] =
                left[row * 3] * right[column] +
                left[row * 3 + 1] * right[3 + column] +
                left[row * 3 + 2] * right[6 + column];
        }
    }
}

__device__ __forceinline__ void igris_c_multiply_at_b(
    const float left[9],
    const float right[9],
    float result[9]
) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result[row * 3 + column] =
                left[row] * right[column] +
                left[3 + row] * right[3 + column] +
                left[6 + row] * right[6 + column];
        }
    }
}

__device__ __forceinline__ void igris_c_rotate(
    const float rotation[9],
    const float vector[3],
    float result[3]
) {
    for (int row = 0; row < 3; ++row) {
        result[row] =
            rotation[row * 3] * vector[0] +
            rotation[row * 3 + 1] * vector[1] +
            rotation[row * 3 + 2] * vector[2];
    }
}

__device__ __forceinline__ void igris_c_rotate_transpose(
    const float rotation[9],
    const float vector[3],
    float result[3]
) {
    for (int column = 0; column < 3; ++column) {
        result[column] =
            rotation[column] * vector[0] +
            rotation[3 + column] * vector[1] +
            rotation[6 + column] * vector[2];
    }
}

__device__ __forceinline__ void igris_c_axis_angle_rotation(
    const float axis[3],
    float angle,
    float rotation[9]
) {
    const float x = axis[0];
    const float y = axis[1];
    const float z = axis[2];
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float one_minus_cosine = 1.0f - cosine;
    rotation[0] = cosine + x * x * one_minus_cosine;
    rotation[1] = x * y * one_minus_cosine - z * sine;
    rotation[2] = x * z * one_minus_cosine + y * sine;
    rotation[3] = y * x * one_minus_cosine + z * sine;
    rotation[4] = cosine + y * y * one_minus_cosine;
    rotation[5] = y * z * one_minus_cosine - x * sine;
    rotation[6] = z * x * one_minus_cosine - y * sine;
    rotation[7] = z * y * one_minus_cosine + x * sine;
    rotation[8] = cosine + z * z * one_minus_cosine;
}

__device__ __forceinline__ void igris_c_cross(
    const float left[3],
    const float right[3],
    float result[3]
) {
    result[0] = left[1] * right[2] - left[2] * right[1];
    result[1] = left[2] * right[0] - left[0] * right[2];
    result[2] = left[0] * right[1] - left[1] * right[0];
}

// Computes every link pose and each active joint's world origin/axis once.
// The generated joint table is topologically ordered from the world link.
__device__ __noinline__ void igris_c_forward_model(
    const float q[IGRIS_C_DIM],
    IgrisCTransform link_poses[IGRIS_C_LINK_COUNT],
    float joint_origins[IGRIS_C_DIM][3],
    float joint_axes[IGRIS_C_DIM][3]
) {
    igris_c_identity3(link_poses[0].rotation);
    for (int component = 0; component < 3; ++component) {
        link_poses[0].translation[component] = 0.0f;
    }

    for (int joint_index = 0;
         joint_index < IGRIS_C_TREE_JOINT_COUNT;
         ++joint_index) {
        const IgrisCJointSpec &joint = igris_c_tree_joints[joint_index];
        const IgrisCTransform &parent = link_poses[joint.parent_link];
        IgrisCTransform &child = link_poses[joint.child_link];

        float translated_origin[3];
        igris_c_rotate(
            parent.rotation,
            joint.origin_translation,
            translated_origin
        );
        for (int component = 0; component < 3; ++component) {
            child.translation[component] =
                parent.translation[component] + translated_origin[component];
        }

        float origin_rotation[9];
        igris_c_multiply3(
            parent.rotation,
            joint.origin_rotation,
            origin_rotation
        );

        float axis_world[3];
        igris_c_rotate(origin_rotation, joint.axis, axis_world);
        if (joint.configuration_index >= 0) {
            for (int component = 0; component < 3; ++component) {
                joint_origins[joint.configuration_index][component] =
                    child.translation[component];
                joint_axes[joint.configuration_index][component] =
                    axis_world[component];
            }
        }

        if (joint.type == 1) {
            const float displacement = q[joint.configuration_index];
            for (int component = 0; component < 3; ++component) {
                child.translation[component] +=
                    axis_world[component] * displacement;
            }
            for (int index = 0; index < 9; ++index) {
                child.rotation[index] = origin_rotation[index];
            }
        } else if (joint.type == 2) {
            float joint_rotation[9];
            igris_c_axis_angle_rotation(
                joint.axis,
                q[joint.configuration_index],
                joint_rotation
            );
            igris_c_multiply3(
                origin_rotation,
                joint_rotation,
                child.rotation
            );
        } else {
            for (int index = 0; index < 9; ++index) {
                child.rotation[index] = origin_rotation[index];
            }
        }
    }
}

__device__ __forceinline__ void igris_c_frame_kinematics_from_model(
    int link_index,
    const IgrisCTransform link_poses[IGRIS_C_LINK_COUNT],
    const float joint_origins[IGRIS_C_DIM][3],
    const float joint_axes[IGRIS_C_DIM][3],
    IgrisCTransform &pose,
    float position_jacobian[3 * IGRIS_C_DIM],
    float angular_jacobian[3 * IGRIS_C_DIM]
) {
    for (int index = 0; index < 9; ++index) {
        pose.rotation[index] = link_poses[link_index].rotation[index];
    }
    for (int component = 0; component < 3; ++component) {
        pose.translation[component] = link_poses[link_index].translation[component];
    }
    for (int index = 0; index < 3 * IGRIS_C_DIM; ++index) {
        position_jacobian[index] = 0.0f;
        angular_jacobian[index] = 0.0f;
    }

    const unsigned long long ancestors =
        igris_c_link_ancestor_masks[link_index];
    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
        if ((ancestors & (1ULL << joint)) == 0ULL) {
            continue;
        }
        if (igris_c_configuration_types[joint] == 1) {
            for (int component = 0; component < 3; ++component) {
                position_jacobian[component * IGRIS_C_DIM + joint] =
                    joint_axes[joint][component];
            }
        } else {
            float displacement[3];
            for (int component = 0; component < 3; ++component) {
                displacement[component] = pose.translation[component] -
                    joint_origins[joint][component];
            }
            float linear[3];
            igris_c_cross(joint_axes[joint], displacement, linear);
            for (int component = 0; component < 3; ++component) {
                position_jacobian[component * IGRIS_C_DIM + joint] =
                    linear[component];
                angular_jacobian[component * IGRIS_C_DIM + joint] =
                    joint_axes[joint][component];
            }
        }
    }
}

__device__ __noinline__ void igris_c_task_frame_kinematics(
    const float q[IGRIS_C_DIM],
    int task_frame,
    IgrisCTransform &pose,
    float position_jacobian[3 * IGRIS_C_DIM],
    float angular_jacobian[3 * IGRIS_C_DIM]
) {
    IgrisCTransform link_poses[IGRIS_C_LINK_COUNT];
    float joint_origins[IGRIS_C_DIM][3];
    float joint_axes[IGRIS_C_DIM][3];
    igris_c_forward_model(q, link_poses, joint_origins, joint_axes);
    igris_c_frame_kinematics_from_model(
        igris_c_task_link_indices[task_frame],
        link_poses,
        joint_origins,
        joint_axes,
        pose,
        position_jacobian,
        angular_jacobian
    );
}

__device__ __forceinline__ void igris_c_center_of_mass_from_model(
    const IgrisCTransform link_poses[IGRIS_C_LINK_COUNT],
    const float joint_origins[IGRIS_C_DIM][3],
    const float joint_axes[IGRIS_C_DIM][3],
    float center_of_mass[3],
    float jacobian[3 * IGRIS_C_DIM]
) {
    for (int component = 0; component < 3; ++component) {
        center_of_mass[component] = 0.0f;
    }
    for (int index = 0; index < 3 * IGRIS_C_DIM; ++index) {
        jacobian[index] = 0.0f;
    }

    for (int link = 0; link < IGRIS_C_LINK_COUNT; ++link) {
        const float mass = igris_c_link_masses[link];
        if (mass <= 0.0f) {
            continue;
        }
        float rotated_local_com[3];
        igris_c_rotate(
            link_poses[link].rotation,
            igris_c_link_com_local[link],
            rotated_local_com
        );
        float world_com[3];
        for (int component = 0; component < 3; ++component) {
            world_com[component] =
                link_poses[link].translation[component] +
                rotated_local_com[component];
            center_of_mass[component] += mass * world_com[component];
        }

        const unsigned long long ancestors =
            igris_c_link_ancestor_masks[link];
        for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
            if ((ancestors & (1ULL << joint)) == 0ULL) {
                continue;
            }
            float linear[3];
            if (igris_c_configuration_types[joint] == 1) {
                for (int component = 0; component < 3; ++component) {
                    linear[component] = joint_axes[joint][component];
                }
            } else {
                float displacement[3];
                for (int component = 0; component < 3; ++component) {
                    displacement[component] = world_com[component] -
                        joint_origins[joint][component];
                }
                igris_c_cross(joint_axes[joint], displacement, linear);
            }
            for (int component = 0; component < 3; ++component) {
                jacobian[component * IGRIS_C_DIM + joint] +=
                    mass * linear[component];
            }
        }
    }

    const float inverse_mass = 1.0f / IGRIS_C_TOTAL_MASS_KG;
    for (int component = 0; component < 3; ++component) {
        center_of_mass[component] *= inverse_mass;
    }
    for (int index = 0; index < 3 * IGRIS_C_DIM; ++index) {
        jacobian[index] *= inverse_mass;
    }
}

__device__ __noinline__ void igris_c_center_of_mass_kinematics(
    const float q[IGRIS_C_DIM],
    float center_of_mass[3],
    float jacobian[3 * IGRIS_C_DIM]
) {
    IgrisCTransform link_poses[IGRIS_C_LINK_COUNT];
    float joint_origins[IGRIS_C_DIM][3];
    float joint_axes[IGRIS_C_DIM][3];
    igris_c_forward_model(q, link_poses, joint_origins, joint_axes);
    igris_c_center_of_mass_from_model(
        link_poses,
        joint_origins,
        joint_axes,
        center_of_mass,
        jacobian
    );
}

// Returns T_l_grasp^-1 * T_r_grasp. The angular Jacobian is the relative
// spatial angular velocity expressed in l_grasp coordinates.
__device__ __forceinline__ void igris_c_bimanual_from_model(
    const IgrisCTransform link_poses[IGRIS_C_LINK_COUNT],
    const float joint_origins[IGRIS_C_DIM][3],
    const float joint_axes[IGRIS_C_DIM][3],
    IgrisCTransform &relative_pose,
    float position_jacobian[3 * IGRIS_C_DIM],
    float angular_jacobian[3 * IGRIS_C_DIM]
) {
    IgrisCTransform left_pose;
    IgrisCTransform right_pose;
    float left_position_jacobian[3 * IGRIS_C_DIM];
    float right_position_jacobian[3 * IGRIS_C_DIM];
    float left_angular_jacobian[3 * IGRIS_C_DIM];
    float right_angular_jacobian[3 * IGRIS_C_DIM];
    igris_c_frame_kinematics_from_model(
        igris_c_task_link_indices[IGRIS_C_L_GRASP_FRAME],
        link_poses,
        joint_origins,
        joint_axes,
        left_pose,
        left_position_jacobian,
        left_angular_jacobian
    );
    igris_c_frame_kinematics_from_model(
        igris_c_task_link_indices[IGRIS_C_R_GRASP_FRAME],
        link_poses,
        joint_origins,
        joint_axes,
        right_pose,
        right_position_jacobian,
        right_angular_jacobian
    );

    igris_c_multiply_at_b(
        left_pose.rotation,
        right_pose.rotation,
        relative_pose.rotation
    );
    float world_displacement[3];
    for (int component = 0; component < 3; ++component) {
        world_displacement[component] =
            right_pose.translation[component] -
            left_pose.translation[component];
    }
    igris_c_rotate_transpose(
        left_pose.rotation,
        world_displacement,
        relative_pose.translation
    );

    for (int joint = 0; joint < IGRIS_C_DIM; ++joint) {
        float left_angular[3];
        float relative_world_linear[3];
        float relative_world_angular[3];
        for (int component = 0; component < 3; ++component) {
            left_angular[component] =
                left_angular_jacobian[component * IGRIS_C_DIM + joint];
            relative_world_linear[component] =
                right_position_jacobian[component * IGRIS_C_DIM + joint] -
                left_position_jacobian[component * IGRIS_C_DIM + joint];
            relative_world_angular[component] =
                right_angular_jacobian[component * IGRIS_C_DIM + joint] -
                left_angular_jacobian[component * IGRIS_C_DIM + joint];
        }
        float rotating_frame_term[3];
        igris_c_cross(left_angular, world_displacement, rotating_frame_term);
        for (int component = 0; component < 3; ++component) {
            relative_world_linear[component] -= rotating_frame_term[component];
        }
        float relative_local_linear[3];
        float relative_local_angular[3];
        igris_c_rotate_transpose(
            left_pose.rotation,
            relative_world_linear,
            relative_local_linear
        );
        igris_c_rotate_transpose(
            left_pose.rotation,
            relative_world_angular,
            relative_local_angular
        );
        for (int component = 0; component < 3; ++component) {
            position_jacobian[component * IGRIS_C_DIM + joint] =
                relative_local_linear[component];
            angular_jacobian[component * IGRIS_C_DIM + joint] =
                relative_local_angular[component];
        }
    }
}

__device__ __noinline__ void igris_c_bimanual_kinematics(
    const float q[IGRIS_C_DIM],
    IgrisCTransform &relative_pose,
    float position_jacobian[3 * IGRIS_C_DIM],
    float angular_jacobian[3 * IGRIS_C_DIM]
) {
    IgrisCTransform link_poses[IGRIS_C_LINK_COUNT];
    float joint_origins[IGRIS_C_DIM][3];
    float joint_axes[IGRIS_C_DIM][3];
    igris_c_forward_model(q, link_poses, joint_origins, joint_axes);
    igris_c_bimanual_from_model(
        link_poses,
        joint_origins,
        joint_axes,
        relative_pose,
        position_jacobian,
        angular_jacobian
    );
}

}  // namespace ppln::collision

# IGRIS-C planning model, constraints, and collision backend

이 디렉터리는 공개 IGRIS-C description을 현재 플래너에 연결하는 재현 가능한
**35-DoF planning model**을 관리한다. 원본 `igris_c_description_public`은
수정하지 않는다.

## 생성과 검증

```bash
python3 resources/igris_c/prepare_planning_urdf.py
python3 resources/igris_c/prepare_planning_urdf.py --check
python3 resources/igris_c/generate_kinematics_header.py
python3 resources/igris_c/generate_kinematics_header.py --check
python3 resources/igris_c/generate_kinematics_reference.py
python3 resources/igris_c/generate_kinematics_reference.py --check
python3 resources/igris_c/generate_shelf_problem.py --check
python3 resources/igris_c/generate_collision_header.py
python3 resources/igris_c/generate_collision_header.py --check
cmake --build build --target validate_igris_c_kinematics \
  validate_igris_c_constraints validate_igris_c_collision
./build/validate_igris_c_kinematics
./build/validate_igris_c_constraints
./build/validate_igris_c_collision
```

첫 생성기는 xacro를 전개해 planning URDF, 모델 메타데이터, constraint 계약을
만든다. 각 `--check`는 소스에서 결과를 다시 계산하여 커밋된 산출물과 byte
단위로 비교한다.

## 손 모델과 35차원 configuration

소스 xacro mapping은 `base_type=pelvis`, `parallel=false`,
`end_effector=hand`다. 따라서 양쪽 관절형 손의 geometry, inertial, collision을
포함한다. 다만 손가락 22축은 planning 변수에서 제외하고 다음 box-grasp
자세를 각 fixed joint origin에 bake한다.

- 양손 엄지: proximal `+0.65/-0.65`, middle `0.55`, distal `0.45` rad
- 양손 검지·중지·약지·소지: middle `0.55`, distal `0.35` rad
- 목 yaw/pitch: `0` rad에서 fixed

따라서 생성 모델은 69 links, 68 joints, 56 inertials, 59 collision
geometries를 가지지만 movable configuration은 계속 35차원이다. 추가된 하나는
운반 박스용 massless collision-only link이며, 고정된 손을 포함한 로봇 질량은
계속 `58.312 kg`이다.

| index | block | joints |
|---:|---|---|
| 0–5 | floating base | `world_to_x`, `x_to_y`, `y_to_z`, `z_to_roll`, `roll_to_pitch`, `pitch_to_yaw` |
| 6–11 | left leg | `l_hip_pitch`, `l_hip_roll`, `l_hip_yaw`, `l_knee_pitch`, `l_ankle_pitch`, `l_ankle_roll` |
| 12–17 | right leg | `r_hip_pitch`, `r_hip_roll`, `r_hip_yaw`, `r_knee_pitch`, `r_ankle_pitch`, `r_ankle_roll` |
| 18–20 | waist | `waist_pitch`, `waist_roll`, `waist_yaw` |
| 21–27 | left arm | `l_shoulder_pitch`, `l_shoulder_roll`, `l_shoulder_yaw`, `l_elbow_pitch`, `l_wrist_yaw`, `l_wrist_roll`, `l_wrist_pitch` |
| 28–34 | right arm | `r_shoulder_pitch`, `r_shoulder_roll`, `r_shoulder_yaw`, `r_elbow_pitch`, `r_wrist_yaw`, `r_wrist_roll`, `r_wrist_pitch` |

가상 베이스는 기존 G1과 같은 `Tx * Ty * Tz * Rx * Ry * Rz` 직렬 체인이다.
이 여섯 값은 모터 명령이 아니라 world 좌표계의 generalized coordinate다.
실기에서는 추정한 base pose와 encoder 관절값으로 현재 configuration을 만들고,
planning 결과는 전신 균형 제어기와 관절 trajectory로 실행해야 한다.

## Task frame과 constraint

- `l_sole`, `r_sole`: 각 발 collision box의 바닥 중심 `(0.048, 0, -0.071) m`
- `l_grasp`, `r_grasp`: 각 wrist connector 아래 `(-0.025025642, 0, -0.124) m`

관절형 손은 실제 모델에 포함된다. grasp task frame은 기존 wrist-mount
위치에서 손 진행 방향(local `-z`)으로 `0.10 m`, local `-x` 방향으로
`0.025025642 m` 이동하여 손바닥 안에 배치한다. 후자의 offset은 시작 grasp
중심을 world `x=0`에 맞춘 값이다. 양손에 동일한 local offset을 사용하므로
34 cm 상대 pose는 유지된다.

Bimanual equality는 다음 6차원 상대 pose를 고정한다.

```text
inverse(T_world_l_grasp) * T_world_r_grasp
rotation    = identity
translation = [0, -0.34, 0] m
```

Bimanual axis equality는 왼쪽 grasp frame의 local `+X`축을 world `+Z`축에
평행하게 유지한다. 독립 residual은 해당 축의 world `x`, `y` 성분 2개이며,
기존 bimanual 상대 pose equality를 통해 오른손에도 같은 조건이 적용된다.
따라서 손끝 방향(local `-Z`)은 경로 전체에서 지면과 수평을 유지한다.

Foot equality는 양쪽 sole의 시작 world pose를 각각 6차원으로 고정한다.
시작 자세의 target은 다음과 같다.

```text
l_sole: quaternion_wxyz=[1,0,0,0], xyz=[-0.0115,  0.1897, 0] m
r_sole: quaternion_wxyz=[1,0,0,0], xyz=[-0.0115, -0.1897, 0] m
```

CoM inequality는 양발 support polygon을 모든 외곽 경계에서 `0.05 m` 줄인
영역에 적용한다.

```text
raw:  x=[-0.1115, 0.0885], y=[-0.2247, 0.2247] m
5 cm: x=[-0.0615, 0.0385], y=[-0.1747, 0.1747] m
```

합성 CoM은 고정 손을 포함한 `58.312 kg` 로봇과 양 grasp 원점 중점의
`0.15 kg` point-mass 박스를 합친 `58.462 kg` 시스템이다. CoM이 polygon
안에 있으면 inequality의 2개 residual/Jacobian 행은 0이고, 밖에서는 가장
가까운 경계점으로 복귀시키는 XY 행만 활성화된다.

Tangent space는 foot 12행, bimanual 6행, bimanual axis 2행을 쌓은 `20×35`
equality Jacobian의 null space이며, 정상 rank에서 15차원이다. Projection은
foot, bimanual, bimanual axis, CoM을 하나의 `22×35` Jacobian으로 구성해
동시에 수행한다.
`igris_c_project_motion`은 planner가 전달하는 node projection 옵션에 따라
waypoint smoothness와 `granularity * projection_smoothness_threshold`를
사용한다.

## Shelf-lift problem

`scripts/igris_c_problems.json`의 `igris_c_shelf_lift`는 박스를 잡은 시작
자세에서 선반 높이의 목표 자세로 이동한다. 두 endpoint 모두 foot/bimanual
equality와 5 cm CoM margin을 만족한다.
선반 3개 층의 world-frame z 중심은 각각 `0.45`, `0.84`, `1.25 m`이다.
각 선반 판의 world-frame x 중심은 `0.65 m`, 뒷판은 `0.80 m`이다.

- 시작 grasp 중심: `(0.000000, 0.379410, 0.810000) m`
- 시작 grasp 방위: local +X가 world +Z를 향하는 bimanual-axis equality 유지
- 시작 torso world RPY: `(0, 0, 60) deg` (수치 오차 약 `2.7e-6 deg`)
- 시작 무릎 굽힘: 왼쪽 `78.83 deg`, 오른쪽 `70.69 deg`
- 시작 허리 RPY joint: `(-37.15, -15.49, -20.00) deg`
- 시작 팔꿈치 굽힘: 양쪽 `-45.00 deg`
- 목표 grasp 중심: `(0.500000, 0, 1.074974) m`
- 시작 합성 CoM: `(-0.058500250, -0.048470586, 0.718590288) m`
- 목표 합성 CoM: `(-0.012301934, -0.000381692, 0.848256161) m`

박스의 실제 크기는 높이 21 cm, 양팔 간격 방향 가로 34 cm, 세로
25 cm이다. grasp frame의 local +X가 world 위쪽을 향하므로 MuJoCo geom의
local `(x, y, z)` half extent는 `(0.105, 0.17, 0.125) m`이다. 질량은
`0.15 kg`이다. collision-only 박스는 `l_grasp` 기준 `(0, -0.17, 0) m`에
고정되며 12개의 fine sphere(2×3×2)로 보수적으로 근사된다. 이 link에는 inertial을 넣지 않고,
박스 질량은 기존처럼 CoM constraint에서 한 번만 반영한다.

## FK/Jacobian과 collision model

`src/robots/igris_c_kinematics.cuh`는 생성된 69-link tree로 task frame,
robot/payload CoM, bimanual relative pose와 각 `×35` analytic Jacobian을
계산한다. 독립 Pinocchio 중앙차분 기준은 `kinematics_reference.json`에 있다.

`src/robots/igris_c_collision.cuh`는 58개 robot collision mesh와 하나의
attached-payload box의 link-frame axis-aligned bounding box를 robot은 최대
10.5 cm, payload는 최대 12.5 cm cell로 나눈 뒤 각 cell을 둘러싸는 sphere로
보수적으로 근사한다. 결과는 다음과 같다.

- fine collision spheres: 190개(그중 payload 12개), inflation 2 mm
- approximate rigid-group spheres: 30개
- self-collision candidate pairs: 14,303개
- conservative link-pair broad-phase groups: 1,358개
- 제외: 같은 link, 같은 fixed body, tree distance 2 이하, 허용 seed/start/goal에서 고정적으로 겹치는 proxy pair
- approximate sphere가 environment 또는 허용 self link-pair와 겹칠 때만
  fine check를 요청하므로 false negative 없이 빈 공간 계산을 줄임

`validate_igris_c_collision`은 start/goal의 self collision, 빈 환경, 선반
환경을 검사하고, payload sphere에 의도적으로 겹친 obstacle이 실제로 차단되는지
GPU에서 검사한다.

## Planner 등록

기존 planner 알고리즘과 공통 실행 흐름은 유지하고 IGRIS용 타입 dispatch만
추가했다. `RobotCollisionTraits`, constraint parameter 전달, tangent/projection
hook, 35축 Halton 순서와 explicit template instantiation이 PATACON/AORRTC에
등록되어 있다.

```bash
./build/single_mbm igris_c igris_c_shelf_lift 1 --no-print-path
./build/single_mbm igris_c igris_c_shelf_lift 1 \
  --aorrtc --time 30 --no-print-path
./build/single_mbm igris_c igris_c_shelf_lift 1 --visualize
```

## 시각화

`--visualize`는 G1과 같은 방식으로 planner가 찾은 경로를 임시 JSON으로
전달하고 `scripts/visualize_igris_c.py`의 `mujoco.viewer.launch_passive`에서
start→goal 순서로 반복 재생한다. 35개 planning coordinate를 qpos에 적용하며,
손가락 22축은 지정 grasp 자세로 고정한다. 0.15 kg 박스는 매 frame 양 grasp
site의 중점과 왼손 frame 방향을 따라간다. 선반과 5 cm support polygon도
같은 live viewer에 표시한다.

GUI 없이 viewer 입력만 검사할 때는 planner가 만든 trajectory JSON에 대해
`PATACON_MUJOCO_VALIDATE_ONLY=1`을 설정하면 된다.

# PATACON: GPU-Parallel RRT-Connect

[![arXiv Paper](https://img.shields.io/badge/arXiv-2503.06757-b31b1b.svg)](https://arxiv.org/abs/2503.06757)

PATACON is a GPU-parallel RRT-Connect motion planner for constrained robotic
manipulation. Its main features are:

- concurrent sampling and bidirectional tree expansion on the GPU;
- SIMT-optimized edge and collision validation;
- constrained projection and tangent-space planning;
- optional AORRTC anytime cost optimization;
- MuJoCo replay, continuous G1 replanning, and multi-view video export.

The associated ICRA 2026 evaluation reports a 10x average speedup, a 5.4x
reduction in solution-time standard deviation, and a 1.4x improvement in
initial path cost over the compared planners.

## Supported robots

Use the CLI identifiers in this table exactly as written.

| Robot | CLI identifier | DoF | Default problem file | Default problem |
| --- | --- | ---: | --- | --- |
| Franka single arm | `franka_single` | 7 | `scripts/franka_single_problems.json` | `demo` |
| Franka dual arm | `franka` | 14 | `scripts/franka_problems.json` | `demo` |
| FFW-SG2 fixed base | `ffw_sg2` | 15 | `scripts/ffw_sg2_problems.json` | `tray_lift` |
| FFW-SG2 mobile base | `ffw_sg2_mobility` | 18 | `scripts/ffw_sg2_mobility_problems.json` | `tray_lift` |
| Unitree G1 | `g1` | 35 | `scripts/g1_problems.json` | `humanoid_shelf` |
| IGRIS-C | `igris_c` | 35 | `scripts/igris_c_problems.json` | `igris_c_shelf_lift` |

G1 uses six floating-base coordinates and 29 actuated joints, for a total of
35 planning DoF. MuJoCo reports `nq=36` because its free joint stores rotation
as a quaternion, while its velocity dimension remains `nv=35`.

### Robot descriptions

All descriptions required by PATACON are checked out inside this repository.
No robot loads a description from another workspace.

| Robot | Repository-local source | Adapter documentation |
| --- | --- | --- |
| Franka | `franka_description` | `resources/franka/README.md` |
| FFW-SG2 | `ai_worker/ffw_description` | `resources/ffw_sg2/README.md` |
| G1 | `unitree_ros/robots/g1_description` | `resources/g1/README.md` |
| IGRIS-C | `igris_c_description_public` | `resources/igris_c/README.md` |

The G1 backend uses the official Unitree `g1_29dof` URDF, MJCF, and meshes.
Its checked-in collision model contains 133 body spheres and 16 conservative
spheres per rubber hand.

## Build

Run all commands from the PATACON repository root.

### Initial setup

```bash
git submodule update --init --recursive

python3 resources/franka/prepare_description.py --check
python3 resources/g1/prepare_description.py --check
python3 resources/ffw_sg2/prepare_planning_urdf.py
python3 resources/ffw_sg2/prepare_collision_models.py
python3 resources/ffw_sg2/prepare_mujoco_scenes.py

cmake --preset patacon
cmake --build --preset patacon
```

### Incremental builds

Build only the frontend being used:

```bash
cmake --build --preset patacon-single
cmake --build --preset patacon-evaluate
```

`src/planning/PATACON.cu` is the shared CUDA translation unit for the standard
planner and AORRTC. The resulting `patacon_planners` library is linked into
both frontends. Reuse `build_patacon`; deleting it forces an expensive clean
CUDA build.

CUDA split compilation reduces clean-build time but increases peak memory
usage. Disable it on a lower-memory machine with:

```bash
cmake --preset patacon -DPATACON_CUDA_SPLIT_COMPILE=OFF
cmake --build --preset patacon
```

## Run the planner

PATACON provides two frontends:

- `single_mbm`: run one indexed problem;
- `evaluate_mbm`: run every problem in a problem JSON file.

Problem indices are one-based. `run_name` labels evaluation output and does
not need to match a problem name. Without `--problem-file`, each frontend uses
the default problem file listed above.

All commands below include `--no-print-path` to keep terminal output compact.
Remove it only when the full waypoint path is needed in stdout.

### Command forms

```bash
./build_patacon/single_mbm <robot> <problem_name> <problem_index> \
  [options] --no-print-path

./build_patacon/evaluate_mbm <robot> <run_name> \
  [options] --no-print-path
```

### Run one bundled problem

```bash
./build_patacon/single_mbm franka_single demo 1 --no-print-path
./build_patacon/single_mbm franka demo 1 --no-print-path
./build_patacon/single_mbm ffw_sg2 tray_lift 1 --no-print-path
./build_patacon/single_mbm ffw_sg2_mobility tray_lift 1 --no-print-path
./build_patacon/single_mbm g1 humanoid_shelf 1 --no-print-path
./build_patacon/single_mbm igris_c igris_c_shelf_lift 1 --no-print-path
```

### Evaluate a bundled problem set

```bash
./build_patacon/evaluate_mbm franka_single franka_single_default \
  --no-print-path --axis
./build_patacon/evaluate_mbm franka franka_dual_default \
  --no-print-path
./build_patacon/evaluate_mbm ffw_sg2 ffw_sg2_default \
  --no-print-path
./build_patacon/evaluate_mbm ffw_sg2_mobility ffw_sg2_mobility_default \
  --no-print-path
./build_patacon/evaluate_mbm g1 g1_default \
  --no-print-path
./build_patacon/evaluate_mbm igris_c igris_c_default \
  --no-print-path
```

### Run from a standalone JSON entry point

`single_mbm --config` can infer the robot backend, dimension, problem name,
query, world, and constraints from JSON:

```bash
./build_patacon/single_mbm \
  --config scripts/franka_single_problems.json --no-print-path

./build_patacon/single_mbm \
  --config scripts/g1_problems.json --visualize --no-print-path

./build_patacon/single_mbm \
  --config scripts/igris_c_problems.json --validate-config --no-print-path
```

For a collection containing multiple problem groups, add
`"selected_problem": {"name": "...", "index": 1}` at the top level. A
direct single-query file can use this compact form:

```json
{
  "schema_version": 1,
  "robot": {"model": "franka_single", "dimension": 7},
  "name": "direct_demo",
  "query": {
    "start": [1.018291, -0.276863, -0.648974, -0.990170, -0.539716, 2.329168, -2.144403],
    "goals": [[-0.064490, -0.725273, -0.044019, -2.522677, -0.413973, 3.385743, -1.982525]]
  },
  "world": {"sphere": [], "cylinder": [], "box": []}
}
```

`robot.dimension` is checked against the compiled backend; it does not resize
CUDA arrays. See the bundled problem files for `fixed_frame_pose`,
`relative_pose`, `axis_alignment`, and `com_support` constraint examples.

### Common options

| Option | Purpose |
| --- | --- |
| `--problem-file PATH` | Select a problem JSON file. |
| `--save-json PATH` | Save planner results. |
| `--run N`, `--runs N` | Repeat a query `N` times. |
| `--range VALUE` | Override the RRT-Connect extension range. |
| `--axis` | Enable the task's axis rows. |
| `--com` | Enable CoM constraints for `ffw_sg2_mobility`. |
| `--aorrtc` | Continue with anytime cost optimization. |
| `--time SECONDS` | Set the AORRTC budget; requires `--aorrtc`. |
| `--no-print-path` | Suppress the full waypoint list. |

Important `single_mbm`-only options:

| Option | Purpose |
| --- | --- |
| `--seed N` | Set the first random seed. |
| `--plot` | Save an ECDF or AORRTC convergence plot. |
| `--visualize` | Open the robot-specific MuJoCo replay. |
| `--no-path-smoothing` | Replay raw planner waypoints. |
| `--replanning` | Run continuous G1 replanning. |
| `--diagnostics` | Enable constrained-planner diagnostics; incompatible with `--aorrtc`. |

`evaluate_mbm` additionally accepts `--max-problems N`. Its `--plot` option
requires `--aorrtc`. Evaluation visualization supports Franka, G1, and both
FFW-SG2 models; IGRIS-C evaluation visualization is not enabled.

### Repeated runs

```bash
./build_patacon/single_mbm ffw_sg2 tray_lift 1 \
  --runs 500 --seed 1 --plot \
  --save-json logs/patacon_500runs.json \
  --no-print-path
```

The default seed is `1`; repeated runs use consecutive seeds. Without
`--aorrtc`, `--plot` writes a log-scale ECDF of cumulative solved runs versus
planner kernel time.

### AORRTC anytime optimization

```bash
./build_patacon/single_mbm ffw_sg2 tray_lift 1 \
  --aorrtc --time 10 --no-print-path
```

The default AORRTC budget is five seconds. PATACON first finds a feasible path,
then performs fresh-tree cost-bounded searches until the budget expires. A GPU
expansion round cannot be interrupted, so wall time can slightly exceed the
requested budget. See `AORRTC_IMPLEMENTATION.md` for the algorithm-to-code
mapping and output fields.

Both frontends report end-to-end timing statistics and solved-path length/cost
statistics after repeated runs. Tangent-space backtracking prevention is
always enabled.

## Reproducible random-pair benchmarks

### Franka single and dual arm

Build and regenerate the four datasets:

```bash
cmake --build --preset patacon --target \
  generate_franka_single_random_pairs \
  generate_franka_dual_random_pairs

./build_patacon/generate_franka_single_random_pairs
./build_patacon/generate_franka_single_random_pairs --axis
./build_patacon/generate_franka_dual_random_pairs
./build_patacon/generate_franka_dual_random_pairs --axis
```

Evaluate them with `franka_single` or `franka`; `franka_dual` is not a valid
planner identifier.

```bash
./build_patacon/evaluate_mbm franka_single franka_single_random_pairs \
  --problem-file scripts/franka_single_random_pairs_no_axis_100.json \
  --no-print-path

./build_patacon/evaluate_mbm franka_single franka_single_random_pairs_rigid \
  --problem-file scripts/franka_single_random_pairs_axis_100.json \
  --axis \
  --no-print-path

./build_patacon/evaluate_mbm franka franka_dual_random_pairs \
  --problem-file scripts/franka_dual_random_pairs_no_axis_100.json \
  --no-print-path

./build_patacon/evaluate_mbm franka franka_dual_random_pairs_rigid \
  --problem-file scripts/franka_dual_random_pairs_axis_100.json \
  --axis \
  --no-print-path
```

Add `--visualize` to an evaluation command to replay every solved path in one
MuJoCo window.

### FFW-SG2 fixed base

Generate the regular and axis datasets:

```bash
cmake --build --preset patacon --target generate_ffw_sg2_random_pairs

./build_patacon/generate_ffw_sg2_random_pairs
./build_patacon/generate_ffw_sg2_random_pairs \
  --axis \
  --output scripts/ffw_sg2_random_pairs_axis_100.json \
  --problem-name tray_lift_random_pairs_axis
```

Run one pair or evaluate each complete dataset:

```bash
./build_patacon/single_mbm ffw_sg2 tray_lift_random_pairs 1 \
  --problem-file scripts/ffw_sg2_random_pairs_100.json \
  --no-print-path

./build_patacon/evaluate_mbm ffw_sg2 random_pairs_100 \
  --problem-file scripts/ffw_sg2_random_pairs_100.json \
  --no-print-path

./build_patacon/evaluate_mbm ffw_sg2 random_pairs_axis_100 \
  --problem-file scripts/ffw_sg2_random_pairs_axis_100.json \
  --axis \
  --no-print-path
```

### FFW-SG2 mobile base

This dataset varies base `x/y/yaw`, lift, and arm joints and enforces the
dual-arm, axis, and payload-aware CoM constraints.

```bash
cmake --build --preset patacon --target \
  generate_ffw_sg2_mobility_random_pairs
./build_patacon/generate_ffw_sg2_mobility_random_pairs

./build_patacon/evaluate_mbm \
  ffw_sg2_mobility mobility_random_pairs_com_axis_100 \
  --problem-file scripts/ffw_sg2_mobility_random_pairs_com_axis_100.json \
  --com \
  --axis \
  --no-print-path
```

### Unitree G1

Both G1 datasets contain 100 reproducible 35-DoF start-goal pairs. The base
variant enforces fixed feet, payload-aware CoM support, and bimanual relative
pose. The axis-enabled variant also keeps the carried object's local +Z axis parallel
to world Z while leaving yaw free.

```bash
cmake --build --preset patacon --target generate_g1_random_pairs

./build_patacon/generate_g1_random_pairs
./build_patacon/generate_g1_random_pairs --axis

./build_patacon/evaluate_mbm g1 random_pairs_no_axis_100 \
  --problem-file scripts/g1_random_pairs_no_axis_100.json \
  --no-print-path

./build_patacon/evaluate_mbm g1 random_pairs_axis_100 \
  --problem-file scripts/g1_random_pairs_axis_100.json \
  --axis \
  --no-print-path
```

## MuJoCo visualization

Install the optional visualization dependencies:

```bash
python3 -m pip install -r requirements-visualization.txt
```

### Interactive replay

```bash
./build_patacon/single_mbm franka_single demo 1 \
  --visualize --no-print-path --axis
./build_patacon/single_mbm franka demo 1 \
  --visualize --no-print-path
./build_patacon/single_mbm ffw_sg2 tray_lift 1 \
  --visualize --no-print-path
./build_patacon/single_mbm ffw_sg2_mobility tray_lift 1 \
  --visualize --no-print-path
./build_patacon/single_mbm g1 humanoid_shelf 1 \
  --visualize --no-print-path
./build_patacon/single_mbm igris_c igris_c_shelf_lift 1 \
  --visualize --no-print-path
```

Interactive viewers repeat the start-to-goal trajectory until closed. Unless
`--no-path-smoothing` is selected, replay uses this pipeline:

```text
constrained shortcutting
  -> C2 quintic Hermite spline
  -> CUDA projection, joint-limit, and collision revalidation
  -> TOPP-RA time parameterization
  -> MuJoCo replay
```

Franka and FFW-SG2 replay construct all world obstacles at runtime from the
selected problem JSON's `sphere`, `cylinder`, and `box` arrays. Their MuJoCo XML
adapters contain the robot and task objects, but no hardcoded shelves, tables,
kitchen furniture, racks, or obstacle objects.

Benchmark statistics and saved planner JSON retain the raw planner path; this
post-processing changes only the replay copy. To inspect raw waypoints:

```bash
./build_patacon/single_mbm igris_c igris_c_shelf_lift 1 \
  --visualize --no-path-smoothing --no-print-path
```

### Continuous G1 replanning

```bash
./build_patacon/single_mbm g1 humanoid_shelf 1 \
  --replanning --no-print-path
```

Continuous replanning keeps one PATACON process and its GPU allocations alive.
Each request resets search state, uses a five-second planning limit, and
advances the random seed. The MuJoCo controller schedules a latency-aware
handoff to the accepted replacement path.

The moving red obstacle has a `0.04 m` planning radius, moves along world Y at
`0.05 m/s`, and pauses for two seconds at each endpoint. Each planning request
uses its current position as a static obstacle; future motion is not predicted.
During replanning, only the floor and moving obstacle are active world
collisions. The payload moves between `(0.3, 0.0, 0.5)` m and
`(0.3, 0.0, 1.0)` m.

### Replay modes

`single_mbm --visualize` selects a fixed replay mode for each robot. Fixed-base
FFW-SG2 uses direct `qpos` replay. Mobile FFW-SG2 and IGRIS-C use actuator
control with dynamics, while G1 uses torque-PD control. The former `--real`
mode-selection option is no longer exposed.

### Video export

Set `PATACON_VIDEO` to render MP4 directly from MuJoCo. Set
`PATACON_VIDEO_VIEWS=all` for synchronized `front`, `front_left`,
`front_right`, and `overhead` outputs.

```bash
PATACON_VIDEO=logs/simul/g1/g1_humanoid_shelf.mp4 \
PATACON_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm g1 humanoid_shelf 1 \
    --visualize --axis --no-print-path
```

Use a comma-separated subset such as `front,left,right,rear` for selected
views. MP4 output requires `ffmpeg`. The aliases `PATACON_FRANKA_VIDEO`,
`PATACON_FFW_SG2_VIDEO`, `PATACON_G1_VIDEO`, and `PATACON_IGRIS_C_VIDEO` are
also supported.

Render a saved G1 result without replanning:

```bash
python3 scripts/replay_g1_result.py \
  --environment resources/g1/g1_humanoid_shelf.xml \
  --result traces/g1_humanoid_shelf_1_result.json \
  --video logs/g1_humanoid_shelf.mp4 \
  --video-views all --fps 60 --video-width 1280 --video-height 720
```

G1 loads its 29 actuated-joint velocity limits from the repository-local
official URDF. The six floating-base coordinates use a `1.4 m/s` or `rad/s`
fallback. `PATACON_G1_VELOCITY_SCALE` scales replay speed.

## Validation

Build all robot-specific validation executables:

```bash
cmake --build --preset patacon --target \
  validate_ffw_sg2_projection \
  validate_franka_constraints validate_franka_collision \
  validate_g1_constraints validate_g1_collision \
  validate_igris_c_kinematics validate_igris_c_constraints \
  validate_igris_c_collision
```

Run the validators:

```bash
./build_patacon/validate_ffw_sg2_projection

./build_patacon/validate_franka_constraints
./build_patacon/validate_franka_collision franka_single
./build_patacon/validate_franka_collision franka

./build_patacon/validate_g1_constraints
./build_patacon/validate_g1_constraints --axis
./build_patacon/validate_g1_collision
./build_patacon/validate_g1_collision \
  scripts/g1_problems.json --axis

./build_patacon/validate_igris_c_kinematics
./build_patacon/validate_igris_c_constraints
./build_patacon/validate_igris_c_collision
```

`validate_franka_collision` accepts `franka_single` or `franka` as its first
argument and an optional problem JSON as its second. The G1 collision validator
requires the problem path before `--axis`.

## Advanced features

### FFW-SG2 attached-object collision

`ffw_sg2_mobility` can represent an attached object with sphere proxies in the
grasp frame. The planner checks those spheres against the world and all robot
spheres except configured contact links. `box_sphere_grid` creates a
conservative voxel-centered cover of a box:

```json
"attached_object_collision": {
  "enabled": true,
  "world_offset": [0.1, 0.0, 0.0],
  "box_sphere_grid": {
    "half_extents": [0.12, 0.11, 0.06],
    "counts": [7, 7, 5],
    "radius_padding": 0.001
  }
}
```

See `scripts/ffw_sg2_mobility_problems.json` for contact-link and ignored-sphere
examples.

### Tree trace export

```bash
./build_patacon/single_mbm ffw_sg2 tray_lift 1 \
  --trace-mode tree \
  --html-trace-mode tree \
  --html-max-tree-nodes 0 \
  --no-print-path
```

Outputs default to `traces/`. Use `--save-json`, `--graphml`, or `--html` for
explicit paths. A positive `--html-max-tree-nodes` samples the HTML view while
retaining the full JSON and GraphML tree; zero embeds every node.

### Planner configuration

The main compile-time settings are defined in the planning settings and robot
backends.

| Setting | Meaning |
| --- | --- |
| `max_samples` | Maximum total tree samples. |
| `max_iters` | Maximum planning iterations. |
| `num_new_configs` | Samples generated per iteration. |
| `range` | Maximum RRT-Connect extension length. |
| `granularity` | Discretized collision checks per edge. Must match the robot collision kernel batch size. |
| `balance` | `0`: none, `1`: distributed, `2`: single-sided tree balancing. |
| `tree_ratio` | Smaller-tree threshold; normally `0.5` for balance mode 1 and `1` for mode 2. |
| `dynamic_domain` | Enable dynamic-domain sampling. |
| `dd_alpha` | Dynamic-domain radius update factor. |
| `dd_radius` | Initial dynamic-domain radius. |
| `dd_min_radius` | Minimum dynamic-domain radius. |

MotionBenchMaker-compatible problem JSON files can be generated following the
[upstream resource workflow](https://github.com/KavrakiLab/vamp/blob/35080be604aabd4373cc7db8608297afaa446878/resources/README.md#motionbenchmaker-problems).

## Add a robot

1. Create fine and conservative approximate sphere models from the robot URDF.
   [Foam](https://github.com/CoMMALab/foam) can generate the spherized URDFs.
2. Create an SRDF describing disabled self-collision pairs.
3. Add the robot inputs under `resources/<robot>`.
4. Generate CUDA FK/collision code with
   [Cricket](https://github.com/CoMMALab/cricket/tree/gpu-cc-early-exit).
5. Add the generated implementation under `src/robots`.
6. Define the robot dimension, limits, and collision constants in
   `src/planning/Robots.hh` and `src/planning/RobotCollisionTraits.hh`.
7. Register the backend and explicit planner instantiations in
   `src/planning/PATACON.cu`.
8. Add a problem JSON, validation executable, and CMake targets.
9. Rebuild PATACON and validate FK, constraints, environment collision, and
   self-collision before adding benchmarks.

The FFW-SG2 files in `resources/ffw_sg2` provide a complete example of the
planning URDF, fine/approximate collision descriptions, Cricket configuration,
metadata, post-processing, and validation flow.

# PATACON: GPU-Parallel Tangent-Bundle RRT-Connect for Constrained Motion Planning

PATACON (PArallel TAngent-bundle RRT-CONnect) is a GPU-parallel,
bidirectional constrained motion planner. It combines local tangent-space
guidance with RRT-Connect-style tree growth so candidate motions follow the
constraint manifold before numerical projection. Its main algorithmic
components are:

- Tangent-Guided Sampling (TGS) with tree-specific shared tangent-space banks;
- Tangent-Guided Multi-Edge Projection (TGMP), which generates several
  consecutive candidate edges from one sample and nearest-neighbor query;
- concurrent projection and collision validation of TGMP candidate edges;
- insertion of only the longest consecutively valid candidate prefix;
- target-directed CONNECT using the same EXTEND operation;
- optional AO-PATACON anytime cost optimization based on AORRTC;
- MuJoCo replay, continuous G1 replanning, and multi-view video export.

The accompanying paper evaluates eight constrained-planning regimes from
7-DoF manipulation to 35-DoF humanoid whole-body planning. It reports that
the advantage over cpRRTC increases with configuration-space dimension and
constraint restrictiveness, reaching up to an 11.25x speedup with a 100%
success rate.

## Algorithm overview

PATACON maintains a start tree and a goal tree, together with a separate
tangent-space bank for each tree. Every stored tree node has a projected
manifold configuration, a parent, a tangent-space index, and a local
tangent-space representation. CUDA blocks explore these shared trees and
tangent-space banks concurrently.

### Algorithm 1: PATACON search

The top-level search follows the paper's pseudocode in this order:

```text
initialize start/goal trees and their root tangent spaces

for each CUDA block in parallel:
    choose the grow tree and opposite tree using tree balancing
    select a ready tangent space from the grow tree's bank
    sample q_rand in that tangent space
    find q_near among nodes associated with that tangent space
    EXTEND the grow tree using q_rand - q_near

    if EXTEND advanced:
        select the closest node q_tar in the opposite tree
        CHECK_CONNECTION(q_end, q_tar)

        while not connected and the CONNECT limit is not reached:
            project q_tar - q_cur onto q_cur's tangent space
            EXTEND toward that fixed target
            CHECK_CONNECTION(q_end, q_tar)

        if connected:
            trace both parent chains and return the solution path
```

Tree balancing selects the smaller tree when the size imbalance exceeds the
configured threshold and otherwise distributes blocks between the two trees.
A block dynamically scans its selected tree's tangent-space bank until it
finds a fully initialized entry; tangent spaces are not permanently assigned
to individual blocks.

TGS samples a block-local Halton direction in the selected tangent space and
performs a tangent-space-local nearest-neighbor query. For every non-root
tangent space, the frontend enables the paper's backtracking-prevention rule:
a sampled direction pointing toward the parent tangent-space region is
reversed into the forward half-space.

### Algorithm 2: EXTEND with TGMP

Exploration and CONNECT both call the same EXTEND operation. Given a tangent
direction, EXTEND:

1. normalizes the direction and generates up to `Kmax` consecutive tangent
   candidates at step size `xi`;
2. stops candidate generation after including the first candidate whose
   constraint residual `EM` exceeds `epsilon_M`;
3. projects and collision-checks the candidate edges;
4. inserts only the longest valid prefix, returning `Trapped` when the first
   edge is invalid and `Advanced` otherwise;
5. creates a new tangent space only when the complete candidate batch is
   accepted and its last tangent candidate crosses the renewal threshold.

Different TGMP candidate edges are processed concurrently by cooperative GPU
thread groups. Within one edge, discretized waypoints are projected in their
original sequential order because each projection starts from the preceding
projected waypoint. Joint limits, constraint convergence, self-collision, and
environment collision are checked before a projected edge can be inserted.
During CONNECT, an edge must additionally reduce the distance to the fixed
opposite-tree target by at least `epsilon_prog`.

### Algorithm 3: connection and bridge validation

The paper defines `CHECK_CONNECTION` as follows: first require the two selected
nodes to be within `epsilon_con`, then project and validate the remaining bridge
motion, and only then trace the two parent chains and report `Connected`.

The current repository intentionally retains the earlier compatibility rule in
`ConnectionCheck.cuh`: it reports `Connected` using only the
`epsilon_con` distance test and does not project or collision-check a separate
bridge segment. The control-flow placement of `CHECK_CONNECTION` matches the
paper—once after exploratory EXTEND and after every successful CONNECT
EXTEND—but this final bridge-validation detail is not yet paper-equivalent.

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

cmake --preset patacon
cmake --build --preset patacon
```

### Incremental builds

Build only the frontend being used:

```bash
cmake --build --preset patacon-single
cmake --build --preset patacon-evaluate
```

The standard planner and AORRTC are separate CUDA translation units:
`src/planning/PATACON.cu` and
`src/planning/AORRTCOptimization.cu`. Both objects are archived in the
`patacon_planners` library linked by the frontends. A PATACON-only source
change therefore reuses the compiled AORRTC object, and an AORRTC-only change
reuses the PATACON object. Reuse `build_patacon`; deleting it still forces an
expensive clean CUDA build of both objects.

CUDA split compilation reduces clean-build time but increases peak memory
usage. Disable it on a lower-memory machine with:

```bash
cmake --preset patacon -DPATACON_CUDA_SPLIT_COMPILE=OFF
cmake --build --preset patacon
```

### Source boundaries

Shared types, adapters, and CUDA implementation boundaries are kept out of
`scripts/`:

| Path | Responsibility |
| --- | --- |
| `src/config/PlanningProblemJson.hh` | Normalized planning-problem model and JSON loading |
| `src/config/RobotRegistry.hh` | CUDA-independent robot names, dimensions, and default problem files |
| `src/config/PrimitiveEnvironmentJson.hh` | JSON primitives to collision-environment conversion |
| `src/planning/RobotDispatch.hh` | CLI robot name to compiled backend dispatch |
| `src/planning/PlannerResult.hh` | Planner result and validation data types |
| `src/io/PlannerResultJson.hh` | Versioned planner-result serialization |
| `src/planning/PATACON.cu` | Pseudocode-shaped PATACON search flow, public `solve()` facade, and PATACON template instantiations |
| `src/planning/AORRTCOptimization.cu` | Independent AO-PATACON CUDA translation-unit entry and its PATACON device-helper boundary |
| `src/planning/AORRTCOptimization.cuh` | AORRTC-based cost-bounded restart and anytime optimization implementation |
| `src/planning/patacon/RuntimeControl.cpp` | Host-owned device-reset, workspace-reuse, and time-budget state |
| `src/planning/patacon/GpuRuntime.cuh` | Device globals, diagnostics, distance helpers, and random/Halton sampling |
| `src/planning/patacon/DeviceEnvironment.cuh` | Device primitive ownership, reuse, cleanup, and request reset |
| `src/planning/patacon/Projection.cuh` | Robot-specific waypoint projection, tangent-basis construction, and tangent-space sampling primitives |
| `src/planning/patacon/CollisionValidation.cuh` | Tangent-space initialization and projected-edge feasibility/collision validation |
| `src/planning/patacon/SearchContext.cuh` | Kernel argument view, block-local shared state, and step-result types |
| `src/planning/patacon/TreeOperations.cuh` | Tree and tangent-space bookkeeping helpers |
| `src/planning/patacon/TreeSelection.cuh` | Existing tree balancing and tangent-space selection policy |
| `src/planning/patacon/Exploration.cuh` | TGS `q_rand` sampling, tangent-space-local nearest-neighbor selection, and `v_ext` preparation |
| `src/planning/patacon/Extend.cuh` | Pseudocode-shaped shared Algorithm 2 EXTEND entry point |
| `src/planning/patacon/ExtendStages.cuh` | TGMP candidate generation, parallel projection/validation, valid-prefix insertion, and tangent-space renewal stages used by EXTEND |
| `src/planning/patacon/ConnectionCheck.cuh` | CHECK_CONNECTION placement from Algorithm 3 with the current distance-only compatibility rule |
| `src/planning/patacon/Connect.cuh` | Algorithm 1 fixed-target selection and repeated target-directed EXTEND orchestration |
| `src/planning/patacon/PathExtraction.cuh` | Device-side solution claiming and parent-chain extraction into two path segments |
| `src/planning/patacon/PathAssembly.cuh` | Host-side download and assembly of device path segments into the planner result |
| `src/planning/patacon/PathPostprocessing.cuh` | Solution tracing plus visualization path simplification and validation |
| `src/planning/patacon/SolveWorkspace.cuh` | Reusable solver workspace allocation, ownership, and cleanup |
| `src/planning/patacon/SolveStages.cuh` | Host-side validation, CUDA setup, launch, result collection, and cleanup implementation stages |
| `src/planning/patacon/Solve.cuh` | Pseudocode-shaped host solve orchestration over the implementation stages |
| `src/planning/patacon/ExplicitInstantiations.cuh` | Compiled robot and visualization template instantiations |

`single_mbm` and `evaluate_mbm` remain application frontends. The compatibility
header `scripts/planner_result_json.hh` forwards to the new IO location so
existing local tools do not break. The CUDA implementation fragments above are
internal headers, not independent backend APIs. `PATACON.cu` owns the standard
planner state and includes the full flow. `AORRTCOptimization.cu` compiles the
shared device primitives it needs into a separate, self-contained CUDA object
and calls PATACON's public host API for its initial feasible-path search. The
search kernel in `PATACON.cu` reads in Algorithm 1 order: tree balancing,
tangent-space selection, TGS sampling and nearest-neighbor selection, TGMP
EXTEND, the pre-CONNECT connection check, repeated CONNECT EXTEND calls, and
path extraction. Allocation and low-level CUDA details stay in the internal
headers.

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

### AO-PATACON anytime optimization

```bash
./build_patacon/single_mbm ffw_sg2 tray_lift 1 \
  --aorrtc --time 10 --no-print-path
```

The paper calls this anytime planner AO-PATACON. The current CLI and source API
retain the `--aorrtc` and `AORRTC` names because AO-PATACON embeds PATACON in
an AORRTC-based optimization framework. PATACON first finds a feasible path,
uses its configuration-space path length as the incumbent cost bound, and then
restarts fresh-tree cost-bounded searches. Each improved solution tightens the
bound and starts another search. The paper evaluates a 10-second optimization
budget; the executable default is five seconds unless `--time` overrides it.

A GPU expansion round cannot be interrupted, so wall time can slightly exceed
the requested budget. See `AORRTC_IMPLEMENTATION.md` for the algorithm-to-code
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

Configure, build, and run the complete pre-refactor behavior contract with:

```bash
python3 -m pip install -r requirements-test.txt
cmake --preset patacon
cmake --build --preset patacon
ctest --preset patacon
```

CTest runs one deterministic solved planning case for every supported robot
backend, validates each saved result against
`schemas/planner-result-v1.schema.json`, and checks configuration dimensions,
finite values, path counts, and start/goal endpoints. Separate contract tests
cover AORRTC, repeated-run bundles, and `evaluate_mbm` problem-set output.
CUDA tests share a CTest resource lock so `ctest -j` does not run multiple
planner kernels concurrently on the same GPU.

The version-1 result contracts are documented in `schemas/README.md`. Breaking
changes require a new `format` value and schema version rather than modifying
the meaning of an existing version.

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

`validate_ffw_sg2_projection` is registered as
`validation.ffw_sg2.projection_known_failure` but disabled: before this test
suite was introduced, its configuration checks passed while only 1 of 3
motion-segment checks passed. This known failure is recorded explicitly instead
of being treated as a successful baseline.

The pre-refactor performance record is
`benchmarks/baseline-rtx5090-2026-09-30.json`. It contains 20 runs per bundled
robot case, seeds 1 through 20, hardware/build metadata, solved rates, and
kernel-time distributions. See `benchmarks/README.md` for the reproduction
command and comparison policy; performance timing is not a hardware-independent
CTest pass/fail condition.

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

The main settings are defined in `PATACON_settings.hh`. The paper notation maps
to the implementation as follows.

| Setting | Paper symbol | Meaning |
| --- | --- | --- |
| `max_samples` | — | Maximum node capacity per search tree. |
| `max_tangent_spaces` | — | Maximum tangent-space capacity per tree. |
| `max_iters` | `Imax` | Maximum planning iterations executed by a CUDA block. |
| `num_new_configs` | `Nblk` | Number of CUDA blocks, and therefore concurrent tree-expansion attempts. |
| `range` | `xi` | Step length between consecutive TGMP tangent candidates. |
| `max_concon_nodes` | `Kmax` | Maximum candidate edges generated by one EXTEND call. |
| `em_threshold` | `epsilon_M` | Constraint-residual threshold that stops candidate generation and requests tangent-space renewal. |
| `granularity` | `G` | Projected waypoints per candidate edge; must match the robot collision kernel batch size. |
| `max_connect_concon_chunks` | `Jmax` | Maximum target-directed EXTEND calls in one CONNECT attempt. |
| `connect_progress_epsilon` | `epsilon_prog` | Minimum target-distance reduction required from a CONNECT edge. |
| `connect_reached_tolerance` | `epsilon_con` | Distance threshold that triggers CHECK_CONNECTION. |
| `balance` | — | `0`: none, `1`: distributed, `2`: single-sided tree balancing. |
| `tree_ratio` | — | Smaller-tree threshold; normally `0.5` for balance mode 1 and `1` for mode 2. |

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

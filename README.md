# pRRTC: GPU-Parallel RRT-Connect

[![arXiv VAMP](https://img.shields.io/badge/arXiv-2503.06757-b31b1b.svg)](https://arxiv.org/abs/2503.06757)
[![Project Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://commalab.org/papers/pRRTC/)

This repository holds the code for the ICRA 2026 paper [pRRTC: GPU-Parallel RRT-Connect for Fast, Consistent, and Low-Cost Motion Planning.](https://arxiv.org/abs/2503.06757)

We introduce pRRTC, a GPU-based, parallel RRT-Connect-based algorithm. Our approach has three key improvements: 
- Concurrent sampling, expansion and connection of start and goal trees via GPU multithreading
- SIMT-optimized collision checking to quickly validate edges, inspired by the SIMD-optimized validation of [Vector-Accelerated Motion Planning](https://github.com/KavrakiLab/vamp/tree/main)
- Efficient memory management between block- and thread-level parallelism, reducing expensive memory transfer overheads

Our empirical evaluations show that pRRTC achieves a 10x average speedup on constrained reaching tasks. pRRTC also demonstrates a 5.4x reduction in solution time standard deviation and 1.4x improvement in initial path costs compared to state-of-the-art motion planners in complex environments.

## Supported Robots

The benchmark executables accept the following robot identifiers. Use these
exact identifiers on the command line; in particular, the dual-arm Franka is
`franka`, the mobile FFW-SG2 is `ffw_sg2_mobility`, and IGRIS is `igris_c`.

| Robot | CLI identifier | DoF | Default problem file | Default problem |
| --- | --- | ---: | --- | --- |
| Franka single arm | `franka_single` | 7 | `scripts/franka_single_problems.json` | `demo` |
| Franka dual arm | `franka` | 14 | `scripts/franka_problems.json` | `demo` |
| FFW-SG2 fixed-base dual arm | `ffw_sg2` | 15 | `scripts/ffw_sg2_problems.json` | `tray_lift` |
| FFW-SG2 mobile dual arm | `ffw_sg2_mobility` | 18 | `scripts/ffw_sg2_mobility_problems.json` | `tray_lift` |
| Unitree G1 | `g1` | 35 | `scripts/g1_problems.json` | `humanoid_shelf` |
| IGRIS-C | `igris_c` | 35 | `scripts/igris_c_problems.json` | `igris_c_shelf_lift` |

Forward-kinematics and collision-checking code is generated or imported per
robot. The FFW-SG2 kernels were generated using
[Cricket](https://github.com/CoMMALab/cricket.git).

The G1 collision model augments VAMP's 133 body spheres with 16 conservative
spheres per rubber hand. These hand spheres participate in every robot-to-world
check; only attached-payload-to-robot checks omit them because the grasped
payload intentionally overlaps the hands.

## Building Code
To build pRRTC, follow the instructions below:

```bash
git clone https://github.com/gahyun0425/cpRRTC_TB-RRT.git PATACON
cd PATACON
cmake --preset patacon
cmake --build --preset patacon
```

For normal iteration, build only the frontend being used. Both presets keep
the existing `build_patacon` object files and rebuild only changed inputs:

```bash
cmake --build --preset patacon-single
cmake --build --preset patacon-evaluate
```

`pRRTC.cu` and `AORRTC.cu` form one large CUDA translation unit. CMake builds
that unit once as the shared `patacon_planners` static library, then links it
into both frontends. CUDA 13.1 split compilation is enabled for this library
with `--split-compile=0`, and relocatable device code remains enabled because
it produced the lowest measured end-to-end clean backend build time on the
current 16-core machine:

| Planner backend configuration | Measured time |
| --- | ---: |
| split OFF, separable ON | 2345.64 s |
| split ON, separable ON | 628.10 s |
| split ON, separable OFF | 800.95 s |

With separable compilation ON, linking both frontends added 45.41 seconds in
the measured clean build. The default therefore reduced the measured total
from more than 39 minutes to about 11 minutes 14 seconds. A repeated target
build with no source changes is a no-op. Split compilation trades memory for
wall time and used roughly 31--34 GiB at peak in these measurements. On a
lower-memory machine it can be disabled with:

```bash
cmake --preset patacon -DPATACON_CUDA_SPLIT_COMPILE=OFF
```

`PATACON_CUDA_SEPARABLE_COMPILATION=OFF` is also supported and was verified
to compile, link, and run, but was slower in the measurement above. Do not
delete `build_patacon` between builds unless a clean rebuild is intentional.

## Running Code

The repository provides `single_mbm` for one indexed problem and
`evaluate_mbm` for every problem in a problem JSON file. Run commands from the
repository root. Problem indices are one-based.

```bash
./build_patacon/single_mbm <robot> <problem_name> <problem_index> [options]
./build_patacon/evaluate_mbm <robot> <run_name> [options]
```

`run_name` labels evaluation output; it does not have to match a problem name.
Without `--problem-file`, both executables load
`scripts/<robot>_problems.json`.

### Basic commands for the current robot models

Run the bundled problem once:

```bash
./build_patacon/single_mbm franka_single demo 1
./build_patacon/single_mbm franka demo 1
./build_patacon/single_mbm ffw_sg2 tray_lift 1
./build_patacon/single_mbm ffw_sg2_mobility tray_lift 1
./build_patacon/single_mbm g1 humanoid_shelf 1
./build_patacon/single_mbm igris_c igris_c_shelf_lift 1
```

### Standalone planning JSON

`single_mbm` can infer the compiled robot backend, dimension, problem name,
start, goals, world, and constraints from one JSON entry point:

```bash
./build_patacon/single_mbm --config scripts/franka_single_problems.json
./build_patacon/single_mbm --config scripts/g1_problems.json --visualize
./build_patacon/single_mbm --config scripts/igris_c_problems.json --validate-config
```

The existing collection layout remains supported. If a collection contains
multiple problem groups, add `"selected_problem": {"name": "...", "index": 1}`
at the top level. A direct, single-query file uses this layout:

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

`robot.dimension` is validated against the selected compiled backend; it does
not override CUDA array dimensions. Constraint declarations can keep the
legacy robot parameter object or use the common `items` representation:

```json
{
  "constraints": {
    "tolerance_squared": 0.000001,
    "items": [
      {
        "type": "fixed_frame_pose",
        "frame": "left_foot",
        "target": {
          "quaternion_wxyz": [1, 0, 0, 0],
          "position": [0.12, 0.175, 0]
        }
      },
      {
        "type": "fixed_frame_pose",
        "frame": "right_foot",
        "target": {
          "quaternion_wxyz": [1, 0, 0, 0],
          "position": [0.12, -0.175, 0]
        }
      },
      {
        "type": "relative_pose",
        "frame_a": "left_hand",
        "frame_b": "right_hand",
        "target": [1, 0, 0, 0, 0, -0.303, 0]
      },
      {
        "type": "axis_alignment",
        "frame": "left_hand",
        "local_axis": [1, 0, 0],
        "target_world_axis": [0, 0, 1]
      },
      {
        "type": "com_support",
        "support_frames": ["left_foot", "right_foot"],
        "support_polygon": [0.105, -0.169, 0.105, 0.169, 0.005, 0.166, 0.005, -0.166],
        "support_margin_m": 0.05,
        "payload_mass_kg": 0.0
      }
    ]
  }
}
```

The common JSON layer normalizes these declarations into the existing
robot-specific CUDA constraint parameters. FK, collision, projection, and
tangent-space execution remain in the precompiled robot backends, so adding a
new task needs only JSON while adding an entirely new robot still requires a
backend implementation and rebuild.

Evaluate every problem in each bundled problem file:

```bash
./build_patacon/evaluate_mbm franka_single franka_single_default
./build_patacon/evaluate_mbm franka franka_dual_default
./build_patacon/evaluate_mbm ffw_sg2 ffw_sg2_default
./build_patacon/evaluate_mbm ffw_sg2_mobility ffw_sg2_mobility_default
./build_patacon/evaluate_mbm g1 g1_default
./build_patacon/evaluate_mbm igris_c igris_c_default
```

### Command-line options

Both benchmark executables accept `--problem-file PATH`, `--save-json PATH`,
`--run N`/`--runs N`, `--range VALUE`, `--aorrtc`, `--time SECONDS`,
`--no-print-path`, `--rigid-orientation`, and `--com`. `--time` requires
`--aorrtc`, and `--com` is valid only for `ffw_sg2_mobility`.

`single_mbm` additionally accepts:

- `--seed N` and `--plot` for repeated-run pRRTC timing ECDFs.
- `--visualize` for `franka_single`, `franka`, `ffw_sg2`,
  `ffw_sg2_mobility`, `g1`, and `igris_c`.
- `--no-path-smoothing` with `--visualize` to replay the raw planner waypoint
  path without shortcutting, a spline, or TOPP-RA.
- `--real` with `--visualize` for `ffw_sg2_mobility` and `igris_c` only.
- `--object-mass-kg KG` and `--support-margin M` for the mobile FFW-SG2.
  These options require `--com` or `--real`; `--object-mass-kg` is also
  accepted for IGRIS-C real-dynamics replay.
- `--diagnostics`, `--no-waypoint-smoothing`, and `--max-concon-nodes N` for
  constrained-planner experiments.
  `--diagnostics` cannot be combined with `--aorrtc`.
- `--trace-mode`, `--html-trace-mode`, `--html-max-tree-nodes`, `--graphml`,
  `--html`, `--path-key`, and `--patacon-root` for trace export.
- Real-dynamics tuning options `--real-speed`, `--real-settle-steps`,
  `--real-initial-settle-steps`, `--real-payload-mode`, `--real-base-kp-xy`,
  `--real-base-kp-yaw`, `--real-base-max-speed`,
  `--real-base-max-yaw-rate`, `--real-steer-rate-limit`, and
  `--real-drive-accel-limit`.

`evaluate_mbm` additionally accepts `--max-problems N`. Its `--plot` option
requires `--aorrtc`. Its `--visualize` option supports `g1`, `franka_single`,
`franka`, `ffw_sg2`, and `ffw_sg2_mobility`; IGRIS-C evaluation visualization
is intentionally excluded.

Tangent-Space backtracking prevention is always enabled for both the standard
pRRTC path and `--aorrtc`; no command-line option is required.

Repeat planning with `--run N` (`--runs N` is also accepted):

```bash
./build_patacon/single_mbm ffw_sg2 tray_lift 1 --run 100 --no-print-path
```

The base pRRTC seed defaults to `1`. Repeated runs use consecutive seeds,
starting at the value passed to `--seed`. Use `--plot` to save a log-scale
ECDF of cumulative solved runs versus pRRTC kernel planning time:

```bash
./build_patacon/single_mbm ffw_sg2 tray_lift 1 \
  --runs 500 --seed 1 --plot \
  --save-json logs/prrtc_500runs.json --no-print-path
```

The plot is written to
`logs/prrtc_<robot>_<problem>_<index>_<runs>runs_ecdf.png`. When `--aorrtc`
is also present, `--plot` retains its AORRTC anytime-cost convergence meaning.

### Optional AORRTC anytime optimization

`single_mbm` keeps the original first-solution planner as the default. Enable
the cost-augmented AORRTC path with `--aorrtc`:

```bash
./build_patacon/single_mbm ffw_sg2 tray_lift 1 --aorrtc
```

The AORRTC search budget defaults to 5 seconds. Override it with `--time`:

```bash
./build_patacon/single_mbm ffw_sg2 tray_lift 1 --aorrtc --time 10
```

`--time` is accepted only together with `--aorrtc`. The first search is the
current bidirectional planner without a cost bound. Every node records its
cost-to-come. After a solution is found, subsequent fresh-tree searches use
AORRTC cost sampling, cost-aware nearest-neighbour selection, lower-cost parent
resampling, and the remaining solution-cost budget during CONNECT. When a
better solution is found, both trees and Tangent-Space membership are cleared
and the search restarts with the tighter cost bound. GPU allocations and the
RNG/Halton sequence are reused, but tree nodes are not reused.

The AORRTC optimization loop itself uses identity path simplification; its
reported and saved planner path is therefore unchanged. When `--visualize` is
used, a separate constrained shortcut pass is applied only to the replay path,
unless `--no-path-smoothing` is specified.
See `AORRTC_IMPLEMENTATION.md` for the algorithm-to-code mapping and output
fields.
A single GPU expansion round cannot be interrupted halfway, so observed wall
time can exceed the requested search budget slightly.

After the runs finish, both benchmark executables print the average, minimum,
maximum, and population standard deviation of end-to-end planner time as
`time_sec_avg`, `time_sec_min`, `time_sec_max`, and `time_sec_std`. They also
print `path_length_avg/min/max` and `cost_avg/min/max` over solved runs.

### Franka single and dual random start-goal benchmarks

Build the Franka single- and dual-arm generators with:

```bash
cmake --build --preset patacon --target generate_franka_single_random_pairs
cmake --build --preset patacon --target generate_franka_dual_random_pairs
```

The default generator invocations create the non-rigid-orientation datasets.
The `--rigid-orientation` variants select the corresponding rigid-orientation
output filename, problem name, and seed automatically:

```bash
./build_patacon/generate_franka_single_random_pairs
./build_patacon/generate_franka_single_random_pairs --rigid-orientation

./build_patacon/generate_franka_dual_random_pairs
./build_patacon/generate_franka_dual_random_pairs --rigid-orientation
```

For dual-arm datasets, both randomized endpoints are projected onto the source
demo's fixed bimanual relative pose. The attached tray therefore keeps the
planner's fixed left-end-effector transform while the right gripper remains at
the source grasp pose.

Evaluate the generated datasets with the planner robot identifiers
`franka_single` and `franka`; `franka_dual` is not a valid planner identifier.

```bash
./build_patacon/evaluate_mbm franka_single franka_single_random_pairs \
  --problem-file scripts/franka_single_random_pairs_no_rigid_orientation_100.json \
  --no-print-path

./build_patacon/evaluate_mbm franka_single franka_single_random_pairs_rigid \
  --problem-file scripts/franka_single_random_pairs_rigid_orientation_100.json \
  --rigid-orientation --no-print-path

./build_patacon/evaluate_mbm franka franka_dual_random_pairs \
  --problem-file scripts/franka_dual_random_pairs_no_rigid_orientation_100.json \
  --no-print-path

./build_patacon/evaluate_mbm franka franka_dual_random_pairs_rigid \
  --problem-file scripts/franka_dual_random_pairs_rigid_orientation_100.json \
  --rigid-orientation --no-print-path
```

Add `--visualize` to any Franka `evaluate_mbm` command to cycle through every
solved path in one MuJoCo window. The visualizer uses the same fixed
end-effector-to-object transforms as the CUDA collision model instead of
deriving a transform from the object's default XML world pose.

Run either generator with `--help` to see its dataset controls, including
`--template`, `--output`, `--source-name`, `--source-index`, `--problem-name`,
`--count`, `--seed`, sampling sigmas, workspace regions, and
`--max-candidates`.

### FFW-SG2 random start-goal benchmark

`scripts/ffw_sg2_random_pairs_100.json` contains 100 reproducible 15-DoF
start-goal pairs for the fixed-base dual-arm FFW-SG2. The world and attached
object are copied unchanged from `tray_lift`. Joint states are sampled around
the reference endpoints, projected onto the dual-arm relative-pose constraint,
and retained only when the robot and attached object are collision-free.

The attached-object center is restricted to `[0.53, 0.61] x [-0.05, 0.05] x
[1.04, 1.12]` m at the start and to the shelf task region `[0.75, 0.77] x
[-0.03, 0.03] x [1.47, 1.56]` m at the goal. Regenerate the same set with:

```bash
cmake --build --preset patacon --target generate_ffw_sg2_random_pairs
./build_patacon/generate_ffw_sg2_random_pairs
```

Run one pair or evaluate all 100 pairs with:

```bash
./build_patacon/single_mbm ffw_sg2 tray_lift_random_pairs 1 \
  --problem-file scripts/ffw_sg2_random_pairs_100.json

./build_patacon/evaluate_mbm ffw_sg2 random_pairs_100 \
  --problem-file scripts/ffw_sg2_random_pairs_100.json \
  --no-print-path
```

For the 8-dimensional manifold that also constrains the left-gripper axis,
generate and evaluate `scripts/ffw_sg2_random_pairs_rigid_orientation_100.json`
with:

```bash
./build_patacon/generate_ffw_sg2_random_pairs \
  --rigid-orientation \
  --output scripts/ffw_sg2_random_pairs_rigid_orientation_100.json \
  --problem-name tray_lift_random_pairs_rigid_orientation

./build_patacon/evaluate_mbm ffw_sg2 random_pairs_rigid_orientation_100 \
  --problem-file scripts/ffw_sg2_random_pairs_rigid_orientation_100.json \
  --rigid-orientation \
  --no-print-path
```

For the 18-DoF mobility model, the reproducible benchmark in
`scripts/ffw_sg2_mobility_random_pairs_com_axis_100.json` varies base
`x/y/yaw`, lift, and arm joints. Every endpoint satisfies the 8-dimensional
dual-arm-plus-axis equality constraint and the CoM support inequalities for a
25 kg object with a 0.07 m support margin. Generate and evaluate it with:

```bash
cmake --build --preset patacon --target generate_ffw_sg2_mobility_random_pairs
./build_patacon/generate_ffw_sg2_mobility_random_pairs

./build_patacon/evaluate_mbm ffw_sg2_mobility mobility_random_pairs_com_axis_100 \
  --problem-file scripts/ffw_sg2_mobility_random_pairs_com_axis_100.json \
  --com \
  --rigid-orientation \
  --no-print-path
```

### G1 random start-goal benchmark without rigid orientation

`scripts/g1_random_pairs_no_rigid_orientation_100.json` contains 100
reproducible 35-DoF G1 start-goal pairs. Each endpoint is projected onto the
20-dimensional base constraint (fixed feet, payload-aware CoM support, and
bimanual relative pose), while the two rigid-orientation axis rows remain
disabled. Robot, shelf, and attached-object collision checks are applied to
every endpoint.

The payload center varies inside `[-0.0535, 0.0665] x [0.2684, 0.4284] x
[0.5589, 0.6789]` m at the start and inside the shelf task region `[0.3471,
0.4471] x [-0.0947, 0.0653] x [0.8400, 0.9600]` m at the goal. Regenerate and
evaluate the set without `--rigid-orientation`:

```bash
cmake --build --preset patacon --target generate_g1_random_pairs
./build_patacon/generate_g1_random_pairs

./build_patacon/evaluate_mbm g1 random_pairs_no_rigid_orientation_100 \
  --problem-file scripts/g1_random_pairs_no_rigid_orientation_100.json \
  --no-print-path
```

For the rigid-orientation variant,
`scripts/g1_random_pairs_rigid_orientation_100.json` adds two axis residuals
that keep the carried object's local +Z axis parallel to world Z while leaving
yaw free. This gives a 22-dimensional total constraint, 20 equality rows, and
a 15-dimensional tangent space. Generate and evaluate it with:

```bash
./build_patacon/generate_g1_random_pairs --rigid-orientation

./build_patacon/evaluate_mbm g1 random_pairs_rigid_orientation_100 \
  --problem-file scripts/g1_random_pairs_rigid_orientation_100.json \
  --rigid-orientation \
  --no-print-path
```

### MuJoCo visualization

Install the visualization dependencies, including SciPy and TOPP-RA, if they
are not already available:

```bash
python3 -m pip install -r requirements-visualization.txt
```

`single_mbm --visualize` plans the selected problem and replays the returned
start-to-goal path. It supports Franka single/dual, both FFW-SG2 planner
models, G1, and IGRIS-C:

```bash
./build_patacon/single_mbm franka_single demo 1 --visualize
./build_patacon/single_mbm franka demo 1 --visualize
./build_patacon/single_mbm ffw_sg2 tray_lift 1 --visualize
./build_patacon/single_mbm ffw_sg2_mobility tray_lift 1 --visualize
./build_patacon/single_mbm g1 humanoid_shelf 1 --visualize
./build_patacon/single_mbm igris_c igris_c_shelf_lift 1 --visualize
```

Interactive viewers repeat the start-to-goal trajectory until their window is
closed. MP4 export renders one traversal instead.

For problem sets, `evaluate_mbm --visualize` cycles every solved run in one
window for Franka single/dual, G1, fixed-base FFW-SG2, and mobile FFW-SG2.
Each transition resets MuJoCo and applies that problem's start state and
attached-object frame offset. G1 problem-set replay explicitly uses `qpos`;
single-run G1 visualization keeps its existing torque-PD default. IGRIS-C is
not enabled for `evaluate_mbm --visualize`.

For continuous G1 replanning with the PATACON planner, run:

```bash
./build_patacon/single_mbm g1 humanoid_shelf 1 \
  --replanning --no-print-path
```

`--replanning` is currently supported only for one G1 run and automatically
opens the dedicated CTRL-mode MuJoCo visualizer. The first path and every
subsequent path are produced by this repository's PATACON planner. A persistent
planner process stays alive between requests, and each replanning request has a
five-second planning limit. The initial plan uses `--seed` (default `1`), and
continuous replanning requests use consecutive seeds beginning at `seed + 1`;
each candidate prints its seed.
In PATACON mode, the process also keeps the G1 tree,
tangent-space, RNG/Halton, collision-scratch, and pinned signaling allocations
alive across requests. Search state is reset for each request, and the workspace
is released once when the replanning server exits. The visualizer predicts the
continuous commanded state at least 0.10 seconds ahead on the active path. It
adapts that lookahead to the observed planner latency plus the 0.10-second
tracking-stability window, then switches to the accepted replacement path at
the scheduled handoff. This latency-driven handoff allows faster planners to
produce a correspondingly higher path-update rate. Every accepted handoff
prints the running `update_hz_sim` and `update_hz_wall`; these count paths that
were actually applied, not merely planner responses. Replanning replay runs at
0.3 times the normal path speed.
The CTRL follower uses the path's desired joint velocity with MuJoCo's implicit
joint damping and adds world-position feedback for the attached payload. It
uses a default joint gain scale of `16.0` and payload position/velocity gains
of `600/70`, tuned against the replanning start-to-goal and return motion. It
also removes sub-0.003 planning-coordinate duplicate nodes and lengthens short
segments for a 4.0 planning-coordinate/s^2 timing limit, preventing tiny raw
planner edges from producing large command-acceleration spikes.

The red spherical obstacle has an `0.08 m` rendered/physical diameter and an
`0.04 m` planning collision radius (`0.08 m` diameter). It starts at
`(0.400, 0.200, 0.800)` m
and moves at `0.05 m/s` along the world Y axis using offsets from `0.000` m to
`-0.200` m, so its absolute Y range is `y=0.200` m to `y=0.000` m while
keeping `x=0.400` m and `z=0.800` m fixed. It pauses for `2.0 s` at each Y-axis
endpoint before reversing direction. Its
current position is captured as a static obstacle at each request; obstacle
velocity and future motion are not predicted. During `--replanning`, shelf and
other fixed-world collision is removed from both PATACON and MuJoCo. The floor
and red sphere remain physical and collision-checked; each planning request
replaces the sphere's previous position with the current mocap position. The
payload center moves vertically from `(0.3, 0.0, 0.5)` m to
`(0.3, 0.0, 1.0)` m at a fixed orientation. The carried object's active path is shown as one connected green
line (the existing prefix through the handoff plus the newly planned suffix),
and its actual motion trail is blue. When the active path completes, the
controller holds its last CTRL target for one second and reverses the start and
goal regardless of the remaining physical tracking error.

The regular `--visualize` path is unchanged and continues to use
`scripts/visualize_g1.py`; continuous replanning uses
`scripts/visualize_g1_replanning.py`.

Before replay, every supported robot automatically runs the same constrained
path-shortcut pass. A shortcut is accepted only after the robot-specific CUDA
projection, joint-limit, self-collision, attached-object collision, and
environment-collision checks succeed and its configuration-space arclength is
lower. This post-processing changes only the visualization copy of the final
run; benchmark statistics and saved planner-result JSON retain the raw planner
path. The console prints attempted/accepted shortcut counts and the before/after
cost.

The automatic visualization pipeline is:

```text
simplified path
  -> degree-5 quintic Hermite spline (SciPy BPoly, Bernstein basis; not BSpline)
  -> full CUDA projection/joint-limit/collision revalidation
  -> TOPP-RA velocity/acceleration parameterization
  -> MuJoCo replay
```

To inspect the planner output without that post-processing, combine
`--no-path-smoothing` with `--visualize`:

```bash
./build_patacon/single_mbm igris_c igris_c_shelf_lift 1 \
  --visualize --no-path-smoothing
```

This skips the shortcut pass, quintic Hermite construction and full spline
revalidation, and TOPP-RA for every robot supported by `--visualize`. The raw
planner waypoint polyline is sampled linearly at constant velocity within each
segment only to produce replay frames. Velocity changes can therefore be
discontinuous at waypoints, and no acceleration limit or time-optimality is
enforced. This option does not alter benchmark statistics or saved planner
JSON, and is unrelated to `--no-waypoint-smoothing`, which controls projection
behavior inside the constrained planner.

The Hermite spline is C2 and matches position, first derivative, and second
derivative data at every knot. If its initial tangents leave the valid
constraint manifold, their scale is reduced and the complete sampled spline is
revalidated. The accepted projection displacement uses the planner's own
`projection_task_tolerance` as its scale. Both the projected configurations and
the original, unprojected spline samples are collision-checked; the latter are
also checked against joint limits. TOPP-RA runs only after a spline has passed
revalidation. A spline or TOPP-RA failure aborts visualization instead of
silently returning to the old cubic trajectory.

The fixed-base FFW visualizer loads `ffw_lift/ffw_sg2_lift.xml`; the mobile
visualizer loads
`ffw_lift/ffw_sg2_rack_upper_to_lower.xml`. Joint values are mapped to MuJoCo
`qpos` entries by name.

Real-dynamics replay is available only for the mobile FFW-SG2 and IGRIS-C and
must be combined with `--visualize`:

```bash
./build_patacon/single_mbm ffw_sg2_mobility tray_lift 1 \
  --visualize --real --object-mass-kg 25 --support-margin 0.07

./build_patacon/single_mbm igris_c igris_c_shelf_lift 1 --visualize --real
```

Use the `--real-*` options listed above to tune replay speed, settling, mobile
base tracking, and payload behavior. Without `--real`, these two robots use
kinematic trajectory replay. G1 visualization defaults to torque-PD actuator
control through MuJoCo `ctrl`; set `PRRTC_G1_CONTROL_MODE=qpos` to use direct
kinematic replay instead. Playback continues until the MuJoCo window is closed.

Render an MP4 directly from MuJoCo instead of recording the viewer window by
setting `PRRTC_VIDEO` when invoking `single_mbm`. This works for every robot
supported by `--visualize`, including `--real` playback:

```bash
PRRTC_VIDEO=logs/simul/franka_single/franka_single_demo.mp4 \
PRRTC_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm franka_single demo 1 --visualize --rigid-orientation

PRRTC_VIDEO=logs/simul/franka_dual/franka_demo.mp4 \
PRRTC_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm franka demo 1 --visualize

PRRTC_VIDEO=logs/simul/franka_dual_rigid/franka_demo.mp4 \
PRRTC_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm franka demo 1 --visualize --rigid-orientation

PRRTC_VIDEO=logs/simul/ffw_sg2/ffw_sg2_tray_lift.mp4 \
PRRTC_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm ffw_sg2 tray_lift 1 --visualize

PRRTC_VIDEO=logs/simul/ffw_sg2_rigid/ffw_sg2_tray_lift.mp4 \
PRRTC_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm ffw_sg2 tray_lift 1 --visualize --rigid-orientation

PRRTC_VIDEO=logs/simul/ffw_sg2_mobile/ffw_sg2_mobility_tray_lift.mp4 \
PRRTC_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm ffw_sg2_mobility tray_lift 1 --visualize --rigid-orientation --com

PRRTC_VIDEO=logs/simul/g1/g1_humanoid_shelf.mp4 \
PRRTC_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm g1 humanoid_shelf 1 --visualize

PRRTC_VIDEO=logs/simul/g1_rigid/g1_humanoid_shelf.mp4 \
PRRTC_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm g1 humanoid_shelf 1 --visualize --rigid-orientation

PRRTC_VIDEO=logs/simul/igris/igris_c_shelf_lift.mp4 \
PRRTC_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm igris_c igris_c_shelf_lift 1 --visualize

PRRTC_VIDEO=logs/simul/replanning/g1_humanoid_shelf.mp4 \
PRRTC_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm g1 humanoid_shelf 1   --replanning --no-print-path

```

To save synchronized videos from four camera directions, add
`PRRTC_VIDEO_VIEWS=all`:

```bash
PRRTC_VIDEO=logs/simul/franka_single_demo.mp4 \
PRRTC_VIDEO_VIEWS=all \
  ./build_patacon/single_mbm franka_single demo 1 --visualize
```

This creates:

```text
logs/simul/franka_single_demo_front.mp4
logs/simul/franka_single_demo_front_left.mp4
logs/simul/franka_single_demo_front_right.mp4
logs/simul/franka_single_demo_overhead.mp4
```

Use a comma-separated subset such as
`PRRTC_VIDEO_VIEWS=front,left,right,rear` when only selected directions are
needed. `all` selects the generally unobstructed `front`, `front_left`,
`front_right`, and `overhead` views; the side and rear views remain available
explicitly. With no `PRRTC_VIDEO_VIEWS`, the original single front-view output
path is used unchanged. Multi-view rendering advances the trajectory or
physics only once and records every camera at the same simulation frame.

The robot-specific aliases `PRRTC_FRANKA_VIDEO`, `PRRTC_FFW_SG2_VIDEO`,
`PRRTC_G1_VIDEO`, and `PRRTC_IGRIS_C_VIDEO` are also accepted. All MP4 paths
require the `ffmpeg` executable. The visualizers render one start-to-goal pass,
include short start/end holds, create missing output directories, and then
exit. Video duration is determined from the generated frame count and `--fps`
(60 by default), not wall-clock rendering speed.

Each visualizer also accepts `--video OUTPUT.mp4`, `--video-views`,
`--video-width`, and `--video-height` when invoked directly. For example, a
saved G1 planner result can be rendered without rerunning the planner:

```bash
python3 scripts/replay_g1_result.py \
  --environment resources/g1/g1_humanoid_shelf.xml \
  --result traces/g1_humanoid_shelf_1_result.json \
  --video logs/g1_humanoid_shelf.mp4 \
  --video-views all --fps 60 --video-width 1280 --video-height 720
```

Unless `--no-path-smoothing` is used, all robot visualizers reconstruct the
validated C2 quintic Hermite geometric path and pass it to TOPP-RA with zero
initial and final path velocity. Default playback limits are fixed at 2x
without an extra command-line option: Franka, FFW-SG2, and IGRIS-C use a `1.4`
per-coordinate velocity limit and a `5.6` acceleration limit. Real-dynamics
FFW-SG2 and IGRIS-C use a lower default velocity limit of `0.20`.

For G1, the 29 actuated-joint velocity limits are loaded by name from
`g1_29dof.urdf` and doubled for playback; use `--urdf PATH` to select a
different matching description. The six floating-base coordinates use a
doubled `1.4` m/s or rad/s fallback because they have no URDF joint limits.
`--velocity-scale` is a scaling factor in `(0, 1]` for these doubled velocity
limits and defaults to `1.0`; `--speed` remains an alias. A scale of `0.5`
restores the original URDF-rate playback. `PRRTC_G1_VELOCITY_SCALE` sets the
same value when G1 is launched indirectly through `single_mbm --visualize`.
The URDF has no acceleration limits, so `--acceleration` is an explicit
per-coordinate fallback and defaults to `5.6` m/s^2 or rad/s^2. Both options
are optional:

```bash
python3 scripts/replay_g1_result.py \
  --environment resources/g1/g1_humanoid_shelf.xml \
  --result traces/g1_humanoid_shelf_1_result.json
```

Specify `--velocity-scale` to slow G1 below the fixed 2x playback rate, or
`--acceleration` to override a visualizer's acceleration fallback. For every
robot launched through `single_mbm --visualize` without
`--no-path-smoothing`, the geometric spline is constraint- and
collision-revalidated before TOPP-RA. The time parameterizer changes only
traversal speed and does not replace the validated geometric path.

Direct trajectory invocation of `visualize_franka.py`,
`visualize_ffw_sg2.py`, `sim_ffw_sg2_rack_upper_to_lower.py`,
`visualize_g1.py`, and `visualize_igris_c.py` accepts the same video options.

### FFW-SG2 mobility attached object collision

`ffw_sg2_mobility` problem entries can include `attached_object_collision`.
The object is represented by sphere proxies in the attached object frame. Its
frame is reconstructed from the midpoint and orientation of the two gripper
sites, with `world_offset` interpreted in that attached object frame so it moves
with the grippers during replay.
The planner checks these object spheres against world collision geometry and
against all robot spheres except the contact links.
The bundled `ffw_sg2_mobility` `tray_lift` problem uses `box_sphere_grid` to
generate a dense conservative sphere cover of the MuJoCo `object_collision`
box. The generated spheres are placed at voxel centers, and each radius is the
voxel half-diagonal plus `radius_padding`, so the sphere union covers the full
box instead of leaving shell gaps.

Example:

```json
"attached_object_collision": {
  "enabled": true,
  "world_offset": [0.1, 0.0, 0.0],
  "box_sphere_grid": {
    "half_extents": [0.12, 0.11, 0.06],
    "counts": [7, 7, 5],
    "radius_padding": 0.001
  },
  "contact_links": [
    "gripper_l_rh_p12_rn_base",
    "gripper_l_rh_p12_rn_r1",
    "gripper_l_rh_p12_rn_r2",
    "gripper_l_rh_p12_rn_l1",
    "gripper_l_rh_p12_rn_l2",
    "gripper_r_rh_p12_rn_base",
    "gripper_r_rh_p12_rn_r1",
    "gripper_r_rh_p12_rn_r2",
    "gripper_r_rh_p12_rn_l1",
    "gripper_r_rh_p12_rn_l2"
  ]
}
```

If `contact_links` is omitted, both grippers are used as the default contact
set. Advanced cases can add `ignored_robot_spheres` or
`ignored_robot_approx_spheres` with explicit runtime sphere indices.

### CUDA validation commands

Build all robot-specific validation executables with:

```bash
cmake --build --preset patacon --target \
  validate_ffw_sg2_projection \
  validate_franka_constraints validate_franka_collision \
  validate_g1_constraints validate_g1_collision \
  validate_igris_c_kinematics validate_igris_c_constraints \
  validate_igris_c_collision
```

Run the FFW-SG2 and Franka checks with:

```bash
./build_patacon/validate_ffw_sg2_projection

./build_patacon/validate_franka_constraints
./build_patacon/validate_franka_collision franka_single
./build_patacon/validate_franka_collision franka
```

`validate_franka_collision` accepts an optional problem JSON as its second
argument. The first argument must be `franka_single` or `franka`.

Run the G1 checks with:

```bash
./build_patacon/validate_g1_constraints
./build_patacon/validate_g1_constraints --rigid-orientation

./build_patacon/validate_g1_collision
./build_patacon/validate_g1_collision scripts/g1_problems.json --rigid-orientation
```

`validate_g1_collision` accepts the problem JSON as its first argument, so the
path must be present when enabling `--rigid-orientation`.

Run the IGRIS-C checks with:

```bash
./build_patacon/validate_igris_c_kinematics
./build_patacon/validate_igris_c_constraints
./build_patacon/validate_igris_c_collision
```

The optional first argument overrides
`resources/igris_c/kinematics_reference.json` for the kinematics check and
`scripts/igris_c_problems.json` for either constraint or collision check.

### Tree trace HTML

Export the complete bidirectional planning tree to JSON, GraphML, and a
PATACON-style self-contained HTML viewer with:

```bash
./build_patacon/single_mbm ffw_sg2 tray_lift 1 \
  --trace-mode tree \
  --html-trace-mode tree \
  --html-max-tree-nodes 0
```

Outputs are written under `traces/` unless `--save-json`, `--graphml`, or
`--html` supplies an explicit path. A positive
`--html-max-tree-nodes` samples the HTML view while retaining the full tree in
JSON and GraphML; zero embeds every tree node and can produce a large file.
The exporter searches `PATACON_ROOT`, a sibling `patacon` checkout, and
`~/gh_ws/tb_rrt_ws/src/patacon`, or accepts `--patacon-root PATH`.

The [MotionBenchMaker](https://github.com/KavrakiLab/motion_bench_maker) JSON files are generated using the script detailed [here](https://github.com/KavrakiLab/vamp/blob/35080be604aabd4373cc7db8608297afaa446878/resources/README.md#motionbenchmaker-problems).

## Planner Configuration
pRRTC has the following parameters which can be modified in the benchmarking scripts:
- <ins>**max_samples**</ins>: maximum number of samples in trees
- <ins>**max_iters**</ins>: maximum number of planning iterations
- <ins>**num_new_configs**</ins>: amount of new samples generated per iteration
- <ins>**range**</ins>: maximum RRT-Connect extension range
- <ins>**granularity**</ins>: number of discretized motions along an edge during collision checking. Note: this parameter must match the BATCH_SIZE parameter in the robot's collision header (for example, `ffw_sg2.cuh`) for correct results.
- <ins>**balance**</ins>: whether to enable tree balancing -- 0 for no balancing; 1 for distributed balancing where each iteration may generate samples for one or two trees; 2 for single-sided balancing where each iteration generate samples for one tree only
- <ins>**tree_ratio**</ins>: the threshold for distinguishing which tree is smaller in size -- if balance set to 1, then set tree_ratio to 0.5; if balance set to 2, then set tree_ratio to 1
- <ins>**dynamic_domain**</ins>: whether to enable [dynamic domain sampling](https://ieeexplore.ieee.org/abstract/document/1570709) -- 0 for false; 1 for true
- <ins>**dd_alpha**</ins>: extent to which each radius is enlarged or shrunk per modification
- <ins>**dd_radius**</ins>: starting radius for dynamic domain sampling
- <ins>**dd_min_radius**</ins>: minimum radius for dynamic domain sampling

## Adding a Robot
### Generating FKCC kernels
1. Use [Foam](https://github.com/CoMMALab/foam) to generate two spherized urdfs:
- one with approximate geometry, i.e. 1 sphere per link. Ex. [here](https://github.com/CoMMALab/cricket/blob/gpu-cc-early-exit/resources/panda/panda_spherized_1.urdf)
- one with fine geometry. Ex. [here](https://github.com/CoMMALab/cricket/blob/gpu-cc-early-exit/resources/panda/panda_spherized_1.urdf)

2. Clone [Cricket](https://github.com/CoMMALab/cricket.git) and switch to the `gpu-cc-early-exit` branch.

3. Create a folder under `resources/<robot name>`

4. Add the two spherized urdfs and the robot srdf file to this folder.

5. Create a json config file for approximate fkcc kernel generation. Ex. `resources/robot_approx.json`:
```
{
    "name": "Robot",
    "urdf": "robot/robot_spherized_approx.urdf",
    "srdf": "robot/robot.srdf",
    "end_effector": "robot_grasptarget",
    "batch_size": 16,
    "template": "templates/prrtc_approx_template.hh",
    "subtemplates": [],
    "output": "robot_prrtc_approx.hh"
}
```
Make sure to reference the approximate urdf. Batch size should be equal to the number of discretized collision checks on each extension of pRRTC.

6. Repeat step 5 and create a config file for the main fkcc generation. Ex. `resources/robot_main.json`. See the FFW-SG2 config files for examples.

7. After building cricket run the script `gpu_fkcc_gen.sh robot`. This will put the generated code into a file `robot_fk.hh`.

8. Add this to pRRTC as `src/robot/robot.cuh`, and include it in `src/planning/pRRTC.cu`.

### Integrating the generated code
9. Add a template instantiation for your robot to the bottom of `src/planning/pRRTC.cu`.
10. Add your robot to `src/planning/Robots.hh`.

    a. Generate the robot struct from cricket with `build/fkcc_gen robot_struct.json`.
    Ex. config file `robot_struct.json`:
    ```
    {
        "name": "robot",
        "urdf": "robot/robot_spherized.urdf",
        "srdf": "robot/robot.srdf",
        "end_effector": "robot_grasptarget",
        "resolution": 32,
        "template": "templates/prrtc_robot_template.hh",
        "subtemplates": [],
        "output": "robot_struct.hh"
    }
    ```

    b. copy the generated struct into `src/planning/Robots.hh`.
11. Recompile pRRTC.

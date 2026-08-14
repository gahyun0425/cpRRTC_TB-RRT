# FFW-SG2 planning model

This directory contains the model inputs and reproducible generation support for
the FFW-SG2 integration in pRRTC. Phases 1 and 2 create the planning and
collision models; Phase 3 adds the generated CUDA implementation to the planner.

## Source and generated files

- Source URDF: `../../ffw_lift/ffw_sg2.urdf` (kept unchanged)
- Planning URDF: `ffw_sg2_planning.urdf` (generated)
- Self-collision semantics: `ffw_sg2.srdf`
- Generator: `prepare_planning_urdf.py`
- Fine collision data: `ffw_sg2_fine_spheres.json`
- Fine sphere URDF: `ffw_sg2_spherized.urdf` (generated)
- Conservative approximate URDF: `ffw_sg2_spherized_approx.urdf` (generated)
- Collision-model metadata: `collision_model_metadata.json` (generated)
- Collision-model generator: `prepare_collision_models.py`
- Cricket output validator/postprocessor: `postprocess_cricket_header.py`
- Integrated CUDA collision implementation: `../../src/robots/ffw_sg2.cuh`
- 8-DoF model generator: `prepare_single_arm_models.py`
- 8-DoF Cricket postprocessor: `postprocess_cricket_single_header.py`
- Integrated 8-DoF CUDA implementation: `../../src/robots/ffw_sg2_single.cuh`

Regenerate the planning URDF from the repository root with:

```bash
python3 resources/ffw_sg2/prepare_planning_urdf.py
```

The generator removes visual and inertial elements, keeps collision geometry,
changes robot mesh paths to repository-relative paths, and fixes non-planning
joints at their URDF zero pose. The camera visual mesh is therefore not needed;
its collision box remains in the planning model.

## Canonical 15-DoF configuration

The pRRTC configuration vector must use this exact order:

```text
[lift_joint,
 arm_l_joint1, arm_l_joint2, arm_l_joint3, arm_l_joint4,
 arm_l_joint5, arm_l_joint6, arm_l_joint7,
 arm_r_joint1, arm_r_joint2, arm_r_joint3, arm_r_joint4,
 arm_r_joint5, arm_r_joint6, arm_r_joint7]
```

Joint limits in the same order are:

```text
lower = [-0.5,
         -3.14, 0.0, -3.14, -2.9361, -3.14, -1.57, -1.8201,
         -3.14, -3.14, -3.14, -2.9361, -3.14, -1.57, -1.5804]
upper = [ 0.0,
          3.14, 3.14, 3.14, 1.0786, 3.14, 1.57, 1.5804,
          3.14, 0.0, 3.14, 1.0786, 3.14, 1.57, 1.8201]
```

The head, both grippers, and all wheel steering/drive joints are fixed at
`q = 0`. This makes the planning model deterministic and prevents those joints
from silently increasing the pRRTC state dimension.

## Right-arm-only 8-DoF model

The `ffw_sg2_single` configuration vector uses this exact order:

```text
[lift_joint,
 arm_r_joint1, arm_r_joint2, arm_r_joint3, arm_r_joint4,
 arm_r_joint5, arm_r_joint6, arm_r_joint7]
```

The seven left-arm joints are fixed at `q = 0`, but all 124 fine and 27
approximate spheres are retained. The fixed left arm therefore remains a
self-collision obstacle for the moving lift/right arm. The 19 world cuboids and
projected `tray_lift` start/goal are also retained from the 15-DoF problem.

Regenerate the 8-DoF planning/collision URDFs, problem file, and exact memory
metadata with:

```bash
python3 resources/ffw_sg2/prepare_single_arm_models.py
```

Its Cricket inputs are `../ffw_sg2_single_main.json`,
`../ffw_sg2_single_approx.json`, and `../ffw_sg2_single_struct.json`. After
Cricket produces `ffw_sg2_single_fk.hh`, reproduce the integrated header with:

```bash
python3 resources/ffw_sg2/postprocess_cricket_single_header.py \
  --input /path/to/cricket/ffw_sg2_single_fk.hh
```

At `batch_size = 16`, the 8-DoF model-dependent shared-memory total is 30,592
bytes: 23,808 bytes for fine positions, 5,184 for approximate positions, 576
for joint flags, and 1,024 for transforms. The reduced joint-flag stride (9)
and one transform slot are generated from the reduced kinematic structure,
rather than reusing the larger 15-DoF allocation.

## Self-collision policy

The SRDF disables only adjacent links and known rigid-subassembly pairs. In
particular, left-arm/right-arm collision pairs remain enabled. The list is an
initial conservative policy and must be checked with representative poses during
the CUDA integration and validation phase.

## Fine and approximate collision models

The fine sphere set is imported from the existing FFW-SG2 cuRobo collision
model. Its expected source SHA-256 is recorded in the generator, and the
normalized JSON is checked into this directory so normal regeneration has no
dependency on another workspace.

Regenerate both collision URDFs and the memory metadata with:

```bash
python3 resources/ffw_sg2/prepare_collision_models.py
```

The fine model contains 124 spheres over 27 links. The approximate model has
exactly one sphere for each of those 27 links. Each approximate sphere contains
all fine spheres on the same link, with a `1e-6 m` numeric margin. Therefore an
approximate collision-free result can safely skip the fine test relative to the
fine sphere model. This does not prove that the original mesh is fully covered
by the inherited cuRobo sphere set.

The inherited model intentionally omits the fixed base, six wheel links,
`head_link1`, and both camera links. Fine and approximate models use the same
27-link set so the early-exit test cannot disagree merely because one level has
additional links.

At the all-zero 15-DoF pose, the fine model has no unignored self-collision.
The conservative approximate model reports 33 broad-phase candidate link pairs,
mostly because the single sphere enclosing the long `lift_link` overlaps arm
bounding spheres. These are safe false positives: they trigger the fine test
rather than accepting an invalid state. Runtime benchmarking must measure this
fallback rate. Reducing it safely would require splitting the long lift into
multiple fixed collision segments, which is a deliberate deviation from the
one-sphere-per-original-link policy and is not done here.

At `batch_size = 16`, the model requires 5,952 fine position floats, 1,296
approximate position floats, 256 joint-flag integers, and 512 transform floats.
The model-dependent shared-memory total is 32,064 bytes. This is 6,544 bytes
less than the former fixed buffers (38,608 bytes). Exact counts are written to
`collision_model_metadata.json`.

## Cricket generation configs

The following files mirror Cricket's Panda/Baxter pRRTC resource layout:

- `../ffw_sg2_main.json`
- `../ffw_sg2_approx.json`
- `../ffw_sg2_struct.json`

All three use `batch_size: 16`, the same value as Panda, Fetch, and Baxter and
the required pRRTC edge granularity. The selected end-effector is the left
gripper base; as with Baxter, this does not remove the other arm from the
branched kinematic model.

The CUDA code was generated from CoMMALab/Cricket's `gpu-cc-early-exit` branch
at commit `98582c35d81c6ed0d8c4badb7fdf78327523524c`. The raw combined header has
SHA-256 `1d07485ad5763a55bcbc5a3baaf5ffc81a74cdb130a58dc098448b582bd31950`.
After Cricket generates `ffw_sg2_fk.hh`, validate and reproduce the checked-in
header from the pRRTC repository root with:

```bash
python3 resources/ffw_sg2/postprocess_cricket_header.py \
  --input /path/to/cricket/ffw_sg2_fk.hh
```

The validator checks the 124/27 sphere counts, 16 generated joint indices,
two transform slots, and every hard-coded Cricket stride site before replacing
the raw `20 * batch_ind` expressions with the FFW-SG2 stride of 16.

The planner uses `RobotCollisionTraits.hh` to size all four model-dependent
shared buffers at compile time. Panda and Fetch use one transform slot; Baxter
and FFW-SG2 use two. FFW-SG2 allocates a 16-entry flag slice per configuration,
while the existing generated robot headers retain their required 20-entry
stride. `solve()` rejects a granularity other than the generated batch size so
the CUDA indexing contract cannot silently diverge.

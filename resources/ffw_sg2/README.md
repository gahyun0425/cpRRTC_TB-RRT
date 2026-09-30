# FFW-SG2 planning model

This directory contains the PATACON adapters and generated collision inputs for
the FFW-SG2 integration. The robot description itself comes from ROBOTIS'
official `ai_worker` repository, checked out as a submodule inside PATACON.

## Source and generated files

- Upstream repository: `../../ai_worker` (`jazzy`, pinned by the submodule
  gitlink)
- Upstream package: `../../ai_worker/ffw_description` version `2.2.8`
- Source URDF:
  `../../ai_worker/ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf`
- Source MuJoCo model:
  `../../ai_worker/ffw_description/mujoco/ffw_sg2/ffw_sg2.xml`
- Planning URDF: `ffw_sg2_planning.urdf` (generated)
- Self-collision semantics: `ffw_sg2.srdf`
- Planning generator: `prepare_planning_urdf.py`
- Fine collision data: `ffw_sg2_fine_spheres.json`
- Fine sphere URDF: `ffw_sg2_spherized.urdf` (generated)
- Conservative approximate URDF: `ffw_sg2_spherized_approx.urdf` (generated)
- Collision-model metadata: `collision_model_metadata.json` (generated)
- Collision-model generator: `prepare_collision_models.py`
- PATACON-only MuJoCo task overlays: `mujoco/*.xml`
- MuJoCo task-scene generator: `prepare_mujoco_scenes.py`
- Generated task scenes: `../../ffw_lift/ffw_sg2_lift.xml` and
  `../../ffw_lift/ffw_sg2_rack_upper_to_lower.xml`
- Cricket output validator/postprocessor: `postprocess_cricket_header.py`
- Integrated CUDA collision implementation: `../../src/robots/ffw_sg2.cuh`

Initialize the official source and regenerate all description-derived files
from the repository root with:

```bash
git submodule update --init ai_worker
python3 resources/ffw_sg2/prepare_planning_urdf.py
python3 resources/ffw_sg2/prepare_collision_models.py
python3 resources/ffw_sg2/prepare_mujoco_scenes.py
```

The planning generator removes visual and inertial elements, keeps collision
geometry, removes Gazebo/transmission/control data, changes robot mesh paths to
paths inside PATACON, and fixes non-planning joints at their URDF zero pose. The
MuJoCo generator combines the official robot with PATACON's task-only overlays:
the visualization floor, attached object, and optional CoM markers. No FFW-SG2
generation step reads `cptbrrt_pkg`, VAMP, or another workspace.

FFW-SG2 world geometry has one source of truth: the selected planning problem's
`sphere`, `cylinder`, and `box` arrays. Both planner frontends copy these arrays
into the replay document. The standard replay and real-dynamics simulator use
the shared `scripts/mujoco_primitive_environment.py` adapter to validate and
insert the primitives into MuJoCo at runtime. The generated XML files contain
no hardcoded kitchen furniture or racks. Evaluation replay requires every
bundled trajectory to use the same primitive environment.

## Canonical 15-DoF configuration

The PATACON configuration vector must use this exact order:

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
         -3.14, 0.0, -3.14, -2.9361, -3.14, -1.57, -1.5804,
         -3.14, -3.14, -3.14, -2.9361, -3.14, -1.57, -1.8201]
upper = [ 0.0,
          3.14, 3.14, 3.14, 1.0786, 3.14, 1.57, 1.8201,
          3.14, 0.0, 3.14, 1.0786, 3.14, 1.57, 1.5804]
```

The head, both grippers, and all wheel steering/drive joints are fixed at
`q = 0`. This makes the planning model deterministic and prevents those joints
from silently increasing the PATACON state dimension.

## Self-collision policy

The SRDF disables only adjacent links and known rigid-subassembly pairs. In
particular, left-arm/right-arm collision pairs remain enabled. The list is an
initial conservative policy and must be checked with representative poses during
the CUDA integration and validation phase.

## Fine and approximate collision models

The canonical fine sphere set is the checked-in
`ffw_sg2_fine_spheres.json`. The generator has no external import option and
normal regeneration depends only on files within PATACON. The JSON retains the
SHA-256 of its historical source as provenance, not as an input path.

Regenerate both collision URDFs and the memory metadata with:

```bash
python3 resources/ffw_sg2/prepare_collision_models.py
```

The fine model contains 124 spheres over 27 links. The approximate model has
exactly one sphere for each of those 27 links. Each approximate sphere contains
all fine spheres on the same link, with a `1e-6 m` numeric margin. Therefore an
approximate collision-free result can safely skip the fine test relative to the
fine sphere model. This does not prove that the original mesh is fully covered
by the local sphere abstraction.

The local sphere abstraction intentionally omits the fixed base, six wheel
links, `head_link1`, `camera_left_link`, `camera_right_link`, and
`zed_camera_center`. Fine and approximate models use the same 27-link set so
the early-exit test cannot disagree merely because one level has additional
links.

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

The following files configure Cricket code generation:

- `../ffw_sg2_main.json`
- `../ffw_sg2_approx.json`
- `../ffw_sg2_struct.json`

All three use `batch_size: 16`, matching the required PATACON edge granularity.
The selected end-effector is the left gripper base; this does not remove the
other arm from the branched kinematic model.

The CUDA code was generated from CoMMALab/Cricket's `gpu-cc-early-exit` branch
at commit `98582c35d81c6ed0d8c4badb7fdf78327523524c`. The raw combined header has
SHA-256 `dbd1af2d4375affad4b51bcea53c8dfd0a97fda492f0e5f873e8287d095c1a9f`.
After Cricket generates `ffw_sg2_fk.hh`, validate and reproduce the checked-in
header from the PATACON repository root with:

```bash
python3 resources/ffw_sg2/postprocess_cricket_header.py \
  --input /path/to/cricket/ffw_sg2_fk.hh
```

The validator checks the 124/27 sphere counts, 16 generated joint indices,
two transform slots, and every hard-coded Cricket stride site before replacing
the raw `20 * batch_ind` expressions with the FFW-SG2 stride of 16. When the
checked-in output already exists, it also preserves PATACON's two custom
early-exit collision functions while replacing the generated core.

The planner uses `RobotCollisionTraits.hh` to size all four model-dependent
shared buffers at compile time. FFW-SG2 uses two transform slots and allocates
a 16-entry flag slice per configuration. `solve()` rejects a granularity other
than the generated batch size so the CUDA indexing contract cannot silently
diverge.

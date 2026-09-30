# Official Franka FER description adapter

PATACON uses the official
[`frankarobotics/franka_description`](https://github.com/frankarobotics/franka_description)
repository as its Franka source. The submodule is pinned to release `2.9.0`
at commit `7aeeddc449edf8d62b594f9e36a81da53e7796f9`. The upstream submodule is
kept unmodified.

Initialize and verify it from the PATACON root:

```bash
git submodule update --init franka_description
python3 resources/franka/prepare_description.py --check
```

To regenerate the committed adapters after an intentional source update:

```bash
python3 resources/franka/prepare_description.py
python3 resources/franka/prepare_description.py --check
```

The generator expands the official FER Xacro with `franka_hand`, then creates:

- `fer_single.urdf`: one arm mounted at `(0, 0, 0)`;
- `fer_dual.urdf`: PATACON's parallel arms at `(0, +0.2, 0.6)` and
  `(0, -0.2, 0.6)` m;
- `mujoco/fer_single.xml` and `mujoco/fer_dual.xml`: visualization adapters
  that reference the cloned official meshes directly and contain no task
  furniture or world obstacles;
- `model_metadata.json`: the pinned revision, source hashes, generated hashes,
  joint contract, mount positions, TCP offset, and sphere counts.

The official single-arm joint names are namespaced as `fer0_joint1` through
`fer0_joint7`; the second arm uses `fer1_joint1` through `fer1_joint7`.
Planning remains 7/14 dimensional, while the two gripper finger joints per arm
are visualization-only coordinates. The task frame is the official
`franka_hand` TCP at `0.1034 m` from the hand frame.

The official visual meshes are Collada files, which the installed MuJoCo
runtime does not decode. The MuJoCo adapters therefore render the official
collision STLs for links 0--7 and the hand; finger geometry uses the four
official URDF collision boxes. The generated URDFs retain the official visual
mesh references for ROS-compatible consumers.

Franka world geometry has one source of truth: the selected planning problem's
`sphere`, `cylinder`, and `box` arrays. `single_mbm` and `evaluate_mbm` copy
those arrays into the replay document. The shared
`scripts/mujoco_primitive_environment.py` adapter validates and inserts the
primitives into MuJoCo at runtime. The XML keeps only the robot, the
visualization floor, and the attached task object. An evaluation replay
requires every bundled trajectory to use the same primitive environment.

`src/robots/franka_fer.cuh` keeps PATACON's 59-sphere fine and 11-sphere
conservative approximate model per arm. Those proxies remain valid because
the official FER 2.9.0 collision STLs and link frames match the replaced
model. `prepare_description.py --check` binds the sphere header hash and all
official collision mesh hashes into the metadata; CUDA collision and
constraint behavior is covered by `validate_franka_collision` and
`validate_franka_constraints`.

The dual-arm layout is a PATACON task model, not Franka's angled FR3 Duo. A
future change from FER to FR3 or FR3 Duo requires new joint limits, collision
spheres, base transforms, task constraints, and benchmark endpoints.

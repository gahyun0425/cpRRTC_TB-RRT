# Official Unitree G1 description adapter

PATACON uses Unitree Robotics' official
[`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros)
repository as its G1 description source. The repository-local `unitree_ros`
submodule is pinned to commit
`da52948f035165aae2709d30255f5cd3e62875a0`. The upstream submodule is kept
unmodified.

Initialize and verify it from the PATACON root:

```bash
git submodule update --init unitree_ros
python3 resources/g1/prepare_description.py --check
```

To regenerate the task-scene adapter and metadata after an intentional source
update:

```bash
python3 resources/g1/prepare_description.py
python3 resources/g1/prepare_description.py --check
```

The selected upstream model is
`unitree_ros/robots/g1_description/g1_29dof.{urdf,xml}`. It preserves PATACON's
35-dimensional configuration contract: six floating-base coordinates followed
by the 29 actuated joints in official URDF order. Unitree now labels this
generic model deprecated in favor of hardware-specific `mode_machine` models,
but it is retained here because it exactly matches PATACON's existing robot,
joint-limit, collision, constraint, and benchmark contracts. Moving to a
specific mode is a separate model change rather than a description-path update.

`g1_humanoid_shelf.xml` is a PATACON environment overlay. It includes the
official MJCF and meshes directly from the repository-local submodule. The G1
visualizers resolve the same local MJCF and URDF relative to their own script
location, so they do not depend on the current working directory or another
workspace.

PATACON retains its checked-in 133-body-sphere model, 16 spheres per rubber
hand, 6,888 self-collision pairs, and CUDA FK/constraint kernels. The official
URDF and MJCF in the pinned checkout are byte-identical to the Unitree files
against which this planning model was established. `model_metadata.json` binds
the official source files and meshes to the checked-in planner kernels and
records the complete joint/limit/sphere contract.

No G1 generation, build, validation, planning, or visualization step reads
`cptbrrt_pkg`, VAMP, an environment-selected description root, or any path
outside PATACON.

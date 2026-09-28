# Constraint layer

This directory is the boundary between declarative planning problems and the
unchanged PATACON planner kernels.

- `json/` parses reusable constraint declarations such as fixed foot poses,
  bimanual relative poses, axis alignment, and center-of-mass support.
- `*ConstraintConfig.hh` converts normalized JSON into the existing
  trivially-copyable per-backend CUDA parameter blocks.
- `RobotConstraintAdapter.hh` selects the appropriate parameter converter at
  the frontend boundary.
- `backends/` contains the compiled residual, Jacobian, projection, and
  tangent-basis implementations. The historical `src/robots/*_constraint.cuh`
  files are compatibility forwarding headers.

The JSON layer may be extended without changing `pRRTC.cu` or `AORRTC.cu`.
Adding an entirely new robot still requires a compiled kinematics/collision
backend and planner template instantiation because robot dimensions and CUDA
workspace sizes remain compile-time properties.

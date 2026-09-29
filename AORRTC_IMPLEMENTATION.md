# AORRTC integration for cpRRTC / TB-RRT

## Run

The existing planner is unchanged when `--aorrtc` is not supplied.

```bash
./build/single_mbm ffw_sg2 tray_lift 1 --no-print-path
```

Run the AORRTC anytime wrapper with the default 5-second planning budget:

```bash
./build/single_mbm ffw_sg2 tray_lift 1 --aorrtc --no-print-path
```

Set another total AORRTC planning budget:

```bash
./build/single_mbm ffw_sg2 tray_lift 1 --aorrtc --time 10 --no-print-path
```

`--time` is valid only together with `--aorrtc`.

## Algorithm mapping

The implementation uses the repository's current bidirectional cpRRTC/TB-RRT search as the satisficing search inside AORRTC. Its Tangent-Space sampling, ConCon expansion, projection, collision checking, Dynamic Domain and balanced tree-selection logic are retained.

AORRTC-specific behavior is isolated in
`src/planning/AORRTCOptimization.cuh`, which is included by
`src/planning/PATACON.cu`:

1. **Initial search:** calls the same PATACON first-feasible-path implementation used when `--aorrtc` is absent. The resulting path cost becomes the initial `c_max`.
2. **Augmented vertex cost:** each tree node stores `g_T(v)`, with root cost zero and child cost `parent_cost + edge_cost`.
3. **Informed/rejection sampling:** after a first solution, a sampled configuration is rejected unless its admissible start/goal lower bound can beat the current best cost.
4. **Random cost bound:** for a feasible sample, `c_rand` is sampled between the root-to-sample lower bound and `c_max - h_hat(sample)`.
5. **Cost-aware nearest neighbour:** candidates must satisfy the sampled cost bound and are ranked by weighted configuration distance plus weighted cost distance.
6. **Cost-bound resampling:** after a bounded-search vertex is validated, lower cost bounds are sampled to look for a cheaper valid parent. A finite guard (`aorrtc_max_parent_resamples`, default 1000) prevents an accidental infinite device loop.
7. **CONNECT budget:** the opposite-tree target must leave enough total cost budget for a path strictly better than `c_max`.
8. **Anytime restart:** the first solution of each fresh bounded search becomes the new best; the trees and Tangent-Space membership are cleared and a new search starts with the tighter bound. GPU allocations and RNG/Halton state are reused, but tree nodes are not reused.
9. **Time budget:** the fresh-search loop continues until `--time` expires or `max_iters` is reached.
10. **PATACON forward half-space:** every non-root Tangent Space records its parent chart. A sampled tangent direction that points back toward the parent root is flipped before joint-limit scaling. This rule is enabled automatically for AORRTC in the PATACON worktree.

## Path simplification

The paper's pseudocode applies `simplify()` after each found solution. This repository does not currently contain a manifold-safe path shortcut/B-spline simplifier for its projected constrained edges. Therefore this integration uses **identity simplification** (the found path is used directly) rather than applying an unsafe ambient-space shortcut. The core AORRTC cost-augmented search and fresh-tree tightening loop remain active.

## Output

When `--save-json` is used with `--aorrtc`, the result is tagged as `AORRTC` and includes:

- initial and best solution times,
- initial and current best cost,
- number of solution updates,
- number of fresh-search restarts,
- solution-improvement history and paths (capped at 1024 records; the overflow flag is set if more improvements occur).

## Build note

`src/planning/PATACON.cu` and its included
`src/planning/AORRTCOptimization.cuh` form one CUDA translation unit and one
`PATACON.cu.o`. The initial search owns the common sampling, projection,
collision, and Tangent-Space helpers; the AORRTC phase reuses them instead of
defining another copy. Both benchmark frontends link the resulting
`patacon_planners` library.

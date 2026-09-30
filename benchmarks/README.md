# Performance baseline

`baseline-rtx5090-2026-09-30.json` records the pre-refactor behavior of the six
bundled planning cases. Each case uses seeds 1 through 20 after one CUDA
warmup. `kernel_ms` is the primary metric because it excludes process startup,
CUDA context creation, and result serialization. `wall_ms`, solution cost,
waypoint count, and iteration count are retained as supporting signals.

Re-record a baseline with:

```bash
python3 benchmarks/record_baseline.py \
  --executable build_patacon/single_mbm \
  --runs 20 --seed 1 \
  --output benchmarks/baseline-local.json
```

Performance data is intentionally not a normal CTest pass/fail condition.
Compare results only on equivalent GPU/driver/build configurations, and first
check solved rate and path validity. Timing regressions should be investigated
using the distribution (especially median and p95), not one run or a strict
millisecond threshold.

The baseline also records the executable hash and Git state. A dirty-tree
baseline can be useful during development, but a release baseline should be
captured from a clean commit.

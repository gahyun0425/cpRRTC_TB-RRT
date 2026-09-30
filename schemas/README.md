# Result JSON contracts

`planner-result-v1.schema.json` is the contract for `single_mbm` single-run
and repeated-run output. `problem-set-results-v1.schema.json` is the contract
for `evaluate_mbm` output.

Version 1 is frozen. A change that removes a field, changes a field's type or
meaning, or makes an optional field mandatory requires a new format name and
a new schema file. For example, use `PATACON_result_v2` together with
`planner-result-v2.schema.json`. Readers should dispatch on the `format`
field, not infer a version from whichever fields happen to be present.

The schema describes both solved and unsolved planner results. The planning
regression tests additionally require a solved result and check semantic
invariants that JSON Schema cannot express, including configuration dimension,
path counts, finite values, and start-to-goal endpoint consistency.

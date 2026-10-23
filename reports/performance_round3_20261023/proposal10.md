# Proposal 10: vectorize substitution-pattern classification

Decision: **reject for this performance round**. There is no supported CLI
invocation of `calc_substitution_patterns` in the audited source at baseline
`624bb8a6dae536d0fa10891f666684ef1ad3dab0` or in the working tree.

The proposed change would replace repeated row-wise `apply(np.sort)` operations
with one array sort and first-seen row factorization. It could only improve a
direct caller of the helper, for which this investigation found no in-repository
use. No speedup or memory-reduction claim is made.

## Evidence and reproduction

From the repository root, using Python 3.14.7 on macOS arm64:

```sh
python reports/performance_round3_20261023/proposal10.py
git grep -n -E 'calc_substitution_patterns|S_sub_pattern_id|N_sub_pattern_id' -- . ':!reports'
```

The script reads the preserved baseline identified by `CSUBST_PERF_BASELINE`, or
by the temporary baseline pointer shared by this investigation. It validates
the three relevant snapshot file hashes against the named Git commit. It
parses every package Python source file, records identifier and literal-name
references, and checks the CLI registration and dispatch with the AST.
The [machine-readable audit](proposal10.json) stores all matches and hashes.

Only two symbol matches exist outside reports:

- `csubst/combination.py:1401` defines the helper.
- `csubst/main_analyze.py:279` contains its commented-out former call.

There are no executable identifier references or literal-name lookups in
package source. No tracked source, test, documentation, or tool outside reports
mentions either literal output column name (`S_sub_pattern_id` or
`N_sub_pattern_id`). `search` and its legacy `analyze` alias register
`command_search` (`csubst/cli.py:1321`–`1322`), which invokes
`main_analyze` (`csubst/cli.py:295`); the call inside that orchestrator is a
comment. Relevant source hashes and symbol matches agree with the preserved
baseline.

## Limits and adoption condition

This is a static reachability assessment, not exhaustive runtime execution of
every CLI parameter combination or an audit of external Python callers.
The helper remains importable, and its behavior is preserved by making no
production edit. Before considering an optimization, establish an actual
supported caller and a representative workload. Then compare stdout, first-seen
IDs, row/index alignment, floats, NaNs, empty and invalid inputs, and use warmups
plus repeated interleaved wall-time and peak-memory measurements.

No prototype, timing workload, memory measurement, or regression suite was run:
there is no production patch and no supported CLI benefit to validate.
`proposal10.patch` is intentionally empty. The audit itself parses both source
trees and asserts its findings when reproduced; it does not import numerical
libraries or consume the shared benchmark slot.

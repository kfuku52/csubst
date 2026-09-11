# Final review and main-checkout verification

Reviewed the accumulated endpoint optimization, joint defaults, MG model,
scan mixture, cross-command model/observation, and bootstrap warning changes.
The final review corrected these additional boundary cases:

- Combination branch subtotals resolve numbered columns through their
  corresponding `branch_id_<slot>`, report per-slot eligible-site counts, and
  distinguish absent observations (`NA`) from observed zero events.
- Derived ratios become `NA` when a requested combination has no joint
  observation support, even if other branches have observations.
- Endpoint caches rebuild when event consumers change (ASRV diagnostics,
  p-values, epistasis, training branches, site filtering or clade permutations),
  preventing reuse of incompatible compressed tensors. Shape configuration is
  also fingerprinted.
- MG simulation rejects an explicit alignment-frequency override instead of
  silently ignoring it; fitted nucleotide frequencies remain authoritative.
- The earlier joint-default report now identifies its missing-tip policy as a
  historical snapshot superseded by cross-command unification.

Regression tests exercise these cases, including storage transitions and
missing branches whose IDs differ from their column slots.

The main checkout rebuilt the changed Cython extension and passed
`make test test-native lint typecheck` using Python 3.10:
**2,198 tests passed, 5 optional-dependency skips; 16 native checks passed**
(native checks overlap the full suite). Lint, repository hygiene, local
documentation and configured type checks passed. The main checkout includes
15 tests belonging to separate uncommitted work; those files were not included
in this commit. All 35 pre-existing unrelated files retained identical SHA-256
hashes. The external wiki was not checked. Dependency warnings in the test
runtime are retained in [the log](final-review-checks.log).

The temporary build environment initially had setuptools below the repository's
required version. Installing the declared `setuptools>=77` requirement resolved
that environment error; project build metadata was unchanged.

Earlier benchmark/verification JSON files preserve the source hashes and
measurements from their respective stages. This final review makes no new
runtime or statistical-calibration claim. Previously documented unsupported
scan modes remain outside this task's scope.

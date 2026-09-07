# Follow-up implementation audit, 2026-09-08

Starting point: clean `master` at `37df36e`, CSUBST 1.14.10.
Environment remains macOS 26.5.2 ARM64, Python 3.14.0, NumPy 2.4.2,
pandas 2.3.3, and SciPy 1.17.1.

This round inspected subprocess and temporary-file ownership, file replacement,
parallel execution helpers, pseudocount calculations, rate calculations,
sequence translation, ancestral gap reconstruction, branch-length rescaling,
dataset materialization, BLAST response handling, and TSV serialization.
The confirmed fixes below are backed by executable regressions; inspection of
the other areas does not establish correctness of every path.

## Confirmed defects

| Area | Reproduction and impact | Fix |
| --- | --- | --- |
| File replacement | Source and destination reached through a symlinked directory can designate the same file. Replacing the destination and then unlinking the source deletes the newly written output. | Detect filesystem identity before staging; same-file aliases are no-ops, including file symlinks and hardlinks. Distinct-file replacement remains covered. |
| Child process lifecycle | A broken output pipe or `KeyboardInterrupt` during tee forwarding bypasses `wait()` and pipe closure, leaving a child running. | Terminate and reap the direct child on exceptional exit; after a five-second termination timeout, kill and reap it. Close stdout on every path and propagate the original output exception. |
| Numeric broadcasting | Scalar inputs to raw rate/omega helpers raise `TypeError`. A one-dimensional numerator broadcast over a matrix masks rows instead of columns, silently zeroing valid ratios. | Apply the zero-numerator rule with broadcast-aware `np.where`, preserving the existing threshold and ratio semantics. |
| Empty TSV output | The zero-column special case writes plain text to `.gz` destinations and rejects caller-owned text/binary streams. | Use the same pandas writer as nonempty tables, with zero rows to retain the historical one-header-newline behavior. |

Added 21 cases. Before fixing their respective code paths, the first runtime
and numerical checks produced 11 failures, and the TSV checks produced three
failures. Remaining new cases verify unchanged behavior and adjacent cases.
Process checks include a real Python child interrupted during output, normal
and nonzero child exits, merged stdout/stderr with invalid UTF-8, and simulated
termination timeout/forced-kill behavior. Filesystem tests use isolated
temporary directories.

## Validation commands and results

Run from the repository root:

```sh
python -m pytest -q tests/unit/test_runtime_io.py tests/unit/test_omega_statistics_arrays.py tests/unit/test_tsv.py
python -m pytest -q -n 4 --dist worksteal -m 'not process'
python -m pytest -q -m process
CSUBST_STRICT_EXTENSIONS=1 python -m pytest -q -m native
CSUBST_DISABLE_EXTENSIONS=1 python -m pytest -q -n 4 --dist worksteal -m 'not process'
CSUBST_DISABLE_EXTENSIONS=1 python -m pytest -q -m process
make lint typecheck
CSUBST_STRICT_EXTENSIONS=1 python .github/scripts/sites_parity_check.py \
  --numerical-only --workdir reports/generated/audit_20260908/followup_parity \
  --output reports/generated/audit_20260908/followup_parity_metrics.tsv
git diff --check
```

The executed parity run used a temporary output directory rather than the
equivalent ignored report directory shown above.

- Focused runtime, numeric-array, and TSV suite: 25 passed.
- Full suite: 1,457 parallel-safe tests and four process tests passed.
- Strict native selection: seven passed.
- Source-only fallback: 1,425 parallel-safe tests and four process tests
  passed; 32 compiled-only tests skipped.
- Ruff, repository hygiene, local documentation checks, all 13 configured
  mypy targets, and diff whitespace checks passed.
- Real bundled PGK/PEPC `analyze` and `sites` checks matched the established
  numerical expectations: omegaCany2spe 1.975050 and 0.049466 respectively,
  with convergent/divergent/blank counts 5/7/390 and 0/2/954.

Not exercised: other operating systems/Python versions, fresh package builds
or installed-wheel tests, sanitizers, real external downloads, real GPU/model
inference, real PyMOL rendering, or remote Wiki links. Process cleanup tests
cover the direct child, not arbitrary descendant process trees. Linux
performance ceilings were disabled on macOS; no speedup is claimed.

The work is a local commit only, without a version bump or push.

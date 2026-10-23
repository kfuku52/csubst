# Additional performance investigation, 2026-10-23

Baseline: `c371a1b54b317831f923bfc0024fef65d039f636`. Four small changes are
adopted from five additional proposals. Scientific defaults, arithmetic,
calibration assignments/seeds, output schemas and error checks are preserved.

| Proposal | Decision and evidence |
| --- | --- |
| Avoid predictive `ones_like` arrays when only observed endpoints are requested | Adopt. The scan path constructed and discarded these arrays for every branch, block and rate category. Prediction callbacks still receive the same arrays in the same order when prediction is requested. Timing benefit is small/noisy; allocation counters provide the direct evidence. |
| Defer legacy exposure diagnostics until the full output path | Adopt. Score-only calibration no longer creates a Boolean missingness vector and a Unicode reason vector. Legacy rate calculations and full-output diagnostics are shared with the existing path. |
| Reuse the branch-axis bounds check within an assignment | Adopt. An all-valid axis gathers directly from a read-only candidate vector. Invalid/missing IDs retain the existing zero-fill path. Only one `(vector length, all-valid)` tuple is retained per assignment; changing vector lengths rechecks bounds. No event vector or growing cache is retained. |
| Use scalar predicates in `calibration_score` | Adopt. The score has already been converted to a Python float; `math.isfinite/isnan` avoid NumPy scalar dispatch. Explicit no-test outcomes and invalid-score failures remain unchanged. |
| Cache endpoint-context branch-order validation | Defer. The current profile contains about 103,000 comparisons in the 1,000-trial PEPC scan. Skipping them would require an additional immutability contract for externally supplied contexts or validation invalidation machinery. The existing check remains on every call. |

The diagnostic [profile](profile.json) prioritizes real workloads; its instrumented
times are not speed measurements. The PEPC run made 103,103 `_rate_summary`
calls, including 103,000 candidate-vector selections.

## Measurements

Apple M2 Max (12 logical CPUs), macOS 27.0.1 arm64; Python 3.14.7, NumPy 2.5.3,
SciPy 1.17.1, pandas 3.0.5. Native binaries are identical in both source trees.
BLAS/OpenMP thread limits are one. CLI seed is 8 and output precision is 12
digits. The parallel compatibility case uses two CSUBST workers.

The [isolated phase benchmark](phase.py) captures 206 score-only calls from
actual fixed-fit PGK/PEPC scans. Arguments, candidate vectors and exposure
caches are reused, with one excluded warmup and seven alternating-order
repetitions. These timings exclude startup and are not whole-command gains.
Every captured score dictionary and every endpoint record matched exactly.

| Isolated phase | CPU median before → after | Change |
| --- | --- | --- |
| PEPC branch selection, 20,600 rate summaries/sample | 0.3383 → 0.2842 s | −16.0% |
| PGK legacy diagnostics, 20,600 rate summaries/sample | 0.2879 → 0.2250 s | −21.9% |
| PEPC scalar score validation, 206,000 scores/sample | 0.0956 → 0.0188 s | −80.4% |
| PEPC observed endpoint inference, 3,102 records/sample | 1.8993 → 1.8546 s | −2.4%; overlapping ranges |

Raw samples, wall times, ranges, input hashes and source hashes are in
[phase.json](phase.json). [variants.py](variants.py) builds each isolated
proposal from the baseline, without applying unrelated proposals.

[Constructor instrumentation](allocations.json) confirms that one PEPC endpoint
inference avoids 12,320 `ones_like` allocations totaling 265,354,880 bytes
(253 MiB of cumulative allocation, not retained or peak memory). A single
206-call PGK sample avoids 206 Unicode reason arrays (1,226,112 bytes) and 206
Boolean missingness arrays (12,772 bytes). Prepared branch selection removes
206 zero-filled event vectors (230,720 bytes). These counters cover only the
listed NumPy constructors, not all Python, native or ufunc allocations.

The initially noisy search timings prompted a separate nine-repeat
[predictive-phase check](predictive_phase.json). CPU medians were
0.5385 → 0.5354 seconds for PGK and 2.9246 → 2.9236 seconds for PEPC, with
overlapping ranges. Every endpoint record matched exactly. This supports
equivalent prediction performance; it does not establish a search speed gain.

The whole-CLI benchmark freezes candidate sources before measuring, copies
bundled alignments/trees/foregrounds and IQ-TREE sidecars to fresh temporary
directories, excludes one warmup per side, then alternates before/after order
for five repetitions. CPU is child user+system time and RSS is macOS
`/usr/bin/time -l` peak resident bytes. A shared lock serializes measured work
and tests. Other host activity is not controlled; wall times are visibly noisy.

Initial results are in [cli.json](cli.json); the independent confirmation and
parallel/full-scan coverage are in [cli_followup.json](cli_followup.json).
The shorter PGK commands do not establish a whole-command speed improvement.
Peak RSS ranges overlap; no reduction in peak RSS is claimed.

For PEPC's 1,000-trial candidate-fixed scan, CPU median fell from 7.76 to
6.83 seconds in the first group (−12.0%), and from 7.27 to 6.73 seconds in the
confirmation group (−7.4%). All ten measured pairs used less CPU after the
change. Confirmation wall time was 8.71 → 8.81 seconds, so a consistent
wall-time improvement is not established.

Across both groups, 48 before/after pairs (96 CLI executions, including
warmups) matched all scientific tables and JSON. The additional
[initial artifact audit](output_audit.json) and
[confirmation artifact audit](output_audit_followup.json) compare every
non-log output file, including fitted endpoint metadata, foreground files and
completed-run status. Only table `elapsed_sec` and search `run_id` fields are
excluded. `audit_report.py` checks the final source fingerprints, isolated
proposal transformations, report links, file sizes and strict JSON syntax.

<!-- BEGIN CLI RESULTS -->

| CLI case / group | CPU before → after (s) | Wall before → after (s) | Peak RSS before → after (MiB) |
| --- | --- | --- | --- |
| `pepc_fixed1000` / initial | 7.76 → 6.83 | 10.43 → 7.35 | 580.9 → 580.1 |
| `pgk_marginal200` / initial | 1.89 → 1.91 | 2.97 → 2.72 | 203.8 → 207.5 |
| `pgk_called50` / initial | 1.74 → 1.84 | 2.09 → 2.35 | 203.3 → 207.1 |
| `pgk_search` / initial | 2.21 → 2.35 | 2.74 → 2.70 | 258.2 → 263.8 |
| `pepc_search` / initial | 5.97 → 5.52 | 7.52 → 6.59 | 449.8 → 457.2 |
| `pepc_fixed1000` / confirmation | 7.27 → 6.73 | 8.71 → 8.81 | 577.5 → 585.5 |
| `pepc_fixed_parallel200` / confirmation | 19.27 → 17.94 | 20.36 → 23.18 | 618.4 → 639.9 |
| `pgk_full20` / confirmation | 3.17 → 3.22 | 4.13 → 5.20 | 332.4 → 311.5 |

<!-- END CLI RESULTS -->

## Correctness and coverage

New regression cases cover independent exhaustive enumeration of transformed
observed and predictive endpoint distributions, rate mixtures including a
zero-rate category, selected/unselected branches, partial blocks, all legacy
exposures with zero target/control lengths, score-only/full summaries, ordered
and duplicate IDs, negative/out-of-range/missing IDs, vector-size changes,
read-only candidate ownership, and finite/undefined/invalid scalar scores.

[checks.json](checks.json) records repository checks and exit codes. The first
focused command named a nonexistent test file and collected no tests; the
corrected selection passed. Subsequent full lanes cover the final tests.
Verification uses Python 3.12.15; the focused selection also runs on Python
3.14.7. Its temporary interpreter path is normalized to `<PYTHON312>` in the
recorded commands. The Makefile typecheck covers its declared module list only.

Whole-command comparisons reuse fitted bundled models. They do not recompute
IQ-TREE fits or establish biological calibration. This is source-tree evidence,
not installed-wheel/sdist, Linux, or a complete supported-Python matrix proof.
Optional-dependency skips are reported in the check results. No fixture,
reference value, threshold, tolerance or version is changed.

Python 3.12 results: `make test-fast` passed 1,704 tests (6 skipped); `make test`
passed 2,249 tests (9 skipped), followed by 5 serial process tests; and
`make test-native` passed 15 tests. All six compiled modules were explicitly
imported and checked. `make lint typecheck` passed. The Python 3.14 focused
selection passed 394 tests. The 9 full-suite skips are 4 Torch, 2 gemmi and 3
real IQ-TREE cases. Eight existing Matplotlib `set_bad` pending-deprecation
warnings appeared in the Python 3.12 ordinary suites. Detailed versions are in
[environment.json](environment.json).

With `CSUBST_DISABLE_EXTENSIONS=1`, the complete Makefile lane passed 2,219
tests (39 skipped), followed by 5 serial process tests. The extra 30 skips
require native extensions; the same 9 optional-backend cases remain unverified.
These are source-only fallback runs with extension dispatch disabled, rather
than a rebuilt sdist install.

## Reproduction

Start in the repository root with the development environments described in
[CONTRIBUTING.md](../../CONTRIBUTING.md). Create a baseline source snapshot and
copy the same locally compiled native modules into it:

```bash
PERF_BASELINE="$(mktemp -d)"
git archive c371a1b csubst | tar -x -C "$PERF_BASELINE"
python -c 'import glob, shutil, sys; [shutil.copy2(p, sys.argv[1]+"/csubst") for p in glob.glob("csubst/*.so")]' "$PERF_BASELINE"
python reports/performance_round4_20261023/phase.py --baseline "$PERF_BASELINE" --result /tmp/round4-phase.json
python reports/performance_round4_20261023/phase.py --baseline "$PERF_BASELINE" --result /tmp/round4-allocations.json --allocations-only
python reports/performance_round4_20261023/phase.py --baseline "$PERF_BASELINE" --result /tmp/round4-predictive.json --cases pgk_search pepc_search --repeats 9
python reports/performance_round4_20261023/compare_cli.py --baseline "$PERF_BASELINE" --workdir "$(mktemp -d)/runs" --result /tmp/round4-cli.json --cases pepc_fixed1000 pgk_marginal200 pgk_called50 pgk_search pepc_search --repeats 5
python reports/performance_round4_20261023/compare_cli.py --baseline "$PERF_BASELINE" --workdir "$(mktemp -d)/runs" --result /tmp/round4-followup.json --cases pepc_fixed1000 pepc_fixed_parallel200 pgk_full20 --repeats 5
```

Use a fresh `--workdir` for each group. `audit_outputs.py --workdir <WORKDIR>
--result <JSON>` audits every non-log artifact from a completed group. The
benchmark reuses the tracked comparison/locking helpers in
`reports/performance_round3_20261023/` and
`reports/performance_round2_implementation_20261022/`.

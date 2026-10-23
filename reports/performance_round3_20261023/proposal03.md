# Proposal 03: scalar finite checks in rate_score

Decision: promising for full-CLI follow-up. The isolated scalar workload uses
less CPU; the real PEPC calibration chunk suggests a smaller gain. Wall-time
noise and unresolved PGK results do not support a whole-CLI speedup claim.
The accompanying patch is suitable for the parent's deeper verification.

`rate_score` already converts all four arguments to Python floats. Replace
the four-element NumPy finite-check allocation and reduction with four
`math.isfinite` calls. Keep negative-count/zero-exposure validation, all
log-space calculations, return values, and float-conversion errors unchanged.
There is no new retained storage or algorithmic memory change.

Reproduce from repository root, with an extracted baseline snapshot containing
the package and bundled datasets:

```sh
CSUBST_PERF_BASELINE='<BASELINE_624bb8a>' python reports/performance_round3_20261023/proposal03.py
```

The current session alternatively uses `/tmp/csubst-round3-baseline-path`.
Baseline: `624bb8a6dae536d0fa10891f666684ef1ad3dab0`. Environment: macOS
27.0.1 ARM64, 12 logical CPUs; Python 3.14.7, NumPy 2.5.3, SciPy 1.17.1,
pandas 3.0.5. Native BLAS/OpenMP variables are set to one before imports.
All computation runs under this round's shared benchmark lock. Each comparison
has five untimed warmups per variant followed by seven measured repetitions,
alternating the variant order. CPU time uses `process_time`; wall time uses
`perf_counter`. Samples and input hashes are in `proposal03.json`.

| Workload | Before CPU median | After CPU median | Reduction |
| --- | ---: | ---: | ---: |
| 30,000 scalar calls, mixed enrichment/non-enrichment/invalid/extreme inputs | 0.076897 s | 0.047427 s | 38.3% |
| PGK candidate_fixed50 chunk, 12 observed candidates | 0.079575 s | 0.080076 s | -0.6% |
| PEPC candidate_fixed50 chunk, 103 observed candidates | 0.293962 s | 0.272092 s | 7.4% |

Every scalar paired CPU comparison improved by 30.8–42.3%. PEPC paired CPU
changes range from a 2.9% regression to a 15.4% improvement, with a median
improvement of 5.2%. PGK has no resolved gain. Wall medians respectively were
0.357283→0.206783, 0.237747→0.235461, and 0.400320→0.364162 seconds, but
wide within-variant ranges show substantial scheduling noise. These wall
medians are observations, not reliable speedup estimates.

The PGK/PEPC runs load copied bundled fits through the real CLI and intercept
its calibration chunk. They repeat the original chunk with fresh worker
caches, holding fitted tensors and model state fixed. Both variants finish all
50 assignments without failure. Every comparison preserves candidate keys,
scores, maxima, and assignment diagnostics exactly. Timings exclude CLI
startup, parsing, ancestral inference, initial discovery, and final output;
they are not end-to-end timings.

Direct validation compares 25,822 cases exactly: 25,798 results and 24 matching
conversion exceptions. It includes the four-position Cartesian grid over
zero, signed zero, negatives, subnormals, large finite values, infinities and
NaNs; Decimal/Fraction/NumPy scalar/string inputs; overflow-sized integers;
5,000 seeded wide-log random cases; and adjacent enrichment boundaries.
Scores, diagnostic P values, exception classes/messages, and function warnings
all match. Scientific defaults and numerical formulas remain unchanged.

Verification: the harness completed successfully; `python -m py_compile` and
`python -m ruff check` passed for `proposal03.py`; `git diff --check` passed.
Production modules and tests were not edited by this proposal agent. The
parent will run affected repository checks after integration. Full CLI,
installed-wheel, peak-RSS, native/fallback parity, and scientific calibration
were not measured by this proposal; no claim is made for those scopes.

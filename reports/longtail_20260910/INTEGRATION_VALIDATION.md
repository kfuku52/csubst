# Independent-null default: integration validation

2026-09-10. Integrated onto main checkout baseline
`16350b4aa2afc4564f24762b948c7c62b3408112`, preserving the ASRV training,
native 3Di expectation, and site-filtering changes made since the original study.

## Default and supported scope

`--longtail_method` now defaults to `independent_null`, consistently in the CLI,
validation, calibration, P-value dispatch, summary metadata, and missing-row
recalculation. Calibration remains opt-in (`--calibrate_longtail yes`). The
empirical method remains available explicitly. Independent-null calibration
requires urn expectations, base output statistics, no nonzero pseudocounts,
and fixed dispersion for negative-binomial sampling. Unsupported combinations
fail before analysis. Both methods retain the documented exhaustive-arity limit.

## Fixes and regression coverage

- Freeze references across displayed subsets and missing foreground batches.
- Use separate fit/test random streams and the same map for observed/null rates.
- Preserve uncalibrated columns and recompute calibrated P/Q values; report
  unavailable whole-population tests or testing families explicitly.
- Retain nonfinite targets and diagnose insufficient reference samples.
- Share tensor summaries within each pass and compute Poisson category means
  across combinations once, avoiding repeated category expectation calculations.
  Cache lifetime is limited to one statistic/pass, never persisted on run state.
- Check exact cached/uncached draws for four count engines and four base
  statistics, with both uniform and channel-specific ASRV. PEPC additionally
  exercises sparse tensors and branch-specific ASRV.
- Test the omitted method option through both observed and null-test paths.

## Checks

Using the local Miniforge Python 3.10 interpreter, with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

- `make test`: 1,678 non-process tests passed, plus 4 process tests passed.
  Three structural-prediction tests skipped because installed Torch 2.2.2 is
  older than their required 2.6. xdist used one worker
  (`PYTEST_XDIST_AUTO_NUM_WORKERS=1`); all test families were retained.
- `make test-native`: 7 strict native-extension tests passed.
- `make lint typecheck`: passed (including repository hygiene and local docs).
- `make package`: source distribution and native wheel built; twine checks passed.
- Earlier full sequential run: 1,680 passed, 3 skipped, before the final two
  omitted-method regression tests and Poisson cache improvement.

The existing requests dependency warning was present during testing.

## PEPC benchmark

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python .github/scripts/longtail_benchmark.py --outdir /tmp/longtail-benchmark
```

Bundled PEPC alignment, rooted tree and IQ-TREE outputs; no foreground file;
all 971 sites; urn/Wallenius, ASRV `each`, arity 2, `any2spe,any2any`, seed
20260910, no pseudocounts, 1,000 calibration draws. Calibration is enabled with
the method option omitted, exercising the new default. P values are disabled
in this full-data timing run; actual P-value engines are covered separately by
the integration tests above.

For three spread-out pairs (row indices 0, 4154, 8307), both statistics and both
N/S channels, **all cached and uncached draw arrays were exactly equal** across
two repetitions. Cached timing includes initialization of means for all 8,308
combinations, although only three are sampled in this comparison.

| Engine | Uncached, seconds | Cached, seconds |
| --- | --- | --- |
| Poisson, repetition 1 | 31.136 | 13.661 |
| Poisson, repetition 2 | 31.289 | 11.595 |
| Hypergeom, repetition 1 | 5.049 | 5.061 |
| Hypergeom, repetition 2 | 4.922 | 5.339 |

The Poisson speedup comes from shared category expectations. Summary caching
alone gave no meaningful speed improvement, and no hypergeom improvement is
claimed. Cached Poisson means require additional memory proportional to
combinations times categories; this is separate from per-row draw arrays.

The full Poisson calibration completed for all **8,308 combinations** and both
statistics in **79.284 seconds** (calibration only; excludes the preceding
benchmark and tree/state preparation). The resulting table was checked:
method diagnostics select `independent_null` throughout, denominators increased
in 3,319 `any2spe` and 4,779 `any2any` rows, and calibrated finite omegaC never
exceeds its corresponding uncalibrated value. Raw measurements and input hashes
are in [performance.json](performance.json).

# Long-tail calibration

Long-tail calibration is an optional **sensitivity analysis**, disabled by
default. When enabled, `--longtail_method independent_null` is the default method.
It raises dSC using a quantile map and can reduce omegaC. It is not a
correction with a universal biological or statistical threshold at omegaC=1.
Compare the uncalibrated and calibrated columns and their respective P values.

## Running a comparison

Use the same inputs, site filtering and expectation settings in separate output
directories. For example, the following commands compare uncalibrated rates,
the empirical full-table reference and an independently sampled null reference:

```bash
csubst search --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --expectation_method urn --output_stat any2any,any2spe \
  --calc_omega_pvalue yes --omega_pvalue_niter_schedule 1000 \
  --random_seed 1 --calibrate_longtail no --outdir uncalibrated

csubst search --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --expectation_method urn --output_stat any2any,any2spe \
  --calc_omega_pvalue yes --omega_pvalue_niter_schedule 1000 \
  --random_seed 1 --calibrate_longtail yes --longtail_method empirical \
  --outdir empirical

csubst search --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --expectation_method urn --output_stat any2any,any2spe \
  --calc_omega_pvalue yes --omega_pvalue_niter_schedule 1000 \
  --random_seed 1 --calibrate_longtail yes --longtail_method independent_null \
  --longtail_null_niter 1000 --outdir independent_null
```

Both modes currently apply only to arities at or below `--exhaustive_until`
(default 2). A foreground file does not restrict the calibration population
when the search is exhaustive. Above that limit calibration is disabled; the
previous arity's calibrated omegaC can still affect candidate selection through
`--cutoff_stat`. Results from differently selected candidate sets should not be
interpreted as matched power comparisons.

## Empirical reference

For a finite reference population of size n, S midranks define
`q = (rank(dSC) - 0.5) / n`. The replacement denominator is
`max(dSC, quantile(dNC, q))`, with average ranks for ties and linear N quantiles.
For at least two finite pairs, fitting and applying the map to the original
population reproduces the earlier quantile calculation, including its rounding
order. Zero or one finite pair now leaves the rates unchanged and reports
`insufficient_reference`. The numerical minimum of two does not establish that
a small or highly tied reference is statistically representative.

The S values/midranks and sorted N values are stored together for each arity and
category. Missing foreground-permutation rows use this original frozen map;
they cannot fit a new map from just the missing batch. For S values between
reference knots, interpolate the midrank linearly. Values outside the observed
support use the N minimum/maximum before applying the one-sided maximum.
This defines previously unobserved S values explicitly. A new complete analysis
that changes the reference population can still change the empirical map.

When computing calibrated empirical P values, the map is re-estimated in every
null replicate on the same full row population. All rows remain active through
the final configured simulation stage. Freezing a data-fitted observed map for
these tests would be a different procedure. If only missing foreground rows are
being recomputed, their calibrated empirical P/Q values are unavailable because
the full-population null is not present; the output records this reason rather
than testing against a map fitted to the missing subset.

## Independently sampled null reference (experimental)

`independent_null` fits one map **per branch combination and category**, using
`--longtail_null_niter` calibration draws (default 1000, numerical minimum 100).
The reference samples come from the existing selected count-null engine. Their
N/S rates use the same fixed ECN/ECS denominators as the observed rates. This
avoids pooling rows with different exposures, and fixes both the source S
midranks and target N quantiles.

The fit and test streams have distinct purpose labels. Their seeds also include
the canonical branch IDs and category. Consequently, with the same tensors,
expectations and configuration, display subsets, row/category order and missing
row batches cannot change a row's reference or calibrated P value. A fresh
model fit or a changed input tensor can still change the result. Null generation
per row provides marginal tests; it does not preserve a joint null over all
overlapping branch combinations for a family-wide test.

The observed row and each test replicate use exactly the same frozen map. The
test uses the final value of `--omega_pvalue_niter_schedule` for every row,
without adaptive stopping. The maps are regenerated deterministically when
testing and their hashes must match the hashes used for the observation.
`--longtail_test_block_size` (default 256) controls statistic evaluation blocks;
it does not change the generated draws. Draw arrays are generated per row, so
simulation memory does not scale with the product of table size and draw budget.
Tensor summaries are shared within one statistic and calibration/test pass.
For Poisson nulls, category means are computed together for all combinations;
this additional cache scales with combinations times substitution categories.
Caches are discarded between passes and never reused after input/model changes.
These optimizations preserve the count draws and fitted reference hashes.

Supported combinations:

- `--expectation_method urn` and the base categories `any2any`, `any2spe`,
  `spe2any`, `spe2spe`.
- `hypergeom`, `poisson`, `poisson_full`, and `nbinom` with an explicitly fixed
  `--omega_pvalue_nbinom_alpha`.
- No nonzero pseudocount smoothing. Mode `none`, a fixed effective alpha of
  zero and report-only settings are allowed.

Derived `dif` categories and nonzero pseudocounts are rejected pending the joint
category-null and shared observed/null smoothing work. Automatic negative
binomial dispersion is also rejected. These errors are not relaxed by filtering
invalid null draws or substituting another null model.

“Independent” describes calibration versus test **random draws**, conditional
on the existing fitted count model and exposures. It does not mean the model
parameters were estimated independently of the biological observations. It does
not validate ASR, posterior dependence, fitted site rates or structural omegaC.

## Output and migration

- The method default is `independent_null`; omit `--longtail_method` to use it.
  Its supported settings are documented above and validated before analysis.
- The default changed from `--calibrate_longtail yes` to `no`. Explicitly choose
  `yes --longtail_method empirical` for the previous empirical sensitivity
  analysis, with the new small-reference guard and missing-row fix. The change
  can alter higher-arity candidates even when their own calibration is disabled.
- With calibration enabled, `dSC*_nocalib` and `omegaC*_nocalib` retain the input
  to the long-tail step, including skipped categories. They may already include
  pseudocount smoothing. `_raw` and `_smoothed` have their separate existing
  meanings; `_nocalib` does not imply unsmoothed.
- `pomegaC*_nocalib` and `qomegaC*_nocalib` retain uncalibrated inference when
  requested. Unsuffixed P/Q columns are recomputed for the calibrated statistic.
  Existing limitations of uncalibrated pseudocount/dif P values remain separate.
- Per-category `calibration_*` columns record method, map hash, empirical
  population hash (or `per_combination`), finite reference pairs, unique S
  values, status, actual denominator increase, resolved null seed, P-value
  status, valid test draws and undefined test draws. Equal denominators are
  not counted as increased. Nonfinite target rates retain their original omega.
- Newly added independent-null foreground rows can have their marginal P values
  recomputed. Their calibrated Q values are left undefined because the missing
  batch is not the complete original testing family. A Q value must not be
  copied between different testing families.

## Reproducible comparison figures

Run `python .github/scripts/longtail_comparison.py` from the repository root.
It writes PNG/PDF figures, TSV summaries and version/seed/source metadata to
`reports/longtail_20260910`. The experiment uses 1000 independent datasets per
condition, 1000 fit draws and 999 test draws. The legacy formula is preserved in
the script, including its former singleton behavior, and checked against the
production formula on multi-row arrays. Frozen and independent-null maps use
the production `QuantileMap` implementation.

The known-parameter Poisson study separates row count (1, 4, 32), expected S
(0.5, 8), other-row signal prevalence (0, 50, 100%) and focal signal (N x1 or
x4). Other-row counts are rounded to the nearest attainable number of rows.
All methods share the same observed/test draws. Each condition tests one focal
row per independent dataset, allowing exact 95% binomial intervals without
treating overlapping branch combinations as independent biological replicates.
The omega>=5 pass rate is recorded separately from P<=0.05 rejection.

These figures validate behavior under that count model only. End-to-end
sequence/tree simulations with ASR and model refitting, heterogeneous site-rate
models and biological data are still needed before broader calibration claims
or making the independent-null method a default.

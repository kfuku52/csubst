# Issue #46 progress report (2026-02-25)

## Scope covered
- Root-cause analysis for anti-conservative empirical p-values under true null.
- Added safety controls for empirical p-value calculation.
- Added/benchmarked alternative null generators (`hypergeom`, `poisson`, `poisson_full`).
- Added true-null diagnostics, runtime, and peak RAM summaries.

## Reproducible scripts
- `reports/issue46/run_safe_guard_compare.sh`
- `reports/issue46/run_null_model_compare.sh`

## Main artifacts
- `reports/issue46/autoguard_vs_old_true_null_pvalue_compare.png`
- `reports/issue46/safe_guard_pvalue_cdf_hist_compare.png`
- `reports/issue46/null_model_pvalue_cdf_hist_compare.png`
- `reports/issue46/autoguard_vs_old_summary.tsv`
- `reports/issue46/safe_guard_summary_all.tsv`
- `reports/issue46/null_model_summary_all.tsv`
- `reports/issue46/autoguard_vs_old_runtime_peak.tsv`
- `reports/issue46/safe_guard_runtime_all.tsv`
- `reports/issue46/null_model_runtime_all.tsv`
- `reports/issue46/SHA256SUMS.txt`

## Headline results (true null)
- Baseline (`min_sub_pp=0`) is strongly anti-conservative.
- Safety thresholding (`min_sub_pp>=0.05`) substantially improves calibration behavior.
- New continuous-null variants (`poisson`, `poisson_full`) reduce runtime, but do not fix anti-conservativeness in this dataset without thresholding.

## Current default policy
- Empirical p-value calculation remains default OFF (`--calc_omega_pvalue no`).
- No auto-adjustment is applied to `--min_sub_pp` for empirical p-value mode.
- If thresholding is required, set `--min_sub_pp` explicitly.

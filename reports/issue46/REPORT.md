# Issue #46 progress report (2026-02-26)

## Scope covered
- Root-cause analysis for anti-conservative empirical p-values under true null.
- Removed `--omega_pvalue_safe_min_sub_pp` safeguard (no backward compatibility).
- Added denominator-validity control for empirical omega_C p-values via `--omega_pvalue_min_expected_S`.
- Added/benchmarked alternative null generators (`hypergeom`, `poisson`, `poisson_full`, `nbinom`).
- Added true-null diagnostics, runtime, and peak RAM summaries.

## Reproducible scripts
- `reports/issue46/run_safe_guard_compare.sh`
- `reports/issue46/run_null_model_compare.sh`

## Main artifacts
- `reports/issue46/autoguard_vs_old_true_null_pvalue_compare.png`
- `reports/issue46/safe_guard_pvalue_cdf_hist_compare.png`
- `reports/issue46/null_model_pvalue_cdf_hist_compare.png`
- `reports/issue46/null_model_cdf_after_ecs_filter.png`
- `reports/issue46/null_model_frac_le_0_05_after_ecs_filter.png`
- `reports/issue46/ecs_filter_effect_before_after.png`
- `reports/issue46/null_model_runtime_after_ecs_filter.png`
- `reports/issue46/autoguard_vs_old_summary.tsv`
- `reports/issue46/safe_guard_summary_all.tsv`
- `reports/issue46/null_model_summary_all.tsv`
- `reports/issue46/autoguard_vs_old_runtime_peak.tsv`
- `reports/issue46/safe_guard_runtime_all.tsv`
- `reports/issue46/null_model_runtime_all.tsv`
- `reports/issue46/SHA256SUMS.txt`

## Headline results (true null)
- Without `--omega_pvalue_safe_min_sub_pp`, baseline (`min_sub_pp=0`) remained anti-conservative for `any2spe`.
- New denominator-validity rule (`--omega_pvalue_min_expected_S 0.01`) improved `any2spe` calibration:
  `frac(p<=0.05)` was ~0.05 for `hypergeom`, `poisson`, and `nbinom`.
- `poisson_full` remained anti-conservative (`any2spe` `frac(p<=0.05)` ~0.28).
- Runtime impact was small except legacy `hypergeom` in `any2spe` (still slow).

## Current default policy
- Empirical p-value calculation remains default OFF (`--calc_omega_pvalue no`).
- No auto-adjustment is applied to `--min_sub_pp` for empirical p-value mode.
- Rows with very small expected synonymous counts (`ECS < --omega_pvalue_min_expected_S`, default `0.01`) are set to `NaN` in empirical p/q output.

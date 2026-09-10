# Pseudocounts and conditional omegaC P values

`--calc_omega_pvalue yes` with `--expectation_method urn` compares the observed
omegaC to a fitted **count** null. Observed counts and every null replicate use
the same transformation, including pseudocounts and any long-tail step. This is
not a bootstrap of alignment generation, ASR, site-rate fitting or search
selection, and does not add P values to the codon/3Di model-based expectations.

## Supported settings

| Statistic / smoothing | Count null | Long-tail setting |
| --- | --- | --- |
| Base categories; none or fixed symmetric alpha | hypergeom, poisson, poisson_full, nbinom | Off or empirical; independent_null requires fixed nbinom dispersion |
| Any category including dif; none or fixed symmetric alpha | poisson with compatible category means | Off, empirical or independent_null |
| Empirical prior and/or alpha=auto | poisson with compatible category means | Off or empirical |

For `independent_null`, fixed symmetric pseudocounts are applied to both the
independent calibration samples and test samples before applying the frozen
map. Data-dependent priors/alpha are rejected with this method: a frozen map
cannot silently reuse parameters fitted to the tested observation. See
[long-tail calibration](LONGTAIL_CALIBRATION.md).

The default output categories include `any2dif`, while the default count null is
`hypergeom`. To request P values for the default categories, explicitly select
`--omega_pvalue_null_model poisson`. To retain the hypergeom null, explicitly
select base categories such as `--output_stat any2any,any2spe`. No null model is
switched automatically. These restrictions concern P values; they do not
remove untested effect-size categories from ordinary runs.

```bash
csubst search --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --expectation_method urn --calc_omega_pvalue yes \
  --omega_pvalue_null_model poisson --omega_pvalue_niter_schedule 1000 \
  --pseudocount_mode symmetric --pseudocount_alpha 1 \
  --calibrate_longtail no --random_seed 1 --outdir smoothed

csubst search --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --expectation_method urn --calc_omega_pvalue yes \
  --omega_pvalue_null_model poisson --omega_pvalue_niter_schedule 1000 \
  --pseudocount_mode empirical --pseudocount_alpha auto \
  --calibrate_longtail yes --longtail_method empirical \
  --random_seed 1 --outdir refitted
```

## Statistic and estimation order

For each category, calculate

```
dNC = (OCN + alpha_ON) / (ECN + alpha_EN)
dSC = (OCS + alpha_OS) / (ECS + alpha_ES)
omegaC = dNC / dSC
```

`--pseudocount_target observed|expected|both` selects which alphas are nonzero.
Derived counts are formed before smoothing. Ratios are never subtracted to
construct a dif statistic. A nonzero prior, however small, is included in the
count ratio; an exact 0/0 rate is zero. The existing omega tolerance convention
maps dNC below `float_tol` to omegaC zero. Positive/zero ratios remain infinite;
undefined ratios remain NaN. With alpha zero or mode `none`, the raw tolerance
conventions are unchanged.

Fixed symmetric alpha is shared by observation and null. For empirical priors
or `alpha=auto`, every null replicate refits the prior/alpha from a complete
pseudo dataset using the same count columns, requested fitting categories and
row population as the observation. Expected counts remain fixed. Empirical-
Bayes deterministic thinning is independent of row/category ordering. This
procedure includes smoothing-parameter estimation, but does not refit E, ASRV,
negative-binomial dispersion, or the upstream phylogenetic model.

Empirical long-tail calibration follows smoothing and is refitted within each
replicate on the same full row population. A missing-row subset cannot refit
an observation's full-population prior or calibration statistic. Supplemental
foreground-permutation rows use the original fitted prior/alpha for effect
sizes; their data-dependent P/Q values remain unavailable with
`pvalue_status_<stat>=unavailable_full_population_null` (zero test draws).
A fresh `get_omega` call fits its own current counts/settings; explicitly
reusing a fitted context is reserved for supplemental effect sizes. Independent-
null calibration instead uses its same independently trained map for the
observed and test draws.

Smoothed, calibrated and joint-category tests use the **final** scheduled repetition count
for every row. They do not select active rows using interim P values. Joint
Poisson draws stream in repetition blocks (`--longtail_test_block_size`);
changing block size, row order, category order, or earlier schedule entries
preserves the results. Fixed-smoothing marginal tests use canonical 128-draw
sampling blocks. Alpha-zero, uncalibrated base-statistic tests retain the
existing marginal sampler and staged schedule for compatibility.

## Joint Poisson category null

The four disjoint count categories are `spe2spe`, `spe2dif`, `dif2spe`, and
`dif2dif`. Their fitted means are obtained from the four base-category means:

```
mu_spe2spe = ECspe2spe
mu_spe2dif = ECspe2any - ECspe2spe
mu_dif2spe = ECany2spe - ECspe2spe
mu_dif2dif = ECany2any - ECspe2any - ECany2spe + ECspe2spe
```

Draw each disjoint component once, and form every reported category as a sum
of those same draws. This preserves the Poisson marginal means and the
covariance implied by shared components. If all events are specific, dif is
identically zero. Materially negative or infinite component means cause an
error; only subtraction round-off is zeroed. Category-specific ASRV or another
fitted-mean construction can produce incompatible marginals, in which case
this joint null is unavailable. No negative simulated counts are clipped, and
no invalid dif trials are dropped to manufacture a P value.

This is an explicit coupling of the **combination-count** Poisson marginals,
not a reconstruction of branch histories. N/S channels and distinct rows are
independent conditional on their means. It does not preserve shared-branch
covariance across combinations. The other marginal count engines do not yet
specify a supported joint category law; dif and data-dependent smoothing P
values are rejected for those engines.

## P values, diagnostics and limits

With B valid draws and r upper-tail exceedances including ties, the reported
value is `(r+1)/(B+1)`. Relative machine-precision ties count in the upper tail;
no absolute tolerance equates tiny positive statistics with zero. NaN draws
are counted as undefined; infinities remain comparable. The plus-one formula
does not by itself validate the fitted null. Q values are recomputed by BH
from the new P values within the existing reported family; this does not
establish validity after search selection or under arbitrary dependence.

`pvalue_*_<stat>` records `statistic` (raw/smoothed, optionally calibrated),
`n` (valid draws), `undefined`, `alpha` and `prior` (fixed/refit), `expectation`
(fixed fitted count null), and `joint` (independent Poisson atoms or marginal).
When long-tail calibration follows an initial test, the initial metadata and
P/Q values are preserved with `_nocalib`; the unsuffixed columns describe the
recomputed calibrated test. `_smoothed` effect-size columns precede long-tail
calibration; `_nocalib` does not mean unsmoothed.

Tests cover identical-count P=1, zero-alpha compatibility, nonfinite values,
small enumerated count spaces, category nesting/covariance, naive replicate
refitting, and real dense/sparse urn means. A reproducible
[conditional-count experiment](../reports/scientific_review_20260910/validate_pseudocount_pvalues.py)
uses independently generated observed datasets. Its intervals quantify that
experiment only; biological false-positive calibration still needs independent
full-pipeline simulations, including ASR and any data-driven selection.

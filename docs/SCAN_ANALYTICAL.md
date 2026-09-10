# Analytical endpoint-enrichment inference

`--scan_analytic_pvalue endpoint_mixture` adds a simulation-free test that
integrates ancestral states and rate categories on the full codon tree. It
avoids interpreting fractional posterior event mass as an integer Poisson
count. It is opt-in: fitted-model uncertainty is still uncalibrated, and the
finite-sample construction can lose substantial power with few foregrounds.

```bash
csubst scan --alignment_file alignment.fa --rooted_tree_file rooted.nwk \
  --foreground foreground.tsv --iqtree_model GY+FQ \
  --scan_match any2spe,spe2spe \
  --scan_pvalue_calibration none --scan_n_permutations 0 \
  --scan_analytic_pvalue endpoint_mixture
```

The initial implementation accepts the codon models supported by joint endpoint
inference (GY, ECMK07, ECMrest, supported frequency modifiers and discrete
G/R/I rate mixtures), and fixed amino-acid recodings, with `--ml_anc no`. It supports `any2any`,
`any2spe`, `spe2any`, and `spe2spe`. `dif` contrasts and 3Di require separate
verified models and are rejected. The reported IQ-TREE category distribution
is integrated; posterior-mean site rates are not plugged into the test.
`--scan_site_plot_filter analytical` uses the new P when this option is active.

## Null, alternative, and calculation

Let `P0(D)` be the probability of the observed tip states at one site under the
fixed codon tree, branch lengths, root frequencies, Q, and rate distribution.
For a prespecified state contrast and foreground clades, define four alternative
transition matrices. On each foreground branch and in each rate category,
multiply the matching **endpoint** transition probabilities by 2, 10, or 100,
and normalize every row to sum to one. The fourth alternative is the infinite-tilt
limit: condition each row on a matching endpoint. If that row has zero matching
probability, its finite tilts and their limit all equal the original row. This
removes an artificial maximum effect size without introducing infinities. Background matrices are unchanged.
Each multiplier defines a probability distribution `Pa(D)` on the same data.
They enrich endpoints; they are not fitted branch-rate multipliers in a CTMC Q.

Scaled tree pruning computes the null and four alternative likelihoods,
summing over shared ancestors and the site-wide rate category. It uses tip
observations, not reconstructed ancestral marginals or called substitutions.
Missing tips contribute an all-ones likelihood vector. Foreground branches
come from the topology and phenotype-defined clades, including branches with
no ASR row; candidate support cannot select which branches enter the test.

The equal-weight, prespecified mixture yields

```
E(D) = (P2(D) + P10(D) + P100(D) + Pconditional(D)) / (4 P0(D))
p_endpoint_enrichment_analytic = min(1, 1/E(D))
```

Under a **fixed correct null**, `E0[E] <= 1`. Markov's inequality gives
`Pr0(p <= alpha) <= alpha`. This is a conservative finite-sample P, not an
asymptotic chi-square tail. The alternatives are averaged, never selected by
maximizing the observed likelihood. Calculations use log likelihoods, and
`log_e_endpoint_enrichment` retains evidence when P underflows float64.
There is no Monte Carlo resolution floor.

This tests compatibility with the specified **site-wide evolutionary null** in
a foreground-endpoint direction. It does not establish adaptation, identify a
unique historical substitution, or test a nuisance-free equality of foreground
and background Poisson rates. A different state contrast at an alternative
site need not remain a true null. The original `p_rate_enrichment_asymptotic`
and score retain their diagnostic meaning and are emitted separately.

## Selection and FDR

The hypothesis count is fixed **before site filtering or candidate discovery**:

```
m = input_sites * traits * foreground_targets
    * sum(contrasts_per_match)
```

There is currently one foreground target per trait. With `S` states the match
counts are 1 (`any2any`), S (`any2spe`), S (`spe2any`), and S(S−1) (`spe2spe`).
For 20 amino acids, 300 sites, two traits and `any2spe,spe2spe`, m=240,000,
even if only three candidates are emitted. Unsupported or zero-exposure
Poisson diagnostics do not remove endpoint hypotheses from this universe.

Unreported candidates have P=1 (equivalently, set their original e-value to
zero before testing). This can only decrease evidence, so arbitrary
support/site filtering cannot inflate the e-value expectation. The new BH
column uses m, not the number of emitted rows. BH on these reciprocal e-values
is equivalent to the base **e-BH** procedure for rejection levels below one;
therefore it supports FDR control under arbitrary hypothesis dependence when
the fixed-null e-value assumptions hold. This stronger statement is specific
to this construction and does not apply to the old asymptotic P values.
The additional BY column is a more conservative generic P-value correction.

Columns:

- `p_endpoint_enrichment_analytic`: reciprocal mixture likelihood-ratio P.
- `log_e_endpoint_enrichment`: natural-log evidence, before P clipping.
- `scan_analytic_status`: explicitly marks fitted-model-conditional inference.
- `scan_analytic_family_size`: full per-run hypothesis count, repeated on rows.
- `q_endpoint_enrichment_analytic_bh`: BH/e-BH adjustment across that full run.
- `q_endpoint_enrichment_analytic_by`: optional interpretation under generic BY.

`scan_inference.json` contains an `analytical_endpoint` section with the family
size even for empty scans, fixed alternatives, category rates/weights, target
branches, the numerical null model, and limitations. Across orthogroups, concatenate the new P columns
and use the **sum of per-run family sizes**, counting each run once, including
empty runs. Do not sum a repeated per-row family-size column or count only
emitted candidates. Do not combine the old diagnostic P and new endpoint P
into one interchangeable column. GeneGalleon's existing asymptotic-only
consumer is not automatically switched by this opt-in CSUBST feature.

## What remains unresolved

The mathematical result assumes parameters fixed independently of the tested
data and a correctly specified model. IQ-TREE estimates topology/lengths,
frequencies, Q parameters and rate mixtures from sequence data. Freezing those
estimates computationally does **not** make them statistically independent.
Neither reciprocal likelihood ratios nor BH repairs that nuisance estimation
or model misspecification. Production results are model-conditional evidence,
not a claim of universally calibrated FDR. The current validation uses known
parameters; a CLI smoke test with a fitted model checks integration, not FDR.

The sparse alternative mixture also trades power for finite-sample validity.
See [the measured calibration and runtime report](../reports/scan_analytic_20260910/REPORT.md)
before choosing this mode. It remains opt-in rather than silently replacing
the existing diagnostic or changing GeneGalleon output semantics.

## Reproducible validation

- `tests/unit/test_scan_analytic.py` independently enumerates every hidden and
  observed state on a small tree with unequal lengths and mixed rates. It
  checks the likelihoods, unit mean of E and superuniform P at every attainable
  threshold, as well as FDR-family accounting and empty output.
- `tools/benchmark_scan_analytic.py` repeats full discovery under the independent
  four-codon CTMC/pruner, with null and injected-signal sites, site filtering,
  uncertain ancestry and unequal branches. It compares both selected-family
  and full-family versions of the old diagnostic against the new method.
- `tools/benchmark_scan_analytic_codon.py` measures the 61-state kernel and
  illustrates attainable small P values in strong, prespecified patterns.

References: [Felsenstein's tree likelihood algorithm (1981)](https://pubmed.ncbi.nlm.nih.gov/7288891/)
and [Wang and Ramdas, FDR control with e-values (2022)](https://academic.oup.com/jrsssb/article/84/3/822/7056146).
The particular endpoint alternatives here are an implementation choice, not a
method taken from either paper.

## Independently trained alternative mixtures

Add `--scan_analytic_profile profile.json` to use a frozen mixture trained on
independent data. Without it, the equal four-component mixture above remains
the default. Each version-1 JSON atom contains `multiplier` (finite >=1, or
JSON `null` for the infinite limit), `participation` (0..1), and `weight`
(nonnegative; weights sum to one). For example:

```json
{"version": 1, "atoms": [
  {"multiplier": 10, "participation": 1, "weight": 0.7},
  {"multiplier": null, "participation": 0.5, "weight": 0.3}
]}
```

Participation rho replaces each foreground transition by
`rho * P_tilt + (1-rho) * P_null`. This marginalizes independent participation
of individual foreground branches, conditional on the shared mixture atom.
Every kernel and the weighted mixture remain normalized. The evidence is
`sum(weight_a * P_a(D)) / P0(D)`, so the fixed-correct-null result is unchanged.
The exact atoms and optional training provenance are saved in the inference
JSON. Never train weights, select a profile, or retune effect ranges using the
tested alignment or its discovery results.

`tools/train_scan_analytic.py` trains on independently generated CTMC histories,
freezes a hashed profile, and then evaluates untouched seeds. Foreground Q
matrices increase K-to-N jump rates; tip states are never overwritten.
The candidate basis includes eight-point Gauss–Legendre approximations to
a log-uniform endpoint multiplier on [1,1000], full/half participation, and
infinite-limit alternatives. Training maximizes mean predictive log likelihood
over all training signal sites. It does not optimize selected-hit counts.
This four-codon study does not establish transfer to arbitrary codon models.

The paired study uses the *same selected trait×match BH* for the Poisson
diagnostic, default endpoint mixture and learned mixture. That comparison is
diagnostic and differs from the production full-family correction above.
See [the independent training/holdout report](../reports/scan_analytic_training_20260910/REPORT.md).


When combined with `--scan_observation joint` or `bridge`, the analytical
likelihood retains the original tip emissions, including missingness and
partial ambiguity. It never treats imputed tip posteriors as fresh observations.
The observation mode affects candidate discovery and the exploratory rate score;
the analytical P remains a test of endpoints under its frozen null.
Joint/bridge mode's uniform-model restrictions still apply.

# Scan scores and inference

An opt-in simulation-free alternative is documented in
[analytical endpoint-enrichment inference](SCAN_ANALYTICAL.md). It has its own
P columns and prespecified full-family correction; the diagnostics below retain
their existing interpretation.

`scan` discovers recurrent foreground substitutions. The discovery threshold,
foreground support and rate contrast use the same data. Consequently, the
analytic tail and its BH adjustment are **exploratory diagnostics**, not
selection-adjusted significance or guaranteed FDR control.

## Score, posterior mass and sparse observations

`score_rate_enrichment` is the negative base-10 logarithm of the one-sided
Poisson asymptotic tail. Larger values indicate greater foreground rate
enrichment. Non-enrichment has score zero. The log-tail calculation preserves
ordering even when its exponentiated diagnostic underflows to zero.

`p_rate_enrichment_asymptotic` retains that approximate tail for inspection.
For equal exposures and foreground/background counts 2/0, it is about
0.04794548. For **independent integer Poisson observations with fixed
exposures**, conditioning on the total instead gives an exact binomial upper
tail of 0.25. That comparison is a small-sample diagnostic, not a replacement
test for scan output.

For marginal/joint observations, both `called` and `posterior_sum` sum
fractional posterior event mass. Bridge observations sum posterior mean jump
counts, which can exceed one; see [joint/bridge scan](SCAN_CTMC.md).
`called` sums only events that pass the event threshold; it does not convert
them to integer observations. Neither rounding the mass nor switching to
`called` establishes the assumptions of an exact integer-count test.

Keep the effect size (`rate_ratio`), foreground/background mass, exposure and
support alongside the score. Zero exposure produces undefined scores and is marked
`scan_rate_testable=false`, with `scan_rate_undefined_reason` identifying the
no-test outcome. The instantaneous-Q exposure can be zero for multi-step
endpoint changes. A rate contrast is not identifiable in that case.
This does not repair the instantaneous-rate/endpoint mismatch itself. Rate enrichment alone does not identify selection or a
causal phenotypic effect.

## BH family: trait × match

BH is applied separately to **output candidates within each trait and
`scan_match`**. Sites and particular from/to state identities are pooled
within that group. For example, all output `any2spe` candidates for one trait
share a BH family, including candidates for different destination residues.
`spe2spe` candidates belong to a separate family. Match classes overlap and
are not disjoint events.

The two BH output columns are:

- `q_rate_enrichment_asymptotic_by_trait_match`
- `q_rate_enrichment_empirical_by_trait_match`

`scan_bh_family_id` is a JSON pair `[trait, match]`. `scan_bh_family_size`
counts all selected rows in that family; `scan_bh_valid_asymptotic_count` and
`scan_bh_valid_empirical_count` record how many finite values were adjusted.
Undefined inputs retain undefined q values.

These adjustments do not include discovery selection, comparison across match
classes, different genes/runs, or choosing settings after inspecting results.
Even candidate-wise empirical P values need a valid selection-aware null and
appropriate dependence assumptions before BH can be interpreted as FDR
control. Applying BH or a more conservative multiplicity adjustment cannot
repair invalid individual P values.

## Calibration modes

| `--scan_pvalue_calibration` | Repeated procedure | Output and scope |
| --- | --- | --- |
| `none` | No null replicates | Exploratory score and asymptotic/BH diagnostics |
| `candidate_fixed` | Change foreground clades; retest observed candidates against the fixed posterior tensors | Candidate-wise `p_rate_enrichment_empirical`; discovery selection is not corrected |
| `full_scan` (default) | Change foreground clades; repeat discovery/support selection against the fixed posterior tensors | Candidate-wise empirical values plus `p_rate_enrichment_empirical_maxT`, compared with the maximum over all testable output candidates in each replicate |
| `parametric` | Hold the fitted model/lengths fixed; simulate new tips, repeat exact ASR and joint/bridge discovery | Global `p_rate_enrichment_empirical_maxT`; see [fixed-model limits](SCAN_CTMC.md#parametric-calibration) |
| `parametric_bootstrap` | Simulate a fitted uniform GY codon null; refit IQ-TREE/ASR and repeat recoding, filtering, exposure, discovery and support selection | `p_rate_enrichment_bootstrap_maxT`, compared with each complete replicate's maximum score |

All maximum-score families cover **all traits, requested matches and testable
candidates in that one scan run**, as recorded by `scan_maxT_scope`. This is
separate from the trait × match BH family. No BH adjustment is applied to an
already maximum-score-adjusted P value.

The clade methods require exchangeability under their actual sampling rules.
They use a fixed, uniform space of eligible, size-binned nonoverlapping clade
assignments, including the observed assignment, with no retry under a different
space. They currently accept one trait per run; multi-trait bootstrap and
uncalibrated candidate listing remain available. `full_scan` repeats discovery
but does not simulate sequences or rerun ASR. Its conditional assignment
inference is not a claim of universal FWER control. In candidate-wise
`full_scan` output, absence of that candidate in a successful null draw is a
no-test outcome. See [clade calibration](SCAN_CALIBRATION.md) for the exact
assignment law, eligibility, enumeration and diagnostics.

Calibration compares the **same score** in observation and null replicates.
Upper-tail ties are included, with a relative/absolute tolerance of 1e-12,
and the Monte Carlo estimate is `(1 + exceedances) / (B + 1)`. If the clade
assignment space is completely enumerated, the tail is `exceedances / M`
with no extra +1; bootstrap always uses Monte Carlo. No foreground
enrichment returns P=1. This formula alone does not establish exchangeability
or account for estimated nuisance parameters.

## Fitted parametric bootstrap

Example:

```sh
csubst scan --alignment_file alignment.fa --rooted_tree_file rooted.nwk \
  --foreground foreground.tsv --iqtree_model GY+F \
  --scan_match any2spe,spe2spe \
  --scan_pvalue_calibration parametric_bootstrap \
  --scan_n_permutations 999 --scan_permutation_seed 51 \
  --scan_site_plot_filter parametric_bootstrap --scan_site_plot_alpha 0.05 \
  --outdir scan_bootstrap
```

The current fitted-null implementation supports **`GY+F` and `GY+FQ`, with
uniform site rates**, ordinary amino acids and codon-to-AA recodings. Other
codon families, rate mixtures, model selection and 3Di are explicitly rejected
in this mode. Existing exploratory and clade modes remain available for
their previous inputs. A codon null does not automatically validate a 3Di
observation/prediction process.

The procedure is:

1. Fit the supplied alignment on its supplied topology with no special
   foreground effect. Before analysis filtering, capture full alignment
   length, the fitted generator, fitted branch lengths and missingness mask.
2. Read kappa/omega from the retained IQ-TREE checkpoint, and empirical
   frequencies from the verbose log (or use equal frequencies for `+FQ`).
   Reconstruct the generator in the explicit codon order. Independently
   recompute the alignment likelihood by scaled pruning. A finite mismatch
   with IQ-TREE's reported likelihood emits a warning and bootstrap continues
   with the reconstructed generator. Record both likelihoods, their absolute
   difference, tolerance and check status in provenance. Nonfinite likelihoods
   remain errors. Old cached runs without the required
   precise model output are refitted.
3. Generate each complete alignment by the CTMC transition `exp(Q × length)`;
   include invariant sites. Fix topology, root position, foreground,
   alignment length and the original fully-missing-codon mask. No fitted
   foreground enrichment is injected.
4. In a fresh directory, refit model parameters and branch lengths and
   reconstruct ancestors. Re-estimate automatic recoding from that replicate
   when requested. A separately supplied training alignment remains a fixed
   external input. Repeat the original site filter, branch rescaling,
   exposure and candidate/support selection. Do not reuse observed ASR or
   an observed mask of retained sites.
5. Use the maximum score across that run's candidates for the null reference.
   A successfully processed dataset without candidates remains in the
   reference. When every site is excluded by the configured scan filter,
   scan writes an empty result and records that no-test outcome.

Joint/bridge scan observations are supported with endpoint exposure and raw
model lengths. The original observation mode and threshold are repeated in
every bootstrap child. These modes use checkpoint/log model precision in the
observed and each simulated analysis.

An explicit set of precomputed IQ-TREE intermediate inputs is not supported
for bootstrap: it would not define a reproducible refitting procedure for
new pseudo-alignments. Unambiguous sense codons and fully missing codons
(`---`, `NNN`, `???`) are supported. Partial ambiguity needs an explicit
observation model and is rejected. Model reconstruction and the likelihood
check are validated with IQ-TREE 2.3.6; incompatible checkpoint/log formats
fail explicitly rather than using rounded parameters.

`--scan_n_permutations` denotes the number of null datasets in this mode.
Replicates currently run sequentially; each IQ-TREE fit uses `--threads`.
Bootstrap requires a nonnegative `--scan_permutation_seed`.
Seeds are deterministic functions of the base seed and replicate index and
are supplied to both the simulator and IQ-TREE. Scan fits use verbose logs
so the observation and each replicate read parameters at the same precision.

The fitted generating parameters are held fixed during simulation, but are
re-estimated during analysis of each pseudo-alignment. This is a
**model-conditional parametric bootstrap**, not an exact test. Its quality
depends on the fitted null and nuisance-parameter estimation. Complete-null
calibration does not by itself prove strong FWER control under partial
alternatives, robustness to model misspecification, or validity across
multiple independently selected runs.

## Diagnostics, failures and plotting

`csubst_scan_inference.json` is written even for an empty candidate table.
It records the score, BH and maximum-score scopes, calibration mode, requested
replicate count, seed and no-test reason. Row-level `scan_inference_status`
separates exploratory diagnostics, clade diagnostics, pending bootstrap,
completed model-conditional bootstrap and failed calibration.

Bootstrap creates a unique `csubst_scan_bootstrap_*` directory under the
output directory, containing `model.npz`, `manifest.json`, and per-replicate
alignments, commands, IQ-TREE outputs, scan tables and logs. It does not reuse
these directories on a later run. The manifest records successful empty
draws, seeds, model-likelihood validation and any failure. Keep these files
with the result for reproducibility; failed draws are never dropped or
replaced with new draws to improve a P value.

If any clade replicate fails, its calibration P/q columns remain undefined;
success and failure counts are still reported. Bootstrap stops at its first
failed draw and likewise leaves bootstrap P values undefined. It does not
divide by the number of successful draws. A candidate with finite nonnegative mass and zero target or control exposure
is explicitly untestable; its score for selection is minus infinity and the
replicate stays in the reference. Its output P values remain undefined.
NaN/infinite/negative inputs and other unexplained undefined scores are
failures. They cannot be reclassified as empty successful families.

`--scan_site_plot_filter` controls display only. `analytical` uses the
asymptotic diagnostic; `empirical` and `full_scan` use the corresponding
clade diagnostics; `parametric_bootstrap` requires that calibration mode.
The inclusive `--scan_site_plot_alpha` cutoff does not turn diagnostic values
into validated significance tests. An unrestricted plot remains the default.

## Validation and output migration

Unit/integration checks cover the sparse 2/0 example, unequal exposure,
fractional mass, extreme tails, trait × match grouping, successful empty
draws, failed draws, inclusive ties, precision of serialized scores and
observed/replicate procedure identity.

`tools/validate_scan_calibration.py` measures repeated-discovery error rates
under an independent four-codon CTMC with exact marginal-posterior pruning.
It covers sparse changes, unequal branches, uncertain ancestry/nonuniform
known site rates, two traits/two matches, site filtering and partial
alternatives. It uses separate reference and validation datasets and reports
binomial confidence intervals. It **does not refit IQ-TREE parameters**, so
its operating characteristics must not be reported as fitted-bootstrap FWER.
Real IQ-TREE bootstrap runs are a separate end-to-end validation.

Output migration:

- `p_rate_enrichment` → `p_rate_enrichment_asymptotic`.
- `q_rate_enrichment_by_trait_match` →
  `q_rate_enrichment_asymptotic_by_trait_match`.
- Global and trait-only q columns were removed; empirical BH is also only
  `q_rate_enrichment_empirical_by_trait_match`.
- Use `score_rate_enrichment` for ranking and resampling. Its TSV serialization
  preserves floating-point precision; diagnostic P/q values use scientific
  notation. Never combine old q values with newly calibrated P values.

## Statistical references

- [R exact Poisson test documentation](https://www.stat.ethz.ch/R-manual/R-devel/library/stats/html/poisson.test.html): conditional binomial test for two integer Poisson counts.
- [Benjamini & Hochberg (1995)](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1995.tb02031.x): BH and its original assumptions.
- [Winkler et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/): permutation inference and exchangeability.
- [IQ-TREE 2.3.6 ModelCodon](https://github.com/iqtree/iqtree2/blob/v2.3.6/model/modelcodon.cpp) and [alignment frequencies](https://github.com/iqtree/iqtree2/blob/v2.3.6/alignment/alignment.cpp): fitted codon model and empirical-frequency output.

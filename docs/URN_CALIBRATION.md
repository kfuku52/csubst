# Empirical urn expectations and pipeline validation

`--expectation_method urn` defines a fitted count null. Site weights, branch
totals, ancestral-state estimates, filtering and recoding can all depend on
the same alignment. Conditional randomization under those fitted quantities
does not establish false-positive control for the entire analysis.

These options remain experimental. The ordinary default expectation method
is unchanged. A fixed amino-acid grouping tests convergence to that group;
it does not test identity of individual amino acids or structural states.

## Learning ASRV weights

The default `--asrv_training_branches all` preserves the original training
population. Alternative training sets apply to `pool`, `sn`, `each` and
`file_each` with urn expectations:

- `background`: exclude the union of foreground and marginal target branches
  across all traits, and exclude the root. Requires at least one target and
  at least one remaining training branch.
- A comma-separated list of numerical branch IDs, such as `1,4,7`: use
  exactly those incoming branches. Duplicate, unknown and root IDs fail.

The selection changes the site-mass summary only. Evaluation branch totals
and observed counts retain all branches. Both dense and sparse tensors follow
the same rule. The evaluation branch's nonmissing-site mask is applied after
learning the shared profile. An explicit numerical training set must be
chosen for the current input tree; it is not portable across different trees.

For example, with an existing foreground definition:

```bash
csubst search --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --foreground foreground.tsv --expectation_method urn --asrv each \
  --asrv_training_branches background --asrv_concentration 2 --asrv_report yes
```

Excluding foreground branches removes their direct contribution to the site
mass profile. It does not make ancestral-state estimates independent: ASR
still uses the complete alignment and tree. Custom training/concentration with active
epistasis is currently rejected because its separate fitting still uses all
branches and per-site alpha (review ID 3). This is a sensitivity analysis,
not a claim of independence or proof of calibration. A site-rate file fitted
to the tested alignment is also data-dependent.

## Two different ASRV concentration conventions

In `sn/each/file_each`, the legacy convention adds
`--asrv_dirichlet_alpha` (default 1) to **each valid site**. For branch b:

`p[b,s] = (c[s] + alpha) / (sum_valid c + alpha * number_valid_sites)`.

Thus a category with total mass 2 on 100 valid sites receives total prior
mass 100 at alpha=1. This is a choice of regularization, not a universal
optimal prior. `pool` and `file` do not use that alpha.

The optional `--asrv_concentration TAU` instead adds a fixed **total** uniform
prior mass on each evaluation branch's valid sites:

`p[b,s] = (c[s] + tau / number_valid_sites) / (sum_valid c + tau)`.

It overrides the per-site alpha; it is not added to it. Zero empirical mass
with positive concentration yields the stated uniform prior. With zero
concentration it yields zero weights, explicitly identified in diagnostics.
An empty evaluation mask always yields zero weights. Negative or nonfinite
masses and concentration are rejected.

No automatic tau fitting or empirical prior has been introduced. ASRV
concentration is separate from `--pseudocount_alpha`, which acts on omega
counts, and from `--omega_pvalue_nbinom_alpha`, which models overdispersion.

`file_each` retains the original definition `c[s] = empirical_mass[s] *
file_rate[s]` before smoothing. With a fixed prior mass, scaling all file
rates changes shrinkage strength. The file-rate units therefore matter;
this hybrid is not a pure rate-profile prior. Zero file rates can receive
positive probability from the uniform prior. Neither convention is silently
replaced by a pooled or file prior.

## Wallenius precision

`--urn_wallenius_expectation auto` uses exact boundary and uniform cases,
exact one/two-draw formulas, and the original exact sequential subset
enumeration domain of at most 20 positive-weight sites. Larger nonuniform
problems with more than two draws use the approximate mean equation. The
small-draw formulas extend exactness without expanding subset enumeration
across thousands of categories. Scaling all weights leaves the
answer unchanged, including very small weights.

`--urn_wallenius_expectation exact` rejects a case that would need the
approximation. It is available only for the Wallenius urn expectation model.
No universal error bound is asserted for `auto`. Detailed provenance records
the methods encountered for the input categories and branch draw sizes.
Stochastic rounding uses both neighboring integer draw sizes.

The former 21-site, two-draw case with weights `[20,1,...,1]` returned heavy
site inclusion probability 0.730583. The exact ordered-pair calculation and
the new implementation return 0.756410. This fixes that numerical example;
it does not establish biological calibration of the urn null.

For an external reference, [BiasedUrn's manual](https://cran.r-universe.dev/BiasedUrn/doc/manual.html)
also distinguishes approximate multivariate means from full enumeration.

## Meaning of Poisson and negative-binomial nulls

`poisson` obtains a mean from the selected urn overlap model and shared
ASRV profile, then draws Poisson **counts**. `poisson_full` instead obtains
branch-specific profiles and totals directly from observed branch/site
mass, passes them through the same urn-overlap calculation, and then draws
Poisson counts. It bypasses ASRV smoothing for that null mean. Its mean can
differ from the EC used in the observed effect-size calculation.

Neither option simulates a full phylogeny, nor a joint set of site events.
The urn overlap may depend on rounding and the inclusion-probability
approximation. The word "full" refers to retaining observed branch/site
association, not to a complete generative evolutionary model. Observed
concentration can be absorbed into the null. Changing ASRV affects the
ordinary EC even when the `poisson_full` null mean bypasses it.

`nbinom auto` estimates count overdispersion from observed residuals. A real
signal can change that estimate. Its estimation uncertainty and the behavior
of dependent branch combinations need separate validation. This work does
not change the negative-binomial estimator or the conditional P-value engine.

## Learning and applying automatic recoding

`srchisq6` minimizes a maximum taxon-composition chi-square criterion.
`kgbauto6` minimizes a conductance criterion using a same-site residue
co-occurrence matrix; that matrix is not a phylogenetic rate generator.
Neither method directly optimizes omega or its P value.

By default they learn from the analysis alignment. Use
`--nonsyn_recode_training_alignment training.fa` to learn the grouping from
a separate codon alignment, then apply it to the analysis alignment. It is
accepted only for these automatic methods. The training alignment need not
have the same taxa or site count, but it uses the configured genetic code.
Its independence is the investigator's design responsibility.

The training cache includes the alignment SHA-256, amino-acid order and
genetic code. Replacing content at the same path invalidates the cached
statistics. Each full pipeline replicate runs in a fresh process and
relearns the grouping unless a separate training alignment was specified.

`csubst_nonsyn_recoding.metadata.json` accompanies the existing grouping TSV
and records groups, training source/hash, objective score, seed and number
of random starts. Neither determinism nor a separate filename establishes
selection-adjusted inference.

## Provenance output

Urn searches write `csubst_urn_provenance.json` (respecting output prefix).
It records the fitted-null scope, training IDs, concentration convention,
rate-file source, urn/rounding policy, recoding provenance and null model.
`pipeline_calibrated` is always false for this conditional analysis.

With `--asrv_report yes`, a serial audit also covers the four base categories
and reports training mass, empty masks, zero empirical/normalized mass,
minimum/maximum prior fraction and effective site count `1 / sum(p^2)`.
Methods used by Wallenius, including the `poisson_full` means when requested,
are collected in that audit. Without the detailed audit, worker-local method
records can be incomplete and are explicitly marked as such. The report
requires extra calculations and does not affect the statistic.

## Independent pipeline validation

`tools/evaluate_urn_pipeline.py` is a validation harness, not a replacement
for `pomegaC*`. It reruns IQ-TREE, ASR, recoding, ASRV and search for every
configuration and independent simulated dataset. The predeclared statistic
is the maximum omega in the eligible branch family, then the maximum over
all declared configurations. A prespecified `branch_ids` list can restrict
the family to one combination. Eligibility thresholds and `exclude_branch_ids` are repeated in every
dataset; an empty family has statistic negative infinity. Undefined values
in an eligible family fail instead of being silently discarded.

Independent null **calibration** datasets supply the reference distribution.
Separate null and alternative **validation** datasets are ranked against it
using inclusive ties and `(1 + exceedances)/(1 + calibration_count)`.
Binomial intervals count independent datasets, not correlated branch rows.
The summary reports whether the upper 95% interval bound is below the
prespecified FPR limit, provided the Monte Carlo resolution reaches the test
level. This is conditional on the chosen simulator and reference sample; it
does not prove validity under model misspecification or all real datasets.

Generate a small smoke dataset and execute it:

```bash
python tools/generate_urn_validation.py --outdir /tmp/urn-input \
  --calibration 3 --validation 3 --alternatives 2 --sites 300
python tools/evaluate_urn_pipeline.py --manifest /tmp/urn-input/manifest.json \
  --outdir /tmp/urn-validation
```

The small counts above test execution only: minimum possible P is 0.25.
The generator defaults (1000 calibration, 2000 null validation, 200
alternative datasets) are intended for a more substantial assessment.
Inspect resource needs before launching that many full IQ-TREE analyses.

The generator uses a symmetric single-step sense-codon CTMC with mean rate
one, independent of the urn/tensor code. `--heterogeneous`, `--missing`,
`--sites` and `--signal-sites` vary selected stress conditions. Alternatives
inject shared terminal GCT states in tips a and e and serve only as a
synthetic sensitivity control, not a fitted alternative evolutionary model.
The generated manifest excludes root/root-adjacent branches by their known
topological IDs, because CSUBST intentionally leaves their EC undefined.
The supplied configurations compare all-branch ASRV, background ASRV with
fixed total concentration, and automatic recoding with background ASRV.

For another independent simulator, supply manifest schema 1 with:

- `simulator`: description/settings and `independent_replicates: true`;
  this is an input contract, not a check that proves independence.
- `analysis_options`: common CLI option/value mapping.
- `configurations`: objects with unique `name` and `options` mappings.
- `selection`: `statistic`, `arity`, optional `minimum_counts` mapping and
  optional `branch_ids` / `exclude_branch_ids` lists. Declare these before inspecting validation.
- `replicates`: unique `id`, `role` (`calibration`/`validation`), `truth`
  (`null`/`alternative`), `alignment`, `tree`, optional `foreground` and
  optional `training_alignment`. Relative paths resolve beside the manifest.

Run-local inputs, output paths and seeds cannot be overridden. Inputs are
copied into fresh output directories; the runner never overwrites a prior
run or writes next to the supplied inputs. `command.json` and `process.log`
record every execution. The harness disables conditional omega P values,
and validates the complete selected effect-size statistic directly.

Outputs are `replicate_scores.tsv`, `validation.tsv`, `summary.json` and
all individual run artifacts. The raw per-configuration scores remain
available to examine signal attenuation as well as selection.

## Remaining dependencies and acceptance scope

The [pseudocount/omega-P implementation](PSEUDOCOUNT_PVALUES.md) shares
observed/null smoothing and provides a compatible-mean joint Poisson category
null (scientific review IDs 2 and 4). Its conditional-count checks do not
replace full-pipeline validation with independently generated alignments.
Epistasis cross-validation (ID 3), 3Di measurement (ID 9), long-tail
calibration (ID 10), site filtering (ID 11) and search-family selection
(ID 13) require their own appropriate configurations and independent tests.

This change supplies controlled training, provenance, a numerical correction
and a repeatable full-pipeline validation mechanism. It does not designate a
new default alpha/tau, assert uniform FPR/power guarantees, or certify
automatic recoding from numerical unit tests alone.

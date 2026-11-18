# Empirical urn expectations and pipeline validation

`--expectation_method urn` defines a fitted count null. Site weights, branch
totals, ancestral-state estimates, filtering and recoding can all depend on
the same alignment. Conditional randomization under those fitted quantities
does not establish false-positive control for the entire analysis.

The ordinary search expectation is `codon_model`. A fixed amino-acid
grouping tests convergence to that group. Consult the effective `asrv`
and the urn settings in `csubst_urn_provenance.json` for every urn run.

## Default empirical urn

With `--expectation_method urn`, the default `--asrv auto` selects
`empirical_all` for joint codon expectations under the Wallenius urn with
`any2spe`, `any2any`, or `any2dif` output. This fits one shared profile to
all eligible branches, **including the evaluated branches**, for K=2 and
higher arities. The root and its immediate children do not contribute to
training or receive expected counts. Adding a higher arity or selecting a
foreground does not change the training population.

The optional pairwise mode `--asrv empirical` assigns each branch to one
of four folds by its ID modulo four. A pair uses only branches outside
its members' folds. Under `auto`, source-specific statistics, marginal
ASR, recoding, other urn laws, count P values, long-tail calibration and
custom site-weight settings select `each`. Explicitly requesting an
incompatible empirical mode raises an error.

Each training branch's N+S endpoint masses contribute to a site-rate
profile. The mixture with uniform site weights is fitted by
leave-one-training-branch-out log scoring. N weights combine these
rates with empirical parent-to-destination amino-acid rates learned
with a 0.5-count destination-pooled prior. Synonymous site weights combine
parental codon compatibility with the empirical site profile based on a
0.5-count per-site prior.

In `empirical_all`, **each branch keeps its own observed N and S totals**.
These totals are sums of joint endpoint posterior probabilities and can be
fractional. For N, sum the site weights for each destination amino acid on
the branch. Allocate the branch's observed N total in proportion to these
sums across destinations. A branch with zero observed N mass receives zero
N draws. Thus ECN asks how much overlap is expected given the amount of
amino-acid change on each branch. Constraints that relax at the same sites
on several branches can still change where the observed events occur.

For S, the branch total includes all synonymous families. Destination rates
are fully pooled across eligible branches. For a branch `b` and destination
codon `d`, the opportunity
`U[b,d]` sums the S site weights times the parental probability of another
synonymous codon at eligible sites. If `S[b]` is the branch's observed S mass
and `L[b]` its number of eligible sites, the rate for `d` is its observed
tree-wide mass divided by `sum_b [(S[b] / L[b]) * U[b,d]]`. Allocate `S[b]`
in proportion to `U[b,d]` times these rates. A destination with no training
mass receives zero draws. This fit is reused at each arity. The pairwise
cross-fitted `empirical` mode uses observed branch-by-destination S masses
and predicts N totals from branch N+S mass and amino-acid opportunities.
It scales the predicted N mass to the observed N mass across training branches.
[PEPC accuracy and timing comparison](../reports/omega_low_count_20261116/PEPC_FULL_POOLING.md)

For each site and destination, the K-way overlap is the product of the K
branches’ Wallenius inclusion probabilities.
`any2spe` sums overlaps within destinations. `any2any` sums overlaps
over destination combinations within the same substitution group, so
`any2dif` is their difference. In the absence of eligible training
branches for a pair's held-out folds, the corresponding expected counts
are unavailable. The ASR conditions on the entire alignment and the fitted
codon model. Folds affect the learning of urn weights and N totals.
Provenance records the training scope, eligible branch IDs and either a
single fitted uniform weight or diagnostics for the held-out folds.
Including evaluated branches can absorb some of their signal into the
expected counts; see the [PEPC all-branch comparison](../reports/omega_low_count_20261116/PEPC_ALL_BRANCH_URN.md).
[PEPC estimator study](../reports/omega_low_count_20261116/URN_ECN_BIAS.md)

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
structure weighting is rejected because it defines its own clade-specific
training sets and per-site alpha. See [structure weighting](STRUCTURE_WEIGHTING.md). This is a sensitivity analysis,
not a claim of independence or proof of calibration. A site-rate file fitted
to the tested alignment is also data-dependent.

With joint endpoint posteriors, each S or N urn uses the eligible branch-sites
of its corresponding substitution tensor. Fully ambiguous tips with no
observation likelihood information cannot receive urn weight on their terminal
branches. Event thresholds change the retained mass and branch totals.

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
counts, and from `--omegaC_pvalue_nbinom_alpha`, which models overdispersion.

`file_each` retains the original definition `c[s] = empirical_mass[s] *
file_rate[s]` before smoothing. With a fixed prior mass, scaling all file
rates changes shrinkage strength. The file-rate units therefore matter;
this hybrid is not a pure rate-profile prior. Zero file rates can receive
positive probability from the uniform prior. Neither convention is silently
replaced by a pooled or file prior.

`file` and `file_each` use the posterior mean `Rate` column of the IQ-TREE
`.rate` file when present and use `C_Rate` otherwise. With a fitted `+R` or
`+G` model these modes require site-rate rows. The chosen column is recorded as
`site_rate_column` in the urn provenance.

`--asrv file --asrv_posterior_draws N` with joint Wallenius urn expectations
uses the fitted codon rate-category posterior. It draws one category per
retained site, shares that assignment across the tree and the S/N channels,
and averages conditional expected overlaps over the assignments. Branch
totals remain fixed at their inferred values. Use this experimental option
for sensitivity analysis of rate-category uncertainty. The default of zero
draws uses fixed site weights. The option supports EC and omegaC for codon
or amino-acid recoding and uses `--random_seed` for reproducibility.
The conditional P-value, long-tail
calibration, and fixed-weight ASRV report options require fixed site weights.

## Wallenius precision

For Wallenius urn expectations, the default `--urn_wallenius_expectation auto`
uses exact boundary and uniform cases,
exact one/two-draw formulas, exact subset enumeration for at most 20
positive-weight sites, and bounded exact recurrences for two to four distinct
positive weights. The recurrences track selected counts from all groups
except the largest and share each group's mean across its sites.
They run for up to 256 draws and 8,192 count-state updates, bounded by
`draws * product(1 + min(draws, group_size))` over the tracked groups. Other larger nonuniform
problems use the approximate mean equation. Scaling all weights leaves the
answer unchanged, including very small weights. [Fog (2008)](https://doi.org/10.1080/03610910701790269)

The approximate mean equation uses a short bounded clock when the draw
size is small relative to the sum of the weights scaled to a maximum of
one. The general solver uses log time with bounds derived from the
smallest and largest positive weights. Both solve for inclusion
probabilities that sum to the requested draw size.

`--urn_wallenius_expectation exact` rejects a case that would need the
approximation. It is available only for the Wallenius urn expectation model.
`--urn_wallenius_expectation quadrature` checks the numerical precision of
ECN and ECS when weights concentrate on a few sites. It keeps the exact
cases. For the remaining profiles, it integrates the site's exponential
arrival density against the probability that fewer than
the required number of other sites have arrived. A truncated count recurrence
computes this probability. The adaptive quadrature checks convergence and
that inclusion probabilities sum to the draw size. Runtime grows with the
number of sites and the shorter of the selected and omitted counts.
The solver raises an error if it fails to converge or one integration point
would require more than 64 MiB for the count table. The quadrature error
check combines relative and absolute tolerances.
No universal error bound is asserted for `auto`. Detailed provenance records
the methods encountered for the input categories and branch draw sizes.
Stochastic rounding uses both neighboring integer draw sizes.

The former 21-site, two-draw case with weights `[20,1,...,1]` returned heavy
site inclusion probability 0.730583. The exact ordered-pair calculation and
the new implementation return 0.756410. This fixes that numerical example;
it does not establish biological calibration of the urn null.

For an external reference, [BiasedUrn's manual](https://cran.r-universe.dev/BiasedUrn/doc/manual.html)
also distinguishes approximate multivariate means from full enumeration.

## Fisher subset weights

Fisher urn expectations and sampling use the same recurrence for sums of
fixed-size subset weights. These sums are accumulated in log space, including
when dividing a positive site weight by the largest weight would round it to
zero. The recurrence applies to samples weighted by the product of the selected
site weights. [Chen, Dempster and Liu (1994)](https://doi.org/10.1093/biomet/81.3.457)

When most positive-weight sites are selected, inclusion probabilities sum
over the smaller omitted subset with reciprocal weights. A site's inclusion
probability is computed by summing the omitted subsets that exclude it.
Dense samples use this recurrence for each forward choice as well.
For equal weights, the next inclusion chance is the remaining draw size
divided by the number of remaining sites.
When a Fisher suffix table would exceed 64 MiB, expectation and sampling
retain rows at block boundaries and reconstruct one block at a time.
The extra recurrence pass reduces peak memory on large draws.

## Hypergeometric site-subset draws

Each null replicate samples a subset independently for each branch,
conditional on the fitted site weights and the rounded branch draw size.
Combinations containing the same branch share that branch's subset within
the replicate. Equal weights and draw sizes on distinct branches still
produce independent subsets, including with deterministic rounding.

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
Training mass is `null` if the raw total exceeds the representable range.
The audit still reports finite prior fractions and effective site counts.
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
The default configurations compare all-branch ASRV, background ASRV with
fixed total concentration, and automatic recoding with background ASRV.
For a paired comparison of `file`, `each`, and `file_each` under Wallenius, use
`--compare-freerate-asrv` on the generator. It simulates three rate classes
and fits `GY+FQ+R3` for every replicate and configuration. `summary.json`
reports null rejection rates and alternative detection rates with binomial
intervals in `by_configuration`, using a separate empirical reference for
each method. The top-level summary continues to use the maximum across all
configurations and evaluates selection over the entire family.
`--branch-scale`, `--rate-spread`, and `--missing` vary the simulated
branch lengths, rate dispersion, and missingness. Use independent calibration
and validation datasets within each scenario.
To compare precision or shared-rate assignments, set `asrv: "file"` in the
configuration options and add `urn_wallenius_expectation: "quadrature"` or
`asrv_posterior_draws: 64` to the configurations that use those methods.

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

Empirical ASRV modes (`pool`, `sn`, `each`, `file_each`) load full-tree states,
including in foreground-only searches with branch output disabled. Training
branches and their parent states must be available independently of output
options; this also applies to background/explicit training sets and structure
cross-fitting. Fixed weights (`no`, `file`) can still use selective loading.

# Analysis-site selection and convergence counts

CSUBST retains all sites by default (`--drop_invariant_tip_sites no`).
The optional filters change the analysis target; they are not mathematically
equivalent accelerators for an all-site analysis.

| Criterion | Sites selected for exclusion |
| --- | --- |
| `no` | None |
| `tip_invariant` | Identical unambiguous states across non-missing tips, including a column with only one observed tip. Direct 3Di uses its structural tip states; other routes use codons. |
| `zero_sub_mass` | Observed N and S mass on every analyzed branch is at or below `float_tol`, before `min_sub_pp` thresholding. |

Identical tips do not imply identical ancestors. Even a site with zero observed
substitution mass can have positive model-expected convergence. Removing it can
change ECN/ECS. Codon branch lengths are rescaled using observed counts and the
number of available sites: filtering can change expectations on retained sites,
too. For urn expectations, changing the site universe can change normalized
weights and overlap probabilities. The direction of the change in omegaC is not
fixed.

N and S use the same retained columns. In direct 3Di, a structurally invariant
column may contain synonymous codon changes. The structural mask is a definition
of the selected target, not proof that the column has zero N/S signal. Inspect
these contributions before interpreting a filtered N/S ratio.

Direct 3Di fits and infers ancestral states on all columns for both
`codon_model` and `urn`; requested analysis filtering happens afterward. Cache
format 6 records the direct structural mask for both routes and rejects caches
from the former urn prefiltering workflow. `--sa_state_cache auto` rebuilds
incompatible caches, and `yes` rejects them. No published fixed 3Di Q matrix is
required by this change.

## Fixed-model contribution report

Run an all-site search with the diagnostic enabled:

```bash
csubst search --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --expectation_method codon_model --drop_invariant_tip_sites no \
  --site_filter_report yes --max_arity 2 --outdir all-sites
```

The report currently requires `search`/`analyze`, `--cb yes`, and
`--expectation_method codon_model`, including its native 3Di routing. It does
not attribute urn expectations to sites; that requires an explicit definition
for fixed branch totals and site weights. Unsupported combinations are rejected.

With the default output prefix, the additional files are:

- `csubst_site_filter_sites.tsv`: one-based original alignment position, number
  of tips with nonzero codon-state mass (including ambiguous states with mass),
  structural/codon mask basis, and the two candidate exclusion masks.
- `csubst_site_filter.json`: reference and selection definition, site counts,
  analyzed child/parent branch pairs, tolerance, and inference scope. The mask
  and metadata files are also written for explicitly filtered searches.
- `csubst_site_filter_counts_K.tsv`: for each analyzed K-branch combination and
  criterion, `all`, `retained`, and `excluded` raw OCN/OCS/ECN/ECS counts for the
  requested output categories. Each criterion's all-site row uses the same
  reference. Empty partitions have zero contributions; unavailable expectations
  retain the main analysis's NaN convention.

Q, state posteriors, rates and branch lengths remain those of the all-site run.
Within that fixed reference, all = retained + excluded, up to numerical precision.
The report is computed before pseudocounts and long-tail calibration. It contains
counts, not a separately recalibrated omegaC or selection-adjusted P-value. It
reuses the primary analysis's expected projections without changing them.
Diagnostic projection reductions require additional computation and storage.

The report covers combinations reached by the all-site search. A heuristic
higher-order search does not automatically cover combinations found only by a
filtered run. Use exhaustive searches at the relevant arities when feasible,
or report the intersection and mode-specific candidate sets separately.

## Pipeline sensitivity comparison

Run the same inputs and settings with `no`, `tip_invariant`, and `zero_sub_mass`
in separate output directories. Keep `min_sub_pp`, ASR, recoding, expectation
method, pseudocounts, calibration and search options equal. Compare by original
branch IDs, not row number.

Report candidate-set differences, site counts, all four count components,
uncalibrated and calibrated omegaC, undefined ratios, and rank changes.
A difference between two refitted/rescaled expectations is not an excluded-site
contribution. Obtain the latter from the fixed-model report above. If all columns
would be excluded, CSUBST rejects the filtered analysis; do not substitute an
unrequested site set.

## Null-calibration contract and remaining work

Existing empirical P-values are urn count-null calculations conditional on the
analysis site universe and fitted count-null ingredients. They do not regenerate
tip states or repeat site selection. This implementation does not add empirical
P-values for native 3Di model expectations and does not claim selection-adjusted
FPR control.

A full-pipeline null study must generate complete pseudo-alignments and repeat
ASR, the requested filter, codon rescaling, ASRV/prior estimation, search and the
same final statistic in each replicate. Store the replicate's own mask and
candidate set. Do not reuse the observed mask as if it reproduced selection;
a deliberately fixed-mask conditional analysis is a different inferential target.
For 3Di, independent synthetic N and S alignments do not establish the joint null
of structural and codon changes. State the joint generation assumption or include
the predictor pipeline and validate it separately.

Pseudocount transformations and dif-category joint randomization must be made
consistent first (scientific review IDs 2 and 4). Long-tail calibration and
data-estimated weights/priors need the same treatment in observations and null
replicates (IDs 10 and 12). Record all-sites-excluded, undefined and failed
replicates rather than silently discarding them. For each predeclared setting,
report the number of replicates, rejection rate and binomial confidence interval;
define acceptable calibration error before running the study. Validate power
separately. Numerical count additivity is not evidence of FPR calibration.

Likelihood ascertainment correction such as IQ-TREE's `+ASC` conditions the
likelihood on variable sites. It does not replace this downstream selection/null
contract; full-site fitting here should not receive `+ASC` merely because a
post-fit analysis filter is requested. See the
[IQ-TREE model documentation](https://www.iqtree.org/doc/Substitution-Models#ascertainment-bias-correction).

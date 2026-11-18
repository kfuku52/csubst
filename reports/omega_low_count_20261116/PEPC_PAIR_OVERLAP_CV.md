# Choosing urn destination pooling by held-out branch-pair overlaps

On independent simulations, pooling S destinations reduced ECS error at every
tested arity when aggregated by condition, and cut K=6 ECS WAPE by 46–59%.
The S-only variant increased all-pairs raw omega log-RMSE by about 4%.
Predicting held-out pair overlaps gave little additional gain over complete
pooling or over predicting individual branches. The evidence favors borrowing
destination information for ECS. It does not favor making the N estimator
preserve the observed N total on each branch, or adding this CV procedure to
the default search.

## Oracle and empirical results on the two influential seeds

K=6 expected S summed over the 84 stem combinations divided by the generating
joint null (ideal 1), using the production S site weights:

| Mixed-rate seed | Current urn | Q composition, keep observed branch S total | All Q destination counts | Empirical pair-CV, S only |
|---:|---:|---:|---:|---:|
| 1931 | 5.714 | 0.217 | 0.377 | 0.800 |
| 3271 | 29.653 | 1.037 | 0.556 | 2.913 |

At seed 1931 the share attributable to CCA and CCT fell from 99.996% to
13.7% when replacing all branch destination counts with Q marginals, with
the true joint share at 23.6%. At seed 3271 the GAG share fell from 99.84%
to 38.2% (true joint share 24.0%). The destination counts account for much
of the extreme concentration. With exact unconditional branch counts the
urn still underestimated the generating K=6 overlap in both replicates.
The empirical correction reduced WAPE from 5.595 to 0.445 and from 29.534
to 2.346, respectively. The pair-CV strength reached full pooling on S in
both cases. The N oracle was less consistent: replacing N margins raised
ECN WAPE from 0.307 to 0.389 on seed 1931 and reduced it from 0.604 to
0.493 on seed 3271.
[Oracle diagnostics](pepc_pair_overlap_cv_oracle.tsv)

## Independent evaluation

Each entry is an error ratio to the current urn (lower is better). Pairs
use all eligible branches, K=6 uses the nine designated stems. All three
variants here use the current S site-rate profile.

| Application | Condition | Pair ECN WAPE | Pair ECS WAPE | Pair raw omega log-RMSE | K=6 ECN WAPE | K=6 ECS WAPE |
|---|---|---:|---:|---:|---:|---:|
| S only | Unit | 1.000 | 0.716 | 1.039 | 1.000 | 0.410 |
| S only | Mixed | 1.000 | 0.745 | 1.038 | 1.000 | 0.536 |
| S only | Directional | 1.000 | 0.720 | 1.037 | 1.000 | 0.449 |
| N and S, both observed channel totals | Unit | 1.472 | 0.716 | 1.094 | 1.466 | 0.410 |
| N and S, both observed channel totals | Mixed | 1.288 | 0.745 | 1.115 | 0.965 | 0.536 |
| N and S, both observed channel totals | Directional | 1.476 | 0.720 | 1.088 | 1.134 | 0.449 |
| N and S, production N total | Unit | 1.000 | 0.716 | 1.039 | 1.020 | 0.410 |
| N and S, production N total | Mixed | 1.000 | 0.745 | 1.038 | 0.916 | 0.536 |
| N and S, production N total | Directional | 1.000 | 0.720 | 1.037 | 1.030 | 0.449 |

Each condition contributes 67,016 pair-replicates and 672 K=6 combination-
replicates. The S-only raw omega error ratio had a worst one-sided 95%
upper bound of 1.044 across all-pairs conditions. On the true realized
positive-count subsets its raw omega error ratio instead ranged from
0.980 to 0.992. On all-pairs ECN, applying the same correction to N with
observed branch N totals increased WAPE by 29–48%. Preserving the existing
N total made the pairwise metrics virtually identical to S-only. At K=6,
that anchored N version reduced mixed-rate ECN WAPE by 8.4%, but increased
it by 2–3% in the two unit-rate conditions, with the directional upper
bound at 1.069. It supplied no consistent additional gain. In the
common-site-rate sensitivity analysis, the S-only all-pairs raw omega
upper bound stayed below 1.047. See the linked tables for all controls.

S-only K=6 ECS WAPE on the independent panel was 0.525, 0.559 and 0.575
(unit, mixed and biased). Complete pooling gave 0.505, 0.561 and 0.572.
The matched single-branch loss gave 0.527, 0.560 and 0.575. Ten of 24
S fits selected infinity, and the other fourteen selected a prior mass
of at least 256. Even with the improvement, pooled mixed-rate K=6 ECS
was only 0.564 of the true expectation (current 0.251). Across these
alignments no K=5 or K=6 stem combination had a true S convergence or
had both inferred counts above tolerance. High-order raw omega accuracy
cannot be established from these observations.

On the retrospective panel the S-only K=6 WAPE ratios were 0.618, 0.170
and 0.604 (unit, mixed and biased). The all-pairs raw omega error increased
by 3.7–4.0%. These retrospective checks include a previously identified
extreme seed. Eight new seeds per condition give limited information about
the tail of the seed distribution.

[All metrics and paired intervals](pepc_pair_overlap_cv_summary.tsv) ·
[Per-seed scores](pepc_pair_overlap_cv_results.tsv) ·
[Selected strengths](pepc_pair_overlap_cv_strengths.tsv) ·
[Held-out losses by candidate](pepc_pair_overlap_cv_losses.tsv)

## Study design

Use the same 71-tip PEPC topology, ECMK07 simulations, 520 codons, joint ASR,
and three scenarios described in the [symmetry study](PEPC_URN_SYMMETRY.md).
Each alignment gets its own IQ-TREE ECMK07_GY0K+F fit, with +R4 for mixed
rates. The prospective panel uses eight previously unused seeds per scenario
(4099, 4241, 4381, 4519, 4657, 4793, 4931, 5077). A retrospective panel
reuses the last eight seeds per scenario from the symmetry study. These
panels are summarized separately. The two previously identified influential
mixed-rate seeds (1931 and 3271) are case studies, not a random sample.

Score all 8,377 eligible independent pairs, allowing sisters, and all subsets
of the nine specified stems at K=2 through 6 (36, 84, 126, 126 and 84).
Fitting uses all 138 eligible branches, with no foreground labels or simulation
parameters. Use pooled expected-count ratios to assess signed bias, WAPE for
combination-wise count accuracy, and RMSE of log(1+omega) on the same rows with
both observed posterior counts at least 1e-9. Also use true endpoint counts on
the subset with both realized counts positive. The oracle omega from generating
signal and null expectations is the reference for each row. Keep the symmetry
study's practical 5% error margin and its 20,000 alignment-level paired
bootstrap replicates (one-sided 95% upper error-ratio bound). Tiny posterior
counts can still make raw omega very noisy.

## Diagnostic with known Q

In the two homogeneous mixed-rate alignments, propagate the generating root
distribution down the simulated tree within each rate partition. For each
stem edge, integrate the probability of a change to each destination over
its parent state and sum across sites. This supplies the unconditional
single-branch draw counts. Hold the fitted ancestral posteriors and urn
site profiles fixed and recompute K=6 overlaps, using either (i) only the Q
destination proportions with the original channel total on each branch, or
(ii) the complete Q destination counts. For N the original total is the
production predicted total. For S it is observed. Compare to the joint
generating K=6 expectation, which integrates shared ancestral states.
Identical branch Q and rate partitions gave the same nine-by-81 marginal
count arrays across both seeds. The legacy control reproduces the original
targeted CLI ECN/ECS. Exact margins alone do not impose the joint oracle's
site alignment or ancestor correlations on the urn.

## Empirical estimator

Given observed counts C_bd for branch b and destination d, branch total n_b
and a pooled destination probability p_bd, predict the fraction going to d
by (C_bd + alpha p_bd) / (n_b + alpha). Zero uses the observed composition,
infinity uses the opportunity-based composition. A branch with zero observed
mass uses the pooled composition and zero draws in the observed-total variant.
Fit S and N strengths separately using the same objective. S retains the
observed S total. Test N both with its observed N total and with the existing
predicted N total as the scale, leaving the observed-count information in the
composition formula. The latter recovers the production ECN at infinity.

S uses compatible synonymous parental codons times site rate as urn weights.
Estimate destination coefficients outside the held branches as counts divided
by parental opportunities weighted by S mass per eligible site. Multiply
the destination opportunities of each evaluation branch by those coefficients
and normalize to get its pooled composition. N uses the production parent
amino-acid route estimator, including the route prior and exposure adjustment.
Test the existing S site rate (N+S, fixed 0.5 prior) and the common
cross-validated N+S site rate; N uses its production cross-validated rate.
Wallenius inclusion probabilities and stochastic rounding match the CLI.

### Cross-validation

* Randomly and evenly split the eligible branches into four folds, independent
  of bias labels. Score the 2,021 independent pairs whose members belong to
  the same held fold. Train the site-rate and destination profiles using only
  the other three branch folds.
* Randomly split the sites into two equal folds with the same reproducible
  RNG stream (seed 20261118). On each held branch, estimate its composition
  using one half of the sites, extrapolate its channel total by eligible
  site count, and predict on the opposite half. For anchored N, estimate
  branch effort from training-site N+S and scale the training-branch route
  model by that effort. Training-branch observations at all sites inform
  their site's baseline rate. Both halves are scored in turn.
* The predicted count for each held pair and destination is the sum of the
  products of its two site inclusion probabilities over validation sites.
  The response is the analogous overlap of joint endpoint posterior masses.
  Score every pair and destination, including zeros, by the Poisson mean
  loss mu - y log(mu + 1e-12). This is a predictive loss for fractional
  counts. The method makes no independent-Poisson sampling claim. Compare
  a matched single-branch destination loss under the same folds and a
  fixed complete-pooling control.
* Select from 0, 1, 4, 16, 64, 256, 1024, 4096 and infinity per alignment
  and channel. Refit on all sites and branches at the chosen strengths.
  The IQ-TREE model and parent posteriors are conditional on the whole
  alignment. The experiment does not refit IQ-TREE inside folds. The
  simulation Q is accessed for scoring only after selecting the strengths.

By construction, held-site counts on the focal branches enter the calibration
as responses only, conditional on the fitted ASR. Summing held-site/destination
responses recovers production OCN and OCS for every selected pair. Fractional
predicted draws interpolate the adjacent integer Wallenius inclusion profiles.
An independent numerical comparison covered zero, small, and support-limited
draws and matched the production solver. All final overlaps use the ordinary
production routines. The control checks all scored CLI ECN, ECS and supported
omega rows, with the tolerances from the preceding study.

## Reproduction

CSUBST 1.17.5 working tree based on b782efd, Python 3.12.15, IQ-TREE 3.1.4,
macOS arm64. Use the repository root and fresh working directories. PREVIOUS
and FIRST refer to the last and first eight-seed simulation panels from the
[symmetry study](PEPC_URN_SYMMETRY.md).

```bash
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
script=reports/omega_low_count_20261116/pepc_pair_overlap_cv.py
panel=$(mktemp -d)
scores=$(mktemp -d)
prior_scores=$(mktemp -d)
python "$script" simulate "$panel" --iqtree iqtree --jobs 7
python "$script" score "$panel" --work "$scores" --jobs 5
python "$script" score "$PREVIOUS" --retrospective --work "$prior_scores" --jobs 4
repo=$PWD
(cd "$(mktemp -d)" && python "$repo/$script" diagnose "$FIRST" --case rate_mixture-1931)
(cd "$(mktemp -d)" && python "$repo/$script" diagnose "$PREVIOUS" --case rate_mixture-3271)
(cd "$(mktemp -d)" && python "$repo/$script" one "$FIRST/rate_mixture-1931")
```

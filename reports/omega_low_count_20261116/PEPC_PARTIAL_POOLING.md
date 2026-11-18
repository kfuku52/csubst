# Site-validated pooling of synonymous destinations on the PEPC tree

On six new seeds per condition, pooling reduced combination-level ECS
error at every tested arity. Cross-validation gave nearly all the weight
to the whole-tree destination distribution. Full pooling gave comparable
results. Positive-count raw omega accuracy at K=2 worsened in every
condition. This experiment supports empirical destination pooling as a
candidate for improving ECS, with insufficient evidence for changing the
default omega estimator.

## ECS on the new simulations

Each entry gives summed ECS / generating-null S (ideal 1), followed by
combination-level WAPE (ideal 0). Each row aggregates six alignments.

| Condition | K | Current urn | Site-validated pooling | Full pooling |
|---|---:|---:|---:|---:|
| Unit rate | 2 | 0.847 / 0.350 | 0.876 / 0.226 | 0.876 / 0.226 |
|  | 3 | 0.702 / 0.657 | 0.809 / 0.274 | 0.809 / 0.274 |
|  | 4 | 0.458 / 0.883 | 0.778 / 0.321 | 0.778 / 0.321 |
|  | 5 | 0.256 / 0.979 | 0.798 / 0.372 | 0.799 / 0.371 |
|  | 6 | 0.083 / 0.961 | 0.868 / 0.471 | 0.870 / 0.470 |
| Mixed rates | 2 | 0.946 / 0.383 | 0.996 / 0.280 | 0.995 / 0.280 |
|  | 3 | 0.808 / 0.621 | 0.988 / 0.355 | 0.984 / 0.355 |
|  | 4 | 0.655 / 0.862 | 0.992 / 0.443 | 0.981 / 0.438 |
|  | 5 | 0.552 / 1.083 | 1.025 / 0.587 | 1.002 / 0.573 |
|  | 6 | 0.416 / 1.154 | 1.043 / 0.786 | 1.006 / 0.753 |
| Directional bias | 2 | 0.984 / 0.399 | 1.076 / 0.298 | 1.076 / 0.299 |
|  | 3 | 0.936 / 0.724 | 1.131 / 0.385 | 1.133 / 0.385 |
|  | 4 | 0.877 / 1.042 | 1.225 / 0.493 | 1.230 / 0.493 |
|  | 5 | 0.725 / 1.168 | 1.352 / 0.611 | 1.363 / 0.613 |
|  | 6 | 0.519 / 1.179 | 1.495 / 0.747 | 1.515 / 0.752 |

The cross-validated allocation reduced WAPE relative to the current urn
in 89 of the 90 paired seed-by-arity comparisons. At mixed-rate K=6
it helped in five of six seeds. Each leave-one-seed-out aggregate gave
a lower WAPE (0.671–0.865 versus 1.034–1.193). Even so, seed-specific
ECS ratios ranged from 0.469 to 1.850. Under directional bias, pooling
reduced absolute combination error but moved the aggregate K=6 ratio
from 0.519 to 1.495.

Nine of 18 alignments selected full pooling. On the other nine, the
median weight on the whole-tree distribution across the nine stems
ranged from 94.6% to 99.7%. Repeating the strength fit with a second
five-fold site split (shuffle seed 20261119) changed the
median weight by at most 2.2 percentage points. This second fit did
not enter the ECS or omega estimates. Predicting single-branch
destinations rewards sharing information when branch counts are
sparse. Its objective does not score errors in K-branch overlaps.
[Strengths, validation losses and branch weights](pepc_partial_pooling_fit.tsv)

## Comparison with the earlier mixed-rate seeds

Reapplying the same procedure to the three earlier mixed-rate seeds
(741, 852, 963) yielded K=6 ECS ratios of 0.261, 1.411 and 1.408 and
WAPEs of 1.073, 1.300 and 1.300 for the current urn, cross-validated
pooling and full pooling, respectively. On seed 852 alone the
cross-validated estimate was 3.27 times the reference and its WAPE
rose from 1.342 to 2.736. WAPE fell on both other seeds. Combining
all nine mixed-rate seeds descriptively gave K=6 WAPEs of 1.127,
0.958 and 0.935. Only the six new seeds constitute prospective validation.

## Raw omegaCany2spe

Compute RMSE on the log1p scale against the omega value implied by
the generating signal and null Q. Include only combinations with both
inferred observed counts at or above the search tolerance (1e-9).
This is a conditional assessment. Estimated counts just over the
tolerance can still yield very unstable ratios.

| K=2 condition | Supported / evaluated | Current urn | Site-validated pooling |
|---|---:|---:|---:|
| Unit rate | 181 / 216 | 3.231 | 3.277 |
| Mixed rates | 214 / 216 | 1.451 | 1.554 |
| Directional bias | 192 / 216 | 3.361 | 3.465 |

K=5 and K=6 had zero combinations with any realized synonymous
convergence in the true ancestral sequences across the new alignments,
and none had both inferred observed counts above tolerance. Thus the
direct omega comparison provides no evidence of a high-arity improvement.
[Per-seed ECS errors, count support and omega losses](pepc_partial_pooling_scores.tsv)

## Experiment

Simulate 520 codons under empirical ECMK07 on the bundled PEPC tree. Use
six new seeds (1103, 1217, 1321, 1433, 1549, 1657) in each of three
conditions from the [initial experiment](PEPC_HIGH_ARITY_S_TOTALS.md).
The baseline has unit rates, the mixed-rate null has equally frequent
rates 0, 0.2, 0.8 and 3, and the directional case has a shared target
amino acid on the nine stems at 20% of unit-rate sites. Fit
ECMK07_GY0K+F to each alignment with IQ-TREE, adding +R4 for the
mixed-rate condition. Each method uses the same inferred joint endpoint
posteriors, parental opportunity, site rates and branch-wide S totals.
The fixed nine stems define 36, 84, 126, 126 and 84 combinations at
K = 2, 3, 4, 5 and 6. All 138 eligible branches enter training, including
the nine evaluated branches. The estimator receives no bias labels.

Let C_bd denote the posterior count of synonymous changes on branch b to
codon d, and n_b the sum over d. Estimate the whole-tree destination
coefficient by dividing the total count to d by the sum of parental
opportunities for d weighted by each branch's S count per eligible site.
Opportunity sums the site rate times the probability of a compatible
synonymous parent. Multiplying each branch's opportunities by the fitted
coefficients and normalizing across destinations gives p_bd.
The replacement draw total for a destination is

```text
n_b * (C_bd + alpha * p_bd) / (n_b + alpha).
```

Zero strength returns the current urn. Infinite strength gives an
allocation of n_b * p_bd. The finite-strength weight alpha / (alpha + n_b)
is higher on branches with fewer observations. Recompute Wallenius
site inclusion probabilities using these destination totals.

Choose one strength per alignment by five-fold destination prediction.
Randomly partition sites into balanced folds (fixed shuffle seed 20261118)
and hold out each fold on all branches simultaneously. Fit branch counts,
branch effort and whole-tree destination coefficients using only the
training sites. Score the predicted fractions by held-out S count
times negative log probability, summed across all eligible branches
and destinations. Compare zero, powers of two from 2^-8 to 2^18 in
quarter-octave steps, and infinity. Scoring uses a common uniform
mixture of weight 1e-12 to give finite log loss to unseen destinations.
Each fold shares the alignment-wide IQ-TREE parameter fit. The site-local
S rates have a common normalization that cancels in the forecast. Refit
the destination allocation on all sites using the selected strength.
The simulation Q and realized ancestral sequences enter only the assessment.

Use the joint generating-null S expectation for each combination as the
ECS reference. ECS ratio is the sum of predictions divided by the sum of
references (ideal 1). WAPE is the sum of absolute combination errors
divided by the same sum (ideal 0). Pool the six seeds before division.
Seed-level comparisons pair the same combination sets and alignments.
The independent unit of replication is the alignment, since many
combinations share stems.

The original urn reproduced the CLI ECS for every one of the 8,208 new
combination-by-alignment evaluations (relative tolerance 1e-10 plus
6e-18 for printed output). Both allocation endpoints also reproduced
the earlier mixed-rate pilot at seed 852 for all five arities. The
simulation replay checked tips, true ancestral sequences, and the 36
pairwise Q expectations per alignment. The joint K=2–6 reference passed
direct ancestral-state enumeration on a small tree.

## Reproduction

CSUBST 1.17.5 working tree based on b782efd, IQ-TREE 3.1.4, Python 3.12.15,
macOS arm64. Run from the repository root in the CSUBST analysis
environment, with IQ-TREE available on PATH.

```bash
work=$(mktemp -d)
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/pepc_partial_pooling.py panel "$work" --iqtree iqtree --jobs 6
python reports/omega_low_count_20261116/pepc_partial_pooling.py score "$work" --jobs 4
```

To reproduce the retrospective comparison, set PANEL and HIGH_K to the
two work directories in the [earlier study](PEPC_HIGH_ARITY_S_TOTALS.md).

```bash
prior=$(mktemp -d)
python reports/omega_low_count_20261116/pepc_partial_pooling.py pilot-mixed "$prior" --pilot-panel "$PANEL" --pilot-high-k "$HIGH_K"
```

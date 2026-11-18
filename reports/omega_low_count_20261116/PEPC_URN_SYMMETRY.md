# How much of the empirical urn can N and S share?

On this PEPC topology, a single cross-validated site rate for N and S retained
pairwise accuracy for ECN, ECS and raw omega in two independent eight-seed
panels. Keeping the existing branch draw totals was important for raw omega.
Using empirical routes and predicting the joint N/S split improved ECS
considerably, but raised all-pairs raw omega error. None of the tested
alternatives met the 5% criterion simultaneously for all-pairs raw omega and
both high-arity count errors across conditions in the last validation panel.
The experiment made no change to search defaults.
[All condition and arity estimates with paired intervals](pepc_urn_symmetry_summary.tsv)
and [per-seed sufficient statistics](pepc_urn_symmetry_seeds.tsv).

## Initial independent comparison

Each condition contributes 67,016 pair-replicates on the first eight new seeds.
The table gives the largest error ratio to the current urn across the three
conditions, followed by the largest one-sided 95% bootstrap upper bound.
Ratios below one indicate lower error. Maxima for different columns can
come from different conditions.

| Method | ECN WAPE | ECS WAPE | Observed omega log-RMSE |
|---|---:|---:|---:|
| `current` | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| `shared_cv` | 1.000 (1.000) | 1.009 (1.013) | 1.004 (1.004) |
| `shared_fixed` | 1.027 (1.081) | 1.000 (1.000) | 1.005 (1.005) |
| `observed_both` | 2.680 (2.784) | 1.009 (1.013) | 1.239 (1.255) |
| `own_totals` | 1.435 (1.471) | 0.769 (0.790) | 1.093 (1.103) |
| `separate_scales` | 1.000 (1.000) | 0.685 (0.719) | 1.066 (1.073) |
| `joint_totals` | 1.003 (1.005) | 0.680 (0.714) | 1.066 (1.073) |

Sharing the cross-validated rate met the 5% margin for all three endpoints
in all three all-pairs conditions. Moving both channels to the fixed site rate
gave a worst ECN upper bound of 1.081. On the nine stems at K=6, that
fixed-rate variant increased ECN WAPE to 3.056 times the control under unit
rates and 2.611 times under directional bias. Sharing the CV rate instead
preserved ECN exactly, and gave K=6 ECS WAPE ratios of 0.662 (unit),
1.053 (mixed) and 0.890 (biased). Its mixed-rate upper bound was 1.075.

The joint N+S allocation reduced all-pairs ECS WAPE by 32–42% while raising
observed omega log-RMSE by 5.0–6.6%. With mixed rates the observed ratio
was 1.066 (upper bound 1.073). Using true realized ancestral counts on
positive pairs gave 1.107. Replacing the numerators with generating means
gave 0.236. Joint allocation substantially improved the inferred ratio
of expected counts in this diagnostic, while realized sparse counts
changed the ranking for raw omega.

## Independent replication and the smoothing floor

The last eight seeds per condition gave these all-pairs error ratios and
one-sided upper bounds relative to the current estimator.

| Method | Condition | ECN WAPE | ECS WAPE | Observed omega log-RMSE |
|---|---|---:|---:|---:|
| shared_cv | Unit rate | 1.000 (1.000) | 0.978 (0.988) | 0.993 (0.994) |
| shared_cv | Mixed rates | 1.000 (1.000) | 1.003 (1.007) | 1.004 (1.004) |
| shared_cv | Directional bias | 1.000 (1.000) | 0.979 (0.987) | 0.994 (0.994) |
| joint_totals | Unit rate | 1.003 (1.005) | 0.604 (0.622) | 1.061 (1.070) |
| joint_totals | Mixed rates | 1.003 (1.006) | 0.656 (0.682) | 1.067 (1.076) |
| joint_totals | Directional bias | 1.000 (1.002) | 0.590 (0.616) | 1.061 (1.065) |
| shared_floor | Unit rate | 1.000 (1.000) | 0.978 (0.988) | 0.993 (0.994) |
| shared_floor | Mixed rates | 1.014 (1.023) | 1.000 (1.000) | 1.004 (1.005) |
| shared_floor | Directional bias | 1.000 (1.000) | 0.979 (0.987) | 0.994 (0.994) |

Across the 22 alignments per condition in the three seed panels, the fitted
floor bound was active in all mixed-rate replicates and in none of the two
unit-rate conditions. The algorithm used only the inferred site profile.
It therefore follows the existing fixed rate in these mixed-rate alignments
and the learned CV rate in the others.

At K=6 the common CV rate gave ECS WAPE ratios of 0.824 (unit), 1.065
(mixed, upper 1.074) and 0.884 (biased). The floor recovered the current
mixed-rate ECS by construction, with an ECN WAPE ratio of 1.038 (upper
1.086). The mixed-rate K=5 ECN upper bound was 1.087. Earlier mixed-rate
K=6 ECN ratios for the floor were 1.083 on the six reused seeds and 0.956
on the first eight new seeds. High-arity uncertainty persists.

The joint N+S split gave K=6 ECS WAPE ratios of 0.607, 0.149 and 0.482
(unit, mixed, biased). Its all-pairs mixed-rate omega RMSE rose by 6.7%
using inferred observations and 12.1% using true ancestral counts on their
respective supported rows. With generating means as numerators, the same
method reduced the error in log omega by 72.1% in this condition. Accurately
estimating the EC ratio and accurately estimating raw omega from a short
realized alignment are distinct targets. Across all 66 alignments, none
of the K=5 or K=6 stem combinations had a true synonymous convergence, and
none had both observed posterior counts above tolerance. Direct high-arity
raw omega accuracy therefore remains undetermined.

## Design

Compare the current `empirical_all` estimator with the alternatives below on the bundled
71-tip PEPC topology, with 520 codons per replicate. Use unit rates, the four equally
weighted site rates 0, 0.2, 0.8 and 3, and directional bias toward a shared amino acid
on nine stems at 20% of unit-rate sites. Each simulated alignment has its own IQ-TREE
ECMK07_GY0K+F fit (+R4 under mixed rates). Each comparison within an alignment uses
the same tree, joint endpoint posterior masses, observed counts, eligible sites and
138 training branches. All methods include the evaluation branches in training.
The estimator has no access to the foreground labels or generating parameters.

* Reuse the six seeds per condition from the [S allocation experiment](PEPC_PARTIAL_POOLING.md).
* Use eight new seeds per condition (1801, 1931, 2063, 2203, 2339, 2473, 2609 and
  2729) to evaluate the initial comparisons.
* After examining that comparison, add one common-rate variant with a smoothing
  floor and test it on eight further seeds per condition (3001, 3137, 3271, 3407,
  3539, 3673, 3803 and 3931). Keep the earlier seed sets separate from this replication.

Evaluate all 8,377 eligible independent branch pairs per alignment, allowing sister
pairs. Also evaluate the same nine prespecified stems without hit selection, giving
36, 84, 126, 126 and 84 combinations for K=2 through 6. The stem pairs are a subset
of the all-pairs assessment. The null reference integrates the joint distribution
of substitutions to the same destination under the generating background Q on
the evaluated edges. The signal Q includes the simulated directional bias.

## Estimators

The current N rule uses an empirical parent-to-destination amino-acid route table,
a site rate estimated from N+S endpoint mass with leave-one-branch-out selection
of a uniform smoothing weight, and the branch's N+S mass per eligible site to
predict destination counts. It calibrates total N across training branches.
Current S uses the compatibility of the parent codon with each synonymous
destination, a fixed addition of 0.5 to each site's N+S mass, and observed
per-branch per-destination S counts. All alternatives use the same Wallenius
inclusion and K-way overlap calculation as the control.

| Method in the tables | Site rates | Destination weights and draw counts |
|---|---|---|
| current | Existing N and S rates | Existing N and S rules |
| shared_cv | Existing cross-validated N rate for both | Existing N and S rules |
| shared_fixed | Existing fixed-prior S rate for both | Existing N and S rules, refitting N routes with this rate |
| observed_both | Common CV rate | Parent compatibility and observed destination counts for each channel |
| own_totals | Common CV rate | Learn routes for both channels and predict destination composition, holding measured branch N and S totals separately |
| separate_scales | Common CV rate | Learn routes and use the measured branch N+S mass per eligible site to predict both channels, calibrating each whole-tree total separately |
| joint_totals | Common CV rate | Also constrain each branch's predicted N and S to sum to its measured N+S, preserving both whole-tree channel totals |
| shared_floor | Common CV rate with the existing 0.5-per-site prior as a lower bound on smoothing | Existing N and S rules, refitting N routes with the shared rate |

For learned routes, fit the same estimator with the same 0.5-count prior toward
the destination's pooled rate. The N table covers amino-acid changes, and the S
tables cover distinct codons within each synonymous family. A route's exposure
uses branch effort, parental state probability and site rate. Under separate
scales, multiply each branch's opportunity-weighted route profile by its measured
N+S mass per eligible site and calibrate the channel to its observed tree total.
Under joint totals, fit one N-versus-S log odds offset to the training branches
and split each measured branch total using those probabilities. The script
checks both branch and tree conservation.

For the smoothing floor, let x be the empirical site profile normalized to mean 1.
Both existing profiles take the form (1-w)x + w. The fixed 0.5 prior corresponds
to w0 = 0.5 / (mean site N+S mass + 0.5) on the complete alignments here.
The CV log loss is convex in w, so its minimum under w >= w0 is max(w_CV, w0).
This imposes the same minimum smoothing on both channels. No truth or explicit
rate-mixture label enters that choice.

## Accuracy measures

Count WAPE is the sum of absolute combination-wise errors divided by the sum
of the true expected counts for the same rows. Bias is the sum of estimates
divided by the corresponding sum of references. For omega, the target is
(muN_signal / muN_null) / (muS_signal / muS_null). The primary omega measure is
RMSE of log(1 + omega) for the common subset where both observed posterior counts
reach the production threshold of 1e-9. Fractional counts near this threshold
can give very noisy ratios. A secondary RMSE uses true ancestral endpoint counts
on rows with both true counts positive. A separate diagnostic substitutes
the generating signal means for the observed numerators to isolate EC errors.
It measures the error attributable to the estimated null counts.

Use a 5% increase in error as a practical margin specified before scoring.
For each condition and arity, resample seeds in pairs across methods (20,000
bootstrap samples), keeping every branch combination in an alignment together.
The 95th percentile of the error ratio gives a one-sided upper bound. Report
an error as within the margin when that bound is at most 1.05. The stored
intervals are 5th to 95th percentiles. These are approximate bounds from
eight seeds on one topology. Empty observed-count strata have no raw omega
RMSE. Assess all-pair results and high-arity results separately.

## Verification and reproduction

Each score run recomputes a standard all-pairs `search` and checks the reconstructed
control ECN and ECS against the CLI for every evaluated row at every arity
(relative tolerance 2e-10 plus 8e-18 for printed output). Raw control omega
agrees to relative tolerance 1e-8 on the common supported rows. Targeted and
all-pairs searches agree on the 36 shared pairs. Source-by-destination S totals
from a full-event joint-posterior pass reproduce the production destination
totals. The Q replay uses the same tips and true ancestral sequences and agrees with
the simulator's public pair reference. Direct ancestral-state enumeration checks
the high-arity oracle on a small tree.

CSUBST 1.17.5 working tree based on b782efd, Python 3.12.15, IQ-TREE 3.1.4,
macOS arm64. From the repository root, set PREVIOUS to the simulation panel
from the [previous experiment](PEPC_PARTIAL_POOLING.md). Use fresh directories
for fits and scores.

```bash
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
script=reports/omega_low_count_20261116/pepc_urn_symmetry.py
panel=$(mktemp -d)
extra=$(mktemp -d)
python "$script" simulate "$panel" --iqtree iqtree --jobs 8
python "$script" simulate "$extra" --replication --iqtree iqtree --jobs 8
for source in "$PREVIOUS" "$panel" "$extra"; do
  scores=$(mktemp -d)
  python "$script" score "$source" --work "$scores" --jobs 4
done
```

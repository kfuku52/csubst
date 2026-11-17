# Parent-state site weights for the PEPC urn

This replay uses nine complete PEPC alignments, with seeds 101, 202, and
303 in three conditions. It uses the same fitted codon generator, branch
lengths, and rate categories as the comparison of all ASRV modes. The
observed N and S counts and the observed per-branch, per-destination
substitution masses are fixed for the main comparisons. The study
includes sister pairs and excludes 139 pairs per alignment whose codon EC
is unavailable near the root. [Original ASRV comparison](URN_ACCURACY.md)

For each branch, site and destination, the urn uses the posterior mean of
the site rate times the indicator that the parental codon permits the
event. An S event requires a different codon for the same amino acid. An N
event requires a different amino acid. This expectation integrates over
the joint posterior of parent codon and rate class. The profile outside
the focal subtree omits all descendant tip likelihoods of that branch
while keeping the fitted model fixed. Each branch uses its own exclusion.

The replay uses the existing Wallenius inclusion-probability solver and
stochastic rounding of branch masses. No inferred total exceeds the
number of sites with positive weight for its destination. The observed
endpoint overlap and fitted codon EC reproduce the saved `search` results
on all eligible pairs. The outside calculation also agrees with explicit
tip masking for three selected terminal or internal branches in every
alignment and a separate small-tree control with an invariant rate
category.

## Main comparison

Each cell pools the four count totals before computing omegaC. The
two background columns pool 25,131 pair-replicates each. The last
column pools 108 pair-replicates with two directionally biased branches.
Values in parentheses give ECN and ECS relative to the generating-Q null.

| Method | Background, unit rates | Background, mixed rates | Two biased branches |
| --- | ---: | ---: | ---: |
| Generating expected omegaC | 1.000 | 1.000 | 3.296 |
| Codon model | 0.879 (1.063, 0.999) | 0.902 (0.991, 0.942) | 4.543 (1.096, 1.033) |
| Urn `each` | 0.136 (0.451, 0.066) | 0.193 (0.205, 0.042) | 0.711 (0.500, 0.074) |
| Urn `file_each` | 0.138 (0.453, 0.067) | 0.319 (0.294, 0.099) | 0.710 (0.501, 0.074) |
| Parent weights for S | 1.831 (0.451, 0.883) | 4.183 (0.205, 0.903) | 8.559 (0.500, 0.888) |
| Parent weights for N | 0.138 (0.445, 0.066) | 0.089 (0.444, 0.042) | 0.726 (0.490, 0.074) |
| Parent weights for N and S | 1.857 (0.445, 0.883) | 1.930 (0.444, 0.903) | 8.739 (0.490, 0.888) |
| Parent weights for N and S, excluding descendants | 1.850 (0.445, 0.880) | 1.929 (0.442, 0.898) | 8.733 (0.490, 0.887) |

With both channels constrained, the pairwise ECS RMSE drops from 0.692
to 0.206 at unit rate and from 0.887 to 0.279 with mixed rates. The
adjusted ECN still lies below half the Q reference. Excluding descendants
changes the pooled omegaC by at most 0.007 in these three comparisons.
The same S-induced increase in omegaC appears under both ways of
estimating parental states. The earlier oracle-parent result therefore
largely carries over to ECS when the parents are inferred.
[Oracle-parent controls](URN_CAUSES.md)

## Isolate the remaining ECN error

Starting from both full-data parent profiles, replace the N destination
totals with the fitted codon-model predictions, or replace only the N
site weights with fitted codon-model event probabilities. The latter
weights each parent and rate by its specific transition to the destination
amino acid and keeps the observed branch totals. The last row uses
model-based site weights in both channels and keeps observed totals.

| Additional comparison | Background, unit rates | Background, mixed rates | Two biased branches |
| --- | ---: | ---: | ---: |
| Replace N totals only | 1.823 (0.453, 0.883) | 1.843 (0.465, 0.903) | 9.177 (0.466, 0.888) |
| Model weights for N, parent weights for S | 0.812 (1.018, 0.883) | 0.928 (0.924, 0.903) | 3.590 (1.192, 0.888) |
| Model weights for N and S | 0.878 (1.018, 0.955) | 0.890 (0.924, 0.866) | 3.905 (1.192, 0.966) |

The strong ECN deficit remains with model totals alone. The specific
transition probabilities supply information about how likely different
allowed destinations are, beyond the parent compatibility indicator.
Using those probabilities in N brings the pooled omegaC of the biased
pairs to 3.590, compared with 4.543 using the usual codon model. Their
realized ratio from the simulated ancestors is 4.681. Across individual
pairs, the codon-model RMSE of `log(1 + omegaC)` against the generating
expected value is 0.849, 0.812 and 1.516 in the three columns. For the
hybrid with the N probability profile it is 1.046, 0.947 and 2.089.
Pooled agreement and accuracy on individual pairs favor different
methods in these simulations. [Per-seed and per-subset results](urn_parent_opportunity_summary.tsv)

The follow-up holds those N profiles fixed and varies the branch totals
and subset law on the original and independently refitted PEPC seeds.
[N draw totals and Fisher odds](URN_CONDITIONING.md)

## Reproduce

Run with editable CSUBST 1.17.5 (`c39484a`), Python 3.12.15 on macOS arm64.
Set `panel` to the original simulation directory, `original` to the
`runs` directory with the archived `codon_model` and `urn-each` searches,
and `additional` to the `runs` directory with `urn-file_each`. The
[ASRV comparison](URN_ACCURACY.md#reproduce) can regenerate both sets in
one directory, which may be supplied for both arguments.

```bash
work="$(mktemp -d)"
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/urn_parent_opportunity.py \
  "$panel" "$original" "$additional" "$work/summary.tsv" --jobs 5
```

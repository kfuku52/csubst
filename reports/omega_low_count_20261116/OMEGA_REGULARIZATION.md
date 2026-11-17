# Regularizing the omegaC point estimate toward one

The experimental estimate gives the prior count the fitted ECN-to-ECS
ratio. It converts OCS to
nonsynonymous count units using `ECN/ECS` and adds the same `A` to the
numerator and denominator. This avoids highly variable ratios when both
channels have little absolute support. The estimator needs positive ECN
and ECS. The comparison is available as a research script.

## Simulation comparison

The replay uses the published PGK (60 alignments) and PEPC (12 alignments)
simulation panels with single-edge joint ASR and separate branch products.
It uses the accepted recovered IQ-TREE fits for the 14 stalled PGK runs
and the archived fixed-input reruns for nine PEPC alignments. The remaining
49 alignments use their saved default fits. There are 199,584 eligible
rows with positive expectations and a numeric generating-Q ratio. The raw
ratio has a nonfinite value in 22 of those rows, and a positive prior of
0.03 provides finite estimates for all 22. The table compares estimators
on the shared set of 199,562 rows with a finite raw ratio.
[Original panel](../simulate_sensitivity_20261116/README.md),
[PGK recovery](../simulate_sensitivity_20261116/RECOVERY.md),
[PEPC fixed-input reruns](URN_ACCURACY.md)

Each entry is RMSE of `log(1 + omegaC)` against the generating expected
omegaC, which integrates over the generating process rather than using
the realized counts as its target. Enriched pairs have a generating
expected omegaC of at least 1.5. The pairs within each alignment share
sites and branches. All-pair results are dominated by near-null cases.

| Method | PGK, all 99,039 | PEPC, all 100,523 | PGK, enriched 672 | PEPC, enriched 108 |
| --- | ---: | ---: | ---: | ---: |
| Raw | 0.843 | 0.845 | 1.805 | 1.516 |
| Regularized, prior 0.01 | 0.318 | 0.412 | 0.945 | 0.979 |
| Regularized, prior 0.03 | 0.239 | 0.311 | 0.845 | 0.892 |
| Regularized, prior 0.10 | 0.159 | 0.201 | 0.759 | 0.810 |
| Existing symmetric pseudocount 0.1 on observed and expected counts | 0.444 | 0.492 | 0.681 | 0.645 |
| Existing symmetric pseudocount 0.3 on observed and expected counts | 0.261 | 0.294 | 0.628 | 0.639 |
| Constant one | 0.064 | 0.025 | 0.711 | 0.769 |

The exposure-scaled prior reduces extreme estimation error but pulls the
enriched cases toward one as well. With prior 0.03, the mean signed log
error is −0.738 on enriched PGK pairs and −0.769 on enriched PEPC pairs.
The existing symmetric correction does better on those enriched cases,
and biases near-null estimates upward. The constant-one control shows why
low all-pair RMSE alone cannot establish recovery of a directional signal.
[Per-seed losses and support subsets](omega_regularization_summary.tsv)

An exploratory urn replay also used the fitted codon-model predictive
feature as a site profile for each branch and destination. It normalized
that profile to the observed branch-by-destination total and took the
linear overlap across each pair. This version uses both N and S profiles.
Its pooled omegaC moves from 6.459 to 4.874 relative to the ordinary codon
model on the enriched pairs of biased seed 101 (generating value 3.337).
On all pairs of rate-mixture seed 202 the same substitution moves omegaC
from 0.834 to 0.762 (generating value 1). The improvements depend on the
case. [Earlier controls for the urn synonymous expectation](URN_CAUSES.md)

## Reproduction

Use the saved [simulation panel](../simulate_sensitivity_20261116/README.md),
the [14 recovered PGK searches](../simulate_sensitivity_20261116/RECOVERY.md),
and the `runs` directory from the [PEPC urn experiment](URN_ACCURACY.md).
The replay uses the saved CSUBST 1.17.5 pair tables, with Python 3.12.15 on
macOS arm64.

```bash
work="$(mktemp -d)"
python reports/omega_low_count_20261116/omega_regularization.py \
  "$panel" "$recovered_pgk" "$pepc_runs" "$work/summary.tsv"
```

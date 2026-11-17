# How much count support does a pairwise omegaC estimate have?

This pilot uses homogeneous null cases from the [PGK and PEPC sensitivity
panel](../simulate_sensitivity_20261116/README.md) and the accepted IQ-TREE
refits from the [recovery check](../simulate_sensitivity_20261116/RECOVERY.md).
The exact count calculation uses **28,311 branch-pair and condition
combinations** with known simulation Q and lengths. The search comparison
has **96,490 pair-replicate rows** from 34 alignments. It uses CSUBST 1.17.5
at `c9f5d89`, Python 3.12.15, and the saved IQ-TREE 3.1.4 fits on macOS arm64.

For a branch pair with IID sites, each site is an N convergence, an S
convergence, or neither. This follows directly from the endpoint event
definition. The Q-based expectations divided by the alignment length give
the two event probabilities. Conditional on the total N or S count, the
N count is binomial. Transforming a central 95% Clopper–Pearson interval
for that probability gives an interval for omegaC, using the ratio of the
Q-based S and N null expectations for the exposure adjustment. When both
realized counts are zero, the interval is [0, ∞]. The conditional
construction also appears in analyses of two Poisson rates.
[Fay 2010](https://journal.r-project.org/articles/RJ-2010-008/)

| Expected S count under Q | P(both counts zero) | P(both counts positive) | P(interval excludes 1 in the positive direction if N quadruples) |
| --- | ---: | ---: | ---: |
| ≤0.1 | 96.54% | 0.0049% | 0.18% |
| 0.1–1 | 69.34% | 0.36% | 1.29% |
| 1–5 | 16.81% | 4.37% | 3.20% |
| >5 | 0.15% | 20.89% | 13.82% |

These are averages of exact probabilities over branch pairs and conditions.
The fourfold case multiplies the per-site N event probability by 4 while
holding the S probability fixed. It is a count-level sensitivity check.
The expected S count alone gives little information about power. Even in
the >5 S bin, 660 of 661 pairs have expected N count at or below 1. The
interval contains the true null ratio of 1 with 99.12–99.96% probability
when averaged within each S bin.

With actual tip-only `search` output, in the ≤0.1 S bin, **34,942**
pair-replicates had zero true N and S counts. The search ratio was finite
in 34,926 of those rows. In 19,430, both fractional search counts were
positive. Direct substitution of those fractional counts into the
interval formula produces mathematically bounded endpoints even though
the simulated realization has zero events in either channel. A bounded
interval may still be extremely wide. The fractional-count calculation
here is a diagnostic comparison. The coverage guarantee for the
conditional binomial construction assumes integer counts.

The [joint-posterior follow-up](POSTERIOR.md) integrates rate categories
and ancestral states jointly within each site and compares count intervals
with the realized counts under uniform rates, heterogeneous rates, and
directional bias. Rows sharing branches or alignments are dependent.

The [Wallenius urn simulation](URN_ACCURACY.md) uses the PEPC alignments
to check expected counts against the generating Q and compare sampled
tail probabilities across the same scenarios.

## Reproduce

Use the saved panel directory and the recovered fit directory from the
linked experiments. The latter has a subdirectory named
`<scenario>-<seed>` containing `csubst_search` for each repaired PGK
fit. Run from the repository root.

```bash
python reports/omega_low_count_20261116/pilot.py \
  "$panel" "$recovered" reports/generated/omega_low_count
```

- [Exact probabilities by S support](oracle_support.tsv).
- [Exact probabilities by N and S support](oracle_joint_support.tsv).
- [True and reconstructed count comparisons by seed](pipeline_support.tsv).

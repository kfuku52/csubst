# Why PEPC urn expectations and tail probabilities fail

These controlled replacements reuse the six unbiased PEPC alignments from
the urn accuracy experiment. Each of the unit-rate and four-rate scenarios
has three seeds, 520 codon sites and 8,377 eligible branch pairs per seed.
The fitted IQ-TREE inputs, CSUBST endpoint posterior and branch-pair family
are identical to the earlier runs. Sister pairs are included. The 139
root-adjacent rows without EC or p-values are excluded from each comparison.
[Urn accuracy experiment](URN_ACCURACY.md)

## Synonymous expectation

For `any2spe`, the urn computes an overlap separately for each derived
codon and sums across codons. Its per-branch draw size is the inferred
number of endpoint changes to the derived codon across all sites. The
default `each` profile adds one unit of uniform prior mass at every site
before normalization. In these complete alignments this adds 520 units per
derived codon, including at sites where that codon cannot be reached by a
synonymous endpoint change from the parental codon. Across the 59
synonymous destination categories, this is 30,680 units of prior mass
versus 2,470–2,964 units of fitted synonymous change mass per alignment.
`--asrv file` weights sites by rates and also assigns the same profile to
all destinations.
[Urn implementation and concentration conventions](../../docs/URN_CALIBRATION.md)

The controlled allocations retain each branch's totals by derived codon
and distribute them linearly in proportion to the indicated profile. For
uniform weights this is the exact Wallenius overlap with stochastic
rounding. The replay checks the result against every saved `--asrv no`
ECS and checks the `each` expectation against the actual sampled overlap
mean. The replacement totals are either realized from the true ancestor
and tip sequences or computed from the generating transition matrix at the
true parental codon and rate at each site.

The amino-acid profile permits only codons synonymous with the parental
amino acid. The codon profile also excludes the parental codon itself.
Both profiles weight allowed codons by the true site rate. The full
transition profile uses the generating Q to weight allowed codons from the
true parental codon on that edge. Averaging those probabilities within
each true rate class preserves each branch's destination totals and their
dependence on rate while removing the site-specific parental composition.
The true-parent profiles are diagnostic controls that use knowledge of the
simulated ancestors.

| Allocation, with fitted branch totals unless specified | Unit rate, ECS/Q | Rate mixture, ECS/Q |
| --- | ---: | ---: |
| Production `each` | 6.56% | 4.17% |
| Mean of the actual 999 urn draws | 6.56% | 4.17% |
| `each` profile, linear | 6.58% | 4.19% |
| `each` profile, replace totals with true realized counts | 6.52% | 4.21% |
| `each` profile, replace totals with Q totals | 6.65% | 4.19% |
| True rates alone | 4.84% | 6.89% |
| Restrict derived codon to the parental amino-acid family | 66.16% | 83.18% |
| Also exclude the unchanged parental codon | 88.62% | 97.84% |
| Full transition from the parental codon | 96.26% | 93.70% |
| Full transition and Q totals | 97.23% | 95.62% |
| Full transition averaged within each rate class, Q totals | 4.94% | 6.83% |

The rate-class average keeps each branch's destination totals fixed and
reduces the overlap by roughly an order of magnitude. The shared site
constraint is therefore sufficient to explain most of the ECS deficit.
With fitted branch totals, the simple parental codon restriction also cuts
pairwise ECS RMSE from 0.692 to 0.205 and from 0.887 to 0.296. The full Q
profile offers a stronger control with the Q totals, with RMSE of 0.033
and 0.066. A ratio of pooled counts near one can still coexist with error
at individual pairs. The urn's sampled S totals (630.25 and 542.30) agree
closely with the production ECS (630.29 and 543.43), so the numerical
Wallenius approximation explains little of the deficit.
[Per-seed overlap and RMSE inputs](urn_causes_ecs.tsv)

The follow-up replaces true parental codons by fitted ancestral posteriors
and also tests N-specific site profiles.
[Parent-state urn weights](URN_PARENT_OPPORTUNITY.md)

## Tail probabilities

The rerun captures the exact fractional observed counts, EC and sampled
counts in memory. Baseline p- and q-values are compared to the original
search for every eligible row. Each controlled replacement reuses the 999
draws while changing the inputs shown, and applies the same inclusive
upper-tail rank and BH calculation. The generating-Q draws use the exact
rate-specific pair probabilities. At each site the pair can have a matching
N endpoint, a matching S endpoint, or neither. The counts in each rate
class follow the corresponding multinomial. Summing the means over rate
classes reproduces `simulate`'s generating-Q counts for every pair. This
comparison simulates the correct marginal null for each pair independently.
It does not reproduce cross-pair dependency in a new full alignment. The
single-channel replacements mix arrays from the two nulls to isolate their
marginal effects, without preserving their original joint distributions.

| Change to the original urn test | Unit rate, q ≤ .05 | Rate mixture, q ≤ .05 |
| --- | ---: | ---: |
| Original fractional counts and urn draws | 50.33% | 76.06% |
| Use generating-Q ECN and ECS in the statistic, same draws | 50.33% | 76.06% |
| Use true observed S, same draws | 50.38% | 76.06% |
| Use true observed N, same draws | 0% | 0% |
| Replace sampled S only with generating-Q counts | 50.47% | 76.19% |
| Replace sampled N only with generating-Q counts | 45.76% | 63.88% |
| Replace both sampled counts with generating-Q counts | 46.27% | 64.22% |
| Use both true observed counts and generating-Q draws | 0% | 0% |

The urn draws have zero N in 99.65% of the unit-rate and 99.67% of the
rate-mixture trials. Once the observed statistic is positive, any draw
with zero N is below it, including draws that also have zero S. For a
positive observed omega and M valid draws, the upper-tail p-value is
bounded by `(1 + number of positive N draws) / (M + 1)`. Among the adjusted
urn calls, 6,112 and 9,576 pairs never draw positive N in any of the 999
trials and receive the minimum p-value of .001. Every adjusted omega call
is also a call under the upper tail of N alone in this replay.

With the same urn counts in the test statistic, changing only EC to the
generating-Q values leaves rejection counts unchanged. For positive ECN
and ECS, the factor `ECS/ECN` multiplies the observed and sampled ratios
equally, away from numerical zero decisions. The urn's lower rate of
positive N draws accounts for most of the extra rejections relative to the
generating-Q draws with fractional observations. The mismatch between
fractional and integer N remains even under the generating-Q draws. The
two integer-observation runs with zero adjusted calls also have raw
`p ≤ .05` fractions of 0.61% and 1.26% with the urn and 0.62% and 1.02%
with the generating-Q draws. These are descriptive rejection rates across
six sets of correlated branch pairs. [Per-seed tail counts](urn_causes_tails.tsv)

The calibration follow-up simulates whole tip alignments under Q and
reconstructs their ancestors before forming the null statistic.
[Fixed-model ASR calibration](ASR_BOOTSTRAP.md)

## Reproduction

Use CSUBST 1.17.5 (`c39484a`) in the editable Python 3.12.15 environment
on macOS arm64, with one CSUBST and BLAS thread per search. Set `panel`
to the panel directory containing `PEPC`, and `previous_runs` to the `runs`
directory produced by the earlier urn accuracy experiment. The replay
checks the archived `urn-each` and `urn-no` fits and tables from that run.

```bash
work="$(mktemp -d)"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/urn_causes.py \
  "$panel" "$previous_runs" "$work/replay" --jobs 3
```

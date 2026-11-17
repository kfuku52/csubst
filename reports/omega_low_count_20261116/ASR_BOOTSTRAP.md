# Fixed-model calibration of omegaC after ancestral-state reconstruction

Generate new tip alignments under background Q, and reconstruct each with
the same joint parent–child endpoint calculation used by `search`. The
reference statistic is the posterior fractional `omegaCany2spe`, with the
default product across focal edges. It uses the usual codon-model EC,
computed anew from each replicate's posterior. The generator, branch
lengths and per-site rates are fixed at their known simulation values for
both generation and reconstruction. The model used to score alternatives
is also the background model. The known parameters isolate the effect of
applying ASR to the reference statistic.
[Origin of the comparison](URN_CAUSES.md)

The PEPC tree has 71 tips and the alignments have 520 codons. The two rate
conditions use either unit rates or a fixed cycle of 0, 0.2, 0.8 and 3.
Each condition has 399 independently simulated background alignments for
calibration (seeds 10000–10398), and 96 null and 96 alternative alignments
held out from calibration (seeds 20000–20095 and 30000–30095). The alternative
uses the same 90% nonsynonymous target-rate bias used by the original PEPC
panel on the first 104 codons in the designated foreground branches.
The target amino acid varies randomly by site and replicate. The actual
targeted rate fraction can vary across amino acids. Of the 8,377 eligible
branch pairs, 36 contain two foreground edges. Both families are defined
in advance and include sister pairs. Root-adjacent branches are excluded.
[Original PEPC simulation setup](../simulate_sensitivity_20261116/README.md)

An upper-tail probability is `(1 + number of calibration scores at least
as high as observed) / 400`. The same 399 calibrations define each family's
reference distribution of its maximum. The screened version reapplies the
default higher-order search cutoffs (`OCN ≥ 2` and `omegaC ≥ 5`) within each
replicate before taking the maximum. Empty families receive negative
infinity. Pairwise BH is reported for context. The smallest attainable
pairwise p-value, 0.0025, cannot pass BH at 5% for a single hit in the family
of 36 or 8,377. The maximum test calibrates at the alignment level and
keeps the dependence between branch pairs.

The `urn` comparison applies the production Wallenius site-subset kernels
with `each` weights, stochastic rounding and 399 draws to the same
posterior features, on the first 12 null and 12 alternative seeds in each
rate condition. Two controls rank posterior `omegaCany2spe` against the
fixed generating-Q expectations, or rank only the posterior `OCN`.
They use the same calibration alignments.

## Results

For the 12 held-out background alignments evaluated by both methods in
each scenario, the raw branch-pair rejection fraction at 0.05 falls from
54.13% to 4.77% at unit rate and from 80.96% to 4.90% with mixed rates.
With all 96 background alignments, the corresponding ASR calibration
fractions are 4.82% and 4.95%. All prior urn comparisons are paired on the
same posterior edge features and test seeds.

| Family and test at p ≤ .05 | Unit-rate null | Unit-rate alternative | Mixed-rate null | Mixed-rate alternative |
| --- | ---: | ---: | ---: | ---: |
| Maximum over 36 prespecified foreground pairs | 2/96 | 12/96 | 5/96 | 21/96 |
| Maximum over all 8,377 pairs | 4/96 | 2/96 | 4/96 | 5/96 |
| Screened maximum over 36 foreground pairs | 0/96 | 0/96 | 1/96 | 2/96 |
| Screened maximum over all 8,377 pairs | 3/96 | 9/96 | 6/96 | 10/96 |

The full unscreened family gives an observed null rejection rate of
4.17% in each condition (exact 95% binomial interval 1.15–10.33%,
conditional on the sampled reference). Alternative rejection within the 36
prespecified pairs is 12.5% and 21.9% before applying the higher-order gates.
In the alternative, only 131 and 246 of 3,456 foreground pair-replicates
have any realized matching N endpoint, respectively. At the individual
pair level the raw foreground rejection rate rises from 5.03% to 7.75%
and from 5.18% to 9.26%. Within those pairs, ranking OCN alone gives 7.52%
and 8.91% alternative rejection. Fixing EC at the generating-Q means
gives almost the same pair-level result as recomputing it from ASR.

The fixed-model result isolates the benefit of ranking the same
fractional statistic on both sides of the test. Detection remains weak
for this PEPC alternative when the pairs have to be found across the whole
tree. Calibration after estimating Q, branch lengths and latent site
rates from each alignment remains to be checked. The confidence intervals
and power counts treat alignments as replicates.
[Per-alignment and per-family results](asr_bootstrap_summary.tsv)

## Reproduction

Run on CSUBST 1.17.5 (`c39484a`), Python 3.12.15 on macOS arm64, with one
BLAS thread per process. Set `panel` to the panel directory containing
`PEPC`, and `previous_runs` to the `runs` directory from the archived urn
accuracy experiment. The script checks observed and expected counts
against the saved unit-rate and rate-mixture searches on their archived
IQ-TREE fits before processing new replicates.

```bash
work="$(mktemp -d)"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/asr_bootstrap.py \
  "$panel" "$previous_runs" "$work/bootstrap" --jobs 8
```

The archived summary concatenates `baseline.tsv` and `rate_mixture.tsv`.
In pandas, read with `keep_default_na=False, na_values=['NA']` to retain
the literal condition label `null`.

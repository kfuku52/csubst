# Separating site rates and synonymous destination totals in the high-arity PEPC urn

This diagnostic reuses the [high-arity panel](PEPC_HIGH_ARITY_COMPARISON.md):
nine alignments, 520 codons, nine fixed stems, three seeds per condition,
ECMK07_GY0K+F ASR (+R4 for mixed rates). At K=6 there are 84 subsets per
seed. Parent posteriors, masks and each evaluation branch's total S mass
remain fixed. No variant uses fitted Q to predict S rates or totals;
all variants use the same fitted ancestral states.

**Allocation**: The existing urn draws each branch's observed number of
events ending in each synonymous destination (fractional endpoint mass).
The complete-pooling diagnostic uses one destination coefficient for
all eligible branches. Estimate it as the pooled count for that destination
divided by pooled opportunity: sum the compatible parental codon mass times
the site rate, with each training branch weighted by S mass per eligible
site. On each evaluation branch distribute its original S total among
destinations in proportion to opportunity times coefficient. This borrows
information about destination frequencies; it does not change which
parental codons can give a synonymous event.

**Site profile**: Compare the existing S rate (eligible-branch N+S site
mass with fixed 0.5 prior), the existing N cross-validated site profile
reused for S, and the actual rate of each simulated site (diagnostic
only). Changing site rate also changes the opportunities in the pooled
allocation. One global fit uses all 138 eligible branches. The original
combination reproduces the previous CLI ECS for all K=2..6
(rtol 1e-10 plus 6e-18 for output rounding).

Each K=6 cell below gives summed predicted ECS / summed true null S
(target 1), followed by sum of absolute combination errors / summed true
null S (target 0). Seed and combination rows are pooled before dividing.

| Site rate | Destination totals | Unit | Mixed | Biased |
|---|---|---:|---:|---:|
| Current | Observed | 1.480 (1.942) | 0.261 (1.073) | 1.308 (1.834) |
| Current | Pooled | 1.734 (0.949) | 1.408 (1.300) | 1.709 (0.818) |
| Cross-validated | Observed | 0.772 (1.333) | 0.286 (1.088) | 0.671 (1.275) |
| Cross-validated | Pooled | 0.790 (0.378) | 1.539 (1.386) | 0.976 (0.429) |
| Known | Observed | 0.995 (1.546) | 0.263 (1.030) | 0.650 (1.263) |
| Known | Pooled | 0.770 (0.369) | 1.323 (0.945) | 0.988 (0.439) |

In mixed rates, replacing the site rate with its true value hardly
changes the pooled K=6 deficit when destination totals stay observed.
Reallocating the same S mass instead raises the ratio above one and
increases combination-level absolute error in this condition. This
identifies an influential component, not a superior production estimator.
With the current site rate, K=3 mixed-rate ECS changes from 0.839 to
0.989 of the reference and the absolute error drops from 0.695 to 0.351.
The effects depend on arity and vary widely across seeds.

A next experiment is partial pooling: shrink noisy observed destination
frequencies toward opportunity-based predictions while preserving branch S
mass. Select the strength using out-of-sample prediction (train the pooled
profile without the validation observations), then evaluate on new simulation
seeds, reporting signed EC bias and combination-level errors at K=2..6.
Accurate ECS alone does not make raw omega estimable when there are no
high-order observed substitutions. This pilot changed no search defaults.
[All per-seed counts and losses](pepc_high_arity_s_totals.tsv)

## Reproduction

CSUBST 1.17.5 working tree based on b782efd; Python 3.12.15, macOS arm64.
PANEL is the existing ECMK07 panel with IQ-TREE fits and HIGH_K is the
completed simulation work directory from the linked high-arity comparison.

```bash
work=$(mktemp -d)
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/pepc_high_arity_s_totals.py "$PANEL" "$HIGH_K" "$work"
```

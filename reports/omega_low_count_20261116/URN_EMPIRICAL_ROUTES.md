# Empirical parent-to-destination urn weights on PEPC

Each branch keeps its observed posterior substitution mass for each
destination amino acid or synonymous codon. The urn allocates those
totals to sites with Wallenius inclusion probabilities and stochastic
rounding. The comparison uses 18 complete PEPC alignments from the two
existing sets of fitted models. The archived panel uses seeds 101, 202,
and 303 and the second panel uses seeds 411, 522, and 633. Each seed has
unit-rate background, mixed-rate background, and directionally biased
alignments. There are 8,377 eligible pairs per alignment. [Panel and fit preparation](URN_CONDITIONING.md#reproduction)

## Empirical weights

For each ordered amino-acid route `a → d`, `C[a,d]` is the total N
endpoint posterior mass on the training branches. The exposure `E[a]`
sums the parental amino-acid probability across those branches and sites,
weighted by branch effort and relative site rate. In the empirical
version the branch effort is the mean N plus S endpoint mass across sites.
The site rate is proportional to the total N plus S endpoint mass across
training branches, plus 0.5 per site, and has mean one.

The unpooled rate is `C[a,d] / E[a]`. To share information for rare
routes, the pooled variant uses `(C[a,d] + k) / (E[a] + k / mu[d])`,
where `k = 0.5` and `mu[d]` is the pooled rate toward `d` from all other
parental amino acids. The diagonal is zero. The N site weight is the
average empirical rate toward the destination over the parental-state
posterior, multiplied by the site rate. S uses parental synonymous
compatibility times the same site rate. The compatibility control uses
the same inputs and only distinguishes whether the parent amino acid can
undergo the N endpoint event. A separate control uses fitted branch
lengths and the fitted rate-category posterior for the exposures and
site rates with the learned N route table. That control and the Q-profile
comparator use the previous fitted-rate S urn.

The shared background fit uses 129 nonforeground branches. In the
cross-fit, four fixed folds follow branch ID modulo four. Training for a
pair excludes both of its folds and all foreground branches, for both the
route table and the empirical site and branch weights. The underlying
ASR still conditions on the whole alignment with the fitted codon model.
The empirical weights use those posteriors and counts to form EC,
without using predicted probabilities of particular Q transitions. The
Q-profile row uses such predicted probabilities as a comparator.

## Results

Each cell shows pooled omegaC with per-pair RMSE of `log(1 + omegaC)` in
parentheses. The two background columns each pool 25,131 pair-replicates
per panel. The last column pools the 108 pairs with both branches biased.
Generating expected omegaC is 1 in both background conditions and is
3.296 and 3.283 in the biased subset of the two panels, respectively.
RMSE uses rows with a finite ratio for all compared methods.

| Archived panel | Unit rate | Mixed rates | Both branches biased |
| --- | ---: | ---: | ---: |
| Codon model | 0.879 (0.849) | 0.902 (0.812) | 4.543 (1.516) |
| Q-based N profile, observed totals | 0.812 (1.046) | 0.928 (0.947) | 3.590 (2.089) |
| Cross-fit compatibility only | 1.634 (1.135) | 1.803 (1.045) | 7.731 (2.209) |
| Empirical routes, shared background | 0.891 (1.057) | 1.030 (0.965) | 3.892 (2.112) |
| Empirical routes, cross-fit | 0.934 (1.063) | 1.089 (0.974) | 4.118 (2.122) |
| Empirical routes, unpooled shared background | 0.709 (1.032) | 0.838 (0.939) | 3.082 (2.083) |

| Second panel | Unit rate | Mixed rates | Both branches biased |
| --- | ---: | ---: | ---: |
| Codon model | 0.907 (0.880) | 1.119 (0.887) | 7.130 (2.665) |
| Q-based N profile, observed totals | 0.812 (1.063) | 1.126 (1.052) | 7.135 (2.551) |
| Cross-fit compatibility only | 1.712 (1.162) | 2.232 (1.169) | 15.035 (2.709) |
| Empirical routes, shared background | 0.880 (1.073) | 1.174 (1.061) | 8.061 (2.585) |
| Empirical routes, cross-fit | 0.927 (1.080) | 1.264 (1.077) | 8.387 (2.589) |
| Empirical routes, unpooled shared background | 0.704 (1.047) | 0.955 (1.028) | 5.965 (2.505) |

The empirical cross-fit raises ECN from 45–62% of the generating-Q count
with compatibility alone to 75–117%. ECS stays the same within that
comparison. The pairwise omegaC error falls in all six cells relative to
compatibility alone. The codon model has lower pairwise omegaC error in
five of the six cells. Pooled ratios and pairwise loss respond differently
to the regularization. For example, removing pooling under the shared
background gives the largest error in ECN among the empirical route
variants on the second panel, while its pooled omegaC and pairwise ratio
loss can come closer to the generating target. The realized omegaC on
the two biased branches is 4.681 and 7.000 in the two panels. Site-rate
and route-weight variants are available in the full summary.
[Per-seed and per-subset data](urn_empirical_routes_summary.tsv)

The same 18 alignments were refitted with additional codon models to
compare the observed and expected counts independently.
[Codon-model comparison](CODON_MODEL_COMPARISON.md)

## Reproduction

Run the editable CSUBST 1.17.5 checkout based on `c39484a` with Python
3.12.15. This run used macOS arm64 and one BLAS thread per worker. Use the
original simulation panel, the original fixed-fit codon runs and the
second PEPC panel prepared in the [N conditioning comparison](URN_CONDITIONING.md#reproduction).

```bash
work="$(mktemp -d)"
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/urn_empirical_routes.py \
  "$panel" "$work/archived.tsv" --archived-fits "$runs"
python reports/omega_low_count_20261116/urn_empirical_routes.py \
  "$second_panel" "$work/fresh.tsv"
```

The archived summary concatenates the two outputs. For every alignment
the script checks the archived ASR and model counts and verifies that
the empirical route counts sum to the observed destination totals on
background branches. The Q-profile control also reproduces all 126
matching seed-and-subset rows of expected counts and EC losses from the
N conditioning comparison.

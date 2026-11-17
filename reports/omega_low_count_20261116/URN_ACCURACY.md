# Wallenius urn estimates and sampled tails on PEPC simulations

Compare fixed-fit codon predictions with Wallenius urn expectations on
PEPC. Use the complete sequences from the background-Q unit-rate,
background-Q rate-mixture, and directional-bias conditions, with seeds
101, 202, and 303 in each condition. The urn and codon runs share the
fitted codon Q, lengths, rate mixture, and single-edge joint ASR.
Both use products across branches for the observed counts. Sister pairs
are retained. Pairs with unavailable codon-model predictions on the
root-adjacent edges are excluded consistently from all summaries.
[Original simulation panel](../simulate_sensitivity_20261116/README.md)

The urn profiles are uniform (`--asrv no`), pooled N and S substitution
mass (`pool`), separate N and S substitution mass (`sn`),
destination-specific empirical weights (`each`), fitted site rates
(`file`), and fitted site rates multiplied by destination-specific
empirical mass (`file_each`). The rate-mixture experiment also supplies
the true simulated site-rate file to `--asrv file`. This changes the urn
weights while retaining the fitted endpoint posterior and branch totals.
The `each` runs use either the default prior mass of one per eligible
site or `--asrv_concentration 2` total prior mass. Both calculate empirical
tail probabilities from 999 Wallenius weighted site-subset draws, using
stochastic rounding and `--omega_pvalue_null_model hypergeom`. In `auto`,
the point expectation uses exact special cases and the mean-equation
approximation when needed.
[Urn definitions](../../docs/URN_CALIBRATION.md)

## Expected counts and omegaC

Each row pools 25,131 pair-replicates. ECN/Q and ECS/Q divide the pooled
expectations by the corresponding generating-Q null counts. Pooled
omegaC divides the observed N and S totals by their respective expected
totals before taking the ratio.

| Generating scenario | Codon ECN/Q, ECS/Q | Default urn ECN/Q, ECS/Q | Codon omegaC | Default urn omegaC | Generating expected omegaC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Background Q, unit rate | 1.063, 0.999 | 0.451, 0.066 | 0.879 | 0.136 | 1.000 |
| Directional bias, unit rate | 1.079, 1.031 | 0.451, 0.070 | 1.032 | 0.166 | 1.008 |
| Background Q, site-rate mixture | 0.991, 0.942 | 0.205, 0.042 | 0.902 | 0.193 | 1.000 |

Across the three scenarios, the default urn's branch-pair RMSE against
the generating-Q null was 2.28–3.22 times the codon-model RMSE for ECN
and 4.38–4.63 times for ECS.

Uniform site weights use exact uniform overlap and give ECS/Q of 0.048,
0.051, and 0.028. In the mixture, using fitted rates raises this to 0.065
and using the true rates raises it to 0.069. This rules out the
mean-equation approximation or inaccurate site rates alone as an
explanation for the synonymous deficit in these runs.

Reducing smoothing to total concentration 2 changes ECS/Q to 1.113,
1.148, and 0.978. ECN/Q rises to 4.915, 5.109, and 2.640, respectively.
On the 108 directionally biased foreground pair-replicates, generating
expected omegaC is 3.296 and the realized ratio with the generating null
is 4.681. Codon-model omegaC is 4.543, compared with 0.711 for the default
urn and 0.956 with concentration 2. Ancestral-state inference is held
fixed in these contrasts.

### All ASRV modes

Each cell shows pooled omegaC followed by ECN/Q and ECS/Q. The two
background columns pool 25,131 pair-replicates each. The final column
pools the same 108 directionally biased foreground pair-replicates.
Generating expected omegaC is 1, 1, and 3.296, respectively.

| Method | Background, unit rate | Background, rate mixture | Two biased foreground branches |
| --- | ---: | ---: | ---: |
| Codon model | 0.879 (1.063, 0.999) | 0.902 (0.991, 0.942) | 4.543 (1.096, 1.033) |
| `no` | 0.109 (0.415, 0.048) | 0.146 (0.185, 0.028) | 0.566 (0.459, 0.054) |
| `pool` | 0.109 (0.525, 0.061) | 0.146 (0.491, 0.075) | 0.566 (0.572, 0.067) |
| `sn` | 0.110 (0.527, 0.062) | 0.190 (0.320, 0.064) | 0.570 (0.576, 0.068) |
| `each` | 0.136 (0.451, 0.066) | 0.193 (0.205, 0.042) | 0.711 (0.500, 0.074) |
| `file` | 0.109 (0.417, 0.049) | 0.145 (0.421, 0.064) | 0.566 (0.460, 0.054) |
| `file_each` | 0.138 (0.453, 0.067) | 0.319 (0.294, 0.099) | 0.710 (0.501, 0.074) |
| `file` with generating site rates | — | 0.146 (0.449, 0.069) | — |
| `each` with total concentration 2 | 0.212 (4.915, 1.113) | 0.351 (2.641, 0.978) | 0.956 (5.772, 1.145) |

`pool` increases ECN and ECS by similar factors relative to `no`, giving
nearly the same ratio. `file_each` gives the largest pooled omegaC among
the six standard ASRV modes under the rate mixture. Its expected S count
is 9.9% of the generating-Q reference and its expected N count is 29.4%.
[Per-seed and subset summaries](urn_accuracy_summary.tsv)

## Empirical tail calls

| Generating scenario and scope | Pairs | Default urn `p ≤ .05` | Default urn `q ≤ .05` | Concentration 2 `q ≤ .05` |
| --- | ---: | ---: | ---: | ---: |
| Background Q, unit rate, all pairs | 25,131 | 13,037 | 12,649 (50.3%) | 5,992 (23.8%) |
| Background Q, site-rate mixture, all pairs | 25,131 | 19,226 | 19,114 (76.1%) | 10,149 (40.4%) |
| Directional bias, generating expected omegaC ≥ 1.5 | 108 | 53 | 52 (48.1%) | 18 (16.7%) |

Of the default-urn adjusted calls in the two background panels, 12,648 of
12,649 in the unit-rate case and 19,097 of 19,114 in the mixture have
observed N mass below one. The corresponding counts with zero realized N
are 12,530 and 18,812. Both background simulations have generating
expected omegaC 1. Applying the default cutoff for higher-order search
(`OCNany2spe ≥ 2` and `omegaCany2spe ≥ 5`) to the pair tables leaves zero
pairs across all nine alignments and tested urn configurations, including
the biased pairs.

The fractional posterior observed counts are compared with integer site
draws. In the raw statistic, zero N draw mass produces omegaC zero, so
even a small positive posterior estimate can lie high in the discrete
null tail. All 999 draws are defined for the eligible pairs in these
summaries. With fixed positive expectations the same EC scale multiplies
the observed and randomized ratios, so the error in the expected totals
alone does not explain the sampled-tail behavior.
[Per-seed and subset summaries](urn_accuracy_summary.tsv)

The generating-Q expectation and empirical urn expectation are different
targets. The urn conditions on inferred substitution totals and fitted
site weights. Comparison to the generating alignment assesses this full
procedure. Pair-replicates in the same alignment share branches and sites,
so rejection fractions describe these simulations.
[Controlled replacements of EC and sampled tails](URN_CAUSES.md)

## Reproduce

Use the original PEPC simulations and the prepared complete-data fits
from the missingness experiment. Each method, including the codon-model
reference, was rerun from the same saved fit files. The generator hash,
branch lengths and rate categories agree exactly across methods. Observed
counts match within the saved-table numerical tolerance. The analysis uses
CSUBST 1.17.5 (`b3ca4ea` for the original modes, `c39484a` for the added
`pool`, `sn`, and `file_each` runs) in the editable Python 3.12.15
environment on macOS arm64 with one CSUBST and BLAS thread per search.
[Fit preparation](MISSINGNESS.md#reproduce)

```bash
work="$(mktemp -d)"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/urn_accuracy.py \
    "$panel" "$cases" "$work/runs" "$work/summary.tsv" --jobs 3 --draws 999 \
    --modes no pool sn each file file_each oracle_file each_tau2
```

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

The three urn profiles are uniform (`--asrv no`), fitted site rates
(`--asrv file`), and destination-specific empirical weights with default
regularization (`--asrv each`). The rate-mixture experiment also supplies
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

## Reproduce

Use the original PEPC simulations and the prepared complete-data fits
from the missingness experiment. Each method, including the codon-model
reference, was rerun from the same saved fit files. The generator hash,
branch lengths and rate categories agree exactly across methods. Observed
counts match within the saved-table numerical tolerance. The analysis uses
CSUBST 1.17.5 (`b3ca4ea`) in the editable Python 3.12.15 environment on
macOS arm64 with one CSUBST and BLAS thread per search.
[Fit preparation](MISSINGNESS.md#reproduce)

```bash
work="$(mktemp -d)"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/urn_accuracy.py \
    "$panel" "$cases" "$work/runs" "$work/summary.tsv" --jobs 3 --draws 999 \
    --modes no file oracle_file each each_tau2
```

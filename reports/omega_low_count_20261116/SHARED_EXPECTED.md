# Shared ancestry and rate in predictive ECN/ECSany2spe

The joint observed-count option shares the ancestry and rate of the two
branches. This replay tests the corresponding change to the predictive
counts in the denominator. It uses 75,393 eligible pair-replicates from
nine complete PEPC alignments and 414 sister pair-replicates from six
masked alignments. Each simulation condition uses seeds 101, 202, and
303. Each contrast keeps the fitted Q, topology, lengths, rate mixture,
and observed counts fixed. [Simulation panel](../simulate_sensitivity_20261116/README.md),
[Masking design](MISSINGNESS.md)

## Prediction

For each focal edge, the predictive feature at parent codon `a` and rate
category `c` is the probability of a change to a given amino acid (N) or
codon (S) under the raw fitted transition for that edge. For pairs with
different parents, we move these features from the parents to the common
ancestor with tip-conditioned transitions along the connecting paths. The
two focal endpoints are predicted from the raw kernels. The tips affect
the fitted model and the posterior over the parent states and site rates.

We compare the product of separately averaged edge features, the product
when only the rate is shared, and the product averaged over both a shared
ancestral state and a shared rate. The fitted conditional predictions and
the generating-Q reference differ in what they condition on. The reference
averages over ancestral states before the tips are observed and uses the
generating site rates. For the biased simulations, the simulated reference
compares the generating model with a null that replaces the two focal-edge
kernels by the background kernel. Upstream evolution is retained.

The masked intervals lie entirely outside the biased partition. We remove
exactly the background-Q contribution of each excluded site from both
generating references. The complete-biased totals can include site-specific
contributions even with uniform rates. Summing counts before taking ratios
gives pooled comparisons. Pairs from the same alignment share data.

## All eligible pairs on complete alignments

EC/Q divides the pooled fitted predictive counts by the corresponding
pooled generating null expectations. The omegaC columns use the same
joint observed N and S numerators for all three expectation methods.

| Generating condition | ECN/Q, separate → shared rate → joint | ECS/Q, separate → shared rate → joint | omegaC, separate → shared rate → joint | Generating expected omegaC |
| --- | ---: | ---: | ---: | ---: |
| Background Q, unit rate | 1.063 → 1.083 → 1.083 | 0.999 → 1.014 → 1.014 | 0.873 → 0.870 → 0.870 | 1.000 |
| Directional bias, unit rate | 1.079 → 1.086 → 1.086 | 1.031 → 1.035 → 1.036 | 1.021 → 1.018 → 1.019 | 1.008 |
| Background Q, site-rate mixture | 0.991 → 1.031 → 1.032 | 0.942 → 0.972 → 0.980 | 0.868 → 0.861 → 0.867 | 1.000 |

Sharing the rate explains almost all of the pooled ECN change. In the
mixture simulations it recovers some of the missing ECS. ECN moves from
slightly below the generating reference to slightly above it. These
effects nearly cancel in the pooled omegaC ratio. For comparison, using
the generating-Q denominator with the same observed counts gives 0.929,
1.069, and 0.913 for the three conditions. Using the realized endpoint
counts with the generating-Q denominator gives 0.943, 1.058, and 0.894.

On complete biased foreground pairs, the pooled omegaC is 4.511 with
separate expectations and 4.502 with joint expectations. The generating
expected ratio is 3.296 and the realized ratio is 4.681 for these seeds.

## Sister pairs

In the site-rate mixture condition, sharing the rate and parent brings
pooled ECN/Q from **0.942 to 0.989** and ECS/Q from **0.855 to 0.965**.
The sister-pair ECS RMSE falls from 0.391 to 0.332 counts and improves
in all three seeds. Pooling with joint observed counts gives omegaC 0.715
with separate expectations and 0.769 with joint expectations. The
realized ratio under the generating denominator is 0.751, with a
generating expected ratio of 1.000. The 207 pair-replicates in this
subset include sister pairs that the default search excludes.

The masked unit-rate controls do not show that improvement. For biased
sisters, joint expectations move ECN/Q from 1.089 to 1.100 and ECS/Q
from 1.046 to 1.073. The pooled omegaC increases from 1.144 to 1.160
with the same observed numerators. Their realized N count is zero and
their generating expected omegaC is 0.997. Changing the prediction
leaves the parent-state uncertainty in the N numerator intact.
[Per-seed and subset summaries](shared_expected_summary.tsv), [N numerator diagnosis](SISTER_N.md)

## Verification and reproduction

The three predictive formulas agree with exhaustive finite-state
enumeration on a small tree with sister and non-sister pairs, an invariant
rate category, and missing and ambiguous tips. The separate calculation
matches saved search ECN and ECS within 5.1e-13 across all tested inputs.
For homogeneous generating Q, replacing the tips with likelihood ones and
using the known rates reproduces the generating-Q reference within 3.7e-14
over 51,096 pair-replicates in the baseline and site-rate mixture runs.

Run from the repository root with the editable CSUBST 1.17.5 environment.
The replay uses `fa7cd07`, Python 3.12.15, and one BLAS thread per worker.
Set `panel` and `cases` to the original simulations and the prepared
PEPC masking experiment. [Fixture preparation](MISSINGNESS.md#reproduce)

```bash
work="$(mktemp -d)"
scripts=reports/omega_low_count_20261116
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python "$scripts/verify_shared_expected.py"
python "$scripts/shared_expected.py" "$panel" "$cases" "$work/complete.tsv"
python "$scripts/shared_expected.py" "$panel" "$cases" "$work/masked.tsv" \
  --masked-sisters --scenarios baseline biased
```

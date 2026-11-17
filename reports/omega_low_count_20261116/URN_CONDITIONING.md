# N draw sizes and Fisher odds in the PEPC urn

In the earlier PEPC comparison, replacing N site weights by the fitted
codon-model event probabilities corrected much of the pooled ECN deficit.
The branches still supplied draw totals from observed events. Individual
pair estimates retained substantial error. We vary the draw totals and
the sampling law while keeping that N prediction and the S comparison
fixed. [Parent-state urn comparison](URN_PARENT_OPPORTUNITY.md)

## Comparison

The two panels use the same 71-tip PEPC topology and three conditions at
520 codons. One panel reuses seeds 101, 202, and 303. The validation panel
generates new alignments with seeds 411, 522, and 633 and refits
`ECMK07_GY0K+F+R4` with IQ-TREE 3.1.4. Each panel includes a background
unit-rate condition, a background four-rate condition, and the original
directional-bias condition. The search includes sister pairs. The
two background columns each pool 25,131 pair-replicates per panel.
The two-foreground subset of the biased condition pools 108
pair-replicates per panel. [Simulation design](../simulate_sensitivity_20261116/README.md)

The S urn uses the previous full-data parent compatibility profile and
observed branch-by-destination masses in every urn comparison. For each
branch and destination amino acid, let `p[s]` be the fitted predictive
probability of the N endpoint event at site `s`. The N draw totals are
the observed ASR mass, the model-predicted mass `sum(p)`, or their equal
average. We integrate floor and ceiling draw sizes with stochastic
rounding probabilities. Wallenius uses `p` as the weights. Fisher uses
either `p` or the odds `p / (1 - p)`. The Fisher `p` control isolates the
change of subset law from the change of weights. [Fisher calculation](../../docs/URN_CALIBRATION.md#fisher-subset-weights)

At a fixed integer total, independent Bernoulli events with probabilities
`p` have a conditional subset distribution proportional to the product
of those odds. Exhaustive enumeration checks this identity and verifies
that averaging over all possible totals under the Bernoulli model
recovers each site's `p`. The study also reports the unconditioned N
expectation from the ordinary codon calculation with the same urn S.
Holding the total near `sum(p)` through rounding is a separate comparison
to averaging over the full total distribution.

The replay verifies the observed and predicted counts against saved
`search` outputs for every alignment. The archived Wallenius comparison
with observed totals reproduces the earlier N-weight and S-parent
treatment across all 63 seed-and-subset groups. Pairwise omegaC loss is
RMSE on the `log(1 + omegaC)` scale against the generating expected
omegaC, over the rows with finite ratios in every treatment. ECN and ECS
are also scored independently against the generating-Q null. The pooled
omegaC ratio is computed after adding the contributing counts across the
given pairs. [Per-seed and per-subset results](urn_conditioning_summary.tsv)

## Results

The table shows pairwise log RMSE, with pooled omegaC in parentheses.
The validation panel uses new fits and the same prespecified settings.

| N draw rule, validation panel | Background, unit rate | Background, mixed rates | Two biased foreground branches |
| --- | ---: | ---: | ---: |
| Ordinary codon model | 0.880 (0.907) | 0.887 (1.119) | 2.665 (7.130) |
| Wallenius, observed totals | 1.063 (0.812) | 1.052 (1.126) | 2.551 (7.135) |
| Wallenius, half observed and half model totals | 0.846 (0.808) | 0.839 (1.115) | 2.580 (6.555) |
| Wallenius, model totals | 0.859 (0.798) | 0.850 (1.086) | 2.594 (6.097) |
| Fisher odds, observed totals | 1.063 (0.810) | 1.051 (1.119) | 2.551 (7.119) |
| Fisher odds, half observed and half model totals | 0.846 (0.805) | 0.838 (1.104) | 2.579 (6.537) |
| Fisher odds, model totals | 0.859 (0.795) | 0.850 (1.074) | 2.594 (6.078) |

On the archived panel, the same Wallenius half-and-half comparison moves
pairwise log RMSE from 1.046 to 0.822, 0.947 to 0.789, and 2.089 to 1.538.
The improvement over observed totals repeats on both background
conditions in the validation panel. On its biased pairs the pooled
ratio moves from 7.135 to 6.555 while the individual-pair RMSE increases
slightly. That panel has generating expected omegaC 3.283 and realized
omegaC 7.000 from the true simulated ancestors. The archived biased pairs
have generating expected omegaC 3.296 and realized omegaC 4.681. The
nine new fits provide limited resolution for comparing performance on
individual biased pairs.

On the validation panel, Wallenius half-and-half changes ECN RMSE from
0.0101 to 0.0052, 0.0185 to 0.0105, and 0.0079 to 0.0041. Full model
totals bring ECN RMSE close to the unconditioned codon N calculation;
the pooled results also agree closely when S is held fixed. Across the
two panels, switching from Wallenius to Fisher odds with the same total
changes the scenario-level omegaC RMSE by less than 0.0006 and pooled
omegaC by less than 0.019. Fisher with `p` gives the same practical
comparison. The large effect in this experiment is how the N draw totals
are estimated. [Per-seed and per-subset results](urn_conditioning_summary.tsv)

The empirical follow-up learns parent-to-destination site weights from
background endpoint counts, keeping the observed urn draw totals.
[Empirical route weights](URN_EMPIRICAL_ROUTES.md)

## Reproduction

The editable CSUBST 1.17.5 checkout is based on `c39484a`. The study ran
with Python 3.12.15 on macOS arm64 with one BLAS thread per worker. Set
`panel` to the original simulation directory and `runs` to the saved
same-fit codon runs from the [ASRV study](URN_ACCURACY.md#reproduce).
Set `IQTREE_EXE` to IQ-TREE 3.1.4 for the fresh validation fits.

```bash
work="$(mktemp -d)"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHONPATH=reports/simulate_sensitivity_20261116 python - "$work/new" "$IQTREE_EXE" <<'PY'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from run import CONDITIONS, MODEL, prepare, search, simulate

panel, iqtree = Path(sys.argv[1]), sys.argv[2]
inputs = prepare(panel, 'PEPC')

def fit(case):
    scenario, seed = case
    run = panel / 'PEPC' / f'{scenario}-{seed}'
    run.mkdir()
    simulate(run, 'PEPC', inputs, CONDITIONS[scenario], seed)
    search(run, MODEL + '+R4', seed, iqtree, 'default')

cases = [(scenario, seed) for scenario in ('baseline', 'rate_mixture', 'biased')
         for seed in (411, 522, 633)]
with ThreadPoolExecutor(max_workers=9) as pool:
    list(pool.map(fit, cases))
PY
python reports/omega_low_count_20261116/urn_conditioning.py \
  "$panel" "$work/archived.tsv" --archived-fits "$runs"
python reports/omega_low_count_20261116/urn_conditioning.py \
  "$work/new" "$work/fresh.tsv"
```

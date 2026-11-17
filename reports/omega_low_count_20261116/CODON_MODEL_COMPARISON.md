# Codon-model comparison for PEPC omegaCany2spe

Fit `GY+F+R4`, `MGK+F3X4+R4`, and `ECMK07+F+R4` to the 18 complete PEPC
alignments from the earlier urn experiments. Compare with the saved
`ECMK07_GY0K+F+R4` fits on the same alignments. All four models use the
same rooted topology and four FreeRate categories. Each model gets its
own fitted branch lengths, category rates, category weights and fitted
substitution parameters. MGK uses counted position-specific nucleotide
frequencies and the other models use empirical codon frequencies. Search
uses joint single-edge endpoint posteriors and multiplies across the
focal edges for `any2spe`, with sister pairs included. The generating
background Q is based on `ECMunrest` with a rescaled nonsynonymous
component. The model comparison tests end-to-end accuracy on these
simulations. [Simulation and archived fit preparation](URN_CONDITIONING.md#reproduction)

The three conditions use background Q with unit rates, background Q
with mixed rates, and directional bias on foreground branches. Each
condition has six seeds across the two panels. The two background
conditions each pool 50,262 eligible pair-replicates, and the directional
bias comparison pools 216 pair-replicates with both branches affected.
The archived and second panels also appear separately in the summary.

Pooled omegaC uses the ratio of normalized counts after summing the
contributing OCN, OCS, ECN and ECS. ECN and ECS are compared with the
generating-Q null. Observed count RMSE uses the realized counts from
the true simulated ancestors. Pairwise omegaC RMSE is measured on
`log(1 + omegaC)` against the generating expected omega, on the common
finite rows. The oracle-EC calculation keeps a model's reconstructed
observed counts and uses the generating-Q denominators. This measures
how the result changes when the expectation error is removed. The score
inputs include the IQ-TREE reported log-likelihood and AIC.

For each of the four ancestral-state fits, compare three ways to compute
ECN and ECS while holding OCN and OCS fixed. The codon-model expectation
uses the fitted Q. Wallenius `--asrv each` allocates each branch's observed
destination totals over sites using destination-specific counts pooled
across branches and a pseudocount of one per site. The empirical urn
also holds branch destination totals fixed. It learns weights from the
inferred parental states and substitutions on all eligible branches
outside the two evaluated branches' folds. Four folds are assigned by
branch ID modulo four. On this tree, 138 branches enter the training pool
before the fold exclusion. The simulator's foreground labels only identify
the biased pairs when scoring the results. The ASR for all three methods
uses the same fitted codon model and full alignment.
[Empirical urn design](URN_EMPIRICAL_ROUTES.md)

## Results

Each cell contains the pooled omegaC. The ratios in parentheses give
ECN and ECS relative to their generating-Q null expectations.

| ASR model | Expected counts | Background, unit rate | Background, mixed rates | Both branches directionally biased |
| --- | --- | ---: | ---: | ---: |
| Generating model | Generating expected omegaC | 1.000 | 1.000 | 3.290 |
| True simulated ancestors | Realized omegaC | 0.966 | 1.005 | 5.948 |
| `ECMK07_GY0K+F+R4` | Fitted Q | 0.894 (1.090, 1.010) | 1.013 (1.002, 0.959) | 5.923 (1.130, 1.073) |
| `ECMK07_GY0K+F+R4` | Urn `each` | 0.145 (0.460, 0.069) | 0.230 (0.210, 0.046) | 1.009 (0.461, 0.075) |
| `ECMK07_GY0K+F+R4` | Empirical urn | 0.923 (1.049, 1.004) | 1.168 (0.795, 0.877) | 5.957 (1.080, 1.031) |
| `GY+F+R4` | Fitted Q | 0.407 (3.018, 1.280) | 0.489 (2.537, 1.212) | 2.959 (3.034, 1.373) |
| `GY+F+R4` | Urn `each` | 0.151 (0.471, 0.074) | 0.258 (0.220, 0.056) | 1.084 (0.464, 0.077) |
| `GY+F+R4` | Empirical urn | 0.934 (1.080, 1.050) | 1.184 (0.849, 0.982) | 6.308 (1.088, 1.050) |
| `MGK+F3X4+R4` | Fitted Q | 0.464 (2.330, 1.113) | 0.549 (2.017, 1.046) | 3.304 (2.377, 1.193) |
| `MGK+F3X4+R4` | Urn `each` | 0.148 (0.469, 0.071) | 0.237 (0.218, 0.049) | 1.072 (0.463, 0.075) |
| `MGK+F3X4+R4` | Empirical urn | 0.928 (1.072, 1.025) | 1.166 (0.835, 0.919) | 6.300 (1.084, 1.037) |
| `ECMK07+F+R4` | Fitted Q | 0.117 (4.410, 0.522) | 0.152 (3.487, 0.466) | 0.712 (4.900, 0.554) |
| `ECMK07+F+R4` | Urn `each` | 0.143 (0.463, 0.067) | 0.223 (0.213, 0.042) | 0.992 (0.465, 0.073) |
| `ECMK07+F+R4` | Empirical urn | 0.912 (1.060, 0.981) | 1.155 (0.802, 0.814) | 5.865 (1.092, 1.018) |

Across ASR models, `each` gives 4–8% of the generating-Q ECS. The empirical
urn gives 98–105% at unit rate and 81–98% in the mixed-rate condition.
Its ECN is still about 15–21% low with mixed rates, and pooled omegaC
exceeds one for all four empirical urn fits in that condition. Within each
ASR model, the empirical urn has higher pairwise log omegaC RMSE than the
fitted Q in all three conditions, on rows with finite omegaC for all three
expectation methods of that model.
[Per-seed counts and errors for both urns](compare_urn_asr_unlabeled_summary.tsv)

Including the simulated foreground in the pool of eligible training
branches reduced pooled empirical urn omegaC by 0.8–0.9% at unit rate,
1.2% at mixed rates, and 2.2% on pairs with directional bias on both
branches. In the biased pairs, ECN rose by about 1.8% and ECS fell by
about 0.4%. These changes use the same ASR and observed pair counts in
the two runs. [Previous per-seed counts with the foreground held out](compare_urn_asr_summary.tsv)

The ECN follow-up separates site weighting from destination totals and
compares a cross-validated site profile with pooled N+S totals.
[ECN bias analysis](URN_ECN_BIAS.md)

Among the four codon models, the hybrid has the smallest pairwise ECN and ECS RMSE in all three
comparisons. For mixed rates, ECN RMSE is 0.00815 for the hybrid, 0.05142
for GY, 0.03468 for MGK and 0.07295 for ECMK07. Replacing just the expected
counts with the common generating-Q denominators gives pooled omegaC
of 0.961–0.985 and 1.023–1.137 in the two background conditions, and
6.240–6.582 on the biased pairs. Thus most of the separation among the
pooled ratios comes from expected counts. The hybrid's observed S RMSE
is also the smallest in all three comparisons. It has the lowest AIC
in each of the 18 alignments.

The ranking of pairwise log omegaC loss differs from expectation-count
accuracy. At unit rate, for mixed rates, and in the two-foreground subset,
log RMSE is 0.865, 0.844 and 2.168 for the hybrid. The corresponding
values are 0.825, 0.789 and 2.012 for GY, 0.849, 0.810 and 2.081 for MGK,
and 0.805, 0.776 and 2.014 for ECMK07. The statistic is evaluated on
50,214, 50,244 and 216 common finite pairs, respectively. MGK's pooled
foreground value is close to the generating expected ratio alongside an
ECN overestimate of 2.38 times and elevated realized N in these six seeds.
The hybrid's value is close to the ratio calculated from the realized
true endpoints. The mean across possible simulations and the realization
of these seeds are distinct targets in this comparison.
[Per-panel and per-seed results](codon_model_comparison_summary.tsv)

## Reproduction

Use the saved original simulation panel, its complete codon-model runs,
and the same-topology second PEPC panel from the [prior study](URN_CONDITIONING.md#reproduction).
The replay uses editable CSUBST 1.17.5 based on `c39484a`, Python 3.12.15,
and IQ-TREE 3.1.4 on macOS arm64. Each search fits on one thread.

```bash
work="$(mktemp -d)"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/compare_codon_models.py \
  "$panel" "$runs" "$second_panel" "$work/fits" --iqtree "$IQTREE_EXE" --jobs 11
```

Apply both urn estimators to the saved fits with the following command.
For the pooled values, sum the counts for `all` in the background
conditions and `two_foreground` in the biased condition.

```bash
python reports/omega_low_count_20261116/compare_urn_asr.py \
  "$panel" "$runs" "$second_panel" "$work/fits" "$work/urn-asr-unlabeled.tsv" --jobs 8
```

The previous run held out the simulated foreground from all urn training
folds. To reproduce it, add `--exclude-known-foreground` and select a
different output path. [Previous per-seed comparison](compare_urn_asr_summary.tsv)

The replay checks each generator hash and reproduces the codon-model
search's observed and expected counts for every fitted alignment. The
standard urn was also checked pair by pair against independent
`search --expectation_method urn --urn_model wallenius --asrv each` runs
using each model's saved seed 411 unit-rate fit.

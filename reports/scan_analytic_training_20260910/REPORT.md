# Independent CTMC training and untouched evaluation

The learned alternative improves sensitivity strongly in sparse four-foreground
trees, but it is not a general replacement for the fixed mixture. Under the
requested selected trait×match BH comparison, its complete-null false-positive
probability increases from 1% to 6% in that setting. Two-foreground sensitivity
does not improve; unequal and uncertain trees show little benefit or small
losses. The profile remains explicitly opt-in.

## Implemented method

The default four equally weighted endpoint alternatives are preserved.
`--scan_analytic_profile JSON` loads prespecified weighted endpoint alternatives,
each with an effect multiplier and a per-branch participation probability.
Participation mixes the tilted and null transition matrix on each foreground
branch. This integrates over independently participating branches rather than
requiring identical changes on every foreground. All transition rows and the
mixture normalize; scaled pruning integrates hidden ancestors and shared rate
categories. The reciprocal e-value construction still holds for a fixed,
correct null and independently fixed mixture.

The candidate components were: default four atoms; eight-point Gauss–Legendre
integration of a log-uniform multiplier from 1 to 1000 with full participation;
the same with half participation; infinite-limit half participation; and
infinite-limit full participation. These are endpoint alternatives, whereas the
training generator alters actual CTMC jump rates.

Independent training used 600 datasets / 4,800 sites, spanning three geometries
and two/four foregrounds. Each foreground K-to-N Q entry is multiplied by a
log-uniform effect between 2 and 64. Half of sites share an effect across
foregrounds and the rest have heterogeneous effects. Q diagonals are recomputed;
tip observations are generated from Gillespie histories, never overwritten.
Mean predictive log likelihood across all training sites determines weights;
candidate discovery is not involved in training.

Fitted component weights are 0.772862557 for log-uniform/full participation and
0.227137413 for infinite/half participation. The other three components retain
the optimizer lower bound of approximately 1e-8. The frozen profile SHA-256 is
`6185061068421747d45c62c9638d2915c475a5921ba3aa5fbe27646e2324f1c6`.
The profile was saved before any holdout scoring and did not change afterward.

## Paired held-out design

There are 1,800 independent held-out datasets: 100 replicates for every
geometry × foreground count × regime. Regimes are complete null, homogeneous
foreground boost 16, and alternating boosts 2/64. Each dataset has 12 sites;
alternative datasets contain three process-changed sites. The same dataset is
evaluated with and without invariant-tip-site filtering, giving 3,600 scans.
The model has four codons (AAA,AAG,AAC,AAT); the uncertain geometry has a latent
three-category site-rate mixture integrated by the independent null pruner.

All three methods receive exactly the same discovered rows and the same
selected trait×match BH denominator at alpha=.05. This is the comparison
requested, not a globally valid discovery correction. Production endpoint
q-values still use the full prespecified family.

The table pools the two alternative regimes for sensitivity (600 changed sites
per geometry/foreground/filter). Process sensitivity counts changed sites with
a significant trait1-to-N candidate. Realized endpoint sensitivity conditions
on at least two foreground branches actually ending K-to-N in the simulated
history. Jump-conditioned sensitivity is separately available in the CSV.
Null any false is the probability of at least one discovery in a complete-null
dataset (100 replicates). These are not per-test type-I errors.

| Geometry | Foregrounds | Filter | Null any false: Poisson / fixed / learned | Process sensitivity: Poisson / fixed / learned | Realized endpoint sensitivity: Poisson / fixed / learned |
|---|---:|---|---|---|---|
| sparse | 2 | False | 0.0% / 2.0% / 2.0% | 2.7% / 4.8% / 4.8% | 50.0% / 90.6% / 90.6% |
| sparse | 2 | True | 0.0% / 0.0% / 0.0% | 2.7% / 4.0% / 4.0% | 37.5% / 75.0% / 75.0% |
| sparse | 4 | False | 0.0% / 1.0% / 6.0% | 2.8% / 8.2% / 27.2% | 9.1% / 26.2% / 87.2% |
| sparse | 4 | True | 0.0% / 0.0% / 1.0% | 2.8% / 6.8% / 18.5% | 6.4% / 21.9% / 59.4% |
| uncertain | 2 | False | 5.0% / 3.0% / 3.0% | 4.8% / 4.0% / 4.0% | 21.6% / 17.6% / 17.6% |
| uncertain | 2 | True | 5.0% / 1.0% / 1.0% | 4.8% / 3.8% / 3.8% | 20.0% / 16.8% / 16.8% |
| uncertain | 4 | False | 5.0% / 0.0% / 0.0% | 0.8% / 3.2% / 3.0% | 1.8% / 6.3% / 6.0% |
| uncertain | 4 | True | 5.0% / 0.0% / 0.0% | 0.8% / 2.8% / 2.8% | 1.8% / 5.6% / 5.6% |
| unequal | 2 | False | 0.0% / 0.0% / 0.0% | 0.8% / 0.0% / 0.0% | 6.5% / 0.0% / 0.0% |
| unequal | 2 | True | 0.0% / 0.0% / 0.0% | 0.8% / 0.0% / 0.0% | 6.5% / 0.0% / 0.0% |
| unequal | 4 | False | 12.0% / 2.0% / 2.0% | 0.7% / 8.5% / 8.0% | 1.5% / 18.8% / 17.6% |
| unequal | 4 | True | 12.0% / 2.0% / 2.0% | 0.7% / 8.3% / 7.8% | 1.5% / 18.4% / 17.3% |

Sparse four-foreground, unfiltered process sensitivity rises from 8.2% to 27.2%;
realized endpoint sensitivity rises from 49/187 (26.2%) to 163/187 (87.2%).
The paired process-sensitivity increase is 12.7 percentage points for homogeneous
effects (approximate dataset-level 95% interval 9.1–16.2), and 25.3 points for
heterogeneous effects (20.6–30.1). These intervals are descriptive, normal
approximations with no adjustment for multiple comparisons.

The same complete-null setting has 6/100 learned-model datasets with a false
discovery: exact binomial 95% interval 2.2–12.6%. Thus these data neither establish
5% control nor precisely locate the false-positive probability. In heterogeneous
alternative datasets the learned method has mean null-site FDP 6.83%, versus
1.0% for the fixed mixture; the homogeneous counterpart is 1.83% versus 0%.
Null-site FDP counts only hypotheses at unchanged sites: off-target traits or
directions at process-changed sites are not labeled true nulls by assumption.

The profile was not retuned after these results. More foregrounds and a more
flexible alternative do not fix selection-induced denominator reduction or
fitted-null uncertainty. These results support an experimental option, not
automatic promotion to the default.

## Validation and limits

- Host: 212 related scan/endpoint tests passed; one compiled projection test was
  skipped because the host extension is not built.
- GeneGalleon Docker runtime: 62 related tests passed.
- GY+FQ+G4 with joint posterior ran end to end in the container, emitted eight
  candidates, and retained the exact frozen profile in inference JSON.
- New tests independently enumerate weighted/partial-participation likelihoods,
  verify unit null expectation, compare Gillespie endpoints against expm, and
  check latent-category posterior integration against clamped pruning.
- Ruff passed on changed analytical/training files; mypy passed for the
  analytical engine; git diff whitespace check passed.
- SIF was not tested. This known-parameter four-codon calibration does not
  establish validity after IQ-TREE nuisance fitting or transfer to 61-codon
  biological datasets.

An initial evaluation invocation stopped on a candidate column-name typo
before producing holdout scores. The correction resumed the already-frozen
profile and identical seeds; no retraining took place. Summary parsing was
also corrected to preserve the literal regime name 'null' rather than treating
it as a missing CSV value. Neither correction altered simulation or inference.

## Reproduction and artifacts

Run from the CSUBST checkout:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CSUBST_DISABLE_EXTENSIONS=1 python tools/train_scan_analytic.py --output /tmp/fresh-training-study
python tools/summarize_scan_training.py /tmp/fresh-training-study
```

The output directory must be fresh. `--resume-frozen` resumes an interrupted
evaluation with the saved weights and verifies matching configuration.
[Preregistered settings](frozen_v1/preregistered.json),
[frozen profile](frozen_v1/profile.json), [all paired rows](frozen_v1/paired_results.csv),
[detailed metrics and binomial intervals](frozen_v1/summary.csv), and
[paired differences](frozen_v1/paired_differences.csv) are retained.
`run.json` records evaluation elapsed time; when resuming it excludes the
already completed training phase.

Previous reports used direct tip-state signal injection. Their power numbers
must not be interpreted as a before/after comparison with this new generator.

## Runtime cost

61 codons, four rate categories, 100 candidates, balanced 16/64/256-tip trees,
and up to eight target branches. Each measurement uses a fresh process, one
candidate warmup and four repeated runs on identical seeds. Null model
construction, simulation, IQ-TREE and candidate discovery are outside the
timed kernel. Fixed and learned runs were sequential, after holdout simulation
finished, using Python 3.10.14 on macOS ARM64 with BLAS/OMP threads set to one.
These methods intentionally produce different evidence; this measures the cost
of the added mixture, not an equivalent-output speedup.

| Tips | Fixed median seconds | Learned median seconds | Ratio | Fixed peak MiB | Learned peak MiB |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.658 | 1.637 | 2.49× | 135.0 | 150.3 |
| 64 | 1.601 | 2.923 | 1.83× | 134.3 | 151.6 |
| 256 | 5.541 | 8.411 | 1.52× | 137.9 | 158.2 |

Reproduce with `tools/benchmark_scan_analytic_codon.py --candidates 100 --output FILE`,
first without a profile and then with `--profile frozen_v1/profile.json`.
All per-repeat timings, peak memory and strong-pattern checks are in
[fixed results](runtime_fixed.json) and [learned results](runtime_learned.json).
The retained 21 alternative atoms increase compute and memory relative to four.
Strong hand-constructed endpoint patterns in these files are illustrations,
not empirical power estimates.

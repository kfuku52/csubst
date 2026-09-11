# Issue #46: end-to-end null calibration

This experiment measures the existing omegaC tests. It does not recalibrate
them or tune the settings using their measured rejection rates.

## Prespecified design

Two hundred independent 400-codon alignments in each of two eight-tip tree
regimes (400 alignments total). The long-tree regime scales every branch of
the short tree by three. All branches and sites share omega=0.2 and kappa=2.5;
site rates follow four discrete-Gamma categories with shape 0.6. Positional
nucleotide frequencies define a stationary F3X4 codon distribution. The
vendored Pyvolve generator has no foreground-specific process or imposed
convergent sites. Simulation and inference use separate seeded streams.

Each alignment goes through IQ-TREE 3.1.4 fitting of GY+F3X4+G4, branch lengths,
omega, kappa, codon frequencies and Gamma shape, followed by ASR and site-rate
estimation. Only the known rooted topology, without any branch lengths, is
passed from simulation to inference. No ancestral truth or true parameter is
supplied. The model family and tree topology are fixed; their selection,
misspecification, recombination and alignment error are outside this design.
The executable comes from the [official IQ-TREE release](https://github.com/iqtree/iqtree3/releases/tag/v3.1.4).

Each fitted alignment is used by four independent CSUBST search executions:

| Setting | Count null | min_sub_pp | Pseudocount | Long-tail |
| --- | --- | --- | --- | --- |
| hypergeom_pp0 | hypergeom | 0 | none | off |
| hypergeom_pp005 | hypergeom | 0.05 | none | off |
| poisson_symmetric_pp005 | poisson | 0.05 | symmetric alpha=1, both | off |
| poisson_independent_pp005 | poisson | 0.05 | symmetric alpha=1, both | independent_null |

All settings use marginal endpoint posteriors, urn expectations, ASRV=each,
stochastic rounding, any2spe, one thread and 3,999 fixed test draws. Independent
long-tail maps use 1,000 additional training draws. Model/ASR fits are shared
only between settings on the same alignment; every independent alignment is
fitted anew. The production source is frozen before the experiment so that
other workspace changes cannot change the running analysis.

The actual search evaluates all eligible nonsister independent branch pairs
(K=2), then applies the default OCNany2spe>=2 and omegaCany2spe>=5 cutoff to
generate K=3 candidates. Empty candidate sets remain valid outcomes. Max
combination=10,000; exhaustive_until=2; max_arity=3. Output precision is 12
decimal places to avoid classification changes from four-digit TSV rounding.
The final reporting rule applies the same effect-size cutoffs to evaluated
rows. This is a stated analysis rule, not an automatic universal discovery
rule imposed by CSUBST.

## Estimands and denominators

The primary measure is the fraction of independent alignments with at least
one cutoff-selected row whose production q<=0.05, across evaluated K=2 and
selected K=3. Under this complete null, every discovery is false, so this
probability is both the family-wise error rate and the FDR of this reporting
rule for one search run. Production BH families remain per arity; we do not
pretend they provide a joint adjustment across arities.

Secondary measures are any evaluated p<=0.05 or q<=0.05, any cutoff-selected
p<=0.05, and p<=0.05 for the prespecified a/e tip-branch pair. The latter is a
single prespecified hypothesis; an unadjusted minimum over many rows is not.
The winner diagnostic takes the highest omegaC among selected rows separately
at each arity and records whether either winner has p<=0.05.

Exact two-sided 95% binomial intervals use independent alignments, never
correlated branch rows. Intervals are pointwise, not simultaneous across
settings/regimes. Missing P/Q values count as no reported rejection in the
unconditional dataset indicators, with finite counts and conditional
prespecified-pair results reported separately. Failed executions are not
silently excluded. No stopping or threshold changes are based on interim FPR.

## Reproduction

From the repository root, with the reported source revision and IQ-TREE binary:

```bash
python .github/scripts/omega_pipeline_fpr.py \
  --workdir /tmp/csubst_issue46_pipeline_full_20260910 \
  --iqtree /path/to/iqtree3 --replicates 200 --workers 4 --seed 4609201
python -m pytest -q tests/unit/test_omega_pipeline_fpr_script.py
```

The work directory must be new. It retains every simulated alignment, fitted
IQ-TREE artifact, CSUBST output, command, timing and per-alignment result.
The pilot used different seeds and is excluded from the 400 confirmatory
replicates. A preliminary IQ-TREE 2.3.6 executable left supplied branch lengths
unchanged on the pilot; 3.1.4 was independently checked to optimize lengths
before starting the experiment. The existing installation was not modified.

## Results

**After candidate selection, using unadjusted P<=0.05 reached a 13.0% run-level false-positive rate in the long-tree hypergeom/min_sub_pp=0.05 setting (26/200; 95% CI 8.67–18.47%). Using production Q<=0.05 gave 0/200 in each of the eight regime/setting cells (pointwise 95% CI 0–1.83%).**

The zero Q result also holds before applying the effect-size cutoffs: no evaluated row had Q<=0.05. Thus that result is not solely an artifact of empty final candidate sets. The Poisson/symmetric settings produced no cutoff-qualified candidates; their power under a convergence alternative was not assessed.

| Regime | Setting | Selected P<=0.05: probability per search (95% CI) | Any P<=0.05, before cutoff | Prespecified pair P<=0.05 | Datasets with cutoff-qualified pairs |
| --- | --- | --- | --- | --- | --- |
| short | hypergeom_pp0 | 2/200 = 1.0% (0.12–3.57%) | 58/200 | 1/200 | 4/200 |
| short | hypergeom_pp005 | 1/200 = 0.5% (0.01–2.75%) | 69/200 | 1/200 | 4/200 |
| short | poisson_symmetric_pp005 | 0/200 = 0.0% (0.00–1.83%) | 14/200 | 0/200 | 0/200 |
| short | poisson_independent_pp005 | 0/200 = 0.0% (0.00–1.83%) | 10/200 | 0/200 | 0/200 |
| long | hypergeom_pp0 | 14/200 = 7.0% (3.88–11.47%) | 62/200 | 1/200 | 14/200 |
| long | hypergeom_pp005 | 26/200 = 13.0% (8.67–18.47%) | 77/200 | 2/200 | 26/200 |
| long | poisson_symmetric_pp005 | 0/200 = 0.0% (0.00–1.83%) | 18/200 | 0/200 | 0/200 |
| long | poisson_independent_pp005 | 0/200 = 0.0% (0.00–1.83%) | 11/200 | 0/200 | 0/200 |

All 83,200 evaluated P/Q pairs were finite. Every search evaluated 52 branch pairs, yielding 10,400 rows in each regime/setting cell. The prespecified pair was finite in all 200 replicates per cell. These correlated rows were never used as independent Bernoulli trials.

**Higher-arity limitation:** K=3 was requested, and the production candidate-selection step ran in every search, but no eligible triplet was generated in any of the 1,600 searches. This experiment measures branch-pair selection and its stopping behavior. It supplies no empirical calibration evidence for post-selected K>=3 P/Q values.

For the prespecified a/e pair, rejection was 0–2/200 (0–1%). Across 52 screened pairs, any unadjusted P<=0.05 occurred in 5–38.5% of searches. This distinction is why the unadjusted minimum or a selected winner cannot be treated as a prespecified 5% test. Increasing min_sub_pp to 0.05 did not control the selected-P reporting rule at 5% in the long-tree case. The paired comparison between threshold settings was not separately tested.

The observed Q results support conservative run-level behavior for this matched-model, eight-tip experiment. They do not establish general calibration or useful power, particularly for larger trees, model/topology selection, model misspecification, or K>=3. Keep #46 open for those questions.

## Fitting, computation and audit

Tested source: [72c3c45](https://github.com/kfuku52/csubst/commit/72c3c4543e66e21ae48aaa7bec80b66e953d18da) (CSUBST 1.15.2). A frozen snapshot of 101 tracked source files was checked against that commit. IQ-TREE 3.1.4 was used for all confirmatory runs. Exact software versions and hashes are in [environment.json](environment.json) and [metadata.json](metadata.json).

The 400 alignments required 400 fresh IQ-TREE fits and 1,600 CSUBST searches. Total outer wall time was 30.08 minutes with four workers. Runtime is descriptive on a shared host; this is not a before/after optimization comparison.

| Regime | Median fitted omega | Median fitted kappa | Median fitted Gamma shape | Median total fitted branch length | Mean-max ASR posterior: median (range) |
| --- | --- | --- | --- | --- | --- |
| short | 0.200 | 2.571 | 0.597 | 1.984 | 0.950 (0.931–0.964) |
| long | 0.200 | 2.555 | 0.603 | 5.943 | 0.825 (0.779–0.860) |

| Regime | Setting | Median search seconds | Maximum search RSS (MiB) |
| --- | --- | --- | --- |
| short | hypergeom_pp0 | 3.28 | 237.2 |
| short | hypergeom_pp005 | 2.51 | 208.5 |
| short | poisson_symmetric_pp005 | 2.43 | 156.7 |
| short | poisson_independent_pp005 | 3.96 | 161.8 |
| long | hypergeom_pp0 | 7.12 | 304.4 |
| long | hypergeom_pp005 | 3.34 | 207.1 |
| long | poisson_symmetric_pp005 | 2.79 | 160.0 |
| long | poisson_independent_pp005 | 4.96 | 161.9 |

All 400 alignments and fitted artifacts were hash-verified. All 1,600 output tables passed an independent BH recalculation and agreement with the production cutoff counts. There were zero failed or omitted replicates. See [audit.json](audit.json).

Full repository tests: **2,120 passed, 5 skipped**, on Python 3.10.14. Three skips require PyTorch>=2.6, and two require gemmi. One existing requests dependency warning remains. Ruff, repository hygiene, documentation checks and configured mypy targets passed. The new audit/aggregation tests cover missing P values, independent-dataset denominators, selection, Q discrepancies and seed separation.

## Retained evidence

- [Summary and pointwise intervals](summary.json)
- [Per-alignment records](records.json.gz)
- [Every evaluated row](tested_rows.tsv.gz)
- [Simulated alignments](simulated_alignments.fasta.gz)
- [Generating parameters](generator.json)
- [Public-comment replicate event lists](replicate_events.json)
- [Full test log](pytest.log)

Export and independently audit a finished run with:

```bash
python .github/scripts/omega_pipeline_fpr_report.py \
  --workdir /tmp/csubst_issue46_pipeline_full_20260910 \
  --outdir reports/issue46_pipeline_20260910
```

## Issue update

Posted and verified: [Issue #46 comment](https://github.com/kfuku52/csubst/issues/46#issuecomment-5619521815).
The issue remains open.

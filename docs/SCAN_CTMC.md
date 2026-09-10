# Joint posterior and CTMC bridge scan

`scan` supports joint endpoint probabilities and posterior CTMC jump counts.
Use the shared fitted bootstrap to refit the model, branch lengths and ASR for
every simulated alignment, then repeat candidate selection:

```bash
python -m csubst scan \
  --alignment_file alignment.fa --rooted_tree_file rooted.nwk \
  --foreground foreground.tsv --iqtree_model GY+FQ \
  --scan_observation bridge --scan_min_event_count 0.5 \
  --scan_rate_exposure endpoint --scan_rate_length raw \
  --scan_rate_event_mode posterior_sum --min_sub_pp 0 \
  --scan_pvalue_calibration parametric_bootstrap --scan_n_permutations 999 \
  --scan_site_plot_filter parametric_bootstrap --threads 1 --outdir scan_bridge
```

Use `--scan_observation joint --scan_min_event_pp 0.5` for joint endpoint
probabilities. The default `marginal` observation and `q_weighted` exposure
retain the legacy calculation. These methods estimate different quantities;
their event counts need not agree.

## Observation and exposure definitions

Let `a` be the likelihood message above an edge, including its siblings and the
root prior, and `b` the likelihood message below the edge. With generator `Q`,
length `t`, and `L = a exp(Qt) b`, joint endpoint probabilities are

```text
Pr(parent=i, child=j | all tips) = a[i] exp(Qt)[i,j] b[j] / L.
```

Scaled Felsenstein pruning computes these messages from leaf emissions. The
internal-node posterior values imported from IQ-TREE are ignored in this
calculation. The root prior is the parsed fitted stationary codon distribution. The result
is the exact joint posterior for the supplied Q and emission likelihoods;
GY+F and GY+FQ scans read checkpoint parameters and full-precision log
frequencies, including each bootstrap refit. Other supported codon families
retain the existing report-based input precision. Explicit GY intermediate
files require their matching checkpoint/log; automatic inputs are refitted
when these are absent.
Codon-pair masses are grouped only after calculation. Same reported-state
endpoints are excluded from the joint nonsynonymous event tensor.

Bridge observations instead use

```text
E[N_ij | all tips] = Q[i,j] / L * integral_0^t
                     (a exp(Qu))[i] (exp(Q(t-u)) b)[j] du, i != j.
```

This counts every within-edge jump, including changes on paths that return to
the initial state. Synonymous/within-group jumps are excluded when accumulating
the nonsynonymous tensor. Spectral divided differences evaluate the integral
for reversible Q; ill-conditioned boundary likelihoods use a direct adjoint
Frechet derivative. Calculations process bounded site blocks, without storing
a full branch × site × codon × codon posterior array. The final grouped event
tensor and pruning messages are held in memory.

`endpoint` exposure with `joint` observations uses the finite-time endpoint
opportunity described in [SCAN_ENDPOINT.md](SCAN_ENDPOINT.md). With `bridge`
observations the same option instead integrates expected **jump counts**:

```text
opportunity = sum_k,l (parent_posterior integral_0^t exp(Qu) du)[k] Q[k,l]
              over the candidate's allowed, different reported-state pairs.
```

It allows intermediate jumps before reaching a candidate's source state. The
integrated transition matrix is calculated through a block matrix exponential,
not an inverse of singular Q. Exposure conditions on the inferred parent
state, but not on the child evidence; the observation conditions on all tips.
Their ratio and Poisson LRT remain exploratory scores. Calibration repeats
this entire calculation under the null, including estimating parent states
from the same data used in the score.

## Thresholds and output units

Bridge values can exceed one. `--scan_min_event_count` thresholds posterior
mean jump counts for discovery and foreground-unit support; it may exceed one.
`--scan_min_event_pp` applies to marginal/joint observations. Neither threshold
filters the event mass summed for the rate statistic (`posterior_sum`).

| Column | Interpretation |
|---|---|
| `scan_observation_method` | `marginal_posterior_product`, `joint_endpoint_posterior`, or `posterior_mean_jump_count` |
| `scan_event_units` | `endpoint_probability` or `posterior_mean_jump_count` |
| `scan_event_threshold` | Discovery/support threshold in those units |
| `candidate_event_mass_sum` | Sum of matching called event masses |
| `support_mass_sum`, `support_mass_mean` | Sum/mean of per-unit maximal matching event mass |
| `target_event_count`, `other_event_count` | All matching posterior event mass on each branch set |
| `scan_exposure_units` | `expected_endpoint_events` for joint, `expected_jump_count` for bridge |
| `score_rate_enrichment` | Stable negative log10 asymptotic tail, used for ranking and calibration |
| `scan_inference_status` | Exploratory, conditional clade calibration, or completed fitted bootstrap status |

For bridge output, legacy probability-named columns `candidate_event_pp_sum`,
`support_pp_sum`, `support_pp_mean`, and `scan_min_event_pp` are undefined.
Use the generic mass/threshold columns. Plotting uses the selected observation
tensor and its appropriate threshold.

## Parametric calibration

Both sequence-simulation modes use the same stable maximum score and inclusive
Monte Carlo tail, `(1 + number of null maxima >= observed score) / (B + 1)`.
They repeat ASR, event inference, discovery and support selection over all
traits/matches/sites in the run. Empty or explicitly untestable candidate
families remain in the reference. Invalid statistics are failures, never
silently removed from the denominator. No-enrichment scores return P=1.

| Mode | Model fitting in each replicate | Calibrated column | Plot filter |
|---|---|---|---|
| `parametric_bootstrap` | Refit GY parameters and branch lengths with IQ-TREE, then repeat recoding and site filtering | `p_rate_enrichment_bootstrap_maxT` | `parametric_bootstrap` |
| `parametric` | Hold Q, topology and lengths fixed; simulate new tip data and recompute ASR | `p_rate_enrichment_empirical_maxT` | `full_scan` |

The first mode uses the main [scan bootstrap implementation](SCAN_INFERENCE.md).
It preserves `scan_observation`, the appropriate discovery threshold and exposure
settings in the child command. Every joint/bridge replicate reads its own
precise fitted model. Calibration manifests identify the observation method,
record model checks and retain failed/empty trials. The supplied topology,
root, foreground and alignment length remain fixed; topology search is not
repeated. This is model-conditional inference, not an exact test or a guarantee
under model misspecification.

The second mode is retained for controlled fixed-model experiments, including
uniform ECM models. It replays missingness and the explicit partial-ambiguity
observation model described below. It does not quantify model/branch-length
estimation uncertainty. Candidate-wise empirical columns are undefined in this
mode; use the global maxT column directly, without another BH correction.

`p_rate_enrichment_asymptotic` and its trait × match BH column remain exploratory
diagnostics. `full_scan` calibration itself still permutes foreground clades
against fixed ASR tensors; it does not simulate sequences or refit ancestors.
All simulation modes run serially and have minimum Monte Carlo P `1/(B+1)`.

## Supported data and explicit limits

The same uniform GY/MG/ECMK07/ECMrest codon-model restrictions as
[endpoint exposure](SCAN_ENDPOINT.md#supported-inputs) apply. Bridge requires
reversibility and positive stationary frequencies. Mixture models and native
3Di are not supported. Raw model lengths, `--ml_anc no` and `--min_sub_pp 0` are required.
Use `--scan_observation` for scan; the separate `--substitution_posterior`
option controls search/analyze and must remain `marginal` here. IQ-TREE documents
codon-model lengths per codon site, rather than the DNA per-nucleotide scale
([model documentation](https://iqtree.github.io/doc/Substitution-Models#codon-models)).

A completely missing leaf/site has an all-ones emission likelihood and is
integrated out. Its latent branch events can have positive posterior mass;
missing tips are not assigned a zero substitution count. Bootstrap simulations
preserve these missing entries. Conservation annotations summarize the
recomputed tip posteriors; `*_valid_tip_count` counts posterior rows with mass,
including imputed missing tips, rather than the number of resolved input calls.

For fixed-model `parametric` calibration, partial IUPAC ambiguity is handled
as a specified nucleotide-coarsening model:
resolved positions remain resolved; an ambiguous nucleotide subset and its
complement define two categories; N has one category. A simulated codon emits
the corresponding category's compatible sense codons. The observed support
must be exactly reproduced by this partition, with uniform emission weights.
This retains partial information, but assumes this coarsening mechanism is
fixed, rather than modeling sequencing errors or data-dependent base calling.
IQ-TREE itself treats any ambiguous codon as wholly unknown, so retaining
partial codon information here can change ASR marginals even before replacing
the marginal product. See the [IQ-TREE codon input specification](https://iqtree.github.io/doc/Tutorial#using-codon-models).
The benchmark fit also reports frequencies with limited text precision;
[the independent comparison](../reports/scan_ctmc_20260910/README.md#independent-comparison-with-iq-tree-asr)
separates those effects.

Fitted `parametric_bootstrap` currently supports GY+F/GY+FQ, unambiguous sense
codons and completely missing codons. It rejects partial ambiguity and replays
the configured invariant-site filter on each simulated alignment.

Data-dependent invariant-site filtering is rejected for fixed `parametric`
calibration (`--drop_invariant_tip_sites no` is required). This prevents using
an observed-data-selected site set without replaying that selection. Candidate
selection within scan itself is replayed in every replicate.

## Reproducible comparisons

The [integration comparison](../reports/scan_integration_20260910/README.md)
compares the latest main implementation with joint/bridge under the same
fitted GY null, including real CLI refits and nested codon-model calibration.
The earlier fixed-model experiments remain available below.


```bash
python .github/scripts/scan_ctmc_benchmark.py \
  --fit-dir /tmp/pepc-uniform-fit --outdir /tmp/ctmc-benchmark --repeats 3
python .github/scripts/scan_ctmc_accuracy.py --out /tmp/ctmc-accuracy.json
```

Add `--calibration pipeline --niter 3` to the performance command to time
legacy full-scan permutations versus parametric ASR/discovery replicates.
Three replicates are a runtime workload, not a significance analysis.

The performance script uses the same PEPC fit, independent child processes,
fixed numerical-library thread limits, excluded warmups and rotating run
order. It records CLI/scan time, peak RSS, input hashes, and differences in
candidate scores. It compares uncalibrated scans because label permutations
and parametric bootstrap perform different work.

The accuracy experiment records full simulated jump histories on a small
two-state tree, repeats exact ASR and the production scan's candidate selection,
and uses independently seeded calibration/null-validation/alternative samples.
It reports count RMSE, family-wise false-positive rates, power, and binomial
intervals. `legacy_default` uses marginal products with N-rescaled lengths;
`marginal` additionally isolates the same product with raw Q-weighted exposure.
The offline bootstrap is applied to each observation method for a controlled
comparison; it is not the legacy foreground-label permutation procedure.
The shared calibration reference introduces uncertainty beyond the conditional
binomial intervals. This experiment does not substitute for codon-model
misspecification validation.

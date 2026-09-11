# Joint endpoint posteriors

Joint endpoint posteriors are the default for search/analyze, sites, inspect,
benchmark and scan. Use `--substitution_posterior marginal` to retain the previous
product of node marginals for reproducibility and comparisons. Changing the
estimator can change counts, candidate rankings and root-adjacent eligibility;
it is not a numerical optimization of the marginal estimator.

For `scan`, use [joint/bridge observations](SCAN_CTMC.md) through
`--scan_observation joint` or `bridge`, with endpoint exposure and raw model
lengths. The common `--substitution_posterior` switch also controls scan;
`--scan_observation` is a scan-specific override and additionally accepts `bridge`.
Unspecified exposure/length options resolve to endpoint/raw for joint or bridge,
and q_weighted/n_rescaled for marginal. Explicit incompatible settings fail.
There is no automatic fallback to marginal for unsupported models.

```bash
csubst search --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --substitution_posterior joint --endpoint_block_size 64
```

For a zero-length branch with parent and child both uncertain between A and B,
the joint distribution has no off-diagonal mass. Multiplying their marginal
probabilities can incorrectly report a change probability of 0.5.

## Estimator and supported inputs

Joint inference conditions on the complete tip observations, rooted tree and
parsed fitted model. Scaled pruning and parent-to-child conditioning compute
endpoint distributions without sampling ancestral histories. An inserted root
is inferred from the model. This can make root-adjacent branches eligible that
were previously excluded because the root had no ASR row.

- Codons: ECMK07, ECMrest, GY, MG and MGK, with uniform rates or an IQ-TREE discrete
  rate-category table. Categories use posterior weights given the whole site,
  not a posterior-mean site rate. Mixtures of different Q matrices and
  ascertainment-corrected models are rejected. MG/MGK require counted F1X4/F3X4
  nucleotide frequencies; other MG variants and frequency schemes are rejected.
- Conventional and recoded N sum the codon joint over both state axes. S contains
  distinct codons within the same synonymous group. Recoding does not assume a
  Markov process on the reduced alphabet.
- Native 3Di N uses its independent fitted uniform GTR context. Codon S and the
  AA stream for codon exposure/VESM remain independent of 3Di N. Direct ASR and
  the existing fitted-context cache are required; translate ASR is unsupported.
- `--ml_anc yes` is incompatible. Tip ambiguities are observation likelihoods.
  Missing tips contribute likelihood one to pruning. Unobserved child subtrees
  and wholly missing sites are excluded by a separate eligibility mask. Numerical
  event tensors store zero for excluded entries; output tables render them as `NA`. Internal
  marginals are recomputed; tip observation arrays are retained for rebuilding.

All commands use the same fitted codon model loader. When all five IQ-TREE
intermediate paths are explicitly supplied, their reported model is authoritative;
`--iqtree_model` selects a new fit only with `--iqtree_redo` or inferred inputs. GY+F/GY+FQ (also with
rate categories) use matching checkpoint parameters and precise log frequencies
when available. Report-only fits use the same reported precision in every
command; scan does not independently refit or replace their model. Fitted
bootstrap still requires precise sidecars; finite likelihood mismatches warn
and are recorded in provenance while bootstrap continues.
The posterior is conditional on the selected parameters. The 3Di generator
retains checkpoint precision. Sources and the generator hash, rates and branch
lengths are recorded in `*_endpoint_model.json`. Fresh GY/MG fits retain their
checkpoint and verbose log. Unsupported missing frequency schemes fail instead
of being replaced with empirical codon frequencies.

### Observation eligibility and TSV output

Search, sites and joint/bridge scan exclude branches whose child subtree has no
informative tip observation at that site. Partial ambiguity remains evidence;
zero or constant likelihood rows are unobserved. Root rows are not branches.
Latent states are still integrated during pruning. Scan applies the same mask
to event mass, synonymous summaries, exposure and analytical target branches;
bootstrap replays the original missingness and repeats discovery.

Tables distinguish a real zero from unavailable observations: `NA` means
excluded, not that a latent posterior cannot be inferred. S/N summary tables
include `S_eligible_count` and `N_eligible_count` (branch-sites, or eligible
sites for a branch combination). Across-combination site summaries also include
`S_combination_eligible_count` / `N_combination_eligible_count`; sites adds `S_eligible_<branch>` and
`N_eligible_<branch>`. The count is zero when the corresponding summary is
unavailable. Numeric tensors remain finite and masked, so reducers do not
propagate NaNs. Sparse candidate/event tables omit excluded events.


## MG model correction

MG/MGK use the target nucleotide frequency at the changed codon position;
GY uses the target codon frequency. Earlier CSUBST releases routed MG through
GY-style frequency weighting. This is corrected for both posterior modes and
simulation, so `marginal` retains the old estimator but does not reproduce that
MG matrix bug. F1X4/F3X4 frequencies are counted from the original fitted
alignment, excluding unknown codons as IQ-TREE does, and their sense-codon
products are checked against reported codon frequencies when available.
Stop-codon exclusion is included in stationary-frequency normalization.
An incompatible alignment/report pair fails rather than guessing parameters.
Fresh MG fits request verbose logs for fitted omega/kappa precision; existing
rounded logs retain their stated precision. Simulation true-ASR bundles preserve
MG nucleotide frequencies explicitly, so reimport does not estimate them from
the simulated tips. Select `--substitution_posterior marginal` when evaluating
exported true ancestral states; joint recomputes ancestral uncertainty and
requires a fitted likelihood model/rate distribution. For joint analysis of
simulated sequences, refit the simulated tip alignment with IQ-TREE.

The MG/GY frequency distinction follows [IQ-TREE model documentation](https://iqtree.github.io/doc/Substitution-Models#codon-models)
and its [codon implementation](https://github.com/iqtree/iqtree2/blob/master/model/modelcodon.cpp).

## Expectations and downstream outputs

For `expectation_method=codon_model`, joint mode uses

`sum_c P(c | D) P(parent=a | D,c) exp(Q * rate_c * fitted_length)[a,d]`.

This is a fitted conditional endpoint prediction with the same S/N
classification as the observed joint. It retains the parent state in the
transition calculation, rather than multiplying it by an already mixed child
marginal. Fitted lengths are used; diagnostic lengths rescaled from observed
counts do not replace them. The endpoint engine shares nonnegative transition
powers across lengths/categories (uniformization), and uses SciPy `expm` when
the bounded power cache is insufficient. `--expected_state_backend` controls
the legacy marginal route.

Production model expectations share the joint sparse reducer. Legacy
`get_exp_state` and the legacy fused parent/child-marginal helper reject joint
mode: a child marginal cannot represent its prediction. Tests independently
materialize the full pair projection and compare every reduced category.

The observed tensors feed search, site summaries, scan and VESM extraction.
`event_pp` in joint mode is an endpoint-pair probability conditional on the
parsed model and data. Existing thresholds remove events without renormalizing.
In search, `zero_sub_mass` filtering uses joint mass before thresholds. Filtering
invalidates cached tensors/projections and rebuilds them on the retained axis.

`urn` consumes the new observed weights through its existing fitted count null.
Neither urn nor the model prediction is a complete ASR-pipeline bootstrap.
Thresholding posterior events is data-dependent selection; the analytical
prediction does not calibrate that selection. Scan exposure and scan P-value
calibration are unchanged.

Higher-order candidate selection is shared by both estimators; see the
[higher-order search guide](HIGHER_ORDER_SEARCH.md) for foreground handling,
cutoff priority, and the candidate cap.

## Memory and time

For N nodes, C categories, K states and block size B, the principal inference
workspace is O(N C B K). Parent-to-child marginals use matrix multiplication;
they do not need a site × from × to array. The transition cache is capped at
approximately 32 MiB, with an additional power cache of at most 8 MiB for the
supported codon/3Di state spaces. No ancestral scenarios are sampled.

Search with codon-model expectations retains only the requested observed
projections and branch/site counts when individual events are unnecessary.
This compact path supports `any2any`, `spe2any`, `any2spe` and derived statistics
that use these projections. The default branch table is supported by retaining
only each site's maximum-probability event for its substitution string.

With `--b no` and no auxiliary AA stream, observed and expected projections are
computed directly from factored endpoints: S visits only distinct codons in the
same synonymous group; N sums transitions between different recoding groups.
The S reduction has a Cython implementation and an equivalent NumPy path when
the extension is unavailable. Both sum nonnegative terms, avoiding subtraction
of near-unit unchanged probabilities and preserving tiny changes. No posterior
probability is discarded for the optimization.

For arity=2, the engine can go one step further: accumulate branch-pair Gram
matrices for each site block and discard the block's projections. This avoids
all-site projection CSR storage, its sorting/index buffers, and temporary disk
payloads. Branch/site counts and optional branch substitution strings are still
available. The pair matrices, block features, counts and temporary product are
estimated before allocation; this route is used only within a 64 MiB workspace
budget. This budget does not include input state arrays or pruning workspace.
Large trees exceeding it retain the projection route, avoiding an uncontrolled
quadratic allocation. Higher arities, site-filter reports and clade permutations
also retain projections because their later consumers can need them.

Near-marginal runtime is currently limited to the measured arity=2 search
without branch or site tables. Disabling site tables alone is insufficient:
`--b yes` needs individual event maxima for branch substitution strings, and
higher arities retain larger projections. PEPC measurements found joint about
1.5–2.2 times slower for exhaustive arity=3, 3.5–4.1 times slower for targeted
arity=4, and 3.7 times slower for arity=2 with the branch table enabled.
`--threads` alone does not parallelize the projected joint reducer;
`--blas_threads` can accelerate its matrix products. See the
[arity/CPU/output benchmark](../reports/endpoint_scaling_20260910/README.md)
for the workload, timing ranges, numerical checks and platform limitations.

Ordinary unrecoded joint search reads tip observations and the state-file header,
skipping the internal probability rows that pruning will overwrite. Existing
model/input checks and tip ambiguity handling still apply. Other recoding and
site/scan paths retain their existing loading contracts.

Once the final expected reducer is consumed, both the active reducer and its
endpoint-cache reference are released when the analysis declares there will be
no further reuse. Clade permutations retain their reusable cache.

Transition powers use `R = I + Q/mu`, where `mu = max(-diag(Q))`. The engine
forms `exp(-mu*t) sum_n (mu*t)^n/n! R^n`. Reusing the powers avoids a matrix
exponential per length/category, while nonnegative terms preserve rare
transition probabilities. The remaining Poisson tail is checked against the
smallest positive partial matrix entry. Very large times or insufficient cache
capacity use `expm`; no branch length or posterior probability is truncated.

Consumers needing full events or additional summaries retain the full sparse
representation: sites/scan, `spe2spe`, CS/CBS tables, positive `min_sub_pp`, urn,
P-values, long-tail calibration, epistasis, ASRV diagnostics, and restricted ASRV
training branches. Observed and predictive events are accumulated directly
in the requested S/N/AA event spaces using native accumulation.
Full sparse event storage is retained for these consumers; no event probabilities
are dropped to obtain this speedup. Temporary space remains bounded by the site
block size and event axes. Dense Gram products use NumPy, sharing its BLAS
runtime with inference instead of potentially initializing a second runtime
through SciPy. Projected predictive N summaries cache the transition-to-group
contraction before applying site likelihoods. This cache retains at most 32 MiB
of array payload (including referenced transition matrices) and 1,024 entries;
its transition arrays are never modified during inference.

CSR outputs are spooled to temporary files and allocated once. Temporary disk
must accommodate those payloads. Final projections (or full sparse outputs),
ordinary state arrays, and downstream reduction buffers still occupy RAM.
`*_endpoint_model.json` records `observed_storage` (`pairwise`, `projections`, or `full_events`) and
`direct_projection` for each fitted model so the selected route can be inspected.

Optimization comparison against a saved pre-change source tree:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python .github/scripts/benchmark_endpoint_optimization.py \
  --baseline-root /tmp/csubst-endpoint-before \
  --workdir /tmp/csubst-endpoint-optimization --repeats 3 \
  --result endpoint_optimization.json
```

Both source trees must have compatible built extensions. This benchmark
alternates old marginal, pre-optimization joint, and optimized joint in isolated
processes and captures the unrounded CB table as well as the normal TSV. Counts
and annotations are checked directly; ratio differences are checked against
the numerical differences in their already-validated numerator/denominator.
Ratios with a very small denominator can amplify floating-point summation
differences, including changes in the final displayed digit. Add
`--branch-table` to include the default branch substitution strings.

Reproduce isolated-process measurements (one warmup, three measured runs):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python .github/scripts/benchmark_endpoints.py \
  --workdir /tmp/csubst-endpoint-benchmark --result endpoint_benchmark.json --repeats 3
```

CLI comparisons use supplied PGK/PEPC files, one CPU/BLAS thread and no long-tail
calibration. They include input reading, events, expectations, CB reduction and
output, but not an IQ-TREE refit. Synthetic kernel comparisons give the old
method precomputed accurate marginals and the new method tip likelihoods.
Fixture generation is excluded; the added inference work is included. Simulated
endpoint truth supports Brier-score comparisons; exhaustive enumeration tests
establish numerical accuracy. Neither establishes omegaC false-positive rates
or biological calibration.

For a before/after joint comparison through PEPC arity 6, use the scaling
harness with `--baseline-mode joint` and a saved pre-change source tree:

```bash
python .github/scripts/benchmark_endpoint_scaling.py \
  --baseline-root /tmp/csubst-endpoint-before --baseline-mode joint \
  --scenarios heuristic6 heuristic6_b --cpus 1 4 --blas 1 --repeats 3 \
  --workdir /tmp/csubst-endpoint-scaling --result endpoint_scaling.json
```

The JSON labels the saved implementation `baseline_joint` and the current one
`joint`. Unrounded branch and CB tables are retained for output comparison.
For supported projected searches, branch tables accumulate classified observed
events directly and use smaller projections for expectations. Rate categories
are mixed before selecting the highest-probability event at each branch/site.
The native event kernel traverses contiguous sites for each codon pair. Pruning
messages are retained in the forward-posterior workspace until consumed,
avoiding a second transition multiplication without another tree-sized buffer.

Compare both datasets and output modes (default/statistics including `spe2spe`, site tables,
recoding, arity 6, and site block sizes 8/64/256):

```bash
python .github/scripts/benchmark_endpoint_conditions.py \
  --baseline-root /tmp/csubst-endpoint-before \
  --workdir /tmp/csubst-endpoint-conditions --result endpoint_conditions.json
```

Both trees need compatible built extensions. The harness alternates versions,
uses one warmup and three measurements per condition, and verifies all captured
unrounded output tables and branch/cutoff decisions. `--profile` instead runs
one diagnostic baseline profile per condition; it is separate from timing runs.

## Remaining statistical scope

CB/CS/CBS still multiply per-branch scores. Exact single-edge joints do not make
that product a joint posterior across branches. Endpoints also do not count
multiple hits or reversions. Whole-tree posterior sampling and stochastic
mapping are separate extensions, not enabled by this option.

Related methods: [SubRecon](https://doi.org/10.1093/bioinformatics/bty101) for
adjacent-node joint probabilities and [Minin & Suchard](https://doi.org/10.1007/s00285-007-0120-8)
for labeled CTMC transition counts.

Combination tables report per-slot observation coverage as `S_eligible_count_1`,
`N_eligible_count_1`, etc., using the corresponding `branch_id_1` (and subsequent
slots). A branch subtotal with no eligible sites is `NA`; an observed zero remains
zero. MG simulation requires `--simulate_eq_freq auto` or `iqtree`, preserving the
fitted nucleotide frequencies; an explicit `alignment` override is rejected.

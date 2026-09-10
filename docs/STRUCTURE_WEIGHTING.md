# Exploratory structure weighting

The legacy `epistasis_*` option names now implement **exploratory conditional
count prediction**, not a test or a mechanistic correction for biological
epistasis. They remain off by default. Positive selected beta is not evidence
of epistasis. The former response-weighted branch context has been removed.

## Supported input contract

Active fixed or automatic beta requires `--expectation_method urn --asrv sn`,
independently obtained branch context, and site features supplied through
`--epistasis_degree_file` or `--epistasis_pdb`. Other ASRV modes, custom ASRV
training/concentration, `--asrv_report`, and `--calc_omega_pvalue yes` are rejected.
Use the automatic `csubst_epistasis.json` report for fold-specific diagnostics.
No new 3Di model expectation or calibrated omega P value is provided.

```bash
csubst search --alignment_file alignment.fa --rooted_tree_file tree.nwk \
  --foreground foreground.tsv --expectation_method urn --asrv sn \
  --epistasis_beta auto --epistasis_apply_to N \
  --epistasis_degree_file site_features.tsv \
  --epistasis_context_file independent_context.tsv \
  --epistasis_context_source "Independent experiment accession and preprocessing description" \
  --epistasis_cv_clades 5
```

The context TSV has exactly `branch_key, context_1` columns (tab-separated),
or `branch_key, context_1, context_2` when the resolved site feature mode has
two dimensions. `branch_key` is a JSON array of sorted descendant taxon names
for the corresponding rooted incoming branch: `["taxon_A"]` for a terminal
branch or `["taxon_A","taxon_B"]` for their ancestral branch. When writing CSV
with a library, let it escape embedded JSON quotes. Include every non-root
branch exactly once; roots, unknown clades, duplicates, nonfinite values, and
wrong dimensions are rejected. Tip names must be unique. The context columns
match the resolved site-feature dimensions in order (single selected metric,
or degree then proximity for paired features). Context is used as supplied;
there is no response-based scaling or reconstruction.

The provenance string and input hash are recorded. They are declarations,
not an automatic proof of independence. Do not construct context from the
scored branch's substitutions, from selected convergence hits, or from a
structure annotated using those hits. Parent ASR states reconstructed using
all tips are not automatically independent either. Supplying synonymous
substitutions as context for nonsynonymous counts does not establish independence.
Use independent measurements or genuinely pre-branch information. The simulator
below uses oracle pre-branch states that are normally unavailable in real data.

## Folds and predictions

The rooted topology is cut into monophyletic blocks. Starting from the root's
children, the largest splittable clade is split until the requested block
count is reached; descendant taxa break ties. Polytomies can yield more blocks
than requested, and a tree with fewer tips can yield fewer blocks. Branch row
order, numerical labels, and internal names do not determine the folds.
Ancestral connectors above the blocks are prediction-only buffers: their
counts never enter any fit or outer score. Their predictions use the terminal
blocks, and their diagnostic rows have `scored=false`.

Each outer block is excluded from both site-prior estimation and parameter
selection. Inner leave-one-clade-out CV selects beta, ASRV per-site alpha
and/or clip from the remaining blocks. In every fold, prior site mass is the
sum of training counts plus alpha, masked and normalized for the prediction
branch. Zero-event branches remain in the fold; all-missing rows are excluded
using the supplied mask. Thus altering a held-out block's response cannot
alter its own predictions or parameters, conditional on features, context,
tree, masks and folds. It can change predictions for other outer blocks.

Inner scores are the equal-clade average of log score per event; a zero-event
clade contributes zero. Beta candidates are 0 to 3 in steps of 0.1. Ties within
1e-12 prefer smaller beta, then smaller alpha, then smaller clip. Fixed beta
must be nonnegative. `--epistasis_clip auto` now selects a clip by inner CV,
not a quantile of all branches. Joint-auto alpha and clip grids retain their
CLI settings; fixed clip stays fixed. The beta=0 comparator selects its own
alpha using the same inner folds. Outer scores are not reused for tuning.
All-undefined inner scores cause an explicit error; use a positive alpha or
more training data. Zero context selects beta=0.

All resulting per-branch probabilities are used directly for the selected
N and/or S channel's urn expectations and categories. A full-data ASRV prior
is not reapplied after CV. Consequently `auto` selecting beta=0 still uses
cross-fitted priors, whereas `off` or an explicitly fixed zero disables the
entire feature and retains ordinary ASRV. Compare the outer matched beta=0
scores in the JSON report when isolating the effect of structure weighting.
Reported scalar beta/alpha/clip in `cb_stats` are branch means, not a single
full-data refit; JSON contains actual fold values and grids' boundary flags.

`--epistasis_beta_partition branch_depth` uses topology/branch lengths to
partition depths; equal depths stay together, so the requested bin count is
an upper bound. Each bin needs at least three nonempty clade blocks for tuning
or two when all parameters are fixed. Insufficient blocks fail explicitly.
There is no fallback to random rows or to response-derived context.

This removes direct response reuse at the count layer. It does not make
posteriors reconstructed from all tips independent, eliminate phylogenetic
dependence across the frontier, or calibrate a downstream hypothesis test.
The supplied masks and structural features are conditioned on. A full pipeline
validation must repeat ASR, data-adaptive feature construction and selection
under the intended holdout procedure. Re-fitting on all branches and reporting
these CV scores as that model's independent performance is not supported.

## Independent validation

```bash
python tools/validate_epistasis.py --replicates 1000 --seed 20260910 \
  --scenarios iid,binary_null,binary_epistatic,codon_null,codon_epistatic,codon_convergence,codon_epistatic_convergence \
  --output validation.json
```

The IID experiment draws one uniform event per branch and independent context.
The other experiments evolve a sequence by a Gillespie CTMC with symmetric
single-site mutation proposals and a fitness containing single-site fields
and residue-pair interactions. Rate is mutation rate times exp(fitness change/2).
Nulls set pair couplings to zero while retaining site fields and heterogeneous
mutation rates. Codon experiments use all 61 sense codons, permit only single
nucleotide moves and record N/S histories separately. Root sequences are
uniform, explicitly not assumed to be at mutation-selection equilibrium.
Convergence scenarios independently change fields on two preselected clades.
Site features are fixed graph degrees specified before evolution. No realized
convergence label enters their construction. Context is the mean phenotype
of the actual parent sequence, fixed before the branch evolves.

Tests compare a small binary process with its exact generator, detailed balance,
and matrix exponential, and check codon moves and synonymous rates. This is a
controlled history-level simulator, not a full biological fitness model or an
ASR validation. It need not produce a positive beta for true epistasis: this
low-dimensional predictor may not capture the generating interaction.
Reported beta frequencies are selection frequencies, not false-positive rates.
Outer log-score differences and their Monte Carlo standard errors describe
prediction only. Omega P/q-value calibration awaits joint null generation and
full refitting (scientific-review IDs 2 and 4).

`tools/evaluate_epistasis_simulation.py` remains an artificial feature-label
stress test. Its weighted modes now require explicit independent context and
use ASRV sn. It must not be cited as an independent epistatic simulator.

## Residue-state-specific pair oracle

`csubst/pair_epistasis.py` is a separate research API for two interacting sites
on two disjoint branches with known parent states and a known fitness landscape.
It evolves both sites jointly, computes exact endpoint-count distributions for
all nine any/spe/dif categories, and compares them with an independent-site
baseline matched to true marginal composition and mean per-site rates.

`tools/validate_pair_epistasis.py` checks this oracle using an independently
coded Gillespie simulator. The [complete validation report](../reports/pair_epistasis_oracle_20260910/README.md)
includes the fixed 27-cell grid and its limits. This is the known-parameter
proof-of-concept stage: it does not estimate interactions from observed
convergence, handle ASR uncertainty, or modify production omega/P-values.

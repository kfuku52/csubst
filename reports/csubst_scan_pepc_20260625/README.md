# csubst scan PEPC report

Date: 2026-06-26

This report records a smoke/example analysis for the new `csubst scan` command on the bundled PEPC dataset, and compares the intended scope of `csubst scan` with PCOC and ESL-PSC Toolkit.

## Commands

The bundled PEPC foreground file encodes all C4 taxa as a single lineage value. This is correct for ordinary foreground/background analysis, but `csubst scan --scan_min_support 2` asks how many foreground units share a substitution; therefore the bundled file yields one foreground unit and no candidate can pass support >= 2.

```bash
python csubst/csubst scan \
  --alignment_file csubst/dataset/PEPC.alignment.fa \
  --rooted_tree_file csubst/dataset/PEPC.tree.nwk \
  --foreground csubst/dataset/PEPC.foreground.txt \
  --iqtree_treefile csubst/dataset/PEPC.alignment.fa.treefile \
  --iqtree_state csubst/dataset/PEPC.alignment.fa.state \
  --iqtree_rate csubst/dataset/PEPC.alignment.fa.rate \
  --iqtree_iqtree csubst/dataset/PEPC.alignment.fa.iqtree \
  --iqtree_log csubst/dataset/PEPC.alignment.fa.log \
  --scan_match any2spe \
  --scan_min_event_pp 0.5 \
  --scan_min_support 2 \
  --scan_pvalue_calibration none \
  --float_digit 8 \
  --threads 1 \
  --outdir reports/csubst_scan_pepc_20260625/default
```

For recurrent substitutions among the 12 C4 leaves, I used `PEPC.foreground.independent.txt`, which assigns each C4 leaf to its own foreground unit:

```bash
python csubst/csubst scan \
  --alignment_file csubst/dataset/PEPC.alignment.fa \
  --rooted_tree_file csubst/dataset/PEPC.tree.nwk \
  --foreground reports/csubst_scan_pepc_20260625/PEPC.foreground.independent.txt \
  --iqtree_treefile csubst/dataset/PEPC.alignment.fa.treefile \
  --iqtree_state csubst/dataset/PEPC.alignment.fa.state \
  --iqtree_rate csubst/dataset/PEPC.alignment.fa.rate \
  --iqtree_iqtree csubst/dataset/PEPC.alignment.fa.iqtree \
  --iqtree_log csubst/dataset/PEPC.alignment.fa.log \
  --scan_match any2spe \
  --scan_min_event_pp 0.5 \
  --scan_min_support 2 \
  --scan_pvalue_calibration none \
  --float_digit 8 \
  --threads 1 \
  --outdir reports/csubst_scan_pepc_20260625/independent_any2spe
```

The same independent-foreground run with all nine pattern classes:

```bash
python csubst/csubst scan \
  --alignment_file csubst/dataset/PEPC.alignment.fa \
  --rooted_tree_file csubst/dataset/PEPC.tree.nwk \
  --foreground reports/csubst_scan_pepc_20260625/PEPC.foreground.independent.txt \
  --iqtree_treefile csubst/dataset/PEPC.alignment.fa.treefile \
  --iqtree_state csubst/dataset/PEPC.alignment.fa.state \
  --iqtree_rate csubst/dataset/PEPC.alignment.fa.rate \
  --iqtree_iqtree csubst/dataset/PEPC.alignment.fa.iqtree \
  --iqtree_log csubst/dataset/PEPC.alignment.fa.log \
  --scan_match all \
  --scan_min_event_pp 0.5 \
  --scan_min_support 2 \
  --scan_pvalue_calibration none \
  --float_digit 8 \
  --threads 1 \
  --outdir reports/csubst_scan_pepc_20260625/independent_all
```

## Output Overview

| Run | Foreground units | `--scan_match` | Output rows | Candidate `scan_id`s | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| `default` | 1 | `any2spe` | 0 | 0 | Bundled foreground has one lineage value, so support >= 2 is impossible. |
| `independent_any2spe` | 12 | `any2spe` | 86 | 86 | One foreground row per candidate. |
| `independent_all` | 12 | all 9 classes | 716 | 716 | One foreground row per candidate. |

`independent_all` candidate rows by match class:

| Match | Rows |
| --- | ---: |
| `any2any` | 137 |
| `spe2any` | 120 |
| `any2dif` | 103 |
| `any2spe` | 86 |
| `spe2spe` | 82 |
| `dif2any` | 61 |
| `spe2dif` | 61 |
| `dif2dif` | 56 |
| `dif2spe` | 10 |

## Top PEPC Candidates

Top foreground rows from `independent_any2spe`, sorted by support count, slow-site rank (`site_rate_quantile` ascending), then rate-enrichment P value:

| Change | Alignment codon site | FG units | FG fraction | FG events | FG exposure | Other events | Other exposure | P | q | Site-rate quantile | Background AA conservation | Supporting units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 778S | 778 | 7 | 0.5833 | 6.6758 | 0.0940 | 2.3145 | 0.6643 | 9.79e-06 | 0.000842 | 0.8075 | 1.0000 | 1,2,4,5,6,7,11 |
| 625I | 625 | 5 | 0.4167 | 4.9857 | 0.2365 | 1.0228 | 1.2022 | 0.000165 | 0.004200 | 0.6538 | 1.0000 | 1,4,5,6,7 |
| 663N | 663 | 5 | 0.4167 | 5.0912 | 0.0415 | 7.9369 | 0.2304 | 0.0187 | 0.0596 | 0.6914 | 0.6792 | 1,4,5,6,7 |
| 759A | 759 | 5 | 0.4167 | 5.2137 | 0.1130 | 2.0083 | 0.6960 | 0.000194 | 0.004200 | 0.7887 | 0.9828 | 1,2,4,6,11 |
| 538T | 538 | 4 | 0.3333 | 3.9687 | 0.0566 | 0.0311 | 0.2710 | 0.000114 | 0.004200 | 0.5649 | 0.9811 | 4,5,6,7 |
| 571N | 571 | 4 | 0.3333 | 3.9619 | 0.0189 | 0.0383 | 0.0539 | 0.000671 | 0.006400 | 0.5994 | 0.6792 | 1,3,4,6 |
| 577T | 577 | 4 | 0.3333 | 3.9798 | 0.0394 | 2.3767 | 0.2901 | 0.001300 | 0.010900 | 0.6056 | 0.6604 | 4,5,6,7 |
| 628K | 628 | 4 | 0.3333 | 3.9944 | 0.1360 | 4.1110 | 0.5554 | 0.0297 | 0.0724 | 0.6569 | 0.3019 | 4,6,9,11 |
| 8V | 8 | 3 | 0.2500 | 3.1785 | 0.0378 | 3.6621 | 0.1820 | 0.0385 | 0.0828 | 0.0408 | 0.4828 | 1,2,7 |
| 115L | 115 | 3 | 0.2500 | 2.9452 | 0.1432 | 0.0614 | 0.8568 | 0.000489 | 0.006400 | 0.1485 | 0.9091 | 2,8,9 |

The very slow-site candidates in this table are especially interesting under the working hypothesis that convergent substitutions at slowly evolving/conserved sites are more likely to be phenotype-relevant than substitutions at fast sites. For example, 8V and 115L have much lower site-rate quantiles than 778S, but lower foreground support.

Top foreground rows from `independent_all` show how relaxed pattern classes cluster around the same positions:

| Change | Alignment codon site | Match | FG units | P | q | Site-rate quantile | Background AA conservation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 571:any2dif | 571 | `any2dif` | 7 | 0.001500 | 0.026900 | 0.5994 | 0.6792 |
| 571:dif2dif | 571 | `dif2dif` | 7 | 0.001900 | 0.033200 | 0.5994 | 0.6792 |
| 571:dif2any | 571 | `dif2any` | 7 | 0.041600 | 0.108300 | 0.5994 | 0.6792 |
| 571:any2any | 571 | `any2any` | 7 | 0.047900 | 0.112800 | 0.5994 | 0.6792 |
| A778S | 778 | `spe2spe` | 7 | 9.77e-06 | 0.001400 | 0.8075 | 1.0000 |
| 778S | 778 | `any2spe` | 7 | 9.79e-06 | 0.001400 | 0.8075 | 1.0000 |
| A778 | 778 | `spe2any` | 7 | 9.80e-06 | 0.001400 | 0.8075 | 1.0000 |
| 778:any2any | 778 | `any2any` | 7 | 0.000151 | 0.007700 | 0.8075 | 1.0000 |
| 582:any2dif | 582 | `any2dif` | 6 | 2.12e-06 | 0.000758 | 0.6109 | 0.7925 |
| 582:dif2dif | 582 | `dif2dif` | 6 | 2.12e-06 | 0.000758 | 0.6109 | 0.7925 |

## Comparison With Existing Methods

### PCOC

PCOC, Profile Change with One Change, is a likelihood model for detecting sites associated with repeated phenotype transitions. The paper defines PCOC as a combination of amino-acid profile changes and a OneChange component, and compares a convergent model to a null model at each site. It requires a rooted tree and an amino-acid alignment. Sources: [Rey et al. 2018, MBE](https://academic.oup.com/mbe/article/35/9/2296/5050468), [PCOC GitHub README](https://github.com/CarineRey/pcoc).

| Axis | `csubst scan` | PCOC |
| --- | --- | --- |
| Primary question | Which explicit nonsynonymous-state substitutions recur on foreground branches? | Which sites fit a phenotype-transition amino-acid profile shift model? |
| State space | Codon-derived amino-acid or recoded nonsynonymous states; supports `--nonsyn_recode`. | Amino-acid alignment and profile categories. |
| Event localization | Reports branch/site/from/to posterior event mass. | Imposes at least one change on phenotype-transition branches, but output is mainly site-model evidence. |
| Foreground definition | Reads CSUBST foreground lineages; reports foreground candidate rows. | Requires convergent scenario/node labels for phenotype transitions. |
| Branch length use | Default P-value exposure uses CSUBST N-rescaled branch length with state- and Q-weighted opportunity; raw and S+N lengths are optional. | Branch lengths are part of the likelihood model. |
| Exact AA convergence | Directly available with `spe2spe` or `any2spe`. | Not limited to identical amino acids; profile shifts can capture chemically related residues. |
| Screening/statistics | One-sided Poisson LRT enrichment from event counts and state/Q-weighted exposure; useful for ranking, not a full generative convergent-profile model. | Full site likelihood comparison under PCOC/PC/OC models. |
| Best use | Transparent foreground candidate substitution table, branch auditing, recoding/3Di-friendly scans. | Model-based detection of sites under repeated phenotype-transition scenarios, especially when exact identical substitutions are too strict. |

### ESL-PSC Toolkit

ESL-PSC uses paired species contrast and sparse predictive modeling to identify genes/sites associated with convergent traits. The 2026 Toolkit adds a GUI/CLI environment, interactive pair selection, command preview, live execution, ranked gene/site exploration, a complementary substitution-counting method, and continuous-trait analysis. Sources: [Allard and Kumar 2026 arXiv Toolkit preprint](https://arxiv.org/abs/2605.27677), [Allard and Kumar 2025 Nature Communications](https://www.nature.com/articles/s41467-025-58428-8), [ESL-PSC GitHub README](https://github.com/kumarlabgit/ESL-PSC).

| Axis | `csubst scan` | ESL-PSC / ESL-PSC Toolkit |
| --- | --- | --- |
| Primary question | Which substitutions occurred repeatedly on specified phylogenetic branches? | Which genes/sites predict convergent trait states across paired species contrasts? |
| Scale | Single-gene or codon alignment-centered, but can be batched gene-wise externally. | Designed for multi-gene/proteome-scale feature selection. |
| Model | Ancestral-state posterior events plus branch-exposure enrichment. | Sparse group-lasso predictive model; Nature Communications paper notes ESL does not estimate substitution rates, among-site rates, or branch lengths. |
| Phylogeny use | Explicit rooted tree, branch IDs, and foreground branches. | Phylogeny-informed pair/contrast design masks background convergence. |
| Output | Candidate substitution rows with from/to states, supporting foreground units, P/q values, site rate, and conservation. | Gene ranks, site/residue weights, species prediction scores, GUI summaries. |
| Interpretability | Very high at the branch-event level. | High at gene/site feature level, but not an ancestral-event table. |
| Best use | Auditing mechanistic candidate substitutions after defining foreground branches. | Discovering sparse genetic predictors of convergent traits across many loci. |

### Practical Positioning

`csubst scan` is not a replacement for PCOC or ESL-PSC. It fills a different gap:

- It is more explicit than PCOC about individual inferred substitutions, because it reports from-state, to-state, site, branch support, and foreground-vs-control event counts.
- It is less model-complete than PCOC for site likelihoods, because the current P value is a screening enrichment statistic rather than a full convergent-profile likelihood.
- It is more branch-history aware than ESL-PSC, because it depends on ancestral states and branch lengths.
- It is much narrower than ESL-PSC for proteome-scale prediction, because it does not fit a sparse model across many genes or predict phenotypes.
- It is complementary to both: use ESL-PSC for proteome-scale candidate gene/site discovery, PCOC for amino-acid profile-shift site evidence, and `csubst scan` to audit exactly which substitutions occurred in foreground branches and how they compare with control branches.

## Implemented Fixes During This Work

- Added `csubst scan` as an independent subcommand.
- Added all nine `--scan_match` classes: `any2any`, `any2spe`, `any2dif`, `spe2any`, `spe2spe`, `spe2dif`, `dif2any`, `dif2spe`, `dif2dif`.
- Foreground candidate rows are reported directly; marginal/combined target rows were intentionally removed because the scan is clade-centered.
- Added `--scan_rate_exposure q_weighted` as the default; it applies state-aware opportunity and weights each allowed parent-to-derived transition by the nonsynonymous instantaneous rate matrix. With `--scan_rate_length n_rescaled`, the Q weight is normalized to the conditional nonsynonymous transition probability so nonsynonymous branch length is not double-weighted by total outgoing rate.
- Added `--scan_rate_event_mode called|posterior_sum`; default `posterior_sum` keeps candidate discovery/support thresholded but uses all matching posterior event mass for rate P values.
- Added `--scan_other_scope all|sister` so the control branch set can be made explicit for foreground-vs-control comparisons.
- `state_aware` remains available when a model-rate-weighted exposure is not desired.
- Added stratified q-value columns: `q_rate_enrichment_by_trait` and `q_rate_enrichment_by_trait_match`.
- Added `--scan_pvalue_calibration none|candidate_fixed|full_scan`; `full_scan` is the default and reports maxT-style empirical P values from foreground-clade permutations.
- Added `--scan_n_permutations`, `--scan_permutation_seed`, `--scan_permutation_sample_original`, and `--scan_permutation_retry_sample_original`.
- Added `scan_permutation_failure_reasons` so failed empirical-calibration iterations are no longer silent.
- Added `--scan_rate_length raw|sn_rescaled|n_rescaled`; default is `n_rescaled`, matching the nonsynonymous-state focus of `csubst scan`.
- Added IQ-TREE site rates, site-rate quantiles, amino-acid conservation, and nonsyn-state conservation.
- Added empty-output headers so a no-candidate scan still writes a machine-readable `csubst_scan.tsv`.
- Updated README command list and CLI tests.

## Additional Features Or Fixes Still Worth Adding

1. Branch-level companion table: write one row per `scan_id` x branch x target class with event posterior, parent-state opportunity, chosen exposure, raw length, SN length, and N length. This would make reversion/secondary-substitution auditing much easier.
2. Calibration: add parametric-simulation calibration for `p_rate_enrichment`; clade-permutation calibration is now available.
3. PCOC bridge: export candidate transition branches/sites in a PCOC-friendly scenario format, and import PCOC site scores into the scan table for side-by-side ranking.
4. ESL-PSC bridge: export per-site/per-state binary features or summary scores from `csubst scan` so they can be used as interpretable features alongside ESL-PSC gene/site weights.
5. Precision policy: recommend or auto-use higher precision for P/q columns, because the global `--float_digit 4` default can round small P values to `0.0000`.
6. Multi-gene summary: add a small collector to aggregate `csubst_scan.tsv` outputs across genes, because ESL-PSC's strongest advantage is proteome-scale ranking.

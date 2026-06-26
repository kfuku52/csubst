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

For recurrent substitutions among the C4 lineages, I used `PEPC.foreground.independent.txt`, which keeps most C4 leaves as independent foreground units but groups the Setaria/Zea/Sorghum PEPC clade as one foreground unit so its stem branch is analyzed as the foreground branch:

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
  --scan_site_plot no \
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
  --scan_site_plot no \
  --float_digit 8 \
  --threads 1 \
  --outdir reports/csubst_scan_pepc_20260625/independent_all
```

## Output Overview

| Run | Foreground units | `--scan_match` | Output rows | Candidate `scan_id`s | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| `default` | 1 | `any2spe` | 0 | 0 | Bundled foreground has one lineage value, so support >= 2 is impossible. |
| `independent_any2spe` | 10 | `any2spe` | 78 | 78 | One foreground row per candidate. |
| `independent_all` | 10 | all 9 classes | 526 | 526 | One foreground row per candidate. |

`independent_all` candidate rows by match class:

| Match | Rows |
| --- | ---: |
| `any2any` | 104 |
| `spe2any` | 96 |
| `any2spe` | 78 |
| `spe2spe` | 77 |
| `any2dif` | 63 |
| `spe2dif` | 40 |
| `dif2any` | 33 |
| `dif2dif` | 29 |
| `dif2spe` | 6 |

## Top PEPC Candidates

Top foreground rows from `independent_any2spe`, sorted by support count, slow-site rank (`site_rate_quantile` ascending), then rate-enrichment P value:

| Change | Alignment codon site | FG units | FG fraction | FG events | FG exposure | Other events | Other exposure | P | q | Site-rate quantile | Background AA conservation | Supporting units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 778S | 778 | 8 | 0.8000 | 7.2397 | 0.0979 | 1.7506 | 0.6605 | 2e-06 | 0.0001558 | 0.8075 | 1.0000 | 1,2,4,5,6,7,8,10 |
| 663N | 663 | 6 | 0.6000 | 6.0573 | 0.0443 | 6.9709 | 0.2276 | 0.00551 | 0.02287 | 0.6914 | 0.6792 | 1,4,5,6,7,8 |
| 759A | 759 | 6 | 0.6000 | 5.7717 | 0.1166 | 1.4503 | 0.6924 | 3.98e-05 | 0.0009985 | 0.7887 | 0.9828 | 1,2,4,6,8,10 |
| 625I | 625 | 5 | 0.5000 | 4.9857 | 0.1881 | 1.0228 | 1.2507 | 5.12e-05 | 0.0009985 | 0.6538 | 1.0000 | 1,4,5,6,7 |
| 538T | 538 | 4 | 0.4000 | 3.9687 | 0.0457 | 0.0311 | 0.2819 | 4.62e-05 | 0.0009985 | 0.5649 | 0.9811 | 4,5,6,7 |
| 570Q | 570 | 4 | 0.4000 | 3.9384 | 0.0991 | 3.1536 | 0.5064 | 0.00878 | 0.03112 | 0.5983 | 0.7736 | 2,4,6,8 |
| 571N | 571 | 4 | 0.4000 | 3.9619 | 0.0141 | 0.0383 | 0.0587 | 0.000192 | 0.002988 | 0.5994 | 0.6792 | 1,3,4,6 |
| 577T | 577 | 4 | 0.4000 | 3.9798 | 0.0343 | 2.3767 | 0.2952 | 0.000727 | 0.007088 | 0.6056 | 0.6604 | 4,5,6,7 |
| 729V | 729 | 4 | 0.4000 | 4.0215 | 0.1643 | 5.2512 | 1.2026 | 0.00851 | 0.03112 | 0.7573 | 0.8475 | 4,6,7,8 |
| 8V | 8 | 3 | 0.3000 | 2.9928 | 0.0336 | 3.8479 | 0.1862 | 0.0382 | 0.07087 | 0.0408 | 0.4828 | 1,2,7 |

The very slow-site candidates in this table are especially interesting under the working hypothesis that convergent substitutions at slowly evolving/conserved sites are more likely to be phenotype-relevant than substitutions at fast sites. For example, 8V has a much lower site-rate quantile than 778S, but lower foreground support.

Top foreground rows from `independent_all` show how relaxed pattern classes cluster around the same positions:

| Change | Alignment codon site | Match | FG units | P | q | Site-rate quantile | Background AA conservation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 778S | 778 | `any2spe` | 8 | 2e-06 | 0.0003527 | 0.8075 | 1.0000 |
| A778S | 778 | `spe2spe` | 8 | 2e-06 | 0.0003527 | 0.8075 | 1.0000 |
| A778 | 778 | `spe2any` | 8 | 2.01e-06 | 0.0003527 | 0.8075 | 1.0000 |
| 778:any2any | 778 | `any2any` | 8 | 6.9e-06 | 0.0009068 | 0.8075 | 1.0000 |
| 663:any2dif | 663 | `any2dif` | 7 | 0.000717 | 0.01199 | 0.6914 | 0.6792 |
| 663:dif2dif | 663 | `dif2dif` | 7 | 0.000742 | 0.01199 | 0.6914 | 0.6792 |
| 663:any2any | 663 | `any2any` | 7 | 0.00202 | 0.02091 | 0.6914 | 0.6792 |
| 663:dif2any | 663 | `dif2any` | 7 | 0.00206 | 0.02091 | 0.6914 | 0.6792 |
| 570:any2any | 570 | `any2any` | 6 | 0.00059 | 0.01111 | 0.5983 | 0.7736 |
| 570:any2dif | 570 | `any2dif` | 6 | 0.000752 | 0.01199 | 0.5983 | 0.7736 |

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

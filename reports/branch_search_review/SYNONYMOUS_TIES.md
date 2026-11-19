# Does synonymous evidence resolve OCN ties at K=4?

Using the smallest OCS/ECS to resolve an exact OCN tie did not improve recovery of the planted quartet. On 36 long alignments the ordinary index fell from 6 to 4 recoveries, and the group-refined index fell from 6 to 5. The rule increased selected ωC even with no plant. Retain vote order in the research index; there is no change to `csubst search`.

## Paired experiment

Generate the original vote-ordered proposal list **once** per method and replicate. Both selections use descending cutoff `OCNany2spe` (site PP ≥ 0.5). The baseline preserves index votes and the original seeded order for tied votes. The alternative keeps that order only after breaking equal OCN by ascending full-alignment `OCSany2spe / ECSany2spe`. Zero OCS with positive ECS is a valid zero; an undefined S rate ranks last. No rounding, tolerance ties, pseudocount or change to the final ωC estimator. For top-64 membership only the tie split by the budget needs S scores. For reported planted ranks the entire OCN group of that plant is ordered by the same rule. Labels and final ωC are used only for evaluation.

Each planted alignment has 256 independent branches, a single planted quartet and four planted sites. All planted quartets passed OCN ≥ 2 and ωC ≥ 5. An entry below counts **seeds recovering that quartet**, not unrelated threshold crossings. The 65,536-draw limit is identical in each arm. Seeds 1–24 at 8,192 sites were used in the previous proposal study; seeds 25–36 were held out from that study. Each length is evaluated separately.

| Sites | Seeds | Ordinary index: proposed | Top 64: votes → S | Group refinement: proposed | Top 64: votes → S |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2,048 | 1–12 | 12 | 12 → 12 | 12 | 12 → 12 |
| 8,192 | 1–12 | 7 | 3 → 3 | 7 | 3 → 3 |
| 8,192 | 13–24 | 4 | 1 → 0 | 8 | 1 → 1 |
| 8,192 | 25–36 | 3 | 2 → 1 | 4 | 2 → 1 |

The four quartets newly proposed by group refinement in the earlier study all had OCN = 4. At 8,192 sites the generating-null ECS is 8.808 for **every** quartet, so OCS/ECS orders this tie exactly as OCS would. The simulation imposes directional changes on N and does not plant an S deficit.

| Seed | Number tied | Planted OCS | Rank by votes | Rank using S |
| ---: | ---: | ---: | ---: | ---: |
| 18 | 380 | 10 | 102 | 253 |
| 20 | 339 | 5 | 165 | 47 |
| 22 | 346 | 9 | 175 | 193 |
| 23 | 343 | 8 | 210 | 145 |

In the same block seed 15 moved from 24 to 145. In the held-out block seed 26 moved from 19 to 235 with group refinement. A change in which plant is recovered can therefore be hidden by unchanged total recovery.

## Negative controls and fitted codon panels

With no plant, 12 replicates (seeds 25–36; 256 branches × 8,192 sites) used the same background law and paired seeds as the held-out planted runs. The expected background OCN is only 0.107 per unselected quartet. Selecting the strongest N overlaps from tens of thousands of proposals already strongly enriches ωC. The table takes the median of the 12 within-replicate medians. Counts are summed over 12 × 64 slots **per method**.

| Proposal method | Median selected ωC: votes → S | Replicates with increase | Selected groups exchanged | Threshold crossings: votes → S | Selected OCS = 0: votes → S |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary index | 41.32 → 66.11 | 12/12 | 509/768 | 768 → 768 | 0 → 0 |
| Group refinement | 41.32 → 66.11 | 12/12 | 510/768 | 768 → 768 | 0 → 0 |

OCN ≥ 2 and ωC ≥ 5 are descriptive cutoffs, not a calibrated test. The binary endpoint was already saturated before reranking. The larger ωC values after reranking are a selection effect on the denominator, even without any zero OCS. These numbers do not estimate a genome-wide false-positive rate.

On the bundled PGK tree we regenerated three strong directional, three moderate directional and three relaxation alignments, with 520 codons each, using the earlier enriched panel. IQ-TREE fitted `ECMK07_GY0K+F+R4` to the simulated tips. Exhaustive CSUBST scores on 25 independent stems gave 12,650 quartets per replicate. Across the three strong replicates there were 210 reference hits; each method retained 192 under either ranking (all available slots). Both rankings recovered the only moderate reference hit and proposed no relaxation quartet. The strong panels had **no exact OCN tie crossing rank 64**; the other panels had at most two proposals. This panel therefore checks that continuous posteriors and undersubscribed runs are handled correctly, but cannot distinguish two tie breakers when there is no decision to make. The synthetic panel has branch-invariant ECS; the present experiments do not exclude a use for S with heterogeneous ECS and actual boundary ties.

## Reproduction and data

Base `d56ffe5` plus the [research proposal implementations](GROUP_REFINEMENT.md); macOS arm64, Python 3.14.7, NumPy 2.5.3, SciPy 1.17.1, pandas 3.0.5 and IQ-TREE 3.1.4. [Scale results](synonymous_ties_scale.tsv) and [PGK results](synonymous_ties_pgk.tsv) contain paired results for each seed and method. Empty planted ranks in a planted run mean the candidate was never proposed; all planted fields are empty in `no_plant` rows. The counts of proposals, eligible plants/reference hits, hits proposed and baseline top-64 hits were cross-checked against the earlier [fixed-draw results](group_refinement_runs.tsv) for all 72 overlapping synthetic and 18 PGK rows. No elapsed-time comparison was made.

From the repository root, using a fresh output directory for analyses:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
driver=reports/branch_search_review/synonymous_ties.py
python "$driver" scale long 2048 $(seq 1 12)
python "$driver" scale long 8192 $(seq 1 36)
python "$driver" scale null 8192 $(seq 25 36)
python "$driver" pgk strong 5749 "$panel_dir"
python -m pytest -q reports/branch_search_review/test_synonymous_ties.py reports/branch_search_review/test_group_refinement.py
```

For each PGK case follow the [PGK preparation recipe](BRANCH_PAIR_INDEX.md#reproduction) with `--isolate_tips`, using the simulation seed also as the `search --random_seed` for the fit. Run the `pgk` example for `strong`, `moderate` and `relax` and seeds 5749, 5881 and 6151 on the corresponding panel directories. The benchmark independently reconstructs each candidate's cutoff OCN from the per-branch destination probabilities before reading its synonymous counts from the exhaustive table.

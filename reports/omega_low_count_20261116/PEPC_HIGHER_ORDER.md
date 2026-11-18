# Higher-order search on the bundled PEPC alignment

CSUBST 1.17.5 ran from the working tree based on `b782efd` on macOS arm64
with Python 3.12.15. Runs used the bundled 71-tip, 971-codon PEPC
alignment and rooted tree, the saved ECMK07+F+R4 IQ-TREE fit, joint
endpoint posteriors, and one CSUBST and BLAS thread. Sisters were excluded.
Pairs were enumerated exhaustively. Subsequent arities used the ordinary
candidate expansion with a cap of 10,000 retained rows.

## Default cutoffs

The entries give evaluated rows / rows meeting both `OCNany2spe >= 2`
and `omegaCany2spe >= 5`. With the bundled C4 foreground and
`--fg_exclude_wg no`, the count refers to all rows in the corresponding
output table. The sole K=6 hit in that run was entirely foreground.

| K | Codon model, no foreground | Codon model, C4 foreground | Urn, no foreground |
| ---: | ---: | ---: | ---: |
| 2 | 8,446 / 85 | 8,446 / 85 | 8,446 / 16 |
| 3 | 416 / 45 | 69 / 30 | 8 / 1 |
| 4 | 87 / 20 | 41 / 18 | — |
| 5 | 14 / 6 | 13 / 6 | — |
| 6 | 1 / 1 | 1 / 1 | — |

In both codon-model runs the K=6 combination was `1,26,33,35,40,102`.
Its observed nonsynonymous count was 2.498896562655 and the expected
nonsynonymous count was 9.535e-9. The observed synonymous count was
0 at output precision and the expected count was 1.5508e-6. The raw
`omegaCany2spe` was infinite. With symmetric alpha 0.5 applied to this
combination, the estimate was 5.997812. The alpha comparison used OCN
alone during expansion.

The K=6 branches were *Alternanthera pungens* (1), *Cyperus sanguinolentus*
(26), *Eleocharis vivipara* (33), *Fimbristylis littoralis* (35), *Flaveria
australasica* (40) and the branch leading to the *Bienertia sinuspersici* /
*Suaeda aralocaspica* clade (102, `Node49`).

## Sensitivity to candidate selection

With only `OCNany2spe >= 2` and no foreground restriction, the maximum
remained K=6 with the same combination. With OCN >= 1, both the global
and C4 searches reached K=7 (`1,26,33,35,40,102,124`, OCN 1.220351).
Branch 124 (`Node58`) leads to the *Setaria italica* / *Sorghum bicolor* /
*Zea mays* clade. Its raw omegaC was infinite and the symmetric alpha 0.5
estimate was 3.440702. At OCN >= 0.5 the C4 search reached K=8
(`1,5,26,33,35,40,102,103`, OCN 0.540787). Branches 5 and 103 are
*Amaranthus hypochondriacus* and *Suaeda eltonica*, respectively.

At OCN >= 0.5 without a foreground restriction, raising the search cap
to K=20 gave two passing K=12 combinations. Each had its entire OCN
contribution at H663N. The one evaluated K=13 combination scored 0.448621.

| K=12 branch IDs | OCN | C4 branches |
| --- | ---: | ---: |
| `1,8,26,33,35,40,54,99,102,111,112,124` | 0.794009 | 7 |
| `1,8,26,33,35,40,54,86,99,111,112,124` | 0.509643 | 6 |

The global OCN >= 0.5 search evaluated / retained 1,437 / 933 at K=8,
733 / 401 at K=9, 287 / 122 at K=10, 78 / 23 at K=11, 13 / 2 at K=12,
and 1 / 0 at K=13.

The table gives site-wise OCN contributions from `csubst sites`. The
substitutions use 1-based codon-alignment coordinates. Each column sums
to the matching search count within output rounding.

| Substitution | K=6 | K=7 | K=8 (C4) |
| --- | ---: | ---: | ---: |
| V625I | 0.967652 | 0.000000414 | 0 |
| H663N | 0.842298 | 0.827620 | 0 |
| A778S | 0.688946 | 0.392731 | 0.540787 |

For the global OCN-only thresholds of 2, 1 and 0.5, an independent check
confirmed that every candidate whose K−1 subsets all passed the OCN
cutoff was evaluated. The candidate cap did not apply. Adding a
branch multiplies each site/destination term in OCN by a probability in
[0, 1], so the observed count gives a pruning bound for this fixed ASR.
This bound applies to OCN; omegaC can increase as K increases.

## Urn compatibility

`--expectation_method urn --asrv auto --max_arity 10` resolved to `each`.
Among the 15 pairs within the K=6 combination, the default two cutoffs
retained 14 under the codon model, 2 with `each`, and 15 in a separate
pairwise `empirical` run. The latter retained 263 of the 8,446 evaluated
pairs overall. The default `each` search reached K=3 with one passing
triple (`1,26,124`, OCN 2.334371). With C4 foreground and OCN-only
expansion at OCN >= 2, `each` also recovered the same K=6, with identical
OCN and ECN 1.02852e-7. Empirical urn expectations currently support
K=2, so the K=6 urn comparison uses `each`.

The analysis uses fitted endpoint probabilities and score cutoffs on one
tree. The counts and pseudocount estimates characterize the candidates;
the run did not perform a pipeline-wide significance test.

## Reproduction

Run from the repository root. Use a new output directory for each setting.

```bash
fit=csubst/dataset/PEPC.alignment.fa
work=$(mktemp -d)
common=(--alignment_file "$fit" --rooted_tree_file csubst/dataset/PEPC.tree.nwk
        --iqtree_model ECMK07+F+R4 --iqtree_state "$fit.state" --iqtree_rate "$fit.rate"
        --iqtree_iqtree "$fit.iqtree" --iqtree_log "$fit.log" --iqtree_treefile "$fit.treefile"
        --threads 1 --blas_threads 1 --float_digit 12)
fg=(--foreground csubst/dataset/PEPC.foreground.txt --fg_exclude_wg no)
python -m csubst search "${common[@]}" --max_arity 10 --b no --random_seed 8 \
  --expectation_method codon_model --outdir "$work/codon"
python -m csubst search "${common[@]}" "${fg[@]}" --max_arity 10 --b no --random_seed 8 \
  --expectation_method codon_model --cutoff_stat OCNany2spe,0.5 --outdir "$work/c4"
python -m csubst search "${common[@]}" --max_arity 20 --b no --random_seed 8 \
  --expectation_method codon_model --cutoff_stat OCNany2spe,0.5 --outdir "$work/global"
python -m csubst sites "${common[@]}" --branch_id 1,26,33,35,40,102 \
  --cb_file "$work/codon/csubst_cb_6.tsv" --tree_site_plot no \
  --site_summary_plot no --site_state_plot no --outdir "$work/sites"
```

To reproduce these historical urn runs under the new default, use
`--expectation_method urn --asrv each` for higher arities or
`--expectation_method urn --asrv empirical --max_arity 2` for cross-fitted pairs. For the pseudocount
comparison, use the desired OCN-only cutoff and add
`--pseudocount_mode symmetric --pseudocount_alpha 0.5`.

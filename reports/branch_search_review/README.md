# Branch search and regional urn verification

The comparison uses CSUBST 1.18.6 at `1aabfbd` and the local implementation
based on that commit. The runs used macOS arm64, Python 3.12.15, NumPy 2.5.5,
SciPy 1.18.1 and IQ-TREE 3.1.4.

## Higher-order search on PEPC

The reference enumerates every combination of the nine specified C4 stems
through K=6. All three searches used the same PEPC alignment, foreground
file and fitted `ECMK07_GY0K+F+R4` parameters. A hit requires
OCNany2spe ≥ 2 and omegaCany2spe ≥ 5 at the reported arity.

| K | Reference combinations | Codon-model hits, all_cutoffs / ocn_bound / reference | Empirical urn hits, all_cutoffs / ocn_bound / reference |
|---:|---:|---:|---:|
| 2 | 36 | 33 / 33 / 33 | 20 / 20 / 20 |
| 3 | 84 | 32 / 32 / 32 | 26 / 32 / 32 |
| 4 | 126 | 18 / 18 / 18 | 17 / 18 / 18 |
| 5 | 126 | 6 / 6 / 6 | 6 / 6 / 6 |
| 6 | 84 | 1 / 1 / 1 | 1 / 1 / 1 |

With the default cap of 10,000, `ocn_bound` recovered the reference **branch IDs** at every arity for both methods.
For K=3 the codon model scored 32 combinations with `ocn_bound` and 81 with `all_cutoffs`.
For example, the empirical urn now recovers branches 1, 5 and 103 at K=3.
The result concerns these nine stems. The candidate cap and trait constraints
can affect larger searches.

Explicit `--high_order_pruning ocn_bound` and `auto` selected
the same rows and hits in both models. Across the 36 pairwise comparisons,
median parent-node distance (`dist_bl`) was 2.0053 versus 2.2774 for the
historical ordered metric (`dist_bl_legacy`). For each pair the difference
equaled the length of its first branch. Changing the distance definition
preserved OCN, OCS, ECN, ECS and ωC for every shared row at K=2–6.

Low synonymous counts matter especially at high arity. At K=6, 79 of the 84
reference combinations had infinite legacy ωC and a finite four-count ratio
under both expectations. At K=5 the count was 60 of 126. Positive OCS below
the tolerance can produce this distinction. Finite arithmetic alone does not
increase the evidence supplied by such a small synonymous count.

To reproduce, copy the bundled PEPC FASTA, tree and foreground file to a
fresh directory. Fit the default IQ-TREE model once and reuse the fit across
runs. Use the two values of `--expectation_method` (`codon_model` and `urn`)
and the two pruning modes. For the exhaustive foreground reference, set
`--high_order_pruning all_cutoffs --cutoff_stat OCNany2spe,0`, then apply the
hit thresholds to the resulting tables.

```bash
python -m csubst search --alignment_file PEPC.alignment.fa --rooted_tree_file PEPC.tree.nwk \
  --foreground PEPC.foreground.txt --exhaustive_until 1 --fg_exclude_wg no \
  --exclude_sister_pair no --max_arity 6 --output_stat any2spe --float_digit 17 \
  --high_order_pruning ocn_bound --expectation_method urn
```

## Candidate limits and alignment length

PEPC has 71 tips and 971 codon sites. Each cell below gives
`all_cutoffs` / `ocn_bound`. All runs used the default thresholds.
The nine-stem reference enumerated 84 triples and found 32 hits. Tree-wide
searches used the default independent-pair exclusions, with 139 distinct
branches represented in 8,446 pairs. Tree-wide hit counts are counts from
the indicated searches. Shared triples had identical OCN, OCS, ECN, ECS and ωC.

| PEPC search at K=3 | Parent cap | Eligible pairs | Independent triples generated | Hits |
| --- | ---: | ---: | ---: | ---: |
| Nine C4 stems, urn | 20 | 20 / 33 | 47 / 51 | 26 / 22 |
| Whole tree, codon model | 100 | 307 / 458 | 530 / 584 | 54 / 56 |
| Whole tree, codon model | 10,000 | 307 / 458 | 3,048 / 6,011 | 65 / 75 |

At cap 20, nine of the nine-stem hits were unique to `all_cutoffs` and
five were unique to `ocn_bound`. At cap 100 on the whole tree the respective
counts were nine and eleven. With the full budget, `ocn_bound` generated
more triples but needed expected counts for only 75 of them on the whole tree.
Median wall time for K=2–3 was 6.91 / 5.69 seconds and median peak RSS was
624 / 666 MiB. These are three runs per mode after one warmup, alternating
the order and reusing the fitted model (one thread, macOS arm64,
`/usr/bin/time -l`). All three K=3 tables matched their same-mode warmup.
To reproduce the targeted comparison, use the command above with
`--max_arity 3 --max_combination 20 --branch_dist no` and each pruning mode.
For the tree-wide comparison, omit the foreground, sister-pair and exhaustive
overrides, select `--expectation_method codon_model`, and set the indicated cap.

The following controlled workload isolates branch count and alignment length.
At each site, each independent terminal branch has probability 0.04 of an N
event and 0.04 of an S event, with one shared destination per channel at
that site. For uniform independent sites, the expected pair overlap is the
product of the two branch totals divided by sequence length. Smaller panels
are subsets of the same random draw (seed 20261119). The workload runs
CSUBST's subset selection and union generation with the default cap. It
measures candidate pressure, without fitting a codon model or scoring triples.

| Independent branches | Sites | Eligible pairs, all_cutoffs / ocn_bound | Generated triples, all_cutoffs / ocn_bound |
| ---: | ---: | ---: | ---: |
| 64 | 512 | 158 / 345 | 773 / 3,425 |
| 64 | 2,048 | 96 / 1,692 | 253 / 38,551 |
| 256 | 512 | 2,924 / 6,520 | 67,946 / 299,120 |
| 256 | 2,048 | 1,476 / 27,533 | 16,796 / 625,602 |

In the last case `ocn_bound` discarded 17,533 eligible pairs before generating
triples. Reproduce with `python reports/branch_search_review/pruning_pressure.py`.

The [site-intersection pilot](SITE_INTERSECTION.md) compares direct K=4
proposals with parent expansion at matched quartet-evaluation budgets.

## Conditioning the empirical urn on a prespecified region

The 24 simulated alignments cross two trees, two domain widths (20% and 40%
of 520 codons), three matched seeds (5749, 5881, 6151), and two foreground
processes. Eight terminal branches of length 0.25 receive a local increase
in the ratio of summed nonsynonymous to synonymous Q entries (0.1 to 0.9).
The directional treatment also favors a common derived amino acid at each
treated site. The nondirectional treatment preserves relative nonsynonymous
preferences. Each analysis searches all eligible branch pairs without
foreground labels. The three estimates reuse one IQ-TREE fit per alignment.

Each cell gives the whole-alignment estimate, then the estimate using the
prespecified generating domain, then the estimate with that interval shifted
by half its width. Each hit fraction has 168 focal pair-replicates
(28 pairs × two widths × three seeds).

| Tree | ECN / generating mean N under local relaxation | Local-relaxation hits | Directional hits |
|---|---:|---:|---:|
| PEPC | 0.605 / 0.957 / 0.637 | 0 / 0 / 0 | 108 / 108 / 108 |
| PGK | 0.563 / 0.888 / 0.591 | 1 / 1 / 1 | 93 / 92 / 93 |

ECN calibration uses the ratio of **summed** ECN to summed
`muNany2spe_sim` under nondirectional relaxation. This includes the chance
coincidences induced by the elevated local N rate. The separately retained
`muNany2spe_null` uses the background generator on the two branches.
All three analyses gave identical OCN, OCS and ECS row by row.
None of the other eligible pairs passed in these 24 alignments.
The domain-specific correction improves expected counts when the interval
is accurate. This panel shows little change in threshold crossing and does
not estimate a general false-positive rate. Pairs share branches and sites,
and the two domain widths reuse seeds.

Run the panel in a fresh destination with the following command.
[Replicate-level results](panel.tsv) · [Simulation and analysis script](region_panel.py)

```bash
python reports/branch_search_review/region_panel.py /tmp/csubst-region-panel --iqtree iqtree3 --jobs 3
```

## Runtime and numerical parity

Wall time is the median of three alternating runs after warm-up. Peak RSS
is the median of the three per-run maxima from macOS `/usr/bin/time -l`.
The sites and simulate runs used the bundled fits, one CSUBST thread, one
BLAS thread and native extensions. Files were written to fresh directories.
Background machine load varied between runs.

| Workload | Baseline seconds | Current seconds | Time saved | Baseline / current peak MiB |
|---|---:|---:|---:|---:|
| PGK sites, branches 23 and 51 | 6.02 | 4.60 | 24% | 430 / 424 |
| PEPC sites, branches 9 and 108 | 18.50 | 12.32 | 33% | 1,046 / 1,122 |
| PEPC simulate, file rates, 192 codons | 8.73 | 6.94 | 21% | 664 / 677 |

For sites, enable `--site_state_plot yes --tree_site_plot no` and supply
the two branch IDs with `--branch_id`. For simulate, use
`--num_simulated_site 192 --background_omega 0.2 --simulate_asrv file
--simulate_seed 101 --export_expected_omegaC yes`. Both commands used
`--iqtree_model ECMK07+F+R4` and explicit copies of the five bundled
IQ-TREE files. All old data columns in the site and state TSVs agreed
exactly in the timing comparison. The historical `max_dist_bl` values
are now reported in `max_dist_bl_legacy`. Simulated tips, ancestors,
ASR state/rate tables and tree matched
byte-for-byte. Across 8,516 pairs, the largest absolute difference in the
four Q-based expected counts was 3.11 × 10⁻¹⁵.

Four real IQ-TREE bootstrap refits with two total threads took 20.06 s with
one worker and 5.49 s with two (medians of three runs per setting).
The first configuration assigns two threads to each fit, the second assigns
one. Both used one BLAS thread and a cached fit for the observed alignment.
All 48 candidate bootstrap P values matched. Using the same replicate seed
with different IQ-TREE thread counts changed a maximum score by at most
8.6 × 10⁻¹¹ in the measured runs. Four refits test reproducibility and
execution time; the minimum attainable bootstrap P here is 0.2.

Use the [eight-tip alignment](bootstrap.fa) and the following commands in a
fresh directory. Set the IQ-TREE executable to the installed version.
Reuse the observed fit and change the output directory for each run.

```bash
printf '%s\n' '(((a:.3,b:.3):.2,(c:.3,d:.3):.2):.2,((e:.3,f:.3):.2,(g:.3,h:.3):.2):.2)R;' > tree.nwk
printf '1\ta\n2\te\n' > fg.tsv
python -m csubst scan --alignment_file bootstrap.fa --rooted_tree_file tree.nwk \
  --foreground fg.tsv --iqtree_model GY+FQ --iqtree_exe iqtree3 --iqtree_outdir fit \
  --scan_pvalue_calibration parametric_bootstrap --scan_bootstrap_workers 2 \
  --scan_n_permutations 4 --scan_permutation_seed 1 --scan_observation joint \
  --scan_min_event_pp 0.1 --scan_min_support 1 --scan_rate_length raw \
  --scan_rate_exposure endpoint --scan_site_plot no --threads 2 --blas_threads 1
```

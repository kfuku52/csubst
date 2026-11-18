# Distinguishing relaxation from directional convergence on PEPC

The comparison asks whether a branch pair passes the ordinary search filter when both branches have more nonsynonymous substitutions but have no new preference for a particular amino acid. The evaluation uses the 71-tip PEPC tree and 36 prespecified pairs among its nine independent C4 stems. Each search scores all 8,308 eligible pairs without foreground labels. Another 1,070 pairs have one focal stem and 7,202 have none. The endpoint statistic is `any2spe` throughout.

## Results

At the original PEPC lengths all three primary methods and the regional diagnostic gave the same candidate counts among the 288 target pair-replicates per condition. The baseline and both nondirectional relaxations gave zero candidates with either endpoint treatment. With a shared directional preference, ordinary search recovered **1/288** and the true endpoints recovered **15/288**. In the 14 missed pairs, true OCN was exactly two. Ten had inferred OCN between 1.999 and 2. The filter on the posterior mean explains the loss here. Estimated ECN and ECS did not change the set passing the OCN gate. From the generating Q, the expected numbers of target pairs per alignment attaining even two true N coincidences are 0.0013 for baseline, 0.0128 for whole-branch relaxation, and 0.0384 for local relaxation. Eight zero-call alignments at these lengths place little constraint on differences in directional specificity.

With the fourfold stems, the cells below show **inferred / known-endpoint** candidate counts out of 288 per row. For the codon model, replacing ancestors holds the tip-fitted Q, lengths and posterior site-rate weights fixed. For the urns the known branch changes are also used to re-estimate their empirical rates and draw totals.

| Generating condition | Branch-fixed urn | Predicted-N urn | Codon model | Regional N |
|---|---:|---:|---:|---:|
| Baseline | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Whole-branch relaxation | 0 / 1 | 2 / 8 | 2 / 8 | 0 / 1 |
| Local relaxation | 2 / 5 | 10 / 25 | 10 / 23 | 1 / 1 |
| Directional preference | 194 / 219 | 194 / 220 | 194 / 220 | 191 / 212 |

With inferred endpoints, branch-fixed N cut local-relaxation calls from 3.47% to 0.69% at the same 67.36% directional recovery as predicted-N and the codon model. Local relaxation produced a branch-fixed candidate in 2 of 8 alignments, versus 6 of 8 with predicted-N. Directional recovery ranged from 16 to 31 of 36 pairs per seed for the three primary methods. All 194 branch-fixed directional candidates had finite omega, with OCS at least one in 193. Both local-relaxation candidates met that OCS constraint. Among the 8,272 other eligible pairs per alignment, total inferred candidates across eight seeds were 0, 3, 2 and 2 in table order for each of the three primary methods. The regional diagnostic scores the 36 prespecified pairs.

Under the two nondirectional treatments the true number of **chance** coincidences is available directly from the generating Q. The table gives the sum of inferred ECN across target pairs and seeds divided by the sum of these generating expectations. A ratio of one is calibrated for this directional-null question. The denominator includes the relaxed rates. It differs from the simulator's separate background-only null column.

| Stem lengths | Relaxation | Branch-fixed urn | Predicted-N urn | Codon model | Regional N |
|---|---|---:|---:|---:|---:|
| Original | Whole branch | 0.98 | 0.35 | 0.37 | 0.97 |
| Original | Shared domain | 0.55 | 0.19 | 0.20 | 1.00 |
| Fourfold | Whole branch | 1.03 | 0.38 | 0.51 | 1.03 |
| Fourfold | Shared domain | 0.58 | 0.21 | 0.29 | 1.01 |

Fixing branch N absorbs the increased chance of recurrence caused by globally elevated N. The same number of N substitutions concentrated on a shared set of sites produces more coincidences. Fixing N separately in the known region and its complement accounts for this extra opportunity. The empirical S denominator is identical under all three urn analyses. Under local relaxation its ratio to the generating S expectation was 0.96 on the original lengths and 0.94 on the longer stems.

## Experimental design

Each condition has eight matched seeds and 1,040 codons. The main panel uses the bundled PEPC branch lengths. A sensitivity panel lengthens just the nine stems fourfold, across all sites and conditions. Under the original lengths the generating process produces only 1.32 N coincidences in total across 36 stem pairs per alignment under local relaxation. The longer stems provide more events for evaluating a per-pair threshold of two. The two panels have separate estimates.

| Generating condition | Changes on the nine focal stems |
|---|---|
| Baseline | The same background Q on every branch and site |
| Whole-branch relaxation | All nonsynonymous rates increase by a common factor at all 1,040 codons, relative to synonymous rates |
| Local relaxation | The same nondirectional increase on the first 208 codons of every focal stem |
| Directional preference | At those same 208 codons, one target amino acid per site is shared by all focal stems |

The background uses PEPC-derived codon frequencies and ECMK07 exchangeabilities. The input ratio of summed nonsynonymous to synonymous Q entries is 0.1 on the background and 0.9 in the local-relaxation block. The simulator normalizes Q after changing its rates. This operational definition of relaxation preserves the relative nonsynonymous exchange preferences and can also change the absolute synonymous rate. The block represents a hypothetical shared domain. CSUBST's `--convergent_amino_acids random0` supplies the nondirectional treatment and `random1 --percent_biased_sub 90` supplies the directional treatment. Background site rates are uniform. In the two local treatments the untreated 832 codons are identical for corresponding tips and ancestors at each seed.

Rates for the whole-branch and directional conditions were chosen before inference by equating the expected number of nonsynonymous endpoint changes summed over the nine stems to that of local relaxation. For each stem the calculation integrates the ancestral codon frequency times the transition probability to a different amino acid. The directional calibration uses that seed's actual target amino acids. An independent replay reads the matrices and lengths directly from Pyvolve, propagates codon probabilities through the tree, and checks all nine margins. Equal expected totals allow the allocation of N events across sites and destinations to vary. Realized integer totals fluctuate. The matched totals are 107.56 N changes at the original lengths and 409.76 at fourfold stem lengths. Baseline expectations are 60.16 and 236.91, respectively. The directional and locally relaxed per-stem expectations differ by at most 1.04% and 3.65%, respectively, within the two panels. The input whole-branch rate ratios are 0.196513 and 0.191438. The seed-specific directional ratios range from 0.9477 to 0.9606 and from 1.0828 to 1.0996.

## Estimators and endpoints

* **Branch-fixed urn** uses the current `--expectation_method urn` defaults, including empirical site weights, pooled route rates and separate observed branch N and S totals.
* **Predicted-N urn** uses the same all-branch fit, site profile and synonymous estimator. Only the N allocation changes. It uses all-branch route rates and effort estimated from each branch's combined N+S events. The two urns have identical ECS within each alignment and endpoint treatment.
* **Codon model** uses the ordinary fitted ECMK07_GY0K+F+R4 expectation. Each alignment has its own IQ-TREE fit from tips on the known topology. All three methods share that fit.

With **inferred endpoints**, observed N/S counts and parent states come from the default joint parent–child endpoint posterior. With **known endpoints**, we read the true parent and child codons for every site and branch, recompute observed counts and refit both urns to the true changes. For the codon expectation we substitute the true parent codon while retaining Q, fitted lengths and the tip-derived posterior weights of the four rate categories. As in default search, per-edge destination probabilities are multiplied within sites, then summed. Thus the endpoint comparison holds the tip-derived phylogenetic fit fixed.

A candidate requires both `OCNany2spe >= 2` and `omegaCany2spe >= 5`. Infinite omega with positive OCN counts under this filter. The output also records candidates with finite omega and with OCS at least one. A candidate under relaxation is a false directional call even if the simulated pair really experienced coincident substitutions. Under directional bias the target-pair pass fraction measures recovery of treated pairs. These are screening fractions. The replicate unit is one independently simulated alignment per condition, with branch and site sharing among its pairs.

## Regional diagnostic

The first four seeds of the long-stem panel showed fewer whole-branch relaxation calls using branch-fixed N, with residual calls under local relaxation. We therefore evaluated an additional N-only correction. For every branch we partition the sites into the predefined 208-codon region and its 832-codon complement, fix each partition's N total, allocate that total to destinations using the same all-branch routes and site weights, then add the two partitions' expected overlaps. The S estimator is identical to the default. No treated branches are identified in the fit. This is a best-case assessment for an independently supplied domain boundary. The last four seeds are reported separately because the first four motivated the follow-up.

On the four longer-stem seeds reserved for confirmation (144 pair-replicates per condition), regional N changed ordinary-search local-relaxation candidates from **1 to 0** and directional candidates from **86 to 85**. With known endpoints the corresponding changes were 2 to 0 and 95 to 94. Across all eight seeds regional N changed local ECN calibration from 0.58 to 1.01, but reduced directional recovery by three pairs. Branch-fixed N remains the supported general-purpose choice here. The regional estimate is informative as a sensitivity analysis when boundaries are available independently. Only one tree, one domain size and one background constraint were tested, with four confirmation seeds for the regional comparison. These data do not establish a general false-positive rate or a safe method for choosing domains from the same convergence signal.

The [pooled results](pepc_relaxation_summary.tsv), [seed-level scores](pepc_relaxation_seeds.tsv), [comparison of true and inferred OCN support](pepc_relaxation_asr.tsv), and [per-stem N calibration and realized counts](pepc_relaxation_branch_N.tsv) contain both panels. The seed-level and pooled tables include expected and observed N and S alongside cutoff counts.

## Reproduction and verification

Source revision `37a5acc`, CSUBST 1.17.5, IQ-TREE 3.1.4, Python 3.12.15, macOS arm64. The generator reads the bundled PEPC tree, alignment, foreground list and IQ-TREE input bundle. All fits and simulated sequences are written to fresh temporary directories.

```bash
export PYTHONPATH="$PWD" MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
script=reports/omega_low_count_20261116/pepc_relaxation.py
for scale in 1 4
do
    work=$(mktemp -d)
    python "$script" calibrate "$work" --stem-scale "$scale"
    python "$script" run "$work" --iqtree iqtree --jobs 5
    python "$script" audit "$work"
    python "$script" score "$work" --jobs 3
    python "$script" regional "$work" --jobs 3
    python "$script" summarize "$work"
done
```

The per-alignment scorer checks clade identity for every tree node, independently counts realized events from the FASTAs, and compares them with the simulator oracle on all 8,308 pairs. It checks fitted parameters and branch lengths against the ordinary codon search. In all four scenarios of both panels, seed 5209 also has an independent executable comparison with the standard urn search. The audit repeats these eight simulations and compares tip and ancestral sequences and the per-stem generating-Q margins. Summary checks verify each seed's actual target-amino-acid assignments and the shared untreated sequence block. The focused oracle, empirical urn and endpoint-posterior tests passed (14 tests).

All 64 tip-only IQ-TREE fits and all scoring assertions completed. The gate calculation for the nondirectional generators was checked independently against a binomial convolution. This study adds research scripts and tables. Its source checks cover the estimators and endpoints used here. The full package and installed-wheel suites were outside this analysis.

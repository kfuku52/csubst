# Support-guided group removal in the branch-pair index

The research driver compares two ways to propose four independent branches
after finding two site/destination patterns that share at least four branches.
Both methods start from the same index. The original samples up to 24
quartets per intersection. The alternative retains up to four paths. When
more than six branches remain, it also tries intersecting them with a third
site/destination pattern. This can remove several branches in one step. It
considers up to 24 distinct patterns per parent, prioritizing repeated
patterns per branch removed, and up to 24 one- or two-branch deletion moves.
When four to six branches remain the final subsets can be scored directly.

Each move is scored on the whole alignment. For each site, multiply the
branch probabilities of reaching a given derived amino acid and sum over
amino acids. Count the sites for which this sum reaches 0.5. Among paths
that remove different numbers of branches, prefer the largest gain in
supported sites per branch removed from the initial intersection. Sample
ties independently of the planted labels. For known events, intersections
of packed bit masks give the same site counts. A bounded score cache
reuses repeated posterior calculations on the PGK panels. The comparison
driver applies PP ≥ 0.5 to OCN selection and ranking for both methods.
The ωC cutoff always uses the full-alignment counts.

The group-removal rule is inspired by MaxAlign's greedy improvement per
sequence removed. The CSUBST prototype proposes multiple candidates at a
specified arity. Final scoring uses CSUBST's convergence criteria.
[MaxAlign method](https://doi.org/10.1186/1471-2105-8-312) ·
[Baseline index and panel definitions](BRANCH_PAIR_INDEX.md)

## Recovery and cost

Each cell gives the original index followed by the support-guided index.
In each known-event simulation, the planted K branches shared changes at
four sites and passed the final hit thresholds. The table counts seeds in
which the planted combination entered the candidates or ranked in the top 64.
Times and peak process RSS are paired-panel medians at the same maximum of
65,536 proposals. The trials at K=5 and K=6 plant five and six convergent
branches respectively.

| Sites | K | Seeds | Proposed | Top 64 | Seconds | Peak MiB |
| ---: | ---: | --- | --- | --- | --- | --- |
| 2,048 | 4 | 1–12 | 12 / 12 | 12 / 12 | 0.232 / 0.240 | 166 / 165 |
| 8,192 | 4 | 1–12 | 7 / 7 | 3 / 3 | 0.981 / 1.092 | 233 / 233 |
| 8,192 | 4 | 13–24 | 4 / 8 | 1 / 1 | 0.906 / 0.989 | 233 / 233 |
| 8,192 | 5 | 1–12 | 12 / 12 | 12 / 12 | 0.789 / 0.902 | 228 / 230 |
| 8,192 | 6 | 1–12 | 12 / 12 | 12 / 12 | 0.654 / 0.618 | 224 / 223 |

At K=4 and 8,192 sites both methods recovered exactly the same four planted
quartets in the top 64 across 24 seeds. Support-guided proposals added the
quartet for seeds 18, 20, 22 and 23. Each had four supporting sites. In those
four runs respectively, 18, 13, 15 and 9 proposals had greater N support,
and 380, 339, 346 and 343 had equal support (including the plant).
The vote-based tie breaker and random exact ties placed the planted
quartets at ranks 102, 165, 175 and 210. All four could have entered the top
64 under a different ordering of these ties. At K=5 and K=6 both methods
reached the maximum possible recovery. The planted combination had zero
synonymous coincidences in 1/12 and 4/12 seeds, respectively.

For K=4, 82.6% of the sampled eligible site pairs already intersected in
exactly four branches at both lengths, leaving just one possible proposal.
The fraction with seven or more branches, where a third site can drive a
larger group removal, was 0.31% at 2,048 sites and 0.34% at 8,192 sites.
These fractions use all sampled site pairs before the proposal limit.

Under a shared 0.25-second deadline at 2,048 sites, both methods proposed
and recovered all 12 plants. Under a 1.0-second deadline at 8,192 sites,
the index and group refinement proposed 4 and 3 plants respectively and
both recovered 2/12. The original index had no finished checkpoint in one
run and the refinement had none in two. Deadline outcomes depend on machine
load. The fixed-proposal runs above give reproducible candidate sets.

Across the three strong PGK alignments, parent-state uncertainty gave
210 exhaustive reference hits. The original index proposed all 210 and
group refinement proposed 209. Both recovered 192 in the 192 available
slots. Median candidate generation and ranking took 0.246 and 0.206 seconds,
respectively. In the moderate directional panels both proposed four
quartets and recovered the one reference hit. The relaxation panels had
zero reference hits and neither method proposed a quartet. These PGK
experiments assess K=4 on a prespecified set of independent stems. The
higher-arity results use known events on the synthetic branches.
[Per-seed proposals, recovery, time and memory](group_refinement_runs.tsv)

## Validation protocol

The scale experiment uses the same four planted sites, quartet, 256
independent branches, two sequence lengths and 12 seeds as the baseline.
One arm allows up to 65,536 quartet proposals across the alignment and ranks
distinct candidates by whole-alignment N support, with votes breaking ties.
Another arm records the last available top 64 under a common wall-clock
deadline. The index, proposal processing and ranking all consume that time.
Both methods may emit up to 131,072 proposals in this arm. A complete ranked
answer is retained every 4,096 draws and at completion, using a copy of the
RNG for random ties. A checkpoint completing after the deadline cannot
contribute candidates. Data generation, interpreter startup and scoring the
known quartet for the reference are outside both clocks.

The PGK comparisons share the saved IQ-TREE fit, eight categorical posterior
draws and the same randomly ordered site pairs per draw. Cutoff OCN was
recomputed for every proposed quartet and checked against the exhaustive
CSUBST table (relative tolerance 1e-10, absolute tolerance 1e-12). Each method
ranked up to 64 quartets per seed using cutoff OCN. The hit cutoffs were
OCNany2spe ≥ 2 and ωCany2spe ≥ 5. Elapsed time covers candidate generation
and OCN ranking after loading inputs. Exhaustive reference checks are outside
the clock.

## Reproduction

The base commit is `d56ffe5`. The benchmark ran on macOS arm64 with Python
3.12.15, NumPy 2.5.5 and SciPy 1.18.1. Use one BLAS thread and a fresh
process per run. For the scale panel alternate method order across seeds
1–12 after a warm-up (seed 101). Both methods used 0.25 seconds at 2,048 sites
and 1.0 second at 8,192, chosen after exploratory runs on seeds 98–100.
The additional K=4 trials used seeds 13–24 and the same proposal limit.
The PGK cases use the same strong directional,
moderate directional and relaxation panels as the baseline, with seeds
5749, 5881 and 6151. The `pgk` argument is the directory containing
`posteriors.npz` and `reference/csubst_cb_4.tsv`.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
driver=reports/branch_search_review/group_refinement_benchmark.py
for method in indexed group_refined; do
    python "$driver" scale "$method" 8192 1
    python "$driver" scale "$method" 8192 1 --arity 6
    python "$driver" scale "$method" 8192 1 --seconds 1.0
    python "$driver" pgk "$method" "$panel_dir"
done
python "$driver" witness-sizes 8192 1
python "$driver" planted-rank 8192 18
python -m pytest -q reports/branch_search_review/test_group_refinement.py
```

[Follow-up: reranking equal OCN by synonymous counts](SYNONYMOUS_TIES.md) tests the four additional proposals, independent seeds and negative controls.

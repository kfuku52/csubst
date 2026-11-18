# Estimating empirical urn weights from all PEPC branches

A research prototype fitted the empirical site rates, amino-acid routes and N
calibration scale using all 138 eligible branches, including the evaluated
branches. This changes the training sample of the pairwise empirical urn,
not the underlying Wallenius inclusion probability calculation. Branch N+S
effort and observed S destination totals remain branch-specific. Site-rate
regularization still uses leave-one-training-branch-out scores *within* the
training set. For each destination, the K-branch expected overlap is the
sum over sites of the product of the K inclusion probabilities. Summing
these overlaps over destinations gives ECN or ECS for `any2spe`.

## Search without a specified foreground

We replayed the search from all independent pairs. Only branch IDs and
observed N/S counts were read from a search with the looser OCN-only cutoff;
its expected counts, omega and candidate rankings were discarded. At each
K the ordinary CSUBST candidate generator and cutoff selector expanded the
previous retained rows. Both cutoffs were required: OCN >= 2 and raw
omegaC >= 5. The 10,000-row cap never applied. The last column uses only
the K-subsets of the final six branches, comparing the all-branch ECN to an
independent fit that excluded exactly the K evaluated branches.

| K | Candidates generated | Retained | Focal subsets compared | Median ECN(all) / ECN(exclude K) |
|---:|---:|---:|---:|---:|
| 2 | 8,446 | 250 | 15 | 1.031 |
| 3 | 2,046 | 63 | 20 | 1.073 |
| 4 | 121 | 20 | 15 | 1.129 |
| 5 | 14 | 6 | 6 | 1.189 |
| 6 | 1 | 1 | 1 | 1.227 |

The one K=6 hit was `1,26,33,35,40,102`, also found by the codon model in
the [earlier PEPC search](PEPC_HIGHER_ORDER.md). The high-K default urn
(`each`) previously stopped at K=3. The replay above was performed before changing the default. A subsequent
real `csubst search --expectation_method urn` on the same fit, with arity
limits 2 and 10, selected `empirical_all` in both runs. All reported EC
columns for the 8,446 pairs agreed between runs at 17 decimal places.
The higher-arity run reproduced the candidate counts above and the
prototype ECN/ECS for all 57 focal subsets (relative tolerance 1e-11
plus 6e-18 for output rounding). The previous pairwise fit, explicitly
requested with `--asrv empirical`, retained 263 pairs.

See the [direct comparison with fitted codon Q](PEPC_HIGH_ARITY_COMPARISON.md)
for natural-data hit overlap, timings and simulation-based high-arity accuracy.

## Influence of the six branches on their own expectation

| K=6 training set | Training branches | ECN | ECS |
|---|---:|---:|---:|
| All eligible | 138 | 1.13776e-8 | 1.35959e-7 |
| Exclude exactly the six evaluated branches | 132 | 9.27640e-9 | 1.45183e-7 |
| Exclude six random nonfocal branches (median of 40, seed 8) | 132 | 1.11451e-8 | 1.38207e-7 |

With six random nonfocal branches omitted the 10th–90th percentile range
for ECN was 9.99839e-9 to 1.22724e-8. Including the focal branches raised
their ECN by 22.7% and lowered their ECS by 6.4% relative to omitting
them. OCN was 2.498897; OCS was zero at the reported precision and raw
omegaC was infinite with both fits. The expected counts therefore show an
influence that the reported ratio cannot resolve. These are comparisons on
one selected natural-data hit, not estimates of error against a true null,
false positive rate or power. All 57 focal subsets passed under both fits.
[Per-subset scores](pepc_all_branch_urn_scores.tsv)

## Reproduce

CSUBST 1.17.5 working tree based on b782efd, macOS arm64, Python 3.12.15.
The 71-tip, 971-codon PEPC alignment and bundled ECMK07+F+R4 IQ-TREE fit
were used unchanged. The fitted joint endpoint model, observed counts and
eligibility masks are common across training choices. The prototype checks
both ECN and ECS against the shipped K=2 implementation on all 15 focal
pairs with identical four-fold training sets (relative tolerance 1e-11).

From the repository root in the editable development environment:

```bash
repo=$PWD
work=$(mktemp -d)
mkdir "$work/inputs"
cp csubst/dataset/PEPC.alignment.fa* csubst/dataset/PEPC.tree.nwk "$work/inputs/"
cd "$work"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
fit=inputs/PEPC.alignment.fa
python -m csubst search --alignment_file "$fit" --rooted_tree_file inputs/PEPC.tree.nwk \
  --iqtree_model ECMK07+F+R4 --iqtree_state "$fit.state" --iqtree_rate "$fit.rate" \
  --iqtree_iqtree "$fit.iqtree" --iqtree_log "$fit.log" --iqtree_treefile "$fit.treefile" \
  --b no --max_arity 10 --threads 1 --blas_threads 1 --random_seed 8 --float_digit 17 \
  --cutoff_stat OCNany2spe,2 --outdir observed
python "$repo/reports/omega_low_count_20261116/pepc_all_branch_urn.py" inputs observed scores.tsv
```

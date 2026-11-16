# Estimating an ECMK07 nonsynonymous multiplier

The default `ECMK07+F+R4` uses ECMK07's empirical exchangeability matrix,
codon frequencies from the current tip alignment, and fitted branch lengths
and FreeRate parameters. It fixes the multiplier for the nonsynonymous
entries of that matrix at one. The earlier
[conditional decomposition](EC_DECOMPOSITION.md) identified this
restriction as the main driver of the denominator error in PEPC.

`ECMK07_GY0K+F+R4` fits one nonsynonymous multiplier in IQ-TREE. It
preserves the empirical exchangeabilities, including rates for pairs of
codons differing by more than one nucleotide. IQ-TREE estimates the
multiplier by maximizing the tip likelihood on the given topology.
CSUBST reads the coefficient from the matching checkpoint and uses it to
reconstruct the Q for both ancestral inference and expected counts.
The report's `omega_E` is a derived stationary N/S flux statistic.
It is checked against the reconstructed Q and does not supply the
exchangeability multiplier. See the
[IQ-TREE 3.1.4 codon implementation](https://github.com/iqtree/iqtree3/blob/v3.1.4/model/modelcodon.cpp).

The PEPC alignments and topologies are the same six as in the
[simulation report](README.md). Both refits re-estimate branch lengths and
the rate mixture from the same tips. The table sums counts over the 36
foreground pairs within each replicate and the three replicates of each
condition. Pooled omegaC uses the sums of the model's own observed and
expected counts.

| Condition | Model | ECN | ECS | Pooled omegaC |
| --- | --- | ---: | ---: | ---: |
| Background | Q-based reference | 2.192 | 110.815 | 1.00 |
| Background | ECMK07 | 10.218 | 57.004 | 0.10 |
| Background | ECMK07 with fitted multiplier | 2.233 | 112.341 | 0.91 |
| Biased | Q-based reference | 2.192 | 110.815 | 3.32 |
| Biased | ECMK07 | 9.492 | 56.129 | 0.30 |
| Biased | ECMK07 with fitted multiplier | 2.074 | 109.554 | 2.61 |

The reference rows show Q-based expected omegaC under the corresponding
simulation condition and the Q-based EC denominators. The realized,
known-ancestor omegaC values with the
same null denominator were 0.86 and 2.36. The fitted ECMK07 multipliers
ranged from 0.309 to 0.338, against a generator value of 0.33038.
Stationary N flux was 17.0–18.8% for fitted ECMK07 and 17.9% for the generator.
The fitted hybrid had higher tip likelihood than the fixed ECMK07 model
in all six replicates. Independent pruning with the imported Q and the
IQ-TREE checkpoint rates reproduced the hybrid IQ-TREE likelihood to
within 4.84 × 10⁻⁵ log-likelihood units across the six fits. Search
itself retains its existing use of the rate-category table in the IQ-TREE
report.

The hybrid estimates a single multiplier for the whole alignment. The
simulation uses an ECMK07 backbone with a global N multiplier, so this
comparison tests recovery of the generated denominators on PEPC under
that model.

`GY+F+R4` was also tried. Only biased seeds 101 and 202 finished within
the exploratory run. Four other GY fits were stopped after more than 11
minutes in optimization. On the same 72 foreground pair rows, GY gave
ECN/ECS of 3.615/96.227, versus 6.294/38.417 for fixed ECMK07 and
1.335/74.486 for the fitted ECMK07. The Q-based counts were 1.461/73.876.
GY retained a larger ECN and increased ECS above the reference on this
subset. Use the [matched comparison](codon_model_gy_matched.tsv) for GY
and the [full six-run comparison](codon_model_comparison.tsv) for ECMK07.

## Reproduce

Use the six simulation and baseline search outputs from
[the original run](README.md#reproduce). In a new temporary directory
create a subdirectory for each replicate, named as in the original run,
and copy its `sim.fa` and `sim_true_asr.treefile`. Run IQ-TREE 3.1.4
from each replicate directory. The report uses CSUBST 1.17.5 with this
change, Python 3.14.7, and the same macOS arm64 environment as the
simulation report.

```bash
mkdir -p hybrid_fit
iqtree -s sim.fa -te sim_true_asr.treefile -m ECMK07_GY0K+F+R4 \
  --seqtype CODON1 -T 1 --ancestral --rate -v --redo --prefix hybrid_fit/pepc -seed "$seed"
python -m csubst search --alignment_file sim.fa --rooted_tree_file sim_true_asr.treefile \
  --iqtree_iqtree hybrid_fit/pepc.iqtree --iqtree_log hybrid_fit/pepc.log \
  --iqtree_treefile hybrid_fit/pepc.treefile --iqtree_rate hybrid_fit/pepc.rate \
  --iqtree_state hybrid_fit/pepc.state --random_seed "$seed" \
  --max_arity 2 --exhaustive_until 2 --exclude_sister_pair no --output_stat any2spe \
  --float_digit 12 --b no --outdir hybrid_search
```

For the exploratory comparison, run GY on biased seeds 101 and 202:

```bash
python -m csubst search --alignment_file sim.fa --rooted_tree_file sim_true_asr.treefile \
  --iqtree_model GY+F+R4 --iqtree_redo yes --iqtree_outdir iqtree --random_seed "$seed" \
  --max_arity 2 --exhaustive_until 2 --exclude_sister_pair no --output_stat any2spe \
  --float_digit 12 --b no --outdir search
```

For direct use, `csubst search --iqtree_model ECMK07_GY0K+F+R4 --iqtree_redo yes`
also runs IQ-TREE and retains the checkpoint and verbose log.

From the repository root, set `original` and `refits` to the directories
containing the original and new replicate directories, then summarize:

```bash
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python reports/simulate_expected_omega_20261116/compare_fitted_models.py \
  "$original" "$refits" reports/generated/pepc_model_refits
```

- [Counts by seed](codon_model_comparison_by_seed.tsv)
- [Fitted parameters and reconstructed likelihoods](codon_model_fit_diagnostics.tsv)

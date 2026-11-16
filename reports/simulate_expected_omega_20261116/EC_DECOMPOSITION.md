# Why PEPC search increases ECN and decreases ECS

The dominant cause is the difference in the underlying N/S exchangeability
of the two codon generators. The simulator lowers nonsynonymous
exchangeabilities from the ECMK07 matrix when it builds its background Q.
Search with `ECMK07+F+R4` uses the empirical exchangeabilities and the
codon frequencies of each simulated tip alignment. In these PEPC runs it
does not fit the simulator's nonsynonymous multiplier. Branch length
estimation contributes to the reduction in ECS and partially offsets the
increase in ECN. The FreeRate fit has a much smaller effect.

The background generator multiplied the nonsynonymous exchangeabilities
by **0.33038** to attain the simulator's unweighted N/S rate-mass ratio of
0.2. After stationary total-rate normalization, N events account for
**17.9%** of instantaneous flux under that Q and **37.8–42.0%** under
the six fitted search generators. S accounts for the remaining flux.
See [the fitted inputs by seed](ec_fit_diagnostics.tsv), the
[CSUBST Q builder](../../csubst/parser_misc.py), and
[IQ-TREE's empirical codon model documentation](https://www.iqtree.org/doc/Substitution-Models).

The table sums EC counts over the 36 pairs of foreground branches and
the three biased replicates. Entries starting with a sign are changes in
raw counts. The first row uses the generator-based exact pair expectation.

| Source or change | ECN | ECS |
| --- | ---: | ---: |
| Simulation null | 2.192 | 110.815 |
| Change to search's parent conditioning and pair aggregation | −0.014 | −1.377 |
| Q | +8.087 | −45.466 |
| Branch lengths | −0.681 | −8.438 |
| Site rates | −0.091 | +0.596 |
| Full search | 9.492 | 56.129 |

Q accounts for 83% of the net ECS decrease. The fitted foreground length
sum ranges from 0.538 to 0.581 across these three replicates, compared
with 0.593 in simulation. The posterior mean site-rate averages are
0.99994–1.00005. The category mix is still heterogeneous and enters the
replay through the full site-category posterior.

At the fitted branch lengths and rate categories, restoring the simulated
N multiplier alone, while keeping the fitted frequencies, changes
ECN from **9.492 to 1.994** and ECS from **56.129 to 104.274**.
An order-averaged split of the Q effect at these fitted lengths and rates
assigns **+7.386 ECN and −46.950 ECS** to the exchangeability change and
**+0.189 ECN and +3.485 ECS** to the frequency change. This is a
conditional decomposition of Q at fitted lengths and rates. Its sum
can differ from the averaged Q contribution in the main table.

The background-only simulations show the same pattern. Their
foreground pairs go from 2.192 and 110.815 to 10.218 and 57.004.
Q contributes +8.487 and −48.617. The direction also holds across
all eligible branch pairs and each of the six individual foreground summaries.

## How the comparison was made

The analysis uses the six completed searches in the [PEPC study](README.md).
It reconstructs the fitted Q and checks its hash against
`csubst_endpoint_model.json`. It keeps the fitted joint posterior
`P(category, parent codon | tip data)` fixed. Each
counterfactual predicts the transition with either the simulated
background or fitted Q, the actual simulation or fitted branch lengths,
and either unit site rates or the fitted category rates. When rates are
set to one, category probabilities are still marginalized under the
original fitted posterior. The pair calculation then matches destination
states using the product of branch predictions as in search. The effect
of each factor is its average marginal change over all six replacement
orders, so interactions are distributed across the factors. The
counterfactuals are conditional predictions, with branch lengths,
rate-category posteriors and Q held as specified in each comparison.

For the two parent-state comparisons, one prediction uses the exact
realized codons at both parents and the other uses the fitted parent
posteriors. Both use the simulation Q, lengths and unit rates.
For the biased foreground pairs these give ECN/ECS of
2.177/109.547 and 2.178/109.438. Their difference is small.
The comparison to the exact simulation null additionally includes finite
sampling of parent codons and the branch-pair dependence in the simulation.

Both EC columns are `NA` in search for 139 pairs per replicate that include
a root-adjacent branch. The decomposition uses identical row masks for
all comparisons. All 36 foreground pairs are eligible. Over all pair
groups in the three biased replicates, this mask removes 9.943 ECN and
497.490 ECS from the original simulation totals. The fully fitted replay
agrees with the production values for all **50,262 reportable pair rows**,
with a maximum absolute error of `5.02e-13`.

The earlier interpretation of denominator bias in this experiment depends
on the fitted model. These values quantify a conditional substitution of
inputs for one PEPC topology and six alignments. Refitting under a
different codon model can change lengths, rates and ancestral posteriors.

## Reproduce

Generate the six simulation and search outputs using the
[reproduction commands](README.md#reproduce). Run from the repository root
in the same Python environment, with `work` set to the temporary
directory containing those runs.

```bash
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python reports/simulate_expected_omega_20261116/decompose_expectation.py \
  "$work" reports/generated/pepc_ec_decomposition
```

- [Effects over three seeds](ec_effects.tsv) and [effects by seed](ec_effects_by_seed.tsv).
- [All counterfactual counts](ec_corners.tsv). The four digits indicate use
  of fitted N exchangeabilities, frequencies, lengths and rates in that
  order. Zero means use the simulation input for that component.
- [Fit and replay diagnostics](ec_fit_diagnostics.tsv).

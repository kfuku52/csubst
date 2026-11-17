# Parent amino-acid uncertainty in inferred sister-pair N counts

The PEPC comparison contains 69 eligible sister pairs in each of six
simulations. We replayed the complete and clade-masked tips with their
respective fitted models and matched both CLI observed N totals for every
pair. The primary comparison is the three directional-bias seeds under
clade masking, with 96,408 eligible pair-sites and zero realized N
convergences. [Alignment and masking design](MISSINGNESS.md)

## Sharing the parent state accounts for the change

For each site and destination amino acid, the shared-rate count averages
the product of two separately averaged parental states. The full joint
count averages the product over a single parent state and rate category.
The difference is the posterior rate-weighted sum of the parent-state
covariances in the two conditional change probabilities.

| Parent and rate treatment | Pooled observed N estimate |
| --- | ---: |
| Separate parent states and rates | 0.10173 |
| Shared rate, separate parent states | 0.10185 |
| Shared parent and shared rate | 2.00982 |
| Shared rate and parent codon conditional on the true parental amino acid | 0.00642 |

Of the fitted joint total, **2.00573 of 2.00982** comes from parent
states whose amino acid differs from the simulated truth. At sites where
both true child amino acids equal the true parent amino acid, the increase
is 1.90750 of the total increase of 1.90809. About 81% of the increase
occurs outside the directionally biased site partition. In the unmasked
replays of the same three seeds, the observed N estimates change from
0.109 to 2.172 for the same realized total of zero.

At seed 202, site 33, the Sesbania and Vitis tips both carry GTT (Val),
as does their true parent. The fitted posterior assigns the parent
93.57% Val and 6.43% other amino acids, mostly Ile. The three tips in the
neighboring sister subtree all carry Ile. Under a shared parent, the
inferred probability of N convergence is 6.43%. With separate parents
it is `0.0643² = 0.00413`. The largest single increase is at seed 202,
site 415, with true codons ATT, ATT, and ATC at the parent and the two
endpoints. All encode Ile. The five tips in the neighboring subtree carry
Val, and the shared parent has 18.05% posterior probability of Val.
[Largest site contributions](sister_n_top_sites.tsv)

## Effects of known generating parameters

| Model for masked directional-bias runs | Joint observed N |
| --- | ---: |
| Fitted model | 2.00982 |
| Replace fitted site-rate mixture with the generating unit rate | 2.00307 |
| Replace fitted lengths with the generating lengths | 1.98628 |
| Replace both lengths and rates | 1.98193 |
| Also use the generating background Q | 1.95411 |
| Full generating model with the logged target amino acid and foreground edges per site | 1.82744 |

The oracle uses the simulator's background Q, root frequencies, and the
site-specific foreground generator reconstructed from the simulation log.
Its unconditional expected N counts reproduce the simulated-Q counts in
the export for each selected pair. Across the three masked replicates,
the probability of zero realized eligible sister N events under that model
is 19.0% before observing the tips and 15.6% conditional on the tips.
We calculate the probability of no event jointly across eligible sister
pairs within each site by pruning with those events excluded. Probabilities
then multiply across independent sites and alignments.

The background-Q controls illustrate the opposite realized error. For
the six true N convergences under masking, the fitted separate and joint
totals are 0.075 and 1.560, respectively. The generating-model posterior
mean is 1.572. Conditioning the fitted model on the true parent amino
acid brings the total to 5.972. Both results reflect uncertainty at the
common parent, with different realized ancestral states and event totals
across the six simulated alignments. The comparisons concern observed
endpoint counts. Expected counts are shared between the two search modes.
[Model and site summaries](sister_n_totals.tsv), [zero-event probabilities](sister_n_zero.tsv)

## Reproduce

Use the PEPC panel and `cases` directory from the masking analysis, with
CSUBST 1.17.5 (`fa7cd07`) in the editable Python environment. Run from the
repository root. The verification script uses independent finite-state
enumeration to check the decomposition and the probability of zero events
with a rate mixture, nested sister pairs, and missing tips.

```bash
scripts=reports/omega_low_count_20261116
work="$(mktemp -d)"
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python "$scripts/verify_sister_n.py"
python "$scripts/sister_n_diagnostics.py" "$panel" "$cases" "$work" --jobs 4
```

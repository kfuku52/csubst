# PEPC comparison of the default codon and empirical urn expectations

Both methods used the search default joint endpoint posterior from
ECMK07_GY0K+F+R4. The codon expectation was selected by omitting
`--expectation_method`. The urn search added only `--expectation_method urn`,
selecting the Wallenius urn and `empirical_all` with the observed N and S
totals fixed separately on each branch. All values below use `any2spe`.

## Accuracy with known generating Q

The independent panel has 71 tips, 520 codons, and eight seeds in each
condition. Directional bias acts toward a shared amino acid at 20% of sites
on the nine designated stems. Unit and mixed rates without directional
bias are controls. The new fits use four estimated rate categories even
for simulations generated with a constant rate. Each alignment has one fit
shared between both expectation methods. The 24 alignments are the same as
in the [branch-fixed urn evaluation](PEPC_FULL_POOLING.md).

The ordinary K=2 search used its default statistics and sister-branch
exclusion. Of 8,446 rows per alignment, 8,308 have eligible expected counts
for both models. Each condition contributes 66,464 pair-replicates. The
previous report also allowed sister pairs and fitted homogeneous models to
the unit-rate simulations. Numerical differences between reports can follow
from both choices.

Count error is WAPE against the generating *null* Q, summed over eligible
combinations and replicates. Omega error is the RMSE of
`log(1 + omegaCany2spe)` against the generating signal/null omega. It uses
the same pairs for both methods, with both observed counts at least 1e-9 and
both estimates finite. The respective pair counts are 58,405, 63,015 and
57,371 in table order. Each cell gives **codon model / urn**. Smaller is better.

| Condition | Pair ECN WAPE | Pair ECS WAPE | Pair omega error | K=6 ECN WAPE | K=6 ECS WAPE |
|---|---:|---:|---:|---:|---:|
| Unit rates | 0.226 / 0.313 | 0.214 / 0.261 | 1.538 / 1.569 | 0.802 / 0.600 | 0.850 / 0.508 |
| Mixed rates | 0.227 / 0.302 | 0.222 / 0.279 | 1.281 / 1.339 | 0.593 / 0.498 | 0.772 / 0.561 |
| Directional bias | 0.231 / 0.317 | 0.221 / 0.267 | 1.535 / 1.573 | 1.082 / 0.535 | 0.780 / 0.573 |

The codon model had lower pairwise ECN and ECS WAPE on every one of the
24 alignments. Omega error was lower on 18 of the 24. Replacing the
inferred observations by true simulated endpoint counts on pairs with
both realized counts positive gave omega errors of 1.041 / 1.057,
0.782 / 0.799 and 0.990 / 1.011. This separate comparison uses 888, 1,679
and 937 pair-replicates.

For K=2–6 the additional targeted analysis scored all subsets of the nine
prespecified stems irrespective of cutoff. The K=6 sample is 84 combinations
per seed, or 672 per condition. The urn had lower aggregate K=6 WAPE for
both channels in all conditions. Individual alignments varied widely. For
K=6 ECN, the codon model had lower error on 7, 5 and 5 of the eight seeds,
respectively. Some large errors drive the pooled result. For example, in the
mixed-rate condition aggregate K=6 ECS divided by the generating expectation
was 1.330 / 0.555. Neither K=5 nor K=6 had a supported simulated raw omega.
The generating null is unconditional on the observed branch N and S totals.
These simulations do not directly measure specificity under branch-specific
relaxation of selection.

[All arities, aggregate signed ratios, errors and counts](pepc_search_defaults_simulation.tsv) ·
[Per-seed results](pepc_search_defaults_seeds.tsv)

## Observed PEPC alignment

The 971-codon alignment was refitted under the current default model. The
packaged ECMK07+F+R4 fit uses a different base model. The shared ASR gives
summed OCN 3,392.82 and OCS 20,468.84 across the 8,308 eligible pairs.
The omega median uses the 8,271 pairs with both observed counts at least
1e-9. The ordinary cutoff requires OCN ≥ 2 and raw omega ≥ 5.

| Observed PEPC result | Codon model | Urn |
|---|---:|---:|
| Sum of pair ECN | 847.95 | 1,120.52 |
| Sum of pair ECS | 19,967.61 | 20,164.52 |
| Pair omega median | 0.682 | 0.472 |
| Pairs passing cutoff | 307 | 217 |
| Omega median among the 36 prespecified C4 stem pairs | 64.88 | 6.72 |
| C4 stem pairs passing cutoff | 33 | 20 |

Of the candidates, 174 pairs were shared, 133 occurred only with the codon
expectation and 43 only with the urn. Within the 36 C4 stem pairs all 20
urn candidates also passed with the codon expectation. As an example, stem
pair 1,5 had common OCN 10.228 and OCS 3.703. ECN was 0.204 / 1.173,
ECS was 7.987 / 3.550 and omega was 108.03 / 8.36. Both methods retained
one prespecified K=6 combination, 1,26,33,35,40,102. Its OCS was below
1e-9 and both reported infinite omega. There is no known true omega for
the observed PEPC alignment.

[Observed results by arity](pepc_search_defaults_observed.tsv) ·
[Observed branch combinations with both estimates and cutoff flags](pepc_search_defaults_branches.tsv)

## Reproduction and checks

CSUBST 1.17.5 working tree based on b782efd, IQ-TREE 3.1.4, Python 3.12.15
and macOS arm64. Start with the independent panel produced by the
[PEPC simulation protocol](PEPC_PAIR_OVERLAP_CV.md#reproduction). The command
uses fresh input copies. The eight matching mixed-rate fits are reused.
The other 16 simulations and the observed alignment require new fits.
Both methods use the same per-alignment fitted parameters, tree and site rates.
The all-pairs searches set one thread, a reproducible seed and 17 decimal places
for output. The targeted searches additionally set the foreground, K and
enumeration options to score the same subsets for both methods.

```bash
script=reports/omega_low_count_20261116/pepc_search_defaults.py
work=$(mktemp -d)
python "$script" run "$panel" "$work" --iqtree iqtree --jobs 5
python "$script" score "$panel" "$work" reports/generated/pepc_search_defaults
```

Scoring verifies completed manifests, joint ASR, all four rate categories,
identical fitted-generator provenance, branch lengths, observation counts,
branch IDs and support masks. It also checks the observed-total urn provenance
and the expected oracle row for every simulated combination. There are 288
simulation score rows before aggregation. On the observed alignment, the
IQ-TREE 3 hybrid checkpoint uses empirical frequencies of fully resolved
sense codons. The matching CSUBST frequency check now excludes partially
ambiguous codons when that model's report omits the frequency vector.
The resolved-codon estimate agreed with the checkpoint within 5e-12 across
all 61 codons. All 24 simulated alignments contain only A/C/G/T.

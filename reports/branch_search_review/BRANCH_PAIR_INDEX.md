# Branch-pair indexing and site-level OCN support

For each site and derived amino acid, the prototype indexes all pairs of branches
with that event. If two such lists share four branches, they share at least six
distinct branch pairs. A blocked sparse matrix product finds the site pairs
meeting this condition. The search samples eligible site pairs and emits at
most 65,536 quartet draws. A site containing many branches is expensive to
index because it contributes one entry per branch pair.

## Recovery and cost on 256 independent branches

The two site-based methods use the same seeded inputs, quartet draw limit and final budget
of 64. Both rank distinct proposals by whole-alignment N support, with the
original votes breaking ties. The comparator probes up to four million site
pairs. The index processes all eligible site pairs before sampling them.
The planted quartet passed the OCN and ωC thresholds in all 12 seeds.

| Codon sites | Candidate generator | Planted quartet proposed | In top 64 | Median seconds | Median peak RSS (MiB) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2,048 | Direct site-pair probes | 12/12 | 12/12 | 0.234 | 152 |
| 2,048 | Branch-pair index | 12/12 | 12/12 | 0.175 | 162 |
| 8,192 | Direct site-pair probes | 5/12 | 4/12 | 0.261 | 188 |
| 8,192 | Branch-pair index | 7/12 | 3/12 | 0.747 | 223 |

On the same inputs we scored all 32,640 branch pairs per seed using the
generating site probabilities and known events. At both lengths, zero pairs
met both OCNany2spe ≥ 2 and ωCany2spe ≥ 5 in every seed. The default
`all_cutoffs` search therefore had no parents to extend and recovered the
planted quartet in 0/12 seeds at either length. The timing table compares
the two site-based generators only.

At 8,192 sites the index contained a median of 1,016,820 entries and found
about 488,000 eligible site pairs. More planted quartets entered the candidate
list, but four of those seven fell below rank 64. The timing covers candidate
generation and N-support ranking. Peak RSS includes the entire Python process.
Each seed and method ran in a separate process, alternating method order after
one warm-up per method. These measurements support keeping the index experimental.

## PGK codon simulations with inferred ancestors

We used the 33-tip PGK tree and 520 codons. For each alignment, IQ-TREE fitted
`ECMK07_GY0K+F+R4` from tip sequences on the supplied topology. CSUBST computed
joint parent–child posteriors and combined these separately across branches.
The reference enumerated independent quartets and applied OCNany2spe ≥ 2 and
ωCany2spe ≥ 5. All comparisons used seeds 5749, 5881 and 6151.

The strong positive control isolated the eight treated tips in a prespecified
panel of 25 independent stems (21 terminal and four internal). This gives
12,650 quartets per seed. Directional bias affected 40% of sites, with
background ω of 0.1, foreground ω of 2 and a foreground branch-length multiplier
of 8. Each seed had exactly 70 hits, the four-branch subsets of the eight treated
branches. All 70 also passed the site-PP threshold of 0.5.

| Method at site PP ≥ 0.5 | Parent cap per step | Distinct proposals across seeds | Reference hits proposed (of 210) | Hits in top 64 per seed (of 192 slots) |
| --- | ---: | ---: | ---: | ---: |
| `all_cutoffs` | 10,000 (default) | 210 | 210 | 192 |
| `ocn_bound` | 10,000 | 210 | 210 | 192 |
| `all_cutoffs` | 32 | 136 | 136 | 136 |
| `ocn_bound` | 32 | 136 | 136 | 136 |
| Branch-pair index, eight categorical posterior draws | — | 210 | 210 | 192 |
| Branch-pair index, per-branch N probability bound | — | 364 | 210 | 192 |

At the default parent cap both parent methods proposed every reference hit.
Reducing the cap to 32 excluded 74 hits. The site procedures use different
proposal budgets. To evaluate ranking in all methods we looked up the
candidate's exact cutoff OCN in the exhaustive table and retained up to 64,
then applied the ωC threshold. The per-branch bound uses one list per site
containing every branch with N event probability ≥ 0.5. A quartet can reach
site PP 0.5 only if all four branches appear in that list. Shared destination
and the exact posterior contribution are checked during scoring.

In the moderate-effect panel, foreground ω was 0.9 on 20% of sites. A
topology-only selection of 24 stems (nine internal) yielded zero reference hits
in either the directional or relaxation simulations. Isolating the eight treated
tips yielded one hit across the three directional seeds and zero under relaxation.
Every method recovered the one directional hit. These sparse controls provide
little information about relative sensitivity.

## Effect of the PP threshold

In the diffuse-signal toy scenario, the same four branches had an event
probability of 0.55 each at 32 sites. Each site's contribution was about 0.092.
All 12 planted quartets passed the raw thresholds and zero passed when site
PP ≥ 0.5 was required. Across all quartets the corresponding totals were 51 and
39. The threshold deliberately removes this type of distributed support.

On the nine PEPC C4 stems, the default `all_cutoffs` search produced the
following hit counts with the same fit and the default OCN and ωC cutoffs.

| Arity | Codon model, PP 0 / 0.5 | Empirical urn, PP 0 / 0.5 |
| ---: | ---: | ---: |
| 2 | 33 / 33 | 20 / 20 |
| 3 | 32 / 31 | 26 / 25 |
| 4 | 18 / 18 | 17 / 17 |

Raw OCN, OCS, ECN, ECS and ωC agreed on shared rows. With PP 0.5 the
`ocn_bound` codon-model search recovered the same branch identities as full
enumeration at K=2, 3 and 4. An exact joint branch-pair search also preserved
the raw statistics within numerical tolerance.

## Reproduction

Environment: macOS arm64, Python 3.12.15, NumPy 2.5.5, SciPy 1.18.1,
IQ-TREE 3.1.4 and a CSUBST source tree based on `1aabfbd`.
Use seeds 1–12 and lengths 2,048 and 8,192 for the timing runs. The final two
fields are elapsed seconds and `ru_maxrss` (bytes on macOS).

```bash
OPENBLAS_NUM_THREADS=1 python reports/branch_search_review/branch_pair_index.py indexed 256 8192 1
OPENBLAS_NUM_THREADS=1 python reports/branch_search_review/branch_pair_index.py two_sites 256 8192 1
```

To reproduce the parent-cutoff comparison, run the following from the repository root.

```bash
PYTHONPATH=reports/branch_search_review python - <<'PY'
import site_intersection_pilot as pilot

for length in (2048, 8192):
    for seed in range(1, 13):
        state, quality, qn, qs, _ = pilot.realization("long", 256, length, seed)
        pairs = pilot.reference(state, quality, qn, qs, 2)
        assert len(pairs) == 32640 and not pilot.qualifying(pairs).any()
PY
```

The [regional panel script](region_panel.py) prepares the PGK inputs and
moderate cases. For the strong controls run its PGK simulation with the same
seeds, `--percent_foreground_sites 40 --foreground_omega 2
--foreground_branch_length_scale 8 --target_amino_acids random1`. Fit each
simulated alignment with `search`, using `sim/csubst.fa` as the alignment,
`sim/csubst_true_asr.treefile` as the rooted topology and `fit` as the
IQ-TREE output directory. From the repository root, pass the resulting case
directory (containing `sim` and `fit`) to the comparison script.

```bash
python reports/branch_search_review/branch_pair_index_codon.py \
  "$case_dir" "$new_output_dir" --isolate_tips "$inputs_dir/fg.tsv"
```

Omit `--isolate_tips` for the topology-only panel. Each run writes
`recovery.tsv`, the exhaustive tables and the per-branch destination
probabilities used to propose candidates. The observed scaling and codon
simulations cover K=4. Performance at higher K remains unmeasured.

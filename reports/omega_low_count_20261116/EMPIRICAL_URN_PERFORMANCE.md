# Empirical and legacy Wallenius urn runtime

Compare `search --expectation_method urn --asrv empirical` with
`--asrv each` before and after streaming the empirical observed counts and
training totals. The baseline is `631d3eb`. Each test uses a saved
IQ-TREE ECMK07 fit on the 71-tip PEPC topology and evaluates 8,516 branch
pairs, including sisters. The 520-codon input uses the mixed-rate
`rate_mixture-741` alignment from the [ECMK07 study](EMPIRICAL_URN_ECMK07.md).
The 1,200-codon input is `neutral-1013` from the
[distance study](EPISTASIS_DISTANCE.md). The comparison uses the usual
destination-based search statistics, with the optional branch table
disabled. CSUBST reuses all five IQ-TREE sidecars in each search.

The machine was an Apple M2 Max with 64 GiB of RAM on macOS arm64.
Editable CSUBST 1.17.5 used Python 3.12.15 and the same compiled
extensions in both source snapshots. The saved fits came from IQ-TREE
3.1.4. Search and BLAS used one thread. We ran the modes sequentially
and alternated their order. Each row below reports the median across
three measured runs after one untimed warmup per mode. Wall time covers
process startup, input loading, computation, and writing the search
outputs. RSS is the peak reported by BSD `time -l` for the process.

| Codons | ASRV | Before wall (s) | After wall (s) | Before peak RSS (MiB) | After peak RSS (MiB) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 520 | `each` | 6.39 | 6.39 | 670 | 682 |
| 520 | `empirical` | 6.82 | 3.58 | 707 | 350 |
| 1,200 | `each` | 11.85 | 11.78 | 1,440 | 1,422 |
| 1,200 | `empirical` | 14.07 | 6.27 | 1,400 | 429 |

Empirical wall time fell by 47.5% at 520 codons and 55.4% at 1,200.
With the optimized checkout, empirical wall time was 44% and 47% lower
than the corresponding `each` times. The peak empirical RSS fell by
50.5% and 69.3%. The legacy times stayed within 1%. Per-run variation
and warmups are available in the [recorded timings](empirical_urn_timing.tsv).

For the 1,200-codon baseline, profiling attributed 3.9 seconds to
CSR index sorting and 5.1 seconds to computing Wallenius inclusion
profiles under profiler overhead. The empirical path now accumulates
observed Gram matrices together with branch-by-site totals, N route
totals and S destination totals during endpoint inference. The folded
expectations use dense Gram products when the pair set is dense. The
Wallenius calculation chooses and records its numerical method in one
pass. Zero-draw contributions reuse the zero-initialized output.

Every one of the eight before/after `each` pairs had a byte-identical
branch-combination TSV. In each empirical before/after comparison, row
IDs, schema, categorical results, missing-value masks and the recorded
set of Wallenius methods matched. The empirical maximum absolute changes
were 1.1e-12 for observed counts and ECS, 3.3e-8 for `ECNany2spe`, and
2.7e-7 among all expected counts. Floating columns agreed at `rtol=2e-6,
atol=3e-12`, with the largest relative differences in ratios with small
denominators. The streamed sums change the last digits of some fold-level
inputs to the bounded site-weight optimizer.

## Reproduction

Use the script from the optimized checkout. Set `small_fa`, `small_tree`,
and `small_fit` to the 520-codon simulated alignment, sim tree, and saved
IQ-TREE prefix. Set `large_fa`, `large_tree`, and `large_fit` from the
1,200-codon neutral simulation. Fit prefixes exclude the `.iqtree`
suffix. Run once for a checkout of `631d3eb` and once for the optimized
checkout using a fresh output directory for each.

```bash
python reports/omega_low_count_20261116/benchmark_empirical_urn.py \
  "$work" --source "$checkout" \
  --case pepc520 "$small_fa" "$small_tree" "$small_fit" \
  --case pepc1200 "$large_fa" "$large_tree" "$large_fit" --repeats 3
```

## Batched one-draw profiles and bounded mean clocks

The `b782efd` profile for 1,200 codons spent 4.0 seconds computing
Wallenius inclusion profiles under profiler overhead. About 1.9 seconds
were in the approximate mean solver. The next change normalizes
fractional draws up to one in batches, and uses a bounded direct solve
of the same mean equation when all inclusion probabilities at the
solution are at most one half. More concentrated profiles and larger
depletion continue through the log-clock solver. Exact methods and
quadrature keep their existing numerical rules.

We reran the fixed-fit benchmark with the same two alignments, native
extensions and thread settings. Within each case and mode we alternated
the execution order of the two source snapshots. The table gives medians
over three runs per snapshot after a warmup. The paired measurements
carry the `b782efd` and `bounded_clock` labels in the
[recorded timings](empirical_urn_timing.tsv).

| Codons | ASRV | Previous wall (s) | New wall (s) | Previous peak RSS (MiB) | New peak RSS (MiB) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 520 | `each` | 7.04 | 7.24 | 694 | 683 |
| 520 | `empirical` | 3.91 | 3.35 | 354 | 350 |
| 1,200 | `each` | 12.80 | 12.86 | 1,412 | 1,337 |
| 1,200 | `empirical` | 6.64 | 4.93 | 433 | 432 |

The empirical wall time fell by 14.5% and 25.7% at the two sizes.
The legacy medians differed by at most 2.8% in the paired runs.
Across all 16 before/after TSV comparisons, including the
warmups, `any2spe` observed and expected count columns agreed exactly
at the 12-digit TSV precision. The maximum absolute change in any
count column was 1.1e-12. The fitted fold weights, missing-value masks,
and sets of Wallenius methods matched.

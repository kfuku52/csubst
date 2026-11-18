# Complete pooling of synonymous destinations in the empirical urn

Both estimators in this comparison fit tree-wide synonymous destination
rates and preserve each branch's observed S total. The earlier default,
`s_full`, predicts the branch's N mass from its N+S mass, amino-acid
opportunities and the fitted tree-wide N scale. The current `empirical_all`
default, `both_full`, allocates the branch's observed N total using the
same pooled amino-acid route profile. This conditions the overlap on how
much N change was inferred on each branch. The S estimates are identical.

## Accuracy on the independent PEPC panel

The three conditions each contain eight independently simulated 520-codon
alignments on the 71-tip PEPC tree. The reference is the expected count
from the generating Q, integrated over shared ancestral states. Each
condition has 67,016 eligible all-pair observations and 672 K=6 observations
from the nine designated stems. WAPE is summed absolute error divided by
summed generating expectation. Raw omega error is the RMSE of log1p(omega)
against the generating omega on pairs with both observed posterior counts
at least 1e-9. Smaller values indicate lower error.

| Condition | K=2 ECS WAPE, previous → s_full | K=6 ECS WAPE, previous → s_full | K=2 ECN WAPE, s_full → both_full | K=6 ECN WAPE, s_full → both_full | K=2 raw omega error, previous → s_full → both_full |
|---|---:|---:|---:|---:|---:|
| Unit rates | 0.364 → 0.261 | 1.281 → 0.505 | 0.213 → 0.313 | 0.413 → 0.601 | 1.442 → 1.499 → 1.579 |
| Mixed rates | 0.374 → 0.279 | 1.044 → 0.561 | 0.235 → 0.302 | 0.509 → 0.498 | 1.198 → 1.243 → 1.336 |
| Directional bias | 0.370 → 0.267 | 1.281 → 0.572 | 0.214 → 0.317 | 0.475 → 0.534 | 1.451 → 1.505 → 1.578 |

S pooling reduced K=2 ECS WAPE by 25–28% and K=6 ECS WAPE by 46–61%.
Raw pairwise omega error increased by 3.7–4.0% with inferred counts.
Using realized N and S counts on the subset where both were positive,
S pooling reduced the omega error by 0.9–2.0%. No K=6 combination had usable positive counts for raw
omega on this panel. Anchoring N to observed N increased all-pair ECN WAPE
by 29–48%, beyond the earlier 5% practical margin. In the mixed-rate
condition the small K=6 ECN improvement coexisted with a 29% all-pair loss.
[Independent estimates, paired intervals and support](pepc_pair_overlap_cv_summary.tsv)

The production CLI replay checks ECN/ECS sums, absolute errors, pair
support and raw omega squared errors against the independent implementation
for each seed at all pairs and K=2–6 stems. These are 48 searches and
144 seed-by-scope-by-arity comparisons for the branch-fixed `both_full`
default. The 24 exhaustive branch-pair tables also matched the original
observed-N candidate byte for byte. All 144 tables retained the observed
counts and ECS values printed by the earlier `s_full` default.
[Per-seed reference](pepc_pair_overlap_cv_results.tsv) · [Replay check](pepc_full_pooling_check.py)

## Single-threaded search time

CSUBST 1.17.5 on the working tree based on b782efd, Python 3.12.15,
NumPy 2.5.5, SciPy 1.18.1, Apple M2 Max, macOS 27.0.1 arm64. Each cell
shows the median wall time in seconds across six separate CLI processes,
followed by the range. The first run for each workload and method warmed
caches. The six measured rounds used all six method orders. Wall time
includes Python startup, joint ASR, the urn fit, scoring and writing
the cb tables. Each process read the same previously fitted IQ-TREE files.

| Workload | Previous | s_full | both_full |
|---|---:|---:|---:|
| Observed PEPC, 971 codons, 8,516 independent pairs | 4.07 (3.94–4.34) | 4.11 (3.98–4.45) | 4.06 (3.95–4.33) |
| Mixed-rate PEPC, 520 codons, 9 stems, K=2–6 | 2.87 (2.72–3.15) | 2.77 (2.71–3.08) | 2.84 (2.74–2.99) |

Median peak RSS (previous, s_full, both_full) was 392, 391, 391 MiB for
all pairs and 305, 305, 305 MiB for the stems. The N candidate saved
about 0.05 s on pairs and cost about 0.06 s on stems. Neither workload
shows a consistent speed advantage large enough to offset the N error.
Branch combinations and observed counts agreed across all versions. ECN
agreed between the previous and s_full versions and ECS agreed between
s_full and both_full. [Individual timings, warmups and peak RSS](pepc_full_pooling_timing.tsv)

## Reproduction

Use the independent panel from the [simulation protocol](PEPC_PAIR_OVERLAP_CV.md#reproduction)
and an editable CSUBST install. The stored method labels `current`, `s_full`
and `both_full` in that protocol refer to the three estimators above.
To reconstruct the historical controls in an isolated research copy, use
the `_pool_n_totals` allocation in `_score` with the all-branch training
fit for `s_full`. For `current`, also set `rates.s_draw_totals = None` in
`fit_all_branches` to use observed S destination counts.

Run the new default and verify it against the original independent estimates:

```bash
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/pepc_full_pooling_check.py "$panel" "$(mktemp -d)" --jobs 3
```

For timing, run each estimator in a fresh directory using a copied alignment
and its matching five IQ-TREE files (`.iqtree`, `.state`, `.log`, `.rate`,
`.treefile`), plus the matching checkpoint for ECMK07_GY0K. With the shell
variables set to absolute input paths and a selected package snapshot on
`PYTHONPATH`, the all-pairs timing command is:

```bash
/usr/bin/time -lp python -m csubst search --alignment_file "$alignment" --rooted_tree_file "$tree" \
  --iqtree_iqtree "$prefix.iqtree" --iqtree_state "$prefix.state" --iqtree_log "$prefix.log" \
  --iqtree_rate "$prefix.rate" --iqtree_treefile "$prefix.treefile" \
  --expectation_method urn --output_stat any2spe --exclude_sister_pair no --b no \
  --threads 1 --blas_threads 1 --float_digit 17 --random_seed 4099 --outdir out \
  --exhaustive_until 2 --max_arity 2
```

Use the bundled PEPC alignment and fit for all pairs. For the specified stems,
use the mixed-rate seed 4099 alignment and its ECMK07_GY0K+F+R4 fit, with
`--exhaustive_until 1 --max_arity 6 --foreground "$foreground" --fg_exclude_wg no --cutoff_stat OCNany2spe,0`.

## Source checks

On Python 3.12.15 the focused urn, ASRV and numerical parity selection passed
119 tests. With IQ-TREE 3.1.4 on PATH, `make test` passed 2,381 tests
in the main lane and five process tests. Six skips require optional torch
or gemmi. The eight warnings
came from the plotting tests. `make lint typecheck`, `make docs-check` and
`git diff --check` passed. Validation used the source tree and the local
native build. Installed-wheel and source-only fallback builds were outside
this estimator comparison.

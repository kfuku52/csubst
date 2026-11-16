# Optimizing exact any2spe counts across branch pairs

A [second benchmark](PAIR_PATH_COMPOSITION.md) measures the effect of choosing
the direction of propagation and composing transition matrices along paths.

The prototype evaluates the same joint branch-pair posterior as the first
benchmark. It sums the observed nonsynonymous and synonymous convergence
counts across sites after conditioning on the shared ancestral state and
rate category. The benchmark uses all eligible pairs from arity-two
`search` with its default sister-pair exclusion. [First benchmark](PERFORMANCE.md)

## Implementation

- At a terminal branch with a called codon, each convergence type has one
  possible destination. Propagate two vectors of parent-state probabilities
  along the path. The terminal–terminal contraction needs matching destination
  labels. The terminal–dense contraction selects the matching destination
  from the dense feature matrix before the dot product. Internal and ambiguous
  branches use all destination channels.
- Keep the parent-state dimension contiguous and propagate paths sharing an
  edge in a single matrix multiplication, in tiles of at most 16 paths.
  Process descendants before ancestors and release projections after their
  final use. The experimental benchmark streams four patterns at a time.
- Group identical fully called tip patterns and weight each by its original
  multiplicity. The simulated input sizes of 520, 520, and 1,000 columns
  become 478, 406, and 996 patterns, respectively, in PGK, 520-codon PEPC,
  and 1,000-codon PEPC. With ambiguous observations the prototype processes
  all columns and retains dense terminal features on the affected branches.

## Measurements

| Input | Pairs | Prior exact (s) | Optimized exact (s) | Speedup | Prior peak RSS (MiB) | Optimized peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PGK, 520 codons | 1,682 | 13.89 | 6.05 | 2.30× | 623 | 347 |
| PEPC, 520 codons | 8,446 | 40.50 | 15.73 | 2.58× | 1,574 | 609 |
| PEPC, 1,000 codons | 8,446 | 78.70 | 35.27 | 2.23× | 1,501 | 645 |

Times and memory are medians of three fresh-process trials after a warmup,
except for the prior 1,000-codon PEPC result, which is a single trial.
The four-pattern block size, single CSUBST and BLAS threads, platform,
versions, and cached fitted inputs match the first benchmark. Times include
input parsing and output. The two exact implementations calculate the
observed counts only. Current `search` also reports the expected counts.
The full default `search` took 2.85 s and 4.35 s on 520- and 1,000-codon
PEPC, respectively. [Per-trial measurements](pair_optimization_timing.tsv)

On 520-codon PEPC, reducing the block size to two patterns gave 16.67 s
and 407 MiB in a single trial. This setting trades about one second for
roughly 200 MiB compared with the four-pattern median.

The PGK profile of the original four-site-block prototype spent 3.40 s
propagating paths and 3.21 s packing, weighting, and copying the inputs
to its joint contractions, out of 12.28 s profile time. The compact-tip
prototype alone took 7.87 s on PGK and 25.34 s on PEPC (one trial per
input). Additional pilots compared full and split contractions, site
block sizes, the axis order, and grouping projections by edge. Factoring
the conditional transition into the site-independent transition and two
site-specific factors reduced retained buffers but took 21.57 s on
520-codon PEPC versus 16.87 s for the grouped contractions without tiles.
The selected implementation uses the grouped contractions and tiled
projections.

## Verification and reproduction

All 18,574 eligible pairs match the original exact totals in every trial
(`rtol=1e-9`, `atol=6e-12`). The largest absolute difference in a saved
count was about `1e-12`. The 520-codon PEPC result at block size two passed
the same comparison. Exhaustive state enumeration covers missing and
ambiguous tips, multiple rate categories including rate zero, a
multifurcation, and repeated patterns with unequal multiplicity. The
independent per-pair calculation checks 64 sampled pairs from each real
fit. The endpoint unit and numerical parity selection passed 85 tests.

Use the original fitted PGK and PEPC inputs. The results retain the fitted
IQ-TREE generator, branch lengths, rate classes, and search pair list, and
the runner checks the generator hash. Run from the repository root with
the editable CSUBST installation. [Fixture preparation](PERFORMANCE.md#reproduce)

```bash
work="$(mktemp -d)"
scripts=reports/omega_low_count_20261116
bench="$scripts/benchmark_pair_cost.py"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python "$scripts/verify_posterior.py"
python "$bench" "$work/optimized" "$panel" "$pepc_fit" --mode optimized --repeats 3
python "$bench" "$work/memory" "$panel" "$pepc_fit" --mode optimized --datasets pepc --block-size 2 --repeats 0
python "$bench" "$work/reference" "$panel" "$pepc_fit" --mode released --repeats 0
bench="$PWD/$bench"
(cd "$work/optimized/pgk" && python "$bench" --verify optimized-1.tsv)
(cd "$work/optimized/pepc" && python "$bench" --verify optimized-1.tsv)
(cd "$work/optimized/pepc1000" && python "$bench" --verify optimized-1.tsv)
for dataset in pgk pepc pepc1000; do
  python "$bench" --compare-tables "$work/reference/$dataset/released-0.tsv" \
    "$work/optimized/$dataset/optimized-1.tsv"
done
```

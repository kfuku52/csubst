# Composing paths for exact branch-pair convergence counts

A follow-up benchmarks compact terminal events when some sites are
ineligible because of missing data. [Masked-pair benchmark](MASKED_PAIR_PERFORMANCE.md)

The second optimization keeps the observed `OCNany2spe` and `OCSany2spe`
estimands and the fitted endpoint model from the first optimization.
[Previous benchmark](PAIR_OPTIMIZATION.md)

For a split whose one side contains only called terminal branches, each
terminal event has a single destination label at that site. Multiply its
two conditional feature vectors by the common-ancestor posterior, send
the resulting joint mass down the opposite subtree, and contract with
the corresponding destination at each focal branch. Applying the
conditional transition to mass in this direction gives the same inner
product as applying it to the event feature in the other direction.
Mixed pairs on splits with dense features on both sides continue to use
the original contraction. Sending mass down for all mixed pairs increased
runtime in the pilot.

For the remaining dense projections, only materialize a feature at the
nodes that participate in a contraction. Compose the intervening
conditional transitions and apply the product in one matrix multiplication.
On 1,000-codon PEPC, the number of distinct dense projections per site
block drops from 538 to 403 after the change in direction, then to 235
after composing transitions. The transition matrices stay in float64.
The normalization computes reciprocals once per state and uses the
existing log-space conditional kernel for rare rows.

## Measurements

| Input | Pairs | Previous exact (s) | Composed paths (s) | Previous peak RSS (MiB) | Composed peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PGK, 520 codons | 1,682 | 6.05 | 5.89 | 347 | 355 |
| PEPC, 520 codons | 8,446 | 15.73 | 11.96 | 609 | 565 |
| PEPC, 1,000 codons | 8,446 | 35.27 | 26.15 | 645 | 597 |

Entries are medians of three independent processes after one warmup,
with four site patterns per block. The two PEPC cases took 24% and 26%
less time. The PGK difference was small. The previous values use the
archived three-trial measurements from the same inputs and host.

On 1,000-codon PEPC, a single eight-pattern run took 23.45 s and
895 MiB. Two patterns took 30.15 s and 415 MiB. Four patterns gave
the measured balance of runtime and memory shown above.

The one-thread measurements use CSUBST 1.17.5 (`c9f5d89`), Python 3.12.15,
NumPy 2.5.5 with Accelerate, an Apple M2 Max, and the same cached IQ-TREE
3.1.4 models, alignments, eligible arity-two pairs, and default sister-pair
exclusion as the previous work. The timings include model loading and TSV
output. The exact prototype computes the observed counts. The current
`search` with its complete default outputs took 4.35 s on 1,000-codon PEPC.
[Per-trial measurements](pair_path_timing.tsv)

## Verification and reproduction

The four-pattern result matches the previous exact totals for all 18,574
eligible pairs in every trial (`rtol=1e-9`, `atol=6e-12`). The two- and
eight-pattern outputs pass the same check on 1,000-codon PEPC. The maximum
absolute difference across the saved tables is about `1e-12`.
Small-tree enumeration covers both propagation directions, ambiguous and
missing observations, and zero-rate categories. Additional checks exercise
an intermediate node skipped by composition and the log-space recovery
of a subnormal conditional row. The endpoint and numerical parity
selection passed 85 tests. An independent per-pair replay checks 64
sampled pairs from each fitted input.

Prepare the original fitted PGK and PEPC fixtures and run from the repository
root with the editable package installed. [Fixture preparation](PERFORMANCE.md#reproduce)

```bash
work="$(mktemp -d)"
scripts=reports/omega_low_count_20261116
bench="$scripts/benchmark_pair_cost.py"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python "$scripts/verify_posterior.py"
python "$bench" "$work/previous" "$panel" "$pepc_fit" --mode optimized --repeats 3
python "$bench" "$work/composed" "$panel" "$pepc_fit" --mode composed --repeats 3
for block in 2 8; do
  python "$bench" "$work/block-$block" "$panel" "$pepc_fit" --mode composed \
    --datasets pepc1000 --block-size "$block" --repeats 0
done
bench="$PWD/$bench"
for dataset in pgk pepc pepc1000; do
  python "$bench" --compare-tables "$work/previous/$dataset/optimized-1.tsv" \
    "$work/composed/$dataset/composed-1.tsv"
  (cd "$work/composed/$dataset" && python "$bench" --verify composed-1.tsv)
done
```

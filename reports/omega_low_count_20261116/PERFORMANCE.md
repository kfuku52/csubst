# Cost of exact branch-pair convergence posteriors

The [optimization follow-up](PAIR_OPTIMIZATION.md) measures a faster exact
implementation against the timings below.

The exact branch-pair estimator for `any2spe` is too expensive for the
`search` default on the tested workloads. Current `search` already uses
a joint parent–child posterior within each branch. Its `any2spe` counts
multiply the corresponding event probabilities across branches. The exact
pair calculation integrates each product over the two branches’ shared
ancestral state and site-rate category.

## Benchmark

Both calculations used the same simulated tips, fixed topology, fitted
`ECMK07_GY0K+F+R4` generator, branch lengths, and site-rate mixture within
each case. All runs used `max_arity=2` and the default exclusion of sister
pairs. The paired baseline runs `search` with `--output_stat any2spe --b no`,
including expected counts and the output tables. The exact prototype times
the observed `OCNany2spe` and `OCSany2spe` totals, including input loading
and output. It requires those fitted inputs and a pair list from `search`.

| Alignment | Codons | Pairs | `search` (s) | Exact (s) | Time ratio | `search` peak RSS (MiB) | Exact peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PGK, background | 520 | 1,682 | 1.67 | 13.89 | 8.3× | 209 | 623 |
| PEPC, site-rate mixture | 520 | 8,446 | 2.16 | 40.50 | 18.8× | 299 | 1,574 |
| PEPC, background | 1,000 | 8,446 | 2.91 | 78.70 | 27.1× | 360 | 1,501 |

The PGK and 520-codon PEPC results are medians of three timed trials after
one warmup. The wall times for the exact kernel, excluding setup and output,
were 12.98 s on PGK and 39.58 s on 520-codon PEPC. The repeated runs
alternated execution order. The 1,000-codon PEPC comparison was a single
trial per implementation, with a 77.71 s exact kernel time.

With the default output settings (`any2any,any2dif,any2spe`, `--b yes`),
the current search took 2.85 s on 520-codon PEPC and 4.35 s on
1,000-codon PEPC (three-trial medians).

For a 64-pair subset sampled with seed 940, the exact pass took median
wall times of 7.00 s on PGK, 12.58 s on 520-codon PEPC, and 23.77 s on
1,000-codon PEPC. Peak RSS fell to 359, 506, and 522 MiB, respectively.
The calculation still processes every site and propagates probabilities
on the paths needed to condition the sampled pairs.

The exact kernel groups pairs by common ancestor and evaluates weighted
matrix products in four-site blocks. Releasing branch-path projections
after their last use reduced peak memory relative to the pilot that stored
them for the entire block. The remaining work includes propagating codon
event probabilities to the common ancestor and combining branches at each
shared ancestral state and rate category.

The cache-all pilot used 902 MiB on PGK and 2,381 MiB on 520-codon PEPC.
The release schedule cut that peak by about a third in both cases. It
did not demonstrate a runtime reduction.

The [per-trial measurements](pair_timing.tsv) include process wall time,
peak RSS, and the exact kernel time separately. The benchmark ran on an
Apple M2 Max, macOS 27.0.1 arm64, with CSUBST 1.17.5 (`c9f5d89`), Python
3.12.15, NumPy 2.5.5 (Accelerate), and cached IQ-TREE 3.1.4 fits. Both
CSUBST and BLAS used one thread. Each timed run launched a fresh process.

## Numerical checks

The exact counts from the memory-saving and cache-all kernels agree on
all 18,574 eligible pairs across the three inputs (`rtol=1e-9`,
`atol=6e-12`). Repeated full-pair runs produced identical output tables. The
64-pair subset totals agree with those of the full-pair calculation. The
current search gives the same observed and expected `any2spe` counts with
its default outputs and with the outputs restricted to `any2spe`.

Both exact kernels also match exhaustive enumeration on a small tree.
The per-pair reference calculation checks the exact and factorized totals
on 64 pairs per input. These checks cover the observed-count prototype and
the timed cases.

## Reproduce

Use `PGK/baseline-101` and `PEPC/rate_mixture-101` from the
[sensitivity panel](../simulate_sensitivity_20261116/README.md#reproduce)
for `panel`. For `pepc_fit`, use `null-101` from the
[1,000-codon PEPC simulation](../simulate_expected_omega_20261116/README.md#reproduce)
with the input pair and `hybrid_fit` laid out as in the
[codon model refit](../simulate_expected_omega_20261116/CODON_MODEL_REFIT.md#reproduce).
Set `panel` and `pepc_fit` to the two fixture directory paths and run from
the repository root with an editable installation.

```bash
work="$(mktemp -d)"
scripts=reports/omega_low_count_20261116
bench="$scripts/benchmark_pair_cost.py"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python "$scripts/verify_posterior.py"
python "$bench" "$work/pilot" "$panel" "$pepc_fit" --mode compare --repeats 0
python "$bench" "$work/paired" "$panel" "$pepc_fit" --mode released --datasets pgk pepc
python "$bench" "$work/targeted" "$panel" "$pepc_fit" --mode targeted
python "$bench" "$work/default" "$panel" "$pepc_fit" --mode baseline --datasets pepc-default pepc1000-default
python "$bench" "$work/larger" "$panel" "$pepc_fit" --mode released --datasets pepc1000 --repeats 0
bench="$PWD/$bench"
(cd "$work/paired/pgk" && python "$bench" --verify released-1.tsv)
(cd "$work/paired/pepc" && python "$bench" --verify released-1.tsv)
(cd "$work/larger/pepc1000" && python "$bench" --verify released-0.tsv)
```

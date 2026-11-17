# Compact terminal branches with ineligible sites

For a terminal branch whose eligible sites each have a called codon, the
observed endpoint event can use one destination label per site. The pair
kernel keeps that representation when other sites on the branch are
ineligible, and sets their event features to zero. Original tip emissions,
including missing observations, still enter tree pruning. Ambiguous or
missing emissions on eligible sites use the dense feature representation.

## Measurement

| Alignment | Before wall time | After wall time | Before peak RSS | After peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Complete | 15.19 s | 15.09 s | 650 MiB | 650 MiB |
| Clade-masked | 18.73 s | 15.28 s | 664 MiB | 628 MiB |

The medians show an 18% reduction in masked runtime and a 5.5% reduction
in peak memory. On the complete alignment the difference was small. The
saved observed and expected counts agree for all pairs and trials. The
saved `dNC`, `dSC`, and `omegaC` pass the same numeric tolerance check.

The benchmark runs `search --pair_posterior joint --output_stat any2spe`
with the PEPC directional-bias seed 202. Its complete and clade-masked
alignments both have 520 codons and 71 tips. Both trials for each condition
use the same cached IQ-TREE fit and the same 8,516 branch pairs, including
sisters. The mask removes 208 codons from 18 tips. The benchmark loads
`endpoint_pair.py` from `fa7cd07` for the before runs and uses the working
checkout for the after runs. It measures fresh-process wall time and peak
RSS on Apple M2 Max, macOS arm64, Python 3.12.15, NumPy 2.5.5 with
Accelerate, and one CSUBST and BLAS thread. It warms up each method once
and alternates the order of subsequent runs.
[Mask preparation](MISSINGNESS.md#reproduce), [Per-trial measurements](masked_pair_timing.tsv)

## Reproduce

Run from the repository root with an editable installation. Set `case`
to the `biased-202` directory from the masking experiment.

```bash
work="$(mktemp -d)"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
git show fa7cd07:csubst/endpoint_pair.py > "$work/before.py"
python reports/omega_low_count_20261116/benchmark_masked_pairs.py \
    "$case" "$work/runs" "$work/before.py"
```

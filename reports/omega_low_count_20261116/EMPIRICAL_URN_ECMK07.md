# Default empirical urn versus fitted codon expectations on ECMK07 simulations

Nine PEPC alignments use the bundled 71-tip topology and 520 codon sites.
The simulator uses ECMK07 exchangeabilities and PEPC equilibrium codon
frequencies. We compute the simulator's N/S mass parameter directly from
that Q (0.605366042955631), which leaves the background N exchangeabilities
at their original scale. The scenarios use seeds 741, 852, and 963.
The baseline has unit rates. The mixture repeats rates 0, 0.2, 0.8,
and 3 at equal frequencies. The biased condition adds a preferred amino
acid on the designated branches at 20% of unit-rate sites. The null
reference comes from the simulator's background Q.

Search fits `ECMK07+F`, `ECMK07_GY0K+F`, `GY+F`, and `MGK+F3X4` on the
known topology, with `+R4` added in the mixed-rate condition. Each fit
estimates its parameters from the simulated tips. The codon and urn
searches reuse the same IQ-TREE fit, including the branch lengths and
rate categories. The paired searches agree on OCN and OCS and use the
same eligible pairs. Each urn run omits `--asrv` and verifies that the
resolved mode in provenance is `empirical`. Simulation foreground
labels are used to score the biased pairs after the searches.

## Pooled omegaC and individual pair error

Each cell gives `omegaCany2spe` from the pooled counts, with the pairwise
RMSE of `log(1 + omegaC)` in parentheses. RMSE uses the same finite pairs
across all eight combinations of fit and expectation. There are 25,131
eligible pair-replicates per background condition and 108 with both
branches biased. The common finite populations are 25,041, 25,093,
and 108 respectively.

| Ancestral-state fit | Expectations | Unit rate | Mixed rates | Both branches biased |
| --- | --- | ---: | ---: | ---: |
| Generating model | Generating ratio | 1.000 | 1.000 | 3.285 |
| True simulated ancestors | Realized ratio | 0.969 | 1.002 | 3.222 |
| ECMK07+F | Fitted codon Q | 1.034 (1.365) | 1.111 (1.307) | 3.394 (3.042) |
| ECMK07+F | Empirical urn | 1.061 (1.292) | 1.053 (1.206) | 3.579 (3.080) |
| ECMK07_GY0K+F | Fitted codon Q | 0.979 (1.361) | 1.086 (1.305) | 3.247 (3.037) |
| ECMK07_GY0K+F | Empirical urn | 1.061 (1.297) | 1.053 (1.210) | 3.577 (3.088) |
| GY+F | Fitted codon Q | 0.601 (1.274) | 0.667 (1.173) | 1.853 (2.816) |
| GY+F | Empirical urn | 1.068 (1.295) | 1.045 (1.173) | 3.572 (3.017) |
| MGK+F3X4 | Fitted codon Q | 0.645 (1.350) | 0.707 (1.236) | 2.065 (3.020) |
| MGK+F3X4 | Empirical urn | 1.062 (1.354) | 1.040 (1.221) | 3.609 (3.192) |

## Expected counts relative to the generating Q

Each cell gives the ECN and ECS ratios after summing the corresponding
counts. Each ratio has target one.

| Ancestral-state fit | Expectations | Unit rate | Mixed rates | Both branches biased |
| --- | --- | ---: | ---: | ---: |
| ECMK07+F | Fitted codon Q | 1.000 / 1.047 | 0.937 / 1.020 | 1.102 / 1.153 |
| ECMK07+F | Empirical urn | 0.937 / 1.005 | 0.864 / 0.891 | 0.988 / 1.090 |
| ECMK07_GY0K+F | Fitted codon Q | 1.033 / 1.022 | 0.950 / 1.009 | 1.131 / 1.132 |
| ECMK07_GY0K+F | Empirical urn | 0.937 / 1.004 | 0.865 / 0.890 | 0.989 / 1.090 |
| GY+F | Fitted codon Q | 2.540 / 1.549 | 1.996 / 1.325 | 2.974 / 1.701 |
| GY+F | Empirical urn | 0.969 / 1.051 | 0.940 / 0.978 | 1.006 / 1.110 |
| MGK+F3X4 | Fitted codon Q | 2.002 / 1.301 | 1.677 / 1.166 | 2.294 / 1.440 |
| MGK+F3X4 | Empirical urn | 0.962 / 1.030 | 0.921 / 0.942 | 1.000 / 1.098 |

For GY and MGK, the urn reduces the large ECN overestimate in the
fitted-Q calculation. The mixed-rate GY ECN RMSE falls from 0.13580
to 0.02883. With the matched ECMK07 fit, mixed-rate ECN RMSE rises from
0.02571 to 0.03109, and the pooled ratio benefits from ECN and ECS both
being low. On the directionally biased subset the fitted-Q omegaC has
lower pairwise log RMSE for all four ASR models. These comparisons use
one generating family, one topology, and three seeds per condition.
[Per-seed expected counts and loss](compare_empirical_urn_summary.tsv)

## Reproduction

Run from the CSUBST root with editable CSUBST 1.17.5 based on `c39484a`
and the empirical urn implementation in this checkout. The simulations
and fits ran on macOS arm64 with Python 3.12.15 and IQ-TREE 3.1.4. Set
`IQTREE_EXE` to the path to that executable.

```bash
work="$(mktemp -d)"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/compare_empirical_urn.py \
  "$work" --stage fit --iqtree "$IQTREE_EXE" --jobs 8
python reports/omega_low_count_20261116/compare_empirical_urn.py \
  "$work" --stage urn --jobs 8
```

The report uses `summary.tsv` from that work directory. To rescore
completed searches, run the same script with `--stage score`.

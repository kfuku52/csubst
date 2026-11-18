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
rate categories. Within each fit, the three searches agree on OCN, OCS and eligible
pairs. The earlier urn (`--asrv empirical`) trains outside the focal pair's
folds. The new default (`empirical_all`) uses all 138 eligible branches,
including the focal pair. Its 36 additional searches omitted `--asrv`;
each provenance file confirms the all-branch training scope. Foreground
labels are used only to select the biased pairs for evaluation.

## Pooled omegaC and individual pair error

Each cell gives `omegaCany2spe` from the pooled counts, with the pairwise
RMSE of `log(1 + omegaC)` in parentheses. RMSE uses the same finite pairs
across all twelve combinations of fit and expectation. There are 25,131
eligible pair-replicates per background condition and 108 with both
branches biased. The common finite populations are 25,041, 25,093,
and 108 respectively.

| Ancestral-state fit | Expectations | Unit rate | Mixed rates | Both branches biased |
| --- | --- | ---: | ---: | ---: |
| Generating model | Generating ratio | 1.000 | 1.000 | 3.285 |
| True simulated ancestors | Realized ratio | 0.969 | 1.002 | 3.222 |
| ECMK07+F | Fitted codon Q | 1.034 (1.365) | 1.111 (1.307) | 3.394 (3.042) |
| ECMK07+F | Urn `empirical` (four folds) | 1.061 (1.292) | 1.053 (1.206) | 3.579 (3.080) |
| ECMK07+F | Urn `empirical_all` | 0.984 (1.280) | 1.045 (1.204) | 3.303 (3.059) |
| ECMK07_GY0K+F | Fitted codon Q | 0.979 (1.361) | 1.086 (1.305) | 3.247 (3.037) |
| ECMK07_GY0K+F | Urn `empirical` (four folds) | 1.061 (1.297) | 1.053 (1.210) | 3.577 (3.088) |
| ECMK07_GY0K+F | Urn `empirical_all` | 0.984 (1.285) | 1.045 (1.208) | 3.302 (3.067) |
| GY+F | Fitted codon Q | 0.601 (1.274) | 0.667 (1.173) | 1.853 (2.816) |
| GY+F | Urn `empirical` (four folds) | 1.068 (1.295) | 1.045 (1.173) | 3.572 (3.017) |
| GY+F | Urn `empirical_all` | 0.992 (1.282) | 1.036 (1.171) | 3.302 (2.995) |
| MGK+F3X4 | Fitted codon Q | 0.645 (1.350) | 0.707 (1.236) | 2.065 (3.020) |
| MGK+F3X4 | Urn `empirical` (four folds) | 1.062 (1.354) | 1.040 (1.221) | 3.609 (3.192) |
| MGK+F3X4 | Urn `empirical_all` | 0.987 (1.341) | 1.031 (1.219) | 3.336 (3.170) |

## Expected counts relative to the generating Q

Each cell gives the ECN and ECS ratios after summing the corresponding
counts. Each ratio has target one.

| Ancestral-state fit | Expectations | Unit rate | Mixed rates | Both branches biased |
| --- | --- | ---: | ---: | ---: |
| ECMK07+F | Fitted codon Q | 1.000 / 1.047 | 0.937 / 1.020 | 1.102 / 1.153 |
| ECMK07+F | Urn `empirical` (four folds) | 0.937 / 1.005 | 0.864 / 0.891 | 0.988 / 1.090 |
| ECMK07+F | Urn `empirical_all` | 0.959 / 0.955 | 0.891 / 0.912 | 1.022 / 1.041 |
| ECMK07_GY0K+F | Fitted codon Q | 1.033 / 1.022 | 0.950 / 1.009 | 1.131 / 1.132 |
| ECMK07_GY0K+F | Urn `empirical` (four folds) | 0.937 / 1.004 | 0.865 / 0.890 | 0.989 / 1.090 |
| ECMK07_GY0K+F | Urn `empirical_all` | 0.960 / 0.954 | 0.892 / 0.911 | 1.022 / 1.040 |
| GY+F | Fitted codon Q | 2.540 / 1.549 | 1.996 / 1.325 | 2.974 / 1.701 |
| GY+F | Urn `empirical` (four folds) | 0.969 / 1.051 | 0.940 / 0.978 | 1.006 / 1.110 |
| GY+F | Urn `empirical_all` | 0.993 / 1.000 | 0.971 / 1.002 | 1.041 / 1.061 |
| MGK+F3X4 | Fitted codon Q | 2.002 / 1.301 | 1.677 / 1.166 | 2.294 / 1.440 |
| MGK+F3X4 | Urn `empirical` (four folds) | 0.962 / 1.030 | 0.921 / 0.942 | 1.000 / 1.098 |
| MGK+F3X4 | Urn `empirical_all` | 0.985 / 0.980 | 0.951 / 0.965 | 1.033 / 1.048 |

For all four ASR fits, switching from four-fold to all-branch training
brings pooled omegaC closer to the generating ratio in all three conditions.
With bias on both branches, the range across fits goes from 3.572–3.609
to 3.302–3.336 (target 3.285). The pairwise log RMSE also decreases in
each of the 36 matched seed-by-fit comparisons, but the gain in mixed rates
is small (about 0.002 after pooling). In the biased subset, fitted codon Q
still gives lower pairwise log RMSE than either urn for all four ASR fits.
The pooled ratio and pairwise error measure different aspects of accuracy.

The mixed-rate ECN/ECS deficits persist for the matched ECMK07 fit
(0.891 / 0.912 with all branches); when both are underestimated their
ratio can look well calibrated. With GY, the large mixed-rate ECN excess
under fitted Q (1.996) is largely removed by the all-branch urn (0.971).
All comparisons are for K=2 on one topology, one generating codon model,
and three seeds per condition; the four ASR fits reuse the same alignments.
[Per-seed expected counts and loss](compare_empirical_urn_summary.tsv)

## Reproduction

The original simulations and fits used editable CSUBST 1.17.5 based on
`c39484a`, macOS arm64, Python 3.12.15 and IQ-TREE 3.1.4. The all-branch
extension used the working tree based on `b782efd` with `empirical_all`
implemented, and reused copies of the original alignments, fits, and older
search outputs. Rescoring reproduced all 72 archived per-seed/method rows
unchanged. To run the complete comparison from the repository root, set
`IQTREE_EXE` to the path to that executable.

```bash
work="$(mktemp -d)"
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python reports/omega_low_count_20261116/compare_empirical_urn.py \
  "$work" --stage fit --iqtree "$IQTREE_EXE" --jobs 8
python reports/omega_low_count_20261116/compare_empirical_urn.py \
  "$work" --stage urn --jobs 8
```

The report uses `summary.tsv` from that work directory. To extend a copy of
the old two-method panel, run only `--stage empirical_all`; to rescore
all three completed searches without fitting, use `--stage score`.

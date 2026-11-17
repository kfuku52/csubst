# Missing codon blocks in one PEPC clade

Six existing PEPC simulations provide known endpoint states for a test of
clustered missing observations. Three use the background Q throughout.
Three favor a derived amino acid on foreground branches at 20% of sites.
Each condition uses seeds 101, 202, and 303, 71 tips, 520 codons, and a
uniform generating site rate. The study uses the original known topology.
[Simulation setup](../simulate_sensitivity_20261116/README.md)

Mask a contiguous block of 208 codons beginning at alignment column 174
in the clade whose tip count is closest to one-quarter of the tree. This
removes 10.1% of the tip codons, all from the same 18 tips. The
complete inputs use the saved IQ-TREE 3.1.4 `ECMK07_GY0K+F+R4` fits.
The masked inputs receive new fits using the same model and seeds. Within
each fit, the CLI compares `--pair_posterior separate` and `joint` with
the same pair list, Q, branch lengths, rate mixture, and expected counts.
We retain sister pairs and report them separately. Subroot rows with
unavailable expected counts are excluded from the summary.

A separate numerical replay holds the complete-data Q, branch lengths,
and rate mixture fixed while masking the same sites. This separates the
effect of losing tip observations from the combined effect of missingness
and refitting. Replaying the complete data reproduces the saved observed
and expected counts. The CLI validates the fitted frequencies against its
alignment and rejects an unmodified complete-data IQ-TREE bundle on the
masked input, so the fixed-model control evaluates the supplied model
directly.

True N and S counts use the simulated tip and internal sequences and
exclude sites where either edge has no observed descendants. The truth on
complete data reproduces every exported true count. This report scales the
Q-based null totals by each pair's retained fraction. This is exact for
the homogeneous background-Q runs. For the biased runs it approximates
site-specific null contributions, since the foreground generator can
change upstream ancestral-state distributions. The expectation follow-up
removes the generating contribution of the omitted sites directly.
[Shared expected counts](SHARED_EXPECTED.md)

Pooled summaries sum each count before taking ratios. Rows sharing branches or an
alignment are dependent, so results are descriptive across these seeds.

## Results

For each scenario and input condition, the summary retains 25,131
pair-replicates, including 207 sister pairs across the three seeds.
The block changes site eligibility for 11,601 pair-replicates per
scenario. Both search modes use the same eligibility target.

| Generating scenario, masked and refitted | True S | Observed S, separate → joint | S RMSE, separate → joint | True N | Observed N, separate → joint | N RMSE, separate → joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Background Q, sisters | 90 | 22.93 → 96.41 | 0.885 → 0.784 | 6 | 0.075 → 1.560 | 0.1700 → 0.1664 |
| Directional bias, sisters | 83 | 24.66 → 100.56 | 0.728 → 0.672 | 0 | 0.102 → 2.010 | 0.0026 → 0.0250 |

The sister-pair S RMSE improved in five of the six masked replicates.
For differently parented pairs, S RMSE fell by about 1% in both scenarios
(0.253 → 0.250 and 0.255 → 0.252). The zero realized N count in the biased
sister sample is compatible with positive fitted posterior means. The
joint calculation assigned more probability to shared-parent changes,
increasing N RMSE in every biased seed. In the masked fixed-fit replay,
the same 207 biased sister pairs had S RMSE 0.732 → 0.673 and N RMSE
0.0027 → 0.0256. This difference therefore also occurs without refitting.
The site-level follow-up separates uncertainty in parent amino acids,
site rates, branch lengths, and the generator. [Sister-pair N diagnosis](SISTER_N.md)

In the background-Q mask, pooled ECN relative to the generating-Q
expectation was 1.068 with the complete-data fit and 1.032 after refitting.
The ECS ratio moved from 1.001 to 1.017. The pooled realized omegaC using
the Q-based expected totals was 0.850 after masking. The fitted values
with the refit were 0.818 (separate) and 0.813 (joint), so the improvement
in S counts did not carry over to this ratio. The joint option uses the
same expected counts as the separate option within each fitted model.

For pairs with both branches in the biased foreground, the pooled realized
omegaC calculated from known endpoints and the Q-based null was 4.68 on
the full alignment and 6.40 on the retained sites. The site-eligibility
change alone therefore alters the realized target. [Per-seed summaries](missingness_summary.tsv)

## Reproduce

Run from the repository root with Python 3.12.15 and the editable CSUBST
installation. The analysis used CSUBST 1.17.5 (`109c94a`) on macOS arm64 and one
CSUBST and BLAS thread per job. Set `panel` to the directory from the
PEPC simulation panel and `IQTREE_EXE` to the IQ-TREE 3.1.4 executable.
The commands write result tables and IQ-TREE intermediates to a new workspace.

```bash
work="$(mktemp -d)"
scripts=reports/omega_low_count_20261116
export MPLBACKEND=Agg VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python "$scripts/missingness.py" "$panel" "$work/cases" "$work/refit.tsv" --iqtree "$IQTREE_EXE"
python "$scripts/fixed_mask_replay.py" "$panel" "$work/fixed.tsv"
```

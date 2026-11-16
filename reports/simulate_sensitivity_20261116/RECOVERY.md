# Recovery of stalled ECMK07 multiplier fits

The [sensitivity panel](README.md) found 14 PGK simulations where IQ-TREE
left the estimated nonsynonymous multiplier at 1. All 14 old inferred fits
were copied into fresh IQ-TREE directories with matching provenance and then
analyzed with the revised `csubst search`. These runs used CSUBST 1.17.5 at
base revision `ed229ac` with the recovery change, Python 3.12.15,
IQ-TREE 3.1.4, and macOS arm64.

A nearby multiplier improved the conditional likelihood in all 14 cases.
Refitting with `--inc-zero-freq --min-freq 0.0001` recovered an improved
fit in every case, with log-likelihood gains of 5.86–540.93. For the
1,651 eligible pairs in each replicate, the summed ECN moved closer to the
Q reference in all 14 cases. The summed ECS moved closer in 13 cases.
[Per-seed recovery counts](omega_recovery_by_seed.tsv)

| PGK case | Q ECN / ECS | Original ECN / ECS | Recovered ECN / ECS |
| --- | ---: | ---: | ---: |
| Baseline 202 | 7.51 / 530.91 | 44.06 / 279.70 | 8.41 / 595.33 |
| Strong constraint 303 | 0.64 / 695.17 | 43.44 / 233.51 | 0.56 / 744.56 |

Separate CLI checks used PGK baseline seed 202 with an empty fit directory
and IQ-TREE 3.1.4, and with a stalled 2.4.0 cache. The checks recovered
multipliers of 0.28459 and 0.28328 respectively. The PEPC baseline seed
101 cache was reused without running IQ-TREE. Explicit frequency-floor fits
from both versions could also be supplied directly. IQ-TREE 3.1.4 records
the fitted codon frequencies in its checkpoint. IQ-TREE 2.4.0 records them
in its verbose log.

The verification covers the exact-1 stalls observed on these datasets.
The likelihood probes test local changes with other fitted inputs held
fixed. The study tests neither convergence from an arbitrary fitted
multiplier nor calibration of pair-level uncertainty.

## Reproduce on a preserved inferred fit

In a fresh directory, copy the simulated alignment and its rooted tree.
Copy the corresponding `.iqtree`, `.log`, `.state`, `.rate`, `.treefile`,
`.ckp.gz`, and `.state.csubst-manifest.json` under the prefix reported
by `runtime.infer_iqtree_output_prefix` for the new alignment path and
IQ-TREE directory.

```bash
python -m csubst search --alignment_file sim.fa --rooted_tree_file sim_true_asr.treefile \
  --iqtree_exe "${IQTREE_EXE:-iqtree}" --iqtree_outdir csubst_iqtree \
  --random_seed 202 --max_arity 2 --exhaustive_until 2 --exclude_sister_pair no \
  --output_stat any2spe --float_digit 12 --b no
```

The search log records whether CSUBST accepted a frequency-floor refit.
The completed search run record, endpoint-model JSON, and pair TSV supply
the data used in the comparison.

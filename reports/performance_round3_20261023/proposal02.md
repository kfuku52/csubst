# Proposal 02: NumPy classified endpoint events

The broad grouped-codon implementation is rejected. It does not improve CPU time on the realistic 61-state S+N and recoded workloads. A selected-pair specialization for the real N-only 20-state structural endpoint route is promising. It preserves the grouped fallback; the parent completed exact-patch kernel verification; repository integration checks remain required.

Baseline: `624bb8a`. Python 3.14.7; native library threads set to one before imports; optional endpoint event extension disabled for these fallback measurements. Every workload received one excluded warmup and seven alternating baseline/candidate runs, serialized with this report directory's benchmark lock. Inputs are deterministically generated likelihoods at the bundled PGK (417 sites) and PEPC (971 sites) shapes, including resolved, partially ambiguous, and fully missing observations. These are kernel measurements, not full analyses or scientific calibration.

| Workload | Baseline wall s | Candidate wall s | Wall speedup | CPU speedup |
| --- | ---: | ---: | ---: | ---: |
| PGK_shape_SN64 | 0.3125 | 0.2484 | 1.258 | 0.997 |
| PEPC_shape_SN64 | 0.6609 | 0.8851 | 0.747 | 0.953 |
| PEPC_shape_SN8 | 0.5890 | 0.7122 | 0.827 | 0.881 |
| PEPC_shape_SN256 | 1.5278 | 1.3871 | 1.101 | 0.983 |
| PEPC_shape_S_only64 | 0.6822 | 0.0280 | 24.394 | 16.196 |
| PEPC_shape_recode64 | 0.4497 | 0.4967 | 0.905 | 0.824 |

The host also ran unrelated repository jobs. Interleaving reduces drift, and process CPU time makes the broad rejection clear despite a misleading wall-time improvement in the PGK-shaped case. The S-only speedup is not an adoption argument: no current production `_build` route requests S alone (the codon side of 3Di also requests AA).

The candidate gathers only classified codon pair products and aggregates duplicate reported events with `numpy.add.reduceat`. It preserves the native branch and delegates arbitrary, fractional, and overlapping mappings to the original `_events` implementation. Across 36 comparisons (3/61 states; 0/1/8/64/129/256 sites; contiguous/strided read-only arrays; recoding), maximum absolute difference was 4.44e-16 at the existing rtol=1e-14/atol=1e-15. Five independent 3-state enumeration checks covered partition, identity, singleton, fractional, and overlapping mappings. A native sentinel check verified unchanged dispatch.

Fresh-process peak RSS is retained in JSON. At 4,096 sites the rejected full S+N candidate used 467.5 MB versus 499.1 MB baseline; the unrouted S-only case used 211.6 MB versus 446.4 MB. These are one-shot stress measurements including imports, not repeated memory statistics or representative default-block improvements. No production memory benefit is claimed.

Reproduce from the repository root with an extracted baseline source tree:

```sh
python reports/performance_round3_20261023/proposal02.py --baseline-root BASELINE_TREE
python reports/performance_round3_20261023/proposal02.py --baseline-root BASELINE_TREE --phase narrow
```

`--baseline` and `CSUBST_PERF_BASELINE` are equivalent explicit baseline selectors; the session's temporary marker is only a convenience. The source tree must contain the original `csubst/endpoint_io.py`. No production modules or test files were edited. Native output numerics, full CLI tables, installed artifacts, and statistical calibration have not been verified by this prototype report; any adopted specialization needs the parent's affected checks.

## Narrow production route

`endpoint_io._build(structural=True)` requests only N with a 20-state identity mapping. Here every selected codon/state pair maps to exactly one reported event, so the new path needs no grouped reduction or identity matrix multiplication. `proposal02.patch` limits the specialization to N-only one-to-one permutations; grouped, fractional, overlapping, and native paths retain the original code.

| 20-state structural workload | Baseline CPU s | Final patch CPU s | CPU reduction |
| --- | ---: | ---: | ---: |
| 3di20_identity64 | 0.309024 | 0.188241 | 39.1% |
| 3di20_identity256 | 0.376731 | 0.227430 | 39.6% |

The parent reran `--phase narrow` against the exact final patch with seven alternating
repetitions, one warmup, 971 deterministically generated sites and 160 complete site
passes per sample. Float64 event arrays were bitwise equal. The CPU ranges separated
for both block sizes: 0.291–0.357 vs 0.174–0.218 s, and 0.325–0.392 vs 0.219–0.245 s.
Wall ranges overlapped on this busy host; no dependable wall or whole-CLI speedup is claimed.

Final validation passed 96 identity/permutation and dtype cases, including mixed dtypes,
float32/float64/int16/complex128, empty/one-site, strided and read-only inputs, plus
independent small-state permutation enumeration. The broader mapping/native-dispatch
controls also passed. No narrow-path memory benefit is claimed.

Measured final source SHA-256: `73f672608c73f7217d563d9d65f3439cd9d62b95fa36ca79e8994a07669e36df`.
Unapplied patch SHA-256: `1489632a070c5099fece9297a17d4019db28231bff31eb471cc723c3590fd05d`.
The initial narrow prototype differed in dtype handling; its preliminary timing values
are superseded by the exact-patch measurements in the saved JSON. Required integration
and source/native/fallback checks are recorded by the parent in this round’s main report.

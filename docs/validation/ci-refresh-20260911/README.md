# CI fixture and source-artifact refresh

The 1.16.1 [CI run](https://github.com/kfuku52/csubst/actions/runs/34591725612)
and its 1.16.0 [predecessor](https://github.com/kfuku52/csubst/actions/runs/34563902544)
had the same missing-sdist-support and PGK/PEPC reference failures. The native
3Di integration changes did not introduce those failures.

The parity snapshots still described marginal substitution posteriors with
`--drop_invariant_tip_sites tip_invariant`. Current defaults use joint
posteriors and retain invariant-tip sites. Under the current code, explicitly
selecting the old settings reproduces the old snapshots (`legacy.tsv`).
The current defaults agree between compiled and Pure-Python execution
(`native.tsv`, `fallback.tsv`). All branch IDs, category counts and omegaC
values are checked. Scientific unit/integration checks additionally compare
the underlying joint inference against independent enumerations and pruning.

These local numerical checks used the ARM64 GeneGalleon validation runtime,
Python 3.12.14, and the installed CSUBST wheel containing commit 6963896's code.
They ran the public `.github/scripts/sites_parity_check.py --installed
--numerical-only` commands with `CSUBST_STRICT_EXTENSIONS=1` or
`CSUBST_DISABLE_EXTENSIONS=1`. The legacy control appended
`--substitution_posterior marginal --drop_invariant_tip_sites tip_invariant`
to both analyze and sites commands and checked the former snapshots.
Concurrent build activity makes those local timings unsuitable as baselines;
only the scientific columns are retained here.

The source distribution now includes the Python tools and null-pilot script
used by its tests. Its complete extracted fallback suite passes 2,210 tests
in the parallel lane and 4 process tests (43 skips). Artifact validation
requires the formerly omitted scripts as well.

A separate, explicitly requested `record-parity-baseline` job collects three
Linux hosted-runner measurements after verifying the current scientific
references. The normal parity job retains its absolute and baseline-relative
limits throughout baseline collection. See TESTING.md for the procedure.

## Linux baseline

The [manual run](https://github.com/kfuku52/csubst/actions/runs/34592796250)
on commit `7a4079d195101c009eeb71b0c50fc298335d2856` completed
`record-parity-baseline` successfully. All three installed-wheel replicates
passed the scientific checks on Ubuntu/Python 3.12. The raw artifacts are
`replicate-1.tsv` through `replicate-3.tsv`; the versioned baseline uses
median wall times and maximum peak RSS from these three runs.

| Dataset | Analyze median seconds | Analyze peak KiB | Sites median seconds | Sites peak KiB |
| --- | ---: | ---: | ---: | ---: |
| PGK | 2.62 | 214116 | 7.11 | 371364 |
| PEPC | 7.28 | 358772 | 17.17 | 908180 |

The historical baseline was measured for different inference/filter defaults,
so this is a new reference workload, not a performance improvement claim.
The normal parity job in that same run passed the scientific checks but
rejected the old PEPC time/RSS baseline; it was not bypassed by collection.
All absolute limits and the 2.0 wall-time/1.75 RSS multipliers remain unchanged.
All other CI jobs, including full native and fallback suites, passed.

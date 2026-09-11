# Issue #46: current implementation and remaining calibration work

2026-09-10. Issue: <https://github.com/kfuku52/csubst/issues/46>.
The omegaC P-value feature is implemented, but this issue should stay open if
its goal is a generally calibrated biological test of excess convergence.

Commit review, 2026-09-11: this report retains the original experiment and
revision below. The current checkout passed the 16 count/pipeline/calibration
script tests and `make lint typecheck`. The subsequent pipeline report's 400
archived replicate records reproduce its aggregate, and its 83,200 retained
rows reproduce the rejection/selection summaries and an independent BH check.
The full simulation was not rerun; these historical results do not validate
later changes to joint inference.

## Status

- `calc_omega_pvalue` supplies conditional fitted-count P/Q values.
- Fixed/refitted pseudocount transformations and independent-null long-tail
  references have already been implemented. Documentation describes their
  scope in [PSEUDOCOUNT_PVALUES](../../docs/PSEUDOCOUNT_PVALUES.md).
- Independent full-pipeline simulations covering fractional posterior mass,
  ASR/model fitting and search selection are still needed. A successful
  known-count-null check cannot establish those properties. `scan` tests a
  different event-level hypothesis and does not close this omegaC issue.

## Changes in this audit

Added a seeded, independent Poisson-atom null/power regression, collected by
pytest. It covers raw statistics, fixed symmetric alpha=1, and alpha=1 with
independent-null long-tail maps. Sparse/dense S scenarios test excess rejection;
a fourfold N-count alternative tests that the implementation can detect signal.
Independent exposures are sampled log-uniformly between 0.5 and 2; each row has
disjoint branch IDs and independent atom observations. Expectations remain
known and fixed during inference. There is no ASR or selection in this model.
The reported statistic is any2spe; requesting any2dif alongside it exercises
the production joint Poisson count engine. It does not exercise the marginal
hypergeometric/Poisson samplers or data-dependent smoothing.

Null regression gates use a one-sided exact binomial test against 5%, with
0.001/6 per prespecified null case. The power gate is 50% for this fixed strong
alternative. These are regression alarms, not proof of calibration. Three
runtime repetitions reuse identical observations/seeds and are not counted as
additional independent statistical trials. Full P-value hashes verify repeat
identity. Independent-null map training uses 1,000 additional draws per row.

The historical PGK sensitivity script previously inherited the new
`independent_null` default despite historical empirical-calibration limits.
It now explicitly selects `empirical` and seed 46. Its measured rejection
fractions and broad regression caps are not FPR estimates: PGK is an observed
alignment with dependent branch combinations. No caps were loosened.

## Reproduction

Run from the repository root with an installed development environment:

```bash
python .github/scripts/omega_count_null_check.py --trials 512 --draws 999 \
  --repeats 3 --output reports/issue46_20260910/count_null.json
python .github/scripts/omega_pvalue_calibration_check.py \
  --output reports/issue46_20260910/summary.tsv \
  --runtime-output reports/issue46_20260910/runtime.tsv \
  --workdir /tmp/csubst_issue46_empirical_20260910 --niter 100
python -m pytest -q tests/unit/test_omega_count_null_script.py
make test lint typecheck
```

Runtime is descriptive, not a before/after speedup claim: production numerical
code was not changed. Peak RSS in the count report covers the whole process,
not individual settings. PGK uses bundled alignment/tree/precomputed IQ-TREE
outputs, one thread, 1,620 branch combinations, hypergeom, and 100 test draws.
The initial implicit independent-null PGK run was interrupted after 254.07 s
in its first condition; it did not complete and is not a timing comparison.

## Measured results

Python 3.10.14; macOS-26.6.2-x86_64-i386-64bit; NumPy 1.26.4, SciPy 1.15.2. Current checkout commit: `1a83b170816fa5bc01057dfe3f72c7bfc1927589` plus this audit diff.

| Setting | Regime | Rejected / 512 | Rate (95% CI) | Median seconds (range) |
| --- | --- | --- | --- | --- |
| raw | sparse_S | 0 | 0.00% (0.00–0.72%) | 0.48 (0.46–0.48) |
| raw | dense | 27 | 5.27% (3.50–7.58%) | 0.60 (0.59–0.64) |
| raw | enriched | 424 | 82.81% (79.26–85.98%) | 0.57 (0.57–0.58) |
| symmetric | sparse_S | 18 | 3.52% (2.10–5.50%) | 0.42 (0.42–0.43) |
| symmetric | dense | 25 | 4.88% (3.18–7.12%) | 0.59 (0.58–0.59) |
| symmetric | enriched | 467 | 91.21% (88.42–93.52%) | 0.59 (0.57–0.64) |
| independent_null | sparse_S | 18 | 3.52% (2.10–5.50%) | 8.80 (8.55–9.57) |
| independent_null | dense | 23 | 4.49% (2.87–6.66%) | 9.77 (9.41–9.96) |
| independent_null | enriched | 471 | 91.99% (89.29–94.19%) | 9.79 (9.76–10.18) |

Whole-process peak RSS: 334.4 MiB. Times use 999 test draws, 512 rows, two jointly generated categories, and three identical seeded repetitions. Other work may run on this shared host; these are descriptive costs, not isolated speedup measurements.

All six null cases passed the prespecified excess-rejection gate. The three alternative cases exceeded 50% power. The 95% intervals are pointwise, without multiplicity adjustment. Sparse-S raw inference was strongly conservative. Results support only this known-mean count model.

[Machine-readable count results](count_null.json), [PGK rejection fractions](summary.tsv), [PGK runtime/RAM](runtime.tsv). PGK completed all four historical sensitivity checks (3.17–3.97 s each, maximum RSS 245.1–255.1 MiB). Its any2spe rejection fraction remained 23.8–24.0% with min_sub_pp=0.05; this real-data result cannot be interpreted as FPR.

## Verification

Full sequential suite: **2,095 passed, 5 skipped** in 76.20 seconds on Python
3.10.14, including all ten new regression tests. Three skips require PyTorch
>=2.6 and two require gemmi. One existing requests dependency warning remains.
`make test` could not start its parallel lane because pytest-xdist is absent;
the supported sequential `python -m pytest -q` completed instead. The configured
Python 3.12 environment lacks pytest, so the Python 3.12 verification lane was
not run. Ruff, repository hygiene, documentation validation and all configured
mypy targets passed. No production numerical implementation was changed; no
commit, push, issue comment or closure was performed.

[Full test log](full_tests.log), [static checks](final_checks.log),
[type checks](typecheck.log), [input/source hashes](input_source_sha256.json).

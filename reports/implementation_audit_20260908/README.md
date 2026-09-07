# Implementation audit, 2026-09-08

Audited the clean `master` worktree at `5713ffb` (CSUBST 1.14.10).
The review combined the full test suite, static checks, inspection of numeric
identifier parsing, table conversion, FASTA I/O, empirical statistics, CLI
input protection, resource locking, and cache validation, plus targeted
boundary cases. This is evidence for the exercised behavior, not a claim
that every execution path is defect-free.

Environment: macOS 26.5.2 ARM64, Python 3.14.0, NumPy 2.4.2,
pandas 2.3.3, SciPy 1.17.1. Existing compiled extensions were available.

## Confirmed defects and fixes

| Defect | Reproduction | Corrected behavior |
| --- | --- | --- |
| Branch/site sorting happened before numeric conversion | Mixed string/integer branch columns raise `TypeError`; string-only keys sort lexicographically | Normalize integer-like columns before sorting within combinations and across rows |
| Integer text was parsed through floating point | `9007199254740993` becomes `9007199254740992`; signed-int64 maximum with `.0` overflows | Parse the validated integer portion directly in table, branch, site, and manifest parsers |
| Table casts accepted values outside int64 | Unsigned or floating `2**63` can become a negative identifier | Reject out-of-range identifiers before conversion |
| Integral-column detection skipped missing remainders | `[1.0, NaN]` or `[1.0, inf]` reaches an invalid integer cast | Retain missing, nonfinite, fractional, and out-of-range substitution columns without lossy conversion |
| A resource destination could equal the cache root | Updating that destination removes unrelated cached files | Require a strict descendant after resolving aliases; reject the root before population |
| Cache reuse checked only the saved manifest | A new requested structure filename returns a nonexistent path; changed expected checksums do not trigger repair | Check current required files and expected sizes/checksums at every reuse decision; repair online or fail offline |

The initial added tests reproduced 13 table/cache failures and 26 identifier
parser failures before the corresponding fixes. Two further cases cover a
cache-root symlink and expected-size validation without hashing, giving 41
new regression cases. Filesystem deletion tests use isolated pytest temporary
directories; structure/model download tests use deterministic local fakes.

## Validation

Run from the repository root:

```sh
python -m pytest -q -n 4 --dist worksteal -m 'not process'
python -m pytest -q -m process
CSUBST_STRICT_EXTENSIONS=1 python -m pytest -q -m native
CSUBST_DISABLE_EXTENSIONS=1 python -m pytest -q -n 4 --dist worksteal -m 'not process'
CSUBST_DISABLE_EXTENSIONS=1 python -m pytest -q -m process
make lint typecheck
CSUBST_STRICT_EXTENSIONS=1 python .github/scripts/sites_parity_check.py \
  --numerical-only --workdir reports/generated/audit_20260908/parity \
  --output reports/generated/audit_20260908/parity_metrics.tsv
git diff --check
```

The audit's parity run used an isolated temporary output directory; the
command above uses the repository's ignored report-output location instead.

- Standard suite: 1,436 parallel-safe tests and 4 process tests passed.
- Strict native selection: 7 tests passed.
- Source-only fallback: 1,404 parallel-safe tests and 4 process tests passed;
  32 compiled-extension-only cases skipped as expected.
- Focused table/parser/cache/structure checks: 82 passed, 2 process cases
  excluded (covered by the process lane).
- Ruff, repository hygiene, local documentation checks, and all 13 configured
  mypy targets passed. Whitespace validation passed.
- Bundled PGK and PEPC data passed the real `analyze`/`sites` numerical checks:

| Dataset | Branches | omegaCany2spe | Convergent sites | Divergent sites | Blank sites |
| --- | --- | ---: | ---: | ---: | ---: |
| PGK | 23, 51 | 1.975050 | 5 | 7 | 390 |
| PEPC | 9, 108 | 0.049466 | 0 | 2 | 954 |

Not exercised: other OS/Python combinations, fresh wheel/sdist installation,
sanitizer builds, live external downloads, real PyMOL rendering, GPU/model
inference, or remote Wiki validation. Linux performance thresholds were
disabled for the macOS parity run; no performance improvement is claimed.
No version bump or push is part of this local-commit audit.

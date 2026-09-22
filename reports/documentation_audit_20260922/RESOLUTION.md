# Cython diagnostic follow-up — 2026-09-22

The B finding in [the audit](README.md) is resolved in 1.16.9.
Cython remains an isolated build requirement; it is no longer listed among the
runtime distributions checked by `param._format_dependency_versions`.
Dependency metadata, extension loading, and scientific calculations are unchanged.

A regression test exercises package-metadata lookup with Cython absent, both
with all runtime packages present and with NumPy absent. The first reports no
missing packages; the second still reports NumPy. This prevents suppressing
real runtime dependency failures along with the false Cython diagnostic.

Verification on macOS arm64 / Python 3.14.7:

- `python -m pytest -q tests/unit/test_param_core.py`: 34 passed.
- `make test-fast`: 1,583 passed, 6 skipped. Repeated with
  `PYTEST_ADDOPTS=-rs` to identify the skips: four require torch and two gemmi.
- `make lint typecheck`: passed, including repository documentation checks.
- Reinstalled the changed source non-editably with `python -m pip install
  --no-deps .` into the audit's isolated venv. Verified installed version 1.16.9
  and that Cython distribution metadata is absent.
- In a fresh temporary directory, executed `csubst dataset --name PGK` then
  `csubst search --alignment_file alignment.fa.gz --rooted_tree_file tree.nwk
  --foreground foreground.txt`. Both exited 0. Asserted the log reports
  `CSUBST missing dependency packages: none`, has no `cython=not installed`,
  and the search run record has status `complete`.
- `git diff --check`: passed.

No full integration/parity or other-platform lane was run for this diagnostic-only
change. The installed PGK smoke test reused the bundled fit; it does not verify
an external IQ-TREE refit or optional structural backends.

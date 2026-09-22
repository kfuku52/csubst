# Testing CSUBST

The test suite is organized by the kind of feedback each test provides:

- `tests/unit/` contains focused tests for calculations, parsers, validation,
  rendering helpers, and other single-module behavior.
- `tests/integration/` covers orchestration across modules, files, processes,
  caches, and command implementations.
- `tests/cli/` covers the command-line parser and a focused set of executable
  entry-point subprocess checks.
- `tests/parity/` compares independent implementations and execution backends.
- `tests/support/` contains reusable factories and fakes. Files in this
  directory must not use the `test_*.py` naming pattern.

Install the test dependencies with the editable package:

```bash
python -m pip install -e '.[test]'
```

Use `.[dev]` instead for the complete contributor toolchain, including lint,
type checking, and packaging. `make` invokes every tool through `$(PYTHON)`;
use `make PYTHON=/path/to/python test-fast` to select an environment explicitly.

The ordinary sequential full suite remains supported:

```bash
pytest -q
```

For the shortest complete run, execute parallel-safe tests first and keep the
few tests that create their own worker processes in a serial lane:

```bash
pytest -q -n auto --dist worksteal -m "not process"
pytest -q -m process
```

The repository caps `-n auto` at four workers to avoid collection overhead,
memory pressure, and nested scheduling contention.

To verify the source-only fallback locally:

```bash
CSUBST_SKIP_EXTENSIONS=1 python -m pip install -e '.[test]'
CSUBST_DISABLE_EXTENSIONS=1 pytest -q -m "not process"
CSUBST_DISABLE_EXTENSIONS=1 pytest -q -m process
```

For a shorter local feedback loop matching the Python-version CI matrix, run:

```bash
pytest -q -n auto --dist worksteal tests/unit tests/cli -m "not slow and not parity and not process"
```

Individual suites and markers can be selected directly:

```bash
pytest -q tests/unit
pytest -q tests/integration
pytest -q tests/cli
pytest -q -m parity
pytest -q -m slow
pytest -q -m process
```

`integration`, `cli`, and `parity` markers are assigned from the directory
taxonomy. Add `@pytest.mark.slow` only after a duration report shows that a test
materially exceeds the normal unit-test feedback time. Tests that require a
compiled extension may use `@pytest.mark.requires_cython`. Add
`@pytest.mark.process` only when a test creates nested worker processes and
therefore must run outside pytest-xdist.

The `native` marker identifies tests that must exercise the compiled path,
including read-only NumPy arrays, memmaps, and pandas-owned arrays:

```bash
make test-native
```

This sets `CSUBST_STRICT_EXTENSIONS=1`. Tests that intentionally inject an
accelerator failure to verify fallback remain outside this selection. CI also
imports all six extensions explicitly, so missing binaries cannot turn the
native check into a successful all-skipped run.

Keep test modules aligned with one source responsibility or one observable
workflow. Prefer a shared factory in `tests/support/` when setup is reused
across files, but keep case-specific inputs and expected values next to the
assertions. As a review guideline, split a test module before it grows much
beyond roughly 600 lines unless keeping a cohesive scenario together is
clearer.

Keep a test when removing it would hide a realistic regression worth its runtime,
flakiness, and maintenance cost. Prefer observable results and independent
references over copied algorithms, fake-backend parity, or internal constants.
Consolidate duplicate checks at the boundary that catches the failure; exercise
independent option dimensions without automatically taking their Cartesian product.
When pruning tests, record the lost or retained guarantee, run the affected suite,
and then run the full suite. Collection counts describe the change, not a target.
The Cython sanitizer job separately exercises sparse and expected-state parity tests.

Performance parity uses `.github/performance_baseline.tsv`. Update it only from
a successful, representative Linux hosted-runner result and review numerical
parity at the same time. Absolute ceilings remain as a safety net; baseline
ratios catch smaller regressions.

After an intentional inference-default change, first verify the scientific
reference values. Dispatch `Pytest` with `record_performance_baseline=true`
to collect three numerically checked Linux runs in the separate
`record-parity-baseline` job. Its `parity-baseline-measurements` artifact can
support a reviewed baseline update; use per-dataset median wall time and peak
RSS across replicates. The ordinary `parity` job still enforces all existing
performance limits during this dispatch. Never update a baseline to hide an
unexplained regression.

## Choosing checks

Run from the repository root using the environment from
[CONTRIBUTING.md](CONTRIBUTING.md). Every command below must exit zero; inspect
pytest's skip summary as well. Missing optional dependencies are unverified
coverage, not a successful test of that backend. `make typecheck` covers only
the modules listed in `TYPECHECK_TARGETS`, not the entire package.

| Changed behavior | Minimum affected checks |
| --- | --- |
| Documentation or command examples | `make docs-check`; run any changed example separately on copied inputs. The checker parses commands but does not execute analyses. |
| Parser, parameters, aliases | `python -m pytest -q tests/cli`; relevant `tests/unit/test_param_*.py`; `make docs-check` |
| Paths, logging, output lifecycle | `python -m pytest -q tests/cli/test_cli_input_safety.py tests/cli/test_output_lifecycle.py tests/unit/test_runtime_io.py tests/unit/test_safe_workdir.py`; affected command integration tests |
| Numerical kernels, estimators, sparse/dense behavior | Relevant unit tests and `python -m pytest -q tests/parity`; check the corresponding method document/reference values |
| `main_*` orchestration, scan, caches, rendering | Matching unit and integration modules; e.g. `python -m pytest -q tests/integration/test_main_analyze.py` for search orchestration |
| Cython or accelerator dispatch | Matching unit/parity tests, native and source-only fallback lanes above; verify binaries exist before interpreting native skips |
| Packaging or dependencies | `make package` and the installed-wheel procedure below; source tests cannot validate wheel contents |

Find consumers and tests with `rg -n 'changed_symbol' csubst tests` rather than
assuming the test filename matches the implementation. For ordinary code changes,
follow focused checks with `make test-fast lint typecheck`. Use `make test` when
shared behavior or multiple workflows are affected; it includes integration,
parity, slow tests, and the separate serial process lane. Do not repeat successful
checks without a relevant new edit or unresolved concern.

### Local tests versus external validation

The pytest suite uses small fixtures, temporary output directories, and fakes for
many external services. `integration` does not mean a real IQ-TREE fit or a model
download, and `slow` is a runtime marker, not a network permission switch.
Inspect the selected test when real-backend coverage matters. Keep new output
tests on `tmp_path` and use `tests/support/cli_runner.py` for subprocess tests so
both source and installed-artifact modes continue to work.

Real CLI examples must run in a fresh directory created with `mktemp -d`, with
copied inputs and an installed package available outside the checkout. The README
dataset example is an analysis, not a substitute for a short unit-test run.
The scripts in `tools/` and scientific calibration/performance scripts in
`.github/scripts/` have their own workload and resource requirements; do not run
them all as a smoke test. Confirm data size, model availability, executable
requirements, and expected runtime first. Keep model downloads, external-service
checks, and large simulations separate from routine validation.

`make lint` includes the offline documentation check. Wiki coverage requires an
explicit local clone (`WIKI_DIR`, see CONTRIBUTING.md); remote links and external
resources are not validated by a repository-only pass. CI's installed-wheel,
coverage, sanitizer, and Linux performance lanes remain additional evidence.

## Testing the installed wheel

Source tests normally select the checkout. Artifact tests must use a separate,
non-editable installation and run outside the checkout. `CSUBST_TEST_INSTALLED=1`
disables the source override and checks the package and all six native modules
against the active environment prefix, both before and after the suite.

Process targets used by tests live in `tests/support/process_workers.py`.
Spawned interpreters can import that support module without adding the source
package to their path; pytest's synthetic importlib-mode test modules are not
importable by a fresh child interpreter.

The following reproduces the installed-wheel CI lane in a fresh environment
(start with an empty `dist/`, or use a clean checkout):

```bash
python -m build --sdist --outdir dist
CSUBST_USE_CYTHON=0 python -m pip wheel --no-deps dist/csubst-*.tar.gz --wheel-dir dist/from-sdist
python .github/scripts/package_artifact_check.py
CSUBST_REPO="$PWD"
CSUBST_TEST_ROOT="$(mktemp -d)"
python -m venv "$CSUBST_TEST_ROOT/venv"
"$CSUBST_TEST_ROOT/venv/bin/python" .github/scripts/install_test_artifact.py
cd "$CSUBST_TEST_ROOT"
CSUBST_TEST_INSTALLED=1 "$CSUBST_TEST_ROOT/venv/bin/python" -m pytest "$CSUBST_REPO/tests" --import-mode=importlib -q -n auto --dist worksteal -m 'not process'
CSUBST_TEST_INSTALLED=1 "$CSUBST_TEST_ROOT/venv/bin/python" -m pytest "$CSUBST_REPO/tests" --import-mode=importlib -q -m process
CSUBST_STRICT_EXTENSIONS=1 "$CSUBST_TEST_ROOT/venv/bin/python" "$CSUBST_REPO/.github/scripts/sites_parity_check.py" --installed --numerical-only
```

`--installed` reads the datasets bundled in the wheel and removes source
`PYTHONPATH` overrides from child CLI processes. `--numerical-only` keeps all
scientific assertions but skips Linux-specific runtime/RSS thresholds; the
Linux CI parity lane omits this option and still enforces those thresholds.

## CI responsibilities

The Python 3.12 packaging job produces one sdist and one wheel rebuilt from its
generated C. Full coverage and scientific parity reuse that run's wheel; the
fallback lane installs the sdist without extensions and runs its complete
extracted test suite. Sanitizer builds remain isolated. Python 3.10, 3.11,
3.13, and 3.14 each retain source/native compatibility checks.

Default-branch pushes and pull requests run CI without duplicating branch-push
checks. New commits cancel superseded runs. The macOS Intel installation/native
check runs for build, dependency, runtime, and CI changes; unknown diffs,
weekly schedules, and manual dispatch always include it. Release wheel builds
also retain both macOS architectures and Linux, with installed PGK/PEPC parity
for every supported Python version. Major/minor release automation calls the
wheel workflow explicitly for the tested tag; patch versions do not create a
release.

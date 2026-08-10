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
```

For a shorter local feedback loop matching the Python-version CI matrix, run:

```bash
pytest -q -n auto --dist worksteal tests/unit tests/cli -m "not slow and not parity"
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

Keep test modules aligned with one source responsibility or one observable
workflow. Prefer a shared factory in `tests/support/` when setup is reused
across files, but keep case-specific inputs and expected values next to the
assertions. As a review guideline, split a test module before it grows much
beyond roughly 600 lines unless keeping a cohesive scenario together is
clearer.

When moving or consolidating tests, compare collection counts before and after
the change, run the affected suite, and then run the full suite. The Cython
sanitizer job separately exercises sparse and expected-state parity tests.

Performance parity uses `.github/performance_baseline.tsv`. Update it only from
a successful, representative Linux hosted-runner result and review numerical
parity at the same time. Absolute ceilings remain as a safety net; baseline
ratios catch smaller regressions.

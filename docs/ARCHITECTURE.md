# Architecture

CSUBST is a staged command-line analysis pipeline. `csubst.cli` owns parsing and
dispatch. Each `main_*` module orchestrates one command, while numerical and I/O
work lives in focused modules such as `substitution`, `omega`, `foreground`,
`sequence_io`, and `resource_cache`.

CLI parent-parser factories own shared argument groups, and each command has a
registration function. `_build_parser` only composes those groups. `cli_io`
checks log/input path collisions before opening a log, including parse-error
paths, without importing numerical libraries. See [CLI_SAFETY.md](CLI_SAFETY.md)
for the observable failure behavior.

## Analysis context

Command arguments are validated by `param.get_global_parameters` and wrapped in
`runtime.RunContext`. Its immutable `config` view contains validated inputs;
derived arrays, tables, caches, and counters are written to `runtime_state`.
The combined mapping interface remains for compatibility. New stage boundaries
should accept `config_types.AnalysisConfig`, avoid introducing ad-hoc keys, and
place new keys in the appropriate grouped `TypedDict` first.

Parameter normalization is divided into ordered input, model, epistasis,
search, state, simulation, site, structure, execution, recoding, and output
stages. The entry point wraps the configuration only after all stages succeed.
Keep cross-stage dependencies explicit and preserve validation order when
moving rules. Type checking covers these boundaries and the new I/O/statistics
modules; it does not yet cover every historical context key.

## Statistics and rendering

`omega_statistics` owns array-only rates, dS calibration, empirical upper-tail
counts/p-values, and FDR. Its typed array inputs/outputs do not depend on the
pipeline context, I/O, or accelerator dispatch. `omega` retains null-model and
expected-count orchestration and re-exports the existing helper names.

`site_tree_plot` owns tree/site selection, layout, labels, heatmaps, and figure
output for prepared site tables. `main_sites` retains data preparation,
structure integration, and command orchestration, and re-exports existing
rendering entry points. Shared lazy Matplotlib initialization lives in
`plotting`, so non-rendering commands still avoid importing the plotting stack.

## Acceleration

Six optional Cython modules accelerate combination generation, sparse tensor
construction, recoding, IQ-TREE parsing, and omega calculations. Python/NumPy
implementations are the correctness reference. The runtime modes are:

- default: use compiled modules when present and warn once on a failed fast path;
- `CSUBST_DISABLE_EXTENSIONS=1`: do not import compiled modules;
- `CSUBST_STRICT_EXTENSIONS=1`: re-raise a fast-path error, used by performance
  parity CI to prevent unnoticed slowdowns.

`--threads` controls process/task parallelism. `--blas_threads` independently
limits native BLAS/OpenMP threads per process and defaults to 1, preventing
multiplicative oversubscription.

Set backend environment variables before importing NumPy/SciPy. Accelerate uses
`VECLIB_MAXIMUM_THREADS`. Read-only numerical inputs stay read-only across
Python/Cython boundaries; input memoryviews should be `const` unless mutation
is part of the documented contract.

## Verification layers

Unit tests check module contracts, integration tests check pipelines, CLI tests
check entry-point and logging behavior, and parity tests compare implementations.
The full CI lane enforces branch coverage, a pure-Python lane validates source
archives without binaries, and versioned PGK/PEPC baselines detect numerical,
runtime, and peak-memory regressions. Cython sanitizer tests cover native memory
safety. See [../TESTING.md](../TESTING.md) for commands.

# Architecture

CSUBST is a staged command-line analysis pipeline. `csubst.cli` owns parsing and
dispatch. Each `main_*` module orchestrates one command, while numerical and I/O
work lives in focused modules such as `substitution`, `omega`, `foreground`,
`sequence_io`, and `resource_cache`.

## Analysis context

Command arguments are validated by `param.get_global_parameters` and wrapped in
`runtime.RunContext`. Its immutable `config` view contains validated inputs;
derived arrays, tables, caches, and counters are written to `runtime_state`.
The combined mapping interface remains for compatibility. New stage boundaries
should accept `config_types.AnalysisConfig`, avoid introducing ad-hoc keys, and
place new keys in the appropriate grouped `TypedDict` first.

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

## Verification layers

Unit tests check module contracts, integration tests check pipelines, CLI tests
check entry-point and logging behavior, and parity tests compare implementations.
The full CI lane enforces branch coverage, a pure-Python lane validates source
archives without binaries, and versioned PGK/PEPC baselines detect numerical,
runtime, and peak-memory regressions. Cython sanitizer tests cover native memory
safety. See [../TESTING.md](../TESTING.md) for commands.

# Contributing

Use Python 3.12 for the complete local verification lane and keep compatibility
with Python 3.10–3.14. Install an editable development environment with:

```bash
python -m pip install -e '.[test]'
```

The common commands are available through `make`:

```bash
make test-fast
make lint
make typecheck
make test
make package
```

Keep command orchestration in `csubst/cli.py`, pure I/O in focused modules, and
numerical kernels behind small Python contracts. New optional Cython code must
retain a tested Python implementation. Use `CSUBST_DISABLE_EXTENSIONS=1` to
exercise it and `CSUBST_STRICT_EXTENSIONS=1` to turn an accelerator failure into
an error instead of a warning and fallback.

Do not add direct dependencies unless production code, configuration, tests, or
documentation imports or invokes them. Prefer compatibility fixes over upper
bounds; lock files and deployment constraints should be maintained separately
from library metadata. See [TESTING.md](TESTING.md),
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and
[reports/README.md](reports/README.md) for the detailed contracts.

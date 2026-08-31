# Contributing

Use Python 3.12 for the complete local verification lane and keep compatibility
with Python 3.10–3.14. Install an editable development environment with:

```bash
python -m pip install -e '.[dev]'
```

The common commands are available through `make`:

```bash
make test-fast
make lint
make docs-check
make typecheck
make test
make test-native
make package
```

The `dev` extra includes the test runner, Ruff, mypy, build, and Twine used by
these commands. The smaller `test` extra remains available for running tests
without development and packaging tools. `make PYTHON=/path/to/python ...`
selects a particular environment for every tool.

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

When changing CLI options, output formats, or file names, update examples and
both documentation repositories in the same change. Keep the top-level README
short, preserve its useful inline figures, and place extended guides in the
Wiki. `make lint` checks current repository documentation without network access;
it parses documented commands but never executes them.

Before publishing a Wiki update, use a local clone with the same check:

```bash
CSUBST_WIKI_DIR="$(mktemp -d)"
git clone https://github.com/kfuku52/csubst.wiki.git "$CSUBST_WIKI_DIR"
make docs-check WIKI_DIR="$CSUBST_WIKI_DIR"
```

Point `WIKI_DIR` at an existing edited clone to check those edits instead.
The check covers argument syntax and local file/image/Wiki links, not every
remote URL, heading anchor, scientific statement, or optional dependency.
Run affected examples on a small bundled dataset as well and inspect their
output manifests before copying file names into the guides. A basic check uses
`csubst dataset --name PGK` in an empty directory, followed by `csubst sites`
with alignment `alignment.fa.gz`, tree `tree.nwk`, and branch IDs `23,51`.
Its default manifest is `csubst_sites/csubst.branch_id23,51/csubst.outputs.tsv`.

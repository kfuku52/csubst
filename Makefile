.PHONY: test-fast test lint typecheck package clean

test-fast:
	pytest -q -n auto --dist worksteal tests/unit tests/cli -m "not slow and not parity"

test:
	pytest -q -n auto --dist worksteal -m "not process"
	pytest -q -m process

lint:
	ruff check csubst tests setup.py .github/scripts
	python .github/scripts/repository_hygiene_check.py

typecheck:
	@for target in csubst/config_types.py csubst/runtime.py csubst/sequence_io.py csubst/expected_sparse.py csubst/tsv.py; do \
		mypy --follow-imports=skip "$$target" || exit 1; \
	done

package:
	python -m build
	python -m twine check dist/*

clean:
	python -c 'from pathlib import Path; import shutil; [shutil.rmtree(path) for path in (Path("build"), Path("dist")) if path.exists()]'

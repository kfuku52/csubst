PYTHON ?= python
TYPECHECK_TARGETS = csubst/config_types.py csubst/runtime.py csubst/cli_io.py csubst/sequence_io.py csubst/param.py csubst/omega_statistics.py csubst/site_tree_plot.py csubst/plotting.py csubst/expected_sparse.py csubst/tsv.py csubst/main_scan.py csubst/main_analyze.py csubst/main_download.py

.PHONY: test-fast test test-native lint typecheck package clean

test-fast:
	$(PYTHON) -m pytest -q -n auto --dist worksteal tests/unit tests/cli -m "not slow and not parity and not process"

test:
	$(PYTHON) -m pytest -q -n auto --dist worksteal -m "not process"
	$(PYTHON) -m pytest -q -m process

test-native:
	CSUBST_STRICT_EXTENSIONS=1 $(PYTHON) -m pytest -q -m native

lint:
	$(PYTHON) -m ruff check csubst tests setup.py .github/scripts
	$(PYTHON) .github/scripts/repository_hygiene_check.py

typecheck:
	@for target in $(TYPECHECK_TARGETS); do \
		$(PYTHON) -m mypy --follow-imports=skip "$$target" || exit 1; \
	done

package:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean:
	$(PYTHON) -c 'from pathlib import Path; import shutil; [shutil.rmtree(path) for path in (Path("build"), Path("dist")) if path.exists()]'

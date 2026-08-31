import os
import pathlib
import sys
import importlib.util

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_INSTALLED = os.environ.get('CSUBST_TEST_INSTALLED') == '1'
TEST_SUPPORT = ROOT / "tests" / "support"
if str(TEST_SUPPORT) not in sys.path:
    sys.path.insert(0, str(TEST_SUPPORT))
if TEST_INSTALLED:
    sys.path[:] = [p for p in sys.path if pathlib.Path(p or '.').resolve() != ROOT]
    sys.path.insert(0, str(ROOT / '.github' / 'scripts'))
    from _installed_package import require_installed_package
    require_installed_package()
else:
    # Source tests deliberately select the checkout; artifact tests must not.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if "csubst" in sys.modules:
        for module_name in list(sys.modules):
            if module_name == "csubst" or module_name.startswith("csubst."):
                del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        "csubst",
        ROOT / "csubst" / "__init__.py",
        submodule_search_locations=[str(ROOT / "csubst")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["csubst"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

# Keep Matplotlib's font cache writable without forcing an expensive rebuild on
# every pytest invocation.  The cache is ignored by git and reused by local
# sequential/xdist runs; an explicit caller-provided MPLCONFIGDIR still wins.
if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache = (
        ROOT
        / ".pytest_cache"
        / "matplotlib"
        / "py{}{}".format(sys.version_info.major, sys.version_info.minor)
    )
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)


@pytest.fixture(autouse=True)
def _set_random_seed():
    np.random.seed(0)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config):
    """Cap auto workers to avoid collection overhead and memory contention."""
    if hasattr(os, "sched_getaffinity"):
        available_cpus = len(os.sched_getaffinity(0))
    else:
        available_cpus = os.cpu_count() or 1
    return max(1, min(4, int(available_cpus)))


def pytest_configure(config):
    """Build a cold font cache once in the xdist controller, not per worker."""
    is_worker = hasattr(config, "workerinput")
    uses_xdist = getattr(config.option, "numprocesses", None) is not None
    if uses_xdist and not is_worker:
        from matplotlib import font_manager  # noqa: F401


def pytest_sessionfinish(session, exitstatus):
    if TEST_INSTALLED:
        require_installed_package()


def pytest_collection_modifyitems(items):
    """Apply suite markers from the directory-based test taxonomy."""
    for item in items:
        path_parts = item.path.parts
        if "cli" in path_parts:
            item.add_marker(pytest.mark.cli)
        if "integration" in path_parts:
            item.add_marker(pytest.mark.integration)
        if "parity" in path_parts:
            item.add_marker(pytest.mark.parity)

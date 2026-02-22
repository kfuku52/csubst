import os
import pathlib
import sys
import tempfile
import importlib.util

import numpy as np
import pytest

# Ensure tests import the local checkout rather than an installed csubst package.
ROOT = pathlib.Path(__file__).resolve().parents[1]
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

# main_site imports matplotlib at module import time; ensure cache is writable.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))


@pytest.fixture(autouse=True)
def _set_random_seed():
    np.random.seed(0)

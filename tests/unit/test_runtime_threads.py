import os
import json
from pathlib import Path
import subprocess
import sys

import pytest

from csubst import runtime


def test_configure_native_threads_sets_all_supported_environment_variables(monkeypatch):
    expected = {'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'}
    for variable_name in expected:
        monkeypatch.delenv(variable_name, raising=False)

    assert runtime.configure_native_threads(2) == 2

    assert {os.environ[name] for name in expected} == {'2'}


def test_configure_native_threads_rejects_non_positive_value():
    with pytest.raises(ValueError, match='blas_threads'):
        runtime.configure_native_threads(0)


def test_accelerate_thread_limit_is_set_before_numpy_import(tmp_path):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(runtime.__file__).resolve().parents[1])
    env.pop('VECLIB_MAXIMUM_THREADS', None)
    code = '''import json, os, sys
from csubst import runtime
assert 'numpy' not in sys.modules
runtime.configure_native_threads(1)
import numpy as np
assert np.dot(np.ones((2, 2)), np.ones((2, 2))).tolist() == [[2., 2.], [2., 2.]]
print(json.dumps({'limit': os.environ.get('VECLIB_MAXIMUM_THREADS')}))
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {'limit': '1'}

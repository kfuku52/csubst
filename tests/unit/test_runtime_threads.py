import os

import pytest

from csubst import runtime


def test_configure_native_threads_sets_all_supported_environment_variables(monkeypatch):
    for variable_name in runtime._NATIVE_THREAD_ENV_VARS:
        monkeypatch.delenv(variable_name, raising=False)

    assert runtime.configure_native_threads(2) == 2

    assert {os.environ[name] for name in runtime._NATIVE_THREAD_ENV_VARS} == {'2'}


def test_configure_native_threads_rejects_non_positive_value():
    with pytest.raises(ValueError, match='blas_threads'):
        runtime.configure_native_threads(0)

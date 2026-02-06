import argparse

import pytest

from csubst import param
from csubst import substitution


def _args(**kwargs):
    defaults = {
        "threads": 1,
        "float_type": 64,
        "float_digit": 4,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_get_global_parameters_defaults_sub_tensor_backend_to_auto():
    g = param.get_global_parameters(_args())
    assert g["sub_tensor_backend"] == "auto"


def test_get_global_parameters_normalizes_sub_tensor_backend_case():
    g = param.get_global_parameters(_args(sub_tensor_backend="SpArSe"))
    assert g["sub_tensor_backend"] == "sparse"


def test_get_global_parameters_rejects_invalid_sub_tensor_backend():
    with pytest.raises(ValueError, match="sub_tensor_backend"):
        param.get_global_parameters(_args(sub_tensor_backend="not-a-backend"))


def test_resolve_sub_tensor_backend_auto_to_dense():
    g = {"sub_tensor_backend": "auto"}
    assert substitution.resolve_sub_tensor_backend(g) == "dense"
    assert g["resolved_sub_tensor_backend"] == "dense"


def test_resolve_sub_tensor_backend_sparse_falls_back_to_dense():
    g = {"sub_tensor_backend": "sparse"}
    assert substitution.resolve_sub_tensor_backend(g) == "dense"
    assert g["resolved_sub_tensor_backend"] == "dense"


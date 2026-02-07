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
    assert g["sub_tensor_sparse_density_cutoff"] == pytest.approx(0.15)
    assert g["parallel_backend"] == "auto"
    assert g["parallel_chunk_factor"] == 1
    assert g["parallel_chunk_factor_reducer"] == 4


def test_get_global_parameters_normalizes_sub_tensor_backend_case():
    g = param.get_global_parameters(_args(sub_tensor_backend="SpArSe"))
    assert g["sub_tensor_backend"] == "sparse"


def test_get_global_parameters_rejects_invalid_sub_tensor_backend():
    with pytest.raises(ValueError, match="sub_tensor_backend"):
        param.get_global_parameters(_args(sub_tensor_backend="not-a-backend"))


def test_get_global_parameters_rejects_invalid_sparse_density_cutoff():
    with pytest.raises(ValueError, match="sub_tensor_sparse_density_cutoff"):
        param.get_global_parameters(_args(sub_tensor_sparse_density_cutoff=1.5))


def test_get_global_parameters_rejects_invalid_parallel_backend():
    with pytest.raises(ValueError, match="parallel_backend"):
        param.get_global_parameters(_args(parallel_backend="invalid"))


def test_get_global_parameters_rejects_invalid_chunk_factors():
    with pytest.raises(ValueError, match="parallel_chunk_factor"):
        param.get_global_parameters(_args(parallel_chunk_factor=0))
    with pytest.raises(ValueError, match="parallel_chunk_factor_reducer"):
        param.get_global_parameters(_args(parallel_chunk_factor_reducer=0))


def test_resolve_sub_tensor_backend_auto_to_dense():
    g = {"sub_tensor_backend": "auto"}
    assert substitution.resolve_sub_tensor_backend(g) == "dense"
    assert g["resolved_sub_tensor_backend"] == "dense"


def test_resolve_sub_tensor_backend_sparse_is_sparse():
    g = {"sub_tensor_backend": "sparse"}
    assert substitution.resolve_sub_tensor_backend(g) == "sparse"
    assert g["resolved_sub_tensor_backend"] == "sparse"

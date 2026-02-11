import argparse

import pytest

from csubst import param
from csubst import output_stat
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


def test_get_global_parameters_parses_output_stat_and_required_base_stats():
    g = param.get_global_parameters(_args(output_stat="ANY2ANY,any2dif,any2spe,any2any"))
    assert g["output_stats"] == ["any2any", "any2dif", "any2spe"]
    assert g["output_base_stats"] == ["any2any", "any2spe"]
    assert g["output_dif_stats"] == ["any2dif"]


def test_get_global_parameters_tracks_required_intermediate_dif_stats():
    g = param.get_global_parameters(_args(output_stat="dif2dif"))
    assert g["output_base_stats"] == ["any2any", "spe2any", "any2spe", "spe2spe"]
    assert g["output_dif_stats"] == ["any2dif", "dif2dif", "spe2dif"]


def test_get_global_parameters_rejects_invalid_output_stat():
    with pytest.raises(ValueError, match="output_stat"):
        param.get_global_parameters(_args(output_stat="any2any,not_a_stat"))


def test_drop_unrequested_stat_columns_removes_helper_stats():
    cb = {
        "OCNany2any": [1.0],
        "OCNany2dif": [0.5],
        "OCNany2spe": [0.5],
        "omegaCany2any": [1.0],
        "omegaCany2dif": [1.0],
    }
    import pandas as pd

    out = output_stat.drop_unrequested_stat_columns(pd.DataFrame(cb), ["any2dif"])
    assert "OCNany2dif" in out.columns
    assert "omegaCany2dif" in out.columns
    assert "OCNany2any" not in out.columns
    assert "OCNany2spe" not in out.columns
    assert "omegaCany2any" not in out.columns


def test_get_global_parameters_adjusts_default_cutoff_stat_for_output_subset():
    g = param.get_global_parameters(
        _args(
            output_stat="any2any",
            cutoff_stat="OCNany2spe,2.0|omegaCany2spe,5.0",
        )
    )
    assert g["cutoff_stat"] == "OCNany2any,2.0|omegaCany2any,5.0"


def test_get_global_parameters_rejects_incompatible_custom_cutoff_stat():
    with pytest.raises(ValueError, match='requires --output_stat to include "any2spe"'):
        param.get_global_parameters(
            _args(
                output_stat="any2any",
                cutoff_stat="OCNany2spe,2.0|omegaCany2any,5.0",
            )
        )


def test_get_global_parameters_rejects_nonpositive_threads():
    with pytest.raises(ValueError, match="threads"):
        param.get_global_parameters(_args(threads=0))
    with pytest.raises(ValueError, match="threads"):
        param.get_global_parameters(_args(threads=-1))


def test_get_global_parameters_requires_foreground_for_exhaustive_until_one():
    with pytest.raises(ValueError, match="exhaustive_until 1"):
        param.get_global_parameters(_args(exhaustive_until=1, foreground=None))


def test_get_global_parameters_requires_foreground_for_clade_permutation():
    with pytest.raises(ValueError, match="fg_clade_permutation"):
        param.get_global_parameters(_args(fg_clade_permutation=1, foreground=None))


def test_get_global_parameters_rejects_invalid_percent_biased_sub():
    with pytest.raises(ValueError, match="percent_biased_sub"):
        param.get_global_parameters(_args(percent_biased_sub=-1))
    with pytest.raises(ValueError, match="percent_biased_sub"):
        param.get_global_parameters(_args(percent_biased_sub=100))


def test_get_global_parameters_rejects_invalid_tree_site_plot_values():
    with pytest.raises(ValueError, match="tree_site_plot_max_sites"):
        param.get_global_parameters(_args(tree_site_plot_max_sites=0))
    with pytest.raises(ValueError, match="tree_site_plot_min_prob"):
        param.get_global_parameters(_args(tree_site_plot_min_prob=1.1))


def test_get_global_parameters_rejects_invalid_simulate_ranges():
    with pytest.raises(ValueError, match="num_simulated_site"):
        param.get_global_parameters(_args(num_simulated_site=0))
    with pytest.raises(ValueError, match="num_simulated_site"):
        param.get_global_parameters(_args(num_simulated_site=-2))
    with pytest.raises(ValueError, match="percent_convergent_site"):
        param.get_global_parameters(_args(percent_convergent_site=-1))
    with pytest.raises(ValueError, match="percent_convergent_site"):
        param.get_global_parameters(_args(percent_convergent_site=101))


def test_get_global_parameters_rejects_negative_simulate_scalars():
    with pytest.raises(ValueError, match="tree_scaling_factor"):
        param.get_global_parameters(_args(tree_scaling_factor=-0.1))
    with pytest.raises(ValueError, match="foreground_scaling_factor"):
        param.get_global_parameters(_args(foreground_scaling_factor=-0.1))
    with pytest.raises(ValueError, match="background_omega"):
        param.get_global_parameters(_args(background_omega=-0.1))
    with pytest.raises(ValueError, match="foreground_omega"):
        param.get_global_parameters(_args(foreground_omega=-0.1))


def test_get_global_parameters_validates_convergent_amino_acids():
    with pytest.raises(ValueError, match="randomN"):
        param.get_global_parameters(_args(convergent_amino_acids="random-1"))
    with pytest.raises(ValueError, match="randomN"):
        param.get_global_parameters(_args(convergent_amino_acids="randomX"))
    with pytest.raises(ValueError, match="0 <= N <= 20"):
        param.get_global_parameters(_args(convergent_amino_acids="random21"))
    with pytest.raises(ValueError, match="unsupported amino acids"):
        param.get_global_parameters(_args(convergent_amino_acids="Z"))
    g = param.get_global_parameters(_args(convergent_amino_acids="AQ"))
    assert g["convergent_amino_acids"] == "AQ"


def test_resolve_sub_tensor_backend_auto_to_dense():
    g = {"sub_tensor_backend": "auto"}
    assert substitution.resolve_sub_tensor_backend(g) == "dense"
    assert g["resolved_sub_tensor_backend"] == "dense"


def test_resolve_sub_tensor_backend_sparse_is_sparse():
    g = {"sub_tensor_backend": "sparse"}
    assert substitution.resolve_sub_tensor_backend(g) == "sparse"
    assert g["resolved_sub_tensor_backend"] == "sparse"

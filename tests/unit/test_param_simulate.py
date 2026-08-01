from factories import make_args as _args


import pytest

from csubst import param




def test_get_global_parameters_requires_untrimmed_cds_for_export2chimera():
    with pytest.raises(ValueError, match="--untrimmed_cds"):
        param.get_global_parameters(_args(export2chimera=True, untrimmed_cds=None))
    g = param.get_global_parameters(_args(export2chimera=True, untrimmed_cds="genes.fa"))
    assert g["export2chimera"] is True


def test_get_global_parameters_rejects_invalid_simulate_ranges():
    with pytest.raises(ValueError, match="num_simulated_site"):
        param.get_global_parameters(_args(num_simulated_site=0))
    with pytest.raises(ValueError, match="num_simulated_site"):
        param.get_global_parameters(_args(num_simulated_site=-2))
    with pytest.raises(ValueError, match="percent_convergent_site"):
        param.get_global_parameters(_args(percent_convergent_site=-1))
    with pytest.raises(ValueError, match="percent_convergent_site"):
        param.get_global_parameters(_args(percent_convergent_site=101))


def test_get_global_parameters_validates_simulate_seed():
    with pytest.raises(ValueError, match="simulate_seed"):
        param.get_global_parameters(_args(simulate_seed=-2))
    g = param.get_global_parameters(_args(simulate_seed=-1))
    assert g["simulate_seed"] == -1
    g = param.get_global_parameters(_args(simulate_seed=123))
    assert g["simulate_seed"] == 123


def test_get_global_parameters_allows_empty_true_asr_prefix_for_runtime_default():
    g = param.get_global_parameters(_args(export_true_asr=True, true_asr_prefix=""))
    assert g["true_asr_prefix"] == ""


def test_get_global_parameters_validates_simulate_asrv_mode():
    g = param.get_global_parameters(_args(simulate_asrv="no"))
    assert g["simulate_asrv"] == "no"
    g = param.get_global_parameters(_args(simulate_asrv="FiLe"))
    assert g["simulate_asrv"] == "file"
    with pytest.raises(ValueError, match="simulate_asrv"):
        param.get_global_parameters(_args(simulate_asrv="maybe"))


def test_get_global_parameters_validates_simulate_eq_freq_mode():
    g = param.get_global_parameters(_args(simulate_eq_freq="auto"))
    assert g["simulate_eq_freq"] == "auto"
    g = param.get_global_parameters(_args(simulate_eq_freq="IQTREE"))
    assert g["simulate_eq_freq"] == "iqtree"
    g = param.get_global_parameters(_args(simulate_eq_freq="alignment"))
    assert g["simulate_eq_freq"] == "alignment"
    with pytest.raises(ValueError, match="simulate_eq_freq"):
        param.get_global_parameters(_args(simulate_eq_freq="unsupported"))


def test_get_global_parameters_rejects_negative_simulate_scalars():
    with pytest.raises(ValueError, match="tree_scaling_factor"):
        param.get_global_parameters(_args(tree_scaling_factor=-0.1))
    with pytest.raises(ValueError, match="foreground_scaling_factor"):
        param.get_global_parameters(_args(foreground_scaling_factor=-0.1))
    with pytest.raises(ValueError, match="background_omega"):
        param.get_global_parameters(_args(background_omega=-0.1))
    with pytest.raises(ValueError, match="foreground_omega"):
        param.get_global_parameters(_args(foreground_omega=-0.1))


def test_get_global_parameters_accepts_optional_background_omega():
    g = param.get_global_parameters(_args(background_omega=None))
    assert g["background_omega"] is None
    g = param.get_global_parameters(_args(background_omega="iqtree"))
    assert g["background_omega"] is None


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

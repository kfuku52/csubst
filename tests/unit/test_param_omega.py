from factories import make_args as _args


import pytest

from csubst import param




def test_get_global_parameters_rejects_nonpositive_threads():
    with pytest.raises(ValueError, match="threads"):
        param.get_global_parameters(_args(threads=0))
    with pytest.raises(ValueError, match="threads"):
        param.get_global_parameters(_args(threads=-1))


def test_get_global_parameters_parses_download_prostt5_bool():
    g = param.get_global_parameters(_args(download_prostt5="yes"))
    assert g["download_prostt5"] is True


def test_get_global_parameters_rejects_invalid_download_prostt5_bool():
    with pytest.raises(ValueError, match="download_prostt5"):
        param.get_global_parameters(_args(download_prostt5="maybe"))


@pytest.mark.parametrize("name", ["resource_lock_poll", "resource_lock_timeout"])
def test_get_global_parameters_rejects_nonpositive_resource_lock_option(name):
    with pytest.raises(ValueError, match=name):
        param.get_global_parameters(_args(**{name: 0}))


def test_get_global_parameters_requires_foreground_for_exhaustive_until_one():
    with pytest.raises(ValueError, match="exhaustive_until 1"):
        param.get_global_parameters(_args(exhaustive_until=1, foreground=None))


def test_get_global_parameters_requires_foreground_for_clade_permutation():
    with pytest.raises(ValueError, match="fg_clade_permutation"):
        param.get_global_parameters(_args(fg_clade_permutation=1, foreground=None))


def test_get_global_parameters_rejects_calc_omega_pvalue_without_urn_expectation():
    with pytest.raises(ValueError, match='--expectation_method "urn"'):
        param.get_global_parameters(_args(calc_omega_pvalue=True, expectation_method="codon_model"))


def test_get_global_parameters_sets_expectation_defaults():
    g = param.get_global_parameters(_args())
    assert g["expectation_method"] == "codon_model"
    assert g["urn_model"] == "wallenius"


def test_get_global_parameters_accepts_fisher_urn_model():
    g = param.get_global_parameters(_args(expectation_method="urn", urn_model="fisher"))
    assert g["expectation_method"] == "urn"
    assert g["urn_model"] == "fisher"


def test_get_global_parameters_sets_omega_pvalue_defaults():
    g = param.get_global_parameters(_args())
    assert g["calc_omega_pvalue"] is False
    assert g["omega_pvalue_null_model"] == "hypergeom"
    assert g["omega_pvalue_niter_schedule"] is None
    assert g["omega_pvalue_refine_upper_edge_bins"] == 2
    assert g["omega_pvalue_rounding"] == "stochastic"


def test_get_global_parameters_rejects_removed_omega_pvalue_niter():
    with pytest.raises(ValueError, match="omega_pvalue_niter was removed"):
        param.get_global_parameters(_args(omega_pvalue_niter=1000))


def test_get_global_parameters_parses_omega_pvalue_niter_schedule_auto_alias():
    g = param.get_global_parameters(_args(omega_pvalue_niter_schedule="auto"))
    assert g["omega_pvalue_niter_schedule"] is None


def test_get_global_parameters_parses_custom_omega_pvalue_niter_schedule():
    g = param.get_global_parameters(_args(omega_pvalue_niter_schedule="200,2000"))
    assert g["omega_pvalue_niter_schedule"] == [200, 2000]


def test_get_global_parameters_rejects_invalid_omega_pvalue_niter_schedule():
    with pytest.raises(ValueError, match="omega_pvalue_niter_schedule"):
        param.get_global_parameters(_args(omega_pvalue_niter_schedule="100,abc"))
    with pytest.raises(ValueError, match="omega_pvalue_niter_schedule"):
        param.get_global_parameters(_args(omega_pvalue_niter_schedule="1000,100"))
    with pytest.raises(ValueError, match="omega_pvalue_niter_schedule"):
        param.get_global_parameters(_args(omega_pvalue_niter_schedule="200,20000"))


def test_get_global_parameters_accepts_omega_pvalue_refine_upper_edge_bins():
    g = param.get_global_parameters(_args(omega_pvalue_refine_upper_edge_bins=5))
    assert g["omega_pvalue_refine_upper_edge_bins"] == 5


def test_get_global_parameters_rejects_invalid_omega_pvalue_refine_upper_edge_bins():
    with pytest.raises(ValueError, match="omega_pvalue_refine_upper_edge_bins"):
        param.get_global_parameters(_args(omega_pvalue_refine_upper_edge_bins=-1))


def test_get_global_parameters_rejects_removed_omega_pvalue_refine_threshold():
    with pytest.raises(ValueError, match="omega_pvalue_refine_threshold was removed"):
        param.get_global_parameters(_args(omega_pvalue_refine_threshold=0.1))


def test_get_global_parameters_rejects_removed_omega_pvalue_refine_ci_alpha():
    with pytest.raises(ValueError, match="omega_pvalue_refine_ci_alpha was removed"):
        param.get_global_parameters(_args(omega_pvalue_refine_ci_alpha=0.1))


def test_get_global_parameters_rejects_invalid_omega_pvalue_rounding():
    with pytest.raises(ValueError, match="omega_pvalue_rounding"):
        param.get_global_parameters(_args(omega_pvalue_rounding="invalid"))


def test_get_global_parameters_rejects_invalid_omega_pvalue_null_model():
    with pytest.raises(ValueError, match="omega_pvalue_null_model"):
        param.get_global_parameters(_args(omega_pvalue_null_model="invalid"))


def test_get_global_parameters_accepts_omega_pvalue_poisson_full_model():
    g = param.get_global_parameters(_args(omega_pvalue_null_model="poisson_full"))
    assert g["omega_pvalue_null_model"] == "poisson_full"


def test_get_global_parameters_accepts_omega_pvalue_nbinom_model():
    g = param.get_global_parameters(_args(omega_pvalue_null_model="nbinom"))
    assert g["omega_pvalue_null_model"] == "nbinom"


def test_get_global_parameters_sets_omega_pvalue_nbinom_alpha_defaults():
    g = param.get_global_parameters(_args())
    assert g["omega_pvalue_nbinom_alpha"] == "auto"


def test_get_global_parameters_accepts_fixed_omega_pvalue_nbinom_alpha():
    g = param.get_global_parameters(_args(omega_pvalue_nbinom_alpha=0.5))
    assert g["omega_pvalue_nbinom_alpha"] == pytest.approx(0.5)


def test_get_global_parameters_rejects_invalid_omega_pvalue_nbinom_alpha():
    with pytest.raises(ValueError, match="omega_pvalue_nbinom_alpha"):
        param.get_global_parameters(_args(omega_pvalue_nbinom_alpha=-0.1))
    with pytest.raises(ValueError, match="omega_pvalue_nbinom_alpha"):
        param.get_global_parameters(_args(omega_pvalue_nbinom_alpha=float("nan")))


def test_get_global_parameters_keeps_min_sub_pp_unchanged_for_omega_pvalue(capsys):
    g = param.get_global_parameters(
        _args(
            calc_omega_pvalue=True,
            expectation_method="urn",
            min_sub_pp=0,
            ml_anc="no",
        )
    )
    captured = capsys.readouterr()
    assert g["min_sub_pp"] == pytest.approx(0.0)
    assert "auto-set to" not in captured.err


def test_get_global_parameters_rejects_removed_omega_pvalue_safe_min_sub_pp():
    with pytest.raises(ValueError, match="was removed"):
        param.get_global_parameters(_args(omega_pvalue_safe_min_sub_pp=0.05))


def test_get_global_parameters_keeps_explicit_min_sub_pp_for_omega_pvalue(capsys):
    g = param.get_global_parameters(
        _args(
            calc_omega_pvalue=True,
            expectation_method="urn",
            min_sub_pp=0.2,
            ml_anc="no",
        )
    )
    captured = capsys.readouterr()
    assert g["min_sub_pp"] == pytest.approx(0.2)
    assert "auto-set to" not in captured.err


def test_get_global_parameters_does_not_auto_set_min_sub_pp_when_ml_anc_yes(capsys):
    g = param.get_global_parameters(
        _args(
            calc_omega_pvalue=True,
            expectation_method="urn",
            min_sub_pp=0,
            ml_anc="yes",
        )
    )
    captured = capsys.readouterr()
    assert g["min_sub_pp"] == pytest.approx(0.0)
    assert "auto-set to" not in captured.err


def test_get_global_parameters_rejects_invalid_min_sub_pp():
    with pytest.raises(ValueError, match="min_sub_pp"):
        param.get_global_parameters(_args(min_sub_pp=-0.1))
    with pytest.raises(ValueError, match="min_sub_pp"):
        param.get_global_parameters(_args(min_sub_pp=1.1))


def test_get_global_parameters_accepts_file_each_asrv_and_dirichlet_alpha():
    g = param.get_global_parameters(_args(asrv="FILE_EACH", asrv_dirichlet_alpha=0.25))
    assert g["asrv"] == "file_each"
    assert g["asrv_dirichlet_alpha"] == pytest.approx(0.25)


def test_get_global_parameters_sets_dirichlet_alpha_default_to_one():
    g = param.get_global_parameters(_args(asrv="each"))
    assert g["asrv_dirichlet_alpha"] == pytest.approx(1.0)


def test_get_global_parameters_rejects_invalid_asrv_dirichlet_alpha():
    with pytest.raises(ValueError, match="asrv_dirichlet_alpha"):
        param.get_global_parameters(_args(asrv_dirichlet_alpha=-0.1))


def test_get_global_parameters_rejects_invalid_asrv_mode():
    with pytest.raises(ValueError, match="--asrv"):
        param.get_global_parameters(_args(asrv="hybrid"))

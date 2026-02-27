import argparse

import numpy as np
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
    assert g["sub_tensor_auto_sparse_min_elements"] == 100000000
    assert g["parallel_backend"] == "auto"
    assert g["parallel_chunk_factor"] == 1
    assert g["parallel_chunk_factor_reducer"] == 4
    assert g["parallel_min_items_sub_tensor"] == 256
    assert g["parallel_min_items_per_job_sub_tensor"] == 64
    assert g["parallel_min_items_cb"] == 20000
    assert g["parallel_min_rows_cbs"] == 200000
    assert g["parallel_min_items_branch_dist"] == 20000
    assert g["parallel_min_items_expected_state"] == 50000000
    assert g["parallel_min_items_per_job_expected_state"] == 10000000


def test_get_global_parameters_rejects_unsupported_float_type():
    with pytest.raises(ValueError, match="float_type"):
        param.get_global_parameters(_args(float_type=128))


def test_get_global_parameters_normalizes_sub_tensor_backend_case():
    g = param.get_global_parameters(_args(sub_tensor_backend="SpArSe"))
    assert g["sub_tensor_backend"] == "sparse"


def test_get_global_parameters_prints_dependency_versions(capsys):
    param.get_global_parameters(_args())
    captured = capsys.readouterr()
    assert "CSUBST dependency versions:" in captured.out
    assert "CSUBST missing dependency packages:" in captured.out
    for package_name in param.DEPENDENCY_DISTRIBUTIONS:
        assert "{}=".format(package_name) in captured.out


def test_get_global_parameters_reports_missing_dependency_packages(monkeypatch, capsys):
    missing_package = param.DEPENDENCY_DISTRIBUTIONS[0]

    def _mock_get_dependency_version(distribution_name):
        if distribution_name == missing_package:
            return "not installed"
        return "1.0.0"

    monkeypatch.setattr(param, "_get_dependency_version", _mock_get_dependency_version)
    param.get_global_parameters(_args())
    captured = capsys.readouterr()
    txt = "CSUBST missing dependency packages: {}".format(missing_package)
    assert txt in captured.out


def test_get_global_parameters_rejects_invalid_sub_tensor_backend():
    with pytest.raises(ValueError, match="sub_tensor_backend"):
        param.get_global_parameters(_args(sub_tensor_backend="not-a-backend"))


def test_get_global_parameters_rejects_invalid_sparse_density_cutoff():
    with pytest.raises(ValueError, match="sub_tensor_sparse_density_cutoff"):
        param.get_global_parameters(_args(sub_tensor_sparse_density_cutoff=1.5))
    with pytest.raises(ValueError, match="sub_tensor_auto_sparse_min_elements"):
        param.get_global_parameters(_args(sub_tensor_auto_sparse_min_elements=-1))


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"sub_tensor_sparse_density_cutoff": float("nan")}, "sub_tensor_sparse_density_cutoff"),
        ({"min_single_prob": float("nan")}, "min_single_prob"),
        ({"min_combinat_prob": float("inf")}, "min_combinat_prob"),
        ({"percent_biased_sub": float("inf")}, "percent_biased_sub"),
        ({"database_timeout": float("inf")}, "database_timeout"),
        ({"mafft_op": float("nan")}, "mafft_op"),
    ],
)
def test_get_global_parameters_rejects_non_finite_float_values(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        param.get_global_parameters(_args(**kwargs))


def test_get_global_parameters_rejects_invalid_parallel_backend():
    with pytest.raises(ValueError, match="parallel_backend"):
        param.get_global_parameters(_args(parallel_backend="invalid"))


def test_get_global_parameters_rejects_invalid_chunk_factors():
    with pytest.raises(ValueError, match="parallel_chunk_factor"):
        param.get_global_parameters(_args(parallel_chunk_factor=0))
    with pytest.raises(ValueError, match="parallel_chunk_factor_reducer"):
        param.get_global_parameters(_args(parallel_chunk_factor_reducer=0))


def test_get_global_parameters_rejects_invalid_parallel_auto_thresholds():
    with pytest.raises(ValueError, match="parallel_min_items_sub_tensor"):
        param.get_global_parameters(_args(parallel_min_items_sub_tensor=-1))
    with pytest.raises(ValueError, match="parallel_min_items_per_job_sub_tensor"):
        param.get_global_parameters(_args(parallel_min_items_per_job_sub_tensor=0))
    with pytest.raises(ValueError, match="parallel_min_rows_per_job_cbs"):
        param.get_global_parameters(_args(parallel_min_rows_per_job_cbs=0))
    with pytest.raises(ValueError, match="parallel_min_items_expected_state"):
        param.get_global_parameters(_args(parallel_min_items_expected_state=-1))
    with pytest.raises(ValueError, match="parallel_min_items_per_job_expected_state"):
        param.get_global_parameters(_args(parallel_min_items_per_job_expected_state=0))


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


@pytest.mark.parametrize(
    "cutoff_stat,expected",
    [
        ("OCNany2spe|omegaCany2spe,5.0", "Invalid --cutoff_stat token"),
        ("OCN[any2spe,2.0", "Invalid cutoff regex"),
        ("OCNany2spe,nan", "finite"),
    ],
)
def test_get_global_parameters_rejects_malformed_cutoff_stat(cutoff_stat, expected):
    with pytest.raises(ValueError, match=expected):
        param.get_global_parameters(_args(cutoff_stat=cutoff_stat))


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


def test_get_global_parameters_rejects_incompatible_qc_cutoff_stat():
    with pytest.raises(ValueError, match='requires --output_stat to include "any2spe"'):
        param.get_global_parameters(
            _args(
                output_stat="any2any",
                cutoff_stat="QCNany2spe,0.95|omegaCany2any,5.0",
            )
        )


def test_get_global_parameters_rejects_incompatible_regex_cutoff_stat():
    with pytest.raises(ValueError, match='requires --output_stat to include "any2spe,dif2spe"'):
        param.get_global_parameters(
            _args(
                output_stat="any2any",
                cutoff_stat=r"OCN(any|dif)2spe,2.0|omegaCany2any,5.0",
            )
        )


def test_get_global_parameters_accepts_regex_cutoff_stat_when_output_stats_cover_all_matches():
    g = param.get_global_parameters(
        _args(
            output_stat="any2any,any2spe,dif2spe",
            cutoff_stat=r"OCN(any|dif)2spe,2.0|omegaCany2any,5.0",
        )
    )
    assert g["output_stats"] == ["any2any", "any2spe", "dif2spe"]


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


def test_get_global_parameters_requires_foreground_for_exhaustive_until_one():
    with pytest.raises(ValueError, match="exhaustive_until 1"):
        param.get_global_parameters(_args(exhaustive_until=1, foreground=None))


def test_get_global_parameters_requires_foreground_for_clade_permutation():
    with pytest.raises(ValueError, match="fg_clade_permutation"):
        param.get_global_parameters(_args(fg_clade_permutation=1, foreground=None))


def test_get_global_parameters_rejects_calc_quantile_without_modelfree():
    with pytest.raises(ValueError, match='--omegaC_method "modelfree"'):
        param.get_global_parameters(_args(calc_quantile=True, omegaC_method="submodel"))


def test_get_global_parameters_rejects_calc_omega_pvalue_without_modelfree():
    with pytest.raises(ValueError, match='--omegaC_method "modelfree"'):
        param.get_global_parameters(_args(calc_omega_pvalue=True, omegaC_method="submodel"))


def test_get_global_parameters_sets_omega_pvalue_defaults():
    g = param.get_global_parameters(_args())
    assert g["calc_omega_pvalue"] is False
    assert g["omega_pvalue_null_model"] == "poisson"
    assert g["omega_pvalue_niter"] == 1000
    assert g["omega_pvalue_rounding"] == "round"


def test_get_global_parameters_rejects_invalid_omega_pvalue_niter():
    with pytest.raises(ValueError, match="omega_pvalue_niter"):
        param.get_global_parameters(_args(omega_pvalue_niter=0))
    with pytest.raises(ValueError, match="omega_pvalue_niter"):
        param.get_global_parameters(_args(omega_pvalue_niter=100000))


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
            omegaC_method="modelfree",
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
            omegaC_method="modelfree",
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
            omegaC_method="modelfree",
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


def test_get_global_parameters_sets_epistasis_defaults():
    g = param.get_global_parameters(_args())
    assert g["epistasis_apply_to"] == "N"
    assert g["epistasis_site_metric"] == "off"
    assert g["epistasis_beta_auto"] is False
    assert g["epistasis_beta_value"] == pytest.approx(0.0)
    assert g["epistasis_clip_auto"] is False
    assert g["epistasis_clip_value"] == pytest.approx(3.0)
    assert g["epistasis_beta_partition"] == "global"
    assert g["epistasis_branch_depth_bins"] == 3
    assert g["epistasis_feature_mode"] == "single"
    assert g["epistasis_joint_auto"] is False
    assert g["epistasis_joint_alpha_grid"] == [0.0, 0.5, 1.0, 2.0]
    assert g["epistasis_joint_clip_grid"] == [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    assert g["epistasis_requested"] is False


def test_get_global_parameters_parses_epistasis_auto_and_apply_to_s():
    g = param.get_global_parameters(
        _args(
            epistasis_apply_to="S",
            epistasis_site_metric="proximity",
            epistasis_beta="auto",
            epistasis_clip="auto",
            epistasis_beta_partition="branch_depth",
            epistasis_branch_depth_bins=4,
            epistasis_feature_mode="paired",
            epistasis_joint_auto="yes",
            epistasis_joint_alpha_grid="0,0.25,1",
            epistasis_joint_clip_grid="1.5,2,3",
        )
    )
    assert g["epistasis_apply_to"] == "S"
    assert g["epistasis_site_metric"] == "proximity"
    assert g["epistasis_beta_auto"] is True
    assert g["epistasis_clip_auto"] is True
    assert g["epistasis_beta_partition"] == "branch_depth"
    assert g["epistasis_branch_depth_bins"] == 4
    assert g["epistasis_feature_mode"] == "paired"
    assert g["epistasis_joint_auto"] is True
    assert g["epistasis_joint_alpha_grid"] == [0.0, 0.25, 1.0]
    assert g["epistasis_joint_clip_grid"] == [1.5, 2.0, 3.0]
    assert g["epistasis_requested"] is True


def test_get_global_parameters_auto_promotes_site_metric_when_epistasis_is_active():
    g = param.get_global_parameters(_args(epistasis_beta="0.5"))
    assert g["epistasis_beta_auto"] is False
    assert g["epistasis_beta_value"] == pytest.approx(0.5)
    assert g["epistasis_site_metric"] == "auto"
    assert g["epistasis_requested"] is True


def test_get_global_parameters_rejects_invalid_epistasis_options():
    with pytest.raises(ValueError, match="epistasis_apply_to"):
        param.get_global_parameters(_args(epistasis_apply_to="X"))
    with pytest.raises(ValueError, match="epistasis_site_metric"):
        param.get_global_parameters(_args(epistasis_site_metric="X"))
    with pytest.raises(ValueError, match="epistasis_beta"):
        param.get_global_parameters(_args(epistasis_beta="-0.1"))
    with pytest.raises(ValueError, match="epistasis_clip"):
        param.get_global_parameters(_args(epistasis_clip="0"))
    with pytest.raises(ValueError, match="epistasis_beta_partition"):
        param.get_global_parameters(_args(epistasis_beta_partition="invalid"))
    with pytest.raises(ValueError, match="epistasis_branch_depth_bins"):
        param.get_global_parameters(_args(epistasis_branch_depth_bins=0))
    with pytest.raises(ValueError, match="epistasis_feature_mode"):
        param.get_global_parameters(_args(epistasis_feature_mode="invalid"))
    with pytest.raises(ValueError, match="epistasis_joint_alpha_grid"):
        param.get_global_parameters(_args(epistasis_joint_alpha_grid="-1,0"))
    with pytest.raises(ValueError, match="epistasis_joint_clip_grid"):
        param.get_global_parameters(_args(epistasis_joint_clip_grid="0,1"))


def test_get_global_parameters_sets_pseudocount_defaults():
    g = param.get_global_parameters(_args())
    assert g["pseudocount_alpha"] == pytest.approx(0.0)
    assert g["pseudocount_alpha_auto"] is False
    assert g["pseudocount_mode"] == "none"
    assert g["pseudocount_target"] == "both"
    assert g["pseudocount_enabled"] is False
    assert g["pseudocount_add_output_columns"] is False


def test_get_global_parameters_accepts_auto_pseudocount_alpha():
    g = param.get_global_parameters(
        _args(
            pseudocount_alpha="auto",
            pseudocount_mode="symmetric",
            pseudocount_target="both",
        )
    )
    assert g["pseudocount_alpha"] == pytest.approx(0.0)
    assert g["pseudocount_alpha_auto"] is True
    assert g["pseudocount_enabled"] is True


def test_get_global_parameters_accepts_pseudocount_options():
    g = param.get_global_parameters(
        _args(
            pseudocount_alpha=0.5,
            pseudocount_mode="empirical",
            pseudocount_target="expected",
            pseudocount_report=True,
        )
    )
    assert g["pseudocount_alpha"] == pytest.approx(0.5)
    assert g["pseudocount_mode"] == "empirical"
    assert g["pseudocount_target"] == "expected"
    assert g["pseudocount_enabled"] is True
    assert g["pseudocount_add_output_columns"] is True


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"pseudocount_alpha": -0.1}, "pseudocount_alpha"),
        ({"pseudocount_alpha": float("nan")}, "pseudocount_alpha"),
        ({"pseudocount_alpha": "abc"}, "pseudocount_alpha"),
        ({"pseudocount_mode": "invalid"}, "pseudocount_mode"),
        ({"pseudocount_target": "invalid"}, "pseudocount_target"),
    ],
)
def test_get_global_parameters_rejects_invalid_pseudocount_options(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        param.get_global_parameters(_args(**kwargs))


def test_get_global_parameters_rejects_removed_pseudocount_strength_option():
    with pytest.raises(ValueError, match="pseudocount_strength"):
        param.get_global_parameters(_args(pseudocount_strength=2.0))


def test_get_global_parameters_sets_quantile_refine_defaults():
    g = param.get_global_parameters(_args())
    assert g["quantile_refine_edge_bins"] == 2
    assert g["quantile_niter_schedule"] == [100, 1000, 10000]


def test_get_global_parameters_parses_quantile_schedule_auto_alias():
    g = param.get_global_parameters(_args(quantile_niter_schedule="auto"))
    assert g["quantile_niter_schedule"] == [100, 1000, 10000]


def test_get_global_parameters_parses_custom_quantile_schedule():
    g = param.get_global_parameters(_args(quantile_niter_schedule="200,2000,10000"))
    assert g["quantile_niter_schedule"] == [200, 2000, 10000]


def test_get_global_parameters_rejects_invalid_quantile_schedule():
    with pytest.raises(ValueError, match="integers"):
        param.get_global_parameters(_args(quantile_niter_schedule="100,abc"))
    with pytest.raises(ValueError, match="strictly increasing"):
        param.get_global_parameters(_args(quantile_niter_schedule="1000,100"))
    with pytest.raises(ValueError, match="upper bound"):
        param.get_global_parameters(_args(quantile_niter_schedule="100,100000"))


def test_get_global_parameters_rejects_negative_quantile_refine_edge_bins():
    with pytest.raises(ValueError, match="quantile_refine_edge_bins"):
        param.get_global_parameters(_args(quantile_refine_edge_bins=-1))


def test_get_global_parameters_rejects_invalid_percent_biased_sub():
    with pytest.raises(ValueError, match="percent_biased_sub"):
        param.get_global_parameters(_args(percent_biased_sub=-1))
    with pytest.raises(ValueError, match="percent_biased_sub"):
        param.get_global_parameters(_args(percent_biased_sub=100))


def test_get_global_parameters_rejects_invalid_site_probability_thresholds():
    with pytest.raises(ValueError, match="tree_site_plot_max_sites"):
        param.get_global_parameters(_args(tree_site_plot_max_sites=0))
    with pytest.raises(ValueError, match="min_single_prob"):
        param.get_global_parameters(_args(min_single_prob=1.1))
    with pytest.raises(ValueError, match="min_single_prob"):
        param.get_global_parameters(_args(min_single_prob=-0.1))
    with pytest.raises(ValueError, match="min_combinat_prob"):
        param.get_global_parameters(_args(min_combinat_prob=1.1))
    with pytest.raises(ValueError, match="min_combinat_prob"):
        param.get_global_parameters(_args(min_combinat_prob=-0.1))


def test_get_global_parameters_parses_uniprot_include_redundant_string_bool():
    g_true = param.get_global_parameters(_args(uniprot_include_redundant="true"))
    g_false = param.get_global_parameters(_args(uniprot_include_redundant="false"))
    assert g_true["uniprot_include_redundant"] is True
    assert g_false["uniprot_include_redundant"] is False


def test_get_global_parameters_rejects_invalid_uniprot_include_redundant_string():
    with pytest.raises(ValueError, match="uniprot_include_redundant"):
        param.get_global_parameters(_args(uniprot_include_redundant="maybe"))


def test_get_global_parameters_parses_drop_invariant_tip_sites_single_option():
    g_no = param.get_global_parameters(_args(drop_invariant_tip_sites="no"))
    g_tip = param.get_global_parameters(_args(drop_invariant_tip_sites="tip_invariant"))
    g_zero = param.get_global_parameters(_args(drop_invariant_tip_sites="zero_sub_mass"))
    assert g_no["drop_invariant_tip_sites"] is False
    assert g_no["drop_invariant_tip_sites_mode"] == "tip_invariant"
    assert g_tip["drop_invariant_tip_sites"] is True
    assert g_tip["drop_invariant_tip_sites_mode"] == "tip_invariant"
    assert g_zero["drop_invariant_tip_sites"] is True
    assert g_zero["drop_invariant_tip_sites_mode"] == "zero_sub_mass"


def test_get_global_parameters_rejects_invalid_drop_invariant_tip_sites_string():
    with pytest.raises(ValueError, match="drop_invariant_tip_sites"):
        param.get_global_parameters(_args(drop_invariant_tip_sites="maybe"))
    with pytest.raises(ValueError, match="drop_invariant_tip_sites"):
        param.get_global_parameters(_args(drop_invariant_tip_sites="yes"))
    with pytest.raises(ValueError, match="drop_invariant_tip_sites"):
        param.get_global_parameters(_args(drop_invariant_tip_sites="true"))
    with pytest.raises(ValueError, match="drop_invariant_tip_sites"):
        param.get_global_parameters(_args(drop_invariant_tip_sites="false"))


def test_get_global_parameters_parses_sa_asr_mode_values():
    g_default = param.get_global_parameters(_args())
    g_translate = param.get_global_parameters(_args(sa_asr_mode="translate"))
    g_direct = param.get_global_parameters(_args(sa_asr_mode="direct"))
    assert g_default["sa_asr_mode"] == "direct"
    assert g_translate["sa_asr_mode"] == "translate"
    assert g_direct["sa_asr_mode"] == "direct"


def test_get_global_parameters_rejects_invalid_sa_asr_mode():
    with pytest.raises(ValueError, match="sa_asr_mode"):
        param.get_global_parameters(_args(sa_asr_mode="invalid"))
    with pytest.raises(ValueError, match="sa_iqtree_model"):
        param.get_global_parameters(_args(sa_iqtree_model=""))


def test_get_global_parameters_parses_sa_smoke_max_branches():
    g = param.get_global_parameters(_args(sa_smoke_max_branches=5))
    assert g["sa_smoke_max_branches"] == 5


def test_get_global_parameters_rejects_negative_sa_smoke_max_branches():
    with pytest.raises(ValueError, match="sa_smoke_max_branches"):
        param.get_global_parameters(_args(sa_smoke_max_branches=-1))


def test_get_global_parameters_parses_prostt5_options():
    g = param.get_global_parameters(
        _args(
            prostt5_local_dir=" /tmp/prostt5 ",
            prostt5_no_download="yes",
            prostt5_device="MPS",
            prostt5_cache="yes",
            prostt5_cache_file=" /tmp/prostt5_cache.tsv ",
        )
    )
    assert g["prostt5_local_dir"] == "/tmp/prostt5"
    assert g["prostt5_no_download"] is True
    assert g["prostt5_device"] == "mps"
    assert g["prostt5_cache"] is True
    assert g["prostt5_cache_file"] == "/tmp/prostt5_cache.tsv"


def test_get_global_parameters_rejects_invalid_prostt5_no_download():
    with pytest.raises(ValueError, match="prostt5_no_download"):
        param.get_global_parameters(_args(prostt5_no_download="maybe"))


def test_get_global_parameters_rejects_invalid_prostt5_cache():
    with pytest.raises(ValueError, match="prostt5_cache"):
        param.get_global_parameters(_args(prostt5_cache="maybe"))


def test_get_global_parameters_rejects_empty_prostt5_cache_file():
    with pytest.raises(ValueError, match="prostt5_cache_file"):
        param.get_global_parameters(_args(prostt5_cache_file=""))


def test_get_global_parameters_sets_default_prostt5_cache_file():
    g = param.get_global_parameters(_args())
    assert g["prostt5_cache_file"] == "csubst_prostt5_cache.tsv"


def test_get_global_parameters_parses_sa_state_cache_options():
    g_default = param.get_global_parameters(_args())
    assert g_default["sa_state_cache"] == "auto"
    assert g_default["sa_state_cache_file"] == "csubst_3di_state_cache.npz"

    g_custom = param.get_global_parameters(
        _args(
            sa_state_cache="YES",
            sa_state_cache_file=" /tmp/3di_state_cache.npz ",
        )
    )
    assert g_custom["sa_state_cache"] == "yes"
    assert g_custom["sa_state_cache_file"] == "/tmp/3di_state_cache.npz"


def test_get_global_parameters_rejects_invalid_sa_state_cache_mode():
    with pytest.raises(ValueError, match="sa_state_cache"):
        param.get_global_parameters(_args(sa_state_cache="maybe"))


def test_get_global_parameters_rejects_empty_sa_state_cache_file():
    with pytest.raises(ValueError, match="sa_state_cache_file"):
        param.get_global_parameters(_args(sa_state_cache_file=""))


def test_get_global_parameters_parses_plot_nonsyn_recode_pca_3di20_bool():
    g_default = param.get_global_parameters(_args())
    assert g_default["plot_nonsyn_recode_pca_3di20"] is False
    g_yes = param.get_global_parameters(_args(plot_nonsyn_recode_pca_3di20="yes"))
    assert g_yes["plot_nonsyn_recode_pca_3di20"] is True
    with pytest.raises(ValueError, match="plot_nonsyn_recode_pca_3di20"):
        param.get_global_parameters(_args(plot_nonsyn_recode_pca_3di20="maybe"))


def test_get_global_parameters_requires_full_cds_alignment_for_3di20():
    with pytest.raises(ValueError, match="full_cds_alignment_file"):
        param.get_global_parameters(_args(nonsyn_recode="3di20"))


def test_get_global_parameters_disables_alignment_file_for_3di20():
    with pytest.raises(ValueError, match="alignment_file is disabled"):
        param.get_global_parameters(
            _args(
                nonsyn_recode="3di20",
                alignment_file="trimmed.fa",
                full_cds_alignment_file="full.fa",
            )
        )
    g = param.get_global_parameters(
        _args(
            nonsyn_recode="3di20",
            alignment_file="",
            full_cds_alignment_file="full.fa",
        )
    )
    assert g["alignment_file"] == "full.fa"
    g_alias = param.get_global_parameters(
        _args(
            nonsyn_recode="3di",
            alignment_file="",
            full_cds_alignment_file="full_alias.fa",
        )
    )
    assert g_alias["nonsyn_recode"] == "3di20"
    assert g_alias["alignment_file"] == "full_alias.fa"


def test_get_global_parameters_infers_iqtree_paths_from_gz_alignment():
    g = param.get_global_parameters(
        _args(
            alignment_file="input.fa.gz",
            iqtree_treefile="infer",
            iqtree_state="infer",
            iqtree_rate="infer",
            iqtree_iqtree="infer",
        )
    )
    assert g["iqtree_treefile"] == "input.fa.treefile"
    assert g["iqtree_state"] == "input.fa.state"
    assert g["iqtree_rate"] == "input.fa.rate"
    assert g["iqtree_iqtree"] == "input.fa.iqtree"


def test_get_global_parameters_validates_database_timeout():
    g = param.get_global_parameters(_args(database_timeout=12))
    assert g["database_timeout"] == pytest.approx(12.0)
    with pytest.raises(ValueError, match="database_timeout"):
        param.get_global_parameters(_args(database_timeout=0))
    with pytest.raises(ValueError, match="database_timeout"):
        param.get_global_parameters(_args(database_timeout=-1))


def test_get_global_parameters_validates_site_database_and_pymol_ranges():
    g = param.get_global_parameters(
        _args(
            database_evalue_cutoff=1.0,
            database_minimum_identity=0.25,
            mafft_op=-1,
            mafft_ep=0.2,
            pymol_gray=80,
            pymol_transparency=0.65,
            pymol_max_num_chain=20,
        )
    )
    assert g["database_evalue_cutoff"] == pytest.approx(1.0)
    assert g["database_minimum_identity"] == pytest.approx(0.25)
    assert g["mafft_op"] == pytest.approx(-1.0)
    assert g["mafft_ep"] == pytest.approx(0.2)
    assert g["pymol_gray"] == 80
    assert g["pymol_transparency"] == pytest.approx(0.65)
    assert g["pymol_max_num_chain"] == 20

    with pytest.raises(ValueError, match="database_evalue_cutoff"):
        param.get_global_parameters(_args(database_evalue_cutoff=0))
    with pytest.raises(ValueError, match="database_evalue_cutoff"):
        param.get_global_parameters(_args(database_evalue_cutoff=-1))

    with pytest.raises(ValueError, match="database_minimum_identity"):
        param.get_global_parameters(_args(database_minimum_identity=-0.1))
    with pytest.raises(ValueError, match="database_minimum_identity"):
        param.get_global_parameters(_args(database_minimum_identity=1.1))

    with pytest.raises(ValueError, match="mafft_op"):
        param.get_global_parameters(_args(mafft_op=-2))
    with pytest.raises(ValueError, match="mafft_ep"):
        param.get_global_parameters(_args(mafft_ep=-2))

    with pytest.raises(ValueError, match="pymol_gray"):
        param.get_global_parameters(_args(pymol_gray=-1))
    with pytest.raises(ValueError, match="pymol_gray"):
        param.get_global_parameters(_args(pymol_gray=101))

    with pytest.raises(ValueError, match="pymol_transparency"):
        param.get_global_parameters(_args(pymol_transparency=-0.1))
    with pytest.raises(ValueError, match="pymol_transparency"):
        param.get_global_parameters(_args(pymol_transparency=1.1))

    with pytest.raises(ValueError, match="pymol_max_num_chain"):
        param.get_global_parameters(_args(pymol_max_num_chain=0))


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


def test_resolve_sub_tensor_backend_auto_to_dense():
    g = {"sub_tensor_backend": "auto"}
    assert substitution.resolve_sub_tensor_backend(g) == "dense"
    assert g["resolved_sub_tensor_backend"] == "dense"


def test_resolve_sub_tensor_backend_sparse_is_sparse():
    g = {"sub_tensor_backend": "sparse"}
    assert substitution.resolve_sub_tensor_backend(g) == "sparse"
    assert g["resolved_sub_tensor_backend"] == "sparse"


def test_resolve_sub_tensor_backend_auto_can_switch_to_sparse_for_large_tensors():
    g = {"sub_tensor_backend": "auto", "sub_tensor_auto_sparse_min_elements": 1000}
    state = np.zeros((2, 2, 20), dtype=float)
    assert substitution.resolve_sub_tensor_backend(g=g, state_tensor=state, mode="asis") == "sparse"
    assert g["resolved_sub_tensor_backend"] == "sparse"


def test_resolve_sub_tensor_backend_auto_keeps_dense_for_small_tensors():
    g = {"sub_tensor_backend": "auto", "sub_tensor_auto_sparse_min_elements": 100000}
    state = np.zeros((2, 2, 20), dtype=float)
    assert substitution.resolve_sub_tensor_backend(g=g, state_tensor=state, mode="asis") == "dense"
    assert g["resolved_sub_tensor_backend"] == "dense"

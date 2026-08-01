from factories import make_args as _args

import subprocess
import sys

import numpy as np
import pytest

from csubst import param
from csubst import output_stat
from csubst import runtime




def test_get_global_parameters_sets_fixed_numerical_defaults():
    g = param.get_global_parameters(_args())
    assert g["infile_type"] == "iqtree"
    assert g["random_seed"] == 1
    assert g["expected_state_backend"] == "auto"
    assert g["float_type"] is np.float64
    assert g["float_tol"] == pytest.approx(1e-9)
    assert "sub_tensor_backend" not in g
    assert "parallel_backend" not in g


@pytest.mark.slow
def test_importing_param_does_not_eagerly_import_numerical_recoding_module():
    code = "import sys; import csubst.param; assert 'csubst.recoding' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_get_global_parameters_sets_vesm_defaults():
    g = param.get_global_parameters(_args())
    assert g["vep_model"] == "none"
    assert g["vep_min_event_pp"] == pytest.approx(0.8)
    assert g["vep_plot"] is True
    assert g["vep_site_aggregate"] == "most_deleterious"
    assert g["vep_device"] == "auto"
    assert g["vep_cache"] is True
    assert g["pymol_color_by"] == "auto"


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_get_global_parameters_validates_vesm_event_pp(value):
    with pytest.raises(ValueError, match="vep_min_event_pp"):
        param.get_global_parameters(_args(vep_min_event_pp=value))


def test_get_global_parameters_validates_vesm_related_choices():
    with pytest.raises(ValueError, match="vep_model"):
        param.get_global_parameters(_args(vep_model="unknown"))
    with pytest.raises(ValueError, match="vep_site_aggregate"):
        param.get_global_parameters(_args(vep_site_aggregate="median"))
    with pytest.raises(ValueError, match="vep_device"):
        param.get_global_parameters(_args(vep_device="tpu"))
    with pytest.raises(ValueError, match="pymol_color_by vesm requires"):
        param.get_global_parameters(_args(pymol_color_by="vesm", vep_model="none"))


def test_get_global_parameters_builds_run_context_and_output_namespace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = param.get_global_parameters(_args(outdir="results", output_prefix="scan"))
    assert isinstance(g, runtime.RunContext)
    assert g["outdir"] == str((tmp_path / "results").resolve())
    assert g["output_prefix"] == "scan"
    assert g["log_file"] == str((tmp_path / "results" / "scan.log").resolve())
    assert g.config["threads"] == 1
    g["threads"] = 8
    assert g["threads"] == 8
    assert g.config["threads"] == 1


@pytest.mark.parametrize("seed", [-2, 1.5, "1.5", True])
def test_get_global_parameters_rejects_invalid_random_seed(seed):
    with pytest.raises(ValueError, match="random_seed"):
        param.get_global_parameters(_args(random_seed=seed))


def test_get_global_parameters_rejects_output_prefix_paths():
    with pytest.raises(ValueError, match="output_prefix"):
        param.get_global_parameters(_args(output_prefix="nested/run"))


def test_get_global_parameters_normalizes_expected_state_backend_case():
    g = param.get_global_parameters(_args(expected_state_backend="EiGeN"))
    assert g["expected_state_backend"] == "eigen"


def test_get_global_parameters_rejects_invalid_expected_state_backend():
    with pytest.raises(ValueError, match="expected_state_backend"):
        param.get_global_parameters(_args(expected_state_backend="invalid"))


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


@pytest.mark.parametrize(
    "kwargs,expected",
    [
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

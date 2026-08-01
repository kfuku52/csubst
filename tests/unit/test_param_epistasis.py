from factories import make_args as _args


import pytest

from csubst import param




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

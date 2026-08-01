from factories import make_args as _args


import pytest

from csubst import param
from csubst import runtime




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


def test_get_global_parameters_parses_write_instantaneous_rate_matrix_bool():
    g_default = param.get_global_parameters(_args())
    assert g_default["write_instantaneous_rate_matrix"] is False
    g_yes = param.get_global_parameters(_args(write_instantaneous_rate_matrix="yes"))
    assert g_yes["write_instantaneous_rate_matrix"] is True
    with pytest.raises(ValueError, match="write_instantaneous_rate_matrix"):
        param.get_global_parameters(_args(write_instantaneous_rate_matrix="maybe"))


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


def test_get_global_parameters_infers_iqtree_paths_from_gz_alignment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = param.get_global_parameters(
        _args(
            alignment_file="input.fa.gz",
            iqtree_treefile="infer",
            iqtree_state="infer",
            iqtree_rate="infer",
            iqtree_iqtree="infer",
        )
    )
    prefix = runtime.infer_iqtree_output_prefix(
        alignment_file="input.fa.gz",
        iqtree_outdir=str(tmp_path / "csubst_iqtree"),
    )
    assert g["iqtree_treefile"] == prefix + ".treefile"
    assert g["iqtree_state"] == prefix + ".state"
    assert g["iqtree_rate"] == prefix + ".rate"
    assert g["iqtree_iqtree"] == prefix + ".iqtree"


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
            pymol_surface_quality=-1,
            pymol_max_num_chain=20,
        )
    )
    assert g["database_evalue_cutoff"] == pytest.approx(1.0)
    assert g["database_minimum_identity"] == pytest.approx(0.25)
    assert g["mafft_op"] == pytest.approx(-1.0)
    assert g["mafft_ep"] == pytest.approx(0.2)
    assert g["pymol_gray"] == 80
    assert g["pymol_transparency"] == pytest.approx(0.65)
    assert g["pymol_surface_quality"] == -1
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

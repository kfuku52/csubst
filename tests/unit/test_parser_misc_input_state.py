import numpy as np
import pytest

from csubst import parser_misc
from csubst import sequence
from csubst import ete
from csubst import tree


def test_read_input_submodel_rejects_unsupported_substitution_model(monkeypatch):
    def fake_get_input_information(local_g):
        local_g["substitution_model"] = "UNSUPPORTED+F+R4"
        return local_g

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_input_information", fake_get_input_information)
    g = {
        "infile_type": "iqtree",
        "expectation_method": "codon_model",
    }
    with pytest.raises(ValueError, match="Unsupported substitution model"):
        parser_misc.read_input(g)


def test_read_input_submodel_detects_reverse_signed_rate_sum_mismatch(monkeypatch):
    def fake_get_input_information(local_g):
        local_g.update(
            {
                "substitution_model": "GY+F+R4",
                "omega": 1.0,
                "kappa": 1.0,
                "equilibrium_frequency": np.array([0.5, 0.5], dtype=float),
                "codon_orders": np.array(["AAA", "AAC"]),
                "amino_acid_orders": np.array(["K", "N"]),
                "codon_table": [("K", "AAA"), ("N", "AAC")],
                "reconstruction_codon_table": [("K", "AAA"), ("N", "AAC")],
                "synonymous_indices": {"K": [0], "N": [1]},
                "matrix_groups": {"K": ["AAA"], "N": ["AAC"]},
            }
        )
        return local_g

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_input_information", fake_get_input_information)
    monkeypatch.setattr(
        parser_misc,
        "get_mechanistic_instantaneous_rate_matrix",
        lambda g: np.array([[-0.75, 0.75], [0.75, -0.75]], dtype=float),
    )
    monkeypatch.setattr(
        parser_misc,
        "cdn2pep_matrix",
        lambda inst_cdn, g: np.array([[-2.0, 2.0], [2.0, -2.0]], dtype=float),
    )

    def fake_get_rate_tensor(inst, mode, g):
        if mode == "syn":
            return np.array([[[1.0, 0.0], [0.0, 0.0]]], dtype=float)
        if mode == "asis":
            return np.array([[[0.0, 0.5], [0.0, 0.0]]], dtype=float)
        raise AssertionError("unexpected mode")

    monkeypatch.setattr(parser_misc, "get_rate_tensor", fake_get_rate_tensor)
    monkeypatch.setattr(parser_misc.np, "savetxt", lambda *args, **kwargs: None)

    g = {
        "infile_type": "iqtree",
        "expectation_method": "codon_model",
        "float_tol": 1e-12,
    }
    with pytest.raises(AssertionError, match="Sum of rates did not match"):
        parser_misc.read_input(g)


def test_read_input_writes_instantaneous_rate_matrix_only_when_enabled(monkeypatch, tmp_path):
    writes = []

    def fake_get_input_information(local_g):
        local_g.update(
            {
                "substitution_model": "GY+F+R4",
                "omega": 1.0,
                "kappa": 1.0,
                "equilibrium_frequency": np.array([0.5, 0.5], dtype=float),
                "codon_orders": np.array(["AAA", "AAC"]),
                "amino_acid_orders": np.array(["K", "N"]),
                "codon_table": [("K", "AAA"), ("N", "AAC")],
                "reconstruction_codon_table": [("K", "AAA"), ("N", "AAC")],
                "synonymous_indices": {"K": [0], "N": [1]},
                "matrix_groups": {"K": ["AAA"], "N": ["AAC"]},
                "float_tol": 1e-12,
                "outdir": str(tmp_path),
                "output_prefix": "run1",
            }
        )
        return local_g

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_input_information", fake_get_input_information)
    monkeypatch.setattr(parser_misc, "_initialize_and_report_nonsyn_recode", lambda g: g)
    monkeypatch.setattr(
        parser_misc,
        "get_mechanistic_instantaneous_rate_matrix",
        lambda g: np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=float),
    )
    monkeypatch.setattr(parser_misc, "cdn2pep_matrix", lambda inst_cdn, g: np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=float))
    monkeypatch.setattr(parser_misc, "cdn2nsy_matrix", lambda inst_cdn, g: np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=float))

    def fake_get_rate_tensor(inst, mode, g):
        if mode == "syn":
            return np.zeros((1, 2, 2), dtype=float)
        if mode == "asis":
            return np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=float)
        raise AssertionError("unexpected mode")

    monkeypatch.setattr(parser_misc, "get_rate_tensor", fake_get_rate_tensor)
    monkeypatch.setattr(
        parser_misc.np,
        "savetxt",
        lambda path, arr, delimiter="\t": writes.append((str(path), np.array(arr, copy=True))),
    )

    g_disabled = {
        "infile_type": "iqtree",
        "expectation_method": "codon_model",
        "float_tol": 1e-12,
        "write_instantaneous_rate_matrix": False,
        "nonsyn_recode": "no",
    }
    parser_misc.read_input(g_disabled)
    assert writes == []

    g_enabled = {
        "infile_type": "iqtree",
        "expectation_method": "codon_model",
        "float_tol": 1e-12,
        "write_instantaneous_rate_matrix": True,
        "nonsyn_recode": "no",
    }
    parser_misc.read_input(g_enabled)
    assert len(writes) == 1
    assert writes[0][0].endswith("run1_instantaneous_rate_matrix.tsv")


def test_prep_state_3di20_translate_uses_translate_builder(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_node = len(list(tr.traverse()))
    state_cdn = np.zeros((num_node, 2, 3), dtype=float)
    state_cdn[:, :, 0] = 1.0
    state_pep = np.zeros((num_node, 2, 20), dtype=float)
    state_pep[:, :, 0] = 1.0
    state_nsy = np.zeros((num_node, 2, 20), dtype=float)
    state_nsy[:, :, 1] = 1.0
    state_orders = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_state_tensor", lambda g, selected_branch_ids=None: state_cdn)
    monkeypatch.setattr(sequence, "cdn2pep_state", lambda state_cdn, g, selected_branch_ids=None: state_pep)
    monkeypatch.setattr(
        sequence,
        "cdn2nsy_state",
        lambda state_cdn, g, selected_branch_ids=None: (_ for _ in ()).throw(AssertionError("unexpected cdn2nsy")),
    )
    called = {"translate": False}

    def _fake_translate_builder(g, state_pep, selected_branch_ids=None):
        called["translate"] = True
        return state_nsy, state_orders, {1: "VV", 2: "VV"}

    monkeypatch.setattr(parser_misc.structural_alphabet, "build_3di_state_from_state_pep", _fake_translate_builder)
    g = {
        "tree": tr,
        "infile_type": "iqtree",
        "input_data_type": "cdn",
        "nonsyn_recode": "3di20",
        "sa_asr_mode": "translate",
    }
    out = parser_misc.prep_state(g)
    assert called["translate"] is True
    assert out["state_nsy"].shape == state_nsy.shape
    assert out["nonsyn_state_orders"].tolist() == state_orders.tolist()
    assert "_3di_alignment_by_branch_id" in out


def test_prep_state_rejects_nucleotide_input_before_iqtree_loader(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))

    def _unexpected_loader(g, selected_branch_ids=None):
        raise AssertionError("unexpected nucleotide loader")

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_state_tensor", _unexpected_loader)
    g = {
        "tree": tr,
        "infile_type": "iqtree",
        "input_data_type": "nuc",
        "nonsyn_recode": "no",
    }

    with pytest.raises(NotImplementedError, match="Non-codon input is obsolete"):
        parser_misc.prep_state(g)


def test_prep_state_default_nonsynonymous_state_reuses_peptide_storage(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_node = len(list(tr.traverse()))
    state_cdn = np.zeros((num_node, 2, 3), dtype=float)
    state_pep = np.zeros((num_node, 2, 20), dtype=float)
    state_pep[:, :, 0] = 1.0

    monkeypatch.setattr(
        parser_misc.parser_iqtree,
        "get_state_tensor",
        lambda g, selected_branch_ids=None: state_cdn,
    )
    monkeypatch.setattr(
        sequence,
        "cdn2pep_state",
        lambda state_cdn, g, selected_branch_ids=None: state_pep,
    )
    monkeypatch.setattr(
        sequence,
        "cdn2nsy_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("duplicate conversion")),
    )

    out = parser_misc.prep_state(
        {
            "tree": tr,
            "infile_type": "iqtree",
            "input_data_type": "cdn",
            "nonsyn_recode": "no",
        }
    )

    assert out["state_nsy"] is out["state_pep"]


def test_prep_state_3di20_direct_uses_direct_builder(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_node = len(list(tr.traverse()))
    state_cdn = np.zeros((num_node, 2, 3), dtype=float)
    state_cdn[:, :, 0] = 1.0
    state_pep = np.zeros((num_node, 2, 20), dtype=float)
    state_pep[:, :, 0] = 1.0
    state_nsy = np.zeros((num_node, 2, 20), dtype=float)
    state_nsy[:, :, 2] = 1.0
    state_orders = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_state_tensor", lambda g, selected_branch_ids=None: state_cdn)
    monkeypatch.setattr(sequence, "cdn2pep_state", lambda state_cdn, g, selected_branch_ids=None: state_pep)
    monkeypatch.setattr(
        parser_misc.structural_alphabet,
        "build_3di_state_from_state_pep",
        lambda g, state_pep, selected_branch_ids=None: (_ for _ in ()).throw(AssertionError("unexpected translate")),
    )
    called = {"direct": False}

    def _fake_direct_builder(g, selected_branch_ids=None, predictor=None):
        called["direct"] = True
        return state_nsy, state_orders, {"A": "VV", "B": "VV"}

    monkeypatch.setattr(parser_misc.structural_alphabet, "build_3di_state_direct", _fake_direct_builder)
    g = {
        "tree": tr,
        "infile_type": "iqtree",
        "input_data_type": "cdn",
        "nonsyn_recode": "3di20",
        "sa_asr_mode": "direct",
    }
    out = parser_misc.prep_state(g)
    assert called["direct"] is True
    assert out["state_nsy"].shape == state_nsy.shape
    assert out["nonsyn_state_orders"].tolist() == state_orders.tolist()
    assert "_3di_tip_alignment_by_leaf" in out


def test_prep_state_3di20_prefers_sa_inference_branch_ids(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_node = len(list(tr.traverse()))
    state_cdn = np.zeros((num_node, 2, 3), dtype=float)
    state_cdn[:, :, 0] = 1.0
    state_pep = np.zeros((num_node, 2, 20), dtype=float)
    state_pep[:, :, 0] = 1.0
    state_nsy = np.zeros((num_node, 2, 20), dtype=float)
    state_nsy[:, :, 3] = 1.0
    state_orders = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_state_tensor", lambda g, selected_branch_ids=None: state_cdn)
    monkeypatch.setattr(sequence, "cdn2pep_state", lambda state_cdn, g, selected_branch_ids=None: state_pep)
    called_selected = {"ids": None}

    def _fake_translate_builder(g, state_pep, selected_branch_ids=None):
        called_selected["ids"] = np.asarray(selected_branch_ids, dtype=np.int64)
        return state_nsy, state_orders, {0: "VV", 1: "VV", 2: "VV"}

    monkeypatch.setattr(parser_misc.structural_alphabet, "build_3di_state_from_state_pep", _fake_translate_builder)
    g = {
        "tree": tr,
        "infile_type": "iqtree",
        "input_data_type": "cdn",
        "nonsyn_recode": "3di20",
        "sa_asr_mode": "translate",
        "state_loaded_branch_ids": np.array([0, 1, 2], dtype=np.int64),
        "sa_inference_branch_ids": np.array([0, 2], dtype=np.int64),
    }
    parser_misc.prep_state(g)
    np.testing.assert_array_equal(called_selected["ids"], np.array([0, 2], dtype=np.int64))


def test_prep_state_3di20_auto_writes_and_reuses_cache(tmp_path, monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_node = len(list(tr.traverse()))
    state_cdn = np.zeros((num_node, 2, 3), dtype=np.float64)
    state_cdn[:, :, 0] = 1.0
    state_pep = np.zeros((num_node, 2, 20), dtype=np.float64)
    state_pep[:, :, 0] = 1.0
    state_nsy = np.zeros((num_node, 2, 20), dtype=np.float64)
    state_nsy[:, :, 4] = 1.0
    state_orders = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    cache_file = tmp_path / "3di_state_cache.npz"
    full_cds_file = tmp_path / "full_cds.fa"
    iqtree_state_file = tmp_path / "input.state"
    full_cds_file.write_text(">A\nATGATG\n>B\nATGATG\n", encoding="utf-8")
    iqtree_state_file.write_text("# dummy\n", encoding="utf-8")

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_state_tensor", lambda g, selected_branch_ids=None: state_cdn)
    monkeypatch.setattr(sequence, "cdn2pep_state", lambda state_cdn, g, selected_branch_ids=None: state_pep)
    called = {"n": 0}

    def _fake_translate_builder(g, state_pep, selected_branch_ids=None):
        called["n"] += 1
        return state_nsy, state_orders, {0: "VV", 1: "VV", 2: "VV"}

    monkeypatch.setattr(parser_misc.structural_alphabet, "build_3di_state_from_state_pep", _fake_translate_builder)
    g_base = {
        "tree": tr,
        "infile_type": "iqtree",
        "input_data_type": "cdn",
        "nonsyn_recode": "3di20",
        "sa_asr_mode": "translate",
        "sa_state_cache": "auto",
        "sa_state_cache_file": str(cache_file),
        "full_cds_alignment_file": str(full_cds_file),
        "alignment_file": str(full_cds_file),
        "path_iqtree_state": str(iqtree_state_file),
        "float_type": np.float64,
    }
    out_first = parser_misc.prep_state(dict(g_base))
    assert called["n"] == 1
    assert cache_file.exists() is True
    assert out_first["state_nsy"].shape == state_nsy.shape

    monkeypatch.setattr(
        parser_misc.structural_alphabet,
        "build_3di_state_from_state_pep",
        lambda g, state_pep, selected_branch_ids=None: (_ for _ in ()).throw(
            AssertionError("3Di builder should not run when cache is reusable")
        ),
    )
    out_second = parser_misc.prep_state(dict(g_base))
    assert called["n"] == 1
    assert out_second["nonsyn_state_orders"].tolist() == state_orders.tolist()
    assert out_second["state_nsy"].shape == state_nsy.shape


def test_prep_state_can_defer_site_filtering(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_node = len(list(tr.traverse()))
    state_cdn = np.zeros((num_node, 2, 3), dtype=float)
    state_cdn[:, :, 0] = 1.0
    state_pep = np.zeros((num_node, 2, 2), dtype=float)
    state_pep[:, :, 0] = 1.0
    state_nsy = np.zeros((num_node, 2, 2), dtype=float)
    state_nsy[:, :, 0] = 1.0

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_state_tensor", lambda g, selected_branch_ids=None: state_cdn)
    monkeypatch.setattr(sequence, "cdn2pep_state", lambda state_cdn, g, selected_branch_ids=None: state_pep)
    monkeypatch.setattr(sequence, "cdn2nsy_state", lambda state_cdn, g, selected_branch_ids=None: state_nsy)

    def fake_drop_invariant_tip_sites(local_g):
        local_g["state_cdn"] = local_g["state_cdn"][:, 1:2, :]
        local_g["state_pep"] = local_g["state_pep"][:, 1:2, :]
        local_g["state_nsy"] = local_g["state_nsy"][:, 1:2, :]
        local_g["site_index_alignment"] = np.array([1], dtype=np.int64)
        return local_g

    monkeypatch.setattr(parser_misc, "drop_invariant_tip_sites", fake_drop_invariant_tip_sites)

    g = {
        "tree": tr,
        "infile_type": "iqtree",
        "input_data_type": "cdn",
        "nonsyn_recode": "no",
        "drop_invariant_tip_sites": True,
    }
    out_unfiltered = parser_misc.prep_state(dict(g), apply_site_filtering=False)
    assert out_unfiltered["state_cdn"].shape[1] == 2
    np.testing.assert_array_equal(out_unfiltered["site_index_alignment"], np.array([0, 1], dtype=np.int64))

    out_filtered = parser_misc.apply_site_filters(out_unfiltered)
    assert out_filtered["state_cdn"].shape[1] == 1
    np.testing.assert_array_equal(out_filtered["site_index_alignment"], np.array([1], dtype=np.int64))


def test_3di_state_cache_context_tracks_model_selection(tmp_path):
    full_cds_file = tmp_path / "full_cds.fa"
    iqtree_state_file = tmp_path / "input.state"
    full_cds_file.write_text(">A\nATGATG\n", encoding="utf-8")
    iqtree_state_file.write_text("# dummy\n", encoding="utf-8")

    g_translate = {
        "sa_asr_mode": "translate",
        "infile_type": "iqtree",
        "input_data_type": "cdn",
        "full_cds_alignment_file": str(full_cds_file),
        "alignment_file": str(full_cds_file),
        "path_iqtree_state": str(iqtree_state_file),
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": "",
        "sa_iqtree_model": "GTR",
        "ml_anc": False,
        "drop_invariant_tip_sites": False,
        "drop_invariant_tip_sites_mode": "tip_invariant",
    }
    ctx_translate_a = parser_misc._get_3di_state_cache_context(
        g=g_translate,
        selected_branch_ids=np.array([0, 2], dtype=np.int64),
        state_cdn_shape=(3, 4, 5),
    )
    ctx_translate_b = parser_misc._get_3di_state_cache_context(
        g=dict(g_translate, prostt5_model="custom-model"),
        selected_branch_ids=np.array([0, 2], dtype=np.int64),
        state_cdn_shape=(3, 4, 5),
    )
    assert ctx_translate_a != ctx_translate_b

    g_direct = dict(g_translate, sa_asr_mode="direct")
    ctx_direct_a = parser_misc._get_3di_state_cache_context(
        g=g_direct,
        selected_branch_ids=np.array([1], dtype=np.int64),
        state_cdn_shape=(3, 4, 5),
    )
    ctx_direct_b = parser_misc._get_3di_state_cache_context(
        g=dict(g_direct, sa_iqtree_model="JC"),
        selected_branch_ids=np.array([1], dtype=np.int64),
        state_cdn_shape=(3, 4, 5),
    )
    assert ctx_direct_a != ctx_direct_b


def test_prep_state_3di20_yes_requires_compatible_cache(tmp_path, monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    num_node = len(list(tr.traverse()))
    state_cdn = np.zeros((num_node, 2, 3), dtype=np.float64)
    state_cdn[:, :, 0] = 1.0
    state_pep = np.zeros((num_node, 2, 20), dtype=np.float64)
    state_pep[:, :, 0] = 1.0
    cache_file = tmp_path / "missing_cache.npz"
    full_cds_file = tmp_path / "full_cds.fa"
    iqtree_state_file = tmp_path / "input.state"
    full_cds_file.write_text(">A\nATGATG\n>B\nATGATG\n", encoding="utf-8")
    iqtree_state_file.write_text("# dummy\n", encoding="utf-8")

    monkeypatch.setattr(parser_misc.parser_iqtree, "get_state_tensor", lambda g, selected_branch_ids=None: state_cdn)
    monkeypatch.setattr(sequence, "cdn2pep_state", lambda state_cdn, g, selected_branch_ids=None: state_pep)
    monkeypatch.setattr(
        parser_misc.structural_alphabet,
        "build_3di_state_from_state_pep",
        lambda g, state_pep, selected_branch_ids=None: (_ for _ in ()).throw(
            AssertionError("3Di builder should not run when --sa_state_cache yes has no cache")
        ),
    )
    g = {
        "tree": tr,
        "infile_type": "iqtree",
        "input_data_type": "cdn",
        "nonsyn_recode": "3di20",
        "sa_asr_mode": "translate",
        "sa_state_cache": "yes",
        "sa_state_cache_file": str(cache_file),
        "full_cds_alignment_file": str(full_cds_file),
        "alignment_file": str(full_cds_file),
        "path_iqtree_state": str(iqtree_state_file),
        "float_type": np.float64,
    }
    with pytest.raises(ValueError, match="sa_state_cache yes"):
        parser_misc.prep_state(g)

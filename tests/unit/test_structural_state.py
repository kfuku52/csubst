import numpy as np
import pytest
from collections import OrderedDict

from csubst import ete
from csubst import structural_alphabet
from csubst import tree


def _build_test_tree():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    node_by_name = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    return tr, node_by_name


def _build_state_pep(node_count, aa_orders, seq_by_node, float_tol=1e-12):
    state_pep = np.zeros((node_count, 3, len(aa_orders)), dtype=float)
    aa_lookup = {aa: i for i, aa in enumerate(aa_orders.tolist())}
    for node_id, seq in seq_by_node.items():
        for site, aa in enumerate(seq):
            if aa == "-":
                continue
            state_pep[node_id, site, aa_lookup[aa]] = 1.0
    return state_pep


def test_build_3di_state_from_state_pep_projects_gaps_and_one_hot():
    tr, node_by_name = _build_test_tree()
    aa_orders = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    seq_by_node = {
        node_by_name["A"]: "AC-",
        node_by_name["B"]: "AD-",
    }
    state_pep = _build_state_pep(
        node_count=len(list(tr.traverse())),
        aa_orders=aa_orders,
        seq_by_node=seq_by_node,
    )
    g = {
        "tree": tr,
        "amino_acid_orders": aa_orders,
        "float_tol": 1e-12,
    }

    def _fake_predictor(aa_sequences, _g):
        out = dict()
        for key, seq in aa_sequences.items():
            out[key] = "VV" if len(seq) == 2 else ("V" * len(seq))
        return out

    state_3di, state_orders, aligned = structural_alphabet.build_3di_state_from_state_pep(
        g=g,
        state_pep=state_pep,
        predictor=_fake_predictor,
    )
    assert state_3di.shape == (len(list(tr.traverse())), 3, 20)
    assert state_orders.tolist() == list("ACDEFGHIKLMNPQRSTVWY")
    assert aligned[node_by_name["A"]] == "VV-"
    assert aligned[node_by_name["B"]] == "VV-"
    v_index = state_orders.tolist().index("V")
    assert state_3di[node_by_name["A"], 0, v_index] == pytest.approx(1.0)
    assert state_3di[node_by_name["A"], 1, v_index] == pytest.approx(1.0)
    assert state_3di[node_by_name["A"], 2, :].sum() == pytest.approx(0.0)


def test_build_3di_state_from_state_pep_raises_on_length_mismatch():
    tr, node_by_name = _build_test_tree()
    aa_orders = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    seq_by_node = {
        node_by_name["A"]: "AC-",
        node_by_name["B"]: "AD-",
    }
    state_pep = _build_state_pep(
        node_count=len(list(tr.traverse())),
        aa_orders=aa_orders,
        seq_by_node=seq_by_node,
    )
    g = {
        "tree": tr,
        "amino_acid_orders": aa_orders,
        "float_tol": 1e-12,
    }

    def _bad_predictor(aa_sequences, _g):
        out = dict()
        for key, seq in aa_sequences.items():
            out[key] = "V" * (len(seq) + 1)
        return out

    with pytest.raises(ValueError, match="length mismatch"):
        structural_alphabet.build_3di_state_from_state_pep(
            g=g,
            state_pep=state_pep,
            predictor=_bad_predictor,
        )


def test_build_3di_state_from_state_pep_includes_root_when_selected():
    tr, node_by_name = _build_test_tree()
    root_id = node_by_name["R"]
    tip_a_id = node_by_name["A"]
    aa_orders = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    seq_by_node = {
        root_id: "AC-",
        tip_a_id: "AC-",
    }
    state_pep = _build_state_pep(
        node_count=len(list(tr.traverse())),
        aa_orders=aa_orders,
        seq_by_node=seq_by_node,
    )
    g = {
        "tree": tr,
        "amino_acid_orders": aa_orders,
        "float_tol": 1e-12,
    }

    def _fake_predictor(aa_sequences, _g):
        return {key: ("V" * len(seq)) for key, seq in aa_sequences.items()}

    state_3di, state_orders, aligned = structural_alphabet.build_3di_state_from_state_pep(
        g=g,
        state_pep=state_pep,
        selected_branch_ids=np.array([root_id, tip_a_id], dtype=np.int64),
        predictor=_fake_predictor,
    )
    v_index = state_orders.tolist().index("V")
    assert root_id in aligned
    assert tip_a_id in aligned
    assert state_3di[root_id, 0, v_index] == pytest.approx(1.0)
    assert state_3di[root_id, 1, v_index] == pytest.approx(1.0)


def test_build_tip_aa_and_3di_alignment_from_full_cds(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    aln_path = tmp_path / "full_cds.fa"
    aln_path.write_text(">A\nAAA---AAG\n>B\nGCT---NNN\n", encoding="utf-8")
    g = {
        "tree": tr,
        "full_cds_alignment_file": str(aln_path),
        "codon_table": [
            ("K", "AAA"),
            ("K", "AAG"),
            ("A", "GCT"),
        ],
    }
    aa = structural_alphabet.build_tip_aa_alignment_from_full_cds(g=g)
    assert aa["A"] == "K-K"
    assert aa["B"] == "A-X"

    def _fake_predictor(aa_sequences, _g):
        return {name: ("V" * len(seq)) for name, seq in aa_sequences.items()}

    out_path = tmp_path / "tip_3di.fa"
    threed = structural_alphabet.build_tip_3di_alignment_from_full_cds(
        g=g,
        predictor=_fake_predictor,
        output_path=str(out_path),
    )
    assert threed["A"] == "V-V"
    assert threed["B"] == "V-V"
    text = out_path.read_text(encoding="utf-8")
    assert ">A" in text
    assert "V-V" in text


def test_encode_tip_3di_alignment_for_morph_maps_states_and_gaps():
    tip_3di = {
        "A": "ACD-",
        "B": "MNPY",
    }
    encoded = structural_alphabet._encode_tip_3di_alignment_for_morph(
        tip_3di_by_name=tip_3di,
        output_path=None,
    )
    assert encoded["A"] == "012-"
    assert encoded["B"] == "ABCJ"


def test_normalize_direct_iqtree_model_maps_gtr20_to_gtr():
    model, remapped = structural_alphabet._normalize_direct_iqtree_model("GTR20")
    assert model == "GTR"
    assert remapped is True
    model_ok, remapped_ok = structural_alphabet._normalize_direct_iqtree_model("MK")
    assert model_ok == "MK"
    assert remapped_ok is False


def test_read_direct_3di_state_tensor_accepts_morph_state_columns(tmp_path):
    rooted_tree = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    treefile_path = tmp_path / "direct.treefile"
    tree.write_tree(rooted_tree, outfile=str(treefile_path), add_numerical_label=False)
    direct_tree = tree.standardize_node_names(ete.PhyloNode(treefile_path.read_text(), format=1))
    root_name = [n.name for n in direct_tree.traverse() if ete.is_root(n)][0]
    state_path = tmp_path / "direct.state"
    morph_orders = list("0123456789ABCDEFGHIJ")
    row1 = np.zeros(20, dtype=float)
    row2 = np.zeros(20, dtype=float)
    row1[:2] = [0.10, 0.90]
    row2[:2] = [0.80, 0.20]
    state_path.write_text(
        "\n".join(
            [
                "Node\tSite\tState\t" + "\t".join("p_" + value for value in morph_orders),
                "{}\t1\t1\t{}".format(root_name, "\t".join(str(value) for value in row1)),
                "{}\t2\t0\t{}".format(root_name, "\t".join(str(value) for value in row2)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    g = {
        "rooted_tree": rooted_tree,
        "float_type": np.float64,
        "ml_anc": False,
    }
    tip_3di_by_name = {"A": "AC", "B": "CA"}
    paths = {
        "treefile": str(treefile_path),
        "state": str(state_path),
        "state_symbol_mode": "morph",
    }
    state_tensor, state_orders = structural_alphabet._read_direct_3di_state_tensor(
        g=g,
        paths=paths,
        tip_3di_by_name=tip_3di_by_name,
        selected_branch_ids=None,
    )
    node_by_name = {node.name: int(ete.get_prop(node, "numerical_label")) for node in rooted_tree.traverse()}
    root_id = node_by_name["R"]
    tip_a_id = node_by_name["A"]
    idx_a = state_orders.tolist().index("A")
    idx_c = state_orders.tolist().index("C")
    assert state_tensor[root_id, 0, idx_a] == pytest.approx(0.10)
    assert state_tensor[root_id, 0, idx_c] == pytest.approx(0.90)
    assert state_tensor[root_id, 1, idx_a] == pytest.approx(0.80)
    assert state_tensor[root_id, 1, idx_c] == pytest.approx(0.20)
    assert state_tensor[tip_a_id, 0, idx_a] == pytest.approx(1.0)
    assert state_tensor[tip_a_id, 1, idx_c] == pytest.approx(1.0)


def test_run_iqtree_direct_3di_uses_morph_and_remaps_gtr20(tmp_path, monkeypatch):
    rooted_tree = ete.PhyloNode("(A:1,B:1)R;", format=1)
    tip_alignment = tmp_path / "csubst_alignment_3di_tip_morph.fa"
    tip_alignment.write_text(">A\n0\n>B\n1\n", encoding="utf-8")
    prefix = str(tip_alignment.resolve())
    captured = {"command": None}

    def _fake_run(command, cwd=None):
        captured["command"] = list(command)
        for ext in ["treefile", "state", "iqtree", "log"]:
            (tmp_path / ("csubst_alignment_3di_tip_morph.fa.{}".format(ext))).write_text(
                "stub\n", encoding="utf-8"
            )
        return 0

    monkeypatch.setattr(structural_alphabet.runtime, "run_subprocess_tee", _fake_run)
    g = {
        "rooted_tree": rooted_tree,
        "iqtree_exe": "iqtree",
        "threads": 2,
        "iqtree_redo": True,
        "sa_iqtree_model": "GTR20",
    }
    paths = structural_alphabet._run_iqtree_direct_3di(g=g, tip_alignment_path=prefix)
    command = captured["command"]
    assert command is not None
    seqtype_index = command.index("--seqtype")
    model_index = command.index("-m")
    assert command[seqtype_index + 1] == "MORPH"
    assert command[model_index + 1] == "GTR"
    assert paths["state_symbol_mode"] == "morph"


def test_build_3di_state_direct_prefilters_tip_invariant_sites_when_enabled(monkeypatch):
    tip_full = OrderedDict([("A", "AAC"), ("B", "ABC"), ("C", "A-C")])
    captured = {"encoded_tip": None}
    num_node = 3
    num_state = 20
    reduced_tensor = np.zeros((num_node, 1, num_state), dtype=float)
    reduced_tensor[:, :, 0] = 0.25

    monkeypatch.setattr(
        structural_alphabet,
        "build_tip_3di_alignment_from_full_cds",
        lambda g, predictor=None, output_path=None: tip_full,
    )

    def _fake_encode(tip_3di_by_name, output_path=None):
        captured["encoded_tip"] = OrderedDict(tip_3di_by_name)
        return tip_3di_by_name

    monkeypatch.setattr(structural_alphabet, "_encode_tip_3di_alignment_for_morph", _fake_encode)
    monkeypatch.setattr(
        structural_alphabet,
        "_run_iqtree_direct_3di",
        lambda g, tip_alignment_path: {"treefile": "x.tree", "state": "x.state", "state_symbol_mode": "morph"},
    )
    monkeypatch.setattr(
        structural_alphabet,
        "_read_direct_3di_state_tensor",
        lambda g, paths, tip_3di_by_name, selected_branch_ids=None: (
            reduced_tensor.copy(),
            np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object),
        ),
    )

    g = {
        "drop_invariant_tip_sites": True,
        "drop_invariant_tip_sites_mode": "tip_invariant",
    }
    state_tensor, state_orders, tip_out = structural_alphabet.build_3di_state_direct(g=g)
    assert tip_out == tip_full
    assert captured["encoded_tip"] == OrderedDict([("A", "A"), ("B", "B"), ("C", "-")])
    assert "_precomputed_tip_invariant_site_mask" in g
    np.testing.assert_array_equal(g["_precomputed_tip_invariant_site_mask"], np.array([True, False, True]))
    assert state_tensor.shape == (num_node, 3, num_state)
    np.testing.assert_allclose(state_tensor[:, 0, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(state_tensor[:, 1, 0], 0.25, atol=1e-12)
    np.testing.assert_allclose(state_tensor[:, 2, :], 0.0, atol=1e-12)
    assert state_orders.tolist() == list("ACDEFGHIKLMNPQRSTVWY")


def test_build_3di_state_direct_does_not_prefilter_when_mode_is_zero_sub_mass(monkeypatch):
    tip_full = OrderedDict([("A", "AAA"), ("B", "AAA")])
    captured = {"encoded_tip": None}
    monkeypatch.setattr(
        structural_alphabet,
        "build_tip_3di_alignment_from_full_cds",
        lambda g, predictor=None, output_path=None: tip_full,
    )

    def _fake_encode(tip_3di_by_name, output_path=None):
        captured["encoded_tip"] = OrderedDict(tip_3di_by_name)
        return tip_3di_by_name

    monkeypatch.setattr(structural_alphabet, "_encode_tip_3di_alignment_for_morph", _fake_encode)
    monkeypatch.setattr(
        structural_alphabet,
        "_run_iqtree_direct_3di",
        lambda g, tip_alignment_path: {"treefile": "x.tree", "state": "x.state", "state_symbol_mode": "morph"},
    )
    monkeypatch.setattr(
        structural_alphabet,
        "_read_direct_3di_state_tensor",
        lambda g, paths, tip_3di_by_name, selected_branch_ids=None: (
            np.ones((3, 3, 20), dtype=float),
            np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object),
        ),
    )
    g = {
        "drop_invariant_tip_sites": True,
        "drop_invariant_tip_sites_mode": "zero_sub_mass",
    }
    structural_alphabet.build_3di_state_direct(g=g)
    assert captured["encoded_tip"] == tip_full
    assert "_precomputed_tip_invariant_site_mask" not in g

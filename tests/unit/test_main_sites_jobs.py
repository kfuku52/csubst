import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from csubst import main_sites
from csubst import runtime
from csubst import tree
from csubst import ete


def test_translate_and_write_fasta(tmp_path):
    g = {"matrix_groups": {"K": ["AAA"], "N": ["AAC"]}}
    assert main_sites.translate("AAAAAC", g) == "KN"
    out = tmp_path / "toy.fa"
    main_sites.write_fasta(str(out), "sample", "KN")
    assert out.read_text(encoding="utf-8") == ">sample\nKN\n"


def test_translate_rejects_non_triplet_sequence_length():
    g = {"matrix_groups": {"K": ["AAA"]}}
    with pytest.raises(ValueError, match="multiple of 3"):
        main_sites.translate("AAAAA", g)


def test_translate_rejects_unknown_codon():
    g = {"matrix_groups": {"K": ["AAA"]}}
    with pytest.raises(ValueError, match='Unknown codon "AAT"'):
        main_sites.translate("AAT", g)


def test_resolve_chimera_line_for_site_handles_missing_any2dif_column():
    df = pd.DataFrame({"codon_site_seq1": [1], "OCNany2spe": [0.25]})
    out = main_sites._resolve_chimera_line_for_site(df=df, codon_site_col="codon_site_seq1", seq_site=1)
    assert out == "\t:1\t0.2500\n"


def test_resolve_chimera_line_for_site_handles_missing_any2spe_column():
    df = pd.DataFrame({"codon_site_seq1": [1], "OCNany2dif": [0.4]})
    out = main_sites._resolve_chimera_line_for_site(df=df, codon_site_col="codon_site_seq1", seq_site=1)
    assert out == "\t:1\t-0.4000\n"


def test_export2chimera_rejects_non_triplet_sequence_before_writing(tmp_path):
    cds = tmp_path / "untrimmed.fa"
    cds.write_text(">seq1\nAAAA\n", encoding="utf-8")
    df = pd.DataFrame(
        {
            "codon_site_seq1": [1],
            "OCNany2spe": [0.3],
            "OCNany2dif": [0.1],
        }
    )
    g = {
        "untrimmed_cds": str(cds),
        "site_outdir": str(tmp_path),
        "matrix_groups": {"K": ["AAA"]},
    }
    with pytest.raises(ValueError, match='multiple of 3 for Chimera export'):
        main_sites.export2chimera(df=df, g=g)
    assert not (tmp_path / "csubst_sites_seq1.chimera.txt").exists()
    assert not (tmp_path / "csubst_sites_seq1.fasta").exists()


def test_get_parent_branch_ids(tiny_tree):
    bids = []
    for node in tiny_tree.traverse():
        if node.name in {"A", "C"}:
            bids.append(ete.get_prop(node, "numerical_label"))
    out = main_sites.get_parent_branch_ids(np.array(bids), {"tree": tiny_tree})
    assert len(out) == 2
    # Both A and C have internal node X as parent.
    x_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "X"][0]
    assert set(out.values()) == {x_id}


def test_build_aln_gene_match_for_leaf_rejects_non_triplet_untrimmed_cds():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    leaf_a = [n for n in tr.traverse() if n.name == "A"][0]
    num_node = len(list(tr.traverse()))
    g = {
        "state_cdn": np.zeros((num_node, 1, 1), dtype=float),
        "codon_orders": np.array(["AAA"]),
    }
    with pytest.raises(ValueError, match='length for "A" should be multiple of 3'):
        main_sites._build_aln_gene_match_for_leaf(leaf=leaf_a, seq="AAAA", num_site=1, g=g)


def test_add_states_handles_root_branch_without_parent_index_error(tiny_tree):
    root_id = int(ete.get_prop(tiny_tree, "numerical_label"))
    num_node = max(int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()) + 1
    num_site = 5
    state_cdn = np.zeros((num_node, num_site, 2), dtype=float)
    state_pep = np.zeros((num_node, num_site, 2), dtype=float)
    state_cdn[root_id, :, 0] = 1.0
    state_pep[root_id, :, 0] = 1.0
    df = pd.DataFrame({"site": np.arange(num_site, dtype=int)})
    g = {
        "tree": tiny_tree,
        "state_cdn": state_cdn,
        "state_pep": state_pep,
        "codon_orders": np.array(["AAA", "AAG"]),
        "amino_acid_orders": np.array(["K", "N"]),
    }
    out = main_sites.add_states(df=df, branch_ids=np.array([root_id], dtype=np.int64), g=g, add_hydrophobicity=False)
    assert (out["cdn_{}_anc".format(root_id)] == "").all()
    assert (out["aa_{}_anc".format(root_id)] == "").all()


def test_resolve_site_jobs_intersection_mode_preserves_branch_set_and_outdir_prefix(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "intersection", "branch_id": "{},{}".format(labels["A"], labels["C"])}
    out = main_sites.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    np.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], [labels["A"], labels["C"]])
    assert out["site_jobs"][0]["site_outdir"].startswith("./csubst_sites.branch_id")


def test_resolve_site_jobs_honors_output_namespace(tmp_path, tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {
        "tree": tiny_tree,
        "mode": "intersection",
        "branch_id": "{},{}".format(labels["A"], labels["C"]),
        "outdir": str(tmp_path / "sites"),
        "output_prefix": "result",
    }
    out = main_sites.resolve_site_jobs(g)
    expected = tmp_path / "sites" / "result.branch_id{},{}".format(labels["A"], labels["C"])
    assert out["site_jobs"][0]["site_outdir"] == str(expected)


def test_resolve_site_jobs_rejects_duplicate_branch_ids(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "intersection", "branch_id": "{},{}".format(labels["A"], labels["A"])}
    with pytest.raises(ValueError, match="duplicate IDs"):
        main_sites.resolve_site_jobs(g)


def test_maybe_relocate_site_log_file_keeps_open_log_at_original_path(tmp_path, monkeypatch):
    default_log = Path(runtime.default_site_log_path(base_dir=tmp_path, create_dir=True))
    default_log.write_text("hello\n", encoding="utf-8")
    site_outdir = tmp_path / "csubst_sites.branch_id1,2"
    g = {
        "log_file": str(default_log),
        "site_jobs": [{"site_outdir": str(site_outdir)}],
    }

    monkeypatch.chdir(tmp_path)
    out = main_sites._maybe_relocate_site_log_file(g)

    relocated_log = site_outdir / "csubst.log"
    assert out["log_file"] == str(default_log.resolve())
    assert not relocated_log.exists()
    assert default_log.read_text(encoding="utf-8") == "hello\n"


def test_normalize_branch_ids_rejects_non_integer_like_values():
    with pytest.raises(ValueError, match="integer-like"):
        main_sites._normalize_branch_ids(np.array([1.5]))
    with pytest.raises(ValueError, match="integer-like"):
        main_sites._normalize_branch_ids(np.array([True]))


def test_resolve_site_jobs_lineage_mode_returns_ancestor_to_descendant_path(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "lineage", "branch_id": "{},{}".format(labels["X"], labels["C"])}
    out = main_sites.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    np.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], [labels["X"], labels["C"]])
    assert out["site_jobs"][0]["site_outdir"] == "./csubst_sites.lineage.branch_id{},{}".format(labels["X"], labels["C"])
    assert not out["site_jobs"][0]["single_branch_mode"]


def test_resolve_site_jobs_lineage_mode_rejects_non_ancestor_pairs(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "lineage", "branch_id": "{},{}".format(labels["B"], labels["C"])}
    with pytest.raises(ValueError, match="ancestor"):
        main_sites.resolve_site_jobs(g)


def test_resolve_site_jobs_accepts_vesm_for_intersection_and_lineage(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    intersection = main_sites.resolve_site_jobs(
        {
            "tree": tiny_tree,
            "mode": "intersection",
            "branch_id": str(labels["A"]),
            "vep_model": "vesm-35m",
            "nonsyn_recode": "no",
        }
    )
    assert intersection["site_jobs"][0]["branch_ids"].tolist() == [labels["A"]]
    lineage = main_sites.resolve_site_jobs(
        {
            "tree": tiny_tree,
            "mode": "lineage",
            "branch_id": "{},{}".format(labels["X"], labels["C"]),
            "vep_model": "vesm-35m",
            "nonsyn_recode": "no",
        }
    )
    assert lineage["site_jobs"][0]["branch_ids"].tolist() == [labels["X"], labels["C"]]


def test_resolve_site_jobs_rejects_vesm_set_mode_and_recoding(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    with pytest.raises(ValueError, match="set mode is not yet supported"):
        main_sites.resolve_site_jobs(
            {
                "tree": tiny_tree,
                "mode": "set,any,{}|{}".format(labels["A"], labels["C"]),
                "branch_id": "unused",
                "vep_model": "vesm-35m",
                "nonsyn_recode": "no",
            }
        )
    with pytest.raises(ValueError, match="requires --nonsyn_recode no"):
        main_sites.resolve_site_jobs(
            {
                "tree": tiny_tree,
                "mode": "intersection",
                "branch_id": str(labels["A"]),
                "vep_model": "vesm-35m",
                "nonsyn_recode": "charge",
            }
        )


@pytest.mark.parametrize("mode_name", ["total", "each", "all", "clade"])
def test_resolve_site_jobs_rejects_removed_modes(tiny_tree, mode_name):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {
        "tree": tiny_tree,
        "mode": mode_name,
        "branch_id": "{},{}".format(labels["A"], labels["C"]),
    }
    with pytest.raises(ValueError, match="intersection,lineage,set"):
        main_sites.resolve_site_jobs(g)


def test_resolve_site_jobs_set_mode_extracts_expression_branch_ids(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    root_id = int(ete.get_prop(tiny_tree, "numerical_label"))
    g = {"tree": tiny_tree, "mode": "set,any,({}|{})-{}".format(labels["A"], labels["C"], root_id), "branch_id": "unused"}
    out = main_sites.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    np.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], sorted([labels["A"], labels["C"]]))
    assert out["mode"] == "set"
    assert out["set_stat_type"] == "any"
    assert out["mode_expression"] == "({}|{})-{}".format(labels["A"], labels["C"], root_id)
    assert out["site_jobs"][0]["site_outdir"].startswith("./csubst_sites.set.any.expr")
    assert "_or_" in out["site_jobs"][0]["site_outdir"]
    assert "_minus_" in out["site_jobs"][0]["site_outdir"]


def test_resolve_site_jobs_set_mode_with_all_other_symbol_in_label(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "set,any,({}|{})-A".format(labels["A"], labels["C"]), "branch_id": "unused"}
    out = main_sites.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    np.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], sorted([labels["A"], labels["C"]]))
    assert out["site_jobs"][0]["site_outdir"].startswith("./csubst_sites.set.any.expr")
    assert "_all_other" in out["site_jobs"][0]["site_outdir"]


def test_resolve_site_jobs_set_mode_without_branch_id(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "set,any,{}|{}".format(labels["A"], labels["B"])}
    out = main_sites.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    np.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], sorted([labels["A"], labels["B"]]))


def test_resolve_site_jobs_set_mode_rejects_unknown_branch_ids(tiny_tree):
    g = {"tree": tiny_tree, "mode": "set,any,999|1", "branch_id": "unused"}
    with pytest.raises(ValueError, match="unknown branch IDs"):
        main_sites.resolve_site_jobs(g)


def test_resolve_site_jobs_set_mode_rejects_invalid_expression_syntax(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "set,any,({}|{}".format(labels["A"], labels["B"])}
    with pytest.raises(ValueError, match="Unbalanced parentheses"):
        main_sites.resolve_site_jobs(g)


def test_resolve_site_jobs_set_mode_rejects_missing_stat_type(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "set,{}|{}".format(labels["A"], labels["B"])}
    with pytest.raises(ValueError, match="set,<substitution_type>,<expression>"):
        main_sites.resolve_site_jobs(g)


def test_resolve_site_jobs_set_mode_rejects_invalid_stat_type(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "set,unknown,{}|{}".format(labels["A"], labels["B"])}
    with pytest.raises(ValueError, match="any,spe"):
        main_sites.resolve_site_jobs(g)


@pytest.mark.parametrize("legacy_stat_type", ["any2any", "any2spe", "spe2any", "spe2spe"])
def test_resolve_site_jobs_set_mode_rejects_legacy_stat_types(tiny_tree, legacy_stat_type):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    g = {"tree": tiny_tree, "mode": "set,{},{}|{}".format(legacy_stat_type, labels["A"], labels["B"])}
    with pytest.raises(ValueError, match="no longer supported|was removed"):
        main_sites.resolve_site_jobs(g)

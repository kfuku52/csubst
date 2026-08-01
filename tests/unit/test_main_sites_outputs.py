import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from csubst import main_sites
from csubst import substitution_sparse
from csubst import ete


def test_plot_state_writes_outputs_when_enabled(tmp_path, tiny_tree):
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()}
    num_node = max(labels.values()) + 1
    on = np.zeros((num_node, 5, 1, 2, 2), dtype=float)
    os = np.zeros((num_node, 5, 1, 2, 2), dtype=float)
    for site in range(5):
        on[labels["A"], site, 0, 0, 1] = 0.8 - 0.1 * site
        on[labels["C"], site, 0, 0, 1] = 0.7 - 0.1 * site
        os[labels["A"], site, 0, 0, 1] = 0.6 - 0.05 * site
        os[labels["C"], site, 0, 0, 1] = 0.9 - 0.05 * site
    g = {
        "tree": tiny_tree,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "amino_acid_orders": np.array(["A", "B"]),
        "matrix_groups": {"grp": ["AA", "AB"]},
        "site_state_plot": True,
    }
    out_paths = main_sites.plot_state(on, os, np.array([labels["A"], labels["C"]], dtype=int), g)
    expected = {
        str(tmp_path / "csubst_sites.state.pdf"),
        str(tmp_path / "csubst_sites.state_N.tsv"),
        str(tmp_path / "csubst_sites.state_S.tsv"),
    }
    assert set(out_paths) == expected
    for path in expected:
        assert Path(path).exists()


def test_get_df_ad_uses_nonsyn_state_orders_for_recoded_tensor():
    g = {
        "amino_acid_orders": np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object),
        "nonsyn_state_orders": np.array(["AGPST", "DENQ", "HKR", "ILMV", "FWY", "C"], dtype=object),
        "matrix_groups": {"grp": ["AAA", "AAT"]},
        "min_combinat_prob": 0.5,
    }
    sub_tensor = np.zeros((2, 2, 1, 6, 6), dtype=float)
    sub_tensor[0, 0, 0, 0, 5] = 0.75
    sub_tensor[1, 1, 0, 5, 0] = 0.25

    df_ad = main_sites.get_df_ad(sub_tensor=sub_tensor, g=g, mode="nsy")
    df_ad = main_sites.add_site_stats(df_ad, sub_tensor, g, "nsy", method="rank1")
    df_ad = main_sites.add_has_target_high_combinat_prob_site(df_ad, sub_tensor, g, "nsy")

    row = df_ad.loc[(df_ad["state_from"] == "AGPST") & (df_ad["state_to"] == "C"), :]
    assert row.shape[0] == 1
    assert float(row["value"].iloc[0]) == pytest.approx(0.75, abs=1e-12)
    assert bool(row["has_target_high_combinat_prob_site"].iloc[0]) is True
    assert "A" not in set(df_ad["state_from"])
    assert "DENQ" in set(df_ad["state_from"])


def test_plot_state_accepts_recoded_nonsyn_state_axis(tmp_path, tiny_tree):
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()}
    num_node = max(labels.values()) + 1
    on = np.zeros((num_node, 3, 1, 6, 6), dtype=float)
    os = np.zeros((num_node, 3, 1, 2, 2), dtype=float)
    for site in range(3):
        on[labels["A"], site, 0, 0, 5] = 0.8 - 0.1 * site
        on[labels["C"], site, 0, 5, 0] = 0.7 - 0.1 * site
        os[labels["A"], site, 0, 0, 1] = 0.6 - 0.05 * site
        os[labels["C"], site, 0, 1, 0] = 0.9 - 0.05 * site
    g = {
        "tree": tiny_tree,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "amino_acid_orders": np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object),
        "nonsyn_state_orders": np.array(["AGPST", "DENQ", "HKR", "ILMV", "FWY", "C"], dtype=object),
        "matrix_groups": {"grp": ["AAA", "AAT"]},
        "site_state_plot": True,
    }

    out_paths = main_sites.plot_state(on, os, np.array([labels["A"], labels["C"]], dtype=int), g)

    assert str(tmp_path / "csubst_sites.state_N.tsv") in out_paths
    state_n = pd.read_csv(tmp_path / "csubst_sites.state_N.tsv", sep="\t")
    assert "AGPST" in set(state_n["state_from"])
    assert "C" in set(state_n["state_to"])


def test_plot_state_accepts_sparse_substitution_tensors(tmp_path, tiny_tree):
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()}
    num_node = max(labels.values()) + 1
    on = np.zeros((num_node, 5, 1, 2, 2), dtype=float)
    os = np.zeros((num_node, 5, 1, 2, 2), dtype=float)
    for site in range(5):
        on[labels["A"], site, 0, 0, 1] = 0.8 - 0.1 * site
        on[labels["C"], site, 0, 0, 1] = 0.7 - 0.1 * site
        os[labels["A"], site, 0, 0, 1] = 0.6 - 0.05 * site
        os[labels["C"], site, 0, 0, 1] = 0.9 - 0.05 * site
    g = {
        "tree": tiny_tree,
        "site_outdir": str(tmp_path),
        "float_format": "%.4f",
        "amino_acid_orders": np.array(["A", "B"]),
        "matrix_groups": {"grp": ["AA", "AB"]},
        "site_state_plot": True,
    }
    sparse_on = substitution_sparse.SparseSubstitutionTensor.from_dense(on)
    sparse_os = substitution_sparse.SparseSubstitutionTensor.from_dense(os)

    out_paths = main_sites.plot_state(sparse_on, sparse_os, np.array([labels["A"], labels["C"]], dtype=int), g)

    assert set(out_paths) == {
        str(tmp_path / "csubst_sites.state.pdf"),
        str(tmp_path / "csubst_sites.state_N.tsv"),
        str(tmp_path / "csubst_sites.state_S.tsv"),
    }
    for path in out_paths:
        assert Path(path).exists()


def test_plot_state_skips_outputs_when_disabled(tmp_path):
    g = {"site_state_plot": False, "site_outdir": str(tmp_path)}
    out_paths = main_sites.plot_state(
        ON_tensor=np.zeros((1, 1, 1, 1, 1), dtype=float),
        OS_tensor=np.zeros((1, 1, 1, 1, 1), dtype=float),
        branch_ids=np.array([0], dtype=int),
        g=g,
    )
    assert out_paths == []
    assert list(tmp_path.glob("csubst_sites.state*")) == []


def test_write_site_output_manifest_records_files_and_parameters(tmp_path):
    existing = tmp_path / "csubst_sites.tsv"
    existing.write_text("x\n", encoding="utf-8")
    g = {
        "site_outdir": str(tmp_path),
        "single_branch_mode": False,
        "tree_site_plot": True,
        "site_state_plot": False,
        "tree_site_plot_format": "pdf",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 60,
        "pdb": None,
    }
    rows = list()
    branch_ids = np.array([23, 51], dtype=int)
    main_sites.add_site_output_manifest_row(rows, str(existing), "site_table_tsv", g, branch_ids)
    main_sites.add_site_output_manifest_row(
        rows, str(tmp_path / "csubst_sites.state.pdf"), "state_pattern_pdf", g, branch_ids, note="skipped"
    )
    manifest_path = main_sites.write_site_output_manifest(rows, g, branch_ids)
    assert Path(manifest_path).exists()
    out_df = pd.read_csv(manifest_path, sep="\t")
    assert set(["output_kind", "output_file", "file_exists", "site_state_plot"]).issubset(set(out_df.columns))
    assert (out_df.loc[:, "output_kind"] == "output_manifest").any()
    manifest_row = out_df.loc[out_df.loc[:, "output_kind"] == "output_manifest", :].iloc[0]
    assert manifest_row["file_exists"] == "Y"
    assert int(manifest_row["file_size_bytes"]) == Path(manifest_path).stat().st_size
    site_row = out_df.loc[out_df.loc[:, "output_kind"] == "site_table_tsv", :].iloc[0]
    assert site_row["file_exists"] == "Y"
    assert int(site_row["file_size_bytes"]) > 0
    skipped_row = out_df.loc[out_df.loc[:, "output_kind"] == "state_pattern_pdf", :].iloc[0]
    assert skipped_row["file_exists"] == "N"
    assert skipped_row["note"] == "skipped"


def test_write_site_output_manifest_accepts_scalar_branch_id(tmp_path):
    existing = tmp_path / "csubst_sites.tsv"
    existing.write_text("x\n", encoding="utf-8")
    g = {
        "site_outdir": str(tmp_path),
        "single_branch_mode": True,
        "tree_site_plot": True,
        "site_state_plot": False,
        "tree_site_plot_format": "pdf",
        "min_combinat_prob": 0.5,
        "min_single_prob": 0.8,
        "tree_site_plot_max_sites": 60,
        "pdb": None,
    }
    rows = list()
    main_sites.add_site_output_manifest_row(rows, str(existing), "site_table_tsv", g, np.int64(23))
    manifest_path = main_sites.write_site_output_manifest(rows, g, np.int64(23))
    out_df = pd.read_csv(manifest_path, sep="\t")
    site_row = out_df.loc[out_df.loc[:, "output_kind"] == "site_table_tsv", :].iloc[0]
    assert str(site_row["branch_ids"]) == "23"
    assert int(site_row["branch_count"]) == 1


def test_get_df_ad_add_site_stats_and_target_flag():
    g = {"amino_acid_orders": np.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    sub_tensor = np.zeros((2, 2, 1, 2, 2), dtype=float)
    # A->B occurs at site 0 and site 1 with totals [2,1].
    sub_tensor[0, 0, 0, 0, 1] = 2.0
    sub_tensor[0, 1, 0, 0, 1] = 1.0
    # B->A occurs with totals [3,1].
    sub_tensor[1, 0, 0, 1, 0] = 3.0
    sub_tensor[1, 1, 0, 1, 0] = 1.0

    df_ad = main_sites.get_df_ad(sub_tensor=sub_tensor, g=g, mode="nsy")
    assert df_ad[["group", "state_from", "state_to"]].to_records(index=False).tolist() == [
        ("nsy", "A", "B"),
        ("nsy", "B", "A"),
    ]
    np.testing.assert_allclose(df_ad["value"].to_numpy(), [3.0, 4.0], atol=1e-12)

    df_ad = main_sites.add_site_stats(df_ad, sub_tensor, g, "nsy", method="tau")
    df_ad = main_sites.add_site_stats(df_ad, sub_tensor, g, "nsy", method="tsi")
    df_ad = main_sites.add_site_stats(df_ad, sub_tensor, g, "nsy", method="rank1")
    df_ad = main_sites.add_site_stats(df_ad, sub_tensor, g, "nsy", method="rank2")
    np.testing.assert_allclose(df_ad["site_tau"].to_numpy(), [0.5, 2.0 / 3.0], atol=1e-12)
    np.testing.assert_allclose(df_ad["site_tsi"].to_numpy(), [2.0 / 3.0, 3.0 / 4.0], atol=1e-12)
    np.testing.assert_allclose(df_ad["site_rank1"].to_numpy(), [2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(df_ad["site_rank2"].to_numpy(), [1.0, 1.0], atol=1e-12)

    flagged = main_sites.add_has_target_high_combinat_prob_site(df_ad, sub_tensor, g, "nsy")
    assert flagged["has_target_high_combinat_prob_site"].tolist() == [True, True]

    scaled_sub_tensor = sub_tensor * 0.1
    scaled_df_ad = main_sites.get_df_ad(sub_tensor=scaled_sub_tensor, g=g, mode="nsy")
    g_threshold = dict(g)
    g_threshold["min_combinat_prob"] = 0.25
    scaled_flagged = main_sites.add_has_target_high_combinat_prob_site(
        scaled_df_ad,
        scaled_sub_tensor,
        g_threshold,
        "nsy",
    )
    assert scaled_flagged["has_target_high_combinat_prob_site"].tolist() == [False, True]


def test_site_summary_stats_support_sparse_substitution_tensors(monkeypatch):
    g = {"amino_acid_orders": np.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    dense_tensor = np.zeros((2, 2, 1, 2, 2), dtype=float)
    dense_tensor[0, 0, 0, 0, 1] = 2.0
    dense_tensor[0, 1, 0, 0, 1] = 1.0
    dense_tensor[1, 0, 0, 1, 0] = 3.0
    dense_tensor[1, 1, 0, 1, 0] = 1.0
    sparse_tensor = substitution_sparse.SparseSubstitutionTensor.from_dense(dense_tensor)

    monkeypatch.setattr(
        substitution_sparse.SparseSubstitutionTensor,
        "sum",
        lambda self, axis=None: (_ for _ in ()).throw(AssertionError("dense fallback should not run")),
    )

    df_ad = main_sites.get_df_ad(sub_tensor=sparse_tensor, g=g, mode="nsy")
    df_ad = main_sites.add_site_stats(df_ad, sparse_tensor, g, "nsy", method="tau")
    df_ad = main_sites.add_site_stats(df_ad, sparse_tensor, g, "nsy", method="tsi")
    flagged = main_sites.add_has_target_high_combinat_prob_site(df_ad, sparse_tensor, g, "nsy")

    np.testing.assert_allclose(flagged["value"].to_numpy(), [3.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(flagged["site_tau"].to_numpy(), [0.5, 2.0 / 3.0], atol=1e-12)
    np.testing.assert_allclose(flagged["site_tsi"].to_numpy(), [2.0 / 3.0, 3.0 / 4.0], atol=1e-12)
    assert flagged["has_target_high_combinat_prob_site"].tolist() == [True, True]


def test_add_branch_sub_prob_accepts_sparse_substitution_tensors():
    dense = np.zeros((3, 3, 1, 2, 2), dtype=float)
    dense[1, 0, 0, 0, 1] = 0.2
    dense[1, 0, 0, 1, 0] = 0.3
    dense[1, 2, 0, 0, 1] = 0.4
    dense[2, 1, 0, 1, 0] = 0.8
    sparse = substitution_sparse.SparseSubstitutionTensor.from_dense(dense)
    branch_ids = np.array([1, 2], dtype=np.int64)
    df = main_sites.initialize_site_df(num_site=3)

    dense_out = main_sites.add_branch_sub_prob(df=df.copy(), branch_ids=branch_ids, sub_tensor=dense, attr="N")
    sparse_out = main_sites.add_branch_sub_prob(df=df.copy(), branch_ids=branch_ids, sub_tensor=sparse, attr="N")

    np.testing.assert_allclose(
        sparse_out[["N_sub_1", "N_sub_2"]].to_numpy(dtype=float),
        dense_out[["N_sub_1", "N_sub_2"]].to_numpy(dtype=float),
        atol=1e-12,
    )


def test_add_site_stats_hg_ignores_zero_probabilities():
    g = {"amino_acid_orders": np.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    sub_tensor = np.zeros((2, 2, 1, 2, 2), dtype=float)
    # A->B totals per site = [1, 0], so entropy should be 0, not NaN.
    sub_tensor[0, 0, 0, 0, 1] = 1.0
    df_ad = main_sites.get_df_ad(sub_tensor=sub_tensor, g=g, mode="nsy")
    out = main_sites.add_site_stats(df_ad, sub_tensor, g, "nsy", method="hg")
    assert pytest.approx(float(out["site_hg"].iloc[0]), abs=1e-12) == 0.0
    assert pd.isna(out["site_hg"].iloc[1])


def test_add_site_stats_tau_single_site_returns_zero_not_nan():
    g = {"amino_acid_orders": np.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    sub_tensor = np.zeros((2, 1, 1, 2, 2), dtype=float)
    # Single-site profile; tau denominator would otherwise be 0.
    sub_tensor[0, 0, 0, 0, 1] = 1.0
    sub_tensor[1, 0, 0, 0, 1] = 1.0
    df_ad = main_sites.get_df_ad(sub_tensor=sub_tensor, g=g, mode="nsy")
    out = main_sites.add_site_stats(df_ad, sub_tensor, g, "nsy", method="tau")
    assert pytest.approx(float(out["site_tau"].iloc[0]), abs=1e-12) == 0.0
    assert pd.isna(out["site_tau"].iloc[1])


def test_add_site_stats_rank_overflow_returns_zero_not_error():
    g = {"amino_acid_orders": np.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    sub_tensor = np.zeros((2, 2, 1, 2, 2), dtype=float)
    sub_tensor[0, 0, 0, 0, 1] = 2.0
    sub_tensor[0, 1, 0, 0, 1] = 1.0
    df_ad = main_sites.get_df_ad(sub_tensor=sub_tensor, g=g, mode="nsy")
    out = main_sites.add_site_stats(df_ad, sub_tensor, g, "nsy", method="rank5")
    np.testing.assert_allclose(out["site_rank5"].to_numpy(), [0.0, np.nan], atol=1e-12, equal_nan=True)


def test_get_highest_identity_chain_name_handles_empty_identity_dict_without_pymol_import():
    g = {"aa_identity_means": {}}
    out = main_sites.get_highest_identity_chain_name(g)
    assert out["highest_identity_chain_name"] is None


def test_get_df_dist_reports_max_distance_for_multi_branch_substitutions(tiny_tree):
    g = {"tree": tiny_tree, "amino_acid_orders": np.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    sub_tensor = np.zeros((num_node, 1, 1, 2, 2), dtype=float)
    a_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "A"][0]
    c_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "C"][0]
    b_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "B"][0]
    sub_tensor[a_id, 0, 0, 0, 1] = 0.6
    sub_tensor[c_id, 0, 0, 0, 1] = 0.6
    sub_tensor[b_id, 0, 0, 1, 0] = 0.7
    out = main_sites.get_df_dist(sub_tensor=sub_tensor, g=g, mode="nsy")
    row_ab = out.loc[(out["state_from"] == "A") & (out["state_to"] == "B"), :]
    row_ba = out.loc[(out["state_from"] == "B") & (out["state_to"] == "A"), :]
    assert pytest.approx(float(row_ab["max_dist_bl"].iloc[0]), rel=0, abs=1e-12) == 1.0
    assert pd.isna(row_ba["max_dist_bl"].iloc[0])

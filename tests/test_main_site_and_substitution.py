import numpy
import pandas
import pytest
from pathlib import Path

from csubst import main_site
from csubst import substitution
from csubst import substitution_cy
from csubst import tree
from csubst import ete


@pytest.fixture
def tiny_tree():
    tr = ete.PhyloNode("(B:1,(A:1,C:2)X:3)R;", format=1)
    return tree.add_numerical_node_labels(tr)


def test_get_gapsite_rate_matches_manual_count():
    state_tensor = numpy.array(
        [
            [[1, 0], [0, 0], [1, 0]],
            [[0, 1], [0, 0], [0, 1]],
            [[1, 0], [1, 0], [0, 0]],
        ],
        dtype=float,
    )
    # site-wise gap rates: [0/3, 2/3, 1/3]
    out = main_site.get_gapsite_rate(state_tensor)
    numpy.testing.assert_allclose(out, [0.0, 2.0 / 3.0, 1.0 / 3.0], atol=1e-12)


def test_extend_site_index_edge_fills_missing_edges():
    sites = pandas.Series([2, 3, 7], dtype=int)
    out = main_site.extend_site_index_edge(sites, num_extend=2)
    # Gap between 3 and 7 is filled by 5 and 6.
    assert out.tolist() == [2, 3, 5, 6, 7]


def test_initialize_site_df_columns_are_correct():
    df = main_site.initialize_site_df(4)
    assert df["codon_site_alignment"].tolist() == [0, 1, 2, 3]
    assert df["nuc_site_alignment"].tolist() == [1, 4, 7, 10]


def test_combinatorial2single_columns_removes_combination_columns():
    df = pandas.DataFrame(
        {
            "OCSany2any": [1],
            "OCSany2spe": [2],
            "OCNspe2dif": [3],
            "kept": [4],
        }
    )
    out = main_site.combinatorial2single_columns(df)
    assert list(out.columns) == ["kept"]


def test_get_yvalues_for_supported_modes():
    df = pandas.DataFrame(
        {
            "S_sub": [0.0, 0.5],
            "N_sub": [1.0, 2.0],
            "S_sub_1": [0.0, 0.3],
            "S_sub_2": [0.0, 0.4],
            "N_sub_1": [0.7, 0.1],
            "N_sub_2": [0.3, 0.2],
            "OCNany2spe": [0.2, 0.4],
            "OCSany2spe": [0.1, 0.2],
        }
    )
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_sub", "S"), [0.0, 2.5], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_sub", "N"), [1.0, 2.0], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "_sub_", "S"), [0.0, 1.0], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "any2spe", "S"), [0.3, 0.6], atol=1e-12)
    numpy.testing.assert_allclose(main_site.get_yvalues(df, "any2spe", "N"), [0.2, 0.4], atol=1e-12)


def test_translate_and_write_fasta(tmp_path):
    g = {"matrix_groups": {"K": ["AAA"], "N": ["AAC"]}}
    assert main_site.translate("AAAAAC", g) == "KN"
    out = tmp_path / "toy.fa"
    main_site.write_fasta(str(out), "sample", "KN")
    assert out.read_text(encoding="utf-8") == ">sample\nKN\n"


def test_get_parent_branch_ids(tiny_tree):
    bids = []
    for node in tiny_tree.traverse():
        if node.name in {"A", "C"}:
            bids.append(ete.get_prop(node, "numerical_label"))
    out = main_site.get_parent_branch_ids(numpy.array(bids), {"tree": tiny_tree})
    assert len(out) == 2
    # Both A and C have internal node X as parent.
    x_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "X"][0]
    assert set(out.values()) == {x_id}


def test_get_state_orders():
    g = {"amino_acid_orders": numpy.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    orders_nsy, keys_nsy = main_site.get_state_orders(g, "nsy")
    assert keys_nsy == ["nsy"]
    assert list(orders_nsy["nsy"]) == ["A", "B"]
    orders_syn, keys_syn = main_site.get_state_orders(g, "syn")
    assert keys_syn == ["grp"]
    assert orders_syn["grp"] == ["AA", "AB"]


def test_get_df_ad_add_site_stats_and_target_flag():
    g = {"amino_acid_orders": numpy.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    sub_tensor = numpy.zeros((2, 2, 1, 2, 2), dtype=float)
    # A->B occurs at site 0 and site 1 with totals [2,1].
    sub_tensor[0, 0, 0, 0, 1] = 2.0
    sub_tensor[0, 1, 0, 0, 1] = 1.0
    # B->A occurs with totals [3,1].
    sub_tensor[1, 0, 0, 1, 0] = 3.0
    sub_tensor[1, 1, 0, 1, 0] = 1.0

    df_ad = main_site.get_df_ad(sub_tensor=sub_tensor, g=g, mode="nsy")
    assert df_ad[["group", "state_from", "state_to"]].to_records(index=False).tolist() == [
        ("nsy", "A", "B"),
        ("nsy", "B", "A"),
    ]
    numpy.testing.assert_allclose(df_ad["value"].to_numpy(), [3.0, 4.0], atol=1e-12)

    df_ad = main_site.add_site_stats(df_ad, sub_tensor, g, "nsy", method="tau")
    df_ad = main_site.add_site_stats(df_ad, sub_tensor, g, "nsy", method="tsi")
    df_ad = main_site.add_site_stats(df_ad, sub_tensor, g, "nsy", method="rank1")
    df_ad = main_site.add_site_stats(df_ad, sub_tensor, g, "nsy", method="rank2")
    numpy.testing.assert_allclose(df_ad["site_tau"].to_numpy(), [0.5, 2.0 / 3.0], atol=1e-12)
    numpy.testing.assert_allclose(df_ad["site_tsi"].to_numpy(), [2.0 / 3.0, 3.0 / 4.0], atol=1e-12)
    numpy.testing.assert_allclose(df_ad["site_rank1"].to_numpy(), [2.0, 3.0], atol=1e-12)
    numpy.testing.assert_allclose(df_ad["site_rank2"].to_numpy(), [1.0, 1.0], atol=1e-12)

    flagged = main_site.add_has_target_high_combinat_prob_site(df_ad, sub_tensor, g, "nsy")
    assert flagged["has_target_high_combinat_prob_site"].tolist() == [True, True]


def test_get_df_dist_reports_max_distance_for_multi_branch_substitutions(tiny_tree):
    g = {"tree": tiny_tree, "amino_acid_orders": numpy.array(["A", "B"]), "matrix_groups": {"grp": ["AA", "AB"]}}
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    sub_tensor = numpy.zeros((num_node, 1, 1, 2, 2), dtype=float)
    a_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "A"][0]
    c_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "C"][0]
    b_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "B"][0]
    sub_tensor[a_id, 0, 0, 0, 1] = 0.6
    sub_tensor[c_id, 0, 0, 0, 1] = 0.6
    sub_tensor[b_id, 0, 0, 1, 0] = 0.7
    out = main_site.get_df_dist(sub_tensor=sub_tensor, g=g, mode="nsy")
    row_ab = out.loc[(out["state_from"] == "A") & (out["state_to"] == "B"), :]
    row_ba = out.loc[(out["state_from"] == "B") & (out["state_to"] == "A"), :]
    assert pytest.approx(float(row_ab["max_dist_bl"].iloc[0]), rel=0, abs=1e-12) == 1.0
    assert pandas.isna(row_ba["max_dist_bl"].iloc[0])


def test_get_substitution_tensor_asis_matches_manual_outer_products():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    state = numpy.zeros((3, 2, 2), dtype=float)
    state[labels["R"], :, :] = [[1.0, 0.0], [0.5, 0.5]]
    state[labels["A"], :, :] = [[0.0, 1.0], [1.0, 0.0]]
    state[labels["B"], :, :] = [[1.0, 0.0], [0.0, 1.0]]
    g = {"tree": tr, "ml_anc": "yes", "float_tol": 1e-12}
    mmap_file = Path("tmp.csubst.sub_tensor.toy.mmap")
    try:
        out = substitution.get_substitution_tensor(state_tensor=state, mode="asis", g=g, mmap_attr="toy")
        # Branch A, site 0: parent state 0 -> child state 1 with prob 1.
        numpy.testing.assert_allclose(out[labels["A"], 0, 0, :, :], [[0.0, 1.0], [0.0, 0.0]], atol=1e-12)
        # Branch A, site 1: parent [0.5, 0.5], child state 0 => only 1->0 survives diag masking.
        numpy.testing.assert_allclose(out[labels["A"], 1, 0, :, :], [[0.0, 0.0], [0.5, 0.0]], atol=1e-12)
    finally:
        if mmap_file.exists():
            mmap_file.unlink()


def test_apply_min_sub_pp_threshold():
    g = {"min_sub_pp": 0.3, "ml_anc": False}
    sub = numpy.array([[[[[0.2, 0.4], [0.1, 0.5]]]]], dtype=float)
    out = substitution.apply_min_sub_pp(g, sub)
    numpy.testing.assert_allclose(out, [[[[[0.0, 0.4], [0.0, 0.5]]]]], atol=1e-12)


def test_get_s_get_cs_and_get_bs_match_manual_values():
    # shape = [branch, site, group, from, to]
    sub = numpy.zeros((2, 2, 1, 2, 2), dtype=float)
    sub[0, 0, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    sub[1, 0, 0, :, :] = [[0.0, 0.5], [0.2, 0.0]]
    sub[0, 1, 0, :, :] = [[0.0, 0.4], [0.0, 0.0]]
    sub[1, 1, 0, :, :] = [[0.0, 0.1], [0.3, 0.0]]

    s = substitution.get_s(sub, attr="N")
    numpy.testing.assert_allclose(s["N_sub"].to_numpy(), [1.0, 0.8], atol=1e-12)

    cs = substitution.get_cs(numpy.array([[0, 1]]), sub, attr="N")
    numpy.testing.assert_allclose(cs["OCNany2any"].to_numpy(), [0.21, 0.16], atol=1e-12)
    numpy.testing.assert_allclose(cs["OCNspe2any"].to_numpy(), [0.12, 0.04], atol=1e-12)
    numpy.testing.assert_allclose(cs["OCNany2spe"].to_numpy(), [0.12, 0.04], atol=1e-12)
    numpy.testing.assert_allclose(cs["OCNspe2spe"].to_numpy(), [0.12, 0.04], atol=1e-12)

    bs = substitution.get_bs(S_tensor=sub, N_tensor=sub * 2.0)
    # first branch, first site
    row0 = bs.loc[(bs["branch_id"] == 0) & (bs["site"] == 0), :].iloc[0]
    assert pytest.approx(float(row0["S_sub"]), abs=1e-12) == 0.3
    assert pytest.approx(float(row0["N_sub"]), abs=1e-12) == 0.6


def test_get_b_uses_tree_numerical_labels_for_branch_ids(tiny_tree):
    num_node = max(ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()) + 1
    sub = numpy.zeros((num_node, 1, 1, 2, 2), dtype=float)
    a_id = [ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse() if n.name == "A"][0]
    sub[a_id, 0, 0, 0, 1] = 0.5

    out = substitution.get_b(g={"tree": tiny_tree, "num_node": num_node}, sub_tensor=sub, attr="S", sitewise=False)
    expected_ids = sorted([ete.get_prop(n, "numerical_label") for n in tiny_tree.traverse()])
    assert out["branch_id"].tolist() == expected_ids
    assert set(out["branch_name"]) == set([n.name for n in tiny_tree.traverse()])


def test_add_dif_column_and_add_dif_stats():
    cb = pandas.DataFrame(
        {
            "OCSany2any": [1.0, 1.0],
            "OCSany2spe": [0.4, 1.2],
            "OCSspe2any": [0.5, 0.2],
            "OCSspe2spe": [0.4, 0.2],
            "OCNany2any": [1.0, 1.0],
            "OCNany2spe": [0.5, 0.2],
            "OCNspe2any": [0.5, 0.2],
            "OCNspe2spe": [0.1, 0.3],
        }
    )
    out = substitution.add_dif_column(cb.copy(), "tmp", "OCSany2any", "OCSany2spe", tol=1e-6)
    numpy.testing.assert_allclose(out["tmp"].to_numpy(), [0.6, numpy.nan], equal_nan=True)

    out2 = substitution.add_dif_stats(cb.copy(), tol=1e-6, prefix="OC")
    assert "OCSany2dif" in out2.columns
    assert "OCNdif2spe" in out2.columns


def test_get_substitution_tensor_syn_matches_manual_groupwise_products():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    # codon state order: [AAA, AAG, TTT, TTC]
    # synonymous groups: K=[AAA,AAG], F=[TTT,TTC]
    state = numpy.zeros((3, 1, 4), dtype=float)
    state[labels["R"], 0, :] = [0.6, 0.4, 0.0, 0.0]
    state[labels["A"], 0, :] = [0.1, 0.9, 0.0, 0.0]
    state[labels["B"], 0, :] = [0.8, 0.2, 0.0, 0.0]
    g = {
        "tree": tr,
        "ml_anc": "yes",
        "float_tol": 1e-12,
        "amino_acid_orders": ["K", "F"],
        "synonymous_indices": {"K": [0, 1], "F": [2, 3]},
        "max_synonymous_size": 2,
    }
    mmap_file = Path("tmp.csubst.sub_tensor.toy_syn.mmap")
    try:
        out = substitution.get_substitution_tensor(state_tensor=state, mode="syn", g=g, mmap_attr="toy_syn")
        # Branch A, group K: diag masked outer product of parent [0.6,0.4] and child [0.1,0.9].
        numpy.testing.assert_allclose(out[labels["A"], 0, 0, :, :], [[0.0, 0.54], [0.04, 0.0]], atol=1e-12)
        # Group F has no support in this toy example.
        numpy.testing.assert_allclose(out[labels["A"], 0, 1, :, :], [[0.0, 0.0], [0.0, 0.0]], atol=1e-12)
        # Branch B, group K:
        numpy.testing.assert_allclose(out[labels["B"], 0, 0, :, :], [[0.0, 0.12], [0.32, 0.0]], atol=1e-12)
    finally:
        if mmap_file.exists():
            mmap_file.unlink()


def _toy_sub_tensor():
    # shape = [branch, site, group, from, to]
    sub = numpy.zeros((3, 2, 1, 2, 2), dtype=float)
    sub[0, 0, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    sub[1, 0, 0, :, :] = [[0.0, 0.5], [0.2, 0.0]]
    sub[2, 0, 0, :, :] = [[0.0, 0.4], [0.3, 0.0]]
    sub[0, 1, 0, :, :] = [[0.0, 0.1], [0.0, 0.0]]
    sub[1, 1, 0, :, :] = [[0.0, 0.1], [0.3, 0.0]]
    sub[2, 1, 0, :, :] = [[0.0, 0.2], [0.1, 0.0]]
    return sub


def test_sub_tensor2cb_mmap_chunk_writer_matches_non_mmap(tmp_path):
    sub = _toy_sub_tensor()
    ids = numpy.array([[2, 0], [1, 2]], dtype=numpy.int64)
    expected = substitution.sub_tensor2cb(ids, sub, mmap=False, df_mmap=None, mmap_start=0, float_type=numpy.float64)

    arity = ids.shape[1]
    mmap_path = tmp_path / "cb_writer.mmap"
    mmap_out = numpy.memmap(mmap_path, dtype=numpy.float64, mode="w+", shape=(ids.shape[0] + 1, arity + 4))
    mmap_out[:] = 0.0
    substitution.sub_tensor2cb(ids, sub, mmap=True, df_mmap=mmap_out, mmap_start=1, float_type=numpy.float64)
    observed = numpy.array(mmap_out[1 : 1 + ids.shape[0], :], copy=True)
    del mmap_out

    numpy.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cbs_mmap_chunk_writer_matches_non_mmap(tmp_path):
    sub = _toy_sub_tensor()
    ids = numpy.array([[2, 0], [1, 2]], dtype=numpy.int64)
    expected = substitution.sub_tensor2cbs(ids, sub, mmap=False, df_mmap=None, mmap_start=0)

    arity = ids.shape[1]
    num_site = sub.shape[1]
    mmap_path = tmp_path / "cbs_writer.mmap"
    mmap_rows = (ids.shape[0] + 1) * num_site
    mmap_out = numpy.memmap(mmap_path, dtype=numpy.float64, mode="w+", shape=(mmap_rows, arity + 5))
    mmap_out[:] = 0.0
    substitution.sub_tensor2cbs(ids, sub, mmap=True, df_mmap=mmap_out, mmap_start=1)
    row_start = num_site
    row_end = row_start + expected.shape[0]
    observed = numpy.array(mmap_out[row_start:row_end, :], copy=True)
    del mmap_out

    numpy.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cb_cython_fastpath_matches_python_fallback(monkeypatch):
    if not hasattr(substitution_cy, "calc_combinatorial_sub_double_arity2"):
        pytest.skip("Cython dense reducer fast path is unavailable")
    rng = numpy.random.default_rng(7)
    sub = rng.random((5, 4, 2, 3, 3), dtype=numpy.float64)
    ids = numpy.array([[0, 1], [2, 3], [4, 1]], dtype=numpy.int64)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cb", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cb(ids, sub, mmap=False, df_mmap=None, mmap_start=0, float_type=numpy.float64)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cb", lambda *args, **kwargs: True)
    observed = substitution.sub_tensor2cb(ids, sub, mmap=False, df_mmap=None, mmap_start=0, float_type=numpy.float64)

    numpy.testing.assert_allclose(observed, expected, atol=1e-12)


def test_sub_tensor2cbs_cython_fastpath_matches_python_fallback(monkeypatch):
    if not hasattr(substitution_cy, "calc_combinatorial_sub_by_site_double_arity2"):
        pytest.skip("Cython dense reducer fast path is unavailable")
    rng = numpy.random.default_rng(11)
    sub = rng.random((4, 3, 2, 3, 3), dtype=numpy.float64)
    ids = numpy.array([[0, 1], [2, 3]], dtype=numpy.int64)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cbs", lambda *args, **kwargs: False)
    expected = substitution.sub_tensor2cbs(ids, sub, mmap=False, df_mmap=None, mmap_start=0)

    monkeypatch.setattr(substitution, "_can_use_cython_dense_cbs", lambda *args, **kwargs: True)
    observed = substitution.sub_tensor2cbs(ids, sub, mmap=False, df_mmap=None, mmap_start=0)

    numpy.testing.assert_allclose(observed, expected, atol=1e-12)


def test_get_cb_matches_sum_of_get_cs_per_combination():
    sub = _toy_sub_tensor()
    ids = numpy.array([[2, 0], [1, 2]], dtype=numpy.int64)
    cb = substitution.get_cb(ids, sub, {"threads": 1, "float_type": numpy.float64}, attr="OCN")

    for combo in ids:
        c1, c2 = sorted(combo.tolist())
        row = cb.loc[(cb["branch_id_1"] == c1) & (cb["branch_id_2"] == c2), :].iloc[0]
        cs = substitution.get_cs(numpy.array([[c1, c2]], dtype=numpy.int64), sub, attr="N")
        assert pytest.approx(float(row["OCNany2any"]), abs=1e-12) == float(cs["OCNany2any"].sum())
        assert pytest.approx(float(row["OCNspe2any"]), abs=1e-12) == float(cs["OCNspe2any"].sum())
        assert pytest.approx(float(row["OCNany2spe"]), abs=1e-12) == float(cs["OCNany2spe"].sum())
        assert pytest.approx(float(row["OCNspe2spe"]), abs=1e-12) == float(cs["OCNspe2spe"].sum())


def test_get_cbs_grouped_sum_matches_get_cb():
    sub = _toy_sub_tensor()
    ids = numpy.array([[2, 0], [1, 2]], dtype=numpy.int64)

    cb = substitution.get_cb(ids, sub, {"threads": 1, "float_type": numpy.float64}, attr="OCN")
    cbs = substitution.get_cbs(ids, sub, attr="N", g={"threads": 1})
    cols = ["OCNany2any", "OCNspe2any", "OCNany2spe", "OCNspe2spe"]
    summed = cbs.groupby(["branch_id_1", "branch_id_2"], as_index=False)[cols].sum()
    merged = cb.merge(summed, on=["branch_id_1", "branch_id_2"], suffixes=("_cb", "_cbs"))

    for col in cols:
        numpy.testing.assert_allclose(
            merged[f"{col}_cb"].to_numpy(),
            merged[f"{col}_cbs"].to_numpy(),
            atol=1e-12,
        )

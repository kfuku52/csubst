import numpy as np
import pandas as pd
import pytest

from csubst import ete
from csubst import main_scan
from csubst import substitution_sparse
from csubst import substitution_scan
from csubst import tree


def test_normalize_scan_matches_all_expands_to_nine_classes():
    assert substitution_scan.normalize_scan_matches("all") == list(substitution_scan.SCAN_MATCHES)
    assert len(substitution_scan.normalize_scan_matches("all")) == 9
    assert substitution_scan.normalize_scan_matches("any2spe,spe2spe,any2spe") == [
        "any2spe",
        "spe2spe",
    ]


def test_build_candidates_supports_scan_match_specific_grouping():
    events = pd.DataFrame(
        {
            "branch_id": [1, 2, 3],
            "site": [0, 0, 0],
            "from_state_id": [0, 2, 0],
            "to_state_id": [1, 1, 3],
            "event_pp": [0.8, 0.7, 0.6],
        }
    )
    state_orders = np.array(["A", "K", "C", "N"], dtype=object)

    candidates = substitution_scan.build_candidates(
        events=events,
        scan_matches=["any2spe", "dif2spe", "spe2spe"],
        state_orders=state_orders,
    )

    any_to_k = candidates.loc[
        (candidates["scan_match"] == "any2spe") & (candidates["to_state"] == "K")
    ].iloc[0]
    assert any_to_k["from_state"] == "any"
    assert any_to_k["state_change"] == "1K"

    dif_to_k = candidates.loc[
        (candidates["scan_match"] == "dif2spe") & (candidates["to_state"] == "K")
    ].iloc[0]
    assert dif_to_k["from_state"] == "dif"

    spe_to_spe = candidates.loc[candidates["scan_match"] == "spe2spe", :]
    assert set(spe_to_spe["state_change"].tolist()) == {"A1K", "A1N", "C1K"}


def test_unit_support_counts_foreground_lineages_once_and_rejects_non_foreground_targets():
    candidate_events = pd.DataFrame(
        {
            "branch_id": [1, 3, 4],
            "site": [0, 0, 0],
            "from_state_id": [0, 0, 0],
            "to_state_id": [1, 1, 1],
            "event_pp": [0.9, 0.8, 0.7],
        }
    )
    units = pd.DataFrame(
        {
            "unit_id": [1, 2],
            "fg_branch_ids": ["1", "2"],
        }
    )

    fg = substitution_scan._summarize_unit_support(
        candidate_events=candidate_events,
        units_df=units,
        target_class="fg",
        min_event_pp=0.5,
    )

    assert fg["support_unit_ids"] == "1"
    assert fg["support_branch_ids"] == "1"
    with pytest.raises(ValueError, match="Only foreground target class"):
        substitution_scan._summarize_unit_support(
            candidate_events=candidate_events,
            units_df=units,
            target_class="mg",
            min_event_pp=0.5,
        )


def test_extract_candidate_posterior_events_supports_sparse_tensors():
    dense = np.zeros((4, 1, 1, 2, 2), dtype=float)
    dense[1, 0, 0, 0, 1] = 0.4
    dense[2, 0, 0, 0, 1] = 0.7
    dense[3, 0, 0, 1, 1] = 0.9
    sparse = substitution_sparse.SparseSubstitutionTensor.from_dense(dense)

    events = substitution_scan.extract_candidate_posterior_events(
        sub_tensor=sparse,
        site=0,
        from_ids=np.array([0], dtype=np.int64),
        to_ids=np.array([1], dtype=np.int64),
    )

    assert events["branch_id"].tolist() == [1, 2]
    assert events["event_pp"].tolist() == pytest.approx([0.4, 0.7])


def test_rate_summary_state_aware_excludes_completed_clade_branches_but_allows_reversion_opportunity():
    branch_meta = pd.DataFrame(
        {
            "branch_id": [1, 2, 3, 4],
            "parent_id": [0, 1, 2, 3],
            "raw_length": [1.0, 1.0, 1.0, 1.0],
            "sn_rescaled_length": [10.0, 10.0, 10.0, 10.0],
            "n_rescaled_length": [5.0, 5.0, 5.0, 5.0],
        }
    )
    state_nsy = np.zeros((5, 1, 2), dtype=float)
    state_nsy[0, 0, :] = [1.0, 0.0]
    state_nsy[1, 0, :] = [0.0, 1.0]
    state_nsy[2, 0, :] = [1.0, 0.0]
    state_nsy[3, 0, :] = [1.0, 0.0]
    state_nsy[4, 0, :] = [0.0, 1.0]
    candidate_events = pd.DataFrame(
        {
            "branch_id": [1, 3],
            "site": [0, 0],
            "from_state_id": [0, 0],
            "to_state_id": [1, 1],
            "event_pp": [1.0, 1.0],
        }
    )

    state_aware = substitution_scan._rate_summary(
        candidate_events=candidate_events,
        branch_meta=branch_meta,
        state_nsy=state_nsy,
        site=0,
        from_ids=np.array([0], dtype=np.int64),
        to_ids=np.array([1], dtype=np.int64),
        target_branch_ids=np.array([1, 2, 3], dtype=np.int64),
        rate_length="raw",
        rate_exposure="state_aware",
    )
    raw_exposure = substitution_scan._rate_summary(
        candidate_events=candidate_events,
        branch_meta=branch_meta,
        state_nsy=state_nsy,
        site=0,
        from_ids=np.array([0], dtype=np.int64),
        to_ids=np.array([1], dtype=np.int64),
        target_branch_ids=np.array([1, 2, 3], dtype=np.int64),
        rate_length="raw",
        rate_exposure="raw_branch_length",
    )

    assert state_aware["target_raw_branch_length"] == pytest.approx(3.0)
    assert state_aware["target_exposure_branch_length"] == pytest.approx(2.0)
    assert state_aware["other_exposure_branch_length"] == pytest.approx(1.0)
    assert raw_exposure["target_exposure_branch_length"] == pytest.approx(3.0)


def test_rate_summary_q_weighted_exposure_uses_instantaneous_nsy_rates():
    branch_meta = pd.DataFrame(
        {
            "branch_id": [1, 2],
            "parent_id": [0, 1],
            "raw_length": [1.0, 2.0],
            "sn_rescaled_length": [1.0, 2.0],
            "n_rescaled_length": [1.0, 2.0],
        }
    )
    state_nsy = np.zeros((3, 1, 2), dtype=float)
    state_nsy[0, 0, :] = [1.0, 0.0]
    state_nsy[1, 0, :] = [0.5, 0.5]
    state_nsy[2, 0, :] = [0.0, 1.0]
    q_matrix = np.array([[-2.0, 2.0], [5.0, -5.0]], dtype=float)
    candidate_events = pd.DataFrame(
        {
            "branch_id": [1, 2],
            "site": [0, 0],
            "from_state_id": [0, 0],
            "to_state_id": [1, 1],
            "event_pp": [1.0, 1.0],
        }
    )

    out = substitution_scan._rate_summary(
        candidate_events=candidate_events,
        branch_meta=branch_meta,
        state_nsy=state_nsy,
        site=0,
        from_ids=np.array([0], dtype=np.int64),
        to_ids=np.array([1], dtype=np.int64),
        target_branch_ids=np.array([1], dtype=np.int64),
        other_branch_ids=np.array([2], dtype=np.int64),
        rate_length="raw",
        rate_exposure="q_weighted",
        q_matrix=q_matrix,
    )

    assert out["target_exposure_branch_length"] == pytest.approx(2.0)
    assert out["other_exposure_branch_length"] == pytest.approx(2.0)


def test_rate_summary_q_weighted_normalizes_rates_for_n_rescaled_lengths():
    branch_meta = pd.DataFrame(
        {
            "branch_id": [1, 2],
            "parent_id": [0, 1],
            "raw_length": [6.0, 6.0],
            "sn_rescaled_length": [6.0, 6.0],
            "n_rescaled_length": [6.0, 6.0],
        }
    )
    state_nsy = np.zeros((3, 1, 3), dtype=float)
    state_nsy[0, 0, :] = [1.0, 0.0, 0.0]
    state_nsy[1, 0, :] = [1.0, 0.0, 0.0]
    state_nsy[2, 0, :] = [0.0, 1.0, 0.0]
    q_matrix = np.array(
        [
            [-3.0, 1.0, 2.0],
            [4.0, -5.0, 1.0],
            [1.0, 1.0, -2.0],
        ],
        dtype=float,
    )
    candidate_events = pd.DataFrame(
        {
            "branch_id": [1, 2],
            "site": [0, 0],
            "from_state_id": [0, 0],
            "to_state_id": [1, 1],
            "event_pp": [1.0, 1.0],
        }
    )

    out = substitution_scan._rate_summary(
        candidate_events=candidate_events,
        branch_meta=branch_meta,
        state_nsy=state_nsy,
        site=0,
        from_ids=np.array([0], dtype=np.int64),
        to_ids=np.array([1], dtype=np.int64),
        target_branch_ids=np.array([1], dtype=np.int64),
        other_branch_ids=np.array([2], dtype=np.int64),
        rate_length="n_rescaled",
        rate_exposure="q_weighted",
        q_matrix=q_matrix,
    )

    assert out["target_exposure_branch_length"] == pytest.approx(2.0)
    assert out["other_exposure_branch_length"] == pytest.approx(2.0)


def _set_state(state, branch_id, site, state_id):
    state[int(branch_id), int(site), :] = 0.0
    state[int(branch_id), int(site), int(state_id)] = 1.0


def _toy_scan_context():
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("((A:1,B:1)X:1,(C:1,D:1)Y:1)R;", format=1)
    )
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    num_node = max(labels.values()) + 1
    for node in tr.traverse():
        ete.set_prop(node, "SNdist", 0.25)
        ete.set_prop(node, "Ndist", 0.10)
    fg_leaf_names = {"trait": [["A"], ["C"]]}
    for i, names in enumerate(fg_leaf_names["trait"], start=1):
        name_set = set(names)
        for node in tr.traverse():
            node_leaf_names = set(ete.get_leaf_names(node))
            ete.add_features(
                node,
                **{"is_lineage_fg_trait_{}".format(i): node_leaf_names.issubset(name_set)},
            )
    for node in tr.traverse():
        ete.add_features(node, is_fg_trait=False)
    state_nsy = np.zeros((num_node, 1, 2), dtype=float)
    state_pep = np.zeros((num_node, 1, 2), dtype=float)
    for node_id in labels.values():
        _set_state(state_nsy, node_id, 0, 0)
        _set_state(state_pep, node_id, 0, 0)
    for name in ["A", "C"]:
        _set_state(state_nsy, labels[name], 0, 1)
        _set_state(state_pep, labels[name], 0, 1)
        ete.add_features(next(node for node in tr.traverse() if node.name == name), is_fg_trait=True)
    on_tensor = np.zeros((num_node, 1, 1, 2, 2), dtype=float)
    on_tensor[labels["A"], 0, 0, 0, 1] = 0.9
    on_tensor[labels["C"], 0, 0, 0, 1] = 0.8
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"name": ["A", "C"], "trait": [1, 2]}),
        "fg_leaf_names": fg_leaf_names,
        "fg_ids": {"trait": np.array([labels["A"], labels["C"]], dtype=np.int64)},
        "fg_stem_only": True,
        "scan_sister_stem_only": True,
        "state_nsy": state_nsy,
        "state_pep": state_pep,
        "nonsyn_state_orders": np.array(["A", "K"], dtype=object),
        "amino_acid_orders": np.array(["A", "K"], dtype=object),
        "iqtree_rate_values": np.array([0.25], dtype=float),
        "float_tol": 1e-12,
        "nonsyn_recode": "no",
        "scan_match": "any2spe",
        "scan_min_event_pp": 0.5,
        "scan_min_support": "2",
        "scan_rate_length": "raw",
        "scan_rate_exposure": "state_aware",
        "scan_rate_event_mode": "posterior_sum",
        "scan_other_scope": "all",
        "scan_pvalue_calibration": "none",
        "scan_n_permutations": 0,
        "scan_permutation_seed": 1,
        "scan_permutation_sample_original": False,
        "scan_permutation_retry_sample_original": True,
        "min_clade_bin_count": 1,
    }
    return g, on_tensor


def test_scan_substitutions_outputs_foreground_rows():
    g, on_tensor = _toy_scan_context()

    scan_df, units = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    assert units.shape[0] == 2
    assert scan_df.shape[0] == 1
    assert scan_df["scan_id"].nunique() == 1
    assert set(scan_df["target_class"].tolist()) == {"fg"}
    fg_row = scan_df.iloc[0]
    assert fg_row["state_change"] == "1K"
    assert fg_row["support_unit_count"] == 2
    assert fg_row["target_event_count"] == pytest.approx(1.7)
    assert fg_row["target_exposure_branch_length"] == pytest.approx(2.0)
    assert fg_row["site_rate"] == pytest.approx(0.25)


def test_build_scan_site_plot_table_uses_supporting_branch_event_pp():
    g, on_tensor = _toy_scan_context()
    scan_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in g["tree"].traverse()}

    site_df, branch_ids = substitution_scan.build_scan_site_plot_table(
        scan_df=scan_df,
        g=g,
        ON_tensor=on_tensor,
    )

    assert branch_ids.tolist() == [labels["A"], labels["C"]]
    assert site_df["codon_site_alignment"].tolist() == [1]
    assert site_df["OCNany2spe"].iloc[0] == pytest.approx(0.9)
    assert site_df["N_sub_{}".format(labels["A"])].iloc[0] == pytest.approx(0.9)
    assert site_df["N_sub_{}".format(labels["C"])].iloc[0] == pytest.approx(0.8)


def test_write_scan_site_plot_reuses_sites_plotter(monkeypatch, tmp_path):
    g, on_tensor = _toy_scan_context()
    g["outdir"] = str(tmp_path)
    g["tree_site_plot_format"] = "svg"
    scan_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in g["tree"].traverse()}
    g["fg_ids"] = {"trait": np.array([labels["X"], labels["A"], labels["C"]], dtype=np.int64)}
    calls = []

    def fake_plot_tree_site(df, g):
        calls.append((df.copy(deep=True), dict(g)))
        return [str(tmp_path / "csubst_scan.tree_site.svg")]

    monkeypatch.setattr(main_scan.main_sites, "plot_tree_site", fake_plot_tree_site)

    out_paths = main_scan._write_scan_site_plot(g=g, scan_df=scan_df, ON_tensor=on_tensor)

    assert out_paths == [str(tmp_path / "csubst_scan.tree_site.svg")]
    assert len(calls) == 1
    plot_df, plot_g = calls[0]
    assert plot_g["mode"] == "lineage"
    assert plot_g["site_outdir"] == str(tmp_path)
    assert plot_g["tree_site_plot_prefix"] == "csubst_scan"
    assert plot_g["tree_site_plot_format"] == "svg"
    assert plot_g["tree_site_output_table"] is False
    assert plot_g["tree_site_branch_color_mode"] == "single"
    assert plot_g["tree_site_highlight_branch_ids"].tolist() == sorted([labels["X"], labels["A"], labels["C"]])
    assert plot_df["codon_site_alignment"].tolist() == [1]


def test_scan_state_change_uses_alignment_site_coordinate():
    g, on_tensor = _toy_scan_context()
    g["site_index_alignment"] = np.array([9], dtype=np.int64)

    scan_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    row = scan_df.iloc[0]
    assert row["site"] == 0
    assert row["codon_site_alignment"] == 10
    assert row["state_change"] == "10K"


def test_scan_substitutions_supports_sparse_substitution_tensor_end_to_end():
    g, on_tensor = _toy_scan_context()
    sparse_tensor = substitution_sparse.SparseSubstitutionTensor.from_dense(on_tensor)

    scan_df, units = substitution_scan.scan_substitutions(
        g=g,
        ON_tensor=sparse_tensor,
        rate_ON_tensor=sparse_tensor,
    )

    assert units.shape[0] == 2
    assert scan_df.shape[0] == 1
    assert scan_df.loc[scan_df["target_class"] == "fg", "target_event_count"].iloc[0] == pytest.approx(1.7)


def test_scan_posterior_sum_uses_unthresholded_rate_tensor_when_called_tensor_is_thresholded():
    g, raw_tensor = _toy_scan_context()
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in g["tree"].traverse()}
    raw_tensor[labels["B"], 0, 0, 0, 1] = 0.2
    called_tensor = raw_tensor.copy()
    called_tensor[called_tensor < 0.5] = 0.0
    g["scan_rate_event_mode"] = "posterior_sum"

    scan_df, _ = substitution_scan.scan_substitutions(
        g=g,
        ON_tensor=called_tensor,
        rate_ON_tensor=raw_tensor,
    )

    assert scan_df.iloc[0]["target_event_count"] == pytest.approx(1.7)
    assert scan_df.iloc[0]["other_event_count"] == pytest.approx(0.2)


def test_scan_substitutions_handles_multiple_traits_and_stratified_qvalues():
    g, on_tensor = _toy_scan_context()
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in g["tree"].traverse()}
    g["fg_df"]["trait2"] = [1, 2]
    g["fg_leaf_names"]["trait2"] = [["A"], ["C"]]
    g["fg_ids"]["trait2"] = np.array([labels["A"], labels["C"]], dtype=np.int64)
    for i, names in enumerate(g["fg_leaf_names"]["trait2"], start=1):
        name_set = set(names)
        for node in g["tree"].traverse():
            node_leaf_names = set(ete.get_leaf_names(node))
            ete.add_features(
                node,
                **{"is_lineage_fg_trait2_{}".format(i): node_leaf_names.issubset(name_set)},
            )
            ete.add_features(node, is_fg_trait2=node.name in {"A", "C"})

    scan_df, units = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    assert set(scan_df["trait"].tolist()) == {"trait", "trait2"}
    assert units.groupby("trait").size().to_dict() == {"trait": 2, "trait2": 2}
    assert scan_df.groupby("trait").size().to_dict() == {"trait": 1, "trait2": 1}
    q_cols = [
        "q_rate_enrichment",
        "q_rate_enrichment_by_trait",
        "q_rate_enrichment_by_trait_match",
    ]
    for col in q_cols:
        assert col in scan_df.columns
        assert np.isfinite(scan_df[col].to_numpy(dtype=float)).all()
    assert np.allclose(
        scan_df["q_rate_enrichment_by_trait_match"].to_numpy(dtype=float),
        scan_df["p_rate_enrichment"].to_numpy(dtype=float),
    )


def test_scan_full_scan_permutation_adds_empirical_maxt_pvalues():
    g, on_tensor = _toy_scan_context()
    g["scan_pvalue_calibration"] = "full_scan"
    g["scan_n_permutations"] = 4
    g["scan_permutation_seed"] = 3

    scan_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    assert scan_df.shape[0] == 1
    row = scan_df.iloc[0]
    assert row["scan_pvalue_calibration"] == "full_scan"
    assert row["scan_permutation_success_count"] == 4
    assert row["scan_permutation_failure_count"] == 0
    assert row["scan_permutation_failure_reasons"] == ""
    assert np.isfinite(float(row["p_rate_enrichment_empirical_maxT"]))
    assert 0 < float(row["p_rate_enrichment_empirical_maxT"]) <= 1


def test_scan_candidate_fixed_permutation_adds_empirical_pvalues():
    g, on_tensor = _toy_scan_context()
    g["scan_pvalue_calibration"] = "candidate_fixed"
    g["scan_n_permutations"] = 4
    g["scan_permutation_seed"] = 3

    scan_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    row = scan_df.iloc[0]
    assert row["scan_pvalue_calibration"] == "candidate_fixed"
    assert row["scan_permutation_success_count"] == 4
    assert np.isfinite(float(row["p_rate_enrichment_empirical"]))
    assert np.isnan(float(row["p_rate_enrichment_empirical_maxT"]))


def test_scan_permutation_failures_report_reasons(monkeypatch, capsys):
    g, on_tensor = _toy_scan_context()
    g["scan_pvalue_calibration"] = "candidate_fixed"
    g["scan_n_permutations"] = 2
    g["scan_permutation_seed"] = 3

    def fail_permutation_context(*args, **kwargs):
        raise RuntimeError("permutation context boom")

    monkeypatch.setattr(
        substitution_scan,
        "_build_permuted_context_with_seed",
        fail_permutation_context,
    )

    scan_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    captured = capsys.readouterr()
    row = scan_df.iloc[0]
    assert row["scan_permutation_success_count"] == 0
    assert row["scan_permutation_failure_count"] == 2
    assert "RuntimeError: permutation context boom" in row["scan_permutation_failure_reasons"]
    assert "2 of 2 permutations failed" in captured.out
    assert np.isnan(float(row["p_rate_enrichment_empirical"]))


def test_scan_permutations_use_parallel_backend_and_chunks(monkeypatch):
    g, on_tensor = _toy_scan_context()
    g["scan_pvalue_calibration"] = "candidate_fixed"
    g["scan_n_permutations"] = 4
    g["scan_permutation_seed"] = 3
    g["threads"] = 2
    calls = []

    def fake_run_starmap(func, args_iterable, n_jobs, backend="multiprocessing", chunksize=None):
        args = list(args_iterable)
        calls.append((len(args), n_jobs, backend, chunksize))
        return [func(*arg) for arg in args]

    monkeypatch.setattr(substitution_scan.parallel, "run_starmap", fake_run_starmap)

    scan_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    assert calls == [(2, 2, "multiprocessing", None)]
    row = scan_df.iloc[0]
    assert row["scan_permutation_backend"] == "multiprocessing"
    assert row["scan_permutation_n_jobs"] == 2
    assert row["scan_permutation_success_count"] == 4


def test_scan_parallel_permutation_matches_single_thread_result():
    serial_g, serial_tensor = _toy_scan_context()
    serial_g["scan_pvalue_calibration"] = "full_scan"
    serial_g["scan_n_permutations"] = 4
    serial_g["scan_permutation_seed"] = 3
    serial_g["threads"] = 1
    parallel_g, parallel_tensor = _toy_scan_context()
    parallel_g["scan_pvalue_calibration"] = "full_scan"
    parallel_g["scan_n_permutations"] = 4
    parallel_g["scan_permutation_seed"] = 3
    parallel_g["threads"] = 2

    serial_df, _ = substitution_scan.scan_substitutions(g=serial_g, ON_tensor=serial_tensor)
    parallel_df, _ = substitution_scan.scan_substitutions(g=parallel_g, ON_tensor=parallel_tensor)

    assert parallel_df.iloc[0]["scan_permutation_n_jobs"] == 2
    assert serial_df.iloc[0]["p_rate_enrichment_empirical"] == pytest.approx(
        parallel_df.iloc[0]["p_rate_enrichment_empirical"]
    )
    assert serial_df.iloc[0]["p_rate_enrichment_empirical_maxT"] == pytest.approx(
        parallel_df.iloc[0]["p_rate_enrichment_empirical_maxT"]
    )


def test_scan_rejects_negative_permutation_count_even_without_calibration():
    g, on_tensor = _toy_scan_context()
    g["scan_pvalue_calibration"] = "none"
    g["scan_n_permutations"] = -1

    with pytest.raises(ValueError, match="scan_n_permutations"):
        substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)


def test_empirical_pvalue_is_clamped_to_one_for_duplicate_null_values():
    out = substitution_scan._empirical_p_from_values(
        p_obs=0.05,
        values=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        denominator_count=4,
    )

    assert out == pytest.approx(1.0)


def test_scan_rate_event_mode_posterior_sum_keeps_low_pp_background_mass_for_rates():
    g, on_tensor = _toy_scan_context()
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in g["tree"].traverse()}
    on_tensor[labels["B"], 0, 0, 0, 1] = 0.2

    g["scan_rate_event_mode"] = "posterior_sum"
    posterior_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)
    g["scan_rate_event_mode"] = "called"
    called_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    posterior_row = posterior_df.iloc[0]
    called_row = called_df.iloc[0]
    assert posterior_row["target_event_count"] == pytest.approx(1.7)
    assert posterior_row["other_event_count"] == pytest.approx(0.2)
    assert called_row["target_event_count"] == pytest.approx(1.7)
    assert called_row["other_event_count"] == pytest.approx(0.0)


def test_scan_other_scope_limits_foreground_control_branches_to_sisters():
    g, on_tensor = _toy_scan_context()

    g["scan_other_scope"] = "all"
    all_df, units = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)
    g["scan_other_scope"] = "sister"
    sister_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    assert "sister_branch_ids" in units.columns
    assert all_df.iloc[0]["other_raw_branch_length"] == pytest.approx(4.0)
    assert sister_df.iloc[0]["other_raw_branch_length"] == pytest.approx(2.0)


def test_scan_substitutions_empty_result_preserves_output_schema(capsys):
    g, on_tensor = _toy_scan_context()
    g["scan_min_support"] = "3"

    scan_df, units = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    captured = capsys.readouterr()
    assert "--scan_min_support resolved to 3" in captured.out
    assert units.shape[0] == 2
    assert scan_df.empty
    assert list(scan_df.columns) == list(substitution_scan.SCAN_OUTPUT_COLUMNS)

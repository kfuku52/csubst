
import numpy as np
import pandas as pd
import pytest

from csubst import ete
from csubst import foreground
from csubst import main_scan
from csubst import substitution_scan
from csubst import substitution
from csubst import tree


def _set_state(state, branch_id, site, state_id):
    state[int(branch_id), int(site), :] = 0.0
    state[int(branch_id), int(site), int(state_id)] = 1.0


@pytest.mark.parametrize(
    ("match", "from_ids", "to_ids"),
    [
        ("any2spe", [0, 1, 2], [2]),
        ("spe2any", [0], [0, 1, 2]),
        ("any2any", [0, 1, 2], [0, 1, 2]),
    ],
)
def test_scan_sparse_rate_projection_matches_block_extraction(match, from_ids, to_ids):
    dense = np.zeros((3, 2, 1, 3, 3), dtype=np.float64)
    dense[0, 1, 0, 0, 2] = 0.2
    dense[0, 1, 0, 1, 2] = 0.3
    dense[1, 1, 0, 0, 1] = 0.4
    dense[2, 1, 0, 0, 0] = 0.9  # Diagonal mass must be excluded.
    sparse = substitution.dense_to_sparse_sub_tensor(dense)
    projection = substitution_scan._build_scan_rate_projection(sparse, [match])
    assert match not in projection

    expected = substitution_scan.extract_candidate_posterior_events(
        sub_tensor=sparse,
        site=1,
        from_ids=from_ids,
        to_ids=to_ids,
    )
    observed = substitution_scan.extract_candidate_posterior_events(
        sub_tensor=sparse,
        site=1,
        from_ids=from_ids,
        to_ids=to_ids,
        projection=projection,
    )

    pd.testing.assert_frame_equal(observed, expected, check_exact=True)


@pytest.mark.parametrize(
    ("match", "from_ids", "to_ids"),
    [
        ("any2spe", [0, 1, 2], [2]),
        ("spe2any", [0], [0, 1, 2]),
        ("any2any", [0, 1, 2], [0, 1, 2]),
    ],
)
def test_scan_sparse_rate_projection_uses_fast_path_for_zero_diagonal(match, from_ids, to_ids):
    dense = np.zeros((3, 2, 1, 3, 3), dtype=np.float64)
    dense[0, 1, 0, 0, 2] = 0.2
    dense[0, 1, 0, 1, 2] = 0.3
    dense[1, 1, 0, 0, 1] = 0.4
    sparse = substitution.dense_to_sparse_sub_tensor(dense)
    projection = substitution_scan._build_scan_rate_projection(sparse, [match])
    assert match in projection

    expected = substitution_scan.extract_candidate_posterior_events(
        sub_tensor=sparse,
        site=1,
        from_ids=from_ids,
        to_ids=to_ids,
    )
    observed = substitution_scan.extract_candidate_posterior_events(
        sub_tensor=sparse,
        site=1,
        from_ids=from_ids,
        to_ids=to_ids,
        projection=projection,
    )

    pd.testing.assert_frame_equal(observed, expected, check_exact=True)


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


def test_scan_called_rate_mode_does_not_build_unused_projection(monkeypatch):
    g, on_tensor = _toy_scan_context()
    g["scan_rate_event_mode"] = "called"
    monkeypatch.setattr(
        substitution_scan,
        "_build_scan_rate_projection",
        lambda *args, **kwargs: pytest.fail("called mode should not build a rate projection"),
    )

    context = substitution_scan._build_scan_static_context(
        g=g,
        ON_tensor=on_tensor,
        rate_ON_tensor=on_tensor,
    )

    assert context["rate_event_projection"] is None


def _toy_clade_scan_context():
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("(((A:1,B:1)X:1,(C:1,D:1)Y:1)W:1,(E:1,F:1)Z:1)R;", format=1)
    )
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    num_node = max(labels.values()) + 1
    fg_leaf_names = {"trait": [["A", "B"], ["C", "D"]]}
    for node in tr.traverse():
        ete.set_prop(node, "SNdist", 1.0)
        ete.set_prop(node, "Ndist", 1.0)
        ete.add_features(node, is_fg_trait=node.name in {"X", "Y"})
    for i, names in enumerate(fg_leaf_names["trait"], start=1):
        name_set = set(names)
        for node in tr.traverse():
            node_leaf_names = set(ete.get_leaf_names(node))
            ete.add_features(
                node,
                **{"is_lineage_fg_trait_{}".format(i): node_leaf_names.issubset(name_set)},
            )
    state_nsy = np.zeros((num_node, 1, 2), dtype=float)
    state_pep = np.zeros((num_node, 1, 2), dtype=float)
    for node_id in labels.values():
        _set_state(state_nsy, node_id, 0, 0)
        _set_state(state_pep, node_id, 0, 0)
    for name in ["X", "Y", "A", "B", "C", "D"]:
        _set_state(state_nsy, labels[name], 0, 1)
        _set_state(state_pep, labels[name], 0, 1)
    on_tensor = np.zeros((num_node, 1, 1, 2, 2), dtype=float)
    on_tensor[labels["X"], 0, 0, 0, 1] = 0.9
    on_tensor[labels["Y"], 0, 0, 0, 1] = 0.8
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame(
            {"name": ["A", "B", "C", "D"], "trait": [1, 1, 2, 2]}
        ),
        "fg_leaf_names": fg_leaf_names,
        "fg_ids": {"trait": np.array([labels["X"], labels["Y"]], dtype=np.int64)},
        "fg_stem_only": True,
        "scan_sister_stem_only": False,
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
    }
    return g, on_tensor, labels


def _toy_binary_unit_modes_context():
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("(((A:1,B:1)X:1,E:1)U:1,((C:1,D:1)Y:1,F:1)V:1)R;", format=1)
    )
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    num_node = max(labels.values()) + 1
    for node in tr.traverse():
        ete.set_prop(node, "SNdist", 1.0)
        ete.set_prop(node, "Ndist", 1.0)
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame(
            {"name": ["A", "B", "C", "D", "E", "F"], "trait": [1, 1, 1, 1, 0, 0]}
        ),
        "fg_stem_only": True,
    }
    g = foreground.get_foreground_ids(g=g, write=False)
    state_nsy = np.zeros((num_node, 1, 2), dtype=float)
    state_pep = np.zeros((num_node, 1, 2), dtype=float)
    for node_id in labels.values():
        _set_state(state_nsy, node_id, 0, 0)
        _set_state(state_pep, node_id, 0, 0)
    on_tensor = np.zeros((num_node, 1, 1, 2, 2), dtype=float)
    on_tensor[labels["A"], 0, 0, 0, 1] = 0.9
    on_tensor[labels["C"], 0, 0, 0, 1] = 0.8
    g.update(
        {
            "scan_sister_stem_only": False,
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
            "min_clade_bin_count": 1,
        }
    )
    return g, on_tensor, labels


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


def test_scan_unit_modes_split_binary_foreground_and_control_internal_branch_coverage():
    g, on_tensor, labels = _toy_binary_unit_modes_context()

    g["scan_unit_mode"] = "lineage"
    lineage_df, lineage_units = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)
    g["scan_unit_mode"] = "stem"
    stem_df, stem_units = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)
    g["scan_unit_mode"] = "clade"
    clade_df, clade_units = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    assert lineage_units.shape[0] == 1
    assert lineage_units.iloc[0]["unit_mode"] == "lineage"
    assert set(substitution_scan._parse_id_list(lineage_units.iloc[0]["stem_branch_ids"])) == {
        labels["X"],
        labels["Y"],
    }
    assert lineage_df.empty

    assert stem_units.shape[0] == 2
    assert set(stem_units["unit_mode"].tolist()) == {"stem"}
    assert {
        tuple(substitution_scan._parse_id_list(value).tolist())
        for value in stem_units["fg_branch_ids"]
    } == {(labels["X"],), (labels["Y"],)}
    assert stem_df.empty

    assert clade_units.shape[0] == 2
    assert set(clade_units["unit_mode"].tolist()) == {"clade"}
    clade_branch_sets = {
        frozenset(substitution_scan._parse_id_list(value).tolist())
        for value in clade_units["fg_branch_ids"]
    }
    assert clade_branch_sets == {
        frozenset([labels["X"], labels["A"], labels["B"]]),
        frozenset([labels["Y"], labels["C"], labels["D"]]),
    }
    assert clade_df.shape[0] == 1
    assert clade_df.iloc[0]["scan_unit_mode"] == "clade"
    assert clade_df.iloc[0]["support_unit_count"] == 2


@pytest.mark.parametrize(
    ("unit_mode", "expected_unit_count", "expected_branch_counts"),
    [
        ("lineage", 1, [2]),
        ("stem", 2, [1, 1]),
        ("clade", 2, [3, 3]),
    ],
)
def test_scan_permutation_context_uses_the_same_unit_mode_as_observed(
    monkeypatch,
    unit_mode,
    expected_unit_count,
    expected_branch_counts,
):
    g, _, labels = _toy_binary_unit_modes_context()
    g["scan_unit_mode"] = unit_mode
    trait_cache = foreground._get_trait_clade_permutation_cache(g=g, trait_name="trait")
    selected_flags = np.zeros_like(trait_cache["is_fg_stem"], dtype=bool)
    selected_flags[trait_cache["branch_id_to_index"][labels["X"]]] = True
    selected_flags[trait_cache["branch_id_to_index"][labels["Y"]]] = True
    monkeypatch.setattr(
        substitution_scan.foreground,
        "_randomize_foreground_stem_flags_from_plan",
        lambda **kwargs: selected_flags,
    )

    context = substitution_scan._build_permuted_trait_context(
        g=g,
        trait_name="trait",
        valid_branch_ids=trait_cache["branch_ids"],
        sample_original_foreground=True,
    )

    assert context["units"].shape[0] == expected_unit_count
    assert context["units"]["unit_mode"].unique().tolist() == [unit_mode]
    branch_counts = sorted(
        substitution_scan._parse_id_list(value).shape[0]
        for value in context["units"]["fg_branch_ids"]
    )
    assert branch_counts == sorted(expected_branch_counts)


def test_scan_zero_event_probability_does_not_count_as_support_at_zero_threshold():
    event_pp, branches = substitution_scan._support_for_unit(
        branch_event={},
        branch_ids=np.array([2, 3], dtype=np.int64),
        min_event_pp=0.0,
    )

    assert event_pp == 0.0
    assert branches == []


def test_scan_zero_threshold_still_requires_an_observed_event_in_each_support_unit():
    g, on_tensor = _toy_scan_context()
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in g["tree"].traverse()}
    on_tensor[labels["C"], 0, 0, 0, 1] = 0.0
    g["scan_min_event_pp"] = 0.0

    scan_df, units = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    assert units.shape[0] == 2
    assert scan_df.empty


def test_scan_3di_q_weighted_fallback_completes_without_codon_q_context(capsys):
    g, on_tensor = _toy_scan_context()
    g["nonsyn_recode"] = "3di20"
    g["scan_rate_exposure"] = "q_weighted"

    scan_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    assert scan_df.iloc[0]["scan_rate_exposure"] == "state_aware"
    assert "does not define 3Di-state transition rates" in capsys.readouterr().out


def test_scan_rate_target_uses_whole_foreground_clades_when_discovery_is_stem_only():
    g, on_tensor, labels = _toy_clade_scan_context()

    scan_df, units = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    row = scan_df.iloc[0]
    assert row["target_event_count"] == pytest.approx(1.7)
    assert row["target_raw_branch_length"] == pytest.approx(6.0)
    assert row["target_exposure_branch_length"] == pytest.approx(2.0)
    assert row["other_raw_branch_length"] == pytest.approx(4.0)
    expected_clade_ids = {
        labels["X"], labels["A"], labels["B"],
        labels["Y"], labels["C"], labels["D"],
    }
    observed_clade_ids = set()
    for value in units["fg_clade_branch_ids"].tolist():
        observed_clade_ids.update(int(v) for v in str(value).split(",") if v != "")
    assert observed_clade_ids == expected_clade_ids


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


def test_select_scan_plot_rows_keeps_one_best_candidate_per_site():
    scan_df = pd.DataFrame(
        {
            "target_class": ["fg", "fg", "fg"],
            "codon_site_alignment": [10, 10, 20],
            "support_unit_count": [2, 4, 3],
            "support_pp_sum": [1.8, 3.2, 2.7],
            "candidate_event_pp_sum": [1.8, 3.2, 2.7],
            "p_rate_enrichment_empirical_maxT": [0.2, 0.1, 0.3],
            "p_rate_enrichment": [0.02, 0.01, 0.03],
        }
    )

    out = substitution_scan._select_scan_plot_rows(scan_df)

    assert out["codon_site_alignment"].tolist() == [10, 20]
    assert out["support_unit_count"].tolist() == [4, 3]


def test_filter_scan_site_plot_candidates_uses_full_scan_maxt_pvalue():
    scan_df = pd.DataFrame(
        {
            "codon_site_alignment": [10, 20, 30],
            "p_rate_enrichment": [1e-6, 1e-3, 0.2],
            "p_rate_enrichment_empirical": [0.01, 0.03, 0.04],
            "p_rate_enrichment_empirical_maxT": [0.02, 0.05, 0.2],
        }
    )
    g = {
        "scan_site_plot_filter": "full_scan",
        "scan_site_plot_alpha": 0.05,
        "scan_pvalue_calibration": "full_scan",
    }

    out = substitution_scan.filter_scan_site_plot_candidates(scan_df=scan_df, g=g)

    assert out["codon_site_alignment"].tolist() == [10, 20]


def test_filter_scan_site_plot_candidates_rejects_full_scan_filter_without_full_scan_calibration():
    scan_df = pd.DataFrame(
        {
            "codon_site_alignment": [10],
            "p_rate_enrichment_empirical_maxT": [0.01],
        }
    )
    g = {
        "scan_site_plot_filter": "full_scan",
        "scan_site_plot_alpha": 0.05,
        "scan_pvalue_calibration": "candidate_fixed",
    }

    with pytest.raises(ValueError, match="requires --scan_pvalue_calibration full_scan"):
        substitution_scan.filter_scan_site_plot_candidates(scan_df=scan_df, g=g)


def test_validate_scan_configuration_rejects_zero_permutations_for_calibration():
    with pytest.raises(ValueError, match="should be > 0"):
        substitution_scan.validate_scan_configuration(
            {
                "scan_pvalue_calibration": "full_scan",
                "scan_n_permutations": 0,
                "scan_site_plot_filter": "all",
            }
        )


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

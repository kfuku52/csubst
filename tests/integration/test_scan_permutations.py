
import numpy as np
import pandas as pd
import pytest

from csubst import ete
from csubst import foreground
from csubst import substitution_scan
from csubst import tree


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


def test_scan_permutation_retries_contexts_that_lose_analyzable_units(monkeypatch):
    g, _ = _toy_scan_context()
    calls = []

    def retry_then_succeed(*args, **kwargs):
        calls.append(bool(kwargs["sample_original_foreground"]))
        if len(calls) < 3:
            raise substitution_scan._RetryableScanPermutationError("lost unit")
        return {"units": pd.DataFrame()}

    monkeypatch.setattr(substitution_scan, "_build_permuted_scan_context", retry_then_succeed)

    out = substitution_scan._build_permuted_context_with_seed(
        g=g,
        trait_names=["trait"],
        valid_branch_ids=np.arange(g["state_nsy"].shape[0], dtype=np.int64),
        permutation_index=1,
    )

    assert out["units"].empty
    assert calls == [False, False, False]


def test_permuted_trait_context_rejects_a_selected_stem_without_analyzable_state(monkeypatch):
    g, _ = _toy_scan_context()
    trait_cache = foreground._get_trait_clade_permutation_cache(g=g, trait_name="trait")
    branch_id_to_index = trait_cache["branch_id_to_index"]
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in g["tree"].traverse()}
    selected_flags = np.zeros_like(trait_cache["is_fg_stem"], dtype=bool)
    selected_flags[branch_id_to_index[labels["A"]]] = True
    selected_flags[branch_id_to_index[labels["B"]]] = True
    monkeypatch.setattr(
        substitution_scan.foreground,
        "_randomize_foreground_stem_flags_from_plan",
        lambda **kwargs: selected_flags,
    )
    valid_branch_ids = np.array(
        [branch_id for branch_id in trait_cache["branch_ids"] if int(branch_id) != labels["B"]],
        dtype=np.int64,
    )

    with pytest.raises(substitution_scan._RetryableScanPermutationError, match="retained 1 of 2"):
        substitution_scan._build_permuted_trait_context(
            g=g,
            trait_name="trait",
            valid_branch_ids=valid_branch_ids,
            sample_original_foreground=False,
        )


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


@pytest.mark.slow
@pytest.mark.process
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

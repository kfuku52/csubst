import itertools
import json

import numpy as np
import pandas as pd
import pytest

from csubst import ete, main_scan, substitution_scan
from scan_fixtures import make_scan_context


def _assign_groups(g, groups):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in g["tree"].traverse()}
    names = [name for group in groups for name in group]
    g["fg_df"] = pd.DataFrame({"name": names, "trait": [i + 1 for i, group in enumerate(groups) for _ in group]})
    g["fg_leaf_names"] = {"trait": [list(group) for group in groups]}
    g["fg_ids"] = {"trait": np.array([labels[name] for name in names])}
    for node in g["tree"].traverse():
        tips = set(ete.get_leaf_names(node))
        flags = [tips.issubset(set(group)) for group in groups]
        ete.set_prop(node, "is_fg_trait", any(flags))
        for i, flag in enumerate(flags, 1):
            ete.set_prop(node, "is_lineage_fg_trait_" + str(i), flag)
    return labels


@pytest.mark.parametrize("mode", ["candidate_fixed", "full_scan"])
@pytest.mark.parametrize("bin_count", [1, 10])
def test_six_assignment_null_has_correct_exact_tail(mode, bin_count):
    rejections = 0
    for pair in itertools.combinations("ABCD", 2):
        g, tensor = make_scan_context()
        _assign_groups(g, [[name] for name in pair])
        g.update(scan_pvalue_calibration=mode, scan_n_permutations=99, min_clade_bin_count=bin_count)
        scan, _ = substitution_scan.scan_substitutions(g, tensor)
        if scan.empty:
            assert g["scan_calibration_diagnostics"]["status"] == "no_observed_candidates"
            continue
        column = "p_rate_enrichment_empirical_maxT" if mode == "full_scan" else "p_rate_enrichment_empirical"
        assert pair == ("A", "C")
        assert scan.iloc[0][column] == pytest.approx(1 / 6)
        assert scan.iloc[0]["scan_permutation_sampling"] == "exact"
        assert scan.iloc[0]["scan_permutation_success_count"] == 6
        assert scan.iloc[0]["scan_permutation_unique_count"] == 6
        assert scan.iloc[0]["scan_pvalue_resolution"] == pytest.approx(1 / 6)
        assert g["scan_calibration_diagnostics"]["original_configuration_count"] == 1
        rejections += int(scan.iloc[0][column] <= 0.05)
    # Old foreground exclusion rejected 1/6 assignments at nominal alpha=.05.
    assert rejections == 0


def test_full_scan_reselects_competing_candidates():
    results = {}
    for mode in ["candidate_fixed", "full_scan"]:
        g, tensor = make_scan_context()
        labels = _assign_groups(g, [["A"], ["C"]])
        for key in ["state_nsy", "state_pep"]:
            g[key] = np.repeat(g[key], 2, axis=1)
            g[key][:, 1, :] = [1, 0]
            g[key][[labels["B"], labels["D"]], 1, :] = [0, 1]
        tensor = np.repeat(tensor, 2, axis=1)
        tensor[:, 1] = 0
        tensor[labels["B"], 1, 0, 0, 1] = 0.9
        tensor[labels["D"], 1, 0, 0, 1] = 0.8
        g["iqtree_rate_values"] = np.array([0.25, 0.25])
        g.update(scan_pvalue_calibration=mode, scan_n_permutations=99)
        results[mode], _ = substitution_scan.scan_substitutions(g, tensor)
    assert results["candidate_fixed"].iloc[0]["p_rate_enrichment_empirical"] == pytest.approx(1 / 6)
    assert results["full_scan"].iloc[0]["p_rate_enrichment_empirical"] == pytest.approx(1 / 6)
    assert results["full_scan"].iloc[0]["p_rate_enrichment_empirical_maxT"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("mode", ["candidate_fixed", "full_scan"])
def test_undefined_candidate_invalidates_calibration(mode):
    g, tensor = make_scan_context()
    if mode == "full_scan":
        labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in g["tree"].traverse()}
        for key in ["state_nsy", "state_pep"]:
            g[key] = np.repeat(g[key], 2, axis=1)
            g[key][:, 1, :] = [1, 0]
            g[key][[labels["B"], labels["D"]], 1, :] = [0, 1]
        tensor = np.repeat(tensor, 2, axis=1)
        tensor[:, 1] = 0
        tensor[labels["B"], 1, 0, 0, 1] = 0.9
        tensor[labels["D"], 1, 0, 0, 1] = 0.8
        g["iqtree_rate_values"] = np.array([0.25, 0.25])
    for node in g["tree"].traverse():
        if node.name in ["B", "D"]:
            node.dist = 0.0
    g.update(scan_pvalue_calibration=mode, scan_n_permutations=99)
    scan, _ = substitution_scan.scan_substitutions(g, tensor)
    row = scan.iloc[0]
    assert row["scan_calibration_status"] == "unavailable_failed_trials"
    assert row["scan_permutation_failure_count"] == 1
    assert row["scan_permutation_success_count"] == 5
    assert np.isnan(row["p_rate_enrichment_empirical"])
    assert np.isnan(row["q_rate_enrichment_empirical"])
    assert np.isnan(row["p_rate_enrichment_empirical_maxT"])
    trials = g["scan_calibration_diagnostics"]["trials"]
    failed = [trial for trial in trials if trial["status"] == "failed"]
    assert failed[0]["candidate_count"] == 1
    assert failed[0]["finite_pvalue_count"] == 0
    assert failed[0]["configuration_id"] is not None


def test_one_exception_does_not_condition_the_null_on_success(monkeypatch):
    g, tensor = make_scan_context()
    g.update(scan_pvalue_calibration="full_scan", scan_n_permutations=99)
    original = substitution_scan._build_permuted_context_with_seed

    def fail_one(*args, **kwargs):
        if kwargs["permutation_index"] == 3:
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(substitution_scan, "_build_permuted_context_with_seed", fail_one)
    scan, _ = substitution_scan.scan_substitutions(g, tensor)
    assert scan.iloc[0]["scan_permutation_success_count"] == 5
    assert scan.iloc[0]["scan_permutation_failure_count"] == 1
    assert np.isnan(scan.iloc[0]["p_rate_enrichment_empirical_maxT"])
    assert len(g["scan_calibration_diagnostics"]["trials"]) == 6


@pytest.mark.parametrize("unit_mode", ["stem", "clade", "lineage"])
@pytest.mark.parametrize("other_scope", ["all", "sister"])
def test_original_configuration_restores_the_statistic(unit_mode, other_scope):
    g, tensor = make_scan_context()
    _assign_groups(g, [["A", "D"], ["C"]])
    g.update(scan_unit_mode=unit_mode, scan_other_scope=other_scope)
    static = substitution_scan._build_scan_static_context(g, tensor)
    observed, _ = substitution_scan._scan_substitutions_core(g, tensor, scan_static=static)
    plan = substitution_scan._get_scan_trait_plan(g, "trait", static["valid_branch_ids"])
    context = substitution_scan._build_permuted_scan_context(
        g, ["trait"], static["valid_branch_ids"], configurations={"trait": plan["sampler"].observed},
    )
    restored, _ = substitution_scan._scan_substitutions_core(g, tensor, scan_context=context, scan_static=static)
    cols = ["site", "support_unit_count", "target_event_count", "other_event_count",
            "target_exposure_branch_length", "other_exposure_branch_length", "p_rate_enrichment"]
    pd.testing.assert_frame_equal(observed[cols], restored[cols], check_exact=True)


def test_empty_scan_still_writes_independent_diagnostics(tmp_path, monkeypatch):
    g, tensor = make_scan_context()
    g.update(scan_pvalue_calibration="full_scan", scan_n_permutations=99, scan_min_support="3")
    scan, _ = substitution_scan.scan_substitutions(g, tensor)
    assert scan.empty
    monkeypatch.setattr(main_scan.runtime, "output_path", lambda g, name: str(tmp_path / name))
    path = main_scan._write_scan_calibration(g)
    diagnostic = json.loads(open(path, encoding="utf-8").read())
    assert diagnostic["status"] == "no_observed_candidates"
    assert diagnostic["observed"]["configuration_id"]
    assert diagnostic["plan"]["valid_space_size"] == 6
    assert diagnostic["trials"] == []


@pytest.mark.parametrize("setting,value", [
    ("scan_permutation_sample_original", False), ("scan_permutation_retry_sample_original", True),
])
def test_legacy_sampling_laws_are_rejected(setting, value):
    g, tensor = make_scan_context()
    g.update(scan_pvalue_calibration="full_scan", scan_n_permutations=10)
    g[setting] = value
    with pytest.raises(ValueError, match=setting):
        substitution_scan.scan_substitutions(g, tensor)


def test_multiple_traits_require_a_joint_null():
    g, tensor = make_scan_context()
    g["fg_df"]["trait2"] = [1, 2]
    g.update(scan_pvalue_calibration="full_scan", scan_n_permutations=10)
    with pytest.raises(ValueError, match="joint assignment null"):
        substitution_scan.scan_substitutions(g, tensor)


def test_nonfinite_values_are_not_treated_as_nonextreme():
    assert np.isnan(substitution_scan._empirical_p_from_values(0.01, [np.nan] * 99, 99))
    assert substitution_scan._empirical_p_from_values(0.01, [0.01] * 99, 99) == 1.0
    assert substitution_scan._empirical_p_from_values(0.01, [0.01] * 99, 99, exact=True) == 1.0

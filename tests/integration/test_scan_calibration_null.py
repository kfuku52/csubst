import itertools
import json

import numpy as np
import pandas as pd
import pytest

from csubst import ete, foreground, main_scan, substitution_scan, trait_ctmc
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


def test_trait_ctmc_rediscovery_uses_fixed_data_and_retains_empty_trials(tmp_path, monkeypatch):
    newick = "(((A:0.1,B:0.1)AB:1.9,(C:1.9,D:1.9)CD:0.1)X:1,((E:1,F:1)EF:1,(G:1,H:1)GH:1)Y:1)R;"
    g, tensor = make_scan_context(newick, event_branches=("AB", "E"))
    states = tmp_path / "traits.tsv"
    states.write_text("name\ttrait\nA\t1\nB\t1\nE\t1\n")
    time_tree = tmp_path / "time.nwk"
    time_tree.write_text(newick)
    g.update(foreground=str(states), fg_format=2, fg_trait_tree=str(time_tree),
             fg_trait_match_tip_count=True, scan_unit_mode="stem",
             scan_pvalue_calibration="full_scan", scan_n_permutations=40)
    foreground.get_foreground_branch(g, write=False)
    assert len(g["fg_leaf_names"]["trait"]) == 2
    scan, units = substitution_scan.scan_substitutions(g, tensor)
    assert len(units) == 2
    diagnostic = g["scan_calibration_diagnostics"]
    assert diagnostic["status"] == "conditional_trait_model"
    assert diagnostic["sampling"] == "monte_carlo"
    assert diagnostic["plan"]["conditioned_origin_count"] == 2
    assert diagnostic["plan"]["condition_on_positive_tip_count"] is True
    assert diagnostic["plan"]["gain_rate"] == diagnostic["plan"]["loss_rate"]
    assert diagnostic["success_count"] == 40
    assert diagnostic["unique_configuration_count"] > 1
    assert any(trial["candidate_count"] == 0 for trial in diagnostic["trials"])
    max_exceedances = sum(trial["max_score"] is not None and
                          trial["max_score"] >= diagnostic["observed"]["max_score"]
                          for trial in diagnostic["trials"])
    assert scan.iloc[0]["p_rate_enrichment_empirical_maxT"] == pytest.approx((1 + max_exceedances) / 41)
    again, _ = substitution_scan.scan_substitutions(g, tensor)
    pd.testing.assert_frame_equal(scan, again)
    assert g["scan_calibration_diagnostics"] == diagnostic

    def cannot_sample(*args):
        raise trait_ctmc.SamplingError()

    monkeypatch.setattr(trait_ctmc, "sample_stems", cannot_sample)
    scan, _ = substitution_scan.scan_substitutions(g, tensor)
    assert scan.iloc[0]["scan_calibration_status"] == "unavailable_failed_trials"
    assert np.isnan(scan.iloc[0]["p_rate_enrichment_empirical_maxT"])
    diagnostic = g["scan_calibration_diagnostics"]
    assert diagnostic["failure_count"] == 40
    assert diagnostic["sampling_attempts"] == 40 * trait_ctmc.MAX_PROPOSALS


@pytest.mark.parametrize("mode", ["candidate_fixed", "full_scan"])
def test_zero_exposure_is_a_successful_no_test_configuration(mode):
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
    assert row["scan_calibration_status"] == "conditional_assignment"
    assert row["scan_permutation_failure_count"] == 0
    assert row["scan_permutation_success_count"] == 6
    assert np.isfinite(row["p_rate_enrichment_empirical"])
    assert np.isfinite(row["q_rate_enrichment_empirical_by_trait_match"])
    trials = g["scan_calibration_diagnostics"]["trials"]
    no_tests = [trial for trial in trials if trial["candidate_count"] == 1 and trial["testable_candidate_count"] == 0]
    assert len(no_tests) == 1
    assert no_tests[0]["status"] == "success"
    assert no_tests[0]["max_score"] is None
    assert no_tests[0]["configuration_id"] is not None


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


@pytest.mark.parametrize("mode", ["candidate_fixed", "full_scan"])
def test_invalid_null_score_cannot_be_reclassified_as_no_test(monkeypatch, mode):
    g, tensor = make_scan_context()
    g.update(scan_pvalue_calibration=mode, scan_n_permutations=99)
    injected = []
    if mode == "candidate_fixed":
        original = substitution_scan._candidate_fixed_permutation_scores

        def invalid_fixed(*args, **kwargs):
            values = original(*args, **kwargs)
            if values and not injected:
                values[next(iter(values))] = np.nan
                injected.append(True)
            return values

        monkeypatch.setattr(substitution_scan, "_candidate_fixed_permutation_scores", invalid_fixed)
    else:
        original = substitution_scan._scan_substitutions_core

        def invalid_discovery(*args, **kwargs):
            frame, units = original(*args, **kwargs)
            if kwargs.get("scan_context") is not None and not frame.empty and not injected:
                frame = frame.copy()
                frame.loc[frame.index[0], ["score_rate_enrichment", "target_event_count"]] = np.nan
                injected.append(True)
            return frame, units

        monkeypatch.setattr(substitution_scan, "_scan_substitutions_core", invalid_discovery)
    frame, _ = substitution_scan.scan_substitutions(g, tensor)
    assert injected == [True]
    row = frame.iloc[0]
    assert row["scan_permutation_failure_count"] == 1
    assert row["scan_permutation_success_count"] == 5
    assert row["scan_calibration_status"] == "unavailable_failed_trials"
    assert row["scan_inference_status"] == "failed_null_replicates"
    assert frame[["p_rate_enrichment_empirical", "p_rate_enrichment_empirical_maxT",
                  "q_rate_enrichment_empirical_by_trait_match"]].isna().all().all()


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
            "target_exposure_branch_length", "other_exposure_branch_length", "score_rate_enrichment"]
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
    with pytest.raises(ValueError, match="undefined"):
        substitution_scan._empirical_p_from_scores(2., [np.nan] * 99, 99)
    assert substitution_scan._empirical_p_from_scores(0.01, [0.01] * 99, 99) == 1.0
    assert substitution_scan._empirical_p_from_scores(0.01, [0.01] * 99, 99, exact=True) == 1.0

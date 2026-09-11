
import pandas as pd
import numpy as np
import pytest

from csubst import ete
from csubst import foreground
from csubst import substitution_scan
from scan_fixtures import make_scan_context as _toy_scan_context


@pytest.mark.parametrize("calibration", ["candidate_fixed", "full_scan"])
def test_endpoint_scan_dense_sparse_and_parallel_permutations_agree(calibration):
    from csubst import substitution, site_storage

    g, on_tensor = _toy_scan_context()
    g.update(scan_rate_exposure="endpoint", state_cdn=g["state_nsy"].copy(),
             instantaneous_codon_rate_matrix=np.array([[-1., 1.], [1., -1.]]),
             equilibrium_frequency=np.array([.5, .5]), substitution_model="GY",
             nonsynonymous_indices={"A": [0], "K": [1]},
             iqtree_rate_values=np.ones(1), scan_n_permutations=4,
             scan_pvalue_calibration=calibration, threads=1)
    dense, _ = substitution_scan.scan_substitutions(g=dict(g), ON_tensor=on_tensor)
    sparse, _ = substitution_scan.scan_substitutions(
        g=dict(g, threads=2), ON_tensor=substitution.dense_to_sparse_sub_tensor(on_tensor))
    cols = [c for c in dense if c not in ["scan_permutation_n_jobs", "scan_permutation_backend"]]
    pd.testing.assert_frame_equal(dense[cols], sparse[cols])
    stored = site_storage.SiteEventTensor(on_tensor.shape)
    stored.write_block(0, np.moveaxis(on_tensor, 1, 0))
    stored.seal()
    disk, _ = substitution_scan.scan_substitutions(g=dict(g, threads=2), ON_tensor=stored)
    pd.testing.assert_frame_equal(dense[cols], disk[cols])
    assert dense.iloc[0]["scan_permutation_success_count"] == 4
    assert dense.iloc[0]["scan_exposure_units"] == "expected_endpoint_events"


@pytest.mark.parametrize("calibration", ["candidate_fixed", "full_scan"])
def test_endpoint_undefined_permutation_is_not_treated_as_no_candidates(monkeypatch, calibration):
    monkeypatch.setattr(substitution_scan, "_build_permuted_context_with_seed", lambda **kw: dict(configuration_id="a", configuration=[], sampling_attempts=1))
    monkeypatch.setattr(substitution_scan, "_candidate_fixed_permutation_scores", lambda **kw: {"x": np.nan})
    monkeypatch.setattr(substitution_scan, "_scan_substitutions_core", lambda **kw: (pd.DataFrame({"score_rate_enrichment": [np.nan]}), None))
    monkeypatch.setattr(substitution_scan, "_scan_row_key", lambda row: "x")
    result = substitution_scan._run_scan_permutation(
        permutation_index=0, g={}, observed_df=None, observed_keys={"x"}, calibration=calibration,
        trait_names=[], branch_meta=None, valid_branch_ids=[], ON_tensor=None,
        rate_ON_tensor=None, scan_static={"rate_exposure": "endpoint"})
    assert not result["success"]
    assert "undefined" in result["failure_reason"].lower()


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


def test_scan_permutation_does_not_retry_a_failed_context(monkeypatch):
    g, _ = _toy_scan_context()
    calls = []

    def fail_context(*args, **kwargs):
        calls.append(kwargs["sample_original_foreground"])
        raise ValueError("lost unit")

    monkeypatch.setattr(substitution_scan, "_build_permuted_scan_context", fail_context)
    with pytest.raises(ValueError, match="lost unit"):
        substitution_scan._build_permuted_context_with_seed(
            g=g, trait_names=["trait"],
            valid_branch_ids=np.arange(g["state_nsy"].shape[0]), permutation_index=1,
        )
    assert calls == [True]


def test_scan_eligibility_is_fixed_before_sampling():
    g, _ = _toy_scan_context()
    g["scan_unit_mode"] = "stem"
    trait_cache = foreground._get_trait_clade_permutation_cache(g=g, trait_name="trait")
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in g["tree"].traverse()}
    valid_branch_ids = np.array([bid for bid in trait_cache["branch_ids"] if bid != labels["B"]])
    plan = substitution_scan._get_scan_trait_plan(g, "trait", valid_branch_ids)
    excluded = trait_cache["branch_id_to_index"][labels["B"]]
    assert not plan["eligible"][excluded]
    assert all(excluded not in group for config in plan["sampler"].configurations for group in config)
    assert plan["sampler"].observed in plan["sampler"].configurations

    invalid_observed_ids = np.array([bid for bid in valid_branch_ids if bid != labels["A"]])
    with pytest.raises(ValueError, match="observed foreground component"):
        substitution_scan._get_scan_trait_plan(g, "trait", invalid_observed_ids)


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


def test_empirical_pvalue_rejects_an_inconsistent_denominator():
    with pytest.raises(ValueError, match="exactly one null statistic"):
        substitution_scan._empirical_p_from_scores(
            score_obs=2., values=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06], denominator_count=4,
        )


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

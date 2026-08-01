
import numpy as np
import pandas as pd
import pytest

from csubst import substitution_sparse
from csubst import substitution_scan


def test_parse_scan_support_threshold_distinguishes_integer_counts_and_exact_fractions():
    assert substitution_scan.parse_scan_support_threshold("1", total_units=10) == 1
    assert substitution_scan.parse_scan_support_threshold("1.0", total_units=10) == 10
    assert substitution_scan.parse_scan_support_threshold("0.07", total_units=100) == 7


def test_rank_quantiles_assigns_average_rank_to_ties():
    out = substitution_scan._rank_quantiles(np.array([1.0, 2.0, 2.0, 4.0]))
    assert out.tolist() == pytest.approx([0.25, 0.625, 0.625, 1.0])


def test_3di_q_weighted_exposure_resolves_to_state_aware(capsys):
    g = {"scan_rate_exposure": "q_weighted", "nonsyn_recode": "3di20"}

    resolved = substitution_scan.resolve_scan_rate_exposure(g)

    assert resolved == "state_aware"
    assert "3Di" in capsys.readouterr().out


def test_normalize_scan_matches_all_expands_to_nine_classes():
    assert substitution_scan.normalize_scan_matches("all") == list(substitution_scan.SCAN_MATCHES)
    assert len(substitution_scan.normalize_scan_matches("all")) == 9
    assert substitution_scan.normalize_scan_matches("any2spe,spe2spe,any2spe") == [
        "any2spe",
        "spe2spe",
    ]


def test_normalize_scan_unit_mode_defaults_to_clade_and_rejects_unknown_values():
    assert substitution_scan.normalize_scan_unit_mode(None) == "clade"
    assert substitution_scan.normalize_scan_unit_mode(" STEM ") == "stem"
    assert substitution_scan.normalize_scan_unit_mode("clade") == "clade"
    with pytest.raises(ValueError, match="scan_unit_mode"):
        substitution_scan.normalize_scan_unit_mode("branch")


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


def test_q_weighted_opportunity_uses_parent_codon_posterior_not_equal_codon_weighting():
    branch_meta = pd.DataFrame(
        {
            "branch_id": [1, 2],
            "parent_id": [0, 1],
            "raw_length": [1.0, 1.0],
            "sn_rescaled_length": [1.0, 1.0],
            "n_rescaled_length": [1.0, 1.0],
        }
    )
    state_cdn = np.zeros((3, 1, 3), dtype=float)
    state_cdn[0, 0, 0] = 1.0
    state_cdn[1, 0, 1] = 1.0
    q_codon = np.array(
        [
            [-1.0, 0.0, 1.0],
            [0.0, -9.0, 9.0],
            [1.0, 1.0, -2.0],
        ]
    )
    codon_state_ids = np.array([0, 0, 1], dtype=np.int64)

    opportunity = substitution_scan._q_weighted_opportunity(
        branch_meta=branch_meta,
        state_nsy=np.zeros((3, 1, 2), dtype=float),
        state_cdn=state_cdn,
        site=0,
        from_ids=np.array([0], dtype=np.int64),
        to_ids=np.array([1], dtype=np.int64),
        q_matrix=None,
        codon_q_matrix=q_codon,
        codon_state_ids=codon_state_ids,
        rate_length="raw",
    )

    assert opportunity.tolist() == pytest.approx([1.0, 9.0])


def test_q_weighted_codon_opportunity_normalizes_only_nonsynonymous_rates_for_n_rescaled_length():
    branch_meta = pd.DataFrame(
        {
            "branch_id": [1],
            "parent_id": [0],
            "raw_length": [1.0],
            "sn_rescaled_length": [1.0],
            "n_rescaled_length": [1.0],
        }
    )
    state_cdn = np.zeros((2, 1, 4), dtype=float)
    state_cdn[0, 0, 0] = 1.0
    q_codon = np.array(
        [
            [-9.0, 5.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    codon_state_ids = np.array([0, 0, 1, 2], dtype=np.int64)

    opportunity = substitution_scan._q_weighted_opportunity(
        branch_meta=branch_meta,
        state_nsy=np.zeros((2, 1, 3), dtype=float),
        state_cdn=state_cdn,
        site=0,
        from_ids=np.array([0], dtype=np.int64),
        to_ids=np.array([1], dtype=np.int64),
        q_matrix=None,
        codon_q_matrix=q_codon,
        codon_state_ids=codon_state_ids,
        rate_length="n_rescaled",
    )

    assert opportunity.tolist() == pytest.approx([0.25])


import csv
import io

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


@pytest.mark.parametrize('size', [0, 1, 3, 20, 61])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64])
def test_candidate_distribution_retains_pandas_ties_and_display(size, dtype):
    rng = np.random.default_rng(84)
    orders = np.array(['state{}'.format(i) for i in range(size)])
    indices = rng.permutation(size)
    masses = rng.integers(0, 4, size=size).astype(dtype)
    if dtype != np.int64:
        masses[::5] = np.nan
    values = list(zip(indices, masses))
    expected = ','.join('{}:{:.6g}'.format(orders[int(index)], float(value))
                        for index, value in pd.Series(masses, index=indices).sort_values(
                            ascending=False).items())
    assert substitution_scan._candidate_distribution_label(values, orders) == expected


@pytest.mark.parametrize('ids,masses,expected', [
    ([2, 1], [.25, .75], (.75, .25)),
    ([2, 1], [.25, np.nan], (0., .25)),
    ([1, np.nan, 2], [.5, .9, .25], (.5, .25)),
    ([1, 1, 1, 2], [1e16, 1., 1., .25], (1e16 + 2., .25)),
    ([], [], (0., 0.)),
])
@pytest.mark.parametrize('prepared', [False, True])
def test_rate_summary_retains_branch_aggregation_and_missing_values(ids, masses, expected, prepared):
    events = pd.DataFrame({'branch_id': ids, 'event_pp': masses})
    original = events.copy(deep=True)
    meta = pd.DataFrame({'branch_id': [1, 2], 'parent_id': [0, 0], 'raw_length': [.1, .2],
                         'sn_rescaled_length': [.1, .2], 'n_rescaled_length': [.1, .2]})
    branches = substitution_scan._ScanRateMetadata(meta).prepare([1], None) if prepared else None
    result = substitution_scan._rate_summary(
        events, meta, np.ones((3, 1, 2)), 0, [0], [1], [1], 'raw', 'raw_branch_length',
        prepared_branch=branches)
    assert (result['target_event_count'], result['other_event_count']) == expected
    pd.testing.assert_frame_equal(events, original, check_exact=True)


def test_prepared_rates_keep_assignment_masks_and_length_choices_independent():
    meta = pd.DataFrame({'branch_id': [7, 1, 3], 'parent_id': [0, 0, 0],
                         'raw_length': [.5, .25, .25], 'sn_rescaled_length': [1., .5, .5],
                         'n_rescaled_length': [2., 1., 1.]})
    events = pd.DataFrame({'branch_id': [1, 3, 7], 'event_pp': [.75, .125, .25]})
    data = substitution_scan._ScanRateMetadata(meta)
    for target, other, length, expected in (
        ([1, 1, 99], [1, 3, 7], 'raw', (.75, .375, .25, .75)),
        ([7], None, 'sn_rescaled', (.25, .875, 1., 1.)),
        ([1], [3], 'n_rescaled', (.75, .125, 1., 1.)),
    ):
        result = substitution_scan._rate_summary(
            events, meta, np.ones((8, 1, 2)), 0, [0], [1], target, length, 'raw_branch_length',
            other_branch_ids=other, prepared_branch=data.prepare(target, other))
        assert tuple(result[key] for key in ('target_event_count', 'other_event_count',
                                            'target_exposure', 'other_exposure')) == expected


def test_rate_metadata_is_lazy_so_event_column_error_precedes_length_error():
    meta = pd.DataFrame({'branch_id': [1]})
    prepared = substitution_scan._ScanRateMetadata(meta).prepare([1], None)
    with pytest.raises(KeyError, match='event_pp'):
        substitution_scan._rate_summary(
            pd.DataFrame({'branch_id': [1]}), meta, np.ones((2, 1, 2)), 0, [0], [1], [1],
            'raw', 'raw_branch_length', prepared_branch=prepared)


@pytest.mark.parametrize('length', [[], 'unknown'])
def test_rate_metadata_retains_invalid_length_error(length):
    meta = pd.DataFrame({'branch_id': [1]})
    with pytest.raises(ValueError, match='--scan_rate_length'):
        substitution_scan._rate_summary(
            pd.DataFrame({'branch_id': [1], 'event_pp': [.5]}), meta, np.ones((2, 1, 2)),
            0, [0], [1], [1], length, 'raw_branch_length')


@pytest.mark.parametrize('prepared', [False, True])
def test_unit_support_counts_foreground_lineages_once_and_rejects_non_foreground_targets(prepared):
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
            "lineage_value": [7, 11],
            "fg_branch_ids": ["1", "2"],
        }
    )

    fg = substitution_scan._summarize_unit_support(
        candidate_events=candidate_events,
        units_df=units,
        target_class="fg",
        min_event_pp=0.5,
        unit_plan=substitution_scan._scan_unit_support_plan(units) if prepared else None,
    )

    assert fg["support_unit_ids"] == "1"
    assert fg["support_branch_ids"] == "1"
    assert fg["lineage_total"] == 2
    assert fg["support_lineage_count"] == 1
    assert fg["support_lineage_ids"] == "7"
    assert fg["support_lineage_fraction"] == 0.5
    with pytest.raises(ValueError, match="Only foreground target class"):
        substitution_scan._summarize_unit_support(
            candidate_events=candidate_events,
            units_df=units,
            target_class="mg",
            min_event_pp=0.5,
        )


def test_candidate_event_mask_preserves_row_order_and_excludes_diagonal_mass():
    events = pd.DataFrame({
        'site': ['2', '0', '2', '2', '2', '2'],
        'from_state_id': ['0', '0', '1', '2', '0', '0'],
        'to_state_id': ['1', '1', '1', '1', '2', '1'],
    }, index=[8, 4, 3, 3, 1, 0])
    mask = substitution_scan._event_mask_for_candidate(events, 2, [0, 1], [1])
    assert mask.tolist() == [True, False, False, False, False, True]
    assert not substitution_scan._event_mask_for_candidate(events, 3, [0], [1]).any()
    assert not substitution_scan._event_mask_for_candidate(events, 2, [], [1]).any()
    assert substitution_scan._event_mask_for_candidate(events.iloc[:0], 2, [0], [1]).shape == (0,)


@pytest.mark.parametrize("third_branch_pp, expected_count, expected_ids", [
    (0.49, 1, "7"),
    (0.5, 2, "7,11"),
])
def test_lineage_support_groups_disconnected_units_and_uses_analyzable_denominator(
    third_branch_pp, expected_count, expected_ids,
):
    events = pd.DataFrame({"branch_id": [1, 2, 3], "event_pp": [0.9, 0.8, third_branch_pp]})
    units = pd.DataFrame({
        "unit_id": [1, 2, 3, 4],
        "lineage_value": [7, 7, 11, 13],
        "fg_branch_ids": ["1", "2", "3", ""],
    })

    support = substitution_scan._summarize_unit_support(events, units, "fg", 0.5)

    assert support["support_unit_count"] == expected_count + 1
    assert support["unit_total"] == 4
    assert support["lineage_total"] == 2  # ID 13 has no analyzable candidate branch.
    assert support["support_lineage_count"] == expected_count
    assert support["support_lineage_fraction"] == expected_count / 2
    assert support["support_lineage_ids"] == expected_ids


@pytest.mark.parametrize("no_units", [False, True])
def test_lineage_support_with_no_analyzable_branches_is_undefined_fraction(no_units):
    events = pd.DataFrame({"branch_id": [1], "event_pp": [0.9]})
    units = pd.DataFrame({"unit_id": [1], "lineage_value": [7], "fg_branch_ids": [""]})
    if no_units:
        units = units.iloc[:0]

    support = substitution_scan._summarize_unit_support(events, units, "fg", 0.5)

    assert support["lineage_total"] == 0
    assert support["support_lineage_count"] == 0
    assert np.isnan(support["support_lineage_fraction"])
    assert support["support_lineage_ids"] == ""


def test_lineage_support_id_list_round_trips_delimited_identifiers():
    identifiers = ['origin,a', 'origin"b', 'origin\nc', 'origin\rd', 'origin\r\ne']
    branch_ids = list(range(1, len(identifiers) + 1))
    units = pd.DataFrame({
        "unit_id": branch_ids, "lineage_value": identifiers, "fg_branch_ids": list(map(str, branch_ids)),
    })
    events = pd.DataFrame({"branch_id": branch_ids, "event_pp": [0.9] * len(identifiers)})
    support = substitution_scan._summarize_unit_support(events, units, "fg", 0.5)
    assert support["support_lineage_count"] == len(identifiers)
    assert next(csv.reader(io.StringIO(support["support_lineage_ids"]))) == identifiers
    serialized = pd.DataFrame([support]).to_csv(sep="\t", index=False)
    saved = pd.read_csv(io.StringIO(serialized), sep="\t").iloc[0]
    assert saved["support_lineage_ids"] == support["support_lineage_ids"]


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

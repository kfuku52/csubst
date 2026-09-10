import itertools
from collections import Counter

import numpy as np
import pytest

from csubst import scan_permutation


def _four_clade_plan():
    # ((A,B)X,((C,D)Y,(E,F)Z)W): W contains Y and Z, X is disjoint.
    descendants = [{0}, {1}, {2}, {1, 2, 3}]
    return scan_permutation.build_plan(
        bin_array=[0, 0, 0, 0], eligible=[True] * 4,
        observed=((0, 1),), descendants=descendants,
    )


def test_all_valid_configurations_are_enumerated_once():
    plan = _four_clade_plan()
    assert plan.proposal_count == 6
    assert set(plan.configurations) == {((0, 1),), ((0, 2),), ((0, 3),), ((1, 2),)}
    assert len(plan.configurations) == 4
    assert plan.observed in plan.configurations


def test_whole_configuration_rejection_is_uniform(monkeypatch):
    monkeypatch.setattr(scan_permutation, "ENUMERATION_PROPOSAL_LIMIT", 0)
    plan = _four_clade_plan()
    assert plan.configurations is None
    rng = np.random.default_rng(617)
    counts = Counter()
    attempts = []
    for _ in range(12000):
        configuration, count = scan_permutation.sample_configuration(plan, rng)
        counts[configuration] += 1
        attempts.append(count)
    assert set(counts) == {((0, 1),), ((0, 2),), ((0, 3),), ((1, 2),)}
    # The old sequential sampler gave WX probability 1/3 and XY/XZ 5/24.
    assert all(abs(count / 12000 - 0.25) < 0.02 for count in counts.values())
    assert max(attempts) > 1


def test_lineage_allocations_match_independent_exhaustive_assignments():
    bins = np.array([0, 0, 0, 1, 1])
    plan = scan_permutation.build_plan(
        bins, [True] * 5, ((0, 3), (1,)), [{i} for i in range(5)],
    )
    expected = set()
    for lineage_1 in itertools.combinations(range(5), 2):
        if sorted(bins[list(lineage_1)]) != [0, 1]:
            continue
        for lineage_2 in range(3):
            if lineage_2 not in lineage_1:
                expected.add((lineage_1, (lineage_2,)))
    assert set(plan.configurations) == expected
    assert len(plan.configurations) == 12
    assert plan.proposal_count == 12


def test_sampler_never_changes_space_after_exhausting_proposals(monkeypatch):
    monkeypatch.setattr(scan_permutation, "ENUMERATION_PROPOSAL_LIMIT", 0)
    monkeypatch.setattr(scan_permutation, "MAX_REJECTION_DRAWS", 3)
    plan = _four_clade_plan()

    class AlwaysOverlap:
        def choice(self, pool, size, replace):
            return np.array([1, 3])

    with pytest.raises(scan_permutation.SamplingError) as error:
        scan_permutation.sample_configuration(plan, AlwaysOverlap())
    assert error.value.attempts == 3
    assert plan.observed == ((0, 1),)


@pytest.mark.parametrize("observed", [((0, 0),), ((0,), (1,))])
def test_overlapping_observed_clades_are_rejected(observed):
    with pytest.raises(ValueError, match="non-overlapping"):
        scan_permutation.build_plan([0, 0], [True, True], observed, [{0, 1}, {1}])


def test_eligibility_does_not_silently_drop_observed_components():
    with pytest.raises(ValueError, match="observed foreground component"):
        scan_permutation.build_plan([0, 0], [True, False], ((0, 1),), [{0}, {1}])

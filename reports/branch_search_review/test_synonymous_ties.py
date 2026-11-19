import numpy as np
import pytest

from synonymous_ties import s_ratio, target_ranks, top_sets


def test_rank_uses_expected_s_and_keeps_vote_order_for_exact_ties():
    n = np.array([8, 7, 7, 7, 7, 6])
    observed = np.array([999, 1, 2, 0, 1, 0])
    expected = np.array([1, 1, 4, 2, 2, 1])
    # At OCN 7 the S rates are 1, 0.5, 0, 0.5 in original vote order.
    def rates(indices):
        assert list(indices) == [1, 2, 3, 4]
        return s_ratio(observed[indices], expected[indices])

    votes, syn, count = top_sets(n, rates, limit=3)
    assert (list(votes), list(syn), count) == ([0, 1, 2], [0, 3, 2], 4)
    assert target_ranks(n, rates, 4) == (5, 4)


@pytest.mark.parametrize(('support', 'limit'), [([], 1), ([4, 3], 2), ([4, 4, 3], 2), ([5, 4, 3], 2)])
def test_no_need_to_compute_s_when_membership_is_fixed(support, limit):
    def unexpected(_):
        pytest.fail('S lookup on an unsplit tie')
    base, syn, n = top_sets(np.array(support), unexpected, limit)
    assert n == 0
    np.testing.assert_array_equal(base, syn)


def test_boundary_and_planted_ranks_match_full_lexicographic_order():
    rng = np.random.default_rng(417)
    n = rng.integers(0, 5, 200)
    observed = rng.integers(0, 20, 200)
    expected = rng.integers(0, 3, 200)
    def rates(idx):
        return s_ratio(observed[idx], expected[idx])
    idx = np.arange(len(n))
    full = np.lexsort((idx, rates(idx), -n))
    votes, syn, tied = top_sets(n, rates)
    assert tied > 1 and set(syn) == set(full[:64])
    assert list(votes) == sorted(idx, key=lambda i: -n[i])[:64]
    for rank, target in enumerate(full, 1):
        assert target_ranks(n, rates, target)[1] == rank


def test_invalid_s_rates_rank_last_and_zero_remains_valid():
    assert list(s_ratio([0, 1, np.nan, -1, 1], [2, 0, 2, 3, np.inf])) == [0, np.inf, np.inf, np.inf, np.inf]

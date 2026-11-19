from itertools import combinations

import numpy as np
import pytest

import branch_pair_index as index
import group_refinement as refinement
import group_refinement_benchmark as benchmark


def test_joint_site_threshold_counts_each_destination_in_the_same_site():
    # Below .5 for each destination separately, exactly .5 in total at site 0.
    p = np.array([[[.5, .5], [.5, .5]], [[.5, .25], [.5, .25]]])
    assert refinement.posterior_metrics(p, .5)((0, 1)) == (1, .5)


def test_known_support_matches_uncertain_posteriors_at_every_arity():
    rng = np.random.default_rng(12)
    state = rng.integers(0, 4, size=(7, 101))
    quality = np.linspace(.3, 1, state.shape[1])
    p = np.stack([state == d for d in (1, 2)], axis=1) * quality
    fast = refinement.known_support(state, quality, .5)
    dense = refinement.posterior_metrics(p, .5)
    for arity in range(2, 8):
        for group in combinations(range(len(state)), arity):
            assert fast(group) == dense(group)[0]


@pytest.mark.parametrize('target', [4, 5])
def test_removing_several_branches_recovers_a_shared_group(target):
    planted = tuple(range(1, target + 1))
    state = np.zeros((target + 4, 6), dtype=int)
    state[:, :2] = 1  # The initial witness includes all four distractors.
    state[list(planted), 2:] = [1, 2, 1, 2]
    support = refinement.known_support(state, np.ones(6), .5)
    events = np.concatenate([(state == d).T for d in (1, 2)])
    rng = np.random.default_rng(13)
    moves = refinement.site_pattern_moves(events, target, rng)
    assert moves(tuple(range(len(state)))) == (planted,)
    result = refinement.refine_group(range(len(state)), target, 2, support, rng, patterns=moves, move_limit=64)
    assert planted in result
    assert len(set(result)) == len(result) == 2
    assert support(planted) == 6


@pytest.mark.parametrize('size,target', [(26, 25), (30, 4)])
def test_sampled_moves_remove_one_or_two_distinct_branches(size, target):
    group = tuple(range(size))
    children = list(refinement._children(group, target, np.random.default_rng(22), 5))
    assert len(set(children)) == 5
    assert all(len(c) == size - min(2, size - target) and set(c) < set(group) for c in children)


def test_index_draw_limit_and_empty_input():
    events = np.ones((4, 7), dtype=bool)
    def propose(rows):
        return index.rank_witnesses([rows], 1, np.random.default_rng(5), score=len, draw_budget=3)
    keys, _, eligible, draws = propose(events)
    assert draws == len(keys) == 3
    assert eligible == 6
    assert keys == propose(events)[0]
    assert propose(events[:0]) == ([], 0, 0, 0)


@pytest.mark.parametrize('arity', [4, 5, 6])
def test_index_uses_requested_arity_for_witnesses_and_proposals(arity):
    events = np.zeros((3, 7), dtype=bool)
    events[:2, :arity] = True
    events[2, :arity - 1] = True
    for score in (None, len):
        keys, _, eligible, draws = index.rank_witnesses(
            [events], 1, np.random.default_rng(1), score=score, arity=arity
        )
        assert keys == [tuple(range(arity))]
        assert eligible == draws == 1


def test_deadline_retains_last_finished_ranking(monkeypatch):
    clock = iter([0., .2, .4, .8, 1.1])
    monkeypatch.setattr(benchmark, 'perf_counter', lambda: next(clock))

    def two_snapshots(state, quality, rng, *, checkpoint, **kwargs):
        votes = {(0, 1, 2, 3): 1}
        checkpoint(votes, rng)
        votes[(0, 1, 2, 4)] = 1
        checkpoint(votes, rng)

    monkeypatch.setattr(index, 'indexed_candidates', two_snapshots)
    result = benchmark.timed_scale('indexed', 128, 7, 1.)
    assert result['proposed'] == 1
    assert result['last_result_seconds'] == .4


def test_diffuse_signal_respects_site_cutoff_in_the_scale_driver():
    args = ('group_refined', 40, 2048, 1, 'uncertain')
    assert index.run(*args, draw_budget=64, min_ocn_site_pp=0)[5]
    assert not index.run(*args, draw_budget=64)[5]

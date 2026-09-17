"""Regression coverage for heuristic candidate selection and trait isolation."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from csubst import combination, ete, main_analyze, tree


def _config():
    tr = tree.add_numerical_node_labels(ete.PhyloNode('(A:1,B:1,C:1,D:1,E:1,F:1)R;', format=1))
    ids = sorted(int(ete.get_prop(n, 'numerical_label')) for n in ete.iter_leaves(tr))
    return {
        'tree': tr, 'dep_ids': [np.array([bid]) for bid in ids],
        'fg_dep_ids': {'traitA': [], 'traitB': []},
        'fg_df': pd.DataFrame(columns=['name', 'traitA', 'traitB']),
        'foreground': None, 'threads': 1, 'exhaustive_until': 2,
        'max_arity': 6, 'max_combination': 100, 'cutoff_stat': 'score,2',
        'float_tol': 1e-12, 'calibrate_longtail': False, 'branch_dist': False,
        'fg_clade_permutation': 0, 'df_cb_stats_main': pd.DataFrame(),
    }, ids


@pytest.mark.parametrize('exhaustive_until', [1, 2, 3])
@pytest.mark.parametrize('threads', [1, 2])
def test_no_foreground_search_reaches_six_with_real_candidate_generation(monkeypatch, exhaustive_until, threads):
    g, ids = _config()
    g.update(exhaustive_until=exhaustive_until, threads=threads)
    observed = {}

    def counts(combos, tensor, g, attr, selected_base_stats=None):
        return pd.DataFrame(combos, columns=['branch_id_' + str(i + 1) for i in range(combos.shape[1])])

    def omega(cb, OS, ON, g):
        cb['score'] = 3.0
        return cb, g

    def annotate(cb, g):
        for trait in ('traitA', 'traitB'):
            for label in ('fg', 'mf', 'mg'):
                cb['is_' + label + '_' + trait] = 'N'
        return cb, g

    def record(g, cb, arity, start):
        observed[arity] = set(map(tuple, cb.filter(regex='^branch_id_').to_numpy()))
        return g

    monkeypatch.setattr(main_analyze.substitution, 'get_cb', counts)
    monkeypatch.setattr(main_analyze.table, 'merge_tables', lambda a, b: a)
    monkeypatch.setattr(main_analyze.substitution, 'add_dif_stats', lambda cb, *a, **kw: cb)
    monkeypatch.setattr(main_analyze.omega, 'calc_omega', omega)
    monkeypatch.setattr(main_analyze.substitution, 'get_substitutions_per_branch', lambda cb, *a: cb)
    monkeypatch.setattr(main_analyze.table, 'get_linear_regression', lambda cb: cb)
    monkeypatch.setattr(main_analyze.foreground, 'get_foreground_branch_num', annotate)
    monkeypatch.setattr(main_analyze.foreground, 'add_median_cb_stats', record)
    main_analyze.cb_search(g, None, None, None, None, write_cb=False)
    assert observed == {k: set(combinations(ids, k)) for k in range(2, 7)}


def test_candidate_cap_honors_cutoff_order_and_filters_foreground_first():
    g, _ = _config()
    g.update(foreground='traits.tsv', max_combination=2, cutoff_stat='secondary,0|score,0')
    cb = pd.DataFrame({
        'branch_id_1': [0, 0, 0, 0], 'branch_id_2': [1, 2, 3, 4],
        'score': [100, 8, 9, 7], 'secondary': [100, 3, 2, 4],
        'is_fg_traitA': ['N', 'Y', 'Y', 'N'], 'is_mf_traitA': ['N'] * 4,
        'is_mg_traitA': ['N'] * 4, 'is_fg_traitB': ['N'] * 4,
        'is_mf_traitB': ['N'] * 4, 'is_mg_traitB': ['N', 'N', 'N', 'Y'],
    })
    result = main_analyze._select_high_order_candidates(cb, g)
    assert result.branch_id_2.tolist() == [4, 2]
    g['cutoff_stat'] = '(secondary|score),0|secondary,0'
    result = main_analyze._select_high_order_candidates(cb, g)
    assert result.branch_id_2.tolist() == [3, 2]
    assert result.columns.is_unique


@pytest.mark.parametrize('selector', ['target', 'passed'])
@pytest.mark.parametrize('threads', [1, 2])
def test_unrelated_trait_cannot_rescue_dependent_foreground_candidates(selector, threads):
    g, ids = _config()
    a, b, c = ids[:3]
    g.update(foreground='traits.tsv', exhaustive_until=1, threads=threads)
    g['fg_dep_ids']['traitA'] = [np.array([a, b])]
    if selector == 'target':
        kwargs = {'target_id_dict': {'traitA': np.array([a, b]), 'traitB': np.array([c])}, 'arity': 2}
    else:
        cb = pd.DataFrame({'branch_id_1': [a, b], 'branch_id_2': [c, c]})
        for trait in ('traitA', 'traitB'):
            for label in ('fg', 'mf', 'mg'):
                cb['is_' + label + '_' + trait] = 'Y' if (trait, label) == ('traitA', 'fg') else 'N'
        kwargs = {'cb_passed': cb, 'arity': 3}
    _, result = combination.get_node_combinations(g, verbose=False, **kwargs)
    assert result.shape == (0, kwargs['arity'])


def test_dependency_annotation_preserved_when_another_trait_validly_selects_row():
    g, ids = _config()
    a, b = ids[:2]
    g.update(foreground='traits.tsv', exhaustive_until=1)
    g['fg_dep_ids']['traitA'] = [np.array([a, b])]
    targets = {trait: np.array([a, b]) for trait in ('traitA', 'traitB')}
    g, result = combination.get_node_combinations(g, target_id_dict=targets, arity=2, verbose=False)
    np.testing.assert_array_equal(result, [[a, b]])
    np.testing.assert_array_equal(g['fg_dependent_id_combinations']['traitA'], [[a, b]])
    assert g['fg_dependent_id_combinations']['traitB'].size == 0


@pytest.mark.parametrize('arity', [2, 3, 4])
@pytest.mark.parametrize('exhaustive_until', [1, 4])
def test_trait_candidate_union_matches_independent_bruteforce(arity, exhaustive_until):
    rng = np.random.default_rng(917)
    for _ in range(5):
        g, ids = _config()
        g.update(foreground='traits.tsv', exhaustive_until=exhaustive_until)
        global_pair = rng.choice(ids, 2, replace=False)
        g['dep_ids'].append(global_pair)
        for trait in ('traitA', 'traitB'):
            g['fg_dep_ids'][trait] = [rng.choice(ids, 2, replace=False)]
        previous = list(combinations(ids, arity - 1))
        masks = {trait: rng.random(len(previous)) < .55 for trait in ('traitA', 'traitB')}
        cb = pd.DataFrame(previous, columns=['branch_id_' + str(i + 1) for i in range(arity - 1)])
        expected = set()
        for trait, mask in masks.items():
            cb['is_fg_' + trait] = np.where(mask, 'Y', 'N')
            cb['is_mg_' + trait] = cb['is_mf_' + trait] = 'N'
            for left, right in combinations([row for row, keep in zip(previous, mask) if keep], 2):
                union = set(left) | set(right)
                if len(union) != arity or set(global_pair) <= union:
                    continue
                if exhaustive_until < arity and set(g['fg_dep_ids'][trait][0]) <= union:
                    continue
                expected.add(tuple(sorted(union)))
        g, actual = combination.get_node_combinations(g, cb_passed=cb, arity=arity, verbose=False)
        assert set(map(tuple, actual)) == expected
        if actual.size:
            for trait in ('traitA', 'traitB'):
                expected_dependent = {row for row in expected if set(g['fg_dep_ids'][trait][0]) <= set(row)}
                assert set(map(tuple, g['fg_dependent_id_combinations'][trait])) == expected_dependent


@pytest.mark.parametrize('values, expected', [([2., float('nan'), float('inf')], [1, 3]), ([1., 1., 1.], [])])
def test_candidate_cutoff_handles_boundary_nan_inf_and_empty(values, expected):
    g, _ = _config()
    cb = pd.DataFrame({'branch_id_1': [0, 0, 0], 'branch_id_2': [1, 2, 3], 'score': values})
    assert main_analyze._select_high_order_candidates(cb, g).branch_id_2.tolist() == expected

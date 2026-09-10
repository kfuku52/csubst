import itertools
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from csubst import pair_epistasis as pair
spec = importlib.util.spec_from_file_location('validate_pair_epistasis', Path(__file__).resolve().parents[2] / 'tools' / 'validate_pair_epistasis.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def landscape(strength=1.6):
    return pair.PairLandscape([[0., .15, -.1], [-.1, .1, .2]],
                              strength * np.eye(3), [.6, 1.])


def mean_rates(model):
    q, pi, states = model.generator(), model.stationary(), model.states
    return np.array([sum(pi[i] * q[i, j] for i, old in enumerate(states)
                         for j, new in enumerate(states) if old[s] != new[s])
                     for s in range(2)])


def test_generator_balance_and_matched_baseline():
    model = landscape()
    q, pi = model.generator(), model.stationary()
    np.testing.assert_allclose(q.sum(axis=1), 0., atol=1e-15)
    flux = pi[:, None] * q
    np.testing.assert_allclose(flux, flux.T, atol=1e-15)
    baseline = pair.matched_independent_landscape(model)
    for axis in (0, 1):
        np.testing.assert_allclose(pi.reshape(3, 3).sum(axis=axis),
                                   baseline.stationary().reshape(3, 3).sum(axis=axis))
    np.testing.assert_allclose(mean_rates(model), mean_rates(baseline))
    # The same mutation 0 -> 1 has different rates in backgrounds 0 and 1.
    assert q[0, 3] != pytest.approx(q[1, 4])


def test_independent_limit_factorizes_and_zero_time():
    model = landscape(0.)
    baseline = pair.matched_independent_landscape(model)
    np.testing.assert_allclose(model.generator(), baseline.generator(), atol=1e-15)
    singles = []
    for fields, mu in zip(model.fields, model.mutation_rates):
        q = mu / 2 * np.exp((fields[None, :] - fields[:, None]) / 2)
        np.fill_diagonal(q, 0.)
        np.fill_diagonal(q, -q.sum(axis=1))
        singles.append(expm(.7 * q))
    np.testing.assert_allclose(expm(.7 * model.generator()), np.kron(*singles), atol=1e-15)
    for pmf in pair.exact_category_pmfs(model, [[0, 0], [1, 2]], [0., 0.]).values():
        np.testing.assert_array_equal(pmf, [1., 0., 0.])


def test_categories_against_scalar_definition_exhaustively():
    endpoints = np.array(list(itertools.product(landscape().states, repeat=2)))
    for parents in endpoints:
        observed = pair.endpoint_categories(parents, endpoints)
        expected = np.zeros_like(observed)
        for n, tips in enumerate(endpoints):
            for site in range(2):
                a, b = parents[:, site]
                c, d = tips[:, site]
                if a == c or b == d:
                    continue
                for i, category in enumerate(pair.CATEGORIES):
                    origin, destination = category.split('2')
                    match_from = origin == 'any' or (origin == 'spe' and a == b) or (origin == 'dif' and a != b)
                    match_to = destination == 'any' or (destination == 'spe' and c == d) or (destination == 'dif' and c != d)
                    expected[n, i] += int(match_from and match_to)
        np.testing.assert_array_equal(observed, expected)


def test_exact_transition_matches_independent_gillespie():
    model = landscape()
    size = 60000
    draws = tool.simulate_endpoints(model, [0, 1], .8, size, np.random.default_rng(531))
    observed = np.bincount(draws[:, 0] * 3 + draws[:, 1], minlength=9) / size
    expected = expm(.8 * model.generator())[1]
    tolerance = 6 * np.sqrt(expected * (1 - expected) / size) + 1 / size
    assert np.all(abs(observed - expected) < tolerance)


def test_pair_convolution_preserves_dependence_and_valid_tail():
    model = landscape()
    pmf = pair.exact_category_pmfs(model, [[0, 1], [0, 1]], [.5, .65])['any2spe']
    total = pair.repeated_pair_pmf(pmf, 2)
    manual = np.zeros(5)
    for i, j in itertools.product(range(3), repeat=2):
        manual[i + j] += pmf[i] * pmf[j]
    np.testing.assert_allclose(total, manual)
    for level in (.01, .05, .1, .5):
        assert total[pair.upper_tail(total) <= level].sum() <= level + 1e-14


@pytest.mark.parametrize('parents,lengths', [([[0., 0.], [1., 1.]], [1, 1]),
                                            ([[0, 0], [1, 3]], [1, 1]),
                                            ([[0, 0], [1, 1]], [-1, 1]),
                                            ([[0, 0], [1, 1]], [1, np.nan])])
def test_invalid_branch_input(parents, lengths):
    with pytest.raises(ValueError):
        pair.exact_category_pmfs(landscape(), parents, lengths)


@pytest.mark.parametrize('pmf,count', [([.1, .2, .3], 2), ([-.1, .1, 1.], 2),
                                      ([.1, .2, .7], 0), ([.1, .2, .7], True)])
def test_invalid_convolution(pmf, count):
    with pytest.raises(ValueError):
        pair.repeated_pair_pmf(pmf, count)

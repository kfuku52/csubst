import importlib.util
from pathlib import Path

import numpy as np
from scipy.linalg import expm


spec = importlib.util.spec_from_file_location('validate_epistasis', Path(__file__).resolve().parents[2] / 'tools' / 'validate_epistasis.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_binary_generator_obeys_independent_fitness_and_detailed_balance():
    model = tool.Landscape([.1, -.2], [[0., .8], [.8, 0.]], [.3, .4])
    states, q, pi = model.exact_generator()
    np.testing.assert_allclose(q.sum(axis=1), 0., atol=1e-14)
    np.testing.assert_allclose(pi @ q, 0., atol=1e-14)
    np.testing.assert_allclose(pi[:, None] * q, (pi[:, None] * q).T, atol=1e-14)
    for i, x in enumerate(states):
        for j, y in enumerate(states):
            changed = np.flatnonzero(x != y)
            if len(changed) == 1:
                # Independent, full-energy difference, not the simulator's
                # optimized local-background rate calculation.
                old_spin, new_spin = 2 * x - 1, 2 * y - 1
                energy_difference = (.1 * new_spin[0] - .2 * new_spin[1] + .8 * new_spin.prod()) - (.1 * old_spin[0] - .2 * old_spin[1] + .8 * old_spin.prod())
                expected = [.3, .4][changed[0]] * np.exp(energy_difference / 2.)
                np.testing.assert_allclose(q[i, j], expected)


def test_gillespie_transition_distribution_matches_matrix_exponential():
    model = tool.Landscape([.1, -.2], [[0., .8], [.8, 0.]], [.3, .4])
    states, q, _ = model.exact_generator()
    rng = np.random.default_rng(44)
    observed = np.zeros(4)
    for _ in range(10000):
        final, counts = model.evolve(states[0], .7, rng)
        observed[2 * final[0] + final[1]] += 1
        assert counts[1].sum() == 0
    np.testing.assert_allclose(observed / observed.sum(), expm(.7 * q)[0], atol=.015)


def test_null_factorizes_but_coupled_process_depends_on_background():
    for coupling in [0., .7]:
        model = tool.Landscape([.1, -.2], [[0., coupling], [coupling, 0.]], [.3, .4])
        rates_a = model.transitions(np.array([0, 0]))[2]
        rates_b = model.transitions(np.array([0, 1]))[2]
        assert bool(np.isclose(rates_a[0], rates_b[0])) == (coupling == 0.)


def test_codon_generator_has_61_sense_codons_single_nt_moves_and_synonymous_control():
    model = tool.Landscape([.1, -.2], [[0., .7], [.7, 0.]], [.3, .4], 'codon')
    assert len(model.symbols) == 61
    assert not {'TAA', 'TAG', 'TGA'}.intersection(model.symbols)
    initial = np.array([model.symbols.index('AAA'), model.symbols.index('GAA')])
    sites, targets, rates, syn = model.transitions(initial)
    for site, target, rate, synonymous in zip(sites, targets, rates, syn):
        assert sum(a != b for a, b in zip(model.symbols[initial[site]], model.symbols[target])) == 1
        if synonymous:
            np.testing.assert_allclose(rate, model.site_rates[site])
        final = initial.copy()
        final[site] = target
        np.testing.assert_allclose(rate, model.site_rates[site] * np.exp((model.fitness(final) - model.fitness(initial)) / 2.))


def test_simulator_report_is_explicit_about_inference_scope():
    payload = tool.run(2, 1, ['iid', 'binary_null', 'codon_epistatic'])
    assert payload['omega_pvalue_calibrated'] is False
    assert len(payload['replicates']) == 6
    assert all(np.isfinite(v['mean_outer_gain_nats_per_event']) for v in payload['summary'].values())

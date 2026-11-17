from itertools import product

import numpy as np
from scipy.linalg import expm

from csubst import ete, simulate_epistasis, tree
from csubst.simulate_omega import ExpectedOmega


def test_inverse_epistasis_expectation_matches_independent_scalar_sum():
    tr = tree.add_numerical_node_labels(ete.PhyloNode('((A:0.2,C:0.2)X:0.4,(B:0.3,D:0.2)Y:0.1)R;', format=1))
    codes = ['AAA', 'AAG', 'GAA', 'GAG']
    amino = np.array([0, 0, 1, 1])
    pi = np.full(4, .25)
    q = np.array([[-3, 1, 2, 0], [1, -3, 0, 2], [2, 0, -3, 1], [0, 2, 1, -3]], dtype=float)
    g = {'tree': tr, 'codon_table': list(zip('KKEE', codes)), 'pyvolve_codon_orders': codes,
         'eq_freq': pi, 'background_Q': q, 'simulate_seed': 4, 'simulate_rng': np.random.default_rng(4),
         'simulate_site_rates': np.array([1., 0., 1.]), 'num_convergent_site': 2,
         'simulate_epistasis_distance_scale': .5, 'simulate_epistasis_preference_multiplier': 2.,
         'foreground_scaling_factor': 1.}
    focal = {ete.get_prop(n, 'numerical_label') for n in ete.iter_leaves(tr) if n.name in ('A', 'B')}
    oracle = ExpectedOmega(tr, g['codon_table'], codes, 'test')
    sequences = simulate_epistasis.evolve(g, focal, 'inverse', oracle)
    assert len({s[3:6] for s in sequences.values()}) == 1  # invariant site

    expected = np.zeros(4)
    pa, pb, pr = expm(.2*q), expm(.3*q), expm(.5*q)
    conditional_a, conditional_b = [], []
    for favored in range(2):
        tilted = q.copy()
        for i, j in product(range(4), repeat=2):
            if amino[i] != amino[j] == favored:
                tilted[i, j] *= 2.
                tilted[i, i] -= q[i, j]
        conditional_a.append(expm(.2*tilted))
        conditional_b.append(expm(.3*tilted))
    for a, b, c, d in product(range(4), repeat=4):
        n = amino[a] != amino[c] == amino[d] != amino[b]
        s = a != c == d != b and amino[a] == amino[c] == amino[b]
        weight = pi[a] * pr[a, b]
        base = weight * pa[a, c] * pb[b, d]
        expected += np.array([n, s, n, s]) * base
        expected[2:] += np.array([n, s]) * base
        for x, y in product(range(2), repeat=2):
            # Retention is 1/2, leaving a 1/2 component with independent preferences.
            context_weight = .25*(x == y) + .125
            full = weight * context_weight * conditional_a[x][a, c] * conditional_b[y][b, d]
            expected[:2] += np.array([n, s]) * full
    i, j = sorted((oracle.indices['A'], oracle.indices['B']))
    np.testing.assert_allclose(oracle.counts[i, j], expected, rtol=1e-12, atol=1e-14)

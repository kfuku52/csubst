import csv
from itertools import product

import numpy as np
from scipy.linalg import expm

from csubst import ete, tree
from csubst._vendor import pyvolve
from csubst.simulate_omega import ExpectedOmega, _omega


def test_omega_avoids_overflow_and_underflow_of_intermediate_rates():
    n = np.array([1e200, 1e-200, 0.0, 1.0, 1.0])
    s = np.array([2e200, 2e-200, 0.0, 0.0, 1.0])
    null_n = np.array([1e-200, 1e200, 1.0, 1.0, 0.0])
    null_s = np.array([1e-200, 1e200, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(_omega(n, s, null_n, null_s), [0.5, 0.5, np.nan, np.inf, np.nan])


def test_expected_omega_matches_enumeration_with_shared_ancestry(tmp_path):
    codons = ['AAA', 'AAG', 'GAA', 'GAG']
    amino_acids = ['K', 'K', 'E', 'E']
    background = np.array([
        [-3, 1, 2, 0], [1, -3, 0, 2], [2, 0, -3, 1], [0, 2, 1, -3],
    ], dtype=float)
    biased = background.copy()
    biased[:2, :] *= 3
    root, fg = [
        pyvolve.Model(
            'custom', name=name, parameters={'matrix': matrix, 'code': codons},
            save_custom_frequencies=tmp_path / 'frequencies.txt',
        )
        for name, matrix in [('root', background), ('fg', biased)]
    ]
    tr = tree.add_numerical_node_labels(ete.PhyloNode('(A:0.2,(B:0.3,C:0)U:0.4)R;', format=1))
    partition = pyvolve.Partition(models=[root, fg], root_model_name='root', size=3)
    evolver = pyvolve.Evolver(
        partitions=partition, tree=pyvolve.read_tree(tree='(A:0.2,(B:0.3#fg,C:0)U:0.4#fg)R;'),
    )
    evolver(seed=1, seqfile=None, ratefile=None, infofile=None)
    oracle = ExpectedOmega(tr, list(zip(amino_acids, codons)), codons, 'trait')
    oracle.add_evolution(evolver)
    output = tmp_path / 'expected.tsv'
    oracle.write(output, {
        'R': 'AAAGAAAAG', 'U': 'AAAGAAAAG', 'C': 'AAAGAAAAG',
        'A': 'GAAGAGAAG', 'B': 'GAAGAGAAA',
    })
    with output.open() as handle:
        rows = {
            frozenset((row['branch_name_1'], row['branch_name_2'])): row
            for row in csv.DictReader(handle, delimiter='\t')
        }
    assert set(rows) == {
        frozenset(pair) for pair in [('A', 'B'), ('A', 'C'), ('A', 'U'), ('B', 'C')]
    }
    pa, pu = expm(background * 0.2), expm(biased * 0.4)
    pair = rows[frozenset(('A', 'B'))]
    assert float(pair['dist_bl']) == 0.4
    assert float(rows[frozenset(('B', 'C'))]['dist_bl']) == 0.
    assert all(row['dist_bl'] == row['dist_parent_bl'] for row in rows.values())
    for label, pb in [('sim', expm(biased * 0.3)), ('null', expm(background * 0.3))]:
        n = s = 0.0
        for r, u, a, b in product(range(4), repeat=4):
            weight = 3 * 0.25 * pu[r, u] * pa[r, a] * pb[u, b]
            aa_r, aa_u, aa_a, aa_b = [amino_acids[i] for i in (r, u, a, b)]
            n += weight * (aa_r != aa_a and aa_a == aa_b and aa_b != aa_u)
            s += weight * (r != a and a == b and b != u and aa_r == aa_a == aa_u)
        np.testing.assert_allclose([float(pair[f'muNany2spe_{label}']), float(pair[f'muSany2spe_{label}'])],
                                   [n, s], rtol=1e-12)
    assert (pair['OCNany2spe_true'], pair['OCSany2spe_true']) == ('1', '1')
    assert rows[frozenset(('B', 'C'))]['omegaCany2spe_expected'] == 'NA'
    assert float(rows[frozenset(('A', 'U'))]['omegaCany2spe_expected']) != 1.0

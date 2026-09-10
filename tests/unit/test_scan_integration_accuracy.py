from pathlib import Path
import runpy

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / '.github/scripts/scan_integration_accuracy.py'


def test_recorded_jump_truth_matches_endpoints_and_poisson_mean():
    simulate = runpy.run_path(str(SCRIPT))['simulate_with_truth']
    model = dict(topology='(a:.4,e:.4)R;', sites=2000,
                 codons=np.array(['AAA','AAC']), pi=np.array([.5,.5]),
                 q=np.array([[-1.,1.],[1.,-1.]]))
    sequences, truth = simulate(model,np.random.default_rng(1051))
    a = np.array([sequences['a'][i:i+3] for i in range(0,6000,3)])
    e = np.array([sequences['e'][i:i+3] for i in range(0,6000,3)])
    # Every jump switches the amino acid. Endpoint disagreement is exactly the
    # parity of the two root-to-tip counts, including returns and multiple hits.
    np.testing.assert_array_equal(a != e, (truth['a'] + truth['e']) % 2 == 1)
    counts = np.r_[truth['a'],truth['e']]
    assert abs(counts.mean() - .4) < .04
    assert (counts >= 2).any()

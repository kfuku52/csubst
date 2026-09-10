"""Small scientific consistency probes; not an end-to-end calibration study.

Run from repository root: python reports/scientific_review_20260910/reproduce.py
"""
import contextlib
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from csubst import omega, parser_misc, substitution_scan


def main():
    out = {}
    # Same count vector on observation and every null replicate must tie when
    # the same statistic is applied. Inject counts to isolate statistic handling.
    cb = pd.DataFrame({'branch_id_1': [0], 'branch_id_2': [1],
                       'OCNany2spe': [1.], 'ECNany2spe': [10.],
                       'OCSany2spe': [10.], 'ECSany2spe': [10.]})
    g = dict(output_stats=['any2spe'], float_tol=1e-12,
             pseudocount_mode='symmetric', pseudocount_alpha=1.,
             pseudocount_target='both', calc_omega_pvalue=True,
             expectation_method='urn', omega_pvalue_niter_schedule=[1000])
    def identical_counts(**kwargs):
        value = 1. if kwargs['SN'] == 'N' else 10.
        return np.full((1, kwargs['niter']), value)
    with contextlib.redirect_stdout(io.StringIO()):
        smoothed = omega.get_omega(cb.copy(), g)
        with patch.object(omega, '_get_mode_permutation_count_matrix', identical_counts):
            result = omega.add_omega_empirical_pvalues(smoothed, None, None, g)
    out['pseudocount_identical_null'] = result[['omegaCany2spe', 'pomegaCany2spe']].iloc[0].to_dict()
    assert result['pomegaCany2spe'].iloc[0] < .002

    # IID uniform single events have no structure dependence. Their own feature
    # nevertheless makes held-out scoring favour a positive coefficient.
    rng = np.random.default_rng(20260910)
    counts = np.zeros((500, 20))
    counts[np.arange(500), rng.integers(0, 20, size=500)] = 1
    feature = np.linspace(-1, 1, 20)
    mask = np.ones_like(counts, dtype=bool)
    base = np.full_like(counts, 1 / 20)
    context = omega._calc_epistasis_branch_context(counts, feature, mask, 1e-12)
    beta, diag = omega._fit_epistasis_beta_cv(counts, base, context, feature, mask, 3., 1e-12)
    out['iid_uniform_epistasis_cv'] = dict(selected_beta=beta, diagnostics=diag)

    # Equal any/spe means with all events specific: a valid nested null has
    # identically zero differences. Independent mode draws do not.
    a = rng.poisson(1, (1, 200000)).astype(float)
    s = rng.poisson(1, (1, 200000)).astype(float)
    d = omega._compose_permutation_count_matrix('any2dif', {'any2any': a, 'any2spe': s}, 1e-12)
    out['independent_nested_null'] = dict(undefined_fraction=float(np.isnan(d).mean()),
                                         positive_fraction=float((d > 0).mean()))

    out['sparse_poisson_test'] = dict(lrt=substitution_scan._poisson_lrt_pvalue(2, 1, 0, 1),
                                     conditional_exact=binomtest(2, 2, .5, alternative='greater').pvalue)
    calibrated, _, _ = omega._calibrate_dsc_vector([10.], [1.])
    out['single_row_longtail'] = dict(raw_omega=10., calibrated_omega=float(10 / calibrated[0]))
    out['marginal_product_zero_length'] = dict(marginal_product_change_mass=float(1 - np.square([.5, .5]).sum()),
                                              joint_change_mass=0.)

    # A state can be reached via an intermediate state although Q_02=0.
    q = np.array([[-1., 1., 0.], [1., -2., 1.], [0., 1., -1.]])
    opp = substitution_scan._q_weighted_opportunity(pd.DataFrame({'parent_id': [0]}),
        np.array([[[1., 0., 0.]]]), 0, [0], [2], q, 'raw')
    out['instantaneous_vs_endpoint'] = dict(exposure=float(opp[0]), endpoint_probability=float(expm(q)[0, 2]))

    # 3Di initialisation currently gives its symbols amino-acid codon groups.
    with contextlib.redirect_stdout(io.StringIO()):
        init = parser_misc._initialize_and_report_nonsyn_recode(dict(nonsyn_recode='3di20',
            amino_acid_orders=np.array(['A', 'C']), synonymous_indices={'A': [0], 'C': [1]},
            matrix_groups={'A': ['GCT'], 'C': ['TGT']}))
    try:
        out['3di_initial_q'] = parser_misc.cdn2nsy_matrix(np.array([[-2., 2.], [1., -1.]]), init).tolist()
    except ValueError as exc:
        # After issue 1 is fixed, this invalid conversion must be rejected.
        out['3di_initial_q'] = {'blocked': True, 'reason': str(exc)}
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

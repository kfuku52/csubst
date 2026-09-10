"""Real dense/sparse urn means, count samplers, and smoothing (no ASR fixtures)."""
import itertools

import numpy as np
import pandas as pd
import pytest

from csubst import omega, omega_null, output_stat, substitution_sparse


def inputs(arity, asrv):
    tensor = np.zeros((4, 8, 1, 3, 3))
    for branch in range(4):
        for k, (a, d) in enumerate([(0, 1), (0, 2), (1, 2)]):
            tensor[branch, (branch+k) % 8, 0, a, d] = .2
    ids = np.array(list(itertools.combinations(range(4), arity)))
    cb = pd.DataFrame(ids, columns=['branch_id_'+str(i+1) for i in range(arity)])
    weights = np.arange(1, 9, dtype=float)
    weights /= weights.sum()
    g = dict(expectation_method='urn', urn_model='wallenius', asrv=asrv,
             sub_sites={'no': np.full(8, .125), 'N': weights, 'S': weights[::-1]},
             threads=1, float_type=np.float64, float_tol=1e-12,
             calc_omega_pvalue=True, omega_pvalue_null_model='poisson',
             omega_pvalue_niter_schedule=[20], output_stats=list(output_stat.ALL_OUTPUT_STATS),
             pseudocount_mode='symmetric', pseudocount_alpha=.5, random_seed=37)
    for ch in ['N', 'S']:
        summary = tensor.sum(axis=(0, 1))
        g[ch+'_ind_nomissing_gad'] = np.where(summary != 0)
        g[ch+'_ind_nomissing_ga'] = np.where(summary.sum(axis=2) != 0)
        g[ch+'_ind_nomissing_gd'] = np.where(summary.sum(axis=1) != 0)
        for sub, axes in [('any2any', (3, 4)), ('spe2any', (4,)), ('any2spe', (3,)), ('spe2spe', ())]:
            projection = tensor.sum(axis=axes) if axes else tensor
            cb['OC'+ch+sub] = [np.prod(projection[row], axis=0).sum() for row in ids]
            cb['EC'+ch+sub] = omega.calc_E_stat(cb, tensor, sub, SN=ch, g=g)
        for prefix in ['OC', 'EC']:
            atoms = omega_null.atomic_means({s: cb[prefix+ch+s] for s in output_stat.BASE_OUTPUT_STATS}, 1e-12)
            for s in output_stat.ALL_OUTPUT_STATS:
                cb[prefix+ch+s] = omega_null.project_atoms(atoms, s)
    return tensor, cb, g


@pytest.mark.parametrize('arity', [2, 3])
@pytest.mark.parametrize('asrv', ['no', 'sn'])
def test_real_urn_means_joint_null_dense_sparse_parity(arity, asrv):
    tensor, cb, g = inputs(arity, asrv)
    sparse = substitution_sparse.dense_to_sparse_substitution_tensor(tensor)
    # Hide unused base E columns as normal selected-output pipelines do.
    g['output_stats'] = ['any2any', 'any2spe', 'any2dif']
    cb = cb.drop(columns=['EC'+ch+s for ch in ['N', 'S'] for s in ['spe2any', 'spe2spe']])
    cb = omega.get_omega(cb, g)
    dense_result = omega.add_omega_empirical_pvalues(cb.copy(), tensor, tensor, dict(g))
    sparse_result = omega.add_omega_empirical_pvalues(cb.copy(), sparse, sparse, dict(g))
    for s in g['output_stats']:
        np.testing.assert_array_equal(dense_result['pomegaC'+s], sparse_result['pomegaC'+s])
        assert dense_result['pomegaC'+s].between(1/21, 1).all()
        assert (dense_result['pvalue_undefined_'+s] == 0).all()


@pytest.mark.parametrize('model', ['hypergeom', 'poisson', 'poisson_full', 'nbinom'])
def test_fixed_smoothing_real_engines_schedule_invariance(model):
    tensor, cb, g = inputs(2, 'sn')
    g.update(output_stats=['any2spe'], omega_pvalue_null_model=model,
             omega_pvalue_nbinom_alpha=.3, omega_pvalue_niter_schedule=[20])
    cb = omega.get_omega(cb, g)
    a = omega.add_omega_empirical_pvalues(cb.copy(), tensor, tensor, dict(g))
    g.update(omega_pvalue_niter_schedule=[5, 20], longtail_test_block_size=3)
    b = omega.add_omega_empirical_pvalues(cb.iloc[::-1].copy(), tensor, tensor, dict(g)).iloc[::-1]
    np.testing.assert_array_equal(a.pomegaCany2spe, b.pomegaCany2spe)
    assert (a.pvalue_n_any2spe == 20).all()


@pytest.mark.parametrize('mode,alpha', [('symmetric', 1.), ('empirical', 'auto')])
def test_cli_run_context_preserves_smoothing_through_inference(mode, alpha):
    from csubst.runtime import RunContext
    tensor, cb, g = inputs(2, 'no')
    g.update(output_stats=['any2any', 'any2spe', 'any2dif'],
             pseudocount_mode=mode, pseudocount_alpha=alpha,
             omega_pvalue_niter_schedule=[5])
    outputs = []
    for context in [dict(g), RunContext(config=g)]:
        observed = omega.get_omega(cb.copy(), context)
        out = omega.add_omega_empirical_pvalues(observed, tensor, tensor, context)
        assert (out['pvalue_statistic_any2spe'] == 'smoothed').all()
        outputs.append(out)
    pd.testing.assert_frame_equal(outputs[0], outputs[1])

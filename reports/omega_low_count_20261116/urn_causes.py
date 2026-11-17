"""Replay PEPC urn tails and replace the components of synonymous overlap."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.linalg import expm

from csubst import cli, endpoint_io, ete, genetic_code, main_simulate, omega, omega_statistics, sequence, tree
from csubst._vendor.pyvolve.evolver import ZERO
from csubst.simulate_omega import ExpectedOmega

from pilot import EXPECTED, KEY, NULL, OBS, TRUTH
from replay_posterior import oracle_generator
from urn_accuracy import PVALUE, QVALUE, load_search, search_argv


@dataclass(eq=False)
class BackgroundModel:
    name: str
    matrix: np.ndarray
    params: dict


def capture_search(root, previous, seed):
    """Collect inputs to the actual CLI empirical-tail calculation."""
    fields = ('sim.fa', 'tree.nwk', *(f'fit.{ext}' for ext in ('iqtree', 'state', 'rate', 'treefile', 'log', 'ckp.gz')))
    root.mkdir(parents=True)
    for name in fields:
        shutil.copyfile(previous / name, root / name)
    outdir, argv = search_argv('each', seed, 999)
    components = omega._prepare_substitution_permutation_components
    counts = omega._get_mode_permutation_count_matrix
    pvalues = omega.add_omega_empirical_pvalues
    captured = {}

    def record_components(sub_tensor, mode, SN, g):
        result = components(sub_tensor, mode, SN, g)
        if SN == 'S':
            branch, site = result[:2]
            s_by_codon = np.zeros((branch.shape[0], len(g['codon_orders'])))
            site_by_codon = np.zeros((site.shape[0], len(g['codon_orders'])))
            for i, aa in enumerate(g['amino_acid_orders']):
                indices = g['synonymous_indices'][aa]
                s_by_codon[:, indices] = branch[:, i, :len(indices)]
                site_by_codon[:, indices] = site[:, i, :len(indices)]
            captured.update(branch=s_by_codon, site=site_by_codon, codons=list(g['codon_orders']),
                            tol=g['float_tol'])
        return result

    def record_samples(**kwargs):
        values = counts(**kwargs)
        captured[kwargs['SN']] = values
        captured['pairs'] = kwargs['cb_ids'].copy()
        return values

    def record_pvalue_inputs(cb, ON_tensor, OS_tensor, g):
        captured['inputs'] = cb[KEY + OBS + EXPECTED].copy()
        return pvalues(cb, ON_tensor, OS_tensor, g)

    old = Path.cwd()
    try:
        os.chdir(root)
        with (patch.object(omega, '_prepare_substitution_permutation_components', record_components),
              patch.object(omega, '_get_mode_permutation_count_matrix', record_samples),
              patch.object(omega, 'add_omega_empirical_pvalues', record_pvalue_inputs)):
            if cli.main(argv):
                raise RuntimeError(f'Urn replay failed in {root}.')
    finally:
        os.chdir(old)
    table, _ = load_search(root, outdir)
    return captured, table


def oracle(rep, frame, codons):
    """Reconstruct rate-specific pair probabilities and realized branch endpoints."""
    sim = rep / 'sim'
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(sim / 'sim_true_asr.treefile'), format=1))
    parents, _ = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    original = rep.parent / 'inputs/PEPC.tree.nwk'
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in tr.traverse()}
    lengths = pd.Series({node.name: node.dist for node in tr.traverse()})
    original_tree = tree.add_numerical_node_labels(ete.PhyloNode(str(original), format=1))
    orig_parents, times = endpoint_io._tree_arrays({'tree': original_tree}, structural=False)
    np.testing.assert_array_equal(parents, orig_parents)
    np.testing.assert_allclose(times[parents >= 0], lengths.loc[[names[i] for i in np.where(parents >= 0)[0]]],
                               rtol=2e-5, atol=2e-8)
    for side in (1, 2):
        assert frame[f'branch_id_{side}'].map(names).eq(frame[f'branch_name_{side}']).all()
    sequences = sequence.read_fasta(sim / 'sim.fa') | sequence.read_fasta(sim / 'sim_true_asr.anc.fa')
    lookup = {codon: i for i, codon in enumerate(codons)}
    rates = pd.read_csv(sim / 'sim_true_asr.rate', sep='\t')['C_Rate'].to_numpy()
    states = np.array([[lookup[sequences[names[i]][s:s+3]] for s in range(0, 3 * rates.size, 3)]
                       for i in range(len(parents))])
    code = genetic_code.get_codon_table(1)
    pair_model = ExpectedOmega(tr, code, codons, '')
    vendor_tree = main_simulate.get_pyvolve_tree(tr, 1., '')
    by_name = {names[i]: times[i] for i in np.where(parents >= 0)[0]}
    pending = list(vendor_tree.children)
    while pending:
        node = pending.pop()
        node.branch_length = by_name[node.name]
        pending.extend(node.children)
    q, pi = oracle_generator(rep, np.asarray(codons))
    left, right = (frame[c].map(pair_model.indices).to_numpy() for c in ('branch_name_1', 'branch_name_2'))
    probabilities = []
    parent_states = states[parents]
    q_s = np.zeros((*states.shape, len(codons)))
    for rate in np.unique(rates):
        model = BackgroundModel('root', q * rate, {'state_freqs': pi})
        evolver = SimpleNamespace(full_tree=vendor_tree, scale_tree=1.,
                                  _exponentiate_matrix=lambda matrix, time: expm(matrix * time))
        pair_model.counts.fill(0)
        pair_model._add_model_group(evolver, model, [model], 1)
        mask = rates == rate
        probabilities.append((int(mask.sum()), pair_model.counts[left, right, :2].copy()))
        for b in np.where((parents >= 0) & (times > ZERO))[0]:
            transition = expm(q * (times[b] * rate))
            q_s[b, mask] = (transition * pair_model.synonymous)[parent_states[b, mask]]
    np.testing.assert_allclose(sum(size * p for size, p in probabilities), frame[NULL], rtol=4e-9, atol=2e-11)
    true_s = (states != parent_states) & (pair_model.aa[states] == pair_model.aa[parent_states])
    true_s[parents < 0] = False
    a, b = frame[KEY].to_numpy().T
    np.testing.assert_array_equal((true_s[a] & true_s[b] & (states[a] == states[b])).sum(axis=1), frame[TRUTH[1]])
    realized_totals = np.stack([(true_s & (states == d)).sum(axis=1) for d in range(len(codons))], axis=1)
    return probabilities, q_s, realized_totals, parent_states, pair_model.aa, rates


def overlap(categories, a, b):
    flat = categories.reshape(len(categories), -1)
    return (flat @ flat.T)[a, b]


def linear_allocation(profile, totals):
    mass = profile.sum(axis=-2, keepdims=True)
    return np.divide(profile, mass, out=np.zeros_like(profile), where=mass > 0) * totals[:, None, :]


def ecs_cases(frame, capture, q_s, realized, parent, aa, rates):
    a, b = frame[KEY].to_numpy().T
    fitted = capture['branch']
    oracle_totals = q_s.sum(axis=1)
    np.testing.assert_allclose(fitted.sum(axis=1)[a], frame['S_sub_1'], rtol=1e-9, atol=1e-10)
    each = capture['site'] + 1.
    same_aa = aa[parent][..., None] == aa
    possible = same_aa & (parent[..., None] != np.arange(len(aa)))
    by_rate = np.empty_like(q_s)
    for r in np.unique(rates):
        mask = rates == r
        by_rate[:, mask] = q_s[:, mask].mean(axis=1, keepdims=True)
    uniform = (fitted @ fitted.T)[a, b] / len(rates)
    cases = {'production_each': frame[EXPECTED[1]], 'uniform_fitted': uniform,
             'uniform_realized': (realized @ realized.T)[a, b] / len(rates),
             'uniform_q_totals': (oracle_totals @ oracle_totals.T)[a, b] / len(rates),
             'opportunities_averaged_within_rate': overlap(by_rate, a, b),
             'q_parent_full': overlap(q_s, a, b)}
    for label, profile in [('each', each), ('rate_only', rates[:, None] * np.ones(len(aa))),
                           ('parent_aa', same_aa * rates[:, None]),
                           ('parent_codon', possible * rates[:, None]), ('q_parent', q_s)]:
        for source, totals in [('fitted', fitted), ('realized', realized), ('q_totals', oracle_totals)]:
            cases[f'{label}_{source}_linear'] = overlap(linear_allocation(profile, totals), a, b)
    truth = frame[NULL[1]].to_numpy()
    return [dict(method=label, n_pairs=len(truth), q_ecs=truth.sum(), ecs=values.sum(),
                 ecs_sse=np.square(values - truth).sum()) for label, values in cases.items()]


def upper_tail(obs, exp, ns, ss, tol):
    dn = omega_statistics._calc_raw_rate(obs[:, 0], exp[:, 0], tol)
    ds = omega_statistics._calc_raw_rate(obs[:, 1], exp[:, 1], tol)
    score = omega_statistics._calc_raw_omega(dn, ds, tol)
    ge, valid = omega_statistics._calc_omega_empirical_upper_tail_counts(score, *exp.T, ns, ss, tol)
    return omega_statistics._calc_omega_empirical_upper_tail_pvalues_from_counts(score, exp[:, 1], ge, valid)


def tail_cases(frame, capture, probabilities, indices, seed):
    obs, realized, u, q = (frame[cols].to_numpy() for cols in (OBS, TRUTH, EXPECTED, NULL))
    tol = capture['tol']
    ns, ss = [capture[channel][indices] for channel in ('N', 'S')]
    rng = np.random.default_rng(seed + 20261117)
    qn, qs = np.zeros_like(ns), np.zeros_like(ss)
    for size, p in probabilities:
        p_n, p_s = p.T[:, :, None]
        # N and S are mutually exclusive per site; rates are fixed by simulation.
        n = rng.binomial(size, p_n, size=qn.shape)
        qn += n
        qs += rng.binomial(size - n, p_s / (1 - p_n))
    trials = [('urn', 'posterior', obs, u, ns, ss), ('urn_rescale_ec', 'posterior', obs, q, ns, ss),
              ('urn', 'true_n', np.column_stack((realized[:, 0], obs[:, 1])), u, ns, ss),
              ('urn', 'true_s', np.column_stack((obs[:, 0], realized[:, 1])), u, ns, ss),
              ('urn', 'true_both', realized, u, ns, ss),
              ('q_n_urn_s', 'posterior', obs, q, qn, ss), ('urn_n_q_s', 'posterior', obs, q, ns, qs),
              ('generating_q', 'posterior', obs, q, qn, qs),
              ('generating_q', 'true_both', realized, q, qn, qs)]
    rows = []
    # Use the actual low-count implementation in blocks for all counterfactuals.
    blocks = [slice(i, i + 400) for i in range(0, len(frame), 400)]
    for null, observed, values, exp, nsim, ssim in trials:
        p = np.concatenate([upper_tail(values[sl], exp[sl], nsim[sl], ssim[sl], tol) for sl in blocks])
        adjusted = omega._calc_bh_fdr_qvalues(p)
        if null == 'urn' and observed == 'posterior':
            np.testing.assert_allclose(p, frame[PVALUE], rtol=0, atol=7e-13)
            np.testing.assert_allclose(adjusted, frame[QVALUE], rtol=0, atol=7e-13)
        only_n = (1. + (nsim >= values[:, 0, None]).sum(axis=1)) / (1. + nsim.shape[1])
        rejected = adjusted <= .05
        rejected_n = omega._calc_bh_fdr_qvalues(only_n) <= .05
        rows.append(dict(null=null, observed=observed, n_pairs=len(p), finite_p=int(np.isfinite(p).sum()),
                         raw=(p <= .05).sum(), adjusted=rejected.sum(), n_only_raw=(only_n <= .05).sum(),
                         n_only_adjusted=rejected_n.sum(),
                         adjusted_n_only_overlap=(rejected & rejected_n).sum(),
                         adjusted_true_n0=(rejected & (realized[:, 0] == 0)).sum(),
                         adjusted_no_n_draw=(rejected & (nsim.max(axis=1) == 0)).sum(),
                         adjusted_p_min=(rejected & (p == .001)).sum(),
                         nsim_sum=nsim.sum(), ssim_sum=ssim.sum(),
                         nsim_zero=(nsim == 0).sum(), ssim_zero=(ssim == 0).sum()))
    return rows


def analyze(job):
    rep, prior, root = job
    scenario, seed = rep.name.rsplit('-', 1)
    seed = int(seed)
    capture, fitted = capture_search(root, prior, seed)
    previous, _ = load_search(prior, 'urn-each')
    uniform, _ = load_search(prior, 'urn-no')
    pd.testing.assert_frame_equal(fitted, previous, check_exact=True)
    np.testing.assert_array_equal(capture['pairs'], fitted[KEY])
    truth = pd.read_csv(rep / 'sim/sim_expected_omega.tsv', sep='\t')
    frame = fitted.merge(truth, on=KEY, validate='one_to_one', sort=False)
    np.testing.assert_array_equal(frame[KEY], fitted[KEY])
    eligible = frame[EXPECTED].notna().all(axis=1).to_numpy()
    assert frame.loc[~eligible, PVALUE].isna().all()
    frame = frame.loc[eligible]
    probabilities, q_s, realized, parent, aa, rates = oracle(rep, frame, capture['codons'])
    np.testing.assert_array_equal(fitted[KEY], uniform[KEY])
    a, b = frame[KEY].to_numpy().T
    branch = capture['branch']
    np.testing.assert_allclose((branch @ branch.T)[a, b] / len(rates), uniform.loc[eligible, EXPECTED[1]],
                               rtol=1e-9, atol=7e-13)
    # Preserve the input side of exact near-zero decisions and ties.
    np.testing.assert_array_equal(capture['inputs'][KEY], fitted[KEY])
    frame[OBS + EXPECTED] = capture['inputs'].loc[eligible, OBS + EXPECTED].to_numpy()
    ecs = ecs_cases(frame, capture, q_s, realized, parent, aa, rates)
    tails = tail_cases(frame, capture, probabilities, eligible, seed)
    print(f'{rep.name}: {len(frame)} pairs checked against previous search', flush=True)
    return [[dict(scenario=scenario, seed=seed, **row) for row in rows] for rows in (ecs, tails)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('prior_runs', type=Path)
    parser.add_argument('work', type=Path)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--scenarios', nargs='+', choices=['baseline', 'rate_mixture'],
                        default=['baseline', 'rate_mixture'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 202, 303])
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir()
    names = [f'{scenario}-{seed}' for scenario in args.scenarios for seed in args.seeds]
    jobs = [(args.panel.resolve() / 'PEPC' / name, args.prior_runs.resolve() / name, work / name) for name in names]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(analyze, jobs))
    for label, rows in zip(('ecs', 'tails'), zip(*results)):
        pd.DataFrame([row for seed_rows in rows for row in seed_rows]).to_csv(
            work / f'{label}.tsv', sep='\t', index=False, float_format='%.12g')


if __name__ == '__main__':
    main()

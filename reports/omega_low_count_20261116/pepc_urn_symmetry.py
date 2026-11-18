"""Compare common site rates, empirical routes and branch draw totals for N and S."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from csubst import cli, empirical_urn, endpoint_io, omega, param, parser_misc, runtime
from csubst.omega_statistics import _calc_raw_omega, _calc_raw_rate

from compare_empirical_urn import background_omega, prepare, run_cli
from pepc_high_arity_accuracy import EXT, FITS, SCENARIOS, STEMS, check_joint_oracle, read_cb
from pepc_partial_pooling import SEEDS as PREVIOUS_SEEDS, panel_case, run_in

NEW_SEEDS = (1801, 1931, 2063, 2203, 2339, 2473, 2609, 2729)
REPLICATION_SEEDS = (3001, 3137, 3271, 3407, 3539, 3673, 3803, 3931)


def inputs(rep):
    sim = rep / 'sim'
    fit, = (rep / FITS[1] / 'iqtree').glob('*.iqtree')
    args = ['search', '--alignment_file', str(sim / 'sim.fa'), '--rooted_tree_file', str(sim / 'sim_true_asr.treefile'),
            '--expectation_method', 'urn', '--asrv', 'empirical_all', '--output_stat', 'any2spe',
            '--exclude_sister_pair', 'no', '--random_seed', rep.name.split('-')[1], '--b', 'no',
            '--threads', '1', '--blas_threads', '1', '--float_digit', '17']
    for ext in EXT:
        args.extend(['--iqtree_' + ext, str(fit.with_suffix('.' + ext))])
    run_cli(Path.cwd(), 'control.log', [*args, '--max_arity', '2', '--exhaustive_until', '2', '--outdir', 'pairs'])
    def context():
        g = param.get_global_parameters(cli._build_parser().parse_args([*args, '--max_arity', '6']))
        g = runtime.ensure_output_layout(g, create_dir=True)
        g = parser_misc.prepare_input_context(g, include_foreground=True, include_marginal=True,
                                              resolve_state_subset=True, prepare_state=True)
        endpoint_io.prepare(g)
        return g
    g = context()
    data, rates = empirical_urn.fit_all_branches(g, g['_endpoint_tensors']['N'], g['_endpoint_tensors']['S'])
    # Obtain missing source codon counts from a separate full-event pass.
    # Use the ordinary projected search fit for all other inputs.
    with patch.object(endpoint_io, '_use_projected_search', return_value=False):
        syn = context()['_endpoint_tensors']['S']
    ids = g['_asrv_training_ids']
    prov = json.loads(Path('pairs/csubst_urn_provenance.json').read_text())
    np.testing.assert_array_equal(ids, prov['training_branch_ids'])
    assert prov['asrv'] == 'empirical_all' and len(ids) == 138
    counts = np.zeros((syn.num_group, syn.num_state_from, syn.num_state_to))
    for b, _, group, a, d, v in syn.iter_branch_coordinates():
        if data.in_scope[b]:
            ind = (group * syn.num_state_from + a) * syn.num_state_to + d
            counts += np.bincount(ind, weights=v, minlength=counts.size).reshape(counts.shape)
    np.testing.assert_allclose(counts.sum(axis=1), data.syn_totals[ids].sum(axis=0), atol=1e-12)
    return g, data, rates, counts


def reference(rep, ids):
    pair = read_cb(Path('pairs'), 2)
    pair = pair[pair.branch_id_1.isin(ids) & pair.branch_id_2.isin(ids)]
    assert len(pair) == 8377
    q2 = pd.read_table(rep / 'sim/sim_expected_omega.tsv')
    q2.index = q2[['branch_id_1', 'branch_id_2']].astype(str).agg(','.join, axis=1)
    qk = pd.read_table(rep / 'truth/replay/sim_expected_omega.tsv').set_index('branches')
    def make(cb, ref, branches):
        ix = np.searchsorted(ids, branches)
        np.testing.assert_array_equal(ids[ix], branches)
        assert cb.index.is_unique and ref.index.is_unique
        return ix, cb, ref.loc[cb.index]
    sets = {('all', 2): make(pair, q2, pair[['branch_id_1', 'branch_id_2']].to_numpy())}
    for k in range(2, 7):
        branches = np.array(list(combinations(STEMS, k)))
        keys = [','.join(map(str, x)) for x in branches]
        target_cb = read_cb(rep / 'control/out', k)
        assert set(target_cb.index) == set(keys)
        cb = target_cb.loc[keys]
        if k == 2:
            for col in ('ECNany2spe', 'ECSany2spe', 'OCNany2spe', 'OCSany2spe'):
                np.testing.assert_allclose(cb[col], pair.loc[keys, col], rtol=1e-10, atol=1e-16)
        sets['stem', k] = make(cb, qk, branches)
    return sets


def raw_draws(profile, effort, tree_count):
    raw = profile.sum(axis=1) * effort[:, None]
    return raw * (tree_count / raw.sum())


def allocate(totals, weights):
    denom = weights.sum(axis=1)
    assert np.all((denom > 0) | (totals == 0))
    scale = np.divide(totals, denom, out=np.zeros_like(totals), where=denom > 0)
    return scale[:, None] * weights


def joint_draws(total, n, s, tree_n):
    # One fitted relative scale preserves both training channel totals when
    # conditionally normalizing N and S to the measured total on each branch.
    use = total > 0
    assert (n[use].sum(axis=1) > 0).all() and (s[use].sum(axis=1) > 0).all()
    odds = np.log(n[use].sum(axis=1)) - np.log(s[use].sum(axis=1))
    delta = brentq(lambda offset: total[use] @ expit(odds + offset) - tree_n, -50., 50.)
    count_n = np.zeros_like(total)
    count_n[use] = total[use] * expit(odds + delta)
    return allocate(count_n, n), allocate(total - count_n, s)


def overlap(g, sets, weights, counts):
    assert np.isfinite(counts).all() and (counts >= 0).all()
    out = {key: np.zeros(len(ix)) for key, (ix, _, _) in sets.items()}
    # Other than the all-pairs matrix product, each subset has only nine stems.
    for w, c in zip(weights.transpose(2, 0, 1), counts.T):
        if not c.any():
            continue
        profiles = omega.wallenius_inclusion_profiles(w, c, g)
        for key, (ix, _, _) in sets.items():
            out[key] += empirical_urn._overlaps(ix, profiles)
    return out


def fit_channels(g, data, rates, syn_routes, sets):
    ids = g['_asrv_training_ids']
    effort = data.effort[ids]
    eligible = data.eligible[ids]
    parent_aa = data.aa_parent[ids]
    route_n = data.route_totals[ids].sum(axis=0)
    seen_n = data.route_totals[ids].sum(axis=1)
    seen_s = np.concatenate([data.syn_totals[ids, i, :len(gr)] for i, gr in enumerate(data.groups)
                             if len(gr) > 1], axis=1)
    branch_n, branch_s = seen_n.sum(axis=1), seen_s.sum(axis=1)
    branch_total = data.ns_counts[ids].sum(axis=1)
    np.testing.assert_allclose(branch_n + branch_s, branch_total, rtol=1e-12)
    codon_state = g['state_cdn'][data.parents[ids]]
    fitted = {}
    for label, rate in [('fixed', rates.s_rate), ('cv', rates.n_rate)]:
        site = eligible * rate
        exposure = (effort @ (site[:, :, None] * parent_aa).sum(axis=1))
        route = empirical_urn._route_rates(route_n, exposure)
        n_profile = (parent_aa @ route) * site[:, :, None]
        n = raw_draws(n_profile, effort, branch_n.sum())
        fit_n = overlap(g, sets, n_profile, n)
        compat, learned = [], []
        for i, group in enumerate(data.groups):
            if len(group) == 1:
                continue
            parent = codon_state[:, :, group]
            compat.append((parent @ (1. - np.eye(len(group)))) * site[:, :, None])
            if label == 'cv':
                group_exposure = effort @ (parent * site[:, :, None]).sum(axis=1)
                group_rate = empirical_urn._route_rates(syn_routes[i, :len(group), :len(group)], group_exposure)
                learned.append((parent @ group_rate) * site[:, :, None])
        fit_s = overlap(g, sets, np.concatenate(compat, axis=2), seen_s)
        if label == 'fixed':
            fixed_n, fixed_s = fit_n, fit_s
            continue
        fitted['current'] = fit_n, fixed_s
        fitted['shared_fixed'] = fixed_n, fixed_s
        fitted['shared_cv'] = fit_n, fit_s
        # Move N to the S rule, including compatibility and observed destination counts.
        compat_n = (parent_aa @ (1. - np.eye(route_n.shape[0]))) * site[:, :, None]
        fitted['observed_both'] = overlap(g, sets, compat_n, seen_n), fit_s
        s_profile = np.concatenate(learned, axis=2)
        s = raw_draws(s_profile, effort, branch_s.sum())
        fitted['own_totals'] = (overlap(g, sets, n_profile, allocate(branch_n, n)),
                                overlap(g, sets, s_profile, allocate(branch_s, s)))
        fitted['separate_scales'] = fit_n, overlap(g, sets, s_profile, s)
        total_n, total_s = joint_draws(branch_total, n, s, branch_n.sum())
        np.testing.assert_allclose(total_n.sum(axis=1) + total_s.sum(axis=1), branch_total, rtol=1e-12)
        np.testing.assert_allclose([total_n.sum(), total_s.sum()], [branch_n.sum(), branch_s.sum()], rtol=1e-12)
        fitted['joint_totals'] = overlap(g, sets, n_profile, total_n), overlap(g, sets, s_profile, total_s)
    # On this complete PEPC panel both rates are mixtures of the same
    # empirical site profile and the uniform profile. Impose the existing
    # 0.5-per-site prior as the minimum uniform weight in the CV fit.
    support = eligible.sum(axis=0)
    assert (support > 0).all()
    site_mass = data.ns_counts[ids].sum(axis=0) * len(ids) / support
    uniform_floor = 0.5 / (site_mass.mean() + 0.5)
    empirical = site_mass / site_mass.mean()
    np.testing.assert_allclose((1-uniform_floor)*empirical + uniform_floor, rates.s_rate)
    np.testing.assert_allclose((1-rates.uniform)*empirical + rates.uniform, rates.n_rate)
    chosen = 'shared_fixed' if rates.uniform < uniform_floor else 'shared_cv'
    fitted['shared_floor'] = fitted[chosen]
    return fitted, rates.uniform, uniform_floor


def score_case(rep):
    scenario, seed = rep.name.rsplit('-', 1)
    g, data, rates, syn_routes = inputs(rep)
    sets = reference(rep, g['_asrv_training_ids'])
    fitted, cv_uniform, uniform_floor = fit_channels(g, data, rates, syn_routes, sets)
    rows = []
    for key, (_, cb, q) in sets.items():
        tn, ts = q.muNany2spe_null.to_numpy(), q.muSany2spe_null.to_numpy()
        mn, ms = q.muNany2spe_sim.to_numpy(), q.muSany2spe_sim.to_numpy()
        assert np.all((tn > 0) & (ts > 0) & (mn > 0) & (ms > 0))
        target = (mn / tn) / (ms / ts)
        on, os = cb.OCNany2spe.to_numpy(), cb.OCSany2spe.to_numpy()
        true_n, true_s = q.OCNany2spe_true.to_numpy(), q.OCSany2spe_true.to_numpy()
        tol = g['float_tol']
        supported = (on >= tol) & (os >= tol)
        true_positive = (true_n > 0) & (true_s > 0)
        for method, (all_n, all_s) in fitted.items():
            en, es = all_n[key], all_s[key]
            assert (en >= 0).all() and (es >= 0).all() and np.isfinite(en + es).all()
            omega_hat = _calc_raw_omega(_calc_raw_rate(on, en, tol), _calc_raw_rate(os, es, tol), tol)
            if method == 'current':
                np.testing.assert_allclose(en, cb.ECNany2spe, rtol=2e-10, atol=8e-18)
                np.testing.assert_allclose(es, cb.ECSany2spe, rtol=2e-10, atol=8e-18)
                np.testing.assert_allclose(omega_hat[supported], cb.omegaCany2spe.to_numpy()[supported], rtol=1e-8)
            def log_sse(obs, keep):
                with np.errstate(invalid='ignore'):
                    loss = (np.log1p(obs[keep]) - np.log1p(target[keep])) ** 2
                return loss.sum() if np.isfinite(loss).all() else np.inf
            with np.errstate(divide='ignore', invalid='ignore'):
                true_hat = (true_n / true_s) * (es / en)
                mean_hat = (mn / ms) * (es / en)
            rows.append(dict(scenario=scenario, seed=int(seed), scope=key[0], k=key[1], method=method,
                             cv_uniform=cv_uniform, uniform_floor=uniform_floor, combos=len(q), ref_n=tn.sum(), ref_s=ts.sum(), ec_n=en.sum(), ec_s=es.sum(),
                             abs_n=np.abs(en-tn).sum(), abs_s=np.abs(es-ts).sum(),
                             supported=supported.sum(), raw_sse=log_sse(omega_hat, supported),
                             true_supported=true_positive.sum(), true_sse=log_sse(true_hat, true_positive),
                             true_s_positive=(true_s > 0).sum(), mean_sse=log_sse(mean_hat, np.ones(len(q), bool)),
                             zero_n=(en == 0).sum(), zero_s=(es == 0).sum()))
    pd.DataFrame(rows).to_csv('results.tsv', sep='\t', float_format='%.15g', index=False)


def score_panel(panel, work, jobs):
    cases = sorted(p for p in panel.iterdir() if p.is_dir() and (p / 'sim/sim.fa').is_file())
    seeds = {int(c.name.rsplit('-', 1)[1]) for c in cases}
    assert seeds in (set(NEW_SEEDS), set(PREVIOUS_SEEDS), set(REPLICATION_SEEDS))
    assert {c.name for c in cases} == {f'{sc}-{s}' for sc in SCENARIOS for s in seeds}
    with ThreadPoolExecutor(jobs) as pool:
        futures = [pool.submit(run_in, work / c.name, Path(__file__).resolve(), 'one', c) for c in cases]
        for future in futures:
            future.result()
    pd.concat([pd.read_table(work / c.name / 'results.tsv') for c in cases], ignore_index=True).to_csv(
        work / 'results.tsv', sep='\t', index=False, float_format='%.13g')


def summarize(work):
    df = pd.read_table(work / 'results.tsv')
    rows = []
    for (scope, scenario, k), group in df.groupby(['scope', 'scenario', 'k']):
        methods = {method: block.sort_values('seed') for method, block in group.groupby('method')}
        base = methods['current']
        n = len(base)
        # Resample alignments as whole units, with the same weights for each
        # statistic and method. Branch combinations within a tree are dependent.
        resamples = np.random.default_rng(20261118).multinomial(n, np.full(n, 1/n), size=20000)
        for method, d in methods.items():
            np.testing.assert_array_equal(d[['seed', 'combos', 'supported', 'true_supported', 'ref_n', 'ref_s']],
                                          base[['seed', 'combos', 'supported', 'true_supported', 'ref_n', 'ref_s']])
            row = dict(scope=scope, scenario=scenario, k=k, method=method, seeds=n, combos=d.combos.sum(),
                       supported=d.supported.sum(), true_supported=d.true_supported.sum(),
                       true_s_positive=d.true_s_positive.sum(), zero_n=d.zero_n.sum(), zero_s=d.zero_s.sum(),
                       n_bias=d.ec_n.sum()/d.ref_n.sum(), s_bias=d.ec_s.sum()/d.ref_s.sum())
            for metric, num, den, power in [('n_wape','abs_n','ref_n',1), ('s_wape','abs_s','ref_s',1),
                                            ('omega','raw_sse','supported',.5),
                                            ('omega_true','true_sse','true_supported',.5),
                                            ('omega_mean','mean_sse','combos',.5)]:
                a, b = d[num].to_numpy(), base[num].to_numpy()
                if not d[den].sum():
                    continue
                row[metric] = (a.sum()/d[den].sum())**power
                if not (np.isfinite(a).all() and np.isfinite(b).all() and b.sum() > 0):
                    continue
                row[metric+'_rel'] = (a.sum()/b.sum())**power
                br = resamples @ b
                ok = br > 0
                ratios = ((resamples[ok] @ a)/br[ok])**power
                row[metric+'_lo'], row[metric+'_hi'] = np.quantile(ratios, [.05,.95])
            rows.append(row)
    pd.DataFrame(rows).to_csv(work / 'summary.tsv', sep='\t', float_format='%.10g', na_rep='NA', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('simulate', 'score', 'one', 'summarize'))
    parser.add_argument('panel', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--iqtree', default='iqtree')
    parser.add_argument('--replication', action='store_true', help='Simulate the independent validation for the smoothing floor.')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    panel = args.panel.resolve()
    if args.stage == 'simulate':
        sources = prepare(panel, 'PEPC')
        ratio = background_omega(sources)
        (panel / 'background-omega.txt').write_text(f'{ratio:.17g}\n')
        check_joint_oracle()
        with ThreadPoolExecutor(args.jobs) as pool:
            futures = [pool.submit(panel_case, panel, sources, ratio, args.iqtree, sc, seed)
                       for seed in (REPLICATION_SEEDS if args.replication else NEW_SEEDS) for sc in SCENARIOS]
            for future in futures:
                future.result()
    elif args.stage == 'summarize':
        summarize(panel)
    elif args.stage == 'one':
        score_case(panel)
    else:
        if args.work is None:
            parser.error('score requires --work')
        score_panel(panel, args.work.resolve(), args.jobs)
        summarize(args.work.resolve())


if __name__ == '__main__':
    main()

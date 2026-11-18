"""Site-validated partial pooling of PEPC synonymous destination counts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from csubst import cli, empirical_urn, endpoint_io, omega, param, parser_misc, runtime
from csubst.omega_statistics import _calc_raw_omega, _calc_raw_rate
from compare_empirical_urn import background_omega, fit_case, make_case, prepare
from pepc_high_arity_accuracy import EXT, FITS, SCENARIOS, SEEDS as PILOT_SEEDS, STEMS, check_joint_oracle, read_cb

SEEDS = (1103, 1217, 1321, 1433, 1549, 1657)
HERE = Path(__file__).resolve().parent


def run_in(directory, script, *args):
    directory.mkdir(parents=True)
    with (directory / 'job.log').open('w') as log:
        subprocess.run([sys.executable, str(script), *map(str, args)], cwd=directory,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def panel_case(work, sources, ratio, iqtree, scenario, seed):
    case = f'{scenario}-{seed}'
    rep = work / case
    make_case(sources, rep, scenario, seed, ratio)
    fit_case(rep, scenario, seed, FITS[1], iqtree)
    script = HERE / 'pepc_high_arity_accuracy.py'
    run_in(rep / 'truth', script, 'oracle', work, scenario, seed)
    run_in(rep / 'control', script, 'target', work, case, FITS[1], 'urn')
    print('Completed panel', case, flush=True)


FOLDS = 5
ALPHAS = np.r_[0., 2. ** np.arange(-8., 18.25, .25), np.inf]


def site_statistics(rep):
    sim = rep / 'sim'
    fit, = (rep / FITS[1] / 'iqtree').glob('*.iqtree')
    args = ['search', '--alignment_file', str(sim / 'sim.fa'), '--rooted_tree_file', str(sim / 'sim_true_asr.treefile'),
            '--expectation_method', 'urn', '--asrv', 'empirical_all', '--output_stat', 'any2spe',
            '--max_arity', '6', '--b', 'no']
    for ext in EXT:
        args.extend(['--iqtree_' + ext, str(fit.with_suffix('.' + ext))])
    g = param.get_global_parameters(cli._build_parser().parse_args(args))
    g = runtime.ensure_output_layout(g, create_dir=True)
    g = parser_misc.prepare_input_context(g, include_foreground=True, include_marginal=True,
                                          resolve_state_subset=True, prepare_state=True)
    endpoint_io.prepare(g)
    syn = g['_endpoint_tensors']['S']
    data, rates = empirical_urn.fit_all_branches(g, g['_endpoint_tensors']['N'], syn)
    train = g['_asrv_training_ids']
    cli_prov = json.loads((rep / 'control/out/csubst_urn_provenance.json').read_text())
    np.testing.assert_array_equal(train, cli_prov['training_branch_ids'])
    assert cli_prov['asrv'] == 'empirical_all' and len(train) == 138
    stems = np.searchsorted(train, STEMS)
    np.testing.assert_array_equal(train[stems], STEMS)
    sites = np.arange(syn.num_site)
    membership = np.concatenate([np.eye(FOLDS)[np.random.default_rng(seed).permutation(sites) % FOLDS]
                                 for seed in (20261118, 20261119)], axis=1)
    # Projected sparse columns have site as their fastest varying index.
    projection = syn.project('any2spe')[train].reshape((-1, len(sites))).tocsr()
    counts = (projection @ membership).reshape((*data.syn_totals[train].shape, membership.shape[1]))
    for partition in np.split(counts, 2, axis=-1):
        np.testing.assert_allclose(partition.sum(axis=-1), data.syn_totals[train], rtol=1e-11, atol=1e-14)
    eligible = data.eligible[train]
    n_eligible = eligible @ membership
    group_counts, opportunities, stem_weights = [], [], []
    for i, group in enumerate(data.groups):
        if len(group) < 2:
            continue
        parent = g['state_cdn'][data.parents[train, None, None], sites[None, :, None], group]
        # The S rate is site-local up to a common scalar, which cancels in all
        # per-destination coefficients. Retain the CLI's scale for final ECS.
        weights = (parent @ (1. - np.eye(len(group)))) * (rates.s_rate * eligible)[:, :, None]
        opportunities.append(weights.transpose(0, 2, 1) @ membership)
        group_counts.append(counts[:, i, :len(group)])
        stem_weights.append(weights[stems])
    compact = np.concatenate(group_counts, axis=1)
    np.testing.assert_allclose(compact[:, :, :FOLDS].sum(axis=(1, 2)), data.syn_totals[train].sum(axis=(1, 2)))
    return g, stems, compact, np.concatenate(opportunities, axis=1), n_eligible, np.concatenate(stem_weights, axis=2)


def pooled_prob(counts, opp, eligible):
    s = counts.sum(axis=1)
    effort = np.divide(s, eligible, out=np.zeros_like(s), where=eligible > 0.)
    exposure = effort @ opp
    mass = counts.sum(axis=0)
    assert np.all((exposure > 0.) | (mass == 0.))
    coef = np.divide(mass, exposure, out=np.zeros_like(mass), where=exposure > 0.)
    raw = opp * coef
    denom = raw.sum(axis=1, keepdims=True)
    assert (denom > 0.).all()  # Every eligible PEPC branch has parental opportunity.
    return raw / denom


def dest_prob(counts, pool, alpha):
    if np.isinf(alpha):
        return pool
    denom = counts.sum(axis=1, keepdims=True) + alpha
    return np.divide(counts + alpha * pool, denom, out=pool.copy(), where=denom > 0.)


def select_alpha(counts, opp, eligible):
    losses = np.zeros(len(ALPHAS))
    for f in range(FOLDS):
        other = np.arange(FOLDS) != f
        train = counts[:, :, other].sum(axis=2)
        pool = pooled_prob(train, opp[:, :, other].sum(axis=2), eligible[:, other].sum(axis=1))
        held = counts[:, :, f]
        positive = held > 0.
        for j, alpha in enumerate(ALPHAS):
            p = dest_prob(train, pool, alpha)
            # Common scoring-only floor makes the endpoints comparable under
            # unseen destinations and preserves row normalization.
            prob = (1. - 1e-12) * p[positive] + 1e-12 / p.shape[1]
            losses[j] -= held[positive] @ np.log(prob)
    losses /= counts.sum()
    i = np.argmin(losses)
    return ALPHAS[i], losses[[0, i, -1]]


def score_case(rep):
    scenario, seed = rep.name.split('-')
    g, stems, counts, opp, eligible, weights = site_statistics(rep)
    second_alpha, _ = select_alpha(counts[..., FOLDS:], opp[..., FOLDS:], eligible[..., FOLDS:])
    counts, opp, eligible = counts[..., :FOLDS], opp[..., :FOLDS], eligible[..., :FOLDS]
    alpha, losses = select_alpha(counts, opp, eligible)
    train = counts.sum(axis=-1)
    pool = pooled_prob(train, opp.sum(axis=-1), eligible.sum(axis=-1))[stems]
    branch = train[stems]
    totals = branch.sum(axis=1, keepdims=True)
    weight = np.ones_like(totals) if np.isinf(alpha) else alpha / (alpha + totals)
    second_weight = np.ones_like(totals) if np.isinf(second_alpha) else second_alpha / (second_alpha + totals)
    diagnostics = dict(scenario=scenario, seed=seed, alpha=alpha, weight_min=weight.min(),
                       weight_median=np.median(weight), weight_max=weight.max(),
                       second_alpha=second_alpha, second_weight_median=np.median(second_weight),
                       cv_loss_observed=losses[0], cv_loss_selected=losses[1], cv_loss_pooled=losses[2])
    rows = []
    # Truth is opened only after prediction strength and profiles are fitted.
    q = pd.read_table(rep / 'truth/replay/sim_expected_omega.tsv').set_index('branches')
    for method, a in [('observed', 0.), ('partial_cv', alpha), ('pooled', np.inf)]:
        draws = totals * dest_prob(branch, pool, a)
        np.testing.assert_allclose(draws.sum(axis=1), totals.ravel(), rtol=1e-13)
        assert np.all(np.isfinite(draws) & (draws >= 0.))
        profiles = [omega.wallenius_inclusion_profiles(w, d, g)
                    for w, d in zip(weights.transpose(2, 0, 1), draws.T) if d.any()]
        for k in range(2, 7):
            ix = np.array(list(combinations(range(len(stems)), k)))
            keys = [','.join(map(str, b)) for b in np.array(STEMS)[ix]]
            combo = read_cb(rep / 'control/out', k)
            assert set(keys) == set(combo.index)
            cb = combo.loc[keys]
            truth = q.loc[keys]
            true_s = truth.muSany2spe_null.to_numpy()
            ec = sum(omega._calc_cb_site_overlap(ix, profile, np.float64) for profile in profiles)
            assert (true_s > 0.).all() and (ec >= 0.).all() and np.isfinite(ec).all()
            if method == 'observed':
                np.testing.assert_allclose(ec, cb.ECSany2spe, rtol=1e-10, atol=6e-18)
            # Use the production rate/ratio definitions on common supported rows.
            tol = g['float_tol']
            obs_s = cb.OCSany2spe.to_numpy()
            obs_n = cb.OCNany2spe.to_numpy()
            supported = (obs_n >= tol) & (obs_s >= tol)
            d_n = _calc_raw_rate(obs_n, cb.ECNany2spe, tol)
            raw = _calc_raw_omega(d_n, _calc_raw_rate(obs_s, ec, tol), tol)
            if supported.any() and method == 'observed':
                np.testing.assert_allclose(raw[supported], cb.omegaCany2spe.to_numpy()[supported], rtol=1e-8)
            oracle_omega = ((truth.muNany2spe_sim / truth.muNany2spe_null)
                            / (truth.muSany2spe_sim / truth.muSany2spe_null)).to_numpy()
            assert np.isfinite(raw[supported]).all()
            rows.append(dict(scenario=scenario, seed=seed, k=k, method=method, combos=len(ix),
                             ecs_sum=ec.sum(), target_sum=true_s.sum(), abs_err_sum=abs(ec - true_s).sum(),
                             true_oc_s_positive=(truth.OCSany2spe_true > 0).sum(), omega_supported=supported.sum(),
                             omega_log1p_sse=np.square(np.log1p(raw[supported]) - np.log1p(oracle_omega[supported])).sum()))
    pd.DataFrame(rows).to_csv('scores.tsv', sep='\t', index=False)
    pd.DataFrame([diagnostics]).to_csv('fit.tsv', sep='\t', index=False)


def score_all(work, cases, jobs):
    with ThreadPoolExecutor(jobs) as pool:
        futures = []
        for scenario, seed in cases:
            rep = work / f'{scenario}-{seed}'
            futures.append(pool.submit(run_in, rep / 'scoring', Path(__file__).resolve(), 'one', rep))
        for job in futures:
            job.result()
    summarize(work, cases)


def summarize(work, cases):
    for name in ('scores', 'fit'):
        per_case = [pd.read_table(work / f'{sc}-{s}/scoring/{name}.tsv')
                    for sc, s in cases]
        pd.concat(per_case, ignore_index=True).to_csv(work / f'{name}.tsv', sep='\t', float_format='%.12g', index=False)

    scores = pd.read_table(work / 'scores.tsv')
    metrics = scores.groupby(['scenario', 'k', 'method']).sum(numeric_only=True).drop(columns='seed')
    metrics['ecs_ratio'] = metrics.ecs_sum / metrics.target_sum
    metrics['ecs_wape'] = metrics.abs_err_sum / metrics.target_sum
    metrics['omega_log1p_rmse_supported'] = np.sqrt(
        metrics.omega_log1p_sse / metrics.omega_supported.where(metrics.omega_supported > 0))
    metrics.to_csv(work / 'summary.tsv', sep='\t', float_format='%.12g', na_rep='NA')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('panel', 'score', 'one', 'pilot-mixed'))
    p.add_argument('work', type=Path)
    p.add_argument('--iqtree', default='iqtree')
    p.add_argument('--pilot-panel', type=Path)
    p.add_argument('--pilot-high-k', type=Path)
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()
    work = args.work.resolve()
    cases = [(scenario, seed) for seed in SEEDS for scenario in SCENARIOS]
    if args.stage == 'panel':
        check_joint_oracle()
        sources = prepare(work, 'PEPC')
        ratio = background_omega(sources)
        (work / 'background-omega.txt').write_text(f'{ratio:.17g}\n')
        with ThreadPoolExecutor(args.jobs) as pool:
            futures = [pool.submit(panel_case, work, sources, ratio, args.iqtree, *case) for case in cases]
            for job in futures:
                job.result()
    elif args.stage == 'pilot-mixed':
        if args.pilot_panel is None or args.pilot_high_k is None:
            p.error('pilot-mixed requires --pilot-panel and --pilot-high-k')
        cases = [('rate_mixture', seed) for seed in PILOT_SEEDS]
        for sc, seed in cases:
            rep = work / f'{sc}-{seed}'
            rep.mkdir()
            panel = args.pilot_panel.resolve() / rep.name
            prior = args.pilot_high_k.resolve() / rep.name
            for name, source in [('sim', panel / 'sim'), (FITS[1], panel / FITS[1]), ('truth', prior),
                                 ('control', prior / FITS[1] / 'urn/target')]:
                (rep / name).symlink_to(source, target_is_directory=True)
        score_all(work, cases, args.jobs)
    elif args.stage == 'one':
        score_case(work)
    else:
        score_all(work, cases, args.jobs)


if __name__ == '__main__':
    main()

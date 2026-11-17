"""Factor empirical-urn ECN into site weights and destination totals on PEPC."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from asr_bootstrap import omega_score
from compare_codon_models import MODELS
from compare_urn_asr import fit_parser, fit_tasks
from pilot import EXPECTED, KEY, OBS
from urn_accuracy import load_search
from urn_empirical_routes import crossfit_groups, learned_rates, parent_evidence
from urn_parent_opportunity import load_fitted_events, metrics, opportunity_features, urn_expected


def route_weights(parent_aa, site_rate, train, effort, counts, branches, shape=.5):
    exposure = np.einsum('b,s,bsa->a', effort, site_rate, parent_aa[train])
    route = learned_rates(counts, exposure, shape)
    return (parent_aa[branches] @ route) * site_rate[None, :, None]


def normalized_site_rate(mass):
    return mass / mass.mean()


def cross_validated_site_rate(branch_site_counts):
    """Estimate the uniform mixture weight by leave-one-training-branch-out scoring."""
    site_counts = branch_site_counts.sum(axis=0)
    others = site_counts - branch_site_counts
    empirical = others / others.sum(axis=1, keepdims=True)
    uniform = 1. / len(site_counts)
    supported = branch_site_counts > 0.
    score_mass = branch_site_counts[supported]
    reference = empirical[supported]

    def score(weight):
        p = (1. - weight) * reference + weight * uniform
        if not p.all():
            return np.inf
        return -float(score_mass @ np.log(p))

    result = minimize_scalar(score, bounds=(0., 1.), method='bounded', options={'xatol': 1e-7})
    if not result.success:
        raise ValueError('Could not fit the site-weight regularization.')
    weight = min((0., result.x, 1.), key=score)
    return (1. - weight) * normalized_site_rate(site_counts) + weight, weight


def pooled_ns_totals(eval_weights, train_weights, train_effort, n_total, eval_effort):
    train_flux = train_weights.sum(axis=(1, 2))
    scale = n_total.sum() / (train_effort @ train_flux)
    return eval_weights.sum(axis=1) * (scale * eval_effort)[:, None]


def group_expectations(pairs, train, parent, parent_aa, route_counts, aa, empirical_site_counts, fitted_site_rate,
                       actual_site_rate, observed_totals, predicted_n):
    ids, inverse = np.unique(pairs, return_inverse=True)
    pairs = inverse.reshape(pairs.shape)
    n_total, s_total = observed_totals
    n_mass, s_mass = empirical_site_counts
    joint_mass = n_mass[train] + s_mass[train]
    site_mass = joint_mass.sum(axis=0)
    effort = joint_mass.mean(axis=1)
    count = route_counts[train].sum(axis=0)
    rate = normalized_site_rate(site_mass + .5)
    n_weights = route_weights(parent_aa, rate, train, effort, count, ids)
    _, s_weights = opportunity_features(parent[ids] * rate[None, :, None], aa)
    es = urn_expected(pairs, s_weights, s_total[ids])

    # Conditional expectation given only the observed branch-wide N count.
    destination_mass = n_weights.sum(axis=1)
    destination_share = destination_mass / destination_mass.sum(axis=1, keepdims=True)
    pooled_dest = destination_share * n_total[ids].sum(axis=1, keepdims=True)
    # Fit the N fraction from the other branches and condition on the branch-wide N+S count.
    train_weights = route_weights(parent_aa, rate, train, effort, count, train)
    branch_effort = (n_mass[ids] + s_mass[ids]).mean(axis=1)
    pooled_ns = pooled_ns_totals(n_weights, train_weights, effort, n_total[train], branch_effort)
    cv_rate, cv_uniform_weight = cross_validated_site_rate(joint_mass)

    n_cases = {
        'empirical_current': (n_weights, n_total[ids]),
        'empirical_pool_destinations': (n_weights, pooled_dest),
        'empirical_pool_ns': (n_weights, pooled_ns),
        'fitted_q_totals': (n_weights, predicted_n[ids].sum(axis=1)),
        'fitted_q_profile': (predicted_n[ids], n_total[ids]),
        'empirical_unpooled_routes': (route_weights(parent_aa, rate, train, effort, count, ids, shape=0.),
                                      n_total[ids]),
    }
    rate_cases = {
        'empirical_no_site_pseudocount': normalized_site_rate(site_mass),
        'empirical_s_site_counts': normalized_site_rate(s_mass[train].sum(axis=0) + .5),
        'empirical_cv_site_rate': cv_rate,
        'fitted_site_rates': fitted_site_rate,
        'known_site_rates': actual_site_rate,
    }
    for name, weights in rate_cases.items():
        n_cases[name] = (route_weights(parent_aa, weights, train, effort, count, ids), n_total[ids])
    for rate_name, label in [('empirical_cv_site_rate', 'empirical_cv_pool_ns'),
                             ('empirical_no_site_pseudocount', 'empirical_raw_pool_ns')]:
        profile = n_cases[rate_name][0]
        train_profile = route_weights(parent_aa, rate_cases[rate_name], train, effort, count, train)
        n_cases[label] = (profile, pooled_ns_totals(profile, train_profile, effort, n_total[train], branch_effort))
    ecn = {tag: urn_expected(pairs, weights, mass) for tag, (weights, mass) in n_cases.items()}
    # Test the two corrections together, conditioning on the branch-wide N total.
    fit_weights = n_cases['fitted_site_rates'][0]
    fit_mass = fit_weights.sum(axis=1)
    fit_dest = fit_mass / fit_mass.sum(axis=1, keepdims=True) * n_total[ids].sum(axis=1, keepdims=True)
    ecn['fitted_rates_pool_destinations'] = urn_expected(pairs, fit_weights, fit_dest)
    return ecn, es, {'empirical_current': rate, **rate_cases}, cv_uniform_weight


def analyze(task):
    panel, rep, label, source, outdir, report, tree_path = task
    table, meta = load_search(source, outdir)
    assert meta['parameter_source']['model'] == MODELS[label]
    frame, _, model, tips, aa, observed, predicted = load_fitted_events(rep, table, meta, report, tree_path)
    pairs = frame[KEY].to_numpy()
    training = np.unique(pairs)
    membership, parent, parent_rate, routes = parent_evidence(model, tips, aa, training)
    parent_aa = parent @ membership
    rate_by_branch = parent_rate[training].sum(axis=-1)
    np.testing.assert_allclose(rate_by_branch, np.broadcast_to(rate_by_branch[0], rate_by_branch.shape), atol=1e-10)
    fitted_rate = normalized_site_rate(rate_by_branch[0])
    site_counts = [channel.sum(axis=2) for channel in observed]
    totals = [channel.sum(axis=1) for channel in observed]
    np.testing.assert_allclose(routes[training].sum(axis=1), totals[0][training], rtol=1e-10, atol=1e-12)
    actual = pd.read_csv(rep / 'sim/sim_true_asr.rate', sep='\t')['C_Rate'].to_numpy()
    assert actual.shape == fitted_rate.shape
    scenario, seed = rep.name.rsplit('-', 1)
    keys = dict(panel=panel, scenario=scenario, seed=int(seed), model=label)

    methods = {'codon_model': frame[EXPECTED].to_numpy()}
    rate_rows = []
    cv_weights = []
    for fold, (keep, train) in enumerate(crossfit_groups(pairs, training)):
        en, es, rates, cv_weight = group_expectations(pairs[keep], train, parent, parent_aa, routes, aa, site_counts,
                                                     fitted_rate, normalized_site_rate(actual), totals, predicted[0])
        cv_weights.append(cv_weight)
        for name, values in en.items():
            if name not in methods:
                methods[name] = np.full((len(frame), 2), np.nan)
            methods[name][keep] = np.column_stack((values, es))
        if scenario == 'rate_mixture':
            for name, rate in rates.items():
                for true_rate in np.unique(actual):
                    mask = actual == true_rate
                    rate_rows.append(dict(**keys, fold=fold, method=name, true_rate=true_rate, n_sites=int(mask.sum()),
                                          estimated_rate=rate[mask].mean(), estimated_variance=rate[mask].var()))

    assert all(np.isfinite(values).all() for values in methods.values())
    observed_pairs = frame[OBS].to_numpy()
    common = np.logical_and.reduce([np.isfinite(omega_score(observed_pairs, x)) for x in methods.values()])
    foreground = (frame['foreground_1'].eq(1) & frame['foreground_2'].eq(1)).to_numpy()
    select = foreground if scenario == 'biased' else np.ones(len(frame), dtype=bool)
    print(f'{panel}/{rep.name}/{label}: evaluated {select.sum()} pairs', flush=True)
    rows = [dict(**keys, method=name, cv_uniform_weight=np.mean(cv_weights) if name.startswith('empirical_cv_') else np.nan,
                 **metrics(frame, values, select, common)) for name, values in methods.items()]
    return rows, rate_rows


def main():
    parser = fit_parser(__doc__)
    parser.add_argument('--site-rates-output', required=True)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(analyze, fit_tasks(args)))
    for i, path in enumerate((args.output, args.site_rates_output)):
        pd.DataFrame(row for result in results for row in result[i]).to_csv(path, sep='\t', index=False,
                                                                           float_format='%.12g')


if __name__ == '__main__':
    main()

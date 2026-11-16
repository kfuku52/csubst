"""Pilot on count support for pairwise omegaC under homogeneous simulation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binom


PANEL_REPORT = Path(__file__).resolve().parents[1] / 'simulate_sensitivity_20261116'
HOMOGENEOUS = {
    'PGK': {
        'baseline': 520, 'short_alignment': 120, 'long_alignment': 1400,
        'short_branches': 520, 'long_branches': 520, 'strong_constraint': 520, 'weak_constraint': 520,
    },
    'PEPC': {'baseline': 520, 'long_branches': 520},
}
KEY = ['branch_id_1', 'branch_id_2']
NULL = ['muNany2spe_null', 'muSany2spe_null']
OBS = ['OCNany2spe', 'OCSany2spe']
EXPECTED = ['ECNany2spe', 'ECSany2spe']
TRUTH = ['OCNany2spe_true', 'OCSany2spe_true']
ALPHA_TAIL = .025


def upper_tail_cutoff(total, p):
    cutoff = binom.isf(ALPHA_TAIL, total, p)
    # A confidence endpoint equal to the null stays inside the interval.
    return cutoff + (binom.sf(cutoff, total, p) == ALPHA_TAIL)


def integrate_count_law(n_sites, p_n, p_s):
    """Under IID sites, condition on the total N-or-S count."""
    support = p_n + p_s
    if not ((p_n > 0) & (p_s > 0) & (4 * p_n + p_s < 1)).all():
        raise ValueError('Invalid probabilities for the fourfold-N sensitivity check.')
    p = p_n / support
    alt_support = 4 * p_n + p_s
    alt_p = 4 * p_n / alt_support
    # Drop at most one part in a trillion from either count distribution.
    max_total = int(binom.isf(1e-12, n_sites, alt_support).max())
    total = np.arange(max_total + 1)[None, :]
    p, alt_p = p[:, None], alt_p[:, None]
    pmf = binom.pmf(total, n_sites, support[:, None])
    low = binom.ppf(ALPHA_TAIL, total, p)
    high = upper_tail_cutoff(total, p)
    cover = (pmf * (binom.cdf(high, total, p) - binom.cdf(low - 1, total, p))).sum(axis=1)
    alt_pmf = binom.pmf(total, n_sites, alt_support[:, None])
    alt_power = (alt_pmf * binom.sf(high, total, alt_p)).sum(axis=1)
    n_zero = np.exp(n_sites * np.log1p(-p_n))
    s_zero = np.exp(n_sites * np.log1p(-p_s))
    both_zero = np.exp(n_sites * np.log1p(-support))
    return dict(
        exact_null_coverage=cover,
        truth_p_both_zero=both_zero,
        truth_p_both_positive=np.clip(1 - n_zero - s_zero + both_zero, 0, 1),
        power_n_fourfold=alt_power,
    )


def count_interval_at_null(n, s, e_n, e_s):
    """Test interval coverage on the binomial scale, without odds overflow."""
    lower_p = np.zeros(len(n))
    upper_q = np.zeros(len(s))
    has_n, has_s = n > 0, s > 0
    lower_p[has_n] = beta.ppf(ALPHA_TAIL, n[has_n], s[has_n] + 1)
    # Work with the complementary probability to avoid cancellation near 1.
    upper_q[has_s] = beta.ppf(ALPHA_TAIL, s[has_s], n[has_s] + 1)
    total = e_n + e_s
    covers_one = (lower_p <= e_n / total) & (upper_q <= e_s / total)
    return covers_one, has_n & has_s


def support_bin(reference_s):
    return pd.cut(reference_s, [-np.inf, .1, 1, 5, np.inf],
                  labels=('<=0.1', '0.1-1', '1-5', '>5'))


def pair_data(rep, fit):
    manifest = json.loads((fit / 'csubst_search_run.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError(f'Incomplete fitted run in {fit}')
    sim = pd.read_csv(rep / 'sim/sim_expected_omega.tsv', sep='\t')
    search = pd.read_csv(fit / 'csubst_cb_2.tsv', sep='\t')
    frame = sim.merge(search[KEY + OBS + EXPECTED + ['omegaCany2spe']], on=KEY,
                      how='outer', indicator=True, validate='one_to_one')
    if not frame.pop('_merge').eq('both').all():
        raise ValueError(f'Missing branch pairs in {rep}')
    frame = frame.dropna(subset=OBS + EXPECTED).copy()
    if (frame[NULL + EXPECTED] <= 0).any().any():
        raise ValueError(f'Nonpositive exposures in {rep}')
    if not frame['omegaCany2spe_expected'].eq(1).all():
        raise ValueError(f'Expected a homogeneous null in {rep}')
    frame['s_bin'] = support_bin(frame[NULL[1]])
    return frame


def summarize(panel, recovered):
    repairs = pd.read_csv(PANEL_REPORT / 'omega_recovery_by_seed.tsv', sep='\t')
    replacements = {('PGK', r.scenario, r.seed) for r in repairs.itertuples(index=False)}
    exact, exact_joint, pipeline = [], [], []
    for dataset, scenarios in HOMOGENEOUS.items():
        for scenario, sites in scenarios.items():
            replicates = sorted((panel / dataset).glob(f'{scenario}-[0-9]*'))
            if len(replicates) != (4 if dataset == 'PGK' else 3):
                raise ValueError(f'Missing replicates for {dataset}/{scenario}')
            baseline = None
            for rep in replicates:
                seed = int(rep.name.rsplit('-', 1)[1])
                repaired = (dataset, scenario, seed) in replacements
                fit = recovered / rep.name / 'csubst_search' if repaired else rep / 'default'
                frame = pair_data(rep, fit)
                exposures = frame.set_index(KEY).sort_index()[NULL]
                if baseline is None:
                    baseline = exposures
                    probs = exposures / sites
                    stats = integrate_count_law(sites, *probs.to_numpy().T)
                    unique = exposures.assign(**stats, n_bin=support_bin(exposures[NULL[0]]),
                                              s_bin=support_bin(exposures[NULL[1]]))
                    for keys, output in ((['s_bin'], exact), (['s_bin', 'n_bin'], exact_joint)):
                        grouped = unique.groupby(keys, observed=True)
                        means = grouped[list(stats)].mean()
                        means['pairs'] = grouped.size()
                        output.append(means.reset_index().assign(dataset=dataset, scenario=scenario, sites=sites))
                else:
                    pd.testing.assert_frame_equal(exposures, baseline)
                for method, obs_keys, exp_keys in (
                    ('true_counts_q', TRUTH, NULL),
                    ('true_counts_fitted_E', TRUTH, EXPECTED),
                    ('fractional_counts_q', OBS, NULL),
                    ('fractional_counts_fitted_E', OBS, EXPECTED),
                ):
                    obs, exp = frame[obs_keys].to_numpy(), frame[exp_keys].to_numpy()
                    covers_one, finite_two_sided = count_interval_at_null(*obs.T, *exp.T)
                    truth_both_zero = frame[TRUTH].eq(0).all(axis=1).to_numpy()
                    comparison = frame[['s_bin']].assign(
                        pair_rows=1, covers_one=covers_one,
                        finite_two_sided=finite_two_sided, truth_both_zero=truth_both_zero,
                        finite_two_sided_with_truth_zero=finite_two_sided & truth_both_zero,
                        finite_search_omega_with_truth_zero=np.isfinite(frame.omegaCany2spe) & truth_both_zero,
                    )
                    stats = comparison.groupby('s_bin', observed=True).sum().reset_index()
                    pipeline.append(stats.assign(dataset=dataset, scenario=scenario, seed=seed, method=method))
    return {name: pd.concat(rows, ignore_index=True) for name, rows in (
        ('oracle_support.tsv', exact), ('oracle_joint_support.tsv', exact_joint), ('pipeline_support.tsv', pipeline),
    )}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('recovered', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, table in summarize(args.panel, args.recovered).items():
        table.to_csv(args.output / filename, sep='\t', index=False, float_format='%.10g')


if __name__ == '__main__':
    main()

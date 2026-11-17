"""Test a fixed-model null with the same endpoint inference as observed PEPC."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

from csubst import endpoint, endpoint_io, ete, genetic_code, main_simulate, omega, omega_statistics, tree
from csubst._vendor.pyvolve.evolver import ZERO

from pilot import EXPECTED, KEY, NULL, OBS
from replay_posterior import fitted_generator, load_tips, oracle_generator
from urn_causes import upper_tail


SCENARIOS = ('baseline', 'rate_mixture')
SITE_COUNT = 520
TOL = 1e-9
CONDITION_SEEDS = {'calibration': 10000, 'null': 20000, 'alternative': 30000}


def projection(codons):
    code = dict((codon, aa) for aa, codon in genetic_code.get_codon_table(1))
    amino, aa_ids = np.unique([code[c] for c in codons], return_inverse=True)
    indices = [np.flatnonzero(aa_ids == i) for i in range(amino.size)]
    config = dict(amino_acid_orders=amino, synonymous_indices=dict(zip(amino, indices)),
                  max_synonymous_size=max(map(len, indices)))
    mapping = np.eye(amino.size)[aa_ids]
    args = (config, ['N', 'S'], {'N': mapping}, {'any2spe'})
    return (endpoint_io._projection_transform(*args),
            endpoint_io._projection_transform(*args, predictive=True)), indices, aa_ids


def feature_buffers(model, size, n_amino):
    return np.zeros((model.parents.size, size, n_amino)), np.zeros((model.parents.size, size, model.pi.size))


def edge_features(model, tips, transforms, syn_indices):
    size = next(iter(tips.values())).shape[0]
    features = [feature_buffers(model, size, len(syn_indices)) for _ in range(2)]
    records = model.iter_blocks(tips, block_size=64, joint=False, transform=transforms[0],
                                predictive=True, predictive_transform=transforms[1])
    for record in records:
        if record.parent < 0:
            continue
        sl = slice(record.start, record.stop)
        b = record.child
        for (n, s), block in zip(features, (record.reduced, record.reduced_predictive)):
            n[b, sl] = block['N', 'any2spe'][:, 0]
            values = block['S', 'any2spe']
            for group, idx in enumerate(syn_indices):
                s[b, sl, idx] = values[:, group, :len(idx)].T
    return features


def pair_counts(features, pairs):
    a, b = pairs.T
    flat = features.reshape(len(features), -1)
    return (flat @ flat.T)[a, b]


def omega_score(obs, expected):
    n, s = omega_statistics._calc_raw_rate(obs, expected, TOL).T
    return omega_statistics._calc_raw_omega(n, s, TOL)


def check_search(panel, previous):
    """Verify that the projection reproduces saved search on the same fit."""
    for scenario in SCENARIOS:
        name = f'{scenario}-101'
        rep, fit = panel / name, previous / name
        source = fit / 'codon_model'
        meta = json.loads((source / 'csubst_endpoint_model.json').read_text())['codon']
        tr = tree.add_numerical_node_labels(ete.PhyloNode(str(fit / 'tree.nwk'), format=1))
        parents, _ = endpoint_io._tree_arrays({'tree': tr}, structural=False)
        q, pi, codons, _ = fitted_generator(rep, fit / 'fit.iqtree', meta)
        model = endpoint.EndpointModel(parents, meta['branch_lengths'], q, pi, meta['rates'], meta['weights'])
        transforms, syn_indices, _ = projection(codons)
        observed, predicted = edge_features(model, load_tips(rep, codons, tr), transforms, syn_indices)
        reference = pd.read_csv(source / 'csubst_cb_2.tsv', sep='\t')
        pairs = reference[KEY].to_numpy()
        for features, columns in ((observed, OBS), (predicted, EXPECTED)):
            valid = reference[columns].notna().all(axis=1)
            counts = np.column_stack([pair_counts(values, pairs[valid]) for values in features])
            np.testing.assert_allclose(counts, reference.loc[valid, columns], rtol=1e-9, atol=6e-12)
        print(f'{name}: matched saved search observed and expected counts', flush=True)


def cdf(matrices):
    if np.min(matrices) < -1e-12:
        raise ValueError('Negative transition probability.')
    np.testing.assert_allclose(matrices.sum(axis=-1), 1., rtol=1e-10, atol=1e-10)
    stop = np.cumsum(matrices.clip(0.), axis=-1)
    stop /= stop[..., -1:]
    stop[..., -1] = 1
    return stop


class Panel:
    def __init__(self, panel, scenario):
        rep = panel / f'{scenario}-101'
        frame = pd.read_csv(rep / 'sim/sim_expected_omega.tsv', sep='\t')
        forward = pd.read_csv(panel / 'biased-101/sim/sim_expected_omega.tsv', sep='\t')
        frame = frame.merge(forward[KEY + ['foreground_1', 'foreground_2']], on=KEY, validate='one_to_one',
                            suffixes=('', '_alt'))
        tr = tree.add_numerical_node_labels(ete.PhyloNode(str(panel / 'inputs/PEPC.tree.nwk'), format=1))
        parents, lengths = endpoint_io._tree_arrays({'tree': tr}, structural=False)
        simulated_tr = tree.add_numerical_node_labels(ete.PhyloNode(str(rep / 'sim/sim_true_asr.treefile'), format=1))
        simulated_parents, simulated_lengths = endpoint_io._tree_arrays({'tree': simulated_tr}, structural=False)
        np.testing.assert_array_equal(parents, simulated_parents)
        np.testing.assert_allclose(lengths, simulated_lengths, rtol=2e-5, atol=2e-8)
        lengths[lengths <= ZERO] = 0
        self.codons = main_simulate.get_pyvolve_codon_order()
        self.transforms, self.syn_indices, self.aa = projection(self.codons)
        q, pi = oracle_generator(rep, self.codons)
        actual_rates = pd.read_csv(rep / 'sim/sim_true_asr.rate', sep='\t')['C_Rate'].to_numpy()
        assert len(actual_rates) == SITE_COUNT
        rates, self.assignments = np.unique(actual_rates, return_inverse=True)
        self.site_groups = [np.flatnonzero(self.assignments == i) for i in range(len(rates))]
        self.models = [endpoint.EndpointModel(parents, lengths, q, pi, rates=(r,)) for r in rates]
        self.order, self.parents, self.pi = self.models[0].order, parents, pi
        self.eye = np.eye(len(pi))
        self.leaves = sorted(self.models[0].leaves)
        eligible = (parents >= 0) & (parents != self.models[0].root)
        a, b = frame[KEY].to_numpy().T
        kept = eligible[a] & eligible[b]
        frame = frame.loc[kept].reset_index(drop=True)
        self.pairs = frame[KEY].to_numpy()
        self.target = (frame['foreground_1_alt'].eq(1) & frame['foreground_2_alt'].eq(1)).to_numpy()
        self.expected = frame[NULL].to_numpy()
        assert (self.expected > 0).all() and self.target.any()
        branches = set(frame.loc[frame.foreground_1_alt.eq(1), 'branch_id_1'])
        branches.update(frame.loc[frame.foreground_2_alt.eq(1), 'branch_id_2'])
        self.foreground = np.array(sorted(branches), dtype=int)
        self.foreground_idx = {branch: i for i, branch in enumerate(self.foreground)}
        self.base = cdf(np.array([[m.transition(i, 0) for i in range(len(parents))] for m in self.models]))
        code = genetic_code.get_codon_table(1)
        syn = main_simulate.get_synonymous_codon_substitution_index({'codon_table': code}, self.codons)
        nonsyn = main_simulate.get_nonsynonymous_codon_substitution_index(syn)
        biased = []
        for aa in sorted({aa for aa, _ in code if aa != '*'}):
            target = main_simulate.get_biased_nonsynonymous_substitution_index([aa], code, self.codons)
            new_q = main_simulate.apply_percent_biased_sub(q, 90, target, [aa], code, self.codons,
                                                          nonsyn, syn, .2, eq_freq=pi)
            biased.append([[expm(new_q * (lengths[i] * r)) for i in self.foreground] for r in rates])
        self.alt = cdf(np.array(biased))

    def simulate(self, seed, alternative):
        rng = np.random.default_rng(seed)
        sites = np.arange(SITE_COUNT)
        target_sites = sites[:SITE_COUNT // 5]
        targets = rng.integers(self.alt.shape[0], size=len(target_sites)) if alternative else None
        values = np.empty((len(self.parents), SITE_COUNT), dtype=np.int64)
        values[self.order[0]] = rng.choice(len(self.pi), SITE_COUNT, p=self.pi)
        for child in self.order[1:]:
            parent = values[self.parents[child]]
            prob = self.base[self.assignments, child, parent]
            if alternative and child in self.foreground_idx:
                prob[target_sites] = self.alt[targets, self.assignments[target_sites],
                                              self.foreground_idx[child], parent[target_sites]]
            values[child] = (rng.random(SITE_COUNT)[:, None] >= prob).sum(axis=1)
        return values

    def summarize(self, values):
        a, b = self.pairs.T
        parent = values[self.parents]
        aa_child, aa_parent = self.aa[values], self.aa[parent]
        n = aa_child != aa_parent
        s = (values != parent) & ~n
        ns = np.count_nonzero(n[a] & n[b] & (aa_child[a] == aa_child[b]), axis=1)
        ss = np.count_nonzero(s[a] & s[b] & (values[a] == values[b]), axis=1)
        return np.column_stack([ns, ss])

    def infer(self, values):
        features = [feature_buffers(self.models[0], SITE_COUNT, len(self.syn_indices)) for _ in range(2)]
        for sites, model in zip(self.site_groups, self.models):
            if model.rates[0] == 0:
                continue
            tips = {i: self.eye[values[i, sites]] for i in self.leaves}
            blocks = edge_features(model, tips, self.transforms, self.syn_indices)
            for (n, s), (block_n, block_s) in zip(features, blocks):
                n[:, sites], s[:, sites] = block_n, block_s
        return features


def urn_tails(observed, features, pairs, seed, draws):
    rng = np.random.default_rng(seed + 450000)
    sampled, expected = [], []
    config = {'omega_pvalue_rounding': 'stochastic', 'urn_model': 'wallenius'}
    for channel in features:
        overlap = np.zeros((len(pairs), draws), dtype=np.int32)
        mean = np.zeros(len(pairs))
        for destination in channel.transpose(2, 0, 1):
            totals = destination.sum(axis=1)
            if not totals.any():
                continue
            profile = destination.sum(axis=0) + 1.
            profile /= profile.sum()
            sizes = omega._prepare_permutation_branch_sizes(totals, draws, config, rng=rng)
            overlap += omega._get_permutations_fast(pairs, sizes, profile, draws, rng=rng)
            mean += omega._calc_wallenius_expected_overlap(pairs, profile, totals, config, np.float64)
        sampled.append(overlap)
        expected.append(mean)
    expected = np.column_stack(expected)
    return np.concatenate([upper_tail(observed[sl], expected[sl], sampled[0][sl], sampled[1][sl], TOL)
                           for sl in (slice(i, i + 400) for i in range(0, len(pairs), 400))])


_PANEL = None


def initialize(panel, scenario):
    global _PANEL
    _PANEL = Panel(panel, scenario)


def replicate(job, urn_draws):
    condition, seed, urn = job
    panel = _PANEL
    states = panel.simulate(seed, condition == 'alternative')
    features, predictions = panel.infer(states)
    counts = np.column_stack([pair_counts(channel, panel.pairs) for channel in features])
    predicted = np.column_stack([pair_counts(channel, panel.pairs) for channel in predictions])
    observed = omega_score(counts, predicted)
    fixed_ec = omega_score(counts, panel.expected)
    assert not (np.isnan(observed) | np.isnan(fixed_ec)).any()
    truth = panel.summarize(states)
    p_urn = urn_tails(counts, features, panel.pairs, seed, urn_draws) if urn else None
    return condition, seed, counts, observed, fixed_ec, truth, p_urn


def p_rank(values, reference):
    return (1 + (reference >= values).sum(axis=0)) / (1 + len(reference))


def trial_summary(simulations, target, scenario):
    calibrations = [r for r in simulations if r[0] == 'calibration']
    calibration_scores = np.stack([r[3] for r in calibrations])
    calibration_fixed_ec = np.stack([r[4] for r in calibrations])
    calibration_n = np.stack([r[2][:, 0] for r in calibrations])
    families = {'target': target, 'all': np.ones_like(target)}
    max_null = {}
    screened_null = np.where((calibration_n >= 2.) & (calibration_scores >= 5.), calibration_scores, -np.inf)
    for family, selected in families.items():
        max_null[family] = (calibration_scores[:, selected].max(axis=1), screened_null[:, selected].max(axis=1))
    rows = []
    for condition, seed, counts, observed, fixed_ec, truth, p_urn in simulations:
        if condition == 'calibration':
            continue
        scores = {'asr_omega': p_rank(observed, calibration_scores), 'fixed_ec_omega': p_rank(fixed_ec, calibration_fixed_ec),
                  'asr_n_only': p_rank(counts[:, 0], calibration_n)}
        if p_urn is not None:
            scores['urn'] = p_urn
        eligible = (counts[:, 0] >= 2.) & (observed >= 5.)
        for name, mask in families.items():
            for method, p in scores.items():
                n = mask.sum()
                bh = omega._calc_bh_fdr_qvalues(p[mask])
                t = np.where(eligible[mask], observed[mask], -np.inf)
                max_p = p_rank(observed[mask].max(), max_null[name][0]) if method == 'asr_omega' else np.nan
                screened_p = p_rank(t.max(), max_null[name][1]) if method == 'asr_omega' else np.nan
                rows.append(dict(scenario=scenario, condition=condition, seed=seed, family=name, method=method,
                                 n_pairs=n, true_n_positive=((truth[:, 0] > 0) & mask).sum(),
                                 true_n_sum=truth[mask, 0].sum(), true_s_sum=truth[mask, 1].sum(),
                                 obs_n_sum=counts[mask, 0].sum(), obs_s_sum=counts[mask, 1].sum(),
                                 reported_support=(eligible & mask).sum(), raw=(p[mask] <= .05).sum(),
                                 adjusted=(bh <= .05).sum(), max_p=max_p, screened_max_p=screened_p))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('previous_runs', type=Path)
    parser.add_argument('work', type=Path)
    parser.add_argument('--calibration', type=int, default=399)
    parser.add_argument('--validation', type=int, default=96)
    parser.add_argument('--urn-validation', type=int, default=12)
    parser.add_argument('--urn-draws', type=int, default=399)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=SCENARIOS)
    args = parser.parse_args()
    if (not 1 <= args.calibration < 10000 or not 1 <= args.validation < 10000
            or not 0 <= args.urn_validation <= args.validation or args.urn_draws < 1):
        parser.error('Use nonoverlapping seed ranges and an urn subset within validation.')
    args.work.mkdir()
    panel = args.panel.resolve() / 'PEPC'
    check_search(panel, args.previous_runs.resolve())
    task = partial(replicate, urn_draws=args.urn_draws)
    config = [('calibration', args.calibration), ('null', args.validation), ('alternative', args.validation)]
    jobs = [(condition, CONDITION_SEEDS[condition] + i, condition != 'calibration' and i < args.urn_validation)
            for condition, count in config for i in range(count)]
    for scenario in args.scenarios:
        data = []
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=initialize, initargs=(panel, scenario)) as pool:
            for i, row in enumerate(pool.map(task, jobs, chunksize=1), 1):
                data.append(row)
                if i == len(jobs) or i % 50 == 0:
                    print(f'{scenario}: {i}/{len(jobs)} replicates', flush=True)
        selected = Panel(panel, scenario)
        pd.DataFrame(trial_summary(data, selected.target, scenario)).to_csv(
            args.work / f'{scenario}.tsv', sep='\t', index=False, float_format='%.12g', na_rep='NA')


if __name__ == '__main__':
    main()

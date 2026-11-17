"""Decompose sister-pair N endpoint probabilities against simulated truth."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.special import logsumexp

from csubst import endpoint, endpoint_io, ete, fitted_model, genetic_code, main_simulate, sequence, tree
from csubst._vendor.pyvolve.evolver import ZERO

from endpoint_posterior import ConditionalEdges
from missingness import tip_mask
from pilot import EXPECTED, KEY, OBS, TRUTH
from replay_posterior import fitted_generator, oracle_generator


FIELDS = ('separate', 'shared_rate', 'joint', 'true_parent_aa', 'wrong_parent_aa_mass', 'parent_aa_pp')


class ForegroundModel(endpoint.EndpointModel):
    """Use one alternate generator on known foreground edges of a site partition."""

    def __init__(self, parents, lengths, q, pi, foreground_edges, foreground_q):
        super().__init__(parents, lengths, q, pi)
        self.foreground = {}
        by_time = {}
        for child in foreground_edges:
            length = lengths[child]
            if length not in by_time:
                matrix = np.maximum(expm(foreground_q * length), 0.)
                by_time[length] = matrix / matrix.sum(axis=1, keepdims=True)
            self.foreground[child] = by_time[length]

    def transition(self, child, category):
        if child in self.foreground:
            return self.foreground[child]
        return super().transition(child, category)


def scores(model, tips, amino, pairs, parent_aa, missing, both):
    """Resolve the contributions from node state, rate class, and endpoint features."""
    size = len(next(iter(tips.values())))
    n_rate = len(model.rates)
    membership = np.eye(amino.max() + 1)[amino]
    different = amino[:, None] != amino
    parent_ids = model.parents[pairs[:, 0]]
    children = set(pairs.ravel())
    shared_nodes = set(parent_ids)
    values = np.zeros((len(pairs), size, len(FIELDS)))
    records = model.iter_blocks(tips, block_size=16, branch_ids=children, joint=False,
                                transform=ConditionalEdges(n_rate), transform_rate_context=True,
                                category_nodes=True, rate_posterior=True, tip_missing=missing)
    for _, block in groupby(records, key=lambda record: record.start):
        direct, node = {}, {}
        for record in block:
            sl = slice(record.start, record.stop)
            if record.parent < 0:
                rate = record.rate_posterior
            if record.child in shared_nodes:
                node[record.child] = record.category_node
            if record.child in children:
                k = np.stack([record.reduced[i] for i in range(n_rate)])
                direct[record.child] = (k * different) @ membership
        for i, ((a, b), parent) in enumerate(zip(pairs, parent_ids)):
            p = node[parent]
            f, g = direct[a], direct[b]
            mf = np.einsum('csk,cska->csa', p, f)
            mg = np.einsum('csk,cska->csa', p, g)
            sep = (mf.sum(axis=0) * mg.sum(axis=0)).sum(axis=-1)
            inv = np.divide(1., rate, out=np.zeros_like(rate), where=rate > 0.)
            share_rate = (mf * mg * inv[..., None]).sum(axis=(0, 2))
            mass = p * np.einsum('cska,cska->csk', f, g)
            joint = mass.sum(axis=(0, 2))
            known = amino == parent_aa[i, sl, None]
            known_pp = (p * known).sum(axis=(0, 2))
            known_joint = (mass * known).sum(axis=(0, 2))
            clamped = np.divide(known_joint, known_pp, where=known_pp > 0.,
                                out=np.full_like(known_pp, np.nan))
            wrong = (mass * ~known).sum(axis=(0, 2))
            values[i, sl] = np.column_stack((sep, share_rate, joint, clamped, wrong, known_pp))
    values[~both] = 0
    return values


def log_evidence(model, tips, amino, restrictions):
    """Likelihood with zero events at designated sister pairs, optionally unconditioned on tips."""
    sites = len(next(iter(tips.values())))
    states = len(amino)
    membership = np.eye(amino.max() + 1)[amino]
    different = amino[:, None] != amino
    logs = []
    for category, weight in enumerate(model.weights):
        inside = np.zeros((len(model.parents), sites, states))
        scale = np.zeros(inside.shape[:2])
        for child, emission in tips.items():
            inside[child] = emission
        for node in reversed(model.order):
            children = model.children[node]
            if not children:
                continue
            message = [inside[child] @ model.transition(child, category).T for child in children]
            row = np.prod(message, axis=0)
            if node in restrictions:
                mask = restrictions[node]
                if len(children) != 2:
                    raise ValueError('This diagnostic requires bifurcating sister pairs.')
                m = [(model.transition(child, category) * inside[child, mask, None, :] * different) @ membership
                     for child in children]
                excluded = (m[0] * m[1]).sum(axis=-1)
                difference = row[mask] - excluded
                if (difference < -1e-12).any():
                    raise ValueError('Restricted likelihood exceeds total likelihood.')
                row[mask] = np.maximum(difference, 0.)
            norm = row.max(axis=1)
            inside[node] = row / np.where(norm > 0., norm, 1.)[:, None]
            with np.errstate(divide='ignore'):
                scale[node] = scale[children].sum(axis=0) + np.log(norm)
        with np.errstate(divide='ignore'):
            logs.append(np.log(weight) + scale[model.root] + np.log(inside[model.root] @ model.pi))
    return logsumexp(logs, axis=0)


def log_no_event(model, tips, amino, pairs, both, missing):
    likelihoods = {leaf: np.where(missing[leaf][:, None], 1., rows) for leaf, rows in tips.items()}
    constraints = {int(model.parents[a]): both[i] for i, (a, _) in enumerate(pairs)}
    if len(constraints) != len(pairs):
        raise ValueError('Nonunique sister parent in zero-event diagnostic.')
    log_observed = log_evidence(model, likelihoods, amino, {})
    log_restricted = log_evidence(model, likelihoods, amino, constraints)
    delta = log_restricted - log_observed
    if np.isnan(delta).any() or (delta > 1e-11).any():
        raise ValueError('Invalid zero-event probability.')
    return float(delta.sum())


def generating_models(rep, codons, parents, lengths, sites, foreground_edges):
    q, pi = oracle_generator(rep, codons)
    log = (rep / 'sim/sim.log').read_text()
    targets = re.findall(r'^Codon site (\d+); Biased amino acids = ([A-Z]);', log, flags=re.MULTILINE)
    if [int(i) for i, _ in targets] != list(range(1, len(targets) + 1)):
        raise ValueError('Generating partitions were not logged consecutively.')
    amino_targets = np.array([aa for _, aa in targets] + [''] * (sites - len(targets)))
    code = genetic_code.get_codon_table(1)
    context = {'codon_table': code}
    synonymous = main_simulate.get_synonymous_codon_substitution_index(context, codons)
    nonsynonymous = main_simulate.get_nonsynonymous_codon_substitution_index(synonymous)
    models = []
    for aa in np.unique(amino_targets):
        mask = amino_targets == aa
        if aa:
            target = main_simulate.get_biased_nonsynonymous_substitution_index([aa], code, codons)
            biased = main_simulate.apply_percent_biased_sub(
                q, 90, target, [aa], code, codons, nonsynonymous, synonymous, .2, eq_freq=pi,
            )
            model = ForegroundModel(parents, lengths, q, pi, foreground_edges, biased)
        else:
            model = endpoint.EndpointModel(parents, lengths, q, pi)
        models.append((mask, model))
    return models, amino_targets != '', q, pi


def summarize(values, truth, valid, sectors):
    out = []
    for name, mask in sectors.items():
        select = valid & mask
        if select.any():
            rows = values[select]
            expected = rows.sum(axis=0)
            finite = np.isfinite(rows[:, 3])
            out.append(dict(
                subset=name, pair_sites=int(select.sum()), realized_n=int(truth[select].sum()),
                **dict(zip(FIELDS, expected)),
                true_parent_aa_unsupported=int((~finite).sum()),
                # Compute pair-level rather than site-level error in the count totals.
                joint_pair_sse=float(np.square(((values[..., 2] - truth) * select).sum(axis=1)).sum()),
                separate_pair_sse=float(np.square(((values[..., 0] - truth) * select).sum(axis=1)).sum()),
            ))
    return out


def analyze_case(job):
    rep, case, condition = job
    sim = rep / 'sim'
    scenario, seed = rep.name.rsplit('-', 1)
    folder = case / (condition + '-joint')
    meta = json.loads((folder / 'csubst_endpoint_model.json').read_text())['codon']
    if condition == 'complete':
        report = case / 'fit.iqtree'
        alignment = case / 'complete.fa'
    else:
        report, = (case / 'masked_fit').glob('*.iqtree')
        alignment = case / 'masked.fa'
    q, pi, codons, amino = fitted_generator(rep, report, meta, alignment)
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(sim / 'sim_true_asr.treefile'), format=1))
    original = tree.add_numerical_node_labels(ete.PhyloNode(str(rep.parent / 'inputs/PEPC.tree.nwk'), format=1))
    parents, lengths = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    truth_parents, gen_lengths = endpoint_io._tree_arrays({'tree': original}, structural=False)
    np.testing.assert_array_equal(truth_parents, parents)
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in tr.traverse()}
    for node in ete.iter_leaves(original):
        assert names[int(ete.get_prop(node, 'numerical_label'))] == node.name
    fit = endpoint.EndpointModel(parents, meta['branch_lengths'], q, pi, meta['rates'], meta['weights'])
    np.testing.assert_allclose(lengths, gen_lengths, rtol=2e-5, atol=2e-8)
    gen_lengths[gen_lengths <= ZERO] = 0
    complete = sequence.read_fasta(sim / 'sim.fa')
    all_truth = complete | sequence.read_fasta(sim / 'sim_true_asr.anc.fa')
    sites = len(next(iter(complete.values()))) // 3
    actual_rate = pd.read_csv(sim / 'sim_true_asr.rate', sep='\t')['C_Rate'].to_numpy()
    np.testing.assert_array_equal(actual_rate, 1.)
    labels = {codon: index for index, codon in enumerate(codons)}
    node_states = np.array([[labels[all_truth[names[i]][s:s+3]] for s in range(0, 3 * sites, 3)]
                            for i in range(len(parents))])
    node_aa = amino[node_states]
    emissions = np.zeros((len(parents), sites, len(amino)))
    eye = np.eye(len(amino))
    for leaf in fit.leaves:
        emissions[leaf] = eye[node_states[leaf]]
    clade, mask = tip_mask(tr, complete, sites)
    if condition != 'complete':
        for leaf in fit.leaves:
            if names[leaf] in clade:
                emissions[leaf, mask] = 0.
    observed, eligible = fitted_model.observation_masks(tr, emissions)
    tips = {leaf: emissions[leaf] for leaf in fit.leaves}
    missing = {leaf: ~observed[leaf] for leaf in fit.leaves}
    input_by_name = sequence.read_fasta(alignment)
    for leaf in tips:
        visible = ''.join('---' if skip else codon for skip, codon in zip(missing[leaf], codons[node_states[leaf]]))
        assert visible == input_by_name[names[leaf]]
    simulated = pd.read_csv(sim / 'sim_expected_omega.tsv', sep='\t')
    joint = pd.read_csv(folder / 'csubst_cb_2.tsv', sep='\t')
    separate = pd.read_csv(case / (condition + '-separate') / 'csubst_cb_2.tsv', sep='\t')
    records = simulated.merge(joint[KEY + EXPECTED + OBS], on=KEY, validate='one_to_one')
    records = records.merge(separate[KEY + OBS], on=KEY, suffixes=('_joint', '_separate'), validate='one_to_one')
    a, b = records[KEY].to_numpy().T
    records = records.loc[(parents[a] == parents[b]) & records[EXPECTED].notna().all(axis=1)].copy()
    pairs = records[KEY].to_numpy()
    a, b = pairs.T
    for i in (1, 2):
        assert records[f'branch_id_{i}'].map(names).eq(records[f'branch_name_{i}']).all()
    pa = node_aa[parents[a]]
    truth_all = (node_aa[a] == node_aa[b]) & (node_aa[a] != pa)
    np.testing.assert_array_equal(truth_all.sum(axis=1), records[TRUTH[0]])
    both = eligible[a] & eligible[b]
    truth = truth_all & both

    foreground = set(np.loadtxt(sim / 'sim_foreground_branch.txt', dtype=int, ndmin=1))
    partitions, biased_sites, gq, gpi = generating_models(rep, codons, parents, gen_lengths, sites, foreground)
    n_prior = np.zeros(len(pairs))
    conditional = np.zeros((*truth.shape, len(FIELDS)))
    prior_no_n = 0.
    post_no_n = 0.
    for indices, model in partitions:
        sub_tips = {leaf: rows[indices] for leaf, rows in tips.items()}
        sub_missing = {leaf: rows[indices] for leaf, rows in missing.items()}
        full_prior = {leaf: np.ones((1, len(amino))) for leaf in tips}
        full_ok = np.ones((len(pairs), 1), dtype=bool)
        none_missing = {leaf: np.zeros(1, dtype=bool) for leaf in tips}
        prior = scores(model, full_prior, amino, pairs, pa[:, :1], none_missing, full_ok)[:, 0, 2]
        n_prior += indices.sum() * prior
        posterior = scores(model, sub_tips, amino, pairs, pa[:, indices], sub_missing, both[:, indices])
        conditional[:, indices] = posterior
        unrestricted = {leaf: np.ones_like(rows) for leaf, rows in sub_tips.items()}
        prior_no_n += log_no_event(model, unrestricted, amino, pairs, both[:, indices],
                                  {leaf: np.zeros(len(rows), dtype=bool) for leaf, rows in unrestricted.items()})
        post_no_n += log_no_event(model, sub_tips, amino, pairs, both[:, indices], sub_missing)
    np.testing.assert_allclose(n_prior, records['muNany2spe_sim'], rtol=5e-8, atol=1e-9)

    variants = {
        'fitted': scores(fit, tips, amino, pairs, pa, missing, both),
        'true_site_rate': scores(endpoint.EndpointModel(parents, meta['branch_lengths'], q, pi),
                                 tips, amino, pairs, pa, missing, both),
        'true_lengths': scores(endpoint.EndpointModel(parents, gen_lengths, q, pi, meta['rates'], meta['weights']),
                               tips, amino, pairs, pa, missing, both),
        'true_lengths_and_rate': scores(endpoint.EndpointModel(parents, gen_lengths, q, pi),
                                        tips, amino, pairs, pa, missing, both),
        'generating_background': scores(endpoint.EndpointModel(parents, gen_lengths, gq, gpi),
                                         tips, amino, pairs, pa, missing, both),
        'generating_model': conditional,
    }
    fitted = variants['fitted']
    np.testing.assert_allclose(fitted[..., 0].sum(axis=1), records[OBS[0] + '_separate'], rtol=1e-9, atol=6e-12)
    np.testing.assert_allclose(fitted[..., 2].sum(axis=1), records[OBS[0] + '_joint'], rtol=1e-9, atol=6e-12)
    is_tip_pair = np.array([(x in tips, y in tips) for x, y in pairs]).sum(axis=1)
    tips_equal = (is_tip_pair[:, None] == 2) & (node_aa[a] == node_aa[b])
    np.testing.assert_allclose(fitted[..., 0][tips_equal & both], np.square(fitted[..., 2][tips_equal & both]),
                               rtol=1e-9, atol=6e-13)
    sectors = {
        'all': np.ones_like(both),
        'two_tips': is_tip_pair[:, None] == 2,
        'one_tip': is_tip_pair[:, None] == 1,
        'two_internal': is_tip_pair[:, None] == 0,
        'both_true_aa_unchanged': (node_aa[a] == pa) & (node_aa[b] == pa),
        'same_true_child_aa': node_aa[a] == node_aa[b],
        'biased_site_partition': biased_sites,
        'background_site_partition': ~biased_sites,
        'either_focal_foreground': np.isin(pairs, list(foreground)).any(axis=1)[:, None],
        'unperturbed_focal_edges': ~np.isin(pairs, list(foreground)).any(axis=1)[:, None],
    }
    summaries = []
    tag = dict(scenario=scenario, seed=int(seed), condition=condition)
    for name, values in variants.items():
        summaries.extend(dict(**tag, model=name, **row) for row in summarize(values, truth, both, sectors))
    changes = fitted[..., 2] - fitted[..., 0]
    top = []
    letters = {c: aa for aa, c in genetic_code.get_codon_table(1)}
    for idx in np.argsort(changes.ravel())[-12:][::-1]:
        i, s = np.unravel_index(idx, changes.shape)
        u, v = pairs[i]
        top.append(dict(
            **tag, branch_1=names[u], branch_2=names[v], site=int(s + 1), biased_site=bool(biased_sites[s]),
            true_parent=codons[node_states[parents[u], s]], true_child_1=codons[node_states[u, s]],
            true_child_2=codons[node_states[v, s]],
            true_parent_amino=letters[codons[node_states[parents[u], s]]],
            pair_type=int(is_tip_pair[i]), realized_n=int(truth[i, s]),
            **dict(zip(FIELDS, fitted[i, s])), generating_joint=conditional[i, s, 2],
        ))
    zero = dict(**tag, realized_n=int(truth.sum()), eligible_pairs=len(pairs), generating_prior_p0=np.exp(prior_no_n),
                generating_posterior_p0=np.exp(post_no_n),
                fitted_posterior_p0=np.exp(log_no_event(fit, tips, amino, pairs, both, missing)))
    return summaries, top, zero


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('cases', type=Path)
    parser.add_argument('outdir', type=Path)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--conditions', nargs='+', choices=['complete', 'clade_block'],
                        default=['complete', 'clade_block'])
    args = parser.parse_args()
    jobs = [(args.panel.resolve() / 'PEPC' / name, args.cases.resolve() / name, condition)
            for name in [f'{scenario}-{seed}' for scenario in ('baseline', 'biased') for seed in (101, 202, 303)]
            for condition in args.conditions]
    args.outdir.mkdir(parents=True, exist_ok=True)
    totals, top, zero = [], [], []
    with ProcessPoolExecutor(args.jobs) as pool:
        for job, (sums, rows, probability) in zip(jobs, pool.map(analyze_case, jobs)):
            totals.extend(sums)
            top.extend(rows)
            zero.append(probability)
            print(job[0].name, job[2], probability, flush=True)
    for name, table in [('sister_n_totals', totals), ('sister_n_top_sites', top), ('sister_n_zero', zero)]:
        pd.DataFrame(table).to_csv(args.outdir / (name + '.tsv'), sep='\t', index=False, float_format='%.12g', na_rep='NA')


if __name__ == '__main__':
    main()

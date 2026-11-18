"""Try shared rate and ancestral-state predictions for ECN/ECSany2spe."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from csubst import endpoint, endpoint_io, ete, fitted_model, sequence, tree
from csubst.endpoint_pair import (
    _called_codons, _compress_patterns, _edge_kernel, _EdgeEvidence, _pack_dense,
    count_any2spe, make_plan,
)

from pilot import EXPECTED, KEY, NULL, OBS, TRUTH
from missingness import count_truth, tip_mask
from replay_posterior import fitted_generator, load_tips


METHODS = ('separate', 'shared_rate', 'joint')
SIM = ['muNany2spe_sim', 'muSany2spe_sim']


def predicted_counts(model, tips, amino, pairs, block_size=4, *, tip_missing=None, eligible=None):
    """Combine predictive edge features under three parent/rate factorizations.

    The endpoints of the two focal branches are drawn from raw transitions.
    Internal paths up to their parents carry tip-conditioned transitions.
    """
    sites = len(next(iter(tips.values())))
    multiplicity = np.ones(sites)
    if not tip_missing and eligible is None:
        called = {i: codons for i, values in tips.items() if (codons := _called_codons(values)) is not None}
        tips, _, multiplicity = _compress_patterns(tips, called)

    groups = make_plan(model.parents, pairs)
    focal = sorted(set(pairs.ravel()))
    paths = [path for _, left, right, _ in groups for path in (*left, *right)]
    prefixes = {path[:i] for path in paths for i in range(2, len(path) + 1)}
    uses = Counter(paths)
    uses.update(path[:-1] for path in prefixes)
    required_edges = {path[-1] for path in prefixes}
    nodes = {int(model.parents[i]) for i in focal} | {ancestor for ancestor, *_ in groups}
    grouped = defaultdict(list)
    for group in groups:
        grouped[group[0]].append(group)
    paths_by_node = defaultdict(lambda: defaultdict(list))
    for path in sorted(prefixes):
        edge = path[-1]
        paths_by_node[model.parents[edge]][edge].append(path)
    schedule = [(paths_by_node[node], grouped[node]) for node in reversed(model.order)
                if node in paths_by_node or node in grouped]

    num_aas = int(amino.max()) + 1
    membership = np.eye(num_aas)[amino]
    different = amino[:, None] != amino
    syn = ~different & ~np.eye(len(amino), dtype=bool)
    channels = (slice(0, num_aas), slice(num_aas, num_aas + len(amino)))
    rate_count = len(model.rates)
    transitions = {i: np.stack([model.transition(i, c) for c in range(rate_count)])
                   for i in set(focal) | required_edges}
    raw = {i: np.ascontiguousarray(np.concatenate(((transitions[i] * different) @ membership,
                                                    transitions[i] * syn), axis=-1).swapaxes(-1, -2))
           for i in focal}
    pos = {i: j for j, i in enumerate(focal)}
    a, b = np.array([[pos[i], pos[j]] for i, j in pairs]).T
    totals = np.zeros((3, len(pairs), 2))
    records = model.iter_blocks(tips, block_size=block_size, branch_ids=required_edges, joint=False,
                                transform=_EdgeEvidence(rate_count), transform_rate_context=True,
                                category_nodes=True, tip_missing=tip_missing, rate_posterior=True)

    for start, block in groupby(records, key=lambda record: record.start):
        state, edges = {}, {}
        for record in block:
            sl = slice(start, record.stop)
            child = record.child
            if child == model.root:
                rate_pp = record.rate_posterior
            if child in nodes:
                state[child] = record.category_node
            if child in required_edges:
                rights = np.stack([record.reduced[c] for c in range(rate_count)])
                edges[child] = _edge_kernel(transitions[child], rights).swapaxes(-1, -2)

        size = sl.stop - start
        direct = {i: np.broadcast_to(f[:, None, :, :], (rate_count, size, *f.shape[1:]))
                  for i, f in raw.items()}
        if eligible is not None:
            for i in focal:
                if not eligible[i, sl].all():
                    direct[i] = direct[i] * eligible[i, sl][None, :, None, None]
        marginals = np.stack([np.einsum('csk,csdk->csd', state[model.parents[i]], direct[i])
                              for i in focal])
        sqrt_mult = np.sqrt(multiplicity[sl])
        scale = np.divide(sqrt_mult[None, :], np.sqrt(rate_pp), out=np.zeros_like(rate_pp), where=rate_pp > 0.)
        separate = marginals.sum(axis=1) * sqrt_mult[:, None]
        shared_rate = marginals * scale[..., None]
        for method, features in enumerate((separate, shared_rate)):
            for channel, section in enumerate(channels):
                flat = np.ascontiguousarray(features[..., section]).reshape(len(focal), -1)
                totals[method, :, channel] += (flat @ flat.T)[a, b]

        cached = {}
        remaining = uses.copy()

        def consume(path):
            if len(path) == 1:
                return direct[path[0]]
            remaining[path] -= 1
            return cached.pop(path) if remaining[path] == 0 else cached[path]

        for steps, crosses in schedule:
            for edge, pending in steps.items():
                for start_path in range(0, len(pending), 16):
                    tile = pending[start_path:start_path+16]
                    inputs = [consume(path[:-1]) for path in tile]
                    width = inputs[0].shape[-2]
                    rhs = np.concatenate(inputs, axis=-2) if len(inputs) > 1 else inputs[0]
                    moved = rhs @ edges[edge]
                    for j, path in enumerate(tile):
                        cached[path] = moved[..., j*width:(j+1)*width, :]
                    del inputs, rhs, moved
            for ancestor, left, right, (target, li, ri) in crosses:
                weight = np.sqrt(state[ancestor] * multiplicity[sl][None, :, None])
                x, y = ([consume(path) for path in side] for side in (left, right))
                for channel, section in enumerate(channels):
                    lf = _pack_dense(x, weight, section)
                    rf = _pack_dense(y, weight, section)
                    totals[2, target, channel] += (lf @ rf.T)[li, ri]
                    del lf, rf
                del x, y
    return totals


def aggregate(frame, parents, values, fixed_observed):
    pairs = frame[KEY].to_numpy()
    sisters = parents[pairs[:, 0]] == parents[pairs[:, 1]]
    masks = {
        'all': np.ones(len(frame), dtype=bool),
        'sisters': sisters,
        'different_parents': ~sisters,
        'two_foreground': (frame.foreground_1.eq(1) & frame.foreground_2.eq(1)).to_numpy(),
        'low_expected_s': frame[NULL[1]].le(.1).to_numpy(),
        'expected_s_over_one': frame[NULL[1]].gt(1.).to_numpy(),
    }
    null = frame[NULL].to_numpy()
    for tag, include in masks.items():
        if not include.any():
            continue
        for method, expected in zip(METHODS, values):
            row = dict(subset=tag, method=method, pair_count=int(include.sum()))
            for label, data in [('null', null), ('sim', frame[SIM].to_numpy()), ('truth', frame[TRUTH].to_numpy()),
                                ('obs_separate', frame[OBS].to_numpy()), ('obs_joint', fixed_observed),
                                ('predicted', expected)]:
                for suffix, column in zip(('n', 's'), data[include].T):
                    row[label + '_' + suffix] = column.sum()
            row['ec_n_sse'], row['ec_s_sse'] = np.square(expected[include] - null[include]).sum(axis=0)
            yield row


def mask_sister_input(rep, tr, model, tips, frame, alignment):
    complete = sequence.read_fasta(rep / 'sim/sim.fa')
    ancestors = sequence.read_fasta(rep / 'sim/sim_true_asr.anc.fa')
    sites = len(next(iter(tips.values())))
    clade, mask = tip_mask(tr, complete, sites)
    ends = sequence.read_fasta(alignment)
    status = np.zeros((len(model.parents), sites, len(model.pi)))
    for node in ete.iter_leaves(tr):
        i = int(ete.get_prop(node, 'numerical_label'))
        status[i] = tips[i]
        if node.name in clade:
            status[i, mask] = 0.
        edited = ''.join('---' if node.name in clade and excluded else complete[node.name][s*3:s*3+3]
                         for s, excluded in enumerate(mask))
        if ends[node.name] != edited:
            raise ValueError('Masked alignment differs from the original design.')
    observed, eligible = fitted_model.observation_masks(tr, status)
    missing = {i: ~observed[i] for i in tips}
    counts, retained = count_truth(tr, model.parents, frame[KEY].to_numpy(), complete | ancestors, clade, mask)
    frame[TRUTH] = counts

    # The missing block is wholly past the directionally biased partition.
    # Remove background-Q expectations only for the excluded eligible sites.
    affected = [int(site) for site in re.findall(r'^Codon site (\d+);', (rep / 'sim/sim.log').read_text(), re.MULTILINE)]
    if mask[np.asarray(affected, dtype=int) - 1].any():
        raise ValueError('Mask overlaps the directionally biased partition.')
    baseline = rep.with_name('baseline-' + rep.name.rsplit('-', 1)[1]) / 'sim'
    for source in (rep / 'sim', baseline):
        rates = pd.read_csv(source / 'sim_true_asr.rate', sep='\t')['C_Rate']
        if len(rates) != sites or not rates.eq(1.).all():
            raise ValueError('The masking contrast requires unit generating rates.')
        log = (source / 'sim.log').read_text()
        if any(flag not in log for flag in ('--background_omega 0.2 ', '--tree_branch_length_scale 1.0 ',
                                            '--foreground_branch_length_scale 1.0 ')):
            raise ValueError('The PEPC reference fits must use the same background-Q setup.')
        if (source / 'sim_true_asr.treefile').read_bytes() != (baseline / 'sim_true_asr.treefile').read_bytes():
            raise ValueError('The masked and reference simulation use different trees.')
    ref = frame[KEY].merge(pd.read_csv(baseline / 'sim_expected_omega.tsv', sep='\t')[KEY + NULL],
                           on=KEY, how='left', validate='one_to_one')
    removed = (1. - retained / sites)[:, None] * ref[NULL].to_numpy()
    for cols in (NULL, SIM):
        corrected = frame[cols].to_numpy() - removed
        if not np.isfinite(corrected).all() or (corrected < -1e-10).any():
            raise ValueError('Invalid generating reference after masking.')
        frame[cols] = np.maximum(corrected, 0.)
    return eligible, missing


def analyze(job):
    rep, cases, masked = job
    scenario, seed = rep.name.rsplit('-', 1)
    case = cases / rep.name
    alignment = None
    if masked:
        reference = case / 'clade_block-separate'
        report, = (case / 'masked_fit').glob('*.iqtree')
        saved_joint = case / 'clade_block-joint/csubst_cb_2.tsv'
        alignment = case / 'masked.fa'
    elif case.exists():
        reference = case / 'complete-separate'
        report = case / 'fit.iqtree'
        saved_joint = case / 'complete-joint/csubst_cb_2.tsv'
    else:
        reference = rep / 'default'
        report, = (rep / 'iqtree-default').glob('*.iqtree')
        saved_joint = None
    metadata = json.loads((reference / 'csubst_endpoint_model.json').read_text())['codon']
    q, pi, codons, amino = fitted_generator(rep, report, metadata, alignment)
    tr = tree.add_numerical_node_labels(ete.PhyloNode(str(rep / 'sim/sim_true_asr.treefile'), format=1))
    parents, _ = endpoint_io._tree_arrays({'tree': tr}, structural=False)
    model = endpoint.EndpointModel(parents, metadata['branch_lengths'], q, pi, metadata['rates'], metadata['weights'])
    tips = load_tips(rep, codons, tr)
    search = pd.read_csv(reference / 'csubst_cb_2.tsv', sep='\t')
    truth = pd.read_csv(rep / 'sim/sim_expected_omega.tsv', sep='\t')
    frame = truth.merge(search[KEY + EXPECTED + OBS], on=KEY, how='outer', validate='one_to_one', indicator=True)
    if not frame.pop('_merge').eq('both').all():
        raise ValueError('Pair sets disagree for ' + str(rep))
    frame = frame.dropna(subset=EXPECTED).sort_values(KEY)
    names = {int(ete.get_prop(node, 'numerical_label')): node.name for node in tr.traverse()}
    for i in (1, 2):
        if not frame[f'branch_id_{i}'].map(names).eq(frame[f'branch_name_{i}']).all():
            raise ValueError('Simulated and fitted branch IDs differ.')
    pairs = frame[KEY].to_numpy()
    eligible, missing = None, None
    if masked:
        frame = frame.loc[parents[pairs[:, 0]] == parents[pairs[:, 1]]].copy()
        pairs = frame[KEY].to_numpy()
        eligible, missing = mask_sister_input(rep, tr, model, tips, frame, alignment)
    values = predicted_counts(model, tips, amino, pairs, tip_missing=missing, eligible=eligible)
    np.testing.assert_allclose(values[0], frame[EXPECTED], rtol=1e-9, atol=6e-12)
    largest = np.abs(values[0] - frame[EXPECTED].to_numpy()).max()
    if saved_joint is None:
        fixed_observed = count_any2spe(model, tips, amino, pairs, 4)
    else:
        fixed_observed = frame[KEY].merge(pd.read_csv(saved_joint, sep='\t')[KEY + OBS], on=KEY,
                                           how='left', validate='one_to_one')[OBS].to_numpy()
        if not np.isfinite(fixed_observed).all():
            raise ValueError('Joint observed counts are missing for a saved pair.')
    print(rep.name, len(pairs), 'factorized EC max abs difference', largest, flush=True)
    return [dict(scenario=scenario, seed=int(seed), condition='masked_sisters' if masked else 'complete', **row)
            for row in aggregate(frame, parents, values, fixed_observed)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('cases', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--scenarios', nargs='+', choices=['baseline', 'biased', 'rate_mixture'],
                        default=['baseline', 'biased', 'rate_mixture'])
    parser.add_argument('--masked-sisters', action='store_true')
    args = parser.parse_args()
    if args.masked_sisters and 'rate_mixture' in args.scenarios:
        parser.error('--masked-sisters requires the baseline and/or biased scenarios.')
    jobs = [(args.panel.resolve() / 'PEPC' / f'{scenario}-{seed}', args.cases.resolve(), args.masked_sisters)
            for scenario in args.scenarios for seed in (101, 202, 303)]
    rows = []
    with ProcessPoolExecutor(args.jobs) as pool:
        for results in pool.map(analyze, jobs):
            rows.extend(results)
    pd.DataFrame(rows).to_csv(args.output, sep='\t', index=False, float_format='%.12g', na_rep='NA')


if __name__ == '__main__':
    main()

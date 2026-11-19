"""Compare OCN tie breakers on the very same indexed proposals (research only)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import branch_pair_index as index
import group_refinement as refinement
from group_refinement_benchmark import pgk_proposals
import site_intersection_pilot as pilot


METHODS = ('indexed', 'group_refined')


def s_ratio(observed, expected):
    """Undefined S rates rank last; zero observations with positive ECS rank first."""
    observed, expected = np.broadcast_arrays(observed, expected)
    valid = np.isfinite(observed) & (observed >= 0) & np.isfinite(expected) & (expected > 0)
    return np.divide(observed, expected, out=np.full(expected.shape, np.inf), where=valid)


def top_sets(support, ratio, limit=64):
    """Preserve descending OCN and original vote order. Score S only for a split tie.

    Returns membership, not a complete S-sorted order above the boundary.
    """
    if limit < 1:
        raise ValueError('limit must be positive')
    base = np.argsort(-support, kind='stable')[:limit]
    if len(support) <= limit:
        return base, base, 0
    value = support[base[-1]]
    group = np.flatnonzero(support == value)
    above = base[support[base] > value]
    slots = limit - len(above)
    if len(group) == slots:
        return base, base, 0
    order = np.argsort(ratio(group), kind='stable')
    return base, np.concatenate([above, group[order[:slots]]]), len(group)


def target_ranks(support, ratio, target):
    value = support[target]
    group = np.flatnonzero(support == value)
    offset = np.count_nonzero(support > value) + 1
    position = group.searchsorted(target)
    values = ratio(group)
    better = np.count_nonzero(values < values[position])
    tied_before = np.count_nonzero(values[:position] == values[position])
    return int(offset + position), int(offset + better + tied_before)


def scale_case(sites, seed, scenario):
    state, quality, qn, qs, plant = pilot.realization(scenario, 256, sites, seed)
    score = refinement.known_support(state, quality, .5)
    syn = pilot.pack_rows(state == 3)
    ecs = np.sum(qs ** 4)
    ecn = np.sum((qn * quality) ** 4) * (.8 ** 4 + .2 ** 4)
    rows = []
    for method in METHODS:
        keys, *_ = index.indexed_candidates(state, quality, np.random.default_rng(seed + 9000),
                                            score=score if method == 'group_refined' else None)
        support = np.fromiter(map(score, keys), int, count=len(keys))

        def oc_s(indices):
            return pilot.overlap(syn, [keys[i] for i in indices], 4)

        def ratio(indices):
            return s_ratio(oc_s(indices), ecs)

        base, reranked, tied = top_sets(support, ratio)
        row = dict(scenario='no_plant' if scenario == 'null' else scenario, sites=sites, seed=seed,
                   method=method, proposals=len(keys), split_tie_size=tied,
                   exchanged=len(set(base) - set(reranked)), selected=len(base),
                   plant_pass=None, rank_votes=None, rank_syn=None, plant_ocs=None)
        if scenario != 'null':
            truth = pilot.reference(state[list(plant)], quality, qn, qs, 4)
            row['plant_pass'] = int(pilot.qualifying(truth).item())
            if plant in keys:
                pos = keys.index(plant)
                row['rank_votes'], row['rank_syn'] = target_ranks(support, ratio, pos)
                assert (row['rank_votes'] <= 64) == (pos in base)
                assert (row['rank_syn'] <= 64) == (pos in reranked)
            row['plant_ocs'] = pilot.overlap(syn, [plant], 4).item()
        for label, selected in [('votes', base), ('syn', reranked)]:
            ocs = oc_s(selected)
            with np.errstate(divide='ignore', invalid='ignore'):
                omega = (support[selected] / ecn) / (ocs / ecs)
            row['passes_' + label] = int(np.count_nonzero((support[selected] >= 2) & (omega >= 5)))
            row['median_omega_' + label] = np.median(omega)
            row['zero_ocs_' + label] = int(np.count_nonzero(ocs == 0))
        rows.append(row)
    return rows


def pgk_case(panel, condition, seed):
    with np.load(panel / 'posteriors.npz') as f:
        ids, p = f['ids'], f['p']
    ref = pd.read_table(panel / 'reference/csubst_cb_4.tsv').set_index([f'branch_id_{i}' for i in range(1, 5)])
    passes = (ref.cutoff_OCNany2spe >= 2) & (ref.omegaCany2spe >= 5)
    rows = []
    for method in METHODS:
        keys, support, _ = pgk_proposals(method, ids, p)
        table = ref.loc[keys, :]
        np.testing.assert_allclose(support, table.cutoff_OCNany2spe, rtol=1e-10, atol=1e-12)
        ocs, ecs = table.OCSany2spe.to_numpy(), table.ECSany2spe.to_numpy()
        base, reranked, tied = top_sets(support, lambda group: s_ratio(ocs[group], ecs[group]))
        hits = passes.loc[keys].to_numpy()
        rows.append(dict(scenario=condition, seed=seed, method=method, reference_hits=int(passes.sum()),
                         proposals=len(keys), proposed_hits=int(hits.sum()), split_tie_size=tied,
                         selected=len(base), exchanged=len(set(base) - set(reranked)),
                         passes_votes=hits[base].sum(), passes_syn=hits[reranked].sum()))
    return rows


def as_table(rows):
    frame = pd.DataFrame(rows)
    for column in ('plant_pass', 'rank_votes', 'rank_syn', 'plant_ocs'):
        if column in frame:
            frame[column] = frame[column].astype('Int64')
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    scale = commands.add_parser('scale')
    scale.add_argument('scenario', choices=['long', 'null'])
    scale.add_argument('sites', type=int)
    scale.add_argument('seeds', type=int, nargs='+')
    pgk = commands.add_parser('pgk')
    pgk.add_argument('condition')
    pgk.add_argument('seed', type=int)
    pgk.add_argument('panel', type=Path)
    args = parser.parse_args()
    if args.command == 'scale':
        results = [row for seed in args.seeds for row in scale_case(args.sites, seed, args.scenario)]
    else:
        results = pgk_case(args.panel, args.condition, args.seed)
    print(as_table(results).to_csv(sep='\t', index=False), end='')

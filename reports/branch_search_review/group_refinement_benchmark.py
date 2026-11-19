"""Compare sampled and support-guided quartets on matching inputs."""
import argparse
from collections import defaultdict
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
import pandas as pd

import branch_pair_index as index
import group_refinement as refinement
import site_intersection_pilot as pilot


def timed_scale(method, length, seed, seconds, arity=4):
    """Measure the last complete ranked answer available at the common deadline."""
    state, quality, qn, qs, plant = pilot.realization('long', 256, length, seed, arity=arity)
    passes = pilot.qualifying(pilot.reference(state[list(plant)], quality, qn, qs, arity)).item()
    last = dict(proposed=0, hits_proposed=0, hits_top64=0, last_result_seconds=None)
    start = perf_counter()
    support = refinement.known_support(state, quality, .5)
    buckets = defaultdict(list)

    class DeadlineReached(Exception):
        pass

    def checkpoint(votes, rng):
        if perf_counter() - start > seconds:
            raise DeadlineReached
        # Insertions preserve order. Only new keys need whole-alignment scores.
        for group in islice(reversed(votes), len(votes) - last['proposed']):
            buckets[support(group)].append(group)
        top = []
        for value in sorted(buckets, reverse=True):
            tied = buckets[value]
            remaining = 64 - len(top)
            if len(tied) > remaining:
                # Votes break N-support ties. Draw random exact ties independently
                # of branch IDs without consuming the proposal generator's RNG.
                tied_votes = {group: votes[group] for group in tied}
                tied = index.rank_votes(tied_votes, deepcopy(rng))[:remaining]
            top.extend(tied)
            if len(top) == 64:
                break
        elapsed = perf_counter() - start
        if elapsed > seconds:
            raise DeadlineReached
        last.update(proposed=len(votes), hits_proposed=int(plant in votes), hits_top64=int(plant in top),
                    last_result_seconds=elapsed)

    try:
        index.indexed_candidates(state, quality, np.random.default_rng(9000 + seed),
                                 score=support if method == 'group_refined' else None,
                                 draw_budget=2 * index.DEFAULT_DRAWS, checkpoint=checkpoint, arity=arity)
    except DeadlineReached:
        pass
    if not passes:
        last.update(hits_proposed=0, hits_top64=0)
    return dict(last, reference_hits=int(passes), deadline_seconds=seconds)


def pgk_proposals(method, ids, p):
    """Return branch IDs in vote order, cutoff OCN and the number of draws."""
    metrics = refinement.posterior_metrics(p, .5)
    cumulative = p.cumsum(axis=1)
    draw_rng = np.random.default_rng(9109)

    def postings():
        for _ in range(pilot.ROUNDS):
            dest = (draw_rng.random((len(ids), 1, p.shape[2])) >= cumulative).sum(axis=1)
            yield np.concatenate([(dest == d).T for d in range(p.shape[1])])

    keys, _, _, draws = index.rank_witnesses(
        postings(), pilot.ROUNDS, np.random.default_rng(9309),
        score=(lambda group: metrics(group)[0]) if method == 'group_refined' else None,
        witness_seeds=np.random.SeedSequence(9209).spawn(pilot.ROUNDS),
    )
    branch_ids = [tuple(ids[list(key)]) for key in keys]
    support = np.array([metrics(key)[1] for key in keys])
    return branch_ids, support, draws


def pgk(method, panel):
    with np.load(panel / 'posteriors.npz') as data:
        ids, p = data['ids'], data['p']
    cols = [f'branch_id_{i + 1}' for i in range(4)]
    ref = pd.read_csv(panel / 'reference/csubst_cb_4.tsv', sep='\t').set_index(cols)
    hits = set(ref.index[(ref.cutoff_OCNany2spe >= 2) & (ref.omegaCany2spe >= 5)])
    start = perf_counter()
    branch_ids, support, draws = pgk_proposals(method, ids, p)
    order = np.argsort(-support, kind='stable')[:64]
    top = [branch_ids[i] for i in order]
    elapsed = perf_counter() - start
    if branch_ids:
        np.testing.assert_allclose(support, ref.loc[branch_ids, 'cutoff_OCNany2spe'], rtol=1e-10, atol=1e-12)
    return dict(reference_hits=len(hits), proposed=len(branch_ids), hits_proposed=len(hits.intersection(branch_ids)),
                hits_top64=len(hits.intersection(top)), draws=draws, seconds=elapsed)


def witness_sizes(length, seed):
    state, *_ = pilot.realization('long', 256, length, seed)
    events = np.concatenate([(state == d).T for d in (1, 2)])
    events = events[events.sum(axis=1) >= pilot.K]
    sample, _, eligible = index.witness_sample(events, index.DEFAULT_DRAWS, np.random.default_rng(9000 + seed))
    counts = np.zeros(events.shape[1] + 1, dtype=int)
    for start in range(0, len(sample), 4096):
        left, right = np.divmod(sample[start:start + 4096], len(events))
        sizes = (events[left] & events[right]).sum(axis=1)
        counts += np.bincount(sizes, minlength=len(counts))
    return dict(eligible=eligible, sampled=len(sample), four=int(counts[4]), five=int(counts[5]),
                six=int(counts[6]), seven_or_more=int(counts[7:].sum()))


def planted_rank(length, seed):
    state, quality, _, _, planted = pilot.realization('long', 256, length, seed)
    support = refinement.known_support(state, quality, .5)
    keys, *_ = index.indexed_candidates(state, quality, np.random.default_rng(seed + 9000), score=support)
    values = np.fromiter(map(support, keys), int, count=len(keys))
    value = support(planted)
    rank = None
    if planted in keys:
        position = keys.index(planted)
        rank = 1 + int((values > value).sum()) + int((values[:position] == value).sum())
    return dict(support=value, above=int((values > value).sum()), tied=int((values == value).sum()), rank=rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='panel_type', required=True)
    scale = sub.add_parser('scale')
    codon = sub.add_parser('pgk')
    geometry = sub.add_parser('witness-sizes')
    ranking = sub.add_parser('planted-rank')
    for command in (scale, codon):
        command.add_argument('method', choices=['indexed', 'group_refined'])
    for command in (scale, geometry, ranking):
        command.add_argument('sites', type=int)
        command.add_argument('seed', type=int)
    scale.add_argument('--seconds', type=float)
    scale.add_argument('--arity', type=int, default=4, choices=range(2, 257), metavar='K')
    codon.add_argument('panel', type=Path)
    args = parser.parse_args()
    if args.panel_type == 'pgk':
        result = pgk(args.method, args.panel)
    elif args.panel_type == 'witness-sizes':
        result = witness_sizes(args.sites, args.seed)
    elif args.panel_type == 'planted-rank':
        result = planted_rank(args.sites, args.seed)
    elif args.seconds is not None:
        if args.seconds <= 0:
            parser.error('--seconds must be positive')
        result = timed_scale(args.method, args.sites, args.seed, args.seconds, args.arity)
    else:
        row = index.run(args.method, 256, args.sites, args.seed, 'long', arity=args.arity, min_ocn_site_pp=.5)
        result = dict(reference_hits=int(row[5]), hits_proposed=int(row[5] and row[6]),
                      hits_top64=int(row[5] and row[7]),
                      proposed=row[8], draws=row[11], seconds=row[12])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_rss_bytes'] = rss * 1024 if sys.platform.startswith('linux') else rss
    print(json.dumps(result))

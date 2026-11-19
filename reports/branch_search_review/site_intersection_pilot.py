"""K=4 site-based search pilot on synthetic independent terminal branches (NumPy 2+)."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from functools import lru_cache
from itertools import combinations
from math import comb, gcd
from pathlib import Path
import io

import numpy as np
import pandas as pd
from csubst import combination, main_analyze

K = 4
PARENT_CAP = 32
CUTOFF = 'OCNany2spe,2|omegaCany2spe,5'
ROUNDS, PER_POSTING, PER_ROUND = 8, 24, 8192
SCENARIOS = {'short': (24, 512), 'long': (40, 2048), 'hot': (40, 2048),
             'uncertain': (40, 2048), 'null': (40, 2048)}


@lru_cache
def ids(n, k):
    return list(combinations(range(n), k))


def realization(scenario, n, length, seed):
    rng = np.random.default_rng(seed)
    qn = np.full(length, .075)
    qs = np.full(length, .11)
    qs[rng.choice(length, length // 16, replace=False)] = .35
    if scenario == 'hot':
        qn[rng.choice(length, length // 40, replace=False)] = .53
    quality = np.ones(length)
    if scenario == 'uncertain':
        quality[rng.random(length) < .65] = .55
    # Sites have two possible derived N states and one shared derived S state.
    u = rng.random((n, length))
    state = np.select([u < .8 * qn, u < qn, u < qn + qs], [1, 2, 3], default=0)
    planted = tuple(sorted(rng.choice(n, K, replace=False).tolist()))
    if scenario != 'null':
        sites = np.flatnonzero(quality == (.55 if scenario == 'uncertain' else 1.))
        targets = rng.choice(sites, 32 if scenario == 'uncertain' else 4, replace=False)
        state[np.ix_(planted, targets)] = rng.choice([1, 2], len(targets))
    return state, quality, qn, qs, planted


def pack_rows(mask):
    return [int.from_bytes(row, 'little') for row in np.packbits(mask, axis=1, bitorder='little')]


def overlap(masks, branches, k):
    if k == 2:
        return np.fromiter(((masks[a] & masks[b]).bit_count() for a, b in branches), np.int32, len(branches))
    if k == 3:
        return np.fromiter(((masks[a] & masks[b] & masks[c]).bit_count() for a, b, c in branches), np.int32, len(branches))
    counts = ((masks[a] & masks[b] & masks[c] & masks[d]).bit_count() for a, b, c, d in branches)
    return np.fromiter(counts, np.int32, len(branches))


def reference(state, quality, qn, qs, k):
    branches = ids(len(state), k)
    on = np.zeros(len(branches))
    for prob in np.unique(quality):
        for dest in (1, 2):
            on += prob ** k * overlap(pack_rows((quality == prob) & (state == dest)), branches, k)
    os = overlap(pack_rows(state == 3), branches, k)
    en = ((qn * quality) ** k).sum() * (.8 ** k + .2 ** k)
    es = (qs ** k).sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        omega = (on / en) / (os / es)
    df = pd.DataFrame(branches, columns=[f'branch_id_{i+1}' for i in range(k)])
    return df.assign(OCNany2spe=on, omegaCany2spe=omega)


def existing(tabs, mode, *, parent_cap=PARENT_CAP):
    parents = tabs[2]
    g = dict(cutoff_stat=CUTOFF, high_order_pruning=mode, foreground=None, max_combination=parent_cap)
    for k in (3, 4):
        with redirect_stdout(io.StringIO()):
            selected = main_analyze._select_high_order_candidates(parents, g)
        cols = [f'branch_id_{i+1}' for i in range(k)]
        sets = combination._generate_union_candidates_by_shared_subset(selected[cols[:-1]].to_numpy(), k)
        if k == 4:
            return set(map(tuple, sets))
        parents = tabs[k].merge(pd.DataFrame(sets, columns=cols), on=cols, how='inner')


def sample_groups(members, count, rng):
    if comb(len(members), K) <= count:
        return combinations(members, K)
    # Uniform subset of combinations without materializing a dense posting's tuples.
    chosen = set()
    while len(chosen) < count:
        group = rng.choice(members, K, replace=False)
        chosen.add(tuple(sorted(group.tolist())))
    return sorted(chosen)


def site_candidates(state, quality, rng, hard_only=False):
    votes = Counter()
    draws = 0
    for _ in range(ROUNDS):
        if hard_only or np.all(quality == 1.):
            keep = quality == 1.
        else:
            # One categorical branch outcome per site: chosen destination or no N event.
            keep = rng.random(state.shape) < quality
        events = [np.ascontiguousarray(((state == d) & keep).T) for d in (1, 2)]
        postings = [(d, int(i)) for d, arr in enumerate(events) for i in np.flatnonzero(arr.sum(axis=1) >= K)]
        evidence = Counter()
        remaining = PER_ROUND
        for j in rng.permutation(len(postings)):
            d, i = postings[j]
            members = np.flatnonzero(events[d][i]).tolist()
            groups = list(sample_groups(members, min(PER_POSTING, remaining), rng))
            evidence.update(groups)
            remaining -= len(groups)
            if remaining == 0:
                break
        votes.update({group: count - 1 for group, count in evidence.items() if count >= 2})
        draws += PER_ROUND - remaining
    # Uniform random tie breaking independent of the planted branch IDs.
    keys = sorted(votes)
    rng.shuffle(keys)
    return sorted(keys, key=votes.get, reverse=True), draws


def two_site_candidates(state, quality, rng, probe_budget=4_000_000):
    n = len(state)
    votes = Counter()
    proposals = probes = 0
    rounds = ROUNDS if np.any(quality < 1) else 1
    for _ in range(rounds):
        keep = True if rounds == 1 else rng.random(state.shape) < quality
        events = np.concatenate([((state == d) & keep).T for d in (1, 2)])
        events = events[events.sum(axis=1) >= K]
        m = len(events)
        if m < 2:
            continue
        # One destination per branch/site, so two postings at the same site
        # have empty intersection. Pad to machine words for fast batch popcounts.
        events = events[rng.permutation(m)]
        packed = np.packbits(np.pad(events, ((0, 0), (0, (-n) % 64))), axis=1, bitorder='little')
        masks = np.ascontiguousarray(packed).view(np.uint64)
        pair_count = m * (m - 1) // 2
        max_probes = min(pair_count, probe_budget // rounds)
        max_draws = ROUNDS * PER_ROUND // rounds
        # Permute pair ranks without building an M x M matrix. The starting
        # rank and shuffled postings vary by replicate; no pair is visited twice.
        step = int(rng.integers(1, max(2, pair_count)))
        while gcd(step, pair_count) != 1:
            step += 1
        start = int(rng.integers(pair_count))
        used = 0
        for offset in range(0, max_probes, 4096):
            stop = min(offset + 4096, max_probes)
            rank = (start + step * np.arange(offset, stop, dtype=np.int64)) % pair_count
            right = ((1 + np.sqrt(1 + 8 * rank)) * .5).astype(np.int64)
            left = rank - right * (right - 1) // 2
            intersections = masks[left] & masks[right]
            sizes = np.bitwise_count(intersections).sum(axis=1)
            probes += len(rank)
            for idx in np.flatnonzero(sizes >= K):
                members = np.flatnonzero(np.unpackbits(intersections[idx].view(np.uint8), bitorder='little')).tolist()
                groups = list(sample_groups(members, min(PER_POSTING, max_draws - used), rng))
                votes.update(groups)
                used += len(groups)
                if used == max_draws:
                    break
            if used == max_draws:
                break
        proposals += used
    keys = sorted(votes)
    rng.shuffle(keys)
    return sorted(keys, key=votes.get, reverse=True), probes, proposals


def qualifying(frame):
    return frame.OCNany2spe.ge(2) & frame.omegaCany2spe.ge(5)


def small_panel(seeds):
    rows = []
    for scenario, (branches, length) in SCENARIOS.items():
        for seed in seeds:
            state, quality, qn, qs, plant = realization(scenario, branches, length, seed)
            tables = {k: reference(state, quality, qn, qs, k) for k in (2, 3, 4)}
            passing = tables[K].loc[qualifying(tables[K])]
            hits = set(map(tuple, passing.iloc[:, :K].to_numpy()))
            first, draws = site_candidates(state, quality, np.random.default_rng(seed + 5000))
            second, probes, draws2 = two_site_candidates(state, quality, np.random.default_rng(seed + 7000))
            alternatives = [('one_site', first, 0, draws), ('two_sites', second, probes, draws2)]
            if scenario == 'uncertain':
                hard, count = site_candidates(state, quality, np.random.default_rng(seed + 5000), hard_only=True)
                alternatives.append(('one_site_hard_calls', hard, 0, count))
            for mode in ('all_cutoffs', 'ocn_bound'):
                old = existing(tables, mode)
                budget = len(old)
                choices = [(mode, old, 0, 0)]
                choices.extend((name, set(keys[:budget]), p, d) for name, keys, p, d in alternatives)
                for method, checked, pair_probes, tuple_draws in choices:
                    rows.append(dict(scenario=scenario, seed=seed, baseline=mode, method=method,
                                     budget=budget, evaluated=len(checked), hits=len(hits & checked),
                                     exhaustive_hits=len(hits), plant_hit=scenario != 'null' and plant in hits,
                                     plant_found=scenario != 'null' and plant in hits & checked,
                                     pair_probes=pair_probes, tuple_draws=tuple_draws))
    return pd.DataFrame(rows)


def scale_panel(seeds):
    rows = []
    for branches, length in [(40, 2048), (128, 2048), (256, 2048), (256, 8192)]:
        for seed in seeds:
            state, quality, qn, qs, plant = realization('long', branches, length, seed)
            plant_hit = qualifying(reference(state[list(plant)], quality, qn, qs, K)).item()
            first, count = site_candidates(state, quality, np.random.default_rng(seed + 5000))
            second, probes, count2 = two_site_candidates(state, quality, np.random.default_rng(seed + 7000))
            # Additional work on hard calls only. Check N overlap across the whole
            # alignment for every proposed quartet before spending a full score.
            support = sum(overlap(pack_rows(state == dest), second, K) for dest in (1, 2))
            by_support = [second[i] for i in np.argsort(-support, kind='stable')]
            methods = [('one_site', first, 0, count), ('two_sites', second, probes, count2),
                       ('two_sites_bitset_rank', by_support, probes, count2)]
            for name, keys, p, d in methods:
                rows.append(dict(branches=branches, sites=length, seed=seed, method=name,
                                 plant_proposed=plant in keys, proposed=len(keys),
                                 plant_hit=plant_hit, plant_found=plant_hit and plant in keys[:64],
                                 evaluated=min(len(keys), 64), pair_probes=p, tuple_draws=d,
                                 extra_ocn_evaluations=len(second) if name == 'two_sites_bitset_rank' else 0))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=12)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    args.outdir.mkdir(parents=True, exist_ok=True)
    seeds = range(1, args.seeds + 1)
    small_panel(seeds).to_csv(args.outdir / 'small.tsv', sep='\t', index=False)
    scale_panel(seeds).to_csv(args.outdir / 'scale.tsv', sep='\t', index=False)

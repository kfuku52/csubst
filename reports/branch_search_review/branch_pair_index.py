"""Index a site/destination posting by every unordered pair of branches it contains."""
from math import comb
import argparse
import resource
import time

import numpy as np
from scipy import sparse

import site_intersection_pilot as pilot


def witness_sample(events, limit, rng):
    """Block the sparse product and keep a uniform sample of supported site pairs."""
    num_events, branches = events.shape
    indices = []
    indptr = [0]
    for event in events:
        members = np.flatnonzero(event).astype(np.int64)
        left, right = np.triu_indices(len(members), 1)
        large = members[right]
        pairs = large * (large - 1) // 2 + members[left]
        indices.append(pairs)
        indptr.append(indptr[-1] + len(pairs))
    if not indices:
        return [], 0, 0
    pair_codes = np.concatenate(indices)
    pair_count = comb(branches, 2)
    # A sparse tree may have very few observed branch pairs. Avoid allocating
    # a row pointer for every possible pair when transposing the index.
    if pair_count > len(pair_codes):
        distinct, pair_codes = np.unique(pair_codes, return_inverse=True)
        pair_count = len(distinct)
    dtype = np.int32 if pair_count <= np.iinfo(np.int32).max else np.int64
    index = sparse.csr_matrix((np.ones(indptr[-1], dtype=dtype), pair_codes, indptr),
                              shape=(num_events, pair_count))
    index.sort_indices()
    inverted = index.T.tocsr()
    priorities = np.empty(0)
    sample = np.empty(0, dtype=np.int64)
    eligible = 0
    # Bound the number of possible site-pair entries in each product block.
    width = max(1, min(128, 524288 // max(1, num_events)))
    for start in range(0, num_events, width):
        product = (index[start:start + width] @ inverted).tocoo()
        left = product.row.astype(np.int64) + start
        keep = (left < product.col) & (product.data >= comb(pilot.K, 2))
        keys = left[keep] * num_events + product.col[keep]
        eligible += len(keys)
        if not len(keys):
            continue
        priorities = np.concatenate((priorities, rng.random(len(keys))))
        sample = np.concatenate((sample, keys))
        if len(sample) > limit:
            top = np.argpartition(priorities, limit - 1)[:limit]
            priorities, sample = priorities[top], sample[top]
    return sample[np.argsort(priorities)], index.nnz, eligible


def rank_witnesses(posting_rounds, rounds, rng):
    votes = {}
    budget = pilot.ROUNDS * pilot.PER_ROUND // rounds
    index_entries = witnesses = draws = 0
    for events in posting_rounds:
        events = events[events.sum(axis=1) >= pilot.K]
        sample, nnz, eligible = witness_sample(events, budget, rng)
        index_entries += nnz
        witnesses += eligible
        used = 0
        for left, right in zip(*np.divmod(np.asarray(sample, dtype=np.int64), max(1, len(events)))):
            members = np.flatnonzero(events[left] & events[right]).tolist()
            for group in pilot.sample_groups(members, min(pilot.PER_POSTING, budget-used), rng):
                votes[group] = votes.get(group, 0) + 1
                used += 1
            if used == budget:
                break
        draws += used
    keys = sorted(votes)
    rng.shuffle(keys)
    return sorted(keys, key=votes.get, reverse=True), index_entries, witnesses, draws


def indexed_candidates(state, quality, rng):
    rounds = pilot.ROUNDS if np.any(quality < 1) else 1

    def postings():
        for _ in range(rounds):
            keep = True if rounds == 1 else rng.random(state.shape) < quality
            yield np.concatenate([((state == d) & keep).T for d in (1, 2)])
    return rank_witnesses(postings(), rounds, rng)


def run(method, branches, length, seed, scenario):
    state, quality, qn, qs, plant = pilot.realization(scenario, branches, length, seed)
    plant_passes = pilot.qualifying(pilot.reference(state[list(plant)], quality, qn, qs, pilot.K)).item()
    start = time.perf_counter()
    if method == 'indexed':
        keys, entries, witnesses, draws = indexed_candidates(state, quality, np.random.default_rng(seed + 9000))
    else:
        keys, entries, draws = pilot.two_site_candidates(state, quality, np.random.default_rng(seed + 7000))
        witnesses = -1
    # Use the same whole-alignment N support and the same full-score budget on both arms.
    support = np.zeros(len(keys))
    for prob in np.unique(quality):
        for dest in (1, 2):
            rows = pilot.pack_rows((state == dest) & (quality == prob))
            support += prob ** pilot.K * pilot.overlap(rows, keys, pilot.K)
    order = np.argsort(-support, kind='stable')[:64]
    top = [keys[i] for i in order]
    elapsed = time.perf_counter() - start
    return (scenario, branches, length, seed, method, plant_passes, plant in keys, plant in top,
            len(keys), entries, witnesses, draws, elapsed, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['indexed','two_sites'])
    parser.add_argument('branches', type=int)
    parser.add_argument('sites', type=int)
    parser.add_argument('seed', type=int)
    parser.add_argument('--scenario', choices=list(pilot.SCENARIOS), default='long')
    args = parser.parse_args()
    print(*run(args.method, args.branches, args.sites, args.seed, args.scenario), sep='\t')

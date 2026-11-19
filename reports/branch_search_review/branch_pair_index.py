"""Index a site/destination posting by every unordered pair of branches it contains."""
from math import comb
import argparse
import resource
import time

import numpy as np
from scipy import sparse

import group_refinement
import site_intersection_pilot as pilot


DEFAULT_DRAWS = pilot.ROUNDS * pilot.PER_ROUND


def witness_sample(events, limit, rng, *, arity=pilot.K):
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
        keep = (left < product.col) & (product.data >= comb(arity, 2))
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


def rank_witnesses(posting_rounds, rounds, rng, *, score=None, draw_budget=DEFAULT_DRAWS,
                   witness_seeds=None, checkpoint=None, arity=pilot.K):
    votes = {}
    budget = draw_budget // rounds
    if budget < 1:
        raise ValueError("draw budget must allow at least one proposal per round")
    index_entries = witnesses = draws = 0
    for i, events in enumerate(posting_rounds):
        events = events[events.sum(axis=1) >= arity]
        sample_rng = rng if witness_seeds is None else np.random.default_rng(witness_seeds[i])
        sample, nnz, eligible = witness_sample(events, budget, sample_rng, arity=arity)
        patterns = group_refinement.site_pattern_moves(events, arity, rng) if score is not None else None
        index_entries += nnz
        witnesses += eligible
        used = 0
        next_checkpoint = 4096
        for left, right in zip(*np.divmod(np.asarray(sample, dtype=np.int64), max(1, len(events)))):
            members = np.flatnonzero(events[left] & events[right]).tolist()
            limit = min(pilot.PER_POSTING, budget - used)
            if score is None:
                groups = pilot.sample_groups(members, limit, rng, arity=arity)
            else:
                groups = group_refinement.refine_group(members, arity, limit, score, rng, patterns=patterns)
            for group in groups:
                votes[group] = votes.get(group, 0) + 1
                used += 1
            if checkpoint is not None and used >= next_checkpoint:
                checkpoint(votes, rng)
                next_checkpoint += 4096
            if used == budget:
                break
        draws += used
    if checkpoint is not None:
        checkpoint(votes, rng)
    return rank_votes(votes, rng), index_entries, witnesses, draws


def rank_votes(votes, rng):
    keys = sorted(votes)
    rng.shuffle(keys)
    return sorted(keys, key=votes.get, reverse=True)


def indexed_candidates(state, quality, rng, *, score=None, draw_budget=DEFAULT_DRAWS, checkpoint=None, arity=pilot.K):
    rounds = pilot.ROUNDS if np.any(quality < 1) else 1

    def postings():
        for _ in range(rounds):
            keep = True if rounds == 1 else rng.random(state.shape) < quality
            yield np.concatenate([((state == d) & keep).T for d in (1, 2)])
    return rank_witnesses(postings(), rounds, rng, score=score, draw_budget=draw_budget, checkpoint=checkpoint,
                          arity=arity)


def run(method, branches, length, seed, scenario, *, draw_budget=DEFAULT_DRAWS, arity=pilot.K,
        min_ocn_site_pp=None):
    if not 2 <= arity <= branches:
        raise ValueError("arity must be between 2 and the number of branches")
    if method == 'two_sites' and arity != pilot.K:
        raise ValueError("the direct site-pair pilot supports only K=4")
    if min_ocn_site_pp is None:
        min_ocn_site_pp = .5 if method == 'group_refined' else 0.
    if not 0 <= min_ocn_site_pp <= 1:
        raise ValueError("site PP must be between 0 and 1")
    state, quality, qn, qs, plant = pilot.realization(scenario, branches, length, seed, arity=arity)
    planted_state = state[list(plant)]
    reference = pilot.reference(planted_state, quality, qn, qs, arity)
    # Apply the PP minimum only to the observed-count cutoff. The full
    # alignment still supplies the reference omega, including at weak sites.
    pp = quality ** arity
    shared = (planted_state == 1).all(axis=0) | (planted_state == 2).all(axis=0)
    cutoff_ocn = pp[shared & (pp >= min_ocn_site_pp)].sum()
    plant_passes = bool(cutoff_ocn >= 2 and reference.omegaCany2spe.item() >= 5)
    start = time.perf_counter()
    if method in ('indexed', 'group_refined'):
        score = group_refinement.known_support(state, quality, min_ocn_site_pp) if method == 'group_refined' else None
        keys, entries, witnesses, draws = indexed_candidates(
            state, quality, np.random.default_rng(seed + 9000), score=score, draw_budget=draw_budget, arity=arity
        )
    else:
        keys, entries, draws = pilot.two_site_candidates(state, quality, np.random.default_rng(seed + 7000))
        witnesses = -1
    # Use the same whole-alignment N support and the same full-score budget on both arms.
    support = np.zeros(len(keys))
    for prob in np.unique(quality):
        if prob ** arity < min_ocn_site_pp:
            continue
        for dest in (1, 2):
            rows = pilot.pack_rows((state == dest) & (quality == prob))
            support += prob ** arity * pilot.overlap(rows, keys, arity)
    order = np.argsort(-support, kind='stable')[:64]
    top = [keys[i] for i in order]
    elapsed = time.perf_counter() - start
    return (scenario, branches, length, seed, method, plant_passes, plant in keys, plant in top,
            len(keys), entries, witnesses, draws, elapsed, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['indexed','two_sites','group_refined'])
    parser.add_argument('branches', type=int)
    parser.add_argument('sites', type=int)
    parser.add_argument('seed', type=int)
    parser.add_argument('--scenario', choices=list(pilot.SCENARIOS), default='long')
    parser.add_argument('--arity', type=int, default=pilot.K)
    parser.add_argument('--min-ocn-site-pp', type=float,
                        help='Minimum site PP for OCN selection and ranking. Defaults to 0.5 for group_refined and 0 otherwise.')
    parser.add_argument('--draw-budget', type=int, default=DEFAULT_DRAWS,
                        help='Maximum emitted branch combinations across indexed proposal rounds.')
    args = parser.parse_args()
    if args.method == 'two_sites' and args.draw_budget != DEFAULT_DRAWS:
        parser.error('--draw-budget applies to the index methods')
    print(*run(args.method, args.branches, args.sites, args.seed, args.scenario,
               draw_budget=args.draw_budget, arity=args.arity, min_ocn_site_pp=args.min_ocn_site_pp), sep='\t')

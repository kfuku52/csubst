"""MaxAlign-inspired group removal for site-index proposals."""
from functools import lru_cache
from itertools import combinations
from math import comb, isqrt

import numpy as np

import site_intersection_pilot as pilot


def known_support(state, quality, minimum):
    """Count supported sites for any branch subset under the pilot's event model."""
    event_rows = np.concatenate([state == d for d in (1, 2)], axis=1)
    branch_masks = pilot.pack_rows(event_rows)

    @lru_cache(maxsize=None)
    def site_mask(arity):
        keep = np.tile(quality ** arity >= minimum, 2)
        return pilot.pack_rows(keep[None, :])[0]

    def score(group):
        shared = branch_masks[group[0]] & site_mask(len(group))
        for branch in group[1:]:
            shared &= branch_masks[branch]
        return shared.bit_count()

    return score


def posterior_metrics(p, minimum):
    """Count passing sites and sum their shared-destination probabilities."""
    @lru_cache(maxsize=8192)
    def score(group):
        joint = p[group[0]].copy()
        for branch in group[1:]:
            joint *= p[branch]
        by_site = joint.sum(axis=0)
        passing = by_site[by_site >= minimum]
        return len(passing), passing.sum()

    return score


def _children(group, target, rng, move_limit):
    # Bounded deletion moves also explore subsets absent from the site patterns.
    remove = min(2, len(group) - target)
    count = comb(len(group), remove)
    if count <= move_limit:
        return combinations(group, len(group) - remove)
    ranks = rng.choice(count, move_limit, replace=False)
    children = []
    for rank in ranks.tolist():
        if remove == 1:
            children.append(group[:rank] + group[rank + 1:])
        else:
            right = (1 + isqrt(1 + 8 * rank)) // 2
            left = rank - right * (right - 1) // 2
            children.append(group[:left] + group[left + 1:right] + group[right + 1:])
    return children


def site_pattern_moves(events, target, rng, limit=24):
    """Propose removing all branches that block one additional site/destination."""
    @lru_cache(maxsize=256)
    def moves(group):
        flags = events[:, group]
        sizes = np.count_nonzero(flags, axis=1)
        viable = flags[(sizes >= target) & (sizes < len(group))]
        if not len(viable):
            return ()
        patterns, frequency = np.unique(np.packbits(viable, axis=1), axis=0, return_counts=True)
        masks = np.unpackbits(patterns, axis=1, count=len(group)).astype(bool)
        if len(masks) > limit:
            order = rng.permutation(len(masks))
            gain = frequency / (len(group) - masks.sum(axis=1))
            masks = masks[order[np.argsort(-gain[order], kind='stable')[:limit]]]
        members = np.asarray(group)
        return tuple(tuple(members[mask].tolist()) for mask in masks)

    return moves


def refine_group(members, target, count, score, rng, *, patterns=None, width=4, move_limit=24):
    """Keep several branch subsets with the most whole-alignment supporting sites."""
    group = tuple(members)
    if len(group) == target:
        return [group]
    if len(group) <= target + 2:
        children = list(_children(group, target, rng, move_limit))
        keep = min(count, width)
        if len(children) > keep:
            rng.shuffle(children)
            children.sort(key=score, reverse=True)
        return children[:keep]
    frontier = [group]
    while True:
        unique = {}
        for parent in frontier:
            if len(parent) == target:
                unique[parent] = None
                continue
            unique.update(dict.fromkeys(_children(parent, target, rng, move_limit)))
            if patterns is not None and len(parent) > target + 2:
                unique.update(dict.fromkeys(patterns(parent)))
        children = list(unique)
        sizes = {len(child) for child in children}
        finished = sizes == {target}
        keep = min(width, count) if finished else width
        if len(children) > keep:
            rng.shuffle(children)
            if len(sizes) == 1:
                children.sort(key=score, reverse=True)
            else:
                initial = score(group)
                children.sort(key=lambda c: (score(c) - initial) / (len(group) - len(c)), reverse=True)
            children = children[:keep]
        if finished or all(len(child) == target for child in children):
            return children[:count]
        frontier = children

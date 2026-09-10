"""Conditional, uniform sampling of non-overlapping foreground configurations.

This is the scan assignment null, not a model of sequence evolution. Bin
membership and eligibility must be fixed independently of the foreground
assignment. Each lineage has a fixed number of components in each bin.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np


ENUMERATION_PROPOSAL_LIMIT = 10_000
MAX_REJECTION_DRAWS = 10_000

Configuration = tuple[tuple[int, ...], ...]


class SamplingError(RuntimeError):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = int(attempts)


@dataclass
class CladePlan:
    pools: tuple[tuple[int, ...], ...]
    counts: tuple[tuple[int, ...], ...]
    descendants: tuple[frozenset[int], ...]
    observed: Configuration
    proposal_count: int
    configurations: tuple[Configuration, ...] | None


def nonoverlapping(configuration, descendants):
    selected = [int(i) for group in configuration for i in group]
    if len(selected) != len(set(selected)):
        return False
    selected_set = set(selected)
    return all(not ((descendants[i] - {i}) & selected_set) for i in selected)


def _bin_allocations(pool, counts):
    """Enumerate unordered subsets for labeled groups, without replacement."""
    if not counts:
        yield ()
        return
    for selected in itertools.combinations(pool, counts[0]):
        selected_set = set(selected)
        remaining = tuple(i for i in pool if i not in selected_set)
        for tail in _bin_allocations(remaining, counts[1:]):
            yield (selected,) + tail


def build_plan(bin_array, eligible, observed, descendants):
    bin_array = np.asarray(bin_array, dtype=np.int64)
    eligible = np.asarray(eligible, dtype=bool)
    observed = tuple(tuple(sorted(int(i) for i in group)) for group in observed)
    descendants = tuple(frozenset(int(i) for i in values) for values in descendants)
    selected = [i for group in observed for i in group]
    if not selected:
        raise ValueError("No foreground components are available for scan calibration.")
    if not all(eligible[i] for i in selected):
        raise ValueError("An observed foreground component has no analyzable target branches.")
    if not nonoverlapping(observed, descendants):
        raise ValueError("Scan calibration requires non-overlapping observed foreground clades.")
    used_bins = sorted(set(int(bin_array[i]) for i in selected))
    pools = tuple(
        tuple(np.flatnonzero(eligible & (bin_array == bin_no)).tolist())
        for bin_no in used_bins
    )
    counts = tuple(
        tuple(sum(int(bin_array[i]) == bin_no for i in group) for group in observed)
        for bin_no in used_bins
    )
    proposal_count = 1
    for pool, bin_counts in zip(pools, counts):
        remaining = len(pool)
        for count in bin_counts:
            proposal_count *= math.comb(remaining, count)
            remaining -= count
    configurations = None
    if proposal_count <= ENUMERATION_PROPOSAL_LIMIT:
        configurations_list = []
        allocations = [_bin_allocations(pool, count) for pool, count in zip(pools, counts)]
        for allocation in itertools.product(*allocations):
            configuration = tuple(
                tuple(sorted(i for bin_groups in allocation for i in bin_groups[group_index]))
                for group_index in range(len(observed))
            )
            if nonoverlapping(configuration, descendants):
                configurations_list.append(configuration)
        configurations = tuple(sorted(configurations_list))
        if observed not in configurations:
            raise ValueError("The observed configuration is outside the scan assignment space.")
    return CladePlan(pools, counts, descendants, observed, proposal_count, configurations)


def sample_configuration(plan, rng):
    if plan.configurations is not None:
        index = int(rng.integers(len(plan.configurations)))
        return plan.configurations[index], 1
    # Every bin allocation has the same number of orderings. Consequently each
    # complete proposal is uniform, and rejecting the WHOLE overlapping proposal
    # gives a uniform draw from the valid space. Sequentially excluding clades
    # inside a proposal would instead weight configurations unequally.
    for attempt in range(1, MAX_REJECTION_DRAWS + 1):
        groups: list[list[int]] = [[] for _ in plan.observed]
        for pool, counts in zip(plan.pools, plan.counts):
            selected = rng.choice(pool, size=sum(counts), replace=False)
            offset = 0
            for group, count in zip(groups, counts):
                group.extend(int(i) for i in selected[offset:offset + count])
                offset += count
        configuration = tuple(tuple(sorted(group)) for group in groups)
        if nonoverlapping(configuration, plan.descendants):
            return configuration, attempt
    raise SamplingError(
        "No non-overlapping scan configuration was drawn in {} complete proposals; "
        "calibration is unavailable. The assignment space was not changed.".format(MAX_REJECTION_DRAWS),
        attempts=MAX_REJECTION_DRAWS,
    )

"""Exact small-state oracle for background-dependent convergence expectations.

Research API only: two interacting sites, two disjoint branches, known parent
states and a known landscape. Observations are net endpoint substitutions,
not the number of hidden recurrent events. No omega statistic or ASR is fitted.
"""

from dataclasses import dataclass
import itertools

import numpy as np
from scipy.linalg import expm
from scipy.special import softmax


CATEGORIES = ('any2any', 'any2spe', 'spe2any', 'spe2spe', 'any2dif',
              'dif2any', 'dif2spe', 'spe2dif', 'dif2dif')


@dataclass
class PairLandscape:
    fields: np.ndarray
    coupling: np.ndarray
    mutation_rates: np.ndarray

    def __post_init__(self):
        self.fields = np.array(self.fields, dtype=float, copy=True)
        self.coupling = np.array(self.coupling, dtype=float, copy=True)
        self.mutation_rates = np.array(self.mutation_rates, dtype=float, copy=True)
        if self.fields.ndim != 2 or self.fields.shape[0] != 2:
            raise ValueError('Pair fields must have shape (2, alphabet_size).')
        k = self.fields.shape[1]
        if not 2 <= k <= 8 or self.coupling.shape != (k, k) or self.mutation_rates.shape != (2,):
            raise ValueError('Invalid small-state pair landscape dimensions (2 <= alphabet_size <= 8).')
        if not all(np.isfinite(v).all() for v in (self.fields, self.coupling, self.mutation_rates)):
            raise ValueError('Pair landscape parameters must be finite.')
        if np.any(self.mutation_rates <= 0):
            raise ValueError('Mutation rates must be positive.')

    @property
    def alphabet_size(self):
        return self.fields.shape[1]

    @property
    def states(self):
        return np.array(list(itertools.product(range(self.alphabet_size), repeat=2)))

    def fitness(self, states):
        states = np.asarray(states)
        return (self.fields[0, states[..., 0]] + self.fields[1, states[..., 1]] +
                self.coupling[states[..., 0], states[..., 1]])

    def stationary(self):
        return softmax(self.fitness(self.states))

    def generator(self):
        states = self.states
        fitness = self.fitness(states)
        q = np.zeros((len(states), len(states)))
        for i, old in enumerate(states):
            for j, new in enumerate(states):
                changed = np.flatnonzero(old != new)
                if len(changed) == 1:
                    rate = self.mutation_rates[changed[0]] / (self.alphabet_size - 1)
                    q[i, j] = rate * np.exp((fitness[j] - fitness[i]) / 2.)
            q[i, i] = -q[i].sum()
        if not np.isfinite(q).all():
            raise ValueError('Pair transition rates overflowed; reduce fitness differences.')
        return q


def matched_independent_landscape(model):
    """Give the baseline the true marginal composition AND mean rate per site.

    This generous oracle baseline isolates changing local background from
    improvements due merely to site preference or rate heterogeneity.
    """
    k = model.alphabet_size
    pi = model.stationary()
    table = pi.reshape(k, k)
    fields = np.log(np.stack((table.sum(axis=1), table.sum(axis=0))))
    baseline = PairLandscape(fields, np.zeros((k, k)), np.ones(2))
    rates = []
    for q, stationary in ((model.generator(), pi), (baseline.generator(), baseline.stationary())):
        per_site = np.zeros(2)
        for i, old in enumerate(model.states):
            for j, new in enumerate(model.states):
                changed = np.flatnonzero(old != new)
                if len(changed) == 1:
                    per_site[changed[0]] += stationary[i] * q[i, j]
        rates.append(per_site)
    return PairLandscape(fields, np.zeros((k, k)), rates[0] / rates[1])


def validate_parents(parents, alphabet_size):
    values = np.asarray(parents)
    if values.shape != (2, 2) or values.dtype.kind not in 'iu' or np.any(values < 0) or np.any(values >= alphabet_size):
        raise ValueError('Parents must be integer states with shape (2 branches, 2 sites).')
    return values


def endpoint_categories(parents, endpoints):
    """Derive all categories from one pair of endpoint observations per draw."""
    parents, endpoints = np.asarray(parents), np.asarray(endpoints)
    if parents.shape != (2, 2) or endpoints.ndim != 3 or endpoints.shape[1:] != (2, 2):
        raise ValueError('Expected parents (2,2) and endpoints (draws,2,2).')
    both_changed = np.all(endpoints != parents, axis=1)
    same_from = parents[0] == parents[1]
    same_to = endpoints[:, 0] == endpoints[:, 1]
    # Atomic logical intersections preserve every inclusion relationship.
    masks = (both_changed, both_changed & same_to, both_changed & same_from,
             both_changed & same_from & same_to, both_changed & ~same_to,
             both_changed & ~same_from, both_changed & ~same_from & same_to,
             both_changed & same_from & ~same_to, both_changed & ~same_from & ~same_to)
    return np.stack([v.sum(axis=1) for v in masks], axis=1)


def exact_endpoint_distribution(model, parents, lengths):
    parents = validate_parents(parents, model.alphabet_size)
    lengths = np.asarray(lengths, dtype=float)
    if lengths.shape != (2,) or not np.isfinite(lengths).all() or np.any(lengths < 0):
        raise ValueError('Two finite nonnegative branch lengths are required.')
    q = model.generator()
    probabilities = []
    for parent, duration in zip(parents, lengths):
        index = int(parent[0] * model.alphabet_size + parent[1])
        p = expm(duration * q)[index]
        if p.min() < -1e-12 or not np.isclose(p.sum(), 1., atol=1e-12):
            raise ValueError('Invalid matrix-exponential transition probabilities.')
        p = np.maximum(p, 0.)
        probabilities.append(p / p.sum())
    states = model.states
    endpoints = np.array(list(itertools.product(states, states)))
    probability = np.outer(*probabilities).reshape(-1)
    return endpoints, probability


def exact_category_pmfs(model, parents, lengths):
    endpoints, probability = exact_endpoint_distribution(model, parents, lengths)
    counts = endpoint_categories(parents, endpoints)
    return {name: np.bincount(counts[:, i], weights=probability, minlength=3)
            for i, name in enumerate(CATEGORIES)}


def repeated_pair_pmf(pmf, num_pairs):
    """Independent pair blocks; sites WITHIN each pair remain dependent."""
    pmf = np.asarray(pmf, dtype=float)
    if pmf.shape != (3,) or not np.isfinite(pmf).all() or np.any(pmf < 0) or not np.isclose(pmf.sum(), 1.):
        raise ValueError('Expected a valid pair-count PMF on 0,1,2.')
    if isinstance(num_pairs, bool) or not isinstance(num_pairs, (int, np.integer)) or num_pairs < 1:
        raise ValueError('num_pairs must be a positive integer.')
    out = np.array([1.])
    for _ in range(num_pairs):
        out = np.convolve(out, pmf)
    return out / out.sum()


def upper_tail(pmf):
    return np.minimum(1., np.cumsum(np.asarray(pmf)[::-1])[::-1])

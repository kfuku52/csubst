"""Blocked CTMC endpoint inference, conditional on a fixed fitted model.

No ancestral histories or node-by-site-by-state-by-state array is stored.
Rate categories are integrated using their posterior probability at each site.
An edge joint is a distribution of endpoints, not a count of CTMC jumps.
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.sparse.csgraph import shortest_path
from scipy.special import logsumexp


_PRUNING_LOG_THRESHOLD = np.sqrt(np.finfo(float).tiny)


def _conditional_transition(transition, log_right):
    """Normalize P times child evidence separately for each parent state."""
    with np.errstate(divide='ignore'):
        log_mass = np.log(transition) + log_right
    maximum = log_mass.max(axis=1, keepdims=True)
    mass = np.exp(log_mass - np.where(np.isfinite(maximum), maximum, 0.))
    total = mass.sum(axis=1, keepdims=True)
    np.divide(mass, total, out=mass, where=total > 0)
    return mass


def _scaled_outside(parent, denominator):
    # Bridge reducers need an outside message and are invariant to its scale.
    with np.errstate(divide='ignore'):
        log_outside = np.log(parent) - np.log(np.where(denominator > 0, denominator, 1.))
    log_outside[denominator == 0] = -np.inf
    return np.exp(log_outside - log_outside.max(axis=1, keepdims=True))


def _recover_rare_pruning_rows(inside, messages, scales, node, children, sites):
    """Recompute the child product before a lost state can bias the posterior."""
    log_inside = np.zeros((len(sites), inside.shape[-1]))
    log_scale = np.zeros(len(sites))
    with np.errstate(divide='ignore'):
        for child in children:
            log_inside += np.log(messages[child, sites])
            log_scale += scales[child, sites]
    maximum = log_inside.max(axis=1)
    finite = np.isfinite(maximum)
    inside[node, sites] = np.exp(log_inside - np.where(finite, maximum, 0.)[:, None])
    scales[node, sites] = log_scale + maximum


class TransitionSeries:
    """Reuse nonnegative CTMC transition powers across lengths and rates."""

    def __init__(self, q):
        self.q = q
        self.rate = float(np.max(-np.diag(q)))
        # A partial sum must include a path to every reachable pair before a
        # relative tail bound can certify all entries.
        distances = shortest_path(q > 0, directed=True, unweighted=True)
        self._support_steps = int(np.max(distances, where=np.isfinite(distances), initial=0.))
        capacity = min(256, max(1, (8 * 1024**2) // q.nbytes))
        self._powers = np.empty((capacity, *q.shape))
        identity = np.eye(len(q))
        self._powers[0] = identity
        self._count = 1
        self._kernel = identity + q / self.rate if self.rate > 0 else identity
        self._row_norm = float(self._kernel.sum(axis=1).max())
        self._nonnegative = bool((self._kernel >= 0).all())

    def _sum(self, time, *, occupation=False):
        # exp(Qt) is a Poisson-weighted sum of powers of I + Q / rate.
        tau = self.rate * time
        if tau == 0:
            return np.eye(len(self.q))
        capacity = len(self._powers)
        tau_bound = tau * self._row_norm
        if tau_bound > 32 or capacity <= self._support_steps or not self._nonnegative:
            return None
        coefficients = np.empty(capacity)
        coefficients[0] = np.exp(-tau)
        target = 1e-18
        for n in range(1, capacity):
            coefficients[n] = coefficients[n - 1] * tau / n
            if n < self._support_steps or n + 1 <= tau_bound:
                continue
            # The next Poisson terms have a bounded geometric ratio. In the
            # integrated series their weight is spread over lower powers.
            tail = coefficients[n] * max(1., self._row_norm)**n * tau_bound / (n + 1 - tau_bound)
            if tail > target:
                continue
            while self._count <= n:
                i = self._count
                self._powers[i] = self._powers[i - 1] @ self._kernel
                self._count += 1
            weights = coefficients[:n + 1]
            if occupation:
                # For the mean transition over [0, t], integrate each
                # embedded-chain power. At order n this distributes the
                # Poisson mass of j equally over powers 0 through j.
                weights = np.cumsum((weights / np.arange(1, n + 2))[::-1])[::-1]
            matrix = (weights @ self._powers[:n + 1].reshape(n + 1, -1)).reshape(self.q.shape)
            target = float(matrix[matrix > 0].min() * np.finfo(float).eps * .25)
            if tail <= target:
                return matrix
        return None

    def transition(self, time):
        """Return exp(Q * time)."""
        matrix = self._sum(time)
        return expm(self.q * time) if matrix is None else matrix

    def occupation(self, time):
        """Return the integral of exp(Q * u) over [0, time]."""
        matrix = self._sum(time, occupation=True)
        if matrix is not None:
            return time * matrix
        k = len(self.q)
        augmented = np.zeros((2 * k, 2 * k))
        augmented[:k, :k] = self.q
        augmented[:k, k:] = np.eye(k)
        return expm(augmented * time)[:k, k:]


@dataclass
class EndpointBlock:
    start: int
    stop: int
    child: int
    parent: int
    node: np.ndarray
    joint: np.ndarray | None
    predictive: np.ndarray | None
    reduced: dict | None = None
    reduced_predictive: dict | None = None
    category_node: np.ndarray | None = None
    rate_posterior: np.ndarray | None = None


class EndpointModel:
    """Finite-state tree model with a site-independent discrete rate mixture."""

    def __init__(self, parents, lengths, q, pi, rates=(1.0,), weights=(1.0,)):
        self.parents = np.asarray(parents)
        self.lengths = np.asarray(lengths, dtype=float)
        self.q, self.pi = np.asarray(q, dtype=float), np.asarray(pi, dtype=float)
        self.rates, self.weights = np.asarray(rates, dtype=float), np.asarray(weights, dtype=float)
        p = self.parents
        if (p.ndim != 1 or not p.size or p.dtype.kind not in 'iu'
                or np.any((p < -1) | (p >= p.size)) or np.sum(p == -1) != 1):
            raise ValueError('Endpoint parents must contain exactly one root and valid node indices.')
        self.root = int(np.flatnonzero(p == -1)[0])
        if (self.lengths.shape != p.shape or not np.isfinite(self.lengths).all()
                or np.any(self.lengths < 0) or self.lengths[self.root] != 0):
            raise ValueError('Endpoint lengths must be finite/nonnegative with zero at the root.')
        k = self.pi.size
        if (self.pi.ndim != 1 or k < 2 or self.q.shape != (k, k)
                or not np.isfinite(self.q).all() or not np.isfinite(self.pi).all()
                or np.any(self.pi < 0) or not np.isclose(self.pi.sum(), 1, atol=1e-10)):
            raise ValueError('Invalid endpoint Q or root frequencies.')
        off = self.q.copy()
        np.fill_diagonal(off, 0)
        if (np.any(off < 0) or not np.allclose(self.q.sum(axis=1), 0, atol=1e-10)
                or not np.allclose(self.pi @ self.q, 0, atol=1e-9)):
            raise ValueError('Endpoint Q must be a generator stationary for pi.')
        if (self.rates.ndim != 1 or not self.rates.size or self.weights.shape != self.rates.shape
                or not np.isfinite(self.rates).all() or not np.isfinite(self.weights).all()
                or np.any(self.rates < 0) or np.any(self.weights <= 0)
                or not np.isclose(self.weights.sum(), 1, atol=1e-10)):
            raise ValueError('Invalid endpoint rate categories or mixture weights.')
        self.children: list[list[int]] = [[] for _ in p]
        for child, parent in enumerate(p):
            if parent >= 0:
                self.children[parent].append(child)
        self.order: list[int] = []
        visited = set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node in visited:
                raise ValueError('Endpoint tree contains a cycle.')
            visited.add(node)
            self.order.append(node)
            stack.extend(reversed(self.children[node]))
        if len(self.order) != p.size:
            raise ValueError('Endpoint tree is disconnected.')
        self.leaves = {i for i, children in enumerate(self.children) if not children}
        self._transition_series = TransitionSeries(self.q)
        self._transitions = OrderedDict()
        # Cap the transition cache independently of the number of sites/nodes.
        self._cache_items = max(1, (32 * 1024 * 1024) // self.q.nbytes)

    def transition(self, child, category):
        t = float(self.lengths[child] * self.rates[category])
        if t not in self._transitions:
            matrix = self._transition_series.transition(t)
            if matrix.min() < -1e-12 or not np.allclose(matrix.sum(axis=1), 1, atol=1e-10):
                raise ValueError('Invalid CTMC transition matrix.')
            matrix = np.maximum(matrix, 0)
            matrix /= matrix.sum(axis=1, keepdims=True)
            self._transitions[t] = matrix
            if len(self._transitions) > self._cache_items:
                self._transitions.popitem(last=False)
        self._transitions.move_to_end(t)
        return self._transitions[t]

    def iter_blocks(self, tips, block_size=64, branch_ids=None, joint=True, predictive=False, transform=None,
                    predictive_transform=None, category_nodes=False, unselected_transform=None,
                    transform_rate_context=False, tip_missing=None, rate_posterior=False):
        """Yield one edge/site block; root records only contain a node marginal.

        Tips are observation likelihoods, NOT posterior probabilities. Missing
        observations have likelihood one for every state. Predictive records
        use sum_c P(c|D) P(parent=a|D,c) P_c(a,d), a fitted conditional
        endpoint prediction; they are not unconditional null probabilities.
        A separate predictive transform can omit observed-only summaries. Each
        transform's outputs are summed over rate categories before yielding.
        An optional unselected transform retains coarse summaries on branches
        outside branch_ids without constructing their detailed joint events.
        With transform_rate_context, reducers receive unweighted boundary
        messages, category-scaled time and the posterior site-category weights.
        Other observed reducers contract the factored joint. For rare rows,
        the factors may use a conditional transition and unit right message.
        tip_missing can mark rows to read as all-ones emissions without copying
        their full input tip arrays.
        With rate_posterior, the root record exposes the site-category posteriors.
        """
        if isinstance(block_size, bool) or int(block_size) != block_size or block_size < 1:
            raise ValueError('Endpoint block size must be a positive integer.')
        if set(tips) != self.leaves:
            raise ValueError('Endpoint inference requires likelihoods for every leaf.')
        arrays = {node: np.asarray(values) for node, values in tips.items()}
        first = next(iter(arrays.values()))
        if first.ndim != 2 or first.shape[1] != self.pi.size:
            raise ValueError('Endpoint tip likelihoods must have site-by-state shape.')
        missing = {} if tip_missing is None else {
            node: np.asarray(mask, dtype=bool) for node, mask in tip_missing.items()
        }
        if any(node not in self.leaves or mask.shape != (first.shape[0],) for node, mask in missing.items()):
            raise ValueError('Missing tip masks must identify leaf sites in the input alignment.')
        for node, values in arrays.items():
            if values.shape != first.shape:
                raise ValueError('Endpoint tip likelihoods must be equal-sized, finite and nonnegative.')
            if not np.isfinite(values).all() or np.any(values < 0):
                invalid = (~np.isfinite(values) | (values < 0)).any(axis=1)
                if node not in missing or np.any(invalid & ~missing[node]):
                    raise ValueError('Endpoint tip likelihoods must be equal-sized, finite and nonnegative.')
        missing = {node: mask for node, mask in missing.items() if mask.any()}
        selected = set(self.order[1:]) if branch_ids is None else {int(i) for i in branch_ids}
        if not selected.issubset(set(self.order) - {self.root}):
            raise ValueError('Endpoint branch selection contains a root or unknown branch.')
        n, k, nc = self.parents.size, self.pi.size, self.rates.size
        forward_floor = (2 * k) / np.finfo(float).max
        capacity = min(int(block_size), first.shape[0])
        inside_buffer = np.empty((nc, n, capacity, k))
        post_buffer = np.empty_like(inside_buffer)
        scale_buffer = np.empty((nc, n, capacity))
        internal_postorder = [node for node in reversed(self.order) if node not in arrays]
        for start in range(0, first.shape[0], int(block_size)):
            stop = min(start + int(block_size), first.shape[0])
            size = stop - start
            inside = inside_buffer[:, :, :size]
            post = post_buffer[:, :, :size]
            scales = scale_buffer[:, :, :size]
            inside.fill(1)
            scales.fill(0)
            for node, values in arrays.items():
                inside[:, node] = values[start:stop]
                if node in missing:
                    inside[:, node, missing[node][start:stop]] = 1.
            for c in range(nc):
                for node in internal_postorder:
                    children = self.children[node]
                    rare_sites = None
                    for child in children:
                        post[c, child] = inside[c, child] @ self.transition(child, c).T
                        inside[c, node] *= post[c, child]
                        scales[c, node] += scales[c, child]
                        norm = inside[c, node].max(axis=1)
                        if norm.min() < _PRUNING_LOG_THRESHOLD:
                            # Once an intermediate product is very small, an
                            # earlier multiplication can already have lost a
                            # state that matters to a later sibling.
                            rare = norm < _PRUNING_LOG_THRESHOLD
                            rare_sites = rare if rare_sites is None else rare_sites | rare
                        inside[c, node] /= np.where(norm > 0, norm, 1)[:, None]
                        with np.errstate(divide='ignore'):
                            scales[c, node] += np.log(norm)
                    if rare_sites is not None:
                        _recover_rare_pruning_rows(inside[c], post[c], scales[c], node, children,
                                                   np.flatnonzero(rare_sites))
            # Child slots hold pruning messages until the forward traversal
            # consumes them, then the same storage holds node posteriors.
            post[:, self.root].fill(0)
            root_unnormalized = inside[:, self.root] * self.pi
            root_sum = root_unnormalized.sum(axis=2)
            with np.errstate(divide='ignore'):
                log_weight = np.log(self.weights)[:, None] + np.log(root_sum) + scales[:, self.root]
            evidence = logsumexp(log_weight, axis=0)
            if not np.isfinite(evidence).all():
                bad = start + int(np.flatnonzero(~np.isfinite(evidence))[0])
                raise ValueError('Zero likelihood under endpoint model at site {}.'.format(bad + 1))
            class_weight = np.exp(log_weight - evidence)
            np.divide(root_unnormalized, root_sum[:, :, None],
                      out=post[:, self.root], where=root_sum[:, :, None] > 0)
            yield EndpointBlock(start, stop, self.root, -1,
                                np.einsum('cs,csk->sk', class_weight, post[:, self.root]), None, None,
                                category_node=(class_weight[:, :, None] * post[:, self.root]
                                               if category_nodes else None),
                                rate_posterior=class_weight if rate_posterior else None)
            for child in self.order[1:]:
                parent = int(self.parents[child])
                wanted = child in selected
                observed_transform = transform if wanted else unselected_transform
                edge = np.zeros((size, k, k)) if joint and wanted and transform is None else None
                pred = np.zeros((size, k, k)) if predictive and wanted and transform is None else None
                reduced: dict = {}
                reduced_pred: dict = {}
                for c in range(nc):
                    transition = self.transition(child, c)
                    denominator = post[c, child]
                    # Conditioning the child on its parent avoids retaining
                    # outside likelihoods or a joint for every branch.
                    right = inside[c, child]
                    parent_weights = post[c, parent]
                    positive = denominator > 0
                    rare_forward_sites: tuple | np.ndarray = ()
                    if denominator.min() < forward_floor:
                        # A posterior at most one cannot overflow the dot
                        # product when the message exceeds this floor.
                        rare = ((denominator < forward_floor) & positive & (parent_weights > 0)).any(axis=1)
                        if rare.any():
                            rare_forward_sites = np.flatnonzero(rare)
                            positive[rare] = False
                    left = np.divide(parent_weights, denominator, out=np.zeros_like(denominator), where=positive)
                    if len(rare_forward_sites) and observed_transform is not None and transform_rate_context:
                        left[rare_forward_sites] = _scaled_outside(
                            parent_weights[rare_forward_sites], denominator[rare_forward_sites],
                        )
                    post[c, child] = (left @ transition) * right
                    if edge is not None:
                        edge += ((left * class_weight[c, :, None])[:, :, None]
                                 * transition * right[:, None, :])
                    if pred is not None:
                        pred += (class_weight[c, :, None, None]
                                 * post[c, parent, :, :, None] * transition)
                    if observed_transform is not None:
                        reductions = [(reduced, left, right, observed_transform)]
                        if predictive:
                            reductions.append((reduced_pred, post[c, parent], np.ones_like(right),
                                               predictive_transform or observed_transform))
                        for target, lvalues, rvalues, reducer in reductions:
                            if transform_rate_context:
                                values = reducer(lvalues, rvalues, transition,
                                                 self.lengths[child] * self.rates[c], class_weight[c])
                            else:
                                values = reducer(lvalues * class_weight[c, :, None], rvalues, transition)
                            for key, value in values.items():
                                if key not in target:
                                    target[key] = value
                                else:
                                    target[key] += value
                    for site in rare_forward_sites:
                        with np.errstate(divide='ignore'):
                            log_right = np.log(right[site])
                        conditional = _conditional_transition(transition, log_right)
                        post[c, child, site] = parent_weights[site] @ conditional
                        weighted_parent = parent_weights[site] * class_weight[c, site]
                        if edge is not None:
                            edge[site] += weighted_parent[:, None] * conditional
                        if observed_transform is not None and not transform_rate_context:
                            # A reducer's factored joint is unchanged by
                            # supplying the conditional kernel and unit right.
                            for key, value in observed_transform(
                                weighted_parent[None, :], np.ones((1, k)), conditional,
                            ).items():
                                reduced[key][site] += value[0]
                yield EndpointBlock(start, stop, child, parent,
                                    np.einsum('cs,csk->sk', class_weight, post[:, child]), edge, pred,
                                    reduced, reduced_pred,
                                    class_weight[:, :, None] * post[:, child] if category_nodes else None)

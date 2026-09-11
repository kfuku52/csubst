"""Blocked CTMC endpoint inference, conditional on a fixed fitted model.

No ancestral histories or node-by-site-by-state-by-state array is stored.
Rate categories are integrated using their posterior probability at each site.
An edge joint is a distribution of endpoints, not a count of CTMC jumps.
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp


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
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node in self.order:
                raise ValueError('Endpoint tree contains a cycle.')
            self.order.append(node)
            stack.extend(reversed(self.children[node]))
        if len(self.order) != p.size:
            raise ValueError('Endpoint tree is disconnected.')
        self.leaves = {i for i, children in enumerate(self.children) if not children}
        self._transitions = OrderedDict()
        self._uniform_rate = float(np.max(-np.diag(self.q)))
        # Share nonnegative uniformization powers across all edge lengths and
        # categories. Cap this cache independently of the transition cache.
        capacity = min(256, max(1, (8 * 1024 * 1024) // (8 * k * k)))
        self._uniform_powers = np.empty((capacity, k, k))
        self._uniform_powers[0] = np.eye(k)
        self._uniform_count = 1
        self._uniform_kernel = (np.eye(k) + self.q / self._uniform_rate
                                if self._uniform_rate > 0 else np.eye(k))
        # Cap the transition cache independently of the number of sites/nodes.
        self._cache_items = max(1, (32 * 1024 * 1024) // (8 * k * k))

    def _uniform_transition(self, t):
        """exp(Qt) = exp(-mu*t) sum_n (mu*t)^n/n! (I + Q/mu)^n.

        All terms are nonnegative. The omitted Poisson tail bounds every
        matrix entry; compare it with the smallest positive partial entry to
        protect relative accuracy even for rare transitions. Use expm outside
        the bounded power workspace, without changing the model or threshold.
        """
        tau = self._uniform_rate * t
        if tau == 0:
            return np.eye(self.pi.size)
        capacity = self._uniform_powers.shape[0]
        if tau > 32 or capacity < self.pi.size:
            return None
        coefficients = np.empty(capacity)
        coefficients[0] = np.exp(-tau)
        target = 1e-18
        for n in range(1, capacity):
            coefficients[n] = coefficients[n - 1] * tau / n
            if n < self.pi.size - 1 or n + 1 <= tau:
                continue
            tail = coefficients[n] * tau / (n + 1 - tau)
            if tail > target:
                continue
            while self._uniform_count <= n:
                i = self._uniform_count
                self._uniform_powers[i] = self._uniform_powers[i - 1] @ self._uniform_kernel
                self._uniform_count += 1
            matrix = (coefficients[:n + 1] @ self._uniform_powers[:n + 1].reshape(n + 1, -1)).reshape(self.q.shape)
            target = float(matrix[matrix > 0].min() * np.finfo(float).eps * 0.25)
            if tail <= target:
                return matrix
        return None

    def transition(self, child, category):
        t = float(self.lengths[child] * self.rates[category])
        if t not in self._transitions:
            matrix = self._uniform_transition(t)
            if matrix is None:
                matrix = expm(self.q * t)
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
                    predictive_transform=None, category_nodes=False):
        """Yield one edge/site block; root records only contain a node marginal.

        Tips are observation likelihoods, NOT posterior probabilities. Missing
        observations have likelihood one for every state. Predictive records
        use sum_c P(c|D) P(parent=a|D,c) P_c(a,d), a fitted conditional
        endpoint prediction; they are not unconditional null probabilities.
        A separate predictive transform can omit observed-only summaries. Each
        transform's outputs are summed over rate categories before yielding.
        """
        if isinstance(block_size, bool) or int(block_size) != block_size or block_size < 1:
            raise ValueError('Endpoint block size must be a positive integer.')
        if set(tips) != self.leaves:
            raise ValueError('Endpoint inference requires likelihoods for every leaf.')
        arrays = {node: np.asarray(values) for node, values in tips.items()}
        first = next(iter(arrays.values()))
        if first.ndim != 2 or first.shape[1] != self.pi.size:
            raise ValueError('Endpoint tip likelihoods must have site-by-state shape.')
        for values in arrays.values():
            if values.shape != first.shape or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Endpoint tip likelihoods must be equal-sized, finite and nonnegative.')
        selected = set(self.order[1:]) if branch_ids is None else {int(i) for i in branch_ids}
        if not selected.issubset(set(self.order) - {self.root}):
            raise ValueError('Endpoint branch selection contains a root or unknown branch.')
        n, k, nc = self.parents.size, self.pi.size, self.rates.size
        capacity = min(int(block_size), first.shape[0])
        inside_buffer = np.empty((nc, n, capacity, k))
        post_buffer = np.empty_like(inside_buffer)
        scale_buffer = np.empty((nc, n, capacity))
        for start in range(0, first.shape[0], int(block_size)):
            stop = min(start + int(block_size), first.shape[0])
            size = stop - start
            inside = inside_buffer[:, :, :size]
            post = post_buffer[:, :, :size]
            scales = scale_buffer[:, :, :size]
            inside.fill(1)
            scales.fill(0)
            for c in range(nc):
                for node in reversed(self.order):
                    if node in arrays:
                        inside[c, node] = arrays[node][start:stop]
                    else:
                        for child in self.children[node]:
                            post[c, child] = inside[c, child] @ self.transition(child, c).T
                            inside[c, node] *= post[c, child]
                            scales[c, node] += scales[c, child]
                            norm = inside[c, node].max(axis=1)
                            inside[c, node] /= np.where(norm > 0, norm, 1)[:, None]
                            with np.errstate(divide='ignore'):
                                scales[c, node] += np.log(norm)
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
                                               if category_nodes else None))
            for child in self.order[1:]:
                parent = int(self.parents[child])
                wanted = child in selected
                edge = np.zeros((size, k, k)) if joint and wanted and transform is None else None
                pred = np.zeros((size, k, k)) if predictive and wanted and transform is None else None
                reduced: dict = {}
                reduced_pred: dict = {}
                for c in range(nc):
                    transition = self.transition(child, c)
                    denominator = post[c, child]
                    # Conditioning the child on its parent avoids retaining
                    # outside likelihoods or a joint for every branch.
                    left = np.divide(post[c, parent], denominator,
                                     out=np.zeros_like(denominator), where=denominator > 0)
                    right = inside[c, child]
                    post[c, child] = (left @ transition) * right
                    if edge is not None:
                        edge += ((left * class_weight[c, :, None])[:, :, None]
                                 * transition * right[:, None, :])
                    if pred is not None:
                        pred += (class_weight[c, :, None, None]
                                 * post[c, parent, :, :, None] * transition)
                    if transform is not None and wanted:
                        for target, lvalues, rvalues, reducer in (
                                (reduced, left, right, transform),
                                (reduced_pred, post[c, parent], np.ones_like(right),
                                 predictive_transform or transform)):
                            if target is reduced_pred and not predictive:
                                continue
                            values = reducer(lvalues * class_weight[c, :, None], rvalues, transition)
                            for key, value in values.items():
                                if key not in target:
                                    target[key] = value
                                else:
                                    target[key] += value
                yield EndpointBlock(start, stop, child, parent,
                                    np.einsum('cs,csk->sk', class_weight, post[:, child]), edge, pred,
                                    reduced, reduced_pred,
                                    class_weight[:, :, None] * post[:, child] if category_nodes else None)

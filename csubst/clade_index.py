"""Linear-size tree intervals with lazy, historically ordered clade listings."""

from collections import OrderedDict
from collections.abc import Mapping, Sequence

import numpy as np

from csubst import ete


class CladeIndex:
    def __init__(self, tree, branch_ids, branch_id_to_index):
        self.branch_ids = branch_ids
        self.branch_id_to_index = branch_id_to_index
        preorder = list(tree.traverse('preorder'))
        positions = {node: index for index, node in enumerate(preorder)}
        sizes = {}
        for node in reversed(preorder):
            sizes[node] = 1 + sum(sizes[child] for child in ete.get_children(node))
        nodes = {int(ete.get_prop(node, 'numerical_label')): node for node in preorder}
        self.starts = np.array([positions[nodes[int(bid)]] for bid in branch_ids], dtype=np.int64)
        self.stops = np.array([positions[nodes[int(bid)]] + sizes[nodes[int(bid)]]
                               for bid in branch_ids], dtype=np.int64)
        self.preorder_indices = np.array([
            branch_id_to_index.get(int(ete.get_prop(node, 'numerical_label')), -1)
            if not ete.is_root(node) else -1 for node in preorder], dtype=np.int64)
        self.breadth_rank = np.empty(len(preorder), dtype=np.int64)
        for rank, node in enumerate(tree.traverse()):
            self.breadth_rank[positions[node]] = rank
        self.leaf_positions = np.array([positions[node] for node in preorder if ete.is_leaf(node)], dtype=np.int64)
        self.leaf_names = tuple(node.name for node in preorder if ete.is_leaf(node))
        self._cache = OrderedDict()
        self._cache_bytes = 0

    def values(self, index, kind):
        key = (int(index), kind)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        start, stop = int(self.starts[index]), int(self.stops[index])
        if kind == 'leaves':
            a, b = np.searchsorted(self.leaf_positions, [start, stop])
            return self.leaf_names[a:b]
        order = np.argsort(self.breadth_rank[start:stop], kind='stable')
        indices = self.preorder_indices[start:stop][order]
        value = self.branch_ids[indices] if kind == 'ids' else indices
        if value.nbytes <= 8 * 1024**2:
            while self._cache and (self._cache_bytes + value.nbytes > 8 * 1024**2 or len(self._cache) >= 256):
                self._cache_bytes -= self._cache.popitem(last=False)[1].nbytes
            self._cache[key] = value
            self._cache_bytes += value.nbytes
        return value

    def __getstate__(self):
        return {**self.__dict__, '_cache': OrderedDict(), '_cache_bytes': 0}


class CladeDescendants(Sequence):
    def __init__(self, index, kind='indices'):
        self.index, self.kind = index, kind

    def __len__(self):
        return len(self.index.branch_ids)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(len(self)))]
        item = int(item)
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError(item)
        return self.index.values(item, self.kind)

    def nonoverlapping(self, selected):
        ordered = sorted((int(self.index.starts[i]), int(self.index.stops[i])) for i in selected)
        return all(right[0] >= left[1] for left, right in zip(ordered, ordered[1:]))


class CladesByBranch(Mapping):
    def __init__(self, sequence):
        self.sequence = sequence

    def __getitem__(self, bid):
        return self.sequence[self.sequence.index.branch_id_to_index[bid]]

    def __len__(self):
        return len(self.sequence)

    def __iter__(self):
        return iter(self.sequence.index.branch_id_to_index)

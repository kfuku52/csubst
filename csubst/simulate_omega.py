"""Exact any2spe endpoint counts under the simulated and background kernels."""

import csv
import os
import tempfile
from collections import Counter
from itertools import combinations

import numpy as np

from csubst import ete, output_safety
from csubst.tree import parent_distance
from csubst._vendor.pyvolve.evolver import ZERO
from csubst.omega_statistics import count_ratio_omega as _omega


class ExpectedOmega:
    """Accumulate site expectations without sampling or fitting tip data."""

    def __init__(self, tree, codon_table, codon_order, trait_name):
        self.nodes = sorted(
            (node for node in tree.traverse() if not ete.is_root(node)),
            key=lambda node: ete.get_prop(node, 'numerical_label'),
        )
        self.indices = {node.name: i for i, node in enumerate(self.nodes)}
        ancestors = [
            {ancestor.name for ancestor in ete.iter_ancestors(node)} for node in self.nodes
        ]
        self.parents = {
            child.name: node.name
            for node in tree.traverse()
            for child in ete.get_children(node)
        }
        self.pairs = np.array([
            (left, right) for left, right in combinations(range(len(self.nodes)), 2)
            if self.nodes[left].name not in ancestors[right]
            and self.nodes[right].name not in ancestors[left]
        ], dtype=np.int64).reshape(-1, 2)
        self.foreground = [
            int(ete.get_prop(node, 'is_fg_' + trait_name, False)) for node in self.nodes
        ]
        self.codons = {codon: i for i, codon in enumerate(codon_order)}
        amino_acids = {codon: aa for aa, codon in codon_table}
        self.aa = np.unique([amino_acids[codon] for codon in codon_order], return_inverse=True)[1]
        self.aa_membership = np.eye(self.aa.max() + 1)[self.aa]
        self.synonymous = self.aa[:, None] == self.aa
        np.fill_diagonal(self.synonymous, False)
        self.identity = np.eye(len(codon_order))
        self.counts = np.zeros((len(self.nodes), len(self.nodes), 4))

    def _event_features(self, transition):
        n = transition @ self.aa_membership
        n[np.arange(self.aa.size), self.aa] = 0
        return np.concatenate((n, transition * self.synonymous), axis=1)

    def add_evolution(self, evolver):
        if self.pairs.size == 0:
            return
        # Simulate supplies a single rate class per Pyvolve partition, with
        # the site rate already applied to the custom model's matrix.
        groups = Counter()
        for part in evolver.partitions:
            groups[part._root_model, tuple(part.models)] += part.size[0]
        for (root_model, models), num_sites in groups.items():
            self._add_model_group(evolver, root_model, models, num_sites)

    def _add_model_group(self, evolver, root_model, models, num_sites):
        models_by_name = {model.name: model for model in models}
        cache = {}
        num_aa = self.aa_membership.shape[1]
        width = num_aa + self.aa.size
        heterogeneous = len(models) > 1
        boundaries = (0, num_aa, width, width + num_aa, 2 * width) if heterogeneous else (0, num_aa, width)

        def edge_matrices(child):
            model = models_by_name[child.model_flag] if heterogeneous else root_model
            length = child.branch_length
            key = (model, length)
            result = cache.get(key)
            if result is None:
                if length <= ZERO:
                    simulated = null = self.identity
                else:
                    time = length * evolver.scale_tree
                    simulated = evolver._exponentiate_matrix(model.matrix, time)
                    null = (
                        simulated if model is root_model
                        else evolver._exponentiate_matrix(root_model.matrix, time)
                    )
                features = self._event_features(simulated)
                if heterogeneous:
                    null_features = features if null is simulated else self._event_features(null)
                    features = np.concatenate((features, null_features), axis=1)
                result = simulated, features
                cache[key] = result
            return result

        def visit(node, state_prob):
            subtrees = []
            for child in node.children:
                transition, direct_features = edge_matrices(child)
                child_ids, child_features = visit(child, state_prob @ transition)
                edge_ids = np.append(child_ids, self.indices[child.name])
                features = np.concatenate((transition @ child_features, direct_features[None, :, :]))
                subtrees.append((edge_ids, features))

            # Each pair occurs at exactly one split. Given the state there,
            # the two subtrees are independent. Integrate over that state
            # only after multiplying the matching endpoint probabilities.
            # Null features use the real transitions on the paths up to
            # each edge's parent, with the background kernel on the edge.
            for (left_ids, left), (right_ids, right) in combinations(subtrees, 2):
                pair_rows = (np.minimum.outer(left_ids, right_ids), np.maximum.outer(left_ids, right_ids))
                for k, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                    weighted = left[:, :, start:end] * state_prob[None, :, None]
                    right_features = right[:, :, start:end].reshape(len(right_ids), -1)
                    counts = weighted.reshape(len(left_ids), -1) @ right_features.T
                    weighted_counts = num_sites * counts
                    self.counts[pair_rows[0], pair_rows[1], k] += weighted_counts
                    if not heterogeneous:
                        self.counts[pair_rows[0], pair_rows[1], k + 2] += weighted_counts

            if not subtrees:
                return np.empty(0, dtype=np.int64), np.empty((0, self.aa.size, boundaries[-1]))
            ids, features = zip(*subtrees)
            return np.concatenate(ids), np.concatenate(features)

        visit(evolver.full_tree, root_model.params['state_freqs'])

    def realized_counts(self, sequences):
        codon_states = {
            name: np.array([self.codons[seq[i:i + 3]] for i in range(0, len(seq), 3)])
            for name, seq in sequences.items()
        }
        derived_n, derived_s = [], []
        for node in self.nodes:
            parent = codon_states[self.parents[node.name]]
            child = codon_states[node.name]
            n_change = self.aa[parent] != self.aa[child]
            derived_n.append(np.where(n_change, self.aa[child], -1))
            derived_s.append(np.where((parent != child) & ~n_change, child, -1))
        counts = np.empty((len(self.pairs), 2), dtype=np.int64)
        for k, derived in enumerate((np.asarray(derived_n), np.asarray(derived_s))):
            for row, (left, right) in enumerate(self.pairs):
                counts[row, k] = np.count_nonzero((derived[left] >= 0) & (derived[left] == derived[right]))
        return counts

    def write(self, path, sequences=None):
        left, right = self.pairs.T
        sim_n, sim_s, null_n, null_s = self.counts[left, right].T
        expected = _omega(sim_n, sim_s, null_n, null_s)
        if sequences is None:
            true_n = true_s = realized = np.full(len(left), np.nan)
        else:
            true_n, true_s = self.realized_counts(sequences).T
            realized = _omega(true_n, true_s, null_n, null_s)
        columns = (
            'branch_id_1', 'branch_id_2', 'branch_name_1', 'branch_name_2',
            'foreground_1', 'foreground_2',
            'muNany2spe_sim', 'muSany2spe_sim', 'muNany2spe_null', 'muSany2spe_null',
            'omegaCany2spe_expected', 'OCNany2spe_true', 'OCSany2spe_true', 'omegaCany2spe_true',
            'dist_parent_bl', 'dist_bl',
        )
        output_safety.validate_destination(path)
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(path))) as directory:
            temporary = os.path.join(directory, 'expected_omega.tsv')
            with open(temporary, 'w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
                writer.writerow(columns)
                for i, (a, b) in enumerate(self.pairs):
                    first, second = self.nodes[a], self.nodes[b]
                    distance = parent_distance(first, second)
                    values = (sim_n[i], sim_s[i], null_n[i], null_s[i], expected[i],
                              true_n[i], true_s[i], realized[i], distance, distance)
                    writer.writerow((
                        ete.get_prop(first, 'numerical_label'), ete.get_prop(second, 'numerical_label'),
                        first.name, second.name, self.foreground[a], self.foreground[b],
                        *('NA' if np.isnan(value) else value for value in values),
                    ))
            os.replace(temporary, path)

"""Small equivalence checks for audit helpers; not production regression tests."""

from collections import Counter
import json

import numpy as np
import pandas as pd

import audit
from csubst import endpoint_io, ete, parser_iqtree, scan_statistics, substitution_scan as scan, tree


def main():
    checks = Counter()
    rng = np.random.default_rng(91)
    matches = scan.normalize_scan_matches('all')
    orders = np.array(['A', 'B', 'C', 'D'])
    for size in (0, 1, 20, 100):
        events = pd.DataFrame({'site': rng.integers(0, 5, size),
                               'from_state_id': rng.integers(0, 4, size),
                               'to_state_id': rng.integers(0, 4, size), 'event_pp': rng.random(size)})
        for selected in (matches, ['any2spe'], ['spe2spe'], ['dif2dif']):
            pd.testing.assert_frame_equal(scan.build_candidates(events, selected, orders),
                                          audit.shared_candidate_groups(events, selected, orders), check_exact=True)
            checks['candidate_tables_exact'] += 1
    for text in ('((A:1,B:1):1,C:1);', '((A:1,B:1,C:1):1,D:1,E:1);',
                 '(((A:1):1,B:1):1,C:1);'):
        tr = tree.add_numerical_node_labels(ete.PhyloNode(text, format=1))
        n = max(int(ete.get_prop(node, 'numerical_label')) for node in tr.traverse()) + 1
        for all_missing in (False, True):
            states = np.zeros((n, 16, 4))
            for leaf in ete.iter_leaves(tr):
                values = rng.dirichlet(np.ones(4), size=16)
                values[rng.random(16) < .5] = 0
                if all_missing:
                    values.fill(0)
                states[int(ete.get_prop(leaf, 'numerical_label'))] = values
            a = parser_iqtree.get_internal_site_nonzero_masks(tr, state_tensor=states)
            b = audit.subtree_masks(tr, states)
            assert a.keys() == b.keys()
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            checks['nonmissing_masks_exact'] += 1
    rows = [[], [{'score_rate_enrichment': 1.}],
            [{'score_rate_enrichment': np.nan, 'target_event_count': .5, 'other_event_count': 0.,
              'target_exposure': 0., 'other_exposure': 1.}],
            [{'score_rate_enrichment': np.nan, 'target_event_count': .5, 'other_event_count': 0.,
              'target_exposure_branch_length': 1., 'other_exposure_branch_length': 0.}],
            [{'score_rate_enrichment': -1.}], [{'score_rate_enrichment': np.inf}],
            [{'score_rate_enrichment': np.nan}],
            [{'score_rate_enrichment': np.nan, 'target_event_count': .5, 'other_event_count': 0.,
              'target_exposure': 1., 'other_exposure': 1.}]]
    for values in rows:
        frame = pd.DataFrame(values)
        results = []
        for function in (scan_statistics.maximum_score, audit.maximum_score):
            try:
                results.append(('value', function(frame)))
            except ValueError as error:
                results.append(('error', str(error)))
        assert results[0] == results[1]
        checks['maximum_score_values_and_failures_exact'] += 1
    g = dict(amino_acid_orders=['X', 'Y'], synonymous_indices={'X': [0, 1], 'Y': [2, 3]}, max_synonymous_size=2)
    mapping = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    for kinds in (['S'], ['N'], ['S', 'N', 'AA']):
        mappings = {'S': None, 'N': mapping, 'AA': mapping}
        original = endpoint_io._event_transform(g, kinds, mappings)
        prototype = audit.predictive_events(g, kinds, mappings, original, Counter())
        for size in (1, 8, 64):
            left = rng.dirichlet(np.ones(4), size=size)
            transition = rng.dirichlet(np.ones(4), size=4)
            right = np.ones_like(left)
            a, b = original(left, right, transition), prototype(left, right, transition)
            for key in a:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-13)
            checks['predictive_transform_tolerance'] += 1
    print(json.dumps(dict(checks), indent=2))


if __name__ == '__main__':
    main()

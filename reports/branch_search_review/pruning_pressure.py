"""Measure K=3 candidate growth on independent branches with known site events."""
from itertools import product

import numpy as np
import pandas as pd

from csubst import combination, main_analyze


def pair_table(draws):
    # Each site has one shared derived outcome per channel. Branches draw
    # independently and both channels have uniform opportunity across sites.
    n_branches, length = draws.shape
    left, right = np.triu_indices(n_branches, 1)
    observed, expected = {}, {}
    for channel, changes in [('N', draws < .04), ('S', (draws >= .04) & (draws < .08))]:
        events = changes.astype(np.int32)
        totals = events.sum(axis=1)
        observed[channel] = (events @ events.T)[left, right]
        expected[channel] = totals[left] * totals[right] / length
    with np.errstate(invalid='ignore', divide='ignore'):
        omega = (observed['N'] / expected['N']) / (observed['S'] / expected['S'])
    return pd.DataFrame({'branch_id_1': left, 'branch_id_2': right,
                         'OCNany2spe': observed['N'], 'omegaCany2spe': omega})


def main():
    draws = np.random.default_rng(20261119).random((256, 2048))
    rows = []
    for branches, length in product((64, 256), (512, 2048)):
        pairs = pair_table(draws[:branches, :length])
        for mode in ('all_cutoffs', 'ocn_bound'):
            stats = pd.DataFrame(index=[0])
            settings = dict(foreground=None, cutoff_stat='OCNany2spe,2|omegaCany2spe,5',
                            max_combination=10000, high_order_pruning=mode, df_cb_stats=stats)
            parents = main_analyze._select_high_order_candidates(pairs, settings)
            triples = combination._generate_union_candidates_by_shared_subset(
                parents[['branch_id_1', 'branch_id_2']].to_numpy(), arity=3)
            rows.append((branches, length, mode, int(stats.at[0, 'expansion_eligible']),
                         len(parents), len(triples)))
    print(pd.DataFrame(rows, columns=['branches', 'sites', 'pruning', 'eligible_pairs',
                                      'retained_pairs', 'generated_triples']).to_string(index=False))


if __name__ == '__main__':
    main()

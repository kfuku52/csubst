"""Audit analysis-site selection without refitting the full-site reference model.

The count report is descriptive. It is not a selection-adjusted null distribution.
"""

import json

import numpy as np
import pandas as pd

from csubst import ete, output_stat, parser_misc, runtime, substitution, tsv


CRITERIA = ('tip_invariant', 'zero_sub_mass')


def validate_report_options(g):
    if not g.get('site_filter_report', False):
        return
    if g.get('subcommand', 'search') not in ('search', 'analyze'):
        raise ValueError('--site_filter_report requires search/analyze.')
    if g.get('drop_invariant_tip_sites', False):
        raise ValueError('--site_filter_report requires --drop_invariant_tip_sites no (full-site reference).')
    if g.get('expectation_method', 'codon_model') != 'codon_model':
        raise ValueError('--site_filter_report currently requires --expectation_method codon_model; '
                         'urn expectations need a separate fixed-weight attribution.')
    if not g.get('cb', True):
        raise ValueError('--site_filter_report requires --cb yes.')


def prepare(g):
    """Capture masks before any site-axis slicing or state release."""
    report = bool(g.get('site_filter_report', False))
    selected = bool(g.get('drop_invariant_tip_sites', False))
    if not (report or selected):
        return
    validate_report_options(g)
    state = g['state_cdn']
    indices = parser_misc.get_site_index_alignment(g, expected_num_site=state.shape[1])
    mode = g.get('drop_invariant_tip_sites_mode', 'tip_invariant') if selected else 'no'
    criteria = CRITERIA if report else (mode,)
    masks = {key: parser_misc.get_site_drop_mask(g, key, indices) for key in criteria}
    structural = ('3di_tip_invariant_mask' in g or '_precomputed_tip_invariant_site_mask' in g)
    basis = 'direct_3di' if structural else 'codon'
    tips = [int(ete.get_prop(n, 'numerical_label')) for n in ete.iter_leaves(g['tree'])]
    tip_counts = (state[tips].sum(axis=2) > 0).sum(axis=0)
    rows = pd.DataFrame({'alignment_site': indices + 1, 'num_tips_with_codon_state': tip_counts})
    for key, mask in masks.items():
        rows[key + '_drop'] = mask
    rows['tip_invariant_basis'] = basis
    tsv.write_dataframe(rows, runtime.output_path(g, 'site_filter_sites.tsv'))
    metadata = {
        'schema_version': 1,
        'analysis_site_selection': mode,
        'analysis_num_sites_before_filter': int(len(indices)),
        'analysis_num_sites_retained': int(len(indices) - (masks[mode].sum() if selected else 0)),
        'tip_invariant_basis': basis,
        'zero_sub_mass_tolerance': float(g['float_tol']),
        'zero_sub_mass_stage': 'before_min_sub_pp',
        'branch_pairs_child_parent': parser_misc._get_drop_site_branch_pairs(g),
        'report_reference': 'all_sites_fixed_model' if report else None,
        'expectation_method': g.get('expectation_method', 'codon_model'),
        'selection_repeated_in_null': False,
        'pvalue_scope': ('fitted_count_null_fixed_analysis_sites'
                         if g.get('calc_omega_pvalue', False) else 'not_requested'),
        'site_coordinate_base': 1,
    }
    with open(runtime.output_path(g, 'site_filter.json'), 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
        handle.write('\n')
    if report:
        g['_site_filter_masks'] = masks
        g['_site_filter_basis'] = basis
    if selected:
        print('Site selection changes the analysis target. Excluded sites need not have zero '
              'expected mass; count-null P-values do not repeat site selection.', flush=True)


def start_count_report(g, cb, ON_tensor, OS_tensor, base_stats):
    """Allocate one arity's report and collect observed projection products."""
    if not g.get('site_filter_report', False):
        return None
    if '_site_filter_masks' not in g:
        raise ValueError('Site-filter report masks must be prepared before state release.')
    id_columns = [col for col in cb.columns if col.startswith('branch_id_')]
    ids = cb[id_columns].to_numpy(dtype=np.int64)
    masks = g['_site_filter_masks']
    parts = {}
    for criterion, drop in masks.items():
        for partition, mask in [('all', np.ones(len(drop), dtype=bool)),
                                ('retained', ~drop), ('excluded', drop)]:
            frame = cb[id_columns].reset_index(drop=True).copy()
            frame['criterion'] = criterion
            frame['partition'] = partition
            frame['num_sites'] = int(mask.sum())
            frame['reference'] = 'all_sites_fixed_model'
            frame['tip_invariant_basis'] = g['_site_filter_basis']
            parts[criterion, partition] = (mask, frame)
    report = {'ids': ids, 'parts': parts, 'base_stats': base_stats}
    for attr, tensor in [('OCN', ON_tensor), ('OCS', OS_tensor)]:
        # Retain at most one additional category projection at a time.
        for stat in base_stats:
            projection = substitution._get_sparse_cb_projection(tensor, stat)
            collect_projection(report, projection, attr + stat)
    return report


def collect_projection(report, projection, column):
    """Project selected columns without changing rates, lengths or branch totals."""
    if report is None:
        return
    all_values = substitution._calc_sparse_projection_products(projection, report['ids'])
    for (criterion, partition), (mask, frame) in report['parts'].items():
        if partition == 'all':
            frame[column] = all_values
            continue
        num_site = len(mask)
        if num_site == 0 or projection.shape[1] % num_site:
            raise ValueError('Projection feature axis does not match site-filter mask.')
        # Projections pack feature * num_site + site. Do not modify a shared
        # cached projection: later arities and the primary analysis reuse it.
        selected = projection.copy()
        selected.data[~mask[selected.indices % num_site]] = 0
        selected.eliminate_zeros()
        frame[column] = substitution._calc_sparse_projection_products(selected, report['ids'])


def finish_count_report(report, g):
    if report is None:
        return
    # Import here to avoid a module cycle with omega's report hooks.
    from csubst import omega

    frames = []
    requested = output_stat.get_required_base_stats(g.get('output_stats', report['base_stats']))
    for mask, frame in report['parts'].values():
        for attr in ('OCN', 'OCS', 'ECN', 'ECS'):
            missing = [attr + stat for stat in requested if attr + stat not in frame]
            if missing:
                raise ValueError('Incomplete site-filter count report: ' + ', '.join(missing))
        for prefix in ('OC', 'EC'):
            frame = substitution.add_dif_stats(frame, g['float_tol'], prefix=prefix,
                                              output_stats=g.get('output_stats'))
        frame = omega.subroot_E2nan(frame, g['tree'])
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    path = runtime.output_path(g, 'site_filter_counts_{}.tsv'.format(report['ids'].shape[1]))
    tsv.write_dataframe(table, path, float_format='%.17g')
    print('Writing fixed-model site-filter count report: ' + path, flush=True)

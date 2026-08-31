"""Tree/site figure rendering, independent of command orchestration.

The caller provides prepared tables and analysis state; this module owns
site selection, tree layout, heatmaps, labels, and figure output.
"""

import os
import re
from collections.abc import MutableMapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from csubst import ete, parser_misc, sequence, tree, tsv
from csubst.plotting import TREE_LINE_CAPSTYLE, font_size, matplotlib, plt


def _normalize_branch_ids(branch_ids):
    if branch_ids is None:
        return np.array([], dtype=np.int64)
    values = np.asarray(branch_ids, dtype=object)
    if values.ndim == 0:
        scalar = values.item()
        if isinstance(scalar, (list, tuple, set, np.ndarray)):
            values = np.asarray(list(scalar), dtype=object)
        else:
            values = np.asarray([scalar], dtype=object)
    flat_values = np.atleast_1d(values).reshape(-1)
    if flat_values.size == 0:
        return np.array([], dtype=np.int64)
    normalized = []
    for value in flat_values.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('branch_ids should be integer-like.')
        if isinstance(value, (int, np.integer)):
            normalized.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            if (not np.isfinite(value)) or (not float(value).is_integer()):
                raise ValueError('branch_ids should be integer-like.')
            normalized.append(int(value))
            continue
        value_txt = str(value).strip()
        if (value_txt == '') or (not bool(re.fullmatch(r'[+-]?[0-9]+(?:\.0+)?', value_txt))):
            raise ValueError('branch_ids should be integer-like.')
        normalized.append(int(float(value_txt)))
    return np.array(normalized, dtype=np.int64)


def _format_branch_id_label(branch_id):
    return 'b{}'.format(int(branch_id))


def _site_output_prefix(g):
    output_prefix = str(g.get('output_prefix', 'csubst_sites')).strip()
    if output_prefix == '':
        return 'csubst_sites'
    return output_prefix


def _oldness_frac_to_rgb(frac):
    frac = min(max(float(frac), 0.0), 1.0)
    if frac <= 0.5:
        t = frac / 0.5
        return (t, t, 1.0 - t)  # blue -> yellow
    t = (frac - 0.5) / 0.5
    return (1.0, 1.0 - t, 0.0)  # yellow -> red


def _get_lineage_midpoint_distances(branch_ids, g):
    if len(branch_ids)==0:
        return [],False
    if ('tree' not in g) or (g['tree'] is None):
        if len(branch_ids)==1:
            return [0.5],False
        return [i/(len(branch_ids)-1) for i in range(len(branch_ids))],False
    node_by_id = dict()
    for node in g['tree'].traverse():
        node_by_id[int(ete.get_prop(node, "numerical_label"))] = node
    lengths = []
    for branch_id in branch_ids:
        node = node_by_id.get(int(branch_id), None)
        bl = float(getattr(node, 'dist', 0.0)) if (node is not None) else 0.0
        lengths.append(max(bl, 0.0))
    total_len = float(sum(lengths))
    if total_len <= 0:
        if len(branch_ids)==1:
            return [0.5],False
        return [i/(len(branch_ids)-1) for i in range(len(branch_ids))],False
    mids = []
    cumul = 0.0
    for bl in lengths:
        mids.append(cumul + bl*0.5)
        cumul += bl
    return mids,True


def _get_lineage_oldness_fracs(branch_ids, g):
    if len(branch_ids)==0:
        return []
    if len(branch_ids)==1:
        return [1.0]
    mid_dists,_ = _get_lineage_midpoint_distances(branch_ids=branch_ids, g=g)
    min_mid = min(mid_dists)
    max_mid = max(mid_dists)
    span = max_mid - min_mid
    if span <= 0:
        return [i/(len(branch_ids)-1) for i in range(len(branch_ids))]
    out = []
    for mid_dist in mid_dists:
        out.append((mid_dist - min_mid) / span)
    return out


def _get_lineage_rgb_by_branch(branch_ids, g):
    fracs = _get_lineage_oldness_fracs(branch_ids=branch_ids, g=g)
    out = dict()
    for branch_id,frac in zip(branch_ids, fracs):
        out[int(branch_id)] = _oldness_frac_to_rgb(frac)
    return out


def get_tree_site_branch_color_by_id(branch_ids, g, default_color='firebrick'):
    branch_ids = _normalize_branch_ids(branch_ids).tolist()
    color_mode = str(g.get('tree_site_branch_color_mode', 'lineage')).strip().lower()
    single_color_modes = {'single', 'uniform', 'one', 'same'}
    if color_mode in single_color_modes:
        color = str(g.get('tree_site_branch_color', default_color)).strip()
        if color == '':
            color = default_color
        return {int(bid): color for bid in branch_ids}
    return _get_lineage_rgb_by_branch(branch_ids=branch_ids, g=g)


def get_highest_identity_chain_name(g):
    if ('highest_identity_chain_name' in g) and g['highest_identity_chain_name']:
        return g
    if 'aa_identity_means' not in g.keys():
        from csubst import parser_pymol
        g = parser_pymol.calc_aa_identity(g)
    aa_identity_means = g.get('aa_identity_means', {})
    if len(aa_identity_means) == 0:
        g['highest_identity_chain_name'] = None
        return g
    mean_keys = np.array(list(aa_identity_means.keys()))
    mean_values = np.array(list(aa_identity_means.values()), dtype=float)
    is_finite = np.isfinite(mean_values)
    if is_finite.any():
        finite_keys = mean_keys[is_finite]
        finite_values = mean_values[is_finite]
        g['highest_identity_chain_name'] = finite_keys[np.argmax(finite_values)]
    else:
        g['highest_identity_chain_name'] = mean_keys[0]
    return g


def get_min_single_prob(g):
    return float(g.get('min_single_prob', 0.8))


def get_min_combinat_prob(g):
    return float(g.get('min_combinat_prob', 0.5))


def get_tree_site_min_prob(g):
    mode = str(g.get('mode', '')).lower()
    if mode == 'lineage':
        return get_min_single_prob(g)
    if g.get('single_branch_mode', False):
        return get_min_single_prob(g)
    return get_min_combinat_prob(g)


def classify_tree_site_categories(df, g):
    if 'codon_site_alignment' not in df.columns:
        raise ValueError('codon_site_alignment column is required.')
    min_prob = get_tree_site_min_prob(g)
    num_site = df.shape[0]
    if g.get('single_branch_mode', False):
        convergent_score = df.loc[:, 'N_sub'].values if 'N_sub' in df.columns else np.zeros(num_site)
        divergent_score = np.zeros(num_site, dtype=float)
    else:
        convergent_score = df.loc[:, 'OCNany2spe'].values if 'OCNany2spe' in df.columns else np.zeros(num_site)
        divergent_score = df.loc[:, 'OCNany2dif'].values if 'OCNany2dif' in df.columns else np.zeros(num_site)
    convergent_score = np.nan_to_num(convergent_score.astype(float), nan=0.0)
    divergent_score = np.nan_to_num(divergent_score.astype(float), nan=0.0)

    category = np.full(shape=(num_site,), fill_value='blank', dtype=object)
    is_convergent = (convergent_score >= min_prob)
    is_divergent = (divergent_score >= min_prob)
    category[is_convergent] = 'convergent'
    category[is_divergent] = 'divergent'

    is_both = is_convergent & is_divergent
    category[is_both & (convergent_score >= divergent_score)] = 'convergent'
    category[is_both & (convergent_score < divergent_score)] = 'divergent'

    out = pd.DataFrame({
        'codon_site_alignment': df.loc[:, 'codon_site_alignment'].values,
        'convergent_score': convergent_score,
        'divergent_score': divergent_score,
        'tree_site_category': category,
    })
    out = out.sort_values(by='codon_site_alignment').reset_index(drop=True)
    return out,min_prob


def get_tree_plot_coordinates(tree):
    root = ete.get_tree_root(tree)
    xcoord = dict()
    ycoord = dict()
    leaf_order = list()

    def assign_x(node, parent_x):
        nl = int(ete.get_prop(node, "numerical_label"))
        xcoord[nl] = float(parent_x)
        for child in ete.get_children(node):
            child_dist = child.dist if child.dist is not None else 0
            assign_x(node=child, parent_x=parent_x + child_dist)

    def assign_y(node, current_y):
        nl = int(ete.get_prop(node, "numerical_label"))
        if ete.is_leaf(node):
            ycoord[nl] = float(current_y)
            leaf_order.append(nl)
            return current_y + 1
        child_ys = list()
        for child in ete.get_children(node):
            current_y = assign_y(node=child, current_y=current_y)
            child_ys.append(ycoord[int(ete.get_prop(child, "numerical_label"))])
        if len(child_ys) == 0:
            ycoord[nl] = float(current_y)
            return current_y + 1
        ycoord[nl] = float(sum(child_ys) / len(child_ys))
        return current_y

    assign_x(node=root, parent_x=0.0)
    _ = assign_y(node=root, current_y=0)
    return xcoord,ycoord,leaf_order


def get_tree_site_plot_max_sites(g):
    max_sites = int(g.get('tree_site_plot_max_sites', 30))
    if max_sites < 1:
        max_sites = 1
    return max_sites


def get_tree_site_tip_label_spacing(g):
    value = g.get('tree_site_tip_label_spacing', 1.35)
    return tree._resolve_tree_tip_label_spacing_factor(
        value=value,
        param_name='--tree_site_tip_label_spacing',
    )


def get_tree_site_fig_max_height(g):
    value = g.get('tree_site_fig_max_height', 24.0)
    return tree._resolve_tree_figure_max_height(
        value=value,
        param_name='--tree_site_fig_max_height',
    )


def get_lineage_display_sites(df, g, min_prob, return_total=False):
    branch_ids = _normalize_branch_ids(g.get('branch_ids', [])).tolist()
    col_pairs = []
    for bid in branch_ids:
        col = 'N_sub_{}'.format(int(bid))
        if col in df.columns:
            col_pairs.append((int(bid), col))
    if len(col_pairs) == 0:
        if return_total:
            return [],0
        return []
    _, cols = zip(*col_pairs)
    branch_values = df.loc[:, list(cols)].to_numpy(dtype=float, copy=True)
    branch_values = np.nan_to_num(branch_values, nan=0.0)
    # Lineage view should show sites meeting or exceeding the configured minimum PP.
    site_ids = df.loc[:, 'codon_site_alignment'].astype(int).to_numpy(copy=True)
    max_branch_prob = branch_values.max(axis=1)
    is_selected = (max_branch_prob >= float(min_prob))
    if not is_selected.any():
        if return_total:
            return [],0
        return []
    selected_sites = site_ids[is_selected]
    selected_scores = max_branch_prob[is_selected]
    selected_total = int(selected_sites.shape[0])
    max_sites = get_tree_site_plot_max_sites(g)
    if selected_sites.shape[0] > max_sites:
        # Pick strongest foreground-substitution sites first.
        order = np.lexsort((selected_sites, -selected_scores))
        selected_sites = selected_sites[order[:max_sites]]
    out_sites = sorted([int(site) for site in selected_sites.tolist()])
    if return_total:
        return out_sites,selected_total
    return out_sites


def get_set_display_sites(df, g, min_prob, return_total=False):
    if 'codon_site_alignment' not in df.columns:
        if return_total:
            return [],0
        return []
    if 'N_set_expr_prob' in df.columns:
        set_scores = df.loc[:, 'N_set_expr_prob'].to_numpy(dtype=float, copy=True)
        set_scores = np.nan_to_num(set_scores, nan=0.0)
    else:
        set_scores = np.zeros(shape=(df.shape[0],), dtype=float)
    if 'N_set_expr' in df.columns:
        set_selected = df.loc[:, 'N_set_expr'].astype(bool).to_numpy(copy=True)
    else:
        # Fallback for backward-compatibility with legacy tables.
        set_selected = (set_scores > float(min_prob))
    # Guard against inconsistent data where set-selected sites have zero score.
    set_selected = set_selected & (set_scores > 0.0)
    if not set_selected.any():
        if return_total:
            return [],0
        return []
    site_ids = df.loc[:, 'codon_site_alignment'].astype(int).to_numpy(copy=True)
    selected_sites = site_ids[set_selected]
    selected_scores = set_scores[set_selected]
    selected_total = int(selected_sites.shape[0])
    max_sites = get_tree_site_plot_max_sites(g)
    if selected_sites.shape[0] > max_sites:
        order = np.lexsort((selected_sites, -selected_scores))
        selected_sites = selected_sites[order[:max_sites]]
    out_sites = sorted([int(site) for site in selected_sites.tolist()])
    if return_total:
        return out_sites,selected_total
    return out_sites


def get_tree_site_display_sites(tree_site_df, g, df=None):
    mode = str(g.get('mode', '')).lower()
    min_prob = get_tree_site_min_prob(g)
    if (mode == 'lineage') and (df is not None):
        lineage_sites = get_lineage_display_sites(df=df, g=g, min_prob=min_prob)
        return [{'site': int(site), 'category': 'lineage'} for site in lineage_sites]
    if (mode == 'set') and (df is not None):
        set_sites = get_set_display_sites(df=df, g=g, min_prob=min_prob)
        return [{'site': int(site), 'category': 'set'} for site in set_sites]
    max_sites = get_tree_site_plot_max_sites(g)
    convergent_df = tree_site_df.loc[tree_site_df.loc[:, 'tree_site_category']=='convergent',:]
    convergent_df = convergent_df.sort_values(by=['convergent_score', 'codon_site_alignment'], ascending=[False, True])
    divergent_df = tree_site_df.loc[tree_site_df.loc[:, 'tree_site_category']=='divergent',:]
    divergent_df = divergent_df.sort_values(by=['divergent_score', 'codon_site_alignment'], ascending=[False, True])
    num_convergent = int(convergent_df.shape[0])
    num_divergent = int(divergent_df.shape[0])

    if (num_convergent + num_divergent) == 0:
        fallback = tree_site_df.copy()
        fallback.loc[:, 'max_score'] = fallback.loc[:, ['convergent_score', 'divergent_score']].max(axis=1)
        fallback = fallback.sort_values(by=['max_score', 'codon_site_alignment'], ascending=[False, True])
        fallback = fallback.iloc[:max_sites, :]
        display_sites = fallback.loc[:, 'codon_site_alignment'].astype(int).tolist()
        display_meta = [{'site': int(site), 'category': 'blank'} for site in display_sites]
        return display_meta

    if (num_convergent > 0) and (num_divergent > 0):
        if max_sites == 1:
            top_conv = float(convergent_df.iloc[0, :].loc['convergent_score'])
            top_div = float(divergent_df.iloc[0, :].loc['divergent_score'])
            if top_conv >= top_div:
                max_conv,max_div = 1,0
            else:
                max_conv,max_div = 0,1
        else:
            max_conv = max_sites // 2
            max_div = max_sites - max_conv
            if max_conv == 0:
                max_conv,max_div = 1,max_sites-1
            if max_div == 0:
                max_conv,max_div = max_sites-1,1
    else:
        if num_convergent > 0:
            max_conv,max_div = max_sites,0
        else:
            max_conv,max_div = 0,max_sites

    max_conv = min(max_conv, num_convergent)
    max_div = min(max_div, num_divergent)
    remaining = max_sites - (max_conv + max_div)
    if remaining > 0:
        add_conv = min(remaining, num_convergent - max_conv)
        max_conv += add_conv
        remaining -= add_conv
    if remaining > 0:
        add_div = min(remaining, num_divergent - max_div)
        max_div += add_div

    convergent_sites = convergent_df.iloc[:max_conv, :].loc[:, 'codon_site_alignment'].astype(int).tolist()
    divergent_sites = divergent_df.iloc[:max_div, :].loc[:, 'codon_site_alignment'].astype(int).tolist()
    convergent_sites = sorted(convergent_sites)
    divergent_sites = sorted(divergent_sites)
    display_meta = [{'site': int(site), 'category': 'convergent'} for site in convergent_sites]
    if (len(convergent_sites) > 0) and (len(divergent_sites) > 0):
        display_meta.append({'site': None, 'category': 'separator'})
    display_meta += [{'site': int(site), 'category': 'divergent'} for site in divergent_sites]
    return display_meta


def get_tree_site_overflow_count(tree_site_df, display_meta, g, df=None):
    plotted_site_count = int(len([item for item in display_meta if item.get('site', None) is not None]))
    mode = str(g.get('mode', '')).lower()
    min_prob = get_tree_site_min_prob(g)
    if (mode == 'lineage') and (df is not None):
        _, total_candidate = get_lineage_display_sites(df=df, g=g, min_prob=min_prob, return_total=True)
        return max(0, int(total_candidate) - plotted_site_count)
    if (mode == 'set') and (df is not None):
        _, total_candidate = get_set_display_sites(df=df, g=g, min_prob=min_prob, return_total=True)
        return max(0, int(total_candidate) - plotted_site_count)
    num_convergent = int((tree_site_df.loc[:, 'tree_site_category'] == 'convergent').sum())
    num_divergent = int((tree_site_df.loc[:, 'tree_site_category'] == 'divergent').sum())
    total_candidate = num_convergent + num_divergent
    if total_candidate == 0:
        total_candidate = int(tree_site_df.shape[0])
    return max(0, int(total_candidate) - plotted_site_count)


def get_tree_site_overflow_label_y(num_alignment_rows, has_structure_track=False, structure_row_y=None, gap_rows=0.5):
    num_alignment_rows = max(int(num_alignment_rows), 0)
    gap_rows = float(gap_rows)
    alignment_bottom = float(num_alignment_rows) - 0.5
    if has_structure_track:
        if structure_row_y is None:
            structure_row_y = float(num_alignment_rows) + 0.5
        structure_bottom = float(structure_row_y) + 0.5
        return structure_bottom + gap_rows
    return alignment_bottom + gap_rows


def get_highlight_leaf_and_branch_ids(tree, branch_ids):
    target_branch_ids = set([int(bid) for bid in branch_ids])
    highlight_branch_ids = set()
    highlight_leaf_ids = set()
    node_by_id = {}
    for node in tree.traverse():
        node_id = int(ete.get_prop(node, "numerical_label"))
        node_by_id[node_id] = node
    for node_id in target_branch_ids:
        node = node_by_id.get(node_id, None)
        if node is None:
            continue
        highlight_branch_ids.add(node_id)
        if ete.is_leaf(node):
            highlight_leaf_ids.add(node_id)
            continue
        for leaf in ete.iter_leaves(node):
            leaf_id = int(ete.get_prop(leaf, "numerical_label"))
            highlight_leaf_ids.add(leaf_id)
    return highlight_leaf_ids,highlight_branch_ids


def get_tree_site_leaf_label_color(leaf, highlight_branch_ids, branch_color_by_id, default_color):
    node = leaf
    while node is not None:
        node_id = int(ete.get_prop(node, "numerical_label"))
        if node_id in highlight_branch_ids:
            return branch_color_by_id.get(node_id, default_color)
        if ete.is_root(node):
            break
        node = node.up
    return default_color


def _get_alignment_to_internal_site_map(g):
    state_tensor = g.get('state_pep', None)
    if state_tensor is None:
        state_tensor = g.get('state_cdn', None)
    expected_num_site = None if state_tensor is None else int(state_tensor.shape[1])
    site_index_alignment = parser_misc.get_site_index_alignment(g=g, expected_num_site=expected_num_site)
    cache = g.get('_alignment_to_internal_site_map_cache', None)
    if isinstance(cache, dict):
        cached_sites = cache.get('site_index_alignment', None)
        if isinstance(cached_sites, np.ndarray) and np.array_equal(cached_sites, site_index_alignment):
            return cache['mapping']
    mapping = {int(aln_site): int(i) for i, aln_site in enumerate(site_index_alignment.tolist())}
    g['_alignment_to_internal_site_map_cache'] = {
        'site_index_alignment': site_index_alignment.copy(),
        'mapping': mapping,
    }
    return mapping


def _resolve_internal_site_index(g, codon_site_alignment):
    try:
        alignment_site = int(codon_site_alignment) - 1
    except (TypeError, ValueError):
        return None
    if alignment_site < 0:
        return None
    alignment_to_internal = _get_alignment_to_internal_site_map(g)
    return alignment_to_internal.get(int(alignment_site), None)


def _is_branch_site_gap(g, branch_id, codon_site_alignment):
    state_pep = g.get('state_pep', None)
    if state_pep is None:
        return False
    try:
        bid = int(branch_id)
    except (TypeError, ValueError):
        return False
    site_index = _resolve_internal_site_index(g=g, codon_site_alignment=codon_site_alignment)
    if site_index is None:
        return True
    if (site_index < 0) or (bid < 0):
        return False
    if (bid >= state_pep.shape[0]) or (site_index >= state_pep.shape[1]):
        return False
    site_state = np.nan_to_num(state_pep[bid, site_index, :], nan=0.0)
    return bool(site_state.sum() == 0)


def get_lineage_site_heatmap_values(df, display_meta, g):
    branch_ids = _normalize_branch_ids(g.get('branch_ids', [])).tolist()
    if len(branch_ids) == 0:
        return np.zeros((0, 0), dtype=float), []
    num_site = len(display_meta)
    values: NDArray[np.float64] = np.full((len(branch_ids), num_site), np.nan, dtype=np.float64)
    if num_site == 0:
        return values, branch_ids
    site_to_row = {
        int(site): i for i,site in enumerate(df.loc[:, 'codon_site_alignment'].astype(int).tolist())
    }
    for col_index,item in enumerate(display_meta):
        site = item.get('site', None)
        if site is None:
            continue
        row_index = site_to_row.get(int(site), None)
        if row_index is None:
            continue
        for row_index_branch,bid in enumerate(branch_ids):
            if _is_branch_site_gap(g=g, branch_id=int(bid), codon_site_alignment=int(site)):
                # Keep as NaN so the heatmap shows a gap-like blank cell, not 0-probability.
                continue
            mode = str(g.get('mode', '')).lower()
            if mode == 'set':
                col = 'N_set_branch_{}_prob'.format(int(bid))
                if col not in df.columns:
                    col = 'N_sub_{}'.format(int(bid))
            else:
                col = 'N_sub_{}'.format(int(bid))
            if col not in df.columns:
                continue
            value = float(df.at[row_index, col])
            if not np.isfinite(value):
                continue
            values[row_index_branch, col_index] = min(max(value, 0.0), 1.0)
    return values, branch_ids


def _get_set_channel_label(set_stat_type, channel_index, state_orders):
    set_stat_type = str(set_stat_type).strip().lower()
    if set_stat_type == 'any':
        return ''
    state_orders = np.asarray(state_orders, dtype=object).reshape(-1)
    if state_orders.shape[0] == 0:
        return str(int(channel_index))

    def _state_label(index):
        if state_orders.shape[0] == 0:
            return str(int(index))
        idx = int(index) % int(state_orders.shape[0])
        return str(state_orders[idx])

    if set_stat_type == 'spe':
        return 'X→{}'.format(_state_label(channel_index))
    return ''


def get_set_heatmap_column_labels(df, display_meta, g):
    if str(g.get('mode', '')).lower() != 'set':
        return {}
    if 'N_set_expr_channel_index' not in df.columns:
        return {}
    site_to_row = {
        int(site): i for i,site in enumerate(df.loc[:, 'codon_site_alignment'].astype(int).tolist())
    }
    set_stat_type = str(g.get('set_stat_type', 'any')).strip().lower()
    state_orders = sequence.get_nonsyn_state_orders(g)
    out = {}
    for item in display_meta:
        site = item.get('site', None)
        if site is None:
            continue
        site = int(site)
        row_index = site_to_row.get(site, None)
        if row_index is None:
            continue
        channel_value = pd.to_numeric(pd.Series([df.at[row_index, 'N_set_expr_channel_index']]), errors='coerce').iloc[0]
        if not np.isfinite(channel_value):
            continue
        channel_index = int(channel_value)
        if channel_index < 0:
            continue
        label = _get_set_channel_label(
            set_stat_type=set_stat_type,
            channel_index=channel_index,
            state_orders=state_orders,
        )
        if label == '':
            continue
        out[site] = label
    return out


def add_heatmap_column_labels(ax_heat, display_meta, label_by_site):
    if len(label_by_site) == 0:
        return None
    text_transform = matplotlib.transforms.blended_transform_factory(ax_heat.transData, ax_heat.transAxes)
    for col_index,item in enumerate(display_meta):
        site = item.get('site', None)
        if site is None:
            continue
        label = label_by_site.get(int(site), '')
        if label == '':
            continue
        ax_heat.text(
            col_index,
            1.02,
            label,
            transform=text_transform,
            ha='center',
            va='bottom',
            fontsize=font_size,
            fontfamily='DejaVu Sans',
            color='black',
            rotation=90,
            clip_on=False,
        )
    return None


def draw_lineage_site_heatmap(ax_heat, heat_values, heat_branch_ids, branch_color_by_id, cmap):
    ax_heat.set_facecolor((1, 1, 1, 0))
    if heat_values.shape[1] == 0:
        ax_heat.axis('off')
        return None
    masked = np.ma.masked_invalid(heat_values)
    im = ax_heat.imshow(
        masked,
        interpolation='nearest',
        aspect='auto',
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        origin='upper',
    )
    ax_heat.set_xlim(-0.5, heat_values.shape[1]-0.5)
    if heat_values.shape[0] > 0:
        ax_heat.set_ylim(heat_values.shape[0]-0.5, -0.5)
        y_ticks = np.arange(heat_values.shape[0], dtype=float)
        ax_heat.set_yticks(y_ticks.tolist())
        ax_heat.set_yticklabels([_format_branch_id_label(bid) for bid in heat_branch_ids], fontsize=font_size)
        for tick,bid in zip(ax_heat.get_yticklabels(), heat_branch_ids):
            tick.set_color(branch_color_by_id.get(int(bid), 'black'))
    else:
        ax_heat.set_yticks([])
    ax_heat.tick_params(axis='x', length=0, labelbottom=False, bottom=False, top=False, labeltop=False)
    ax_heat.tick_params(axis='y', length=0, pad=1)
    ax_heat.set_ylabel('Branch ID', fontsize=font_size)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    return im


def add_lineage_heatmap_colorbar(fig, ax_cb_holder, cmap):
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    ax_cb_holder.set_facecolor((1, 1, 1, 0))
    ax_cb_holder.set_xticks([])
    ax_cb_holder.set_yticks([])
    for spine in ax_cb_holder.spines.values():
        spine.set_visible(False)
    # Keep the scale fixed at 0-1 and place a compact bar in the left holder area.
    # This avoids overlap with the heatmap y-label and alignment top labels.
    cax = ax_cb_holder.inset_axes([0.535, 0.78, 0.15, 0.12])
    cbar = fig.colorbar(scalar, cax=cax, orientation='horizontal')
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=font_size, length=2, labeltop=True, labelbottom=False, pad=1)
    ax_cb_holder.text(
        0.61,
        0.72,
        'Substitution\nposterior\nprobability',
        transform=ax_cb_holder.transAxes,
        ha='center',
        va='top',
        fontsize=font_size,
        color='black',
    )
    return cbar


def _normalize_structure_aa_letter(value):
    if pd.isna(value):
        return ''
    txt = str(value).strip().upper()
    if txt in ('', '-', 'NAN', 'NONE'):
        return ''
    return txt[0]


def _normalize_structure_position_label(value):
    if pd.isna(value):
        return ''
    try:
        position = int(float(value))
    except (TypeError, ValueError):
        return ''
    if position <= 0:
        return ''
    return str(position)


def _select_structure_chain_from_df(df):
    aa_cols = [col for col in df.columns if str(col).startswith('aa_')]
    if len(aa_cols) == 0:
        return None, None, None
    best_chain = None
    best_aa_col = None
    best_position_col = None
    best_score = -1
    for aa_col in aa_cols:
        chain_name = str(aa_col)[3:]
        if chain_name == '':
            continue
        position_col = 'codon_site_' + chain_name
        if position_col not in df.columns:
            fallback_col = 'codon_site_pdb_' + chain_name
            if fallback_col not in df.columns:
                continue
            position_col = fallback_col
        aa_values = df.loc[:, aa_col].fillna('').astype(str).str.strip()
        nonempty_aa = int((aa_values != '').sum())
        pos_values = pd.to_numeric(df.loc[:, position_col], errors='coerce').fillna(0)
        nonzero_pos = int((pos_values > 0).sum())
        score = nonempty_aa + nonzero_pos
        if score > best_score:
            best_score = score
            best_chain = chain_name
            best_aa_col = aa_col
            best_position_col = position_col
    return best_chain, best_aa_col, best_position_col


def get_tree_site_structure_track(df, display_meta, g):
    if g.get('pdb', None) is None:
        return None, g
    g = get_highest_identity_chain_name(g)
    chain_name = g.get('highest_identity_chain_name', None)
    aa_col = None
    position_col = None
    if chain_name:
        aa_col = 'aa_' + str(chain_name)
        if aa_col in df.columns:
            position_col = 'codon_site_' + str(chain_name)
            if position_col not in df.columns:
                fallback_col = 'codon_site_pdb_' + str(chain_name)
                if fallback_col in df.columns:
                    position_col = fallback_col
                else:
                    position_col = None
    if (chain_name is None) or (aa_col is None) or (position_col is None):
        chain_name, aa_col, position_col = _select_structure_chain_from_df(df=df)
        if chain_name is None:
            return None, g
        g['highest_identity_chain_name'] = str(chain_name)
    if 'codon_site_alignment' not in df.columns:
        return None, g
    aa_values_all = df.loc[:, aa_col].fillna('').astype(str).str.strip()
    position_values_all = pd.to_numeric(df.loc[:, position_col], errors='coerce').fillna(0)
    has_structure_content_global = bool((aa_values_all != '').any() or (position_values_all > 0).any())
    if not has_structure_content_global:
        return None, g
    site_to_row = {
        int(site): i for i,site in enumerate(df.loc[:, 'codon_site_alignment'].astype(int).tolist())
    }
    aa_letters = []
    position_labels = []
    for item in display_meta:
        site = item.get('site', None)
        if site is None:
            aa_letters.append('')
            position_labels.append('')
            continue
        row_index = site_to_row.get(int(site), None)
        if row_index is None:
            aa_letters.append('')
            position_labels.append('')
            continue
        aa_letter = _normalize_structure_aa_letter(df.at[row_index, aa_col])
        position_label = _normalize_structure_position_label(df.at[row_index, position_col])
        aa_letters.append(aa_letter)
        position_labels.append(position_label)
    return {
        'chain_name': str(chain_name),
        'aa_letters': aa_letters,
        'position_labels': position_labels,
    }, g


def get_leaf_state_letter(g, leaf_id, codon_site_alignment):
    site_index = _resolve_internal_site_index(g=g, codon_site_alignment=codon_site_alignment)
    if site_index is None:
        return ''
    if (site_index < 0) or (site_index >= g['state_pep'].shape[1]):
        return ''
    state_values = g['state_pep'][leaf_id, site_index, :]
    if np.nan_to_num(state_values, nan=0.0).sum() == 0:
        return ''
    max_index = int(np.argmax(state_values))
    if max_index >= len(g['amino_acid_orders']):
        return ''
    return str(g['amino_acid_orders'][max_index])


def get_amino_acid_colors(g):
    tab20 = plt.get_cmap('tab20')
    aa_colors = {aa: tab20(i % 20) for i,aa in enumerate(g['amino_acid_orders'])}
    aa_colors[''] = (1.0, 1.0, 1.0, 1.0)
    # Match frequent residues in existing prototype style.
    aa_colors['A'] = (1.0, 0.34, 0.0, 1.0)
    aa_colors['V'] = (0.22, 0.04, 0.44, 1.0)
    aa_colors['T'] = (0.00, 0.53, 0.24, 1.0)
    aa_colors['I'] = (0.39, 0.50, 0.06, 1.0)
    return aa_colors


def get_text_color_for_background(rgba):
    r,g,b,_ = rgba
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    return 'black' if luminance > 0.55 else 'white'


def get_nice_scale_length(max_tree_depth):
    max_tree_depth = float(max_tree_depth)
    if max_tree_depth <= 0:
        return 1.0
    target = max_tree_depth * 0.12
    if target <= 0:
        return 1.0
    exponent = np.floor(np.log10(target))
    base = 10 ** exponent
    normalized = target / base
    if normalized <= 1.5:
        scale = 1.0
    elif normalized <= 3.5:
        scale = 2.0
    elif normalized <= 7.5:
        scale = 5.0
    else:
        scale = 10.0
    return scale * base


def _compile_species_regex(species_regex):
    regex_text = '' if species_regex is None else str(species_regex).strip()
    if regex_text == '':
        return None
    try:
        return re.compile(regex_text)
    except re.error as exc:
        txt = '--species_regex is not a valid regular expression: {}'
        raise ValueError(txt.format(exc))


def _extract_species_label(leaf_name, species_pattern):
    if species_pattern is None:
        return None
    label = '' if leaf_name is None else str(leaf_name)
    match = species_pattern.search(label)
    if match is None:
        return None
    if match.lastindex is not None:
        for i in range(1, int(match.lastindex) + 1):
            token = match.group(i)
            if token is None:
                continue
            token = str(token).strip()
            if token != '':
                return token
    token = str(match.group(0)).strip()
    if token == '':
        return None
    return token


def _get_species_by_leaf_id(tree, species_pattern):
    species_by_leaf_id = {}
    is_all_parsed = True
    num_leaf = 0
    for leaf in ete.iter_leaves(tree):
        num_leaf += 1
        leaf_id = int(ete.get_prop(leaf, "numerical_label"))
        species_label = _extract_species_label(leaf_name=(leaf.name or ''), species_pattern=species_pattern)
        if species_label is None:
            is_all_parsed = False
            continue
        species_by_leaf_id[leaf_id] = species_label
    return species_by_leaf_id, (is_all_parsed and (num_leaf > 0))


def get_species_overlap_node_types(tree, species_regex, require_all_tip_labels=False):
    species_pattern = _compile_species_regex(species_regex=species_regex)
    if species_pattern is None:
        return {}
    species_by_leaf_id, all_tip_labels_parsed = _get_species_by_leaf_id(
        tree=tree,
        species_pattern=species_pattern,
    )
    if bool(require_all_tip_labels) and (not all_tip_labels_parsed):
        return {}
    node_type_by_id = {}
    for node in tree.traverse():
        if ete.is_leaf(node):
            continue
        children = ete.get_children(node)
        if len(children) < 2:
            continue
        child_species_sets = []
        missing_species = False
        for child in children:
            species_set = set()
            for leaf in ete.iter_leaves(child):
                leaf_id = int(ete.get_prop(leaf, "numerical_label"))
                species_label = species_by_leaf_id.get(leaf_id, None)
                if species_label is None:
                    missing_species = True
                    break
                species_set.add(species_label)
            if missing_species:
                break
            child_species_sets.append(species_set)
        if missing_species:
            continue
        is_duplication = False
        for i in range(len(child_species_sets)):
            for j in range(i + 1, len(child_species_sets)):
                if len(child_species_sets[i].intersection(child_species_sets[j])) > 0:
                    is_duplication = True
                    break
            if is_duplication:
                break
        node_id = int(ete.get_prop(node, "numerical_label"))
        node_type_by_id[node_id] = 'duplication' if is_duplication else 'speciation'
    return node_type_by_id


def plot_tree_site(df: pd.DataFrame, g: MutableMapping[str, Any]) -> list[str]:
    if not bool(g.get('tree_site_plot', True)):
        print('Skipping tree + site summary outputs (--tree_site_plot no).', flush=True)
        return []
    tree_site_df,min_prob = classify_tree_site_categories(df=df, g=g)
    display_meta = get_tree_site_display_sites(tree_site_df=tree_site_df, g=g, df=df)
    overflow_count = get_tree_site_overflow_count(
        tree_site_df=tree_site_df,
        display_meta=display_meta,
        g=g,
        df=df,
    )
    num_convergence = int(sum(
        1 for item in display_meta
        if (item.get('site', None) is not None) and (str(item.get('category', '')) == 'convergent')
    ))
    num_divergence = int(sum(
        1 for item in display_meta
        if (item.get('site', None) is not None) and (str(item.get('category', '')) == 'divergent')
    ))
    xcoord,ycoord,leaf_order = get_tree_plot_coordinates(tree=g['tree'])
    branch_ids_in_order = _normalize_branch_ids(g['branch_ids']).tolist()
    highlight_branch_ids_in_order = _normalize_branch_ids(
        g.get('tree_site_highlight_branch_ids', branch_ids_in_order)
    ).tolist()
    branch_ids = set(highlight_branch_ids_in_order)
    mode = str(g.get('mode', '')).lower()
    show_branch_heatmap = mode in ('lineage', 'set', 'intersection')
    if mode == 'lineage':
        color_ids_in_order = list(dict.fromkeys(branch_ids_in_order + highlight_branch_ids_in_order))
        branch_color_by_id = get_tree_site_branch_color_by_id(branch_ids=color_ids_in_order, g=g, default_color='firebrick')
    else:
        branch_color_by_id = {int(bid): 'firebrick' for bid in highlight_branch_ids_in_order}
    highlight_leaf_ids,highlight_branch_ids = get_highlight_leaf_and_branch_ids(tree=g['tree'], branch_ids=branch_ids)
    x_values = np.array(list(xcoord.values()), dtype=float)
    x_max = x_values.max() if x_values.shape[0] else 1.0
    if x_max <= 0:
        x_max = 1.0
    tip_label_texts = []
    for leaf in ete.iter_leaves(g['tree']):
        tip_label_texts.append(leaf.name or '')
    max_tip_label_chars = max([len(txt) for txt in tip_label_texts]) if len(tip_label_texts) > 0 else 1
    structure_track, g = get_tree_site_structure_track(df=df, display_meta=display_meta, g=g)
    has_structure_track = structure_track is not None
    structure_row_gap = 0.5 if has_structure_track else 0.0
    structure_row_y = (float(len(leaf_order)) + structure_row_gap) if has_structure_track else None
    # Keep structure-row/tick proximity unchanged even when overflow text is shown.
    overflow_row_space = 2.0 if ((overflow_count > 0) and (not has_structure_track)) else 0.0
    num_site_rows_for_layout = float(len(leaf_order)) + (1.5 if has_structure_track else 0.0) + overflow_row_space

    num_display_site = max(len(display_meta), 1)
    num_leaf = max(num_site_rows_for_layout, 1.0)
    # Dense defaults for compact tree/site output.
    base_tree_panel_width = min(max(5.0, 4.2 + x_max * 0.42), 10.0) * 0.5
    tip_label_width_in = min(max(0.8, max_tip_label_chars * 0.053), 2.8)
    tree_panel_width = base_tree_panel_width + tip_label_width_in
    # Keep a constant physical width per displayed alignment column.
    site_column_width_in = 0.112
    site_panel_width = max(site_column_width_in, num_display_site * site_column_width_in)
    fig_width = tree_panel_width + site_panel_width
    tip_label_spacing = get_tree_site_tip_label_spacing(g)
    tree_site_fig_max_height = get_tree_site_fig_max_height(g)
    fig_height = min(max(2.5, num_leaf * 0.13 * tip_label_spacing + 0.55), tree_site_fig_max_height)
    # Keep one heatmap row (one branch) aligned to one alignment row height.
    alignment_row_height_in = (fig_height - 0.55) / max(num_leaf, 1.0)
    fg_color = 'firebrick'
    bg_branch_color = '#4d4d4d'
    bg_label_color = '#5f6f7f'
    internal_label_color = '#7a7a7a'
    internal_label_size = 4.2
    # Exact RGB colors requested for node types.
    speciation_color = (0.0, 0.0, 1.0)
    duplication_color = (1.0, 0.0, 0.0)
    # Marker diameter reduced to 50% of previous size.
    node_marker_diameter_scale = 0.5
    node_marker_area = 18.0 * (node_marker_diameter_scale ** 2)
    node_marker_size_pt = 4.8 * node_marker_diameter_scale
    node_type_by_id = get_species_overlap_node_types(
        tree=g['tree'],
        species_regex=g.get('species_regex', ''),
        require_all_tip_labels=(str(g.get('species_overlap_node_plot', 'auto')).strip().lower() == 'auto'),
    )
    if str(g.get('species_overlap_node_plot', 'auto')).strip().lower() == 'no':
        node_type_by_id = {}

    if show_branch_heatmap:
        # Keep about half-row visual margin between heatmap and site-number labels.
        heatmap_hspace = 0.22
        heat_panel_height = alignment_row_height_in * max(len(branch_ids_in_order), 1)
        fig_height += heat_panel_height
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[tree_panel_width, site_panel_width],
            height_ratios=[heat_panel_height, fig_height - heat_panel_height],
            wspace=0.01,
            hspace=heatmap_hspace,
        )
        ax_cb_holder = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_tree = fig.add_subplot(gs[1, 0])
        ax_site = fig.add_subplot(gs[1, 1], sharey=ax_tree)
    else:
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(1, 2, width_ratios=[tree_panel_width, site_panel_width], wspace=0.01)
        ax_tree = fig.add_subplot(gs[0, 0])
        ax_site = fig.add_subplot(gs[0, 1], sharey=ax_tree)
        ax_cb_holder = None
        ax_heat = None

    for node in g['tree'].traverse():
        if ete.is_leaf(node):
            continue
        node_id = int(ete.get_prop(node, "numerical_label"))
        children = ete.get_children(node)
        if len(children) <= 1:
            continue
        for child in children:
            child_id = int(ete.get_prop(child, "numerical_label"))
            is_target = ((node_id in highlight_branch_ids) and (child_id in highlight_branch_ids))
            if is_target:
                color = branch_color_by_id.get(child_id, branch_color_by_id.get(node_id, fg_color))
            else:
                color = bg_branch_color
            linewidth = 0.8
            ax_tree.plot([xcoord[node_id], xcoord[node_id]], [ycoord[node_id], ycoord[child_id]],
                         color=color, linewidth=linewidth, zorder=1, solid_capstyle=TREE_LINE_CAPSTYLE)

    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        node_id = int(ete.get_prop(node, "numerical_label"))
        parent_id = int(ete.get_prop(node.up, "numerical_label"))
        is_target = node_id in highlight_branch_ids
        color = branch_color_by_id.get(node_id, fg_color) if is_target else bg_branch_color
        linewidth = 0.8
        ax_tree.plot([xcoord[parent_id], xcoord[node_id]], [ycoord[node_id], ycoord[node_id]],
                     color=color, linewidth=linewidth, zorder=2, solid_capstyle=TREE_LINE_CAPSTYLE)

    root = ete.get_tree_root(g['tree'])
    root_id = int(ete.get_prop(root, "numerical_label"))
    root_stub = max(x_max * 0.03, 0.03)
    root_color = fg_color if (root_id in highlight_branch_ids) else bg_branch_color
    ax_tree.plot([-root_stub, xcoord[root_id]], [ycoord[root_id], ycoord[root_id]],
                 color=root_color, linewidth=0.8, zorder=2, solid_capstyle=TREE_LINE_CAPSTYLE)

    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        node_id = int(ete.get_prop(node, "numerical_label"))
        parent_id = int(ete.get_prop(node.up, "numerical_label"))
        if node_id in highlight_branch_ids:
            node_color = branch_color_by_id.get(node_id, fg_color)
        else:
            node_color = internal_label_color
        # Draw branch IDs on the horizontal branch segment (parent -> node),
        # not at the node point itself.
        parent_x = xcoord[parent_id]
        node_x = xcoord[node_id]
        label_x = parent_x + (node_x - parent_x) * 0.5
        label_y = ycoord[node_id] - 0.08
        ax_tree.text(
            label_x,
            label_y,
            _format_branch_id_label(node_id),
            va='center',
            ha='center',
            fontsize=internal_label_size,
            color=node_color,
            zorder=4,
        )
    if len(node_type_by_id) > 0:
        for node in g['tree'].traverse():
            if ete.is_leaf(node):
                continue
            node_id = int(ete.get_prop(node, "numerical_label"))
            node_type = node_type_by_id.get(node_id, None)
            if node_type is None:
                continue
            marker_color = duplication_color if (node_type == 'duplication') else speciation_color
            ax_tree.scatter(
                [xcoord[node_id]],
                [ycoord[node_id]],
                s=node_marker_area,
                marker='o',
                facecolor=marker_color,
                edgecolor='white',
                linewidth=0.4,
                zorder=5,
            )
        legend_handles = [
            matplotlib.lines.Line2D(
                [0], [0],
                marker='o',
                linestyle='None',
                markerfacecolor=speciation_color,
                markeredgecolor='white',
                markeredgewidth=0.4,
                markersize=node_marker_size_pt,
                label='Speciation node',
            ),
            matplotlib.lines.Line2D(
                [0], [0],
                marker='o',
                linestyle='None',
                markerfacecolor=duplication_color,
                markeredgecolor='white',
                markeredgewidth=0.4,
                markersize=node_marker_size_pt,
                label='Duplication node',
            ),
        ]
        ax_tree.legend(
            handles=legend_handles,
            loc='lower left',
            bbox_to_anchor=(0.0, 1.09),
            frameon=False,
            fontsize=font_size - 1,
            borderaxespad=0.1,
            handletextpad=0.3,
            labelspacing=0.2,
            ncol=1,
        )

    # Increase tree-to-tip-label margin by ~1 character to avoid overlap with terminal branch IDs.
    label_offset = (x_max * 0.02 + 0.04) * 1.05
    one_char_margin = min(max(0.02, x_max * 0.008), 0.04)
    label_offset += one_char_margin
    for leaf in ete.iter_leaves(g['tree']):
        node_id = int(ete.get_prop(leaf, "numerical_label"))
        label = (leaf.name or '')
        if mode == 'lineage':
            if node_id in highlight_leaf_ids:
                label_color = get_tree_site_leaf_label_color(
                    leaf=leaf,
                    highlight_branch_ids=highlight_branch_ids,
                    branch_color_by_id=branch_color_by_id,
                    default_color=fg_color,
                )
            else:
                label_color = bg_label_color
        else:
            is_target_leaf = node_id in highlight_leaf_ids
            label_color = fg_color if is_target_leaf else bg_label_color
        ax_tree.text(x_max + label_offset, ycoord[node_id], label, va='center', ha='left',
                     fontsize=font_size, color=label_color)
    if (len(leaf_order) > 0) or has_structure_track:
        if structure_row_y is not None:
            panel_y_max = float(structure_row_y) + 0.5 + overflow_row_space
        else:
            panel_y_max = float(len(leaf_order)) - 0.5 + overflow_row_space
        ax_tree.set_ylim(panel_y_max, -0.5)
    left_xlim = -root_stub * 1.5
    tip_label_data_span = min(max(0.45, max_tip_label_chars * 0.05), 3.0)
    right_xlim = x_max + tip_label_data_span
    ax_tree.set_xlim(left_xlim, right_xlim)

    scale_length = get_nice_scale_length(x_max)
    scale_x_start = left_xlim + (right_xlim - left_xlim) * 0.03
    scale_x_end = scale_x_start + scale_length
    if scale_x_end > (x_max * 0.95):
        scale_length = get_nice_scale_length(x_max * 0.5)
        scale_x_end = scale_x_start + scale_length
    if len(leaf_order) > 0:
        scale_y = len(leaf_order) - 0.85
    else:
        scale_y = -0.1
    scale_tick = 0.08
    ax_tree.plot([scale_x_start, scale_x_end], [scale_y, scale_y], color='black', linewidth=1.0, zorder=4, solid_capstyle=TREE_LINE_CAPSTYLE)
    ax_tree.plot([scale_x_start, scale_x_start], [scale_y-scale_tick, scale_y+scale_tick], color='black', linewidth=1.0, zorder=4, solid_capstyle=TREE_LINE_CAPSTYLE)
    ax_tree.plot([scale_x_end, scale_x_end], [scale_y-scale_tick, scale_y+scale_tick], color='black', linewidth=1.0, zorder=4, solid_capstyle=TREE_LINE_CAPSTYLE)
    ax_tree.text((scale_x_start + scale_x_end) / 2, scale_y + 0.25, '{:g}'.format(scale_length),
                 va='top', ha='center', fontsize=font_size-1, color='black')

    branch_text = ','.join([str(int(bid)) for bid in branch_ids_in_order])
    title_text = 'Focal branch IDs: {}'.format(branch_text)
    if (num_convergence + num_divergence) > 0:
        min_prob_text = '{:g}'.format(float(min_prob))
        title_text += '; Convergence & Divergence: N={:,}&{:,}, PP \u2265 {}'
        title_text = title_text.format(num_convergence, num_divergence, min_prob_text)
    if mode == 'set':
        mode_expression = str(g.get('mode_expression', '')).strip()
        set_stat_type = str(g.get('set_stat_type', '')).strip()
        set_min_prob_text = '{:.2f}'.format(float(min_prob))
        if mode_expression != '':
            if set_stat_type != '':
                title_text += '; Set operation: {} ({}, PP ≥ {})'.format(
                    mode_expression,
                    set_stat_type,
                    set_min_prob_text,
                )
            else:
                title_text += '; Set operation: {} (PP ≥ {})'.format(
                    mode_expression,
                    set_min_prob_text,
                )
    ax_tree.set_title(title_text, loc='left')
    ax_tree.axis('off')

    aa_colors = get_amino_acid_colors(g)
    separator_color = (0.96, 0.96, 0.96, 1.0)
    for col_idx,item in enumerate(display_meta):
        site = item['site']
        is_separator = (site is None)
        for row_idx,leaf_id in enumerate(leaf_order):
            if is_separator:
                facecolor = separator_color
                aa_letter = ''
            else:
                aa_letter = get_leaf_state_letter(g=g, leaf_id=leaf_id, codon_site_alignment=site)
                facecolor = aa_colors.get(aa_letter, (0.90, 0.90, 0.90, 1.0))
            rect = matplotlib.patches.Rectangle(
                xy=(col_idx-0.5, ycoord[leaf_id]-0.5),
                width=1.0,
                height=1.0,
                facecolor=facecolor,
                edgecolor='white',
                linewidth=0.6,
            )
            ax_site.add_patch(rect)
            if aa_letter != '':
                ax_site.text(col_idx, ycoord[leaf_id], aa_letter, ha='center', va='center',
                             fontsize=font_size, color=get_text_color_for_background(facecolor))
    if structure_row_y is not None:
        for col_idx,item in enumerate(display_meta):
            site = item['site']
            is_separator = (site is None)
            if is_separator:
                facecolor = separator_color
                aa_letter = ''
            else:
                aa_letter = str(structure_track['aa_letters'][col_idx])
                facecolor = aa_colors.get(aa_letter, (0.90, 0.90, 0.90, 1.0))
            rect = matplotlib.patches.Rectangle(
                xy=(col_idx-0.5, structure_row_y-0.5),
                width=1.0,
                height=1.0,
                facecolor=facecolor,
                edgecolor='white',
                linewidth=0.6,
            )
            ax_site.add_patch(rect)
            if aa_letter != '':
                ax_site.text(
                    col_idx,
                    structure_row_y,
                    aa_letter,
                    ha='center',
                    va='center',
                    fontsize=font_size,
                    color=get_text_color_for_background(facecolor),
                )
        structure_label_transform = matplotlib.transforms.blended_transform_factory(ax_site.transAxes, ax_site.transData)
        ax_site.text(
            -0.02,
            structure_row_y,
            str(structure_track['chain_name']),
            transform=structure_label_transform,
            ha='right',
            va='center',
            fontsize=font_size,
            color='black',
            clip_on=False,
        )

    if len(display_meta) == 0:
        ax_site.set_xlim(-0.5, 0.5)
    else:
        ax_site.set_xlim(-0.5, len(display_meta)-0.5)
    if (len(leaf_order) > 0) or has_structure_track:
        if structure_row_y is not None:
            panel_y_max = float(structure_row_y) + 0.5 + overflow_row_space
        else:
            panel_y_max = float(len(leaf_order)) - 0.5 + overflow_row_space
        ax_site.set_ylim(panel_y_max, -0.5)
    tick_positions = [i for i,item in enumerate(display_meta) if item['site'] is not None]
    alignment_tick_labels = [str(display_meta[i]['site']) for i in tick_positions]
    ax_site.set_xticks([])
    ax_site.tick_params(axis='x', bottom=False, labelbottom=False, top=False, labeltop=False)
    alignment_axis = ax_site.secondary_xaxis('top')
    alignment_axis.set_xticks(tick_positions)
    alignment_axis.set_xticklabels(alignment_tick_labels, rotation=90, fontsize=font_size)
    alignment_axis.tick_params(axis='x', length=0, pad=2)
    alignment_axis.set_xlabel('')
    for spine in alignment_axis.spines.values():
        spine.set_visible(False)
    left_axis_label_transform = ax_site.transAxes
    left_axis_label_x = 0.0
    left_label_margin_pt = 4.0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _get_ticklabel_center_y_axes(tick_labels, fallback_y):
        labels = [lab for lab in tick_labels if str(lab.get_text()) != '']
        if len(labels) == 0:
            return float(fallback_y)
        y_min = min([lab.get_window_extent(renderer=renderer).y0 for lab in labels])
        y_max = max([lab.get_window_extent(renderer=renderer).y1 for lab in labels])
        y_center_disp = (float(y_min) + float(y_max)) * 0.5
        _, y_axes = ax_site.transAxes.inverted().transform((0.0, y_center_disp))
        return float(y_axes)
    if show_branch_heatmap:
        alignment_axis.set_xlabel('Alignment position (aa)', fontsize=font_size, labelpad=2)
    else:
        alignment_label_y = _get_ticklabel_center_y_axes(alignment_axis.get_xticklabels(), fallback_y=1.0)
        ax_site.annotate(
            'Alignment position (aa)',
            xy=(left_axis_label_x, alignment_label_y),
            xycoords=left_axis_label_transform,
            xytext=(-left_label_margin_pt, 0.0),
            textcoords='offset points',
            ha='right',
            va='center',
            fontsize=font_size,
            color='black',
            annotation_clip=False,
        )
    if has_structure_track:
        structure_tick_labels = [str(structure_track['position_labels'][i]) for i in tick_positions]
        ax_site.set_xticks(tick_positions)
        ax_site.set_xticklabels(structure_tick_labels, rotation=90, fontsize=font_size)
        ax_site.tick_params(axis='x', length=0, pad=4, bottom=True, labelbottom=True, top=False, labeltop=False)
        ax_site.set_xlabel('')
        structure_label_y = _get_ticklabel_center_y_axes(ax_site.get_xticklabels(), fallback_y=0.0)
        ax_site.annotate(
            'Structure position (aa)',
            xy=(left_axis_label_x, structure_label_y),
            xycoords=left_axis_label_transform,
            xytext=(-left_label_margin_pt, 0.0),
            textcoords='offset points',
            ha='right',
            va='center',
            fontsize=font_size,
            color='black',
            annotation_clip=False,
        )
    else:
        ax_site.set_xticks([])
        ax_site.set_xlabel('')
    ax_site.tick_params(axis='y', left=False, labelleft=False)
    for spine in ax_site.spines.values():
        spine.set_visible(False)

    if show_branch_heatmap:
        heat_values, heat_branch_ids = get_lineage_site_heatmap_values(
            df=df,
            display_meta=display_meta,
            g=g,
        )
        heatmap_cmap = plt.get_cmap('viridis')
        if hasattr(heatmap_cmap, 'copy'):
            heatmap_cmap = heatmap_cmap.copy()
        else:
            heatmap_cmap = matplotlib.colors.ListedColormap(heatmap_cmap(np.linspace(0, 1, 256)))
        # Match alignment gap representation for missing/gap cells in PP heatmap.
        heatmap_cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))
        _ = draw_lineage_site_heatmap(
            ax_heat=ax_heat,
            heat_values=heat_values,
            heat_branch_ids=heat_branch_ids,
            branch_color_by_id=branch_color_by_id,
            cmap=heatmap_cmap,
        )
        _ = add_lineage_heatmap_colorbar(
            fig=fig,
            ax_cb_holder=ax_cb_holder,
            cmap=heatmap_cmap,
        )
        if mode == 'set':
            label_by_site = get_set_heatmap_column_labels(df=df, display_meta=display_meta, g=g)
            _ = add_heatmap_column_labels(
                ax_heat=ax_heat,
                display_meta=display_meta,
                label_by_site=label_by_site,
            )
        ax_heat.set_xlim(ax_site.get_xlim())

    if overflow_count > 0:
        overflow_label = '+{} sites with PP ≥ {:.2f}'.format(int(overflow_count), float(min_prob))
        if has_structure_track:
            # Place overflow text below structure-site tick labels.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ticklabels = [lab for lab in ax_site.get_xticklabels() if str(lab.get_text()) != '']
            overflow_transform = ax_site.transAxes
            if len(ticklabels) > 0:
                y_min = min([lab.get_window_extent(renderer=renderer).y0 for lab in ticklabels])
                _, y_bottom_axes = ax_site.transAxes.inverted().transform((0.0, float(y_min)))
                overflow_label_y = float(y_bottom_axes) - 0.03
            else:
                overflow_label_y = -0.08
        else:
            # Keep overflow text below the alignment track.
            overflow_label_y = get_tree_site_overflow_label_y(
                num_alignment_rows=len(leaf_order),
                has_structure_track=has_structure_track,
                structure_row_y=structure_row_y,
                gap_rows=1.0,
            )
            overflow_transform = matplotlib.transforms.blended_transform_factory(ax_site.transAxes, ax_site.transData)
        ax_site.text(
            0.995,
            overflow_label_y,
            overflow_label,
            transform=overflow_transform,
            ha='right',
            va='center',
            fontsize=font_size,
            color='black',
            fontweight='bold',
            clip_on=False,
        )

    if show_branch_heatmap:
        fig.subplots_adjust(top=0.92, left=0.04, right=0.99, wspace=0.01, hspace=heatmap_hspace)
    else:
        fig.subplots_adjust(top=0.84, left=0.04, right=0.99, wspace=0.01)

    output_prefix = str(g.get('tree_site_plot_prefix', _site_output_prefix(g))).strip()
    if output_prefix == '':
        output_prefix = _site_output_prefix(g)
    fmt = str(g.get('tree_site_plot_format', 'pdf')).lower()
    fig_path = os.path.join(g['site_outdir'], output_prefix + '.tree_site.' + fmt)
    fig.savefig(
        fig_path,
        format=fmt,
        transparent=True,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.02,
    )
    plt.close(fig)
    print('Writing tree + site plot: {}'.format(fig_path), flush=True)

    table_path = os.path.join(g['site_outdir'], output_prefix + '.tree_site.tsv')
    if not bool(g.get('tree_site_output_table', True)):
        if os.path.exists(table_path):
            os.remove(table_path)
            print('Removing stale tree + site category table: {}'.format(table_path), flush=True)
        print('Skipping tree + site category table.', flush=True)
        return [fig_path]

    tree_site_df.loc[:, 'is_plotted'] = False
    tree_site_df.loc[:, 'plot_order'] = np.nan
    current_order = 1
    for item in display_meta:
        site = item['site']
        if site is None:
            continue
        is_site = (tree_site_df.loc[:, 'codon_site_alignment'] == site)
        tree_site_df.loc[is_site, 'is_plotted'] = True
        tree_site_df.loc[is_site, 'plot_order'] = current_order
        current_order += 1
    tree_site_df = expand_site_table_to_alignment(df=tree_site_df, g=g)
    tsv.write_dataframe(
        tree_site_df,
        table_path,
        float_format=g['float_format'],
        chunksize=10000,
    )
    print('Writing tree + site category table: {}'.format(table_path), flush=True)
    return [fig_path, table_path]


def expand_site_table_to_alignment(df, g):
    out = parser_misc.expand_site_axis_table_to_alignment(
        df=df,
        g=g,
        site_col='codon_site_alignment',
        group_cols=[],
        site_is_one_based=True,
    )
    if 'nuc_site_alignment' in out.columns:
        out.loc[:, 'nuc_site_alignment'] = ((out.loc[:, 'codon_site_alignment'].astype(np.int64) * 3) - 2)
    return out

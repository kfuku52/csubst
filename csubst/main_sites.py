import numpy as np
import pandas as pd

import itertools
import os
import re
import sys

from csubst import genetic_code
from csubst import output_manifest
from csubst import parser_misc
from csubst import runtime
from csubst import sequence
from csubst import substitution
from csubst import substitution_sparse
from csubst import tree
from csubst import tsv
from csubst import ete
from csubst import variant_effect

from csubst.plotting import (
    TREE_LINE_CAPSTYLE as TREE_LINE_CAPSTYLE,
    VESM_XTICK_LABEL_GAP_POINTS as VESM_XTICK_LABEL_GAP_POINTS,
    font_size as font_size,
    matplotlib as matplotlib,
    plt as plt,
)
from csubst.site_tree_plot import (
    _compile_species_regex as _compile_species_regex,
    _extract_species_label as _extract_species_label,
    _format_branch_id_label as _format_branch_id_label,
    _get_alignment_to_internal_site_map as _get_alignment_to_internal_site_map,
    _get_lineage_midpoint_distances as _get_lineage_midpoint_distances,
    _get_lineage_oldness_fracs as _get_lineage_oldness_fracs,
    _get_lineage_rgb_by_branch as _get_lineage_rgb_by_branch,
    _get_set_channel_label as _get_set_channel_label,
    _get_species_by_leaf_id as _get_species_by_leaf_id,
    _is_branch_site_gap as _is_branch_site_gap,
    _normalize_branch_ids as _normalize_branch_ids,
    _normalize_structure_aa_letter as _normalize_structure_aa_letter,
    _normalize_structure_position_label as _normalize_structure_position_label,
    _oldness_frac_to_rgb as _oldness_frac_to_rgb,
    _resolve_internal_site_index as _resolve_internal_site_index,
    _select_structure_chain_from_df as _select_structure_chain_from_df,
    _site_output_prefix as _site_output_prefix,
    add_heatmap_column_labels as add_heatmap_column_labels,
    add_lineage_heatmap_colorbar as add_lineage_heatmap_colorbar,
    classify_tree_site_categories as classify_tree_site_categories,
    draw_lineage_site_heatmap as draw_lineage_site_heatmap,
    expand_site_table_to_alignment as expand_site_table_to_alignment,
    get_amino_acid_colors as get_amino_acid_colors,
    get_highest_identity_chain_name as get_highest_identity_chain_name,
    get_highlight_leaf_and_branch_ids as get_highlight_leaf_and_branch_ids,
    get_leaf_state_letter as get_leaf_state_letter,
    get_lineage_display_sites as get_lineage_display_sites,
    get_lineage_site_heatmap_values as get_lineage_site_heatmap_values,
    get_min_combinat_prob as get_min_combinat_prob,
    get_min_single_prob as get_min_single_prob,
    get_nice_scale_length as get_nice_scale_length,
    get_set_display_sites as get_set_display_sites,
    get_set_heatmap_column_labels as get_set_heatmap_column_labels,
    get_species_overlap_node_types as get_species_overlap_node_types,
    get_text_color_for_background as get_text_color_for_background,
    get_tree_plot_coordinates as get_tree_plot_coordinates,
    get_tree_site_branch_color_by_id as get_tree_site_branch_color_by_id,
    get_tree_site_display_sites as get_tree_site_display_sites,
    get_tree_site_fig_max_height as get_tree_site_fig_max_height,
    get_tree_site_leaf_label_color as get_tree_site_leaf_label_color,
    get_tree_site_min_prob as get_tree_site_min_prob,
    get_tree_site_overflow_count as get_tree_site_overflow_count,
    get_tree_site_overflow_label_y as get_tree_site_overflow_label_y,
    get_tree_site_plot_max_sites as get_tree_site_plot_max_sites,
    get_tree_site_structure_track as get_tree_site_structure_track,
    get_tree_site_tip_label_spacing as get_tree_site_tip_label_spacing,
    plot_tree_site as plot_tree_site,
)


def bool2yn(flag):
    return 'Y' if bool(flag) else 'N'


def _get_site_output_manifest_metadata(g):
    effective_min_prob = float(get_tree_site_min_prob(g))
    return {
        'single_branch_mode': bool2yn(g.get('single_branch_mode', False)),
        'tree_site_plot': bool2yn(g.get('tree_site_plot', True)),
        'site_state_plot': bool2yn(g.get('site_state_plot', True)),
        'site_summary_plot': bool2yn(g.get('site_summary_plot', True)),
        'tree_site_plot_format': str(g.get('tree_site_plot_format', 'pdf')).lower(),
        'min_prob_effective': effective_min_prob,
        # Backward-compatible alias for downstream consumers.
        'tree_site_plot_min_prob_effective': effective_min_prob,
        'tree_site_plot_max_sites': int(get_tree_site_plot_max_sites(g)),
        'pdb_mode': bool2yn(g.get('pdb', None) is not None),
        'vep_model': str(g.get('vep_model', 'none')),
        'vep_min_event_pp': float(g.get('vep_min_event_pp', 0.8)),
        'vep_site_aggregate': str(g.get('vep_site_aggregate', 'most_deleterious')),
    }


def add_site_output_manifest_row(manifest_rows, output_path, output_kind, g, branch_ids, note=''):
    return output_manifest.add_output_manifest_row(
        manifest_rows=manifest_rows,
        output_path=output_path,
        output_kind=output_kind,
        note=note,
        base_dir=g['site_outdir'],
        branch_ids=branch_ids,
        extra_fields=_get_site_output_manifest_metadata(g),
    )


def write_site_output_manifest(manifest_rows, g, branch_ids):
    manifest_path = os.path.join(g['site_outdir'], _site_output_prefix(g) + '.outputs.tsv')
    manifest_path = output_manifest.write_output_manifest(
        manifest_rows=manifest_rows,
        manifest_path=manifest_path,
        note='manifest_self_row',
        base_dir=g['site_outdir'],
        branch_ids=branch_ids,
        extra_fields=_get_site_output_manifest_metadata(g),
    )
    print('Writing site output manifest: {}'.format(manifest_path), flush=True)
    return manifest_path


def get_state(node, g):
    seq = ete.get_prop(node, 'sequence', '').upper()
    if seq == '':
        raise AssertionError('Leaf sequence not found for node "{}". Check tree/alignment labels.'.format(node.name))
    if len(seq) % 3 != 0:
        raise AssertionError('Sequence length is not multiple of 3. Node name = ' + node.name)
    state_matrix = np.zeros([g['num_input_site'], g['num_input_state']], dtype=g['float_type'])
    for s in np.arange(g['num_input_site']):
        codon = seq[(s*3):((s+1)*3)]
        codon_index = sequence.get_state_index(state=codon, input_state=g['codon_orders'], ambiguous_table=genetic_code.ambiguous_table)
        for ci in codon_index:
            state_matrix[s,ci] = 1/len(codon_index)
    return(state_matrix)


def add_gapline(df, gapcol, xcol, yvalue, lw, ax):
    x_values = df.loc[:,xcol].values - 0.5
    if x_values.size == 0:
        return None
    y_values = np.ones(x_values.shape) * yvalue
    gap_values = df.loc[:,gapcol].values
    bars = dict()
    bars['x_start'] = list()
    bars['x_end'] = list()
    bars['y'] = list()
    bars['gap'] = list()
    bars['color'] = list()
    current_x = x_values[0]
    current_y = y_values[0]
    current_gap = gap_values[0]
    i_ranges = np.arange(len(x_values))
    i_end = i_ranges[-1]
    for i in i_ranges:
        x_value = x_values[i]
        y_value = y_values[i]
        gap_value = gap_values[i]
        if (i == i_end):
            x_value += 1
        if (gap_value!=current_gap)|(i == i_end):
            bars['x_start'].append(current_x)
            bars['x_end'].append(x_value)
            bars['y'].append(current_y)
            bars['gap'].append(current_gap)
            cval = 1 - current_gap
            bars['color'].append((cval,cval,cval,))
            current_x = x_value
            current_y = y_value
            current_gap = gap_value
    for i in np.arange(len(bars['x_start'])):
        y = bars['y'][i]
        x_start = bars['x_start'][i]
        x_end = bars['x_end'][i]
        color = bars['color'][i]
        ax.hlines(y=y, xmin=x_start, xmax=x_end, linewidth=lw, color=color, zorder=0)


def _zeros_yvalues(num_row):
    return np.zeros(num_row, dtype=float)


def _get_yvalues_sub(df, SN):
    if SN == 'S':
        yvalues = df.loc[:, 'S_sub'].to_numpy(copy=True)
        is_enough_value = (yvalues > 0.01)
        yvalues[is_enough_value] = df.loc[is_enough_value, ['N_sub', 'S_sub']].sum(axis=1).values
        return yvalues
    return df.loc[:, 'N_sub'].to_numpy(copy=True)


def _get_yvalues_set_expr(df, SN):
    if SN == 'S':
        return _zeros_yvalues(df.shape[0])
    if 'N_set_expr_prob' in df.columns:
        return df.loc[:, 'N_set_expr_prob'].to_numpy(copy=True)
    if 'N_set_expr' in df.columns:
        return df.loc[:, 'N_set_expr'].to_numpy(copy=True).astype(float)
    return _zeros_yvalues(df.shape[0])


def _get_yvalues_set_other(df, SN):
    if SN == 'S':
        return _zeros_yvalues(df.shape[0])
    if 'N_set_other' in df.columns:
        return df.loc[:, 'N_set_other'].to_numpy(copy=True).astype(float)
    return _zeros_yvalues(df.shape[0])


def _get_yvalues_sub_branch(df, sub_type, SN):
    branch_id_txt = sub_type.replace('_sub_branch_', '')
    branch_id = int(branch_id_txt)
    n_col = 'N_sub_{}'.format(branch_id)
    s_col = 'S_sub_{}'.format(branch_id)
    nvalues = df.loc[:, n_col].to_numpy(copy=True) if (n_col in df.columns) else _zeros_yvalues(df.shape[0])
    svalues = df.loc[:, s_col].to_numpy(copy=True) if (s_col in df.columns) else _zeros_yvalues(df.shape[0])
    if SN == 'S':
        yvalues = svalues.copy()
        is_enough_value = (yvalues > 0.01)
        yvalues[is_enough_value] = yvalues[is_enough_value] + nvalues[is_enough_value]
        return yvalues
    return nvalues


def _get_yvalues_sub_target(df, col, SN):
    if SN == 'S':
        is_S_cols = df.columns.str.startswith('S_sub_')
        S_cols = df.columns[is_S_cols]
        is_y_cols = is_S_cols | df.columns.str.startswith('N_sub_')
        y_cols = df.columns[is_y_cols]
        yvalues = df.loc[:, S_cols].sum(axis=1).to_numpy(copy=True)
        is_enough_value = (yvalues > 0.01)
        yvalues[is_enough_value] = df.loc[is_enough_value, y_cols].sum(axis=1).values
        return yvalues
    y_cols = df.columns[df.columns.str.startswith(col)]
    return df.loc[:, y_cols].sum(axis=1).values


def _get_yvalues_default(df, sub_type, col, SN):
    if SN == 'S':
        return df.loc[:, ['OCN' + sub_type, 'OCS' + sub_type]].sum(axis=1).values
    return df.loc[:, 'OC' + col].values


def get_yvalues(df, sub_type, SN):
    col = SN + sub_type
    if sub_type == '_sub':
        return _get_yvalues_sub(df=df, SN=SN)
    if sub_type == '_set_expr':
        return _get_yvalues_set_expr(df=df, SN=SN)
    if sub_type == '_set_other':
        return _get_yvalues_set_other(df=df, SN=SN)
    if sub_type.startswith('_sub_branch_'):
        return _get_yvalues_sub_branch(df=df, sub_type=sub_type, SN=SN)
    if sub_type == '_sub_':
        return _get_yvalues_sub_target(df=df, col=col, SN=SN)
    return _get_yvalues_default(df=df, sub_type=sub_type, col=col, SN=SN)


def _add_lineage_distance_colorbar(fig, g):
    branch_ids = _normalize_branch_ids(g.get('branch_ids', [])).tolist()
    if len(branch_ids)==0:
        return None
    mid_dists,is_actual = _get_lineage_midpoint_distances(branch_ids=branch_ids, g=g)
    if len(mid_dists)==0:
        return None
    vmin = float(min(mid_dists))
    vmax = float(max(mid_dists))
    if abs(vmax - vmin) <= 1e-12:
        vmax = vmin + 1.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'lineage_oldness',
        [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
        N=256,
    )
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cax = fig.add_axes([0.18, 0.02, 0.64, 0.03])
    cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    if is_actual:
        cbar_label = 'Branch distance from ancestor (branch-length units)'
    else:
        cbar_label = 'Branch distance from ancestor'
    cbar.set_label(cbar_label, fontsize=font_size)
    if len(mid_dists)==1:
        ticks = [float(mid_dists[0])]
    else:
        ticks = [vmin, (vmin+vmax)*0.5, vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(['{:.3g}'.format(tick) for tick in ticks])
    cbar.ax.tick_params(labelsize=font_size)
    return None


def add_substitution_labels(df, SN, sub_type, SN_colors, ax, g):
    col = 'OC'+ SN + sub_type
    df_sub = df.loc[(df[col] >= get_min_combinat_prob(g)), :].reset_index()
    if df_sub.shape[0] == 0:
        return ax
    g = get_highest_identity_chain_name(g)
    chain_name = g.get('highest_identity_chain_name', None)
    if not chain_name:
        print('Skipping substitution labels because no chain identity information was available.', flush=True)
        return ax
    chain_col = 'codon_site_pdb_' + chain_name
    if chain_col not in df_sub.columns:
        print('Skipping substitution labels because "{}" was not found.'.format(chain_col), flush=True)
        return ax
    anc_cols = df_sub.columns[df_sub.columns.str.startswith('aa_')&df_sub.columns.str.endswith('_anc')]
    des_cols = anc_cols.str.replace('_anc', '')
    x_min_dist = (df_sub.loc[:,'codon_site_alignment'].max()+1) / 35
    x_offset = (df_sub.loc[:,'codon_site_alignment'].max()+1) / 300
    for i in df_sub.index:
        x_value = df_sub.at[i,'codon_site_alignment']
        chain_site = df_sub.at[i,chain_col]
        anc_state = '/'.join(df_sub.loc[i,anc_cols].unique())
        des_state = '/'.join(df_sub.loc[i,des_cols].unique())
        sub_text = anc_state+str(chain_site)+des_state
        ha = 'right'
        x_value2 = x_value
        if (i != 0):
            if ((x_value - df_sub.at[i-1,'codon_site_alignment'])<x_min_dist):
                x_value2 = x_value + x_offset
                ha = 'left'
        ax.text(x=x_value2, y=0.98, s=sub_text, color=SN_colors[SN], fontsize=8, rotation='vertical', ha=ha, va='top')
    return ax


def _get_base_sub_types_and_colors():
    sub_types = {
        '_sub': 'Branch-wise\nsubstitutions\nin the entire tree',
        '_sub_': 'Branch-wise\nsubstitutions\nin the targets',
    }
    SN_color_all = {
        '_sub': {'N': 'black', 'S': 'gainsboro'},
        '_sub_': {'N': 'black', 'S': 'gainsboro'},
    }
    return sub_types, SN_color_all


def _get_branch_sub_type_key(branch_id):
    return '_sub_branch_{}'.format(int(branch_id))


def _add_branch_sub_types(sub_types, SN_color_all, branch_ids, color_by_branch):
    for branch_id in _normalize_branch_ids(branch_ids).tolist():
        key = _get_branch_sub_type_key(branch_id)
        sub_types[key] = 'Substitutions in\nbranch_id {}'.format(int(branch_id))
        SN_color_all[key] = {'N': color_by_branch[int(branch_id)], 'S': 'gainsboro'}
    return sub_types, SN_color_all


def get_plot_sub_types_and_colors(g):
    mode = str(g.get('mode', 'intersection')).lower()
    if mode == 'lineage':
        sub_types, SN_color_all = _get_base_sub_types_and_colors()
        branch_ids = _normalize_branch_ids(g.get('branch_ids', []))
        branch_rgb = _get_lineage_rgb_by_branch(branch_ids=branch_ids.tolist(), g=g)
        sub_types, SN_color_all = _add_branch_sub_types(
            sub_types=sub_types,
            SN_color_all=SN_color_all,
            branch_ids=branch_ids,
            color_by_branch=branch_rgb,
        )
    elif mode == 'set':
        sub_types, SN_color_all = _get_base_sub_types_and_colors()
        tokens = _tokenize_set_expression(g.get('mode_expression', ''))
        branch_ids = _get_set_expression_display_branch_ids(g)
        branch_black = {int(bid): 'black' for bid in branch_ids.tolist()}
        sub_types, SN_color_all = _add_branch_sub_types(
            sub_types=sub_types,
            SN_color_all=SN_color_all,
            branch_ids=branch_ids,
            color_by_branch=branch_black,
        )
        if 'A' in tokens:
            sub_types['_set_other'] = 'Substitutions in\nA'
        mode_expression = str(g.get('mode_expression', '')).strip()
        if mode_expression == '':
            mode_expression = 'set expression'
        sub_types['_set_expr'] = 'Substitutions in\n{}'.format(mode_expression)
        SN_color_all['_set_other'] = {'N': 'black', 'S': 'gainsboro'}
        SN_color_all['_set_expr'] = {'N': 'red', 'S': 'gainsboro'}
    elif g['single_branch_mode']:
        sub_types = {
            '_sub':'Branch-wise\nsubstitutions\nin the entire tree',
            'any2any':'Branch-wise\nsubstitutions\nin the targets', # Identical to branch-wise substitutions in the targets
        }
        SN_color_all = {
            '_sub': {'N':'black', 'S':'gainsboro'},
            'any2any': {'N':'purple', 'S':'gainsboro'}, # Identical to branch-wise substitutions in the targets
        }
    else:
        sub_types = {
            '_sub':'Branch-wise\nsubstitutions\nin the entire tree',
            '_sub_':'Branch-wise\nsubstitutions\nin the targets',
            'any2spe':'Posterior prob.\nof any2spe',
            'any2dif':'Posterior prob.\nof any2dif',
        }
        SN_color_all = {
            '_sub': {'N':'black', 'S':'gainsboro'},
            '_sub_': {'N':'black', 'S':'gainsboro'},
            'any2spe': {'N':'red', 'S':'gainsboro'},
            'any2dif': {'N':'blue', 'S':'gainsboro'},
        }
    return sub_types,SN_color_all


def _configure_barchart_axis(ax, df, sub_type, g, NS_ymax):
    enable_substitution_labels = False
    if sub_type == '_sub':
        ax.set_ylim(0, NS_ymax)
        add_gapline(df=df, gapcol='gap_rate_all', xcol='codon_site_alignment', yvalue=NS_ymax * 0.95, lw=3, ax=ax)
    elif sub_type == '_sub_':
        ymax = df.columns.str.startswith('N_sub_').sum()
        ax.set_ylim(0, ymax)
        add_gapline(df=df, gapcol='gap_rate_target', xcol='codon_site_alignment', yvalue=ymax * 0.95, lw=3, ax=ax)
    elif sub_type.startswith('_sub_branch_'):
        ax.set_ylim(0, 1.0)
        add_gapline(df=df, gapcol='gap_rate_target', xcol='codon_site_alignment', yvalue=0.95, lw=3, ax=ax)
    elif sub_type == '_set_expr':
        ymax = max(float(df.loc[:, 'N_set_expr_prob'].max()) if ('N_set_expr_prob' in df.columns) else 1.0, 1.0)
        ax.set_ylim(0, ymax)
        add_gapline(df=df, gapcol='gap_rate_target', xcol='codon_site_alignment', yvalue=ymax * 0.95, lw=3, ax=ax)
    elif sub_type == '_set_other':
        ax.set_ylim(0, 1.0)
        add_gapline(df=df, gapcol='gap_rate_all', xcol='codon_site_alignment', yvalue=0.95, lw=3, ax=ax)
    else:
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, linestyle='--', linewidth=0.5, color='black', zorder=0)
        add_gapline(df=df, gapcol='gap_rate_target', xcol='codon_site_alignment', yvalue=0.95, lw=3, ax=ax)
        enable_substitution_labels = True
    return enable_substitution_labels


def _draw_barchart_series(ax, df, sub_type, SN, SN_colors, ylabel, g, NS_ymax, is_last_row):
    yvalues = get_yvalues(df, sub_type, SN)
    enable_substitution_labels = _configure_barchart_axis(
        ax=ax,
        df=df,
        sub_type=sub_type,
        g=g,
        NS_ymax=NS_ymax,
    )
    if enable_substitution_labels and (SN == 'N') and (g['pdb'] is not None):
        ax = add_substitution_labels(df, SN, sub_type, SN_colors, ax, g)
    ax.set_ylabel(ylabel, fontsize=font_size)
    xy = pd.DataFrame({'x': df.loc[:, 'codon_site_alignment'].values, 'y': yvalues})
    xy2 = xy.loc[(xy['y'] > 0.01), :]
    ax.bar(xy2['x'], xy2['y'], color=SN_colors[SN])
    if is_last_row:
        ax.set_xlabel('Aligned codon site', fontsize=font_size)
    else:
        ax.set_xlabel('', fontsize=font_size)
    ax.set_xlim(df.loc[:, 'codon_site_alignment'].min() - 0.5, df.loc[:, 'codon_site_alignment'].max() + 0.5)
    return ax


def _draw_barchart_row(ax, df, sub_type, SN_colors, ylabel, g, NS_ymax, is_last_row):
    for SN in ['S', 'N']:
        ax = _draw_barchart_series(
            ax=ax,
            df=df,
            sub_type=sub_type,
            SN=SN,
            SN_colors=SN_colors,
            ylabel=ylabel,
            g=g,
            NS_ymax=NS_ymax,
            is_last_row=is_last_row,
        )
    return ax


def _resolve_barchart_output_base(g):
    if g['pdb'] is None:
        return os.path.join(g['site_outdir'], _site_output_prefix(g))
    return g['pdb_outfile_base']


def _save_barchart_figure(fig, outbase):
    out_path = outbase + ".pdf"
    fig.savefig(out_path, format='pdf', transparent=True)
    #fig.savefig(outbase+".svg", format='svg', transparent=True)
    print("Nonsynonymous and synonymous substitutions are shown in color and gray, respectively.", flush=True)
    print("Alignment gap sites are indicated by gray scale (0% missing = white, 100% missing = black).", flush=True)
    return out_path


def _apply_barchart_layout(fig, g):
    if str(g.get('mode', '')).lower() == 'lineage':
        fig.tight_layout(h_pad=0.5, w_pad=1, rect=[0, 0.09, 1, 1])
        _add_lineage_distance_colorbar(fig=fig, g=g)
        return fig
    fig.tight_layout(h_pad=0.5, w_pad=1)
    return fig


def _create_barchart_figure(num_row):
    fig, axes = plt.subplots(
        nrows=num_row,
        ncols=1,
        figsize=(7.2, 1.2 * num_row),
        sharex=True,
    )
    return fig, np.atleast_1d(axes).reshape(-1)


def _draw_all_barchart_rows(df, g, axes, sub_types, SN_color_all, NS_ymax):
    num_row = len(sub_types)
    for i, (sub_type, ylabel) in enumerate(sub_types.items()):
        _draw_barchart_row(
            ax=axes[i],
            df=df,
            sub_type=sub_type,
            SN_colors=SN_color_all[sub_type],
            ylabel=ylabel,
            g=g,
            NS_ymax=NS_ymax,
            is_last_row=(i == num_row - 1),
        )
    return axes


def plot_barchart(df, g):
    sub_types, SN_color_all = get_plot_sub_types_and_colors(g)
    num_row = len(sub_types)
    fig, axes = _create_barchart_figure(num_row=num_row)
    NS_ymax = df.loc[:, ['N_sub', 'S_sub']].sum(axis=1).max() + 0.5
    _draw_all_barchart_rows(
        df=df,
        g=g,
        axes=axes,
        sub_types=sub_types,
        SN_color_all=SN_color_all,
        NS_ymax=NS_ymax,
    )
    fig = _apply_barchart_layout(fig=fig, g=g)
    outbase = _resolve_barchart_output_base(g)
    return _save_barchart_figure(fig=fig, outbase=outbase)


def plot_lineage_tree(g, outbase):
    if str(g.get('mode', '')).lower() != 'lineage':
        return None
    branch_ids = _normalize_branch_ids(g.get('branch_ids', []))
    if branch_ids.shape[0]==0:
        return None
    branch_rgb = _get_lineage_rgb_by_branch(branch_ids=branch_ids.tolist(), g=g)
    for node in g['tree'].traverse():
        bid = int(ete.get_prop(node, "numerical_label"))
        node_color = branch_rgb.get(bid, 'black')
        # For lineage tree output, color labels by branch color to match the bar/PSE palette.
        ete.set_prop(node, 'color_PLACEHOLDER', node_color)
        ete.set_prop(node, 'labelcolor_PLACEHOLDER', node_color)
    plot_g = {
        'tree': g['tree'],
        'fg_df': pd.DataFrame(columns=['name', 'PLACEHOLDER']),
    }
    tree.plot_branch_category(g=plot_g, file_base=outbase+'.tree', label='all')
    return None


def get_gapsite_rate(state_tensor):
    if state_tensor.shape[0] == 0:
        return np.zeros(shape=(state_tensor.shape[1],), dtype=float)
    num_gapsite = (state_tensor.sum(axis=2)==0).sum(axis=0)
    gapsite_rate = num_gapsite / state_tensor.shape[0]
    return gapsite_rate


def extend_site_index_edge(sites, num_extend):
    new_sites = sites.copy()
    to_append_base = pd.Series(-1 - np.arange(num_extend))
    for i in sites.index[1:]:
        if sites.loc[i]-1 == sites.loc[i-1]:
            continue
        to_append = to_append_base + sites.loc[i]
        new_sites = pd.concat([new_sites, to_append], ignore_index=True)
    new_sites = new_sites.loc[new_sites>=0]
    new_sites = new_sites.drop_duplicates().sort_values().reset_index(drop=True)
    return new_sites


def _resolve_window_sizes(num_gene_site, num_site):
    window_sizes = [100, 50, 10, 5, 4, 3, 2, 1]
    return [w for w in window_sizes if (w < num_gene_site) & (w < num_site)]


def _build_codon_first_state_index(seq, num_gene_site, codon_orders):
    codon_first_index = np.full(shape=(num_gene_site,), fill_value=-1, dtype=np.int64)
    for site in range(num_gene_site):
        codon = seq[(site * 3):((site + 1) * 3)]
        codon_index = sequence.get_state_index(codon, codon_orders, genetic_code.ambiguous_table)
        if len(codon_index) > 0:
            codon_first_index[site] = int(codon_index[0])
    return codon_first_index


def _get_unassigned_window_context(assigned_gene_index, gene_sites, window_size):
    unassigned_aln_sites = np.where(assigned_gene_index == -1)[0].astype(np.int64, copy=False)
    assigned_gene_sites = assigned_gene_index[assigned_gene_index != -1]
    unassigned_gene_sites = np.setdiff1d(gene_sites, assigned_gene_sites, assume_unique=False)
    unassigned_gene_sites = pd.Series(unassigned_gene_sites)
    extended_unassigned_gene_sites = extend_site_index_edge(unassigned_gene_sites, window_size).to_numpy(
        dtype=np.int64,
        copy=False,
    )
    return unassigned_aln_sites, extended_unassigned_gene_sites


def _is_window_state_match(leaf_state_cdn, codon_first_index, uas, ugs, window_size, row_index_cache):
    codon_index_window = codon_first_index[ugs:(ugs + window_size)]
    if (codon_index_window < 0).any():
        # codon may be a stop.
        return False
    row_index = row_index_cache.get(window_size, None)
    if row_index is None:
        row_index = np.arange(window_size, dtype=np.int64)
        row_index_cache[window_size] = row_index
    leaf_window = leaf_state_cdn[uas:(uas + window_size), :]
    return bool((leaf_window[row_index, codon_index_window] != 0).all())


def _has_smaller_following_gene_index(assigned_gene_index, window_aln_end, window_gene_end):
    following_gene_index = assigned_gene_index[window_aln_end:]
    following_gene_index = following_gene_index[following_gene_index != -1]
    if following_gene_index.shape[0] == 0:
        return False
    return bool(following_gene_index.min() < window_gene_end)


def _assign_matching_windows_for_size(
    assigned_gene_index,
    leaf_state_cdn,
    codon_first_index,
    num_site,
    num_gene_site,
    window_size,
    gene_sites,
    row_index_cache,
):
    step_size = max([int(window_size / 5), 1])
    unassigned_aln_sites, extended_unassigned_gene_sites = _get_unassigned_window_context(
        assigned_gene_index=assigned_gene_index,
        gene_sites=gene_sites,
        window_size=window_size,
    )
    txt = 'Window size = {:,}, Number of unassigned alignment site = {:,}'
    print(txt.format(window_size, unassigned_aln_sites.shape[0]), flush=True)
    for k, uas in enumerate(unassigned_aln_sites):
        if (k != 0) and (uas < unassigned_aln_sites[k - 1] + step_size):
            continue
        if (uas + window_size > num_site):
            break
        for ugs in extended_unassigned_gene_sites:
            if (ugs + window_size > num_gene_site):
                break
            if not _is_window_state_match(
                leaf_state_cdn=leaf_state_cdn,
                codon_first_index=codon_first_index,
                uas=uas,
                ugs=ugs,
                window_size=window_size,
                row_index_cache=row_index_cache,
            ):
                continue
            window_aln_end = uas + window_size - 1
            window_gene_end = ugs + window_size - 1
            if _has_smaller_following_gene_index(
                assigned_gene_index=assigned_gene_index,
                window_aln_end=window_aln_end,
                window_gene_end=window_gene_end,
            ):
                continue
            assigned_gene_index[uas:(uas + window_size)] = np.arange(ugs, ugs + window_size, dtype=np.int64)
            break
    return assigned_gene_index


def _report_gene_assignment_summary(assigned_gene_index, aln_sites, has_gene_site_in_aln_value):
    num_gene_site_in_aln = has_gene_site_in_aln_value.sum()
    is_unassigned = (assigned_gene_index == -1)
    txt = 'End. Unassigned alignment site = {:,}, Assigned alignment site = {:,}, '
    txt += 'Alignment site with non-missing gene states: {:,}'
    print(txt.format(is_unassigned.sum(), (~is_unassigned).sum(), num_gene_site_in_aln), flush=True)
    if (~is_unassigned).sum() == num_gene_site_in_aln:
        return
    gene_site_in_aln = set(aln_sites[has_gene_site_in_aln_value])
    gene_site_assigned = set(aln_sites[~is_unassigned])
    only_in_aln = sorted(list(gene_site_in_aln - gene_site_assigned))
    only_in_assigned = sorted(list(gene_site_assigned - gene_site_in_aln))
    txt_base = 'Sites only present in '
    print(txt_base + 'input alignment: {}'.format(','.join([str(v) for v in only_in_aln])), flush=True)
    print(txt_base + 'untrimmed CDS: {}'.format(','.join([str(v) for v in only_in_assigned])), flush=True)


def _build_aln_gene_match_for_leaf(leaf, seq, num_site, g):
    leaf_nn = ete.get_prop(leaf, "numerical_label")
    leaf_state_cdn = g['state_cdn'][leaf_nn, :, :]
    seq = str(seq).replace('-', '').upper()
    if (len(seq) % 3) != 0:
        txt = 'Untrimmed CDS sequence length for "{}" should be multiple of 3 (length={}).'
        raise ValueError(txt.format(leaf.name, len(seq)))
    num_gene_site = int(len(seq) / 3)
    gene_sites = np.arange(num_gene_site, dtype=np.int64)
    aln_sites = np.arange(num_site, dtype=np.int64)
    col_leaf = 'codon_site_' + leaf.name
    assigned_gene_index = np.full(shape=(num_site,), fill_value=-1, dtype=np.int64)
    codon_first_index = _build_codon_first_state_index(
        seq=seq,
        num_gene_site=num_gene_site,
        codon_orders=g['codon_orders'],
    )
    row_index_cache = dict()
    window_sizes = _resolve_window_sizes(num_gene_site=num_gene_site, num_site=num_site)
    for window_size in window_sizes:
        assigned_gene_index = _assign_matching_windows_for_size(
            assigned_gene_index=assigned_gene_index,
            leaf_state_cdn=leaf_state_cdn,
            codon_first_index=codon_first_index,
            num_site=num_site,
            num_gene_site=num_gene_site,
            window_size=window_size,
            gene_sites=gene_sites,
            row_index_cache=row_index_cache,
        )
    has_gene_site_in_aln_value = (leaf_state_cdn.sum(axis=1) > 0)
    _report_gene_assignment_summary(
        assigned_gene_index=assigned_gene_index,
        aln_sites=aln_sites,
        has_gene_site_in_aln_value=has_gene_site_in_aln_value,
    )
    aln_gene_match = pd.DataFrame({
        'codon_site_alignment': aln_sites,
        col_leaf: assigned_gene_index,
    })
    return aln_gene_match


def add_gene_index(df, g):
    seqs = sequence.read_fasta(path=g['untrimmed_cds'])
    num_site = g['state_cdn'].shape[1]
    for leaf in ete.iter_leaves(g['tree']):
        if leaf.name not in seqs:
            continue
        print('Matching untrimmed CDS sequence: {}'.format(leaf.name), flush=True)
        aln_gene_match = _build_aln_gene_match_for_leaf(
            leaf=leaf,
            seq=seqs[leaf.name],
            num_site=num_site,
            g=g,
        )
        df = pd.merge(df, aln_gene_match, on='codon_site_alignment', how='left')
        print('', flush=True)
    return df


def write_fasta(file, label, seq):
    with open(file, 'w') as f:
        f.write('>'+label+'\n')
        f.write(seq+'\n')


def translate(seq, g):
    if (len(seq) % 3) != 0:
        txt = 'Input CDS sequence length should be multiple of 3 for translation (length={}).'
        raise ValueError(txt.format(len(seq)))
    translated_seq = ''
    num_site = int(len(seq)/3)
    codon_to_aa = dict()
    for aa, codons in g['matrix_groups'].items():
        for codon in codons:
            codon_to_aa[str(codon).upper()] = aa
    for s in np.arange(num_site):
        codon = seq[(s*3):((s+1)*3)].upper()
        aa = codon_to_aa.get(codon, None)
        if aa is None:
            txt = 'Unknown codon "{}" was found at codon site {} during translation.'
            raise ValueError(txt.format(codon, s + 1))
        translated_seq += aa
    return translated_seq


def _resolve_chimera_line_for_site(df, codon_site_col, seq_site):
    is_site = (df.loc[:, codon_site_col] == seq_site)
    if is_site.sum() == 0:
        return '\t:{}\t{}\n'.format(seq_site, 'None')
    if 'OCNany2spe' in df.columns:
        Nany2spe = float(df.loc[is_site, 'OCNany2spe'].fillna(0).values[0])
    else:
        Nany2spe = 0.0
    if 'OCNany2dif' in df.columns:
        Nany2dif = float(df.loc[is_site, 'OCNany2dif'].fillna(0).values[0])
    else:
        Nany2dif = 0.0
    Nvalue = Nany2spe if (Nany2spe >= Nany2dif) else -Nany2dif
    return '\t:{}\t{:.4f}\n'.format(seq_site, Nvalue)


def _write_chimera_attribute_file(file_name, seq_sites, df, codon_site_col, header):
    with open(file_name, 'w') as f:
        f.write(header)
        for seq_site in seq_sites:
            line = _resolve_chimera_line_for_site(df=df, codon_site_col=codon_site_col, seq_site=seq_site)
            f.write(line)


def _write_chimera_fasta_for_seq(seq_key, seq, g):
    translated_seq = translate(seq, g)
    file_fasta = os.path.join(g['site_outdir'], _site_output_prefix(g) + '_' + seq_key + '.fasta')
    txt = "Writing amino acid fasta that may be used as a query for homology modeling " \
          "to obtain .pdb file (e.g., with SWISS-MODEL): {}"
    print(txt.format(file_fasta))
    write_fasta(file=file_fasta, label=seq_key, seq=translated_seq)


def export2chimera(df, g):
    header='attribute: condivPP\nmatch mode: 1-to-1\nrecipient: residues\nnone handling: None\n'
    seqs = sequence.read_fasta(path=g['untrimmed_cds'])
    for seq_key in seqs.keys():
        codon_site_col = 'codon_site_' + seq_key
        if codon_site_col not in df.columns:
            print('Sequence not be found in csubst inputs. Skipping: {}'.format(seq_key))
            continue
        seq = seqs[seq_key]
        if (len(seq) % 3) != 0:
            txt = 'Untrimmed CDS sequence length for "{}" should be multiple of 3 for Chimera export (length={}).'
            raise ValueError(txt.format(seq_key, len(seq)))
        seq_num_site = len(seq) // 3
        seq_sites = np.arange(1, seq_num_site + 1)
        file_name = os.path.join(g['site_outdir'], _site_output_prefix(g) + '_' + seq_key + '.chimera.txt')
        txt = 'Writing a file that can be loaded to UCSF Chimera from ' \
              '"Tools -> Structure Analysis -> Define Attribute"'
        print(txt.format(file_name))
        _write_chimera_attribute_file(
            file_name=file_name,
            seq_sites=seq_sites,
            df=df,
            codon_site_col=codon_site_col,
            header=header,
        )
        _write_chimera_fasta_for_seq(seq_key=seq_key, seq=seq, g=g)


def get_parent_branch_ids(branch_ids, g):
    state_ref = g.get('state_cdn', g.get('state_pep', g.get('state_nsy', None)))
    if state_ref is None:
        state_has_mass = None
    else:
        state_has_mass = (state_ref.sum(axis=(1, 2)) > float(g.get('float_tol', 0)))
    parent_branch_ids = dict()
    for node in g['tree'].traverse():
        if ete.get_prop(node, "numerical_label") in branch_ids:
            parent_node = ete.get_effective_state_parent(node, state_has_mass=state_has_mass)
            if parent_node is None:
                continue
            parent_branch_ids[ete.get_prop(node, "numerical_label")] = ete.get_prop(parent_node, "numerical_label")
    return parent_branch_ids


def add_states(df, branch_ids, g, add_hydrophobicity=True):
    parent_branch_ids = get_parent_branch_ids(branch_ids, g)
    seqtypes = ['cdn','pep']
    seqtypes2 = ['cdn','aa']
    order_keys = ['codon_orders','amino_acid_orders']
    for seqtype,seqtype2,order_key in zip(seqtypes,seqtypes2,order_keys):
        for bid in branch_ids:
            col = seqtype2+'_'+str(bid)
            df.loc[:,col] = ''
            for i in df.index:
                states = g['state_'+seqtype][bid,i,:]
                if not states.max()==0:
                    ml_state = g[order_key][states.argmax()]
                    df.at[i,col] = ml_state
        for bid in branch_ids:
            anc_col = seqtype2+'_'+str(bid)+'_anc'
            df.loc[:,anc_col] = ''
            parent_bid = parent_branch_ids.get(int(bid), None)
            if parent_bid is None:
                continue
            parent_bid = int(parent_bid)
            if (parent_bid < 0) or (parent_bid >= g['state_'+seqtype].shape[0]):
                continue
            for i in df.index:
                anc_states = g['state_'+seqtype][parent_bid,i,:]
                if not anc_states.max()==0:
                    ml_anc_state = g[order_key][anc_states.argmax()]
                    df.at[i,anc_col] = ml_anc_state
    if add_hydrophobicity:
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0080635
        aa_hydrophobicity_empirical = {
            'A':129.0, 'R':274.0, 'N':195.0, 'D':193.0, 'C':167.0,
            'E':223.0, 'Q':225.0, 'G':104.0, 'H':224.0, 'I':197.0,
            'L':201.0, 'K':236.0, 'M':224.0, 'F':240.0, 'P':159.0,
            'S':155.0, 'T':172.0, 'W':285.0, 'Y':263.0, 'V':174.0,
            '':np.nan,
        }
        df_aa_hydrophobicity_empirical = pd.DataFrame({
            'aa':aa_hydrophobicity_empirical.keys(),
            'hydrophobicity': aa_hydrophobicity_empirical.values(),
        })
        aa_cols = df.columns[df.columns.str.startswith('aa_')]
        for aa_col in aa_cols:
            hp_col = aa_col+'_'+'hydrophobicity'
            df_aa_hydrophobicity_empirical.columns = [aa_col, hp_col]
            df = pd.merge(df, df_aa_hydrophobicity_empirical, on=aa_col, how='left', sort=False)
        print('')
    return df


def get_state_orders(g, mode):
    if mode=='nsy':
        state_orders = {'nsy':sequence.get_nonsyn_state_orders(g)}
    elif mode=='syn':
        state_orders = g['matrix_groups']
    else:
        raise ValueError('Unsupported state order mode: {}'.format(mode))
    state_keys = list(state_orders.keys())
    return state_orders,state_keys


def validate_state_orders_against_sub_tensor(state_orders, state_keys, sub_tensor, mode):
    group_axis = int(sub_tensor.shape[2])
    state_axis = int(sub_tensor.shape[3])
    if len(state_keys) != group_axis:
        txt = 'State-order group count for mode "{}" ({}) does not match substitution tensor group axis ({}).'
        raise ValueError(txt.format(mode, len(state_keys), group_axis))
    for state_key in state_keys:
        num_state = len(state_orders[state_key])
        if num_state > state_axis:
            txt = 'State-order size for mode "{}" group "{}" ({}) exceeds substitution tensor state axis ({}).'
            raise ValueError(txt.format(mode, state_key, num_state, state_axis))
        if (mode == 'nsy') and (num_state != state_axis):
            txt = 'Nonsynonymous state-order size ({}) does not match substitution tensor state axis ({}).'
            raise ValueError(txt.format(num_state, state_axis))
    return None


def get_df_ad(sub_tensor, g, mode):
    state_orders,state_keys = get_state_orders(g, mode)
    validate_state_orders_against_sub_tensor(state_orders, state_keys, sub_tensor, mode)
    gad, _, _ = substitution.get_group_state_totals(sub_tensor=sub_tensor)
    cols = ['group','state_from','state_to','value']
    nrow = sum([ len(v)**2-len(v) for v in state_orders.values() ])
    df_ad = pd.DataFrame(np.zeros(shape=(nrow, len(cols))))
    df_ad.columns = cols
    df_ad['group'] = df_ad['group'].astype('str')
    df_ad['state_from'] = df_ad['state_from'].astype('str')
    df_ad['state_to'] = df_ad['state_to'].astype('str')
    current_row = 0
    for g in np.arange(gad.shape[0]):
        state_key = state_keys[g]
        for i1,state1 in enumerate(state_orders[state_key]):
            for i2,state2 in enumerate(state_orders[state_key]):
                if (i1==i2):
                    continue
                total_prob = gad[g,i1,i2]
                if (np.isnan(total_prob)):
                    txt = 'Total probability should not be NaN: {}-to-{} substitutions\n'
                    sys.stderr.write(txt.format(state1, state2))
                df_ad.loc[current_row,:] = [state_key, state1, state2, total_prob]
                current_row += 1
    return df_ad


def add_site_stats(df_ad, sub_tensor, g, mode, method='tau'):
    # method = {'tau', 'hg', 'tsi'}
    # https://academic.oup.com/bib/article/18/2/205/2562739
    state_orders,state_keys = get_state_orders(g, mode)
    validate_state_orders_against_sub_tensor(state_orders, state_keys, sub_tensor, mode)
    outcol = 'site_'+method
    df_ad.loc[:,outcol] = np.nan
    sgad = substitution.get_site_group_state_totals(sub_tensor=sub_tensor)
    current_row = 0
    for g in np.arange(sgad.shape[1]):
        state_key = state_keys[g]
        for i1,state1 in enumerate(state_orders[state_key]):
            for i2,state2 in enumerate(state_orders[state_key]):
                if (i1==i2):
                    continue
                x_values = sgad[:,g,i1,i2]
                if (x_values.sum()==0):
                    current_row += 1
                    continue
                if (method=='tau'):
                    if x_values.shape[0] <= 1:
                        value = 0.0
                    else:
                        x_max = x_values.max()
                        x_hat = x_values / x_max
                        value = (1-x_hat).sum() / (x_values.shape[0] - 1)
                elif (method=='hg'):
                    pi = x_values / x_values.sum()
                    pi = pi[pi > 0]
                    value = - (pi * np.log2(pi)).sum() if pi.shape[0] > 0 else 0.0
                elif (method=='tsi'):
                    value = x_values.max() / x_values.sum()
                elif (method.startswith('rank')):
                    rank_no = int(method.replace('rank', ''))
                    temp = x_values.argsort()
                    ranks = np.empty_like(temp)
                    ranks[temp] = np.arange(len(x_values))
                    ranks = np.abs(ranks - ranks.max())+1
                    rank_values = x_values[ranks==rank_no]
                    if rank_values.shape[0] == 0:
                        value = 0.0
                    else:
                        value = float(rank_values[0])
                df_ad.loc[current_row,outcol] = value
                current_row += 1
    return df_ad


def add_has_target_high_combinat_prob_site(df_ad, sub_tensor, g, mode):
    state_orders,state_keys = get_state_orders(g, mode)
    validate_state_orders_against_sub_tensor(state_orders, state_keys, sub_tensor, mode)
    outcol = 'has_target_high_combinat_prob_site'
    df_ad.loc[:,outcol] = False
    sgad = substitution.get_site_group_state_totals(sub_tensor=sub_tensor)
    min_prob = get_min_combinat_prob(g)
    current_row = 0
    for g in np.arange(sgad.shape[1]):
        state_key = state_keys[g]
        for i1,state1 in enumerate(state_orders[state_key]):
            for i2,state2 in enumerate(state_orders[state_key]):
                if (i1==i2):
                    continue
                x_values = sgad[:,g,i1,i2]
                if (x_values >= min_prob).any():
                    df_ad.at[current_row,outcol] = True
                current_row += 1
    return df_ad


def get_df_dist(sub_tensor, g, mode):
    tree_dict = dict()
    for node in g['tree'].traverse():
        tree_dict[ete.get_prop(node, "numerical_label")] = node
    state_orders, state_keys = get_state_orders(g, mode)
    validate_state_orders_against_sub_tensor(state_orders, state_keys, sub_tensor, mode)
    cols = ['group','state_from','state_to','max_dist_bl']
    inds = np.arange(np.array(sub_tensor.shape[2:]).prod()-sub_tensor.shape[4])
    df_dist = pd.DataFrame(columns=cols, index=inds)
    bgad = substitution.get_branch_group_state_totals(sub_tensor=sub_tensor)
    b_index = np.arange(bgad.shape[0])
    g_index = np.arange(bgad.shape[1])
    a_index = np.arange(bgad.shape[2])
    d_index = np.arange(bgad.shape[3])
    current_row = 0
    for g,a,d in itertools.product(g_index, a_index, d_index):
        if (a==d):
            continue
        state_key = state_keys[g]
        if (len(state_orders[state_key])<(a+1))|(len(state_orders[state_key])<(d+1)):
            continue
        state_from = state_orders[state_key][a]
        state_to = state_orders[state_key][d]
        has_enough_sub = (bgad[:,g,a,d] >= 0.5)
        branch_ids = b_index[has_enough_sub]
        if branch_ids.shape[0]==0:
            interbranch_dist = np.nan
        elif branch_ids.shape[0]==1:
            interbranch_dist = np.nan
        elif branch_ids.shape[0]>=2:
            node_dists = list()
            nodes = [ tree_dict[n] for n in branch_ids ]
            for nds in list(itertools.combinations(nodes, 2)):
                node_dist = ete.get_distance(nds[0], nds[1], topology_only=False)
                node_dists.append(node_dist - nds[1].dist)
            interbranch_dist = max(node_dists) # Maximum value among pairwise distances
        df_dist.loc[current_row, :] = [state_key, state_from, state_to, interbranch_dist]
        current_row += 1
    df_dist = df_dist.loc[~df_dist['group'].isnull(),:]
    return df_dist


def plot_state(ON_tensor, OS_tensor, branch_ids, g):
    if not bool(g.get('site_state_plot', True)):
        print('Skipping substitution-pattern summary outputs (--site_state_plot no).', flush=True)
        return []
    fig,axes = plt.subplots(nrows=3, ncols=2, figsize=(7.2, 7.2), sharex=False)
    output_paths = list()
    output_prefix = _site_output_prefix(g)
    outfiles = [output_prefix + '.state_N.tsv', output_prefix + '.state_S.tsv']
    colors = ['red','blue']
    ax_cols = [0,1]
    titles = ['Nonsynonymous substitution','Synonymous substitution']
    iter_items = zip(ax_cols,['nsy','syn'],[ON_tensor,OS_tensor],outfiles,colors,titles)
    for ax_col,mode,sub_tensor,outfile,color,title in iter_items:
        if isinstance(sub_tensor, substitution_sparse.SparseSubstitutionTensor):
            sub_target = substitution.aggregate_sparse_branches(
                sub_tensor=sub_tensor,
                branch_ids=branch_ids,
                operation='sum',
            )
            sub_target_combinat = substitution.aggregate_sparse_branches(
                sub_tensor=sub_tensor,
                branch_ids=branch_ids,
                operation='product',
            )
        else:
            selected_tensor = substitution.get_branches_sub_tensor(sub_tensor=sub_tensor, branch_ids=branch_ids)
            sub_target = np.expand_dims(selected_tensor.sum(axis=0), axis=0)
            sub_target_combinat = np.expand_dims(selected_tensor.prod(axis=0), axis=0)
        df_ad = get_df_ad(sub_tensor=sub_tensor, g=g, mode=mode)
        df_ad_target = get_df_ad(sub_tensor=sub_target, g=g, mode=mode)
        df_ad_combinat = get_df_ad(sub_tensor=sub_target_combinat, g=g, mode=mode)
        df_ad.columns = df_ad.columns.str.replace('value', 'all')
        df_ad.loc[:,'target'] = df_ad_target.loc[:,'value']
        df_ad.loc[:,'target_combinat'] = df_ad_combinat.loc[:,'value']
        df_ad = add_has_target_high_combinat_prob_site(df_ad, sub_target_combinat, g, mode)
        df_ad = add_site_stats(df_ad=df_ad, sub_tensor=sub_tensor, g=g, mode=mode, method='tsi')
        df_ad = add_site_stats(df_ad=df_ad, sub_tensor=sub_tensor, g=g, mode=mode, method='rank1')
        df_ad = add_site_stats(df_ad=df_ad, sub_tensor=sub_tensor, g=g, mode=mode, method='rank2')
        df_ad = add_site_stats(df_ad=df_ad, sub_tensor=sub_tensor, g=g, mode=mode, method='rank3')
        df_ad = add_site_stats(df_ad=df_ad, sub_tensor=sub_tensor, g=g, mode=mode, method='rank4')
        df_ad = add_site_stats(df_ad=df_ad, sub_tensor=sub_tensor, g=g, mode=mode, method='rank5')
        df_dist = get_df_dist(sub_tensor=sub_tensor, g=g, mode=mode)
        df_ad = pd.merge(df_ad, df_dist, on=['group','state_from','state_to'])
        out_path = os.path.join(g['site_outdir'], outfile)
        tsv.write_dataframe(df_ad, out_path, float_format=g['float_format'], chunksize=10000)
        output_paths.append(out_path)
        df_ad.loc[:,'xlabel'] = df_ad.loc[:,'state_from'] + '->' + df_ad.loc[:,'state_to']
        ax = axes[0,ax_col]
        ax.bar(df_ad.loc[:,'xlabel'], df_ad.loc[:,'all'], color='black')
        ax.bar(df_ad.loc[:,'xlabel'], df_ad.loc[:,'target'], color=color)
        ax.get_xaxis().set_ticks([])
        ax.set_xlabel('Substitution category (e.g., {})'.format(df_ad.at[0,'xlabel']), fontsize=font_size)
        ax.set_ylabel('Total substitution\nprobabilities', fontsize=font_size)
        ax.set_title(title, fontsize=font_size)
        ax = axes[1,ax_col]
        bins = np.arange(21)/20
        ax.hist(x=df_ad.loc[:,'site_tsi'].dropna(), bins=bins, color='black')
        is_it = (df_ad.loc[:,'has_target_high_combinat_prob_site'])
        ax.hist(x=df_ad.loc[is_it,'site_tsi'].dropna(), bins=bins, color=color)
        ax.set_xlabel('Site specificity index', fontsize=font_size)
        ax.set_ylabel('Count of\nsubstitution categories', fontsize=font_size)
        ax = axes[2,ax_col]
        bins = np.arange(21) / 20 * df_dist.loc[:,'max_dist_bl'].max()
        ax.hist(x=df_dist.loc[:, 'max_dist_bl'].dropna(), bins=bins, color='black')
        #ax.hist(x=df_dist_target.loc[:, 'max_dist_bl'].dropna(), bins=bins, color=color)
        ax.set_xlabel('Max inter-branch distance of substitution category', fontsize=font_size)
        ax.set_ylabel('Count of\nsubstitution categories', fontsize=font_size)
    fig.tight_layout(h_pad=0.5, w_pad=1)
    outbase = os.path.join(g['site_outdir'], output_prefix + '.state')
    fig_path = outbase + ".pdf"
    fig.savefig(fig_path, format='pdf', transparent=True)
    plt.close(fig)
    output_paths.append(fig_path)
    return output_paths


def get_lineage_site_branch_ids(df, display_meta, g, min_prob):
    if str(g.get('mode', '')).lower() != 'lineage':
        return {}
    branch_ids = _normalize_branch_ids(g.get('branch_ids', [])).tolist()
    col_pairs = []
    for bid in branch_ids:
        col = 'N_sub_{}'.format(int(bid))
        if col in df.columns:
            col_pairs.append((int(bid), col))
    if len(col_pairs) == 0:
        return {}
    bids, cols = zip(*col_pairs)
    branch_values = df.loc[:, list(cols)].to_numpy(dtype=float, copy=True)
    branch_values = np.nan_to_num(branch_values, nan=0.0)
    site_to_row = {
        int(site): i for i,site in enumerate(df.loc[:, 'codon_site_alignment'].astype(int).tolist())
    }
    out = {}
    for item in display_meta:
        site = item.get('site', None)
        if site is None:
            continue
        site = int(site)
        row_index = site_to_row.get(site, None)
        if row_index is None:
            continue
        row_values = branch_values[row_index, :]
        selected = [int(bids[i]) for i,v in enumerate(row_values) if float(v) >= float(min_prob)]
        if len(selected) > 0:
            out[site] = selected
    return out


def get_set_expression_channel_indices(prob_matrix):
    prob_arr = np.nan_to_num(np.asarray(prob_matrix, dtype=float), nan=0.0)
    if prob_arr.ndim == 1:
        prob_arr = prob_arr[:, np.newaxis]
    if prob_arr.ndim != 2:
        raise ValueError('set expression probability matrix should be 1D or 2D.')
    indices = np.full(shape=(prob_arr.shape[0],), fill_value=-1, dtype=np.int64)
    for i,row in enumerate(prob_arr):
        if row.shape[0] == 0:
            continue
        max_value = float(np.max(row))
        if (not np.isfinite(max_value)) or (max_value <= 0):
            continue
        indices[i] = int(np.argmax(row))
    return indices


def get_set_expression_channel_labels(prob_matrix, set_stat_type, state_orders):
    channel_indices = get_set_expression_channel_indices(prob_matrix=prob_matrix)
    labels = np.array([''] * channel_indices.shape[0], dtype=object)
    for i,channel_index in enumerate(channel_indices.tolist()):
        if int(channel_index) < 0:
            continue
        labels[i] = _get_set_channel_label(
            set_stat_type=set_stat_type,
            channel_index=int(channel_index),
            state_orders=state_orders,
        )
    return labels


def add_lineage_site_tick_labels(ax_site, tick_positions, display_meta, site_branch_ids, branch_color_by_id):
    if len(tick_positions) == 0:
        return 0.0
    ax_site.set_xticklabels([''] * len(tick_positions))
    fig = ax_site.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_transform = matplotlib.transforms.blended_transform_factory(ax_site.transData, ax_site.transAxes)
    width_cache = {}
    x_nudge_pt = 0.0
    y_base_offset_pt = 2.0
    label_fontfamily = 'DejaVu Sans Mono'
    max_label_height_pt = 0.0

    def _text_width_pt(text):
        key = (text, font_size, label_fontfamily)
        if key in width_cache:
            return width_cache[key]
        text_artist = matplotlib.text.Text(
            x=0,
            y=0,
            text=text,
            fontsize=font_size,
            fontfamily=label_fontfamily,
        )
        text_artist.set_figure(fig)
        bbox = text_artist.get_window_extent(renderer=renderer)
        width_pt = float(bbox.width) * 72.0 / float(fig.dpi)
        width_cache[key] = width_pt
        return width_pt

    for col_index in tick_positions:
        site = display_meta[col_index].get('site', None)
        if site is None:
            continue
        site = int(site)
        branch_id_list = [int(bid) for bid in site_branch_ids.get(site, [])]
        segments = []
        segments.append(('{}: '.format(site), 'black'))
        for i,bid in enumerate(branch_id_list):
            if i > 0:
                segments.append((',', 'black'))
            segments.append((str(bid), branch_color_by_id.get(bid, 'black')))
        y_offset_pt = y_base_offset_pt
        for seg_text,seg_color in segments:
            ax_site.annotate(
                seg_text,
                xy=(col_index, 1.0),
                xycoords=text_transform,
                xytext=(x_nudge_pt, y_offset_pt),
                textcoords='offset points',
                ha='center',
                va='bottom',
                rotation=90,
                fontsize=font_size,
                fontfamily=label_fontfamily,
                color=seg_color,
                annotation_clip=False,
            )
            y_offset_pt += _text_width_pt(seg_text)
        if y_offset_pt > max_label_height_pt:
            max_label_height_pt = y_offset_pt
    return float(max_label_height_pt)


def initialize_site_df(num_site):
    if np.isscalar(num_site):
        num_site = int(num_site)
        codon_site_alignment = np.arange(num_site, dtype=np.int64)
    else:
        codon_site_alignment = np.asarray(num_site, dtype=np.int64).reshape(-1)
    df = pd.DataFrame()
    df.loc[:,'codon_site_alignment'] = codon_site_alignment
    df.loc[:,'nuc_site_alignment'] = ((df.loc[:,'codon_site_alignment']+1) * 3) - 2
    return df


def remap_codon_site_columns_to_alignment(df, g):
    state_tensor = g.get('state_cdn', None)
    expected_num_site = None if state_tensor is None else int(state_tensor.shape[1])
    site_index_alignment = parser_misc.get_site_index_alignment(g=g, expected_num_site=expected_num_site)
    codon_site_cols = [col for col in df.columns.tolist() if str(col).startswith('codon_site_')]
    if len(codon_site_cols) == 0:
        return df
    out = df.copy(deep=True)
    for col in codon_site_cols:
        values = pd.to_numeric(out.loc[:, col], errors='coerce').to_numpy(dtype=float, copy=True)
        mapped = np.full(shape=values.shape, fill_value=-1, dtype=np.int64)
        is_finite = np.isfinite(values)
        if is_finite.any():
            internal_sites = np.rint(values[is_finite]).astype(np.int64, copy=False)
            is_valid_internal = (internal_sites >= 0) & (internal_sites < site_index_alignment.shape[0])
            mapped_values = np.full(shape=internal_sites.shape, fill_value=-1, dtype=np.int64)
            mapped_values[is_valid_internal] = site_index_alignment[internal_sites[is_valid_internal]]
            mapped[is_finite] = mapped_values
        out.loc[:, col] = mapped
    return out


def add_cs_info(df, branch_ids, sub_tensor, attr):
    cs = substitution.get_cs(id_combinations=branch_ids[np.newaxis,:], sub_tensor=sub_tensor, attr=attr)
    cs.columns = cs.columns.str.replace('site','codon_site_alignment')
    df = pd.merge(df, cs, on='codon_site_alignment')
    df.loc[:,'OC'+attr+'any2dif'] = df.loc[:,'OC'+attr+'any2any'] - df.loc[:,'OC'+attr+'any2spe']
    return df


def add_site_info(df, sub_tensor, attr):
    s = substitution.get_s(sub_tensor, attr=attr)
    s.columns = s.columns.str.replace('site','codon_site_alignment')
    df = pd.merge(df, s, on='codon_site_alignment')
    return df


def add_branch_sub_prob(df, branch_ids, sub_tensor, attr):
    for branch_id in branch_ids:
        sub_probs = substitution.get_branch_site_sub_counts(sub_tensor=sub_tensor, branch_id=branch_id)
        df.loc[:,attr+'_sub_'+str(branch_id)] = sub_probs
    return df


def _parse_branch_ids(branch_id_text):
    if branch_id_text is None:
        raise ValueError('Missing --branch_id.')
    values = [v.strip() for v in str(branch_id_text).split(',') if v.strip()!='']
    if len(values)==0:
        raise ValueError('No branch ID was specified in --branch_id.')
    try:
        branch_ids = np.array([int(v) for v in values], dtype=np.int64)
    except ValueError as exc:
        raise ValueError('--branch_id should be a comma-delimited list of integers.') from exc
    unique_ids, counts = np.unique(branch_ids, return_counts=True)
    duplicated_ids = unique_ids[counts > 1]
    if duplicated_ids.shape[0] > 0:
        txt = '--branch_id contains duplicate IDs: {}'
        raise ValueError(txt.format(','.join([str(int(v)) for v in duplicated_ids.tolist()])))
    return branch_ids


def _is_truthy_fg_value(value):
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    value_txt = str(value).strip().lower()
    return value_txt in ['y', 'yes', 'true', '1', 't']


def _get_node_by_branch_id(g):
    node_by_id = dict()
    for node in g['tree'].traverse():
        branch_id = int(ete.get_prop(node, "numerical_label"))
        node_by_id[branch_id] = node
    return node_by_id


def _validate_existing_branch_ids(branch_ids, node_by_id):
    normalized_branch_ids = _normalize_branch_ids(branch_ids).tolist()
    missing_ids = [int(bid) for bid in normalized_branch_ids if int(bid) not in node_by_id]
    if len(missing_ids)>0:
        txt = '--branch_id contains unknown branch IDs: {}'
        raise ValueError(txt.format(','.join([str(bid) for bid in sorted(missing_ids)])))


def _validate_nonroot_branch_ids(branch_ids, node_by_id):
    _validate_existing_branch_ids(branch_ids, node_by_id)
    normalized_branch_ids = _normalize_branch_ids(branch_ids).tolist()
    root_ids = [int(bid) for bid in normalized_branch_ids if ete.is_root(node_by_id[int(bid)])]
    if len(root_ids)>0:
        txt = '--branch_id should not include root branch IDs: {}'
        raise ValueError(txt.format(','.join([str(bid) for bid in sorted(root_ids)])))


def _read_foreground_branch_combinations(g, node_by_id):
    cb = pd.read_csv(g['cb_file'], sep="\t", index_col=False, header=0)
    bid_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
    if bid_cols.shape[0]==0:
        raise ValueError('No branch_id_* columns were found in --cb_file.')
    is_fg_col = cb.columns.str.startswith('is_fg')
    if is_fg_col.sum()==0:
        raise ValueError('No is_fg* columns were found in --cb_file.')
    fg_mask = cb.loc[:, is_fg_col].apply(lambda col: col.map(_is_truthy_fg_value))
    cb_fg = cb.loc[fg_mask.any(axis=1), :]
    branch_id_list = []
    for i in cb_fg.index:
        bids = _normalize_branch_ids(cb_fg.loc[i, bid_cols].values)
        _validate_nonroot_branch_ids(bids, node_by_id)
        branch_id_list.append(bids)
    if len(branch_id_list)==0:
        raise ValueError('No foreground branch combinations were found in --cb_file.')
    return branch_id_list


def _resolve_lineage_branch_ids(ancestor_id, descendant_id, node_by_id):
    descendant_node = node_by_id[int(descendant_id)]
    lineage_branch_ids = []
    node = descendant_node
    while True:
        node_id = int(ete.get_prop(node, "numerical_label"))
        if not ete.is_root(node):
            lineage_branch_ids.append(node_id)
        if node_id == int(ancestor_id):
            break
        if ete.is_root(node):
            txt = '--mode lineage expects --branch_id ANC,DES where ANC is an ancestor of DES.'
            raise ValueError(txt)
        node = node.up
    lineage_branch_ids = lineage_branch_ids[::-1]
    return np.array(lineage_branch_ids, dtype=np.int64)


def _tokenize_set_expression(mode_expression):
    tokens = []
    i = 0
    txt = str(mode_expression)
    while i < len(txt):
        ch = txt[i]
        if ch.isspace():
            i += 1
            continue
        if ch in ['|', '-', '&', '^', '(', ')']:
            tokens.append(ch)
            i += 1
            continue
        if ch.isdigit():
            j = i + 1
            while (j < len(txt)) and txt[j].isdigit():
                j += 1
            tokens.append(int(txt[i:j]))
            i = j
            continue
        if ch in ['A', 'a']:
            tokens.append('A')
            i += 1
            continue
        raise ValueError('Invalid token in --mode set expression: "{}"'.format(ch))
    if len(tokens)==0:
        raise ValueError('Empty --mode set expression.')
    return tokens


def _extract_set_expression_branch_ids(mode_expression):
    tokens = _tokenize_set_expression(mode_expression)
    branch_ids = sorted(set([token for token in tokens if isinstance(token, int)]))
    if len(branch_ids)==0:
        raise ValueError('--mode set expression should include at least one branch ID.')
    return np.array(branch_ids, dtype=np.int64)


def _get_set_expression_display_branch_ids(g):
    mode_expression = g.get('mode_expression', None)
    branch_ids = _normalize_branch_ids(g.get('branch_ids', [])).tolist()
    if mode_expression is None:
        return np.array(branch_ids, dtype=np.int64)
    tokens = _tokenize_set_expression(mode_expression)
    target_set = set(branch_ids)
    out = []
    seen = set()
    for token in tokens:
        if not isinstance(token, int):
            continue
        bid = int(token)
        if (bid in target_set) and (bid not in seen):
            out.append(bid)
            seen.add(bid)
    for bid in branch_ids:
        if bid not in seen:
            out.append(bid)
            seen.add(bid)
    return np.array(out, dtype=np.int64)


def _get_set_expression_label(mode_expression):
    tokens = _tokenize_set_expression(mode_expression)
    token_label = {
        '|': 'or',
        '&': 'and',
        '^': 'xor',
        '-': 'minus',
        '(': 'lp',
        ')': 'rp',
        'A': 'all_other',
    }
    out = []
    for token in tokens:
        if isinstance(token, int):
            out.append(str(int(token)))
        else:
            out.append(token_label[token])
    mode_expr_label = '_'.join(out)
    mode_expr_label = re.sub(r'_+', '_', mode_expr_label).strip('_')
    if mode_expr_label == '':
        mode_expr_label = 'expr'
    return mode_expr_label


def _get_set_mode_stat_type(set_stat_type):
    stat = str(set_stat_type).strip().lower()
    legacy = {'any2any': 'any', 'any2spe': 'spe'}
    if stat in legacy:
        txt = '--mode set substitution type "{}" is no longer supported. Use "{}" instead.'
        raise ValueError(txt.format(set_stat_type, legacy[stat]))
    if stat in ('spe2any', 'spe2spe'):
        txt = '--mode set substitution type "{}" was removed. Use "spe" or "any".'
        raise ValueError(txt.format(set_stat_type))
    allowed = ('any', 'spe')
    if stat not in allowed:
        txt = '--mode set expects one of [{}] as substitution type, got "{}".'
        raise ValueError(txt.format(','.join(allowed), set_stat_type))
    return stat


def _evaluate_set_expression_boolean(tokens, branch_site_bool):
    branch_site_prob = {}
    for key,value in branch_site_bool.items():
        branch_site_prob[key] = np.zeros(shape=value.shape, dtype=float)
    out_bool,_ = _evaluate_set_expression_boolean_and_prob(
        tokens=tokens,
        branch_site_bool=branch_site_bool,
        branch_site_prob=branch_site_prob,
    )
    return out_bool


def _validate_set_expression_unambiguous_order(tokens):
    operators = {'|', '-', '&', '^'}
    expression_txt = ''.join([str(t) for t in tokens])

    def _validate_operator_sequence(operator_sequence):
        if len(operator_sequence) <= 1:
            return None
        unique_ops = set(operator_sequence)
        if (len(unique_ops) == 1) and (operator_sequence[0] in {'|', '&'}):
            return None
        txt = 'Ambiguous --mode set expression. Use parentheses so operator order is explicit: {}'
        raise ValueError(txt.format(expression_txt))

    operator_stack = [[]]
    for token in tokens:
        if token == '(':
            operator_stack.append([])
            continue
        if token == ')':
            if len(operator_stack) <= 1:
                raise ValueError('Unbalanced parentheses in --mode set expression.')
            operator_sequence = operator_stack.pop()
            _validate_operator_sequence(operator_sequence=operator_sequence)
            continue
        if token in operators:
            operator_stack[-1].append(token)
    if len(operator_stack) != 1:
        raise ValueError('Unbalanced parentheses in --mode set expression.')
    _validate_operator_sequence(operator_sequence=operator_stack[0])
    return None


def _evaluate_set_expression_boolean_and_prob(tokens, branch_site_bool, branch_site_prob):
    operators = ['|', '-', '&', '^']
    _validate_set_expression_unambiguous_order(tokens=tokens)
    operand_stack = []
    operator_stack = []
    expect_operand = True
    operand_shape = None
    for value in branch_site_bool.values():
        operand_shape = value.shape
        break
    if operand_shape is None:
        for value in branch_site_prob.values():
            operand_shape = value.shape
            break
    if operand_shape is None:
        raise ValueError('No branch-site values were provided for set expression evaluation.')

    def _get_operand_arrays(token):
        zero_bool = np.zeros(shape=operand_shape, dtype=bool)
        zero_prob = np.zeros(shape=operand_shape, dtype=float)
        bool_array = branch_site_bool.get(token, zero_bool)
        prob_array = branch_site_prob.get(token, zero_prob)
        bool_array = np.asarray(bool_array, dtype=bool)
        prob_array = np.asarray(prob_array, dtype=float)
        bool_array = np.where(np.isfinite(bool_array), bool_array, False)
        prob_array = np.nan_to_num(prob_array, nan=0.0)
        return bool_array.copy(),prob_array.copy()

    def apply_top_operator():
        if len(operand_stack) < 2:
            raise ValueError('Invalid --mode set expression. Missing operand.')
        rhs_bool,rhs_prob = operand_stack.pop()
        lhs_bool,lhs_prob = operand_stack.pop()
        op = operator_stack.pop()
        lhs_prob_eff = np.where(lhs_bool, lhs_prob, 0.0)
        rhs_prob_eff = np.where(rhs_bool, rhs_prob, 0.0)
        if op == '|':
            out_bool = lhs_bool | rhs_bool
            out_prob = np.where(out_bool, np.maximum(lhs_prob_eff, rhs_prob_eff), 0.0)
        elif op == '-':
            out_bool = lhs_bool & (~rhs_bool)
            out_prob = np.where(out_bool, lhs_prob_eff, 0.0)
        elif op == '&':
            out_bool = lhs_bool & rhs_bool
            out_prob = np.where(out_bool, np.minimum(lhs_prob_eff, rhs_prob_eff), 0.0)
        elif op == '^':
            out_bool = lhs_bool ^ rhs_bool
            out_prob = np.where(out_bool, np.maximum(lhs_prob_eff, rhs_prob_eff), 0.0)
        else:
            raise ValueError('Invalid operator in --mode set expression: {}'.format(op))
        operand_stack.append((out_bool,out_prob))

    for token in tokens:
        if expect_operand:
            if isinstance(token, int):
                operand_stack.append(_get_operand_arrays(token))
                expect_operand = False
            elif token == 'A':
                operand_stack.append(_get_operand_arrays(token))
                expect_operand = False
            elif token == '(':
                operator_stack.append(token)
            else:
                raise ValueError('Invalid --mode set expression near token "{}".'.format(token))
        else:
            if token in operators:
                while (len(operator_stack) > 0) and (operator_stack[-1] in operators):
                    apply_top_operator()
                operator_stack.append(token)
                expect_operand = True
            elif token == ')':
                while (len(operator_stack) > 0) and (operator_stack[-1] != '('):
                    apply_top_operator()
                if (len(operator_stack) == 0) or (operator_stack[-1] != '('):
                    raise ValueError('Unbalanced parentheses in --mode set expression.')
                operator_stack.pop()
            else:
                raise ValueError('Invalid --mode set expression near token "{}".'.format(token))
    if expect_operand:
        raise ValueError('Invalid --mode set expression. Expression ended unexpectedly.')
    while len(operator_stack) > 0:
        if operator_stack[-1] == '(':
            raise ValueError('Unbalanced parentheses in --mode set expression.')
        apply_top_operator()
    if len(operand_stack) != 1:
        raise ValueError('Invalid --mode set expression.')
    out_bool,out_prob = operand_stack[0]
    out_prob = np.where(out_bool, np.nan_to_num(out_prob, nan=0.0), 0.0)
    return out_bool,out_prob


def _validate_set_expression_syntax(mode_expression):
    tokens = _tokenize_set_expression(mode_expression)
    branch_ids = _extract_set_expression_branch_ids(mode_expression)
    branch_site_bool = {int(branch_id): np.zeros(shape=(1,), dtype=bool) for branch_id in branch_ids.tolist()}
    if 'A' in tokens:
        branch_site_bool['A'] = np.zeros(shape=(1,), dtype=bool)
    _evaluate_set_expression_boolean(tokens=tokens, branch_site_bool=branch_site_bool)
    return None


def _get_empty_set_channel_prob(n_site, set_stat_type, ON_tensor=None):
    if ON_tensor is None:
        return np.zeros(shape=(n_site, 1), dtype=float)
    if set_stat_type == 'any':
        n_channel = 1
    elif set_stat_type == 'spe':
        n_channel = int(ON_tensor.shape[4])
    else:
        raise ValueError('Unsupported set substitution type: {}'.format(set_stat_type))
    return np.zeros(shape=(n_site, n_channel), dtype=float)


def _aggregate_set_channels(bool_matrix, prob_matrix):
    bool_arr = np.asarray(bool_matrix, dtype=bool)
    prob_arr = np.nan_to_num(np.asarray(prob_matrix, dtype=float), nan=0.0)
    if bool_arr.ndim == 1:
        prob_arr = np.where(bool_arr, prob_arr, 0.0)
        return bool_arr, prob_arr
    selected = bool_arr.any(axis=1)
    selected_prob = np.where(selected, prob_arr.max(axis=1), 0.0)
    return selected, selected_prob


def add_set_mode_columns(df, g, ON_tensor=None, OS_tensor=None):
    if str(g.get('mode', '')).lower() != 'set':
        return df
    mode_expression = g.get('mode_expression', None)
    if mode_expression is None:
        raise ValueError('Missing set expression for --mode set.')
    set_stat_type = _get_set_mode_stat_type(g.get('set_stat_type', None))
    tokens = _tokenize_set_expression(mode_expression)
    branch_ids = _extract_set_expression_branch_ids(mode_expression)
    n_site = df.shape[0]
    min_single_prob = get_min_single_prob(g)
    if (ON_tensor is None) and (set_stat_type != 'any'):
        txt = '--mode set,{} requires branch-wise substitution tensors. This type is not available from N_sub_* columns only.'
        raise ValueError(txt.format(set_stat_type))
    branch_site_bool = dict()
    branch_site_prob = dict()
    empty_prob = _get_empty_set_channel_prob(
        n_site=n_site,
        set_stat_type=set_stat_type,
        ON_tensor=ON_tensor,
    )
    for branch_id in branch_ids.tolist():
        if ON_tensor is not None:
            bid = int(branch_id)
            if 0 <= bid < ON_tensor.shape[0]:
                n_sub_prob = substitution.get_branch_set_stat_channels(
                    sub_tensor=ON_tensor,
                    branch_id=bid,
                    set_stat_type=set_stat_type,
                )
            else:
                n_sub_prob = empty_prob.copy()
        else:
            col = 'N_sub_{}'.format(int(branch_id))
            if col in df.columns:
                base_prob = np.nan_to_num(df.loc[:, col].to_numpy(dtype=float, copy=True), nan=0.0)
                n_sub_prob = base_prob[:, np.newaxis]
            else:
                n_sub_prob = empty_prob.copy()
        branch_site_prob[int(branch_id)] = n_sub_prob
        branch_site_bool[int(branch_id)] = (n_sub_prob >= min_single_prob)
        branch_prob_arr = np.nan_to_num(np.asarray(n_sub_prob, dtype=float), nan=0.0)
        if branch_prob_arr.ndim == 1:
            branch_prob_max = branch_prob_arr
        else:
            branch_prob_max = branch_prob_arr.max(axis=1)
        branch_prob_max = np.clip(branch_prob_max, 0.0, 1.0)
        df.loc[:, 'N_set_branch_{}_prob'.format(int(branch_id))] = branch_prob_max
    if 'A' in tokens:
        explicit_ids = set([int(bid) for bid in branch_ids.tolist()])
        other_bool_matrix = np.zeros(shape=empty_prob.shape, dtype=bool)
        n_other_prob_matrix = empty_prob.copy()
        s_other_prob = np.zeros(shape=(n_site,), dtype=float)
        if ('tree' in g) and (g['tree'] is not None) and (ON_tensor is not None):
            node_by_id = _get_node_by_branch_id(g)
            other_branch_ids = sorted([
                int(bid) for bid,node in node_by_id.items()
                if (not ete.is_root(node)) and (int(bid) not in explicit_ids)
            ])
            if len(other_branch_ids) > 0:
                other_prob_rows = []
                for other_bid in other_branch_ids:
                    other_prob_rows.append(
                        substitution.get_branch_set_stat_channels(
                            sub_tensor=ON_tensor,
                            branch_id=other_bid,
                            set_stat_type=set_stat_type,
                        )
                    )
                n_other_prob_matrix = np.stack(other_prob_rows, axis=0).max(axis=0)
                other_bool_matrix = (n_other_prob_matrix >= min_single_prob)
                if OS_tensor is not None:
                    other_syn_rows = [
                        substitution.get_branch_site_sub_counts(OS_tensor, branch_id=other_bid)
                        for other_bid in other_branch_ids
                    ]
                    other_syn_probs = np.stack(other_syn_rows, axis=0)
                    if other_syn_probs.ndim == 1:
                        other_syn_probs = other_syn_probs[np.newaxis, :]
                    s_other_prob = other_syn_probs.max(axis=0)
        else:
            other_n_cols = []
            other_s_cols = []
            for col in df.columns[df.columns.str.startswith('N_sub_') | df.columns.str.startswith('S_sub_')].tolist():
                try:
                    if col.startswith('N_sub_'):
                        bid = int(col.replace('N_sub_', ''))
                        is_n_col = True
                    else:
                        bid = int(col.replace('S_sub_', ''))
                        is_n_col = False
                except ValueError:
                    continue
                if bid not in explicit_ids:
                    if is_n_col:
                        other_n_cols.append(col)
                    else:
                        other_s_cols.append(col)
            if len(other_n_cols) > 0:
                other_n_values = df.loc[:, other_n_cols].to_numpy(dtype=float, copy=True)
                other_n_values = np.nan_to_num(other_n_values, nan=0.0)
                n_other_prob_matrix = other_n_values.max(axis=1)[:, np.newaxis]
                other_bool_matrix = (n_other_prob_matrix >= min_single_prob)
            if len(other_s_cols) > 0:
                other_s_values = df.loc[:, other_s_cols].to_numpy(dtype=float, copy=True)
                other_s_values = np.nan_to_num(other_s_values, nan=0.0)
                s_other_prob = other_s_values.max(axis=1)
        branch_site_bool['A'] = other_bool_matrix.astype(bool)
        branch_site_prob['A'] = n_other_prob_matrix.astype(float)
        other_bool_arr = np.asarray(other_bool_matrix, dtype=bool)
        other_prob_arr = np.nan_to_num(np.asarray(n_other_prob_matrix, dtype=float), nan=0.0)
        if other_bool_arr.ndim == 1:
            n_set_other_bool = other_bool_arr
            n_set_other_prob = other_prob_arr
        else:
            n_set_other_bool = other_bool_arr.any(axis=1)
            n_set_other_prob = other_prob_arr.max(axis=1)
        df.loc[:, 'N_set_other'] = n_set_other_bool.astype(bool)
        df.loc[:, 'N_set_other_prob'] = n_set_other_prob.astype(float)
        df.loc[:, 'S_set_other_prob'] = s_other_prob
        # Explicit aliases for easier downstream interpretation in TSV outputs.
        df.loc[:, 'N_set_A'] = df.loc[:, 'N_set_other']
        df.loc[:, 'N_set_A_prob'] = df.loc[:, 'N_set_other_prob']
        df.loc[:, 'S_set_A_prob'] = df.loc[:, 'S_set_other_prob']
    selected_matrix,selected_prob_matrix = _evaluate_set_expression_boolean_and_prob(
        tokens=tokens,
        branch_site_bool=branch_site_bool,
        branch_site_prob=branch_site_prob,
    )
    selected,selected_prob = _aggregate_set_channels(
        bool_matrix=selected_matrix,
        prob_matrix=selected_prob_matrix,
    )
    df.loc[:, 'N_set_expr'] = selected
    df.loc[:, 'N_set_expr_prob'] = np.where(selected, selected_prob, 0.0)
    df.loc[:, 'N_set_expr_channel_index'] = get_set_expression_channel_indices(
        prob_matrix=selected_prob_matrix,
    )
    df.loc[:, 'N_set_expr_channel_label'] = get_set_expression_channel_labels(
        prob_matrix=selected_prob_matrix,
        set_stat_type=set_stat_type,
        state_orders=sequence.get_nonsyn_state_orders(g),
    )
    return df


def should_plot_state(g):
    return _is_intersection_mode(g)


def should_save_pymol_views(g):
    return _is_intersection_mode(g)


def _is_intersection_mode(g):
    mode = str(g.get('mode', 'intersection')).lower()
    return mode == 'intersection'


def _parse_mode_and_expression(raw_mode):
    txt = str(raw_mode).strip()
    parts = [part.strip() for part in txt.split(',')]
    mode = parts[0].lower()
    mode_expression = None
    set_stat_type = None
    if mode == 'set':
        if len(parts) != 3:
            txt = '--mode set expects --mode "set,<substitution_type>,<expression>", e.g., --mode "set,spe,1|3".'
            raise ValueError(txt)
        set_stat_type = _get_set_mode_stat_type(parts[1])
        mode_expression = parts[2]
        if mode_expression == '':
            raise ValueError('--mode set expression is empty.')
    elif len(parts) >= 2:
        mode_expression = ','.join(parts[1:]).strip()
    return mode, mode_expression, set_stat_type


def _build_site_outdir(
    mode,
    branch_txt,
    lineage_input_branch_txt=None,
    mode_expression=None,
    set_stat_type=None,
    base_dir='.',
    output_prefix='csubst_sites',
):
    if mode == 'intersection':
        suffix = '.branch_id' + branch_txt
    elif mode == 'lineage':
        suffix = '.lineage.branch_id' + lineage_input_branch_txt
    elif mode == 'set':
        if set_stat_type is None:
            raise ValueError('Missing set substitution type for --mode set.')
        mode_expr_label = _get_set_expression_label(mode_expression)
        suffix = '.set.' + str(set_stat_type) + '.expr' + mode_expr_label
    else:
        suffix = '.mode' + mode + '.branch_id' + branch_txt
    return os.path.join(str(base_dir), str(output_prefix) + suffix)


def _maybe_relocate_site_log_file(g):
    # The CLI owns an open handle to this path. Moving an open log is not
    # portable (notably on Windows) and makes the output manifest disagree with
    # the active handle. Keep the path resolved before command execution.
    return g


def resolve_site_jobs(g):
    raw_mode = str(g.get('mode', 'intersection')).strip()
    mode, mode_expression, set_stat_type = _parse_mode_and_expression(raw_mode)
    g['mode'] = mode
    g['mode_expression'] = mode_expression
    g['set_stat_type'] = set_stat_type
    if variant_effect.is_enabled(g):
        if mode == 'set':
            raise ValueError('--vep_model currently supports --mode intersection and lineage; set mode is not yet supported.')
        if str(g.get('nonsyn_recode', 'no')).strip().lower() != 'no':
            raise ValueError('--vep_model currently requires --nonsyn_recode no.')
    node_by_id = _get_node_by_branch_id(g)
    branch_id_list = []
    lineage_input_branch_txt = None

    if mode == 'intersection':
        if str(g['branch_id']).lower()=='fg':
            branch_id_list = _read_foreground_branch_combinations(g=g, node_by_id=node_by_id)
        else:
            branch_ids = _parse_branch_ids(g['branch_id'])
            _validate_nonroot_branch_ids(branch_ids, node_by_id)
            branch_id_list = [branch_ids]
    elif mode=='lineage':
        branch_ids = _parse_branch_ids(g['branch_id'])
        if branch_ids.shape[0]!=2:
            raise ValueError('--mode lineage expects --branch_id ANC,DES.')
        lineage_input_branch_txt = '{},{}'.format(int(branch_ids[0]), int(branch_ids[1]))
        _validate_existing_branch_ids(branch_ids, node_by_id)
        descendant_id = int(branch_ids[1])
        if ete.is_root(node_by_id[descendant_id]):
            raise ValueError('--mode lineage expects a non-root DES branch ID.')
        lineage_branch_ids = _resolve_lineage_branch_ids(
            ancestor_id=int(branch_ids[0]),
            descendant_id=descendant_id,
            node_by_id=node_by_id,
        )
        if lineage_branch_ids.shape[0]==0:
            raise ValueError('No non-root branch IDs were found for --mode lineage.')
        branch_id_list = [lineage_branch_ids]
    elif mode=='set':
        if (mode_expression is None) or (mode_expression==''):
            raise ValueError('--mode set expects an expression, e.g., --mode "set,any,1|5".')
        if set_stat_type is None:
            raise ValueError('--mode set expects a substitution type, e.g., --mode "set,any,1|5".')
        _validate_set_expression_syntax(mode_expression=mode_expression)
        expression_branch_ids = _extract_set_expression_branch_ids(mode_expression)
        _validate_existing_branch_ids(expression_branch_ids, node_by_id)
        selected_nonroot = [bid for bid in expression_branch_ids.tolist() if not ete.is_root(node_by_id[int(bid)])]
        if len(selected_nonroot)==0:
            raise ValueError('--mode set expression should include at least one non-root branch ID.')
        branch_id_list = [np.array(sorted(selected_nonroot), dtype=np.int64)]
    else:
        raise ValueError('--mode should be one of intersection,lineage,set or set,<expr>.')

    site_jobs = []
    for branch_ids in branch_id_list:
        branch_ids = _normalize_branch_ids(branch_ids)
        single_branch_mode = (branch_ids.shape[0]==1)
        branch_txt = ','.join([str(int(bid)) for bid in branch_ids.tolist()])
        site_outdir = _build_site_outdir(
            mode=mode,
            branch_txt=branch_txt,
            lineage_input_branch_txt=lineage_input_branch_txt,
            mode_expression=mode_expression,
            set_stat_type=set_stat_type,
            base_dir=g.get('outdir', '.'),
            output_prefix=_site_output_prefix(g),
        )
        site_jobs.append({
            'branch_ids': branch_ids,
            'single_branch_mode': single_branch_mode,
            'site_outdir': site_outdir,
            'mode_expression': mode_expression,
            'set_stat_type': set_stat_type,
        })
    g['site_jobs'] = site_jobs
    g['branch_id_list'] = [job['branch_ids'] for job in site_jobs]
    return g


def add_branch_id_list(g):
    return resolve_site_jobs(g)


def combinatorial2single_columns(df):
    drop_cols = list()
    for SN in ['OCS', 'OCN']:
        for anc in ['any', 'spe', 'dif']:
            for des in ['any', 'spe', 'dif']:
                col = SN + anc + '2' + des
                if col in df.columns:
                    drop_cols.append(col)
    if len(drop_cols) == 0:
        return df
    return df.drop(labels=drop_cols, axis=1)


def _select_vesm_plot_sites(events, max_sites):
    if events.shape[0] == 0:
        return []
    finite = events.loc[
        np.isfinite(pd.to_numeric(events['vesm_llr'], errors='coerce')),
        :,
    ].copy()
    if finite.shape[0] == 0:
        return []
    summary = finite.groupby('codon_site_alignment', as_index=False).agg(
        vep_rank_score=('vesm_llr', 'min'),
        vep_rank_pp=('event_pp', 'max'),
    )
    summary = summary.sort_values(
        by=['vep_rank_score', 'vep_rank_pp', 'codon_site_alignment'],
        ascending=[True, False, True],
        kind='mergesort',
    )
    selected = summary.iloc[:int(max_sites), :]['codon_site_alignment'].astype(int).tolist()
    return sorted(selected)


def _get_vesm_structure_label_by_site(df, sites):
    pdb_cols = [str(col) for col in df.columns if str(col).startswith('codon_site_pdb_')]
    if len(pdb_cols) == 0:
        return {int(site): '' for site in sites}
    best_col = None
    best_count = -1
    for col in pdb_cols:
        values = pd.to_numeric(df[col], errors='coerce').fillna(0).to_numpy(dtype=np.int64, copy=False)
        count = int((values > 0).sum())
        if count > best_count:
            best_col = col
            best_count = count
    if best_col is None:
        return {int(site): '' for site in sites}
    chain_label = best_col.replace('codon_site_pdb_', '', 1)
    row_by_site = {
        int(value): index
        for index,value in zip(df.index.tolist(), df['codon_site_alignment'].astype(int).tolist())
    }
    out = {}
    for site in sites:
        row_index = row_by_site.get(int(site), None)
        if row_index is None:
            out[int(site)] = ''
            continue
        residue = pd.to_numeric(pd.Series([df.at[row_index, best_col]]), errors='coerce').iloc[0]
        if pd.isna(residue) or int(residue) <= 0:
            out[int(site)] = 'unmapped'
        else:
            out[int(site)] = '{}:{}'.format(chain_label, int(residue))
    return out


def _get_vesm_branch_color_by_id(g, branch_ids):
    branch_ids = _normalize_branch_ids(branch_ids).tolist()
    if str(g.get('mode', '')).strip().lower() == 'lineage':
        return get_tree_site_branch_color_by_id(
            branch_ids=branch_ids,
            g=g,
            default_color='crimson',
        )
    return {int(branch_id): 'crimson' for branch_id in branch_ids}


def _draw_vesm_tree_axis(ax, g, selected_branch_ids, branch_color_by_id):
    xcoord,ycoord,leaf_order = get_tree_plot_coordinates(tree=g['tree'])
    selected = set(int(value) for value in _normalize_branch_ids(selected_branch_ids).tolist())
    for node in g['tree'].traverse():
        node_id = int(ete.get_prop(node, 'numerical_label'))
        children = list(ete.get_children(node))
        if len(children) > 0:
            for child in children:
                child_id = int(ete.get_prop(child, 'numerical_label'))
                is_selected_segment = (node_id in selected) and (child_id in selected)
                color = branch_color_by_id.get(child_id, 'crimson') if is_selected_segment else '0.65'
                linewidth = 2.0 if is_selected_segment else 0.8
                ax.plot(
                    [xcoord[node_id], xcoord[node_id]],
                    [ycoord[node_id], ycoord[child_id]],
                    color=color,
                    linewidth=linewidth,
                    solid_capstyle=TREE_LINE_CAPSTYLE,
                )
        if ete.is_root(node):
            continue
        parent_id = int(ete.get_prop(node.up, 'numerical_label'))
        color = branch_color_by_id.get(node_id, 'crimson') if node_id in selected else '0.55'
        linewidth = 2.0 if node_id in selected else 0.8
        ax.plot(
            [xcoord[parent_id], xcoord[node_id]],
            [ycoord[node_id], ycoord[node_id]],
            color=color,
            linewidth=linewidth,
            solid_capstyle=TREE_LINE_CAPSTYLE,
        )
    max_x = max(xcoord.values()) if len(xcoord) else 1.0
    label_offset = max(max_x * 0.02, 0.01)
    for leaf in ete.iter_leaves(g['tree']):
        leaf_id = int(ete.get_prop(leaf, 'numerical_label'))
        ax.text(max_x + label_offset, ycoord[leaf_id], str(leaf.name), va='center', fontsize=6)
    ax.set_title('Selected branches on phylogeny')
    ax.set_axis_off()
    if len(leaf_order) > 0:
        ax.set_ylim(max(ycoord.values()) + 0.5, min(ycoord.values()) - 0.5)
    ax.set_xlim(min(xcoord.values()) if len(xcoord) else 0, max_x + (label_offset * 20))


def _get_vesm_plot_dimensions(num_sites, leaf_count, num_branches, max_height):
    num_sites = max(int(num_sites), 1)
    tick_pitch_points = float(font_size) + VESM_XTICK_LABEL_GAP_POINTS
    site_data_width = num_sites * tick_pitch_points / 72.0
    # Reserve space taken from the site panel by its colorbar and padding.
    site_panel_width = max(1.5, site_data_width / 0.92)
    tree_panel_width = 3.2
    figure_width = max(6.5, tree_panel_width + site_panel_width + 0.5)
    figure_height = min(max(4.5, leaf_count * 0.18, num_branches * 0.35), max_height)
    return figure_width,figure_height,tree_panel_width,site_panel_width


def plot_vesm_tree_site(events, df, g, outbase):
    table_path = outbase + '.vesm_tree_site.tsv'
    plot_format = str(g.get('tree_site_plot_format', 'pdf')).strip().lower()
    fig_path = outbase + '.vesm_tree_site.' + plot_format
    plot_events = events.copy(deep=True)
    selected_sites = _select_vesm_plot_sites(
        events=plot_events,
        max_sites=get_tree_site_plot_max_sites(g),
    )
    plot_events.loc[:, 'is_plotted'] = plot_events['codon_site_alignment'].astype(int).isin(selected_sites)
    plot_order = {int(site): index + 1 for index,site in enumerate(selected_sites)}
    plot_events.loc[:, 'plot_order'] = [
        plot_order.get(int(site), np.nan)
        for site in plot_events['codon_site_alignment'].astype(int).tolist()
    ]
    tsv.write_dataframe(plot_events, table_path, float_format=g['float_format'])
    print('Writing VESM tree + site table: {}'.format(table_path), flush=True)
    if (not bool(g.get('vep_plot', True))) or len(selected_sites) == 0:
        if len(selected_sites) == 0:
            print('Skipping VESM tree + site plot because no scored events passed the PP threshold.', flush=True)
        return [table_path]

    branch_ids = _normalize_branch_ids(g.get('branch_ids', [])).tolist()
    representatives = plot_events.loc[plot_events['is_plotted'], :].sort_values(
        by=['branch_id', 'codon_site_alignment', 'event_pp', 'vesm_llr', 'event_id'],
        ascending=[True, True, False, True, True],
        kind='mergesort',
    ).drop_duplicates(subset=['branch_id', 'codon_site_alignment'], keep='first')
    leaf_count = len(list(ete.iter_leaves(g['tree'])))
    max_height = get_tree_site_fig_max_height(g)
    fig_width,fig_height,tree_panel_width,site_panel_width = _get_vesm_plot_dimensions(
        num_sites=len(selected_sites),
        leaf_count=leaf_count,
        num_branches=len(branch_ids),
        max_height=max_height,
    )
    fig,axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(fig_width, fig_height),
        gridspec_kw={'width_ratios': [tree_panel_width, site_panel_width]},
    )
    ax_tree,ax_grid = axes
    branch_color_by_id = _get_vesm_branch_color_by_id(g=g, branch_ids=branch_ids)
    _draw_vesm_tree_axis(
        ax=ax_tree,
        g=g,
        selected_branch_ids=branch_ids,
        branch_color_by_id=branch_color_by_id,
    )

    x_by_site = {int(site): index for index,site in enumerate(selected_sites)}
    y_by_branch = {int(branch_id): index for index,branch_id in enumerate(branch_ids)}
    finite_scores = representatives['vesm_llr'].to_numpy(dtype=float, copy=False)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    color_limit = max(
        float(g.get('_vep_color_limit', 0.0)),
        float(np.max(np.abs(finite_scores))) if finite_scores.size else 0.0,
        1e-6,
    )
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    cmap = plt.get_cmap('coolwarm_r')
    for row in representatives.itertuples(index=False):
        branch_id = int(row.branch_id)
        site = int(row.codon_site_alignment)
        if branch_id not in y_by_branch or site not in x_by_site:
            continue
        marker_size = 35.0 + (220.0 * float(row.event_pp))
        ax_grid.scatter(
            [x_by_site[site]],
            [y_by_branch[branch_id]],
            s=marker_size,
            c=[float(row.vesm_llr)],
            cmap=cmap,
            norm=norm,
            edgecolors='black',
            linewidths=0.45,
            zorder=3,
        )
        ax_grid.text(
            x_by_site[site],
            y_by_branch[branch_id],
            '{}>{}'.format(row.from_aa, row.to_aa),
            ha='center',
            va='center',
            fontsize=5 if len(selected_sites) <= 18 else 4,
            zorder=4,
        )
    structure_label = _get_vesm_structure_label_by_site(df=df, sites=selected_sites)
    tick_labels = []
    for site in selected_sites:
        label = 'aln {}'.format(int(site))
        if structure_label[int(site)] != '':
            label += '\n' + structure_label[int(site)]
        tick_labels.append(label)
    ax_grid.set_xticks(np.arange(len(selected_sites), dtype=float))
    ax_grid.set_xticklabels(tick_labels, rotation=90, fontsize=font_size)
    ax_grid.set_yticks(np.arange(len(branch_ids), dtype=float))
    ax_grid.set_yticklabels(['b{}'.format(int(branch_id)) for branch_id in branch_ids])
    for tick,branch_id in zip(ax_grid.get_yticklabels(), branch_ids):
        tick.set_color(branch_color_by_id.get(int(branch_id), 'black'))
    ax_grid.set_xlim(-0.6, len(selected_sites) - 0.4)
    ax_grid.set_ylim(len(branch_ids) - 0.4, -0.6)
    ax_grid.set_xlabel('Aligned codon site / mapped structure residue')
    ax_grid.set_ylabel('Selected branch')
    ax_grid.set_title(
        'VESM-35M: marker size = substitution PP; color = LLR (lower = more deleterious)\n'
        'event PP threshold >= {:.3g}'.format(float(g.get('vep_min_event_pp', 0.8)))
    )
    ax_grid.grid(color='0.9', linewidth=0.5, zorder=0)
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax_grid, fraction=0.035, pad=0.03)
    colorbar.set_label('VESM LLR (lower = more deleterious)')
    pp_values = sorted(set([float(g.get('vep_min_event_pp', 0.8)), 1.0]))
    handles = [
        ax_grid.scatter([], [], s=35.0 + (220.0 * value), facecolor='white', edgecolor='black')
        for value in pp_values
    ]
    ax_grid.legend(handles, ['PP {:.2g}'.format(value) for value in pp_values], loc='upper right')
    fig.tight_layout()
    fig.savefig(fig_path, format=plot_format, transparent=False, facecolor='white')
    plt.close(fig)
    print('Writing VESM tree + site plot: {}'.format(fig_path), flush=True)
    return [fig_path, table_path]


def main_sites(g):
    if g['pdb'] is not None:
        from csubst import parser_pymol
    print("Reading and parsing input files.", flush=True)
    vep_enabled = variant_effect.is_enabled(g)
    if vep_enabled and str(g.get('mode', 'intersection')).strip().lower().startswith('set'):
        raise ValueError('--vep_model currently supports --mode intersection and lineage; set mode is not yet supported.')
    if vep_enabled and str(g.get('nonsyn_recode', 'no')).strip().lower() != 'no':
        raise ValueError('--vep_model currently requires --nonsyn_recode no.')
    g = parser_misc.prepare_input_context(
        g,
        include_foreground=False,
        include_marginal=False,
        resolve_state_subset=False,
        prepare_state=(not vep_enabled),
    )
    if vep_enabled:
        # Preserve a compact full-length ancestral context before ordinary site filtering.
        g = parser_misc.prep_state(g, apply_site_filtering=False)
        variant_effect.prepare_ancestral_contexts(g=g)
        g = parser_misc.apply_site_filters(g)
    ON_tensor = substitution.get_substitution_tensor(state_tensor=g['state_nsy'], mode='asis', g=g, mmap_attr='N')
    ON_tensor = substitution.apply_min_sub_pp(g, ON_tensor)
    OS_tensor = substitution.get_substitution_tensor(state_tensor=g['state_cdn'], mode='syn', g=g, mmap_attr='S')
    OS_tensor = substitution.apply_min_sub_pp(g, OS_tensor)
    g = resolve_site_jobs(g)
    g = _maybe_relocate_site_log_file(g)
    for site_job in g['site_jobs']:
        branch_ids = _normalize_branch_ids(site_job['branch_ids'])
        g['single_branch_mode'] = site_job['single_branch_mode']
        g['branch_ids'] = branch_ids
        g['site_outdir'] = site_job['site_outdir']
        g['mode_expression'] = site_job.get('mode_expression', g.get('mode_expression', None))
        g['set_stat_type'] = site_job.get('set_stat_type', g.get('set_stat_type', None))
        site_prefix = _site_output_prefix(g)
        txt = '\nProcessing --mode {} with branch IDs: {}'
        print(txt.format(g['mode'], ','.join([str(int(bid)) for bid in branch_ids.tolist()])), flush=True)
        if (g.get('mode_expression', None) is not None) and (str(g.get('mode', '')).lower() == 'set'):
            print('Set expression ({}): {}'.format(g['set_stat_type'], g['mode_expression']), flush=True)
        if g['single_branch_mode']:
            print('Single branch mode. Substitutions, rather than combinatorial substitutions, will be mapped.')
        if not os.path.exists(g['site_outdir']):
            os.makedirs(g['site_outdir'])
        manifest_rows = list()
        leaf_nn = [ete.get_prop(n, "numerical_label") for n in g['tree'].traverse() if ete.is_leaf(n)]
        num_site = ON_tensor.shape[1]
        df = initialize_site_df(num_site)
        df = add_cs_info(df, g['branch_ids'], sub_tensor=OS_tensor, attr='S')
        df = add_cs_info(df, g['branch_ids'], sub_tensor=ON_tensor, attr='N')
        df.loc[:,'gap_rate_all'] = get_gapsite_rate(state_tensor=g['state_cdn'][leaf_nn,:,:])
        df.loc[:,'gap_rate_target'] = get_gapsite_rate(state_tensor=g['state_cdn'][g['branch_ids'],:,:])
        df = add_site_info(df, sub_tensor=OS_tensor, attr='S')
        df = add_site_info(df, sub_tensor=ON_tensor, attr='N')
        df = add_branch_sub_prob(df, branch_ids=g['branch_ids'], sub_tensor=OS_tensor, attr='S')
        df = add_branch_sub_prob(df, branch_ids=g['branch_ids'], sub_tensor=ON_tensor, attr='N')
        df = add_set_mode_columns(df=df, g=g, ON_tensor=ON_tensor, OS_tensor=OS_tensor)
        df = add_states(df, g['branch_ids'], g)
        vep_events = variant_effect.empty_event_table()
        if vep_enabled:
            from csubst import vesm
            vep_events = variant_effect.extract_atomic_aa_events(g=g, branch_ids=g['branch_ids'])
            if vep_events.shape[0] > 0:
                print(
                    'Scoring {:,} amino-acid event(s) with VESM-35M (event PP >= {}).'.format(
                        int(vep_events.shape[0]),
                        float(g.get('vep_min_event_pp', 0.8)),
                    ),
                    flush=True,
                )
                vep_events = vesm.score_events(events=vep_events, g=g)
            else:
                print(
                    'No amino-acid events passed --vep_min_event_pp {}; VESM model loading was skipped.'.format(
                        float(g.get('vep_min_event_pp', 0.8))
                    ),
                    flush=True,
                )
            df = variant_effect.attach_scores_to_site_table(
                df=df,
                events=vep_events,
                branch_ids=g['branch_ids'],
                g=g,
            )
        if (g['untrimmed_cds'] is not None):
            df = add_gene_index(df, g)
        df = remap_codon_site_columns_to_alignment(df=df, g=g)
        is_site_col = df.columns.str.startswith('codon_site_')
        df.loc[:,is_site_col] += 1
        if (g['untrimmed_cds'] is not None)|(g['export2chimera']):
            export2chimera(df, g)
        if g['run_pdb_sequence_search']:
            from csubst import parser_biodb
            g = parser_biodb.pdb_sequence_search(g)
        if (g['pdb'] is not None):
            id_base = os.path.basename(g['pdb'])
            id_base = re.sub('.pdb$', '', id_base)
            id_base = re.sub('.cif$', '', id_base)
            g['pdb_outfile_base'] = os.path.join(g['site_outdir'], site_prefix + '.' + id_base)
            parser_pymol.initialize_pymol(pdb_id=g['pdb'], g=g)
            num_chain = parser_pymol.get_num_chain()
            if num_chain >= g['pymol_max_num_chain']:
                print(f'Number of chains ({num_chain}) in the PDB file is larger than the maximum number of chains allowed (--pymol_max_num_chain {g["pymol_max_num_chain"]}). PyMOL session image generation is disabled.', flush=True)
                g['pymol_img'] = False
            if g['user_alignment'] is not None:
                g['mafft_add_fasta'] = g['user_alignment']
                print('User protein alignment file is provided. Using it for the coordinate mapping.', flush=True)
                print('Please make sure that the alignment site positions are consistent with the input codon alignment.', flush=True)
                df = parser_pymol.add_coordinate_from_user_alignment(df=df, user_alignment=g['mafft_add_fasta'])
            else:
                g['mafft_add_fasta'] = g['pdb_outfile_base']+'.fa'
                parser_pymol.write_mafft_alignment(g=g)
                df = parser_pymol.add_coordinate_from_mafft_map(df=df, mafft_map_file='tmp.csubst.pdb_seq.fa.map')
            df = parser_pymol.add_pdb_residue_numbering(df=df)
            from csubst import parser_uniprot
            df = parser_uniprot.add_uniprot_site_annotations(df=df, g=g)
            g['session_file_path'] = g['pdb_outfile_base']+'.pymol.pse'
            parser_pymol.write_pymol_session(df=df, g=g)
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=g['session_file_path'],
                output_kind='pymol_session',
                g=g,
                branch_ids=g['branch_ids'],
            )
            if g['pymol_img'] and should_save_pymol_views(g):
                parser_pymol.save_six_views()
                pymol_pdf_path = os.path.join(g['site_outdir'], f'{site_prefix}.{id_base}.pymol.pdf')
                parser_pymol.save_6view_pdf(pdf_filename=pymol_pdf_path)
                add_site_output_manifest_row(
                    manifest_rows=manifest_rows,
                    output_path=pymol_pdf_path,
                    output_kind='pymol_summary_pdf',
                    g=g,
                    branch_ids=g['branch_ids'],
                )
        if vep_enabled:
            if g['pdb'] is None:
                vep_outbase = os.path.join(g['site_outdir'], site_prefix)
            else:
                vep_outbase = g['pdb_outfile_base']
            vep_output_events = variant_effect.add_structure_coordinates_to_events(
                events=vep_events,
                site_df=df,
            )
            vep_table_path = vep_outbase + '.vesm.tsv'
            tsv.write_dataframe(
                vep_output_events,
                vep_table_path,
                float_format=g['float_format'],
            )
            print('Writing VESM event table: {}'.format(vep_table_path), flush=True)
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=vep_table_path,
                output_kind='vesm_event_tsv',
                g=g,
                branch_ids=g['branch_ids'],
            )
            vep_plot_paths = plot_vesm_tree_site(
                events=vep_output_events,
                df=df,
                g=g,
                outbase=vep_outbase,
            )
            for vep_plot_path in vep_plot_paths:
                output_kind = 'vesm_tree_site_tsv' if vep_plot_path.endswith('.tsv') else 'vesm_tree_site_plot'
                add_site_output_manifest_row(
                    manifest_rows=manifest_rows,
                    output_path=vep_plot_path,
                    output_kind=output_kind,
                    g=g,
                    branch_ids=g['branch_ids'],
                )
        if bool(g.get('site_summary_plot', True)):
            barchart_path = plot_barchart(df, g)
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=barchart_path,
                output_kind='site_summary_pdf',
                g=g,
                branch_ids=g['branch_ids'],
            )
        else:
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=_resolve_barchart_output_base(g) + '.pdf',
                output_kind='site_summary_pdf',
                g=g,
                branch_ids=g['branch_ids'],
                note='skipped_by_site_summary_plot',
            )
        if should_plot_state(g):
            state_paths = plot_state(ON_tensor, OS_tensor, g['branch_ids'], g)
            if len(state_paths):
                for state_path in state_paths:
                    file_name = os.path.basename(state_path)
                    if file_name == site_prefix + '.state.pdf':
                        output_kind = 'state_pattern_pdf'
                    elif file_name == site_prefix + '.state_N.tsv':
                        output_kind = 'state_pattern_nonsyn_tsv'
                    elif file_name == site_prefix + '.state_S.tsv':
                        output_kind = 'state_pattern_syn_tsv'
                    else:
                        output_kind = 'state_pattern_misc'
                    add_site_output_manifest_row(
                        manifest_rows=manifest_rows,
                        output_path=state_path,
                        output_kind=output_kind,
                        g=g,
                        branch_ids=g['branch_ids'],
                    )
            else:
                add_site_output_manifest_row(
                    manifest_rows=manifest_rows,
                    output_path=os.path.join(g['site_outdir'], site_prefix + '.state.pdf'),
                    output_kind='state_pattern_pdf',
                    g=g,
                    branch_ids=g['branch_ids'],
                    note='skipped_by_site_state_plot',
                )
                add_site_output_manifest_row(
                    manifest_rows=manifest_rows,
                    output_path=os.path.join(g['site_outdir'], site_prefix + '.state_N.tsv'),
                    output_kind='state_pattern_nonsyn_tsv',
                    g=g,
                    branch_ids=g['branch_ids'],
                    note='skipped_by_site_state_plot',
                )
                add_site_output_manifest_row(
                    manifest_rows=manifest_rows,
                    output_path=os.path.join(g['site_outdir'], site_prefix + '.state_S.tsv'),
                    output_kind='state_pattern_syn_tsv',
                    g=g,
                    branch_ids=g['branch_ids'],
                    note='skipped_by_site_state_plot',
                )
        else:
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=os.path.join(g['site_outdir'], site_prefix + '.state.pdf'),
                output_kind='state_pattern_pdf',
                g=g,
                branch_ids=g['branch_ids'],
                note='skipped_by_mode',
            )
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=os.path.join(g['site_outdir'], site_prefix + '.state_N.tsv'),
                output_kind='state_pattern_nonsyn_tsv',
                g=g,
                branch_ids=g['branch_ids'],
                note='skipped_by_mode',
            )
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=os.path.join(g['site_outdir'], site_prefix + '.state_S.tsv'),
                output_kind='state_pattern_syn_tsv',
                g=g,
                branch_ids=g['branch_ids'],
                note='skipped_by_mode',
            )
        tree_paths = plot_tree_site(df, g)
        tree_site_prefix = str(g.get('tree_site_plot_prefix', site_prefix)).strip() or site_prefix
        if len(tree_paths):
            for tree_path in tree_paths:
                file_name = os.path.basename(tree_path)
                if file_name.startswith(tree_site_prefix + '.tree_site.') and file_name.endswith('.tsv'):
                    output_kind = 'tree_site_table_tsv'
                elif file_name.startswith(tree_site_prefix + '.tree_site.'):
                    output_kind = 'tree_site_plot'
                else:
                    output_kind = 'tree_site_misc'
                add_site_output_manifest_row(
                    manifest_rows=manifest_rows,
                    output_path=tree_path,
                    output_kind=output_kind,
                    g=g,
                    branch_ids=g['branch_ids'],
                )
        else:
            tree_format = str(g.get('tree_site_plot_format', 'pdf')).lower()
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=os.path.join(g['site_outdir'], tree_site_prefix + '.tree_site.' + tree_format),
                output_kind='tree_site_plot',
                g=g,
                branch_ids=g['branch_ids'],
                note='skipped_by_tree_site_plot',
            )
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=os.path.join(g['site_outdir'], tree_site_prefix + '.tree_site.tsv'),
                output_kind='tree_site_table_tsv',
                g=g,
                branch_ids=g['branch_ids'],
                note='skipped_by_tree_site_plot',
            )
        if g['pdb'] is None:
            outbase = os.path.join(g['site_outdir'], site_prefix)
        else:
            outbase = g['pdb_outfile_base']
        if str(g.get('mode', '')).lower() == 'lineage':
            plot_lineage_tree(g=g, outbase=outbase)
        out_path = outbase+'.tsv'
        if g['single_branch_mode']:
            df = combinatorial2single_columns(df)
        df_out = expand_site_table_to_alignment(df=df, g=g)
        tsv.write_dataframe(df_out, out_path, float_format=g['float_format'], chunksize=10000)
        add_site_output_manifest_row(
            manifest_rows=manifest_rows,
            output_path=out_path,
            output_kind='site_table_tsv',
            g=g,
            branch_ids=g['branch_ids'],
        )
        if bool(g.get('output_manifest', True)):
            write_site_output_manifest(manifest_rows=manifest_rows, g=g, branch_ids=g['branch_ids'])
        else:
            print('Skipping site output manifest (--output_manifest no).', flush=True)
    print('To visualize the convergence probability on protein structure, please see: https://github.com/kfuku52/csubst/wiki')
    print('')
    runtime.cleanup_legacy_temp_artifacts()
    return None

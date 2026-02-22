import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import datetime
import itertools
import os
import re
import sys

from csubst import genetic_code
from csubst import parser_misc
from csubst import sequence
from csubst import substitution
from csubst import tree
from csubst import ete

font_size = 8
TREE_LINE_CAPSTYLE = 'round'
matplotlib.rcParams['font.size'] = font_size
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Nimbus Sans', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none' # none, path, or svgfont
matplotlib.rc('xtick', labelsize=font_size)
matplotlib.rc('ytick', labelsize=font_size)
matplotlib.rc('font', size=font_size)
matplotlib.rc('axes', titlesize=font_size)
matplotlib.rc('axes', labelsize=font_size)
matplotlib.rc('xtick', labelsize=font_size)
matplotlib.rc('ytick', labelsize=font_size)
matplotlib.rc('legend', fontsize=font_size)
matplotlib.rc('figure', titlesize=font_size)

def bool2yesno(flag):
    return 'yes' if bool(flag) else 'no'


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

def add_site_output_manifest_row(manifest_rows, output_path, output_kind, g, branch_ids, note=''):
    site_outdir = os.path.abspath(g['site_outdir'])
    output_path_abs = os.path.abspath(output_path)
    exists = os.path.exists(output_path_abs)
    size_bytes = os.path.getsize(output_path_abs) if exists else -1
    normalized_branch_ids = _normalize_branch_ids(branch_ids)
    if output_path_abs.startswith(site_outdir + os.sep):
        output_file = os.path.relpath(output_path_abs, start=site_outdir)
    else:
        output_file = output_path_abs
    effective_min_prob = float(get_tree_site_min_prob(g))
    row = {
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'branch_ids': ','.join([str(int(bid)) for bid in normalized_branch_ids.tolist()]),
        'branch_count': int(normalized_branch_ids.shape[0]),
        'single_branch_mode': bool2yesno(g.get('single_branch_mode', False)),
        'output_kind': str(output_kind),
        'output_file': str(output_file),
        'output_path': output_path_abs,
        'file_exists': bool2yesno(exists),
        'file_size_bytes': int(size_bytes),
        'tree_site_plot': bool2yesno(g.get('tree_site_plot', True)),
        'site_state_plot': bool2yesno(g.get('site_state_plot', True)),
        'tree_site_plot_format': str(g.get('tree_site_plot_format', 'pdf')).lower(),
        'min_prob_effective': effective_min_prob,
        # Backward-compatible alias for downstream consumers.
        'tree_site_plot_min_prob_effective': effective_min_prob,
        'tree_site_plot_max_sites': int(get_tree_site_plot_max_sites(g)),
        'pdb_mode': bool2yesno(g.get('pdb', None) is not None),
        'note': str(note),
    }
    manifest_rows.append(row)
    return manifest_rows

def write_site_output_manifest(manifest_rows, g, branch_ids):
    manifest_path = os.path.join(g['site_outdir'], 'csubst_site.outputs.tsv')
    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_df.shape[0] > 0:
        manifest_df = manifest_df.sort_values(by=['output_kind', 'output_file']).reset_index(drop=True)
    manifest_df.to_csv(manifest_path, sep='\t', index=False, chunksize=10000)
    add_site_output_manifest_row(
        manifest_rows=manifest_rows,
        output_path=manifest_path,
        output_kind='output_manifest',
        g=g,
        branch_ids=branch_ids,
        note='manifest_self_row',
    )
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df = manifest_df.sort_values(by=['output_kind', 'output_file']).reset_index(drop=True)
    manifest_df.to_csv(manifest_path, sep='\t', index=False, chunksize=10000)
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
        return os.path.join(g['site_outdir'], 'csubst_site')
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
    file_fasta = os.path.join(g['site_outdir'], 'csubst_site_' + seq_key + '.fasta')
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
        file_name = os.path.join(g['site_outdir'], 'csubst_site_' + seq_key + '.chimera.txt')
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
    parent_branch_ids = dict()
    for node in g['tree'].traverse():
        if ete.get_prop(node, "numerical_label") in branch_ids:
            parent_branch_ids[ete.get_prop(node, "numerical_label")] = ete.get_prop(node.up, "numerical_label")
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
        state_orders = {'nsy':g['amino_acid_orders']}
    elif mode=='syn':
        state_orders = g['matrix_groups']
    state_keys = list(state_orders.keys())
    return state_orders,state_keys

def get_df_ad(sub_tensor, g, mode):
    state_orders,state_keys = get_state_orders(g, mode)
    gad = sub_tensor.sum(axis=(0,1))
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
    outcol = 'site_'+method
    df_ad.loc[:,outcol] = np.nan
    sgad = sub_tensor.sum(axis=0)
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
    outcol = 'has_target_high_combinat_prob_site'
    df_ad.loc[:,outcol] = False
    sgad = sub_tensor.sum(axis=0)
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
    cols = ['group','state_from','state_to','max_dist_bl']
    inds = np.arange(np.array(sub_tensor.shape[2:]).prod()-sub_tensor.shape[4])
    df_dist = pd.DataFrame(columns=cols, index=inds)
    bgad = sub_tensor.sum(axis=1)
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
    outfiles = ['csubst_site.state_N.tsv', 'csubst_site.state_S.tsv']
    colors = ['red','blue']
    ax_cols = [0,1]
    titles = ['Nonsynonymous substitution','Synonymous substitution']
    iter_items = zip(ax_cols,['nsy','syn'],[ON_tensor,OS_tensor],outfiles,colors,titles)
    for ax_col,mode,sub_tensor,outfile,color,title in iter_items:
        sub_target = sub_tensor[branch_ids,:,:,:,:]
        sub_target_combinat = np.expand_dims(sub_target.prod(axis=0), axis=0)
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
        df_dist_target = get_df_dist(sub_tensor=sub_target, g=g, mode=mode)
        df_ad = pd.merge(df_ad, df_dist, on=['group','state_from','state_to'])
        out_path = os.path.join(g['site_outdir'], outfile)
        df_ad.to_csv(out_path, sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
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
    outbase = os.path.join(g['site_outdir'], 'csubst_site.state')
    fig_path = outbase + ".pdf"
    fig.savefig(fig_path, format='pdf', transparent=True)
    plt.close(fig)
    output_paths.append(fig_path)
    return output_paths

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

def get_lineage_site_heatmap_values(df, display_meta, g):
    branch_ids = _normalize_branch_ids(g.get('branch_ids', [])).tolist()
    if len(branch_ids) == 0:
        return np.zeros((0, 0), dtype=float), []
    num_site = len(display_meta)
    values = np.full((len(branch_ids), num_site), np.nan, dtype=float)
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
            col = 'N_sub_{}'.format(int(bid))
            if col not in df.columns:
                continue
            value = float(df.at[row_index, col])
            if not np.isfinite(value):
                continue
            values[row_index_branch, col_index] = min(max(value, 0.0), 1.0)
    return values, branch_ids


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
        ax_heat.set_yticklabels([str(int(bid)) for bid in heat_branch_ids], fontsize=font_size-1)
        for tick,bid in zip(ax_heat.get_yticklabels(), heat_branch_ids):
            tick.set_color(branch_color_by_id.get(int(bid), 'black'))
    else:
        ax_heat.set_yticks([])
    ax_heat.tick_params(axis='x', length=0, labelbottom=False, bottom=False, top=False, labeltop=False)
    ax_heat.tick_params(axis='y', length=0, pad=1)
    ax_heat.set_ylabel('Branch ID', fontsize=font_size-1)
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
    # Keep the scale fixed at 0-1 and place a compact bar near the heatmap edge.
    cax = ax_cb_holder.inset_axes([0.74, 0.68, 0.14, 0.09])
    cbar = fig.colorbar(scalar, cax=cax, orientation='horizontal')
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.tick_params(labelsize=font_size-1, length=2, labeltop=True, labelbottom=False, pad=1)
    ax_cb_holder.text(
        0.81,
        0.64,
        'Substitution\nposterior\nprobability',
        transform=ax_cb_holder.transAxes,
        ha='center',
        va='top',
        fontsize=font_size-2,
        color='black',
    )
    return cbar


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

def get_leaf_state_letter(g, leaf_id, codon_site_alignment):
    site_index = int(codon_site_alignment) - 1
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

def plot_tree_site(df, g):
    if not bool(g.get('tree_site_plot', True)):
        print('Skipping tree + site summary outputs (--tree_site_plot no).', flush=True)
        return []
    tree_site_df,min_prob = classify_tree_site_categories(df=df, g=g)
    display_meta = get_tree_site_display_sites(tree_site_df=tree_site_df, g=g, df=df)
    xcoord,ycoord,leaf_order = get_tree_plot_coordinates(tree=g['tree'])
    branch_ids_in_order = _normalize_branch_ids(g['branch_ids']).tolist()
    branch_ids = set(branch_ids_in_order)
    mode = str(g.get('mode', '')).lower()
    show_branch_heatmap = mode in ('lineage', 'set')
    if mode == 'lineage':
        branch_color_by_id = _get_lineage_rgb_by_branch(branch_ids=branch_ids_in_order, g=g)
    else:
        branch_color_by_id = {int(bid): 'firebrick' for bid in branch_ids_in_order}
    highlight_leaf_ids,highlight_branch_ids = get_highlight_leaf_and_branch_ids(tree=g['tree'], branch_ids=branch_ids)
    x_values = np.array(list(xcoord.values()), dtype=float)
    x_max = x_values.max() if x_values.shape[0] else 1.0
    if x_max <= 0:
        x_max = 1.0
    tip_label_texts = []
    for leaf in ete.iter_leaves(g['tree']):
        leaf_id = int(ete.get_prop(leaf, "numerical_label"))
        tip_label_texts.append((leaf.name or '') + '|' + str(leaf_id))
    max_tip_label_chars = max([len(txt) for txt in tip_label_texts]) if len(tip_label_texts) > 0 else 1

    num_display_site = max(len(display_meta), 1)
    num_leaf = max(len(leaf_order), 1)
    # Dense defaults for compact tree/site output.
    base_tree_panel_width = min(max(5.0, 4.2 + x_max * 0.42), 10.0) * 0.5
    tip_label_width_in = min(max(0.8, max_tip_label_chars * 0.053), 2.8)
    tree_panel_width = base_tree_panel_width + tip_label_width_in
    # Keep a constant physical width per displayed alignment column.
    site_column_width_in = 0.112
    site_panel_width = max(site_column_width_in, num_display_site * site_column_width_in)
    fig_width = tree_panel_width + site_panel_width
    fig_height = min(max(2.5, num_leaf * 0.13 + 0.55), 8.5)
    fg_color = 'firebrick'
    bg_branch_color = '#4d4d4d'
    bg_label_color = '#5f6f7f'
    internal_label_color = '#7a7a7a'
    internal_label_size = 4.2

    if show_branch_heatmap:
        heat_panel_height = min(max(0.55, len(branch_ids_in_order) * 0.12), 1.9)
        fig_height += heat_panel_height
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[tree_panel_width, site_panel_width],
            height_ratios=[heat_panel_height, fig_height - heat_panel_height],
            wspace=0.01,
            hspace=0.14,
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
            linewidth = 2.0 if is_target else 0.8
            ax_tree.plot([xcoord[node_id], xcoord[node_id]], [ycoord[node_id], ycoord[child_id]],
                         color=color, linewidth=linewidth, zorder=1, solid_capstyle=TREE_LINE_CAPSTYLE)

    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        node_id = int(ete.get_prop(node, "numerical_label"))
        parent_id = int(ete.get_prop(node.up, "numerical_label"))
        is_target = node_id in highlight_branch_ids
        color = branch_color_by_id.get(node_id, fg_color) if is_target else bg_branch_color
        linewidth = 2.0 if is_target else 0.8
        ax_tree.plot([xcoord[parent_id], xcoord[node_id]], [ycoord[node_id], ycoord[node_id]],
                     color=color, linewidth=linewidth, zorder=2, solid_capstyle=TREE_LINE_CAPSTYLE)

    root = ete.get_tree_root(g['tree'])
    root_id = int(ete.get_prop(root, "numerical_label"))
    root_stub = max(x_max * 0.03, 0.03)
    root_color = fg_color if (root_id in highlight_branch_ids) else bg_branch_color
    ax_tree.plot([-root_stub, xcoord[root_id]], [ycoord[root_id], ycoord[root_id]],
                 color=root_color, linewidth=0.8, zorder=2, solid_capstyle=TREE_LINE_CAPSTYLE)

    internal_label_offset = max(x_max * 0.008, 0.008)
    for node in g['tree'].traverse():
        if ete.is_leaf(node):
            continue
        node_id = int(ete.get_prop(node, "numerical_label"))
        if node_id in highlight_branch_ids:
            node_color = branch_color_by_id.get(node_id, fg_color)
        else:
            node_color = internal_label_color
        ax_tree.text(xcoord[node_id] + internal_label_offset, ycoord[node_id] - 0.08, str(node_id),
                     va='center', ha='left', fontsize=internal_label_size, color=node_color, zorder=4)

    # Keep labels close to the tree and size the panel by label text length.
    label_offset = (x_max * 0.02 + 0.04) * 0.85
    for leaf in ete.iter_leaves(g['tree']):
        node_id = int(ete.get_prop(leaf, "numerical_label"))
        label = (leaf.name or '') + '|' + str(node_id)
        if mode == 'lineage':
            label_color = bg_label_color
        else:
            is_target_leaf = node_id in highlight_leaf_ids
            label_color = fg_color if is_target_leaf else bg_label_color
        ax_tree.text(x_max + label_offset, ycoord[node_id], label, va='center', ha='left',
                     fontsize=font_size, color=label_color)

    if len(leaf_order):
        ax_tree.set_ylim(len(leaf_order)-0.5, -0.5)
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
    if mode == 'set':
        mode_expression = str(g.get('mode_expression', '')).strip()
        set_stat_type = str(g.get('set_stat_type', '')).strip()
        if mode_expression != '':
            if set_stat_type != '':
                title_text += '; Operation: {} ({})'.format(mode_expression, set_stat_type)
            else:
                title_text += '; Operation: {}'.format(mode_expression)
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

    if len(display_meta) == 0:
        ax_site.set_xlim(-0.5, 0.5)
    else:
        ax_site.set_xlim(-0.5, len(display_meta)-0.5)
    if len(leaf_order):
        ax_site.set_ylim(len(leaf_order)-0.5, -0.5)
    tick_positions = [i for i,item in enumerate(display_meta) if item['site'] is not None]
    tick_labels = [str(display_meta[i]['site']) for i in tick_positions]
    ax_site.set_xticks(tick_positions)
    ax_site.xaxis.tick_top()
    if mode == 'lineage':
        # Keep labels separated from the alignment letters.
        ax_site.tick_params(axis='x', length=0, pad=1)
    else:
        ax_site.tick_params(axis='x', length=0, pad=1)
    ax_site.set_xticklabels(tick_labels, rotation=90, fontsize=font_size)
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
        ax_heat.set_xlim(ax_site.get_xlim())

    overflow_count = get_tree_site_overflow_count(
        tree_site_df=tree_site_df,
        display_meta=display_meta,
        g=g,
        df=df,
    )
    if overflow_count > 0:
        overflow_label = '+{} sites with PP > {:.2f}'.format(int(overflow_count), float(min_prob))
        if show_branch_heatmap:
            # Place overflow text right below the heatmap panel.
            ax_heat.text(
                0.995,
                -0.02,
                overflow_label,
                transform=ax_heat.transAxes,
                ha='right',
                va='top',
                fontsize=font_size,
                color='black',
                fontweight='bold',
                clip_on=False,
            )
        else:
            ax_site.text(
                0.995,
                0.005,
                overflow_label,
                transform=ax_site.transAxes,
                ha='right',
                va='bottom',
                fontsize=font_size,
                color='black',
                fontweight='bold',
            )

    category_columns = {'convergent': list(), 'divergent': list()}
    for i,item in enumerate(display_meta):
        if item['site'] is None:
            continue
        category = str(item.get('category', ''))
        if category in category_columns:
            category_columns[category].append(i)
    text_transform = matplotlib.transforms.blended_transform_factory(ax_site.transData, ax_site.transAxes)
    min_prob_text = '{:g}'.format(float(min_prob))
    if len(category_columns['convergent']) > 0:
        x_center = (min(category_columns['convergent']) + max(category_columns['convergent'])) / 2
        conv_count = len(category_columns['convergent'])
        ax_site.text(x_center, 1.11, 'Convergent sites\n(N={:,}, PP \u2265 {})'.format(conv_count, min_prob_text), transform=text_transform,
                     va='bottom', ha='center', fontsize=font_size, color='black', fontweight='bold')
    if len(category_columns['divergent']) > 0:
        x_center = (min(category_columns['divergent']) + max(category_columns['divergent'])) / 2
        div_count = len(category_columns['divergent'])
        ax_site.text(x_center, 1.11, 'Divergent sites\n(N={:,}, PP \u2265 {})'.format(div_count, min_prob_text), transform=text_transform,
                     va='bottom', ha='center', fontsize=font_size, color='black', fontweight='bold')

    if show_branch_heatmap:
        fig.subplots_adjust(top=0.94, left=0.04, right=0.99, wspace=0.01, hspace=0.14)
    else:
        fig.subplots_adjust(top=0.86, left=0.04, right=0.99, wspace=0.01)

    fmt = str(g.get('tree_site_plot_format', 'pdf')).lower()
    fig_path = os.path.join(g['site_outdir'], 'csubst_site.tree_site.' + fmt)
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
    table_path = os.path.join(g['site_outdir'], 'csubst_site.tree_site.tsv')
    tree_site_df.to_csv(table_path, sep='\t', index=False, float_format=g['float_format'], chunksize=10000)
    print('Writing tree + site category table: {}'.format(table_path), flush=True)
    return [fig_path, table_path]

def initialize_site_df(num_site):
    df = pd.DataFrame()
    df.loc[:,'codon_site_alignment'] = np.arange(num_site)
    df.loc[:,'nuc_site_alignment'] = ((df.loc[:,'codon_site_alignment']+1) * 3) - 2
    return df

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
        sub_probs = sub_tensor[branch_id,:,:,:,:].sum(axis=(1,2,3))
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
    allowed = ('any2any', 'any2spe', 'spe2any', 'spe2spe')
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


def _get_set_stat_channels_from_branch_tensor(branch_tensor, set_stat_type):
    arr = np.nan_to_num(np.asarray(branch_tensor, dtype=float), nan=0.0)
    if arr.ndim != 4:
        raise ValueError('Branch substitution tensor should be 4D [site,group,anc,des].')
    if set_stat_type == 'any2any':
        out = arr.sum(axis=(1, 2, 3))
        return out[:, np.newaxis]
    if set_stat_type == 'any2spe':
        return arr.sum(axis=(1, 2))
    if set_stat_type == 'spe2any':
        return arr.sum(axis=(1, 3))
    if set_stat_type == 'spe2spe':
        return arr.reshape(arr.shape[0], -1)
    raise ValueError('Unsupported set substitution type: {}'.format(set_stat_type))


def _get_empty_set_channel_prob(n_site, set_stat_type, ON_tensor=None):
    if ON_tensor is None:
        return np.zeros(shape=(n_site, 1), dtype=float)
    if set_stat_type == 'any2any':
        n_channel = 1
    elif set_stat_type == 'any2spe':
        n_channel = int(ON_tensor.shape[4])
    elif set_stat_type == 'spe2any':
        n_channel = int(ON_tensor.shape[3])
    elif set_stat_type == 'spe2spe':
        n_channel = int(ON_tensor.shape[2] * ON_tensor.shape[3] * ON_tensor.shape[4])
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
    if (ON_tensor is None) and (set_stat_type != 'any2any'):
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
                branch_tensor = ON_tensor[bid, :, :, :, :]
                n_sub_prob = _get_set_stat_channels_from_branch_tensor(
                    branch_tensor=branch_tensor,
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
                    other_tensor = ON_tensor[int(other_bid), :, :, :, :]
                    other_prob_rows.append(
                        _get_set_stat_channels_from_branch_tensor(
                            branch_tensor=other_tensor,
                            set_stat_type=set_stat_type,
                        )
                    )
                n_other_prob_matrix = np.stack(other_prob_rows, axis=0).max(axis=0)
                other_bool_matrix = (n_other_prob_matrix >= min_single_prob)
                if OS_tensor is not None:
                    other_syn_probs = OS_tensor[other_branch_ids, :, :, :, :].sum(axis=(2, 3, 4))
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
            txt = '--mode set expects --mode "set,<substitution_type>,<expression>", e.g., --mode "set,any2spe,1|3".'
            raise ValueError(txt)
        set_stat_type = _get_set_mode_stat_type(parts[1])
        mode_expression = parts[2]
        if mode_expression == '':
            raise ValueError('--mode set expression is empty.')
    elif len(parts) >= 2:
        mode_expression = ','.join(parts[1:]).strip()
    return mode, mode_expression, set_stat_type


def _build_site_outdir(mode, branch_txt, lineage_input_branch_txt=None, mode_expression=None, set_stat_type=None):
    if mode == 'intersection':
        return './csubst_site.branch_id' + branch_txt
    if mode == 'lineage':
        return './csubst_site.lineage.branch_id' + lineage_input_branch_txt
    if mode == 'set':
        if set_stat_type is None:
            raise ValueError('Missing set substitution type for --mode set.')
        mode_expr_label = _get_set_expression_label(mode_expression)
        return './csubst_site.set.' + str(set_stat_type) + '.expr' + mode_expr_label
    return './csubst_site.mode' + mode + '.branch_id' + branch_txt


def resolve_site_jobs(g):
    raw_mode = str(g.get('mode', 'intersection')).strip()
    mode, mode_expression, set_stat_type = _parse_mode_and_expression(raw_mode)
    g['mode'] = mode
    g['mode_expression'] = mode_expression
    g['set_stat_type'] = set_stat_type
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
            raise ValueError('--mode set expects an expression, e.g., --mode "set,any2any,1|5".')
        if set_stat_type is None:
            raise ValueError('--mode set expects a substitution type, e.g., --mode "set,any2any,1|5".')
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

def main_site(g):
    if g['pdb'] is not None:
        from csubst import parser_pymol
    print("Reading and parsing input files.", flush=True)
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['genetic_code'])
    g = tree.read_treefile(g)
    g = parser_misc.generate_intermediate_files(g)
    g = parser_misc.annotate_tree(g)
    g = parser_misc.read_input(g)
    g = parser_misc.prep_state(g)
    ON_tensor = substitution.get_substitution_tensor(state_tensor=g['state_pep'], mode='asis', g=g, mmap_attr='N')
    ON_tensor = substitution.apply_min_sub_pp(g, ON_tensor)
    OS_tensor = substitution.get_substitution_tensor(state_tensor=g['state_cdn'], mode='syn', g=g, mmap_attr='S')
    OS_tensor = substitution.apply_min_sub_pp(g, OS_tensor)
    g = resolve_site_jobs(g)
    for site_job in g['site_jobs']:
        branch_ids = _normalize_branch_ids(site_job['branch_ids'])
        g['single_branch_mode'] = site_job['single_branch_mode']
        g['branch_ids'] = branch_ids
        g['site_outdir'] = site_job['site_outdir']
        g['mode_expression'] = site_job.get('mode_expression', g.get('mode_expression', None))
        g['set_stat_type'] = site_job.get('set_stat_type', g.get('set_stat_type', None))
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
        if (g['untrimmed_cds'] is not None):
            df = add_gene_index(df, g)
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
            g['pdb_outfile_base'] = os.path.join(g['site_outdir'], 'csubst_site.' + id_base)
            parser_pymol.initialize_pymol(pdb_id=g['pdb'])
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
                pymol_pdf_path = os.path.join(g['site_outdir'], f'csubst_site.{id_base}.pymol.pdf')
                parser_pymol.save_6view_pdf(pdf_filename=pymol_pdf_path)
                add_site_output_manifest_row(
                    manifest_rows=manifest_rows,
                    output_path=pymol_pdf_path,
                    output_kind='pymol_summary_pdf',
                    g=g,
                    branch_ids=g['branch_ids'],
                )
        barchart_path = plot_barchart(df, g)
        add_site_output_manifest_row(
            manifest_rows=manifest_rows,
            output_path=barchart_path,
            output_kind='site_summary_pdf',
            g=g,
            branch_ids=g['branch_ids'],
        )
        if should_plot_state(g):
            state_paths = plot_state(ON_tensor, OS_tensor, g['branch_ids'], g)
            if len(state_paths):
                for state_path in state_paths:
                    file_name = os.path.basename(state_path)
                    if file_name == 'csubst_site.state.pdf':
                        output_kind = 'state_pattern_pdf'
                    elif file_name == 'csubst_site.state_N.tsv':
                        output_kind = 'state_pattern_nonsyn_tsv'
                    elif file_name == 'csubst_site.state_S.tsv':
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
                    output_path=os.path.join(g['site_outdir'], 'csubst_site.state.pdf'),
                    output_kind='state_pattern_pdf',
                    g=g,
                    branch_ids=g['branch_ids'],
                    note='skipped_by_site_state_plot',
                )
                add_site_output_manifest_row(
                    manifest_rows=manifest_rows,
                    output_path=os.path.join(g['site_outdir'], 'csubst_site.state_N.tsv'),
                    output_kind='state_pattern_nonsyn_tsv',
                    g=g,
                    branch_ids=g['branch_ids'],
                    note='skipped_by_site_state_plot',
                )
                add_site_output_manifest_row(
                    manifest_rows=manifest_rows,
                    output_path=os.path.join(g['site_outdir'], 'csubst_site.state_S.tsv'),
                    output_kind='state_pattern_syn_tsv',
                    g=g,
                    branch_ids=g['branch_ids'],
                    note='skipped_by_site_state_plot',
                )
        else:
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=os.path.join(g['site_outdir'], 'csubst_site.state.pdf'),
                output_kind='state_pattern_pdf',
                g=g,
                branch_ids=g['branch_ids'],
                note='skipped_by_mode',
            )
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=os.path.join(g['site_outdir'], 'csubst_site.state_N.tsv'),
                output_kind='state_pattern_nonsyn_tsv',
                g=g,
                branch_ids=g['branch_ids'],
                note='skipped_by_mode',
            )
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=os.path.join(g['site_outdir'], 'csubst_site.state_S.tsv'),
                output_kind='state_pattern_syn_tsv',
                g=g,
                branch_ids=g['branch_ids'],
                note='skipped_by_mode',
            )
        tree_paths = plot_tree_site(df, g)
        if len(tree_paths):
            for tree_path in tree_paths:
                file_name = os.path.basename(tree_path)
                if file_name.startswith('csubst_site.tree_site.') and file_name.endswith('.tsv'):
                    output_kind = 'tree_site_table_tsv'
                elif file_name.startswith('csubst_site.tree_site.'):
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
                output_path=os.path.join(g['site_outdir'], 'csubst_site.tree_site.' + tree_format),
                output_kind='tree_site_plot',
                g=g,
                branch_ids=g['branch_ids'],
                note='skipped_by_tree_site_plot',
            )
            add_site_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=os.path.join(g['site_outdir'], 'csubst_site.tree_site.tsv'),
                output_kind='tree_site_table_tsv',
                g=g,
                branch_ids=g['branch_ids'],
                note='skipped_by_tree_site_plot',
            )
        if g['pdb'] is None:
            outbase = os.path.join(g['site_outdir'], 'csubst_site')
        else:
            outbase = g['pdb_outfile_base']
        if str(g.get('mode', '')).lower() == 'lineage':
            plot_lineage_tree(g=g, outbase=outbase)
        out_path = outbase+'.tsv'
        if g['single_branch_mode']:
            df = combinatorial2single_columns(df)
        df.to_csv(out_path, sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        add_site_output_manifest_row(
            manifest_rows=manifest_rows,
            output_path=out_path,
            output_kind='site_table_tsv',
            g=g,
            branch_ids=g['branch_ids'],
        )
        if bool(g.get('site_output_manifest', True)):
            write_site_output_manifest(manifest_rows=manifest_rows, g=g, branch_ids=g['branch_ids'])
        else:
            print('Skipping site output manifest (--site_output_manifest no).', flush=True)
    print('To visualize the convergence probability on protein structure, please see: https://github.com/kfuku52/csubst/wiki')
    print('')
    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]
    return None

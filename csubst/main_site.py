import numpy
import matplotlib
import matplotlib.pyplot
import pandas

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

def get_state(node, g):
    seq = ete.get_prop(node, 'sequence', '').upper()
    assert seq != '', 'Leaf sequence not found for node "{}". Check tree/alignment labels.'.format(node.name)
    assert len(seq)%3==0, 'Sequence length is not multiple of 3. Node name = '+node.name
    state_matrix = numpy.zeros([g['num_input_site'], g['num_input_state']], dtype=g['float_type'])
    for s in numpy.arange(g['num_input_site']):
        codon = seq[(s*3):((s+1)*3)]
        codon_index = sequence.get_state_index(state=codon, input_state=g['codon_orders'], ambiguous_table=genetic_code.ambiguous_table)
        for ci in codon_index:
            state_matrix[s,ci] = 1/len(codon_index)
    return(state_matrix)

def add_gapline(df, gapcol, xcol, yvalue, lw, ax):
    x_values = df.loc[:,xcol].values - 0.5
    y_values = numpy.ones(x_values.shape) * yvalue
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
    i_ranges = numpy.arange(len(x_values))
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
    for i in numpy.arange(len(bars['x_start'])):
        y = bars['y'][i]
        x_start = bars['x_start'][i]
        x_end = bars['x_end'][i]
        color = bars['color'][i]
        ax.hlines(y=y, xmin=x_start, xmax=x_end, linewidth=lw, color=color, zorder=0)

def get_yvalues(df, sub_type, SN):
    col = SN + sub_type
    if sub_type == '_sub':
        if SN == 'S':
            yvalues = df.loc[:, 'S' + sub_type].to_numpy(copy=True)
            is_enough_value = (yvalues > 0.01)
            yvalues[is_enough_value] = df.loc[is_enough_value, ['N' + sub_type, 'S' + sub_type]].sum(axis=1).values
        elif SN == 'N':
            yvalues = df.loc[:, col].to_numpy(copy=True)
    elif sub_type == '_set_expr':
        if SN == 'S':
            yvalues = numpy.zeros(df.shape[0], dtype=float)
        elif SN == 'N':
            if 'N_set_expr_prob' in df.columns:
                yvalues = df.loc[:, 'N_set_expr_prob'].to_numpy(copy=True)
            elif 'N_set_expr' in df.columns:
                yvalues = df.loc[:, 'N_set_expr'].to_numpy(copy=True).astype(float)
            else:
                yvalues = numpy.zeros(df.shape[0], dtype=float)
    elif sub_type == '_set_other':
        if SN == 'S':
            yvalues = numpy.zeros(df.shape[0], dtype=float)
        elif SN == 'N':
            if 'N_set_other' in df.columns:
                yvalues = df.loc[:, 'N_set_other'].to_numpy(copy=True).astype(float)
            else:
                yvalues = numpy.zeros(df.shape[0], dtype=float)
    elif sub_type.startswith('_sub_branch_'):
        branch_id_txt = sub_type.replace('_sub_branch_', '')
        branch_id = int(branch_id_txt)
        n_col = 'N_sub_{}'.format(branch_id)
        s_col = 'S_sub_{}'.format(branch_id)
        nvalues = df.loc[:, n_col].to_numpy(copy=True) if (n_col in df.columns) else numpy.zeros(df.shape[0], dtype=float)
        svalues = df.loc[:, s_col].to_numpy(copy=True) if (s_col in df.columns) else numpy.zeros(df.shape[0], dtype=float)
        if SN == 'S':
            yvalues = svalues.copy()
            is_enough_value = (yvalues > 0.01)
            yvalues[is_enough_value] = yvalues[is_enough_value] + nvalues[is_enough_value]
        elif SN == 'N':
            yvalues = nvalues
    elif sub_type=='_sub_':
        if SN == 'S':
            is_S_cols = df.columns.str.startswith('S_sub_')
            S_cols = df.columns[is_S_cols]
            is_y_cols = is_S_cols | df.columns.str.startswith('N_sub_')
            y_cols = df.columns[is_y_cols]
            yvalues = df.loc[:, S_cols].sum(axis=1).to_numpy(copy=True)
            is_enough_value = (yvalues>0.01)
            yvalues[is_enough_value] = df.loc[is_enough_value,y_cols].sum(axis=1).values
        elif SN == 'N':
            y_cols = df.columns[df.columns.str.startswith(col)]
            yvalues = df.loc[:, y_cols].sum(axis=1).values
    else:
        if SN=='S':
            yvalues = df.loc[:,['OCN'+sub_type,'OCS'+sub_type]].sum(axis=1).values
        elif SN=='N':
            yvalues = df.loc[:, 'OC'+col].values
    return yvalues


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
    branch_ids = [int(bid) for bid in numpy.asarray(g.get('branch_ids', [])).tolist()]
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
    from csubst import parser_pymol
    if 'aa_identity_means' not in g.keys():
        g = parser_pymol.calc_aa_identity(g)
    mean_keys = numpy.array(list(g['aa_identity_means'].keys()))
    mean_values = numpy.array(list(g['aa_identity_means'].values()))
    g['highest_identity_chain_name'] = mean_keys[numpy.argmax(mean_values)]
    return g

def add_substitution_labels(df, SN, sub_type, SN_colors, ax, g):
    col = 'OC'+ SN + sub_type
    df_sub = df.loc[(df[col] >= g['pymol_min_combinat_prob']), :].reset_index()
    anc_cols = df_sub.columns[df_sub.columns.str.startswith('aa_')&df_sub.columns.str.endswith('_anc')]
    des_cols = anc_cols.str.replace('_anc', '')
    x_min_dist = (df_sub.loc[:,'codon_site_alignment'].max()+1) / 35
    x_offset = (df_sub.loc[:,'codon_site_alignment'].max()+1) / 300
    for i in df_sub.index:
        x_value = df_sub.at[i,'codon_site_alignment']
        g = get_highest_identity_chain_name(g)
        chain_site = df_sub.at[i,'codon_site_pdb_'+g['highest_identity_chain_name']]
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

def get_plot_sub_types_and_colors(g):
    mode = str(g.get('mode', 'intersection')).lower()
    if mode == 'lineage':
        sub_types = {
            '_sub': 'Branch-wise\nsubstitutions\nin the entire tree',
            '_sub_': 'Branch-wise\nsubstitutions\nin the targets',
        }
        branch_ids = numpy.asarray(g.get('branch_ids', []), dtype=numpy.int64)
        branch_rgb = _get_lineage_rgb_by_branch(branch_ids=branch_ids.tolist(), g=g)
        for branch_id in branch_ids.tolist():
            key = '_sub_branch_{}'.format(int(branch_id))
            sub_types[key] = 'Substitutions in\nbranch_id {}'.format(int(branch_id))
        SN_color_all = {
            '_sub': {'N': 'black', 'S': 'gainsboro'},
            '_sub_': {'N': 'black', 'S': 'gainsboro'},
        }
        for branch_id in branch_ids.tolist():
            key = '_sub_branch_{}'.format(int(branch_id))
            SN_color_all[key] = {'N': branch_rgb[int(branch_id)], 'S': 'gainsboro'}
    elif mode == 'set':
        sub_types = {
            '_sub': 'Branch-wise\nsubstitutions\nin the entire tree',
            '_sub_': 'Branch-wise\nsubstitutions\nin the targets',
        }
        tokens = _tokenize_set_expression(g.get('mode_expression', ''))
        branch_ids = _get_set_expression_display_branch_ids(g)
        for branch_id in branch_ids.tolist():
            key = '_sub_branch_{}'.format(int(branch_id))
            sub_types[key] = 'Substitutions in\nbranch_id {}'.format(int(branch_id))
        if 'A' in tokens:
            sub_types['_set_other'] = 'Substitutions in\nA'
        mode_expression = str(g.get('mode_expression', '')).strip()
        if mode_expression == '':
            mode_expression = 'set expression'
        sub_types['_set_expr'] = 'Substitutions in\n{}'.format(mode_expression)
        SN_color_all = {
            '_sub': {'N': 'black', 'S': 'gainsboro'},
            '_sub_': {'N': 'black', 'S': 'gainsboro'},
            '_set_other': {'N': 'black', 'S': 'gainsboro'},
            '_set_expr': {'N': 'red', 'S': 'gainsboro'},
        }
        for branch_id in branch_ids.tolist():
            key = '_sub_branch_{}'.format(int(branch_id))
            SN_color_all[key] = {'N': 'black', 'S': 'gainsboro'}
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

def plot_barchart(df, g):
    sub_types,SN_color_all = get_plot_sub_types_and_colors(g)
    num_row = len(sub_types)
    fig,axes = matplotlib.pyplot.subplots(nrows=num_row, ncols=1, figsize=(7.2, 1.2*len(sub_types)), sharex=True)
    axes = axes.flat
    i = 0
    NS_ymax = df.loc[:,['N_sub','S_sub']].sum(axis=1).max() + 0.5
    for sub_type in sub_types.keys():
        SN_colors = SN_color_all[sub_type]
        ylabel = sub_types[sub_type]
        ax = axes[i]
        for SN in ['S','N']:
            if sub_type=='_sub':
                yvalues = get_yvalues(df, sub_type, SN)
                ax.set_ylim(0, NS_ymax)
                add_gapline(df=df, gapcol='gap_rate_all', xcol='codon_site_alignment', yvalue=NS_ymax*0.95, lw=3, ax=ax)
            elif sub_type=='_sub_':
                yvalues = get_yvalues(df, sub_type, SN)
                ymax = df.columns.str.startswith('N_sub_').sum()
                ax.set_ylim(0, ymax)
                add_gapline(df=df, gapcol='gap_rate_target', xcol='codon_site_alignment', yvalue=ymax*0.95, lw=3, ax=ax)
            elif sub_type.startswith('_sub_branch_'):
                yvalues = get_yvalues(df, sub_type, SN)
                ax.set_ylim(0, 1.0)
                add_gapline(df=df, gapcol='gap_rate_target', xcol='codon_site_alignment', yvalue=0.95, lw=3, ax=ax)
            elif sub_type == '_set_expr':
                yvalues = get_yvalues(df, sub_type, SN)
                ymax = max(float(df.loc[:, 'N_set_expr_prob'].max()) if ('N_set_expr_prob' in df.columns) else 1.0, 1.0)
                ax.set_ylim(0, ymax)
                add_gapline(df=df, gapcol='gap_rate_target', xcol='codon_site_alignment', yvalue=ymax*0.95, lw=3, ax=ax)
            elif sub_type == '_set_other':
                yvalues = get_yvalues(df, sub_type, SN)
                ax.set_ylim(0, 1.0)
                add_gapline(df=df, gapcol='gap_rate_all', xcol='codon_site_alignment', yvalue=0.95, lw=3, ax=ax)
            else:
                yvalues = get_yvalues(df, sub_type, SN)
                ax.set_ylim(0, 1)
                ax.axhline(y=0.5, linestyle='--', linewidth=0.5, color='black', zorder=0)
                add_gapline(df=df, gapcol='gap_rate_target', xcol='codon_site_alignment', yvalue=0.95, lw=3, ax=ax)
                if (SN=='N')&(g['pdb'] is not None):
                    ax = add_substitution_labels(df, SN, sub_type, SN_colors, ax, g)
            ax.set_ylabel(ylabel, fontsize=font_size)
            xy = pandas.DataFrame({'x':df.loc[:, 'codon_site_alignment'].values, 'y':yvalues})
            xy2 = xy.loc[(xy['y']>0.01),:]
            ax.bar(xy2['x'], xy2['y'], color=SN_colors[SN])
            if (i==num_row-1):
                ax.set_xlabel('Aligned codon site', fontsize=font_size)
            else:
                ax.set_xlabel('', fontsize=font_size)
            ax.set_xlim(df.loc[:,'codon_site_alignment'].min()-0.5, df.loc[:,'codon_site_alignment'].max()+0.5)
        i += 1
    if str(g.get('mode', '')).lower() == 'lineage':
        fig.tight_layout(h_pad=0.5, w_pad=1, rect=[0, 0.09, 1, 1])
        _add_lineage_distance_colorbar(fig=fig, g=g)
    else:
        fig.tight_layout(h_pad=0.5, w_pad=1)
    if g['pdb'] is None:
        outbase = os.path.join(g['site_outdir'], 'csubst_site')
    else:
        outbase = g['pdb_outfile_base']
    fig.savefig(outbase+".pdf", format='pdf', transparent=True)
    #fig.savefig(outbase+".svg", format='svg', transparent=True)
    print("Nonsynonymous and synonymous substitutions are shown in color and gray, respectively.", flush=True)
    print("Alignment gap sites are indicated by gray scale (0% missing = white, 100% missing = black).", flush=True)


def plot_lineage_tree(g, outbase):
    if str(g.get('mode', '')).lower() != 'lineage':
        return None
    branch_ids = numpy.asarray(g.get('branch_ids', []), dtype=numpy.int64)
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
        'fg_df': pandas.DataFrame(columns=['name', 'PLACEHOLDER']),
    }
    tree.plot_branch_category(g=plot_g, file_base=outbase+'.tree', label='all')
    return None


def get_gapsite_rate(state_tensor):
    num_gapsite = (state_tensor.sum(axis=2)==0).sum(axis=0)
    gapsite_rate = num_gapsite / state_tensor.shape[0]
    return gapsite_rate

def extend_site_index_edge(sites, num_extend):
    new_sites = sites.copy()
    to_append_base = pandas.Series(-1 - numpy.arange(num_extend))
    for i in sites.index[1:]:
        if sites.loc[i]-1 == sites.loc[i-1]:
            continue
        to_append = to_append_base + sites.loc[i]
        new_sites = pandas.concat([new_sites, to_append], ignore_index=True)
    new_sites = new_sites.loc[new_sites>=0]
    new_sites = new_sites.drop_duplicates().sort_values().reset_index(drop=True)
    return new_sites

def add_gene_index(df, g):
    seqs = sequence.read_fasta(path=g['untrimmed_cds'])
    num_site = g['state_cdn'].shape[1]
    for leaf in ete.iter_leaves(g['tree']):
        leaf_nn = ete.get_prop(leaf, "numerical_label")
        if leaf.name not in seqs.keys():
            continue
        print('Matching untrimmed CDS sequence: {}'.format(leaf.name), flush=True)
        seq = seqs[leaf.name]
        seq = seq.replace('-','')
        num_gene_site = int(len(seq)/3)
        gene_sites = numpy.arange(num_gene_site)
        aln_sites = numpy.arange(num_site)
        col_leaf = 'codon_site_'+leaf.name
        cols = ['codon_site_alignment',col_leaf]
        aln_gene_match = pandas.DataFrame(-1, index=aln_sites, columns=cols)
        aln_gene_match.loc[:,'codon_site_alignment'] = aln_sites
        window_sizes = [100,50,10,5,4,3,2,1]
        window_sizes = [ w for w in window_sizes if (w<num_gene_site)&(w<num_site) ]
        for window_size in window_sizes:
            step_size = max([int(window_size/5),1])
            is_unassigned = (aln_gene_match.loc[:,col_leaf]==-1)
            unassigned_aln_sites = aln_gene_match.loc[is_unassigned,'codon_site_alignment']
            assigned_gene_sites = aln_gene_match.loc[~is_unassigned,col_leaf]
            unassinged_gene_sites = set(gene_sites) - set(assigned_gene_sites)
            unassinged_gene_sites = pandas.Series(sorted(list(unassinged_gene_sites)))
            extended_unassinged_gene_sites = extend_site_index_edge(unassinged_gene_sites, window_size)
            txt = 'Window size = {:,}, Number of unassigned alignment site = {:,}'
            print(txt.format(window_size, unassigned_aln_sites.shape[0]), flush=True)
            for k,uas in enumerate(unassigned_aln_sites):
                if k!=0:
                    if uas < unassigned_aln_sites.iloc[k-1]+step_size:
                        continue
                if (uas+window_size>num_site):
                    break
                for ugs in extended_unassinged_gene_sites:
                    if (ugs+window_size>num_gene_site):
                        break
                    window_match_flag = True
                    for window_index in numpy.arange(window_size):
                        codon = seq[((ugs+window_index)*3):((ugs+window_index+1)*3)]
                        codon_index = sequence.get_state_index(codon, g['codon_orders'], genetic_code.ambiguous_table)
                        if len(codon_index)==0:
                            window_match_flag = False # codon may be a stop.
                            break
                        ci = codon_index[0] # Take the first codon if ambiguous
                        if g['state_cdn'][leaf_nn,uas+window_index,ci]!=0:
                            continue
                        else:
                            window_match_flag = False
                            break
                    if window_match_flag:
                        window_aln_index = numpy.arange(uas, uas+window_size)
                        window_gene_index = numpy.arange(ugs, ugs+window_size)
                        following_gene_index = aln_gene_match.loc[window_aln_index.max():,col_leaf]
                        min_following_gene_index = following_gene_index.loc[following_gene_index!=-1].min()
                        does_smaller_gene_index_follow = (min_following_gene_index<window_gene_index.max())
                        if does_smaller_gene_index_follow:
                            continue
                        aln_gene_match.loc[window_aln_index,col_leaf] = window_gene_index
                        break
        has_gene_site_in_aln_value = (g['state_cdn'][leaf_nn,:,:].sum(axis=1)>0)
        num_gene_site_in_aln = has_gene_site_in_aln_value.sum()
        is_unassigned = (aln_gene_match.loc[:,col_leaf]==-1)
        txt = 'End. Unassigned alignment site = {:,}, Assigned alignment site = {:,}, '
        txt += 'Alignment site with non-missing gene states: {:,}'
        print(txt.format(is_unassigned.sum(), (~is_unassigned).sum(), num_gene_site_in_aln), flush=True)
        if (~is_unassigned).sum()!=num_gene_site_in_aln:
            gene_site_in_aln = set(aln_sites[has_gene_site_in_aln_value])
            gene_site_assigned = set(aln_gene_match.loc[~is_unassigned,'codon_site_alignment'])
            only_in_aln = sorted(list(gene_site_in_aln - gene_site_assigned))
            only_in_assigned = sorted(list(gene_site_assigned - gene_site_in_aln))
            txt_base = 'Sites only present in '
            print(txt_base+'input alignment: {}'.format(','.join([str(v) for v in only_in_aln])), flush=True)
            print(txt_base+'untrimmed CDS: {}'.format(','.join([str(v) for v in only_in_assigned])), flush=True)
        df = pandas.merge(df, aln_gene_match, on='codon_site_alignment', how='left')
        print('', flush=True)
    return df

def write_fasta(file, label, seq):
    with open(file, 'w') as f:
        f.write('>'+label+'\n')
        f.write(seq+'\n')

def translate(seq, g):
    translated_seq = ''
    num_site = int(len(seq)/3)
    for s in numpy.arange(num_site):
        codon = seq[(s*3):((s+1)*3)]
        for aa in g['matrix_groups'].keys():
            if codon in g['matrix_groups'][aa]:
                translated_seq += aa
                break
    return translated_seq

def export2chimera(df, g):
    header='attribute: condivPP\nmatch mode: 1-to-1\nrecipient: residues\nnone handling: None\n'
    seqs = sequence.read_fasta(path=g['untrimmed_cds'])
    for seq_key in seqs.keys():
        codon_site_col = 'codon_site_'+seq_key
        if codon_site_col not in df.columns:
            print('Sequence not be found in csubst inputs. Skipping: {}'.format(seq_key))
            continue
        seq = seqs[seq_key]
        seq_num_site = int(len(seq)/3)
        seq_sites = numpy.arange(1, seq_num_site+1)
        file_name = os.path.join(g['site_outdir'], 'csubst_site_'+seq_key+'.chimera.txt')
        txt = 'Writing a file that can be loaded to UCSF Chimera from ' \
              '"Tools -> Structure Analysis -> Define Attribute"'
        print(txt.format(file_name))
        with open(file_name, 'w') as f:
            f.write(header)
            for seq_site in seq_sites:
                is_site = (df.loc[:,codon_site_col]==seq_site)
                if is_site.sum()==0:
                    Nvalue = 'None'
                    line = '	:{}	{}\n'.format(seq_site, Nvalue)
                else:
                    Nany2spe = df.loc[is_site,'OCNany2spe'].values[0]
                    Nany2dif = df.loc[is_site,'OCNany2dif'].values[0]
                    Nvalue = Nany2spe if (Nany2spe>=Nany2dif) else -Nany2dif
                    line = '	:{}	{:.4f}\n'.format(seq_site, Nvalue)
                f.write(line)
        translated_seq = translate(seq, g)
        file_fasta = os.path.join(g['site_outdir'], 'csubst_site_'+seq_key+'.fasta')
        txt = "Writing amino acid fasta that may be used as a query for homology modeling " \
              "to obtain .pdb file (e.g., with SWISS-MODEL): {}"
        print(txt.format(file_fasta))
        write_fasta(file=file_fasta, label=seq_key, seq=translated_seq)

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
            for i in df.index:
                anc_states = g['state_'+seqtype][parent_branch_ids[bid],i,:]
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
            '':numpy.nan,
        }
        df_aa_hydrophobicity_empirical = pandas.DataFrame({
            'aa':aa_hydrophobicity_empirical.keys(),
            'hydrophobicity': aa_hydrophobicity_empirical.values(),
        })
        aa_cols = df.columns[df.columns.str.startswith('aa_')]
        for aa_col in aa_cols:
            hp_col = aa_col+'_'+'hydrophobicity'
            df_aa_hydrophobicity_empirical.columns = [aa_col, hp_col]
            df = pandas.merge(df, df_aa_hydrophobicity_empirical, on=aa_col, how='left', sort=False)
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
    df_ad = pandas.DataFrame(numpy.zeros(shape=(nrow, len(cols))))
    df_ad.columns = cols
    df_ad['group'] = df_ad['group'].astype('str')
    df_ad['state_from'] = df_ad['state_from'].astype('str')
    df_ad['state_to'] = df_ad['state_to'].astype('str')
    current_row = 0
    for g in numpy.arange(gad.shape[0]):
        state_key = state_keys[g]
        for i1,state1 in enumerate(state_orders[state_key]):
            for i2,state2 in enumerate(state_orders[state_key]):
                if (i1==i2):
                    continue
                total_prob = gad[g,i1,i2]
                if (numpy.isnan(total_prob)):
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
    df_ad.loc[:,outcol] = numpy.nan
    sgad = sub_tensor.sum(axis=0)
    current_row = 0
    for g in numpy.arange(sgad.shape[1]):
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
                    x_max = x_values.max()
                    x_hat = x_values / x_max
                    value = (1-x_hat).sum() / (x_values.shape[0] - 1)
                elif (method=='hg'):
                    pi = x_values / x_values.sum()
                    value = - (pi * numpy.log2(pi)).sum()
                elif (method=='tsi'):
                    value = x_values.max() / x_values.sum()
                elif (method.startswith('rank')):
                    rank_no = int(method.replace('rank', ''))
                    temp = x_values.argsort()
                    ranks = numpy.empty_like(temp)
                    ranks[temp] = numpy.arange(len(x_values))
                    ranks = numpy.abs(ranks - ranks.max())+1
                    value = x_values[ranks==rank_no]
                df_ad.loc[current_row,outcol] = value
                current_row += 1
    return df_ad

def add_has_target_high_combinat_prob_site(df_ad, sub_tensor, g, mode):
    state_orders,state_keys = get_state_orders(g, mode)
    outcol = 'has_target_high_combinat_prob_site'
    df_ad.loc[:,outcol] = False
    sgad = sub_tensor.sum(axis=0)
    current_row = 0
    for g in numpy.arange(sgad.shape[1]):
        state_key = state_keys[g]
        for i1,state1 in enumerate(state_orders[state_key]):
            for i2,state2 in enumerate(state_orders[state_key]):
                if (i1==i2):
                    continue
                x_values = sgad[:,g,i1,i2]
                if (x_values >= 0.5).any():
                    df_ad.at[current_row,outcol] = True
                current_row += 1
    return df_ad

def get_df_dist(sub_tensor, g, mode):
    tree_dict = dict()
    for node in g['tree'].traverse():
        tree_dict[ete.get_prop(node, "numerical_label")] = node
    state_orders, state_keys = get_state_orders(g, mode)
    cols = ['group','state_from','state_to','max_dist_bl']
    inds = numpy.arange(numpy.array(sub_tensor.shape[2:]).prod()-sub_tensor.shape[4])
    df_dist = pandas.DataFrame(columns=cols, index=inds)
    bgad = sub_tensor.sum(axis=1)
    b_index = numpy.arange(bgad.shape[0])
    g_index = numpy.arange(bgad.shape[1])
    a_index = numpy.arange(bgad.shape[2])
    d_index = numpy.arange(bgad.shape[3])
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
            interbranch_dist = numpy.nan
        elif branch_ids.shape[0]==1:
            interbranch_dist = numpy.nan
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
    fig,axes = matplotlib.pyplot.subplots(nrows=3, ncols=2, figsize=(7.2, 7.2), sharex=False)
    outfiles = ['csubst_site.state_N.tsv', 'csubst_site.state_S.tsv']
    colors = ['red','blue']
    ax_cols = [0,1]
    titles = ['Nonsynonymous substitution','Synonymous substitution']
    iter_items = zip(ax_cols,['nsy','syn'],[ON_tensor,OS_tensor],outfiles,colors,titles)
    for ax_col,mode,sub_tensor,outfile,color,title in iter_items:
        sub_target = sub_tensor[branch_ids,:,:,:,:]
        sub_target_combinat = numpy.expand_dims(sub_target.prod(axis=0), axis=0)
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
        df_ad = pandas.merge(df_ad, df_dist, on=['group','state_from','state_to'])
        out_path = os.path.join(g['site_outdir'], outfile)
        df_ad.to_csv(out_path, sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        df_ad.loc[:,'xlabel'] = df_ad.loc[:,'state_from'] + '→' + df_ad.loc[:,'state_to']
        ax = axes[0,ax_col]
        ax.bar(df_ad.loc[:,'xlabel'], df_ad.loc[:,'all'], color='black')
        ax.bar(df_ad.loc[:,'xlabel'], df_ad.loc[:,'target'], color=color)
        ax.get_xaxis().set_ticks([])
        ax.set_xlabel('Substitution category (e.g., {})'.format(df_ad.at[0,'xlabel']), fontsize=font_size)
        ax.set_ylabel('Total substitution\nprobabilities', fontsize=font_size)
        ax.set_title(title, fontsize=font_size)
        ax = axes[1,ax_col]
        bins = numpy.arange(21)/20
        ax.hist(x=df_ad.loc[:,'site_tsi'].dropna(), bins=bins, color='black')
        is_it = (df_ad.loc[:,'has_target_high_combinat_prob_site'])
        ax.hist(x=df_ad.loc[is_it,'site_tsi'].dropna(), bins=bins, color=color)
        ax.set_xlabel('Site specificity index', fontsize=font_size)
        ax.set_ylabel('Count of\nsubstitution categories', fontsize=font_size)
        ax = axes[2,ax_col]
        bins = numpy.arange(21) / 20 * df_dist.loc[:,'max_dist_bl'].max()
        ax.hist(x=df_dist.loc[:, 'max_dist_bl'].dropna(), bins=bins, color='black')
        #ax.hist(x=df_dist_target.loc[:, 'max_dist_bl'].dropna(), bins=bins, color=color)
        ax.set_xlabel('Max inter-branch distance of substitution category', fontsize=font_size)
        ax.set_ylabel('Count of\nsubstitution categories', fontsize=font_size)
    fig.tight_layout(h_pad=0.5, w_pad=1)
    outbase = os.path.join(g['site_outdir'], 'csubst_site.state')
    fig.savefig(outbase+".pdf", format='pdf', transparent=True)

def get_tree_site_min_prob(g):
    min_prob = g.get('tree_site_plot_min_prob', -1.0)
    if min_prob is None:
        min_prob = -1.0
    min_prob = float(min_prob)
    if min_prob >= 0:
        return min_prob
    if g.get('single_branch_mode', False):
        return float(g.get('pymol_min_single_prob', 0.8))
    return float(g.get('pymol_min_combinat_prob', 0.5))

def classify_tree_site_categories(df, g):
    if 'codon_site_alignment' not in df.columns:
        raise ValueError('codon_site_alignment column is required.')
    min_prob = get_tree_site_min_prob(g)
    num_site = df.shape[0]
    if g.get('single_branch_mode', False):
        convergent_score = df.loc[:, 'N_sub'].values if 'N_sub' in df.columns else numpy.zeros(num_site)
        divergent_score = numpy.zeros(num_site, dtype=float)
    else:
        convergent_score = df.loc[:, 'OCNany2spe'].values if 'OCNany2spe' in df.columns else numpy.zeros(num_site)
        divergent_score = df.loc[:, 'OCNany2dif'].values if 'OCNany2dif' in df.columns else numpy.zeros(num_site)
    convergent_score = numpy.nan_to_num(convergent_score.astype(float), nan=0.0)
    divergent_score = numpy.nan_to_num(divergent_score.astype(float), nan=0.0)

    category = numpy.full(shape=(num_site,), fill_value='blank', dtype=object)
    is_convergent = (convergent_score >= min_prob)
    is_divergent = (divergent_score >= min_prob)
    category[is_convergent] = 'convergent'
    category[is_divergent] = 'divergent'

    is_both = is_convergent & is_divergent
    category[is_both & (convergent_score >= divergent_score)] = 'convergent'
    category[is_both & (convergent_score < divergent_score)] = 'divergent'

    out = pandas.DataFrame({
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
    max_sites = int(g.get('tree_site_plot_max_sites', 60))
    if max_sites < 1:
        max_sites = 1
    return max_sites

def get_tree_site_display_sites(tree_site_df, g):
    max_sites = get_tree_site_plot_max_sites(g)
    convergent_df = tree_site_df.loc[tree_site_df.loc[:, 'tree_site_category']=='convergent',:]
    convergent_df = convergent_df.sort_values(by=['convergent_score', 'codon_site_alignment'], ascending=[False, True])
    divergent_df = tree_site_df.loc[tree_site_df.loc[:, 'tree_site_category']=='divergent',:]
    divergent_df = divergent_df.sort_values(by=['divergent_score', 'codon_site_alignment'], ascending=[False, True])

    if (convergent_df.shape[0] + divergent_df.shape[0]) == 0:
        fallback = tree_site_df.copy()
        fallback.loc[:, 'max_score'] = fallback.loc[:, ['convergent_score', 'divergent_score']].max(axis=1)
        fallback = fallback.sort_values(by=['max_score', 'codon_site_alignment'], ascending=[False, True])
        fallback = fallback.iloc[:max_sites, :]
        display_sites = fallback.loc[:, 'codon_site_alignment'].astype(int).tolist()
        display_meta = [{'site': int(site), 'category': 'blank'} for site in display_sites]
        return display_meta

    if (convergent_df.shape[0] > 0) and (divergent_df.shape[0] > 0):
        max_conv = max(1, max_sites // 2)
        max_div = max(1, max_sites - max_conv)
    elif convergent_df.shape[0] > 0:
        max_conv = max_sites
        max_div = 0
    else:
        max_conv = 0
        max_div = max_sites

    convergent_sites = convergent_df.iloc[:max_conv, :].loc[:, 'codon_site_alignment'].astype(int).tolist()
    divergent_sites = divergent_df.iloc[:max_div, :].loc[:, 'codon_site_alignment'].astype(int).tolist()
    display_meta = [{'site': int(site), 'category': 'convergent'} for site in convergent_sites]
    if (len(convergent_sites) > 0) and (len(divergent_sites) > 0):
        display_meta.append({'site': None, 'category': 'separator'})
    display_meta += [{'site': int(site), 'category': 'divergent'} for site in divergent_sites]
    return display_meta

def get_highlight_leaf_and_branch_ids(tree, branch_ids):
    highlight_branch_ids = set()
    highlight_leaf_ids = set()
    for node in tree.traverse():
        node_id = int(ete.get_prop(node, "numerical_label"))
        if node_id not in branch_ids:
            continue
        for desc in node.traverse():
            highlight_branch_ids.add(int(ete.get_prop(desc, "numerical_label")))
        for leaf in ete.get_leaves(node):
            highlight_leaf_ids.add(int(ete.get_prop(leaf, "numerical_label")))
    return highlight_leaf_ids,highlight_branch_ids

def get_leaf_state_letter(g, leaf_id, codon_site_alignment):
    site_index = int(codon_site_alignment) - 1
    if (site_index < 0) or (site_index >= g['state_pep'].shape[1]):
        return ''
    state_values = g['state_pep'][leaf_id, site_index, :]
    if numpy.nan_to_num(state_values, nan=0.0).sum() == 0:
        return ''
    max_index = int(numpy.argmax(state_values))
    if max_index >= len(g['amino_acid_orders']):
        return ''
    return str(g['amino_acid_orders'][max_index])

def get_amino_acid_colors(g):
    tab20 = matplotlib.pyplot.get_cmap('tab20')
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
    exponent = numpy.floor(numpy.log10(target))
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
        return None
    tree_site_df,min_prob = classify_tree_site_categories(df=df, g=g)
    display_meta = get_tree_site_display_sites(tree_site_df=tree_site_df, g=g)
    xcoord,ycoord,leaf_order = get_tree_plot_coordinates(tree=g['tree'])
    branch_ids = set([int(v) for v in numpy.asarray(g['branch_ids']).tolist()])
    highlight_leaf_ids,highlight_branch_ids = get_highlight_leaf_and_branch_ids(tree=g['tree'], branch_ids=branch_ids)
    x_values = numpy.array(list(xcoord.values()), dtype=float)
    x_max = x_values.max() if x_values.shape[0] else 1.0
    if x_max <= 0:
        x_max = 1.0

    num_display_site = max(len(display_meta), 1)
    num_leaf = max(len(leaf_order), 1)
    tree_panel_width = min(max(6.4, 5.0 + x_max * 0.55), 14.0)
    site_panel_width = min(max(1.8, num_display_site * 0.15), 8.5)
    fig_width = tree_panel_width + site_panel_width
    fig_height = min(max(3.2, num_leaf * 0.18 + 0.7), 11.0)
    fg_color = 'firebrick'
    bg_branch_color = '#4d4d4d'
    bg_label_color = '#5f6f7f'
    internal_label_color = '#7a7a7a'
    internal_label_size = 4.2

    fig = matplotlib.pyplot.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, 2, width_ratios=[tree_panel_width, site_panel_width], wspace=0.01)
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_site = fig.add_subplot(gs[0, 1], sharey=ax_tree)

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
            color = fg_color if is_target else bg_branch_color
            linewidth = 2.0 if is_target else 0.8
            ax_tree.plot([xcoord[node_id], xcoord[node_id]], [ycoord[node_id], ycoord[child_id]],
                         color=color, linewidth=linewidth, zorder=1)

    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        node_id = int(ete.get_prop(node, "numerical_label"))
        parent_id = int(ete.get_prop(node.up, "numerical_label"))
        is_target = node_id in highlight_branch_ids
        color = fg_color if is_target else bg_branch_color
        linewidth = 2.0 if is_target else 0.8
        ax_tree.plot([xcoord[parent_id], xcoord[node_id]], [ycoord[node_id], ycoord[node_id]],
                     color=color, linewidth=linewidth, zorder=2)

    root = ete.get_tree_root(g['tree'])
    root_id = int(ete.get_prop(root, "numerical_label"))
    root_stub = max(x_max * 0.03, 0.03)
    root_color = fg_color if (root_id in highlight_branch_ids) else bg_branch_color
    ax_tree.plot([-root_stub, xcoord[root_id]], [ycoord[root_id], ycoord[root_id]],
                 color=root_color, linewidth=0.8, zorder=2)

    internal_label_offset = max(x_max * 0.008, 0.008)
    for node in g['tree'].traverse():
        if ete.is_leaf(node):
            continue
        node_id = int(ete.get_prop(node, "numerical_label"))
        node_color = fg_color if (node_id in highlight_branch_ids) else internal_label_color
        ax_tree.text(xcoord[node_id] + internal_label_offset, ycoord[node_id] - 0.08, str(node_id),
                     va='center', ha='left', fontsize=internal_label_size, color=node_color, zorder=4)

    label_offset = x_max * 0.02 + 0.05
    for leaf in ete.iter_leaves(g['tree']):
        node_id = int(ete.get_prop(leaf, "numerical_label"))
        label = (leaf.name or '') + '|' + str(node_id)
        is_target_leaf = node_id in highlight_leaf_ids
        label_color = fg_color if is_target_leaf else bg_label_color
        ax_tree.text(x_max + label_offset, ycoord[node_id], label, va='center', ha='left',
                     fontsize=font_size, color=label_color)

    if len(leaf_order):
        ax_tree.set_ylim(len(leaf_order)-0.5, -0.5)
    left_xlim = -root_stub * 1.5
    right_xlim = x_max * 1.30 + 0.35
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
    ax_tree.plot([scale_x_start, scale_x_end], [scale_y, scale_y], color='black', linewidth=1.0, zorder=4)
    ax_tree.plot([scale_x_start, scale_x_start], [scale_y-scale_tick, scale_y+scale_tick], color='black', linewidth=1.0, zorder=4)
    ax_tree.plot([scale_x_end, scale_x_end], [scale_y-scale_tick, scale_y+scale_tick], color='black', linewidth=1.0, zorder=4)
    ax_tree.text((scale_x_start + scale_x_end) / 2, scale_y + 0.25, '{:g}'.format(scale_length),
                 va='top', ha='center', fontsize=font_size-1, color='black')

    branch_text = ','.join([str(bid) for bid in sorted(branch_ids)])
    ax_tree.set_title('Focal branch IDs: {}'.format(branch_text), loc='left')
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

    ax_site.set_xlim(-0.5, len(display_meta)-0.5)
    if len(leaf_order):
        ax_site.set_ylim(len(leaf_order)-0.5, -0.5)
    tick_positions = [i for i,item in enumerate(display_meta) if item['site'] is not None]
    tick_labels = [str(display_meta[i]['site']) for i in tick_positions]
    ax_site.set_xticks(tick_positions)
    ax_site.set_xticklabels(tick_labels, rotation=90, fontsize=font_size)
    ax_site.xaxis.tick_top()
    ax_site.tick_params(axis='x', length=0, pad=1)
    ax_site.tick_params(axis='y', left=False, labelleft=False)
    for spine in ax_site.spines.values():
        spine.set_visible(False)

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

    fig.subplots_adjust(top=0.86, left=0.04, right=0.99, wspace=0.01)

    fmt = str(g.get('tree_site_plot_format', 'pdf')).lower()
    fig_path = os.path.join(g['site_outdir'], 'csubst_site.tree_site.' + fmt)
    fig.savefig(fig_path, format=fmt, transparent=True, dpi=300)
    matplotlib.pyplot.close(fig)
    print('Writing tree + site plot: {}'.format(fig_path), flush=True)

    tree_site_df.loc[:, 'is_plotted'] = False
    tree_site_df.loc[:, 'plot_order'] = numpy.nan
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
    return None

def initialize_site_df(num_site):
    df = pandas.DataFrame()
    df.loc[:,'codon_site_alignment'] = numpy.arange(num_site)
    df.loc[:,'nuc_site_alignment'] = ((df.loc[:,'codon_site_alignment']+1) * 3) - 2
    return df

def add_cs_info(df, branch_ids, sub_tensor, attr):
    cs = substitution.get_cs(id_combinations=branch_ids[numpy.newaxis,:], sub_tensor=sub_tensor, attr=attr)
    cs.columns = cs.columns.str.replace('site','codon_site_alignment')
    df = pandas.merge(df, cs, on='codon_site_alignment')
    df.loc[:,'OC'+attr+'any2dif'] = df.loc[:,'OC'+attr+'any2any'] - df.loc[:,'OC'+attr+'any2spe']
    return df

def add_site_info(df, sub_tensor, attr):
    s = substitution.get_s(sub_tensor, attr=attr)
    s.columns = s.columns.str.replace('site','codon_site_alignment')
    df = pandas.merge(df, s, on='codon_site_alignment')
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
        branch_ids = numpy.array([int(v) for v in values], dtype=numpy.int64)
    except ValueError as exc:
        raise ValueError('--branch_id should be a comma-delimited list of integers.') from exc
    return branch_ids


def _get_node_by_branch_id(g):
    node_by_id = dict()
    for node in g['tree'].traverse():
        branch_id = int(ete.get_prop(node, "numerical_label"))
        node_by_id[branch_id] = node
    return node_by_id


def _validate_existing_branch_ids(branch_ids, node_by_id):
    missing_ids = [int(bid) for bid in branch_ids.tolist() if int(bid) not in node_by_id]
    if len(missing_ids)>0:
        txt = '--branch_id contains unknown branch IDs: {}'
        raise ValueError(txt.format(','.join([str(bid) for bid in sorted(missing_ids)])))


def _validate_nonroot_branch_ids(branch_ids, node_by_id):
    _validate_existing_branch_ids(branch_ids, node_by_id)
    root_ids = [int(bid) for bid in branch_ids.tolist() if ete.is_root(node_by_id[int(bid)])]
    if len(root_ids)>0:
        txt = '--branch_id should not include root branch IDs: {}'
        raise ValueError(txt.format(','.join([str(bid) for bid in sorted(root_ids)])))


def _read_foreground_branch_combinations(g, node_by_id):
    cb = pandas.read_csv(g['cb_file'], sep="\t", index_col=False, header=0)
    bid_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
    if bid_cols.shape[0]==0:
        raise ValueError('No branch_id_* columns were found in --cb_file.')
    is_fg_col = cb.columns.str.startswith('is_fg')
    if is_fg_col.sum()==0:
        raise ValueError('No is_fg* columns were found in --cb_file.')
    cb_fg = cb.loc[(cb.loc[:,is_fg_col]=='Y').any(axis=1),:]
    branch_id_list = []
    for i in cb_fg.index:
        bids = cb_fg.loc[i,bid_cols].values.astype(numpy.int64)
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
    return numpy.array(lineage_branch_ids, dtype=numpy.int64)


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
    return numpy.array(branch_ids, dtype=numpy.int64)


def _get_set_expression_display_branch_ids(g):
    mode_expression = g.get('mode_expression', None)
    branch_ids = [int(bid) for bid in numpy.asarray(g.get('branch_ids', []), dtype=numpy.int64).tolist()]
    if mode_expression is None:
        return numpy.array(branch_ids, dtype=numpy.int64)
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
    return numpy.array(out, dtype=numpy.int64)


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


def _evaluate_set_expression_boolean(tokens, branch_site_bool):
    operators = ['|', '-', '&', '^']
    operand_stack = []
    operator_stack = []
    expect_operand = True
    operand_shape = None
    for value in branch_site_bool.values():
        operand_shape = value.shape
        break
    if operand_shape is None:
        raise ValueError('No branch-site values were provided for set expression evaluation.')

    def apply_top_operator():
        if len(operand_stack) < 2:
            raise ValueError('Invalid --mode set expression. Missing operand.')
        rhs = operand_stack.pop()
        lhs = operand_stack.pop()
        op = operator_stack.pop()
        if op == '|':
            out = lhs | rhs
        elif op == '-':
            out = lhs & (~rhs)
        elif op == '&':
            out = lhs & rhs
        elif op == '^':
            out = lhs ^ rhs
        else:
            raise ValueError('Invalid operator in --mode set expression: {}'.format(op))
        operand_stack.append(out)

    for token in tokens:
        if expect_operand:
            if isinstance(token, int):
                if token in branch_site_bool:
                    operand_stack.append(branch_site_bool[token].copy())
                else:
                    operand_stack.append(numpy.zeros(shape=operand_shape, dtype=bool))
                expect_operand = False
            elif token == 'A':
                if token in branch_site_bool:
                    operand_stack.append(branch_site_bool[token].copy())
                else:
                    operand_stack.append(numpy.zeros(shape=operand_shape, dtype=bool))
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
    return operand_stack[0]


def _validate_set_expression_syntax(mode_expression):
    tokens = _tokenize_set_expression(mode_expression)
    branch_ids = _extract_set_expression_branch_ids(mode_expression)
    branch_site_bool = {int(branch_id): numpy.zeros(shape=(1,), dtype=bool) for branch_id in branch_ids.tolist()}
    if 'A' in tokens:
        branch_site_bool['A'] = numpy.zeros(shape=(1,), dtype=bool)
    _evaluate_set_expression_boolean(tokens=tokens, branch_site_bool=branch_site_bool)
    return None


def add_set_mode_columns(df, g, ON_tensor=None, OS_tensor=None):
    if str(g.get('mode', '')).lower() != 'set':
        return df
    mode_expression = g.get('mode_expression', None)
    if mode_expression is None:
        raise ValueError('Missing set expression for --mode set.')
    tokens = _tokenize_set_expression(mode_expression)
    branch_ids = _extract_set_expression_branch_ids(mode_expression)
    n_site = df.shape[0]
    branch_site_bool = dict()
    for branch_id in branch_ids.tolist():
        col = 'N_sub_{}'.format(int(branch_id))
        if col in df.columns:
            branch_site_bool[int(branch_id)] = (df.loc[:, col].values >= g['pymol_min_single_prob'])
        else:
            branch_site_bool[int(branch_id)] = numpy.zeros(shape=(n_site,), dtype=bool)
    if 'A' in tokens:
        explicit_ids = set([int(bid) for bid in branch_ids.tolist()])
        other_bool = numpy.zeros(shape=(n_site,), dtype=bool)
        n_other_sum = numpy.zeros(shape=(n_site,), dtype=float)
        s_other_sum = numpy.zeros(shape=(n_site,), dtype=float)
        if ('tree' in g) and (g['tree'] is not None) and (ON_tensor is not None):
            node_by_id = _get_node_by_branch_id(g)
            other_branch_ids = sorted([
                int(bid) for bid,node in node_by_id.items()
                if (not ete.is_root(node)) and (int(bid) not in explicit_ids)
            ])
            if len(other_branch_ids) > 0:
                other_probs = ON_tensor[other_branch_ids, :, :, :, :].sum(axis=(2, 3, 4))
                if other_probs.ndim == 1:
                    other_probs = other_probs[numpy.newaxis, :]
                other_bool = (other_probs >= g['pymol_min_single_prob']).any(axis=0)
                n_other_sum = other_probs.sum(axis=0)
                if OS_tensor is not None:
                    other_syn_probs = OS_tensor[other_branch_ids, :, :, :, :].sum(axis=(2, 3, 4))
                    if other_syn_probs.ndim == 1:
                        other_syn_probs = other_syn_probs[numpy.newaxis, :]
                    s_other_sum = other_syn_probs.sum(axis=0)
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
                other_bool = (df.loc[:, other_n_cols].values >= g['pymol_min_single_prob']).any(axis=1)
                n_other_sum = df.loc[:, other_n_cols].sum(axis=1).to_numpy(copy=True)
            if len(other_s_cols) > 0:
                s_other_sum = df.loc[:, other_s_cols].sum(axis=1).to_numpy(copy=True)
        branch_site_bool['A'] = other_bool.astype(bool)
        df.loc[:, 'N_set_other'] = other_bool.astype(bool)
        df.loc[:, 'N_set_other_prob'] = n_other_sum
        df.loc[:, 'S_set_other_prob'] = s_other_sum
        # Explicit aliases for easier downstream interpretation in TSV outputs.
        df.loc[:, 'N_set_A'] = df.loc[:, 'N_set_other']
        df.loc[:, 'N_set_A_prob'] = df.loc[:, 'N_set_other_prob']
        df.loc[:, 'S_set_A_prob'] = df.loc[:, 'S_set_other_prob']
    selected = _evaluate_set_expression_boolean(tokens=tokens, branch_site_bool=branch_site_bool)
    df.loc[:, 'N_set_expr'] = selected
    df.loc[:, 'N_set_expr_prob'] = 0.0
    selected_any = selected.astype(bool)
    if selected_any.any():
        n_sub_cols = df.columns[df.columns.str.startswith('N_sub_')]
        n_extra = numpy.zeros(shape=(df.shape[0],), dtype=float)
        if ('A' in tokens) and ('N_set_other_prob' in df.columns):
            n_extra = df.loc[:, 'N_set_other_prob'].to_numpy(copy=True)
            if 'N_set_other' in df.columns:
                n_extra = n_extra * df.loc[:, 'N_set_other'].astype(float).to_numpy(copy=True)
        if n_sub_cols.shape[0] > 0:
            df.loc[selected_any, 'N_set_expr_prob'] = (
                df.loc[selected_any, n_sub_cols].sum(axis=1).values + n_extra[selected_any]
            )
        elif ('A' in tokens):
            df.loc[selected_any, 'N_set_expr_prob'] = n_extra[selected_any]
    return df


def should_plot_state(g):
    mode = str(g.get('mode', 'intersection')).lower()
    return (mode == 'intersection')


def should_save_pymol_views(g):
    mode = str(g.get('mode', 'intersection')).lower()
    return (mode == 'intersection')


def resolve_site_jobs(g):
    raw_mode = str(g.get('mode', 'intersection')).strip()
    mode_expression = None
    if ',' in raw_mode:
        mode_prefix,mode_expression = raw_mode.split(',', 1)
        mode = mode_prefix.strip().lower()
        mode_expression = mode_expression.strip()
    else:
        mode = raw_mode.lower()
    g['mode'] = mode
    g['mode_expression'] = mode_expression
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
            raise ValueError('--mode set expects an expression, e.g., --mode "set,1|5".')
        _validate_set_expression_syntax(mode_expression=mode_expression)
        expression_branch_ids = _extract_set_expression_branch_ids(mode_expression)
        _validate_existing_branch_ids(expression_branch_ids, node_by_id)
        selected_nonroot = [bid for bid in expression_branch_ids.tolist() if not ete.is_root(node_by_id[int(bid)])]
        if len(selected_nonroot)==0:
            raise ValueError('--mode set expression should include at least one non-root branch ID.')
        branch_id_list = [numpy.array(sorted(selected_nonroot), dtype=numpy.int64)]
    else:
        raise ValueError('--mode should be one of intersection,lineage,set or set,<expr>.')

    site_jobs = []
    for branch_ids in branch_id_list:
        single_branch_mode = (branch_ids.shape[0]==1)
        branch_txt = ','.join([str(int(bid)) for bid in branch_ids.tolist()])
        if mode == 'intersection':
            site_outdir = './csubst_site.branch_id'+branch_txt
        elif mode == 'lineage':
            site_outdir = './csubst_site.lineage.branch_id'+lineage_input_branch_txt
        elif mode == 'set':
            mode_expr_label = _get_set_expression_label(mode_expression)
            site_outdir = './csubst_site.set.expr'+mode_expr_label
        else:
            site_outdir = './csubst_site.mode'+mode+'.branch_id'+branch_txt
        site_jobs.append({
            'branch_ids': branch_ids,
            'single_branch_mode': single_branch_mode,
            'site_outdir': site_outdir,
            'mode_expression': mode_expression,
        })
    g['site_jobs'] = site_jobs
    g['branch_id_list'] = [job['branch_ids'] for job in site_jobs]
    return g


def add_branch_id_list(g):
    return resolve_site_jobs(g)

def combinatorial2single_columns(df):
    for SN in ['OCS','OCN']:
        for anc in ['any','spe','dif']:
            for des in ['any', 'spe', 'dif']:
                col = SN+anc+'2'+des
                if col in df.columns:
                    df = df.drop(labels=col, axis=1)
    return df

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
        branch_ids = site_job['branch_ids']
        g['single_branch_mode'] = site_job['single_branch_mode']
        g['branch_ids'] = branch_ids
        g['site_outdir'] = site_job['site_outdir']
        g['mode_expression'] = site_job.get('mode_expression', g.get('mode_expression', None))
        txt = '\nProcessing --mode {} with branch IDs: {}'
        print(txt.format(g['mode'], ','.join([str(int(bid)) for bid in branch_ids.tolist()])), flush=True)
        if (g.get('mode_expression', None) is not None) and (str(g.get('mode', '')).lower() == 'set'):
            print('Set expression: {}'.format(g['mode_expression']), flush=True)
        if g['single_branch_mode']:
            print('Single branch mode. Substitutions, rather than combinatorial substitutions, will be mapped.')
        if not os.path.exists(g['site_outdir']):
            os.makedirs(g['site_outdir'])
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
            if g['pymol_img'] and should_save_pymol_views(g):
                parser_pymol.save_six_views()
                parser_pymol.save_6view_pdf(pdf_filename=os.path.join(g['site_outdir'], f'csubst_site.{id_base}.pymol.pdf'))
        plot_barchart(df, g)
        if g['pdb'] is None:
            outbase = os.path.join(g['site_outdir'], 'csubst_site')
        else:
            outbase = g['pdb_outfile_base']
        if str(g.get('mode', '')).lower() == 'lineage':
            plot_lineage_tree(g=g, outbase=outbase)
        if should_plot_state(g):
            plot_state(ON_tensor, OS_tensor, g['branch_ids'], g)
        out_path = outbase+'.tsv'
        if g['single_branch_mode']:
            df = combinatorial2single_columns(df)
        df.to_csv(out_path, sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
    print('To visualize the convergence probability on protein structure, please see: https://github.com/kfuku52/csubst/wiki')
    print('')
    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]
    return None

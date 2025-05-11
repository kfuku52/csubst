import numpy
import matplotlib.pyplot
import pandas

import itertools
import os
import re
import sys

from csubst import genetic_code
from csubst import parser_biodb
from csubst import parser_misc
from csubst import sequence
from csubst import substitution
from csubst import tree

font_size = 8
matplotlib.rcParams['font.size'] = font_size
#matplotlib.rcParams['font.family'] = 'Helvetica'
#matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
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
    seq = node.sequence.upper()
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
            yvalues = df.loc[:, 'S' + sub_type].values
            is_enough_value = (yvalues > 0.01)
            yvalues[is_enough_value] = df.loc[is_enough_value, ['N' + sub_type, 'S' + sub_type]].sum(axis=1).values
        elif SN == 'N':
            yvalues = df.loc[:, col].values
    elif sub_type=='_sub_':
        if SN == 'S':
            is_S_cols = df.columns.str.startswith('S_sub_')
            S_cols = df.columns[is_S_cols]
            is_y_cols = is_S_cols | df.columns.str.startswith('N_sub_')
            y_cols = df.columns[is_y_cols]
            yvalues = df.loc[:, S_cols].sum(axis=1).values
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

def get_highest_identity_chain_name(g):
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

def plot_barchart(df, g):
    if g['single_branch_mode']:
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
    fig.tight_layout(h_pad=0.5, w_pad=1)
    if g['pdb'] is None:
        outbase = os.path.join(g['site_outdir'], 'csubst_site')
    else:
        outbase = g['pdb_outfile_base']
    fig.savefig(outbase+".pdf", format='pdf', transparent=True)
    #fig.savefig(outbase+".svg", format='svg', transparent=True)
    print("Nonsynonymous and synonymous substitutions are shown in color and gray, respectively.", flush=True)
    print("Alignment gap sites are indicated by gray scale (0% missing = white, 100% missing = black).", flush=True)

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
        new_sites = new_sites.append(to_append, ignore_index=True)
    new_sites = new_sites.loc[new_sites>=0]
    new_sites = new_sites.drop_duplicates().sort_values().reset_index(drop=True)
    return new_sites

def add_gene_index(df, g):
    seqs = sequence.read_fasta(path=g['untrimmed_cds'])
    num_site = g['state_cdn'].shape[1]
    for leaf in g['tree'].iter_leaves():
        leaf_nn = leaf.numerical_label
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
        if node.numerical_label in branch_ids:
            parent_branch_ids[node.numerical_label] = node.up.numerical_label
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
        tree_dict[node.numerical_label] = node
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
                node_dist = nds[0].get_distance(target=nds[1], topology_only=False)
                node_dists.append(node_dist - nds[1].dist)
            interbranch_dist = max(node_dists) # Maximum value among pairwise distances
        df_dist.loc[current_row, :] = [state_key, state_from, state_to, interbranch_dist]
        current_row += 1
    df_dist = df_dist.loc[~df_dist['group'].isnull(),:]
    return df_dist

def plot_state(N_tensor, S_tensor, branch_ids, g):
    fig,axes = matplotlib.pyplot.subplots(nrows=3, ncols=2, figsize=(7.2, 7.2), sharex=False)
    outfiles = ['csubst_site.state_N.tsv', 'csubst_site.state_S.tsv']
    colors = ['red','blue']
    ax_cols = [0,1]
    titles = ['Nonsynonymous substitution','Synonymous substitution']
    iter_items = zip(ax_cols,['nsy','syn'],[N_tensor,S_tensor],outfiles,colors,titles)
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

def add_branch_id_list(g):
    if g['branch_id']=='fg':
        g['branch_id_list'] = []
        cb = pandas.read_csv(g['cb_file'], sep="\t", index_col=False, header=0)
        bid_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
        cb_fg = cb.loc[(cb.loc[:,cb.columns.str.startswith('is_fg')]=='Y').any(axis=1),:]
        for i in cb_fg.index:
            bids = cb_fg.loc[i,bid_cols].values.astype(int)
            g['branch_id_list'].append(bids)
    else:
        g['branch_id_list'] = [numpy.array([ int(s) for s in g['branch_id'].split(',')]),]
    return g

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
    N_tensor = substitution.get_substitution_tensor(state_tensor=g['state_pep'], mode='asis', g=g, mmap_attr='N')
    N_tensor = substitution.apply_min_sub_pp(g, N_tensor)
    S_tensor = substitution.get_substitution_tensor(state_tensor=g['state_cdn'], mode='syn', g=g, mmap_attr='S')
    S_tensor = substitution.apply_min_sub_pp(g, S_tensor)
    g = add_branch_id_list(g)
    for branch_ids in g['branch_id_list']:
        print('\nProcessing branch IDs: {}'.format(','.join([ str(bid) for bid in branch_ids ])), flush=True)
        if len(branch_ids)==1:
            print('Single branch mode. Substitutions, rather than combinatorial substitutions, will be mapped.')
            g['single_branch_mode'] = True
        else:
            g['single_branch_mode'] = False
        g['branch_ids'] = branch_ids
        g['site_outdir'] = './csubst_site.branch_id'+','.join([ str(bid) for bid in branch_ids ])
        if not os.path.exists(g['site_outdir']):
            os.makedirs(g['site_outdir'])
        leaf_nn = [ n.numerical_label for n in g['tree'].traverse() if n.is_leaf() ]
        num_site = N_tensor.shape[1]
        df = initialize_site_df(num_site)
        df = add_cs_info(df, g['branch_ids'], sub_tensor=S_tensor, attr='S')
        df = add_cs_info(df, g['branch_ids'], sub_tensor=N_tensor, attr='N')
        df.loc[:,'gap_rate_all'] = get_gapsite_rate(state_tensor=g['state_cdn'][leaf_nn,:,:])
        df.loc[:,'gap_rate_target'] = get_gapsite_rate(state_tensor=g['state_cdn'][g['branch_ids'],:,:])
        df = add_site_info(df, sub_tensor=S_tensor, attr='S')
        df = add_site_info(df, sub_tensor=N_tensor, attr='N')
        df = add_branch_sub_prob(df, branch_ids=g['branch_ids'], sub_tensor=S_tensor, attr='S')
        df = add_branch_sub_prob(df, branch_ids=g['branch_ids'], sub_tensor=N_tensor, attr='N')
        df = add_states(df, g['branch_ids'], g)
        if (g['untrimmed_cds'] is not None):
            df = add_gene_index(df, g)
        is_site_col = df.columns.str.startswith('codon_site_')
        df.loc[:,is_site_col] += 1
        if (g['untrimmed_cds'] is not None)|(g['export2chimera']):
            export2chimera(df, g)
        if g['run_pdb_sequence_search']:
            g = parser_biodb.pdb_sequence_search(g)
        if (g['pdb'] is not None):
            id_base = os.path.basename(g['pdb'])
            id_base = re.sub('.pdb$', '', id_base)
            id_base = re.sub('.cif$', '', id_base)
            g['pdb_outfile_base'] = os.path.join(g['site_outdir'], 'csubst_site.' + id_base)
            parser_pymol.initialize_pymol(pdb_id=g['pdb'])
            num_chain = parser_pymol.get_num_chain()
            if num_chain >= g['pymol_max_num_chain']:
                print(f'Number of chains ({num_chain}) in the PDB file is larger than the maximum number of chains allowed (--pymol_max_num_chain {g['pymol_max_num_chain']}). PyMOL session image generation is disabled.', flush=True)
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
            g['session_file_path'] = g['pdb_outfile_base']+'.pymol.pse'
            parser_pymol.write_pymol_session(df=df, g=g)
            if g['pymol_img']:
                parser_pymol.save_six_views()
                parser_pymol.save_6view_pdf(pdf_filename=os.path.join(g['site_outdir'], f'csubst_site.{id_base}.pymol.pdf'))
        plot_barchart(df, g)
        plot_state(N_tensor, S_tensor, g['branch_ids'], g)
        if g['pdb'] is None:
            out_path = os.path.join(g['site_outdir'], 'csubst_site.tsv')
        else:
            out_path = g['pdb_outfile_base']+'.tsv'
        if g['single_branch_mode']:
            df = combinatorial2single_columns(df)
        df.to_csv(out_path, sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
    print('To visualize the convergence probability on protein structure, please see: https://github.com/kfuku52/csubst/wiki')
    print('')
    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]
    return None

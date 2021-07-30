import numpy
import matplotlib.pyplot
import pandas

import os
import sys

from csubst import genetic_code
from csubst import parser_misc
from csubst import sequence
from csubst import substitution

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
    cmap = 'Greys'
    x_values = df.loc[:,xcol] - 0.5
    y_values = numpy.ones(x_values.shape) * yvalue
    gap_values = df.loc[:,gapcol]
    points = numpy.array([x_values, y_values]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
    lc = matplotlib.collections.LineCollection(segments, cmap=cmap, zorder=0)
    lc.set_array(gap_values)
    lc.set_linewidth(lw)
    ax.add_collection(lc)

def plot_barchart(df):
    sub_types = ['_sub','any2spe','any2dif','any2any','spe2spe']
    num_row = len(sub_types)*2
    fig,axes = matplotlib.pyplot.subplots(nrows=num_row, ncols=1, figsize=(7.2, 9.6), sharex=True)
    axes = axes.flat
    i = 0
    NS_ymax = df.loc[:,['N_sub','S_sub']].max().max() + 0.5
    SN_colors = {'N':'red', 'S':'blue'}
    for sub_type in sub_types:
        for SN in ['N','S']:
            ax = axes[i]
            col = SN+sub_type
            yvalues = df.loc[:,col].values
            if col in ['N_sub','S_sub']:
                if col=='N_sub':
                    ylabel = 'Nonsyn. subst.\nin the tree'
                elif col=='S_sub':
                    ylabel = 'Syn. subst.\nin the tree'
                ax.set_ylabel(ylabel, fontsize=font_size)
                ax.set_ylim(0, NS_ymax)
                add_gapline(df=df, gapcol='gap_rate_all', xcol='codon_site_alignment', yvalue=NS_ymax*0.95, lw=3, ax=ax)
            else:
                ax.set_ylim(0, 1)
                ax.set_ylabel(col, fontsize=font_size)
                ax.axhline(y=0.5, linestyle='--', linewidth=0.5, color='black', zorder=0)
                add_gapline(df=df, gapcol='gap_rate_target', xcol='codon_site_alignment', yvalue=0.95, lw=3, ax=ax)
            ax.bar(df.loc[:,'codon_site_alignment'], yvalues, color=SN_colors[SN])
            if (i==num_row-1):
                ax.set_xlabel('Codon site', fontsize=font_size)
            else:
                ax.set_xlabel('', fontsize=font_size)
            ax.set_xlim(df.loc[:,'codon_site_alignment'].min()-0.5, df.loc[:,'codon_site_alignment'].max()+0.5)
            i += 1
    fig.tight_layout(h_pad=0.5, w_pad=1)
    outbase = 'csubst_site'
    fig.savefig(outbase+".pdf", format='pdf', transparent=True)
    fig.savefig(outbase+".svg", format='svg', transparent=True)
    print("Alignment gap sites are indicated by gray scale (100% missing = black).", flush=True)

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
        file_name = 'csubst_site_'+seq_key+'.chimera.txt'
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
                    Nany2spe = df.loc[is_site,'Nany2spe'].values[0]
                    Nany2dif = df.loc[is_site,'Nany2dif'].values[0]
                    Nvalue = Nany2spe if (Nany2spe>=Nany2dif) else -Nany2dif
                    line = '	:{}	{:.4f}\n'.format(seq_site, Nvalue)
                f.write(line)
        translated_seq = translate(seq, g)
        file_fasta = 'csubst_site_'+seq_key+'.fasta'
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

def add_states(df, branch_ids, g):
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

def plot_state(N_tensor, S_tensor, branch_ids, g):
    fig,axes = matplotlib.pyplot.subplots(nrows=2, ncols=2, figsize=(7.2, 4.8), sharex=False)
    outfiles = ['csubst_state_N.tsv', 'csubst_state_S.tsv']
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
        df_ad.to_csv(outfile, sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
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
        ax.hist(x=df_ad.loc[:,'site_tsi'], bins=bins, color='black')
        ax.hist(x=df_ad.loc[(df_ad.loc[:,'has_target_high_combinat_prob_site']),'site_tsi'], bins=bins, color=color)
        ax.set_xlabel('Site specificity index', fontsize=font_size)
        ax.set_ylabel('Count of\nsubstitution categories', fontsize=font_size)
    fig.tight_layout(h_pad=0.5, w_pad=1)
    outbase = 'csubst_state'
    fig.savefig(outbase+".pdf", format='pdf', transparent=True)
    fig.savefig(outbase+".svg", format='svg', transparent=True)

def main_site(g):
    print("Reading and parsing input files.", flush=True)
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['genetic_code'])
    g = parser_misc.read_treefile(g)
    g = parser_misc.generate_intermediate_files(g)
    g = parser_misc.annotate_tree(g)
    g = parser_misc.read_input(g)
    g,g['state_nuc'],g['state_cdn'],g['state_pep'] = parser_misc.prep_state(g)
    N_tensor = substitution.get_substitution_tensor(state_tensor=g['state_pep'], mode='asis', g=g, mmap_attr='N')
    N_tensor = substitution.apply_min_sub_pp(g, N_tensor)
    S_tensor = substitution.get_substitution_tensor(state_tensor=g['state_cdn'], mode='syn', g=g, mmap_attr='S')
    S_tensor = substitution.apply_min_sub_pp(g, S_tensor)

    branch_ids = numpy.array([ int(s) for s in g['branch_id'].split(',')])
    num_site = N_tensor.shape[1]

    df = pandas.DataFrame()
    df.loc[:,'codon_site_alignment'] = numpy.arange(num_site)
    df.loc[:,'nuc_site_alignment'] = ((df.loc[:,'codon_site_alignment']+1) * 3) - 2

    cs = substitution.get_cs(id_combinations=branch_ids[numpy.newaxis,:], sub_tensor=S_tensor, attr='S')
    cs.columns = cs.columns.str.replace('site','codon_site_alignment')
    df = pandas.merge(df, cs, on='codon_site_alignment')
    df.loc[:,'Sany2dif'] = df.loc[:,'Sany2any'] - df.loc[:,'Sany2spe']
    cs = substitution.get_cs(id_combinations=branch_ids[numpy.newaxis,:], sub_tensor=N_tensor, attr='N')
    cs.columns = cs.columns.str.replace('site','codon_site_alignment')
    df = pandas.merge(df, cs, on='codon_site_alignment')
    df.loc[:,'Nany2dif'] = df.loc[:,'Nany2any'] - df.loc[:,'Nany2spe']
    del cs

    leaf_nn = [ n.numerical_label for n in g['tree'].traverse() if n.is_leaf() ]
    df.loc[:,'gap_rate_all'] = get_gapsite_rate(state_tensor=g['state_cdn'][leaf_nn,:,:])
    df.loc[:,'gap_rate_target'] = get_gapsite_rate(state_tensor=g['state_cdn'][branch_ids,:,:])

    s = substitution.get_s(S_tensor, attr='S')
    s.columns = s.columns.str.replace('site','codon_site_alignment')
    df = pandas.merge(df, s, on='codon_site_alignment')
    s = substitution.get_s(N_tensor, attr='N')
    s.columns = s.columns.str.replace('site','codon_site_alignment')
    df = pandas.merge(df, s, on='codon_site_alignment')
    del s
    df = add_states(df, branch_ids, g)
    if (g['untrimmed_cds'] is not None):
        df = add_gene_index(df, g)
    is_site_col = df.columns.str.startswith('codon_site_')
    df.loc[:,is_site_col] += 1
    if (g['untrimmed_cds'] is not None)|(g['export2chimera']):
        export2chimera(df, g)
    plot_barchart(df)
    plot_state(N_tensor, S_tensor, branch_ids, g)
    df.to_csv('csubst_site.tsv', sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
    print('To visualize the convergence probability on protein structure, please see:')
    print('https://github.com/kfuku52/csubst/wiki/Visualizing-convergence-probabilities-on-protein-structures')
    print('')



    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]

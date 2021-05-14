import numpy
import matplotlib.pyplot
import pandas

import os

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

def get_gapsite_rate(state_tensor):
    num_gapsite = (state_tensor.sum(axis=2)==0).sum(axis=0)
    gapsite_rate = num_gapsite / state_tensor.shape[0]
    return gapsite_rate

def add_gene_index(df, g):
    seqs = sequence.read_fasta(path=g['untrimmed_cds'])
    num_site = g['state_cdn'].shape[1]
    for leaf in g['tree'].get_leaves():
        leaf_nn = leaf.numerical_label
        if leaf.name not in seqs.keys():
            continue
        seq = seqs[leaf.name]
        aln_sites = numpy.arange(num_site)
        aln_gene_match = list()
        current_gene_site = 0
        for aln_site in aln_sites:
            missing_site_flag = True
            for cgs in numpy.arange(current_gene_site, num_site):
                codon = seq[(cgs*3):((cgs+1)*3)]
                codon_index = sequence.get_state_index(state=codon, input_state=g['codon_orders'],
                                                       ambiguous_table=genetic_code.ambiguous_table)
                if len(codon_index)==0:
                    continue
                ci = codon_index[0] # Take the first codon if ambiguous
                if g['state_cdn'][leaf_nn,aln_site,ci]!=0:
                    aln_gene_match.append([aln_site,cgs])
                    current_gene_site = cgs + 1
                    missing_site_flag = False
                    break
            if missing_site_flag:
                aln_gene_match.append([aln_site,-1])
        aln_gene_match = pandas.DataFrame(aln_gene_match)
        aln_gene_match.columns = ['codon_site_alignment','codon_site_'+leaf.name]
        df = pandas.merge(df, aln_gene_match, on='codon_site_alignment')
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
        write_fasta(file=file_fasta, label=seq_key, seq=translated_seq)

def main_site(g):
    print("Reading and parsing input files.", flush=True)
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['genetic_code'])
    g = parser_misc.read_treefile(g)
    g = parser_misc.read_input(g)
    g,g['state_nuc'],g['state_cdn'],g['state_pep'] = parser_misc.prep_state(g)
    N_tensor = substitution.get_substitution_tensor(state_tensor=g['state_pep'], mode='asis', g=g, mmap_attr='N')
    N_tensor = substitution.apply_min_sub_pp(g, N_tensor)
    S_tensor = substitution.get_substitution_tensor(state_tensor=g['state_cdn'], mode='syn', g=g, mmap_attr='S')
    S_tensor = substitution.apply_min_sub_pp(g, S_tensor)

    branch_ids = numpy.array([ int(s) for s in g['branch_id'].split(',')])
    num_site = N_tensor.shape[1]
    parent_branch_ids = dict()
    for node in g['tree'].traverse():
        if node.numerical_label in branch_ids:
            parent_branch_ids[node.numerical_label] = node.up.numerical_label

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

    if (g['untrimmed_cds'] is not None):
        df = add_gene_index(df, g)

    is_site_col = df.columns.str.startswith('codon_site_')
    df.loc[:,is_site_col] += 1

    if (g['untrimmed_cds'] is not None)|(g['export2chimera']):
        export2chimera(df, g)

    plot_barchart(df)
    df.to_csv('csubst_site.tsv', sep="\t", index=False, float_format='%.4f', chunksize=10000)

    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]

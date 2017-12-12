#!/usr/bin/env python

# omega_asrv can be larger than omega_flat when,
# for example, ASRV is less biased in nonSonymous substitutions
# for example, # nonSonymous substs are largely different between branch 1 and branch 2

import argparse

from util import parser_phylobayes
from util import parser_iqtree
from util.genetic_code import *
from util.sequence import *
from util.omega import *
from util.combination import *
from util.table import *
from util.param import *

csubst_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='%(prog)s 0.1') # version
parser.add_argument('--max_arity', metavar='INTEGER', default=2, type=int, help='The maximum combinatorial number of branches. Set 2 for paired substitutions.')
parser.add_argument('--nslots', metavar='INTEGER',default=1, type=int, help='The number of processors for parallel computations.')
parser.add_argument('--ncbi_codon_table', metavar='INTEGER',type=int, required=True, help='NCBI codon table id.')
parser.add_argument('--infile_dir', metavar='PATH',type=str, required=True, help='PATH to the input file directory.')
parser.add_argument('--infile_type', metavar='[phylobayes|iqtree]', default='phylobayes', type=str, help='The input file format.')
parser.add_argument('--aln_file', metavar='PATH', default='', type=str, help='Alignment fasta file. Specify if csubst cannot find it in the infile_dir.')
parser.add_argument('--tre_file', metavar='PATH', default='', type=str, help='Rooted tree newick file. Specify if csubst cannot find it in the infile_dir.')
parser.add_argument('--calc_omega', metavar='INTEGER', default=1, type=int, help='Calculate omega for convergence rate.')
parser.add_argument('--ml_anc', metavar='INTEGER', default=0, type=float, help='Maximum-likelihood-like analysis by binarizing ancestral states.')
parser.add_argument('--min_sub_pp', metavar='FLOAT', default=0, type=float, help='The minimum posterior probability of single substitutions to count. Set 0 for a full Bayesian counting without binarization.')
parser.add_argument('--b', metavar='INTEGER',default=1, type=int, help='Branch output. 0 or 1. Set 1 to get the output.')
parser.add_argument('--s', metavar='INTEGER',default=0, type=int, help='Site output. 0 or 1. Set 1 to get the output.')
parser.add_argument('--cs', metavar='INTEGER',default=0, type=int, help='Combinatorial-site output. 0 or 1. Set 1 to get the output.')
parser.add_argument('--cb', metavar='INTEGER',default=1, type=int, help='Combinatorial-branch output. 0 or 1. Set 1 to get the output.')
parser.add_argument('--bs', metavar='INTEGER',default=0, type=int, help='Branch-site output. 0 or 1. Set 1 to get the output.')
parser.add_argument('--cbs', metavar='INTEGER',default=0, type=int, help='Combinatorial-branch-site output. 0 or 1. Set 1 to get the output.')
parser.add_argument('--target_stat', metavar='[omega_conv_unif|omega_conv_asrvN|omega_conv_asrvNS...]', default='omega_conv_asrvN', type=str, help='The statistics used to explore higher-order branch combinations.')
parser.add_argument('--min_stat', metavar='FLOAT',default=1.0, type=float, help='If a branch combination has the target_stat greater than this value, higher-order combinations are explored.')
parser.add_argument('--min_branch_sub', metavar='FLOAT',default=1.0, type=float, help='Minimum substitutions in a branch. Branches < min_branch_sub are excluded from branch combination analyses.')
parser.add_argument('--min_Nany2spe', metavar='FLOAT',default=1.0, type=float, help='Minimum nonsonymous convergent substitutions. Branch combinations < min_Nany2spe are excluded from higher-order analyses.')
parser.add_argument('--min_NCoD', metavar='FLOAT',default=0.15, type=float, help='Minimum nonsynonymous C/D. C/D < min_NCoD are excluded from higher-order analyses.')
parser.add_argument('--exclude_sisters', metavar='INTEGER',default=1, type=int, help='Set 1 to exclude sister branches in branch combinatioin analysis.')
parser.add_argument('--resampling_size', metavar='INTEGER',default=50000, type=int, help='The number of combinatorial branch resampling to estimate rho in higher-order analyses.')
parser.add_argument('--foreground', metavar='PATH',default=None, type=str, help='Foreground taxa for higher-order analysis.')
parser.add_argument('--exclude_wg', metavar='INTEGER',default=1, type=int, help='Set 1 to exclude branches within foreground lineages in branch combination analysis.')
parser.add_argument('--cb_stats', metavar='PATH',default=None, type=str, help='Use precalculated rho parameters in branch combination analysis.')

args = parser.parse_args()
g = get_global_parameters(args)

start = time.time()
print("Reading and parsing input files.", flush=True)
g['current_arity'] = 2
g['codon_table'] = get_codon_table(ncbi_id=g['ncbi_codon_table'])
if g['infile_type']=='phylobayes':
    g = parser_phylobayes.get_input_information(g)
    if g['input_data_type']=='nuc':
        state_nuc = parser_phylobayes.get_state_tensor(g)
        if g['calc_omega']:
            state_cdn = calc_omega_state(state_nuc=state_nuc, g=g)
            state_pep = cdn2pep_state(state_cdn=state_cdn, g=g)
    if g['input_data_type']=='cdn':
        state_cdn = parser_phylobayes.get_state_tensor(g)
        state_pep = cdn2pep_state(state_cdn=state_cdn, g=g)
elif g['infile_type']=='iqtree':
    g = parser_iqtree.get_input_information(g)
    if g['input_data_type']=='nuc':
        state_nuc = parser_iqtree.get_state_tensor(g)
        if g['calc_omega']:
            state_cdn = calc_omega_state(state_nuc=state_nuc, g=g)
            state_pep = cdn2pep_state(state_cdn=state_cdn, g=g)
    if g['input_data_type']=='cdn':
        state_cdn = parser_iqtree.get_state_tensor(g)
        state_pep = cdn2pep_state(state_cdn=state_cdn, g=g)

if not g['foreground'] is None:
    g = get_foreground_branch(g)
g = get_dep_ids(g)
if not g['cb_stats'] is None:
    g['df_cb_stats'] = pandas.read_csv(g['cb_stats'], sep='\t', header=0)


for key in sorted(list(g.keys())):
    if key=='tree':
        print(key)
        print(g[key].get_ascii(attributes=['name','numerical_label'], show_internal=True))
    else:
        print(key, g[key])

N_tensor = get_substitution_tensor(state_tensor=state_pep, mode='asis', g=g, mmap_attr='N')
sub_branches = numpy.where(N_tensor.sum(axis=(1,2,3,4))!=0)[0].tolist()
if g['calc_omega']:
    S_tensor = get_substitution_tensor(state_tensor=state_cdn, mode='syn', g=g, mmap_attr='S')
    sub_branches = list(set(sub_branches).union(set(numpy.where(S_tensor.sum(axis=(1, 2, 3, 4)) != 0)[0].tolist())))
g['sub_branches'] = sub_branches

id_combinations = prepare_node_combinations(g=g, arity=g['current_arity'], check_attr="name")

S_total = numpy.nan_to_num(S_tensor).sum(axis=(0,1,2,3,4))
N_total = numpy.nan_to_num(N_tensor).sum(axis=(0,1,2,3,4))
num_branch = g['num_node'] -1
num_site = S_tensor.shape[2]
print('Synonymous substitutions / tree =', S_total)
print('Nonsynonymous substitutions / tree =', N_total)
print('Synonymous substitutions / branch =', S_total / num_branch)
print('Nonsynonymous substitutions / branch =', N_total / num_branch)
print('Synonymous substitutions / site =', S_total / num_site)
print('Nonsynonymous substitutions / site =', N_total / num_site)
elapsed_time = int(time.time() - start)
print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

if g['bs']:
    start = time.time()
    print("Making branch-site table.", flush=True)
    bs = get_bs(S_tensor, N_tensor)
    bs = sort_labels(df=bs)
    bs.to_csv("csubst_bs.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
    print(bs.info(verbose=False, max_cols=0, memory_usage=True, null_counts=False), flush=True)
    del bs
    elapsed_time = int(time.time() - start)
    print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

del state_cdn, state_pep

if (g['b'])|(g['cb']):
    start = time.time()
    print("Making branch table.", flush=True)
    bS = get_b(g, S_tensor, attr='S')
    bN = get_b(g, N_tensor, attr='N')
    b = merge_tables(bS, bN)
    del bS, bN
    if g['b']:
        b.to_csv("csubst_b.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
    print(b.info(verbose=False, max_cols=0, memory_usage=True, null_counts=False), flush=True)
    elapsed_time = int(time.time() - start)
    print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

if (g['s'])|(g['cb']):
    start = time.time()
    print("Making site table.", flush=True)
    sS = get_s(S_tensor, attr='S')
    sN = get_s(N_tensor, attr='N')
    s = merge_tables(sS, sN)
    del sS, sN
    if g['s']:
        s.to_csv("csubst_s.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
    print(s.info(verbose=False, max_cols=0, memory_usage=True, null_counts=False), flush=True)
    elapsed_time = int(time.time() - start)
    print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

if g['cs']:
    start = time.time()
    print("Making combinat-site table.", flush=True)
    csS = get_cs(id_combinations, S_tensor, attr='S')
    csN = get_cs(id_combinations, N_tensor, attr='N')
    cs = merge_tables(csS, csN)
    del csS, csN
    cs.to_csv("csubst_cs.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
    print(cs.info(verbose=False, max_cols=0, memory_usage=True, null_counts=False), flush=True)
    del cs
    elapsed_time = int(time.time()-start)
    print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

if g['cbs']:
    start = time.time()
    print("Making combinat-branch-site table.", flush=True)
    cbsS = get_cbs(id_combinations, S_tensor, attr='S', g=g)
    cbsN = get_cbs(id_combinations, N_tensor, attr='N', g=g)
    cbs = merge_tables(cbsS, cbsN)
    del cbsS, cbsN
    cbs.to_csv("csubst_cbs.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
    print(cbs.info(verbose=False, max_cols=0, memory_usage=True, null_counts=False), flush=True)
    del cbs
    elapsed_time = int(time.time() - start)
    print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

if g['cb']:
    #cols = ['arity','method','num_combinat','rhoSconv','rhoNconv','Sany2any','Sany2spe','Nany2any','Nany2spe']
    #df_rho = pandas.DataFrame(index=[], columns=cols)
    end_flag = 0
    df_rho = pandas.DataFrame([])
    for current_arity in numpy.arange(g['current_arity'], g['max_arity']+1):
        start = time.time()
        print("Making combinat-branch table. Arity =", current_arity, flush=True)
        g['current_arity'] = current_arity
        if (current_arity == 2) & (g['foreground'] is None):
            id_combinations = id_combinations
        elif (current_arity==2)&(not g['foreground'] is None):
            id_combinations = prepare_node_combinations(g=g, target_nodes=g['fg_id'], arity=current_arity,
                                                        check_attr='name', foreground=True)
        elif current_arity > 2:
            is_stat_enough = (cb[g['target_stat']]>=g['min_stat'])|(cb[g['target_stat']].isnull())
            is_Nany2spe_enough = (cb['Nany2spe']>=g['min_Nany2spe'])
            is_NCoD_enough = (cb['NCoD']>=g['min_NCoD'])
            is_branch_sub_enough = True
            for a in numpy.arange(current_arity-1):
                target_columns = ['S_sub_'+str(a+1), 'N_sub_'+str(a+1)]
                is_branch_sub_enough = is_branch_sub_enough&(cb.loc[:,target_columns].sum(axis=1)>=g['min_branch_sub'])
            num_branch_ids = (is_stat_enough).sum()
            print('arity =', current_arity, ': qualified combinations =', num_branch_ids)
            id_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
            conditions = (is_stat_enough)&(is_branch_sub_enough)&(is_Nany2spe_enough)&(is_NCoD_enough)
            branch_ids = cb.loc[conditions,id_columns].values
            if len(set(branch_ids.ravel().tolist())) < current_arity:
                end_flag = 1
                break
            del cb
            id_combinations = prepare_node_combinations(g=g, target_nodes=branch_ids, arity=current_arity,
                                                        check_attr='name', foreground=True)
            if id_combinations.shape[0] == 0:
                end_flag = 1
                break
        cbS = get_cb(id_combinations, S_tensor, g, 'S')
        cbN = get_cb(id_combinations, N_tensor, g, 'N')
        cb = merge_tables(cbS, cbN)
        del cbS, cbN
        if (current_arity==2)&(g['foreground'] is None):
            rs = 'all'
            rhoNconv_at_2 = cb['Nany2spe'].sum() / cb['Nany2any'].sum()
            rhoSconv_at_2 = cb['Sany2spe'].sum() / cb['Sany2any'].sum()
        else:
            rs = g['resampling_size']
            #rs_estimated = {'rhoSconv':numpy.sqrt(rhoSconv_at_2)**current_arity, 'rhoNconv':numpy.sqrt(rhoNconv_at_2)**current_arity}
            #print('rhoSconv estimated from arity = 2:', rs_estimated['rhoSconv'])
            #print('rhoNconv estimated from arity = 2:', rs_estimated['rhoNconv'])
        cb,tmp_rho_stats = calc_omega(cb, b, s, S_tensor, N_tensor, g, rs)
        file_name = "csubst_cb_"+str(current_arity)+".tsv"
        cb.to_csv(file_name, sep="\t", index=False, float_format='%.4f', chunksize=10000)
        print(cb.info(verbose=False, max_cols=0, memory_usage=True, null_counts=False), flush=True)
        print('median omega_pair_unif =', numpy.round(cb['omega_pair_unif'].median(),decimals=3), flush=True)
        print('median omega_conv_unif =', numpy.round(cb['omega_conv_unif'].median(),decimals=3), flush=True)
        print('median omega_div_unif  =', numpy.round(cb['omega_div_unif'].median(),decimals=3), flush=True)
        print('median omega_pair_asrvNS =', numpy.round(cb['omega_pair_asrvNS'].median(),decimals=3), flush=True)
        print('median omega_conv_asrvNS =', numpy.round(cb['omega_conv_asrvNS'].median(),decimals=3), flush=True)
        print('median omega_div_asrvNS  =', numpy.round(cb['omega_div_asrvNS'].median(),decimals=3), flush=True)
        print('median omega_pair_asrvN =', numpy.round(cb['omega_pair_asrvN'].median(),decimals=3), flush=True)
        print('median omega_conv_asrvN =', numpy.round(cb['omega_conv_asrvN'].median(),decimals=3), flush=True)
        print('median omega_div_asrvN  =', numpy.round(cb['omega_div_asrvN'].median(),decimals=3), flush=True)
        elapsed_time = int(time.time() - start)
        tmp_rho_stats['elapsed_sec'] = elapsed_time
        tmp_df_rho = pandas.DataFrame(index=[current_arity,], columns=sorted(list(tmp_rho_stats.keys())))
        for c in list(tmp_rho_stats.keys()):
            tmp_df_rho.loc[current_arity,c] = tmp_rho_stats[c]
        df_rho = df_rho.append(tmp_df_rho)
        df_rho.to_csv('csubst_cb_stats.tsv', sep="\t", index=False, float_format='%.4f', chunksize=10000)
        print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)
    if end_flag:
        print('No combination satisfied phylogenetic independency. Ending branch combination analysis.')

tmp_files = [ f for f in os.listdir() if f.startswith('tmp.csubst.') ]
_ = [ os.remove(ts) for ts in tmp_files ]

print("\ncsubst completed. Elapsed time =", int(time.time()-csubst_start), '[sec]',  flush=True)


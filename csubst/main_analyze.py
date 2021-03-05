import numpy
import pandas

import os
import time

from csubst import combination
from csubst import foreground
from csubst import genetic_code
from csubst import omega
from csubst import param
from csubst import parser_misc
from csubst import sequence
from csubst import substitution
from csubst import table
from csubst import tree

def add_median_cb_stats(g, cb, current_arity, start):
    is_arity = (g['df_cb_stats'].loc[:,'arity'] == current_arity)
    suffices = list()
    is_targets = list()
    suffices.append('_all')
    is_targets.append(numpy.ones(shape=cb.shape[0], dtype=numpy.bool))
    target_cols = ['is_fg','is_mg','is_mf','dummy']
    suffix_candidates = ['_fg','_mg','_mf']
    if g['force_exhaustive']:
        suffix_candidates.append('_all')
    for target_col,sc in zip(target_cols,suffix_candidates):
        if target_col in cb.columns:
            suffices.append(sc)
            if sc=='_all':
                is_targets.append(True)
            else:
                is_targets.append(cb.loc[:,target_col]=='Y')
    stats = dict()
    stats['median'] = ['dist_bl','dist_node_num','omega_any2any','omega_any2spe','omega_any2dif']
    stats['total'] = ['Nany2any','ENany2any','Sany2any','ESany2any','Nany2spe','ENany2spe','Sany2spe','ESany2spe',]
    for stat in stats.keys():
        for suffix,is_target in zip(suffices,is_targets):
            for ms in stats[stat]:
                col = stat+'_'+ms+suffix
                if stat=='median':
                    g['df_cb_stats'].loc[is_arity,col] = cb.loc[is_target,ms].median()
                elif stat=='total':
                    g['df_cb_stats'].loc[is_arity,col] = cb.loc[is_target,ms].sum()
            g['df_cb_stats'].loc[is_arity,'num'+suffix] = is_target.sum()
            num_qualified = (cb.loc[is_target,g['cutoff_stat']]>=g['cutoff_stat_min']).sum()
            g['df_cb_stats'].loc[is_arity,'num_qualified'+suffix] = num_qualified
    for key in ['Nany2any','Sany2any','Nany2spe','Sany2spe',]:
        totalN = g['df_cb_stats'].loc[is_arity, 'total_'+key+'_all'].values[0]
        totalEN = g['df_cb_stats'].loc[is_arity, 'total_E'+key+'_all'].values[0]
        txt = 'Total {}/E{} = {:,.1f}/{:,.1f} (Expectation equals to {:,.1f}% of the observation)'
        print(txt.format(key, key, totalN, totalEN, totalEN/totalN*100))
    elapsed_time = int(time.time() - start)
    g['df_cb_stats'].loc[is_arity, 'elapsed_sec'] = elapsed_time
    print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)
    return g

def cb_search(g, b, S_tensor, N_tensor, id_combinations, mode='', write_cb=True):
    end_flag = 0
    g = param.initialize_df_cb_stats(g)
    g['df_cb_stats'].loc[:,'mode'] = mode
    for current_arity in numpy.arange(2, g['max_arity'] + 1):
        start = time.time()
        print("Generating combinat-branch table. Arity = {:,}".format(current_arity), flush=True)
        g['current_arity'] = current_arity
        if (current_arity == 2):
            if (g['foreground'] is not None) & (g['force_exhaustive']==False):
                print('Searching foreground branch combinations.', flush=True)
                g,id_combinations = combination.get_node_combinations(g=g, target_nodes=g['target_id'], arity=current_arity,
                                                          check_attr='name')
            else:
                print('Exhaustively searching independent branch combinations.', flush=True)
                if id_combinations is None:
                    g,id_combinations = combination.get_node_combinations(g=g, arity=current_arity, check_attr="name")
        elif (current_arity > 2):
            is_stat_enough = (cb.loc[:,g['cutoff_stat']] >= g['cutoff_stat_min']) | (cb.loc[:,g['cutoff_stat']].isnull())
            is_combinat_sub_enough = ((cb.loc[:,'Nany2any']+cb.loc[:,'Nany2any']) >= g['min_combinat_sub'])
            is_branch_sub_enough = True
            for a in numpy.arange(current_arity - 1):
                target_columns = ['S_sub_' + str(a + 1), 'N_sub_' + str(a + 1)]
                is_branch_sub_enough = is_branch_sub_enough & (
                        cb.loc[:, target_columns].sum(axis=1) >= g['min_branch_sub'])
            num_branch_ids = (is_stat_enough).sum()
            print('Arity = {:,}: qualified combinations = {:,}'.format(current_arity, num_branch_ids), flush=True)
            id_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
            conditions = (is_stat_enough) & (is_branch_sub_enough) & (is_combinat_sub_enough)
            branch_ids = cb.loc[conditions, id_columns].values
            if len(set(branch_ids.ravel().tolist())) < current_arity:
                end_flag = 1
                break
            del cb
            g,id_combinations = combination.get_node_combinations(g=g, target_nodes=branch_ids, arity=current_arity,
                                                    check_attr='name')
            if id_combinations.shape[0] == 0:
                end_flag = 1
                break
        cbS = substitution.get_cb(id_combinations, S_tensor, g, 'S')
        cbN = substitution.get_cb(id_combinations, N_tensor, g, 'N')
        cb = table.merge_tables(cbS, cbN)
        del cbS, cbN
        cb = tree.get_node_distance(g['tree'], cb)
        cb = substitution.get_substitutions_per_branch(cb, b, g)
        #cb = combination.calc_substitution_patterns(cb)
        cb = substitution.get_any2dif(cb, g['float_tol'], prefix='')
        cb, g = omega.calc_omega(cb, S_tensor, N_tensor, g)
        cb = table.get_linear_regression(cb)
        cb, g = foreground.get_foreground_branch_num(cb, g)
        cb = table.sort_cb(cb)
        if write_cb:
            file_name = "csubst_cb_" + str(current_arity) + ".tsv"
            cb.to_csv(file_name, sep="\t", index=False, float_format='%.4f', chunksize=10000)
            txt = 'Memory consumption of cb table: {:,.1f} Mbytes (dtype={})'
            print(txt.format(cb.values.nbytes/1024/1024, cb.values.dtype), flush=True)
        g = add_median_cb_stats(g, cb, current_arity, start)
        if end_flag:
            print('No combination satisfied phylogenetic independence. Ending branch combination analysis.')
            break
    g['df_cb_stats'] = g['df_cb_stats'].loc[(~g['df_cb_stats'].loc[:,'elapsed_sec'].isnull()),:]
    g['df_cb_stats'] = g['df_cb_stats'].loc[:, sorted(g['df_cb_stats'].columns.tolist())]
    g['df_cb_stats_main'] = pandas.concat([g['df_cb_stats_main'], g['df_cb_stats']], ignore_index=True)
    return g

def main_analyze(g):
    start = time.time()
    print("Reading and parsing input files.", flush=True)
    g['current_arity'] = 2
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['genetic_code'])
    g = parser_misc.generate_intermediate_files(g)
    g = parser_misc.read_input(g)
    g,g['state_nuc'],g['state_cdn'],g['state_pep'] = parser_misc.prep_state(g)

    sequence.write_alignment(g['state_cdn'], g['codon_orders'], 'csubst_alignment_codon.fa', mode='codon', g=g)
    sequence.write_alignment(g['state_pep'], g['amino_acid_orders'], 'csubst_alignment_aa.fa', mode='aa', g=g)

    g = foreground.get_foreground_branch(g)
    g = foreground.get_marginal_branch(g)
    g = combination.get_dep_ids(g)
    tree.write_tree(g['tree'])
    tree.plot_branch_category(g['tree'], file_name='csubst_branch_category.pdf')
    if g['plot_state_aa']:
        plot_state_tree(state=g['state_pep'], orders=g['amino_acid_orders'], mode='aa', g=g)
    if g['plot_state_codon']:
        plot_state_tree(state=g['state_cdn'], orders=g['codon_orders'], mode='codon', g=g)

    N_tensor = substitution.get_substitution_tensor(state_tensor=g['state_pep'], mode='asis', g=g, mmap_attr='N')
    N_tensor = substitution.apply_min_sub_pp(g, N_tensor)
    sub_branches = numpy.where(N_tensor.sum(axis=(1, 2, 3, 4)) != 0)[0].tolist()
    S_tensor = substitution.get_substitution_tensor(state_tensor=g['state_cdn'], mode='syn', g=g, mmap_attr='S')
    S_tensor = substitution.apply_min_sub_pp(g, S_tensor)
    sub_branches = list(set(sub_branches).union(set(numpy.where(S_tensor.sum(axis=(1, 2, 3, 4)) != 0)[0].tolist())))
    g['sub_branches'] = sub_branches

    g = tree.rescale_branch_length(g, S_tensor, N_tensor)

    id_combinations = None

    S_total = S_tensor.sum(axis=(0, 1, 2, 3, 4))
    N_total = N_tensor.sum(axis=(0, 1, 2, 3, 4))
    num_branch = g['num_node'] - 1
    num_site = S_tensor.shape[1]
    print('Synonymous substitutions / tree = {:,.1f}'.format(S_total), flush=True)
    print('Nonsynonymous substitutions / tree = {:,.1f}'.format(N_total), flush=True)
    print('Synonymous substitutions / branch = {:,.1f}'.format(S_total / num_branch), flush=True)
    print('Nonsynonymous substitutions / branch = {:,.1f}'.format(N_total / num_branch), flush=True)
    print('Synonymous substitutions / site = {:,.1f}'.format(S_total / num_site), flush=True)
    print('Nonsynonymous substitutions / site = {:,.1f}'.format(N_total / num_site), flush=True)
    elapsed_time = int(time.time() - start)
    print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['bs']):
        start = time.time()
        print("Generating branch-site table.", flush=True)
        bs = substitution.get_bs(S_tensor, N_tensor)
        bs = table.sort_labels(df=bs)
        bs.to_csv("csubst_bs.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        txt = 'Memory consumption of bs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(bs.values.nbytes/1024/1024, bs.values.dtype), flush=True)
        del bs
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['s']) | (g['cb']):
        start = time.time()
        print("Generating site table.", flush=True)
        sS = substitution.get_s(S_tensor, attr='S')
        sN = substitution.get_s(N_tensor, attr='N')
        s = table.merge_tables(sS, sN)
        g = substitution.get_sub_sites(g, sS, sN, state_tensor=g['state_cdn'])
        del sS, sN
        if (g['s']):
            s.to_csv("csubst_s.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        txt = 'Memory consumption of s table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(s.values.nbytes/1024/1024, s.values.dtype), flush=True)
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['omega_method']!='submodel'):
        g['state_cdn'] = None
        g['state_pep'] = None

    if (g['b']) | (g['cb']):
        start = time.time()
        print("Generating branch table.", flush=True)
        bS = substitution.get_b(g, S_tensor, attr='S')
        bN = substitution.get_b(g, N_tensor, attr='N')
        b = table.merge_tables(bS, bN)
        b.loc[:,'branch_length'] = numpy.nan
        for node in g['tree'].traverse():
            b.loc[node.numerical_label,'branch_length'] = node.dist
        txt = 'Number of {} patterns among {:,} branches={:,}, min={:,}, max={:,}'
        for key in ['S_sub', 'N_sub']:
            p = b.loc[:, key].drop_duplicates().values
            print(txt.format(key, b.shape[0], p.shape[0], p.min(), p.max()), flush=True)
        del bS, bN
        b = foreground.annotate_foreground(b, g)
        g['branch_table'] = b
        if (g['b']):
            b.to_csv("csubst_b.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        txt = 'Memory consumption of b table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(b.values.nbytes/1024/1024, b.values.dtype), flush=True)
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cs']):
        start = time.time()
        print("Generating combinat-site table.", flush=True)
        if id_combinations is None:
            g,id_combinations = combination.get_node_combinations(g=g, arity=g['current_arity'], check_attr="name")
        csS = substitution.get_cs(id_combinations, S_tensor, attr='S')
        csN = substitution.get_cs(id_combinations, N_tensor, attr='N')
        cs = table.merge_tables(csS, csN)
        del csS, csN
        cs.to_csv("csubst_cs.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        txt = 'Memory consumption of cs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(cs.values.nbytes/1024/1024, cs.values.dtype), flush=True)
        del cs
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cbs']):
        start = time.time()
        print("Generating combinat-branch-site table.", flush=True)
        if id_combinations is None:
            g,id_combinations = combination.get_node_combinations(g=g, arity=g['current_arity'], check_attr="name")
        cbsS = substitution.get_cbs(id_combinations, S_tensor, attr='S', g=g)
        cbsN = substitution.get_cbs(id_combinations, N_tensor, attr='N', g=g)
        cbs = table.merge_tables(cbsS, cbsN)
        del cbsS, cbsN
        cbs.to_csv("csubst_cbs.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        txt = 'Memory consumption of cbs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(cbs.values.nbytes/1024/1024, cbs.values.dtype), flush=True)
        del cbs
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cb']):
        g['df_cb_stats_main'] = pandas.DataFrame()
        g = cb_search(g, b, S_tensor, N_tensor, id_combinations, mode='foreground', write_cb=True)
        if (g['fg_random']>0):
            for i in numpy.arange(0, g['fg_random']):
                print('starting foreground randomization round {:,}'.format(i+1), flush=True)
                g,rid_combinations = foreground.set_random_foreground_branch(g)
                print('rid_combinations.shape', rid_combinations.shape)
                g = cb_search(g, b, S_tensor, N_tensor, rid_combinations, mode='randomization_'+str(i+1), write_cb=False)
                print('ending foreground randomization round {:,}\n'.format(i+1), flush=True)

        g['df_cb_stats_main'].to_csv('csubst_cb_stats.tsv', sep="\t", index=False, float_format='%.4f', chunksize=10000)

    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]


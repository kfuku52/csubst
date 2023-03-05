import numpy
import pandas

import os
import shutil
import sys
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

def cb_search(g, b, S_tensor, N_tensor, id_combinations, write_cb=True):
    end_flag = 0
    g = param.initialize_df_cb_stats(g)
    for current_arity in numpy.arange(2, g['max_arity'] + 1):
        start = time.time()
        print("Arity = {:,}: Generating cb table".format(current_arity), flush=True)
        g['current_arity'] = current_arity
        if (current_arity==2):
            if (g['exhaustive_until']<current_arity)&(g['foreground'] is not None):
                txt = 'Arity = {:,}: Targeted search of foreground branch combinations'
                print(txt.format(current_arity), flush=True)
                g['df_cb_stats'].loc[current_arity-1, 'mode'] = 'foreground'
                g,id_combinations = combination.get_node_combinations(g=g, target_nodes=g['target_id'],
                                                                      arity=current_arity, check_attr='name')
            else:
                txt = 'Arity = {:,}: Exhaustive search with all independent branch combinations'
                print(txt.format(current_arity), flush=True)
                g['df_cb_stats'].loc[current_arity-1, 'mode'] = 'exhaustive'
                g,id_combinations = combination.get_node_combinations(g=g, target_nodes=None,
                                                                      arity=current_arity, check_attr="name")
        else:
            if (g['exhaustive_until'] < current_arity):
                is_stat_enough = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
                num_branch_ids = is_stat_enough.sum()
                txt = 'Arity = {:,}: Heuristic search with {:,} K-1 branch combinations that passed cutoff stats ({})'
                print(txt.format(current_arity, num_branch_ids, g['cutoff_stat']), flush=True)
                g['df_cb_stats'].loc[current_arity - 1, 'mode'] = 'branch_and_bound'
            else:
                is_stat_enough = numpy.ones(shape=cb.shape[0], dtype=bool)
                txt = 'Arity = {:,}: Exhaustive search with {:,} K-1 branch combinations'
                print(txt.format(current_arity, cb.shape[0]))
                g['df_cb_stats'].loc[current_arity - 1, 'mode'] = 'exhaustive'
            id_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
            if is_stat_enough.sum()>g['max_combination']:
                txt = 'Arity = {:,}: Search will be limited to {:,} of {:,} K-1 branch combinations (see --max_combination)'
                txt = txt.format(current_arity, g['max_combination'], is_stat_enough.sum())
                print(txt, flush=True)
                cutoff_stat_exp = [ item.split(',')[0] for item in g['cutoff_stat'].split('|') ]
                is_col = False
                for cse in cutoff_stat_exp:
                    is_col |= cb.columns.str.fullmatch(cse, na=False)
                cols = cb.columns[is_col].tolist()
                branch_ids_all = cb.loc[is_stat_enough,:].sort_values(by=cols, ascending=False).loc[:,id_columns].values
                branch_ids = branch_ids_all[0:g['max_combination'],:]
                del branch_ids_all
            else:
                branch_ids = cb.loc[is_stat_enough, id_columns].values
            if len(set(branch_ids.ravel().tolist())) < current_arity:
                end_flag = 1
                cb = pandas.DataFrame()
                break
            g,id_combinations = combination.get_node_combinations(g=g, target_nodes=branch_ids, arity=current_arity,
                                                                  check_attr='name')
            if id_combinations.shape[0] == 0:
                end_flag = 1
                cb = pandas.DataFrame()
                break
        print('Preparing OCS table with {:,} process(es).'.format(g['threads']), flush=True)
        cbS = substitution.get_cb(id_combinations, S_tensor, g, 'OCS')
        print('Preparing OCN table with {:,} process(es).'.format(g['threads']), flush=True)
        cbN = substitution.get_cb(id_combinations, N_tensor, g, 'OCN')
        cb = table.merge_tables(cbS, cbN)
        del cbS, cbN
        cb = substitution.add_dif_stats(cb, g['float_tol'], prefix='OC')
        cb, g = omega.calc_omega(cb, S_tensor, N_tensor, g)
        if (g['calibrate_longtail']):
            if (g['exhaustive_until'] >= current_arity):
                cb = omega.calibrate_dsc(cb)
                g['df_cb_stats'].loc[current_arity - 1, 'dSC_calibration'] = 'Y'
            else:
                txt = '--calibrate_longtail is deactivated for arity = {}. '
                txt += 'This option is effective for the arity range specified by --exhaustive_until.\n'
                sys.stderr.write(txt.format(current_arity))
                g['df_cb_stats'].loc[current_arity - 1, 'dSC_calibration'] = 'N'
        else:
            g['df_cb_stats'].loc[current_arity - 1, 'dSC_calibration'] = 'N'
        if g['branch_dist']:
            cb = tree.get_node_distance(tree=g['tree'], cb=cb, ncpu=g['threads'], float_type=g['float_type'])
        cb = substitution.get_substitutions_per_branch(cb, b, g)
        #cb = combination.calc_substitution_patterns(cb)
        cb = table.get_linear_regression(cb)
        cb, g = foreground.get_foreground_branch_num(cb, g)
        cb = table.sort_cb(cb)
        if write_cb:
            file_name = "csubst_cb_" + str(current_arity) + ".tsv"
            cb.to_csv(file_name, sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
            txt = 'Memory consumption of cb table: {:,.1f} Mbytes (dtype={})'
            print(txt.format(cb.values.nbytes/1024/1024, cb.values.dtype), flush=True)
        g = foreground.add_median_cb_stats(g, cb, current_arity, start)
        if end_flag:
            txt = 'No branch combination satisfied phylogenetic independence. Ending higher-order search at K = {:,}.'
            print(txt.format(current_arity))
            break
    g['df_cb_stats'] = g['df_cb_stats'].loc[(~g['df_cb_stats'].loc[:,'elapsed_sec'].isnull()),:]
    g['df_cb_stats'] = g['df_cb_stats'].loc[:, sorted(g['df_cb_stats'].columns.tolist())]
    g['df_cb_stats_main'] = pandas.concat([g['df_cb_stats_main'], g['df_cb_stats']], ignore_index=True)
    return g,cb

def main_analyze(g):
    start = time.time()
    print("Reading and parsing input files.", flush=True)
    g['current_arity'] = 2
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['genetic_code'])
    g = tree.read_treefile(g)
    g = parser_misc.generate_intermediate_files(g)
    g = parser_misc.annotate_tree(g)
    g = parser_misc.read_input(g)
    g = parser_misc.prep_state(g)

    sequence.write_alignment('csubst_alignment_codon.fa', mode='codon', g=g)
    sequence.write_alignment('csubst_alignment_aa.fa', mode='aa', g=g)

    g = foreground.get_foreground_branch(g)
    g = foreground.get_marginal_branch(g)
    g = combination.get_dep_ids(g)
    tree.write_tree(g['tree'])
    tree.plot_branch_category(g['tree'], file_name='csubst_branch_id.pdf', label='all')
    tree.plot_branch_category(g['tree'], file_name='csubst_branch_id_leaf.pdf', label='leaf')
    tree.plot_branch_category(g['tree'], file_name='csubst_branch_id_nolabel.pdf', label='no')
    if g['plot_state_aa']:
        if os.path.exists('csubst_plot_state_aa'):
            shutil.rmtree('csubst_plot_state_aa')
        os.mkdir('csubst_plot_state_aa')
        os.chdir('csubst_plot_state_aa')
        tree.plot_state_tree(state=g['state_pep'], orders=g['amino_acid_orders'], mode='aa', g=g)
        os.chdir('..')
    if g['plot_state_codon']:
        if os.path.exists('csubst_plot_state_codon'):
            shutil.rmtree('csubst_plot_state_codon')
        os.mkdir('csubst_plot_state_codon')
        os.chdir('csubst_plot_state_codon')
        tree.plot_state_tree(state=g['state_cdn'], orders=g['codon_orders'], mode='codon', g=g)
        os.chdir('..')

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
        print("Generating bs table", flush=True)
        bs = substitution.get_bs(S_tensor, N_tensor)
        bs = table.sort_branch_ids(df=bs)
        bs.to_csv("csubst_bs.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        txt = 'Memory consumption of bs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(bs.values.nbytes/1024/1024, bs.values.dtype), flush=True)
        del bs
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['s']) | (g['cb']):
        start = time.time()
        print("Generating s table", flush=True)
        sS = substitution.get_s(S_tensor, attr='S')
        sN = substitution.get_s(N_tensor, attr='N')
        s = table.merge_tables(sS, sN)
        g = substitution.get_sub_sites(g, sS, sN, state_tensor=g['state_cdn'])
        del sS, sN
        if (g['s']):
            s.to_csv("csubst_s.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        txt = 'Memory consumption of s table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(s.values.nbytes/1024/1024, s.values.dtype), flush=True)
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['omegaC_method']!='submodel'):
        g['state_cdn'] = None
        g['state_pep'] = None

    if (g['b']) | (g['cb']):
        start = time.time()
        print("Generating b table", flush=True)
        bS = substitution.get_b(g, S_tensor, attr='S')
        bN = substitution.get_b(g, N_tensor, attr='N')
        b = table.merge_tables(bS, bN)
        b.loc[:,'branch_length'] = numpy.nan
        for node in g['tree'].traverse():
            b.loc[node.numerical_label,'branch_length'] = node.dist
        txt = 'Number of {} patterns among {:,} branches={:,}, min={:,.1f}, max={:,.1f}'
        for key in ['S_sub', 'N_sub']:
            p = b.loc[:, key].drop_duplicates().values
            print(txt.format(key, b.shape[0], p.shape[0], p.min(), p.max()), flush=True)
        del bS, bN
        b = foreground.annotate_foreground(b, g)
        g['branch_table'] = b
        if (g['b']):
            b.to_csv("csubst_b.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        txt = 'Memory consumption of b table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(b.values.nbytes/1024/1024, b.values.dtype), flush=True)
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cs']):
        start = time.time()
        print("Generating cs table", flush=True)
        if id_combinations is None:
            g,id_combinations = combination.get_node_combinations(g=g, arity=g['current_arity'], check_attr="name")
        csS = substitution.get_cs(id_combinations, S_tensor, attr='S')
        csN = substitution.get_cs(id_combinations, N_tensor, attr='N')
        cs = table.merge_tables(csS, csN)
        del csS, csN
        cs.to_csv("csubst_cs.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        txt = 'Memory consumption of cs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(cs.values.nbytes/1024/1024, cs.values.dtype), flush=True)
        del cs
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cbs']):
        start = time.time()
        print("Generating cbs table", flush=True)
        if id_combinations is None:
            g,id_combinations = combination.get_node_combinations(g=g, arity=g['current_arity'], check_attr="name")
        cbsS = substitution.get_cbs(id_combinations, S_tensor, attr='S', g=g)
        cbsN = substitution.get_cbs(id_combinations, N_tensor, attr='N', g=g)
        cbs = table.merge_tables(cbsS, cbsN)
        del cbsS, cbsN
        cbs.to_csv("csubst_cbs.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        txt = 'Memory consumption of cbs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(cbs.values.nbytes/1024/1024, cbs.values.dtype), flush=True)
        del cbs
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cb']):
        g['df_cb_stats_main'] = pandas.DataFrame()
        g,cb = cb_search(g, b, S_tensor, N_tensor, id_combinations, write_cb=True)
        if (g['fg_clade_permutation']>0):
            g = foreground.clade_permutation(cb, g)
        del cb
        g['df_cb_stats_main'] = table.sort_cb_stats(cb_stats=g['df_cb_stats_main'])
        print('Writing csubst_cb_stats.tsv', flush=True)
        g['df_cb_stats_main'].to_csv('csubst_cb_stats.tsv', sep="\t", index=False, float_format=g['float_format'], chunksize=10000)

    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]


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
from csubst import ete
from csubst import output_stat
from csubst import tree

def cb_search(g, b, OS_tensor, ON_tensor, id_combinations, write_cb=True):
    if int(g['max_arity']) < 2:
        raise ValueError('--max_arity should be >= 2.')
    if int(g['max_combination']) < 1:
        raise ValueError('--max_combination should be >= 1.')
    OS_tensor_reducer = substitution.get_reducer_sub_tensor(sub_tensor=OS_tensor, g=g, label='OS')
    ON_tensor_reducer = substitution.get_reducer_sub_tensor(sub_tensor=ON_tensor, g=g, label='ON')
    for current_arity in numpy.arange(2, g['max_arity'] + 1):
        start = time.time()
        g['current_arity'] = current_arity
        g = param.initialize_df_cb_stats(g)
        print("Arity (K) = {:,}: Generating cb table".format(current_arity), flush=True)
        if (current_arity==2):
            if (g['exhaustive_until']<current_arity)&(g['foreground'] is not None):
                txt = 'Arity (K) = {:,}: Targeted search of foreground branch combinations'
                print(txt.format(current_arity), flush=True)
                g['df_cb_stats'].at[0, 'mode'] = 'foreground'
                g,id_combinations = combination.get_node_combinations(g=g, target_id_dict=g['target_ids'],
                                                                      arity=current_arity, check_attr='name')
            else:
                txt = 'Arity (K) = {:,}: Exhaustive search with all independent branch combinations'
                print(txt.format(current_arity), flush=True)
                g['df_cb_stats'].at[0, 'mode'] = 'exhaustive'
                g,id_combinations = combination.get_node_combinations(g=g, exhaustive=True,
                                                                      arity=current_arity, check_attr="name")
        elif (current_arity >= 3):
            id_columns = cb.columns[cb.columns.str.startswith('branch_id_')].tolist()
            fg_columns = cb.columns[cb.columns.str.startswith('is_fg_')].tolist()
            mf_columns = cb.columns[cb.columns.str.startswith('is_mf_')].tolist()
            mg_columns = cb.columns[cb.columns.str.startswith('is_mg_')].tolist()
            cutoff_stat_entries = table.parse_cutoff_stat(cutoff_stat_str=g['cutoff_stat'])
            cutoff_stat_exp = [item[0] for item in cutoff_stat_entries]
            stat_columns = cb.columns[cb.columns.str.fullmatch('|'.join(cutoff_stat_exp), na=False)].tolist()
            cb_passed_columns = id_columns + fg_columns + mf_columns + mg_columns + stat_columns
            if (g['exhaustive_until'] < current_arity):
                is_stat_enough = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
                num_branch_ids = is_stat_enough.sum()
                txt = 'Arity (K) = {:,}: Heuristic search with {:,} K-1 branch combinations that passed cutoff stats ({})'
                print(txt.format(current_arity, num_branch_ids, g['cutoff_stat']), flush=True)
                g['df_cb_stats'].at[0, 'mode'] = 'branch_and_bound'
                if is_stat_enough.sum() > g['max_combination']:
                    txt = 'Arity (K) = {:,}: Search will be limited to {:,} of {:,} K-1 branch combinations (see --max_combination)\n'
                    txt = txt.format(current_arity, g['max_combination'], is_stat_enough.sum())
                    sys.stderr.write(txt)
                    cb_passed = cb.loc[is_stat_enough, :].sort_values(by=stat_columns, ascending=False).reset_index(drop=True)
                    cb_passed = cb_passed.iloc[:g['max_combination'], :].loc[:, cb_passed_columns].reset_index(drop=True)
                else:
                    cb_passed = cb.loc[is_stat_enough,cb_passed_columns].reset_index(drop=True)
                if len(set(cb_passed.loc[:,id_columns].values.ravel().tolist())) < current_arity:
                    cb = pandas.DataFrame()
                    txt = 'Arity (K) = {:,}: No branch combination satisfied --cutoff_stat. Ending higher-order search at K = {:,}.'
                    print(txt.format(current_arity, current_arity))
                    break
                g,id_combinations = combination.get_node_combinations(g=g, cb_passed=cb_passed, cb_all=False,
                                                                      arity=current_arity, check_attr='name')
            else:
                txt = 'Arity (K) = {:,}: Exhaustive search with {:,} K-1 branch combinations'
                print(txt.format(current_arity, cb.shape[0]))
                g['df_cb_stats'].at[0, 'mode'] = 'exhaustive'
                cb_passed = cb.loc[:,cb_passed_columns].reset_index(drop=True)
                g,id_combinations = combination.get_node_combinations(g=g, cb_passed=cb_passed, cb_all=True,
                                                                      arity=current_arity, check_attr='name')
        else:
            raise Exception('Invalid arity: {}'.format(current_arity))
        if id_combinations.shape[0] == 0:
            cb = pandas.DataFrame()
            txt = 'Arity (K) = {:,}: No branch combination satisfied phylogenetic independence. Ending higher-order search at K = {:,}.'
            print(txt.format(current_arity, current_arity))
            break
        print('Preparing OCS table with {:,} process(es).'.format(g['threads']), flush=True)
        cbOS = substitution.get_cb(
            id_combinations,
            OS_tensor_reducer,
            g,
            'OCS',
            selected_base_stats=g.get('output_base_stats'),
        )
        print('Preparing OCN table with {:,} process(es).'.format(g['threads']), flush=True)
        cbON = substitution.get_cb(
            id_combinations,
            ON_tensor_reducer,
            g,
            'OCN',
            selected_base_stats=g.get('output_base_stats'),
        )
        cb = table.merge_tables(cbOS, cbON)
        del cbOS, cbON
        cb = substitution.add_dif_stats(cb, g['float_tol'], prefix='OC', output_stats=g.get('output_stats'))
        cb, g = omega.calc_omega(cb, OS_tensor_reducer, ON_tensor_reducer, g)
        if (g['calibrate_longtail']):
            if (g['exhaustive_until'] >= current_arity):
                cb = omega.calibrate_dsc(cb, output_stats=g.get('output_stats'))
                g['df_cb_stats'].at[0, 'dSC_calibration'] = 'Y'
            else:
                txt = '--calibrate_longtail is deactivated for arity = {}. '
                txt += 'This option is effective for the arity range specified by --exhaustive_until.\n'
                sys.stderr.write(txt.format(current_arity))
                g['df_cb_stats'].at[0, 'dSC_calibration'] = 'N'
        else:
            g['df_cb_stats'].at[0, 'dSC_calibration'] = 'N'
        if g['branch_dist']:
            cb = tree.get_node_distance(tree=g['tree'], cb=cb, ncpu=g['threads'], float_type=g['float_type'])
        cb = substitution.get_substitutions_per_branch(cb, b, g)
        #cb = combination.calc_substitution_patterns(cb)
        cb = table.get_linear_regression(cb)
        cb = output_stat.drop_unrequested_stat_columns(cb, g.get('output_stats'))
        cb, g = foreground.get_foreground_branch_num(cb, g)
        cb = table.sort_cb(cb)
        if write_cb:
            file_name = "csubst_cb_" + str(current_arity) + ".tsv"
            cb_column_original = cb.columns.tolist()
            cb.columns = cb.columns.str.replace('_PLACEHOLDER', '')
            cb.to_csv(file_name, sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
            cb.columns = cb_column_original
            txt = 'Memory consumption of cb table: {:,.1f} Mbytes (dtype={})'
            print(txt.format(cb.values.nbytes/1024/1024, cb.values.dtype), flush=True)
        g = foreground.add_median_cb_stats(g, cb, current_arity, start)
        if (g['fg_clade_permutation']>0):
            g = foreground.clade_permutation(
                cb=cb,
                g=g,
                OS_tensor_reducer=OS_tensor_reducer,
                ON_tensor_reducer=ON_tensor_reducer,
            )
        g['df_cb_stats'] = g['df_cb_stats'].loc[:, sorted(g['df_cb_stats'].columns.tolist())]
        g['df_cb_stats_main'] = pandas.concat([g['df_cb_stats_main'], g['df_cb_stats']], ignore_index=True)
        if current_arity == g['max_arity']:
            txt = 'Maximum arity (K = {:,}) reached. Ending higher-order search of branch combinations.'
            print(txt.format(g['max_arity']))
            break
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
    g = foreground.get_foreground_branch(g)
    g = foreground.get_marginal_branch(g)
    g = parser_misc.resolve_state_loading(g)
    g = parser_misc.prep_state(g)
    loaded_branch_ids = g.get('state_loaded_branch_ids', None)
    if loaded_branch_ids is not None:
        txt = 'Selective state loading active: writing alignments only for loaded nodes ({:,}).'
        print(txt.format(loaded_branch_ids.shape[0]), flush=True)
    sequence.write_alignment('csubst_alignment_codon.fa', mode='codon', g=g, branch_ids=loaded_branch_ids)
    sequence.write_alignment('csubst_alignment_aa.fa', mode='aa', g=g, branch_ids=loaded_branch_ids)
    g = combination.get_dep_ids(g)
    tree.write_tree(g['tree'])
    tree.plot_branch_category(g, file_base='csubst_branch_id', label='all')
    if g['more_tree_plot']:
        tree.plot_branch_category(g, file_base='csubst_branch_id_leaf', label='leaf')
        tree.plot_branch_category(g, file_base='csubst_branch_id_nolabel', label='no')
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
    ON_tensor = substitution.get_substitution_tensor(state_tensor=g['state_pep'], mode='asis', g=g, mmap_attr='N')
    ON_tensor = substitution.apply_min_sub_pp(g, ON_tensor)
    sub_branches = numpy.where(substitution.get_branch_sub_counts(ON_tensor) != 0)[0].tolist()
    OS_tensor = substitution.get_substitution_tensor(state_tensor=g['state_cdn'], mode='syn', g=g, mmap_attr='S')
    OS_tensor = substitution.apply_min_sub_pp(g, OS_tensor)
    sub_branches = list(set(sub_branches).union(set(numpy.where(substitution.get_branch_sub_counts(OS_tensor) != 0)[0].tolist())))
    g['sub_branches'] = sub_branches
    g = tree.rescale_branch_length(g, OS_tensor, ON_tensor)
    id_combinations = None
    OS_total = substitution.get_total_substitution(OS_tensor)
    ON_total = substitution.get_total_substitution(ON_tensor)
    num_branch = g['num_node'] - 1
    num_site = OS_tensor.shape[1]
    print('Synonymous substitutions / tree = {:,.1f}'.format(OS_total), flush=True)
    print('Nonsynonymous substitutions / tree = {:,.1f}'.format(ON_total), flush=True)
    print('Synonymous substitutions / branch = {:,.1f}'.format(OS_total / num_branch), flush=True)
    print('Nonsynonymous substitutions / branch = {:,.1f}'.format(ON_total / num_branch), flush=True)
    print('Synonymous substitutions / site = {:,.1f}'.format(OS_total / num_site), flush=True)
    print('Nonsynonymous substitutions / site = {:,.1f}'.format(ON_total / num_site), flush=True)
    elapsed_time = int(time.time() - start)
    print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['bs']):
        start = time.time()
        print("Generating bs table", flush=True)
        bs = substitution.get_bs(OS_tensor, ON_tensor)
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
        sOS = substitution.get_s(OS_tensor, attr='S')
        sON = substitution.get_s(ON_tensor, attr='N')
        s = table.merge_tables(sOS, sON)
        g = substitution.get_sub_sites(g, sOS, sON, state_tensor=g['state_cdn'])
        del sOS, sON
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
        bOS = substitution.get_b(g=g, sub_tensor=OS_tensor, attr='S', sitewise=False)
        bON = substitution.get_b(g=g, sub_tensor=ON_tensor, attr='N', sitewise=True)
        b = table.merge_tables(bOS, bON)
        b.loc[:,'branch_length'] = numpy.nan
        for node in g['tree'].traverse():
            b.loc[ete.get_prop(node, "numerical_label"),'branch_length'] = node.dist
        txt = 'Number of {} patterns among {:,} branches={:,}, min={:,.1f}, max={:,.1f}'
        for key in ['S_sub', 'N_sub']:
            p = b.loc[:, key].drop_duplicates().values
            print(txt.format(key, b.shape[0], p.shape[0], p.min(), p.max()), flush=True)
        del bOS, bON
        b = foreground.annotate_b_foreground(b, g)
        g['branch_table'] = b
        if (g['b']):
            b_column_original = b.columns.tolist()
            b.columns = b.columns.str.replace('_PLACEHOLDER', '')
            b.to_csv("csubst_b.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
            b.columns = b_column_original
        txt = 'Memory consumption of b table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(b.values.nbytes/1024/1024, b.values.dtype), flush=True)
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cs']):
        start = time.time()
        print("Generating cs table", flush=True)
        if id_combinations is None:
            g,id_combinations = combination.get_node_combinations(g=g, exhaustive=True, arity=g['current_arity'], check_attr="name")
        reducer_OS_tensor = substitution.get_reducer_sub_tensor(sub_tensor=OS_tensor, g=g, label='csOS')
        reducer_ON_tensor = substitution.get_reducer_sub_tensor(sub_tensor=ON_tensor, g=g, label='csON')
        csOS = substitution.get_cs(id_combinations, reducer_OS_tensor, attr='S')
        csON = substitution.get_cs(id_combinations, reducer_ON_tensor, attr='N')
        cs = table.merge_tables(csOS, csON)
        del csOS, csON
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
            g,id_combinations = combination.get_node_combinations(g=g, exhaustive=True, arity=g['current_arity'], check_attr="name")
        cbsOS = substitution.get_cbs(id_combinations, OS_tensor, attr='S', g=g)
        cbsON = substitution.get_cbs(id_combinations, ON_tensor, attr='N', g=g)
        cbs = table.merge_tables(cbsOS, cbsON)
        del cbsOS, cbsON
        cbs.to_csv("csubst_cbs.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        txt = 'Memory consumption of cbs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(cbs.values.nbytes/1024/1024, cbs.values.dtype), flush=True)
        del cbs
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cb']):
        g['df_cb_stats_main'] = pandas.DataFrame()
        g,cb = cb_search(g, b, OS_tensor, ON_tensor, id_combinations, write_cb=True)
        #if (g['fg_clade_permutation']>0):
        #    g = foreground.clade_permutation(cb, g)
        #del cb
        g['df_cb_stats_main'] = table.sort_cb_stats(cb_stats=g['df_cb_stats_main'])
        print('Writing csubst_cb_stats.tsv', flush=True)
        column_original = g['df_cb_stats_main'].columns
        g['df_cb_stats_main'].columns = pandas.Index(
            [str(col).replace('_PLACEHOLDER', '') for col in column_original]
        )
        g['df_cb_stats_main'].to_csv('csubst_cb_stats.tsv', sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        g['df_cb_stats_main'].columns = column_original

    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]

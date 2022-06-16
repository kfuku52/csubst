import numpy
import pandas

import itertools
import os
import shutil
import sys
import time
import warnings

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

def add_median_cb_stats(g, cb, current_arity, start, verbose=True):
    is_arity = (g['df_cb_stats'].loc[:,'arity'] == current_arity)
    suffices = list()
    is_targets = list()
    suffices.append('_all')
    is_targets.append(numpy.ones(shape=cb.shape[0], dtype=numpy.bool))
    target_cols = ['is_fg','is_mg','is_mf','dummy']
    suffix_candidates = ['_fg','_mg','_mf']
    if g['exhaustive_until']>=current_arity:
        suffix_candidates.append('_all')
    for target_col,sc in zip(target_cols,suffix_candidates):
        if target_col in cb.columns:
            suffices.append(sc)
            if sc=='_all':
                is_targets.append(True)
            else:
                is_targets.append(cb.loc[:,target_col]=='Y')
    stats = dict()
    omega_cols = cb.columns[cb.columns.str.startswith('omegaC')].tolist()
    is_ON = cb.columns.str.startswith('OCNany') | cb.columns.str.startswith('OCNdif') | cb.columns.str.startswith('OCNspe')
    is_OS = cb.columns.str.startswith('OCSany') | cb.columns.str.startswith('OCSdif') | cb.columns.str.startswith('OCSspe')
    is_EN = cb.columns.str.startswith('ECNany') | cb.columns.str.startswith('ECNdif') | cb.columns.str.startswith('ECNspe')
    is_ES = cb.columns.str.startswith('ECSany') | cb.columns.str.startswith('ECSdif') | cb.columns.str.startswith('ECSspe')
    ON_cols = cb.columns[is_ON].tolist()
    OS_cols = cb.columns[is_OS].tolist()
    EN_cols = cb.columns[is_EN].tolist()
    ES_cols = cb.columns[is_ES].tolist()
    stats['median'] = ['dist_bl','dist_node_num',] + omega_cols
    stats['total'] = ON_cols + OS_cols + EN_cols + ES_cols
    is_qualified = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
    for suffix, is_target in zip(suffices, is_targets):
        g['df_cb_stats'].loc[is_arity, 'num' + suffix] = is_target.sum()
        g['df_cb_stats'].loc[is_arity, 'num_qualified' + suffix] = (is_target&is_qualified).sum()
    for stat in stats.keys():
        for suffix,is_target in zip(suffices,is_targets):
            for ms in stats[stat]:
                col = stat+'_'+ms+suffix
                if not col in g['df_cb_stats'].columns:
                    newcol = pandas.DataFrame({col:numpy.zeros(shape=(g['df_cb_stats'].shape[0]))})
                    g['df_cb_stats'] = pandas.concat([g['df_cb_stats'], newcol], ignore_index=False, axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if stat=='median':
                        g['df_cb_stats'].loc[is_arity,col] = cb.loc[is_target,ms].median()
                    elif stat=='total':
                        g['df_cb_stats'].loc[is_arity,col] = cb.loc[is_target,ms].sum()
    if verbose:
        for SN,anc,des in itertools.product(['S','N'], ['any','dif','spe'], ['any','dif','spe']):
            key = SN+anc+'2'+des
            totalON = g['df_cb_stats'].loc[is_arity, 'total_OC'+key+'_all'].values[0]
            totalEN = g['df_cb_stats'].loc[is_arity, 'total_EC'+key+'_all'].values[0]
            if totalON==0:
                percent_value = numpy.nan
            else:
                percent_value = totalEN / totalON * 100
            txt = 'Total OC{}/EC{} = {:,.1f}/{:,.1f} (Expectation equals to {:,.1f}% of the observation.)'
            print(txt.format(key, key, totalON, totalEN, percent_value))
    elapsed_time = int(time.time() - start)
    g['df_cb_stats'].loc[is_arity, 'elapsed_sec'] = elapsed_time
    if verbose:
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)
    return g

def cb_search(g, b, S_tensor, N_tensor, id_combinations, mode='', write_cb=True):
    end_flag = 0
    g = param.initialize_df_cb_stats(g)
    for current_arity in numpy.arange(2, g['max_arity'] + 1):
        start = time.time()
        print("Generating combinat-branch table. Arity = {:,}".format(current_arity), flush=True)
        g['current_arity'] = current_arity
        if (current_arity==2):
            if (g['exhaustive_until']<current_arity)&(g['foreground'] is not None):
                print('Searching foreground branch combinations only.', flush=True)
                g['df_cb_stats'].loc[current_arity-1, 'mode'] = 'foreground'
                g,id_combinations = combination.get_node_combinations(g=g, target_nodes=g['target_id'],
                                                                      arity=current_arity, check_attr='name')
            else:
                print('Exhaustively searching independent branch combinations.', flush=True)
                g['df_cb_stats'].loc[current_arity-1, 'mode'] = 'exhaustive'
                g,id_combinations = combination.get_node_combinations(g=g, target_nodes=None,
                                                                      arity=current_arity, check_attr="name")
        else:
            if (g['exhaustive_until'] < current_arity):
                is_stat_enough = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
                num_branch_ids = is_stat_enough.sum()
                txt = 'Arity = {:,}: Branch combinations at K-1 that passed cutoff stats ({}): {:,}'
                print(txt.format(current_arity, g['cutoff_stat'], num_branch_ids), flush=True)
                g['df_cb_stats'].loc[current_arity - 1, 'mode'] = 'branch_and_bound'
            else:
                is_stat_enough = numpy.ones(shape=cb.shape[0], dtype=bool)
                txt = 'Arity = {:,}: Exhaustive search from {:,} branch combinations at K-1.'
                print(txt.format(current_arity, cb.shape[0]))
                g['df_cb_stats'].loc[current_arity - 1, 'mode'] = 'exhaustive'
            id_columns = cb.columns[cb.columns.str.startswith('branch_id_')]
            if is_stat_enough.sum()>g['max_combination']:
                txt = 'Only {:,} of {:,} branch combinations at arity={} will be used to search arity={}'
                txt = txt.format(g['max_combination'], is_stat_enough.sum(), current_arity-1, current_arity)
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
                break
            del cb
            g,id_combinations = combination.get_node_combinations(g=g, target_nodes=branch_ids, arity=current_arity,
                                                                  check_attr='name')
            if id_combinations.shape[0] == 0:
                end_flag = 1
                break
        print('Preparing the OCS table with {:,} process(es).'.format(g['threads']), flush=True)
        cbS = substitution.get_cb(id_combinations, S_tensor, g, 'OCS')
        print('Preparing the OCN table with {:,} process(es).'.format(g['threads']), flush=True)
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
        g = add_median_cb_stats(g, cb, current_arity, start)
        if end_flag:
            print('No combination satisfied phylogenetic independence. Ending branch combination analysis.')
            break
    g['df_cb_stats'] = g['df_cb_stats'].loc[(~g['df_cb_stats'].loc[:,'elapsed_sec'].isnull()),:]
    g['df_cb_stats'] = g['df_cb_stats'].loc[:, sorted(g['df_cb_stats'].columns.tolist())]
    g['df_cb_stats_main'] = pandas.concat([g['df_cb_stats_main'], g['df_cb_stats']], ignore_index=True)
    return g,cb

def clade_permutation(cb, g):
    print('Starting clade permutation. Note that --fg_random examine the arity of 2.')
    for i in numpy.arange(g['fg_random']):
        start = time.time()
        print('Starting foreground randomization round {:,}'.format(i+1), flush=True)
        g = param.initialize_df_cb_stats(g)
        g['df_cb_stats'] = g['df_cb_stats'].loc[(g['df_cb_stats'].loc[:,'arity']==2),:].reset_index(drop=True)
        g,rid_combinations = foreground.set_random_foreground_branch(g)
        random_mode = 'randomization_iter'+str(i+1)+'_bid'+','.join(g['fg_id'].astype(str))
        bid_columns = [ 'branch_id_'+str(k+1) for k in numpy.arange(rid_combinations.shape[1]) ]
        rid_combinations = pandas.DataFrame(rid_combinations)
        rid_combinations.columns = bid_columns
        rcb = pandas.merge(rid_combinations, cb, how='left', on=bid_columns)
        rcb.loc[:,'is_fg'] = 'Y'
        rcb.loc[:,'is_mg'] = 'N'
        rcb.loc[:,'is_mf'] = 'N'
        g = add_median_cb_stats(g, rcb, 2, start, verbose=False)
        g['df_cb_stats'].loc[:,'mode'] = random_mode
        g['df_cb_stats_main'] = pandas.concat([g['df_cb_stats_main'], g['df_cb_stats']], ignore_index=True)
        print('Ending foreground randomization round {:,}\n'.format(i+1), flush=True)
    is_arity2 = (g['df_cb_stats_main'].loc[:,'arity']==2)
    is_stat_fg = ~g['df_cb_stats_main'].loc[:,'mode'].str.startswith('randomization_')
    is_stat_permutation = g['df_cb_stats_main'].loc[:,'mode'].str.startswith('randomization_')
    obs_value = g['df_cb_stats_main'].loc[is_arity2 & is_stat_fg,'median_omegaCany2spe_fg'].values[0]
    permutation_values = g['df_cb_stats_main'].loc[is_arity2 & is_stat_permutation, 'median_omegaCany2spe_fg'].values
    num_positive = (obs_value<=permutation_values).sum()
    num_all = permutation_values.shape[0]
    pvalue = num_positive / num_all
    obs_ocn = g['df_cb_stats_main'].loc[is_arity2 & is_stat_fg,'total_OCNany2spe_fg'].values[0]
    print('Observed total OCNany2spe in foreground lineages = {:,.3}'.format(obs_ocn))
    permutation_ocns = g['df_cb_stats_main'].loc[is_arity2 & is_stat_permutation, 'total_OCNany2spe_fg'].values
    txt = 'Total OCNany2spe in permutation lineages = {:,.3} ± {:,.3} (mean ± SD)'
    print(txt.format(permutation_ocns.mean(), permutation_ocns.std()))
    print('Observed median omegaCany2spe in foreground lineages = {:,.3}'.format(obs_value))
    txt = 'Median omegaCany2spe in permutation lineages = {:,.3} ± {:,.3} (mean ± SD)'
    print(txt.format(permutation_values.mean(), permutation_values.std()))
    txt = 'P value of foreground convergence (omegaCany2spe) by clade permutations = {} (observation <= permutation = {:,}/{:,})'
    print(txt.format(pvalue, num_positive, num_all))
    return g

def main_analyze(g):
    start = time.time()
    print("Reading and parsing input files.", flush=True)
    g['current_arity'] = 2
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['genetic_code'])
    g = tree.read_treefile(g)
    g = parser_misc.generate_intermediate_files(g)
    g = parser_misc.annotate_tree(g)
    g = parser_misc.read_input(g)
    g,g['state_nuc'],g['state_cdn'],g['state_pep'] = parser_misc.prep_state(g)

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
        print("Generating branch-site table.", flush=True)
        bs = substitution.get_bs(S_tensor, N_tensor)
        bs = table.sort_labels(df=bs)
        bs.to_csv("csubst_bs.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
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
            b.to_csv("csubst_b.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
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
        cs.to_csv("csubst_cs.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
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
        cbs.to_csv("csubst_cbs.tsv", sep="\t", index=False, float_format=g['float_format'], chunksize=10000)
        txt = 'Memory consumption of cbs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(cbs.values.nbytes/1024/1024, cbs.values.dtype), flush=True)
        del cbs
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

    if (g['cb']):
        g['df_cb_stats_main'] = pandas.DataFrame()
        g,cb = cb_search(g, b, S_tensor, N_tensor, id_combinations, mode='foreground', write_cb=True)
        if (g['fg_random']>0):
            g = clade_permutation(cb, g)
        del cb
        g['df_cb_stats_main'] = table.sort_cb_stats(cb_stats=g['df_cb_stats_main'])
        g['df_cb_stats_main'].to_csv('csubst_cb_stats.tsv', sep="\t", index=False, float_format=g['float_format'], chunksize=10000)

    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]


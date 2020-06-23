import time

from csubst import parser_iqtree
from csubst import parser_phylobayes
from csubst.genetic_code import *
from csubst.omega import *
from csubst.param import *
from csubst.sequence import *
from csubst.table import *
from csubst.tree import *
from csubst.substitution import get_sub_sites

def cb_search(g, b, S_tensor, N_tensor, id_combinations, mode='', write_cb=True):
    end_flag = 0
    g = initialize_df_cb_stats(g)
    g['df_cb_stats'].loc[:, 'mode'] = mode
    for current_arity in numpy.arange(2, g['max_arity'] + 1):
        start = time.time()
        print("Making combinat-branch table. Arity = {:,}".format(current_arity), flush=True)
        g['current_arity'] = current_arity
        if (current_arity == 2) & ((g['foreground'] is None)|(g['fg_force_exhaustive'])):
            id_combinations = id_combinations
        elif (current_arity == 2) & (g['foreground'] is not None):
            id_combinations = get_node_combinations(g=g, target_nodes=g['target_id'], arity=current_arity,
                                                    check_attr='name', foreground=True)
        elif (current_arity > 2):
            is_stat_enough = (cb.loc[:,g['target_stat']] >= g['min_stat']) | (cb.loc[:,g['target_stat']].isnull())
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
            id_combinations = get_node_combinations(g=g, target_nodes=branch_ids, arity=current_arity,
                                                    check_attr='name', foreground=True)
            if id_combinations.shape[0] == 0:
                end_flag = 1
                break
        cbS = get_cb(id_combinations, S_tensor, g, 'S')
        cbN = get_cb(id_combinations, N_tensor, g, 'N')
        cb = merge_tables(cbS, cbN)
        del cbS, cbN
        cb = get_node_distance(g['tree'], cb)
        cb = get_substitutions_per_branch(cb, b, g)
        cb = calc_substitution_patterns(cb)
        cb, g = calc_omega(cb, b, S_tensor, N_tensor, g)
        cb, g = get_foreground_branch_num(cb, g)
        if write_cb:
            file_name = "csubst_cb_" + str(current_arity) + ".tsv"
            cb.to_csv(file_name, sep="\t", index=False, float_format='%.4f', chunksize=10000)
            print(cb.info(verbose=False, max_cols=0, memory_usage=True, null_counts=False), flush=True)
        is_arity = (g['df_cb_stats'].loc[:,'arity'] == current_arity)
        med_dist_bl = cb.loc[(cb.loc[:,'fg_branch_num']==current_arity),'dist_bl'].median()
        med_dist_node_num = cb.loc[(cb.loc[:,'fg_branch_num']==current_arity),'dist_node_num'].median()
        g['df_cb_stats'].loc[is_arity,'fg_median_dist_bl'] = med_dist_bl
        g['df_cb_stats'].loc[is_arity,'fg_median_dist_node_num'] = med_dist_node_num
        elapsed_time = int(time.time() - start)
        g['df_cb_stats'].loc[is_arity, 'elapsed_sec'] = elapsed_time
        print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)
        if end_flag:
            print('No combination satisfied phylogenetic independency. Ending branch combination analysis.')
            break
    g['df_cb_stats'] = g['df_cb_stats'].loc[(~g['df_cb_stats'].loc[:,'elapsed_sec'].isnull()),:]
    g['df_cb_stats'] = g['df_cb_stats'].loc[:, sorted(g['df_cb_stats'].columns.tolist())]
    g['df_cb_stats_main'] = pandas.concat([g['df_cb_stats_main'], g['df_cb_stats']], ignore_index=True)
    return g

def csubst_main(g):
    start = time.time()
    print("Reading and parsing input files.", flush=True)
    g['current_arity'] = 2
    g['codon_table'] = get_codon_table(ncbi_id=g['ncbi_codon_table'])
    if g['infile_type'] == 'phylobayes':
        g = parser_phylobayes.get_input_information(g)
        if g['input_data_type'] == 'nuc':
            state_nuc = parser_phylobayes.get_state_tensor(g)
            if (g['calc_omega']):
                state_cdn = calc_omega_state(state_nuc=state_nuc, g=g)
                state_pep = cdn2pep_state(state_cdn=state_cdn, g=g)
        if g['input_data_type'] == 'cdn':
            state_cdn = parser_phylobayes.get_state_tensor(g)
            state_pep = cdn2pep_state(state_cdn=state_cdn, g=g)
    elif g['infile_type'] == 'iqtree':
        g = parser_iqtree.get_input_information(g)
        if g['input_data_type'] == 'nuc':
            state_nuc = parser_iqtree.get_state_tensor(g)
            if (g['calc_omega']):
                state_cdn = calc_omega_state(state_nuc=state_nuc, g=g)
                state_pep = cdn2pep_state(state_cdn=state_cdn, g=g)
        if g['input_data_type'] == 'cdn':
            state_cdn = parser_iqtree.get_state_tensor(g)
            state_pep = cdn2pep_state(state_cdn=state_cdn, g=g)

    write_alignment(state=state_cdn, orders=g['codon_orders'], outfile='csubst_alignment_codon.fa', g=g)
    write_alignment(state=state_pep, orders=g['amino_acid_orders'], outfile='csubst_alignment_aa.fa', g=g)

    g = get_foreground_branch(g)
    g = get_marginal_branch(g)
    g = get_dep_ids(g)
    write_tree(g['tree'])
    plot_branch_category(g)

    N_tensor = get_substitution_tensor(state_tensor=state_pep, mode='asis', g=g, mmap_attr='N')
    sub_branches = numpy.where(N_tensor.sum(axis=(1, 2, 3, 4)) != 0)[0].tolist()
    if (g['calc_omega']):
        S_tensor = get_substitution_tensor(state_tensor=state_cdn, mode='syn', g=g, mmap_attr='S')
        sub_branches = list(set(sub_branches).union(set(numpy.where(S_tensor.sum(axis=(1, 2, 3, 4)) != 0)[0].tolist())))
    g['sub_branches'] = sub_branches

    id_combinations = get_node_combinations(g=g, arity=g['current_arity'], check_attr="name")

    S_total = numpy.nan_to_num(S_tensor).sum(axis=(0, 1, 2, 3, 4))
    N_total = numpy.nan_to_num(N_tensor).sum(axis=(0, 1, 2, 3, 4))
    num_branch = g['num_node'] - 1
    num_site = S_tensor.shape[2]
    print('Synonymous substitutions / tree = {:,}'.format(S_total), flush=True)
    print('Nonsynonymous substitutions / tree = {:,}'.format(N_total), flush=True)
    print('Synonymous substitutions / branch =', S_total / num_branch, flush=True)
    print('Nonsynonymous substitutions / branch =', N_total / num_branch, flush=True)
    print('Synonymous substitutions / site =', S_total / num_site, flush=True)
    print('Nonsynonymous substitutions / site =', N_total / num_site, flush=True)
    elapsed_time = int(time.time() - start)
    print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

    if (g['bs']):
        start = time.time()
        print("Making branch-site table.", flush=True)
        bs = get_bs(S_tensor, N_tensor)
        bs = sort_labels(df=bs)
        bs.to_csv("csubst_bs.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        print(bs.info(verbose=False, max_cols=0, memory_usage=True, null_counts=False), flush=True)
        del bs
        elapsed_time = int(time.time() - start)
        print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

    if (g['s']) | (g['cb']):
        start = time.time()
        print("Making site table.", flush=True)
        sS = get_s(S_tensor, attr='S')
        sN = get_s(N_tensor, attr='N')
        s = merge_tables(sS, sN)
        g = get_sub_sites(g, sS, sN, state_tensor=state_cdn)
        del sS, sN
        if (g['s']):
            s.to_csv("csubst_s.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        print(s.info(verbose=False, max_cols=0, memory_usage=True, null_counts=False), flush=True)
        elapsed_time = int(time.time() - start)
        print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

    del state_cdn, state_pep

    if (g['b']) | (g['cb']):
        start = time.time()
        print("Making branch table.", flush=True)
        bS = get_b(g, S_tensor, attr='S')
        bN = get_b(g, N_tensor, attr='N')
        b = merge_tables(bS, bN)
        txt = 'Number of {} patterns among {:,} branches={:,}, min={:,}, max={:,}'
        for key in ['S_sub', 'N_sub']:
            p = b.loc[:, key].drop_duplicates().values
            print(txt.format(key, b.shape[0], p.shape[0], p.min(), p.max()), flush=True)
        del bS, bN
        b = annotate_foreground(b, g)
        if (g['b']):
            b.to_csv("csubst_b.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        print(b.info(verbose=False, max_cols=0, memory_usage=True, null_counts=False), flush=True)
        elapsed_time = int(time.time() - start)
        print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

    if (g['cs']):
        start = time.time()
        print("Making combinat-site table.", flush=True)
        csS = get_cs(id_combinations, S_tensor, attr='S')
        csN = get_cs(id_combinations, N_tensor, attr='N')
        cs = merge_tables(csS, csN)
        del csS, csN
        cs.to_csv("csubst_cs.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        print(cs.info(verbose=False, max_cols=0, memory_usage=True, null_counts=False), flush=True)
        del cs
        elapsed_time = int(time.time() - start)
        print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

    if (g['cbs']):
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

    if (g['cb']):
        g['df_cb_stats_main'] = pandas.DataFrame()
        g = cb_search(g, b, S_tensor, N_tensor, id_combinations, mode='foreground', write_cb=True)
        if (g['foreground'] is not None)&(g['fg_random']>0):
            for i in numpy.arange(0, g['fg_random']):
                print('starting foreground randomization round {:,}'.format(i+1), flush=True)
                g = get_foreground_branch(g)
                g = randomize_foreground_branch(g)
                g = get_marginal_branch(g)
                g = cb_search(g, b, S_tensor, N_tensor, mode='randomization_'+str(i+1), write_cb=False)
                print('ending foreground randomization round {:,}\n'.format(i+1), flush=True)

        g['df_cb_stats_main'].to_csv('csubst_cb_stats.tsv', sep="\t", index=False, float_format='%.4f', chunksize=10000)

    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]


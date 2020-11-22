import time

#from scipy.stats import chi2_contingency

from csubst import parser_misc
from csubst import genetic_code
from csubst import foreground
from csubst.omega import *
from csubst.param import *
from csubst.sequence import *
from csubst.table import *
from csubst.tree import *
from csubst.substitution import get_sub_sites

def get_linear_regression(cb):
    for prefix in ['S','N']:
        x = cb.loc[:,prefix+'any2any'].values
        y = cb.loc[:,prefix+'any2spe'].values
        x = x[:,numpy.newaxis]
        coef,residuals,rank,s = numpy.linalg.lstsq(x, y, rcond=None)
        cb.loc[:,prefix+'_linreg_residual'] = y - (x[:,0]*coef[0])
    return cb


def chisq_test(x, total_S, total_N):
    obs = x.loc[['Sany2spe','Nany2spe']].values
    if obs.sum()==0:
        return 1
    else:
        contingency_table = numpy.array([obs, [total_S, total_N]])
        out = chi2_contingency(contingency_table, lambda_="log-likelihood")
        return out[1]

def cb_search(g, b, S_tensor, N_tensor, id_combinations, mode='', write_cb=True):
    end_flag = 0
    g = initialize_df_cb_stats(g)
    g['df_cb_stats'].loc[:,'mode'] = mode
    for current_arity in numpy.arange(2, g['max_arity'] + 1):
        start = time.time()
        print("Making combinat-branch table. Arity = {:,}".format(current_arity), flush=True)
        g['current_arity'] = current_arity
        if (current_arity == 2):
            if (g['foreground'] is not None) & (g['fg_force_exhaustive']==False):
                print('Searching foreground branch combinations.', flush=True)
                g,id_combinations = get_node_combinations(g=g, target_nodes=g['target_id'], arity=current_arity,
                                                          check_attr='name')
            else:
                print('Exhaustively searching independent branch combinations.', flush=True)
                if id_combinations is None:
                    g,id_combinations = get_node_combinations(g=g, arity=current_arity, check_attr="name")
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
            g,id_combinations = get_node_combinations(g=g, target_nodes=branch_ids, arity=current_arity,
                                                    check_attr='name')
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
        cb = get_linear_regression(cb)
        #total_S = cb.loc[:,'Sany2spe'].sum()
        #total_N = cb.loc[:,'Nany2spe'].sum()
        #cb.loc[:,'chisq_p'] = cb.apply(chisq_test, args=(total_S, total_N), axis=1)
        cb, g = foreground.get_foreground_branch_num(cb, g)
        if write_cb:
            file_name = "csubst_cb_" + str(current_arity) + ".tsv"
            cb.to_csv(file_name, sep="\t", index=False, float_format='%.4f', chunksize=10000)
            txt = 'Memory consumption of cb table: {:,.1f} Mbytes (dtype={})'
            print(txt.format(cb.values.nbytes/1024/1024, cb.values.dtype), flush=True)
        is_arity = (g['df_cb_stats'].loc[:,'arity'] == current_arity)
        is_fg = (cb.loc[:,'fg_branch_num']==current_arity)
        median_stats = ['dist_bl','dist_node_num','omega_any2any','omega_any2spe','omega_any2dif']
        for ms in median_stats:
            col_all = 'median_'+ms+'_all'
            g['df_cb_stats'].loc[is_arity,col_all] = cb.loc[:,ms].median()
            if any(is_fg):
                col_fg = 'median_'+ms+'_fg'
                g['df_cb_stats'].loc[is_arity,col_fg] = cb.loc[is_fg,ms].median()
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

def main_analyze(g):
    start = time.time()
    print("Reading and parsing input files.", flush=True)
    g['current_arity'] = 2
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['ncbi_codon_table'])
    g = parser_misc.read_input(g)
    g,g['state_nuc'],g['state_cdn'],g['state_pep'] = parser_misc.prep_state(g)

    write_alignment(state=g['state_cdn'], orders=g['codon_orders'], outfile='csubst_alignment_codon.fa', mode='codon', g=g)
    write_alignment(state=g['state_pep'], orders=g['amino_acid_orders'], outfile='csubst_alignment_aa.fa', mode='aa', g=g)

    g = foreground.get_foreground_branch(g)
    g = foreground.get_marginal_branch(g)
    g = get_dep_ids(g)
    write_tree(g['tree'])
    plot_branch_category(g, file_name='csubst_branch_category.pdf')
    if g['plot_state_aa']:
        plot_state_tree(state=g['state_pep'], orders=g['amino_acid_orders'], mode='aa', g=g)
    if g['plot_state_codon']:
        plot_state_tree(state=g['state_cdn'], orders=g['codon_orders'], mode='codon', g=g)

    N_tensor = get_substitution_tensor(state_tensor=g['state_pep'], mode='asis', g=g, mmap_attr='N')
    sub_branches = numpy.where(N_tensor.sum(axis=(1, 2, 3, 4)) != 0)[0].tolist()
    S_tensor = get_substitution_tensor(state_tensor=g['state_cdn'], mode='syn', g=g, mmap_attr='S')
    sub_branches = list(set(sub_branches).union(set(numpy.where(S_tensor.sum(axis=(1, 2, 3, 4)) != 0)[0].tolist())))
    g['sub_branches'] = sub_branches

    print('Branch lengths of the IQ-TREE output are rescaled to match observed-codon-substitutions/codon-site, '
          'rather than nucleotide-substitutions/codon-site.')
    print('Total branch length before rescaling: {:,.3f}'.format(sum([ n.dist for n in g['tree'].traverse() ])))
    for node in g['tree'].traverse():
        if node.is_root():
            node.Sdist = 0
            node.Ndist = 0
            node.SNdist = 0
            continue
        nl = node.numerical_label
        parent = node.up.numerical_label
        num_nonmissing_codon = (g['state_cdn'][(nl,parent),:,:].sum(axis=2).sum(axis=0)!=0).sum()
        if num_nonmissing_codon==0:
            node.Sdist = 0
            node.Ndist = 0
            node.SNdist = 0
            continue
        num_S_sub = S_tensor[nl,:,:,:,:].sum()
        num_N_sub = N_tensor[nl,:,:,:,:].sum()
        node.Sdist = num_S_sub / num_nonmissing_codon
        node.Ndist = num_N_sub / num_nonmissing_codon
        node.SNdist = (num_S_sub + num_N_sub) / num_nonmissing_codon
    print('Total S+N branch length after rescaling: {:,.3f}'.format(sum([ n.SNdist for n in g['tree'].traverse() ])))
    print('Total S branch length after rescaling: {:,.3f}'.format(sum([ n.Sdist for n in g['tree'].traverse() ])))
    print('Total N branch length after rescaling: {:,.3f}'.format(sum([ n.Ndist for n in g['tree'].traverse() ])))

    id_combinations = None

    S_total = numpy.nan_to_num(S_tensor).sum(axis=(0, 1, 2, 3, 4))
    N_total = numpy.nan_to_num(N_tensor).sum(axis=(0, 1, 2, 3, 4))
    num_branch = g['num_node'] - 1
    num_site = S_tensor.shape[1]
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
        txt = 'Memory consumption of bs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(bs.values.nbytes/1024/1024, bs.values.dtype), flush=True)
        del bs
        elapsed_time = int(time.time() - start)
        print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

    if (g['s']) | (g['cb']):
        start = time.time()
        print("Making site table.", flush=True)
        sS = get_s(S_tensor, attr='S')
        sN = get_s(N_tensor, attr='N')
        s = merge_tables(sS, sN)
        g = get_sub_sites(g, sS, sN, state_tensor=g['state_cdn'])
        del sS, sN
        if (g['s']):
            s.to_csv("csubst_s.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        txt = 'Memory consumption of s table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(s.values.nbytes/1024/1024, s.values.dtype), flush=True)
        elapsed_time = int(time.time() - start)
        print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

    if g['omega_method']!='rec':
        g['state_cdn'] = None
        g['state_pep'] = None

    if (g['b']) | (g['cb']):
        start = time.time()
        print("Making branch table.", flush=True)
        bS = get_b(g, S_tensor, attr='S')
        bN = get_b(g, N_tensor, attr='N')
        b = merge_tables(bS, bN)
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
        print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

    if (g['cs']):
        start = time.time()
        print("Making combinat-site table.", flush=True)
        if id_combinations is None:
            g,id_combinations = get_node_combinations(g=g, arity=g['current_arity'], check_attr="name")
        csS = get_cs(id_combinations, S_tensor, attr='S')
        csN = get_cs(id_combinations, N_tensor, attr='N')
        cs = merge_tables(csS, csN)
        del csS, csN
        cs.to_csv("csubst_cs.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        txt = 'Memory consumption of cb table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(cb.values.nbytes/1024/1024, cb.values.dtype), flush=True)
        del cs
        elapsed_time = int(time.time() - start)
        print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

    if (g['cbs']):
        start = time.time()
        print("Making combinat-branch-site table.", flush=True)
        if id_combinations is None:
            g,id_combinations = get_node_combinations(g=g, arity=g['current_arity'], check_attr="name")
        cbsS = get_cbs(id_combinations, S_tensor, attr='S', g=g)
        cbsN = get_cbs(id_combinations, N_tensor, attr='N', g=g)
        cbs = merge_tables(cbsS, cbsN)
        del cbsS, cbsN
        cbs.to_csv("csubst_cbs.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
        txt = 'Memory consumption of cbs table: {:,.1f} Mbytes (dtype={})'
        print(txt.format(cbs.values.nbytes/1024/1024, cbs.values.dtype), flush=True)
        del cbs
        elapsed_time = int(time.time() - start)
        print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]\n", flush=True)

    if (g['cb']):
        g['df_cb_stats_main'] = pandas.DataFrame()
        g = cb_search(g, b, S_tensor, N_tensor, id_combinations, mode='foreground', write_cb=True)
        if (g['foreground'] is not None)&(g['fg_random']>0):
            for i in numpy.arange(0, g['fg_random']):
                print('starting foreground randomization round {:,}'.format(i+1), flush=True)
                g = foreground.get_foreground_branch(g)
                g = foreground.randomize_foreground_branch(g)
                g = foreground.get_marginal_branch(g)
                g = cb_search(g, b, S_tensor, N_tensor, mode='randomization_'+str(i+1), write_cb=False)
                print('ending foreground randomization round {:,}\n'.format(i+1), flush=True)

        g['df_cb_stats_main'].to_csv('csubst_cb_stats.tsv', sep="\t", index=False, float_format='%.4f', chunksize=10000)

    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]


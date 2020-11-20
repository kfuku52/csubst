import numpy
import pandas


def get_global_parameters(args):
    g = dict()
    for attr in [a for a in dir(args) if not a.startswith('_')]:
        g[attr] = getattr(args, attr)
    if (g['iqtree_treefile']=='infer'):
        g['iqtree_treefile'] = g['aln_file']+'.treefile'
    if (g['iqtree_state']=='infer'):
        g['iqtree_state'] = g['aln_file']+'.state'
    if (g['iqtree_rate']=='infer'):
        g['iqtree_rate'] = g['aln_file']+'.rate'
    if (g['iqtree_iqtree']=='infer'):
        g['iqtree_iqtree'] = g['aln_file']+'.iqtree'
    if (g['float_type']==16):
        g['float_type'] = numpy.float16
        g['float_tol'] = 10**-2
    elif (g['float_type']==32):
        g['float_type'] = numpy.float32
        g['float_tol'] = 10**-4
    elif (g['float_type']==64):
        g['float_type'] = numpy.float64
        g['float_tol'] = 10**-9
    return g

def get_dep_ids(g):
    dep_ids = list()
    for leaf in g['tree'].iter_leaves():
        ancestor_nn = [ node.numerical_label for node in leaf.iter_ancestors() if not node.is_root() ]
        dep_id = [leaf.numerical_label,] + ancestor_nn
        dep_id = numpy.sort(numpy.array(dep_id))
        dep_ids.append(dep_id)
    if g['exclude_sisters']:
        for node in g['tree'].traverse():
            children = node.get_children()
            if len(children)>1:
                dep_id = numpy.sort(numpy.array([ node.numerical_label for node in children ]))
                dep_ids.append(dep_id)
    g['dep_ids'] = dep_ids
    if (g['foreground'] is not None)&(g['fg_exclude_wg']):
        fg_dep_ids = list()
        for i in numpy.arange(len(g['fg_leaf_name'])):
            tmp_fg_dep_ids = list()
            for node in g['tree'].traverse():
                is_all_leaf_lineage_fg = all([ ln in g['fg_leaf_name'][i] for ln in node.get_leaf_names() ])
                if is_all_leaf_lineage_fg:
                    is_up_all_leaf_lineage_fg = all([ ln in g['fg_leaf_name'][i] for ln in node.up.get_leaf_names() ])
                    if not is_up_all_leaf_lineage_fg:
                        if node.is_leaf():
                            tmp_fg_dep_ids.append(node.numerical_label)
                        else:
                            descendant_nn = [ n.numerical_label for n in node.get_descendants() ]
                            tmp_fg_dep_ids += [node.numerical_label,] + descendant_nn
            if len(tmp_fg_dep_ids)>1:
                fg_dep_ids.append(numpy.sort(numpy.array(tmp_fg_dep_ids)))
        if (g['fg_sister'])|(g['fg_parent']):
            fg_dep_ids.append(numpy.sort(numpy.array(g['marginal_id'])))
        g['fg_dep_ids'] = fg_dep_ids
    else:
        g['fg_dep_ids'] = numpy.array([])
    return g

def initialize_df_cb_stats(g):
    if g['cb_stats'] is None:
        ind = numpy.arange(0, g['max_arity'])
        cols = ['arity','elapsed_sec','fg_median_dist_bl','fg_median_dist_node_num',]
        cols = cols + ['num_examined','num_fg','num_qualified','num_fg_qualified','fg_conc_factor',]
        g['df_cb_stats'] = pandas.DataFrame(index=ind, columns=cols)
        g['df_cb_stats'].loc[:,'arity'] = ind + 1
        g['df_cb_stats'].loc[:,'target_stat'] = g['target_stat']
        g['df_cb_stats'].loc[:,'min_stat'] = g['min_stat']
    else:
        g['df_cb_stats'] = pandas.read_csv(g['cb_stats'], sep='\t', header=0)
    return(g)

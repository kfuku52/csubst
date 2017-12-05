import itertools
import numpy

def prepare_node_combinations(g, target_nodes=None, arity=2, check_attr=None, verbose=True):
    tree = g['tree']
    all_nodes = [ node for node in tree.traverse() if not node.is_root() ]
    if verbose:
        print("arity:", arity, flush=True)
        print("all nodes:", len(all_nodes), flush=True)
    if (target_nodes is None):
        target_nodes = list()
        for node in all_nodes:
            if (check_attr is None)|(check_attr in dir(node)):
                target_nodes.append(node.numerical_label)
        node_combinations = list(itertools.combinations(target_nodes, arity))
        node_combinations = [set(nc) for nc in node_combinations]
    else:
        target_nodes = [ set(tn) for tn in target_nodes ]
        node_combinations = list()
        for tn1 in target_nodes:
            for tn2 in target_nodes[1:]:
                node_union = tn1.union(tn2)
                if (len(node_union)==arity):
                    if node_union not in node_combinations:
                        node_combinations.append(node_union)
    if verbose:
        print("target nodes:", len(target_nodes), flush=True)
        print("all node combinations: ", len(node_combinations), flush=True)
    node_combinations = numpy.array([list(nc) for nc in node_combinations])
    nc_matrix = numpy.zeros(shape=(len(all_nodes), node_combinations.shape[0]), dtype=numpy.bool_, order='C')
    for i in numpy.arange(node_combinations.shape[0]):
        nc_matrix[node_combinations[i,:],i] = 1
    dep_ids = list()
    for leaf in tree.iter_leaves():
        dep_id = [leaf.numerical_label,] + [ node.numerical_label for node in leaf.iter_ancestors() if not node.is_root() ]
        dep_id = numpy.sort(numpy.array(dep_id))
        dep_ids.append(dep_id)
    if g['exclude_sisters']:
        for node in tree.traverse():
            #if 'get_children' in dir(node):
            children = node.get_children()
            if len(children)>1:
                dep_id = numpy.sort(numpy.array([ node.numerical_label for node in children ]))
                dep_ids.append(dep_id)
    is_dependent_col = False
    for dep_id in dep_ids:
        is_dependent_col = (is_dependent_col)|(nc_matrix[dep_id,:].sum(axis=0)>1)
    nc_matrix = nc_matrix[:,~is_dependent_col]
    rows,cols = numpy.where(nc_matrix==1)
    unique_cols = numpy.unique(cols)
    id_combinations = numpy.zeros(shape=(unique_cols.shape[0], arity), dtype=numpy.int)
    for i in unique_cols:
        id_combinations[i,:] = rows[cols==i]
    id_combinations.sort(axis=1)
    if verbose:
        print("independent node combinations: ", id_combinations.shape[0], flush=True)
    return(id_combinations)

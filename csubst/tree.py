import numpy
import itertools
import re

def add_numerical_node_labels(tree):
    all_leaf_names = tree.get_leaf_names()
    all_leaf_names.sort()
    leaf_numerical_labels = dict()
    for i in range(0, len(all_leaf_names)):
        leaf_numerical_labels[all_leaf_names[i]] = 2**i
    numerical_labels = list()
    for node in tree.traverse():
        leaf_names = node.get_leaf_names()
        numerical_labels.append(sum([leaf_numerical_labels[leaf_name] for leaf_name in leaf_names]))
    argsort_labels = numpy.argsort(numerical_labels)
    short_labels = numpy.arange(len(argsort_labels))
    i=0
    for node in tree.traverse():
        node.numerical_label = short_labels[argsort_labels==i][0]
        i+=1
    return(tree)

def transfer_root(tree_to, tree_from, verbose=False):
    same_leaf_set = len(set(tree_to.get_leaf_names()) - set(tree_from.get_leaf_names())) == 0
    assert same_leaf_set, 'Input tree and iqtree\'s treefile did not have identical leaves.'
    subroot_leaves = [ n.get_leaf_names() for n in tree_from.get_children() ]
    is_n0_bigger_than_n1 = (len(subroot_leaves[0]) > len(subroot_leaves[1]))
    ingroups = subroot_leaves[0] if is_n0_bigger_than_n1 else subroot_leaves[1]
    outgroups = subroot_leaves[0] if not is_n0_bigger_than_n1 else subroot_leaves[1]
    if verbose:
        print('outgroups:', outgroups)
    tree_to.set_outgroup(ingroups[0])
    if (len(outgroups) == 1):
        outgroup_ancestor = [n for n in tree_to.iter_leaves() if n.name == outgroups[0]][0]
    else:
        outgroup_ancestor = tree_to.get_common_ancestor(outgroups)
    tree_to.set_outgroup(outgroup_ancestor)
    subroot_to = tree_to.get_children()
    subroot_from = tree_from.get_children()
    total_subroot_length_to = sum([n.dist for n in subroot_to])
    total_subroot_length_from = sum([n.dist for n in subroot_from])
    for n_to in subroot_to:
        for n_from in subroot_from:
            if (set(n_to.get_leaf_names()) == set(n_from.get_leaf_names())):
                n_to.dist = (n_from.dist / total_subroot_length_from) * total_subroot_length_to
    for n_to in tree_to.traverse():
        if n_to.name == '':
            n_to.name = tree_to.name
            tree_to.name = 'Root'
            break
    return tree_to

def rerooting_by_topology_matching(tree_from, tree_to):
    tree_from.ladderize()
    tree_to.ladderize()
    print('Before rerooting, Robinson-Foulds distance =', tree_from.robinson_foulds(tree_to, unrooted_trees=True)[0])
    outgroup_labels = tree_from.get_descendants()[0].get_leaf_names()
    for node in tree_to.traverse():
        if (node.is_leaf())&(not node.name in outgroup_labels):
            non_outgroup_leaf = node
            break
    tree_to.set_outgroup(non_outgroup_leaf)
    outgroup_ancestor = tree_to.get_common_ancestor(outgroup_labels)
    tree_to.set_outgroup(outgroup_ancestor)
    tree_to = add_numerical_node_labels(tree_to)
    tree_to.ladderize()
    print('After rerooting, Robinson-Foulds distance =', tree_from.robinson_foulds(tree_to, unrooted_trees=False)[0])
    return tree_to

def transfer_internal_node_names(tree_to, tree_from):
    rf_dist = tree_to.robinson_foulds(tree_from, expand_polytomies=True)[0]
    assert rf_dist==0, 'tree topologies are different. RF distance = {}'.format(rf_dist)
    for to in tree_to.traverse():
        if not to.is_leaf():
            for fr in tree_from.traverse():
                if not fr.is_leaf():
                    if set(to.get_leaf_names())==set(fr.get_leaf_names()):
                        to.name = fr.name
    return tree_to

def get_node_distance(tree, cb):
    if not 'numerical_label' in dir(tree):
        tree = add_numerical_node_labels(tree)
    tree_dict = dict()
    for node in tree.traverse():
        tree_dict[node.numerical_label] = node
    cn1 = cb.columns[cb.columns.str.startswith('branch_id_')]
    cn2 = ["dist_node_num", "dist_bl"]
    for cn2_item in cn2:
        cb.loc[:,cn2_item] = numpy.nan
    for i in cb.index:
        nodes = [ tree_dict[n] for n in cb.loc[i,cn1].tolist() ]
        node_dists = list()
        node_nums = list()
        for nds in list(itertools.combinations(nodes, 2)):
            node_dist = nds[0].get_distance(target=nds[1], topology_only=False)
            node_dists.append(node_dist - nds[1].dist)
            node_nums.append(nds[0].get_distance(target=nds[1], topology_only=True))
        node_dist = max(node_dists) # Maximum number among pairwise distances
        node_num = max(node_nums) # Maximum number among pairwise distances
        cb.loc[i,"dist_node_num"] = node_num
        cb.loc[i,"dist_bl"] = node_dist
    return(cb)

def standardize_node_names(tree):
    for node in tree.traverse():
        node.name = re.sub('\[.*', '', node.name)
        node.name = re.sub('/.*', '', node.name)
        node.name = re.sub('^\'', '', node.name)
        node.name = re.sub('\'$', '', node.name)
    return tree

def is_internal_node_labeled(tree):
    is_labeled = True
    for node in tree.traverse():
        if not node.is_root():
            if node.name=='':
                is_labeled = False
    return is_labeled


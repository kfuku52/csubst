import ete3
import numpy

import copy
import itertools
import os
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
                n_to.dist = total_subroot_length_to * (n_from.dist / total_subroot_length_from)
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
    tree_to = tree.add_numerical_node_labels(tree_to)
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

def get_node_distance(tree, cb): # TODO parallel
    if not 'numerical_label' in dir(tree):
        tree = tree.add_numerical_node_labels(tree)
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
        node_dist = max(node_dists) # Maximum value among pairwise distances
        node_num = max(node_nums) # Maximum value among pairwise distances
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

def write_tree(tree, outfile='csubst_tree.nwk', add_numerical_label=True):
    tree2 = copy.deepcopy(tree)
    if add_numerical_label:
        for node in tree2.traverse():
            node.name = node.name + '|' + str(node.numerical_label)
    tree2.write(format=1, outfile=outfile)

def branch_category_layout(node):
    nstyle = ete3.NodeStyle()
    nstyle['size'] = 0
    nstyle["hz_line_width"] = nstyle["vt_line_width"] = 1
    nstyle["hz_line_color"] = node.color
    nstyle["vt_line_color"] = node.color
    nlabel = node.name+'|'+str(node.numerical_label)
    nlabelFace = ete3.TextFace(nlabel, fsize=6, fgcolor=node.color)
    ete3.add_face_to_node(face=nlabelFace, node=node, column=1, aligned=False, position="branch-right")
    node.set_style(nstyle)

def is_ete_plottable():
    try:
        from ete3 import TreeStyle
        from ete3 import NodeStyle
    except ImportError:
        print('TreeStyle and/or NodeStyle are not available in installed ete3. Plotting is skipped.', flush=True)
        return False
    if ('DISPLAY' not in os.environ.keys()):
        print('DISPLAY is not available. Plotting is skipped.', flush=True)
        return False
    return True

def plot_branch_category(tree, file_name):
    if not is_ete_plottable():
        return None
    ts = ete3.TreeStyle()
    ts.mode = 'r'
    ts.show_leaf_name = False
    ts.layout_fn = branch_category_layout
    tree.render(file_name=file_name, tree_style=ts, units='px', dpi=300)

def branch_state_layout(node):
    nstyle = ete3.NodeStyle()
    nstyle['size'] = 0
    nstyle["hz_line_width"] = nstyle["vt_line_width"] = 1
    nstyle["hz_line_color"] = node.color
    nstyle["vt_line_color"] = node.color
    if node.is_leaf():
        nlabel = str(node.state)+'|'+node.name
    else:
        nlabel = str(node.state)
    nlabelFace = ete3.TextFace(nlabel, fsize=6, fgcolor=node.color)
    ete3.add_face_to_node(face=nlabelFace, node=node, column=1, aligned=False, position="branch-right")
    node.set_style(nstyle)

def plot_state_tree(state, orders, mode, g):
    print('Writing ancestral state trees: mode = {}, number of pdf files = {}'.format(mode, state.shape[1]), flush=True)
    if not is_ete_plottable():
        return None
    if mode=='codon':
        missing_state = '---'
    else:
        missing_state = '-'
    ts = ete3.TreeStyle()
    ts.mode = 'r'
    ts.show_leaf_name = False
    ts.layout_fn = branch_state_layout
    ndigit = int(numpy.log10(state.shape[1]))+1
    for i in numpy.arange(state.shape[1]):
        for node in g['tree'].traverse():
            if node.is_root():
                node.state = missing_state
                continue
            nlabel = node.numerical_label
            index = numpy.where(state[nlabel,i,:]==max(state[nlabel,i,:]))[0]
            if len(index)==1:
                node.state = orders[index[0]]
            elif len(index)==0:
                node.state = missing_state
        file_name = 'csubst_state_'+mode+'_'+str(i+1).zfill(ndigit)+'.pdf'
        g['tree'].render(file_name=file_name, tree_style=ts, units='px', dpi=300)

def get_num_adjusted_sites(g, node):
    nl = node.numerical_label
    parent = node.up.numerical_label
    child_states = g['state_cdn'][nl,:,:]
    parent_states = g['state_cdn'][parent,:,:]
    is_child_present = numpy.expand_dims(child_states.sum(axis=1)!=0, axis=1)
    parent_states *= is_child_present
    codon_counts = parent_states.sum(axis=0)
    scaled_Q = numpy.copy(g['instantaneous_codon_rate_matrix'])
    numpy.fill_diagonal(scaled_Q, 0)
    scaled_Q = scaled_Q / numpy.expand_dims(scaled_Q.sum(axis=1), axis=1)
    adjusted_site_S = 0
    adjusted_site_N = 0
    for i in numpy.arange(codon_counts.shape[0]):
        codon = g['codon_orders'][i]
        amino_acid = [ val[0] for val in g['codon_table'] if val[1]==codon ][0]
        synonymous_codons = [ val[1] for val in g['codon_table'] if val[0]==amino_acid ]
        synonymous_codon_index = [ j for j,cdn in enumerate(g['codon_orders']) if cdn in synonymous_codons ]
        prop_S = scaled_Q[i,synonymous_codon_index].sum()
        prop_N = 1 - prop_S
        adjusted_site_S += prop_S * codon_counts[i]
        adjusted_site_N += prop_N * codon_counts[i]
    return adjusted_site_S,adjusted_site_N

def rescale_branch_length(g, S_tensor, N_tensor, denominator='L'):
    print('Branch lengths of the IQ-TREE output are rescaled to match observed-codon-substitutions/codon-site, '
          'rather than nucleotide-substitutions/codon-site.')
    print('Total branch length before rescaling: {:,.3f} nucleotide substitutions / codon site'.format(sum([ n.dist for n in g['tree'].traverse() ])))
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
        # is_S_zero = (num_S_sub==0)
        # is_N_zero = (num_N_sub==0)
        if (denominator=='L'):
            node.Sdist = num_S_sub / num_nonmissing_codon
            node.Ndist = num_N_sub / num_nonmissing_codon
            node.SNdist = node.Sdist + node.Ndist
        elif (denominator=='adjusted_site'): # This option overestimated EN and ES compared with "L"
            adjusted_site_S,adjusted_site_N = get_num_adjusted_sites(g, node)
            #prop_S = adjusted_site_S / (adjusted_site_S + adjusted_site_N)
            #prop_N = adjusted_site_N / (adjusted_site_S + adjusted_site_N)
            #prop_S = num_S_sub / (num_S_sub + num_N_sub)
            #prop_N = num_N_sub / (num_S_sub + num_N_sub)
            adjusted_num_S_sub = num_S_sub / adjusted_site_S
            adjusted_num_N_sub = num_N_sub / adjusted_site_N
            prop_S = adjusted_num_S_sub / (adjusted_num_S_sub + adjusted_num_N_sub)
            prop_N = adjusted_num_N_sub / (adjusted_num_S_sub + adjusted_num_N_sub)
            if num_S_sub==0:
                node.Sdist = 0
            else:
                node.Sdist = node.dist * prop_S
                #node.Sdist = adjusted_site_S / prop_S
            if num_S_sub==0:
                node.Ndist = 0
            else:
                node.Ndist = node.dist * prop_N
                #node.Ndist = adjusted_site_N / prop_N
            node.SNdist = node.Sdist + node.Ndist

    print('Total S+N branch length after rescaling: {:,.3f} codon substitutions / codon site'.format(sum([ n.SNdist for n in g['tree'].traverse() ])))
    print('Total S branch length after rescaling: {:,.3f} codon substitutions / codon site'.format(sum([ n.Sdist for n in g['tree'].traverse() ])))
    print('Total N branch length after rescaling: {:,.3f} codon substitutions / codon site'.format(sum([ n.Ndist for n in g['tree'].traverse() ])))
    return g


import numpy

import copy
import itertools
import re
import sys
import time

from csubst import sequence
from csubst import parallel
from csubst import substitution
from csubst import ete

TREE_FIG_MIN_HEIGHT = 1.8
TREE_FIG_HEIGHT_PER_LEAF = 0.10
TREE_FIG_MAX_HEIGHT = 28.0
TREE_LINE_CAPSTYLE = 'round'

def add_numerical_node_labels(tree):
    all_leaf_names = ete.get_leaf_names(tree)
    all_leaf_names.sort()
    leaf_numerical_labels = dict()
    for i in range(0, len(all_leaf_names)):
        leaf_numerical_labels[all_leaf_names[i]] = 2**i
    numerical_labels = list()
    for node in tree.traverse():
        leaf_names = ete.get_leaf_names(node)
        numerical_labels.append(sum([leaf_numerical_labels[leaf_name] for leaf_name in leaf_names]))
    argsort_labels = numpy.argsort(numerical_labels)
    short_labels = numpy.arange(len(argsort_labels))
    i=0
    for node in tree.traverse():
        ete.set_prop(node, "numerical_label", short_labels[argsort_labels == i][0])
        i+=1
    return tree

def is_consistent_tree(tree1, tree2):
    is_consistent_tree = set(ete.get_leaf_names(tree1)) == set(ete.get_leaf_names(tree2))
    return is_consistent_tree

def transfer_root(tree_to, tree_from, verbose=False):
    for node in tree_to.traverse():
        if node.dist is None:
            node.dist = 0.0
    subroot_leaves = [ete.get_leaf_names(n) for n in ete.get_children(tree_from)]
    is_n0_bigger_than_n1 = (len(subroot_leaves[0]) > len(subroot_leaves[1]))
    ingroups = subroot_leaves[0] if is_n0_bigger_than_n1 else subroot_leaves[1]
    outgroups = subroot_leaves[0] if not is_n0_bigger_than_n1 else subroot_leaves[1]
    if verbose:
        print('outgroups:', outgroups)
    original_root_name = tree_to.name
    tree_to.set_outgroup(ingroups[0])
    if (len(outgroups) == 1):
        outgroup_ancestor = [n for n in ete.iter_leaves(tree_to) if n.name == outgroups[0]][0]
    else:
        outgroup_ancestor = ete.get_common_ancestor(tree_to, outgroups)
    if not set(outgroups) == set(ete.get_leaf_names(outgroup_ancestor)):
        sys.stderr.write('No root bipartition found in --infile. Exiting.\n')
        sys.exit(1)
    tree_to.set_outgroup(outgroup_ancestor)
    subroot_to = ete.get_children(tree_to)
    subroot_from = ete.get_children(tree_from)
    total_subroot_length_to = sum([(n.dist or 0) for n in subroot_to])
    total_subroot_length_from = sum([(n.dist or 0) for n in subroot_from])
    if total_subroot_length_from == 0:
        total_subroot_length_from = 1
    for n_to in subroot_to:
        for n_from in subroot_from:
            if (set(ete.get_leaf_names(n_to)) == set(ete.get_leaf_names(n_from))):
                n_to.dist = total_subroot_length_to * ((n_from.dist or 0) / total_subroot_length_from)
    if original_root_name:
        tree_to.name = original_root_name
    elif not tree_to.name:
        tree_to.name = 'Root'
    return tree_to

def transfer_internal_node_names(tree_to, tree_from):
    rf_dist = tree_to.robinson_foulds(tree_from, expand_polytomies=True)[0]
    assert rf_dist==0, 'tree topologies are different. RF distance = {}'.format(rf_dist)
    for to in tree_to.traverse():
        if not ete.is_leaf(to):
            for fr in tree_from.traverse():
                if not ete.is_leaf(fr):
                    if set(ete.get_leaf_names(to))==set(ete.get_leaf_names(fr)):
                        to.name = fr.name
    return tree_to

def calc_node_dist_chunk(chunk, start, tree_dict, float_type):
    start_time = time.time()
    nrow = chunk.shape[0]
    arr_dist = numpy.zeros(shape=(nrow, 3), dtype=float_type)
    arr_dist[:,0] = numpy.arange(nrow) + start
    for i in numpy.arange(nrow):
        nodes = [ tree_dict[n] for n in chunk[i,:] ]
        node_dists = list()
        node_nums = list()
        for nds in list(itertools.combinations(nodes, 2)):
            node_dist = ete.get_distance(nds[0], nds[1], topology_only=False)
            node_dists.append(node_dist - nds[1].dist)
            node_nums.append(ete.get_distance(nds[0], nds[1], topology_only=True))
        node_dist = max(node_dists) # Maximum value among pairwise distances
        node_num = max(node_nums) # Maximum value among pairwise distances
        arr_dist[i,1] = node_num
        arr_dist[i,2] = node_dist
        if i % 10000 == 0:
            end = start + nrow
            txt = 'Inter-branch distance: {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(i, start, end, int(time.time() - start_time)), flush=True)
    return arr_dist

def get_node_distance(tree, cb, ncpu, float_type):
    txt = 'Starting branch distance calculation. If it takes too long, disable this step by --branch_dist no'
    print(txt, flush=True)
    start_time = time.time()
    if ete.get_prop(tree, "numerical_label") is None:
        tree = add_numerical_node_labels(tree)
    tree_dict = dict()
    for node in tree.traverse():
        tree_dict[ete.get_prop(node, "numerical_label")] = node
    cn1 = cb.columns[cb.columns.str.startswith('branch_id_')]
    id_combinations = cb.loc[:,cn1].values
    n_jobs = parallel.resolve_n_jobs(num_items=id_combinations.shape[0], threads=ncpu)
    if n_jobs == 1:
        out_list = [calc_node_dist_chunk(id_combinations, 0, tree_dict, float_type)]
    else:
        chunks, starts = parallel.get_chunks(id_combinations, n_jobs)
        tasks = [(chunk, start, tree_dict, float_type) for chunk, start in zip(chunks, starts)]
        out_list = parallel.run_starmap(
            func=calc_node_dist_chunk,
            args_iterable=tasks,
            n_jobs=n_jobs,
            backend='threading',
        )
    cb.loc[:, 'dist_node_num'] = -1
    cb.loc[:, 'dist_bl'] = numpy.nan
    for arr_dist in out_list:
        ind = arr_dist[:,0].astype(int)
        cb.loc[ind,'dist_node_num'] = arr_dist[:,1] # This line causes the warning when arr_dist.shape[0] == 1: FutureWarning: In a future version, `df.iloc[:, i] = newvals` will attempt to set the values inplace instead of always setting a new array. To retain the old behavior, use either `df[df.columns[i]] = newvals` or, if columns are non-unique, `df.isetitem(i, newvals)`
        cb.loc[ind,'dist_node_num'] = arr_dist[:,1]
        cb.loc[ind,'dist_bl'] = arr_dist[:,2]
    print('Time elapsed for calculating inter-branch distances: {:,} sec'.format(int(time.time() - start_time)))
    return(cb)

def standardize_node_names(tree):
    for node in tree.traverse():
        node.name = '' if node.name is None else str(node.name)
        node.name = re.sub(r'\[.*', '', node.name)
        node.name = re.sub(r'/.*', '', node.name)
        node.name = re.sub(r'^\'', '', node.name)
        node.name = re.sub(r'\'$', '', node.name)
    leaf_names = ete.get_leaf_names(tree)
    if len(leaf_names)!=len(set(leaf_names)):
        raise ValueError('Leaf names are not unique')
    node_names = [node.name for node in tree.traverse() if (not ete.is_leaf(node)) and (node.name not in [None, ''])]
    if len(node_names)!=len(set(node_names)):
        raise ValueError('Internal node labels are not unique. '
                         'Please provide unique internal node labels or delete them from --rooted_tree_file. '
                         'For the label deletion, nwkit drop may be useful: https://github.com/kfuku52/nwkit/wiki/nwkit-drop')
    return tree

def is_internal_node_labeled(tree):
    is_labeled = True
    for node in tree.traverse():
        if not ete.is_root(node):
            if not node.name:
                is_labeled = False
    return is_labeled

def write_tree(tree, outfile='csubst_tree.nwk', add_numerical_label=True):
    tree2 = copy.deepcopy(tree)
    if add_numerical_label:
        for node in tree2.traverse():
            node.name = (node.name or '') + '|' + str(ete.get_prop(node, "numerical_label"))
    ete.write_tree(tree2, format=1, outfile=outfile)

def is_ete_plottable():
    try:
        _ = _get_pyplot()
    except Exception as exc:
        print('Matplotlib is not available ({}). Plotting is skipped.'.format(exc), flush=True)
        return False
    return True


def _get_pyplot():
    import matplotlib
    try:
        matplotlib.use('Agg')
    except Exception:
        pass
    import matplotlib.pyplot
    return matplotlib.pyplot


def _get_tree_xy(tree):
    root = ete.get_tree_root(tree)
    xcoord = dict()
    ycoord = dict()

    def assign_x(node, x_value):
        xcoord[id(node)] = float(x_value)
        for child in ete.get_children(node):
            branch_len = float(getattr(child, 'dist', 0.0) or 0.0)
            assign_x(child, x_value + max(branch_len, 0.0))

    assign_x(root, 0.0)
    leaves = list(ete.iter_leaves(tree))
    if len(leaves)==0:
        ycoord[id(root)] = 0.0
        return xcoord,ycoord,[root]
    for i,leaf in enumerate(leaves):
        ycoord[id(leaf)] = float(len(leaves)-1-i)

    def assign_y(node):
        node_id = id(node)
        if node_id in ycoord:
            return ycoord[node_id]
        children = ete.get_children(node)
        if len(children)==0:
            ycoord[node_id] = 0.0
            return 0.0
        child_ys = [assign_y(child) for child in children]
        ycoord[node_id] = float(sum(child_ys) / len(child_ys))
        return ycoord[node_id]

    assign_y(root)
    return xcoord,ycoord,leaves


def _is_foreground_stem_branch(node, trait_name):
    if ete.is_root(node):
        return False
    is_fg = bool(ete.get_prop(node, "is_fg_" + trait_name, False))
    is_parent_fg = bool(ete.get_prop(node.up, "is_fg_" + trait_name, False))
    return is_fg and (not is_parent_fg)


def _get_branch_segment_colors(node, trait_name):
    branch_color = ete.get_prop(node, "color_" + trait_name, "black")
    if _is_foreground_stem_branch(node=node, trait_name=trait_name):
        vertical_color = "black"
    else:
        vertical_color = branch_color
    horizontal_color = branch_color
    return vertical_color,horizontal_color


def _render_tree_matplotlib(tree, trait_name, file_name, label='all', state_by_node=None):
    plt = _get_pyplot()
    xcoord,ycoord,leaves = _get_tree_xy(tree)
    labels = []
    for node in tree.traverse():
        if state_by_node is None:
            if label=='no':
                continue
            if (label=='leaf') and (not ete.is_leaf(node)):
                continue
            txt = (node.name or '') + '|' + str(ete.get_prop(node, "numerical_label"))
            font_size = 4
        else:
            nl = ete.get_prop(node, "numerical_label")
            state_txt = str(state_by_node.get(nl, '-'))
            txt = state_txt + '|' + (node.name or '') if ete.is_leaf(node) else state_txt
            font_size = 6
        labels.append((node, txt, font_size))
    max_label_len = max([len(lbl[1]) for lbl in labels], default=0)
    num_leaves = max(len(leaves), 1)
    fig_height = min(max(TREE_FIG_MIN_HEIGHT, num_leaves * TREE_FIG_HEIGHT_PER_LEAF), TREE_FIG_MAX_HEIGHT)
    fig_width = 7.0 + min(7.0, max_label_len * 0.05)
    fig,ax = plt.subplots(figsize=(fig_width, fig_height))
    for node in tree.traverse():
        if ete.is_root(node):
            continue
        parent = node.up
        x_parent = xcoord[id(parent)]
        y_parent = ycoord[id(parent)]
        x_node = xcoord[id(node)]
        y_node = ycoord[id(node)]
        v_color,h_color = _get_branch_segment_colors(node=node, trait_name=trait_name)
        ax.plot([x_parent, x_parent], [y_parent, y_node], color=v_color, linewidth=0.8, solid_capstyle=TREE_LINE_CAPSTYLE)
        ax.plot([x_parent, x_node], [y_node, y_node], color=h_color, linewidth=0.8, solid_capstyle=TREE_LINE_CAPSTYLE)
    xmax = max(xcoord.values()) if len(xcoord)>0 else 0.0
    xspan = max(xmax, 1.0)
    text_offset = xspan * 0.015
    for node,txt,font_size in labels:
        text_color = ete.get_prop(node, "labelcolor_" + trait_name, "black")
        ax.text(
            xcoord[id(node)] + text_offset,
            ycoord[id(node)],
            txt,
            fontsize=font_size,
            color=text_color,
            va='center',
            ha='left',
            clip_on=False,
        )
    if max_label_len>0:
        text_space = max_label_len * xspan * 0.03
    else:
        text_space = xspan * 0.1
    ax.set_xlim(-xspan * 0.02, xmax + text_space + text_offset)
    ax.set_ylim(-0.5, num_leaves - 0.5)
    ax.axis('off')
    fig.savefig(file_name, format='pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)
    return None


def plot_branch_category(g, file_base, label='all'):
    if not is_ete_plottable():
        return None
    trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)]
    for trait_name in trait_names:
        file_name = file_base+'_'+trait_name+'.pdf'
        file_name = file_name.replace('_PLACEHOLDER', '')
        _render_tree_matplotlib(
            tree=g['tree'],
            trait_name=trait_name,
            file_name=file_name,
            label=label,
            state_by_node=None,
        )

def plot_state_tree(state, orders, mode, g):
    print('Writing ancestral state trees: mode = {}, number of pdf files = {}'.format(mode, state.shape[1]), flush=True)
    if not is_ete_plottable():
        return None
    if state.shape[1] == 0:
        print('No sites available for ancestral state tree plotting. Skipping.', flush=True)
        return None
    if mode=='codon':
        missing_state = '---'
    else:
        missing_state = '-'
    trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)]
    for trait_name in trait_names:
        ndigit = int(numpy.log10(state.shape[1]))+1
        for i in numpy.arange(state.shape[1]):
            state_by_node = dict()
            for node in g['tree'].traverse():
                nlabel = ete.get_prop(node, "numerical_label")
                if ete.is_root(node):
                    state_by_node[nlabel] = missing_state
                    continue
                max_prob = max(state[nlabel,i,:])
                index = numpy.where(state[nlabel,i,:]==max_prob)[0]
                if len(index)==1:
                    state_by_node[nlabel] = orders[index[0]]
                elif (len(index)==0)|(max_prob==0):
                    state_by_node[nlabel] = missing_state
            file_name = 'csubst_state_'+trait_name+'_'+mode+'_'+str(i+1).zfill(ndigit)+'.pdf'
            file_name = file_name.replace('_PLACEHOLDER', '')
            _render_tree_matplotlib(
                tree=g['tree'],
                trait_name=trait_name,
                file_name=file_name,
                label='all',
                state_by_node=state_by_node,
            )

def get_num_adjusted_sites(g, node):
    nl = ete.get_prop(node, "numerical_label")
    parent = ete.get_prop(node.up, "numerical_label")
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

def rescale_branch_length(g, OS_tensor, ON_tensor, denominator='L'):
    print('Branch lengths of the IQ-TREE output are rescaled to match observed-codon-substitutions/codon-site, '
          'rather than nucleotide-substitutions/codon-site.')
    print('Total branch length before rescaling: {:,.3f} nucleotide substitutions / codon site'.format(
        sum([(n.dist or 0.0) for n in g['tree'].traverse()])
    ))
    OS_branch_sub = substitution.get_branch_sub_counts(OS_tensor)
    ON_branch_sub = substitution.get_branch_sub_counts(ON_tensor)
    for node in g['tree'].traverse():
        if ete.is_root(node):
            ete.set_prop(node, "Sdist", 0)
            ete.set_prop(node, "Ndist", 0)
            ete.set_prop(node, "SNdist", 0)
            continue
        nl = ete.get_prop(node, "numerical_label")
        parent = ete.get_prop(node.up, "numerical_label")
        num_nonmissing_codon = (g['state_cdn'][(nl,parent),:,:].sum(axis=2).sum(axis=0)!=0).sum()
        if num_nonmissing_codon==0:
            ete.set_prop(node, "Sdist", 0)
            ete.set_prop(node, "Ndist", 0)
            ete.set_prop(node, "SNdist", 0)
            continue
        num_S_sub = OS_branch_sub[nl]
        num_N_sub = ON_branch_sub[nl]
        # is_S_zero = (num_S_sub==0)
        # is_N_zero = (num_N_sub==0)
        if (denominator=='L'):
            sdist = num_S_sub / num_nonmissing_codon
            ndist = num_N_sub / num_nonmissing_codon
            ete.set_prop(node, "Sdist", sdist)
            ete.set_prop(node, "Ndist", ndist)
            ete.set_prop(node, "SNdist", sdist + ndist)
        elif (denominator=='adjusted_site'): # This option overestimated EN and ES compared with "L"
            adjusted_site_S,adjusted_site_N = get_num_adjusted_sites(g, node)
            #prop_S = adjusted_site_S / (adjusted_site_S + adjusted_site_N)
            #prop_N = adjusted_site_N / (adjusted_site_S + adjusted_site_N)
            #prop_S = num_S_sub / (num_S_sub + num_N_sub)
            #prop_N = num_N_sub / (num_S_sub + num_N_sub)
            if adjusted_site_S <= g['float_tol']:
                adjusted_num_S_sub = 0
            else:
                adjusted_num_S_sub = num_S_sub / adjusted_site_S
            if adjusted_site_N <= g['float_tol']:
                adjusted_num_N_sub = 0
            else:
                adjusted_num_N_sub = num_N_sub / adjusted_site_N
            denom = adjusted_num_S_sub + adjusted_num_N_sub
            if denom <= g['float_tol']:
                sdist = 0
                ndist = 0
            else:
                prop_S = adjusted_num_S_sub / denom
                prop_N = adjusted_num_N_sub / denom
                if num_S_sub<g['float_tol']:
                    sdist = 0
                else:
                    sdist = node.dist * prop_S
                    #node.Sdist = adjusted_site_S / prop_S
                if num_N_sub<g['float_tol']:
                    ndist = 0
                else:
                    ndist = node.dist * prop_N
                    #node.Ndist = adjusted_site_N / prop_N
            ete.set_prop(node, "Sdist", sdist)
            ete.set_prop(node, "Ndist", ndist)
            ete.set_prop(node, "SNdist", sdist + ndist)

    print('Total S+N branch length after rescaling: {:,.3f} codon substitutions / codon site'.format(
        sum([ete.get_prop(n, "SNdist", 0) for n in g['tree'].traverse()])
    ))
    print('Total S branch length after rescaling: {:,.3f} codon substitutions / codon site'.format(
        sum([ete.get_prop(n, "Sdist", 0) for n in g['tree'].traverse()])
    ))
    print('Total N branch length after rescaling: {:,.3f} codon substitutions / codon site'.format(
        sum([ete.get_prop(n, "Ndist", 0) for n in g['tree'].traverse()])
    ))
    return g

def read_treefile(g):
    with open(g['rooted_tree_file']) as f:
        rooted_newick = f.read()
    g['rooted_tree'] = ete.PhyloNode(rooted_newick, format=1)
    assert len(ete.get_children(g['rooted_tree']))==2, 'The input tree may be unrooted: {}'.format(g['rooted_tree_file'])
    g['rooted_tree'] = standardize_node_names(g['rooted_tree'])
    g['rooted_tree'] = add_numerical_node_labels(g['rooted_tree'])
    g['num_node'] = len(list(g['rooted_tree'].traverse()))
    print('Using internal node names and branch lengths in --iqtree_treefile '
          'and the root position in --rooted_tree_file.')
    return g

def is_consistent_tree_and_aln(g):
    leaf_names = [l.name for l in ete.get_leaves(g['rooted_tree'])]
    fasta_dict = sequence.read_fasta(path=g['alignment_file'])
    seq_names = list(fasta_dict.keys())
    is_consistent = set(leaf_names) == set(seq_names)
    if not is_consistent:
        dif1 = set(leaf_names) - set(seq_names)
        dif2 = set(seq_names) - set(leaf_names)
        if len(dif1):
            sys.stderr.write('Taxa that are present in tree but not in alignment: {}\n'.format(', '.join(dif1)))
        if len(dif2):
            sys.stderr.write('Taxa that are present in alignment but not in tree: {}\n'.format(', '.join(dif2)))
    return is_consistent

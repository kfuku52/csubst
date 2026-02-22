import numpy as np
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
AA_LOGO_MIN_PROB = 0.03
AA_LOGO_MAX_RESIDUES = 6
AA_LOGO_MIN_WIDTH = 0.04
AA_LOGO_WIDTH_RATIO = 0.015
AA_LOGO_HEIGHT = 0.78
AA_LOGO_COLORS = {
    'A': '#1b9e77',
    'C': '#66a61e',
    'D': '#d95f02',
    'E': '#d95f02',
    'F': '#7570b3',
    'G': '#1b9e77',
    'H': '#7570b3',
    'I': '#e7298a',
    'K': '#377eb8',
    'L': '#e7298a',
    'M': '#e7298a',
    'N': '#66a61e',
    'P': '#1b9e77',
    'Q': '#66a61e',
    'R': '#377eb8',
    'S': '#66a61e',
    'T': '#66a61e',
    'V': '#e7298a',
    'W': '#7570b3',
    'Y': '#7570b3',
}

def add_numerical_node_labels(tree):
    all_leaf_names = sorted(ete.get_leaf_names(tree))
    leaf_numerical_labels = dict()
    for i, leaf_name in enumerate(all_leaf_names):
        # Use Python integers to avoid precision loss for >=64 leaves.
        leaf_numerical_labels[leaf_name] = 1 << i

    nodes = list(tree.traverse())
    clade_signatures = list()
    for node in nodes:
        leaf_names = ete.get_leaf_names(node)
        clade_signature = sum(leaf_numerical_labels[leaf_name] for leaf_name in leaf_names)
        clade_signatures.append(clade_signature)

    # Rank node signatures with pure-Python sorting so large integer signatures
    # remain exact and deterministic across environments.
    sorted_node_indices = sorted(range(len(nodes)), key=lambda idx: clade_signatures[idx])
    rank_by_node_index = {node_index: rank for rank, node_index in enumerate(sorted_node_indices)}
    for node_index, node in enumerate(nodes):
        ete.set_prop(node, "numerical_label", rank_by_node_index[node_index])
    return tree

def is_consistent_tree(tree1, tree2):
    is_consistent_tree = set(ete.get_leaf_names(tree1)) == set(ete.get_leaf_names(tree2))
    return is_consistent_tree

def transfer_root(tree_to, tree_from, verbose=False):
    for node in tree_to.traverse():
        if node.dist is None:
            node.dist = 0.0
    subroot_nodes = ete.get_children(tree_from)
    if len(subroot_nodes) != 2:
        txt = 'Source tree root should be bifurcating for root transfer (found {} children).'
        raise ValueError(txt.format(len(subroot_nodes)))
    subroot_leaves = [ete.get_leaf_names(n) for n in subroot_nodes]
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
        raise ValueError('No root bipartition found in --infile.')
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
    if rf_dist != 0:
        raise AssertionError('tree topologies are different. RF distance = {}'.format(rf_dist))
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
    arr_dist = np.zeros(shape=(nrow, 3), dtype=float_type)
    arr_dist[:,0] = np.arange(nrow) + start
    for i in np.arange(nrow):
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
    cb.loc[:, 'dist_bl'] = np.nan
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
        if (not ete.is_root(node)) and (not ete.is_leaf(node)):
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
    import matplotlib.pyplot as plt
    return plt


def _get_logo_modules():
    import matplotlib.font_manager
    import matplotlib.patches
    import matplotlib.textpath
    import matplotlib.transforms
    font_properties = matplotlib.font_manager.FontProperties(weight='bold')
    return matplotlib.patches,matplotlib.textpath,matplotlib.transforms,font_properties


def _normalize_state_probabilities(probabilities):
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0:
        return None
    probs = np.where(probs < 0, 0.0, probs)
    probs = np.where(np.isfinite(probs), probs, 0.0)
    total = probs.sum()
    if total <= 0:
        return None
    return probs / total


def _select_logo_residues(probabilities, orders):
    probs = _normalize_state_probabilities(probabilities)
    if probs is None:
        return list()
    residues = list()
    for idx, aa in enumerate(orders):
        prob = float(probs[idx])
        if prob >= AA_LOGO_MIN_PROB:
            residues.append((str(aa), prob))
    if len(residues)==0:
        max_idx = int(probs.argmax())
        residues = [(str(orders[max_idx]), float(probs[max_idx]))]
    residues = sorted(residues, key=lambda x: x[1], reverse=True)[:AA_LOGO_MAX_RESIDUES]
    residues = sorted(residues, key=lambda x: x[1])
    return residues


def _draw_aa_logo(ax, x, y, probabilities, orders, logo_width, logo_height,
                  mpl_patches, mpl_textpath, mpl_transforms, font_properties):
    residues = _select_logo_residues(probabilities=probabilities, orders=orders)
    if len(residues)==0:
        return False
    total_height = logo_height * sum([r[1] for r in residues])
    if total_height <= 0:
        return False
    y_cursor = y - (total_height / 2.0)
    for aa,prob in residues:
        if prob <= 0:
            continue
        char = aa[0] if len(aa) else '-'
        glyph = mpl_textpath.TextPath((0, 0), char, size=1.0, prop=font_properties)
        bbox = glyph.get_extents()
        if (bbox.width <= 0) or (bbox.height <= 0):
            continue
        target_height = logo_height * prob
        sx = logo_width / bbox.width
        sy = target_height / bbox.height
        x_shift = x - (bbox.x0 * sx) + ((logo_width - (bbox.width * sx)) / 2.0)
        y_shift = y_cursor - (bbox.y0 * sy)
        tr = mpl_transforms.Affine2D().scale(sx, sy).translate(x_shift, y_shift) + ax.transData
        patch = mpl_patches.PathPatch(
            glyph,
            transform=tr,
            lw=0,
            facecolor=AA_LOGO_COLORS.get(char, 'black'),
            edgecolor='none',
        )
        ax.add_patch(patch)
        y_cursor += target_height
    return True


def _get_nice_scale_length(max_tree_depth):
    max_tree_depth = float(max_tree_depth)
    if max_tree_depth <= 0:
        return 1.0
    target = max_tree_depth * 0.2
    exponent = np.floor(np.log10(target))
    base = 10 ** exponent
    normalized = target / base
    if normalized <= 1.5:
        scale = 1.0
    elif normalized <= 3.5:
        scale = 2.0
    elif normalized <= 7.5:
        scale = 5.0
    else:
        scale = 10.0
    return scale * base


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


def _render_tree_matplotlib(tree, trait_name, file_name, label='all', state_by_node=None,
                            state_prob_by_node=None, state_orders=None, state_mode=None):
    plt = _get_pyplot()
    xcoord,ycoord,leaves = _get_tree_xy(tree)
    use_aa_logo = (state_mode=='aa') and (state_prob_by_node is not None) and (state_orders is not None)
    labels = []
    if use_aa_logo:
        max_label_len = max([len(leaf.name or '') for leaf in leaves], default=0) + 6
    else:
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
    if use_aa_logo:
        mpl_patches,mpl_textpath,mpl_transforms,font_properties = _get_logo_modules()
        logo_width = max(AA_LOGO_MIN_WIDTH, xspan * AA_LOGO_WIDTH_RATIO)
        leaf_name_offset = logo_width + (xspan * 0.01)
        fallback_leaf_name_offset = xspan * 0.03
        for node in tree.traverse():
            node_x = xcoord[id(node)] + text_offset
            node_y = ycoord[id(node)]
            text_color = ete.get_prop(node, "labelcolor_" + trait_name, "black")
            nl = ete.get_prop(node, "numerical_label")
            prob = state_prob_by_node.get(nl, None)
            is_logo_drawn = False
            if prob is not None:
                is_logo_drawn = _draw_aa_logo(
                    ax=ax,
                    x=node_x,
                    y=node_y,
                    probabilities=prob,
                    orders=state_orders,
                    logo_width=logo_width,
                    logo_height=AA_LOGO_HEIGHT,
                    mpl_patches=mpl_patches,
                    mpl_textpath=mpl_textpath,
                    mpl_transforms=mpl_transforms,
                    font_properties=font_properties,
                )
            if not is_logo_drawn:
                fallback = '-'
                if state_by_node is not None:
                    fallback = str(state_by_node.get(nl, '-'))
                ax.text(
                    node_x,
                    node_y,
                    fallback,
                    fontsize=6,
                    color=text_color,
                    va='center',
                    ha='left',
                    clip_on=False,
                )
            if ete.is_leaf(node) and (node.name is not None) and (len(node.name) > 0):
                label_x = node_x + (leaf_name_offset if is_logo_drawn else fallback_leaf_name_offset)
                ax.text(
                    label_x,
                    node_y,
                    node.name,
                    fontsize=5,
                    color=text_color,
                    va='center',
                    ha='left',
                    clip_on=False,
                )
        leaf_label_len = max([len(leaf.name or '') for leaf in leaves], default=0)
        text_space = logo_width + (leaf_label_len * xspan * 0.03) + (xspan * 0.04)
    elif max_label_len>0:
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
        text_space = max_label_len * xspan * 0.03
    else:
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


def _build_state_maps_for_site(tree, site_state, orders, mode, missing_state):
    state_by_node = dict()
    state_prob_by_node = dict() if mode=='aa' else None
    for node in tree.traverse():
        nlabel = ete.get_prop(node, "numerical_label")
        if ete.is_root(node):
            state_by_node[nlabel] = missing_state
            if state_prob_by_node is not None:
                state_prob_by_node[nlabel] = None
            continue
        node_state = site_state[nlabel, :]
        max_prob = float(node_state.max()) if node_state.size > 0 else 0.0
        if max_prob <= 0:
            state_by_node[nlabel] = missing_state
            if state_prob_by_node is not None:
                state_prob_by_node[nlabel] = None
            continue
        index = np.where(node_state==max_prob)[0]
        if len(index)==1:
            state_by_node[nlabel] = orders[index[0]]
        else:
            state_by_node[nlabel] = missing_state
        if state_prob_by_node is not None:
            state_prob_by_node[nlabel] = node_state
    return state_by_node,state_prob_by_node


def _render_state_tree_chunk(tree, trait_name, mode, orders, missing_state, state_chunk, site_indices, ndigit):
    for local_idx,site_index in enumerate(site_indices):
        site_state = state_chunk[:, local_idx, :]
        state_by_node,state_prob_by_node = _build_state_maps_for_site(
            tree=tree,
            site_state=site_state,
            orders=orders,
            mode=mode,
            missing_state=missing_state,
        )
        file_name = 'csubst_state_'+trait_name+'_'+mode+'_'+str(int(site_index)+1).zfill(ndigit)+'.pdf'
        file_name = file_name.replace('_PLACEHOLDER', '')
        _render_tree_matplotlib(
            tree=tree,
            trait_name=trait_name,
            file_name=file_name,
            label='all',
            state_by_node=state_by_node,
            state_prob_by_node=state_prob_by_node,
            state_orders=orders if mode=='aa' else None,
            state_mode=mode,
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
    num_site = int(state.shape[1])
    threads = int(g.get('threads', 1))
    for trait_name in trait_names:
        ndigit = int(np.log10(num_site))+1
        n_jobs = parallel.resolve_n_jobs(num_items=num_site, threads=threads)
        if n_jobs==1:
            site_indices = np.arange(num_site, dtype=np.int64)
            _render_state_tree_chunk(
                tree=g['tree'],
                trait_name=trait_name,
                mode=mode,
                orders=orders,
                missing_state=missing_state,
                state_chunk=state,
                site_indices=site_indices,
                ndigit=ndigit,
            )
            continue
        backend = parallel.resolve_parallel_backend(g=g, task='plot_state_tree')
        chunk_factor = parallel.resolve_chunk_factor(g=g, task='general')
        site_indices = np.arange(num_site, dtype=np.int64)
        site_chunks,_ = parallel.get_chunks(site_indices, threads=n_jobs, chunk_factor=chunk_factor)
        txt = 'Parallel state-tree plotting: trait = {}, mode = {}, workers = {}, backend = {}, chunks = {}'
        print(txt.format(trait_name, mode, n_jobs, backend, len(site_chunks)), flush=True)
        tasks = []
        for chunk in site_chunks:
            chunk = np.asarray(chunk, dtype=np.int64)
            state_chunk = state[:, chunk, :]
            tasks.append((g['tree'], trait_name, mode, orders, missing_state, state_chunk, chunk, ndigit))
        parallel.run_starmap(
            func=_render_state_tree_chunk,
            args_iterable=tasks,
            n_jobs=n_jobs,
            backend=backend,
        )

def get_num_adjusted_sites(g, node):
    nl = ete.get_prop(node, "numerical_label")
    parent = ete.get_prop(node.up, "numerical_label")
    child_states = g['state_cdn'][nl,:,:]
    parent_states = g['state_cdn'][parent,:,:].copy()
    is_child_present = np.expand_dims(child_states.sum(axis=1)!=0, axis=1)
    parent_states *= is_child_present
    codon_counts = parent_states.sum(axis=0)
    scaled_Q = np.copy(g['instantaneous_codon_rate_matrix'])
    np.fill_diagonal(scaled_Q, 0)
    row_sums = scaled_Q.sum(axis=1)
    nonzero_rows = row_sums > 0
    if nonzero_rows.any():
        scaled_Q[nonzero_rows, :] = scaled_Q[nonzero_rows, :] / np.expand_dims(row_sums[nonzero_rows], axis=1)
    if (~nonzero_rows).any():
        scaled_Q[~nonzero_rows, :] = 0
    adjusted_site_S = 0
    adjusted_site_N = 0
    for i in np.arange(codon_counts.shape[0]):
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
    if len(ete.get_children(g['rooted_tree'])) != 2:
        raise AssertionError('The input tree may be unrooted: {}'.format(g['rooted_tree_file']))
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

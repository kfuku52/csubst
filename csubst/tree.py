import numpy as np
import copy
import itertools
import os
import re
import sys
import time

from csubst import sequence
from csubst import parallel
from csubst import substitution
from csubst import ete

TREE_FIG_MIN_HEIGHT = 1.8
TREE_FIG_BASE_HEIGHT = 0.45
TREE_FIG_HEIGHT_PER_LEAF = 0.085
TREE_FIG_MAX_HEIGHT = 180.0
TREE_FIG_WIDTH = 7.12
TREE_FIG_SAVE_PAD_INCHES = 0.0
TREE_FIG_TITLE_X = 0.01
TREE_FIG_TITLE_Y = 0.995
TREE_CONTENT_MIN_WIDTH_RATIO = 0.42
TREE_TIP_LABEL_MIN_DISPLAY_CHARS = 22
TREE_TIP_LABEL_NO_ELLIPSIS_UP_TO_CHARS = 128
TREE_EXACT_TEXT_LAYOUT_MAX_ITEMS = 256
TREE_EXACT_TEXT_LAYOUT_MAX_LEAVES = 192
TREE_LABEL_X_PADDING_RATIO = 0.015
TREE_STATE_X_PADDING_RATIO = 0.002
TREE_ROOT_STATE_EXTRA_X_PADDING_RATIO = 0.006
TREE_STATE_LEAF_LABEL_GAP_RATIO = 0.006
TREE_SCALE_BAR_X_RATIO = 0.03
TREE_SCALE_BAR_Y = -0.28
TREE_SCALE_BAR_TICK_HALF_HEIGHT = 0.07
TREE_SCALE_BAR_LABEL_GAP = 0.03
TREE_SCALE_BAR_UNIT_LABEL = 'subs/codon site'
TREE_LINE_CAPSTYLE = 'projecting'
TREE_LINE_TERMINAL_CAPSTYLE = 'butt'
TREE_LINE_JOINSTYLE = 'miter'
TREE_STATE_TEXT_SIZE = 6
TREE_TIP_LABEL_TEXT_SIZE = 5
TREE_BRANCH_ID_TEXT_SIZE = TREE_TIP_LABEL_TEXT_SIZE * 0.5
TREE_BRANCH_ID_MIN_GAP = 0.03
TREE_SPECIATION_COLOR = (0.0, 0.0, 1.0)
TREE_DUPLICATION_COLOR = (1.0, 0.0, 0.0)
TREE_NODE_MARKER_AREA = 6.5
TREE_NODE_MARKER_EDGE_COLOR = 'white'
TREE_NODE_MARKER_EDGE_WIDTH = 0.35
TREE_NODE_MARKER_ZORDER = 2.0
TREE_STATE_ARTIST_ZORDER = 4.0
AA_LOGO_MIN_PROB = 0.03
AA_LOGO_MAX_RESIDUES = 6
AA_LOGO_MIN_WIDTH = 0.012
AA_LOGO_WIDTH_RATIO = 0.015
AA_LOGO_GAP_RATIO = 0.06
AA_LOGO_HEIGHT_FALLBACK = 0.78
AA_LOGO_SLOT_INNER_WIDTH_RATIO = 0.88
AA_LOGO_MISSING_BOX_FILL_COLOR = '#d0d0d0'
AA_LOGO_MISSING_BOX_EDGE_COLOR = '#7a7a7a'
AA_LOGO_MISSING_TEXT_COLOR = '#4d4d4d'
AA_LOGO_MISSING_BOX_WIDTH_RATIO = 1.0
AA_LOGO_MISSING_BAR_FILL_COLOR = '#4d4d4d'
AA_LOGO_MISSING_BAR_WIDTH_RATIO = 0.46
AA_LOGO_MISSING_BAR_HEIGHT_RATIO = 0.08
AA_LOGO_MISSING_CENTER_SHIFT_RATIO = 0.0
AA_LOGO_LEFT_PADDING_RATIO = 0.0
AA_LOGO_TEXT_WIDTH_RATIO = 0.9
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

_LOGO_MODULES = None
_AA_LOGO_GLYPH_CACHE = dict()


def _format_branch_id_label(branch_id):
    return 'b{}'.format(int(branch_id))

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
    tree_to = _clear_duplicate_internal_node_names(tree_to)
    return tree_to


def _clear_duplicate_internal_node_names(tree):
    seen = set()
    for node in tree.traverse():
        if ete.is_leaf(node):
            continue
        node_name = '' if (node.name is None) else str(node.name)
        if node_name == '':
            continue
        if node_name in seen:
            node.name = ''
            continue
        seen.add(node_name)
    return tree

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

def get_node_distance(
    tree,
    cb,
    ncpu,
    float_type,
    min_items_for_parallel=20000,
    min_items_per_job=5000,
):
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
    n_jobs = parallel.resolve_adaptive_n_jobs(
        num_items=id_combinations.shape[0],
        threads=ncpu,
        min_items_for_parallel=min_items_for_parallel,
        min_items_per_job=min_items_per_job,
    )
    txt = 'Branch-distance scheduler: combinations={}, workers={} (threads={})'
    print(txt.format(id_combinations.shape[0], n_jobs, ncpu), flush=True)
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
    global _LOGO_MODULES
    if _LOGO_MODULES is None:
        import matplotlib.font_manager
        import matplotlib.patches
        import matplotlib.textpath
        import matplotlib.transforms
        font_properties = matplotlib.font_manager.FontProperties(weight='bold')
        _LOGO_MODULES = (matplotlib.patches, matplotlib.textpath, matplotlib.transforms, font_properties)
    return _LOGO_MODULES


def _get_logo_glyph(mpl_textpath, font_properties, char):
    cache_key = (id(mpl_textpath), str(char))
    glyph = _AA_LOGO_GLYPH_CACHE.get(cache_key, None)
    if glyph is None:
        glyph = mpl_textpath.TextPath((0, 0), char, size=1.0, prop=font_properties)
        _AA_LOGO_GLYPH_CACHE[cache_key] = glyph
    return glyph


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
    draw_width = max(float(logo_width) * AA_LOGO_SLOT_INNER_WIDTH_RATIO, 1e-6)
    total_height = logo_height * sum([r[1] for r in residues])
    if total_height <= 0:
        return False
    y_cursor = y - (total_height / 2.0)
    for aa,prob in residues:
        if prob <= 0:
            continue
        char = aa[0] if len(aa) else '-'
        glyph = _get_logo_glyph(
            mpl_textpath=mpl_textpath,
            font_properties=font_properties,
            char=char,
        )
        bbox = glyph.get_extents()
        if (bbox.width <= 0) or (bbox.height <= 0):
            continue
        target_height = logo_height * prob
        sx = draw_width / bbox.width
        sy = target_height / bbox.height
        x_shift = (x - (draw_width / 2.0)) - (bbox.x0 * sx) + ((draw_width - (bbox.width * sx)) / 2.0)
        y_shift = y_cursor - (bbox.y0 * sy)
        tr = mpl_transforms.Affine2D().scale(sx, sy).translate(x_shift, y_shift) + ax.transData
        patch = mpl_patches.PathPatch(
            glyph,
            transform=tr,
            lw=0,
            facecolor=AA_LOGO_COLORS.get(char, 'black'),
            edgecolor='none',
            zorder=TREE_STATE_ARTIST_ZORDER,
        )
        ax.add_patch(patch)
        y_cursor += target_height
    return True


def _draw_logo_placeholder(ax, x, y, text, color, logo_width, logo_height):
    symbol = '-' if text in [None, ''] else str(text)[0]
    draw_width = max(float(logo_width) * AA_LOGO_SLOT_INNER_WIDTH_RATIO, 1e-6)
    box_center_x = float(x) + (float(logo_width) * AA_LOGO_MISSING_CENTER_SHIFT_RATIO)
    box_width = draw_width * AA_LOGO_MISSING_BOX_WIDTH_RATIO
    box_height = float(logo_height)
    box_left = box_center_x - (box_width / 2.0)
    try:
        from matplotlib import patches as mpl_patches
    except Exception:
        mpl_patches = None
    if (mpl_patches is not None) and hasattr(ax, 'add_patch'):
        rect = mpl_patches.Rectangle(
            (box_left, y - (box_height / 2.0)),
            box_width,
            box_height,
            facecolor=AA_LOGO_MISSING_BOX_FILL_COLOR,
            edgecolor=AA_LOGO_MISSING_BOX_EDGE_COLOR,
            linewidth=0.5,
            clip_on=False,
            zorder=TREE_STATE_ARTIST_ZORDER,
        )
        ax.add_patch(rect)
        if symbol == '-':
            bar_width = box_width * AA_LOGO_MISSING_BAR_WIDTH_RATIO
            bar_height = max(box_height * AA_LOGO_MISSING_BAR_HEIGHT_RATIO, 1e-6)
            bar = mpl_patches.Rectangle(
                (box_center_x - (bar_width / 2.0), y - (bar_height / 2.0)),
                bar_width,
                bar_height,
                facecolor=AA_LOGO_MISSING_BAR_FILL_COLOR,
                edgecolor='none',
                clip_on=False,
                zorder=TREE_STATE_ARTIST_ZORDER + 0.1,
            )
            ax.add_patch(bar)
        else:
            ax.text(
                box_center_x,
                y,
                symbol,
                fontsize=TREE_STATE_TEXT_SIZE,
                color=AA_LOGO_MISSING_TEXT_COLOR,
                va='center',
                ha='center',
                clip_on=False,
                zorder=TREE_STATE_ARTIST_ZORDER + 0.1,
            )
    else:
        ax.text(
            box_center_x,
            y,
            symbol,
            fontsize=TREE_STATE_TEXT_SIZE,
            color=AA_LOGO_MISSING_TEXT_COLOR,
            va='center',
            ha='center',
            clip_on=False,
            zorder=TREE_STATE_ARTIST_ZORDER + 0.1,
        )
    return True


def _is_missing_state_text(value):
    if value is None:
        return False
    text = str(value)
    if len(text) == 0:
        return False
    return set(text).issubset({'-'})


def _has_state_probability(probabilities):
    if probabilities is None:
        return False
    arr = np.asarray(probabilities, dtype=np.float64)
    if arr.size == 0:
        return False
    return (float(arr.max()) > 0.0)


def _get_logo_site_count(probabilities):
    if probabilities is None:
        return 0
    arr = np.asarray(probabilities, dtype=np.float64)
    if arr.ndim <= 1:
        return 1 if arr.size > 0 else 0
    return int(arr.shape[0])


def _get_text_height_in_data_units(ax, fontsize, fallback_height=AA_LOGO_HEIGHT_FALLBACK):
    fig = getattr(ax, 'figure', None)
    if fig is None:
        return float(fallback_height)
    try:
        canvas = getattr(fig, 'canvas', None)
        if (canvas is not None) and hasattr(canvas, 'draw'):
            canvas.draw()
        bbox = None
        if (
            (canvas is not None)
            and hasattr(canvas, 'get_renderer')
            and hasattr(ax, 'get_window_extent')
        ):
            bbox = ax.get_window_extent(renderer=canvas.get_renderer())
        if bbox is None:
            bbox = getattr(ax, 'bbox', None)
        if bbox is None:
            return float(fallback_height)
        height_px = float(getattr(bbox, 'height', 0.0))
        if height_px <= 0:
            return float(fallback_height)
        if not hasattr(ax, 'get_ylim'):
            return float(fallback_height)
        ylim = ax.get_ylim()
        y_range = abs(float(ylim[1]) - float(ylim[0]))
        if y_range <= 0:
            return float(fallback_height)
        dpi = float(getattr(fig, 'dpi', 72.0))
        font_height_px = float(fontsize) * dpi / 72.0
        return max(font_height_px * y_range / height_px, 1e-6)
    except Exception:
        return float(fallback_height)


def _get_text_width_measurement_context(ax, enable_exact=True):
    if not enable_exact:
        return None
    fig = getattr(ax, 'figure', None)
    if fig is None:
        return None
    try:
        canvas = getattr(fig, 'canvas', None)
        if (canvas is None) or (not hasattr(canvas, 'draw')) or (not hasattr(canvas, 'get_renderer')):
            return None
        if not hasattr(ax, 'get_window_extent'):
            return None
        canvas.draw()
        renderer = canvas.get_renderer()
        axes_bbox = ax.get_window_extent(renderer=renderer)
        axes_width_px = float(getattr(axes_bbox, 'width', 0.0))
        if axes_width_px <= 0:
            return None
        return {
            'fig': fig,
            'renderer': renderer,
            'axes_width_px': axes_width_px,
            'cache': dict(),
        }
    except Exception:
        return None


def _should_use_exact_text_layout(num_leaves, num_text_items):
    return (
        int(num_leaves) <= TREE_EXACT_TEXT_LAYOUT_MAX_LEAVES
        and int(num_text_items) <= TREE_EXACT_TEXT_LAYOUT_MAX_ITEMS
    )


def _get_text_width_axes_ratio(ax, text, fontsize, fallback_char_ratio, measurement_context=None):
    fallback_ratio = max(len(str(text)), 1) * float(fallback_char_ratio)
    if measurement_context is None:
        return fallback_ratio
    try:
        cache_key = (str(text), float(fontsize))
        cache = measurement_context.get('cache', {})
        if cache_key in cache:
            return cache[cache_key]
        from matplotlib.text import Text as mpl_text
        text_artist = mpl_text(x=0.0, y=0.0, text=str(text), fontsize=fontsize)
        text_artist.set_figure(measurement_context['fig'])
        text_bbox = text_artist.get_window_extent(renderer=measurement_context['renderer'])
        text_width_px = float(getattr(text_bbox, 'width', 0.0))
        if text_width_px <= 0:
            return fallback_ratio
        width_ratio = text_width_px / measurement_context['axes_width_px']
        cache[cache_key] = width_ratio
        measurement_context['cache'] = cache
        return width_ratio
    except Exception:
        return fallback_ratio


def _estimate_text_right_limit(ax, x_left, base_right, text_items, measurement_context=None):
    x_right = float(base_right)
    for item in text_items:
        text = str(item.get('text', ''))
        if len(text) == 0:
            continue
        anchor_x = float(item.get('x', 0.0))
        fontsize = float(item.get('fontsize', TREE_TIP_LABEL_TEXT_SIZE))
        ha = item.get('ha', 'left')
        fallback_char_ratio = float(item.get('fallback_char_ratio', 0.024))
        width_ratio = _get_text_width_axes_ratio(
            ax=ax,
            text=text,
            fontsize=fontsize,
            fallback_char_ratio=fallback_char_ratio,
            measurement_context=measurement_context,
        )
        width_ratio = min(max(width_ratio, 0.0), 0.95)
        if ha == 'center':
            anchor_factor = 0.5
        elif ha == 'right':
            anchor_factor = 0.0
        else:
            anchor_factor = 1.0
        denominator = 1.0 - (anchor_factor * width_ratio)
        if denominator <= 1e-6:
            continue
        candidate = (anchor_x - (anchor_factor * width_ratio * float(x_left))) / denominator
        x_right = max(x_right, candidate)
    return x_right


def _get_content_width_ratio(x_left, content_right, x_right):
    total_width = max(float(x_right) - float(x_left), 1e-12)
    return max(float(content_right) - float(x_left), 0.0) / total_width


def _ellipsize_middle(text, max_chars):
    text = str(text)
    max_chars = int(max(max_chars, 5))
    if len(text) <= max_chars:
        return text
    keep = max_chars - 3
    left = int(np.ceil(keep / 2.0))
    right = max(keep - left, 1)
    return text[:left] + '...' + text[-right:]


def _fit_leaf_label_items(ax, x_left, content_right, static_text_items, leaf_label_items, measurement_context=None):
    if len(leaf_label_items) == 0:
        x_right = _estimate_text_right_limit(
            ax=ax,
            x_left=x_left,
            base_right=content_right,
            text_items=static_text_items,
            measurement_context=measurement_context,
        )
        return list(), x_right

    full_items = [dict(item) for item in leaf_label_items]
    full_x_right = _estimate_text_right_limit(
        ax=ax,
        x_left=x_left,
        base_right=content_right,
        text_items=static_text_items + full_items,
        measurement_context=measurement_context,
    )
    if _get_content_width_ratio(x_left, content_right, full_x_right) >= TREE_CONTENT_MIN_WIDTH_RATIO:
        return full_items, full_x_right

    max_label_len = max([len(str(item.get('text', ''))) for item in leaf_label_items], default=0)
    if max_label_len <= TREE_TIP_LABEL_NO_ELLIPSIS_UP_TO_CHARS:
        return full_items, full_x_right
    min_chars = min(max_label_len, TREE_TIP_LABEL_MIN_DISPLAY_CHARS)
    best_items = None
    best_x_right = None
    low = min_chars
    high = max_label_len
    while low <= high:
        mid = int((low + high) // 2)
        trial_items = [
            dict(item, text=_ellipsize_middle(item.get('text', ''), mid))
            for item in leaf_label_items
        ]
        trial_x_right = _estimate_text_right_limit(
            ax=ax,
            x_left=x_left,
            base_right=content_right,
            text_items=static_text_items + trial_items,
            measurement_context=measurement_context,
        )
        if _get_content_width_ratio(x_left, content_right, trial_x_right) >= TREE_CONTENT_MIN_WIDTH_RATIO:
            best_items = trial_items
            best_x_right = trial_x_right
            low = mid + 1
        else:
            high = mid - 1
    if best_items is None:
        best_items = [
            dict(item, text=_ellipsize_middle(item.get('text', ''), max(min_chars, 5)))
            for item in leaf_label_items
        ]
        best_x_right = _estimate_text_right_limit(
            ax=ax,
            x_left=x_left,
            base_right=content_right,
            text_items=static_text_items + best_items,
            measurement_context=measurement_context,
        )
    return best_items, best_x_right


def _resolve_species_overlap_node_types(tree, species_regex='', species_overlap_node_plot='auto'):
    mode = 'auto' if (species_overlap_node_plot is None) else str(species_overlap_node_plot).strip().lower()
    if mode == 'no':
        return {}
    from csubst import main_sites as _main_sites
    return _main_sites.get_species_overlap_node_types(
        tree=tree,
        species_regex=species_regex,
        require_all_tip_labels=(mode == 'auto'),
    )


def _get_logo_width_from_height(logo_height, xspan, num_leaves, fig_width, fig_height):
    xspan = max(float(xspan), 1e-12)
    num_leaves = max(float(num_leaves), 1.0)
    fig_width = max(float(fig_width), 1e-12)
    fig_height = max(float(fig_height), 1e-12)
    width = float(logo_height)
    width *= AA_LOGO_TEXT_WIDTH_RATIO
    width *= (xspan / num_leaves)
    width *= (fig_height / fig_width)
    return max(float(AA_LOGO_MIN_WIDTH), width)


def _draw_aa_logo_series(ax, x, y, probabilities, orders, logo_width, logo_gap, logo_height,
                         mpl_patches, mpl_textpath, mpl_transforms, font_properties,
                         fallback_text=None, fallback_color='black'):
    arr = np.asarray(probabilities, dtype=np.float64)
    if arr.ndim <= 1:
        drawn = _draw_aa_logo(
            ax=ax,
            x=x,
            y=y,
            probabilities=arr,
            orders=orders,
            logo_width=logo_width,
            logo_height=logo_height,
            mpl_patches=mpl_patches,
            mpl_textpath=mpl_textpath,
            mpl_transforms=mpl_transforms,
            font_properties=font_properties,
        )
        if (not drawn) and (fallback_text is not None):
            drawn = _draw_logo_placeholder(
                ax=ax,
                x=x,
                y=y,
                text=str(fallback_text)[0:1],
                color=fallback_color,
                logo_width=logo_width,
                logo_height=logo_height,
            )
        return drawn, logo_width
    num_site = int(arr.shape[0])
    if num_site <= 0:
        return False, 0.0
    total_width = (num_site * logo_width) + (max(num_site - 1, 0) * logo_gap)
    left_center = x - (total_width / 2.0) + (logo_width / 2.0)
    any_drawn = False
    fallback_chars = list(str(fallback_text)) if fallback_text is not None else list()
    for idx in range(num_site):
        logo_x = left_center + (idx * (logo_width + logo_gap))
        drawn = _draw_aa_logo(
            ax=ax,
            x=logo_x,
            y=y,
            probabilities=arr[idx, :],
            orders=orders,
            logo_width=logo_width,
            logo_height=logo_height,
            mpl_patches=mpl_patches,
            mpl_textpath=mpl_textpath,
            mpl_transforms=mpl_transforms,
            font_properties=font_properties,
        )
        if (not drawn) and (fallback_text is not None):
            fallback_char = '-'
            if idx < len(fallback_chars):
                fallback_char = fallback_chars[idx]
            drawn = _draw_logo_placeholder(
                ax=ax,
                x=logo_x,
                y=y,
                text=fallback_char,
                color=fallback_color,
                logo_width=logo_width,
                logo_height=logo_height,
            )
        any_drawn = any_drawn or drawn
    return any_drawn, total_width


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


def _format_tree_scale_label(scale_length):
    return '{:g} {}'.format(float(scale_length), TREE_SCALE_BAR_UNIT_LABEL)


def _get_tree_figure_size(num_leaves, max_label_len):
    num_leaves = max(int(num_leaves), 1)
    fig_height = min(
        max(TREE_FIG_MIN_HEIGHT, TREE_FIG_BASE_HEIGHT + (num_leaves * TREE_FIG_HEIGHT_PER_LEAF)),
        TREE_FIG_MAX_HEIGHT,
    )
    fig_width = TREE_FIG_WIDTH
    return fig_width,fig_height


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
                            state_prob_by_node=None, state_orders=None, state_mode=None,
                            pdf_pages=None, figure_title=None, node_type_by_id=None):
    plt = _get_pyplot()
    xcoord,ycoord,leaves = _get_tree_xy(tree)
    use_aa_logo = (state_mode=='aa') and (state_prob_by_node is not None) and (state_orders is not None)
    max_logo_sites = 1
    if use_aa_logo:
        counts = [_get_logo_site_count(prob) for prob in state_prob_by_node.values() if prob is not None]
        if len(counts) > 0:
            max_logo_sites = max(counts)
    labels = []
    show_branch_id_labels = (state_by_node is None)
    branch_id_labels = []
    max_branch_id_len = 0
    if show_branch_id_labels:
        for node in tree.traverse():
            if ete.is_root(node):
                continue
            branch_id_labels.append((node, _format_branch_id_label(ete.get_prop(node, "numerical_label"))))
        max_branch_id_len = max([len(lbl[1]) for lbl in branch_id_labels], default=0)
    if use_aa_logo:
        max_label_len = max([len(leaf.name or '') for leaf in leaves], default=0) + 6
    else:
        for node in tree.traverse():
            if state_by_node is None:
                if label=='no':
                    continue
                if (label=='leaf') and (not ete.is_leaf(node)):
                    continue
                txt = node.name or ''
                font_size = 4
            else:
                nl = ete.get_prop(node, "numerical_label")
                state_txt = str(state_by_node.get(nl, '-'))
                if ete.is_root(node) and _is_missing_state_text(state_txt):
                    continue
                txt = state_txt + '|' + (node.name or '') if ete.is_leaf(node) else state_txt
                font_size = TREE_STATE_TEXT_SIZE
            labels.append((node, txt, font_size))
        max_label_len = max([len(lbl[1]) for lbl in labels], default=0)
    num_leaves = max(len(leaves), 1)
    fig_width,fig_height = _get_tree_figure_size(num_leaves=num_leaves, max_label_len=max_label_len)
    fig,ax = plt.subplots(figsize=(fig_width, fig_height))
    if hasattr(fig, 'subplots_adjust'):
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_ylim(-0.5, num_leaves - 0.5)
    static_right_text_items = list()
    leaf_label_items = list()
    branch_nodes = [node for node in tree.traverse() if (not ete.is_root(node))]
    if hasattr(ax, 'add_collection'):
        try:
            from matplotlib.collections import LineCollection
        except Exception:
            LineCollection = None
    else:
        LineCollection = None
    if LineCollection is not None:
        v_segments_by_color = dict()
        h_internal_segments_by_color = dict()
        h_terminal_segments_by_color = dict()
        for node in branch_nodes:
            parent = node.up
            x_parent = xcoord[id(parent)]
            y_parent = ycoord[id(parent)]
            x_node = xcoord[id(node)]
            y_node = ycoord[id(node)]
            v_color,h_color = _get_branch_segment_colors(node=node, trait_name=trait_name)
            v_segments_by_color.setdefault(v_color, list()).append([(x_parent, y_parent), (x_parent, y_node)])
            if ete.is_leaf(node):
                h_terminal_segments_by_color.setdefault(h_color, list()).append([(x_parent, y_node), (x_node, y_node)])
            else:
                h_internal_segments_by_color.setdefault(h_color, list()).append([(x_parent, y_node), (x_node, y_node)])
        for color, segments in v_segments_by_color.items():
            collection = LineCollection(segments, colors=color, linewidths=0.8)
            if hasattr(collection, 'set_capstyle'):
                collection.set_capstyle(TREE_LINE_CAPSTYLE)
            if hasattr(collection, 'set_joinstyle'):
                collection.set_joinstyle(TREE_LINE_JOINSTYLE)
            ax.add_collection(collection)
        for color, segments in h_internal_segments_by_color.items():
            collection = LineCollection(segments, colors=color, linewidths=0.8)
            if hasattr(collection, 'set_capstyle'):
                collection.set_capstyle(TREE_LINE_CAPSTYLE)
            if hasattr(collection, 'set_joinstyle'):
                collection.set_joinstyle(TREE_LINE_JOINSTYLE)
            ax.add_collection(collection)
        for color, segments in h_terminal_segments_by_color.items():
            collection = LineCollection(segments, colors=color, linewidths=0.8)
            if hasattr(collection, 'set_capstyle'):
                collection.set_capstyle(TREE_LINE_TERMINAL_CAPSTYLE)
            if hasattr(collection, 'set_joinstyle'):
                collection.set_joinstyle(TREE_LINE_JOINSTYLE)
            ax.add_collection(collection)
    else:
        for node in branch_nodes:
            parent = node.up
            x_parent = xcoord[id(parent)]
            y_parent = ycoord[id(parent)]
            x_node = xcoord[id(node)]
            y_node = ycoord[id(node)]
            v_color,h_color = _get_branch_segment_colors(node=node, trait_name=trait_name)
            ax.plot([x_parent, x_parent], [y_parent, y_node], color=v_color, linewidth=0.8, solid_capstyle=TREE_LINE_CAPSTYLE)
            horizontal_capstyle = TREE_LINE_TERMINAL_CAPSTYLE if ete.is_leaf(node) else TREE_LINE_CAPSTYLE
            ax.plot([x_parent, x_node], [y_node, y_node], color=h_color, linewidth=0.8, solid_capstyle=horizontal_capstyle)
    if node_type_by_id is None:
        node_type_by_id = {}
    if len(node_type_by_id) > 0 and hasattr(ax, 'scatter'):
        node_marker_coords = {'duplication': (list(), list()), 'speciation': (list(), list())}
        for node in tree.traverse():
            if ete.is_leaf(node):
                continue
            node_id = int(ete.get_prop(node, "numerical_label"))
            node_type = node_type_by_id.get(node_id, None)
            if node_type is None:
                continue
            xs,ys = node_marker_coords.setdefault(node_type, (list(), list()))
            xs.append(xcoord[id(node)])
            ys.append(ycoord[id(node)])
        for node_type,(xs,ys) in node_marker_coords.items():
            if len(xs) == 0:
                continue
            marker_color = TREE_DUPLICATION_COLOR if (node_type == 'duplication') else TREE_SPECIATION_COLOR
            ax.scatter(
                xs,
                ys,
                s=TREE_NODE_MARKER_AREA,
                marker='o',
                facecolor=marker_color,
                edgecolor=TREE_NODE_MARKER_EDGE_COLOR,
                linewidth=TREE_NODE_MARKER_EDGE_WIDTH,
                clip_on=False,
                zorder=TREE_NODE_MARKER_ZORDER,
            )
    xmax = max(xcoord.values()) if len(xcoord)>0 else 0.0
    xspan = max(xmax, 1.0)
    x_left = -xspan * 0.02
    if state_by_node is None:
        text_offset = xspan * TREE_LABEL_X_PADDING_RATIO
    else:
        text_offset = xspan * TREE_STATE_X_PADDING_RATIO
    branch_id_text_space = 0.0
    if show_branch_id_labels and (max_branch_id_len > 0):
        branch_id_text_height = _get_text_height_in_data_units(
            ax=ax,
            fontsize=TREE_BRANCH_ID_TEXT_SIZE,
            fallback_height=AA_LOGO_HEIGHT_FALLBACK * 0.5,
        )
        branch_id_gap = max(branch_id_text_height * 0.2, TREE_BRANCH_ID_MIN_GAP)
        for node,bid_txt in branch_id_labels:
            text_color = ete.get_prop(node, "labelcolor_" + trait_name, "black")
            branch_id_x = (xcoord[id(node.up)] + xcoord[id(node)]) / 2.0
            branch_id_y = ycoord[id(node)] - branch_id_gap
            ax.text(
                branch_id_x,
                branch_id_y,
                bid_txt,
                fontsize=TREE_BRANCH_ID_TEXT_SIZE,
                color=text_color,
                va='top',
                ha='center',
                clip_on=False,
            )
            static_right_text_items.append({
                'x': branch_id_x,
                'text': bid_txt,
                'fontsize': TREE_BRANCH_ID_TEXT_SIZE,
                'ha': 'center',
                'fallback_char_ratio': 0.02,
            })
    label_text_offset = text_offset + branch_id_text_space
    content_right = xmax + label_text_offset
    if use_aa_logo:
        mpl_patches,mpl_textpath,mpl_transforms,font_properties = _get_logo_modules()
        logo_height = _get_text_height_in_data_units(
            ax=ax,
            fontsize=TREE_TIP_LABEL_TEXT_SIZE,
            fallback_height=AA_LOGO_HEIGHT_FALLBACK,
        )
        logo_width = _get_logo_width_from_height(
            logo_height=logo_height,
            xspan=xspan,
            num_leaves=num_leaves,
            fig_width=fig_width,
            fig_height=fig_height,
        )
        logo_gap = logo_width * AA_LOGO_GAP_RATIO
        logo_left_padding = xspan * AA_LOGO_LEFT_PADDING_RATIO
        fallback_leaf_name_offset = xspan * 0.018
        leaf_label_gap = max(logo_width * 0.22, xspan * TREE_STATE_LEAF_LABEL_GAP_RATIO)
        for node in tree.traverse():
            root_logo_extra_offset = 0.0
            if ete.is_root(node):
                root_logo_extra_offset = max(
                    logo_width * 0.35,
                    xspan * TREE_ROOT_STATE_EXTRA_X_PADDING_RATIO,
                )
            node_logo_left = xcoord[id(node)] + label_text_offset + logo_left_padding + root_logo_extra_offset
            node_y = ycoord[id(node)]
            text_color = ete.get_prop(node, "labelcolor_" + trait_name, "black")
            nl = ete.get_prop(node, "numerical_label")
            prob = state_prob_by_node.get(nl, None)
            fallback = '-'
            if state_by_node is not None:
                fallback = str(state_by_node.get(nl, '-'))
            logo_site_count = max(_get_logo_site_count(prob), max(len(fallback), 1))
            logo_target_width = (logo_site_count * logo_width) + (max(logo_site_count - 1, 0) * logo_gap)
            node_x = node_logo_left + (logo_target_width / 2.0)
            content_right = max(content_right, node_logo_left + logo_target_width)
            is_logo_drawn = False
            logo_total_width = logo_width
            if prob is not None:
                is_logo_drawn,logo_total_width = _draw_aa_logo_series(
                    ax=ax,
                    x=node_x,
                    y=node_y,
                    probabilities=prob,
                    orders=state_orders,
                    logo_width=logo_width,
                    logo_gap=logo_gap,
                    logo_height=logo_height,
                    mpl_patches=mpl_patches,
                    mpl_textpath=mpl_textpath,
                    mpl_transforms=mpl_transforms,
                    font_properties=font_properties,
                    fallback_text=fallback,
                    fallback_color=text_color,
                )
            if not is_logo_drawn:
                hide_missing_root = ete.is_root(node) and (not _has_state_probability(prob)) and _is_missing_state_text(fallback)
                if not hide_missing_root:
                    ax.text(
                        node_logo_left,
                        node_y,
                        fallback,
                        fontsize=TREE_STATE_TEXT_SIZE,
                        color=text_color,
                        va='center',
                        ha='left',
                        clip_on=False,
                        zorder=TREE_STATE_ARTIST_ZORDER + 0.1,
                    )
                    static_right_text_items.append({
                        'x': node_logo_left,
                        'text': fallback,
                        'fontsize': TREE_STATE_TEXT_SIZE,
                        'ha': 'left',
                        'fallback_char_ratio': 0.02,
                    })
            if ete.is_leaf(node) and (node.name is not None) and (len(node.name) > 0):
                if is_logo_drawn:
                    label_x = node_logo_left + logo_total_width + leaf_label_gap
                else:
                    label_x = node_logo_left + fallback_leaf_name_offset
                leaf_label_items.append({
                    'x': label_x,
                    'y': node_y,
                    'text': node.name,
                    'fontsize': TREE_TIP_LABEL_TEXT_SIZE,
                    'color': text_color,
                    'va': 'center',
                    'ha': 'left',
                    'clip_on': False,
                    'fallback_char_ratio': 0.024,
                })
    elif max_label_len>0:
        for node,txt,font_size in labels:
            text_color = ete.get_prop(node, "labelcolor_" + trait_name, "black")
            root_label_extra_offset = xspan * TREE_ROOT_STATE_EXTRA_X_PADDING_RATIO if (state_by_node is not None and ete.is_root(node)) else 0.0
            label_x = xcoord[id(node)] + label_text_offset + root_label_extra_offset
            if (state_by_node is None) and ete.is_leaf(node):
                leaf_label_items.append({
                    'x': label_x,
                    'y': ycoord[id(node)],
                    'text': txt,
                    'fontsize': font_size,
                    'color': text_color,
                    'va': 'center',
                    'ha': 'left',
                    'clip_on': False,
                    'fallback_char_ratio': 0.024,
                })
            else:
                ax.text(
                    label_x,
                    ycoord[id(node)],
                    txt,
                    fontsize=font_size,
                    color=text_color,
                    va='center',
                    ha='left',
                    clip_on=False,
                    zorder=TREE_STATE_ARTIST_ZORDER + 0.1,
                )
                static_right_text_items.append({
                    'x': label_x,
                    'text': txt,
                    'fontsize': font_size,
                    'ha': 'left',
                    'fallback_char_ratio': 0.024,
                })
    else:
        for node,txt,font_size in labels:
            text_color = ete.get_prop(node, "labelcolor_" + trait_name, "black")
            root_label_extra_offset = xspan * TREE_ROOT_STATE_EXTRA_X_PADDING_RATIO if (state_by_node is not None and ete.is_root(node)) else 0.0
            label_x = xcoord[id(node)] + label_text_offset + root_label_extra_offset
            if (state_by_node is None) and ete.is_leaf(node):
                leaf_label_items.append({
                    'x': label_x,
                    'y': ycoord[id(node)],
                    'text': txt,
                    'fontsize': font_size,
                    'color': text_color,
                    'va': 'center',
                    'ha': 'left',
                    'clip_on': False,
                    'fallback_char_ratio': 0.024,
                })
            else:
                ax.text(
                    label_x,
                    ycoord[id(node)],
                    txt,
                    fontsize=font_size,
                    color=text_color,
                    va='center',
                    ha='left',
                    clip_on=False,
                    zorder=TREE_STATE_ARTIST_ZORDER + 0.1,
                )
                static_right_text_items.append({
                    'x': label_x,
                    'text': txt,
                    'fontsize': font_size,
                    'ha': 'left',
                    'fallback_char_ratio': 0.024,
                })
    scale_length = _get_nice_scale_length(xmax)
    scale_x_start = xspan * TREE_SCALE_BAR_X_RATIO
    scale_x_end = scale_x_start + scale_length
    if scale_x_end > (xmax * 0.98):
        scale_length = _get_nice_scale_length(xmax * 0.5)
        scale_x_end = scale_x_start + scale_length
    if scale_length > 0:
        scale_label_y = TREE_SCALE_BAR_Y + TREE_SCALE_BAR_TICK_HALF_HEIGHT + TREE_SCALE_BAR_LABEL_GAP
        ax.plot(
            [scale_x_start, scale_x_end],
            [TREE_SCALE_BAR_Y, TREE_SCALE_BAR_Y],
            color='black',
            linewidth=1.0,
            solid_capstyle=TREE_LINE_CAPSTYLE,
        )
        ax.plot(
            [scale_x_start, scale_x_start],
            [TREE_SCALE_BAR_Y - TREE_SCALE_BAR_TICK_HALF_HEIGHT, TREE_SCALE_BAR_Y + TREE_SCALE_BAR_TICK_HALF_HEIGHT],
            color='black',
            linewidth=1.0,
            solid_capstyle=TREE_LINE_CAPSTYLE,
        )
        ax.plot(
            [scale_x_end, scale_x_end],
            [TREE_SCALE_BAR_Y - TREE_SCALE_BAR_TICK_HALF_HEIGHT, TREE_SCALE_BAR_Y + TREE_SCALE_BAR_TICK_HALF_HEIGHT],
            color='black',
            linewidth=1.0,
            solid_capstyle=TREE_LINE_CAPSTYLE,
        )
        ax.text(
            (scale_x_start + scale_x_end) / 2.0,
            scale_label_y,
            _format_tree_scale_label(scale_length),
            fontsize=TREE_TIP_LABEL_TEXT_SIZE,
            color='black',
            va='bottom',
            ha='center',
            clip_on=False,
        )
    fitted_leaf_items,x_right = _fit_leaf_label_items(
        ax=ax,
        x_left=x_left,
        content_right=content_right,
        static_text_items=static_right_text_items,
        leaf_label_items=leaf_label_items,
        measurement_context=_get_text_width_measurement_context(
            ax=ax,
            enable_exact=_should_use_exact_text_layout(
                num_leaves=num_leaves,
                num_text_items=len(static_right_text_items) + len(leaf_label_items),
            ),
        ),
    )
    for item in fitted_leaf_items:
        ax.text(
            item['x'],
            item['y'],
            item['text'],
            fontsize=item['fontsize'],
            color=item['color'],
            va=item['va'],
            ha=item['ha'],
            clip_on=item['clip_on'],
        )
    ax.set_xlim(x_left, x_right + (xspan * 0.004))
    ax.axis('off')
    if figure_title:
        title_kwargs = {
            'fontsize': TREE_STATE_TEXT_SIZE,
            'ha': 'left',
            'va': 'top',
            'clip_on': False,
        }
        if hasattr(ax, 'transAxes'):
            title_kwargs['transform'] = ax.transAxes
        ax.text(
            TREE_FIG_TITLE_X,
            TREE_FIG_TITLE_Y,
            str(figure_title),
            **title_kwargs,
        )
    if pdf_pages is not None:
        pdf_pages.savefig(fig, transparent=True, pad_inches=TREE_FIG_SAVE_PAD_INCHES)
    else:
        fig.savefig(file_name, format='pdf', transparent=True, pad_inches=TREE_FIG_SAVE_PAD_INCHES)
    plt.close(fig)


def _render_tree_matplotlib_with_optional_node_types(node_type_by_id=None, **kwargs):
    render_kwargs = dict(kwargs)
    if node_type_by_id is not None:
        try:
            has_values = (len(node_type_by_id) > 0)
        except TypeError:
            has_values = True
        if has_values:
            render_kwargs['node_type_by_id'] = node_type_by_id
    return _render_tree_matplotlib(**render_kwargs)
    return None


def plot_branch_category(g, file_base, label='all'):
    if not is_ete_plottable():
        return None
    trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)]
    node_type_by_id = _resolve_species_overlap_node_types(
        tree=g['tree'],
        species_regex=g.get('species_regex', ''),
        species_overlap_node_plot=g.get('species_overlap_node_plot', 'auto'),
    )
    for trait_name in trait_names:
        file_name = file_base+'_'+trait_name+'.pdf'
        file_name = file_name.replace('_PLACEHOLDER', '')
        _render_tree_matplotlib_with_optional_node_types(
            tree=g['tree'],
            trait_name=trait_name,
            file_name=file_name,
            label=label,
            state_by_node=None,
            node_type_by_id=node_type_by_id,
        )


def normalize_state_plot_request(value, param_name='state plot selector'):
    if isinstance(value, dict):
        mode = str(value.get('mode', 'none')).strip().lower()
        if mode not in ['none', 'all', 'pages', 'concat']:
            raise ValueError('{} should be one of no, all, SITE,SITE,..., or SITE-SITE-....'.format(param_name))
        site_numbers = tuple(int(v) for v in value.get('site_numbers', tuple()))
        token = str(value.get('token', '')).strip()
        return {
            'mode': mode,
            'site_numbers': site_numbers,
            'token': token,
        }
    if isinstance(value, (bool, np.bool_)):
        mode = 'all' if bool(value) else 'none'
        return {
            'mode': mode,
            'site_numbers': tuple(),
            'token': mode,
        }
    raw = '' if value is None else str(value).strip()
    normalized = raw.lower()
    if normalized in ['', 'no', 'none', 'off', 'false', '0']:
        return {
            'mode': 'none',
            'site_numbers': tuple(),
            'token': 'no',
        }
    if normalized == 'all':
        return {
            'mode': 'all',
            'site_numbers': tuple(),
            'token': 'all',
        }
    if normalized in ['yes', 'y', 'on', 'true', '1']:
        txt = '{} no longer accepts yes/no. Use "all", a comma-separated site list, '
        txt += 'or a hyphen-joined site bundle.'
        raise ValueError(txt.format(param_name))
    has_comma = ',' in normalized
    has_hyphen = '-' in normalized
    if has_comma and has_hyphen:
        txt = '{} should use either commas (multi-page PDF) or hyphens (concatenated single-page PDF), not both.'
        raise ValueError(txt.format(param_name))
    separator = ',' if has_comma else '-'
    raw_tokens = [token.strip() for token in normalized.split(separator) if token.strip() != '']
    if len(raw_tokens) == 0:
        raise ValueError('{} should be "no", "all", or one or more positive site numbers.'.format(param_name))
    site_numbers = list()
    seen = set()
    for token in raw_tokens:
        if not re.fullmatch(r'[0-9]+', token):
            raise ValueError('{} should contain positive integer site numbers.'.format(param_name))
        number = int(token)
        if number <= 0:
            raise ValueError('{} should contain positive integer site numbers.'.format(param_name))
        if number in seen:
            continue
        seen.add(number)
        site_numbers.append(number)
    mode = 'concat' if has_hyphen else 'pages'
    if mode == 'concat':
        token = '-'.join([str(v) for v in site_numbers])
    else:
        token = ','.join([str(v) for v in site_numbers])
    return {
        'mode': mode,
        'site_numbers': tuple(site_numbers),
        'token': token,
    }


def has_state_plot_request(value):
    return normalize_state_plot_request(value=value).get('mode', 'none') != 'none'


def _resolve_state_plot_site_indices(num_site, plot_request, param_name):
    request = normalize_state_plot_request(value=plot_request, param_name=param_name)
    mode = request['mode']
    if mode == 'none':
        return request, np.array([], dtype=np.int64)
    if mode == 'all':
        return request, np.arange(int(num_site), dtype=np.int64)
    site_numbers = np.asarray(request['site_numbers'], dtype=np.int64)
    if site_numbers.size == 0:
        return request, np.array([], dtype=np.int64)
    invalid = site_numbers[(site_numbers < 1) | (site_numbers > int(num_site))]
    if invalid.size > 0:
        invalid_txt = ','.join([str(int(v)) for v in invalid.tolist()])
        txt = '{} included out-of-range site(s): {}. Valid sites are 1-{}.'
        raise ValueError(txt.format(param_name, invalid_txt, int(num_site)))
    return request, (site_numbers - 1).astype(np.int64, copy=False)


def _build_state_maps_for_site(tree, site_state, orders, mode, missing_state):
    state_by_node = dict()
    state_prob_by_node = dict() if mode=='aa' else None
    for node in tree.traverse():
        nlabel = ete.get_prop(node, "numerical_label")
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


def _build_state_maps_for_concatenated_sites(tree, state, site_indices, orders, missing_state):
    state_by_node = dict()
    state_prob_by_node = dict() if len(site_indices) > 0 else None
    for node in tree.traverse():
        nlabel = ete.get_prop(node, "numerical_label")
        symbols = list()
        site_probabilities = list()
        for site_index in site_indices.tolist():
            node_state = state[nlabel, int(site_index), :]
            max_prob = float(node_state.max()) if node_state.size > 0 else 0.0
            site_probabilities.append(np.asarray(node_state, dtype=np.float64))
            if max_prob <= 0:
                symbols.append(missing_state)
                continue
            index = np.where(node_state == max_prob)[0]
            if len(index) == 1:
                symbols.append(str(orders[index[0]]))
            else:
                symbols.append(missing_state)
        state_by_node[nlabel] = ''.join(symbols)
        if state_prob_by_node is not None:
            state_prob_by_node[nlabel] = np.stack(site_probabilities, axis=0)
    return state_by_node,state_prob_by_node


def _render_state_tree_chunk(tree, trait_name, mode, orders, missing_state, state_chunk, site_indices, ndigit,
                             output_dir=None, node_type_by_id=None, site_number_labels=None):
    for local_idx,site_index in enumerate(site_indices):
        site_state = state_chunk[:, local_idx, :]
        state_by_node,state_prob_by_node = _build_state_maps_for_site(
            tree=tree,
            site_state=site_state,
            orders=orders,
            mode=mode,
            missing_state=missing_state,
        )
        site_number = int(site_index) + 1
        if site_number_labels is not None:
            site_number = int(site_number_labels[local_idx])
        file_name = 'csubst_state_'+trait_name+'_'+mode+'_'+str(site_number).zfill(ndigit)+'.pdf'
        file_name = file_name.replace('_PLACEHOLDER', '')
        if output_dir is not None:
            file_name = os.path.join(output_dir, file_name)
        _render_tree_matplotlib_with_optional_node_types(
            tree=tree,
            trait_name=trait_name,
            file_name=file_name,
            label='all',
            state_by_node=state_by_node,
            state_prob_by_node=state_prob_by_node,
            state_orders=orders if mode=='aa' else None,
            state_mode=mode,
            node_type_by_id=node_type_by_id,
        )


def _render_state_tree_bundle(tree, trait_name, mode, orders, missing_state, state, site_indices, output_token,
                              output_dir=None, node_type_by_id=None, site_number_labels=None):
    from matplotlib.backends.backend_pdf import PdfPages

    file_name = 'csubst_state_' + trait_name + '_' + mode + '_' + str(output_token) + '.pdf'
    file_name = file_name.replace('_PLACEHOLDER', '')
    if output_dir is not None:
        file_name = os.path.join(output_dir, file_name)
    with PdfPages(file_name) as pdf_pages:
        for local_idx, site_index in enumerate(site_indices.tolist()):
            site_state = state[:, int(site_index), :]
            state_by_node,state_prob_by_node = _build_state_maps_for_site(
                tree=tree,
                site_state=site_state,
                orders=orders,
                mode=mode,
                missing_state=missing_state,
            )
            site_number = int(site_index) + 1
            if site_number_labels is not None:
                site_number = int(site_number_labels[local_idx])
            _render_tree_matplotlib_with_optional_node_types(
                tree=tree,
                trait_name=trait_name,
                file_name=file_name,
                label='all',
                state_by_node=state_by_node,
                state_prob_by_node=state_prob_by_node,
                state_orders=orders if mode=='aa' else None,
                state_mode=mode,
                pdf_pages=pdf_pages,
                figure_title='Site {}'.format(site_number),
                node_type_by_id=node_type_by_id,
            )
    return file_name


def _render_state_tree_concatenated(tree, trait_name, mode, orders, missing_state, state, site_indices, output_token,
                                    output_dir=None, node_type_by_id=None):
    file_name = 'csubst_state_' + trait_name + '_' + mode + '_' + str(output_token) + '.pdf'
    file_name = file_name.replace('_PLACEHOLDER', '')
    if output_dir is not None:
        file_name = os.path.join(output_dir, file_name)
    state_by_node,state_prob_by_node = _build_state_maps_for_concatenated_sites(
        tree=tree,
        state=state,
        site_indices=site_indices,
        orders=orders,
        missing_state=missing_state,
    )
    _render_tree_matplotlib_with_optional_node_types(
        tree=tree,
        trait_name=trait_name,
        file_name=file_name,
        label='all',
        state_by_node=state_by_node,
        state_prob_by_node=state_prob_by_node if mode == 'aa' else None,
        state_orders=orders if mode == 'aa' else None,
        state_mode=mode,
        figure_title='Sites {}'.format(str(output_token)),
        node_type_by_id=node_type_by_id,
    )
    return file_name


def plot_state_tree(state, orders, mode, g, output_dir=None, plot_request='all', plot_request_name=None):
    if not is_ete_plottable():
        return None
    if state.shape[1] == 0:
        print('No sites available for ancestral state tree plotting. Skipping.', flush=True)
        return None
    if plot_request_name is None:
        plot_request_name = '--plot_state_{}'.format('aa' if mode == 'aa' else mode)
    request,param_name = None,plot_request_name
    request,site_indices = _resolve_state_plot_site_indices(
        num_site=state.shape[1],
        plot_request=plot_request,
        param_name=param_name,
    )
    if site_indices.size == 0:
        print('No sites selected for ancestral state tree plotting. Skipping.', flush=True)
        return None
    print('Writing ancestral state trees: mode = {}, number of pdf files = 1'.format(mode), flush=True)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    if mode=='codon':
        missing_state = '---'
    else:
        missing_state = '-'
    trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)]
    node_type_by_id = _resolve_species_overlap_node_types(
        tree=g['tree'],
        species_regex=g.get('species_regex', ''),
        species_overlap_node_plot=g.get('species_overlap_node_plot', 'auto'),
    )
    out_files = list()
    for trait_name in trait_names:
        if request['mode'] in ['all', 'pages']:
            out_files.append(
                _render_state_tree_bundle(
                    tree=g['tree'],
                    trait_name=trait_name,
                    mode=mode,
                    orders=orders,
                    missing_state=missing_state,
                    state=state,
                    site_indices=site_indices,
                    output_token=request['token'],
                    output_dir=output_dir,
                    node_type_by_id=node_type_by_id,
                )
            )
            continue
        out_files.append(
            _render_state_tree_concatenated(
                tree=g['tree'],
                trait_name=trait_name,
                mode=mode,
                orders=orders,
                missing_state=missing_state,
                state=state,
                site_indices=site_indices,
                output_token=request['token'],
                output_dir=output_dir,
                node_type_by_id=node_type_by_id,
            )
        )
    return out_files


def plot_state_tree_selected_sites(state, orders, mode, g, site_numbers, output_dir=None, plot_request='all', plot_request_name=None):
    if not is_ete_plottable():
        return None
    site_numbers = np.asarray(site_numbers, dtype=np.int64).reshape(-1)
    if site_numbers.size == 0:
        print('No sites available for ancestral state tree plotting. Skipping.', flush=True)
        return None
    if state.shape[1] != site_numbers.shape[0]:
        txt = 'state site axis ({}) did not match provided site_numbers length ({}).'
        raise ValueError(txt.format(state.shape[1], site_numbers.shape[0]))
    if plot_request_name is None:
        plot_request_name = '--plot_state_{}'.format('aa' if mode == 'aa' else mode)
    request = normalize_state_plot_request(value=plot_request, param_name=plot_request_name)
    if request['mode'] == 'none':
        print('No sites selected for ancestral state tree plotting. Skipping.', flush=True)
        return None
    if request['mode'] == 'all':
        requested_site_indices = (site_numbers - 1).astype(np.int64, copy=False)
    else:
        _, requested_site_indices = _resolve_state_plot_site_indices(
            num_site=int(g.get('num_input_site', int(site_numbers.max()) if site_numbers.size else 0)),
            plot_request=plot_request,
            param_name=plot_request_name,
        )
    local_index_by_global = {
        int(global_index): int(local_index)
        for local_index, global_index in enumerate((site_numbers - 1).tolist())
    }
    local_indices = list()
    site_number_labels = list()
    for global_index in requested_site_indices.tolist():
        local_index = local_index_by_global.get(int(global_index), None)
        if local_index is None:
            txt = '{} requested site {} which was not loaded into the selected-site state tensor.'
            raise ValueError(txt.format(plot_request_name, int(global_index) + 1))
        local_indices.append(int(local_index))
        site_number_labels.append(int(global_index) + 1)
    if len(local_indices) == 0:
        print('No sites selected for ancestral state tree plotting. Skipping.', flush=True)
        return None
    local_indices = np.asarray(local_indices, dtype=np.int64)
    site_number_labels = np.asarray(site_number_labels, dtype=np.int64)
    print('Writing ancestral state trees: mode = {}, number of pdf files = 1'.format(mode), flush=True)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    missing_state = '---' if mode == 'codon' else '-'
    trait_names = g['fg_df'].columns[1:len(g['fg_df'].columns)]
    node_type_by_id = _resolve_species_overlap_node_types(
        tree=g['tree'],
        species_regex=g.get('species_regex', ''),
        species_overlap_node_plot=g.get('species_overlap_node_plot', 'auto'),
    )
    out_files = list()
    for trait_name in trait_names:
        if request['mode'] in ['all', 'pages']:
            out_files.append(
                _render_state_tree_bundle(
                    tree=g['tree'],
                    trait_name=trait_name,
                    mode=mode,
                    orders=orders,
                    missing_state=missing_state,
                    state=state,
                    site_indices=local_indices,
                    output_token=request['token'],
                    output_dir=output_dir,
                    node_type_by_id=node_type_by_id,
                    site_number_labels=site_number_labels,
                )
            )
            continue
        out_files.append(
            _render_state_tree_concatenated(
                tree=g['tree'],
                trait_name=trait_name,
                mode=mode,
                orders=orders,
                missing_state=missing_state,
                state=state,
                site_indices=local_indices,
                output_token=request['token'],
                output_dir=output_dir,
                node_type_by_id=node_type_by_id,
            )
        )
    return out_files

def get_num_adjusted_sites(g, node):
    nl = ete.get_prop(node, "numerical_label")
    state_has_mass = (g['state_cdn'].sum(axis=(1, 2)) > float(g.get('float_tol', 0)))
    parent_node = ete.get_effective_state_parent(node, state_has_mass=state_has_mass)
    if parent_node is None:
        return 0, 0
    parent = ete.get_prop(parent_node, "numerical_label")
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
    state_has_mass = (g['state_cdn'].sum(axis=(1, 2)) > float(g.get('float_tol', 0)))
    for node in g['tree'].traverse():
        if ete.is_root(node):
            ete.set_prop(node, "Sdist", 0)
            ete.set_prop(node, "Ndist", 0)
            ete.set_prop(node, "SNdist", 0)
            continue
        nl = ete.get_prop(node, "numerical_label")
        parent_node, branch_length_to_parent = ete.get_effective_state_parent(
            node,
            state_has_mass=state_has_mass,
            accumulate_distance=True,
        )
        if parent_node is None:
            ete.set_prop(node, "Sdist", 0)
            ete.set_prop(node, "Ndist", 0)
            ete.set_prop(node, "SNdist", 0)
            continue
        parent = ete.get_prop(parent_node, "numerical_label")
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
                    sdist = branch_length_to_parent * prop_S
                    #node.Sdist = adjusted_site_S / prop_S
                if num_N_sub<g['float_tol']:
                    ndist = 0
                else:
                    ndist = branch_length_to_parent * prop_N
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

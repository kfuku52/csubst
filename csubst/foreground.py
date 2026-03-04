import numpy as np
import pandas as pd

import copy
import itertools
import re
import sys
import time
import warnings

from csubst import combination
from csubst import omega
from csubst import table
from csubst import param
from csubst import ete
from csubst import substitution
from csubst import tree
from csubst import output_stat


def _get_trait_names(g):
    return g['fg_df'].columns[1:len(g['fg_df'].columns)]


def _invalidate_clade_permutation_cache(g):
    for key in [
        '_node_by_branch_id_cache',
        '_clade_permutation_trait_cache',
        '_clade_permutation_pair_block_cache',
        '_clade_permutation_combination_count_cache',
        '_clade_permutation_randomization_plan_cache',
        '_permutation_cb_pair_lookup_cache',
        '_clade_permutation_fast_stats_plan_cache',
        '_clade_permutation_selected_rows_plan_cache',
    ]:
        if key in g:
            del g[key]
    return g


def _get_node_by_branch_id(g):
    cached = g.get('_node_by_branch_id_cache', None)
    if cached is not None:
        return cached
    node_by_id = dict()
    for node in g['tree'].traverse():
        branch_id = int(ete.get_prop(node, "numerical_label"))
        node_by_id[branch_id] = node
    g['_node_by_branch_id_cache'] = node_by_id
    return node_by_id


def _build_trait_clade_permutation_cache(g, trait_name):
    node_by_id = _get_node_by_branch_id(g)
    tree_obj = g['tree']
    branch_ids = np.array(
        sorted([
            bid
            for bid, node in node_by_id.items()
            if not ete.is_root(node)
        ]),
        dtype=np.int64,
    )
    branch_id_to_index = {int(bid): i for i, bid in enumerate(branch_ids.tolist())}
    leaf_count_by_bid = dict()
    for node in tree_obj.traverse('postorder'):
        bid = int(ete.get_prop(node, "numerical_label"))
        if ete.is_leaf(node):
            leaf_count_by_bid[bid] = 1
            continue
        count = 0
        for child in ete.get_children(node):
            child_bid = int(ete.get_prop(child, "numerical_label"))
            count += int(leaf_count_by_bid.get(child_bid, 0))
        leaf_count_by_bid[bid] = int(count)
    size_array = np.array(
        [leaf_count_by_bid.get(int(bid), 1) for bid in branch_ids.tolist()],
        dtype=np.int64,
    )
    is_fg_stem = np.zeros(shape=(branch_ids.shape[0],), dtype=bool)
    descendant_branch_ids_by_bid = dict()
    descendant_indices_by_bid = dict()
    leaf_names_by_bid = dict()
    for i, bid in enumerate(branch_ids.tolist()):
        node = node_by_id[int(bid)]
        is_fg = ete.get_prop(node, 'is_fg_' + trait_name, False)
        is_parent_fg = ete.get_prop(node.up, 'is_fg_' + trait_name, False)
        is_fg_stem[i] = bool(is_fg and (not is_parent_fg))
        descendant_branch_ids = np.array(
            [
                int(ete.get_prop(d, "numerical_label"))
                for d in node.traverse()
                if not ete.is_root(d)
            ],
            dtype=np.int64,
        )
        descendant_branch_ids_by_bid[int(bid)] = descendant_branch_ids
        descendant_indices_by_bid[int(bid)] = np.array(
            [branch_id_to_index[int(d)] for d in descendant_branch_ids.tolist() if int(d) in branch_id_to_index],
            dtype=np.int64,
        )
        leaf_names_by_bid[int(bid)] = tuple(ete.get_leaf_names(node))
    descendant_branch_ids_by_index = [descendant_branch_ids_by_bid[int(bid)] for bid in branch_ids.tolist()]
    descendant_indices_by_index = [descendant_indices_by_bid[int(bid)] for bid in branch_ids.tolist()]
    leaf_names_by_index = [leaf_names_by_bid[int(bid)] for bid in branch_ids.tolist()]
    df_clade_size_template = pd.DataFrame(index=branch_ids, columns=['branch_id', 'size', 'is_fg_stem_' + trait_name])
    df_clade_size_template.loc[:, 'branch_id'] = branch_ids
    df_clade_size_template.loc[:, 'size'] = size_array
    df_clade_size_template.loc[:, 'is_fg_stem_' + trait_name] = is_fg_stem
    return {
        'branch_ids': branch_ids,
        'branch_id_to_index': branch_id_to_index,
        'size': size_array,
        'is_fg_stem': is_fg_stem,
        'descendant_branch_ids_by_bid': descendant_branch_ids_by_bid,
        'descendant_indices_by_bid': descendant_indices_by_bid,
        'descendant_branch_ids_by_index': descendant_branch_ids_by_index,
        'descendant_indices_by_index': descendant_indices_by_index,
        'leaf_names_by_index': leaf_names_by_index,
        'df_clade_size_template': df_clade_size_template,
    }


def _get_trait_clade_permutation_cache(g, trait_name):
    trait_cache_dict = g.get('_clade_permutation_trait_cache', None)
    if trait_cache_dict is None:
        trait_cache_dict = dict()
        g['_clade_permutation_trait_cache'] = trait_cache_dict
    if trait_name not in trait_cache_dict:
        trait_cache_dict[trait_name] = _build_trait_clade_permutation_cache(g=g, trait_name=trait_name)
    return trait_cache_dict[trait_name]


def _normalize_branch_ids(branch_ids):
    if branch_ids is None:
        return np.array([], dtype=np.int64)
    values = np.asarray(branch_ids)
    values = np.atleast_1d(values).reshape(-1)
    if values.size == 0:
        return np.array([], dtype=np.int64)
    normalized = []
    for value in values.tolist():
        if isinstance(value, bool):
            raise ValueError('Branch IDs should be integer-like.')
        if isinstance(value, (int, np.integer)):
            normalized.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            if (not np.isfinite(value)) or (not float(value).is_integer()):
                raise ValueError('Branch IDs should be integer-like.')
            normalized.append(int(value))
            continue
        value_txt = str(value).strip()
        if (value_txt == '') or (not bool(re.fullmatch(r'[+-]?[0-9]+(?:\.0+)?', value_txt))):
            raise ValueError('Branch IDs should be integer-like.')
        normalized.append(int(float(value_txt)))
    return np.array(normalized, dtype=np.int64)


def _count_branch_memberships(cb, bid_cols, ids):
    if len(bid_cols) == 0:
        return np.zeros(shape=(cb.shape[0],), dtype=np.int64)
    id_list = _normalize_branch_ids(ids).tolist()
    if len(id_list) == 0:
        return np.zeros(shape=(cb.shape[0],), dtype=np.int64)
    bid_matrix = cb.loc[:, bid_cols].to_numpy(copy=False)
    return np.isin(bid_matrix, id_list).sum(axis=1).astype(np.int64)


def _mark_dependent_foreground_rows(cb, bid_cols, trait_name, dependent_id_combinations):
    if len(bid_cols) == 0:
        return cb
    dep = np.asarray(dependent_id_combinations, dtype=np.int64)
    if dep.size == 0:
        return cb
    if dep.size % len(bid_cols) != 0:
        raise ValueError('dependent_id_combinations had an unexpected shape.')
    col_is_fg = 'is_fg_' + trait_name
    # Branch-combination semantics are order-invariant; compare sorted row tuples.
    dep = dep.reshape(-1, len(bid_cols))
    dep_sorted = np.sort(dep, axis=1)
    dep_sorted = np.unique(dep_sorted, axis=0)
    bid_matrix = cb.loc[:, bid_cols].to_numpy(copy=False)
    if bid_matrix.shape[0] == 0:
        return cb
    bid_sorted = np.sort(np.asarray(bid_matrix, dtype=np.int64), axis=1)
    dep_key = np.ascontiguousarray(dep_sorted).view(np.dtype((np.void, dep_sorted.dtype.itemsize * dep_sorted.shape[1]))).reshape(-1)
    bid_key = np.ascontiguousarray(bid_sorted).view(np.dtype((np.void, bid_sorted.dtype.itemsize * bid_sorted.shape[1]))).reshape(-1)
    is_dep = np.isin(bid_key, dep_key)
    cb.loc[is_dep, col_is_fg] = 'N'
    return cb


def _assign_trait_labels(cb, trait_name, arity):
    col_num_fg = 'branch_num_fg_' + trait_name
    col_num_mg = 'branch_num_mg_' + trait_name
    col_is_fg = 'is_fg_' + trait_name
    col_is_mg = 'is_mg_' + trait_name
    col_is_mf = 'is_mf_' + trait_name
    cb.loc[:, col_is_fg] = 'N'
    cb.loc[cb.loc[:, col_num_fg] == arity, col_is_fg] = 'Y'
    cb.loc[:, col_is_mg] = 'N'
    cb.loc[cb.loc[:, col_num_mg] == arity, col_is_mg] = 'Y'
    cb.loc[:, col_is_mf] = 'N'
    is_mf = (cb.loc[:, col_num_fg] > 0) & (cb.loc[:, col_num_mg] > 0)
    is_mf = is_mf & ((cb.loc[:, col_num_fg] + cb.loc[:, col_num_mg]) == arity)
    cb.loc[is_mf, col_is_mf] = 'Y'
    return cb


def _set_target_label_column(df, column_name, positive_index, positive='Y', negative='N'):
    df.loc[:, column_name] = negative
    normalized_index = _normalize_branch_ids(positive_index)
    if normalized_index.shape[0] == 0:
        return df
    if 'branch_id' in df.columns:
        is_target = df.loc[:, 'branch_id'].isin(normalized_index.tolist())
        df.loc[is_target, column_name] = positive
        return df
    valid_labels = [int(v) for v in normalized_index.tolist() if int(v) in df.index]
    if len(valid_labels) > 0:
        df.loc[valid_labels, column_name] = positive
    return df


def _calculate_fg_enrichment(num_enough, num_fg, num_fg_enough, num_all):
    if (num_enough == 0) or (num_fg == 0):
        return 0, 0
    percent_fg_enough = num_fg_enough / num_enough * 100
    enrichment_factor = (num_fg_enough / num_enough) / (num_fg / num_all)
    return percent_fg_enough, enrichment_factor


def combinations_count(n, r):
    # https://github.com/nkmk/python-snippets/blob/05a53ae96736577906a8805b38bce6cc210fe11f/notebook/combinations_count.py#L1-L14
    from operator import mul
    from functools import reduce
    n = int(n)
    r = int(r)
    if n < 0:
        raise ValueError('n should be >= 0.')
    if (r < 0) or (r > n):
        return 0
    r = min(r, n - r)
    numer = reduce(mul, range(n, n - r, -1), 1)
    denom = reduce(mul, range(1, r + 1), 1)
    return numer // denom

def get_df_clade_size(g, trait_name):
    trait_cache = _get_trait_clade_permutation_cache(g=g, trait_name=trait_name)
    return trait_cache['df_clade_size_template'].copy(deep=True)


def _build_clade_permutation_bins_from_arrays(size_array, is_fg_stem, min_clade_bin_count, sample_original_foreground):
    size_array = np.asarray(size_array, dtype=np.int64).reshape(-1)
    is_fg_stem = np.asarray(is_fg_stem, dtype=bool).reshape(-1)
    if size_array.shape[0] == 0:
        return np.array([0, 0], dtype=np.int64), []
    size_min = int(size_array.min())
    size_max = int(size_array.max())
    sizes = np.unique(size_array)[::-1]  # To start counting from rarer (larger) clades
    bins = [int(size_max + 1)]
    count = 0
    counts = []
    for size in sizes:
        is_size = (size_array == int(size))
        if not sample_original_foreground:
            is_size &= (~is_fg_stem)
        count += int(is_size.sum())
        if count >= int(min_clade_bin_count):
            bins.append(int(size))
            counts.append(int(count))
            count = 0
    if len(bins) < 2:
        bins = [int(size_min), int(size_max)]
    return np.array(bins, dtype=np.int64), counts


def _build_clade_permutation_bins(df_clade_size, trait_name, min_clade_bin_count, sample_original_foreground):
    size_array = df_clade_size.loc[:, 'size'].to_numpy(dtype=np.int64, copy=False)
    if size_array.shape[0] == 0:
        return np.array([0, 0], dtype=np.int64), [], size_array
    is_fg_stem = df_clade_size.loc[:, 'is_fg_stem_' + trait_name].to_numpy(dtype=bool, copy=False)
    bins, counts = _build_clade_permutation_bins_from_arrays(
        size_array=size_array,
        is_fg_stem=is_fg_stem,
        min_clade_bin_count=min_clade_bin_count,
        sample_original_foreground=sample_original_foreground,
    )
    return bins, counts, size_array


def _randomize_foreground_flags(before_randomization, sample_original_foreground):
    if sample_original_foreground:
        return np.random.permutation(before_randomization)
    ind_fg = np.where(before_randomization == True)[0]
    ind_nonfg = np.where(before_randomization == False)[0]
    if ind_nonfg.shape[0] < ind_fg.shape[0]:
        txt = 'Not enough non-foreground clades for permutation (required {}, available {}).'
        raise ValueError(txt.format(ind_fg.shape[0], ind_nonfg.shape[0]))
    ind_rfg = np.random.choice(ind_nonfg, ind_fg.shape[0], replace=False)
    after_randomization = np.zeros_like(before_randomization, dtype=bool)
    after_randomization[ind_rfg] = True
    return after_randomization


def _block_randomized_foreground_descendants(df_clade_size, is_bin, node_by_id):
    is_blocked = df_clade_size.loc[:, 'is_blocked'].values
    is_new_fg = is_bin & ~is_blocked & (df_clade_size.loc[:, 'is_fg_stem_randomized'] == True)
    new_fg_bids = df_clade_size.loc[is_new_fg, 'branch_id'].values
    for new_fg_bid in new_fg_bids:
        node = node_by_id.get(int(new_fg_bid), None)
        if node is None:
            continue
        des_bids = [ete.get_prop(d, "numerical_label") for d in node.traverse()]
        df_clade_size.loc[des_bids, 'is_blocked'] = True
    return df_clade_size


def foreground_clade_randomization(df_clade_size, g, trait_name, sample_original_foreground=False):
    bins, counts, size_array = _build_clade_permutation_bins(
        df_clade_size=df_clade_size,
        trait_name=trait_name,
        min_clade_bin_count=g['min_clade_bin_count'],
        sample_original_foreground=sample_original_foreground,
    )
    txt = 'Number of clade permutation bins = {:,} (bin limits = {}, counts = {})'
    print(txt.format(bins.shape[0]-1, ', '.join([ str(s) for s in bins ]), ', '.join([ str(s) for s in counts ])))
    bins = bins[::-1]
    bin_array = np.digitize(size_array, bins, right=False).astype(np.int64, copy=False)
    is_fg = df_clade_size.loc[:, 'is_fg_stem_' + trait_name].to_numpy(dtype=bool, copy=False)
    fg_bins = np.unique(bin_array[is_fg])
    is_fg_stem_randomized = is_fg.copy()
    is_blocked = np.zeros(shape=(df_clade_size.shape[0],), dtype=bool)
    branch_ids = df_clade_size.loc[:, 'branch_id'].to_numpy(dtype=np.int64, copy=False)
    trait_cache = _get_trait_clade_permutation_cache(g=g, trait_name=trait_name)
    branch_id_to_index = trait_cache['branch_id_to_index']
    descendant_indices_by_bid = trait_cache.get('descendant_indices_by_bid', None)
    node_by_id = None
    if descendant_indices_by_bid is None:
        node_by_id = _get_node_by_branch_id(g)
    for bin_no in fg_bins:
        is_bin = (bin_array == int(bin_no))
        is_eligible = is_bin & (~is_blocked)
        before_randomization = is_fg_stem_randomized[is_eligible]
        after_randomization = _randomize_foreground_flags(
            before_randomization=before_randomization,
            sample_original_foreground=sample_original_foreground,
        )
        is_fg_stem_randomized[is_eligible] = after_randomization
        is_new_fg = is_eligible & is_fg_stem_randomized
        new_fg_bids = branch_ids[is_new_fg]
        for new_fg_bid in new_fg_bids.tolist():
            descendant_indices = None
            if descendant_indices_by_bid is not None:
                descendant_indices = descendant_indices_by_bid.get(int(new_fg_bid), None)
            else:
                node = node_by_id.get(int(new_fg_bid), None)
                if node is None:
                    continue
                descendant_bids = [
                    int(ete.get_prop(d, "numerical_label"))
                    for d in node.traverse()
                    if not ete.is_root(d)
                ]
                descendant_indices = np.array(
                    [branch_id_to_index[int(d)] for d in descendant_bids if int(d) in branch_id_to_index],
                    dtype=np.int64,
                )
            if (descendant_indices is not None) and (descendant_indices.shape[0] > 0):
                is_blocked[descendant_indices] = True
    df_clade_size.loc[:, 'bin'] = bin_array
    df_clade_size.loc[:, 'is_fg_stem_randomized'] = is_fg_stem_randomized
    df_clade_size.loc[:, 'is_blocked'] = is_blocked
    return df_clade_size


def _build_clade_permutation_randomization_plan(g, trait_name, sample_original_foreground):
    trait_cache = _get_trait_clade_permutation_cache(g=g, trait_name=trait_name)
    bins, counts = _build_clade_permutation_bins_from_arrays(
        size_array=trait_cache['size'],
        is_fg_stem=trait_cache['is_fg_stem'],
        min_clade_bin_count=g['min_clade_bin_count'],
        sample_original_foreground=sample_original_foreground,
    )
    txt = 'Number of clade permutation bins = {:,} (bin limits = {}, counts = {})'
    print(txt.format(bins.shape[0] - 1, ', '.join([str(s) for s in bins]), ', '.join([str(s) for s in counts])))
    bins_for_digitize = bins[::-1]
    bin_array = np.digitize(trait_cache['size'], bins_for_digitize, right=False).astype(np.int64, copy=False)
    unique_bins = np.unique(bin_array)
    fg_bins = np.unique(bin_array[trait_cache['is_fg_stem']])
    bin_indices = dict()
    for bin_no in unique_bins.tolist():
        bin_indices[int(bin_no)] = np.where(bin_array == int(bin_no))[0].astype(np.int64, copy=False)
    num_possible_permutation_combination = 1
    for bin_no in unique_bins.tolist():
        indices = bin_indices[int(bin_no)]
        num_bin_fg = int(trait_cache['is_fg_stem'][indices].sum())
        if num_bin_fg == 0:
            continue
        if sample_original_foreground:
            num_bin_choice = int(indices.shape[0])
        else:
            num_bin_choice = int((~trait_cache['is_fg_stem'][indices]).sum())
        num_possible_permutation_combination *= combinations_count(n=num_bin_choice, r=num_bin_fg)
    return {
        'bin_array': bin_array,
        'fg_bins': fg_bins.astype(np.int64, copy=False),
        'bin_indices': bin_indices,
        'num_possible_permutation_combination': int(num_possible_permutation_combination),
    }


def _get_clade_permutation_randomization_plan(g, trait_name, sample_original_foreground):
    cache = g.get('_clade_permutation_randomization_plan_cache', None)
    if cache is None:
        cache = dict()
        g['_clade_permutation_randomization_plan_cache'] = cache
    cache_key = (str(trait_name), bool(sample_original_foreground), int(g.get('min_clade_bin_count', 0)))
    if cache_key not in cache:
        cache[cache_key] = _build_clade_permutation_randomization_plan(
            g=g,
            trait_name=trait_name,
            sample_original_foreground=sample_original_foreground,
        )
    return cache[cache_key]


def _randomize_foreground_stem_flags_from_plan(trait_cache, randomization_plan, sample_original_foreground):
    is_fg_stem_randomized = trait_cache['is_fg_stem'].copy()
    is_blocked = np.zeros(shape=(is_fg_stem_randomized.shape[0],), dtype=bool)
    descendant_indices_by_index = trait_cache['descendant_indices_by_index']
    for bin_no in randomization_plan['fg_bins'].tolist():
        indices = randomization_plan['bin_indices'].get(int(bin_no), None)
        if (indices is None) or (indices.shape[0] == 0):
            continue
        eligible_indices = indices[~is_blocked[indices]]
        if eligible_indices.shape[0] == 0:
            continue
        before_randomization = is_fg_stem_randomized[eligible_indices]
        after_randomization = _randomize_foreground_flags(
            before_randomization=before_randomization,
            sample_original_foreground=sample_original_foreground,
        )
        is_fg_stem_randomized[eligible_indices] = after_randomization
        new_fg_indices = eligible_indices[after_randomization]
        for idx in new_fg_indices.tolist():
            descendant_indices = descendant_indices_by_index[int(idx)]
            if descendant_indices.shape[0] > 0:
                is_blocked[descendant_indices] = True
    return is_fg_stem_randomized


def _resolve_randomized_target_ids(trait_cache, randomized_stem_indices, fg_stem_only):
    if randomized_stem_indices.shape[0] == 0:
        return np.array([], dtype=np.int64)
    branch_ids = trait_cache['branch_ids']
    randomized_stem_bids = branch_ids[randomized_stem_indices]
    if fg_stem_only:
        return randomized_stem_bids.astype(np.int64, copy=True)
    descendant_branch_ids_by_index = trait_cache['descendant_branch_ids_by_index']
    concatenated = [descendant_branch_ids_by_index[int(i)] for i in randomized_stem_indices.tolist()]
    if len(concatenated) == 0:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(concatenated).astype(np.int64, copy=False))


def _resolve_randomized_fg_leaf_names(trait_cache, randomized_stem_indices):
    if randomized_stem_indices.shape[0] == 0:
        return []
    leaf_names_by_index = trait_cache['leaf_names_by_index']
    leaf_name_set = set()
    for idx in randomized_stem_indices.tolist():
        leaf_name_set.update(leaf_names_by_index[int(idx)])
    return sorted(leaf_name_set)


def _infer_trait_name_from_clade_size(df_clade_size):
    for col in df_clade_size.columns:
        col_text = str(col)
        if col_text.startswith('is_fg_stem_') and (col_text != 'is_fg_stem_randomized'):
            return col_text.replace('is_fg_stem_', '', 1)
    return None


def get_new_foreground_ids(df_clade_size, g, trait_name=None):
    is_new_fg = (df_clade_size.loc[:,'is_fg_stem_randomized']==True)
    fg_stem_bids = df_clade_size.loc[is_new_fg,'branch_id'].to_numpy(dtype=np.int64, copy=False)
    if fg_stem_bids.shape[0] == 0:
        return np.array([], dtype=np.int64)
    new_fg_ids = list()
    if (g['fg_stem_only']):
        new_fg_ids = fg_stem_bids.astype(np.int64, copy=False)
    else:
        if trait_name is None:
            trait_name = _infer_trait_name_from_clade_size(df_clade_size=df_clade_size)
        descendant_branch_ids_by_bid = None
        if trait_name is not None:
            trait_cache = _get_trait_clade_permutation_cache(g=g, trait_name=trait_name)
            descendant_branch_ids_by_bid = trait_cache.get('descendant_branch_ids_by_bid', None)
        if descendant_branch_ids_by_bid is not None:
            for fg_stem_bid in fg_stem_bids.tolist():
                descendant_branch_ids = descendant_branch_ids_by_bid.get(int(fg_stem_bid), None)
                if descendant_branch_ids is None:
                    continue
                new_fg_ids.append(descendant_branch_ids)
            if len(new_fg_ids) > 0:
                new_fg_ids = np.concatenate(new_fg_ids)
            else:
                new_fg_ids = np.array([], dtype=np.int64)
        else:
            node_by_id = _get_node_by_branch_id(g)
            for fg_stem_bid in fg_stem_bids.tolist():
                node = node_by_id.get(int(fg_stem_bid), None)
                if node is None:
                    continue
                new_lineage_fg_ids = [ete.get_prop(n, "numerical_label") for n in node.traverse()]
                new_fg_ids += new_lineage_fg_ids
    new_fg_ids = np.unique(np.array(new_fg_ids, dtype=np.int64))
    return new_fg_ids

def get_lineage_color_list():
    lineage_colors = [
        'red',
        'blue',
        'darkorange',
        'brown',
        'mediumseagreen',
        'purple',
        'hotpink',
        'olive',
        'darkturquoise',
        'green',
        'darkorchid',
    ]
    return lineage_colors

def get_fg_leaf_names(lineages, trait_name, g):
    fg_leaf_names = []
    all_leaf_names = ete.get_leaf_names(g['tree'])
    for i in np.arange(len(lineages)):
        fg_leaf_names.append([])
        is_lineage = (g['fg_df'].loc[:, trait_name] == lineages[i])
        lineage_regex_names = g['fg_df'].loc[is_lineage, :].loc[:, 'name'].unique().tolist()
        iter = itertools.product(all_leaf_names, lineage_regex_names)
        lineage_leaf_names = [ln for ln, lr in iter if re.match('^' + lr + '$', ln)]
        for lln in lineage_leaf_names:
            match_leaves = [ln for ln in all_leaf_names if lln == ln]
            if len(match_leaves) == 1:
                fg_leaf_names[i].append(match_leaves[0])
            else:
                print('The foreground leaf name cannot be identified:', lln, match_leaves)
    return fg_leaf_names

def annotate_lineage_foreground(lineages, trait_name, g):
    tree = g['tree']
    for i in np.arange(len(lineages)):
        fg_leaf_name_set = set(g['fg_leaf_names'][trait_name][i])
        for node in tree.traverse():
            node_leaf_name_set = set(ete.get_leaf_names(node))
            if len(node_leaf_name_set.difference(fg_leaf_name_set)) == 0:
                ete.add_features(node, **{'is_lineage_fg_'+trait_name+'_'+str(i+1): True})
            else:
                ete.add_features(node, **{'is_lineage_fg_'+trait_name+'_'+str(i+1): False})
    return tree

def _get_lineage_target_ids(lineage_index, trait_name, g):
    fg_leaf_name_set = set(g['fg_leaf_names'][trait_name][lineage_index])
    lineage_fg_ids = list()
    root_id = ete.get_prop(ete.get_tree_root(g['tree']), "numerical_label")
    lineage_flag_key = 'is_lineage_fg_' + trait_name + '_' + str(lineage_index + 1)
    for node in g['tree'].traverse():
        node_leaf_name_set = set(ete.get_leaf_names(node))
        if len(node_leaf_name_set.difference(fg_leaf_name_set)) != 0:
            continue
        if ete.is_root(node) or (not g['fg_stem_only']):
            lineage_fg_ids.append(ete.get_prop(node, "numerical_label"))
        else:
            is_lineage_fg = ete.get_prop(node, lineage_flag_key, False)
            is_parent_lineage_fg = ete.get_prop(node.up, lineage_flag_key, False)
            if (is_lineage_fg == True) & (is_parent_lineage_fg == False):
                lineage_fg_ids.append(ete.get_prop(node, "numerical_label"))
    dif = 1
    while dif:
        num_id = len(lineage_fg_ids)
        lineage_fg_id_set = set(lineage_fg_ids)
        for node in g['tree'].traverse():
            child_ids = [ete.get_prop(child, "numerical_label") for child in ete.get_children(node)]
            if all([id in lineage_fg_id_set for id in child_ids]) & (len(child_ids) != 0):
                node_id = ete.get_prop(node, "numerical_label")
                if node_id not in lineage_fg_id_set:
                    lineage_fg_ids.append(node_id)
        dif = len(lineage_fg_ids) - num_id
    lineage_fg_ids = np.unique(np.array(lineage_fg_ids, dtype=np.int64))
    lineage_fg_ids = lineage_fg_ids[lineage_fg_ids != root_id]
    lineage_fg_ids.sort()
    return lineage_fg_ids

def get_target_ids(lineages, trait_name, g):
    target_ids = np.zeros(shape=(0,), dtype=np.int64)
    for i in np.arange(len(lineages)):
        lineage_fg_ids = _get_lineage_target_ids(lineage_index=i, trait_name=trait_name, g=g)
        target_ids = np.concatenate([target_ids, lineage_fg_ids])
    target_ids = np.unique(target_ids.astype(np.int64))
    target_ids.sort()
    return target_ids

def annotate_foreground(lineages, trait_name, g):
    for node in g['tree'].traverse(): # Initialize
        ete.add_features(node, **{'is_fg_'+trait_name: False})
        ete.add_features(node, **{'foreground_lineage_id_' + trait_name: 0})
        ete.add_features(node, **{'color_'+trait_name: 'black'})
        ete.add_features(node, **{'labelcolor_' + trait_name: 'black'})
    lineage_colors = get_lineage_color_list()
    for i in np.arange(len(lineages)):
        fg_leaf_name_set = set(g['fg_leaf_names'][trait_name][i])
        lineage_color = lineage_colors[i % len(lineage_colors)]
        lineage_prop = 'is_lineage_fg_' + trait_name + '_' + str(i + 1)
        lineage_target_ids = set(_get_lineage_target_ids(lineage_index=i, trait_name=trait_name, g=g).tolist())
        for node in g['tree'].traverse():
            if g['fg_stem_only']:
                if ete.get_prop(node, "numerical_label") in lineage_target_ids:
                    ete.add_features(node, **{'is_fg_'+trait_name: True})
                    ete.add_features(node, **{'foreground_lineage_id_' + trait_name: int(i + 1)})
                    ete.add_features(node, **{'color_'+trait_name: lineage_color})
                    ete.add_features(node, **{'labelcolor_'+trait_name: lineage_color})
                if node.name in fg_leaf_name_set:
                    ete.add_features(node, **{'labelcolor_' + trait_name: lineage_color})
            else:
                is_lineage_fg = ete.get_prop(node, lineage_prop, False)
                if is_lineage_fg == True:
                    ete.add_features(node, **{'is_fg_' + trait_name: True})
                    ete.add_features(node, **{'foreground_lineage_id_' + trait_name: int(i + 1)})
                    ete.add_features(node, **{'color_' + trait_name: lineage_color})
                    ete.add_features(node, **{'labelcolor_' + trait_name: lineage_color})
    return g['tree']

def get_foreground_ids(g, write=True):
    g['fg_leaf_names'] = dict()
    g['target_ids'] = dict()
    g['fg_ids'] = dict()
    for trait_name in _get_trait_names(g):
        lineages = g['fg_df'].loc[:,trait_name].unique()
        lineages = lineages[lineages!=0]
        g['fg_leaf_names'][trait_name] = get_fg_leaf_names(lineages, trait_name, g)
        g['tree'] = annotate_lineage_foreground(lineages, trait_name, g)
        g['target_ids'][trait_name] = get_target_ids(lineages, trait_name, g)
        g['tree'] = annotate_foreground(lineages, trait_name, g)
        g['fg_ids'][trait_name] = copy.deepcopy(g['target_ids'][trait_name]) # marginal_ids may be added to target_id but fg_id won't be changed.
        if write:
            file_name = 'csubst_target_branch_'+trait_name+'.txt'
            file_name = file_name.replace('_PLACEHOLDER', '')
            with open(file_name, 'w') as f:
                for x in g['target_ids'][trait_name]:
                    f.write(str(x)+'\n')
    return g

def read_foreground_file(g):
    if g['fg_format'] == 1:
        fg_df = pd.read_csv(g['foreground'], sep='\t', comment='#', skip_blank_lines=True, header=None)
        if fg_df.shape[1]!=2:
            txt = 'With --fg_format 1, --foreground file should be a tab-separated two-column table without header. '
            txt += 'First column = lineage IDs; Second column = Regex-compatible sequence names'
            raise ValueError(txt)
        fg_df = fg_df.iloc[:,[1,0]]
        fg_df.columns = ['name','PLACEHOLDER']
    elif g['fg_format'] == 2:
        fg_df = pd.read_csv(g['foreground'], sep='\t', comment='#', skip_blank_lines=True, header=0)
        if fg_df.shape[1]<2:
            txt = 'With --fg_format 2, --foreground file should be a tab-separated table with a header line and 2 or more columns. '
            txt += 'Header names should be "name", "TRAIT1", "TRAIT2", ..., where any trait names are allowed. '
            txt += 'First column = Regex-compatible sequence names; Second column and after = lineage IDs (0 = background)'
            raise ValueError(txt)
        fg_df.columns = ['name'] + fg_df.columns[1:len(fg_df.columns)].tolist()
        txt = 'Trait names in --foreground file: {}'.format(', '.join(fg_df.columns[1:len(fg_df.columns)].tolist()))
        print(txt, flush=True)
    return fg_df

def dummy_foreground_annotation(tree, trait_name):
    for node in tree.traverse():
        ete.add_features(node, **{'is_lineage_fg_' + trait_name + '_1': False})
        ete.add_features(node, **{'is_fg_' + trait_name: False})
        ete.add_features(node, **{'foreground_lineage_id_' + trait_name: 0})
        ete.add_features(node, **{'color_' + trait_name: 'black'})
        ete.add_features(node, **{'labelcolor_' + trait_name: 'black'})
    return tree

def get_foreground_branch(g, simulate=False):
    if g['foreground'] is None:
        trait_name = 'PLACEHOLDER'
        g['tree'] = dummy_foreground_annotation(tree=g['tree'], trait_name=trait_name)
        g['fg_df'] = pd.DataFrame(columns=['name', trait_name])
        g['fg_leaf_names'] = {trait_name: []}
        g['fg_ids'] = {trait_name: np.zeros(shape=(0,), dtype=np.int64)}
        g['target_ids'] = {trait_name: np.zeros(shape=(0,), dtype=np.int64)}
    else:
        g['fg_df'] = read_foreground_file(g)
        if simulate:
            if g['fg_format'] == 2:
                first_trait_name = g['fg_df'].columns[1]
                txt = 'With --fg_format 2, only the first trait column ({}) will be used for simulated molecular evolution.\n'
                sys.stderr.write(txt.format(first_trait_name))
            if g['fg_df'].shape[1] > 2:
                g['fg_df'] = g['fg_df'].iloc[:,[0,1]]
            g['fg_df'].columns = ['name','PLACEHOLDER']
        g = get_foreground_ids(g=g, write=True)
    g = _invalidate_clade_permutation_cache(g)
    g['_foreground_annotation_ready'] = True
    return g

def print_num_possible_permuted_combinations(df_clade_size, trait_name, sample_original_foreground):
    num_possible_permutation_combination = 1
    is_fg_stem = df_clade_size.loc[:, 'is_fg_stem_'+trait_name].to_numpy(dtype=bool, copy=False)
    bin_array = df_clade_size.loc[:, 'bin'].to_numpy(dtype=np.int64, copy=False)
    for bin_no in np.unique(bin_array):
        is_bin = (bin_array == int(bin_no))
        num_bin_fg = int((is_bin & is_fg_stem).sum())
        if num_bin_fg == 0:
            continue
        if sample_original_foreground:
            num_bin_choice = int(is_bin.sum())
        else:
            num_bin_choice = int((is_bin & ~is_fg_stem).sum())
        num_possible_permutation_combination_bin = combinations_count(n=num_bin_choice, r=num_bin_fg)
        num_possible_permutation_combination *= num_possible_permutation_combination_bin
    return int(num_possible_permutation_combination)

def randomize_foreground_branch(g, trait_name, sample_original_foreground=False):
    if 'r_target_ids' not in g:
        g['r_target_ids'] = dict()
    if 'r_fg_ids' not in g:
        g['r_fg_ids'] = dict()
    if 'r_fg_leaf_names' not in g:
        g['r_fg_leaf_names'] = dict()
    trait_cache = _get_trait_clade_permutation_cache(g=g, trait_name=trait_name)
    randomization_plan = _get_clade_permutation_randomization_plan(
        g=g,
        trait_name=trait_name,
        sample_original_foreground=sample_original_foreground,
    )
    is_fg_stem_randomized = _randomize_foreground_stem_flags_from_plan(
        trait_cache=trait_cache,
        randomization_plan=randomization_plan,
        sample_original_foreground=sample_original_foreground,
    )
    randomized_stem_indices = np.where(is_fg_stem_randomized)[0].astype(np.int64, copy=False)
    count_cache = g.get('_clade_permutation_combination_count_cache', None)
    if count_cache is None:
        count_cache = dict()
        g['_clade_permutation_combination_count_cache'] = count_cache
    count_cache_key = (str(trait_name), bool(sample_original_foreground))
    if count_cache_key not in count_cache:
        count_cache[count_cache_key] = int(randomization_plan['num_possible_permutation_combination'])
        txt = 'Number of possible clade-permuted combinations without considering branch dependency = {:,}'
        print(txt.format(int(count_cache[count_cache_key])))
    g['r_target_ids'][trait_name] = _resolve_randomized_target_ids(
        trait_cache=trait_cache,
        randomized_stem_indices=randomized_stem_indices,
        fg_stem_only=bool(g.get('fg_stem_only', False)),
    )
    g['r_fg_ids'][trait_name] = g['r_target_ids'][trait_name].copy()
    g['r_fg_leaf_names'][trait_name] = _resolve_randomized_fg_leaf_names(
        trait_cache=trait_cache,
        randomized_stem_indices=randomized_stem_indices,
    )
    return g

def get_marginal_branch(g):
    g['mg_ids'] = dict()
    for trait_name in _get_trait_names(g):
        g['mg_ids'][trait_name] = list()
        target_ids = _normalize_branch_ids(g['target_ids'][trait_name])
        for node in g['tree'].traverse():
            if ete.is_root(node):
                continue
            is_fg = ete.get_prop(node, 'is_fg_' + trait_name, False)
            if (is_fg==False):
                continue
            if (g['mg_parent']):
                is_parent_fg = ete.get_prop(node.up, 'is_fg_' + trait_name, False)
                if (is_parent_fg==False):
                    g['mg_ids'][trait_name].append(ete.get_prop(node.up, "numerical_label"))
            if (g['mg_sister']):
                sisters = ete.get_sisters(node)
                for sister in sisters:
                    if (g['mg_sister_stem_only']):
                        is_sister_fg = ete.get_prop(sister, 'is_fg_' + trait_name, False)
                        if is_sister_fg==False:
                            g['mg_ids'][trait_name].append(ete.get_prop(sister, "numerical_label"))
                    else:
                        for sister_des in sister.traverse():
                            is_sister_des_fg = ete.get_prop(sister_des, 'is_fg_' + trait_name, False)
                            if is_sister_des_fg==False:
                                g['mg_ids'][trait_name].append(ete.get_prop(sister_des, "numerical_label"))
        concat_ids = list(set(g['mg_ids'][trait_name]) - set(target_ids.tolist()))
        g['mg_ids'][trait_name] = np.array(concat_ids, dtype=np.int64)
        g['target_ids'][trait_name] = np.concatenate([target_ids, g['mg_ids'][trait_name]])
        for node in g['tree'].traverse():
            if ete.get_prop(node, "numerical_label") in g['mg_ids'][trait_name]:
                ete.add_features(node, **{'is_marginal_'+trait_name: True})
            else:
                ete.add_features(node, **{'is_marginal_'+trait_name: False})
        file_name = 'csubst_marginal_branch_' + trait_name + '.txt'
        file_name = file_name.replace('_PLACEHOLDER', '')
        if len(g['mg_ids'][trait_name]) > 0:
            with open(file_name, 'w') as f:
                for x in g['mg_ids'][trait_name]:
                    f.write(str(x)+'\n')
    return g

def get_foreground_branch_num(cb, g):
    start_time = time.time()
    bid_cols = cb.columns[cb.columns.str.startswith('branch_id_')]
    arity = len(bid_cols)
    trait_names = _get_trait_names(g)
    for trait_name in trait_names:
        col_num_fg = 'branch_num_fg_' + trait_name
        col_num_mg = 'branch_num_mg_' + trait_name
        col_num_fg_stem = 'branch_num_fg_stem_' + trait_name
        col_is_fg = 'is_fg_' + trait_name
        cb.loc[:, col_num_fg] = _count_branch_memberships(cb=cb, bid_cols=bid_cols, ids=g['fg_ids'][trait_name])
        cb.loc[:, col_num_mg] = _count_branch_memberships(cb=cb, bid_cols=bid_cols, ids=g['mg_ids'][trait_name])
        cb = _assign_trait_labels(cb=cb, trait_name=trait_name, arity=arity)
        cb = _mark_dependent_foreground_rows(
            cb=cb,
            bid_cols=bid_cols,
            trait_name=trait_name,
            dependent_id_combinations=g['fg_dependent_id_combinations'][trait_name],
        )
        df_clade_size = get_df_clade_size(g, trait_name)
        fg_stem_bids = df_clade_size.loc[df_clade_size.loc[:,'is_fg_stem_'+trait_name],'branch_id'].values
        cb.loc[:, col_num_fg_stem] = _count_branch_memberships(cb=cb, bid_cols=bid_cols, ids=fg_stem_bids)
        is_fg = (cb[col_is_fg] == 'Y')
        is_enough_stat = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
        num_enough = is_enough_stat.sum()
        num_fg = is_fg.sum()
        num_fg_enough = (is_enough_stat&is_fg).sum()
        num_all = cb.shape[0]
        percent_fg_enough, enrichment_factor = _calculate_fg_enrichment(
            num_enough=num_enough,
            num_fg=num_fg,
            num_fg_enough=num_fg_enough,
            num_all=num_all,
        )
        txt = 'Arity (K) = {}: Foreground branch combinations with cutoff conditions {} for {} = {:.0f}% ({:,}/{:,}, ' \
              'total examined = {:,}, enrichment factor = {:.1f})'
        txt = txt.format(arity, g['cutoff_stat'], trait_name, percent_fg_enough, num_fg_enough, num_enough,
                         num_all, enrichment_factor)
        print(txt, flush=True)
        if not 'fg_enrichment_factor_'+trait_name in g['df_cb_stats'].columns:
            g['df_cb_stats']['fg_enrichment_factor_'+trait_name] = np.nan
        g['df_cb_stats'].at[0,'fg_enrichment_factor_'+trait_name] = enrichment_factor
    txt = 'Time elapsed for obtaining foreground branch numbers in the cb table: {:,} sec'
    print(txt.format(int(time.time() - start_time)))
    return cb, g

def annotate_b_foreground(b, g):
    trait_names = _get_trait_names(g)
    for trait_name in trait_names:
        col_fg = 'is_fg_' + trait_name
        col_mg = 'is_mg_' + trait_name
        if (g['foreground'] is None):
            b.loc[:, col_fg] = 'Y'
            b.loc[:, col_mg] = 'N'
        else:
            b = _set_target_label_column(
                df=b,
                column_name=col_fg,
                positive_index=g['fg_ids'][trait_name],
                positive='Y',
                negative='N',
            )
            b = _set_target_label_column(
                df=b,
                column_name=col_mg,
                positive_index=g['mg_ids'][trait_name],
                positive='Y',
                negative='N',
            )
    return b

def get_num_foreground_lineages(tree, trait_name):
    num_fl = 0
    prefix = 'is_lineage_fg_' + trait_name + '_'
    for node in tree.traverse():
        if hasattr(node, 'props'):
            keys = node.props.keys()
        elif hasattr(node, '__dict__'):
            keys = node.__dict__.keys()
        else:
            keys = []
        for key in keys:
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if suffix.isdigit():
                num_fl = max(num_fl, int(suffix))
    return num_fl


def _build_clade_permutation_pair_block_cache(g, trait_name):
    node_by_id = _get_node_by_branch_id(g)
    max_branch_id = max([int(bid) for bid in node_by_id.keys()]) if len(node_by_id) > 0 else 0
    multiplier = np.int64(max_branch_id + 1)
    blocked_keys = []
    dep_groups = [np.asarray(dep, dtype=np.int64).reshape(-1) for dep in g.get('dep_ids', [])]
    if int(g.get('exhaustive_until', 0)) < int(g.get('current_arity', 2)):
        dep_groups += [np.asarray(dep, dtype=np.int64).reshape(-1) for dep in g.get('fg_dep_ids', {}).get(trait_name, [])]
    for dep_group in dep_groups:
        dep_group = np.unique(_normalize_branch_ids(dep_group))
        if dep_group.shape[0] < 2:
            continue
        left, right = np.triu_indices(dep_group.shape[0], k=1)
        if left.shape[0] == 0:
            continue
        keys = dep_group[left].astype(np.int64, copy=False) * multiplier + dep_group[right].astype(np.int64, copy=False)
        blocked_keys.append(keys)
    if len(blocked_keys) > 0:
        blocked_keys = np.unique(np.concatenate(blocked_keys).astype(np.int64, copy=False))
    else:
        blocked_keys = np.array([], dtype=np.int64)
    return {
        'multiplier': multiplier,
        'blocked_keys': blocked_keys,
    }


def _get_clade_permutation_pair_block_cache(g, trait_name):
    pair_cache_dict = g.get('_clade_permutation_pair_block_cache', None)
    if pair_cache_dict is None:
        pair_cache_dict = dict()
        g['_clade_permutation_pair_block_cache'] = pair_cache_dict
    cache_key = (
        str(trait_name),
        int(g.get('current_arity', 2)),
        int(g.get('exhaustive_until', 0)),
    )
    if cache_key not in pair_cache_dict:
        pair_cache_dict[cache_key] = _build_clade_permutation_pair_block_cache(g=g, trait_name=trait_name)
    return pair_cache_dict[cache_key]


def _get_randomized_pair_combinations(g, trait_name):
    target_ids = np.unique(_normalize_branch_ids(g['r_target_ids'][trait_name]))
    if target_ids.shape[0] < 2:
        return np.zeros(shape=(0, 2), dtype=np.int64)
    left, right = np.triu_indices(target_ids.shape[0], k=1)
    pair_left = target_ids[left].astype(np.int64, copy=False)
    pair_right = target_ids[right].astype(np.int64, copy=False)
    cache = _get_clade_permutation_pair_block_cache(g=g, trait_name=trait_name)
    blocked_keys = cache['blocked_keys']
    if blocked_keys.shape[0] > 0:
        pair_keys = pair_left * cache['multiplier'] + pair_right
        is_blocked = np.isin(pair_keys, blocked_keys)
        pair_left = pair_left[~is_blocked]
        pair_right = pair_right[~is_blocked]
    if pair_left.shape[0] == 0:
        return np.zeros(shape=(0, 2), dtype=np.int64)
    return np.column_stack([pair_left, pair_right]).astype(np.int64, copy=False)


def _try_randomized_foreground_combinations(g, trait_name, current_arity, sample_original_foreground):
    g = randomize_foreground_branch(g, trait_name, sample_original_foreground=sample_original_foreground)
    if g['r_target_ids'][trait_name].shape[0] < 2:
        return g, None
    if int(current_arity) == 2:
        rid_combinations = _get_randomized_pair_combinations(g=g, trait_name=trait_name)
    else:
        g, rid_combinations = combination.get_node_combinations(
            g=g,
            target_id_dict=g['r_target_ids'],
            arity=current_arity,
            check_attr="name",
            verbose=False,
        )
    if rid_combinations.shape[0] == 0:
        return g, None
    return g, rid_combinations


def _raise_foreground_permutation_failure(num_trial, sample_original_foreground):
    if sample_original_foreground:
        txt = 'Foreground branch permutation failed {:,} times even when allowing sampling from original foreground clades.'
    else:
        txt = 'Foreground branch permutation failed {:,} times. There may not be enough numbers of "similar" clades.'
    raise ValueError(txt.format(num_trial))


def set_random_foreground_branch(g, trait_name, num_trial=100, sample_original_foreground=False):
    # Refresh tree foreground annotations only once; repeated refresh is expensive.
    if not bool(g.get('_foreground_annotation_ready', False)):
        g = get_foreground_branch(g)
    for i in np.arange(num_trial):
        g, rid_combinations = _try_randomized_foreground_combinations(
            g=g,
            trait_name=trait_name,
            current_arity=g['current_arity'],
            sample_original_foreground=sample_original_foreground,
        )
        if rid_combinations is None:
            continue
        print('Foreground branch permutation finished after {:,} trials.'.format(i + 1), flush=True)
        return g, rid_combinations
    _raise_foreground_permutation_failure(
        num_trial=num_trial,
        sample_original_foreground=sample_original_foreground,
    )


def _build_cb_target_masks(cb, trait_names, target_col_prefixes):
    is_targets = dict()
    for trait_name in trait_names:
        for target_col_prefix in target_col_prefixes:
            target_col = target_col_prefix + '_' + trait_name
            if target_col in cb.columns:
                suffix = target_col.replace('is_', '_')
                is_targets[suffix] = (cb[target_col] == 'Y')
            elif target_col_prefix == 'is_all':
                is_targets['_all'] = np.ones(shape=(cb.shape[0],), dtype=bool)
    return is_targets


def _resolve_cb_stats_for_median_and_total(cb):
    omega_cols = cb.columns[cb.columns.str.startswith('omegaC')].tolist()
    is_ON = cb.columns.str.startswith('OCNany') | cb.columns.str.startswith('OCNdif') | cb.columns.str.startswith('OCNspe')
    is_OS = cb.columns.str.startswith('OCSany') | cb.columns.str.startswith('OCSdif') | cb.columns.str.startswith('OCSspe')
    is_EN = cb.columns.str.startswith('ECNany') | cb.columns.str.startswith('ECNdif') | cb.columns.str.startswith('ECNspe')
    is_ES = cb.columns.str.startswith('ECSany') | cb.columns.str.startswith('ECSdif') | cb.columns.str.startswith('ECSspe')
    ON_cols = cb.columns[is_ON].tolist()
    OS_cols = cb.columns[is_OS].tolist()
    EN_cols = cb.columns[is_EN].tolist()
    ES_cols = cb.columns[is_ES].tolist()
    stats = dict()
    stats['median'] = ['dist_bl', 'dist_node_num'] + omega_cols
    stats['total'] = ON_cols + OS_cols + EN_cols + ES_cols
    return stats


def _build_cutoff_column_plan(cb, cutoff_stat_str):
    cutoff_stat_entries = table.parse_cutoff_stat(cutoff_stat_str=cutoff_stat_str)
    cutoff_column_plan = []
    for cutoff_stat_exp, cutoff_stat_value in cutoff_stat_entries:
        is_col = cb.columns.str.fullmatch(cutoff_stat_exp, na=False)
        if is_col.sum() == 0:
            txt = 'The column "{}" was not found in the cb table. '
            txt += 'Check the format of the --cutoff_stat specification ("{}") carefully.'
            raise ValueError(txt.format(cutoff_stat_exp, cutoff_stat_str))
        cutoff_stat_cols = cb.columns[is_col]
        for cutoff_stat_col in cutoff_stat_cols:
            cutoff_column_plan.append((str(cutoff_stat_col), float(cutoff_stat_value)))
    return cutoff_column_plan


def _evaluate_cutoff_column_plan(cb, cutoff_column_plan):
    if len(cutoff_column_plan) == 0:
        return np.ones(shape=(cb.shape[0],), dtype=bool)
    is_enough_stat = np.ones(shape=(cb.shape[0],), dtype=bool)
    for cutoff_col, cutoff_value in cutoff_column_plan:
        values = cb.loc[:, cutoff_col].to_numpy(copy=False)
        if np.issubdtype(values.dtype, np.number):
            values_float = values.astype(np.float64, copy=False)
        else:
            values_float = pd.to_numeric(cb.loc[:, cutoff_col], errors='coerce').to_numpy(dtype=np.float64, copy=False)
        is_enough_stat &= np.isfinite(values_float) & (values_float >= float(cutoff_value))
    return is_enough_stat


def _build_clade_permutation_target_suffix_to_col(cb, trait_names, target_col_prefixes):
    suffix_to_col = dict()
    for trait_name in trait_names:
        for target_col_prefix in target_col_prefixes:
            target_col = target_col_prefix + '_' + trait_name
            if target_col in cb.columns:
                suffix = target_col.replace('is_', '_')
                suffix_to_col[suffix] = target_col
            elif target_col_prefix == 'is_all':
                suffix_to_col['_all'] = None
    if '_all' not in suffix_to_col:
        suffix_to_col['_all'] = None
    return suffix_to_col


def _build_clade_permutation_fast_stats_plan(g, cb, trait_names, target_col_prefixes):
    stats = _resolve_cb_stats_for_median_and_total(cb)
    median_stats = [ms for ms in stats['median'] if ms in cb.columns]
    total_stats = [ms for ms in stats['total'] if ms in cb.columns]
    all_stats = list(dict.fromkeys(median_stats + total_stats))
    stat_to_index = {stat_col: i for i, stat_col in enumerate(all_stats)}
    median_indices = np.array([stat_to_index[ms] for ms in median_stats], dtype=np.int64)
    total_indices = np.array([stat_to_index[ms] for ms in total_stats], dtype=np.int64)
    suffix_to_col = _build_clade_permutation_target_suffix_to_col(
        cb=cb,
        trait_names=trait_names,
        target_col_prefixes=target_col_prefixes,
    )
    median_output_cols = dict()
    total_output_cols = dict()
    count_output_cols = dict()
    for suffix in suffix_to_col.keys():
        median_output_cols[suffix] = ['median_' + ms + suffix for ms in median_stats]
        total_output_cols[suffix] = ['total_' + ms + suffix for ms in total_stats]
        count_output_cols[suffix] = ('num' + suffix, 'num_qualified' + suffix)
    cutoff_column_plan = _build_cutoff_column_plan(cb=cb, cutoff_stat_str=g['cutoff_stat'])
    return {
        'all_stats': all_stats,
        'median_indices': median_indices,
        'total_indices': total_indices,
        'median_output_cols': median_output_cols,
        'total_output_cols': total_output_cols,
        'count_output_cols': count_output_cols,
        'suffix_to_col': suffix_to_col,
        'cutoff_column_plan': cutoff_column_plan,
    }


def _get_clade_permutation_fast_stats_plan(g, cb, trait_names, target_col_prefixes):
    plan_cache = g.get('_clade_permutation_fast_stats_plan_cache', None)
    if plan_cache is None:
        plan_cache = dict()
        g['_clade_permutation_fast_stats_plan_cache'] = plan_cache
    cache_key = (tuple(cb.columns.tolist()), tuple(trait_names), str(g.get('cutoff_stat', '')))
    if cache_key not in plan_cache:
        plan_cache[cache_key] = _build_clade_permutation_fast_stats_plan(
            g=g,
            cb=cb,
            trait_names=trait_names,
            target_col_prefixes=target_col_prefixes,
        )
    return plan_cache[cache_key]


def _build_clade_permutation_selected_rows_plan(g, cb, trait_names, target_col_prefixes):
    base_plan = _build_clade_permutation_fast_stats_plan(
        g=g,
        cb=cb,
        trait_names=trait_names,
        target_col_prefixes=target_col_prefixes,
    )
    all_stats = base_plan['all_stats']
    if len(all_stats) == 0:
        stat_matrix = np.zeros(shape=(cb.shape[0], 0), dtype=np.float64)
    else:
        try:
            stat_matrix = cb.loc[:, all_stats].to_numpy(dtype=np.float64, copy=False)
        except Exception:
            stat_matrix = np.empty(shape=(cb.shape[0], len(all_stats)), dtype=np.float64)
            for i, stat_col in enumerate(all_stats):
                series = cb.loc[:, stat_col]
                if np.issubdtype(series.dtype, np.number):
                    stat_matrix[:, i] = series.to_numpy(dtype=np.float64, copy=False)
                else:
                    stat_matrix[:, i] = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64, copy=False)
    target_arrays_by_suffix = dict()
    for suffix, target_col in base_plan['suffix_to_col'].items():
        if target_col is None:
            continue
        target_arrays_by_suffix[suffix] = (cb.loc[:, target_col].to_numpy(copy=False) == 'Y')
    cutoff_arrays = []
    for cutoff_col, cutoff_value in base_plan['cutoff_column_plan']:
        values = cb.loc[:, cutoff_col].to_numpy(copy=False)
        if np.issubdtype(values.dtype, np.number):
            values_float = values.astype(np.float64, copy=False)
        else:
            values_float = pd.to_numeric(cb.loc[:, cutoff_col], errors='coerce').to_numpy(dtype=np.float64, copy=False)
        cutoff_arrays.append((values_float, float(cutoff_value)))
    out = dict(base_plan)
    out['stat_matrix'] = stat_matrix
    out['target_arrays_by_suffix'] = target_arrays_by_suffix
    out['cutoff_arrays'] = cutoff_arrays
    return out


def _get_clade_permutation_selected_rows_plan(g, cb, trait_names, target_col_prefixes):
    cache = g.get('_clade_permutation_selected_rows_plan_cache', None)
    if cache is None:
        cache = dict()
        g['_clade_permutation_selected_rows_plan_cache'] = cache
    cache_key = (
        id(cb),
        tuple(cb.columns.tolist()),
        tuple(trait_names),
        str(g.get('cutoff_stat', '')),
    )
    if cache_key not in cache:
        cache[cache_key] = _build_clade_permutation_selected_rows_plan(
            g=g,
            cb=cb,
            trait_names=trait_names,
            target_col_prefixes=target_col_prefixes,
        )
    return cache[cache_key]


def _build_cb_target_masks_from_suffix_plan(cb, suffix_to_col):
    masks = dict()
    for suffix, target_col in suffix_to_col.items():
        if target_col is None:
            masks[suffix] = np.ones(shape=(cb.shape[0],), dtype=bool)
        else:
            masks[suffix] = (cb.loc[:, target_col].to_numpy(copy=False) == 'Y')
    return masks


def _compute_aggregate_cb_stat_values_from_plan(cb, plan, is_targets):
    all_stats = plan['all_stats']
    if len(all_stats) == 0:
        return dict()
    try:
        stat_matrix = cb.loc[:, all_stats].to_numpy(dtype=np.float64, copy=False)
    except Exception:
        stat_matrix = np.empty(shape=(cb.shape[0], len(all_stats)), dtype=np.float64)
        for i, stat_col in enumerate(all_stats):
            series = cb.loc[:, stat_col]
            if np.issubdtype(series.dtype, np.number):
                stat_matrix[:, i] = series.to_numpy(dtype=np.float64, copy=False)
            else:
                stat_matrix[:, i] = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64, copy=False)
    median_indices = plan['median_indices']
    total_indices = plan['total_indices']
    aggregated_values = dict()
    for suffix, mask in is_targets.items():
        selected = stat_matrix[mask, :]
        if median_indices.shape[0] > 0:
            median_selected = selected[:, median_indices]
            medians = np.full(shape=(median_indices.shape[0],), fill_value=np.nan, dtype=np.float64)
            if median_selected.shape[0] > 0:
                has_finite = np.any(~np.isnan(median_selected), axis=0)
                if has_finite.any():
                    medians[has_finite] = np.nanmedian(median_selected[:, has_finite], axis=0)
            for out_col, out_value in zip(plan['median_output_cols'][suffix], medians.tolist()):
                aggregated_values[out_col] = float(out_value)
        if total_indices.shape[0] > 0:
            total_selected = selected[:, total_indices]
            if total_selected.shape[0] == 0:
                totals = np.zeros(shape=(total_indices.shape[0],), dtype=np.float64)
            else:
                totals = np.nansum(total_selected, axis=0)
            for out_col, out_value in zip(plan['total_output_cols'][suffix], totals.tolist()):
                aggregated_values[out_col] = float(out_value)
    return aggregated_values


def _compute_aggregate_cb_stat_values_from_selected_matrix(stat_matrix, plan, is_targets):
    if stat_matrix.shape[1] == 0:
        return dict()
    median_indices = plan['median_indices']
    total_indices = plan['total_indices']
    aggregated_values = dict()
    for suffix, mask in is_targets.items():
        selected = stat_matrix[mask, :]
        if median_indices.shape[0] > 0:
            median_selected = selected[:, median_indices]
            medians = np.full(shape=(median_indices.shape[0],), fill_value=np.nan, dtype=np.float64)
            if median_selected.shape[0] > 0:
                has_finite = np.any(~np.isnan(median_selected), axis=0)
                if has_finite.any():
                    medians[has_finite] = np.nanmedian(median_selected[:, has_finite], axis=0)
            for out_col, out_value in zip(plan['median_output_cols'][suffix], medians.tolist()):
                aggregated_values[out_col] = float(out_value)
        if total_indices.shape[0] > 0:
            total_selected = selected[:, total_indices]
            if total_selected.shape[0] == 0:
                totals = np.zeros(shape=(total_indices.shape[0],), dtype=np.float64)
            else:
                totals = np.nansum(total_selected, axis=0)
            for out_col, out_value in zip(plan['total_output_cols'][suffix], totals.tolist()):
                aggregated_values[out_col] = float(out_value)
    return aggregated_values


def _append_cb_target_count_columns(df_cb_stats, is_targets, is_qualified):
    count_values = _compute_target_count_values(is_targets=is_targets, is_qualified=is_qualified)
    if len(count_values) > 0:
        row_index = df_cb_stats.index[0]
        cols = list(count_values.keys())
        df_cb_stats.loc[row_index, cols] = [count_values[col] for col in cols]
    return df_cb_stats


def _ensure_df_cb_stats_columns(df_cb_stats, cols, fill_value=0.0):
    missing_cols = [col for col in cols if col not in df_cb_stats.columns]
    if len(missing_cols) == 0:
        return df_cb_stats
    for col in missing_cols:
        df_cb_stats.loc[:, col] = fill_value
    return df_cb_stats


def _compute_target_count_values(is_targets, is_qualified):
    out = dict()
    is_qualified = np.asarray(is_qualified, dtype=bool)
    for suffix, is_target in is_targets.items():
        is_target = np.asarray(is_target, dtype=bool)
        out['num' + suffix] = int(is_target.sum())
        out['num_qualified' + suffix] = int((is_target & is_qualified).sum())
    return out


def _compute_aggregate_cb_stat_values(cb, stats, is_targets):
    target_masks = {suffix: np.asarray(is_target, dtype=bool) for suffix, is_target in is_targets.items()}
    available_stats = [ms for stat_values in stats.values() for ms in stat_values if ms in cb.columns]
    # Preserve order while removing duplicates.
    available_stats = list(dict.fromkeys(available_stats))
    values_by_ms = dict()
    for ms in available_stats:
        series = cb.loc[:, ms]
        if np.issubdtype(series.dtype, np.number):
            values_by_ms[ms] = series.to_numpy(dtype=np.float64, copy=False)
        else:
            values_by_ms[ms] = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64, copy=False)
    aggregated_values = dict()
    for stat in stats.keys():
        for ms in stats[stat]:
            values = values_by_ms.get(ms, None)
            if values is None:
                continue
            for suffix, mask in target_masks.items():
                col = stat + '_' + ms + suffix
                selected_values = values[mask]
                if stat == 'median':
                    if selected_values.shape[0] == 0:
                        aggregated_values[col] = np.nan
                    elif np.isnan(selected_values).all():
                        aggregated_values[col] = np.nan
                    else:
                        aggregated_values[col] = float(np.nanmedian(selected_values))
                elif stat == 'total':
                    aggregated_values[col] = float(np.nansum(selected_values))
    return aggregated_values


def _aggregate_cb_stats(df_cb_stats, cb, stats, is_targets):
    required_cols = []
    for stat in stats.keys():
        for suffix in is_targets.keys():
            required_cols.extend([stat + '_' + ms + suffix for ms in stats[stat]])
    df_cb_stats = _ensure_df_cb_stats_columns(df_cb_stats=df_cb_stats, cols=required_cols, fill_value=0.0)
    aggregated_values = _compute_aggregate_cb_stat_values(
        cb=cb,
        stats=stats,
        is_targets=is_targets,
    )
    if len(aggregated_values) > 0:
        row_index = df_cb_stats.index[0]
        cols = list(aggregated_values.keys())
        df_cb_stats.loc[row_index, cols] = [aggregated_values[col] for col in cols]
    return df_cb_stats


def _add_median_cb_stats_fast_for_clade_permutation(g, cb, current_arity, start, verbose=True):
    row_defaults = g.get('_clade_permutation_cb_stats_row_defaults', None)
    columns = g.get('_clade_permutation_cb_stats_columns', None)
    if (row_defaults is None) or (columns is None):
        return None
    target_col_prefixes = ['is_fg', 'is_mg', 'is_mf', 'is_all']
    trait_names = _get_trait_names(g)
    stats_plan = _get_clade_permutation_fast_stats_plan(
        g=g,
        cb=cb,
        trait_names=trait_names,
        target_col_prefixes=target_col_prefixes,
    )
    is_targets = _build_cb_target_masks_from_suffix_plan(
        cb=cb,
        suffix_to_col=stats_plan['suffix_to_col'],
    )
    is_qualified = _evaluate_cutoff_column_plan(cb=cb, cutoff_column_plan=stats_plan['cutoff_column_plan'])
    update_values = dict()
    update_values['arity'] = int(current_arity)
    update_values['cutoff_stat'] = g.get('cutoff_stat', '')
    update_values.update(_compute_target_count_values(is_targets=is_targets, is_qualified=is_qualified))
    update_values.update(_compute_aggregate_cb_stat_values_from_plan(cb=cb, plan=stats_plan, is_targets=is_targets))
    elapsed_time = int(time.time() - start)
    update_values['elapsed_sec'] = elapsed_time
    row_values = dict(row_defaults)
    row_values.update(update_values)
    missing_cols = [col for col in update_values.keys() if col not in columns]
    if len(missing_cols) > 0:
        columns = columns + missing_cols
        g['_clade_permutation_cb_stats_columns'] = columns
        for col in missing_cols:
            row_defaults[col] = np.nan
    g['_clade_permutation_current_stat_row'] = row_values
    if verbose:
        g['df_cb_stats'] = pd.DataFrame(
            [{col: row_values.get(col, np.nan) for col in columns}],
            columns=columns,
        )
    if verbose:
        _print_total_oc_ec_summary(df_cb_stats=g['df_cb_stats'])
        print(("Elapsed time for arity = {}: {:,.1f} sec\n".format(current_arity, elapsed_time)), flush=True)
    return g


def _build_clade_permutation_target_masks_for_selected_rows(
    cb,
    selected_rows,
    suffix_to_col,
    trait_name,
    target_arrays_by_suffix=None,
):
    selected_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    num_rows = selected_rows.shape[0]
    if target_arrays_by_suffix is None:
        target_arrays_by_suffix = dict()
    out = dict()
    fg_suffix = '_fg_' + str(trait_name)
    mg_suffix = '_mg_' + str(trait_name)
    mf_suffix = '_mf_' + str(trait_name)
    for suffix, target_col in suffix_to_col.items():
        if suffix == '_all':
            out[suffix] = np.ones(shape=(num_rows,), dtype=bool)
            continue
        if suffix == fg_suffix:
            out[suffix] = np.ones(shape=(num_rows,), dtype=bool)
            continue
        if (suffix == mg_suffix) or (suffix == mf_suffix):
            out[suffix] = np.zeros(shape=(num_rows,), dtype=bool)
            continue
        if target_col is None:
            out[suffix] = np.ones(shape=(num_rows,), dtype=bool)
            continue
        precomputed = target_arrays_by_suffix.get(suffix, None)
        if precomputed is not None:
            out[suffix] = precomputed[selected_rows]
        else:
            out[suffix] = (cb.loc[selected_rows, target_col].to_numpy(copy=False) == 'Y')
    return out


def _evaluate_cutoff_column_plan_for_selected_rows(cb, selected_rows, cutoff_column_plan, cutoff_arrays=None):
    selected_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    if selected_rows.shape[0] == 0:
        return np.zeros(shape=(0,), dtype=bool)
    if len(cutoff_column_plan) == 0:
        return np.ones(shape=(selected_rows.shape[0],), dtype=bool)
    if cutoff_arrays is None:
        cutoff_arrays = []
    is_enough_stat = np.ones(shape=(selected_rows.shape[0],), dtype=bool)
    if len(cutoff_arrays) == len(cutoff_column_plan):
        for values_float, cutoff_value in cutoff_arrays:
            selected_values = values_float[selected_rows]
            is_enough_stat &= np.isfinite(selected_values) & (selected_values >= float(cutoff_value))
        return is_enough_stat
    for cutoff_col, cutoff_value in cutoff_column_plan:
        values = cb.loc[selected_rows, cutoff_col].to_numpy(copy=False)
        if np.issubdtype(values.dtype, np.number):
            values_float = values.astype(np.float64, copy=False)
        else:
            values_float = pd.to_numeric(cb.loc[selected_rows, cutoff_col], errors='coerce').to_numpy(dtype=np.float64, copy=False)
        is_enough_stat &= np.isfinite(values_float) & (values_float >= float(cutoff_value))
    return is_enough_stat


def _set_clade_permutation_stat_row_from_selected_rows(g, cb, selected_rows, trait_name, current_arity, start):
    row_defaults = g.get('_clade_permutation_cb_stats_row_defaults', None)
    columns = g.get('_clade_permutation_cb_stats_columns', None)
    if (row_defaults is None) or (columns is None):
        return None
    trait_names = _get_trait_names(g)
    target_col_prefixes = ['is_fg', 'is_mg', 'is_mf', 'is_all']
    stats_plan = _get_clade_permutation_selected_rows_plan(
        g=g,
        cb=cb,
        trait_names=trait_names,
        target_col_prefixes=target_col_prefixes,
    )
    selected_rows = np.asarray(selected_rows, dtype=np.int64).reshape(-1)
    if selected_rows.shape[0] == 0:
        stat_matrix = np.zeros(shape=(0, stats_plan['stat_matrix'].shape[1]), dtype=np.float64)
    else:
        stat_matrix = stats_plan['stat_matrix'][selected_rows, :]
    is_targets = _build_clade_permutation_target_masks_for_selected_rows(
        cb=cb,
        selected_rows=selected_rows,
        suffix_to_col=stats_plan['suffix_to_col'],
        trait_name=trait_name,
        target_arrays_by_suffix=stats_plan.get('target_arrays_by_suffix', None),
    )
    is_qualified = _evaluate_cutoff_column_plan_for_selected_rows(
        cb=cb,
        selected_rows=selected_rows,
        cutoff_column_plan=stats_plan['cutoff_column_plan'],
        cutoff_arrays=stats_plan.get('cutoff_arrays', None),
    )
    update_values = dict()
    update_values['arity'] = int(current_arity)
    update_values['cutoff_stat'] = g.get('cutoff_stat', '')
    update_values.update(_compute_target_count_values(is_targets=is_targets, is_qualified=is_qualified))
    update_values.update(
        _compute_aggregate_cb_stat_values_from_selected_matrix(
            stat_matrix=stat_matrix,
            plan=stats_plan,
            is_targets=is_targets,
        )
    )
    update_values['elapsed_sec'] = int(time.time() - start)
    row_values = dict(row_defaults)
    row_values.update(update_values)
    missing_cols = [col for col in update_values.keys() if col not in columns]
    if len(missing_cols) > 0:
        columns = columns + missing_cols
        g['_clade_permutation_cb_stats_columns'] = columns
        for col in missing_cols:
            row_defaults[col] = np.nan
    g['_clade_permutation_current_stat_row'] = row_values
    return g


def _print_total_oc_ec_summary(df_cb_stats):
    oc_total_cols = [
        col for col in df_cb_stats.columns
        if str(col).startswith('total_OC') and str(col).endswith('_all')
    ]
    for oc_col in sorted(oc_total_cols):
        key = str(oc_col).replace('total_OC', '').replace('_all', '')
        ec_col = 'total_EC' + key + '_all'
        if ec_col not in df_cb_stats.columns:
            continue
        total_oc = df_cb_stats.at[0, oc_col]
        total_ec = df_cb_stats.at[0, ec_col]
        if total_oc == 0:
            percent_value = np.nan
        else:
            percent_value = total_ec / total_oc * 100
        txt = 'Total OC{}/EC{} = {:,.1f}/{:,.1f} (expectation = {:,.1f}% of observation)'
        print(txt.format(key, key, total_oc, total_ec, percent_value))


def add_median_cb_stats(g, cb, current_arity, start, verbose=True):
    if '_clade_permutation_cb_stats_row_defaults' in g:
        fast_out = _add_median_cb_stats_fast_for_clade_permutation(
            g=g,
            cb=cb,
            current_arity=current_arity,
            start=start,
            verbose=verbose,
        )
        if fast_out is not None:
            return fast_out
    target_col_prefixes = ['is_fg', 'is_mg', 'is_mf', 'is_all']
    trait_names = _get_trait_names(g)
    is_targets = _build_cb_target_masks(
        cb=cb,
        trait_names=trait_names,
        target_col_prefixes=target_col_prefixes,
    )
    stats = _resolve_cb_stats_for_median_and_total(cb)
    is_qualified = table.get_cutoff_stat_bool_array(cb=cb, cutoff_stat_str=g['cutoff_stat'])
    g['df_cb_stats'] = _append_cb_target_count_columns(
        df_cb_stats=g['df_cb_stats'],
        is_targets=is_targets,
        is_qualified=is_qualified,
    )
    g['df_cb_stats'] = _aggregate_cb_stats(
        df_cb_stats=g['df_cb_stats'],
        cb=cb,
        stats=stats,
        is_targets=is_targets,
    )
    if verbose:
        _print_total_oc_ec_summary(df_cb_stats=g['df_cb_stats'])
    elapsed_time = int(time.time() - start)
    g['df_cb_stats'].at[0, 'elapsed_sec'] = elapsed_time
    if verbose:
        print(("Elapsed time for arity = {}: {:,.1f} sec\n".format(current_arity, elapsed_time)), flush=True)
    return g


_ORIGINAL_ADD_MEDIAN_CB_STATS = add_median_cb_stats


def _is_default_add_median_cb_stats_impl():
    return add_median_cb_stats is _ORIGINAL_ADD_MEDIAN_CB_STATS


def _recompute_missing_permutation_rows(g, missing_id_combinations, OS_tensor_reducer, ON_tensor_reducer):
    cbOS = substitution.get_cb(
        missing_id_combinations,
        OS_tensor_reducer,
        g,
        'OCS',
        selected_base_stats=g.get('output_base_stats'),
    )
    cbON = substitution.get_cb(
        missing_id_combinations,
        ON_tensor_reducer,
        g,
        'OCN',
        selected_base_stats=g.get('output_base_stats'),
    )
    cb_missing = table.merge_tables(cbOS, cbON)
    cb_missing = substitution.add_dif_stats(
        cb_missing,
        g['float_tol'],
        prefix='OC',
        output_stats=g.get('output_stats'),
    )
    cb_missing, g = omega.calc_omega(cb_missing, OS_tensor_reducer, ON_tensor_reducer, g)
    if g['calibrate_longtail'] and (g['exhaustive_until'] >= g['current_arity']):
        cb_missing = omega.calibrate_dsc(
            cb_missing,
            output_stats=g.get('output_stats'),
            float_tol=g.get('float_tol', 1e-12),
        )
    if g['branch_dist']:
        cb_missing = tree.get_node_distance(
            tree=g['tree'],
            cb=cb_missing,
            ncpu=g['threads'],
            float_type=g['float_type'],
            min_items_for_parallel=int(g.get('parallel_min_items_branch_dist', 20000)),
            min_items_per_job=int(g.get('parallel_min_items_per_job_branch_dist', 5000)),
        )
    cb_missing = substitution.get_substitutions_per_branch(cb_missing, g['branch_table'], g)
    cb_missing = table.get_linear_regression(cb_missing)
    cb_missing = output_stat.drop_unrequested_stat_columns(cb_missing, g.get('output_stats'))
    cb_missing, g = get_foreground_branch_num(cb_missing, g)
    cb_missing = table.sort_cb(cb_missing)
    return cb_missing, g


def _get_permutation_cb_pair_lookup(cb, bid_columns, g):
    cache = g.get('_permutation_cb_pair_lookup_cache', None)
    cache_key = (id(cb), tuple(bid_columns))
    if (cache is not None) and (cache.get('cache_key', None) == cache_key):
        return cache['lookup'], cache['multiplier']
    bid_values = cb.loc[:, bid_columns].to_numpy(copy=False, dtype=np.int64)
    if bid_values.shape[0] == 0:
        lookup = dict()
        multiplier = np.int64(1)
    else:
        bid_values = np.sort(bid_values, axis=1)
        multiplier = np.int64(int(bid_values.max()) + 1)
        keys = bid_values[:, 0].astype(np.int64, copy=False) * multiplier + bid_values[:, 1].astype(np.int64, copy=False)
        lookup = dict()
        for i, key in enumerate(keys.tolist()):
            lookup.setdefault(int(key), []).append(i)
    g['_permutation_cb_pair_lookup_cache'] = {
        'cache_key': cache_key,
        'lookup': lookup,
        'multiplier': multiplier,
    }
    return lookup, multiplier


def _get_permutation_cb_rows_fast_pairs(rid_combinations, cb, bid_columns, g, build_rcb=True):
    rid_array = np.asarray(rid_combinations, dtype=np.int64)
    if (rid_array.ndim != 2) or (rid_array.shape[1] != 2):
        return None
    if rid_array.shape[0] == 0:
        empty_rows = np.zeros(shape=(0,), dtype=np.int64)
        if build_rcb:
            return cb.iloc[0:0, :].copy(), 0, 0, empty_rows
        return None, 0, 0, empty_rows
    rid_sorted = np.sort(rid_array, axis=1)
    lookup, multiplier = _get_permutation_cb_pair_lookup(cb=cb, bid_columns=bid_columns, g=g)
    keys = rid_sorted[:, 0].astype(np.int64, copy=False) * multiplier + rid_sorted[:, 1].astype(np.int64, copy=False)
    selected_rows = []
    missing_rows = []
    for rid_idx, key in enumerate(keys.tolist()):
        row_ids = lookup.get(int(key), None)
        if row_ids is None:
            missing_rows.append(rid_idx)
            continue
        selected_rows.extend(row_ids)
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    if len(selected_rows) > 0:
        if build_rcb:
            rcb = cb.take(selected_rows, axis=0)
        else:
            rcb = None
    else:
        if build_rcb:
            rcb = cb.iloc[0:0, :].copy()
        else:
            rcb = None
    num_kept = int(selected_rows.shape[0])
    return rcb, rid_array.shape[0] - num_kept, rid_array.shape[0], selected_rows

def _get_permutation_cb_rows(rid_combinations, cb, cb_cache, g, OS_tensor_reducer=None, ON_tensor_reducer=None):
    bid_columns = ['branch_id_' + str(k + 1) for k in np.arange(rid_combinations.shape[1])]
    selected_rows = None
    if cb_cache.shape[0] == 0:
        fast_out = _get_permutation_cb_rows_fast_pairs(
            rid_combinations=rid_combinations,
            cb=cb,
            bid_columns=bid_columns,
            g=g,
            build_rcb=(not _is_default_add_median_cb_stats_impl()),
        )
        if fast_out is not None:
            rcb, dropped_before_recompute, requested_rows, selected_rows = fast_out
            if (dropped_before_recompute == 0) or (OS_tensor_reducer is None) or (ON_tensor_reducer is None):
                if selected_rows is not None:
                    dropped_after_recompute = requested_rows - selected_rows.shape[0]
                elif rcb is not None:
                    dropped_after_recompute = requested_rows - rcb.shape[0]
                else:
                    dropped_after_recompute = requested_rows
                return rcb, cb_cache, g, requested_rows, dropped_before_recompute, dropped_after_recompute, selected_rows
    rid_combinations = pd.DataFrame(rid_combinations, columns=bid_columns)
    rid_combinations = table.sort_branch_ids(rid_combinations)
    if cb_cache.shape[0] == 0:
        cb_pool = cb
    else:
        cb_pool = pd.concat([cb, cb_cache], ignore_index=True)
    rcb = pd.merge(rid_combinations, cb_pool, how='inner', on=bid_columns)
    dropped_before_recompute = rid_combinations.shape[0] - rcb.shape[0]
    if (dropped_before_recompute > 0) and (OS_tensor_reducer is not None) and (ON_tensor_reducer is not None):
        cb_id_rows = cb_pool.loc[:, bid_columns].drop_duplicates().reset_index(drop=True)
        missing = pd.merge(
            rid_combinations,
            cb_id_rows,
            how='left',
            on=bid_columns,
            indicator=True,
        )
        missing = missing.loc[missing.loc[:, '_merge'] == 'left_only', bid_columns].drop_duplicates().reset_index(drop=True)
        if missing.shape[0] != 0:
            missing_id_combinations = missing.loc[:, bid_columns].to_numpy(copy=True, dtype=np.int64)
            cb_missing, g = _recompute_missing_permutation_rows(
                g=g,
                missing_id_combinations=missing_id_combinations,
                OS_tensor_reducer=OS_tensor_reducer,
                ON_tensor_reducer=ON_tensor_reducer,
            )
            if cb_missing.shape[0] != 0:
                cb_cache = pd.concat([cb_cache, cb_missing], ignore_index=True)
                cb_pool = pd.concat([cb, cb_cache], ignore_index=True)
                rcb = pd.merge(rid_combinations, cb_pool, how='inner', on=bid_columns)
    dropped_after_recompute = rid_combinations.shape[0] - rcb.shape[0]
    return rcb, cb_cache, g, rid_combinations.shape[0], dropped_before_recompute, dropped_after_recompute, None

def _clade_permutation_mode_prefix(trait_name):
    return 'randomization_' + str(trait_name) + '_'


def _move_duplicate_columns_to_end(df):
    if df is None:
        return df
    if df.shape[1] == 0:
        return df
    non_placeholder_positions = []
    placeholder_positions = []
    for i, col in enumerate(df.columns.tolist()):
        if str(col).endswith('_PLACEHOLDER'):
            placeholder_positions.append(i)
        else:
            non_placeholder_positions.append(i)
    if len(placeholder_positions) > 0:
        df = df.iloc[:, non_placeholder_positions + placeholder_positions]
    first_positions = []
    duplicate_positions = []
    seen = set()
    for i, col in enumerate(df.columns.tolist()):
        if col in seen:
            duplicate_positions.append(i)
            continue
        seen.add(col)
        first_positions.append(i)
    if len(duplicate_positions) == 0:
        return df
    reordered_positions = first_positions + duplicate_positions
    return df.iloc[:, reordered_positions]


def _build_empty_clade_permutation_cb_stats_template(g):
    source = g.get('df_cb_stats_observed', g.get('df_cb_stats', pd.DataFrame()))
    default_cols = ['arity', 'elapsed_sec', 'fg_enrichment_factor', 'mode', 'dSC_calibration', 'cutoff_stat']
    if (source is None) or (source.shape[1] == 0):
        columns = default_cols
    else:
        columns = source.columns.tolist()
        for default_col in default_cols:
            if default_col not in columns:
                columns.append(default_col)
    # Keep only the first occurrence to match initialize_df_cb_stats-style schema.
    dedup_columns = []
    seen_columns = set()
    for col in columns:
        if col in seen_columns:
            continue
        seen_columns.add(col)
        dedup_columns.append(col)
    columns = dedup_columns
    data = {}
    for col in columns:
        col_txt = str(col)
        if col_txt == 'arity':
            data[col] = np.array([int(g['current_arity'])], dtype=np.int64)
        elif col_txt == 'cutoff_stat':
            data[col] = np.array([str(g.get('cutoff_stat', ''))], dtype=object)
        elif col_txt in ['mode', 'dSC_calibration']:
            data[col] = np.array([''], dtype=object)
        elif col_txt.startswith('clade_permutation_status_'):
            data[col] = np.array([''], dtype=object)
        elif col_txt.startswith('fg_enrichment_factor'):
            data[col] = np.array([np.nan], dtype=np.float64)
        elif (
            col_txt.startswith('num')
            or col_txt.startswith('total_')
            or col_txt.startswith('median_')
            or col_txt == 'elapsed_sec'
        ):
            data[col] = np.array([0.0], dtype=np.float64)
        else:
            data[col] = np.array([np.nan], dtype=np.float64)
    template = pd.DataFrame(data=data, columns=columns)
    return template


def _initialize_current_arity_cb_stats(g):
    template = g.get('_clade_permutation_cb_stats_template', None)
    if template is not None:
        row_defaults = g.get('_clade_permutation_cb_stats_row_defaults', None)
        if row_defaults is not None:
            row_values = dict(row_defaults)
            row_values['arity'] = int(g['current_arity'])
            row_values['cutoff_stat'] = g.get('cutoff_stat', '')
            g['_clade_permutation_current_stat_row'] = row_values
            if (
                ('df_cb_stats' not in g)
                or (not isinstance(g['df_cb_stats'], pd.DataFrame))
                or (g['df_cb_stats'].shape[0] == 0)
            ):
                g['df_cb_stats'] = template.copy(deep=True)
            if 'arity' in g['df_cb_stats'].columns:
                g['df_cb_stats'].at[0, 'arity'] = g['current_arity']
            if 'cutoff_stat' in g['df_cb_stats'].columns:
                g['df_cb_stats'].at[0, 'cutoff_stat'] = g.get('cutoff_stat', '')
            return g
        g['df_cb_stats'] = template.copy(deep=True)
        if 'arity' in g['df_cb_stats'].columns:
            g['df_cb_stats'].at[0, 'arity'] = g['current_arity']
        if 'cutoff_stat' in g['df_cb_stats'].columns:
            g['df_cb_stats'].at[0, 'cutoff_stat'] = g.get('cutoff_stat', '')
        return g
    g = param.initialize_df_cb_stats(g)
    is_arity = (g['df_cb_stats'].loc[:, 'arity'] == g['current_arity'])
    g['df_cb_stats'] = g['df_cb_stats'].loc[is_arity, :].reset_index(drop=True)
    return g


def _build_clade_permutation_mode(trait_name, iteration, randomized_bids, sample_original_foreground):
    mode = _clade_permutation_mode_prefix(trait_name) + 'iter' + str(iteration)
    if sample_original_foreground:
        mode += '_sampleorig'
    normalized_bids = _normalize_branch_ids(randomized_bids)
    mode += '_bid' + ','.join(normalized_bids.astype(str).tolist())
    return mode


def _report_dropped_permutation_rows(dropped_after_recompute, requested_rows, kept_rows):
    if dropped_after_recompute == 0:
        return
    txt = '{:,} ({:,} - {:,}) permuted foreground branch combinations were dropped because they were not included in the cb table.'
    print(txt.format(dropped_after_recompute, requested_rows, kept_rows))


def _set_randomized_trait_flags(rcb, trait_name):
    rcb['is_fg_' + trait_name] = 'Y'
    rcb['is_mg_' + trait_name] = 'N'
    rcb['is_mf_' + trait_name] = 'N'
    return rcb


def _get_current_clade_permutation_stat_row(g):
    row = g.get('_clade_permutation_current_stat_row', None)
    if row is not None:
        return row
    df_cb_stats = g.get('df_cb_stats', None)
    if (df_cb_stats is None) or (not isinstance(df_cb_stats, pd.DataFrame)) or (df_cb_stats.shape[0] == 0):
        return None
    return df_cb_stats.iloc[0].to_dict()


def _append_current_clade_permutation_row_to_buffer(g):
    row = _get_current_clade_permutation_stat_row(g=g)
    if row is None:
        return g
    if '_clade_permutation_rows_buffer' not in g:
        g['_clade_permutation_rows_buffer'] = []
    g['_clade_permutation_rows_buffer'].append(dict(row))
    return g


def _flush_clade_permutation_rows_buffer(g):
    rows_buffer = g.get('_clade_permutation_rows_buffer', None)
    if rows_buffer is None:
        return g
    if len(rows_buffer) == 0:
        return g
    columns = list(g.get('_clade_permutation_cb_stats_columns', []))
    for row in rows_buffer:
        for col in row.keys():
            if col not in columns:
                columns.append(col)
    rows_for_df = [{col: row.get(col, np.nan) for col in columns} for row in rows_buffer]
    perm_df = pd.DataFrame(rows_for_df, columns=columns)
    if ('df_cb_stats_main' not in g) or (g['df_cb_stats_main'] is None) or (g['df_cb_stats_main'].shape[0] == 0):
        g['df_cb_stats_main'] = perm_df
    else:
        g['df_cb_stats_main'] = pd.concat([g['df_cb_stats_main'], perm_df], ignore_index=True)
    rows_buffer.clear()
    return g


def _is_valid_clade_permutation_stat_row(g, trait_name, rid_combinations):
    omega_col = 'median_omegaCany2spe_fg_' + trait_name
    row = _get_current_clade_permutation_stat_row(g=g)
    if row is not None:
        if omega_col not in row:
            txt = 'omegaCany2spe could not be obtained for trait "{}"; skipping this clade permutation trial.\n'
            sys.stderr.write(txt.format(trait_name))
            return False
        omega_value = row.get(omega_col, np.nan)
        if isinstance(omega_value, (np.ndarray, list, tuple)):
            if len(omega_value) == 0:
                txt = 'No clade-permutation stats row was available for trait "{}"; skipping this clade permutation trial.\n'
                sys.stderr.write(txt.format(trait_name))
                return False
            omega_value = np.asarray(omega_value).reshape(-1)[0]
        if not pd.isna(omega_value):
            return True
        print('OmegaCany2spe could not be obtained for permuted foregrounds:')
        print(rid_combinations)
        print('')
        return False
    if omega_col not in g['df_cb_stats'].columns:
        txt = 'omegaCany2spe could not be obtained for trait "{}"; skipping this clade permutation trial.\n'
        sys.stderr.write(txt.format(trait_name))
        return False
    omega_values = g['df_cb_stats'].loc[:, omega_col].values
    if omega_values.shape[0] == 0:
        txt = 'No clade-permutation stats row was available for trait "{}"; skipping this clade permutation trial.\n'
        sys.stderr.write(txt.format(trait_name))
        return False
    if pd.isna(omega_values[0]):
        print('OmegaCany2spe could not be obtained for permuted foregrounds:')
        print(rid_combinations)
        print('')
        return False
    return True


def _resolve_observed_clade_permutation_omega(g, trait_name, obs_col):
    if obs_col not in g['df_cb_stats_observed'].columns:
        txt = 'Observed stats were unavailable for trait "{}"; skipping clade-permutation summary.\n'
        sys.stderr.write(txt.format(trait_name))
        return None, None
    is_arity_obs = (g['df_cb_stats_observed'].loc[:, 'arity'] == g['current_arity'])
    obs_values = g['df_cb_stats_observed'].loc[is_arity_obs, obs_col].dropna().values
    if obs_values.shape[0] == 0:
        txt = 'No observed omegaCany2spe value was found for trait "{}"; skipping clade-permutation summary.\n'
        sys.stderr.write(txt.format(trait_name))
        return None, None
    return obs_values[0], is_arity_obs


def _resolve_permutation_clade_permutation_omega(g, trait_name, obs_col):
    if 'mode' not in g['df_cb_stats_main'].columns:
        txt = 'Permutation rows were unavailable for trait "{}"; skipping clade-permutation summary.\n'
        sys.stderr.write(txt.format(trait_name))
        return None, None, None
    if obs_col not in g['df_cb_stats_main'].columns:
        txt = 'Permutation omegaCany2spe values were unavailable for trait "{}"; skipping clade-permutation summary.\n'
        sys.stderr.write(txt.format(trait_name))
        return None, None, None
    mode_prefix = _clade_permutation_mode_prefix(trait_name)
    is_arity_perm = (g['df_cb_stats_main'].loc[:, 'arity'] == g['current_arity'])
    is_stat_permutation = g['df_cb_stats_main'].loc[:, 'mode'].astype(str).str.startswith(mode_prefix)
    permutation_values = g['df_cb_stats_main'].loc[is_arity_perm & is_stat_permutation, obs_col].dropna().values
    if permutation_values.shape[0] == 0:
        txt = 'No valid clade permutations were available for trait "{}"; p value was not calculated.\n'
        sys.stderr.write(txt.format(trait_name))
        return None, None, None
    return permutation_values, is_arity_perm, is_stat_permutation


def _report_observed_clade_permutation_ocn(g, trait_name, obs_ocn_col, is_arity_obs):
    if obs_ocn_col not in g['df_cb_stats_observed'].columns:
        return
    obs_ocn_values = g['df_cb_stats_observed'].loc[is_arity_obs, obs_ocn_col].dropna().values
    if obs_ocn_values.shape[0] > 0:
        print('Trait {}: Observed total OCNany2spe in foreground lineages = {:,.3}'.format(trait_name, obs_ocn_values[0]))


def _report_permutation_clade_permutation_ocn(g, trait_name, obs_ocn_col, is_arity_perm, is_stat_permutation):
    if obs_ocn_col not in g['df_cb_stats_main'].columns:
        return
    permutation_ocns = g['df_cb_stats_main'].loc[is_arity_perm & is_stat_permutation, obs_ocn_col].dropna().values
    if permutation_ocns.shape[0] == 0:
        return
    finite_mean, finite_std = _get_finite_mean_std(permutation_ocns)
    txt = 'Trait {}: Total OCNany2spe in permutation lineages = {:,.3}; {:,.3} ± {:,.3} (median; mean ± SD excluding inf)'
    print(txt.format(trait_name, np.median(permutation_ocns), finite_mean, finite_std))


def _get_finite_mean_std(values):
    try:
        numeric_values = np.asarray(values, dtype=np.float64)
    except Exception:
        numeric_values = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=np.float64, copy=False)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    if finite_values.shape[0] == 0:
        return np.nan, np.nan
    return finite_values.mean(), finite_values.std()


def _report_clade_permutation_stats(g, trait_name):
    obs_col = 'median_omegaCany2spe_fg_' + trait_name
    obs_ocn_col = 'total_OCNany2spe_fg_' + trait_name
    obs_value, is_arity_obs = _resolve_observed_clade_permutation_omega(
        g=g,
        trait_name=trait_name,
        obs_col=obs_col,
    )
    if obs_value is None:
        return
    permutation_values, is_arity_perm, is_stat_permutation = _resolve_permutation_clade_permutation_omega(
        g=g,
        trait_name=trait_name,
        obs_col=obs_col,
    )
    if permutation_values is None:
        return
    num_positive = (obs_value <= permutation_values).sum()
    num_all = permutation_values.shape[0]
    pvalue = num_positive / num_all
    _report_observed_clade_permutation_ocn(
        g=g,
        trait_name=trait_name,
        obs_ocn_col=obs_ocn_col,
        is_arity_obs=is_arity_obs,
    )
    _report_permutation_clade_permutation_ocn(
        g=g,
        trait_name=trait_name,
        obs_ocn_col=obs_ocn_col,
        is_arity_perm=is_arity_perm,
        is_stat_permutation=is_stat_permutation,
    )
    print('Trait {}: Observed median omegaCany2spe in foreground lineages = {:,.3}'.format(trait_name, obs_value))
    finite_mean, finite_std = _get_finite_mean_std(permutation_values)
    txt = 'Trait {}: Median omegaCany2spe in permutation lineages = {:,.3}; {:,.3} ± {:,.3} (median; mean ± SD excluding inf)'
    print(txt.format(trait_name, np.median(permutation_values), finite_mean, finite_std))
    txt = 'Trait {}: P value of foreground convergence (omegaCany2spe) by clade permutations = {} (observation <= permutation = {:,}/{:,})'
    print(txt.format(trait_name, pvalue, num_positive, num_all))


def _append_clade_permutation_failure_row(g, trait_name, trial_no, reason):
    g = _initialize_current_arity_cb_stats(g)
    mode_prefix = _clade_permutation_mode_prefix(trait_name)
    row = _get_current_clade_permutation_stat_row(g=g)
    if row is None:
        row = dict()
    row = dict(row)
    row['mode'] = mode_prefix + 'iter0_failed_trial' + str(trial_no)
    status_col = 'clade_permutation_status_' + trait_name
    row[status_col] = str(reason)
    g['_clade_permutation_current_stat_row'] = row
    g = _append_current_clade_permutation_row_to_buffer(g=g)
    return g


def _attempt_randomized_combinations(g, trait_name, trial_no, sample_original_foreground):
    try:
        g, rid_combinations = set_random_foreground_branch(
            g,
            trait_name,
            sample_original_foreground=sample_original_foreground,
        )
        return g, rid_combinations, sample_original_foreground, False, False
    except Exception as exc:
        if not sample_original_foreground:
            txt = 'Clade permutation retry for trait "{}": allowing sampling from original foreground clades after trial {:,} failure ({})\n'
            sys.stderr.write(txt.format(trait_name, trial_no + 1, str(exc)))
            return g, None, True, True, False
        g = _append_clade_permutation_failure_row(g, trait_name, trial_no + 1, str(exc))
        txt = 'Clade permutation failed for trait "{}" at trial {:,}: {}\n'
        sys.stderr.write(txt.format(trait_name, trial_no + 1, str(exc)))
        return g, None, sample_original_foreground, False, True


def _resolve_clade_iteration_transition(g, trait_name, iteration, trial_no, max_trials):
    if iteration == g['fg_clade_permutation']:
        txt = 'Clade permutation successfully found {:,} new branch combinations for trait "{}" after {:,} trials.'
        print(txt.format(g['fg_clade_permutation'], trait_name, trial_no + 1))
        return False, True
    if trial_no == (max_trials - 1):
        txt = 'Clade permutation could not find enough number of new branch combinations for trait "{}" even after {:,} trials.\n'
        sys.stderr.write(txt.format(trait_name, max_trials))
    return True, False


def _resolve_clade_permutation_rows_and_mode(
    cb,
    cb_cache,
    g,
    trait_name,
    iteration,
    sample_original_foreground,
    rid_combinations,
    OS_tensor_reducer=None,
    ON_tensor_reducer=None,
):
    random_mode = _build_clade_permutation_mode(
        trait_name=trait_name,
        iteration=iteration,
        randomized_bids=g['r_fg_ids'][trait_name],
        sample_original_foreground=sample_original_foreground,
    )
    rcb, cb_cache, g, requested_rows, _dropped_before_recompute, dropped_after_recompute, selected_rows = _get_permutation_cb_rows(
        rid_combinations=rid_combinations,
        cb=cb,
        cb_cache=cb_cache,
        g=g,
        OS_tensor_reducer=OS_tensor_reducer,
        ON_tensor_reducer=ON_tensor_reducer,
    )
    _report_dropped_permutation_rows(
        dropped_after_recompute=dropped_after_recompute,
        requested_rows=requested_rows,
        kept_rows=(selected_rows.shape[0] if selected_rows is not None else rcb.shape[0]),
    )
    return random_mode, rcb, cb_cache, g, selected_rows


def _finalize_clade_permutation_iteration(
    cb,
    cb_cache,
    g,
    trait_name,
    iteration,
    trial_no,
    max_trials,
    sample_original_foreground,
    rid_combinations,
    start,
    OS_tensor_reducer=None,
    ON_tensor_reducer=None,
):
    random_mode, rcb, cb_cache, g, selected_rows = _resolve_clade_permutation_rows_and_mode(
        cb=cb,
        cb_cache=cb_cache,
        g=g,
        trait_name=trait_name,
        iteration=iteration,
        sample_original_foreground=sample_original_foreground,
        rid_combinations=rid_combinations,
        OS_tensor_reducer=OS_tensor_reducer,
        ON_tensor_reducer=ON_tensor_reducer,
    )
    g['_clade_permutation_current_stat_row'] = None
    if (selected_rows is not None) and _is_default_add_median_cb_stats_impl():
        g = _set_clade_permutation_stat_row_from_selected_rows(
            g=g,
            cb=cb,
            selected_rows=selected_rows,
            trait_name=trait_name,
            current_arity=g['current_arity'],
            start=start,
        )
    else:
        rcb = _set_randomized_trait_flags(rcb=rcb, trait_name=trait_name)
        g = add_median_cb_stats(g, rcb, g['current_arity'], start, verbose=False)
    row = _get_current_clade_permutation_stat_row(g=g)
    if row is not None:
        row = dict(row)
        row['mode'] = random_mode
        g['_clade_permutation_current_stat_row'] = row
    else:
        g['df_cb_stats'].loc[:, 'mode'] = random_mode
    if not _is_valid_clade_permutation_stat_row(g=g, trait_name=trait_name, rid_combinations=rid_combinations):
        return g, cb_cache, False, False
    g = _append_current_clade_permutation_row_to_buffer(g=g)
    print('')
    advance_iter, break_trait = _resolve_clade_iteration_transition(
        g=g,
        trait_name=trait_name,
        iteration=iteration,
        trial_no=trial_no,
        max_trials=max_trials,
    )
    return g, cb_cache, advance_iter, break_trait


def _run_clade_permutation_iteration(
    cb,
    cb_cache,
    g,
    trait_name,
    iteration,
    trial_no,
    max_trials,
    sample_original_foreground,
    OS_tensor_reducer=None,
    ON_tensor_reducer=None,
):
    start = time.time()
    txt = 'Starting foreground clade permutation round {:,} (of {:,}) for trait "{}"'
    print(txt.format(iteration, g['fg_clade_permutation'], trait_name), flush=True)
    g = _initialize_current_arity_cb_stats(g)
    g, rid_combinations, sample_original_foreground, should_continue, break_trait = _attempt_randomized_combinations(
        g=g,
        trait_name=trait_name,
        trial_no=trial_no,
        sample_original_foreground=sample_original_foreground,
    )
    if should_continue:
        return g, cb_cache, sample_original_foreground, False, False
    if break_trait:
        return g, cb_cache, sample_original_foreground, False, True

    g, cb_cache, advance_iter, break_trait = _finalize_clade_permutation_iteration(
        cb=cb,
        cb_cache=cb_cache,
        g=g,
        trait_name=trait_name,
        iteration=iteration,
        trial_no=trial_no,
        max_trials=max_trials,
        sample_original_foreground=sample_original_foreground,
        rid_combinations=rid_combinations,
        start=start,
        OS_tensor_reducer=OS_tensor_reducer,
        ON_tensor_reducer=ON_tensor_reducer,
    )
    return g, cb_cache, sample_original_foreground, advance_iter, break_trait


def _run_clade_permutation_for_trait(
    cb,
    cb_cache,
    g,
    trait_name,
    max_trials,
    OS_tensor_reducer=None,
    ON_tensor_reducer=None,
):
    iteration = 1
    sample_original_foreground = False
    for trial_no in np.arange(max_trials):
        g, cb_cache, sample_original_foreground, advance_iter, break_trait = _run_clade_permutation_iteration(
            cb=cb,
            cb_cache=cb_cache,
            g=g,
            trait_name=trait_name,
            iteration=iteration,
            trial_no=trial_no,
            max_trials=max_trials,
            sample_original_foreground=sample_original_foreground,
            OS_tensor_reducer=OS_tensor_reducer,
            ON_tensor_reducer=ON_tensor_reducer,
        )
        if advance_iter:
            iteration += 1
        if break_trait:
            break
    g = _flush_clade_permutation_rows_buffer(g=g)
    _report_clade_permutation_stats(g, trait_name)
    return g, cb_cache


def clade_permutation(cb, g, OS_tensor_reducer=None, ON_tensor_reducer=None):
    print('Starting foreground clade permutation.')
    trait_names = _get_trait_names(g)
    if len(trait_names) == 0:
        sys.stderr.write('No foreground traits were available for clade permutation.\n')
        return g
    g['df_cb_stats_observed'] = g['df_cb_stats'].copy()
    g['_clade_permutation_cb_stats_template'] = _build_empty_clade_permutation_cb_stats_template(g)
    g['_clade_permutation_cb_stats_columns'] = g['_clade_permutation_cb_stats_template'].columns.tolist()
    g['_clade_permutation_cb_stats_row_defaults'] = g['_clade_permutation_cb_stats_template'].iloc[0].to_dict()
    g['_clade_permutation_rows_buffer'] = []
    g['_clade_permutation_current_stat_row'] = None
    max_trials = g['fg_clade_permutation'] * 10
    cb_cache = pd.DataFrame()
    for trait_name in trait_names:
        g, cb_cache = _run_clade_permutation_for_trait(
            cb=cb,
            cb_cache=cb_cache,
            g=g,
            trait_name=trait_name,
            max_trials=max_trials,
            OS_tensor_reducer=OS_tensor_reducer,
            ON_tensor_reducer=ON_tensor_reducer,
        )
    g['df_cb_stats_main'] = _move_duplicate_columns_to_end(g['df_cb_stats_main'])
    g['df_cb_stats'] = g['df_cb_stats_observed'].copy()
    if '_clade_permutation_cb_stats_template' in g:
        del g['_clade_permutation_cb_stats_template']
    if '_clade_permutation_cb_stats_columns' in g:
        del g['_clade_permutation_cb_stats_columns']
    if '_clade_permutation_cb_stats_row_defaults' in g:
        del g['_clade_permutation_cb_stats_row_defaults']
    if '_clade_permutation_rows_buffer' in g:
        del g['_clade_permutation_rows_buffer']
    if '_clade_permutation_current_stat_row' in g:
        del g['_clade_permutation_current_stat_row']
    if '_clade_permutation_selected_rows_plan_cache' in g:
        del g['_clade_permutation_selected_rows_plan_cache']
    return g

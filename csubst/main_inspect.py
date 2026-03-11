import os
import time
from glob import glob

import numpy as np
import pandas as pd

from csubst import ete
from csubst import genetic_code
from csubst import output_manifest
from csubst import parser_iqtree
from csubst import parser_misc
from csubst import runtime
from csubst import sequence
from csubst import structural_alphabet
from csubst import tree

_FAST_STATE_PLOT_CHUNK_ROWS = 20000


class _FastStatePlotBypassUnavailable(RuntimeError):
    pass


def _plot_state_tree_in_directory(output_dir, state, orders, mode, g, plot_request, plot_request_name):
    pattern = os.path.join(str(output_dir), 'csubst_state_*_' + str(mode) + '_*.pdf')
    for path in glob(pattern):
        if os.path.isfile(path):
            os.remove(path)
    os.makedirs(output_dir, exist_ok=True)
    return tree.plot_state_tree(
        state=state,
        orders=orders,
        mode=mode,
        g=g,
        output_dir=output_dir,
        plot_request=plot_request,
        plot_request_name=plot_request_name,
    )


def _collect_branch_rows(tree_obj):
    rows = []
    node_by_id = {}
    for node in tree_obj.traverse():
        branch_id = int(ete.get_prop(node, "numerical_label"))
        node_by_id[branch_id] = node
    for branch_id in sorted(node_by_id.keys()):
        node = node_by_id[branch_id]
        if ete.is_root(node):
            parent_branch_id = np.nan
        else:
            parent_branch_id = int(ete.get_prop(node.up, "numerical_label"))
        branch_length = 0.0 if (node.dist is None) else float(node.dist)
        row = {
            "branch_id": int(branch_id),
            "parent_branch_id": parent_branch_id,
            "is_root": bool(ete.is_root(node)),
            "is_leaf": bool(ete.is_leaf(node)),
            "node_name": "" if (node.name is None) else str(node.name),
            "branch_length": branch_length,
            "num_descendant_leaves": int(sum([1 for _ in ete.iter_leaves(node)])),
        }
        rows.append(row)
    return rows, node_by_id


def _get_trait_names(g):
    if "fg_df" not in g:
        return []
    fg_df = g["fg_df"]
    if fg_df.shape[1] <= 1:
        return []
    return fg_df.columns[1:].tolist()


def _get_inspect_manifest_rows(g):
    rows = g.get("_inspect_output_manifest_rows", None)
    if rows is None:
        rows = list()
        g["_inspect_output_manifest_rows"] = rows
    return rows


def _get_inspect_output_manifest_metadata(g):
    plot_state_aa = tree.normalize_state_plot_request(
        g.get("plot_state_aa", "no"),
        param_name="--plot_state_aa",
    )
    plot_state_codon = tree.normalize_state_plot_request(
        g.get("plot_state_codon", "no"),
        param_name="--plot_state_codon",
    )
    return {
        "plot_state_aa": str(plot_state_aa.get("token", "no")),
        "plot_state_codon": str(plot_state_codon.get("token", "no")),
        "nonsyn_recode": str(g.get("nonsyn_recode", "no")).strip().lower(),
        "drop_invariant_tip_sites": _normalize_drop_invariant_mode(g),
        "species_overlap_node_plot": str(g.get("species_overlap_node_plot", "auto")).strip().lower(),
        "tree_tip_label_spacing": float(g.get("tree_tip_label_spacing", 1.0)),
        "tree_fig_max_height": float(g.get("tree_fig_max_height", 180.0)),
    }


def _add_inspect_output_manifest_row(manifest_rows, output_path, output_kind, g, note=''):
    return output_manifest.add_output_manifest_row(
        manifest_rows=manifest_rows,
        output_path=output_path,
        output_kind=output_kind,
        note=note,
        base_dir=g["outdir"],
        extra_fields=_get_inspect_output_manifest_metadata(g),
    )


def _record_inspect_output_paths(g, output_paths, output_kind, note=''):
    if output_paths is None:
        return g
    if isinstance(output_paths, (str, os.PathLike)):
        normalized_paths = [str(output_paths)]
    else:
        normalized_paths = [str(path) for path in list(output_paths)]
    if len(normalized_paths) == 0:
        return g
    manifest_rows = _get_inspect_manifest_rows(g)
    for output_path in normalized_paths:
        _add_inspect_output_manifest_row(
            manifest_rows=manifest_rows,
            output_path=output_path,
            output_kind=output_kind,
            g=g,
            note=note,
        )
    return g


def _record_inspect_foreground_branch_files(g):
    for trait_name in _get_trait_names(g):
        output_path = runtime.output_path(g, "foreground_branch_" + str(trait_name) + ".txt")
        if os.path.exists(output_path):
            _record_inspect_output_paths(g=g, output_paths=output_path, output_kind="foreground_branch_txt")
    return g


def _write_inspect_output_manifest(g):
    manifest_path = runtime.output_path(g, "outputs.tsv")
    manifest_path = output_manifest.write_output_manifest(
        manifest_rows=_get_inspect_manifest_rows(g),
        manifest_path=manifest_path,
        note="manifest_self_row",
        base_dir=g["outdir"],
        extra_fields=_get_inspect_output_manifest_metadata(g),
    )
    print("Writing inspect output manifest: {}".format(manifest_path), flush=True)
    return manifest_path


def _strip_placeholder_suffix(text):
    suffix = "_PLACEHOLDER"
    text = str(text)
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def _sanitize_placeholder_columns(df):
    out = df.copy(deep=True)
    out.columns = [_strip_placeholder_suffix(col) for col in out.columns.tolist()]
    return out


def _write_branch_maps(g):
    base_rows, node_by_id = _collect_branch_rows(g["tree"])
    base_df = pd.DataFrame(base_rows).sort_values(by=["branch_id"]).reset_index(drop=True)
    trait_names = _get_trait_names(g)
    combined_df = base_df.copy(deep=True)
    output_paths = list()
    trait_specific_frames = {}
    for trait_name in trait_names:
        target_ids = g.get("target_ids", {}).get(trait_name, np.array([], dtype=np.int64))
        target_set = set([int(x) for x in np.asarray(target_ids).reshape(-1).tolist()])
        trait_rows = []
        for branch_id in combined_df["branch_id"].tolist():
            node = node_by_id[int(branch_id)]
            row = {
                "is_target_branch": bool(int(branch_id) in target_set),
                "is_fg": bool(ete.get_prop(node, "is_fg_" + trait_name, False)),
                "is_mf": bool(ete.get_prop(node, "is_mf_" + trait_name, False)),
                "is_mg": bool(ete.get_prop(node, "is_mg_" + trait_name, False)),
                "foreground_lineage_id": int(ete.get_prop(node, "foreground_lineage_id_" + trait_name, 0)),
                "branch_color": str(ete.get_prop(node, "color_" + trait_name, "black")),
                "label_color": str(ete.get_prop(node, "labelcolor_" + trait_name, "black")),
            }
            trait_rows.append(row)
        trait_df = pd.DataFrame(trait_rows)
        trait_specific_frames[trait_name] = pd.concat([base_df, trait_df], axis=1)
        combined_df.loc[:, "is_target_branch_" + trait_name] = trait_df.loc[:, "is_target_branch"].values
        combined_df.loc[:, "is_fg_" + trait_name] = trait_df.loc[:, "is_fg"].values
        combined_df.loc[:, "is_mf_" + trait_name] = trait_df.loc[:, "is_mf"].values
        combined_df.loc[:, "is_mg_" + trait_name] = trait_df.loc[:, "is_mg"].values
        combined_df.loc[:, "foreground_lineage_id_" + trait_name] = trait_df.loc[:, "foreground_lineage_id"].values
        combined_df.loc[:, "branch_color_" + trait_name] = trait_df.loc[:, "branch_color"].values
        combined_df.loc[:, "label_color_" + trait_name] = trait_df.loc[:, "label_color"].values
    combined_path = runtime.output_path(g, "branch_map.tsv")
    _sanitize_placeholder_columns(combined_df).to_csv(
        combined_path,
        sep="\t",
        index=False,
    )
    output_paths.append(combined_path)
    for trait_name in trait_names:
        out_file = runtime.output_path(g, "branch_map_" + str(trait_name) + ".tsv")
        out_file = out_file.replace("_PLACEHOLDER", "")
        if out_file == combined_path:
            continue
        _sanitize_placeholder_columns(trait_specific_frames[trait_name]).to_csv(out_file, sep="\t", index=False)
        output_paths.append(out_file)
    return output_paths


def _get_aa_state_view_for_inspect(g):
    recode = str(g.get("nonsyn_recode", "no")).strip().lower()
    if (recode != "no") and ("state_nsy" in g) and ("nonsyn_state_orders" in g):
        return g["state_nsy"], g["nonsyn_state_orders"], "nsy"
    return g["state_pep"], g["amino_acid_orders"], "aa"


def _configure_3di_smoke_mode(g):
    g["sa_inference_branch_ids"] = None
    g["sa_smoke_inferred_nonroot_branch_ids"] = None
    max_branches = int(g.get("sa_smoke_max_branches", 0))
    if max_branches <= 0:
        return g
    recode = str(g.get("nonsyn_recode", "no")).strip().lower()
    if recode != "3di20":
        print("Ignoring --sa_smoke_max_branches because --nonsyn_recode is not 3di20.", flush=True)
        return g
    nonroot_branch_ids = list()
    root_id = int(ete.get_prop(ete.get_tree_root(g["tree"]), "numerical_label"))
    for node in g["tree"].traverse():
        if ete.is_root(node):
            continue
        nonroot_branch_ids.append(int(ete.get_prop(node, "numerical_label")))
    nonroot_branch_ids = sorted(nonroot_branch_ids)
    selected_nonroot = nonroot_branch_ids[:max_branches]
    g["sa_smoke_inferred_nonroot_branch_ids"] = np.asarray(selected_nonroot, dtype=np.int64)
    g["sa_inference_branch_ids"] = np.asarray([root_id] + selected_nonroot, dtype=np.int64)
    txt = "3Di smoke mode: limiting inference to {:,} / {:,} non-root branches (plus root)."
    print(txt.format(len(selected_nonroot), len(nonroot_branch_ids)), flush=True)
    if str(g.get("sa_asr_mode", "direct")).strip().lower() == "direct":
        txt = "3Di smoke mode note: --sa_asr_mode direct still runs full IQ-TREE ASR; "
        txt += "branch filtering is applied when importing states."
        print(txt, flush=True)
    return g


def _normalize_drop_invariant_mode(g):
    mode = str(g.get("drop_invariant_tip_sites_mode", g.get("drop_invariant_tip_sites", "tip_invariant"))).strip().lower()
    if mode in ["1", "true", "yes", "on"]:
        return "tip_invariant"
    if mode in ["0", "false", "off", "no"]:
        return "no"
    return mode


def _should_use_fast_state_plot_bypass(g):
    request_modes = [
        tree.normalize_state_plot_request(g.get("plot_state_aa", "no"), param_name="--plot_state_aa")["mode"],
        tree.normalize_state_plot_request(g.get("plot_state_codon", "no"), param_name="--plot_state_codon")["mode"],
    ]
    has_specific_request = any([mode in ["pages", "concat"] for mode in request_modes])
    if not has_specific_request:
        return False
    if any([mode == "all" for mode in request_modes]):
        return False
    if str(g.get("nonsyn_recode", "no")).strip().lower() == "3di20":
        return False
    if _normalize_drop_invariant_mode(g) == "zero_sub_mass":
        return False
    return True


def _prepare_fast_inspect_context(g):
    return parser_misc.prepare_input_context(
        g,
        include_foreground=True,
        include_marginal=True,
        resolve_state_subset=True,
        prepare_state=False,
        state_metadata_only=True,
    )


def _ensure_tree_alignment_for_fast_inspect(g):
    ete.link_to_alignment(g["tree"], alignment=g["alignment_file"], alg_format="fasta")
    leaves = list(ete.iter_leaves(g["tree"]))
    if len(leaves) == 0:
        raise _FastStatePlotBypassUnavailable("tree has no leaves.")
    first_leaf_seq = str(ete.get_prop(leaves[0], "sequence", "")).upper()
    if first_leaf_seq == "":
        msg = 'Failed to map alignment to tree leaves. Check leaf labels in --alignment_file and --rooted_tree_file.'
        raise _FastStatePlotBypassUnavailable(msg)
    if (len(first_leaf_seq) % 3) != 0:
        raise _FastStatePlotBypassUnavailable("Sequence length is not multiple of 3 in alignment file.")
    num_alignment_site = int(len(first_leaf_seq) // 3)
    if num_alignment_site != int(g["num_input_site"]):
        txt = 'The number of codon sites did not match between the alignment and ancestral states.'
        raise _FastStatePlotBypassUnavailable(txt)
    return g


def _get_selected_plot_site_indices(g):
    selected = set()
    for plot_request, plot_request_name in [
        (g.get("plot_state_aa", "no"), "--plot_state_aa"),
        (g.get("plot_state_codon", "no"), "--plot_state_codon"),
    ]:
        request, site_indices = tree._resolve_state_plot_site_indices(
            num_site=int(g["num_input_site"]),
            plot_request=plot_request,
            param_name=plot_request_name,
        )
        if request["mode"] == "none":
            continue
        if request["mode"] == "all":
            raise _FastStatePlotBypassUnavailable('"all" requests should use the standard inspect path.')
        selected.update([int(v) for v in site_indices.tolist()])
    if len(selected) == 0:
        raise _FastStatePlotBypassUnavailable("no site-specific state plot request was found.")
    return np.array(sorted(selected), dtype=np.int64)


def _get_fast_aa_alignment_config(g):
    recode = str(g.get("nonsyn_recode", "no")).strip().lower()
    if recode != "no":
        group_orders = np.asarray(g["nonsyn_state_orders"], dtype=object)
        group_indices = g["nonsynonymous_indices"]
        alignment_orders = np.asarray(sequence._get_nsy_alignment_symbols(g), dtype=object)
        mode = "nsy"
    else:
        group_orders = np.asarray(g["amino_acid_orders"], dtype=object)
        group_indices = g["synonymous_indices"]
        alignment_orders = np.asarray(g["amino_acid_orders"], dtype=object)
        mode = "aa"
    group_matrix = sequence._get_group_sum_matrix(
        group_orders=group_orders,
        group_indices=group_indices,
        num_codon_state=len(g["codon_orders"]),
        dtype=g["float_type"],
    )
    group_index_by_codon = np.full(shape=(len(g["codon_orders"]),), fill_value=-1, dtype=np.int64)
    for group_index, state_name in enumerate(group_orders.tolist()):
        group_members = np.asarray(group_indices[str(state_name)], dtype=np.int64).reshape(-1)
        if group_members.shape[0] == 0:
            continue
        group_index_by_codon[group_members] = int(group_index)
    return {
        "mode": mode,
        "plot_orders": group_orders,
        "alignment_orders": alignment_orders,
        "group_matrix": np.asarray(group_matrix, dtype=g["float_type"]),
        "group_index_by_codon": group_index_by_codon,
    }


def _get_branch_id_matrix_dtype(num_state):
    if int(num_state) <= np.iinfo(np.int16).max:
        return np.int16
    return np.int32


def _stream_internal_states_for_fast_inspect(g, selected_site_indices, aa_config):
    num_node = int(g["num_node"])
    num_site = int(g["num_input_site"])
    num_state = int(g["num_input_state"])
    branch_id_dtype = _get_branch_id_matrix_dtype(num_state=num_state)
    codon_symbol_index = np.full(shape=(num_node, num_site), fill_value=-1, dtype=branch_id_dtype)
    aa_symbol_index = np.full(shape=(num_node, num_site), fill_value=-1, dtype=branch_id_dtype)
    state_cdn_subset = np.zeros(
        shape=(num_node, selected_site_indices.shape[0], num_state),
        dtype=g["float_type"],
    )
    internal_name_to_branch_id = dict()
    mapped_internal_branch_ids = list()
    for node in g["tree"].traverse():
        if ete.is_leaf(node):
            continue
        node_name = "" if (node.name is None) else str(node.name)
        branch_id = int(ete.get_prop(node, "numerical_label"))
        if node_name == "":
            continue
        if node_name in internal_name_to_branch_id:
            txt = 'Duplicate internal node name "{}" remained after tree preparation.'
            raise _FastStatePlotBypassUnavailable(txt.format(node_name))
        internal_name_to_branch_id[node_name] = branch_id
        mapped_internal_branch_ids.append(branch_id)
    if len(mapped_internal_branch_ids) == 0:
        raise _FastStatePlotBypassUnavailable("tree has no internal nodes.")
    mapped_internal_branch_ids = np.array(sorted(set(mapped_internal_branch_ids)), dtype=np.int64)
    seen = np.zeros(shape=(num_node, num_site), dtype=bool)
    selected_local_by_site = np.full(shape=(num_site,), fill_value=-1, dtype=np.int64)
    selected_local_by_site[selected_site_indices] = np.arange(selected_site_indices.shape[0], dtype=np.int64)
    state_columns = g.get("state_probability_columns", None)
    if state_columns is None:
        state_columns = ["p_" + str(state_name) for state_name in g["input_state"]]
    reader = pd.read_csv(
        g["iqtree_state"],
        sep="\t",
        index_col=False,
        header=0,
        comment="#",
        usecols=["Node", "Site"] + list(state_columns),
        chunksize=_FAST_STATE_PLOT_CHUNK_ROWS,
    )
    float_tol = float(g.get("float_tol", 0))
    for chunk in reader:
        site_values = pd.to_numeric(chunk.loc[:, "Site"], errors="coerce").to_numpy(dtype=float, copy=False)
        if not np.isfinite(site_values).all():
            raise _FastStatePlotBypassUnavailable("Non-numeric Site value(s) were found in .state file.")
        rounded_site_values = np.round(site_values)
        if not np.isclose(site_values, rounded_site_values, rtol=0.0, atol=1e-12).all():
            raise _FastStatePlotBypassUnavailable("Non-integer Site value(s) were found in .state file.")
        row_site_indices = rounded_site_values.astype(np.int64) - 1
        if ((row_site_indices < 0) | (row_site_indices >= num_site)).any():
            raise _FastStatePlotBypassUnavailable("State file contained out-of-range Site labels.")
        node_names = chunk.loc[:, "Node"].astype(str).to_numpy(dtype=object, copy=False)
        branch_ids = np.array(
            [internal_name_to_branch_id.get(str(node_name), -1) for node_name in node_names.tolist()],
            dtype=np.int64,
        )
        if (branch_ids < 0).any():
            raise _FastStatePlotBypassUnavailable("State file contained node names that were absent from the annotated tree.")
        if seen[branch_ids, row_site_indices].any():
            raise _FastStatePlotBypassUnavailable("Duplicate Node/Site row(s) were found in .state file.")
        seen[branch_ids, row_site_indices] = True
        row_probs = chunk.loc[:, state_columns].to_numpy(dtype=g["float_type"], copy=False)
        row_probs = np.nan_to_num(row_probs, copy=False)
        if bool(g.get("ml_anc", False)):
            ml_probs = np.zeros_like(row_probs)
            row_argmax = row_probs.argmax(axis=1)
            row_nonmissing = (row_probs.max(axis=1) >= float_tol)
            if row_nonmissing.any():
                ml_probs[row_nonmissing, row_argmax[row_nonmissing]] = 1
            row_probs = ml_probs
        row_max = row_probs.max(axis=1)
        row_argmax = row_probs.argmax(axis=1)
        nonmissing_rows = (row_max >= float_tol)
        if nonmissing_rows.any():
            codon_symbol_index[branch_ids[nonmissing_rows], row_site_indices[nonmissing_rows]] = row_argmax[nonmissing_rows]
        row_group_probs = row_probs.dot(aa_config["group_matrix"])
        row_group_max = row_group_probs.max(axis=1)
        row_group_argmax = row_group_probs.argmax(axis=1)
        nonmissing_group_rows = (row_group_max >= float_tol)
        if nonmissing_group_rows.any():
            aa_symbol_index[branch_ids[nonmissing_group_rows], row_site_indices[nonmissing_group_rows]] = (
                row_group_argmax[nonmissing_group_rows]
            )
        row_selected_local = selected_local_by_site[row_site_indices]
        is_selected_row = (row_selected_local >= 0)
        if is_selected_row.any():
            state_cdn_subset[branch_ids[is_selected_row], row_selected_local[is_selected_row], :] = row_probs[is_selected_row, :]
    is_missing_internal_site = (~seen[mapped_internal_branch_ids, :].all(axis=1))
    if is_missing_internal_site.any():
        missing_branch_ids = mapped_internal_branch_ids[is_missing_internal_site][:10].tolist()
        missing_txt = ",".join([str(int(v)) for v in missing_branch_ids])
        raise _FastStatePlotBypassUnavailable(
            "State file did not contain a complete site set for all internal nodes: {}".format(missing_txt)
        )
    return codon_symbol_index, aa_symbol_index, state_cdn_subset


def _fill_leaf_states_for_fast_inspect(g, selected_site_indices, codon_symbol_index, aa_symbol_index, state_cdn_subset, aa_config):
    num_site = int(g["num_input_site"])
    codon_state_lookup = sequence.build_state_index_lookup(g["codon_orders"])
    selected_local_by_site = np.full(shape=(num_site,), fill_value=-1, dtype=np.int64)
    selected_local_by_site[selected_site_indices] = np.arange(selected_site_indices.shape[0], dtype=np.int64)
    for leaf in ete.iter_leaves(g["tree"]):
        seq = str(ete.get_prop(leaf, "sequence", "")).upper()
        if seq == "":
            raise _FastStatePlotBypassUnavailable('Leaf sequence not found for node "{}".'.format(leaf.name))
        if (len(seq) % 3) != 0:
            raise _FastStatePlotBypassUnavailable('Sequence length is not multiple of 3. Node name = {}'.format(leaf.name))
        num_codon_site = int(len(seq) // 3)
        if num_codon_site != num_site:
            msg = 'Codon site count did not match alignment size for leaf "{}". Expected {}, observed {}.'
            raise _FastStatePlotBypassUnavailable(msg.format(leaf.name, num_site, num_codon_site))
        branch_id = int(ete.get_prop(leaf, "numerical_label"))
        for site_index in range(num_site):
            codon = seq[(site_index * 3):((site_index + 1) * 3)]
            codon_index_direct = codon_state_lookup.get(codon, None)
            if codon_index_direct is not None:
                codon_index = [int(codon_index_direct)]
            else:
                codon_index = sequence.get_state_index(
                    codon,
                    g["codon_orders"],
                    genetic_code.ambiguous_table,
                    state_lookup=codon_state_lookup,
                )
            if len(codon_index) == 0:
                continue
            codon_index_arr = np.asarray(codon_index, dtype=np.int64)
            codon_symbol_index[branch_id, site_index] = int(codon_index_arr.min())
            group_index_arr = aa_config["group_index_by_codon"][codon_index_arr]
            group_index_arr = group_index_arr[group_index_arr >= 0]
            if group_index_arr.shape[0] > 0:
                group_counts = np.bincount(group_index_arr, minlength=aa_config["alignment_orders"].shape[0])
                aa_symbol_index[branch_id, site_index] = int(group_counts.argmax())
            local_index = int(selected_local_by_site[site_index])
            if local_index < 0:
                continue
            state_cdn_subset[branch_id, local_index, codon_index_arr] = 1.0 / codon_index_arr.shape[0]
    return codon_symbol_index, aa_symbol_index, state_cdn_subset


def _apply_fast_missing_site_masks(g, selected_site_indices, codon_symbol_index, aa_symbol_index, state_cdn_subset):
    leaf_ids = np.array(
        [int(ete.get_prop(leaf, "numerical_label")) for leaf in ete.iter_leaves(g["tree"])],
        dtype=np.int64,
    )
    leaf_nonmissing_sites = parser_iqtree._get_leaf_nonmissing_sites(g=g, required_leaf_ids=leaf_ids)
    nonzero_masks = parser_iqtree.get_internal_site_nonzero_masks(
        tree=g["tree"],
        leaf_nonmissing_sites=leaf_nonmissing_sites,
    )
    for branch_id, is_nonzero in nonzero_masks.items():
        codon_symbol_index[int(branch_id), ~is_nonzero] = -1
        aa_symbol_index[int(branch_id), ~is_nonzero] = -1
        if selected_site_indices.shape[0] == 0:
            continue
        selected_mask = is_nonzero[selected_site_indices]
        state_cdn_subset[int(branch_id), ~selected_mask, :] = 0
    return codon_symbol_index, aa_symbol_index, state_cdn_subset


def _remove_legacy_state_plot_dir(g, output_name):
    legacy_dir = runtime.output_path(g, output_name)
    if os.path.isdir(legacy_dir):
        import shutil

        shutil.rmtree(legacy_dir)


def _write_fast_unfiltered_alignments(g, codon_symbol_index, aa_symbol_index, aa_config):
    loaded_branch_ids = g.get("state_loaded_branch_ids", None)
    codon_path = runtime.output_path(g, "alignment_codon.fa")
    sequence.write_alignment_from_symbol_indices(
        codon_path,
        symbol_index_matrix=codon_symbol_index,
        orders=g["codon_orders"],
        missing_state="---",
        g=g,
        branch_ids=loaded_branch_ids,
    )
    _record_inspect_output_paths(g=g, output_paths=codon_path, output_kind="alignment_codon_fa")
    aa_path = runtime.output_path(g, "alignment_aa.fa")
    sequence.write_alignment_from_symbol_indices(
        aa_path,
        symbol_index_matrix=aa_symbol_index,
        orders=aa_config["alignment_orders"],
        missing_state="-",
        g=g,
        branch_ids=loaded_branch_ids,
    )
    _record_inspect_output_paths(g=g, output_paths=aa_path, output_kind="alignment_aa_fa")


def _plot_fast_unfiltered_state_trees(g, selected_site_indices, state_cdn_subset, aa_config):
    site_numbers = (selected_site_indices + 1).astype(np.int64, copy=False)
    if tree.has_state_plot_request(g.get("plot_state_aa", "no")):
        _remove_legacy_state_plot_dir(g, "plot_state_aa")
        if aa_config["mode"] == "nsy":
            aa_state_subset = sequence.cdn2nsy_state(state_cdn=state_cdn_subset, g=g)
        else:
            aa_state_subset = sequence.cdn2pep_state(state_cdn=state_cdn_subset, g=g)
        aa_out_files = tree.plot_state_tree_selected_sites(
            state=aa_state_subset,
            orders=aa_config["plot_orders"],
            mode=aa_config["mode"],
            g=g,
            site_numbers=site_numbers,
            output_dir=g["outdir"],
            plot_request=g["plot_state_aa"],
            plot_request_name="--plot_state_aa",
        )
        _record_inspect_output_paths(g=g, output_paths=aa_out_files, output_kind="state_tree_aa_pdf")
    if tree.has_state_plot_request(g.get("plot_state_codon", "no")):
        _remove_legacy_state_plot_dir(g, "plot_state_codon")
        codon_out_files = tree.plot_state_tree_selected_sites(
            state=state_cdn_subset,
            orders=g["codon_orders"],
            mode="codon",
            g=g,
            site_numbers=site_numbers,
            output_dir=g["outdir"],
            plot_request=g["plot_state_codon"],
            plot_request_name="--plot_state_codon",
        )
        _record_inspect_output_paths(g=g, output_paths=codon_out_files, output_kind="state_tree_codon_pdf")


def _run_fast_unfiltered_outputs(g):
    print('Fast inspect state-plot bypass enabled: streaming selected-site outputs without full state tensors.', flush=True)
    g = _prepare_fast_inspect_context(g)
    g = _record_inspect_foreground_branch_files(g)
    g = _configure_3di_smoke_mode(g)
    g = _ensure_tree_alignment_for_fast_inspect(g)
    selected_site_indices = _get_selected_plot_site_indices(g)
    aa_config = _get_fast_aa_alignment_config(g)
    codon_symbol_index, aa_symbol_index, state_cdn_subset = _stream_internal_states_for_fast_inspect(
        g=g,
        selected_site_indices=selected_site_indices,
        aa_config=aa_config,
    )
    codon_symbol_index, aa_symbol_index, state_cdn_subset = _fill_leaf_states_for_fast_inspect(
        g=g,
        selected_site_indices=selected_site_indices,
        codon_symbol_index=codon_symbol_index,
        aa_symbol_index=aa_symbol_index,
        state_cdn_subset=state_cdn_subset,
        aa_config=aa_config,
    )
    codon_symbol_index, aa_symbol_index, state_cdn_subset = _apply_fast_missing_site_masks(
        g=g,
        selected_site_indices=selected_site_indices,
        codon_symbol_index=codon_symbol_index,
        aa_symbol_index=aa_symbol_index,
        state_cdn_subset=state_cdn_subset,
    )
    _write_fast_unfiltered_alignments(
        g=g,
        codon_symbol_index=codon_symbol_index,
        aa_symbol_index=aa_symbol_index,
        aa_config=aa_config,
    )
    _plot_fast_unfiltered_state_trees(
        g=g,
        selected_site_indices=selected_site_indices,
        state_cdn_subset=state_cdn_subset,
        aa_config=aa_config,
    )
    return g


def _run_standard_unfiltered_outputs(g):
    g = parser_misc.prepare_input_context(
        g,
        include_foreground=True,
        include_marginal=True,
        resolve_state_subset=True,
        prepare_state=False,
    )
    g = _record_inspect_foreground_branch_files(g)
    g = _configure_3di_smoke_mode(g)
    g = parser_misc.prep_state(g, apply_site_filtering=False)
    loaded_branch_ids = g.get("state_loaded_branch_ids", None)
    aa_state_unfiltered, aa_orders_unfiltered, aa_mode_unfiltered = _get_aa_state_view_for_inspect(g)
    codon_state_unfiltered = g["state_cdn"]
    codon_path = runtime.output_path(g, "alignment_codon.fa")
    sequence.write_alignment(
        codon_path,
        mode="codon",
        g=g,
        branch_ids=loaded_branch_ids,
    )
    _record_inspect_output_paths(g=g, output_paths=codon_path, output_kind="alignment_codon_fa")
    aa_path = runtime.output_path(g, "alignment_aa.fa")
    sequence.write_alignment(
        aa_path,
        mode=aa_mode_unfiltered,
        g=g,
        branch_ids=loaded_branch_ids,
    )
    _record_inspect_output_paths(g=g, output_paths=aa_path, output_kind="alignment_aa_fa")
    if str(g.get("nonsyn_recode", "no")).strip().lower() == "3di20":
        nsy_branch_ids = g.get("sa_smoke_inferred_nonroot_branch_ids", None)
        if nsy_branch_ids is None:
            nsy_branch_ids = loaded_branch_ids
        alignment_3di_path = runtime.output_path(g, "alignment_3di.fa")
        sequence.write_alignment(
            alignment_3di_path,
            mode="nsy",
            g=g,
            branch_ids=nsy_branch_ids,
        )
        _record_inspect_output_paths(g=g, output_paths=alignment_3di_path, output_kind="alignment_3di_fa")
    if tree.has_state_plot_request(g.get("plot_state_aa", "no")):
        _remove_legacy_state_plot_dir(g, "plot_state_aa")
        aa_out_files = _plot_state_tree_in_directory(
            output_dir=g["outdir"],
            state=aa_state_unfiltered,
            orders=aa_orders_unfiltered,
            mode=aa_mode_unfiltered,
            g=g,
            plot_request=g["plot_state_aa"],
            plot_request_name="--plot_state_aa",
        )
        _record_inspect_output_paths(g=g, output_paths=aa_out_files, output_kind="state_tree_aa_pdf")
    if tree.has_state_plot_request(g.get("plot_state_codon", "no")):
        _remove_legacy_state_plot_dir(g, "plot_state_codon")
        codon_out_files = _plot_state_tree_in_directory(
            output_dir=g["outdir"],
            state=codon_state_unfiltered,
            orders=g["codon_orders"],
            mode="codon",
            g=g,
            plot_request=g["plot_state_codon"],
            plot_request_name="--plot_state_codon",
        )
        _record_inspect_output_paths(g=g, output_paths=codon_out_files, output_kind="state_tree_codon_pdf")
    return g


def _finalize_inspect_outputs(g):
    g = parser_misc.apply_site_filters(g)
    site_index_map_path = runtime.output_path(g, "site_index_map.tsv")
    if os.path.exists(site_index_map_path):
        _record_inspect_output_paths(g=g, output_paths=site_index_map_path, output_kind="site_index_map_tsv")
    tree_path = runtime.output_path(g, "tree.nwk")
    tree.write_tree(g["tree"], outfile=tree_path)
    _record_inspect_output_paths(g=g, output_paths=tree_path, output_kind="tree_newick")
    branch_all_paths = tree.plot_branch_category(g, file_base=runtime.output_path(g, "branch_id"), label="all")
    _record_inspect_output_paths(g=g, output_paths=branch_all_paths, output_kind="branch_category_all_pdf")
    branch_leaf_paths = tree.plot_branch_category(g, file_base=runtime.output_path(g, "branch_id_leaf"), label="leaf")
    _record_inspect_output_paths(g=g, output_paths=branch_leaf_paths, output_kind="branch_category_leaf_pdf")
    branch_nolabel_paths = tree.plot_branch_category(g, file_base=runtime.output_path(g, "branch_id_nolabel"), label="no")
    _record_inspect_output_paths(g=g, output_paths=branch_nolabel_paths, output_kind="branch_category_nolabel_pdf")
    branch_map_paths = _write_branch_maps(g)
    _record_inspect_output_paths(g=g, output_paths=branch_map_paths, output_kind="branch_map_tsv")
    return g


def main_inspect(g):
    start = time.time()
    g = runtime.ensure_output_layout(g, create_dir=True)
    g["_inspect_output_manifest_rows"] = list()
    g["plot_state_aa"] = tree.normalize_state_plot_request(
        g.get("plot_state_aa", "no"),
        param_name="--plot_state_aa",
    )
    g["plot_state_codon"] = tree.normalize_state_plot_request(
        g.get("plot_state_codon", "no"),
        param_name="--plot_state_codon",
    )
    g["write_site_index_map"] = True
    if bool(g.get("download_prostt5", False)):
        model_source = structural_alphabet.ensure_prostt5_model_files(g=g)
        print("ProstT5 model files are ready: {}".format(model_source), flush=True)
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)
        return
    print("Reading and parsing input files.", flush=True)
    if _should_use_fast_state_plot_bypass(g):
        try:
            g = _run_fast_unfiltered_outputs(g)
        except _FastStatePlotBypassUnavailable as exc:
            txt = 'Fast inspect state-plot bypass was disabled after validation: {}. Falling back to standard state loading.'
            print(txt.format(str(exc)), flush=True)
            g["_inspect_output_manifest_rows"] = list()
            g = _run_standard_unfiltered_outputs(g)
    else:
        g = _run_standard_unfiltered_outputs(g)
    g = _finalize_inspect_outputs(g)
    if bool(g.get("output_manifest", True)):
        _write_inspect_output_manifest(g)
    else:
        print('Skipping inspect output manifest (--output_manifest no).', flush=True)

    elapsed_time = int(time.time() - start)
    print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

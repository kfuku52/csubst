import os
import time
from glob import glob

import numpy as np
import pandas as pd

from csubst import ete
from csubst import parser_misc
from csubst import runtime
from csubst import sequence
from csubst import structural_alphabet
from csubst import tree


def _plot_state_tree_in_directory(output_dir, state, orders, mode, g, plot_request, plot_request_name):
    pattern = os.path.join(str(output_dir), 'csubst_state_*_' + str(mode) + '_*.pdf')
    for path in glob(pattern):
        if os.path.isfile(path):
            os.remove(path)
    os.makedirs(output_dir, exist_ok=True)
    tree.plot_state_tree(
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
    _sanitize_placeholder_columns(combined_df).to_csv(
        runtime.output_path(g, "branch_map.tsv"),
        sep="\t",
        index=False,
    )
    for trait_name in trait_names:
        out_file = runtime.output_path(g, "branch_map_" + str(trait_name) + ".tsv")
        out_file = out_file.replace("_PLACEHOLDER", "")
        if out_file == runtime.output_path(g, "branch_map.tsv"):
            continue
        _sanitize_placeholder_columns(trait_specific_frames[trait_name]).to_csv(out_file, sep="\t", index=False)


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


def main_inspect(g):
    start = time.time()
    g = runtime.ensure_output_layout(g, create_dir=True)
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
    g = parser_misc.prepare_input_context(
        g,
        include_foreground=True,
        include_marginal=True,
        resolve_state_subset=True,
        prepare_state=False,
    )
    g = _configure_3di_smoke_mode(g)
    g = parser_misc.prep_state(g, apply_site_filtering=False)
    loaded_branch_ids = g.get("state_loaded_branch_ids", None)
    aa_state_unfiltered, aa_orders_unfiltered, aa_mode_unfiltered = _get_aa_state_view_for_inspect(g)
    codon_state_unfiltered = g["state_cdn"]
    sequence.write_alignment(
        runtime.output_path(g, "alignment_codon.fa"),
        mode="codon",
        g=g,
        branch_ids=loaded_branch_ids,
    )
    sequence.write_alignment(
        runtime.output_path(g, "alignment_aa.fa"),
        mode=aa_mode_unfiltered,
        g=g,
        branch_ids=loaded_branch_ids,
    )
    if str(g.get("nonsyn_recode", "no")).strip().lower() == "3di20":
        nsy_branch_ids = g.get("sa_smoke_inferred_nonroot_branch_ids", None)
        if nsy_branch_ids is None:
            nsy_branch_ids = loaded_branch_ids
        sequence.write_alignment(
            runtime.output_path(g, "alignment_3di.fa"),
            mode="nsy",
            g=g,
            branch_ids=nsy_branch_ids,
        )
    if tree.has_state_plot_request(g.get("plot_state_aa", "no")):
        legacy_aa_dir = runtime.output_path(g, "plot_state_aa")
        if os.path.isdir(legacy_aa_dir):
            import shutil

            shutil.rmtree(legacy_aa_dir)
        _plot_state_tree_in_directory(
            output_dir=g["outdir"],
            state=aa_state_unfiltered,
            orders=aa_orders_unfiltered,
            mode=aa_mode_unfiltered,
            g=g,
            plot_request=g["plot_state_aa"],
            plot_request_name="--plot_state_aa",
        )
    if tree.has_state_plot_request(g.get("plot_state_codon", "no")):
        legacy_codon_dir = runtime.output_path(g, "plot_state_codon")
        if os.path.isdir(legacy_codon_dir):
            import shutil

            shutil.rmtree(legacy_codon_dir)
        _plot_state_tree_in_directory(
            output_dir=g["outdir"],
            state=codon_state_unfiltered,
            orders=g["codon_orders"],
            mode="codon",
            g=g,
            plot_request=g["plot_state_codon"],
            plot_request_name="--plot_state_codon",
        )

    g = parser_misc.apply_site_filters(g)

    tree.write_tree(g["tree"], outfile=runtime.output_path(g, "tree.nwk"))
    tree.plot_branch_category(g, file_base=runtime.output_path(g, "branch_id"), label="all")
    tree.plot_branch_category(g, file_base=runtime.output_path(g, "branch_id_leaf"), label="leaf")
    tree.plot_branch_category(g, file_base=runtime.output_path(g, "branch_id_nolabel"), label="no")

    _write_branch_maps(g)

    elapsed_time = int(time.time() - start)
    print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

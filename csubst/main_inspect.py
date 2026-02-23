import os
import shutil
import time

import numpy as np
import pandas as pd

from csubst import ete
from csubst import foreground
from csubst import genetic_code
from csubst import parser_misc
from csubst import sequence
from csubst import structural_alphabet
from csubst import tree


def _plot_state_tree_in_directory(output_dir, state, orders, mode, g):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        tree.plot_state_tree(state=state, orders=orders, mode=mode, g=g)
    finally:
        os.chdir(cwd)


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
    _sanitize_placeholder_columns(combined_df).to_csv("csubst_branch_map.tsv", sep="\t", index=False)
    for trait_name in trait_names:
        out_file = ("csubst_branch_map_" + str(trait_name) + ".tsv").replace("_PLACEHOLDER", "")
        if out_file == "csubst_branch_map.tsv":
            continue
        _sanitize_placeholder_columns(trait_specific_frames[trait_name]).to_csv(out_file, sep="\t", index=False)


def _get_aa_state_view_for_inspect(g):
    recode = str(g.get("nonsyn_recode", "no")).strip().lower()
    if (recode != "no") and ("state_nsy" in g) and ("nonsyn_state_orders" in g):
        return g["state_nsy"], g["nonsyn_state_orders"], "nsy"
    return g["state_pep"], g["amino_acid_orders"], "aa"


def main_inspect(g):
    start = time.time()
    if bool(g.get("download_prostt5", False)):
        model_source = structural_alphabet.ensure_prostt5_model_files(g=g)
        print("ProstT5 model files are ready: {}".format(model_source), flush=True)
        elapsed_time = int(time.time() - start)
        print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)
        return
    print("Reading and parsing input files.", flush=True)
    g["codon_table"] = genetic_code.get_codon_table(ncbi_id=g["genetic_code"])
    g = tree.read_treefile(g)
    g = parser_misc.generate_intermediate_files(g)
    g = parser_misc.annotate_tree(g)
    g = parser_misc.read_input(g)
    g = foreground.get_foreground_branch(g)
    g = foreground.get_marginal_branch(g)
    g = parser_misc.resolve_state_loading(g)
    g = parser_misc.prep_state(g)
    loaded_branch_ids = g.get("state_loaded_branch_ids", None)
    sequence.write_alignment("csubst_alignment_codon.fa", mode="codon", g=g, branch_ids=loaded_branch_ids)
    aa_state, aa_orders, aa_mode = _get_aa_state_view_for_inspect(g)
    sequence.write_alignment("csubst_alignment_aa.fa", mode=aa_mode, g=g, branch_ids=loaded_branch_ids)
    if str(g.get("nonsyn_recode", "no")).strip().lower() == "3di20":
        sequence.write_alignment("csubst_alignment_3di.fa", mode="nsy", g=g, branch_ids=loaded_branch_ids)

    tree.write_tree(g["tree"])
    tree.plot_branch_category(g, file_base="csubst_branch_id", label="all")
    tree.plot_branch_category(g, file_base="csubst_branch_id_leaf", label="leaf")
    tree.plot_branch_category(g, file_base="csubst_branch_id_nolabel", label="no")

    if bool(g.get("plot_state_aa", False)):
        _plot_state_tree_in_directory(
            output_dir="csubst_plot_state_aa",
            state=aa_state,
            orders=aa_orders,
            mode=aa_mode,
            g=g,
        )
    if bool(g.get("plot_state_codon", False)):
        _plot_state_tree_in_directory(
            output_dir="csubst_plot_state_codon",
            state=g["state_cdn"],
            orders=g["codon_orders"],
            mode="codon",
            g=g,
        )

    _write_branch_maps(g)

    elapsed_time = int(time.time() - start)
    print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)

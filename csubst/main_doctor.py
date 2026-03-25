import json
import os
import re

import pandas as pd

from csubst import foreground
from csubst import output_manifest
from csubst import parser_iqtree
from csubst import runtime
from csubst import sequence
from csubst import tree
from csubst import ete


_SEVERITY_RANK = {
    "info": 0,
    "warning": 1,
    "error": 2,
}


def _add_check(rows, check_name, severity, status, message, path=""):
    rows.append(
        {
            "check_name": str(check_name),
            "severity": str(severity),
            "status": str(status),
            "path": "" if path is None else str(path),
            "message": str(message),
        }
    )
    return rows


def _pass(rows, check_name, message, path=""):
    return _add_check(rows, check_name, "info", "pass", message, path=path)


def _warn(rows, check_name, message, path=""):
    return _add_check(rows, check_name, "warning", "warn", message, path=path)


def _fail(rows, check_name, message, path=""):
    return _add_check(rows, check_name, "error", "fail", message, path=path)


def _skip(rows, check_name, message, path=""):
    return _add_check(rows, check_name, "info", "skip", message, path=path)


def _normalize_path(value):
    if value is None:
        return ""
    return str(value).strip()


def _check_file_presence(rows, check_name, path, required=True, kind="file"):
    path_txt = _normalize_path(path)
    if path_txt == "":
        if required:
            _fail(rows, check_name, "Required {} path was not provided.".format(kind))
            return False
        _warn(rows, check_name, "Optional {} path was not provided.".format(kind))
        return False
    if not os.path.exists(path_txt):
        _fail(rows, check_name, "{} was not found.".format(kind.capitalize()), path=path_txt)
        return False
    _pass(rows, check_name, "{} exists.".format(kind.capitalize()), path=path_txt)
    return True


def _summarize_alignment(rows, check_name_prefix, path):
    if not _check_file_presence(rows, check_name_prefix + "_path", path, required=True, kind="alignment file"):
        return None
    try:
        seqs = sequence.read_fasta(path)
    except Exception as exc:
        _fail(rows, check_name_prefix + "_parse", "Failed to parse FASTA: {}".format(exc), path=path)
        return None
    if len(seqs) == 0:
        _fail(rows, check_name_prefix + "_records", "Alignment contained no FASTA records.", path=path)
        return None
    names = list(seqs.keys())
    lengths = sorted(set([len(seq) for seq in seqs.values()]))
    _pass(
        rows,
        check_name_prefix + "_records",
        "Parsed {:,} sequences.".format(len(names)),
        path=path,
    )
    if len(lengths) != 1:
        _fail(
            rows,
            check_name_prefix + "_lengths",
            "Sequences do not share the same length: {}.".format(",".join([str(v) for v in lengths])),
            path=path,
        )
        return None
    aln_len = int(lengths[0])
    if aln_len <= 0:
        _fail(rows, check_name_prefix + "_lengths", "Alignment length should be > 0.", path=path)
        return None
    _pass(rows, check_name_prefix + "_lengths", "Alignment length = {:,} nt.".format(aln_len), path=path)
    if (aln_len % 3) != 0:
        _fail(rows, check_name_prefix + "_codon_frame", "Alignment length is not a multiple of 3.", path=path)
    else:
        _pass(
            rows,
            check_name_prefix + "_codon_frame",
            "Alignment length is a multiple of 3 ({:,} codons).".format(aln_len // 3),
            path=path,
        )
    return {
        "path": str(path),
        "num_sequences": int(len(names)),
        "alignment_length": int(aln_len),
        "names": names,
    }


def _summarize_tree(rows, path):
    if not _check_file_presence(rows, "tree_path", path, required=True, kind="tree file"):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            rooted_newick = handle.read()
        tree_obj = ete.PhyloNode(rooted_newick, format=1)
        tree_obj = tree.standardize_node_names(tree_obj)
        tree_obj = tree.add_numerical_node_labels(tree_obj)
    except Exception as exc:
        _fail(rows, "tree_parse", "Failed to parse rooted tree: {}".format(exc), path=path)
        return None
    child_count = len(ete.get_children(tree_obj))
    if child_count != 2:
        _fail(
            rows,
            "tree_rooted",
            "Root node should have exactly 2 children, observed {}.".format(child_count),
            path=path,
        )
    else:
        _pass(rows, "tree_rooted", "Tree root has exactly 2 children.", path=path)
    leaf_names = ete.get_leaf_names(tree_obj)
    if len(set(leaf_names)) != len(leaf_names):
        _fail(rows, "tree_leaf_names", "Leaf names are not unique.", path=path)
    else:
        _pass(rows, "tree_leaf_names", "Tree contains {:,} unique leaves.".format(len(leaf_names)), path=path)
    _pass(
        rows,
        "tree_nodes",
        "Tree contains {:,} total nodes.".format(len(list(tree_obj.traverse()))),
        path=path,
    )
    return {
        "path": str(path),
        "tree": tree_obj,
        "leaf_names": leaf_names,
    }


def _check_tree_alignment_consistency(rows, tree_summary, alignment_summary):
    if (tree_summary is None) or (alignment_summary is None):
        return
    tree_leaf_set = set(tree_summary["leaf_names"])
    aln_name_set = set(alignment_summary["names"])
    if tree_leaf_set == aln_name_set:
        _pass(
            rows,
            "tree_alignment_taxa",
            "Tree leaves and alignment headers match exactly ({:,} taxa).".format(len(tree_leaf_set)),
        )
        return
    only_tree = sorted(tree_leaf_set.difference(aln_name_set))
    only_aln = sorted(aln_name_set.difference(tree_leaf_set))
    pieces = list()
    if len(only_tree):
        pieces.append("tree-only={}".format(",".join(only_tree[:10])))
    if len(only_aln):
        pieces.append("alignment-only={}".format(",".join(only_aln[:10])))
    _fail(rows, "tree_alignment_taxa", "Taxon mismatch detected: {}.".format("; ".join(pieces)))


def _summarize_foreground(rows, g, tree_summary):
    fg_path = g.get("foreground", None)
    if fg_path is None:
        _skip(rows, "foreground_path", "Foreground file was not provided; skipping foreground-specific checks.")
        return None
    if not _check_file_presence(rows, "foreground_path", fg_path, required=True, kind="foreground file"):
        return None
    try:
        fg_df = foreground.read_foreground_file(g)
    except Exception as exc:
        _fail(rows, "foreground_parse", "Failed to parse foreground file: {}".format(exc), path=fg_path)
        return None
    trait_names = fg_df.columns[1:].tolist()
    if len(trait_names) == 0:
        _fail(rows, "foreground_traits", "Foreground file contained no trait columns.", path=fg_path)
        return None
    _pass(
        rows,
        "foreground_traits",
        "Parsed {:,} trait columns: {}.".format(len(trait_names), ",".join([str(t) for t in trait_names])),
        path=fg_path,
    )
    if tree_summary is None:
        _skip(rows, "foreground_tree_mapping", "Skipping foreground-to-tree mapping because tree parsing failed.", path=fg_path)
        return fg_df
    leaf_names = list(tree_summary["leaf_names"])
    for trait_name in trait_names:
        trait_values = fg_df.loc[:, trait_name]
        is_foreground = ~trait_values.map(foreground._is_background_trait_value)
        trait_patterns = fg_df.loc[is_foreground, "name"].astype(str).tolist()
        if len(trait_patterns) == 0:
            _warn(rows, "foreground_trait_" + str(trait_name), 'Trait "{}" had no non-zero foreground rows.'.format(trait_name), path=fg_path)
            continue
        matched_leaves = set()
        unmatched_patterns = list()
        invalid_patterns = list()
        for pattern in trait_patterns:
            try:
                regex = re.compile("^" + pattern + "$")
            except re.error as exc:
                invalid_patterns.append("{} ({})".format(pattern, exc))
                continue
            matches = [leaf_name for leaf_name in leaf_names if regex.match(leaf_name)]
            if len(matches) == 0:
                unmatched_patterns.append(pattern)
                continue
            matched_leaves.update(matches)
        if len(invalid_patterns):
            _fail(
                rows,
                "foreground_trait_" + str(trait_name) + "_regex",
                'Trait "{}" contained invalid regex patterns: {}.'.format(trait_name, "; ".join(invalid_patterns[:5])),
                path=fg_path,
            )
        if len(matched_leaves) == 0:
            _fail(
                rows,
                "foreground_trait_" + str(trait_name) + "_matches",
                'Trait "{}" did not match any tree leaves.'.format(trait_name),
                path=fg_path,
            )
            continue
        if len(unmatched_patterns):
            _warn(
                rows,
                "foreground_trait_" + str(trait_name) + "_matches",
                'Trait "{}" matched {:,} leaves but had unmatched patterns: {}.'.format(
                    trait_name,
                    len(matched_leaves),
                    ",".join(unmatched_patterns[:5]),
                ),
                path=fg_path,
            )
        else:
            _pass(
                rows,
                "foreground_trait_" + str(trait_name) + "_matches",
                'Trait "{}" matched {:,} leaves.'.format(trait_name, len(matched_leaves)),
                path=fg_path,
            )
    return fg_df


def _summarize_iqtree_paths(rows, g):
    files = {
        "iqtree_treefile": g.get("iqtree_treefile", ""),
        "iqtree_state": g.get("iqtree_state", ""),
        "iqtree_rate": g.get("iqtree_rate", ""),
        "iqtree_iqtree": g.get("iqtree_iqtree", ""),
        "iqtree_log": g.get("iqtree_log", ""),
    }
    missing = list()
    for key, path in files.items():
        path_txt = _normalize_path(path)
        if path_txt == "":
            _warn(rows, key, "Path was empty.", path=path_txt)
            missing.append(key)
            continue
        if os.path.exists(path_txt):
            _pass(rows, key, "File exists.", path=path_txt)
        else:
            _warn(rows, key, "File is missing and may need to be generated by IQ-TREE.", path=path_txt)
            missing.append(key)
    return missing


def _check_iqtree_executable(rows, g):
    if not bool(g.get("check_iqtree_exe", True)):
        _skip(rows, "iqtree_exe_check", "Skipped IQ-TREE executable check (--check_iqtree_exe no).")
        return
    try:
        parser_iqtree.check_iqtree_dependency(g)
    except Exception as exc:
        _fail(rows, "iqtree_exe_check", "IQ-TREE executable check failed: {}".format(exc), path=g.get("iqtree_exe", ""))
        return
    _pass(rows, "iqtree_exe_check", "IQ-TREE executable is callable.", path=g.get("iqtree_exe", ""))


def _check_3di_related_inputs(rows, g):
    if str(g.get("nonsyn_recode", "no")).strip().lower() != "3di20":
        _pass(rows, "nonsyn_recode", 'nonsyn_recode="{}"; 3Di-specific checks are not required.'.format(g.get("nonsyn_recode", "no")))
        return
    _pass(rows, "nonsyn_recode", 'nonsyn_recode="3di20"; running 3Di-specific checks.')
    prostt5_local_dir = _normalize_path(g.get("prostt5_local_dir", ""))
    if prostt5_local_dir != "":
        if os.path.isdir(prostt5_local_dir):
            _pass(rows, "prostt5_local_dir", "Local ProstT5 directory exists.", path=prostt5_local_dir)
        else:
            _fail(rows, "prostt5_local_dir", "Local ProstT5 directory was not found.", path=prostt5_local_dir)
    else:
        _pass(rows, "prostt5_model", 'Using remote/local model identifier "{}".'.format(g.get("prostt5_model", "")))
    cache_mode = str(g.get("sa_state_cache", "auto")).strip().lower()
    cache_file = _normalize_path(g.get("sa_state_cache_file", ""))
    if cache_mode == "yes":
        if os.path.exists(cache_file):
            _pass(rows, "sa_state_cache_file", 'Required 3Di state cache exists for --sa_state_cache yes.', path=cache_file)
        else:
            _fail(rows, "sa_state_cache_file", 'Required 3Di state cache is missing for --sa_state_cache yes.', path=cache_file)
    elif cache_mode == "auto":
        if os.path.exists(cache_file):
            _pass(rows, "sa_state_cache_file", "Reusable 3Di state cache exists.", path=cache_file)
        else:
            _skip(rows, "sa_state_cache_file", "3Di state cache does not exist yet; it will be computed on demand.", path=cache_file)
    else:
        _skip(rows, "sa_state_cache_file", '3Di state cache is disabled (--sa_state_cache no).', path=cache_file)


def _write_doctor_outputs(rows, g):
    summary_tsv = runtime.output_path(g, "doctor_summary.tsv")
    summary_json = runtime.output_path(g, "doctor_summary.json")
    df = pd.DataFrame(rows)
    if df.shape[0] > 0:
        df.loc[:, "_severity_rank"] = df["severity"].map(lambda value: _SEVERITY_RANK.get(value, -1))
        df = df.sort_values(
            by=["_severity_rank", "status", "check_name", "path"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
        df = df.drop(columns=["_severity_rank"])
    df.to_csv(summary_tsv, sep="\t", index=False)
    summary_payload = {
        "counts": {
            "pass": int((df["status"] == "pass").sum()) if df.shape[0] > 0 else 0,
            "warn": int((df["status"] == "warn").sum()) if df.shape[0] > 0 else 0,
            "fail": int((df["status"] == "fail").sum()) if df.shape[0] > 0 else 0,
            "skip": int((df["status"] == "skip").sum()) if df.shape[0] > 0 else 0,
        },
        "rows": df.to_dict(orient="records"),
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=True)
    print("Writing doctor summary TSV: {}".format(summary_tsv), flush=True)
    print("Writing doctor summary JSON: {}".format(summary_json), flush=True)
    return df, summary_tsv, summary_json


def _write_doctor_output_manifest(g, summary_tsv, summary_json):
    manifest_rows = list()
    output_manifest.add_output_manifest_row(
        manifest_rows=manifest_rows,
        output_path=summary_tsv,
        output_kind="doctor_summary_tsv",
        base_dir=g["outdir"],
    )
    output_manifest.add_output_manifest_row(
        manifest_rows=manifest_rows,
        output_path=summary_json,
        output_kind="doctor_summary_json",
        base_dir=g["outdir"],
    )
    output_manifest.add_output_manifest_row(
        manifest_rows=manifest_rows,
        output_path=g["log_file"],
        output_kind="doctor_log",
        base_dir=g["outdir"],
    )
    manifest_path = runtime.output_path(g, "outputs.tsv")
    manifest_path = output_manifest.write_output_manifest(
        manifest_rows=manifest_rows,
        manifest_path=manifest_path,
        base_dir=g["outdir"],
    )
    print("Writing doctor output manifest: {}".format(manifest_path), flush=True)
    return manifest_path


def _should_fail(df, fail_level):
    level = str(fail_level).strip().lower()
    if level == "never":
        return False
    if df.shape[0] == 0:
        return False
    max_rank = max([_SEVERITY_RANK.get(sev, 0) for sev in df["severity"].tolist()])
    if level == "warning":
        return max_rank >= _SEVERITY_RANK["warning"]
    return max_rank >= _SEVERITY_RANK["error"]


def main_doctor(g):
    g = runtime.ensure_output_layout(g, create_dir=True)
    rows = list()
    alignment_summary = _summarize_alignment(rows, "alignment", g.get("alignment_file", ""))
    tree_summary = _summarize_tree(rows, g.get("rooted_tree_file", ""))
    _check_tree_alignment_consistency(rows, tree_summary=tree_summary, alignment_summary=alignment_summary)
    _summarize_foreground(rows, g=g, tree_summary=tree_summary)
    _summarize_iqtree_paths(rows, g=g)
    _check_iqtree_executable(rows, g=g)
    _check_3di_related_inputs(rows, g=g)
    df, summary_tsv, summary_json = _write_doctor_outputs(rows=rows, g=g)
    if bool(g.get("output_manifest", True)):
        _write_doctor_output_manifest(g=g, summary_tsv=summary_tsv, summary_json=summary_json)
    num_pass = int((df["status"] == "pass").sum()) if df.shape[0] > 0 else 0
    num_warn = int((df["status"] == "warn").sum()) if df.shape[0] > 0 else 0
    num_fail = int((df["status"] == "fail").sum()) if df.shape[0] > 0 else 0
    num_skip = int((df["status"] == "skip").sum()) if df.shape[0] > 0 else 0
    print("Doctor summary: pass={}, warn={}, fail={}, skip={}".format(num_pass, num_warn, num_fail, num_skip), flush=True)
    if _should_fail(df=df, fail_level=g.get("doctor_fail_level", "error")):
        failing = df.loc[df["status"] != "pass", ["check_name", "message"]]
        message_bits = list()
        for _, row in failing.head(5).iterrows():
            message_bits.append("{}: {}".format(row["check_name"], row["message"]))
        raise ValueError("Doctor checks found issues: {}".format(" | ".join(message_bits)))

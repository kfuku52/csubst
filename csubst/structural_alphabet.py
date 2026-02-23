from collections import OrderedDict
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from csubst import ete
from csubst import sequence
from csubst import tree


_THREEDI_STATE_ORDERS = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
_THREEDI_STATE_SET = frozenset(_THREEDI_STATE_ORDERS.tolist())


def is_3di_recode(g):
    return str(g.get("nonsyn_recode", "no")).strip().lower() == "3di20"


def get_3di_state_orders():
    return _THREEDI_STATE_ORDERS.copy()


def _normalize_branch_ids(branch_ids):
    if branch_ids is None:
        return np.array([], dtype=np.int64)
    arr = np.asarray(branch_ids, dtype=object)
    arr = np.atleast_1d(arr).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    out = list()
    for value in arr.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("selected_branch_ids should be integer-like.")
        if isinstance(value, (int, np.integer)):
            out.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            if (not np.isfinite(value)) or (not float(value).is_integer()):
                raise ValueError("selected_branch_ids should be integer-like.")
            out.append(int(value))
            continue
        value_txt = str(value).strip()
        if value_txt == "":
            raise ValueError("selected_branch_ids should be integer-like.")
        try:
            out.append(int(value_txt))
        except ValueError as exc:
            raise ValueError("selected_branch_ids should be integer-like.") from exc
    return np.array(out, dtype=np.int64)


def _iter_target_branch_ids(tree_obj, selected_branch_ids=None):
    selected_set = None
    if selected_branch_ids is not None:
        selected_set = set(int(v) for v in _normalize_branch_ids(selected_branch_ids).tolist())
    out = list()
    for node in tree_obj.traverse():
        branch_id = int(ete.get_prop(node, "numerical_label"))
        if (selected_set is not None) and (branch_id not in selected_set):
            continue
        out.append(branch_id)
    return out


def _state_pep_to_ml_aa_alignment_by_branch(state_pep, g, selected_branch_ids=None):
    aa_orders = np.asarray(g["amino_acid_orders"], dtype=object).reshape(-1)
    if aa_orders.shape[0] != int(state_pep.shape[2]):
        txt = "amino_acid_orders length ({}) did not match state_pep state axis ({})."
        raise ValueError(txt.format(aa_orders.shape[0], state_pep.shape[2]))
    float_tol = float(g.get("float_tol", 0))
    site_max = state_pep.max(axis=2)
    site_argmax = state_pep.argmax(axis=2)
    out = OrderedDict()
    for branch_id in _iter_target_branch_ids(g["tree"], selected_branch_ids=selected_branch_ids):
        symbols = aa_orders[site_argmax[branch_id, :]]
        missing = site_max[branch_id, :] < float_tol
        if np.any(missing):
            symbols = symbols.copy()
            symbols[missing] = "-"
        out[int(branch_id)] = "".join(symbols.tolist())
    return out


def _sanitize_aa_sequence_for_prostt5(seq):
    seq = str(seq).strip().upper()
    if seq == "":
        return ""
    out = list()
    for aa in seq:
        if aa in _THREEDI_STATE_SET:
            out.append(aa)
        elif aa == "-":
            continue
        else:
            out.append("X")
    return "".join(out)


def _import_prostt5_transformers():
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ModuleNotFoundError as exc:
        txt = "ProstT5 backend requires transformers. Install it before using --nonsyn_recode 3di20."
        raise ImportError(txt) from exc
    return T5Tokenizer, T5ForConditionalGeneration


def _resolve_prostt5_model_options(g):
    model_name = str(g.get("prostt5_model", "Rostlab/ProstT5")).strip()
    if model_name == "":
        raise ValueError("--prostt5_model should be non-empty.")
    local_dir = str(g.get("prostt5_local_dir", "")).strip()
    if local_dir != "":
        if not os.path.isdir(local_dir):
            txt = "--prostt5_local_dir does not exist or is not a directory: {}"
            raise ValueError(txt.format(local_dir))
    no_download = bool(g.get("prostt5_no_download", False))
    return model_name, local_dir, no_download


def _load_prostt5_from_local_only(source, tokenizer_cls, model_cls):
    tokenizer = tokenizer_cls.from_pretrained(
        source,
        do_lower_case=False,
        local_files_only=True,
    )
    model = model_cls.from_pretrained(
        source,
        local_files_only=True,
    )
    return tokenizer, model


def _load_or_download_prostt5(g, tokenizer_cls, model_cls):
    model_name, local_dir, no_download = _resolve_prostt5_model_options(g=g)
    if local_dir != "":
        try:
            tokenizer, model = _load_prostt5_from_local_only(
                source=local_dir,
                tokenizer_cls=tokenizer_cls,
                model_cls=model_cls,
            )
            return tokenizer, model, local_dir
        except Exception as local_exc:
            if no_download:
                txt = (
                    "ProstT5 model files were not found in --prostt5_local_dir. "
                    "Download once with internet access or disable --prostt5_no_download. "
                    "prostt5_local_dir={}"
                )
                raise RuntimeError(txt.format(local_dir)) from local_exc
            print(
                "ProstT5 local files were not found in --prostt5_local_dir; downloading model files.",
                flush=True,
            )
            tokenizer = tokenizer_cls.from_pretrained(
                model_name,
                do_lower_case=False,
            )
            model = model_cls.from_pretrained(model_name)
            tokenizer.save_pretrained(local_dir)
            model.save_pretrained(local_dir)
            return tokenizer, model, local_dir
    try:
        tokenizer, model = _load_prostt5_from_local_only(
            source=model_name,
            tokenizer_cls=tokenizer_cls,
            model_cls=model_cls,
        )
        return tokenizer, model, model_name
    except Exception as local_exc:
        if no_download:
            txt = (
                "ProstT5 model files were not found locally. "
                "Download once with internet access or disable --prostt5_no_download. "
                "model_source={}"
            )
            raise RuntimeError(txt.format(model_name)) from local_exc
        print(
            "ProstT5 local cache was not found; downloading model files (first run only).",
            flush=True,
        )
        tokenizer = tokenizer_cls.from_pretrained(
            model_name,
            do_lower_case=False,
        )
        model = model_cls.from_pretrained(model_name)
        return tokenizer, model, model_name


def ensure_prostt5_model_files(g, tokenizer_cls=None, model_cls=None):
    if tokenizer_cls is None or model_cls is None:
        tokenizer_cls, model_cls = _import_prostt5_transformers()
    tokenizer, model, model_source = _load_or_download_prostt5(
        g=g,
        tokenizer_cls=tokenizer_cls,
        model_cls=model_cls,
    )
    # Drop references immediately; inspect pre-download should not keep them resident.
    del tokenizer
    del model
    return model_source


def _load_prostt5_components(g):
    try:
        import torch
    except ModuleNotFoundError as exc:
        txt = "ProstT5 backend requires torch. Install it before using --nonsyn_recode 3di20."
        raise ImportError(txt) from exc
    T5Tokenizer, T5ForConditionalGeneration = _import_prostt5_transformers()
    tokenizer, model, model_source = _load_or_download_prostt5(
        g=g,
        tokenizer_cls=T5Tokenizer,
        model_cls=T5ForConditionalGeneration,
    )
    device_opt = str(g.get("prostt5_device", "auto")).strip().lower()
    if device_opt == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_opt in ["cpu", "cuda"]:
        device = device_opt
    else:
        raise ValueError('--prostt5_device should be one of "auto", "cpu", "cuda".')
    model = model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _resolve_prostt5_model_source(g):
    model_name, local_dir, no_download = _resolve_prostt5_model_options(g=g)
    if local_dir != "":
        model_source = local_dir
    else:
        model_source = model_name
    return model_source, no_download


def predict_3di_with_prostt5(aa_sequences, g):
    torch, tokenizer, model, device = _load_prostt5_components(g=g)
    out = dict()
    with torch.no_grad():
        for seq_id, raw_seq in aa_sequences.items():
            seq = _sanitize_aa_sequence_for_prostt5(raw_seq)
            if seq == "":
                out[seq_id] = ""
                continue
            prompt = "<AA2fold> " + " ".join(list(seq))
            batch = tokenizer(prompt, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            pred_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                num_beams=1,
                do_sample=False,
                min_new_tokens=int(len(seq)),
                max_new_tokens=int(len(seq)),
            )
            pred = tokenizer.decode(pred_ids[0], skip_special_tokens=True).replace(" ", "")
            pred = str(pred).strip().upper()
            if len(pred) != len(seq):
                txt = "ProstT5 output length mismatch for sequence {}: input={}, output={}."
                raise ValueError(txt.format(seq_id, len(seq), len(pred)))
            invalid = sorted(list(set([ch for ch in pred if ch not in _THREEDI_STATE_SET])))
            if len(invalid) > 0:
                txt = "Unsupported 3Di symbol(s) in ProstT5 output for sequence {}: {}"
                raise ValueError(txt.format(seq_id, ",".join(invalid)))
            out[seq_id] = pred
    return out


def _inject_alignment_gaps(reference_aligned_seq, ungapped_pred_seq, seq_id):
    reference_aligned_seq = str(reference_aligned_seq).strip().upper()
    ungapped_pred_seq = str(ungapped_pred_seq).strip().upper()
    nongap_count = sum([1 for c in reference_aligned_seq if c != "-"])
    if nongap_count != len(ungapped_pred_seq):
        txt = "ProstT5 output length mismatch for sequence {} after gap projection: expected {}, got {}."
        raise ValueError(txt.format(seq_id, nongap_count, len(ungapped_pred_seq)))
    out = list()
    k = 0
    for aa in reference_aligned_seq:
        if aa == "-":
            out.append("-")
            continue
        out.append(ungapped_pred_seq[k])
        k += 1
    if k != len(ungapped_pred_seq):
        txt = "Gap projection mismatch for sequence {}: consumed {}, expected {}."
        raise ValueError(txt.format(seq_id, k, len(ungapped_pred_seq)))
    return "".join(out)


def _build_codon_to_aa_lookup(g):
    codon_table = g.get("codon_table", None)
    if codon_table is None:
        raise ValueError("codon_table is required for 3Di generation.")
    codon_to_aa = dict()
    for aa, codon in codon_table:
        aa = str(aa)
        codon = str(codon).upper().replace("U", "T")
        if aa == "*":
            continue
        codon_to_aa[codon] = aa
    if len(codon_to_aa) == 0:
        raise ValueError("No sense codons were found in codon_table.")
    return codon_to_aa


def _translate_codon_aligned_sequence_to_aa(seq_txt, codon_to_aa):
    seq_txt = str(seq_txt).strip().upper().replace("U", "T")
    if (len(seq_txt) % 3) != 0:
        txt = "Input CDS sequence length should be a multiple of 3 (length={})."
        raise ValueError(txt.format(len(seq_txt)))
    aa_out = list()
    for i in range(0, len(seq_txt), 3):
        codon = seq_txt[i : i + 3]
        if codon == "---":
            aa_out.append("-")
            continue
        if "-" in codon:
            aa_out.append("X")
            continue
        if any([base not in {"A", "C", "G", "T"} for base in codon]):
            aa_out.append("X")
            continue
        aa = codon_to_aa.get(codon, None)
        if aa is None:
            aa_out.append("X")
        else:
            aa_out.append(str(aa))
    return "".join(aa_out)


def _write_fasta_dict(path, seq_dict):
    lines = list()
    for name, seq in seq_dict.items():
        lines.append(">" + str(name))
        lines.append(str(seq))
    with open(path, "w") as f:
        if len(lines) > 0:
            f.write("\n".join(lines) + "\n")


def _get_full_cds_alignment_path(g):
    path = str(g.get("full_cds_alignment_file", "")).strip()
    if path == "":
        path = str(g.get("alignment_file", "")).strip()
    if path == "":
        raise ValueError("3Di mode requires --full_cds_alignment_file.")
    return path


def build_tip_aa_alignment_from_full_cds(g):
    alignment_path = _get_full_cds_alignment_path(g)
    seq_dict = sequence.read_fasta(alignment_path)
    if len(seq_dict) == 0:
        raise ValueError("full CDS alignment is empty.")
    codon_to_aa = _build_codon_to_aa_lookup(g=g)
    tip_names = [leaf.name for leaf in ete.iter_leaves(g["tree"])]
    missing = sorted([name for name in tip_names if name not in seq_dict])
    if len(missing) > 0:
        txt = "Tip sequence(s) missing in full CDS alignment: {}"
        missing_txt = ",".join(missing[:10])
        if len(missing) > 10:
            missing_txt += ",..."
        raise ValueError(txt.format(missing_txt))
    aa_by_tip = OrderedDict()
    for name in tip_names:
        aa_by_tip[str(name)] = _translate_codon_aligned_sequence_to_aa(
            seq_txt=seq_dict[name],
            codon_to_aa=codon_to_aa,
        )
    lengths = sorted(set([len(seq) for seq in aa_by_tip.values()]))
    if len(lengths) != 1:
        raise ValueError("Translated AA alignment should have equal sequence lengths.")
    return aa_by_tip


def build_tip_3di_alignment_from_full_cds(
    g,
    predictor=None,
    output_path="csubst_alignment_3di_tip.fa",
):
    if predictor is None:
        predictor = predict_3di_with_prostt5
    aa_by_tip = build_tip_aa_alignment_from_full_cds(g=g)
    aa_ungapped = {
        name: _sanitize_aa_sequence_for_prostt5(seq.replace("-", ""))
        for name, seq in aa_by_tip.items()
    }
    pred_ungapped = predictor(aa_ungapped, g)
    out = OrderedDict()
    for name, aa_seq in aa_by_tip.items():
        pred_seq = pred_ungapped.get(name, "")
        out[name] = _inject_alignment_gaps(
            reference_aligned_seq=aa_seq,
            ungapped_pred_seq=pred_seq,
            seq_id=name,
        )
    if output_path is not None and str(output_path).strip() != "":
        _write_fasta_dict(path=str(output_path), seq_dict=out)
        print("Writing sequence alignment: {}".format(os.path.abspath(output_path)), flush=True)
    return out


def _build_state_tensor_from_tip_alignment(
    tip_3di_by_name,
    tree_obj,
    state_orders,
    dtype,
):
    num_node = len(list(tree_obj.traverse()))
    seq_lengths = sorted(set([len(str(v)) for v in tip_3di_by_name.values()]))
    if len(seq_lengths) != 1:
        raise ValueError("Tip 3Di alignment should have equal sequence lengths.")
    num_site = int(seq_lengths[0]) if len(seq_lengths) > 0 else 0
    lookup = {str(state): i for i, state in enumerate(np.asarray(state_orders, dtype=object).tolist())}
    state = np.zeros((num_node, num_site, len(lookup)), dtype=dtype)
    for leaf in ete.iter_leaves(tree_obj):
        branch_id = int(ete.get_prop(leaf, "numerical_label"))
        seq_txt = tip_3di_by_name.get(leaf.name, None)
        if seq_txt is None:
            continue
        seq_txt = str(seq_txt)
        if len(seq_txt) != num_site:
            txt = "Tip 3Di sequence length mismatch for {}: expected {}, got {}."
            raise ValueError(txt.format(leaf.name, num_site, len(seq_txt)))
        for i, ch in enumerate(seq_txt):
            idx = lookup.get(ch, None)
            if idx is None:
                continue
            state[branch_id, i, idx] = 1
    return state


def _run_iqtree_direct_3di(g, tip_alignment_path):
    output_prefix = os.path.abspath(str(tip_alignment_path))
    path_treefile = output_prefix + ".treefile"
    path_state = output_prefix + ".state"
    path_iqtree = output_prefix + ".iqtree"
    path_log = output_prefix + ".log"
    required = [path_treefile, path_state, path_iqtree, path_log]
    all_exist = all([os.path.exists(path) for path in required])
    redo = bool(g.get("iqtree_redo", False))
    if all_exist and (not redo):
        print("Direct 3Di IQ-TREE intermediate files exist. Skipping rerun.", flush=True)
        return {
            "treefile": path_treefile,
            "state": path_state,
            "iqtree": path_iqtree,
            "log": path_log,
        }
    file_tree = "tmp.csubst.3di.nwk"
    tree.write_tree(g["rooted_tree"], outfile=file_tree, add_numerical_label=False)
    try:
        model = str(g.get("sa_iqtree_model", "GTR20")).strip()
        if model == "":
            raise ValueError("--sa_iqtree_model should be non-empty.")
        command = [
            g["iqtree_exe"],
            "-s",
            str(tip_alignment_path),
            "-te",
            file_tree,
            "-m",
            model,
            "--seqtype",
            "AA",
            "--threads-max",
            str(int(g.get("threads", 1))),
            "-T",
            "AUTO",
            "--ancestral",
            "--redo",
        ]
        print("Starting direct 3Di IQ-TREE ancestral reconstruction.", flush=True)
        run_iqtree = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
        if run_iqtree.returncode != 0:
            txt = "Direct 3Di IQ-TREE did not finish safely (exit code {})."
            raise AssertionError(txt.format(run_iqtree.returncode))
    finally:
        if os.path.exists(file_tree):
            os.remove(file_tree)
    ckp_path = str(tip_alignment_path) + ".ckp.gz"
    if os.path.exists(ckp_path):
        os.remove(ckp_path)
    return {
        "treefile": path_treefile,
        "state": path_state,
        "iqtree": path_iqtree,
        "log": path_log,
    }


def _read_direct_3di_state_tensor(g, paths, tip_3di_by_name, selected_branch_ids=None):
    with open(paths["treefile"]) as f:
        direct_tree_newick = f.read()
    direct_tree = ete.PhyloNode(direct_tree_newick, format=1)
    direct_tree = tree.standardize_node_names(direct_tree)
    is_consistent = tree.is_consistent_tree(tree1=direct_tree, tree2=g["rooted_tree"])
    if not is_consistent:
        raise ValueError("Direct 3Di IQ-TREE treefile is inconsistent with --rooted_tree_file.")
    direct_tree = tree.transfer_root(tree_to=direct_tree, tree_from=g["rooted_tree"], verbose=False)
    direct_tree = tree.add_numerical_node_labels(direct_tree)
    state_orders = get_3di_state_orders()
    out_dtype = g.get("float_type", np.float64)
    state_tensor = _build_state_tensor_from_tip_alignment(
        tip_3di_by_name=tip_3di_by_name,
        tree_obj=direct_tree,
        state_orders=state_orders,
        dtype=out_dtype,
    )
    state_table = pd.read_csv(paths["state"], sep="\t", index_col=False, header=0, comment="#")
    if state_table.shape[0] == 0:
        return state_tensor, state_orders
    required_columns = ["Node", "Site", "State"]
    missing_columns = [col for col in required_columns if col not in state_table.columns]
    if len(missing_columns) > 0:
        txt = "Direct 3Di .state file is missing required column(s): {}."
        raise ValueError(txt.format(",".join(missing_columns)))
    state_columns = state_table.columns[3:]
    state_lookup = {str(s): i for i, s in enumerate(state_orders.tolist())}
    col_to_state = dict()
    for col in state_columns:
        symbol = str(col).replace("p_", "").strip().upper()
        if symbol in state_lookup:
            col_to_state[str(col)] = int(state_lookup[symbol])
    if len(col_to_state) == 0:
        raise ValueError("No recognized 3Di state columns were found in direct .state file.")
    site_values = pd.to_numeric(state_table.loc[:, "Site"], errors="coerce")
    site_values_arr = site_values.to_numpy(dtype=float)
    if not np.isfinite(site_values_arr).all():
        raise ValueError("Non-numeric Site value(s) were found in direct .state file.")
    rounded = np.round(site_values_arr).astype(np.int64)
    state_table = state_table.copy()
    state_table.loc[:, "Site"] = rounded
    unique_sites = np.sort(state_table.loc[:, "Site"].unique())
    num_site = int(state_tensor.shape[1])
    if unique_sites.shape[0] != num_site:
        txt = "Direct 3Di site count mismatch: expected {}, observed {}."
        raise ValueError(txt.format(num_site, unique_sites.shape[0]))
    site_index_by_label = {int(site_label): i for i, site_label in enumerate(unique_sites.tolist())}
    node_name_to_id = dict()
    for node in direct_tree.traverse():
        node_name_to_id[str(node.name)] = int(ete.get_prop(node, "numerical_label"))
    for node_name, tmp in state_table.groupby("Node", sort=False):
        node_name = str(node_name)
        node_id = node_name_to_id.get(node_name, None)
        if node_id is None:
            continue
        row_sites = tmp.loc[:, "Site"].to_numpy(dtype=np.int64, copy=False)
        row_indices = np.array([site_index_by_label[int(v)] for v in row_sites], dtype=np.int64)
        for col, state_id in col_to_state.items():
            values = pd.to_numeric(tmp.loc[:, col], errors="coerce").to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0)
            state_tensor[node_id, row_indices, state_id] = values.astype(out_dtype, copy=False)
    state_tensor = np.nan_to_num(state_tensor, copy=False)
    if bool(g.get("ml_anc", False)):
        idxmax = np.argmax(state_tensor, axis=2)
        is_nonmissing = state_tensor.sum(axis=2) != 0
        state_tensor_ml = np.zeros(state_tensor.shape, dtype=bool)
        b_idx, s_idx = np.where(is_nonmissing)
        if b_idx.shape[0] > 0:
            state_tensor_ml[b_idx, s_idx, idxmax[b_idx, s_idx]] = True
        state_tensor = state_tensor_ml
    if selected_branch_ids is not None:
        selected_set = set(int(v) for v in _normalize_branch_ids(selected_branch_ids).tolist())
        if len(selected_set) > 0:
            all_ids = np.arange(state_tensor.shape[0], dtype=np.int64)
            drop_ids = all_ids[~np.isin(all_ids, np.array(sorted(selected_set), dtype=np.int64))]
            if drop_ids.shape[0] > 0:
                state_tensor[drop_ids, :, :] = 0
    return state_tensor, state_orders


def build_3di_state_direct(g, selected_branch_ids=None, predictor=None):
    tip_3di_by_name = build_tip_3di_alignment_from_full_cds(
        g=g,
        predictor=predictor,
        output_path="csubst_alignment_3di_tip.fa",
    )
    iqtree_paths = _run_iqtree_direct_3di(g=g, tip_alignment_path="csubst_alignment_3di_tip.fa")
    state_tensor, state_orders = _read_direct_3di_state_tensor(
        g=g,
        paths=iqtree_paths,
        tip_3di_by_name=tip_3di_by_name,
        selected_branch_ids=selected_branch_ids,
    )
    return state_tensor, state_orders, tip_3di_by_name


def build_3di_state_from_state_pep(g, state_pep, selected_branch_ids=None, predictor=None):
    if predictor is None:
        predictor = predict_3di_with_prostt5
    if state_pep.ndim != 3:
        raise ValueError("state_pep should be a 3D tensor.")
    aa_aligned_by_branch = _state_pep_to_ml_aa_alignment_by_branch(
        state_pep=state_pep,
        g=g,
        selected_branch_ids=selected_branch_ids,
    )
    aa_ungapped = {
        int(branch_id): _sanitize_aa_sequence_for_prostt5(seq.replace("-", ""))
        for branch_id, seq in aa_aligned_by_branch.items()
    }
    pred_ungapped = predictor(aa_ungapped, g)
    aligned_3di = OrderedDict()
    for branch_id, aa_aligned_seq in aa_aligned_by_branch.items():
        pred_seq = pred_ungapped.get(int(branch_id), "")
        aligned_3di[int(branch_id)] = _inject_alignment_gaps(
            reference_aligned_seq=aa_aligned_seq,
            ungapped_pred_seq=pred_seq,
            seq_id=branch_id,
        )
    state_orders = get_3di_state_orders()
    state_lookup = {str(state): i for i, state in enumerate(state_orders.tolist())}
    num_node = int(state_pep.shape[0])
    num_site = int(state_pep.shape[1])
    out_dtype = g.get("float_type", state_pep.dtype)
    state_3di = np.zeros((num_node, num_site, state_orders.shape[0]), dtype=out_dtype)
    for branch_id, seq in aligned_3di.items():
        if len(seq) != num_site:
            txt = "Aligned 3Di sequence length mismatch for branch {}: expected {}, got {}."
            raise ValueError(txt.format(branch_id, num_site, len(seq)))
        for i, ch in enumerate(seq):
            state_id = state_lookup.get(ch, None)
            if state_id is None:
                continue
            state_3di[int(branch_id), i, int(state_id)] = 1
    return state_3di, state_orders, aligned_3di

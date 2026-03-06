from collections import OrderedDict
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from csubst import ete
from csubst import runtime
from csubst import sequence
from csubst import tree


_THREEDI_STATE_ORDERS = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
_THREEDI_STATE_SET = frozenset(_THREEDI_STATE_ORDERS.tolist())
_MORPH_STATE_ORDERS = np.array(list("0123456789ABCDEFGHIJ"), dtype=object)
_THREEDI_TO_MORPH_SYMBOL = {
    str(th): str(mo)
    for th, mo in zip(_THREEDI_STATE_ORDERS.tolist(), _MORPH_STATE_ORDERS.tolist())
}
_MORPH_TO_THREEDI_SYMBOL = {v: k for k, v in _THREEDI_TO_MORPH_SYMBOL.items()}
_ASCII_CODE_SIZE = 256
_THREEDI_ASCII_VALID_MASK = np.zeros(shape=(_ASCII_CODE_SIZE,), dtype=bool)
for _state_symbol in _THREEDI_STATE_ORDERS.tolist():
    _THREEDI_ASCII_VALID_MASK[ord(str(_state_symbol).strip().upper())] = True


def is_3di_recode(g):
    return str(g.get("nonsyn_recode", "no")).strip().lower() == "3di20"


def get_3di_state_orders():
    return _THREEDI_STATE_ORDERS.copy()


def _default_prostt5_cache_file():
    return "csubst_prostt5_cache.tsv"


def _to_ascii_matrix(aligned_sequences):
    num_sequence = int(len(aligned_sequences))
    if num_sequence == 0:
        return np.zeros(shape=(0, 0), dtype=np.uint8)
    seq_lengths = sorted(set([len(str(seq).strip().upper()) for seq in aligned_sequences]))
    if len(seq_lengths) != 1:
        raise ValueError("Aligned sequences should have equal lengths.")
    num_site = int(seq_lengths[0])
    if num_site == 0:
        return np.zeros(shape=(num_sequence, 0), dtype=np.uint8)
    out = np.zeros(shape=(num_sequence, num_site), dtype=np.uint8)
    for i, seq in enumerate(aligned_sequences):
        seq_txt = str(seq).strip().upper()
        try:
            seq_ascii = np.frombuffer(seq_txt.encode("ascii"), dtype=np.uint8)
        except UnicodeEncodeError as exc:
            raise ValueError("Sequence contained non-ASCII symbol(s).") from exc
        if seq_ascii.shape[0] != num_site:
            raise ValueError("Aligned sequences should have equal lengths.")
        out[i, :] = seq_ascii
    return out


def _build_ascii_symbol_lookup(symbol_orders):
    out = np.full(shape=(_ASCII_CODE_SIZE,), fill_value=-1, dtype=np.int16)
    for i, symbol in enumerate(np.asarray(symbol_orders, dtype=object).tolist()):
        txt = str(symbol).strip().upper()
        if len(txt) != 1:
            continue
        code = ord(txt)
        if 0 <= code < _ASCII_CODE_SIZE:
            out[code] = int(i)
    return out


def _convert_3di_symbol_to_morph(symbol):
    symbol = str(symbol).strip().upper()
    if symbol == "-":
        return "-"
    out = _THREEDI_TO_MORPH_SYMBOL.get(symbol, None)
    if out is None:
        txt = "Unsupported 3Di symbol for MORPH conversion: {}"
        raise ValueError(txt.format(symbol))
    return out


def _convert_state_symbol_to_3di(symbol, symbol_mode):
    symbol = str(symbol).strip().upper()
    mode = str(symbol_mode).strip().lower()
    if mode == "morph":
        return _MORPH_TO_THREEDI_SYMBOL.get(symbol, None)
    if mode == "aa":
        if symbol in _THREEDI_STATE_SET:
            return symbol
        return None
    if symbol in _THREEDI_STATE_SET:
        return symbol
    return _MORPH_TO_THREEDI_SYMBOL.get(symbol, None)


def _encode_tip_3di_alignment_for_morph(tip_3di_by_name, output_path=None):
    out = OrderedDict()
    for name, seq in tip_3di_by_name.items():
        out[str(name)] = "".join([_convert_3di_symbol_to_morph(ch) for ch in str(seq).strip().upper()])
    if output_path is not None and str(output_path).strip() != "":
        _write_fasta_dict(path=str(output_path), seq_dict=out)
        print("Writing sequence alignment: {}".format(os.path.abspath(output_path)), flush=True)
    return out


def _normalize_direct_iqtree_model(model):
    model_txt = str(model).strip()
    if model_txt == "":
        raise ValueError("--sa_iqtree_model should be non-empty.")
    if model_txt.upper() == "GTR20":
        return "GTR", True
    return model_txt, False


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
    aa_orders = np.asarray(g["amino_acid_orders"], dtype="U1").reshape(-1)
    if aa_orders.shape[0] != int(state_pep.shape[2]):
        txt = "amino_acid_orders length ({}) did not match state_pep state axis ({})."
        raise ValueError(txt.format(aa_orders.shape[0], state_pep.shape[2]))
    float_tol = float(g.get("float_tol", 0))
    site_max = state_pep.max(axis=2)
    site_argmax = state_pep.argmax(axis=2)
    target_branch_ids = np.array(
        _iter_target_branch_ids(g["tree"], selected_branch_ids=selected_branch_ids),
        dtype=np.int64,
    )
    out = OrderedDict()
    if target_branch_ids.shape[0] == 0:
        return out
    symbols_matrix = aa_orders[site_argmax[target_branch_ids, :]]
    missing_matrix = site_max[target_branch_ids, :] < float_tol
    if np.any(missing_matrix):
        symbols_matrix = symbols_matrix.copy()
        symbols_matrix[missing_matrix] = "-"
    for i, branch_id in enumerate(target_branch_ids.tolist()):
        out[int(branch_id)] = "".join(symbols_matrix[i, :].tolist())
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
    device_opt = g.get("prostt5_device", "auto")
    mps_fallback_enabled_preimport = _enable_mps_fallback_for_option_if_needed(
        device_opt=device_opt
    )
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
    device = _resolve_prostt5_device(
        torch_module=torch,
        device_opt=device_opt,
    )
    if (str(device).strip().lower() == "mps") and mps_fallback_enabled_preimport:
        print("Enabled PYTORCH_ENABLE_MPS_FALLBACK=1 for ProstT5 on MPS.", flush=True)
    elif _enable_mps_fallback_if_needed(device=device):
        print("Enabled PYTORCH_ENABLE_MPS_FALLBACK=1 for ProstT5 on MPS.", flush=True)
    model = model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _torch_mps_is_available(torch_module):
    backends = getattr(torch_module, "backends", None)
    if backends is None:
        return False
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None:
        return False
    try:
        if hasattr(mps_backend, "is_built") and (not bool(mps_backend.is_built())):
            return False
        return bool(mps_backend.is_available())
    except Exception:
        return False


def _resolve_prostt5_device(torch_module, device_opt):
    device_opt = str(device_opt).strip().lower()
    cuda_available = bool(getattr(torch_module, "cuda", None) is not None and torch_module.cuda.is_available())
    mps_available = _torch_mps_is_available(torch_module=torch_module)
    if device_opt == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if device_opt == "cuda":
        if not cuda_available:
            raise ValueError('Requested --prostt5_device cuda, but CUDA is not available.')
        return "cuda"
    if device_opt == "mps":
        if not mps_available:
            raise ValueError('Requested --prostt5_device mps, but MPS is not available.')
        return "mps"
    if device_opt == "cpu":
        return "cpu"
    raise ValueError('--prostt5_device should be one of "auto", "cpu", "cuda", "mps".')


def _enable_mps_fallback_if_needed(device):
    if str(device).strip().lower() != "mps":
        return False
    key = "PYTORCH_ENABLE_MPS_FALLBACK"
    current = str(os.environ.get(key, "")).strip()
    if current == "1":
        return False
    os.environ[key] = "1"
    return True


def _enable_mps_fallback_for_option_if_needed(device_opt):
    device_opt = str(device_opt).strip().lower()
    if sys.platform != "darwin":
        return False
    if device_opt not in ["auto", "mps"]:
        return False
    key = "PYTORCH_ENABLE_MPS_FALLBACK"
    current = str(os.environ.get(key, "")).strip()
    if current == "1":
        return False
    os.environ[key] = "1"
    return True


def _resolve_prostt5_model_source(g):
    model_name, local_dir, no_download = _resolve_prostt5_model_options(g=g)
    if local_dir != "":
        model_source = local_dir
    else:
        model_source = model_name
    return model_source, no_download


def _normalize_prostt5_model_cache_key(model_source):
    model_source = str(model_source).strip()
    if os.path.isdir(model_source):
        return os.path.abspath(model_source)
    return model_source


def _load_prostt5_sequence_cache(cache_file, model_key):
    out = dict()
    cache_file = str(cache_file).strip()
    if cache_file == "":
        return out
    if not os.path.exists(cache_file):
        return out
    with open(cache_file, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line == "":
                continue
            cols = line.split("\t")
            if len(cols) != 3:
                continue
            key_txt, seq_txt, pred_txt = cols
            if key_txt != model_key:
                continue
            seq_txt = str(seq_txt).strip().upper()
            pred_txt = str(pred_txt).strip().upper()
            if (seq_txt == "") or (pred_txt == ""):
                continue
            if len(seq_txt) != len(pred_txt):
                continue
            invalid = sorted(list(set([ch for ch in pred_txt if ch not in _THREEDI_STATE_SET])))
            if len(invalid) > 0:
                continue
            out[seq_txt] = pred_txt
    return out


def _append_prostt5_sequence_cache(cache_file, model_key, seq_to_pred):
    if len(seq_to_pred) == 0:
        return
    cache_file = str(cache_file).strip()
    if cache_file == "":
        return
    cache_dir = os.path.dirname(cache_file)
    if cache_dir != "":
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, mode="a", encoding="utf-8") as f:
        for seq_txt, pred_txt in seq_to_pred.items():
            f.write("{}\t{}\t{}\n".format(model_key, seq_txt, pred_txt))


def _is_prostt5_oom_error(exc):
    txt = str(exc).strip().lower()
    return ("out of memory" in txt) or ("oom" in txt) or ("cuda out of memory" in txt) or ("mps out of memory" in txt)


def _clear_torch_device_cache(torch_module, device):
    device = str(device).strip().lower()
    if device == "cuda":
        cuda_ns = getattr(torch_module, "cuda", None)
        if cuda_ns is not None and hasattr(cuda_ns, "empty_cache"):
            try:
                cuda_ns.empty_cache()
            except Exception:
                pass
    if device == "mps":
        mps_ns = getattr(torch_module, "mps", None)
        if mps_ns is not None and hasattr(mps_ns, "empty_cache"):
            try:
                mps_ns.empty_cache()
            except Exception:
                pass


def _resolve_prostt5_auto_batch_size(threads, device, unique_sequence_count):
    threads = max(1, int(threads))
    batch_size = threads
    device = str(device).strip().lower()
    if device in ["cuda", "mps"]:
        # Favor larger batches on accelerators; dynamic backoff handles OOM.
        batch_size = max(16, threads * 4)
        if device == "cuda":
            batch_size = max(batch_size, 32)
        batch_size = min(batch_size, 64)
    if unique_sequence_count > 0:
        batch_size = min(batch_size, unique_sequence_count)
    return max(1, batch_size)


def predict_3di_with_prostt5(aa_sequences, g):
    out = dict()
    seq_to_ids = OrderedDict()
    seq_to_pred = dict()
    for seq_id, raw_seq in aa_sequences.items():
        seq = _sanitize_aa_sequence_for_prostt5(raw_seq)
        if seq == "":
            out[seq_id] = ""
            continue
        if seq not in seq_to_ids:
            seq_to_ids[seq] = list()
        seq_to_ids[seq].append(seq_id)
    length_to_unique_sequences = OrderedDict()
    for seq in seq_to_ids.keys():
        seq_len = int(len(seq))
        if seq_len not in length_to_unique_sequences:
            length_to_unique_sequences[seq_len] = list()
        length_to_unique_sequences[seq_len].append(seq)
    model_source, _ = _resolve_prostt5_model_source(g=g)
    model_key = _normalize_prostt5_model_cache_key(model_source=model_source)
    use_cache = bool(g.get("prostt5_cache", True))
    cache_file = str(g.get("prostt5_cache_file", _default_prostt5_cache_file())).strip()
    cache_hit_count = 0
    if use_cache:
        cached = _load_prostt5_sequence_cache(
            cache_file=cache_file,
            model_key=model_key,
        )
        for seq in seq_to_ids.keys():
            pred = cached.get(seq, None)
            if pred is None:
                continue
            seq_to_pred[seq] = pred
            cache_hit_count += 1
        if cache_hit_count > 0:
            txt = "ProstT5 cache hit: {} / {} unique sequence(s)"
            print(txt.format(cache_hit_count, len(seq_to_ids)), flush=True)
    remaining = [seq for seq in seq_to_ids.keys() if seq not in seq_to_pred]
    if len(remaining) == 0:
        for seq, seq_ids in seq_to_ids.items():
            pred = seq_to_pred[seq]
            for seq_id in seq_ids:
                out[seq_id] = pred
        return out
    seq_to_prompt = {
        seq: "<AA2fold> " + " ".join(seq)
        for seq in remaining
    }
    torch, tokenizer, model, device = _load_prostt5_components(g=g)
    batch_size = _resolve_prostt5_auto_batch_size(
        threads=g.get("threads", 1),
        device=device,
        unique_sequence_count=len(remaining),
    )
    new_cache_entries = dict()
    infer_context = torch.no_grad
    if hasattr(torch, "inference_mode") and callable(getattr(torch, "inference_mode")):
        infer_context = torch.inference_mode
    with infer_context():
        for seq_len, unique_sequences in length_to_unique_sequences.items():
            infer_sequences = [seq for seq in unique_sequences if seq not in seq_to_pred]
            i = 0
            current_batch_size = int(batch_size)
            while i < len(infer_sequences):
                chunk = infer_sequences[i : i + current_batch_size]
                prompts = [seq_to_prompt[seq] for seq in chunk]
                try:
                    batch = tokenizer(prompts, return_tensors="pt", padding=True)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    pred_ids = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask", None),
                        num_beams=1,
                        do_sample=False,
                        min_new_tokens=int(seq_len),
                        max_new_tokens=int(seq_len),
                    )
                except RuntimeError as exc:
                    if _is_prostt5_oom_error(exc) and (current_batch_size > 1):
                        next_batch_size = max(1, current_batch_size // 2)
                        if next_batch_size < current_batch_size:
                            txt = "ProstT5 OOM at batch_size={}, retrying with batch_size={}."
                            print(txt.format(current_batch_size, next_batch_size), flush=True)
                            current_batch_size = next_batch_size
                            _clear_torch_device_cache(torch_module=torch, device=device)
                            continue
                    raise
                if hasattr(tokenizer, "batch_decode") and callable(getattr(tokenizer, "batch_decode")):
                    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                else:
                    pred_texts = [tokenizer.decode(pred_id, skip_special_tokens=True) for pred_id in pred_ids]
                for seq, pred_txt in zip(chunk, pred_texts):
                    pred = str(pred_txt).replace(" ", "")
                    pred = str(pred).strip().upper()
                    if len(pred) != len(seq):
                        txt = "ProstT5 output length mismatch for sequence {}: input={}, output={}."
                        raise ValueError(txt.format(seq, len(seq), len(pred)))
                    invalid = sorted(list(set([ch for ch in pred if ch not in _THREEDI_STATE_SET])))
                    if len(invalid) > 0:
                        txt = "Unsupported 3Di symbol(s) in ProstT5 output for sequence {}: {}"
                        raise ValueError(txt.format(seq, ",".join(invalid)))
                    seq_to_pred[seq] = pred
                    if use_cache:
                        new_cache_entries[seq] = pred
                i += len(chunk)
            # Carry forward the stable per-device batch size to avoid repeated OOM retries
            # across different sequence-length groups in the same run.
            batch_size = min(int(batch_size), int(current_batch_size))
    if use_cache and (len(new_cache_entries) > 0):
        _append_prostt5_sequence_cache(
            cache_file=cache_file,
            model_key=model_key,
            seq_to_pred=new_cache_entries,
        )
        txt = "ProstT5 cache update: wrote {} new unique sequence(s) to {}"
        print(txt.format(len(new_cache_entries), cache_file), flush=True)
    for seq, seq_ids in seq_to_ids.items():
        pred = seq_to_pred.get(seq, None)
        if pred is None:
            txt = "No ProstT5 output was produced for sequence length {}."
            raise ValueError(txt.format(len(seq)))
        for seq_id in seq_ids:
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
    state_orders = np.asarray(state_orders, dtype=object).reshape(-1)
    state = np.zeros((num_node, num_site, state_orders.shape[0]), dtype=dtype)
    lookup_ascii = _build_ascii_symbol_lookup(state_orders)
    leaf_branch_ids = list()
    leaf_sequences = list()
    for leaf in ete.iter_leaves(tree_obj):
        branch_id = int(ete.get_prop(leaf, "numerical_label"))
        seq_txt = tip_3di_by_name.get(leaf.name, None)
        if seq_txt is None:
            continue
        seq_txt = str(seq_txt).strip().upper()
        if len(seq_txt) != num_site:
            txt = "Tip 3Di sequence length mismatch for {}: expected {}, got {}."
            raise ValueError(txt.format(leaf.name, num_site, len(seq_txt)))
        leaf_branch_ids.append(int(branch_id))
        leaf_sequences.append(seq_txt)
    if len(leaf_sequences) == 0 or num_site == 0:
        return state
    seq_ascii = _to_ascii_matrix(leaf_sequences)
    state_ids = lookup_ascii[seq_ascii]
    row_idx, site_idx = np.where(state_ids >= 0)
    if row_idx.shape[0] > 0:
        branch_ids = np.asarray(leaf_branch_ids, dtype=np.int64)
        state_idx = state_ids[row_idx, site_idx].astype(np.int64, copy=False)
        state[branch_ids[row_idx], site_idx, state_idx] = 1
    return state


def _get_tip_alignment_length(tip_alignment):
    lengths = sorted(set([len(str(v)) for v in tip_alignment.values()]))
    if len(lengths) != 1:
        raise ValueError("Tip alignment should have equal sequence lengths.")
    return int(lengths[0]) if len(lengths) > 0 else 0


def _get_tip_invariant_3di_site_mask(tip_3di_by_name):
    num_site = _get_tip_alignment_length(tip_3di_by_name)
    is_tip_invariant = np.zeros(shape=(num_site,), dtype=bool)
    if len(tip_3di_by_name) == 0:
        return is_tip_invariant
    tip_sequences = [str(seq).strip().upper() for seq in tip_3di_by_name.values()]
    tip_ascii = _to_ascii_matrix(tip_sequences)
    if tip_ascii.shape[1] == 0:
        return is_tip_invariant
    is_nonmissing = tip_ascii != ord("-")
    has_invalid_nonmissing = np.any(is_nonmissing & (~_THREEDI_ASCII_VALID_MASK[tip_ascii]), axis=0)
    num_nonmissing = is_nonmissing.sum(axis=0)
    min_values = np.where(is_nonmissing, tip_ascii, np.uint8(255)).min(axis=0)
    max_values = np.where(is_nonmissing, tip_ascii, np.uint8(0)).max(axis=0)
    is_tip_invariant = (num_nonmissing >= 1) & (min_values == max_values) & (~has_invalid_nonmissing)
    return is_tip_invariant


def _slice_tip_alignment_by_site_mask(tip_alignment, keep_mask):
    keep_index = np.where(np.asarray(keep_mask, dtype=bool))[0]
    out = OrderedDict()
    for name, seq in tip_alignment.items():
        seq = str(seq)
        out[str(name)] = "".join([seq[int(i)] for i in keep_index.tolist()])
    return out, keep_index


def _expand_state_tensor_site_axis(state_tensor, keep_site_index, full_num_site):
    keep_site_index = np.asarray(keep_site_index, dtype=np.int64).reshape(-1)
    expanded = np.zeros(
        shape=(state_tensor.shape[0], int(full_num_site), state_tensor.shape[2]),
        dtype=state_tensor.dtype,
    )
    if keep_site_index.shape[0] > 0:
        expanded[:, keep_site_index, :] = state_tensor
    return expanded


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
    file_tree = runtime.temp_path("tmp.csubst.3di.nwk")
    tree.write_tree(g["rooted_tree"], outfile=file_tree, add_numerical_label=False)
    try:
        model, did_remap_gtr20 = _normalize_direct_iqtree_model(g.get("sa_iqtree_model", "GTR"))
        if did_remap_gtr20:
            print(
                "Direct 3Di with --seqtype MORPH remaps --sa_iqtree_model GTR20 to GTR.",
                flush=True,
            )
        command = [
            g["iqtree_exe"],
            "-s",
            str(tip_alignment_path),
            "-te",
            file_tree,
            "-m",
            model,
            "--seqtype",
            "MORPH",
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
        "state_symbol_mode": "morph",
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
    state_symbol_mode = str(paths.get("state_symbol_mode", "auto")).strip().lower()
    for col in state_columns:
        raw_symbol = str(col).replace("p_", "").strip().upper()
        symbol_3di = _convert_state_symbol_to_3di(raw_symbol, symbol_mode=state_symbol_mode)
        if symbol_3di in state_lookup:
            col_to_state[str(col)] = int(state_lookup[symbol_3di])
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
    row_site_labels = state_table.loc[:, "Site"].to_numpy(dtype=np.int64, copy=False)
    row_site_index = np.fromiter(
        (site_index_by_label[int(v)] for v in row_site_labels),
        dtype=np.int64,
        count=row_site_labels.shape[0],
    )
    row_node_names = state_table.loc[:, "Node"].astype(str).to_numpy(dtype=object, copy=False)
    row_node_ids = np.fromiter(
        (node_name_to_id.get(name, -1) for name in row_node_names),
        dtype=np.int64,
        count=row_node_names.shape[0],
    )
    is_valid_row = row_node_ids >= 0
    state_col_names = list(col_to_state.keys())
    state_matrix = state_table.loc[:, state_col_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    state_matrix = np.nan_to_num(state_matrix, nan=0.0)
    for col_idx, col in enumerate(state_col_names):
        values = state_matrix[:, col_idx]
        if not np.any(values):
            continue
        is_nonzero = values != 0
        is_write = is_valid_row & is_nonzero
        if not np.any(is_write):
            continue
        state_id = int(col_to_state[col])
        write_values = values[is_write].astype(out_dtype, copy=False)
        state_tensor[row_node_ids[is_write], row_site_index[is_write], state_id] = write_values
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
    tip_3di_by_name_full = build_tip_3di_alignment_from_full_cds(
        g=g,
        predictor=predictor,
        output_path=runtime.temp_path("csubst_alignment_3di_tip.fa"),
    )
    g.pop("_precomputed_tip_invariant_site_mask", None)
    tip_3di_by_name_direct = tip_3di_by_name_full
    full_num_site = _get_tip_alignment_length(tip_3di_by_name_full)
    keep_site_index = np.arange(full_num_site, dtype=np.int64)
    mode = str(g.get("drop_invariant_tip_sites_mode", "tip_invariant")).strip().lower()
    should_prefilter = bool(g.get("drop_invariant_tip_sites", False)) and (mode == "tip_invariant")
    if should_prefilter and (full_num_site > 0):
        is_drop_site = _get_tip_invariant_3di_site_mask(tip_3di_by_name=tip_3di_by_name_full)
        if bool(is_drop_site.any()):
            keep_mask = ~is_drop_site
            if bool(keep_mask.any()):
                tip_3di_by_name_direct, keep_site_index = _slice_tip_alignment_by_site_mask(
                    tip_alignment=tip_3di_by_name_full,
                    keep_mask=keep_mask,
                )
                g["_precomputed_tip_invariant_site_mask"] = np.asarray(is_drop_site, dtype=bool)
                txt = "Direct 3Di prefilter: dropping {:,} tip-invariant 3Di site(s) before IQ-TREE."
                print(txt.format(int(is_drop_site.sum())), flush=True)
    _encode_tip_3di_alignment_for_morph(
        tip_3di_by_name=tip_3di_by_name_direct,
        output_path=runtime.temp_path("csubst_alignment_3di_tip_morph.fa"),
    )
    iqtree_paths = _run_iqtree_direct_3di(g=g, tip_alignment_path="csubst_alignment_3di_tip_morph.fa")
    state_tensor_direct, state_orders = _read_direct_3di_state_tensor(
        g=g,
        paths=iqtree_paths,
        tip_3di_by_name=tip_3di_by_name_direct,
        selected_branch_ids=selected_branch_ids,
    )
    if keep_site_index.shape[0] != full_num_site:
        state_tensor = _expand_state_tensor_site_axis(
            state_tensor=state_tensor_direct,
            keep_site_index=keep_site_index,
            full_num_site=full_num_site,
        )
    else:
        state_tensor = state_tensor_direct
    return state_tensor, state_orders, tip_3di_by_name_full


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
    projected_by_sequence_pair = dict()
    for branch_id, aa_aligned_seq in aa_aligned_by_branch.items():
        pred_seq = pred_ungapped.get(int(branch_id), "")
        pair_key = (str(aa_aligned_seq), str(pred_seq))
        projected = projected_by_sequence_pair.get(pair_key, None)
        if projected is None:
            projected = _inject_alignment_gaps(
                reference_aligned_seq=aa_aligned_seq,
                ungapped_pred_seq=pred_seq,
                seq_id=branch_id,
            )
            projected_by_sequence_pair[pair_key] = projected
        aligned_3di[int(branch_id)] = projected
    state_orders = get_3di_state_orders()
    state_lookup_ascii = _build_ascii_symbol_lookup(state_orders)
    num_node = int(state_pep.shape[0])
    num_site = int(state_pep.shape[1])
    out_dtype = g.get("float_type", state_pep.dtype)
    state_3di = np.zeros((num_node, num_site, state_orders.shape[0]), dtype=out_dtype)
    if len(aligned_3di) == 0 or num_site == 0:
        return state_3di, state_orders, aligned_3di
    branch_ids = np.array([int(branch_id) for branch_id in aligned_3di.keys()], dtype=np.int64)
    seq_list = [str(seq).strip().upper() for seq in aligned_3di.values()]
    for branch_id, seq in zip(branch_ids.tolist(), seq_list):
        if len(seq) != num_site:
            txt = "Aligned 3Di sequence length mismatch for branch {}: expected {}, got {}."
            raise ValueError(txt.format(branch_id, num_site, len(seq)))
    seq_ascii = _to_ascii_matrix(seq_list)
    state_ids = state_lookup_ascii[seq_ascii]
    row_idx, site_idx = np.where(state_ids >= 0)
    if row_idx.shape[0] > 0:
        state_idx = state_ids[row_idx, site_idx].astype(np.int64, copy=False)
        state_3di[branch_ids[row_idx], site_idx, state_idx] = 1
    return state_3di, state_orders, aligned_3di

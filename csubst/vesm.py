import csv
import hashlib
import os

import numpy as np

from csubst import model_resources
from csubst import resource_cache


MAX_SEQUENCE_RESIDUES = 1022
SCORE_ALGORITHM = "vesm-unmasked-llr-v1"
_CACHE_COLUMNS = (
    "model_resource_id",
    "score_algorithm",
    "window_sha256",
    "window_position",
    "from_aa",
    "to_aa",
    "vesm_llr",
)


def resolve_score_cache_file(g):
    configured = str(g.get("vep_cache_file", "")).strip()
    if configured != "":
        return os.path.abspath(os.path.expanduser(configured))
    cache_root = resource_cache.resolve_cache_dir(g.get("resource_cache_dir", ""))
    return os.path.join(cache_root, "scores", "vesm-35m.tsv")


def _cache_key(window_sha256, window_position, from_aa, to_aa):
    return (
        model_resources.VESM_35M_RESOURCE_ID,
        SCORE_ALGORITHM,
        str(window_sha256),
        int(window_position),
        str(from_aa),
        str(to_aa),
    )


def load_score_cache(cache_file):
    out = {}
    cache_file = str(cache_file).strip()
    if (cache_file == "") or (not os.path.isfile(cache_file)):
        return out
    try:
        with open(cache_file, encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or any(col not in reader.fieldnames for col in _CACHE_COLUMNS):
                return out
            for row in reader:
                try:
                    key = (
                        str(row["model_resource_id"]),
                        str(row["score_algorithm"]),
                        str(row["window_sha256"]),
                        int(row["window_position"]),
                        str(row["from_aa"]),
                        str(row["to_aa"]),
                    )
                    value = float(row["vesm_llr"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    out[key] = value
    except OSError:
        return {}
    return out


def merge_score_cache(
    cache_file,
    score_by_key,
    poll_seconds=resource_cache.DEFAULT_LOCK_POLL_SECONDS,
    timeout_seconds=resource_cache.DEFAULT_LOCK_TIMEOUT_SECONDS,
):
    if len(score_by_key) == 0:
        return
    cache_file = os.path.abspath(os.path.expanduser(str(cache_file)))
    lock_path = resource_cache.resolve_path_lock_path(cache_file, lock_label="vesm-score-cache")
    with resource_cache.acquire_exclusive_lock(
        lock_path=lock_path,
        lock_label="VESM score cache",
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    ):
        merged = load_score_cache(cache_file)
        for key, value in score_by_key.items():
            value = float(value)
            if np.isfinite(value):
                merged[tuple(key)] = value
        lines = ["\t".join(_CACHE_COLUMNS) + "\n"]
        for key in sorted(merged.keys()):
            model_id, algorithm, window_sha, position, from_aa, to_aa = key
            lines.append(
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.17g}\n".format(
                    model_id,
                    algorithm,
                    window_sha,
                    int(position),
                    from_aa,
                    to_aa,
                    float(merged[key]),
                )
            )
        resource_cache.atomic_write_text(cache_file, "".join(lines))


def _torch_mps_is_available(torch_module):
    backends = getattr(torch_module, "backends", None)
    mps_backend = getattr(backends, "mps", None) if backends is not None else None
    if mps_backend is None:
        return False
    is_available = getattr(mps_backend, "is_available", None)
    return bool(callable(is_available) and is_available())


def resolve_device(torch_module, device_opt):
    device_opt = str(device_opt).strip().lower()
    cuda_available = bool(
        getattr(torch_module, "cuda", None) is not None and torch_module.cuda.is_available()
    )
    mps_available = _torch_mps_is_available(torch_module)
    if device_opt == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if device_opt == "cuda" and not cuda_available:
        raise ValueError("Requested --vep_device cuda, but CUDA is not available.")
    if device_opt == "mps" and not mps_available:
        raise ValueError("Requested --vep_device mps, but MPS is not available.")
    if device_opt in ["cpu", "cuda", "mps"]:
        return device_opt
    raise ValueError('--vep_device should be one of "auto", "cpu", "cuda", "mps".')


def _is_oom_error(exc):
    txt = str(exc).strip().lower()
    return ("out of memory" in txt) or ("oom" in txt)


def _clear_device_cache(torch_module, device):
    if device == "cuda" and hasattr(torch_module, "cuda"):
        empty_cache = getattr(torch_module.cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    if device == "mps" and hasattr(torch_module, "mps"):
        empty_cache = getattr(torch_module.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def _load_components(g):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ImportError("VESM-35M inference requires torch. Install `csubst[vep]`.") from exc
    try:
        from transformers import AutoTokenizer, EsmForMaskedLM
    except ModuleNotFoundError as exc:
        raise ImportError("VESM-35M inference requires transformers. Install `csubst[vep]`.") from exc

    paths = model_resources.ensure_vesm35m_resource(
        cache_dir=g.get("resource_cache_dir", ""),
        no_download=bool(g.get("vep_no_download", False)),
        poll_seconds=float(g.get("resource_lock_poll", resource_cache.DEFAULT_LOCK_POLL_SECONDS)),
        timeout_seconds=float(g.get("resource_lock_timeout", resource_cache.DEFAULT_LOCK_TIMEOUT_SECONDS)),
    )
    device = resolve_device(torch_module=torch, device_opt=g.get("vep_device", "auto"))
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    tokenizer = AutoTokenizer.from_pretrained(paths["base_model_dir"], local_files_only=True)
    model = EsmForMaskedLM.from_pretrained(
        paths["base_model_dir"],
        local_files_only=True,
        use_safetensors=True,
    )
    try:
        state_dict = torch.load(paths["checkpoint_path"], map_location="cpu", weights_only=True)
    except TypeError:  # older supported torch releases
        state_dict = torch.load(paths["checkpoint_path"], map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    return torch, tokenizer, model, device


def _window_for_event(sequence, aa_position_1based):
    sequence = str(sequence).strip().upper()
    position = int(aa_position_1based) - 1
    if (position < 0) or (position >= len(sequence)):
        raise ValueError("VESM focal amino-acid position is outside the ancestral context.")
    if len(sequence) <= MAX_SEQUENCE_RESIDUES:
        start = 0
        end = len(sequence)
    else:
        start = position - (MAX_SEQUENCE_RESIDUES // 2)
        start = min(max(start, 0), len(sequence) - MAX_SEQUENCE_RESIDUES)
        end = start + MAX_SEQUENCE_RESIDUES
    return sequence[start:end], position - start, start, end


class Vesm35mScorer:
    def __init__(self, g, components=None):
        self.g = g
        if components is None:
            components = _load_components(g=g)
        self.torch, self.tokenizer, self.model, self.device = components
        self.model_resource_id = model_resources.VESM_35M_RESOURCE_ID
        self.cache_file = resolve_score_cache_file(g=g)
        self.use_cache = bool(g.get("vep_cache", True))
        self.cache = load_score_cache(self.cache_file) if self.use_cache else {}

    def _infer_windows(self, window_to_records):
        sequences = list(window_to_records.keys())
        if len(sequences) == 0:
            return {}
        vocab = self.tokenizer.get_vocab()
        batch_size = 1 if self.device == "cpu" else min(8, len(sequences))
        score_by_key = {}
        infer_context = getattr(self.torch, "inference_mode", self.torch.no_grad)
        index = 0
        while index < len(sequences):
            current_size = min(batch_size, len(sequences) - index)
            chunk = sequences[index : index + current_size]
            try:
                tokens = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=False)
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                with infer_context():
                    logits = self.model(**tokens)["logits"]
                for batch_index, window_sequence in enumerate(chunk):
                    for record in window_to_records[window_sequence]:
                        token_index = int(record["window_position"]) + 1
                        from_id = int(vocab[str(record["from_aa"])])
                        to_id = int(vocab[str(record["to_aa"])])
                        observed_id = int(tokens["input_ids"][batch_index, token_index].item())
                        if observed_id != from_id:
                            raise ValueError(
                                "VESM tokenizer/context mismatch at {}{}{}.".format(
                                    record["from_aa"],
                                    int(record["aa_position_ancestral"]),
                                    record["to_aa"],
                                )
                            )
                        log_probs = self.torch.log_softmax(logits[batch_index, token_index, :], dim=-1)
                        llr = float((log_probs[to_id] - log_probs[from_id]).item())
                        score_by_key[record["cache_key"]] = llr
                index += current_size
            except RuntimeError as exc:
                if _is_oom_error(exc) and current_size > 1:
                    batch_size = max(1, current_size // 2)
                    print(
                        "VESM OOM at batch_size={}; retrying with batch_size={}.".format(
                            current_size, batch_size
                        ),
                        flush=True,
                    )
                    _clear_device_cache(torch_module=self.torch, device=self.device)
                    continue
                raise
        return score_by_key

    def score(self, events):
        if events.shape[0] == 0:
            return events.copy()
        out = events.copy(deep=True)
        records = []
        window_to_records = {}
        for row_index, row in out.iterrows():
            window, window_position, start, end = _window_for_event(
                sequence=row["_context_sequence"],
                aa_position_1based=row["aa_position_ancestral"],
            )
            if window[window_position] != str(row["from_aa"]):
                raise ValueError("VESM context did not contain from_aa at the focal position.")
            window_sha = hashlib.sha256(window.encode("ascii")).hexdigest()
            key = _cache_key(
                window_sha256=window_sha,
                window_position=window_position,
                from_aa=row["from_aa"],
                to_aa=row["to_aa"],
            )
            record = {
                "row_index": row_index,
                "window_position": int(window_position),
                "from_aa": str(row["from_aa"]),
                "to_aa": str(row["to_aa"]),
                "aa_position_ancestral": int(row["aa_position_ancestral"]),
                "cache_key": key,
                "window_start_aa": int(start) + 1,
                "window_end_aa": int(end),
            }
            records.append(record)
            if key not in self.cache:
                window_to_records.setdefault(window, []).append(record)
        new_scores = self._infer_windows(window_to_records=window_to_records)
        if self.use_cache and len(new_scores) > 0:
            merge_score_cache(
                cache_file=self.cache_file,
                score_by_key=new_scores,
                poll_seconds=float(
                    self.g.get("resource_lock_poll", resource_cache.DEFAULT_LOCK_POLL_SECONDS)
                ),
                timeout_seconds=float(
                    self.g.get("resource_lock_timeout", resource_cache.DEFAULT_LOCK_TIMEOUT_SECONDS)
                ),
            )
        self.cache.update(new_scores)
        for record in records:
            llr = float(self.cache[record["cache_key"]])
            row_index = record["row_index"]
            out.at[row_index, "vesm_llr"] = llr
            out.at[row_index, "score_status"] = "scored"
            out.at[row_index, "window_start_aa"] = record["window_start_aa"]
            out.at[row_index, "window_end_aa"] = record["window_end_aa"]
            out.at[row_index, "vesm_model_resource_id"] = self.model_resource_id
        return out


def score_events(events, g):
    if events.shape[0] == 0:
        return events.copy()
    scorer = g.get("_vep_scorer", None)
    if scorer is None:
        scorer = Vesm35mScorer(g=g)
        g["_vep_scorer"] = scorer
    return scorer.score(events=events)

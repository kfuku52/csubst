"""Optional encoder-only AA -> Foldseek 3Di predictors.

ProstT5-CNN follows mheinzinger/ProstT5's predict_3Di_encoderOnly.py.
ESM3Di-35M reconstructs the published DessimozLab/ESM3di LoRA checkpoint
using Transformers/PEFT, without importing upstream training code.
See THIRD_PARTY_NOTICES.md and docs/STRUCTURAL_ALPHABET.md for provenance.
"""

from dataclasses import dataclass
import os
import shutil
from typing import Any
import urllib.request

from csubst import resource_cache
from csubst import structural_alphabet as sa
from csubst.recoding_config import (
    DEFAULT_SA_BACKEND,
    SA_BACKENDS,
    normalize_sa_backend as normalize_backend,
)


BACKENDS = SA_BACKENDS
CNN_REVISION = "3f6c0666ac61d1025ce9473e34d3f67fc893a589"
CNN_SHA256 = "d2cb4150884de095dac3de81b8fe7842ab1265de41dd5e8c9970879a71e62f3e"
ESM3DI_REPO = "cactuskid13/esm2small_3di"
ESM3DI_REVISION = "2227cb08ffa533bc1a7f2b968fd6316f287134d1"
ESM3DI_SHA256 = "044489c3bf8ffa1067dd2e71edf1750b48a2d52d73018ae4908d98ddebbb0587"
ESM_BASE_REPO = "facebook/esm2_t12_35M_UR50D"
ESM_BASE_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"
# The checkpoint contains the entire ESM backbone; only its config/tokenizer
# are needed from the base repository, not another copy of the base weights.
ESM_BASE_FILES = {
    "config.json": (778, "ae07364efa51e0fa1d81da3fe608f6d948abdbbb69785ba6e1df379a458e58ee"),
    "special_tokens_map.json": (125, "3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1"),
    "tokenizer_config.json": (95, "7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad"),
    "vocab.txt": (93, "0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03"),
}


def get_model_cache_key(g):
    backend = normalize_backend(g.get("sa_backend", DEFAULT_SA_BACKEND))
    if backend == "prostt5":
        return sa.get_prostt5_model_cache_key(g)
    if backend == "prostt5-cnn":
        return "prostt5-cnn-v1:{}:cnn@{}".format(sa.get_prostt5_model_cache_key(g), CNN_SHA256)
    return "esm3di-35m-v1:{}@{}:base@{}".format(ESM3DI_REPO, ESM3DI_REVISION, ESM_BASE_REVISION)


def ensure_encoder_resource(g, download_file=None):
    """Atomically prepare and SHA-256 verify all small encoder resources."""
    backend = normalize_backend(g.get("sa_backend", DEFAULT_SA_BACKEND))
    if backend == "prostt5":
        raise ValueError("Encoder resources require prostt5-cnn or esm3di-35m.")
    cache_dir = resource_cache.resolve_cache_dir(g.get("resource_cache_dir", ""))
    root = os.path.join(cache_dir, "models", "3di", backend, "v1")
    if backend == "prostt5-cnn":
        expected = {"cnn.pt": {"size": 3749041, "sha256": CNN_SHA256}}
        revision = CNN_REVISION
    else:
        expected = {"epoch_3.pt": {"size": 136687336, "sha256": ESM3DI_SHA256}}
        expected.update({
            "base_model/" + name: {"size": size, "sha256": digest}
            for name, (size, digest) in ESM_BASE_FILES.items()
        })
        revision = ESM3DI_REVISION

    def populate(stage_dir):
        if backend == "prostt5-cnn":
            url = "https://raw.githubusercontent.com/mheinzinger/ProstT5/{}/cnn_chkpnt/model.pt".format(CNN_REVISION)
            request = urllib.request.Request(url, headers={"User-Agent": "CSUBST"})
            with urllib.request.urlopen(request, timeout=120) as response:
                with open(os.path.join(stage_dir, "cnn.pt"), "wb") as handle:
                    shutil.copyfileobj(response, handle)
            return
        active_download = download_file
        if active_download is None:
            try:
                from huggingface_hub import hf_hub_download
            except ModuleNotFoundError as exc:
                raise ImportError("3Di model downloads require huggingface-hub. Install csubst[3di].") from exc
            active_download = hf_hub_download
        active_download(repo_id=ESM3DI_REPO, revision=ESM3DI_REVISION,
                        filename="epoch_3.pt", local_dir=stage_dir)
        base_dir = os.path.join(stage_dir, "base_model")
        os.makedirs(base_dir, exist_ok=True)
        for name in ESM_BASE_FILES:
            active_download(repo_id=ESM_BASE_REPO, revision=ESM_BASE_REVISION,
                            filename=name, local_dir=base_dir)

    resource_cache.ensure_directory_resource(
        resource_id="{}@{}".format(backend, revision),
        resource_dir=root,
        populate=populate,
        required_files=tuple(expected),
        expected_files=expected,
        manifest_metadata={"backend": backend, "revision": revision},
        cache_dir=cache_dir,
        no_download=bool(g.get("prostt5_no_download", False)),
        verify_existing=True,
        poll_seconds=float(g.get("resource_lock_poll", resource_cache.DEFAULT_LOCK_POLL_SECONDS)),
        timeout_seconds=float(g.get("resource_lock_timeout", resource_cache.DEFAULT_LOCK_TIMEOUT_SECONDS)),
    )
    return root


def _import_torch(g):
    sa._enable_mps_fallback_for_option_if_needed(g.get("prostt5_device", "auto"))
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ImportError("Encoder-only 3Di inference requires torch. Install csubst[3di].") from exc
    version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if version < (2, 6):
        raise ImportError("Encoder-only 3Di checkpoints require torch>=2.6 for safe weights_only loading. Install csubst[3di].")
    return torch


@dataclass
class EncoderPredictor:
    backend: str
    torch: Any
    tokenizer: Any
    model: Any
    device: Any
    labels: tuple
    classifier: Any = None

    def predict_batch(self, sequences):
        return self._predict_batch(sequences, return_logits=False)

    def predict_logits_batch(self, sequences):
        """Return residue-by-state logits in ``self.labels`` order, not posteriors."""
        return self._predict_batch(sequences, return_logits=True)

    def _predict_batch(self, sequences, return_logits):
        if not sequences:
            return []
        if self.backend == "prostt5-cnn":
            prompts = ["<AA2fold> " + " ".join(seq) for seq in sequences]
        else:
            prompts = sequences
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        # Both tokenizers must give exactly one token per residue plus two
        # special tokens. Never silently truncate long inputs or lose residues.
        lengths = batch["attention_mask"].sum(dim=1).tolist()
        if lengths != [len(seq) + 2 for seq in sequences]:
            raise ValueError("{} tokenizer did not preserve one token per residue.".format(self.backend))
        batch = {key: value.to(self.device) for key, value in batch.items()}
        output = self.model(**batch)
        predictions = []
        for index, seq in enumerate(sequences):
            if self.backend == "prostt5-cnn":
                # Retain the zero-masked EOS slot as in upstream singleton inference.
                # Run the tiny CNN on each unpadded embedding: its two biased
                # convolutions otherwise let padding affect terminal residues.
                embedding = output.last_hidden_state[index:index + 1, 1:len(seq) + 2].clone()
                embedding[:, -1] = 0
                features = embedding.permute(0, 2, 1).unsqueeze(-1)
                logits = self.classifier(features)[0, :, :len(seq), 0].transpose(0, 1)
            else:
                logits = output.logits[index, 1:len(seq) + 1]
            if not self.torch.isfinite(logits).all():
                raise ValueError("{} produced non-finite 3Di logits.".format(self.backend))
            if return_logits:
                predictions.append(logits.detach().float().cpu().numpy())
            else:
                indices = logits.argmax(dim=-1).cpu().tolist()
                predictions.append("".join(self.labels[i] for i in indices))
        return predictions


def load_encoder_predictor(g):
    backend = normalize_backend(g.get("sa_backend", DEFAULT_SA_BACKEND))
    if backend == "prostt5":
        raise ValueError("Encoder inference requires prostt5-cnn or esm3di-35m.")
    torch = _import_torch(g)
    device = sa._resolve_prostt5_device(torch, g.get("prostt5_device", "auto"))
    root = ensure_encoder_resource(g)
    labels = tuple("ACDEFGHIKLMNPQRSTVWY")
    classifier = None
    if backend == "prostt5-cnn":
        from transformers import T5EncoderModel, T5Tokenizer
        tokenizer, model, _ = sa._load_or_download_prostt5(g, T5Tokenizer, T5EncoderModel)
        if model.config.d_model != 1024:
            raise ValueError("ProstT5-CNN requires a ProstT5 encoder with 1024-dimensional embeddings.")
        classifier = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),
            torch.nn.ReLU(), torch.nn.Dropout(0.0),
            torch.nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0)),
        )
        checkpoint = torch.load(os.path.join(root, "cnn.pt"), map_location="cpu", weights_only=True)
        classifier.load_state_dict({key.removeprefix("classifier."): value
                                    for key, value in checkpoint["state_dict"].items()}, strict=True)
        classifier = classifier.float().to(device).eval()
    else:
        from transformers import EsmConfig, EsmForTokenClassification, EsmTokenizer
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ModuleNotFoundError as exc:
            raise ImportError("ESM3Di-35M requires peft. Install csubst[3di].") from exc
        base_dir = os.path.join(root, "base_model")
        checkpoint = torch.load(os.path.join(root, "epoch_3.pt"), map_location="cpu", weights_only=True)
        labels = tuple(checkpoint["label_vocab"])
        if len(labels) != 20 or set(labels) != set(sa.get_3di_state_orders()):
            raise ValueError("ESM3Di checkpoint must contain exactly the 20 Foldseek 3Di labels.")
        args = checkpoint["args"]
        if args["hf_model"] != ESM_BASE_REPO:
            raise ValueError("ESM3Di checkpoint does not use the expected ESM2-35M backbone.")
        config = EsmConfig.from_pretrained(base_dir, local_files_only=True)
        config.num_labels = len(labels)
        model = get_peft_model(EsmForTokenClassification(config), LoraConfig(
            task_type=TaskType.TOKEN_CLS, r=args["lora_r"], lora_alpha=args["lora_alpha"],
            lora_dropout=args["lora_dropout"], target_modules=checkpoint["lora_target_modules"],
        ))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # Fold the trained adapters into the backbone once, outside inference.
        model = model.merge_and_unload(safe_merge=True)
        tokenizer = EsmTokenizer.from_pretrained(base_dir, local_files_only=True)
    model = model.float().to(device).eval()
    return EncoderPredictor(backend, torch, tokenizer, model, device, labels, classifier)


def ensure_encoder_model_files(g):
    # Resource preparation should not require an accelerator, even on GPU hosts.
    local_g = dict(g, prostt5_device="cpu")
    predictor = load_encoder_predictor(local_g)
    del predictor
    return os.path.join(resource_cache.resolve_cache_dir(g.get("resource_cache_dir", "")),
                        "models", "3di", normalize_backend(g.get("sa_backend", DEFAULT_SA_BACKEND)), "v1")


def _batch_limit(g, device):
    requested = int(g.get("sa_batch_size", 0))
    if requested < 0:
        raise ValueError("--sa_batch_size should be >= 0.")
    if requested:
        return requested
    # This is intentionally independent of CPU thread count. Bound padded
    # residues as well, so automatic batches shrink for longer sequences.
    return 4 if str(device) == "cpu" else 16


def predict_3di(aa_sequences, g):
    backend = normalize_backend(g.get("sa_backend", DEFAULT_SA_BACKEND))
    if backend == "prostt5":
        return sa.predict_3di_with_prostt5(aa_sequences, g)
    sequences = {key: sa._sanitize_aa_sequence_for_prostt5(seq) for key, seq in aa_sequences.items()}
    unique = list(dict.fromkeys(seq for seq in sequences.values() if seq))
    # The pinned ESM2 backbone uses rotary positions, not a fixed-size learned
    # position table. Do not impose a 1024-token/1022-residue cutoff: full
    # sequences are supported subject to available compute and memory.
    model_key = get_model_cache_key(g)
    use_cache = bool(g.get("prostt5_cache", True))
    cache_file = str(g.get("prostt5_cache_file", sa._default_prostt5_cache_file())).strip()
    cached = sa._load_prostt5_sequence_cache(cache_file, model_key) if use_cache and unique else {}
    predictions = {seq: cached[seq] for seq in unique if seq in cached}
    if predictions:
        print("{} cache hit: {} / {} unique sequence(s)".format(backend, len(predictions), len(unique)), flush=True)
    remaining = sorted((seq for seq in unique if seq not in predictions), key=len, reverse=True)
    if remaining:
        predictor = load_encoder_predictor(g)
        batch_limit = _batch_limit(g, predictor.device)
        print("3Di predictor: {}, device={}, batch size cap={}".format(backend, predictor.device, batch_limit), flush=True)
        new_entries = {}
        offset = 0
        with predictor.torch.inference_mode():
            while offset < len(remaining):
                count = min(batch_limit, max(1, 4096 // (len(remaining[offset]) + 2)))
                chunk = remaining[offset:offset + count]
                try:
                    predicted = predictor.predict_batch(chunk)
                except RuntimeError as exc:
                    if sa._is_prostt5_oom_error(exc) and len(chunk) > 1:
                        batch_limit = max(1, len(chunk) // 2)
                        print("{} OOM; retrying with batch size {}.".format(backend, batch_limit), flush=True)
                        sa._clear_torch_device_cache(predictor.torch, predictor.device)
                        continue
                    raise
                if len(predicted) != len(chunk):
                    raise ValueError("{} returned the wrong number of 3Di sequences.".format(backend))
                for seq, pred in zip(chunk, predicted):
                    if len(pred) != len(seq) or not set(pred).issubset(sa._THREEDI_STATE_SET):
                        raise ValueError("{} returned an invalid 3Di sequence (length or alphabet mismatch).".format(backend))
                    predictions[seq] = pred
                    new_entries[seq] = pred
                offset += len(chunk)
        if use_cache:
            sa._append_prostt5_sequence_cache(
                cache_file, model_key, new_entries,
                poll_seconds=float(g.get("resource_lock_poll", resource_cache.DEFAULT_LOCK_POLL_SECONDS)),
                timeout_seconds=float(g.get("resource_lock_timeout", resource_cache.DEFAULT_LOCK_TIMEOUT_SECONDS)),
            )
    return {key: predictions[seq] if seq else "" for key, seq in sequences.items()}


def predict_3di_records(aa_sequences, g):
    """Retain uncalibrated encoder logits for scientific validation.

    This opt-in API bypasses the character-only cache: a cached argmax cannot
    recover a distribution. It does not change the hard-state ASR path. The
    autoregressive ProstT5 backend returns explicitly hard-only records, never
    fabricated one-hot probabilities. Persist records with structural_validation.
    """
    from csubst.structural_validation import PredictionRecord, STATE_ORDER

    backend = normalize_backend(g.get("sa_backend", DEFAULT_SA_BACKEND))
    sequences = {key: sa._sanitize_aa_sequence_for_prostt5(seq) for key, seq in aa_sequences.items()}
    model_key = get_model_cache_key(g)
    if backend == "prostt5":
        predictions = predict_3di(aa_sequences, dict(g, prostt5_cache=False))
        return {key: PredictionRecord(seq, predictions[key], backend, model_key)
                for key, seq in sequences.items()}

    unique = sorted(set(seq for seq in sequences.values() if seq), key=lambda seq: (-len(seq), seq))
    records = {}
    if unique:
        predictor = load_encoder_predictor(g)
        batch_limit = _batch_limit(g, predictor.device)
        offset = 0
        with predictor.torch.inference_mode():
            while offset < len(unique):
                count = min(batch_limit, max(1, 4096 // (len(unique[offset]) + 2)))
                chunk = unique[offset:offset + count]
                try:
                    logits_batch = predictor.predict_logits_batch(chunk)
                except RuntimeError as exc:
                    if sa._is_prostt5_oom_error(exc) and len(chunk) > 1:
                        batch_limit = max(1, len(chunk) // 2)
                        sa._clear_torch_device_cache(predictor.torch, predictor.device)
                        continue
                    raise
                if len(logits_batch) != len(chunk):
                    raise ValueError("{} returned the wrong number of 3Di logit arrays.".format(backend))
                for seq, logits in zip(chunk, logits_batch):
                    records[seq] = PredictionRecord.from_logits(
                        seq, logits, predictor.labels, backend, model_key,
                    )
                offset += len(chunk)
    if any(not seq for seq in sequences.values()):
        import numpy as np
        records[""] = PredictionRecord.from_logits(
            "", np.empty((0, 20)), STATE_ORDER, backend, model_key,
        )
    return {key: records[seq] for key, seq in sequences.items()}

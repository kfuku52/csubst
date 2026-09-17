import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from csubst import structural_alphabet as sa
from csubst import structural_prediction as sp


class FakePredictor:
    device = "cpu"
    torch = SimpleNamespace(inference_mode=contextlib.nullcontext)

    def __init__(self, fail_above=None, invalid=None):
        self.calls = []
        self.fail_above = fail_above
        self.invalid = invalid

    def predict_batch(self, sequences):
        self.calls.append(list(sequences))
        if self.fail_above is not None and len(sequences) > self.fail_above:
            raise RuntimeError("out of memory")
        if self.invalid is not None:
            return self.invalid
        return ["A" * len(seq) for seq in sequences]


@pytest.mark.parametrize("backend", ["prostt5-cnn", "esm3di-35m"])
def test_encoder_batches_mixed_lengths_deduplicates_and_restores_ids(monkeypatch, backend):
    predictor = FakePredictor()
    monkeypatch.setattr(sp, "load_encoder_predictor", lambda g: predictor)
    inputs = {"short": "M-K", 7: "MKK", "same": "mk", "empty": "--", "unknown": "MUO"}
    result = sa.predict_3di(inputs, {"sa_backend": backend, "prostt5_cache": False})
    assert list(result) == list(inputs)
    assert result == {"short": "AA", 7: "AAA", "same": "AA", "empty": "", "unknown": "AAA"}
    assert predictor.calls == [["MKK", "MXX", "MK"]]


def test_encoder_cache_isolates_backends_and_reuses_without_loading(tmp_path, monkeypatch):
    g = {"sa_backend": "esm3di-35m", "prostt5_cache_file": str(tmp_path / "cache.tsv")}
    predictor = FakePredictor()
    monkeypatch.setattr(sp, "load_encoder_predictor", lambda g: predictor)
    sa._append_prostt5_sequence_cache(g["prostt5_cache_file"], sa.get_prostt5_model_cache_key({}), {"MK": "CC"})
    assert sa.predict_3di({"one": "MK"}, g) == {"one": "AA"}

    def fail(g):
        pytest.fail("A cache hit must not import/load a neural model")

    monkeypatch.setattr(sp, "load_encoder_predictor", fail)
    assert sa.predict_3di({"two": "MK", "empty": ""}, g) == {"two": "AA", "empty": ""}
    assert sa._load_prostt5_sequence_cache(g["prostt5_cache_file"], sa.get_prostt5_model_cache_key({})) == {"MK": "CC"}


def test_encoder_oom_reduces_batch_and_preserves_all_outputs(monkeypatch):
    predictor = FakePredictor(fail_above=1)
    monkeypatch.setattr(sp, "load_encoder_predictor", lambda g: predictor)
    inputs = {"a": "MK", "b": "MKK", "c": "MKKK"}
    result = sa.predict_3di(inputs, {"sa_backend": "prostt5-cnn", "prostt5_cache": False})
    assert result == {key: "A" * len(seq) for key, seq in inputs.items()}
    assert [len(batch) for batch in predictor.calls] == [3, 1, 1, 1]


@pytest.mark.parametrize("invalid", [[], ["ZZ"]])
def test_invalid_encoder_output_is_never_cached(tmp_path, monkeypatch, invalid):
    predictor = FakePredictor(invalid=invalid)
    monkeypatch.setattr(sp, "load_encoder_predictor", lambda g: predictor)
    cache = tmp_path / "cache.tsv"
    with pytest.raises(ValueError, match="invalid 3Di|wrong number"):
        sa.predict_3di({"a": "MK"}, {"sa_backend": "prostt5-cnn", "prostt5_cache_file": str(cache)})
    assert not cache.exists()


@pytest.mark.parametrize('length', [1023, 8192])
def test_esm_accepts_long_sequences_without_truncation(monkeypatch, length):
    predictor = FakePredictor()
    monkeypatch.setattr(sp, "load_encoder_predictor", lambda g: predictor)
    sequence = 'M' * length
    assert sa.predict_3di({'long': sequence}, {'prostt5_cache': False}) == {'long': 'A' * length}
    assert predictor.calls == [[sequence]]


def test_batch_limit_uses_residue_budget_not_threads(monkeypatch):
    predictor = FakePredictor()
    monkeypatch.setattr(sp, "load_encoder_predictor", lambda g: predictor)
    inputs = {i: "M" * (2000 + i) for i in range(4)}
    sa.predict_3di(inputs, {"sa_backend": "prostt5-cnn", "prostt5_cache": False, "threads": 1})
    assert [len(batch) for batch in predictor.calls] == [2, 2]


def test_offline_missing_encoder_resource_does_not_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="download is disabled"):
        sp.ensure_encoder_resource(
            {"sa_backend": "esm3di-35m", "resource_cache_dir": str(tmp_path), "prostt5_no_download": True},
            download_file=lambda **kwargs: pytest.fail("offline mode must not download"),
        )


def test_existing_encoder_resources_are_always_hash_checked(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sp.resource_cache, "ensure_directory_resource", lambda **kwargs: calls.append(kwargs))
    sp.ensure_encoder_resource({"sa_backend": "esm3di-35m", "resource_cache_dir": str(tmp_path), "prostt5_no_download": True})
    assert calls[0]["verify_existing"] is True
    assert calls[0]["no_download"] is True
    assert set(calls[0]["expected_files"]) == set(calls[0]["required_files"])


def test_encoder_download_rejects_corrupt_weights_before_publication(tmp_path):
    def download(**kwargs):
        (Path(kwargs["local_dir"]) / kwargs["filename"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size|SHA|sha|mismatch"):
        sp.ensure_encoder_resource({"sa_backend": "esm3di-35m", "resource_cache_dir": str(tmp_path)}, download_file=download)
    assert not (tmp_path / "models/3di/esm3di-35m/v1").exists()


def test_explicit_prostt5_dispatch_preserves_prostt5_api(monkeypatch):
    expected = {"one": "AC"}
    monkeypatch.setattr(sa, "predict_3di_with_prostt5", lambda aa_sequences, g: expected)
    assert sa.predict_3di({"one": "MK"}, {"sa_backend": "prostt5"}) is expected


def test_default_dispatch_and_cache_identity_use_esm3di(monkeypatch):
    predictor = FakePredictor()
    monkeypatch.setattr(sp, "load_encoder_predictor", lambda g: predictor)
    monkeypatch.setattr(sa, "predict_3di_with_prostt5", lambda *args: pytest.fail("legacy generator selected"))
    assert sa.predict_3di({"one": "MK"}, {"prostt5_cache": False}) == {"one": "AA"}
    assert sa.get_3di_model_cache_key({}) == sa.get_3di_model_cache_key({"sa_backend": "esm3di-35m"})
    assert sa.get_3di_model_cache_key({}) != sa.get_prostt5_model_cache_key({})


@pytest.mark.parametrize("backend", ["prostt5-cnn", "esm3di-35m"])
def test_direct_tip_alignment_uses_selected_encoder_and_restores_gaps(monkeypatch, tmp_path, backend):
    monkeypatch.setattr(sa, "build_tip_aa_alignment_from_full_cds", lambda g: {"A": "M-K", "B": "MK-"})
    predictor = FakePredictor()
    monkeypatch.setattr(sp, "load_encoder_predictor", lambda g: predictor)
    out = tmp_path / "3di.fa"
    result = sa.build_tip_3di_alignment_from_full_cds(
        {"sa_backend": backend, "prostt5_cache": False}, output_path=str(out)
    )
    assert result == {"A": "A-A", "B": "AA-"}
    assert out.read_text() == ">A\nA-A\n>B\nAA-\n"
    assert predictor.calls == [["MK"]]


def _torch_runtime():
    return pytest.importorskip("torch", minversion="2.6")


class TinyTokenizer:
    def __init__(self, torch):
        self.torch = torch

    def __call__(self, prompts, **kwargs):
        seqs = [text.replace("<AA2fold> ", "").replace(" ", "") for text in prompts]
        ids = self.torch.zeros((len(seqs), max(map(len, seqs)) + 2), dtype=self.torch.long)
        mask = ids.clone()
        for i, seq in enumerate(seqs):
            ids[i, :len(seq) + 2] = self.torch.arange(len(seq) + 2) + 1
            mask[i, :len(seq) + 2] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_cnn_batch_padding_matches_unpadded_reference():
    torch = _torch_runtime()
    torch.manual_seed(5)
    classifier = torch.nn.Sequential(
        torch.nn.Conv2d(4, 32, (7, 1), padding=(3, 0)), torch.nn.ReLU(),
        torch.nn.Dropout(0.0), torch.nn.Conv2d(32, 20, (7, 1), padding=(3, 0)),
    ).eval()

    def encoder(input_ids, attention_mask):
        embedding = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
        return SimpleNamespace(last_hidden_state=embedding)

    predictor = sp.EncoderPredictor("prostt5-cnn", torch, TinyTokenizer(torch), encoder,
                                    "cpu", tuple("ACDEFGHIKLMNPQRSTVWY"), classifier)
    inputs = ["M", "MKL", "M" * 17]
    with torch.inference_mode():
        batched = predictor.predict_batch(inputs)
        single = [predictor.predict_batch([seq])[0] for seq in inputs]
    assert batched == single


def test_esm_token_offsets_label_order_and_nonfinite_logits():
    torch = _torch_runtime()
    labels = tuple(reversed("ACDEFGHIKLMNPQRSTVWY"))

    def model(input_ids, attention_mask):
        return SimpleNamespace(logits=torch.nn.functional.one_hot(input_ids % 20, 20).float())

    predictor = sp.EncoderPredictor("esm3di-35m", torch, TinyTokenizer(torch), model, "cpu", labels)
    assert predictor.predict_batch(["MK", "M"]) == [labels[2] + labels[3], labels[2]]
    predictor.model = lambda **batch: SimpleNamespace(logits=torch.full((1, 3, 20), float("nan")))
    with pytest.raises(ValueError, match="non-finite"):
        predictor.predict_batch(["M"])


def test_esm_loader_strict_checkpoint_and_merged_predictions(tmp_path, monkeypatch):
    torch = _torch_runtime()
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    # A tiny real ESM/LoRA model exercises classifier.modules_to_save and the
    # complete checkpoint load, rather than mocking away the trained weights.
    torch.manual_seed(7)
    config = transformers.EsmConfig(vocab_size=33, hidden_size=32, num_hidden_layers=1,
                                    num_attention_heads=4, intermediate_size=64,
                                    pad_token_id=1, mask_token_id=32, num_labels=20,
                                    position_embedding_type="rotary", token_dropout=False)
    base = tmp_path / "base_model"
    config.save_pretrained(base)
    args = dict(hf_model=sp.ESM_BASE_REPO, lora_r=2, lora_alpha=4, lora_dropout=0.0)
    original = peft.get_peft_model(transformers.EsmForTokenClassification(config), peft.LoraConfig(
        task_type=peft.TaskType.TOKEN_CLS, r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=["query", "value"],
    )).eval()
    for name, value in original.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(value, std=0.05)
    checkpoint = {"model_state_dict": original.state_dict(), "args": args,
                  "label_vocab": list("ACDEFGHIKLMNPQRSTVWY"), "lora_target_modules": ["query", "value"]}
    torch.save(checkpoint, tmp_path / "epoch_3.pt")
    monkeypatch.setattr(sp, "ensure_encoder_resource", lambda g: str(tmp_path))
    monkeypatch.setattr(transformers.EsmTokenizer, "from_pretrained", lambda *a, **k: TinyTokenizer(torch))
    predictor = sp.load_encoder_predictor({"sa_backend": "esm3di-35m", "prostt5_device": "cpu"})
    tokens = torch.tensor([[0, 5, 6, 2]])
    mask = torch.ones_like(tokens)
    with torch.inference_mode():
        expected = original(input_ids=tokens, attention_mask=mask).logits
        actual = predictor.model(input_ids=tokens, attention_mask=mask).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert not any("lora_" in name for name, _ in predictor.model.named_parameters())
    del checkpoint["model_state_dict"]["base_model.model.esm.embeddings.word_embeddings.weight"]
    torch.save(checkpoint, tmp_path / "epoch_3.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        sp.load_encoder_predictor({"sa_backend": "esm3di-35m", "prostt5_device": "cpu"})


def test_sequence_cache_expands_home_on_read_and_write(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    g = {'sa_backend': 'esm3di-35m', 'prostt5_cache_file': '~/predictions.tsv'}
    key = sa.get_3di_model_cache_key(g)
    sa._append_prostt5_sequence_cache(g['prostt5_cache_file'], key, {'MK': 'AC'})
    assert (tmp_path / 'predictions.tsv').is_file()
    monkeypatch.setattr(sp, 'load_encoder_predictor', lambda g: pytest.fail('cache hit must not load a model'))
    monkeypatch.setattr(sa, '_load_prostt5_components', lambda g: pytest.fail('cache hit must not load a model'))
    assert sa.predict_3di({'sequence': 'MK'}, g) == {'sequence': 'AC'}

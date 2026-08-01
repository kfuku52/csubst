import os

from csubst import structural_alphabet


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorchRuntime:
    @staticmethod
    def no_grad():
        return _FakeNoGrad()


class _FakeBatchTensor:
    def __init__(self, payload):
        self.payload = payload

    def to(self, _device):
        return self


class _FakeBatchTokenizer:
    decode_calls = []

    @classmethod
    def reset(cls):
        cls.decode_calls = []

    def __call__(self, prompts, return_tensors="pt", padding=True):
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)
        return {
            "input_ids": _FakeBatchTensor(prompt_list),
            "attention_mask": _FakeBatchTensor([1] * len(prompt_list)),
        }

    def decode(self, pred_id, skip_special_tokens=True):
        type(self).decode_calls.append(pred_id)
        return str(pred_id["pred"])


class _FakeBatchTokenizerWithBatchDecode(_FakeBatchTokenizer):
    batch_decode_calls = []

    @classmethod
    def reset(cls):
        cls.decode_calls = []
        cls.batch_decode_calls = []

    def batch_decode(self, pred_ids, skip_special_tokens=True):
        pred_ids = list(pred_ids)
        type(self).batch_decode_calls.append(len(pred_ids))
        return [str(pred_id["pred"]) for pred_id in pred_ids]


class _FakeBatchModel:
    generate_batch_sizes = []

    @classmethod
    def reset(cls):
        cls.generate_batch_sizes = []

    def generate(
        self,
        input_ids,
        attention_mask=None,
        num_beams=1,
        do_sample=False,
        min_new_tokens=0,
        max_new_tokens=0,
    ):
        prompts = list(input_ids.payload)
        type(self).generate_batch_sizes.append(len(prompts))
        return [{"pred": "A" * int(max_new_tokens)} for _ in prompts]


def test_predict_3di_with_prostt5_batches_and_reuses_duplicates(monkeypatch):
    tokenizer = _FakeBatchTokenizer()
    model = _FakeBatchModel()
    _FakeBatchTokenizer.reset()
    _FakeBatchModel.reset()

    monkeypatch.setattr(
        structural_alphabet,
        "_load_prostt5_components",
        lambda g: (_FakeTorchRuntime(), tokenizer, model, "cpu"),
    )
    aa_sequences = {
        "n1": "AC",
        "n2": "AC",
        "n3": "XX",
        "n4": "MNP",
        "n5": "",
        "n6": "--",
    }
    out = structural_alphabet.predict_3di_with_prostt5(
        aa_sequences=aa_sequences,
        g={"threads": 2, "prostt5_cache": False},
    )
    assert out["n1"] == "AA"
    assert out["n2"] == "AA"
    assert out["n3"] == "AA"
    assert out["n4"] == "AAA"
    assert out["n5"] == ""
    assert out["n6"] == ""
    assert _FakeBatchModel.generate_batch_sizes == [2, 1]
    assert len(_FakeBatchTokenizer.decode_calls) == 3


def test_resolve_prostt5_auto_batch_size_uses_threads_by_default():
    batch_size = structural_alphabet._resolve_prostt5_auto_batch_size(
        threads=3,
        device="cpu",
        unique_sequence_count=100,
    )
    assert batch_size == 3


def test_resolve_prostt5_auto_batch_size_can_expand_on_cuda():
    batch_size = structural_alphabet._resolve_prostt5_auto_batch_size(
        threads=4,
        device="cuda",
        unique_sequence_count=100,
    )
    assert batch_size == 32


def test_resolve_prostt5_auto_batch_size_can_expand_on_mps():
    batch_size = structural_alphabet._resolve_prostt5_auto_batch_size(
        threads=2,
        device="mps",
        unique_sequence_count=100,
    )
    assert batch_size == 16


def test_local_prostt5_cache_key_does_not_hash_model_contents(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(b"model-v1")
    monkeypatch.setattr(
        structural_alphabet.resource_cache,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("model contents should not be read")),
    )
    first = structural_alphabet.get_prostt5_model_cache_key(
        {"prostt5_local_dir": str(model_dir)}
    )
    weight_path.write_bytes(b"model-version-2")
    second = structural_alphabet.get_prostt5_model_cache_key(
        {"prostt5_local_dir": str(model_dir)}
    )
    assert first.startswith("local-model@metadata-sha256:")
    assert first != second


def test_local_prostt5_cache_key_distinguishes_model_directories(tmp_path):
    first_dir = tmp_path / "model-a"
    second_dir = tmp_path / "model-b"
    first_dir.mkdir()
    second_dir.mkdir()
    first_weight = first_dir / "model.safetensors"
    second_weight = second_dir / "model.safetensors"
    first_weight.write_bytes(b"AAAA")
    second_weight.write_bytes(b"BBBB")
    shared_mtime_ns = 1_700_000_000_000_000_000
    os.utime(first_weight, ns=(shared_mtime_ns, shared_mtime_ns))
    os.utime(second_weight, ns=(shared_mtime_ns, shared_mtime_ns))

    first = structural_alphabet.get_prostt5_model_cache_key(
        {"prostt5_local_dir": str(first_dir)}
    )
    second = structural_alphabet.get_prostt5_model_cache_key(
        {"prostt5_local_dir": str(second_dir)}
    )

    assert first != second


def test_predict_3di_with_prostt5_uses_cache_without_loading_model(tmp_path, monkeypatch):
    cache_path = tmp_path / "prostt5_cache.tsv"
    model_key = structural_alphabet.get_prostt5_model_cache_key({})
    cache_path.write_text(
        "{}\tAC\tAA\n{}\tXX\tAA\n".format(model_key, model_key),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        structural_alphabet,
        "_load_prostt5_components",
        lambda g: (_ for _ in ()).throw(AssertionError("model should not be loaded on full cache hit")),
    )
    out = structural_alphabet.predict_3di_with_prostt5(
        aa_sequences={"n1": "AC", "n2": "XX"},
        g={"prostt5_cache": True, "prostt5_cache_file": str(cache_path), "threads": 1},
    )
    assert out["n1"] == "AA"
    assert out["n2"] == "AA"


def test_predict_3di_with_prostt5_appends_new_cache_entries(tmp_path, monkeypatch):
    cache_path = tmp_path / "prostt5_cache.tsv"
    tokenizer = _FakeBatchTokenizer()
    model = _FakeBatchModel()
    _FakeBatchTokenizer.reset()
    _FakeBatchModel.reset()
    monkeypatch.setattr(
        structural_alphabet,
        "_load_prostt5_components",
        lambda g: (_FakeTorchRuntime(), tokenizer, model, "cpu"),
    )
    out = structural_alphabet.predict_3di_with_prostt5(
        aa_sequences={"n1": "AC", "n2": "MNP"},
        g={"prostt5_cache": True, "prostt5_cache_file": str(cache_path), "threads": 1},
    )
    assert out["n1"] == "AA"
    assert out["n2"] == "AAA"
    cache_txt = cache_path.read_text(encoding="utf-8")
    model_key = structural_alphabet.get_prostt5_model_cache_key({})
    assert "{}\tAC\tAA".format(model_key) in cache_txt
    assert "{}\tMNP\tAAA".format(model_key) in cache_txt


def test_predict_3di_with_prostt5_prefers_batch_decode_when_available(monkeypatch):
    tokenizer = _FakeBatchTokenizerWithBatchDecode()
    model = _FakeBatchModel()
    _FakeBatchTokenizerWithBatchDecode.reset()
    _FakeBatchModel.reset()
    monkeypatch.setattr(
        structural_alphabet,
        "_load_prostt5_components",
        lambda g: (_FakeTorchRuntime(), tokenizer, model, "cpu"),
    )
    out = structural_alphabet.predict_3di_with_prostt5(
        aa_sequences={"n1": "AC", "n2": "AD"},
        g={"threads": 2, "prostt5_cache": False},
    )
    assert out["n1"] == "AA"
    assert out["n2"] == "AA"
    assert _FakeBatchTokenizerWithBatchDecode.batch_decode_calls == [2]
    assert _FakeBatchTokenizerWithBatchDecode.decode_calls == []

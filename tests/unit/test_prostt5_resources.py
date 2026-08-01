import pytest

from csubst import structural_alphabet


def test_resolve_prostt5_model_source_prefers_local_dir(tmp_path):
    g = {
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": str(tmp_path),
        "prostt5_no_download": True,
    }
    source, no_download = structural_alphabet._resolve_prostt5_model_source(g=g)
    assert source == str(tmp_path)
    assert no_download is True


def test_resolve_prostt5_model_source_uses_model_name_by_default():
    g = {
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": "",
        "prostt5_no_download": False,
    }
    source, no_download = structural_alphabet._resolve_prostt5_model_source(g=g)
    assert source == "Rostlab/ProstT5"
    assert no_download is False


def test_resolve_prostt5_model_source_rejects_missing_local_dir():
    g = {
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": "/path/that/does/not/exist",
        "prostt5_no_download": True,
    }
    with pytest.raises(ValueError, match="prostt5_local_dir"):
        structural_alphabet._resolve_prostt5_model_source(g=g)


class _FakeTokenizer:
    local_sources = set()
    local_only_calls = []
    download_calls = []
    save_calls = []

    def __init__(self, source):
        self.source = str(source)

    @classmethod
    def reset(cls):
        cls.local_sources = set()
        cls.local_only_calls = []
        cls.download_calls = []
        cls.save_calls = []

    @classmethod
    def from_pretrained(cls, source, do_lower_case=False, local_files_only=False, revision=None):
        source = str(source)
        cls.local_only_calls.append((source, bool(local_files_only)))
        if local_files_only:
            if source not in cls.local_sources:
                raise OSError("missing local tokenizer files")
            return cls(source=source)
        cls.download_calls.append(source)
        return cls(source=source)

    def save_pretrained(self, save_directory):
        save_directory = str(save_directory)
        type(self).save_calls.append(save_directory)
        type(self).local_sources.add(save_directory)


class _FakeModel:
    local_sources = set()
    local_only_calls = []
    download_calls = []
    save_calls = []

    def __init__(self, source):
        self.source = str(source)

    @classmethod
    def reset(cls):
        cls.local_sources = set()
        cls.local_only_calls = []
        cls.download_calls = []
        cls.save_calls = []

    @classmethod
    def from_pretrained(cls, source, local_files_only=False, revision=None):
        source = str(source)
        cls.local_only_calls.append((source, bool(local_files_only)))
        if local_files_only:
            if source not in cls.local_sources:
                raise OSError("missing local model files")
            return cls(source=source)
        cls.download_calls.append(source)
        return cls(source=source)

    def save_pretrained(self, save_directory):
        save_directory = str(save_directory)
        type(self).save_calls.append(save_directory)
        type(self).local_sources.add(save_directory)


def test_ensure_prostt5_model_files_downloads_into_local_dir_when_missing(tmp_path):
    _FakeTokenizer.reset()
    _FakeModel.reset()
    g = {
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": str(tmp_path),
        "prostt5_no_download": False,
    }
    model_source = structural_alphabet.ensure_prostt5_model_files(
        g=g,
        tokenizer_cls=_FakeTokenizer,
        model_cls=_FakeModel,
    )
    assert model_source == str(tmp_path)
    assert _FakeTokenizer.download_calls == ["Rostlab/ProstT5"]
    assert _FakeModel.download_calls == ["Rostlab/ProstT5"]
    assert str(tmp_path) in _FakeTokenizer.save_calls
    assert str(tmp_path) in _FakeModel.save_calls


def test_ensure_prostt5_model_files_respects_no_download_for_missing_local_dir(tmp_path):
    _FakeTokenizer.reset()
    _FakeModel.reset()
    g = {
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": str(tmp_path),
        "prostt5_no_download": True,
    }
    with pytest.raises(RuntimeError, match="prostt5_local_dir"):
        structural_alphabet.ensure_prostt5_model_files(
            g=g,
            tokenizer_cls=_FakeTokenizer,
            model_cls=_FakeModel,
        )

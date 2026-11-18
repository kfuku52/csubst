import json
import io
import os
from pathlib import Path

import pytest

from csubst import resource_cache
from csubst import structure_resources


def test_alphafold_latest_tracks_api_and_explicit_pin_reuses_previous_model(tmp_path, monkeypatch):
    root = 'https://alphafold.ebi.ac.uk'
    api = root + '/api/prediction/P12345'
    calls = []
    version = 6

    def open_url(url, timeout):
        assert timeout == 9.0
        calls.append(url)
        pdb_url = f'{root}/files/new-layout-{version}.pdb'
        if url == api:
            predictions = [
                {'entryId': 'AF-P12345-F2', 'latestVersion': 8, 'pdbUrl': root + '/other.pdb'},
                {'modelEntityId': 'AF-P12345-F1', 'latestVersion': version, 'pdbUrl': pdb_url},
            ]
            return io.BytesIO(json.dumps(predictions).encode())
        assert url == pdb_url
        return io.BytesIO(f'MODEL {version}'.encode())

    monkeypatch.setattr(structure_resources.urllib.request, 'urlopen', open_url)
    options = dict(network_timeout=9.0, cache_dir=tmp_path)
    path6 = Path(structure_resources.ensure_alphafold_structure('P12345', 'latest', **options))
    assert structure_resources.ensure_alphafold_structure('P12345', 'latest', **options) == str(path6)
    version = 7
    path7 = Path(structure_resources.ensure_alphafold_structure('P12345', 'latest', **options))
    assert structure_resources.ensure_alphafold_structure('P12345', '6', **options) == str(path6)
    assert calls == [api, root + '/files/new-layout-6.pdb', api, api, root + '/files/new-layout-7.pdb']
    assert path6.read_bytes() == b'MODEL 6' and path7.read_bytes() == b'MODEL 7'
    meta = json.loads((path6.parent / resource_cache.RESOURCE_MANIFEST_NAME).read_text())['metadata']
    assert (meta['model_id'], meta['model_version'], meta['url']) == (
        'AF-P12345-F1', 6, root + '/files/new-layout-6.pdb',
    )


def test_alphafold_old_cache_supports_offline_pin_and_missing_pin_stays_pinned(tmp_path):
    filename = 'AF-P12345-F1-model_v2.pdb'
    base = 'https://alphafold.ebi.ac.uk/files/'
    path2 = structure_resources.ensure_remote_structure(
        'alphafold', 'P12345', base + filename, filename, cache_dir=tmp_path, download_bytes=lambda *_: b'old',
    )
    urls = []

    def unavailable(url, timeout):
        urls.append(url)
        raise FileNotFoundError(url)

    options = dict(network_timeout=5, cache_dir=tmp_path, download_bytes=unavailable)
    assert structure_resources.ensure_alphafold_structure('P12345', '2', **options) == path2
    with pytest.raises(FileNotFoundError):
        structure_resources.ensure_alphafold_structure('P12345', '5', **options)
    assert urls == [base + 'AF-P12345-F1-model_v5.pdb']


def test_remote_structure_is_atomically_cached_and_reused(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    calls = []

    def download(url, timeout_seconds):
        calls.append((url, timeout_seconds))
        return b"data_demo\n#\n"

    kwargs = {
        "source": "rcsb",
        "structure_id": "1ABC",
        "url": "https://files.rcsb.org/download/1ABC.cif",
        "filename": "1abc.cif",
        "cache_dir": cache_dir,
        "network_timeout": 12.5,
        "poll_seconds": 0.01,
        "lock_timeout_seconds": 1.0,
    }
    structure_path = structure_resources.ensure_remote_structure(
        **kwargs,
        download_bytes=download,
    )
    assert structure_path.startswith(str(cache_dir / "structures"))
    assert open(structure_path, mode="rb").read() == b"data_demo\n#\n"
    assert calls == [("https://files.rcsb.org/download/1ABC.cif", 12.5)]
    assert list(workdir.iterdir()) == []

    structure_path_again = structure_resources.ensure_remote_structure(
        **kwargs,
        download_bytes=lambda *_args: (_ for _ in ()).throw(AssertionError("should use cache")),
    )
    assert structure_path_again == structure_path
    manifest_path = os.path.join(os.path.dirname(structure_path), resource_cache.RESOURCE_MANIFEST_NAME)
    manifest = json.loads(open(manifest_path, encoding="utf-8").read())
    assert manifest["metadata"]["source"] == "rcsb"
    assert manifest["metadata"]["structure_id"] == "1ABC"
    assert manifest["files"]["1abc.cif"]["size"] == len(b"data_demo\n#\n")


def test_remote_structure_rejects_path_as_filename(tmp_path):
    with pytest.raises(ValueError, match="file name, not a path"):
        structure_resources.ensure_remote_structure(
            source="rcsb",
            structure_id="1ABC",
            url="https://example.com/1abc.cif",
            filename="nested/1abc.cif",
            cache_dir=tmp_path,
            download_bytes=lambda *_args: b"data_demo\n",
        )


def test_ensure_rcsb_structure_uses_pinned_cache_location_and_url(tmp_path):
    observed = {}

    def download(url, timeout_seconds):
        observed["url"] = url
        observed["timeout"] = timeout_seconds
        return b"data_3zgb\n#\n"

    structure_path = structure_resources.ensure_rcsb_structure(
        pdb_id="3ZGB",
        cache_dir=tmp_path / "cache",
        network_timeout=8,
        poll_seconds=0.01,
        lock_timeout_seconds=1,
        download_bytes=download,
    )
    assert observed == {
        "url": "https://files.rcsb.org/download/3ZGB.cif",
        "timeout": 8.0,
    }
    assert structure_path.endswith(os.path.join("structures", "rcsb", "3zgb", structure_path.split(os.sep)[-2], "3zgb.cif"))

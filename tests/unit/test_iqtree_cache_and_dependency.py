import os
import pathlib

import pytest

from csubst import parser_iqtree
from csubst import runtime
from csubst import ete


def test_infer_iqtree_output_prefix_from_alignment_uses_shared_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    observed = parser_iqtree._infer_iqtree_output_prefix_from_alignment("input.fa.gz")
    assert os.path.dirname(observed) == str((tmp_path / "csubst_iqtree").resolve())
    assert os.path.basename(observed).startswith("input.fa.gz.")
    uncompressed = parser_iqtree._infer_iqtree_output_prefix_from_alignment("input.fa")
    assert os.path.basename(uncompressed).startswith("input.fa.")
    assert uncompressed != observed


def test_infer_iqtree_output_prefix_is_stable_when_work_directory_moves(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = runtime.infer_iqtree_output_prefix(
        alignment_file=first_root / "inputs" / "alignment.fa.gz",
        iqtree_outdir=first_root / "csubst_iqtree",
        base_dir=first_root,
    )
    second = runtime.infer_iqtree_output_prefix(
        alignment_file=second_root / "inputs" / "alignment.fa.gz",
        iqtree_outdir=second_root / "csubst_iqtree",
        base_dir=second_root,
    )
    assert os.path.basename(first) == os.path.basename(second)


def test_check_intermediate_files_infer_uses_shared_iqtree_dir_for_gz_alignment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prefix = parser_iqtree._infer_iqtree_output_prefix_from_alignment("alignment.fa.gz")
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    for ext in ["iqtree", "log", "rate", "state", "treefile"]:
        pathlib.Path(prefix + "." + ext).write_text("x\n", encoding="utf-8")
    g = {
        "alignment_file": "alignment.fa.gz",
        "iqtree_iqtree": "infer",
        "iqtree_log": "infer",
        "iqtree_rate": "infer",
        "iqtree_state": "infer",
        "iqtree_treefile": "infer",
    }
    out, all_exist = parser_iqtree.check_intermediate_files(g)
    assert all_exist is True
    assert out["path_iqtree_iqtree"] == str(prefix) + ".iqtree"
    assert out["path_iqtree_log"] == str(prefix) + ".log"
    assert out["path_iqtree_rate"] == prefix + ".rate"
    assert out["path_iqtree_state"] == str(prefix) + ".state"
    assert out["path_iqtree_treefile"] == str(prefix) + ".treefile"


def test_iqtree_manifest_invalidates_reuse_when_alignment_changes(tmp_path):
    alignment = tmp_path / "alignment.fa"
    alignment.write_text(">A\nAAA\n>B\nAAA\n", encoding="utf-8")
    state_path = tmp_path / "alignment.state"
    g = {
        "alignment_file": str(alignment),
        "path_iqtree_state": str(state_path),
        "rooted_tree": ete.PhyloNode("(A:1,B:1)R;", format=1),
        "iqtree_exe": "/usr/bin/false",
        "iqtree_version": "2.3.6",
        "iqtree_model": "MG",
        "genetic_code": 1,
        "threads": 2,
    }
    parser_iqtree._write_iqtree_manifest(g)
    compatible, reason = parser_iqtree.is_iqtree_manifest_compatible(g)
    assert compatible is True
    assert reason == ""
    alignment.write_text(">A\nAAA\n>B\nAAG\n", encoding="utf-8")
    compatible, reason = parser_iqtree.is_iqtree_manifest_compatible(g)
    assert compatible is False
    assert "changed" in reason


def test_iqtree_manifest_ignores_location_runtime_version_and_threads(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    alignment = first_dir / "alignment.fa"
    alignment.write_text(">A\nAAA\n>B\nAAA\n", encoding="utf-8")
    state_path = first_dir / "alignment.state"
    rooted_tree = ete.PhyloNode("(A:1,B:1)R;", format=1)
    g = {
        "alignment_file": str(alignment),
        "path_iqtree_state": str(state_path),
        "rooted_tree": rooted_tree,
        "iqtree_exe": "/first/iqtree",
        "iqtree_version": "2.3.6",
        "iqtree_model": "MG",
        "genetic_code": 1,
        "threads": 2,
    }
    parser_iqtree._write_iqtree_manifest(g)
    moved_alignment = second_dir / alignment.name
    moved_alignment.write_bytes(alignment.read_bytes())
    moved_state = second_dir / state_path.name
    moved_manifest = second_dir / (state_path.name + ".csubst-manifest.json")
    moved_manifest.write_bytes((first_dir / moved_manifest.name).read_bytes())
    moved_g = dict(g)
    moved_g.update({
        "alignment_file": str(moved_alignment),
        "path_iqtree_state": str(moved_state),
        "iqtree_exe": "/second/iqtree",
        "iqtree_version": "3.0.1",
        "threads": 16,
    })
    compatible, reason = parser_iqtree.is_iqtree_manifest_compatible(moved_g)
    assert compatible is True
    assert reason == ""


def test_parse_iqtree_version_text_for_v2_and_v3():
    v2_txt = "IQ-TREE multicore version 2.3.6 for MacOS Intel 64-bit built Aug  4 2024"
    v3_txt = "IQ-TREE multicore version 3.0.1 for Linux 64-bit built Jan  1 2025"
    assert parser_iqtree._parse_iqtree_version_text(v2_txt) == ("2.3.6", 2)
    assert parser_iqtree._parse_iqtree_version_text(v3_txt) == ("3.0.1", 3)


def test_detect_iqtree_output_version_handles_non_utf8_iqtree_file(tmp_path):
    iqtree_file = tmp_path / "sample.iqtree"
    iqtree_file.write_bytes(b"\xff\xfeIQ-TREE multicore version 2.3.6 for Linux 64-bit\n")
    g = {"path_iqtree_iqtree": str(iqtree_file), "path_iqtree_log": str(tmp_path / "missing.log")}
    out = parser_iqtree.detect_iqtree_output_version(g)
    assert out["iqtree_output_version"] == "2.3.6"
    assert out["iqtree_output_version_major"] == 2


def test_is_version_at_least_handles_numeric_versions():
    assert parser_iqtree._is_version_at_least("2.0.0", "2.0.0")
    assert parser_iqtree._is_version_at_least("2.3.6", "2.0.0")
    assert parser_iqtree._is_version_at_least("3.0.0", "2.0.0")
    assert not parser_iqtree._is_version_at_least("1.6.12", "2.0.0")


def test_check_iqtree_dependency_rejects_nonzero_exit(monkeypatch):
    class _FakeProc:
        returncode = 1
        stdout = b""
        stderr = b""

    monkeypatch.setattr(parser_iqtree.subprocess, "run", lambda *args, **kwargs: _FakeProc())
    with pytest.raises(AssertionError, match="iqtree PATH cannot be found"):
        parser_iqtree.check_iqtree_dependency({"iqtree_exe": "iqtree"})


def test_check_iqtree_dependency_handles_non_utf8_version_output(monkeypatch):
    class _FakeProc:
        returncode = 0
        stdout = b"\xff\xfeIQ-TREE multicore version 2.3.6\n"
        stderr = b""

    monkeypatch.setattr(parser_iqtree.subprocess, "run", lambda *args, **kwargs: _FakeProc())
    g = {"iqtree_exe": "iqtree"}
    parser_iqtree.check_iqtree_dependency(g)
    assert g["iqtree_version"] == "2.3.6"
    assert g["iqtree_version_major"] == 2

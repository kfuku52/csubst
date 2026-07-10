import json

import pandas as pd
import pytest

from csubst import main_doctor


def _base_doctor_config(tmp_path, **overrides):
    iqtree_dir = tmp_path / "csubst_iqtree"
    iqtree_dir.mkdir(exist_ok=True)
    cfg = {
        "alignment_file": str(tmp_path / "alignment.fa"),
        "rooted_tree_file": str(tmp_path / "tree.nwk"),
        "foreground": str(tmp_path / "foreground.txt"),
        "fg_format": 1,
        "fg_stem_only": True,
        "iqtree_exe": "iqtree",
        "iqtree_treefile": str(iqtree_dir / "alignment.fa.treefile"),
        "iqtree_state": str(iqtree_dir / "alignment.fa.state"),
        "iqtree_rate": str(iqtree_dir / "alignment.fa.rate"),
        "iqtree_iqtree": str(iqtree_dir / "alignment.fa.iqtree"),
        "iqtree_log": str(iqtree_dir / "alignment.fa.log"),
        "nonsyn_recode": "no",
        "sa_asr_mode": "direct",
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": "",
        "sa_state_cache": "auto",
        "sa_state_cache_file": str(tmp_path / "csubst_3di_state_cache.npz"),
        "check_iqtree_exe": False,
        "doctor_fail_level": "error",
        "outdir": str(tmp_path / "doctor"),
        "output_prefix": "csubst",
        "log_file": "",
    }
    cfg.update(overrides)
    return cfg


def _write_doctor_inputs(tmp_path, mismatch_taxa=False):
    (tmp_path / "alignment.fa").write_text(">s1\nATGATG\n>s2\nATGATA\n", encoding="utf-8")
    tree_text = "(s1:1,s2:1);\n"
    if mismatch_taxa:
        tree_text = "(s1:1,s3:1);\n"
    (tmp_path / "tree.nwk").write_text(tree_text, encoding="utf-8")
    (tmp_path / "foreground.txt").write_text("1\ts1\n", encoding="utf-8")
    iqtree_dir = tmp_path / "csubst_iqtree"
    iqtree_dir.mkdir(exist_ok=True)
    for suffix in ["treefile", "state", "rate", "iqtree", "log"]:
        (iqtree_dir / ("alignment.fa." + suffix)).write_text("stub\n", encoding="utf-8")


def test_main_doctor_writes_summary_outputs_for_valid_inputs(tmp_path):
    _write_doctor_inputs(tmp_path)
    g = _base_doctor_config(tmp_path)

    main_doctor.main_doctor(g)

    summary_tsv = tmp_path / "doctor" / "csubst_doctor_summary.tsv"
    summary_json = tmp_path / "doctor" / "csubst_doctor_summary.json"
    manifest_tsv = tmp_path / "doctor" / "csubst_outputs.tsv"
    assert summary_tsv.exists() is True
    assert summary_json.exists() is True
    assert manifest_tsv.exists() is True
    df = pd.read_csv(summary_tsv, sep="\t")
    assert (df["status"] == "fail").sum() == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["counts"]["fail"] == 0
    manifest_df = pd.read_csv(manifest_tsv, sep="\t")
    assert (manifest_df["output_kind"] == "doctor_summary_tsv").any()
    assert (manifest_df["output_kind"] == "doctor_summary_json").any()
    assert (manifest_df["output_kind"] == "output_manifest").any()


def test_main_doctor_raises_after_writing_summary_for_taxon_mismatch(tmp_path):
    _write_doctor_inputs(tmp_path, mismatch_taxa=True)
    g = _base_doctor_config(tmp_path)

    with pytest.raises(ValueError, match="Doctor checks found issues"):
        main_doctor.main_doctor(g)

    summary_tsv = tmp_path / "doctor" / "csubst_doctor_summary.tsv"
    assert summary_tsv.exists() is True
    df = pd.read_csv(summary_tsv, sep="\t")
    row = df.loc[df["check_name"] == "tree_alignment_taxa", :].iloc[0]
    assert row["status"] == "fail"


def test_main_doctor_warning_level_allows_explicitly_skipped_optional_checks(tmp_path):
    _write_doctor_inputs(tmp_path)
    g = _base_doctor_config(
        tmp_path,
        doctor_fail_level="warning",
        check_iqtree_exe=False,
    )

    main_doctor.main_doctor(g)

    summary_tsv = tmp_path / "doctor" / "csubst_doctor_summary.tsv"
    df = pd.read_csv(summary_tsv, sep="\t")
    row = df.loc[df["check_name"] == "iqtree_exe_check", :].iloc[0]
    assert row["status"] == "skip"


def test_main_doctor_treats_string_zero_foreground_rows_as_background(tmp_path):
    _write_doctor_inputs(tmp_path)
    (tmp_path / "foreground.txt").write_text("name\ttraitA\ns1\tFG1\ns2\t0\n", encoding="utf-8")
    g = _base_doctor_config(tmp_path, fg_format=2)

    main_doctor.main_doctor(g)

    summary_tsv = tmp_path / "doctor" / "csubst_doctor_summary.tsv"
    df = pd.read_csv(summary_tsv, sep="\t")
    row = df.loc[df["check_name"] == "foreground_trait_traitA_matches", :].iloc[0]
    assert row["status"] == "pass"
    assert "matched 1 leaves" in row["message"]

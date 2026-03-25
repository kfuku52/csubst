import json
import os

import pandas as pd
import pytest

from csubst import main_benchmark
from csubst import runtime


def _base_benchmark_config(tmp_path, **overrides):
    cfg = {
        "outdir": str(tmp_path / "benchmark"),
        "output_prefix": "csubst",
        "log_file": "",
        "expectation_method": "codon_model",
        "asrv": "each",
        "nonsyn_recode": "no",
        "sa_asr_mode": "direct",
        "pseudocount_mode": "none",
        "pseudocount_alpha": "0.0",
        "pseudocount_target": "both",
        "pseudocount_report": False,
        "calc_omega_pvalue": False,
        "full_cds_alignment_file": "",
        "benchmark_expectation_methods": "",
        "benchmark_asrv_modes": "",
        "benchmark_nonsyn_recode_modes": "",
        "benchmark_sa_asr_modes": "",
        "benchmark_pseudocount_modes": "",
        "benchmark_keep_going": True,
        "benchmark_score_column": "omegaCany2spe",
        "benchmark_ocn_column": "OCNany2spe",
        "benchmark_min_score": 5.0,
        "benchmark_min_ocn": 2.0,
        "cb": True,
    }
    cfg.update(overrides)
    return runtime.ensure_run_context(cfg)


def test_iter_benchmark_configs_expands_default_3di_asr_modes():
    g = runtime.ensure_run_context(
        {
            "expectation_method": "codon_model",
            "asrv": "each",
            "nonsyn_recode": "no",
            "sa_asr_mode": "direct",
            "pseudocount_mode": "none",
            "benchmark_expectation_methods": "",
            "benchmark_asrv_modes": "",
            "benchmark_nonsyn_recode_modes": "no,3di20",
            "benchmark_sa_asr_modes": "",
            "benchmark_pseudocount_modes": "",
        }
    )
    configs = main_benchmark._iter_benchmark_configs(g)
    labels = [cfg["label"] for cfg in configs]
    assert any(["recode-no" in label for label in labels])
    assert any([("recode-3di20" in label) and ("sa-direct" in label) for label in labels])
    assert any([("recode-3di20" in label) and ("sa-translate" in label) for label in labels])


def test_iter_benchmark_configs_rejects_invalid_grid_tokens():
    g = runtime.ensure_run_context(
        {
            "expectation_method": "codon_model",
            "asrv": "each",
            "nonsyn_recode": "no",
            "sa_asr_mode": "direct",
            "pseudocount_mode": "none",
            "benchmark_expectation_methods": "bogus",
            "benchmark_asrv_modes": "",
            "benchmark_nonsyn_recode_modes": "",
            "benchmark_sa_asr_modes": "",
            "benchmark_pseudocount_modes": "",
        }
    )
    with pytest.raises(ValueError, match="Unsupported benchmark expectation method"):
        main_benchmark._iter_benchmark_configs(g)


def test_iter_benchmark_configs_deduplicates_asrv_for_codon_model():
    g = runtime.ensure_run_context(
        {
            "expectation_method": "codon_model",
            "asrv": "no",
            "nonsyn_recode": "no",
            "sa_asr_mode": "direct",
            "pseudocount_mode": "none",
            "benchmark_expectation_methods": "codon_model,urn",
            "benchmark_asrv_modes": "no,each,file",
            "benchmark_nonsyn_recode_modes": "",
            "benchmark_sa_asr_modes": "",
            "benchmark_pseudocount_modes": "",
        }
    )

    configs = main_benchmark._iter_benchmark_configs(g)

    codon_model_configs = [cfg for cfg in configs if cfg["expectation_method"] == "codon_model"]
    urn_configs = [cfg for cfg in configs if cfg["expectation_method"] == "urn"]
    assert len(codon_model_configs) == 1
    assert codon_model_configs[0]["asrv"] == "no"
    assert sorted([cfg["asrv"] for cfg in urn_configs]) == ["each", "file", "no"]


def test_main_benchmark_writes_summary_for_pass_and_fail_runs(tmp_path, monkeypatch):
    g = _base_benchmark_config(
        tmp_path,
        benchmark_expectation_methods="codon_model,urn",
    )

    def _fake_main_analyze(local_g):
        if local_g["expectation_method"] == "urn":
            raise ValueError("synthetic failure")
        df = pd.DataFrame(
            {
                "OCNany2spe": [1.0, 3.0],
                "omegaCany2spe": [4.0, 7.0],
            }
        )
        df.to_csv(runtime.output_path(local_g, "cb_2.tsv"), sep="\t", index=False)

    monkeypatch.setattr(main_benchmark.main_analyze, "main_analyze", _fake_main_analyze)

    main_benchmark.main_benchmark(g)

    summary_tsv = tmp_path / "benchmark" / "csubst_benchmark_summary.tsv"
    summary_json = tmp_path / "benchmark" / "csubst_benchmark_summary.json"
    manifest_tsv = tmp_path / "benchmark" / "csubst_outputs.tsv"
    assert summary_tsv.exists() is True
    assert summary_json.exists() is True
    assert manifest_tsv.exists() is True
    summary = pd.read_csv(summary_tsv, sep="\t")
    with open(summary_json, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert summary.shape[0] == 2
    assert payload["counts"] == {"pass": 1, "fail": 1}
    assert payload["score_column"] == "omegaCany2spe"
    assert len(payload["rows"]) == 2
    passed = summary.loc[summary["status"] == "pass", :].iloc[0]
    failed = summary.loc[summary["status"] == "fail", :].iloc[0]
    assert passed["hit_rows"] == 1
    assert passed["score_max"] == 7.0
    assert "synthetic failure" in failed["error_message"]
    assert passed["search_log"] != ""
    assert failed["search_log"] != ""
    assert (summary.loc[:, "search_log"] != summary.loc[:, "run_log"]).all()
    assert pd.Series(summary["search_log"]).map(lambda path: os.path.exists(path)).all()
    manifest = pd.read_csv(manifest_tsv, sep="\t")
    assert (manifest["output_kind"] == "benchmark_summary_tsv").any()
    assert (manifest["output_kind"] == "benchmark_summary_json").any()
    assert (manifest["output_kind"] == "benchmark_run_log").any()
    assert (manifest["output_kind"] == "benchmark_search_log").any()
    assert (manifest["output_kind"] == "output_manifest").any()
    assert (manifest.loc[manifest["output_kind"] == "benchmark_search_log", "file_exists"] == "Y").all()


def test_prepare_run_context_keeps_shared_cache_paths_at_benchmark_root(tmp_path):
    g = _base_benchmark_config(
        tmp_path,
        nonsyn_recode="3di20",
        full_cds_alignment_file="full.fa",
        alignment_file="aln.fa",
        prostt5_cache_file="shared/prostt5.tsv",
        sa_state_cache_file="shared/sa_cache.npz",
    )
    config = {
        "label": "demo",
        "expectation_method": "codon_model",
        "asrv": "each",
        "nonsyn_recode": "3di20",
        "sa_asr_mode": "direct",
        "pseudocount_mode": "none",
    }

    local_a = main_benchmark._prepare_run_context(g, config, str(tmp_path / "benchmark" / "runs" / "001.demo"))
    local_b = main_benchmark._prepare_run_context(g, config, str(tmp_path / "benchmark" / "runs" / "002.demo"))

    expected_prostt5 = str(tmp_path / "benchmark" / "shared" / "prostt5.tsv")
    expected_sa_cache = str(tmp_path / "benchmark" / "shared" / "sa_cache.npz")
    assert local_a["prostt5_cache_file"] == expected_prostt5
    assert local_b["prostt5_cache_file"] == expected_prostt5
    assert local_a["sa_state_cache_file"] == expected_sa_cache
    assert local_b["sa_state_cache_file"] == expected_sa_cache
    assert str(tmp_path / "benchmark" / "runs" / "001.demo") not in local_a["prostt5_cache_file"]
    assert str(tmp_path / "benchmark" / "runs" / "002.demo") not in local_b["sa_state_cache_file"]


def test_prepare_run_context_retargets_inferred_iqtree_paths_for_3di20(tmp_path):
    iqtree_outdir = tmp_path / "iqtree"
    trimmed_prefix = runtime.infer_iqtree_output_prefix(
        alignment_file="trimmed.fa",
        iqtree_outdir=str(iqtree_outdir),
    )
    g = _base_benchmark_config(
        tmp_path,
        alignment_file="trimmed.fa",
        full_cds_alignment_file="full.fa",
        iqtree_outdir=str(iqtree_outdir),
        iqtree_treefile=trimmed_prefix + ".treefile",
        iqtree_state=trimmed_prefix + ".state",
        iqtree_rate=trimmed_prefix + ".rate",
        iqtree_iqtree=trimmed_prefix + ".iqtree",
        iqtree_log=trimmed_prefix + ".log",
        path_iqtree_treefile=trimmed_prefix + ".treefile",
        path_iqtree_state=trimmed_prefix + ".state",
        path_iqtree_rate=trimmed_prefix + ".rate",
        path_iqtree_iqtree=trimmed_prefix + ".iqtree",
        path_iqtree_log=trimmed_prefix + ".log",
    )
    config = {
        "label": "demo",
        "expectation_method": "codon_model",
        "asrv": "each",
        "nonsyn_recode": "3di20",
        "sa_asr_mode": "direct",
        "pseudocount_mode": "none",
    }

    local_g = main_benchmark._prepare_run_context(g, config, str(tmp_path / "benchmark" / "runs" / "001.demo"))

    full_prefix = runtime.infer_iqtree_output_prefix(
        alignment_file="full.fa",
        iqtree_outdir=str(iqtree_outdir),
    )
    assert local_g["alignment_file"] == "full.fa"
    assert local_g["iqtree_treefile"] == full_prefix + ".treefile"
    assert local_g["iqtree_state"] == full_prefix + ".state"
    assert local_g["iqtree_rate"] == full_prefix + ".rate"
    assert local_g["path_iqtree_state"] == full_prefix + ".state"
    assert local_g["path_iqtree_rate"] == full_prefix + ".rate"


def test_main_benchmark_rejects_nonfinite_thresholds(tmp_path):
    g = _base_benchmark_config(
        tmp_path,
        benchmark_min_score=float("nan"),
    )
    with pytest.raises(ValueError, match="benchmark_min_score"):
        main_benchmark.main_benchmark(g)


def test_main_benchmark_requires_cb_output(tmp_path):
    g = _base_benchmark_config(
        tmp_path,
        cb=False,
    )
    with pytest.raises(ValueError, match="requires --cb yes"):
        main_benchmark.main_benchmark(g)


def test_run_single_config_refreshes_search_log_on_rerun(tmp_path, monkeypatch):
    g = _base_benchmark_config(tmp_path)
    config = {
        "label": "demo",
        "expectation_method": "codon_model",
        "asrv": "each",
        "nonsyn_recode": "no",
        "sa_asr_mode": "direct",
        "pseudocount_mode": "none",
    }
    run_dir = str(tmp_path / "benchmark" / "runs" / "001.demo")

    def _fake_first(local_g):
        print("RUN1")

    def _fake_second(local_g):
        print("RUN2")

    monkeypatch.setattr(main_benchmark.main_analyze, "main_analyze", _fake_first)
    first = main_benchmark._run_single_config(g, config, run_dir)
    monkeypatch.setattr(main_benchmark.main_analyze, "main_analyze", _fake_second)
    second = main_benchmark._run_single_config(g, config, run_dir)

    search_log_text = open(second["search_log"], encoding="utf-8").read()
    benchmark_log_text = open(second["run_log"], encoding="utf-8").read()
    assert "RUN1" not in search_log_text
    assert "RUN2" in search_log_text
    assert search_log_text == benchmark_log_text


def test_run_single_config_clears_stale_cb_summary_on_failure(tmp_path, monkeypatch):
    g = _base_benchmark_config(tmp_path)
    config = {
        "label": "demo",
        "expectation_method": "codon_model",
        "asrv": "each",
        "nonsyn_recode": "no",
        "sa_asr_mode": "direct",
        "pseudocount_mode": "none",
    }
    run_dir = str(tmp_path / "benchmark" / "runs" / "001.demo")

    def _fake_first(local_g):
        pd.DataFrame(
            {
                "OCNany2spe": [3.0],
                "omegaCany2spe": [7.0],
            }
        ).to_csv(runtime.output_path(local_g, "cb_2.tsv"), sep="\t", index=False)

    def _fake_second(local_g):
        raise ValueError("boom")

    monkeypatch.setattr(main_benchmark.main_analyze, "main_analyze", _fake_first)
    first = main_benchmark._run_single_config(g, config, run_dir)
    monkeypatch.setattr(main_benchmark.main_analyze, "main_analyze", _fake_second)
    second = main_benchmark._run_single_config(g, config, run_dir)

    assert first["status"] == "pass"
    assert first["cb_rows"] == 1
    assert second["status"] == "fail"
    assert second["cb_rows"] == 0
    assert second["hit_rows"] == 0
    assert pd.isna(second["score_max"])
    assert os.path.exists(second["cb_tsv"]) is False


def test_run_single_config_fails_when_requested_summary_columns_are_missing(tmp_path, monkeypatch):
    g = _base_benchmark_config(
        tmp_path,
        benchmark_score_column="typo_score",
    )
    config = {
        "label": "demo",
        "expectation_method": "codon_model",
        "asrv": "each",
        "nonsyn_recode": "no",
        "sa_asr_mode": "direct",
        "pseudocount_mode": "none",
    }
    run_dir = str(tmp_path / "benchmark" / "runs" / "001.demo")

    def _fake_main_analyze(local_g):
        pd.DataFrame(
            {
                "OCNany2spe": [3.0],
                "omegaCany2spe": [7.0],
            }
        ).to_csv(runtime.output_path(local_g, "cb_2.tsv"), sep="\t", index=False)

    monkeypatch.setattr(main_benchmark.main_analyze, "main_analyze", _fake_main_analyze)
    result = main_benchmark._run_single_config(g, config, run_dir)

    assert result["status"] == "fail"
    assert 'benchmark_score_column "typo_score" was not found' in result["error_message"]
    assert open(result["run_log"], encoding="utf-8").read().count("Benchmark validation failure:") == 1


def test_main_benchmark_writes_logs_for_preparation_failures(tmp_path):
    g = _base_benchmark_config(
        tmp_path,
        benchmark_expectation_methods="codon_model",
        benchmark_nonsyn_recode_modes="3di20",
        benchmark_sa_asr_modes="direct",
        output_manifest=True,
    )

    main_benchmark.main_benchmark(g)

    summary = pd.read_csv(tmp_path / "benchmark" / "csubst_benchmark_summary.tsv", sep="\t")
    manifest = pd.read_csv(tmp_path / "benchmark" / "csubst_outputs.tsv", sep="\t")
    failed = summary.iloc[0]
    assert failed["status"] == "fail"
    assert os.path.exists(failed["run_log"])
    assert os.path.exists(failed["search_log"])
    assert "requires --full_cds_alignment_file" in open(failed["run_log"], encoding="utf-8").read()
    log_rows = manifest.loc[
        manifest["output_kind"].isin(["benchmark_run_log", "benchmark_search_log"]),
        :,
    ]
    assert log_rows.shape[0] == 2
    assert (log_rows.loc[:, "file_exists"] == "Y").all()
    assert (manifest["output_kind"] == "benchmark_cb_tsv").sum() == 0


def test_main_benchmark_preparation_failure_ignores_stale_cb_files(tmp_path):
    g = _base_benchmark_config(
        tmp_path,
        benchmark_expectation_methods="codon_model",
        benchmark_nonsyn_recode_modes="3di20",
        benchmark_sa_asr_modes="direct",
        output_manifest=True,
    )
    run_dir = tmp_path / "benchmark" / "runs" / "001.exp-codon_model.asrv-each.recode-3di20.pc-none.sa-direct"
    run_dir.mkdir(parents=True, exist_ok=True)
    stale_cb = run_dir / "csubst_cb_2.tsv"
    pd.DataFrame(
        {
            "OCNany2spe": [3.0],
            "omegaCany2spe": [7.0],
        }
    ).to_csv(stale_cb, sep="\t", index=False)

    main_benchmark.main_benchmark(g)

    summary = pd.read_csv(tmp_path / "benchmark" / "csubst_benchmark_summary.tsv", sep="\t")
    manifest = pd.read_csv(tmp_path / "benchmark" / "csubst_outputs.tsv", sep="\t")
    failed = summary.iloc[0]
    assert failed["status"] == "fail"
    assert failed["cb_rows"] == 0
    assert failed["hit_rows"] == 0
    assert pd.isna(failed["score_max"])
    assert stale_cb.exists() is False
    assert (manifest["output_kind"] == "benchmark_cb_tsv").sum() == 0

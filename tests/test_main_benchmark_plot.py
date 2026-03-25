import json
import os

import pandas as pd
import pytest

from csubst import main_benchmark_plot
from csubst import runtime


def _write_benchmark_summary(dir_path, rows):
    dir_path.mkdir(parents=True, exist_ok=True)
    summary_path = dir_path / "csubst_benchmark_summary.tsv"
    pd.DataFrame(rows).to_csv(summary_path, sep="\t", index=False)
    return summary_path


def _write_benchmark_manifest(dir_path, summary_path):
    manifest_path = dir_path / "csubst_outputs.tsv"
    pd.DataFrame(
        [
            {
                "output_kind": "benchmark_summary_tsv",
                "output_file": os.path.relpath(summary_path, start=dir_path),
                "output_path": str(summary_path),
                "file_exists": "Y",
            }
        ]
    ).to_csv(manifest_path, sep="\t", index=False)
    return manifest_path


def test_main_benchmark_plot_collects_manifest_and_fallback_outputs(tmp_path):
    bench_a = tmp_path / "bench_a"
    bench_b = tmp_path / "nested" / "bench_b"
    summary_a = _write_benchmark_summary(
        bench_a,
        [
            {
                "label": "exp-codon_model.asrv-each.recode-no.pc-none",
                "status": "pass",
                "expectation_method": "codon_model",
                "asrv": "each",
                "nonsyn_recode": "no",
                "sa_asr_mode": "direct",
                "pseudocount_mode": "none",
                "elapsed_sec": 12.0,
                "hit_rows": 3,
                "score_max": 7.0,
            },
            {
                "label": "exp-urn.asrv-each.recode-no.pc-none",
                "status": "fail",
                "expectation_method": "urn",
                "asrv": "each",
                "nonsyn_recode": "no",
                "sa_asr_mode": "direct",
                "pseudocount_mode": "none",
                "elapsed_sec": None,
                "hit_rows": 0,
                "score_max": None,
            },
        ],
    )
    _write_benchmark_manifest(bench_a, summary_a)
    _write_benchmark_summary(
        bench_b,
        [
            {
                "label": "exp-codon_model.asrv-file.recode-3di20.pc-empirical.sa-translate",
                "status": "pass",
                "expectation_method": "codon_model",
                "asrv": "file",
                "nonsyn_recode": "3di20",
                "sa_asr_mode": "translate",
                "pseudocount_mode": "empirical",
                "elapsed_sec": 20.0,
                "hit_rows": 5,
                "score_max": 9.0,
            }
        ],
    )
    g = runtime.ensure_run_context(
        {
            "benchmark_dir": str(tmp_path),
            "benchmark_plot_recursive": True,
            "benchmark_plot_metrics": "elapsed_sec,hit_rows,score_max",
            "benchmark_plot_format": "png",
            "outdir": str(tmp_path / "plot"),
            "output_prefix": "csubst",
            "log_file": "",
            "output_manifest": True,
        }
    )

    main_benchmark_plot.main_benchmark_plot(g)

    runs_tsv = tmp_path / "plot" / "csubst_benchmark_plot_runs.tsv"
    summary_tsv = tmp_path / "plot" / "csubst_benchmark_plot_summary.tsv"
    summary_json = tmp_path / "plot" / "csubst_benchmark_plot_summary.json"
    overview_png = tmp_path / "plot" / "csubst_benchmark_plot_overview.png"
    manifest_tsv = tmp_path / "plot" / "csubst_outputs.tsv"
    assert runs_tsv.exists() is True
    assert summary_tsv.exists() is True
    assert summary_json.exists() is True
    assert overview_png.exists() is True
    assert overview_png.stat().st_size > 0
    assert manifest_tsv.exists() is True

    runs = pd.read_csv(runs_tsv, sep="\t")
    summary = pd.read_csv(summary_tsv, sep="\t")
    manifest = pd.read_csv(manifest_tsv, sep="\t")
    with open(summary_json, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert runs.shape[0] == 3
    assert set(runs.loc[:, "benchmark_source_name"]) == {"bench_a", "bench_b"}
    codon_model = summary.loc[
        (summary.loc[:, "parameter"] == "expectation_method")
        & (summary.loc[:, "value"] == "codon_model"),
        :
    ].iloc[0]
    urn = summary.loc[
        (summary.loc[:, "parameter"] == "expectation_method")
        & (summary.loc[:, "value"] == "urn"),
        :
    ].iloc[0]
    assert codon_model["total_runs"] == 2
    assert codon_model["pass_runs"] == 2
    assert codon_model["elapsed_sec_mean"] == 16.0
    assert urn["pass_runs"] == 0
    assert urn["fail_runs"] == 1
    assert urn["elapsed_sec_n"] == 0
    assert payload["counts"] == {
        "summary_files": 2,
        "summary_files_skipped": 0,
        "runs": 3,
        "pass": 2,
        "fail": 1,
    }
    assert payload["plot_format"] == "png"
    assert "elapsed_sec" in payload["metrics"]
    assert (manifest.loc[:, "output_kind"] == "benchmark_plot_runs_tsv").any()
    assert (manifest.loc[:, "output_kind"] == "benchmark_plot_summary_tsv").any()
    assert (manifest.loc[:, "output_kind"] == "benchmark_plot_summary_json").any()
    assert (manifest.loc[:, "output_kind"] == "benchmark_plot_overview").any()
    assert (manifest.loc[:, "output_kind"] == "output_manifest").any()


def test_main_benchmark_plot_skips_unreadable_summaries(tmp_path, capsys):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    _write_benchmark_summary(
        good_dir,
        [
            {
                "label": "exp-codon_model.asrv-each.recode-no.pc-none",
                "status": "pass",
                "expectation_method": "codon_model",
                "asrv": "each",
                "nonsyn_recode": "no",
                "sa_asr_mode": "direct",
                "pseudocount_mode": "none",
                "elapsed_sec": 12.0,
                "hit_rows": 3,
                "score_max": 7.0,
            }
        ],
    )
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "csubst_benchmark_summary.tsv").write_text("not a tsv\n", encoding="utf-8")
    g = runtime.ensure_run_context(
        {
            "benchmark_dir": str(tmp_path),
            "benchmark_plot_recursive": True,
            "benchmark_plot_metrics": "elapsed_sec,hit_rows,score_max",
            "benchmark_plot_format": "png",
            "outdir": str(tmp_path / "plot"),
            "output_prefix": "csubst",
            "log_file": "",
            "output_manifest": True,
        }
    )

    main_benchmark_plot.main_benchmark_plot(g)

    captured = capsys.readouterr()
    assert "Skipping unreadable benchmark summary" in captured.out
    runs = pd.read_csv(tmp_path / "plot" / "csubst_benchmark_plot_runs.tsv", sep="\t")
    payload = json.loads((tmp_path / "plot" / "csubst_benchmark_plot_summary.json").read_text(encoding="utf-8"))
    assert runs.shape[0] == 1
    assert payload["counts"] == {
        "summary_files": 1,
        "summary_files_skipped": 1,
        "runs": 1,
        "pass": 1,
        "fail": 0,
    }
    assert payload["skipped_summary_files"] == [str((bad_dir / "csubst_benchmark_summary.tsv").resolve())]


def test_main_benchmark_plot_raises_when_all_summaries_are_unreadable(tmp_path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "csubst_benchmark_summary.tsv").write_text("not a tsv\n", encoding="utf-8")
    g = runtime.ensure_run_context(
        {
            "benchmark_dir": str(tmp_path),
            "benchmark_plot_recursive": True,
            "benchmark_plot_metrics": "elapsed_sec,hit_rows,score_max",
            "benchmark_plot_format": "png",
            "outdir": str(tmp_path / "plot"),
            "output_prefix": "csubst",
            "log_file": "",
            "output_manifest": True,
        }
    )

    with pytest.raises(ValueError, match="No readable benchmark summary TSV files were found"):
        main_benchmark_plot.main_benchmark_plot(g)


def test_main_benchmark_plot_rejects_unknown_metrics(tmp_path):
    good_dir = tmp_path / "good"
    _write_benchmark_summary(
        good_dir,
        [
            {
                "label": "exp-codon_model.asrv-each.recode-no.pc-none",
                "status": "pass",
                "expectation_method": "codon_model",
                "asrv": "each",
                "nonsyn_recode": "no",
                "sa_asr_mode": "direct",
                "pseudocount_mode": "none",
                "elapsed_sec": 12.0,
                "hit_rows": 3,
                "score_max": 7.0,
            }
        ],
    )
    g = runtime.ensure_run_context(
        {
            "benchmark_dir": str(tmp_path),
            "benchmark_plot_recursive": True,
            "benchmark_plot_metrics": "not_a_metric,also_bad",
            "benchmark_plot_format": "png",
            "outdir": str(tmp_path / "plot"),
            "output_prefix": "csubst",
            "log_file": "",
            "output_manifest": False,
        }
    )

    with pytest.raises(ValueError, match="Invalid --benchmark_plot_metrics"):
        main_benchmark_plot.main_benchmark_plot(g)


def test_main_benchmark_plot_skips_structurally_invalid_summaries(tmp_path, capsys):
    good_dir = tmp_path / "good"
    invalid_dir = tmp_path / "invalid"
    _write_benchmark_summary(
        good_dir,
        [
            {
                "label": "exp-codon_model.asrv-each.recode-no.pc-none",
                "status": "pass",
                "expectation_method": "codon_model",
                "asrv": "each",
                "nonsyn_recode": "no",
                "sa_asr_mode": "direct",
                "pseudocount_mode": "none",
                "elapsed_sec": 12.0,
                "hit_rows": 3,
                "score_max": 7.0,
            }
        ],
    )
    invalid_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"foo": "bar", "baz": 1}]).to_csv(
        invalid_dir / "csubst_benchmark_summary.tsv",
        sep="\t",
        index=False,
    )
    g = runtime.ensure_run_context(
        {
            "benchmark_dir": str(tmp_path),
            "benchmark_plot_recursive": True,
            "benchmark_plot_metrics": "elapsed_sec,hit_rows,score_max",
            "benchmark_plot_format": "png",
            "outdir": str(tmp_path / "plot"),
            "output_prefix": "csubst",
            "log_file": "",
            "output_manifest": False,
        }
    )

    main_benchmark_plot.main_benchmark_plot(g)

    captured = capsys.readouterr()
    assert "Skipping unreadable benchmark summary" in captured.out
    runs = pd.read_csv(tmp_path / "plot" / "csubst_benchmark_plot_runs.tsv", sep="\t")
    payload = json.loads((tmp_path / "plot" / "csubst_benchmark_plot_summary.json").read_text(encoding="utf-8"))
    assert runs.shape[0] == 1
    assert set(runs.loc[:, "benchmark_source_name"]) == {"good"}
    assert payload["counts"] == {
        "summary_files": 1,
        "summary_files_skipped": 1,
        "runs": 1,
        "pass": 1,
        "fail": 0,
    }
    assert payload["skipped_summary_files"] == [str((invalid_dir / "csubst_benchmark_summary.tsv").resolve())]


def test_main_benchmark_plot_skips_summaries_with_blank_required_fields(tmp_path, capsys):
    good_dir = tmp_path / "good"
    blank_dir = tmp_path / "blank"
    _write_benchmark_summary(
        good_dir,
        [
            {
                "label": "exp-codon_model.asrv-each.recode-no.pc-none",
                "status": "pass",
                "expectation_method": "codon_model",
                "asrv": "each",
                "nonsyn_recode": "no",
                "sa_asr_mode": "direct",
                "pseudocount_mode": "none",
                "elapsed_sec": 12.0,
                "hit_rows": 3,
                "score_max": 7.0,
            }
        ],
    )
    _write_benchmark_summary(
        blank_dir,
        [
            {
                "label": "",
                "status": "pass",
                "expectation_method": "",
                "asrv": "",
                "nonsyn_recode": "",
                "sa_asr_mode": "",
                "pseudocount_mode": "",
            }
        ],
    )
    g = runtime.ensure_run_context(
        {
            "benchmark_dir": str(tmp_path),
            "benchmark_plot_recursive": True,
            "benchmark_plot_metrics": "elapsed_sec,hit_rows,score_max",
            "benchmark_plot_format": "png",
            "outdir": str(tmp_path / "plot"),
            "output_prefix": "csubst",
            "log_file": "",
            "output_manifest": False,
        }
    )

    main_benchmark_plot.main_benchmark_plot(g)

    captured = capsys.readouterr()
    assert "Skipping unreadable benchmark summary" in captured.out
    runs = pd.read_csv(tmp_path / "plot" / "csubst_benchmark_plot_runs.tsv", sep="\t")
    payload = json.loads((tmp_path / "plot" / "csubst_benchmark_plot_summary.json").read_text(encoding="utf-8"))
    assert runs.shape[0] == 1
    assert set(runs.loc[:, "benchmark_source_name"]) == {"good"}
    assert payload["counts"] == {
        "summary_files": 1,
        "summary_files_skipped": 1,
        "runs": 1,
        "pass": 1,
        "fail": 0,
    }
    assert payload["skipped_summary_files"] == [str((blank_dir / "csubst_benchmark_summary.tsv").resolve())]

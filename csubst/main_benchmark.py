import json
import contextlib
import os
import re
import shutil
import time
import traceback
from itertools import product

import numpy as np
import pandas as pd

from csubst import main_analyze
from csubst import output_manifest
from csubst import pseudocount
from csubst import recoding
from csubst import runtime


_IQTREE_EXTENSIONS = ["iqtree", "log", "rate", "state", "treefile"]


def _split_csv_tokens(value):
    if value is None:
        return list()
    if isinstance(value, (list, tuple)):
        return [str(token).strip() for token in value if str(token).strip() != ""]
    tokens = [token.strip() for token in str(value).split(",") if token.strip() != ""]
    return tokens


def _resolve_grid(value, default_tokens):
    tokens = _split_csv_tokens(value)
    if len(tokens) == 0:
        return list(default_tokens)
    dedup = list()
    for token in tokens:
        if token not in dedup:
            dedup.append(token)
    return dedup


def _sanitize_label_token(value):
    token = str(value).strip().lower()
    token = re.sub(r"[^a-z0-9._-]+", "-", token)
    token = re.sub(r"-+", "-", token).strip("-")
    return token if token != "" else "na"


def _build_config_label(config):
    parts = [
        "exp-{}".format(_sanitize_label_token(config["expectation_method"])),
        "asrv-{}".format(_sanitize_label_token(config["asrv"])),
        "recode-{}".format(_sanitize_label_token(config["nonsyn_recode"])),
        "pc-{}".format(_sanitize_label_token(config["pseudocount_mode"])),
    ]
    if str(config["nonsyn_recode"]).strip().lower() == "3di20":
        parts.append("sa-{}".format(_sanitize_label_token(config["sa_asr_mode"])))
    return ".".join(parts)


def _normalize_expectation_method_token(token):
    normalized = str(token).strip().lower()
    if normalized in ["codon_model", "submodel"]:
        return "codon_model"
    if normalized in ["urn", "modelfree"]:
        return "urn"
    raise ValueError(
        'Unsupported benchmark expectation method "{}". Use codon_model or urn.'.format(token)
    )


def _normalize_asrv_token(token):
    normalized = str(token).strip().lower()
    if normalized not in ["no", "pool", "sn", "each", "file", "file_each"]:
        raise ValueError(
            'Unsupported benchmark ASRV mode "{}".'.format(token)
        )
    return normalized


def _normalize_sa_asr_token(token):
    normalized = str(token).strip().lower()
    if normalized not in ["direct", "translate"]:
        raise ValueError(
            'Unsupported benchmark sa_asr_mode "{}".'.format(token)
        )
    return normalized


def _normalize_pseudocount_mode_token(token):
    normalized = str(token).strip().lower()
    if normalized not in ["none", "symmetric", "empirical"]:
        raise ValueError(
            'Unsupported benchmark pseudocount mode "{}".'.format(token)
        )
    return normalized


def _validate_benchmark_threshold(value, name):
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("{} should be finite.".format(name))
    return float(numeric)


def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    value_txt = str(value).strip().lower()
    if value_txt in ["y", "yes", "t", "true", "on", "1"]:
        return True
    if value_txt in ["n", "no", "f", "false", "off", "0", ""]:
        return False
    return bool(value)


def _iter_benchmark_configs(g):
    default_asrv = _normalize_asrv_token(str(g.get("asrv", "each")).strip().lower())
    expectation_methods = _resolve_grid(
        g.get("benchmark_expectation_methods", ""),
        [str(g.get("expectation_method", "codon_model")).strip().lower()],
    )
    expectation_methods = [_normalize_expectation_method_token(token) for token in expectation_methods]
    asrv_modes = _resolve_grid(
        g.get("benchmark_asrv_modes", ""),
        [str(g.get("asrv", "each")).strip().lower()],
    )
    asrv_modes = [_normalize_asrv_token(token) for token in asrv_modes]
    nonsyn_modes = _resolve_grid(
        g.get("benchmark_nonsyn_recode_modes", ""),
        [str(g.get("nonsyn_recode", "no")).strip().lower()],
    )
    nonsyn_modes = [recoding.normalize_nonsyn_recode(token) for token in nonsyn_modes]
    raw_sa_grid = _split_csv_tokens(g.get("benchmark_sa_asr_modes", ""))
    if len(raw_sa_grid) == 0:
        if any([str(mode).strip().lower() == "3di20" for mode in nonsyn_modes]):
            sa_asr_modes = ["direct", "translate"]
        else:
            sa_asr_modes = [str(g.get("sa_asr_mode", "direct")).strip().lower()]
    else:
        sa_asr_modes = _resolve_grid(raw_sa_grid, [str(g.get("sa_asr_mode", "direct")).strip().lower()])
    sa_asr_modes = [_normalize_sa_asr_token(token) for token in sa_asr_modes]
    pseudocount_modes = _resolve_grid(
        g.get("benchmark_pseudocount_modes", ""),
        [str(g.get("pseudocount_mode", "none")).strip().lower()],
    )
    pseudocount_modes = [_normalize_pseudocount_mode_token(token) for token in pseudocount_modes]
    configs = list()
    for expectation_method, nonsyn_mode, pseudocount_mode in product(
        expectation_methods,
        nonsyn_modes,
        pseudocount_modes,
    ):
        if expectation_method == "urn":
            asrv_mode_iter = asrv_modes
        else:
            asrv_mode_iter = [default_asrv]
        if str(nonsyn_mode).strip().lower() == "3di20":
            sa_mode_iter = sa_asr_modes
        else:
            sa_mode_iter = [str(g.get("sa_asr_mode", "direct")).strip().lower()]
        for asrv_mode, sa_asr_mode in product(asrv_mode_iter, sa_mode_iter):
            config = {
                "expectation_method": str(expectation_method).strip().lower(),
                "asrv": str(asrv_mode).strip().lower(),
                "nonsyn_recode": recoding.normalize_nonsyn_recode(nonsyn_mode),
                "sa_asr_mode": str(sa_asr_mode).strip().lower(),
                "pseudocount_mode": str(pseudocount_mode).strip().lower(),
            }
            config["label"] = _build_config_label(config)
            configs.append(config)
    dedup = list()
    seen = set()
    for config in configs:
        key = tuple([config["expectation_method"], config["asrv"], config["nonsyn_recode"], config["sa_asr_mode"], config["pseudocount_mode"]])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(config)
    return dedup


def _resolve_run_relative_path(run_dir, value):
    value_txt = "" if value is None else str(value).strip()
    if value_txt == "":
        return value_txt
    if os.path.isabs(value_txt):
        return value_txt
    return os.path.join(run_dir, value_txt)


def _resolve_benchmark_shared_path(base_dir, value, default_name):
    value_txt = "" if value is None else str(value).strip()
    if value_txt == "":
        value_txt = str(default_name).strip()
    if os.path.isabs(value_txt):
        return value_txt
    return os.path.join(base_dir, value_txt)


def _get_inferred_iqtree_paths(alignment_file, iqtree_outdir):
    prefix = runtime.infer_iqtree_output_prefix(
        alignment_file=alignment_file,
        iqtree_outdir=iqtree_outdir,
    )
    return {ext: os.path.abspath(prefix + "." + ext) for ext in _IQTREE_EXTENSIONS}


def _retarget_iqtree_paths_for_alignment_change(local_g, original_alignment_file):
    original_alignment = str(original_alignment_file).strip()
    updated_alignment = str(local_g.get("alignment_file", "")).strip()
    if (original_alignment == "") or (updated_alignment == "") or (original_alignment == updated_alignment):
        return local_g
    original_paths = _get_inferred_iqtree_paths(
        alignment_file=original_alignment,
        iqtree_outdir=local_g.get("iqtree_outdir", None),
    )
    updated_paths = _get_inferred_iqtree_paths(
        alignment_file=updated_alignment,
        iqtree_outdir=local_g.get("iqtree_outdir", None),
    )
    for ext in _IQTREE_EXTENSIONS:
        iqtree_key = "iqtree_" + ext
        iqtree_value = str(local_g.get(iqtree_key, "")).strip()
        if (iqtree_value != "") and (iqtree_value.lower() != "infer"):
            if os.path.abspath(iqtree_value) == original_paths[ext]:
                local_g[iqtree_key] = updated_paths[ext]
        path_key = "path_iqtree_" + ext
        path_value = str(local_g.get(path_key, "")).strip()
        if path_value != "":
            if os.path.abspath(path_value) == original_paths[ext]:
                local_g[path_key] = updated_paths[ext]
    return local_g


def _prepare_run_context(base_g, config, run_dir):
    local_g = runtime.ensure_run_context(base_g.copy())
    original_alignment_file = str(local_g.get("alignment_file", "")).strip()
    local_g["subcommand"] = "search"
    local_g["outdir"] = os.path.abspath(run_dir)
    local_g["output_prefix"] = str(base_g.get("output_prefix", "csubst"))
    local_g["log_file"] = os.path.join(local_g["outdir"], local_g["output_prefix"] + ".log")
    local_g["expectation_method"] = config["expectation_method"]
    local_g["omegaC_method"] = "modelfree" if (config["expectation_method"] == "urn") else "submodel"
    local_g["asrv"] = config["asrv"]
    local_g["nonsyn_recode"] = recoding.normalize_nonsyn_recode(config["nonsyn_recode"])
    local_g["sa_asr_mode"] = config["sa_asr_mode"]
    local_g["pseudocount_mode"] = config["pseudocount_mode"]
    local_g["pseudocount_alpha"] = str(local_g.get("pseudocount_alpha", "0.0"))
    if local_g["nonsyn_recode"] == "3di20":
        full_cds = str(local_g.get("full_cds_alignment_file", "")).strip()
        if full_cds == "":
            raise ValueError('benchmark configuration "{}" requires --full_cds_alignment_file.'.format(config["label"]))
        local_g["alignment_file"] = full_cds
        local_g = _retarget_iqtree_paths_for_alignment_change(
            local_g=local_g,
            original_alignment_file=original_alignment_file,
        )
    benchmark_root = os.path.abspath(str(base_g.get("outdir", run_dir)))
    local_g["prostt5_cache_file"] = _resolve_benchmark_shared_path(
        benchmark_root,
        local_g.get("prostt5_cache_file", "csubst_prostt5_cache.tsv"),
        "csubst_prostt5_cache.tsv",
    )
    local_g["sa_state_cache_file"] = _resolve_benchmark_shared_path(
        benchmark_root,
        local_g.get("sa_state_cache_file", "csubst_3di_state_cache.npz"),
        "csubst_3di_state_cache.npz",
    )
    local_g["epistasis_degree_outfile"] = _resolve_run_relative_path(run_dir, local_g.get("epistasis_degree_outfile", "csubst_epistasis_structure_degree.tsv"))
    for path_key in ["prostt5_cache_file", "sa_state_cache_file"]:
        parent_dir = os.path.dirname(str(local_g[path_key]))
        if parent_dir != "":
            os.makedirs(parent_dir, exist_ok=True)
    local_g.update(pseudocount.validate_args(local_g))
    if local_g.get("calc_omega_pvalue", False) and (local_g["expectation_method"] != "urn"):
        raise ValueError("--calc_omega_pvalue yes requires expectation_method urn.")
    return runtime.ensure_output_layout(local_g, create_dir=True)


def _materialize_search_log(local_g, benchmark_log):
    search_log = str(local_g.get("log_file", "")).strip()
    if search_log == "":
        return str(os.path.abspath(benchmark_log))
    search_log_abs = os.path.abspath(search_log)
    benchmark_log_abs = os.path.abspath(benchmark_log)
    if search_log_abs == benchmark_log_abs:
        return search_log_abs
    parent_dir = os.path.dirname(search_log_abs)
    if parent_dir != "":
        os.makedirs(parent_dir, exist_ok=True)
    shutil.copyfile(benchmark_log_abs, search_log_abs)
    return search_log_abs


def _write_preparation_failure_logs(run_dir, output_prefix, config_label, exc):
    os.makedirs(run_dir, exist_ok=True)
    benchmark_log = os.path.abspath(os.path.join(run_dir, "benchmark_run.log"))
    search_log = os.path.abspath(os.path.join(run_dir, str(output_prefix) + ".log"))
    log_text = "Benchmark configuration: {}\n".format(config_label)
    log_text += "Benchmark run directory: {}\n".format(os.path.abspath(run_dir))
    log_text += "Benchmark setup failed before search execution.\n"
    log_text += "Error: {}: {}\n".format(type(exc).__name__, exc)
    log_text += "\n"
    log_text += traceback.format_exc()
    for output_path in [benchmark_log, search_log]:
        parent_dir = os.path.dirname(output_path)
        if parent_dir != "":
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(log_text)
    return benchmark_log, search_log


def _cleanup_run_outputs(run_dir, output_prefix):
    cleanup_paths = [
        os.path.abspath(os.path.join(run_dir, "benchmark_run.log")),
        os.path.abspath(os.path.join(run_dir, str(output_prefix) + ".log")),
        runtime.output_path({"outdir": run_dir, "output_prefix": output_prefix}, "cb_2.tsv"),
    ]
    for output_path in cleanup_paths:
        if os.path.exists(output_path):
            os.remove(output_path)


def _append_benchmark_validation_message(run_log, search_log, message):
    log_paths = list()
    for path in [run_log, search_log]:
        path_txt = "" if path is None else str(path).strip()
        if path_txt == "":
            continue
        path_abs = os.path.abspath(path_txt)
        if path_abs not in log_paths:
            log_paths.append(path_abs)
    for path_abs in log_paths:
        with open(path_abs, "a", encoding="utf-8") as handle:
            handle.write("\nBenchmark validation failure: {}\n".format(message))


def _read_cb_summary(cb_path, score_col, ocn_col, min_score, min_ocn):
    row = {
        "cb_tsv": str(cb_path),
        "cb_rows": 0,
        "score_column_found": "N",
        "ocn_column_found": "N",
        "score_finite": 0,
        "score_max": np.nan,
        "score_median": np.nan,
        "hit_rows": 0,
    }
    if not os.path.exists(cb_path):
        return row
    df = pd.read_csv(cb_path, sep="\t")
    row["cb_rows"] = int(df.shape[0])
    if score_col in df.columns:
        row["score_column_found"] = "Y"
        score = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(score)
        row["score_finite"] = int(finite.sum())
        if row["score_finite"] > 0:
            row["score_max"] = float(np.nanmax(score[finite]))
            row["score_median"] = float(np.nanmedian(score[finite]))
    if ocn_col in df.columns:
        row["ocn_column_found"] = "Y"
    if (score_col in df.columns) and (ocn_col in df.columns):
        score = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
        ocn = pd.to_numeric(df[ocn_col], errors="coerce").to_numpy(dtype=float)
        is_hit = np.isfinite(score) & np.isfinite(ocn)
        is_hit &= (score >= float(min_score)) & (ocn >= float(min_ocn))
        row["hit_rows"] = int(is_hit.sum())
    return row


def _validate_benchmark_result(result, score_col, ocn_col):
    if str(result.get("status", "pass")).strip().lower() != "pass":
        return result
    cb_path = str(result.get("cb_tsv", "")).strip()
    issues = list()
    if (cb_path == "") or (not os.path.exists(cb_path)):
        issues.append("Benchmark requires csubst_cb_2.tsv output. Ensure --cb yes and that search completed successfully.")
    else:
        if str(result.get("score_column_found", "N")).strip().upper() != "Y":
            issues.append(
                'benchmark_score_column "{}" was not found in {}.'.format(score_col, cb_path)
            )
        if str(result.get("ocn_column_found", "N")).strip().upper() != "Y":
            issues.append(
                'benchmark_ocn_column "{}" was not found in {}.'.format(ocn_col, cb_path)
            )
    if len(issues) == 0:
        return result
    message = " ".join(issues)
    result["status"] = "fail"
    existing = str(result.get("error_message", "")).strip()
    if existing == "":
        result["error_message"] = message
    else:
        result["error_message"] = existing + " | " + message
    _append_benchmark_validation_message(
        run_log=result.get("run_log", ""),
        search_log=result.get("search_log", ""),
        message=message,
    )
    return result


def _run_single_config(base_g, config, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    _cleanup_run_outputs(run_dir=run_dir, output_prefix=base_g.get("output_prefix", "csubst"))
    benchmark_log = os.path.join(run_dir, "benchmark_run.log")
    local_g = _prepare_run_context(base_g=base_g, config=config, run_dir=run_dir)
    result = {
        "label": config["label"],
        "status": "pass",
        "error_message": "",
        "run_dir": str(os.path.abspath(run_dir)),
        "run_log": str(benchmark_log),
        "expectation_method": config["expectation_method"],
        "asrv": config["asrv"],
        "nonsyn_recode": config["nonsyn_recode"],
        "sa_asr_mode": config["sa_asr_mode"],
        "pseudocount_mode": config["pseudocount_mode"],
    }
    start = time.time()
    with open(benchmark_log, "w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            print("Benchmark configuration: {}".format(config["label"]), flush=True)
            print("Benchmark run directory: {}".format(os.path.abspath(run_dir)), flush=True)
            print("Benchmark search log: {}".format(local_g["log_file"]), flush=True)
            try:
                main_analyze.main_analyze(local_g)
            except Exception as exc:
                traceback.print_exc()
                result["status"] = "fail"
                result["error_message"] = "{}: {}".format(type(exc).__name__, exc)
    result["elapsed_sec"] = float(time.time() - start)
    result["search_log"] = _materialize_search_log(local_g=local_g, benchmark_log=benchmark_log)
    cb_path = runtime.output_path(local_g, "cb_2.tsv")
    result.update(
        _read_cb_summary(
            cb_path=cb_path,
            score_col=base_g.get("benchmark_score_column", "omegaCany2spe"),
            ocn_col=base_g.get("benchmark_ocn_column", "OCNany2spe"),
            min_score=base_g.get("benchmark_min_score", 5.0),
            min_ocn=base_g.get("benchmark_min_ocn", 2.0),
        )
    )
    result = _validate_benchmark_result(
        result=result,
        score_col=base_g.get("benchmark_score_column", "omegaCany2spe"),
        ocn_col=base_g.get("benchmark_ocn_column", "OCNany2spe"),
    )
    result["search_outdir"] = str(local_g["outdir"])
    return result


def _write_benchmark_output_manifest(g, summary_tsv, summary):
    manifest_rows = list()
    output_manifest.add_output_manifest_row(
        manifest_rows=manifest_rows,
        output_path=summary_tsv,
        output_kind="benchmark_summary_tsv",
        base_dir=g["outdir"],
    )
    summary_json = runtime.output_path(g, "benchmark_summary.json")
    if os.path.exists(summary_json):
        output_manifest.add_output_manifest_row(
            manifest_rows=manifest_rows,
            output_path=summary_json,
            output_kind="benchmark_summary_json",
            base_dir=g["outdir"],
        )
    output_manifest.add_output_manifest_row(
        manifest_rows=manifest_rows,
        output_path=g["log_file"],
        output_kind="benchmark_log",
        base_dir=g["outdir"],
    )
    if summary.shape[0] > 0:
        for _, row in summary.iterrows():
            output_manifest.add_output_manifest_row(
                manifest_rows=manifest_rows,
                output_path=row["run_log"],
                output_kind="benchmark_run_log",
                note=str(row["label"]),
                base_dir=g["outdir"],
                extra_fields={
                    "status": str(row["status"]),
                    "label": str(row["label"]),
                },
            )
            search_log = str(row.get("search_log", "")).strip()
            if search_log != "":
                output_manifest.add_output_manifest_row(
                    manifest_rows=manifest_rows,
                    output_path=search_log,
                    output_kind="benchmark_search_log",
                    note=str(row["label"]),
                    base_dir=g["outdir"],
                    extra_fields={
                        "status": str(row["status"]),
                        "label": str(row["label"]),
                    },
                )
            cb_path = str(row.get("cb_tsv", "")).strip()
            if (cb_path != "") and os.path.exists(cb_path):
                output_manifest.add_output_manifest_row(
                    manifest_rows=manifest_rows,
                    output_path=cb_path,
                    output_kind="benchmark_cb_tsv",
                    note=str(row["label"]),
                    base_dir=g["outdir"],
                    extra_fields={
                        "status": str(row["status"]),
                        "label": str(row["label"]),
                    },
                )
    manifest_path = runtime.output_path(g, "outputs.tsv")
    manifest_path = output_manifest.write_output_manifest(
        manifest_rows=manifest_rows,
        manifest_path=manifest_path,
        base_dir=g["outdir"],
    )
    print("Writing benchmark output manifest: {}".format(manifest_path), flush=True)
    return manifest_path


def _dataframe_records_for_json(df):
    if df.shape[0] == 0:
        return list()
    json_df = df.astype(object).where(pd.notnull(df), None)
    return json_df.to_dict(orient="records")


def _write_benchmark_outputs(g, summary):
    summary_tsv = runtime.output_path(g, "benchmark_summary.tsv")
    summary_json = runtime.output_path(g, "benchmark_summary.json")
    summary.to_csv(summary_tsv, sep="\t", index=False)
    payload = {
        "counts": {
            "pass": int((summary["status"] == "pass").sum()) if summary.shape[0] > 0 else 0,
            "fail": int((summary["status"] != "pass").sum()) if summary.shape[0] > 0 else 0,
        },
        "score_column": str(g.get("benchmark_score_column", "omegaCany2spe")),
        "ocn_column": str(g.get("benchmark_ocn_column", "OCNany2spe")),
        "min_score": float(g.get("benchmark_min_score", 5.0)),
        "min_ocn": float(g.get("benchmark_min_ocn", 2.0)),
        "rows": _dataframe_records_for_json(summary),
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    print("Writing benchmark summary TSV: {}".format(summary_tsv), flush=True)
    print("Writing benchmark summary JSON: {}".format(summary_json), flush=True)
    return summary_tsv, summary_json


def main_benchmark(g):
    g = runtime.ensure_output_layout(g, create_dir=True)
    g["benchmark_min_score"] = _validate_benchmark_threshold(
        g.get("benchmark_min_score", 5.0),
        "--benchmark_min_score",
    )
    g["benchmark_min_ocn"] = _validate_benchmark_threshold(
        g.get("benchmark_min_ocn", 2.0),
        "--benchmark_min_ocn",
    )
    if not _as_bool(g.get("cb", True)):
        raise ValueError("benchmark requires --cb yes because it summarizes csubst_cb_2.tsv.")
    g["benchmark_score_column"] = str(g.get("benchmark_score_column", "omegaCany2spe")).strip()
    if g["benchmark_score_column"] == "":
        raise ValueError("--benchmark_score_column should be non-empty.")
    g["benchmark_ocn_column"] = str(g.get("benchmark_ocn_column", "OCNany2spe")).strip()
    if g["benchmark_ocn_column"] == "":
        raise ValueError("--benchmark_ocn_column should be non-empty.")
    configs = _iter_benchmark_configs(g)
    if len(configs) == 0:
        raise ValueError("No benchmark configurations were generated.")
    runs_root = os.path.join(g["outdir"], "runs")
    os.makedirs(runs_root, exist_ok=True)
    print("Benchmark configurations: {:,}".format(len(configs)), flush=True)
    results = list()
    for i, config in enumerate(configs, start=1):
        run_dir = os.path.join(runs_root, "{:03d}.{}".format(i, config["label"]))
        print("[{}/{}] Running {}".format(i, len(configs), config["label"]), flush=True)
        try:
            result = _run_single_config(base_g=g, config=config, run_dir=run_dir)
        except Exception as exc:
            run_log, search_log = _write_preparation_failure_logs(
                run_dir=run_dir,
                output_prefix=g["output_prefix"],
                config_label=config["label"],
                exc=exc,
            )
            result = {
                "label": config["label"],
                "status": "fail",
                "error_message": "{}: {}".format(type(exc).__name__, exc),
                "run_dir": str(os.path.abspath(run_dir)),
                "run_log": str(run_log),
                "expectation_method": config["expectation_method"],
                "asrv": config["asrv"],
                "nonsyn_recode": config["nonsyn_recode"],
                "sa_asr_mode": config["sa_asr_mode"],
                "pseudocount_mode": config["pseudocount_mode"],
                "elapsed_sec": np.nan,
                "cb_tsv": runtime.output_path({"outdir": run_dir, "output_prefix": g["output_prefix"]}, "cb_2.tsv"),
                "cb_rows": 0,
                "score_column_found": "N",
                "ocn_column_found": "N",
                "score_finite": 0,
                "score_max": np.nan,
                "score_median": np.nan,
                "hit_rows": 0,
                "search_outdir": str(os.path.abspath(run_dir)),
                "search_log": str(search_log),
            }
        results.append(result)
        if (result["status"] != "pass") and (not _as_bool(g.get("benchmark_keep_going", True))):
            break
    summary = pd.DataFrame(results)
    if summary.shape[0] > 0:
        summary = summary.sort_values(by=["status", "elapsed_sec", "label"], ascending=[True, True, True]).reset_index(drop=True)
    summary_tsv, _ = _write_benchmark_outputs(g=g, summary=summary)
    if _as_bool(g.get("output_manifest", True)):
        _write_benchmark_output_manifest(g=g, summary_tsv=summary_tsv, summary=summary)
    num_pass = int((summary["status"] == "pass").sum()) if summary.shape[0] > 0 else 0
    num_fail = int((summary["status"] != "pass").sum()) if summary.shape[0] > 0 else 0
    print("Benchmark summary: pass={}, fail={}".format(num_pass, num_fail), flush=True)
    if (num_fail > 0) and (not _as_bool(g.get("benchmark_keep_going", True))):
        failing = summary.loc[summary["status"] != "pass", :]
        if failing.shape[0] > 0:
            first = failing.iloc[0]
            raise ValueError("Benchmark failed for {}: {}".format(first["label"], first["error_message"]))

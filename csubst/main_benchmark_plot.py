import json
import os

import numpy as np
import pandas as pd

from csubst import output_manifest
from csubst import runtime
from csubst import tsv


_DEFAULT_METRICS = ["elapsed_sec", "hit_rows", "score_max"]
_REQUIRED_SUMMARY_COLUMNS = [
    "label",
    "expectation_method",
    "asrv",
    "nonsyn_recode",
    "sa_asr_mode",
    "pseudocount_mode",
]
_PARAMETER_COLUMNS = [
    "expectation_method",
    "asrv",
    "nonsyn_recode",
    "sa_asr_mode",
    "pseudocount_mode",
]
_PARAMETER_ORDER = {
    "expectation_method": ["codon_model", "urn"],
    "asrv": ["no", "pool", "sn", "each", "file", "file_each"],
    "nonsyn_recode": [
        "no",
        "3di20",
        "dayhoff6",
        "sr6",
        "kgb6",
        "sr4",
        "dayhoff9",
        "dayhoff12",
        "dayhoff15",
        "dayhoff18",
        "srchisq6",
        "kgbauto6",
    ],
    "sa_asr_mode": ["direct", "translate"],
    "pseudocount_mode": ["none", "symmetric", "empirical"],
}


def _split_csv_tokens(value):
    if value is None:
        return list()
    if isinstance(value, (list, tuple)):
        return [str(token).strip() for token in value if str(token).strip() != ""]
    return [token.strip() for token in str(value).split(",") if token.strip() != ""]


def _normalize_metric_list(value):
    metrics = _split_csv_tokens(value)
    if len(metrics) == 0:
        metrics = list(_DEFAULT_METRICS)
    dedup = list()
    for metric in metrics:
        if metric not in dedup:
            dedup.append(metric)
    return dedup


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


def _iter_candidate_paths(root_dir, file_name, recursive):
    root_dir_abs = os.path.abspath(root_dir)
    if recursive:
        for dirpath, _, filenames in os.walk(root_dir_abs):
            if file_name in filenames:
                yield os.path.join(dirpath, file_name)
        return
    candidate = os.path.join(root_dir_abs, file_name)
    if os.path.exists(candidate):
        yield candidate


def _resolve_manifest_output_path(manifest_path, row):
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    output_path = str(row.get("output_path", "")).strip()
    if (output_path != "") and os.path.exists(output_path):
        return os.path.abspath(output_path)
    output_file = str(row.get("output_file", "")).strip()
    if output_file == "":
        return ""
    if os.path.isabs(output_file):
        candidate = output_file
    else:
        candidate = os.path.join(manifest_dir, output_file)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    if output_path != "":
        return os.path.abspath(output_path)
    return os.path.abspath(candidate)


def _collect_manifest_summary_paths(root_dir, recursive):
    summary_paths = list()
    for manifest_path in _iter_candidate_paths(root_dir, "csubst_outputs.tsv", recursive=recursive):
        try:
            manifest_df = pd.read_csv(manifest_path, sep="\t")
        except Exception as exc:
            print("Skipping unreadable manifest {}: {}".format(manifest_path, exc), flush=True)
            continue
        if "output_kind" not in manifest_df.columns:
            continue
        matched = manifest_df.loc[manifest_df["output_kind"] == "benchmark_summary_tsv", :]
        for _, row in matched.iterrows():
            summary_path = _resolve_manifest_output_path(manifest_path=manifest_path, row=row)
            if summary_path != "":
                summary_paths.append(summary_path)
    return summary_paths


def _collect_fallback_summary_paths(root_dir, recursive):
    root_dir_abs = os.path.abspath(root_dir)
    summary_paths = list()
    if recursive:
        for dirpath, _, filenames in os.walk(root_dir_abs):
            for file_name in filenames:
                if not str(file_name).endswith("_benchmark_summary.tsv"):
                    continue
                summary_paths.append(os.path.join(dirpath, file_name))
    else:
        for file_name in os.listdir(root_dir_abs):
            if str(file_name).endswith("_benchmark_summary.tsv"):
                summary_paths.append(os.path.join(root_dir_abs, file_name))
    return summary_paths


def _collect_benchmark_summary_paths(root_dir, recursive):
    summary_paths = list()
    summary_paths.extend(_collect_manifest_summary_paths(root_dir=root_dir, recursive=recursive))
    summary_paths.extend(_collect_fallback_summary_paths(root_dir=root_dir, recursive=recursive))
    dedup = list()
    seen = set()
    for path in summary_paths:
        path_abs = os.path.abspath(path)
        if path_abs in seen:
            continue
        if not os.path.exists(path_abs):
            print("Skipping missing benchmark summary: {}".format(path_abs), flush=True)
            continue
        seen.add(path_abs)
        dedup.append(path_abs)
    return dedup


def _read_benchmark_runs(summary_path):
    try:
        df = pd.read_csv(summary_path, sep="\t")
    except Exception as exc:
        raise ValueError("Failed to read benchmark summary {}: {}".format(summary_path, exc)) from exc
    if df.shape[0] == 0:
        raise ValueError("Benchmark summary had no data rows: {}".format(summary_path))
    missing_columns = [column for column in _REQUIRED_SUMMARY_COLUMNS if column not in df.columns]
    if len(missing_columns) > 0:
        raise ValueError(
            "Benchmark summary is missing required columns {}: {}".format(
                ",".join(missing_columns),
                summary_path,
            )
        )
    required_values = df.loc[:, _REQUIRED_SUMMARY_COLUMNS].fillna("").astype(str)
    required_values = required_values.apply(lambda col: col.str.strip())
    blank_rows = required_values.eq("").any(axis=1)
    if bool(blank_rows.any()):
        raise ValueError(
            "Benchmark summary has blank required fields at row(s) {}: {}".format(
                ",".join([str(int(i) + 1) for i in np.where(blank_rows.to_numpy(dtype=bool))[0].tolist()[:10]]),
                summary_path,
            )
        )
    df.loc[:, "benchmark_summary_tsv"] = os.path.abspath(summary_path)
    df.loc[:, "benchmark_source_dir"] = os.path.dirname(os.path.abspath(summary_path))
    df.loc[:, "benchmark_source_name"] = os.path.basename(os.path.dirname(os.path.abspath(summary_path)))
    return df


def _get_parameter_columns(runs):
    available = list()
    for column in _PARAMETER_COLUMNS:
        if column not in runs.columns:
            continue
        values = runs.loc[:, column].fillna("").astype(str).str.strip()
        if values.eq("").all():
            continue
        available.append(column)
    return available


def _get_metric_columns(runs, requested_metrics):
    available = list()
    missing = list()
    non_numeric = list()
    for metric in requested_metrics:
        if metric not in runs.columns:
            missing.append(metric)
            continue
        values = pd.to_numeric(runs.loc[:, metric], errors="coerce")
        if values.notnull().any():
            available.append(metric)
        else:
            non_numeric.append(metric)
    if (len(missing) > 0) or (len(non_numeric) > 0):
        message_bits = list()
        if len(missing) > 0:
            message_bits.append("missing columns: {}".format(",".join(missing)))
        if len(non_numeric) > 0:
            message_bits.append("non-numeric or empty columns: {}".format(",".join(non_numeric)))
        raise ValueError("Invalid --benchmark_plot_metrics: {}.".format("; ".join(message_bits)))
    return available


def _ordered_values(parameter, series):
    values = [str(value).strip() for value in series.fillna("").tolist() if str(value).strip() != ""]
    unique_values = sorted(set(values))
    preferred = _PARAMETER_ORDER.get(parameter, list())
    ordered = [value for value in preferred if value in unique_values]
    ordered.extend([value for value in unique_values if value not in ordered])
    return ordered


def _build_parameter_summary(runs, parameter_columns, metric_columns):
    rows = list()
    for parameter in parameter_columns:
        values = _ordered_values(parameter, runs.loc[:, parameter])
        for value_order, value in enumerate(values, start=1):
            is_value = runs.loc[:, parameter].fillna("").astype(str).str.strip() == value
            subset = runs.loc[is_value, :].copy()
            total_runs = int(subset.shape[0])
            pass_runs = int((subset.loc[:, "status"] == "pass").sum()) if "status" in subset.columns else total_runs
            fail_runs = int(total_runs - pass_runs)
            pass_subset = subset.loc[subset.loc[:, "status"] == "pass", :] if "status" in subset.columns else subset
            row = {
                "parameter": parameter,
                "value": value,
                "value_order": int(value_order),
                "total_runs": total_runs,
                "pass_runs": pass_runs,
                "fail_runs": fail_runs,
                "pass_rate": (float(pass_runs) / float(total_runs)) if total_runs > 0 else np.nan,
                "source_count": int(subset.loc[:, "benchmark_summary_tsv"].nunique()) if "benchmark_summary_tsv" in subset.columns else 0,
            }
            for metric in metric_columns:
                numeric = pd.to_numeric(pass_subset.loc[:, metric], errors="coerce")
                finite = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
                row["{}_n".format(metric)] = int(finite.shape[0])
                if finite.shape[0] == 0:
                    row["{}_mean".format(metric)] = np.nan
                    row["{}_median".format(metric)] = np.nan
                    row["{}_min".format(metric)] = np.nan
                    row["{}_max".format(metric)] = np.nan
                else:
                    row["{}_mean".format(metric)] = float(finite.mean())
                    row["{}_median".format(metric)] = float(finite.median())
                    row["{}_min".format(metric)] = float(finite.min())
                    row["{}_max".format(metric)] = float(finite.max())
            rows.append(row)
    return pd.DataFrame(rows)


def _dataframe_records_for_json(df):
    if df.shape[0] == 0:
        return list()
    json_df = df.astype(object).where(pd.notnull(df), None)
    return json_df.to_dict(orient="records")


def _humanize_name(value):
    return str(value).replace("_", " ")


def _plot_parameter_overview(runs, parameter_summary, parameter_columns, metric_columns, out_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pass_runs = runs.loc[runs.loc[:, "status"] == "pass", :].copy() if "status" in runs.columns else runs.copy()
    n_rows = max(1, len(metric_columns) + 1)
    n_cols = max(1, len(parameter_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(4.2 * n_cols, 8.0), max(2.8 * n_rows, 4.0)),
        squeeze=False,
        constrained_layout=True,
    )
    point_color = "#2a6f97"
    mean_color = "#111111"
    bar_color = "#7fb069"
    fail_color = "#d95d39"
    for col_index, parameter in enumerate(parameter_columns):
        param_summary = parameter_summary.loc[parameter_summary.loc[:, "parameter"] == parameter, :].copy()
        values = param_summary.loc[:, "value"].astype(str).tolist()
        x_positions = np.arange(len(values), dtype=float)
        ax = axes[0, col_index]
        ax.bar(
            x_positions,
            param_summary.loc[:, "pass_rate"].to_numpy(dtype=float),
            color=bar_color,
            edgecolor="black",
            linewidth=0.6,
        )
        fail_fraction = 1.0 - param_summary.loc[:, "pass_rate"].fillna(0.0).to_numpy(dtype=float)
        ax.bar(
            x_positions,
            fail_fraction,
            bottom=param_summary.loc[:, "pass_rate"].fillna(0.0).to_numpy(dtype=float),
            color=fail_color,
            alpha=0.35,
            edgecolor="black",
            linewidth=0.3,
        )
        for _, row in param_summary.iterrows():
            ax.text(
                float(row["value_order"] - 1),
                min(1.04, float(row["pass_rate"]) + 0.03 if np.isfinite(row["pass_rate"]) else 0.03),
                "{}/{}".format(int(row["pass_runs"]), int(row["total_runs"])),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(0.0, 1.08)
        ax.set_title(_humanize_name(parameter))
        ax.set_ylabel("pass rate")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(values, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for row_index, metric in enumerate(metric_columns, start=1):
            metric_ax = axes[row_index, col_index]
            plotted_means = list()
            for value_index, value in enumerate(values):
                is_value = pass_runs.loc[:, parameter].fillna("").astype(str).str.strip() == value
                numeric = pd.to_numeric(pass_runs.loc[is_value, metric], errors="coerce")
                finite = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
                if finite.shape[0] == 0:
                    plotted_means.append(np.nan)
                    continue
                if finite.shape[0] == 1:
                    x_jitter = np.array([0.0], dtype=float)
                else:
                    x_jitter = np.linspace(-0.18, 0.18, finite.shape[0])
                metric_ax.scatter(
                    np.full(finite.shape[0], float(value_index)) + x_jitter,
                    finite.to_numpy(dtype=float),
                    s=26,
                    color=point_color,
                    alpha=0.8,
                    linewidths=0.0,
                )
                plotted_means.append(float(finite.mean()))
            if any([np.isfinite(value) for value in plotted_means]):
                metric_ax.plot(
                    x_positions,
                    np.asarray(plotted_means, dtype=float),
                    color=mean_color,
                    linewidth=1.3,
                    marker="o",
                    markersize=3.0,
                )
            else:
                metric_ax.text(0.5, 0.5, "No pass data", ha="center", va="center", transform=metric_ax.transAxes)
            metric_ax.set_ylabel(_humanize_name(metric))
            metric_ax.set_xticks(x_positions)
            metric_ax.set_xticklabels(values, rotation=35, ha="right")
            metric_ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "benchmark overview: {} runs across {} benchmark outputs".format(
            int(runs.shape[0]),
            int(runs.loc[:, "benchmark_summary_tsv"].nunique()) if "benchmark_summary_tsv" in runs.columns else 0,
        ),
        fontsize=12,
    )
    fig.savefig(out_path, dpi=200, transparent=False)
    plt.close(fig)


def _write_output_manifest(g, runs_tsv, summary_tsv, summary_json, overview_plot):
    manifest_rows = list()
    for output_path, output_kind in [
        (runs_tsv, "benchmark_plot_runs_tsv"),
        (summary_tsv, "benchmark_plot_summary_tsv"),
        (summary_json, "benchmark_plot_summary_json"),
        (overview_plot, "benchmark_plot_overview"),
        (g["log_file"], "benchmark_plot_log"),
    ]:
        output_manifest.add_output_manifest_row(
            manifest_rows=manifest_rows,
            output_path=output_path,
            output_kind=output_kind,
            base_dir=g["outdir"],
        )
    manifest_path = runtime.output_path(g, "outputs.tsv")
    manifest_path = output_manifest.write_output_manifest(
        manifest_rows=manifest_rows,
        manifest_path=manifest_path,
        base_dir=g["outdir"],
    )
    print("Writing benchmark-plot output manifest: {}".format(manifest_path), flush=True)
    return manifest_path


def main_benchmark_plot(g):
    g = runtime.ensure_output_layout(g, create_dir=True)
    benchmark_dir = os.path.abspath(str(g.get("benchmark_dir", ".")).strip() or ".")
    if not os.path.isdir(benchmark_dir):
        raise ValueError("--benchmark_dir was not found or is not a directory: {}.".format(benchmark_dir))
    recursive = _as_bool(g.get("benchmark_plot_recursive", True))
    plot_format = str(g.get("benchmark_plot_format", "pdf")).strip().lower()
    if plot_format not in ["pdf", "png", "svg"]:
        raise ValueError('--benchmark_plot_format should be one of pdf, png, svg.')
    requested_metrics = _normalize_metric_list(g.get("benchmark_plot_metrics", ""))
    summary_paths = _collect_benchmark_summary_paths(root_dir=benchmark_dir, recursive=recursive)
    if len(summary_paths) == 0:
        raise ValueError("No benchmark summary TSV files were found under {}.".format(benchmark_dir))
    print("Benchmark summary files discovered: {:,}".format(len(summary_paths)), flush=True)
    run_tables = list()
    skipped_summary_paths = list()
    for summary_path in summary_paths:
        try:
            run_tables.append(_read_benchmark_runs(summary_path))
        except ValueError as exc:
            print("Skipping unreadable benchmark summary {}: {}".format(summary_path, exc), flush=True)
            skipped_summary_paths.append(os.path.abspath(summary_path))
    if len(run_tables) == 0:
        raise ValueError("No readable benchmark summary TSV files were found under {}.".format(benchmark_dir))
    runs = pd.concat(run_tables, axis=0, ignore_index=True, sort=False)
    if "status" not in runs.columns:
        runs.loc[:, "status"] = "pass"
    runs.loc[:, "status"] = runs.loc[:, "status"].fillna("unknown").astype(str).str.strip().str.lower()
    parameter_columns = _get_parameter_columns(runs)
    if len(parameter_columns) == 0:
        raise ValueError("No benchmark parameter columns were found in the discovered summaries.")
    metric_columns = _get_metric_columns(runs, requested_metrics=requested_metrics)
    parameter_summary = _build_parameter_summary(
        runs=runs,
        parameter_columns=parameter_columns,
        metric_columns=metric_columns,
    )
    runs_tsv = runtime.output_path(g, "benchmark_plot_runs.tsv")
    summary_tsv = runtime.output_path(g, "benchmark_plot_summary.tsv")
    summary_json = runtime.output_path(g, "benchmark_plot_summary.json")
    overview_plot = runtime.output_path(g, "benchmark_plot_overview.{}".format(plot_format))
    tsv.write_dataframe(runs, runs_tsv)
    tsv.write_dataframe(parameter_summary, summary_tsv)
    payload = {
        "benchmark_dir": benchmark_dir,
        "recursive": bool(recursive),
        "plot_format": plot_format,
        "metrics": list(metric_columns),
        "counts": {
            "summary_files": int(len(run_tables)),
            "summary_files_skipped": int(len(skipped_summary_paths)),
            "runs": int(runs.shape[0]),
            "pass": int((runs.loc[:, "status"] == "pass").sum()),
            "fail": int((runs.loc[:, "status"] != "pass").sum()),
        },
        "skipped_summary_files": list(skipped_summary_paths),
        "parameter_columns": list(parameter_columns),
        "runs": _dataframe_records_for_json(runs),
        "parameter_summary": _dataframe_records_for_json(parameter_summary),
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    _plot_parameter_overview(
        runs=runs,
        parameter_summary=parameter_summary,
        parameter_columns=parameter_columns,
        metric_columns=metric_columns,
        out_path=overview_plot,
    )
    print("Writing benchmark-plot runs TSV: {}".format(runs_tsv), flush=True)
    print("Writing benchmark-plot summary TSV: {}".format(summary_tsv), flush=True)
    print("Writing benchmark-plot summary JSON: {}".format(summary_json), flush=True)
    print("Writing benchmark-plot overview figure: {}".format(overview_plot), flush=True)
    if _as_bool(g.get("output_manifest", True)):
        _write_output_manifest(
            g=g,
            runs_tsv=runs_tsv,
            summary_tsv=summary_tsv,
            summary_json=summary_json,
            overview_plot=overview_plot,
        )
    print(
        "Benchmark-plot summary: {} runs, {} pass, {} fail".format(
            int(runs.shape[0]),
            int((runs.loc[:, "status"] == "pass").sum()),
            int((runs.loc[:, "status"] != "pass").sum()),
        ),
        flush=True,
    )

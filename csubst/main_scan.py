import time
import json

import numpy as np
import pandas as pd

from csubst import parser_misc
from csubst import runtime
from csubst import substitution
from csubst import substitution_scan
from csubst import scan_bootstrap
from csubst import scan_analytic
from csubst import tree
from csubst import main_sites
from csubst import tsv
from csubst.config_types import AnalysisConfig


def _require_foreground(g: AnalysisConfig) -> None:
    if g.get("foreground", None) is None:
        raise ValueError("csubst scan requires --foreground.")
    return None


def _scan_foreground_branch_ids(g, units_df=None):
    fg_ids: list[int] = []
    if (units_df is not None) and ("fg_branch_ids" in units_df.columns):
        for value in units_df["fg_branch_ids"].tolist():
            text = "" if value is None else str(value).strip()
            if text == "":
                continue
            fg_ids.extend(int(v) for v in text.split(",") if str(v).strip() != "")
    for values in g.get("fg_ids", {}).values():
        arr = np.asarray(values, dtype=np.int64).reshape(-1)
        fg_ids.extend(int(v) for v in arr.tolist())
    if len(fg_ids) == 0:
        return np.array([], dtype=np.int64)
    return np.array(sorted(set(fg_ids)), dtype=np.int64)


def _prepare_scan_output_table(scan_df):
    out = scan_df.copy()
    stat_columns = [
        col for col in out.columns
        if str(col).startswith(("p_", "q_", "score_")) or col in ("scan_pvalue_resolution", "log_e_endpoint_enrichment")
    ]
    for col in stat_columns:
        values = np.asarray(pd.to_numeric(out[col], errors="coerce"), dtype=np.float64)
        precision = "{:.17e}" if str(col).startswith(("score_", "log_e_", "p_endpoint_", "q_endpoint_")) else "{:.6e}"
        out[col] = [precision.format(value) if np.isfinite(value) else "" for value in values]
    return out


def _write_scan_calibration(g: AnalysisConfig) -> str:
    """Write diagnostics independently of scan rows, including empty scans."""
    path = runtime.output_path(g, "scan_calibration.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(g["scan_calibration_diagnostics"], handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Writing {}".format(path), flush=True)
    return path


def _write_scan_site_plot(g, scan_df, ON_tensor, units_df=None):
    if not bool(g.get("scan_site_plot", True)):
        print("Skipping scan site visualization (--scan_site_plot no).", flush=True)
        return []
    if scan_df.shape[0] == 0:
        print("Skipping scan site visualization because no candidates passed.", flush=True)
        return []
    filtered_scan_df = substitution_scan.filter_scan_site_plot_candidates(
        scan_df=scan_df,
        g=g,
    )
    filter_mode = str(g.get("scan_site_plot_filter", "all")).strip().lower()
    if filter_mode != "all":
        alpha = float(g.get("scan_site_plot_alpha", 0.05))
        num_sites = (
            filtered_scan_df["codon_site_alignment"].nunique()
            if "codon_site_alignment" in filtered_scan_df.columns
            else 0
        )
        print(
            "Scan site plot filter: {} P <= {:g}; retained {:,}/{:,} candidates across {:,} site(s).".format(
                filter_mode,
                alpha,
                int(filtered_scan_df.shape[0]),
                int(scan_df.shape[0]),
                int(num_sites),
            ),
            flush=True,
        )
    if filtered_scan_df.shape[0] == 0:
        print("Skipping scan site visualization because no candidate passed the plot display filter.", flush=True)
        return []
    site_df, branch_ids = substitution_scan.build_scan_site_plot_table(
        scan_df=filtered_scan_df,
        g=g,
        ON_tensor=ON_tensor,
    )
    if (site_df.shape[0] == 0) or (branch_ids.shape[0] == 0):
        print("Skipping scan site visualization because no supporting branches were available.", flush=True)
        return []
    foreground_branch_ids = _scan_foreground_branch_ids(g, units_df=units_df)
    if foreground_branch_ids.shape[0] == 0:
        foreground_branch_ids = branch_ids
    plot_prefix = str(g.get("output_prefix", "csubst")) + "_scan"
    if filter_mode != "all":
        plot_prefix = "{}_scan.{}_p{:g}".format(
            str(g.get("output_prefix", "csubst")),
            filter_mode,
            float(g.get("scan_site_plot_alpha", 0.05)),
        )
    plot_g = dict(g)
    plot_g.update(
        {
            "branch_ids": branch_ids,
            "tree_site_highlight_branch_ids": foreground_branch_ids,
            "tree_site_branch_color_mode": "single",
            "mode": "lineage",
            "single_branch_mode": False,
            "site_outdir": g["outdir"],
            "tree_site_plot": True,
            "tree_site_plot_prefix": plot_prefix,
            "tree_site_output_table": False,
            "min_single_prob": substitution_scan.scan_event_threshold(g),
            "min_combinat_prob": substitution_scan.scan_event_threshold(g),
        }
    )
    print("Writing scan site visualization from detected candidates.", flush=True)
    return main_sites.plot_tree_site(df=site_df, g=plot_g)


def main_scan(g: AnalysisConfig) -> tuple[AnalysisConfig, pd.DataFrame, pd.DataFrame]:
    start = time.time()
    g = runtime.ensure_output_layout(g, create_dir=True)
    _require_foreground(g)
    substitution_scan.validate_scan_configuration(g)
    scan_analytic.validate_options(g)
    scan_bootstrap.validate_options(g)
    scan_bootstrap.require_precise_fit(g)
    unit_mode = substitution_scan.normalize_scan_unit_mode(g.get("scan_unit_mode", "clade"))
    if unit_mode == "stem":
        g["fg_stem_only"] = True
    elif unit_mode == "clade":
        g["fg_stem_only"] = False
    print("Scan foreground unit mode: {}".format(unit_mode), flush=True)
    print("Reading and parsing input files.", flush=True)
    g["current_arity"] = 2
    g = parser_misc.prepare_input_context(
        g,
        include_foreground=True,
        include_marginal=False,
        resolve_state_subset=True,
        prepare_state=False,
    )
    substitution_scan.validate_scan_configuration(g)
    g = parser_misc.prep_state(g, apply_site_filtering=False)
    bootstrap_model = scan_bootstrap.prepare_model(g) if g.get("scan_pvalue_calibration") == "parametric_bootstrap" else None
    analytic_engine = scan_analytic.prepare(g)
    no_sites = False
    if bool(g.get('drop_invariant_tip_sites', False)):
        mask = parser_misc.get_site_drop_mask(
            g, g.get('drop_invariant_tip_sites_mode', 'tip_invariant'),
            parser_misc.get_site_index_alignment(g, expected_num_site=np.asarray(g['state_cdn']).shape[1]),
        )
        no_sites = bool(mask.size and mask.all())
    if no_sites:
        # A successful scan with no eligible sites is a no-test outcome, not a
        # failed bootstrap draw. Keep full states only for foreground metadata.
        g['scan_no_test_reason'] = 'all_sites_excluded_by_configured_filter'
    else:
        g = parser_misc.apply_site_filters(g)
    print("Generating nonsynonymous substitution tensor for scan.", flush=True)
    if g.get("scan_observation", "marginal") != "marginal":
        from csubst import scan_ctmc
        # Keep emissions for bootstrap missingness; inferred marginals fill missing tips.
        g.update(scan_tip_emissions=np.asarray(g["state_cdn"]))
        if g.get("scan_pvalue_calibration") == "parametric":
            scan_ctmc.validate_parametric_inputs(g)
        updated_g, ON_tensor_rate = scan_ctmc.prepare(g)
        # The CLI also holds this configuration. Replace its old state
        # references instead of leaving the caller's input arrays alive.
        g.update(updated_g)
        del updated_g
        if g.get("scan_pvalue_calibration") != "parametric" and analytic_engine is None:
            # Only model-based simulation and analytical likelihoods reuse
            # the original codon emissions after joint reconstruction.
            g.pop("scan_tip_emissions", None)
    else:
        ON_tensor_rate = substitution.get_substitution_tensor(
            state_tensor=g["state_nsy"],
            mode="asis",
            g=g,
            mmap_attr="scan_N",
        )
    rate_event_mode = substitution_scan.normalize_scan_rate_event_mode(g.get("scan_rate_event_mode", "posterior_sum"))
    ON_tensor_called = ON_tensor_rate
    if float(g.get("min_sub_pp", 0)) != 0:
        if (rate_event_mode == "posterior_sum") and isinstance(ON_tensor_rate, np.ndarray):
            called_path = runtime.temp_path("tmp.csubst.sub_tensor.scan_N_called.mmap")
            ON_tensor_called = np.memmap(
                called_path,
                dtype=ON_tensor_rate.dtype,
                mode="w+",
                shape=ON_tensor_rate.shape,
            )
            np.copyto(ON_tensor_called, ON_tensor_rate)
        ON_tensor_called = substitution.apply_min_sub_pp(g, ON_tensor_called)
    if g.get("scan_observation", "marginal") != "marginal":
        g = scan_ctmc.set_branch_length_summaries(g, ON_tensor_rate)
    else:
        print("Generating synonymous substitution tensor for branch-length context.", flush=True)
        OS_tensor = substitution.get_substitution_tensor(
            state_tensor=g["state_cdn"],
            mode="syn",
            g=g,
            mmap_attr="scan_S",
        )
        OS_tensor = substitution.apply_min_sub_pp(g, OS_tensor)
        g = tree.rescale_branch_length(g, OS_tensor, ON_tensor_called)
        del OS_tensor
    print("Scanning recurrent foreground substitution patterns.", flush=True)
    rate_ON_tensor = ON_tensor_rate if rate_event_mode == "posterior_sum" else ON_tensor_called
    if no_sites:
        scan_df = pd.DataFrame(columns=list(substitution_scan.SCAN_OUTPUT_COLUMNS))
        units_df = substitution_scan.build_scan_units(g, substitution_scan.build_branch_metadata(g))
        scan_df = substitution_scan._calibrate_scan_pvalues(
            g, scan_df, ON_tensor_called, rate_ON_tensor,
            substitution_scan._build_scan_static_context(g, ON_tensor_called, rate_ON_tensor),
        )
    else:
        scan_df, units_df = substitution_scan.scan_substitutions(
            g=g,
            ON_tensor=ON_tensor_called,
            rate_ON_tensor=rate_ON_tensor,
        )
    if bootstrap_model is not None:
        scan_df = scan_bootstrap.calibrate(g, scan_df, bootstrap_model)
    scan_df = scan_analytic.annotate(g, scan_df, units_df, analytic_engine)
    scan_bootstrap.write_inference_report(g, scan_df)
    scan_path = runtime.output_path(g, "scan.tsv")
    units_path = runtime.output_path(g, "scan_units.tsv")
    if scan_df.shape[0] == 0:
        print("No scan candidates passed the configured thresholds.", flush=True)
    scan_output_df = _prepare_scan_output_table(scan_df)
    tsv.write_dataframe(
        scan_output_df,
        scan_path,
        float_format=g["float_format"],
        chunksize=10000,
    )
    tsv.write_dataframe(
        units_df,
        units_path,
        float_format=g["float_format"],
    )
    print("Writing {}".format(scan_path), flush=True)
    print("Writing {}".format(units_path), flush=True)
    _write_scan_calibration(g)
    _write_scan_site_plot(g=g, scan_df=scan_df, ON_tensor=ON_tensor_called, units_df=units_df)
    print(
        "Scan candidates: {:,} rows, {:,} foreground units.".format(
            int(scan_df.shape[0]),
            int(units_df.shape[0]),
        ),
        flush=True,
    )
    elapsed_time = int(time.time() - start)
    print(("Elapsed time: {:,.1f} sec\n".format(elapsed_time)), flush=True)
    runtime.cleanup_legacy_temp_artifacts()
    return g, scan_df, units_df

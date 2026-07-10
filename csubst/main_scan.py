import time

import numpy as np
import pandas as pd

from csubst import parser_misc
from csubst import runtime
from csubst import substitution
from csubst import substitution_scan
from csubst import tree
from csubst import main_sites


def _require_foreground(g):
    if g.get("foreground", None) is None:
        raise ValueError("csubst scan requires --foreground.")
    return None


def _scan_foreground_branch_ids(g):
    fg_ids = []
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
        if str(col).startswith("p_") or str(col).startswith("q_")
    ]
    for col in stat_columns:
        values = np.asarray(pd.to_numeric(out[col], errors="coerce"), dtype=np.float64)
        out[col] = ["{:.6e}".format(value) if np.isfinite(value) else "" for value in values]
    return out


def _write_scan_site_plot(g, scan_df, ON_tensor):
    if not bool(g.get("scan_site_plot", True)):
        print("Skipping scan site visualization (--scan_site_plot no).", flush=True)
        return []
    if scan_df.shape[0] == 0:
        print("Skipping scan site visualization because no candidates passed.", flush=True)
        return []
    site_df, branch_ids = substitution_scan.build_scan_site_plot_table(
        scan_df=scan_df,
        g=g,
        ON_tensor=ON_tensor,
    )
    if (site_df.shape[0] == 0) or (branch_ids.shape[0] == 0):
        print("Skipping scan site visualization because no supporting branches were available.", flush=True)
        return []
    foreground_branch_ids = _scan_foreground_branch_ids(g)
    if foreground_branch_ids.shape[0] == 0:
        foreground_branch_ids = branch_ids
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
            "tree_site_plot_prefix": "csubst_scan",
            "tree_site_output_table": False,
            "min_single_prob": float(g.get("scan_min_event_pp", 0.5)),
            "min_combinat_prob": float(g.get("scan_min_event_pp", 0.5)),
        }
    )
    print("Writing scan site visualization from detected candidates.", flush=True)
    return main_sites.plot_tree_site(df=site_df, g=plot_g)


def main_scan(g):
    start = time.time()
    g = runtime.ensure_output_layout(g, create_dir=True)
    _require_foreground(g)
    print("Reading and parsing input files.", flush=True)
    g["current_arity"] = 2
    g = parser_misc.prepare_input_context(
        g,
        include_foreground=True,
        include_marginal=False,
        resolve_state_subset=True,
        prepare_state=False,
    )
    g = parser_misc.prep_state(g, apply_site_filtering=False)
    g = parser_misc.apply_site_filters(g)
    print("Generating nonsynonymous substitution tensor for scan.", flush=True)
    ON_tensor_rate = substitution.get_substitution_tensor(
        state_tensor=g["state_nsy"],
        mode="asis",
        g=g,
        mmap_attr="scan_N",
    )
    rate_event_mode = substitution_scan.normalize_scan_rate_event_mode(g.get("scan_rate_event_mode", "posterior_sum"))
    ON_tensor_called = ON_tensor_rate
    if float(g.get("min_sub_pp", 0)) != 0:
        if (rate_event_mode == "posterior_sum") and (not substitution._is_sparse_sub_tensor(ON_tensor_rate)):
            called_path = runtime.temp_path("tmp.csubst.sub_tensor.scan_N_called.mmap")
            ON_tensor_called = np.memmap(
                called_path,
                dtype=ON_tensor_rate.dtype,
                mode="w+",
                shape=ON_tensor_rate.shape,
            )
            np.copyto(ON_tensor_called, ON_tensor_rate)
        ON_tensor_called = substitution.apply_min_sub_pp(g, ON_tensor_called)
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
    scan_df, units_df = substitution_scan.scan_substitutions(
        g=g,
        ON_tensor=ON_tensor_called,
        rate_ON_tensor=rate_ON_tensor,
    )
    scan_path = runtime.output_path(g, "scan.tsv")
    units_path = runtime.output_path(g, "scan_units.tsv")
    if scan_df.shape[0] == 0:
        print("No scan candidates passed the configured thresholds.", flush=True)
    scan_output_df = _prepare_scan_output_table(scan_df)
    scan_output_df.to_csv(
        scan_path,
        sep="\t",
        index=False,
        float_format=g["float_format"],
        chunksize=10000,
    )
    units_df.to_csv(
        units_path,
        sep="\t",
        index=False,
        float_format=g["float_format"],
    )
    print("Writing {}".format(scan_path), flush=True)
    print("Writing {}".format(units_path), flush=True)
    _write_scan_site_plot(g=g, scan_df=scan_df, ON_tensor=ON_tensor_called)
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

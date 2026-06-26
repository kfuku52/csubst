import time

from csubst import parser_misc
from csubst import runtime
from csubst import substitution
from csubst import substitution_scan
from csubst import tree


def _require_foreground(g):
    if g.get("foreground", None) is None:
        raise ValueError("csubst scan requires --foreground.")
    return None


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
            ON_tensor_called = ON_tensor_rate.copy()
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
    scan_df.to_csv(
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

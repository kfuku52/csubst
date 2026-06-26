import math
import threading

import numpy as np
import pandas as pd
from scipy.stats import chi2

from csubst import ete
from csubst import foreground
from csubst import parallel
from csubst import parser_misc
from csubst import sequence
from csubst import substitution


SCAN_MATCHES = (
    "any2any",
    "any2spe",
    "any2dif",
    "spe2any",
    "spe2spe",
    "spe2dif",
    "dif2any",
    "dif2spe",
    "dif2dif",
)

SCAN_RATE_EVENT_MODES = ("called", "posterior_sum")
SCAN_RATE_EXPOSURES = ("q_weighted", "state_aware", "raw_branch_length")
SCAN_OTHER_SCOPES = ("all", "sister")
SCAN_PVALUE_CALIBRATIONS = ("none", "candidate_fixed", "full_scan")
_SCAN_PERMUTATION_RANDOM_LOCK = threading.Lock()


SCAN_OUTPUT_COLUMNS = (
    "scan_id",
    "trait",
    "target_class",
    "scan_match",
    "site",
    "nonsyn_recode",
    "from_state",
    "to_state",
    "from_state_ids",
    "to_state_ids",
    "observed_from_state_ids",
    "observed_to_state_ids",
    "from_state_distribution",
    "to_state_distribution",
    "state_change",
    "candidate_event_pp_sum",
    "scan_min_support_count",
    "scan_min_event_pp",
    "scan_rate_length_used",
    "scan_rate_exposure",
    "scan_rate_event_mode",
    "scan_other_scope",
    "scan_pvalue_calibration",
    "scan_n_permutations",
    "scan_permutation_seed",
    "scan_permutation_backend",
    "scan_permutation_n_jobs",
    "scan_permutation_success_count",
    "scan_permutation_failure_count",
    "codon_site_alignment",
    "site_rate",
    "site_rate_quantile",
    "state_all_conservation",
    "state_all_entropy",
    "state_all_major_state",
    "state_all_valid_tip_count",
    "state_bg_conservation",
    "state_bg_entropy",
    "state_bg_major_state",
    "state_bg_valid_tip_count",
    "aa_all_conservation",
    "aa_all_entropy",
    "aa_all_major_state",
    "aa_all_valid_tip_count",
    "aa_bg_conservation",
    "aa_bg_entropy",
    "aa_bg_major_state",
    "aa_bg_valid_tip_count",
    "unit_total",
    "support_unit_count",
    "support_fraction",
    "support_pp_sum",
    "support_pp_mean",
    "support_unit_ids",
    "support_branch_ids",
    "target_event_count",
    "target_event_branch_count",
    "target_raw_branch_length",
    "target_sn_rescaled_length",
    "target_n_rescaled_length",
    "target_exposure_branch_length",
    "other_event_count",
    "other_event_branch_count",
    "other_raw_branch_length",
    "other_sn_rescaled_length",
    "other_n_rescaled_length",
    "other_exposure_branch_length",
    "target_event_rate",
    "other_event_rate",
    "rate_ratio",
    "p_rate_enrichment",
    "p_rate_enrichment_empirical",
    "p_rate_enrichment_empirical_maxT",
    "q_rate_enrichment",
    "q_rate_enrichment_by_trait",
    "q_rate_enrichment_by_trait_match",
)


def normalize_scan_matches(value):
    value_txt = "" if value is None else str(value).strip().lower()
    if value_txt in ["", "all"]:
        return list(SCAN_MATCHES)
    out = list()
    seen = set()
    for token in value_txt.split(","):
        token = token.strip().lower()
        if token == "":
            continue
        if token == "all":
            tokens = SCAN_MATCHES
        else:
            if token not in SCAN_MATCHES:
                txt = "--scan_match should contain values from {}."
                raise ValueError(txt.format(", ".join(SCAN_MATCHES)))
            tokens = (token,)
        for normalized in tokens:
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
    if len(out) == 0:
        raise ValueError("--scan_match should contain at least one value.")
    return out


def normalize_scan_rate_event_mode(value):
    value_txt = "posterior_sum" if value is None else str(value).strip().lower()
    if value_txt not in SCAN_RATE_EVENT_MODES:
        txt = "--scan_rate_event_mode should be one of {}."
        raise ValueError(txt.format(", ".join(SCAN_RATE_EVENT_MODES)))
    return value_txt


def normalize_scan_rate_exposure(value):
    value_txt = "state_aware" if value is None else str(value).strip().lower()
    if value_txt not in SCAN_RATE_EXPOSURES:
        txt = "--scan_rate_exposure should be one of {}."
        raise ValueError(txt.format(", ".join(SCAN_RATE_EXPOSURES)))
    return value_txt


def normalize_scan_other_scope(value):
    value_txt = "all" if value is None else str(value).strip().lower()
    if value_txt not in SCAN_OTHER_SCOPES:
        txt = "--scan_other_scope should be one of {}."
        raise ValueError(txt.format(", ".join(SCAN_OTHER_SCOPES)))
    return value_txt


def normalize_scan_pvalue_calibration(value):
    value_txt = "full_scan" if value is None else str(value).strip().lower()
    if value_txt not in SCAN_PVALUE_CALIBRATIONS:
        txt = "--scan_pvalue_calibration should be one of {}."
        raise ValueError(txt.format(", ".join(SCAN_PVALUE_CALIBRATIONS)))
    return value_txt


def normalize_scan_n_permutations(value):
    n_permutations = int(value)
    if n_permutations < 0:
        raise ValueError("--scan_n_permutations should be >= 0.")
    return n_permutations


def _resolve_scan_permutation_backend(g):
    backend = str(g.get("parallel_backend", "auto")).strip().lower()
    if backend not in ["auto", "multiprocessing", "threading"]:
        raise ValueError("--parallel_backend should be one of auto, multiprocessing, threading.")
    return parallel.resolve_parallel_backend(g=g, task="general")


def _resolve_scan_permutation_n_jobs(g, n_permutations):
    n_permutations = int(n_permutations)
    if n_permutations <= 0:
        return 1
    return parallel.resolve_n_jobs(
        num_items=n_permutations,
        threads=int(g.get("threads", 1)),
    )


def parse_scan_support_threshold(value, total_units, param_name="--scan_min_support"):
    total_units = int(total_units)
    value_txt = str(value).strip().lower()
    if value_txt == "":
        raise ValueError("{} should be non-empty.".format(param_name))
    numeric = float(value_txt)
    if not np.isfinite(numeric):
        raise ValueError("{} should be finite.".format(param_name))
    if numeric < 0:
        raise ValueError("{} should be >= 0.".format(param_name))
    if numeric <= 1.0:
        return int(math.ceil(numeric * total_units))
    if not float(numeric).is_integer():
        raise ValueError("{} should be a fraction <= 1 or an integer count.".format(param_name))
    return int(numeric)


def _state_set_label(indices, state_orders, any_label="any", dif_label="dif"):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.shape[0] == 0:
        return ""
    if indices.shape[0] == state_orders.shape[0]:
        return any_label
    if indices.shape[0] == 1:
        return str(state_orders[int(indices[0])])
    return dif_label


def _state_distribution_label(events, col, state_orders):
    if events.shape[0] == 0:
        return ""
    grouped = events.groupby(col, as_index=True)["event_pp"].sum()
    grouped = grouped.sort_values(ascending=False)
    parts = []
    for idx, value in grouped.items():
        parts.append("{}:{:.6g}".format(str(state_orders[int(idx)]), float(value)))
    return ",".join(parts)


def _is_sparse_sub_tensor(sub_tensor):
    return substitution._is_sparse_sub_tensor(sub_tensor)


def extract_atomic_events(sub_tensor, min_event_pp=0.5, float_tol=1e-12):
    threshold = max(float(min_event_pp), float(float_tol))
    rows = []
    if _is_sparse_sub_tensor(sub_tensor):
        for (_sg, anc, der), mat in sub_tensor.blocks.items():
            if int(anc) == int(der):
                continue
            coo = mat.tocoo()
            if coo.data.shape[0] == 0:
                continue
            keep = np.asarray(coo.data >= threshold, dtype=bool)
            if not keep.any():
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "branch_id": coo.row[keep].astype(np.int64, copy=False),
                        "site": coo.col[keep].astype(np.int64, copy=False),
                        "from_state_id": int(anc),
                        "to_state_id": int(der),
                        "event_pp": coo.data[keep].astype(np.float64, copy=False),
                    }
                )
            )
    else:
        arr = np.asarray(sub_tensor)
        if arr.ndim != 5:
            raise ValueError("sub_tensor should be 5D.")
        branch_id, site, _group, anc, der = np.where(arr >= threshold)
        if branch_id.shape[0] > 0:
            keep = anc != der
            branch_id = branch_id[keep]
            site = site[keep]
            anc = anc[keep]
            der = der[keep]
            event_pp = arr[branch_id, site, 0, anc, der]
            rows.append(
                pd.DataFrame(
                    {
                        "branch_id": branch_id.astype(np.int64, copy=False),
                        "site": site.astype(np.int64, copy=False),
                        "from_state_id": anc.astype(np.int64, copy=False),
                        "to_state_id": der.astype(np.int64, copy=False),
                        "event_pp": event_pp.astype(np.float64, copy=False),
                    }
                )
            )
    if len(rows) == 0:
        return pd.DataFrame(
            columns=["branch_id", "site", "from_state_id", "to_state_id", "event_pp"]
        )
    out = pd.concat(rows, ignore_index=True)
    return out


def extract_candidate_posterior_events(sub_tensor, site, from_ids, to_ids, float_tol=1e-12):
    site = int(site)
    from_ids = np.array(sorted(set(np.asarray(from_ids, dtype=np.int64).reshape(-1).tolist())), dtype=np.int64)
    to_ids = np.array(sorted(set(np.asarray(to_ids, dtype=np.int64).reshape(-1).tolist())), dtype=np.int64)
    if _is_sparse_sub_tensor(sub_tensor):
        num_branch = int(sub_tensor.shape[0])
        num_site = int(sub_tensor.shape[1])
        num_group = int(sub_tensor.shape[2])
        num_from = int(sub_tensor.shape[3])
        num_to = int(sub_tensor.shape[4])
    else:
        arr = np.asarray(sub_tensor)
        if arr.ndim != 5:
            raise ValueError("sub_tensor should be 5D.")
        num_branch, num_site, num_group, num_from, num_to = arr.shape
    if num_group < 1:
        raise ValueError("sub_tensor should contain at least one substitution group.")
    if (site < 0) or (site >= num_site):
        raise ValueError("site is outside the substitution tensor site axis.")
    branch_values = np.zeros(shape=(num_branch,), dtype=np.float64)
    for anc in from_ids.tolist():
        if (int(anc) < 0) or (int(anc) >= num_from):
            continue
        for der in to_ids.tolist():
            if (int(der) < 0) or (int(der) >= num_to) or (int(anc) == int(der)):
                continue
            if _is_sparse_sub_tensor(sub_tensor):
                mat = sub_tensor.blocks.get((0, int(anc), int(der)), None)
                if mat is None:
                    continue
                col = mat.getcol(site).tocoo()
                if col.data.shape[0] == 0:
                    continue
                np.add.at(branch_values, col.row.astype(np.int64, copy=False), col.data.astype(np.float64, copy=False))
            else:
                branch_values += arr[:, site, 0, int(anc), int(der)].astype(np.float64, copy=False)
    keep = np.where(branch_values > float(float_tol))[0]
    if keep.shape[0] == 0:
        return pd.DataFrame(
            columns=["branch_id", "site", "from_state_id", "to_state_id", "event_pp"]
        )
    return pd.DataFrame(
        {
            "branch_id": keep.astype(np.int64, copy=False),
            "site": np.full(shape=keep.shape[0], fill_value=site, dtype=np.int64),
            "from_state_id": np.full(shape=keep.shape[0], fill_value=-1, dtype=np.int64),
            "to_state_id": np.full(shape=keep.shape[0], fill_value=-1, dtype=np.int64),
            "event_pp": branch_values[keep].astype(np.float64, copy=False),
        }
    )


def _get_branch_prop_length_to_effective_parent(node, prop_name, state_has_mass):
    length = max(float(ete.get_prop(node, prop_name, 0.0) or 0.0), 0.0)
    parent = getattr(node, "up", None)
    while parent is not None:
        if ete.node_has_state(parent, state_has_mass=state_has_mass):
            return length
        length += max(float(ete.get_prop(parent, prop_name, 0.0) or 0.0), 0.0)
        parent = getattr(parent, "up", None)
    return length


def build_branch_metadata(g):
    state = np.asarray(g["state_nsy"])
    state_has_mass = state.sum(axis=(1, 2)) > float(g.get("float_tol", 0))
    rows = []
    for node in g["tree"].traverse():
        if ete.is_root(node):
            continue
        branch_id = int(ete.get_prop(node, "numerical_label"))
        if (not ete.is_leaf(node)) and (not ete.node_has_state(node, state_has_mass=state_has_mass)):
            continue
        parent_node, raw_length = ete.get_effective_state_parent(
            node,
            state_has_mass=state_has_mass,
            accumulate_distance=True,
        )
        if parent_node is None:
            continue
        parent_id = int(ete.get_prop(parent_node, "numerical_label"))
        if (branch_id < 0) or (branch_id >= state.shape[0]) or (parent_id < 0) or (parent_id >= state.shape[0]):
            continue
        rows.append(
            {
                "branch_id": branch_id,
                "parent_id": parent_id,
                "raw_length": max(float(raw_length), 0.0),
                "sn_rescaled_length": _get_branch_prop_length_to_effective_parent(
                    node=node,
                    prop_name="SNdist",
                    state_has_mass=state_has_mass,
                ),
                "n_rescaled_length": _get_branch_prop_length_to_effective_parent(
                    node=node,
                    prop_name="Ndist",
                    state_has_mass=state_has_mass,
                ),
            }
        )
    if len(rows) == 0:
        raise ValueError("No analyzable non-root branches were available for scan.")
    return pd.DataFrame(rows).sort_values("branch_id").reset_index(drop=True)


def _lineage_values(g, trait_name):
    values = foreground._get_non_background_lineages(g["fg_df"].loc[:, trait_name])
    return values.tolist()


def _node_by_branch_id(g):
    return {int(ete.get_prop(node, "numerical_label")): node for node in g["tree"].traverse()}


def _lineage_sister_branch_ids(g, trait_name, lineage_index, fg_branch_ids, node_by_id):
    fg_set = set(int(v) for v in np.asarray(fg_branch_ids, dtype=np.int64).reshape(-1).tolist())
    all_fg_set = set(int(v) for v in np.asarray(g["fg_ids"][trait_name], dtype=np.int64).reshape(-1).tolist())
    sister_ids = set()
    lineage_key = "is_lineage_fg_" + str(trait_name) + "_" + str(int(lineage_index) + 1)
    for bid in fg_set:
        node = node_by_id.get(int(bid), None)
        if node is None:
            continue
        for sister in ete.get_sisters(node):
            sister_id = int(ete.get_prop(sister, "numerical_label"))
            is_sister_lineage = bool(ete.get_prop(sister, lineage_key, False))
            if bool(g.get("scan_sister_stem_only", False)):
                if (not is_sister_lineage) and (sister_id not in all_fg_set):
                    sister_ids.add(sister_id)
            else:
                for desc in sister.traverse():
                    desc_id = int(ete.get_prop(desc, "numerical_label"))
                    if desc_id in all_fg_set:
                        continue
                    if not bool(ete.get_prop(desc, lineage_key, False)):
                        sister_ids.add(desc_id)
    sister_ids = sister_ids.difference(fg_set)
    return np.array(sorted(sister_ids), dtype=np.int64)


def _lineage_marginal_branch_ids(g, trait_name, lineage_index, fg_branch_ids, node_by_id):
    fg_set = set(int(v) for v in np.asarray(fg_branch_ids, dtype=np.int64).reshape(-1).tolist())
    mg_ids = set()
    if bool(g.get("mg_parent", False)):
        for bid in fg_set:
            node = node_by_id.get(int(bid), None)
            parent = getattr(node, "up", None) if node is not None else None
            if parent is None or ete.is_root(parent):
                continue
            parent_id = int(ete.get_prop(parent, "numerical_label"))
            if parent_id not in fg_set:
                mg_ids.add(parent_id)
    if bool(g.get("mg_sister", False)):
        sister_ids = _lineage_sister_branch_ids(
            g=g,
            trait_name=trait_name,
            lineage_index=lineage_index,
            fg_branch_ids=fg_branch_ids,
            node_by_id=node_by_id,
        )
        mg_ids.update(int(v) for v in sister_ids.tolist())
    if trait_name in g.get("mg_ids", {}):
        allowed = set(int(v) for v in np.asarray(g["mg_ids"][trait_name], dtype=np.int64).reshape(-1).tolist())
        mg_ids = mg_ids.intersection(allowed)
    mg_ids = mg_ids.difference(fg_set)
    return np.array(sorted(mg_ids), dtype=np.int64)


def build_scan_units(g, branch_meta):
    rows = []
    valid_branches = set(branch_meta["branch_id"].astype(int).tolist())
    node_by_id = _node_by_branch_id(g)
    for trait_name in g["fg_df"].columns[1:].tolist():
        lineages = _lineage_values(g=g, trait_name=trait_name)
        for i, lineage_value in enumerate(lineages):
            fg_ids = foreground._get_lineage_target_ids(
                lineage_index=i,
                trait_name=trait_name,
                g=g,
            )
            fg_ids = np.array([int(v) for v in fg_ids.tolist() if int(v) in valid_branches], dtype=np.int64)
            mg_ids = _lineage_marginal_branch_ids(
                g=g,
                trait_name=trait_name,
                lineage_index=i,
                fg_branch_ids=fg_ids,
                node_by_id=node_by_id,
            )
            mg_ids = np.array([int(v) for v in mg_ids.tolist() if int(v) in valid_branches], dtype=np.int64)
            sister_ids = _lineage_sister_branch_ids(
                g=g,
                trait_name=trait_name,
                lineage_index=i,
                fg_branch_ids=fg_ids,
                node_by_id=node_by_id,
            )
            sister_ids = np.array([int(v) for v in sister_ids.tolist() if int(v) in valid_branches], dtype=np.int64)
            leaf_names = []
            if trait_name in g.get("fg_leaf_names", {}) and i < len(g["fg_leaf_names"][trait_name]):
                leaf_names = sorted([str(v) for v in g["fg_leaf_names"][trait_name][i]])
            rows.append(
                {
                    "trait": trait_name,
                    "unit_id": i + 1,
                    "lineage_value": lineage_value,
                    "fg_leaf_names": ",".join(leaf_names),
                    "fg_branch_ids": ",".join(str(v) for v in fg_ids.tolist()),
                    "mg_branch_ids": ",".join(str(v) for v in mg_ids.tolist()),
                    "sister_branch_ids": ",".join(str(v) for v in sister_ids.tolist()),
                }
            )
    return pd.DataFrame(rows)


def _parse_id_list(value):
    text = "" if value is None else str(value).strip()
    if text == "":
        return np.array([], dtype=np.int64)
    return np.array([int(v) for v in text.split(",") if str(v).strip() != ""], dtype=np.int64)


def _unit_sister_branch_ids(g, fg_branch_ids, all_fg_branch_ids, node_by_id):
    fg_set = set(int(v) for v in np.asarray(fg_branch_ids, dtype=np.int64).reshape(-1).tolist())
    all_fg_set = set(int(v) for v in np.asarray(all_fg_branch_ids, dtype=np.int64).reshape(-1).tolist())
    sister_ids = set()
    for bid in fg_set:
        node = node_by_id.get(int(bid), None)
        if node is None:
            continue
        for sister in ete.get_sisters(node):
            sister_id = int(ete.get_prop(sister, "numerical_label"))
            if bool(g.get("scan_sister_stem_only", False)):
                if sister_id not in all_fg_set:
                    sister_ids.add(sister_id)
            else:
                for desc in sister.traverse():
                    desc_id = int(ete.get_prop(desc, "numerical_label"))
                    if desc_id not in all_fg_set:
                        sister_ids.add(desc_id)
    return np.array(sorted(sister_ids.difference(fg_set)), dtype=np.int64)


def _unit_marginal_branch_ids(g, fg_branch_ids, all_fg_branch_ids, sister_branch_ids, node_by_id):
    fg_set = set(int(v) for v in np.asarray(fg_branch_ids, dtype=np.int64).reshape(-1).tolist())
    all_fg_set = set(int(v) for v in np.asarray(all_fg_branch_ids, dtype=np.int64).reshape(-1).tolist())
    mg_ids = set()
    if bool(g.get("mg_parent", False)):
        for bid in fg_set:
            node = node_by_id.get(int(bid), None)
            parent = getattr(node, "up", None) if node is not None else None
            if parent is None or ete.is_root(parent):
                continue
            parent_id = int(ete.get_prop(parent, "numerical_label"))
            if parent_id not in all_fg_set:
                mg_ids.add(parent_id)
    if bool(g.get("mg_sister", False)):
        mg_ids.update(int(v) for v in np.asarray(sister_branch_ids, dtype=np.int64).reshape(-1).tolist())
    return np.array(sorted(mg_ids.difference(all_fg_set)), dtype=np.int64)


def _selected_stem_fg_branch_ids(g, trait_cache, stem_index):
    stem_index = int(stem_index)
    if bool(g.get("fg_stem_only", False)):
        return np.array([int(trait_cache["branch_ids"][stem_index])], dtype=np.int64)
    return np.array(trait_cache["descendant_branch_ids_by_index"][stem_index], dtype=np.int64)


def _build_permuted_trait_context(g, trait_name, valid_branch_ids, sample_original_foreground):
    if "min_clade_bin_count" not in g:
        g["min_clade_bin_count"] = int(g.get("scan_permutation_min_clade_bin_count", 10))
    trait_cache = foreground._get_trait_clade_permutation_cache(g=g, trait_name=trait_name)
    randomization_plan = foreground._get_clade_permutation_randomization_plan(
        g=g,
        trait_name=trait_name,
        sample_original_foreground=sample_original_foreground,
    )
    stem_flags = foreground._randomize_foreground_stem_flags_from_plan(
        trait_cache=trait_cache,
        randomization_plan=randomization_plan,
        sample_original_foreground=sample_original_foreground,
    )
    stem_indices = np.where(stem_flags)[0].astype(np.int64, copy=False)
    valid_set = set(int(v) for v in np.asarray(valid_branch_ids, dtype=np.int64).reshape(-1).tolist())
    node_by_id = _node_by_branch_id(g)
    unit_fg_arrays = []
    fg_leaf_names = []
    for stem_index in stem_indices.tolist():
        fg_ids = _selected_stem_fg_branch_ids(g=g, trait_cache=trait_cache, stem_index=stem_index)
        fg_ids = np.array([int(v) for v in fg_ids.tolist() if int(v) in valid_set], dtype=np.int64)
        if fg_ids.shape[0] == 0:
            continue
        unit_fg_arrays.append(fg_ids)
        fg_leaf_names.append(sorted([str(v) for v in trait_cache["leaf_names_by_index"][int(stem_index)]]))
    if len(unit_fg_arrays) == 0:
        return {
            "units": pd.DataFrame(columns=["trait", "unit_id", "lineage_value", "fg_leaf_names", "fg_branch_ids", "mg_branch_ids", "sister_branch_ids"]),
            "fg_ids": np.array([], dtype=np.int64),
            "mg_ids": np.array([], dtype=np.int64),
            "fg_leaf_names": [],
        }
    all_fg_ids = np.unique(np.concatenate(unit_fg_arrays).astype(np.int64, copy=False))
    rows = []
    all_mg_ids = []
    for i, fg_ids in enumerate(unit_fg_arrays):
        sister_ids = _unit_sister_branch_ids(
            g=g,
            fg_branch_ids=fg_ids,
            all_fg_branch_ids=all_fg_ids,
            node_by_id=node_by_id,
        )
        sister_ids = np.array([int(v) for v in sister_ids.tolist() if int(v) in valid_set], dtype=np.int64)
        mg_ids = _unit_marginal_branch_ids(
            g=g,
            fg_branch_ids=fg_ids,
            all_fg_branch_ids=all_fg_ids,
            sister_branch_ids=sister_ids,
            node_by_id=node_by_id,
        )
        mg_ids = np.array([int(v) for v in mg_ids.tolist() if int(v) in valid_set], dtype=np.int64)
        all_mg_ids.append(mg_ids)
        rows.append(
            {
                "trait": trait_name,
                "unit_id": i + 1,
                "lineage_value": i + 1,
                "fg_leaf_names": ",".join(fg_leaf_names[i]),
                "fg_branch_ids": ",".join(str(v) for v in fg_ids.tolist()),
                "mg_branch_ids": ",".join(str(v) for v in mg_ids.tolist()),
                "sister_branch_ids": ",".join(str(v) for v in sister_ids.tolist()),
            }
        )
    if len(all_mg_ids) > 0:
        mg_union = np.unique(np.concatenate(all_mg_ids).astype(np.int64, copy=False))
    else:
        mg_union = np.array([], dtype=np.int64)
    return {
        "units": pd.DataFrame(rows),
        "fg_ids": all_fg_ids,
        "mg_ids": mg_union,
        "fg_leaf_names": fg_leaf_names,
    }


def _build_permuted_scan_context(g, trait_names, valid_branch_ids, sample_original_foreground):
    units = []
    fg_ids = {}
    mg_ids = {}
    fg_leaf_names = {}
    for trait_name in trait_names:
        trait_context = _build_permuted_trait_context(
            g=g,
            trait_name=trait_name,
            valid_branch_ids=valid_branch_ids,
            sample_original_foreground=sample_original_foreground,
        )
        units.append(trait_context["units"])
        fg_ids[trait_name] = trait_context["fg_ids"]
        mg_ids[trait_name] = trait_context["mg_ids"]
        fg_leaf_names[trait_name] = trait_context["fg_leaf_names"]
    units_df = pd.concat(units, ignore_index=True) if len(units) > 0 else pd.DataFrame()
    return {
        "units": units_df,
        "fg_ids": fg_ids,
        "mg_ids": mg_ids,
        "fg_leaf_names": fg_leaf_names,
        "trait_names": list(trait_names),
    }


def _candidate_state_sets(match, group, num_state):
    all_states = np.arange(num_state, dtype=np.int64)
    observed_from = np.array(sorted(group["from_state_id"].astype(int).unique().tolist()), dtype=np.int64)
    observed_to = np.array(sorted(group["to_state_id"].astype(int).unique().tolist()), dtype=np.int64)
    anc_token, der_token = match.split("2", 1)
    if anc_token == "any":
        from_set = all_states
    else:
        from_set = observed_from
    if der_token == "any":
        to_set = all_states
    else:
        to_set = observed_to
    if anc_token == "spe" and observed_from.shape[0] != 1:
        return None
    if anc_token == "dif" and observed_from.shape[0] < 2:
        return None
    if der_token == "spe" and observed_to.shape[0] != 1:
        return None
    if der_token == "dif" and observed_to.shape[0] < 2:
        return None
    return from_set, to_set, observed_from, observed_to


def _candidate_group_columns(match):
    if match == "spe2spe":
        return ["site", "from_state_id", "to_state_id"]
    if match in ["any2spe", "dif2spe"]:
        return ["site", "to_state_id"]
    if match in ["spe2any", "spe2dif"]:
        return ["site", "from_state_id"]
    return ["site"]


def _format_state_change(from_label, to_label, site_label, match):
    site_label = int(site_label)
    from_label = str(from_label)
    to_label = str(to_label)
    match = str(match)
    if match == "spe2spe":
        return "{}{}{}".format(from_label, site_label, to_label)
    if to_label not in ["any", "dif"]:
        return "{}{}".format(site_label, to_label)
    if from_label not in ["any", "dif"]:
        return "{}{}".format(from_label, site_label)
    return "{}:{}".format(site_label, match)


def build_candidates(events, scan_matches, state_orders):
    rows = []
    num_state = int(state_orders.shape[0])
    for match in scan_matches:
        if events.shape[0] == 0:
            continue
        group_cols = _candidate_group_columns(match)
        grouped = events.groupby(group_cols, sort=True)
        for key, group in grouped:
            state_sets = _candidate_state_sets(match=match, group=group, num_state=num_state)
            if state_sets is None:
                continue
            from_set, to_set, observed_from, observed_to = state_sets
            site = int(group["site"].iloc[0])
            from_label = _state_set_label(from_set, state_orders)
            to_label = _state_set_label(to_set, state_orders)
            state_change = _format_state_change(
                from_label=from_label,
                to_label=to_label,
                site_label=site + 1,
                match=match,
            )
            rows.append(
                {
                    "scan_match": match,
                    "site": site,
                    "from_state_ids": ",".join(str(v) for v in from_set.tolist()),
                    "to_state_ids": ",".join(str(v) for v in to_set.tolist()),
                    "observed_from_state_ids": ",".join(str(v) for v in observed_from.tolist()),
                    "observed_to_state_ids": ",".join(str(v) for v in observed_to.tolist()),
                    "from_state": from_label,
                    "to_state": to_label,
                    "from_state_distribution": _state_distribution_label(group, "from_state_id", state_orders),
                    "to_state_distribution": _state_distribution_label(group, "to_state_id", state_orders),
                    "state_change": state_change,
                    "candidate_event_pp_sum": float(group["event_pp"].sum()),
                }
            )
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _event_mask_for_candidate(events, site, from_ids, to_ids):
    if events.shape[0] == 0:
        return np.zeros(shape=(0,), dtype=bool)
    from_ids = set(int(v) for v in np.asarray(from_ids, dtype=np.int64).reshape(-1).tolist())
    to_ids = set(int(v) for v in np.asarray(to_ids, dtype=np.int64).reshape(-1).tolist())
    return (
        (events["site"].astype(int) == int(site))
        & events["from_state_id"].astype(int).isin(from_ids)
        & events["to_state_id"].astype(int).isin(to_ids)
        & (events["from_state_id"].astype(int) != events["to_state_id"].astype(int))
    ).to_numpy(dtype=bool)


def _support_for_unit(branch_event, branch_ids, min_event_pp):
    branch_ids = np.asarray(branch_ids, dtype=np.int64).reshape(-1)
    if branch_ids.shape[0] == 0:
        return 0.0, []
    values = []
    supporting = []
    for bid in branch_ids.tolist():
        value = float(branch_event.get(int(bid), 0.0))
        if value >= float(min_event_pp):
            values.append(value)
            supporting.append(int(bid))
    if len(values) == 0:
        return 0.0, []
    return max(values), supporting


def _summarize_unit_support(candidate_events, units_df, target_class, min_event_pp):
    if str(target_class) != "fg":
        raise ValueError("Only foreground target class is supported in scan.")
    if units_df.shape[0] == 0:
        return {
            "unit_total": 0,
            "support_unit_count": 0,
            "support_fraction": np.nan,
            "support_pp_sum": 0.0,
            "support_pp_mean": np.nan,
            "support_unit_ids": "",
            "support_branch_ids": "",
        }
    branch_event = candidate_events.groupby("branch_id")["event_pp"].sum().to_dict()
    support_pp = []
    support_units = []
    support_branches = []
    for _, row in units_df.iterrows():
        fg_ids = _parse_id_list(row.get("fg_branch_ids", ""))
        fg_pp, fg_support_branches = _support_for_unit(branch_event, fg_ids, min_event_pp)
        is_support = fg_pp >= float(min_event_pp)
        pp = fg_pp
        branches = fg_support_branches
        if is_support:
            support_units.append(int(row["unit_id"]))
            support_pp.append(float(pp))
            support_branches.extend([int(v) for v in branches])
    unit_total = int(units_df.shape[0])
    support_count = int(len(support_units))
    return {
        "unit_total": unit_total,
        "support_unit_count": support_count,
        "support_fraction": (support_count / unit_total) if unit_total > 0 else np.nan,
        "support_pp_sum": float(np.sum(support_pp)) if len(support_pp) else 0.0,
        "support_pp_mean": float(np.mean(support_pp)) if len(support_pp) else np.nan,
        "support_unit_ids": ",".join(str(v) for v in support_units),
        "support_branch_ids": ",".join(str(v) for v in sorted(set(support_branches))),
    }


def _target_branch_ids_from_maps(fg_ids_map, mg_ids_map, trait_name, target_class, valid_branch_ids):
    if str(target_class) != "fg":
        raise ValueError("Only foreground target class is supported in scan.")
    valid = set(int(v) for v in valid_branch_ids)
    fg_ids = set(int(v) for v in np.asarray(fg_ids_map.get(trait_name, []), dtype=np.int64).reshape(-1).tolist())
    return np.array(sorted(fg_ids.intersection(valid)), dtype=np.int64)


def _target_branch_ids(g, trait_name, target_class, valid_branch_ids):
    return _target_branch_ids_from_maps(
        fg_ids_map=g.get("fg_ids", {}),
        mg_ids_map=g.get("mg_ids", {}),
        trait_name=trait_name,
        target_class=target_class,
        valid_branch_ids=valid_branch_ids,
    )


def _union_unit_branch_ids(units_df, column):
    out = set()
    if column not in units_df.columns:
        return out
    for value in units_df[column].tolist():
        out.update(int(v) for v in _parse_id_list(value).tolist())
    return out


def _other_branch_ids_from_maps(
    fg_ids_map,
    mg_ids_map,
    trait_name,
    target_class,
    valid_branch_ids,
    scan_other_scope,
    units_df,
):
    if str(target_class) != "fg":
        raise ValueError("Only foreground target class is supported in scan.")
    valid = set(int(v) for v in np.asarray(valid_branch_ids, dtype=np.int64).reshape(-1).tolist())
    target = set(
        int(v)
        for v in _target_branch_ids_from_maps(
            fg_ids_map=fg_ids_map,
            mg_ids_map=mg_ids_map,
            trait_name=trait_name,
            target_class=target_class,
            valid_branch_ids=valid_branch_ids,
        ).tolist()
    )
    sister = _union_unit_branch_ids(units_df=units_df, column="sister_branch_ids").intersection(valid)
    if scan_other_scope == "all":
        out = valid.difference(target)
    elif scan_other_scope == "sister":
        out = sister
    else:
        txt = "--scan_other_scope should be one of {}."
        raise ValueError(txt.format(", ".join(SCAN_OTHER_SCOPES)))
    return np.array(sorted(out.intersection(valid).difference(target)), dtype=np.int64)


def _other_branch_ids(g, trait_name, target_class, valid_branch_ids, scan_other_scope, units_df):
    return _other_branch_ids_from_maps(
        fg_ids_map=g.get("fg_ids", {}),
        mg_ids_map=g.get("mg_ids", {}),
        trait_name=trait_name,
        target_class=target_class,
        valid_branch_ids=valid_branch_ids,
        scan_other_scope=scan_other_scope,
        units_df=units_df,
    )


def _opportunity_states(from_ids, to_ids):
    to_set = set(int(v) for v in np.asarray(to_ids, dtype=np.int64).reshape(-1).tolist())
    out = []
    for anc in np.asarray(from_ids, dtype=np.int64).reshape(-1).tolist():
        anc = int(anc)
        if any(der != anc for der in to_set):
            out.append(anc)
    return np.array(sorted(set(out)), dtype=np.int64)


def _length_column(rate_length):
    if rate_length == "raw":
        return "raw_length"
    if rate_length == "sn_rescaled":
        return "sn_rescaled_length"
    if rate_length == "n_rescaled":
        return "n_rescaled_length"
    raise ValueError("--scan_rate_length should be one of raw, sn_rescaled, n_rescaled.")


def _poisson_lrt_pvalue(x_target, l_target, x_other, l_other):
    x_target = float(x_target)
    x_other = float(x_other)
    l_target = float(l_target)
    l_other = float(l_other)
    if (l_target <= 0) or (l_other <= 0):
        return np.nan
    rate_target = x_target / l_target
    rate_other = x_other / l_other
    if not (rate_target > rate_other):
        return 1.0
    x_total = x_target + x_other
    l_total = l_target + l_other
    if (x_total <= 0) or (l_total <= 0):
        return 1.0

    def term(x, exposure, rate):
        if (x <= 0) or (rate <= 0):
            return -rate * exposure
        return x * math.log(rate) - rate * exposure

    common_rate = x_total / l_total
    ll_alt = term(x_target, l_target, rate_target) + term(x_other, l_other, rate_other)
    ll_null = term(x_target, l_target, common_rate) + term(x_other, l_other, common_rate)
    statistic = max(0.0, 2.0 * (ll_alt - ll_null))
    return min(1.0, float(0.5 * chi2.sf(statistic, df=1)))


def _bh_qvalues(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    qvalues = np.full(shape=pvalues.shape, fill_value=np.nan, dtype=np.float64)
    valid = np.isfinite(pvalues)
    if not valid.any():
        return qvalues
    valid_indices = np.where(valid)[0]
    p = pvalues[valid]
    order = np.argsort(p)
    ranked = p[order]
    n = ranked.shape[0]
    q = np.empty(shape=n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        value = ranked[i] * n / (i + 1)
        prev = min(prev, value)
        q[i] = prev
    out = np.empty(shape=n, dtype=np.float64)
    out[order] = np.minimum(q, 1.0)
    qvalues[valid_indices] = out
    return qvalues


def _assign_grouped_qvalues(scan_df, out_col, group_cols):
    scan_df[out_col] = np.nan
    if scan_df.shape[0] == 0:
        return scan_df
    for _, index_values in scan_df.groupby(group_cols, sort=False, dropna=False).groups.items():
        pvalues = scan_df.loc[index_values, "p_rate_enrichment"].to_numpy(dtype=np.float64)
        scan_df.loc[index_values, out_col] = _bh_qvalues(pvalues)
    return scan_df


def _q_weighted_opportunity(branch_meta, state_nsy, site, from_ids, to_ids, q_matrix):
    if q_matrix is None:
        raise ValueError("--scan_rate_exposure q_weighted requires instantaneous_nsy_rate_matrix.")
    state_nsy = np.asarray(state_nsy, dtype=np.float64)
    q = np.asarray(q_matrix, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("instantaneous_nsy_rate_matrix should be a square matrix.")
    num_state = int(state_nsy.shape[2])
    if q.shape[0] != num_state:
        txt = "instantaneous_nsy_rate_matrix shape ({}) did not match nonsyn state axis ({})."
        raise ValueError(txt.format(q.shape[0], num_state))
    q_positive = np.where(np.isfinite(q) & (q > 0), q, 0.0)
    weights = np.zeros(shape=(num_state,), dtype=np.float64)
    to_ids = np.asarray(to_ids, dtype=np.int64).reshape(-1)
    for anc in np.asarray(from_ids, dtype=np.int64).reshape(-1).tolist():
        anc = int(anc)
        if (anc < 0) or (anc >= num_state):
            continue
        allowed_to = [int(der) for der in to_ids.tolist() if (int(der) != anc) and (0 <= int(der) < num_state)]
        if len(allowed_to) == 0:
            continue
        weights[anc] = float(q_positive[anc, allowed_to].sum())
    parent_ids = branch_meta["parent_id"].astype(int).to_numpy(copy=False)
    return state_nsy[parent_ids, int(site), :].dot(weights)


def _rate_summary(
    candidate_events,
    branch_meta,
    state_nsy,
    site,
    from_ids,
    to_ids,
    target_branch_ids,
    rate_length,
    rate_exposure,
    other_branch_ids=None,
    q_matrix=None,
):
    target_set = set(int(v) for v in np.asarray(target_branch_ids, dtype=np.int64).reshape(-1).tolist())
    branch_ids = branch_meta["branch_id"].astype(int).to_numpy(copy=False)
    is_target = np.isin(branch_ids, np.array(sorted(target_set), dtype=np.int64))
    if other_branch_ids is None:
        other_set = set(int(v) for v in branch_ids.tolist()).difference(target_set)
    else:
        other_set = set(int(v) for v in np.asarray(other_branch_ids, dtype=np.int64).reshape(-1).tolist())
        other_set = other_set.difference(target_set)
    is_other = np.isin(branch_ids, np.array(sorted(other_set), dtype=np.int64))
    event_by_branch = candidate_events.groupby("branch_id")["event_pp"].sum().to_dict()
    event_values = np.array([float(event_by_branch.get(int(bid), 0.0)) for bid in branch_ids], dtype=np.float64)
    length_col = _length_column(rate_length)
    chosen_length = branch_meta[length_col].to_numpy(dtype=np.float64, copy=False)
    raw_length = branch_meta["raw_length"].to_numpy(dtype=np.float64, copy=False)
    sn_length = branch_meta["sn_rescaled_length"].to_numpy(dtype=np.float64, copy=False)
    n_length = branch_meta["n_rescaled_length"].to_numpy(dtype=np.float64, copy=False)
    if rate_exposure == "state_aware":
        opp_states = _opportunity_states(from_ids=from_ids, to_ids=to_ids)
        if opp_states.shape[0] == 0:
            opportunity = np.zeros(shape=branch_ids.shape[0], dtype=np.float64)
        else:
            parent_ids = branch_meta["parent_id"].astype(int).to_numpy(copy=False)
            opportunity = state_nsy[parent_ids, int(site), :][:, opp_states].sum(axis=1)
    elif rate_exposure == "raw_branch_length":
        opportunity = np.ones(shape=branch_ids.shape[0], dtype=np.float64)
    elif rate_exposure == "q_weighted":
        opportunity = _q_weighted_opportunity(
            branch_meta=branch_meta,
            state_nsy=state_nsy,
            site=site,
            from_ids=from_ids,
            to_ids=to_ids,
            q_matrix=q_matrix,
        )
    else:
        txt = "--scan_rate_exposure should be one of {}."
        raise ValueError(txt.format(", ".join(SCAN_RATE_EXPOSURES)))
    exposure = chosen_length * opportunity
    target_event = float(event_values[is_target].sum())
    other_event = float(event_values[is_other].sum())
    target_exposure = float(exposure[is_target].sum())
    other_exposure = float(exposure[is_other].sum())
    target_rate = target_event / target_exposure if target_exposure > 0 else np.nan
    other_rate = other_event / other_exposure if other_exposure > 0 else np.nan
    if np.isfinite(target_rate) and np.isfinite(other_rate):
        if other_rate == 0:
            rate_ratio = np.inf if target_rate > 0 else np.nan
        else:
            rate_ratio = target_rate / other_rate
    else:
        rate_ratio = np.nan
    pvalue = _poisson_lrt_pvalue(
        x_target=target_event,
        l_target=target_exposure,
        x_other=other_event,
        l_other=other_exposure,
    )
    return {
        "target_event_count": target_event,
        "target_event_branch_count": int((event_values[is_target] > 0).sum()),
        "target_raw_branch_length": float(raw_length[is_target].sum()),
        "target_sn_rescaled_length": float(sn_length[is_target].sum()),
        "target_n_rescaled_length": float(n_length[is_target].sum()),
        "target_exposure_branch_length": target_exposure,
        "other_event_count": other_event,
        "other_event_branch_count": int((event_values[is_other] > 0).sum()),
        "other_raw_branch_length": float(raw_length[is_other].sum()),
        "other_sn_rescaled_length": float(sn_length[is_other].sum()),
        "other_n_rescaled_length": float(n_length[is_other].sum()),
        "other_exposure_branch_length": other_exposure,
        "target_event_rate": target_rate,
        "other_event_rate": other_rate,
        "rate_ratio": rate_ratio,
        "p_rate_enrichment": pvalue,
    }


def _rank_quantiles(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(shape=values.shape, fill_value=np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    finite_values = values[finite]
    order = np.argsort(finite_values, kind="mergesort")
    ranks = np.empty(shape=finite_values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, finite_values.shape[0] + 1, dtype=np.float64)
    out[finite] = ranks / finite_values.shape[0]
    return out


def _calc_conservation(state, orders, tip_ids, float_tol):
    state = np.asarray(state)
    orders = np.asarray(orders, dtype=object).reshape(-1)
    tip_ids = np.asarray(tip_ids, dtype=np.int64).reshape(-1)
    num_site = int(state.shape[1])
    prefix = {
        "conservation": np.full(num_site, np.nan, dtype=np.float64),
        "entropy": np.full(num_site, np.nan, dtype=np.float64),
        "major_state": np.array([""] * num_site, dtype=object),
        "valid_tip_count": np.zeros(num_site, dtype=np.int64),
    }
    if tip_ids.shape[0] == 0:
        return prefix
    block = state[tip_ids, :, :]
    valid = block.sum(axis=2) > float(float_tol)
    prefix["valid_tip_count"] = valid.sum(axis=0).astype(np.int64, copy=False)
    totals = block.sum(axis=0)
    total_mass = totals.sum(axis=1)
    valid_sites = total_mass > float(float_tol)
    if valid_sites.any():
        probs = np.zeros_like(totals, dtype=np.float64)
        probs[valid_sites, :] = totals[valid_sites, :] / total_mass[valid_sites, None]
        major_idx = probs.argmax(axis=1)
        prefix["conservation"][valid_sites] = probs[valid_sites, major_idx[valid_sites]]
        prefix["major_state"][valid_sites] = orders[major_idx[valid_sites]].astype(str)
        entropy = np.zeros(num_site, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.where(probs > 0, np.log(probs), 0.0)
        entropy[valid_sites] = -np.sum(probs[valid_sites, :] * log_probs[valid_sites, :], axis=1)
        denom = math.log(max(orders.shape[0], 2))
        prefix["entropy"][valid_sites] = entropy[valid_sites] / denom
    return prefix


def build_site_annotations(g, trait_name, fg_leaf_names_map=None):
    num_site = int(g["state_nsy"].shape[1])
    site_index_alignment = parser_misc.get_site_index_alignment(g=g, expected_num_site=num_site)
    out = pd.DataFrame(
        {
            "site": np.arange(num_site, dtype=np.int64),
            "codon_site_alignment": site_index_alignment.astype(np.int64, copy=False) + 1,
        }
    )
    rate_values = g.get("iqtree_rate_values", None)
    if rate_values is None:
        out["site_rate"] = np.nan
        out["site_rate_quantile"] = np.nan
    else:
        rate_values = np.asarray(rate_values, dtype=np.float64).reshape(-1)
        if rate_values.shape[0] != num_site:
            txt = "iqtree_rate_values length ({}) did not match scan site axis ({})."
            raise ValueError(txt.format(rate_values.shape[0], num_site))
        out["site_rate"] = rate_values
        out["site_rate_quantile"] = _rank_quantiles(rate_values)
    tip_ids = []
    fg_tip_names = set()
    if fg_leaf_names_map is None:
        fg_leaf_names_map = g.get("fg_leaf_names", {})
    for values in fg_leaf_names_map.get(trait_name, []):
        fg_tip_names.update([str(v) for v in values])
    bg_tip_ids = []
    for leaf in ete.iter_leaves(g["tree"]):
        branch_id = int(ete.get_prop(leaf, "numerical_label"))
        tip_ids.append(branch_id)
        if str(leaf.name) not in fg_tip_names:
            bg_tip_ids.append(branch_id)
    float_tol = float(g.get("float_tol", 1e-12))
    state_orders = sequence.get_nonsyn_state_orders(g)
    aa_orders = np.asarray(g.get("amino_acid_orders", []), dtype=object)
    state_all = _calc_conservation(g["state_nsy"], state_orders, tip_ids, float_tol)
    state_bg = _calc_conservation(g["state_nsy"], state_orders, bg_tip_ids, float_tol)
    aa_all = _calc_conservation(g["state_pep"], aa_orders, tip_ids, float_tol)
    aa_bg = _calc_conservation(g["state_pep"], aa_orders, bg_tip_ids, float_tol)
    for label, values in [
        ("state_all", state_all),
        ("state_bg", state_bg),
        ("aa_all", aa_all),
        ("aa_bg", aa_bg),
    ]:
        out[label + "_conservation"] = values["conservation"]
        out[label + "_entropy"] = values["entropy"]
        out[label + "_major_state"] = values["major_state"]
        out[label + "_valid_tip_count"] = values["valid_tip_count"]
    return out


def _scan_substitutions_core(g, ON_tensor, rate_ON_tensor=None, scan_context=None):
    scan_matches = normalize_scan_matches(g.get("scan_match", "any2spe"))
    scan_targets = ["fg"]
    rate_length = str(g.get("scan_rate_length", "n_rescaled")).strip().lower()
    _length_column(rate_length)
    rate_exposure = normalize_scan_rate_exposure(g.get("scan_rate_exposure", "q_weighted"))
    rate_event_mode = normalize_scan_rate_event_mode(g.get("scan_rate_event_mode", "posterior_sum"))
    scan_other_scope = normalize_scan_other_scope(g.get("scan_other_scope", "all"))
    pvalue_calibration = normalize_scan_pvalue_calibration(g.get("scan_pvalue_calibration", "full_scan"))
    n_permutations = normalize_scan_n_permutations(g.get("scan_n_permutations", 1000))
    permutation_seed = int(g.get("scan_permutation_seed", 1))
    permutation_backend = _resolve_scan_permutation_backend(g)
    permutation_n_jobs = _resolve_scan_permutation_n_jobs(g=g, n_permutations=n_permutations)
    min_event_pp = float(g.get("scan_min_event_pp", 0.5))
    if (not np.isfinite(min_event_pp)) or (min_event_pp < 0) or (min_event_pp > 1):
        raise ValueError("--scan_min_event_pp should satisfy 0 <= value <= 1.")
    state_orders = sequence.get_nonsyn_state_orders(g)
    if rate_ON_tensor is None:
        rate_ON_tensor = ON_tensor
    q_matrix = None
    if rate_exposure == "q_weighted":
        q_matrix = g.get("instantaneous_nsy_rate_matrix", None)
        if q_matrix is None:
            raise ValueError("--scan_rate_exposure q_weighted requires instantaneous_nsy_rate_matrix.")
    events = extract_atomic_events(
        sub_tensor=ON_tensor,
        min_event_pp=min_event_pp,
        float_tol=float(g.get("float_tol", 1e-12)),
    )
    branch_meta = build_branch_metadata(g)
    valid_branch_ids = branch_meta["branch_id"].astype(int).to_numpy(copy=False)
    if scan_context is None:
        units = build_scan_units(g=g, branch_meta=branch_meta)
        fg_ids_map = g.get("fg_ids", {})
        mg_ids_map = g.get("mg_ids", {})
        fg_leaf_names_map = g.get("fg_leaf_names", {})
        trait_names = g["fg_df"].columns[1:].tolist()
    else:
        units = scan_context["units"].copy()
        fg_ids_map = scan_context["fg_ids"]
        mg_ids_map = scan_context["mg_ids"]
        fg_leaf_names_map = scan_context["fg_leaf_names"]
        trait_names = list(scan_context["trait_names"])
    if units.shape[0] == 0:
        raise ValueError("No foreground units were available for scan. Use --foreground.")
    rows = []
    scan_id = 0
    for trait_name in trait_names:
        units_trait = units.loc[units["trait"] == trait_name, :].reset_index(drop=True)
        candidate_branch_ids = _target_branch_ids_from_maps(
            fg_ids_map=fg_ids_map,
            mg_ids_map=mg_ids_map,
            trait_name=trait_name,
            target_class="fg",
            valid_branch_ids=valid_branch_ids,
        )
        candidate_events = events.loc[events["branch_id"].astype(int).isin(candidate_branch_ids.tolist()), :].copy()
        candidates = build_candidates(
            events=candidate_events,
            scan_matches=scan_matches,
            state_orders=state_orders,
        )
        if candidates.shape[0] == 0:
            continue
        min_support_count = parse_scan_support_threshold(
            g.get("scan_min_support", "2"),
            total_units=units_trait.shape[0],
        )
        if min_support_count > units_trait.shape[0]:
            if scan_context is None:
                txt = (
                    "Scan warning: --scan_min_support resolved to {} for trait {}, "
                    "but only {} foreground units are available; no candidates can pass."
                )
                print(txt.format(int(min_support_count), trait_name, int(units_trait.shape[0])), flush=True)
            continue
        site_annotations = build_site_annotations(
            g=g,
            trait_name=trait_name,
            fg_leaf_names_map=fg_leaf_names_map,
        )
        site_annotation_by_site = site_annotations.set_index("site", drop=False)
        for _, cand in candidates.iterrows():
            site = int(cand["site"])
            from_ids = np.array([int(v) for v in str(cand["from_state_ids"]).split(",") if v != ""], dtype=np.int64)
            to_ids = np.array([int(v) for v in str(cand["to_state_ids"]).split(",") if v != ""], dtype=np.int64)
            mask = _event_mask_for_candidate(
                events=events,
                site=site,
                from_ids=from_ids,
                to_ids=to_ids,
            )
            matched_events = events.loc[mask, :].copy()
            if rate_event_mode == "called":
                rate_events = matched_events
            elif rate_event_mode == "posterior_sum":
                rate_events = extract_candidate_posterior_events(
                    sub_tensor=rate_ON_tensor,
                    site=site,
                    from_ids=from_ids,
                    to_ids=to_ids,
                    float_tol=float(g.get("float_tol", 1e-12)),
                )
            else:
                txt = "--scan_rate_event_mode should be one of {}."
                raise ValueError(txt.format(", ".join(SCAN_RATE_EVENT_MODES)))
            candidate_support = _summarize_unit_support(
                candidate_events=matched_events,
                units_df=units_trait,
                target_class="fg",
                min_event_pp=min_event_pp,
            )
            if int(candidate_support["support_unit_count"]) < int(min_support_count):
                continue
            scan_id += 1
            site_row = site_annotation_by_site.loc[site, :].to_dict()
            site_label = int(site_row.get("codon_site_alignment", site + 1))
            state_change = _format_state_change(
                from_label=cand["from_state"],
                to_label=cand["to_state"],
                site_label=site_label,
                match=cand["scan_match"],
            )
            for target_class in scan_targets:
                target_branch_ids = _target_branch_ids_from_maps(
                    fg_ids_map=fg_ids_map,
                    mg_ids_map=mg_ids_map,
                    trait_name=trait_name,
                    target_class=target_class,
                    valid_branch_ids=valid_branch_ids,
                )
                other_branch_ids = _other_branch_ids_from_maps(
                    fg_ids_map=fg_ids_map,
                    mg_ids_map=mg_ids_map,
                    trait_name=trait_name,
                    target_class=target_class,
                    valid_branch_ids=valid_branch_ids,
                    scan_other_scope=scan_other_scope,
                    units_df=units_trait,
                )
                support = _summarize_unit_support(
                    candidate_events=matched_events,
                    units_df=units_trait,
                    target_class=target_class,
                    min_event_pp=min_event_pp,
                )
                rate = _rate_summary(
                    candidate_events=rate_events,
                    branch_meta=branch_meta,
                    state_nsy=np.asarray(g["state_nsy"]),
                    site=site,
                    from_ids=from_ids,
                    to_ids=to_ids,
                    target_branch_ids=target_branch_ids,
                    rate_length=rate_length,
                    rate_exposure=rate_exposure,
                    other_branch_ids=other_branch_ids,
                    q_matrix=q_matrix,
                )
                row = {
                    "scan_id": int(scan_id),
                    "trait": trait_name,
                    "target_class": target_class,
                    "scan_match": cand["scan_match"],
                    "site": int(site),
                    "nonsyn_recode": str(g.get("nonsyn_recode", "no")),
                    "from_state": cand["from_state"],
                    "to_state": cand["to_state"],
                    "from_state_ids": cand["from_state_ids"],
                    "to_state_ids": cand["to_state_ids"],
                    "observed_from_state_ids": cand["observed_from_state_ids"],
                    "observed_to_state_ids": cand["observed_to_state_ids"],
                    "from_state_distribution": cand["from_state_distribution"],
                    "to_state_distribution": cand["to_state_distribution"],
                    "state_change": state_change,
                    "candidate_event_pp_sum": float(cand["candidate_event_pp_sum"]),
                    "scan_min_support_count": int(min_support_count),
                    "scan_min_event_pp": float(min_event_pp),
                    "scan_rate_length_used": rate_length,
                    "scan_rate_exposure": rate_exposure,
                    "scan_rate_event_mode": rate_event_mode,
                    "scan_other_scope": scan_other_scope,
                    "scan_pvalue_calibration": pvalue_calibration,
                    "scan_n_permutations": int(n_permutations),
                    "scan_permutation_seed": int(permutation_seed),
                    "scan_permutation_backend": permutation_backend,
                    "scan_permutation_n_jobs": int(permutation_n_jobs),
                    "scan_permutation_success_count": 0,
                    "scan_permutation_failure_count": 0,
                    "p_rate_enrichment_empirical": np.nan,
                    "p_rate_enrichment_empirical_maxT": np.nan,
                }
                row.update(site_row)
                row.update(support)
                row.update(rate)
                rows.append(row)
    if len(rows) == 0:
        scan_df = pd.DataFrame(columns=list(SCAN_OUTPUT_COLUMNS))
    else:
        scan_df = pd.DataFrame(rows)
        sort_cols = [
            "trait",
            "scan_match",
            "site_rate_quantile",
            "target_class",
            "p_rate_enrichment",
            "support_fraction",
            "support_unit_count",
        ]
        for col in sort_cols:
            if col not in scan_df.columns:
                scan_df[col] = np.nan
        scan_df["q_rate_enrichment"] = _bh_qvalues(scan_df["p_rate_enrichment"].to_numpy(dtype=np.float64))
        scan_df = _assign_grouped_qvalues(
            scan_df=scan_df,
            out_col="q_rate_enrichment_by_trait",
            group_cols=["trait"],
        )
        scan_df = _assign_grouped_qvalues(
            scan_df=scan_df,
            out_col="q_rate_enrichment_by_trait_match",
            group_cols=["trait", "scan_match"],
        )
        scan_df = scan_df.sort_values(
            by=["trait", "scan_match", "p_rate_enrichment", "site_rate_quantile", "target_class"],
            ascending=[True, True, True, True, True],
            na_position="last",
        ).reset_index(drop=True)
        scan_df = scan_df.loc[:, [col for col in SCAN_OUTPUT_COLUMNS if col in scan_df.columns]]
    return scan_df, units


def _scan_permutation_seed(base_seed, permutation_index):
    modulus = np.iinfo(np.uint32).max
    return int((int(base_seed) + int(permutation_index)) % modulus)


def _scan_row_key(row):
    return (
        str(row["trait"]),
        str(row["target_class"]),
        str(row["scan_match"]),
        int(row["site"]),
        str(row["from_state_ids"]),
        str(row["to_state_ids"]),
    )


def _empirical_p_from_values(p_obs, values, denominator_count):
    p_obs = float(p_obs)
    denominator_count = int(denominator_count)
    if (not np.isfinite(p_obs)) or (denominator_count <= 0):
        return np.nan
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return min(1.0, float((1 + int((values <= p_obs).sum())) / (1 + denominator_count)))


def _build_permuted_context_with_seed(g, trait_names, valid_branch_ids, permutation_index):
    base_seed = int(g.get("scan_permutation_seed", 1))
    sample_original = bool(g.get("scan_permutation_sample_original", False))
    retry_with_original = bool(g.get("scan_permutation_retry_sample_original", True))
    with _SCAN_PERMUTATION_RANDOM_LOCK:
        random_state = np.random.get_state()
        np.random.seed(_scan_permutation_seed(base_seed=base_seed, permutation_index=permutation_index))
        try:
            try:
                return _build_permuted_scan_context(
                    g=g,
                    trait_names=trait_names,
                    valid_branch_ids=valid_branch_ids,
                    sample_original_foreground=sample_original,
                )
            except Exception:
                if sample_original or (not retry_with_original):
                    raise
                return _build_permuted_scan_context(
                    g=g,
                    trait_names=trait_names,
                    valid_branch_ids=valid_branch_ids,
                    sample_original_foreground=True,
                )
        finally:
            np.random.set_state(random_state)


def _candidate_fixed_permutation_pvalues(
    g,
    observed_df,
    scan_context,
    branch_meta,
    valid_branch_ids,
    ON_tensor,
    rate_ON_tensor,
):
    rate_length = str(g.get("scan_rate_length", "n_rescaled")).strip().lower()
    rate_exposure = normalize_scan_rate_exposure(g.get("scan_rate_exposure", "q_weighted"))
    rate_event_mode = normalize_scan_rate_event_mode(g.get("scan_rate_event_mode", "posterior_sum"))
    scan_other_scope = normalize_scan_other_scope(g.get("scan_other_scope", "all"))
    min_event_pp = float(g.get("scan_min_event_pp", 0.5))
    called_events = None
    if rate_event_mode == "called":
        called_events = extract_atomic_events(
            sub_tensor=ON_tensor,
            min_event_pp=min_event_pp,
            float_tol=float(g.get("float_tol", 1e-12)),
        )
    q_matrix = None
    if rate_exposure == "q_weighted":
        q_matrix = g.get("instantaneous_nsy_rate_matrix", None)
    out = {}
    for _, row in observed_df.iterrows():
        trait_name = str(row["trait"])
        target_class = str(row["target_class"])
        site = int(row["site"])
        from_ids = np.array([int(v) for v in str(row["from_state_ids"]).split(",") if v != ""], dtype=np.int64)
        to_ids = np.array([int(v) for v in str(row["to_state_ids"]).split(",") if v != ""], dtype=np.int64)
        if rate_event_mode == "called":
            mask = _event_mask_for_candidate(
                events=called_events,
                site=site,
                from_ids=from_ids,
                to_ids=to_ids,
            )
            rate_events = called_events.loc[mask, :].copy()
        else:
            rate_events = extract_candidate_posterior_events(
                sub_tensor=rate_ON_tensor,
                site=site,
                from_ids=from_ids,
                to_ids=to_ids,
                float_tol=float(g.get("float_tol", 1e-12)),
            )
        target_branch_ids = _target_branch_ids_from_maps(
            fg_ids_map=scan_context["fg_ids"],
            mg_ids_map=scan_context["mg_ids"],
            trait_name=trait_name,
            target_class=target_class,
            valid_branch_ids=valid_branch_ids,
        )
        units_trait = scan_context["units"].loc[scan_context["units"]["trait"] == trait_name, :].reset_index(drop=True)
        other_branch_ids = _other_branch_ids_from_maps(
            fg_ids_map=scan_context["fg_ids"],
            mg_ids_map=scan_context["mg_ids"],
            trait_name=trait_name,
            target_class=target_class,
            valid_branch_ids=valid_branch_ids,
            scan_other_scope=scan_other_scope,
            units_df=units_trait,
        )
        rate = _rate_summary(
            candidate_events=rate_events,
            branch_meta=branch_meta,
            state_nsy=np.asarray(g["state_nsy"]),
            site=site,
            from_ids=from_ids,
            to_ids=to_ids,
            target_branch_ids=target_branch_ids,
            rate_length=rate_length,
            rate_exposure=rate_exposure,
            other_branch_ids=other_branch_ids,
            q_matrix=q_matrix,
        )
        out[_scan_row_key(row)] = float(rate["p_rate_enrichment"])
    return out


def _run_scan_permutation(
    permutation_index,
    g,
    observed_df,
    observed_keys,
    calibration,
    trait_names,
    branch_meta,
    valid_branch_ids,
    ON_tensor,
    rate_ON_tensor,
):
    try:
        scan_context = _build_permuted_context_with_seed(
            g=g,
            trait_names=trait_names,
            valid_branch_ids=valid_branch_ids,
            permutation_index=permutation_index,
        )
        row_key_to_p = {}
        min_p = np.nan
        if calibration == "candidate_fixed":
            perm_by_key = _candidate_fixed_permutation_pvalues(
                g=g,
                observed_df=observed_df,
                scan_context=scan_context,
                branch_meta=branch_meta,
                valid_branch_ids=valid_branch_ids,
                ON_tensor=ON_tensor,
                rate_ON_tensor=rate_ON_tensor,
            )
            for key, pvalue in perm_by_key.items():
                if key in observed_keys:
                    row_key_to_p.setdefault(key, []).append(float(pvalue))
        else:
            perm_df, _ = _scan_substitutions_core(
                g=g,
                ON_tensor=ON_tensor,
                rate_ON_tensor=rate_ON_tensor,
                scan_context=scan_context,
            )
            if perm_df.shape[0] == 0:
                min_p = 1.0
            else:
                pvalues = perm_df["p_rate_enrichment"].to_numpy(dtype=np.float64)
                finite = pvalues[np.isfinite(pvalues)]
                min_p = float(finite.min()) if finite.shape[0] > 0 else 1.0
                for _, perm_row in perm_df.iterrows():
                    key = _scan_row_key(perm_row)
                    if key in observed_keys:
                        row_key_to_p.setdefault(key, []).append(float(perm_row["p_rate_enrichment"]))
        return {
            "success": True,
            "min_p": min_p,
            "row_key_to_p": row_key_to_p,
        }
    except Exception:
        return {
            "success": False,
            "min_p": np.nan,
            "row_key_to_p": {},
        }


def _run_scan_permutation_chunk(
    permutation_indices,
    g,
    observed_df,
    observed_keys,
    calibration,
    trait_names,
    branch_meta,
    valid_branch_ids,
    ON_tensor,
    rate_ON_tensor,
):
    return [
        _run_scan_permutation(
            permutation_index=int(permutation_index),
            g=g,
            observed_df=observed_df,
            observed_keys=observed_keys,
            calibration=calibration,
            trait_names=trait_names,
            branch_meta=branch_meta,
            valid_branch_ids=valid_branch_ids,
            ON_tensor=ON_tensor,
            rate_ON_tensor=rate_ON_tensor,
        )
        for permutation_index in np.asarray(permutation_indices, dtype=np.int64).reshape(-1).tolist()
    ]


def _calibrate_scan_pvalues(g, observed_df, ON_tensor, rate_ON_tensor):
    calibration = normalize_scan_pvalue_calibration(g.get("scan_pvalue_calibration", "full_scan"))
    if calibration == "none":
        return observed_df
    n_permutations = normalize_scan_n_permutations(g.get("scan_n_permutations", 1000))
    if n_permutations == 0 or observed_df.shape[0] == 0:
        return observed_df
    branch_meta = build_branch_metadata(g)
    valid_branch_ids = branch_meta["branch_id"].astype(int).to_numpy(copy=False)
    trait_names = g["fg_df"].columns[1:].tolist()
    row_key_to_perm_p = {_scan_row_key(row): [] for _, row in observed_df.iterrows()}
    observed_keys = set(row_key_to_perm_p.keys())
    min_perm_p = []
    n_jobs = _resolve_scan_permutation_n_jobs(g=g, n_permutations=n_permutations)
    backend = _resolve_scan_permutation_backend(g=g)
    permutation_indices = np.arange(1, n_permutations + 1, dtype=np.int64)
    chunk_factor = parallel.resolve_chunk_factor(g=g, task="general")
    permutation_chunks, _ = parallel.get_chunks(
        input_data=permutation_indices,
        threads=n_jobs,
        chunk_factor=chunk_factor,
    )
    permutation_args = [
        (
            permutation_chunk,
            g,
            observed_df,
            observed_keys,
            calibration,
            trait_names,
            branch_meta,
            valid_branch_ids,
            ON_tensor,
            rate_ON_tensor,
        )
        for permutation_chunk in permutation_chunks
    ]
    chunk_results = parallel.run_starmap(
        _run_scan_permutation_chunk,
        permutation_args,
        n_jobs=n_jobs,
        backend=backend,
    )
    results = [result for chunk_result in chunk_results for result in chunk_result]
    success_count = 0
    failure_count = 0
    for result in results:
        if not result.get("success", False):
            failure_count += 1
            continue
        success_count += 1
        if calibration == "full_scan":
            min_p = float(result.get("min_p", np.nan))
            min_perm_p.append(min_p if np.isfinite(min_p) else 1.0)
        for key, pvalues in result.get("row_key_to_p", {}).items():
            if key in row_key_to_perm_p:
                row_key_to_perm_p[key].extend(float(v) for v in pvalues)
    observed_df = observed_df.copy()
    observed_df.loc[:, "scan_permutation_backend"] = backend
    observed_df.loc[:, "scan_permutation_n_jobs"] = int(n_jobs)
    observed_df.loc[:, "scan_permutation_success_count"] = int(success_count)
    observed_df.loc[:, "scan_permutation_failure_count"] = int(failure_count)
    if success_count == 0:
        return observed_df
    empirical = []
    empirical_maxT = []
    for _, row in observed_df.iterrows():
        key = _scan_row_key(row)
        p_obs = float(row["p_rate_enrichment"])
        empirical.append(
            _empirical_p_from_values(
                p_obs=p_obs,
                values=row_key_to_perm_p.get(key, []),
                denominator_count=success_count,
            )
        )
        if calibration == "full_scan":
            empirical_maxT.append(
                _empirical_p_from_values(
                    p_obs=p_obs,
                    values=min_perm_p,
                    denominator_count=success_count,
                )
            )
        else:
            empirical_maxT.append(np.nan)
    observed_df.loc[:, "p_rate_enrichment_empirical"] = empirical
    observed_df.loc[:, "p_rate_enrichment_empirical_maxT"] = empirical_maxT
    return observed_df


def scan_substitutions(g, ON_tensor, rate_ON_tensor=None):
    scan_df, units = _scan_substitutions_core(g=g, ON_tensor=ON_tensor, rate_ON_tensor=rate_ON_tensor)
    scan_df = _calibrate_scan_pvalues(
        g=g,
        observed_df=scan_df,
        ON_tensor=ON_tensor,
        rate_ON_tensor=rate_ON_tensor if rate_ON_tensor is not None else ON_tensor,
    )
    if scan_df.shape[0] > 0:
        scan_df = scan_df.loc[:, [col for col in SCAN_OUTPUT_COLUMNS if col in scan_df.columns]]
    return scan_df, units

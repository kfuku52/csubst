import hashlib

import numpy as np
import pandas as pd

from csubst import ete
from csubst import parser_misc
from csubst import sequence


EVENT_COLUMNS = (
    "event_id",
    "branch_id",
    "parent_branch_id",
    "site",
    "codon_site_alignment",
    "aa_position_ancestral",
    "from_aa",
    "to_aa",
    "state_change",
    "from_state_pp",
    "to_state_pp",
    "event_pp",
    "vesm_llr",
    "score_status",
    "context_length_aa",
    "window_start_aa",
    "window_end_aa",
    "context_sha256",
    "vesm_model_resource_id",
    "vep_min_event_pp",
)


def is_enabled(g):
    return str(g.get("vep_model", "none")).strip().lower() != "none"


def empty_event_table():
    return pd.DataFrame(columns=list(EVENT_COLUMNS) + ["_context_sequence"])


def _validate_full_codon_alignment(g, state_pep):
    alignment_path = str(g.get("alignment_file", "")).strip()
    if alignment_path == "":
        raise ValueError("--vep_model requires --alignment_file to be a full-length codon alignment.")
    seq_by_name = sequence.read_fasta(alignment_path)
    if len(seq_by_name) == 0:
        raise ValueError("The VEP alignment is empty: {}".format(alignment_path))
    lengths = sorted(set(len(str(value)) for value in seq_by_name.values()))
    if len(lengths) != 1:
        raise ValueError("VESM requires an aligned full-length CDS FASTA with equal sequence lengths.")
    alignment_length = int(lengths[0])
    if alignment_length % 3 != 0:
        raise ValueError("VESM full-length CDS alignment length should be a multiple of 3.")
    num_site = alignment_length // 3
    if num_site != int(state_pep.shape[1]):
        txt = "VESM alignment codon count ({}) did not match the unfiltered ancestral-state site count ({})."
        raise ValueError(txt.format(num_site, int(state_pep.shape[1])))
    tip_names = [str(node.name) for node in ete.iter_leaves(g["tree"])]
    missing = sorted(name for name in tip_names if name not in seq_by_name)
    if len(missing) > 0:
        missing_txt = ",".join(missing[:10]) + (",..." if len(missing) > 10 else "")
        raise ValueError("Tree tip sequence(s) missing in VEP alignment: {}".format(missing_txt))
    presence_by_tip = {}
    for name in tip_names:
        seq_txt = str(seq_by_name[name]).strip().upper()
        presence = np.zeros(num_site, dtype=bool)
        for site in range(num_site):
            codon = seq_txt[(site * 3) : ((site + 1) * 3)]
            if ("-" in codon) and (codon != "---"):
                txt = "Partial codon gap in VEP alignment: sequence={}, codon_site_alignment={}."
                raise ValueError(txt.format(name, site + 1))
            presence[site] = codon != "---"
        presence_by_tip[name] = presence
    return presence_by_tip, num_site


def infer_ancestral_gap_presence(tree_obj, presence_by_tip, num_node, num_site):
    """Infer codon presence/absence with Fitch parsimony and deterministic tie breaking."""
    fitch_mask = {}
    present_count = {}
    tip_count = {}

    def postorder(node):
        node_key = id(node)
        if ete.is_leaf(node):
            if str(node.name) not in presence_by_tip:
                raise ValueError("Missing tip gap mask: {}".format(node.name))
            present = np.asarray(presence_by_tip[str(node.name)], dtype=bool)
            fitch_mask[node_key] = np.where(present, 2, 1).astype(np.uint8, copy=False)
            present_count[node_key] = present.astype(np.int32, copy=False)
            tip_count[node_key] = np.ones(num_site, dtype=np.int32)
            return
        children = list(ete.get_children(node))
        if len(children) == 0:
            raise ValueError("Internal VEP gap-reconstruction node had no children.")
        for child in children:
            postorder(child)
        child_masks = [fitch_mask[id(child)] for child in children]
        intersection = child_masks[0].copy()
        union = child_masks[0].copy()
        for child_mask in child_masks[1:]:
            intersection = np.bitwise_and(intersection, child_mask)
            union = np.bitwise_or(union, child_mask)
        fitch_mask[node_key] = np.where(intersection != 0, intersection, union).astype(np.uint8, copy=False)
        present_count[node_key] = sum((present_count[id(child)] for child in children), np.zeros(num_site, dtype=np.int32))
        tip_count[node_key] = sum((tip_count[id(child)] for child in children), np.zeros(num_site, dtype=np.int32))

    postorder(tree_obj)
    presence = np.zeros((int(num_node), int(num_site)), dtype=bool)

    def preorder(node, parent_state=None):
        node_key = id(node)
        mask = fitch_mask[node_key]
        state = np.zeros(num_site, dtype=bool)
        state[mask == 2] = True
        ambiguous = mask == 3
        if ambiguous.any():
            if parent_state is None:
                # Majority-tip tie breaking behaves sensibly for both clade insertions and deletions.
                state[ambiguous] = (
                    present_count[node_key][ambiguous] * 2 >= tip_count[node_key][ambiguous]
                )
            else:
                state[ambiguous] = parent_state[ambiguous]
        branch_id = int(ete.get_prop(node, "numerical_label"))
        if (branch_id < 0) or (branch_id >= int(num_node)):
            raise ValueError("Tree numerical_label is outside the ancestral-state tensor: {}".format(branch_id))
        presence[branch_id, :] = state
        for child in ete.get_children(node):
            preorder(child, parent_state=state)

    preorder(tree_obj)
    return presence


def prepare_ancestral_contexts(g):
    state_pep = np.asarray(g.get("state_pep", None))
    if state_pep.ndim != 3:
        raise ValueError("state_pep is required before preparing VESM ancestral contexts.")
    aa_orders = np.asarray(g.get("amino_acid_orders", []), dtype="U1").reshape(-1)
    if aa_orders.shape[0] != state_pep.shape[2]:
        raise ValueError("amino_acid_orders did not match state_pep for VESM context construction.")
    presence_by_tip, num_site = _validate_full_codon_alignment(g=g, state_pep=state_pep)
    presence = infer_ancestral_gap_presence(
        tree_obj=g["tree"],
        presence_by_tip=presence_by_tip,
        num_node=state_pep.shape[0],
        num_site=num_site,
    )
    state_values = np.nan_to_num(state_pep, nan=0.0)
    site_argmax = state_values.argmax(axis=2)
    site_max = state_values.max(axis=2)
    float_tol = float(g.get("float_tol", 1e-12))
    contexts = {}
    for node in g["tree"].traverse():
        branch_id = int(ete.get_prop(node, "numerical_label"))
        aligned_symbols = aa_orders[site_argmax[branch_id, :]].copy()
        aligned_symbols[(presence[branch_id, :]) & (site_max[branch_id, :] <= float_tol)] = "X"
        aligned_symbols[~presence[branch_id, :]] = "-"
        mapping = np.full(num_site, -1, dtype=np.int64)
        nongap_sites = np.flatnonzero(presence[branch_id, :])
        mapping[nongap_sites] = np.arange(nongap_sites.shape[0], dtype=np.int64)
        ungapped = "".join(aligned_symbols[presence[branch_id, :]].tolist())
        contexts[branch_id] = {
            "sequence": ungapped,
            "aligned_sequence": "".join(aligned_symbols.tolist()),
            "alignment_to_ungapped": mapping,
            "presence": presence[branch_id, :].copy(),
        }
    bundle = {
        "contexts": contexts,
        "presence": presence,
        "num_alignment_site": int(num_site),
        "gap_method": "fitch_tip_majority_tiebreak_v1",
    }
    g["_vep_ancestral_contexts"] = bundle
    return bundle


def _effective_parent_ids(g, branch_ids):
    state_pep = np.nan_to_num(np.asarray(g["state_pep"]), nan=0.0)
    state_has_mass = state_pep.sum(axis=(1, 2)) > float(g.get("float_tol", 1e-12))
    branch_set = set(int(value) for value in np.asarray(branch_ids, dtype=np.int64).reshape(-1).tolist())
    out = {}
    for node in g["tree"].traverse():
        branch_id = int(ete.get_prop(node, "numerical_label"))
        if branch_id not in branch_set:
            continue
        parent_node = ete.get_effective_state_parent(node, state_has_mass=state_has_mass)
        if parent_node is not None:
            out[branch_id] = int(ete.get_prop(parent_node, "numerical_label"))
    return out


def extract_atomic_aa_events(g, branch_ids):
    if "_vep_ancestral_contexts" not in g:
        raise ValueError("VESM ancestral contexts were not prepared before event extraction.")
    state_pep = np.nan_to_num(np.asarray(g["state_pep"]), nan=0.0)
    aa_orders = np.asarray(g["amino_acid_orders"], dtype=object).reshape(-1)
    site_index_alignment = parser_misc.get_site_index_alignment(
        g=g,
        expected_num_site=state_pep.shape[1],
    )
    branch_ids = np.asarray(branch_ids, dtype=np.int64).reshape(-1)
    parent_by_branch = _effective_parent_ids(g=g, branch_ids=branch_ids)
    bundle = g["_vep_ancestral_contexts"]
    threshold = float(g.get("vep_min_event_pp", 0.8))
    float_tol = float(g.get("float_tol", 1e-12))
    rows = []
    for branch_id in branch_ids.tolist():
        parent_id = parent_by_branch.get(int(branch_id), None)
        if parent_id is None:
            continue
        parent_context = bundle["contexts"].get(int(parent_id), None)
        child_context = bundle["contexts"].get(int(branch_id), None)
        if (parent_context is None) or (child_context is None):
            continue
        joint = state_pep[parent_id, :, :, np.newaxis] * state_pep[branch_id, :, np.newaxis, :]
        diagonal = np.arange(aa_orders.shape[0], dtype=np.int64)
        joint[:, diagonal, diagonal] = 0.0
        site_ids, from_ids, to_ids = np.where((joint >= threshold) & (joint > float_tol))
        for site, from_id, to_id in zip(site_ids.tolist(), from_ids.tolist(), to_ids.tolist()):
            alignment_site = int(site_index_alignment[int(site)])
            if alignment_site >= int(bundle["num_alignment_site"]):
                continue
            if not (
                bool(parent_context["presence"][alignment_site])
                and bool(child_context["presence"][alignment_site])
            ):
                continue
            aa_position = int(parent_context["alignment_to_ungapped"][alignment_site])
            if aa_position < 0:
                continue
            from_aa = str(aa_orders[int(from_id)])
            to_aa = str(aa_orders[int(to_id)])
            context_chars = list(parent_context["sequence"])
            if aa_position >= len(context_chars):
                continue
            context_chars[aa_position] = from_aa
            context_sequence = "".join(context_chars)
            context_sha256 = hashlib.sha256(context_sequence.encode("ascii")).hexdigest()
            rows.append(
                {
                    "branch_id": int(branch_id),
                    "parent_branch_id": int(parent_id),
                    "site": int(site),
                    "codon_site_alignment": alignment_site + 1,
                    "aa_position_ancestral": aa_position + 1,
                    "from_aa": from_aa,
                    "to_aa": to_aa,
                    "state_change": "{}{}{}".format(from_aa, alignment_site + 1, to_aa),
                    "from_state_pp": float(state_pep[parent_id, site, from_id]),
                    "to_state_pp": float(state_pep[branch_id, site, to_id]),
                    "event_pp": float(joint[site, from_id, to_id]),
                    "vesm_llr": np.nan,
                    "score_status": "pending",
                    "context_length_aa": len(context_sequence),
                    "window_start_aa": 0,
                    "window_end_aa": 0,
                    "context_sha256": context_sha256,
                    "vesm_model_resource_id": "",
                    "vep_min_event_pp": threshold,
                    "_context_sequence": context_sequence,
                }
            )
    if len(rows) == 0:
        return empty_event_table()
    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["branch_id", "site", "from_aa", "to_aa"],
        kind="mergesort",
    ).reset_index(drop=True)
    out.insert(
        0,
        "event_id",
        [
            "b{}.a{}.{}>{}".format(row.branch_id, row.codon_site_alignment, row.from_aa, row.to_aa)
            for row in out.itertuples(index=False)
        ],
    )
    return out.loc[:, list(EVENT_COLUMNS) + ["_context_sequence"]]


def attach_scores_to_site_table(df, events, branch_ids, g):
    out = df.copy(deep=True)
    branch_ids = np.asarray(branch_ids, dtype=np.int64).reshape(-1)
    finite_events = events.loc[np.isfinite(pd.to_numeric(events.get("vesm_llr", np.nan), errors="coerce")), :].copy()
    for branch_id in branch_ids.tolist():
        prefix = "vesm_{}_".format(int(branch_id))
        out.loc[:, prefix + "state_change"] = ""
        out.loc[:, prefix + "event_pp"] = np.nan
        out.loc[:, prefix + "llr"] = np.nan
        out.loc[:, prefix + "event_count"] = 0
        branch_events = finite_events.loc[finite_events["branch_id"].astype(int) == int(branch_id), :]
        for site, group in branch_events.groupby("site", sort=True):
            site = int(site)
            if (site < 0) or (site >= out.shape[0]):
                continue
            ranked = group.sort_values(
                by=["event_pp", "vesm_llr", "event_id"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            representative = ranked.iloc[0]
            out.at[site, prefix + "state_change"] = str(representative["state_change"])
            out.at[site, prefix + "event_pp"] = float(representative["event_pp"])
            out.at[site, prefix + "llr"] = float(representative["vesm_llr"])
            out.at[site, prefix + "event_count"] = int(group.shape[0])

    out.loc[:, "vesm_structure_llr"] = np.nan
    out.loc[:, "vesm_structure_event_id"] = ""
    out.loc[:, "vesm_structure_branch_id"] = -1
    out.loc[:, "vesm_structure_event_count"] = 0
    aggregate = str(g.get("vep_site_aggregate", "most_deleterious"))
    for site, group in finite_events.groupby("site", sort=True):
        site = int(site)
        if (site < 0) or (site >= out.shape[0]):
            continue
        scores = group["vesm_llr"].to_numpy(dtype=float, copy=False)
        if aggregate == "most_deleterious":
            representative = group.sort_values(
                by=["vesm_llr", "event_pp", "event_id"],
                ascending=[True, False, True],
                kind="mergesort",
            ).iloc[0]
            value = float(representative["vesm_llr"])
            event_id = str(representative["event_id"])
            branch_id = int(representative["branch_id"])
        elif aggregate == "pp_weighted_mean":
            weights = group["event_pp"].to_numpy(dtype=float, copy=False)
            value = float(np.average(scores, weights=weights)) if float(weights.sum()) > 0 else float(scores.mean())
            event_id = ""
            branch_id = -1
        elif aggregate == "mean":
            value = float(scores.mean())
            event_id = ""
            branch_id = -1
        else:  # validated by param.get_global_parameters
            raise ValueError("Unsupported --vep_site_aggregate: {}".format(aggregate))
        out.at[site, "vesm_structure_llr"] = value
        out.at[site, "vesm_structure_event_id"] = event_id
        out.at[site, "vesm_structure_branch_id"] = branch_id
        out.at[site, "vesm_structure_event_count"] = int(group.shape[0])
    finite_scores = out["vesm_structure_llr"].to_numpy(dtype=float, copy=False)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    g["_vep_color_limit"] = max(float(np.max(np.abs(finite_scores))) if finite_scores.size else 0.0, 1e-6)
    return out


def add_structure_coordinates_to_events(events, site_df):
    if events.shape[0] == 0:
        return events.drop(columns=["_context_sequence"], errors="ignore").copy()
    coordinate_cols = [
        col
        for col in site_df.columns
        if str(col).startswith("codon_site_") and col != "codon_site_alignment"
    ]
    coordinate_table = site_df.loc[:, ["codon_site_alignment"] + coordinate_cols].copy()
    coordinate_table = coordinate_table.drop_duplicates(subset=["codon_site_alignment"], keep="first")
    out = events.drop(columns=["_context_sequence"], errors="ignore").merge(
        coordinate_table,
        how="left",
        on="codon_site_alignment",
    )
    return out

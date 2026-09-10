import numpy as np
import pandas as pd

from csubst import ete, tree


def _set_state(state, branch_id, site, state_id):
    state[int(branch_id), int(site), :] = 0.0
    state[int(branch_id), int(site), int(state_id)] = 1.0


def make_scan_context():
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("((A:1,B:1)X:1,(C:1,D:1)Y:1)R;", format=1)
    )
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    num_node = max(labels.values()) + 1
    for node in tr.traverse():
        ete.set_prop(node, "SNdist", 0.25)
        ete.set_prop(node, "Ndist", 0.10)
    fg_leaf_names = {"trait": [["A"], ["C"]]}
    for i, names in enumerate(fg_leaf_names["trait"], start=1):
        name_set = set(names)
        for node in tr.traverse():
            node_leaf_names = set(ete.get_leaf_names(node))
            ete.add_features(
                node,
                **{"is_lineage_fg_trait_{}".format(i): node_leaf_names.issubset(name_set)},
            )
    for node in tr.traverse():
        ete.add_features(node, is_fg_trait=False)
    state_nsy = np.zeros((num_node, 1, 2), dtype=float)
    state_pep = np.zeros((num_node, 1, 2), dtype=float)
    for node_id in labels.values():
        _set_state(state_nsy, node_id, 0, 0)
        _set_state(state_pep, node_id, 0, 0)
    for name in ["A", "C"]:
        _set_state(state_nsy, labels[name], 0, 1)
        _set_state(state_pep, labels[name], 0, 1)
        ete.add_features(next(node for node in tr.traverse() if node.name == name), is_fg_trait=True)
    on_tensor = np.zeros((num_node, 1, 1, 2, 2), dtype=float)
    on_tensor[labels["A"], 0, 0, 0, 1] = 0.9
    on_tensor[labels["C"], 0, 0, 0, 1] = 0.8
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"name": ["A", "C"], "trait": [1, 2]}),
        "fg_leaf_names": fg_leaf_names,
        "fg_ids": {"trait": np.array([labels["A"], labels["C"]], dtype=np.int64)},
        "fg_stem_only": True,
        "scan_sister_stem_only": True,
        "state_nsy": state_nsy,
        "state_pep": state_pep,
        "nonsyn_state_orders": np.array(["A", "K"], dtype=object),
        "amino_acid_orders": np.array(["A", "K"], dtype=object),
        "iqtree_rate_values": np.array([0.25], dtype=float),
        "float_tol": 1e-12,
        "nonsyn_recode": "no",
        "scan_match": "any2spe",
        "scan_min_event_pp": 0.5,
        "scan_min_support": "2",
        "scan_rate_length": "raw",
        "scan_rate_exposure": "state_aware",
        "scan_rate_event_mode": "posterior_sum",
        "scan_other_scope": "all",
        "scan_pvalue_calibration": "none",
        "scan_n_permutations": 0,
        "scan_permutation_seed": 1,
        "scan_permutation_sample_original": True,
        "scan_permutation_retry_sample_original": False,
        "min_clade_bin_count": 1,
    }
    return g, on_tensor

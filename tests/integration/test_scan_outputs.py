import os

import numpy as np
import pandas as pd
import pytest

from csubst import ete
from csubst import main_scan
from csubst import substitution_sparse
from csubst import substitution_scan
from csubst import tree


def _set_state(state, branch_id, site, state_id):
    state[int(branch_id), int(site), :] = 0.0
    state[int(branch_id), int(site), int(state_id)] = 1.0


def _toy_scan_context():
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
        "scan_permutation_sample_original": False,
        "scan_permutation_retry_sample_original": True,
        "min_clade_bin_count": 1,
    }
    return g, on_tensor


def test_scan_state_change_uses_alignment_site_coordinate():
    g, on_tensor = _toy_scan_context()
    g["site_index_alignment"] = np.array([9], dtype=np.int64)

    scan_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    row = scan_df.iloc[0]
    assert row["site"] == 0
    assert row["codon_site_alignment"] == 10
    assert row["state_change"] == "10K"


def test_scan_substitutions_supports_sparse_substitution_tensor_end_to_end():
    g, on_tensor = _toy_scan_context()
    sparse_tensor = substitution_sparse.SparseSubstitutionTensor.from_dense(on_tensor)

    scan_df, units = substitution_scan.scan_substitutions(
        g=g,
        ON_tensor=sparse_tensor,
        rate_ON_tensor=sparse_tensor,
    )

    assert units.shape[0] == 2
    assert scan_df.shape[0] == 1
    assert scan_df.loc[scan_df["target_class"] == "fg", "target_event_count"].iloc[0] == pytest.approx(1.7)


def test_scan_worker_tensor_descriptor_reopens_memmap_without_serializing_data(tmp_path):
    path = tmp_path / "tensor.mmap"
    tensor = np.memmap(path, dtype=np.float64, mode="w+", shape=(2, 3))
    tensor[:, :] = np.arange(6, dtype=float).reshape(2, 3)

    packed = substitution_scan._pack_scan_tensor_for_worker(tensor)
    reopened = substitution_scan._unpack_scan_tensor_for_worker(packed)

    assert packed["__scan_memmap__"] is True
    assert isinstance(reopened, np.memmap)
    assert reopened.mode == "r"
    np.testing.assert_allclose(reopened, tensor)


def test_scan_worker_context_memmaps_large_state_arrays_and_drops_unused_states():
    state_nsy = np.ones((128, 128, 16), dtype=np.float64)
    state_cdn = np.ones((128, 128, 16), dtype=np.float64)
    g = {
        "state_nsy": state_nsy,
        "state_cdn": state_cdn,
        "state_pep": np.ones((128, 128, 16), dtype=np.float64),
        "state_nuc": np.ones((1,), dtype=np.float64),
    }
    scan_static = {
        "q_context": {
            "q_matrix": None,
            "state_cdn": state_cdn,
            "codon_q_matrix": np.eye(16),
            "codon_state_ids": np.arange(16),
        },
        "observed_site_annotations": {"trait": pd.DataFrame({"site": [0]})},
    }

    worker_g, worker_static, owned_paths = substitution_scan._pack_scan_worker_context(
        g=g,
        scan_static=scan_static,
    )
    try:
        assert worker_g["state_nsy"]["__scan_memmap__"] is True
        assert worker_g["state_cdn"]["__scan_memmap__"] is True
        assert "state_pep" not in worker_g
        assert "state_nuc" not in worker_g
        assert worker_static["observed_site_annotations"] == {}
        unpacked_g, unpacked_static = substitution_scan._unpack_scan_worker_context(
            g=worker_g,
            scan_static=worker_static,
        )
        assert isinstance(unpacked_g["state_nsy"], np.memmap)
        assert isinstance(unpacked_g["state_cdn"], np.memmap)
        assert unpacked_static["q_context"]["state_cdn"] is unpacked_g["state_cdn"]
        np.testing.assert_allclose(unpacked_g["state_nsy"], state_nsy)
        np.testing.assert_allclose(unpacked_g["state_cdn"], state_cdn)
    finally:
        for path in owned_paths:
            if os.path.exists(path):
                os.remove(path)


def test_scan_posterior_sum_uses_unthresholded_rate_tensor_when_called_tensor_is_thresholded():
    g, raw_tensor = _toy_scan_context()
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in g["tree"].traverse()}
    raw_tensor[labels["B"], 0, 0, 0, 1] = 0.2
    called_tensor = raw_tensor.copy()
    called_tensor[called_tensor < 0.5] = 0.0
    g["scan_rate_event_mode"] = "posterior_sum"

    scan_df, _ = substitution_scan.scan_substitutions(
        g=g,
        ON_tensor=called_tensor,
        rate_ON_tensor=raw_tensor,
    )

    assert scan_df.iloc[0]["target_event_count"] == pytest.approx(1.7)
    assert scan_df.iloc[0]["other_event_count"] == pytest.approx(0.2)


def test_scan_substitutions_handles_multiple_traits_and_stratified_qvalues():
    g, on_tensor = _toy_scan_context()
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in g["tree"].traverse()}
    g["fg_df"]["trait2"] = [1, 2]
    g["fg_leaf_names"]["trait2"] = [["A"], ["C"]]
    g["fg_ids"]["trait2"] = np.array([labels["A"], labels["C"]], dtype=np.int64)
    for i, names in enumerate(g["fg_leaf_names"]["trait2"], start=1):
        name_set = set(names)
        for node in g["tree"].traverse():
            node_leaf_names = set(ete.get_leaf_names(node))
            ete.add_features(
                node,
                **{"is_lineage_fg_trait2_{}".format(i): node_leaf_names.issubset(name_set)},
            )
            ete.add_features(node, is_fg_trait2=node.name in {"A", "C"})

    scan_df, units = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    assert set(scan_df["trait"].tolist()) == {"trait", "trait2"}
    assert units.groupby("trait").size().to_dict() == {"trait": 2, "trait2": 2}
    assert scan_df.groupby("trait").size().to_dict() == {"trait": 1, "trait2": 1}
    q_cols = [
        "q_rate_enrichment",
        "q_rate_enrichment_by_trait",
        "q_rate_enrichment_by_trait_match",
    ]
    for col in q_cols:
        assert col in scan_df.columns
        assert np.isfinite(scan_df[col].to_numpy(dtype=float)).all()
    assert np.allclose(
        scan_df["q_rate_enrichment_by_trait_match"].to_numpy(dtype=float),
        scan_df["p_rate_enrichment"].to_numpy(dtype=float),
    )


def test_scan_full_scan_permutation_adds_empirical_maxt_pvalues():
    g, on_tensor = _toy_scan_context()
    g["scan_pvalue_calibration"] = "full_scan"
    g["scan_n_permutations"] = 4
    g["scan_permutation_seed"] = 3

    scan_df, _ = substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    assert scan_df.shape[0] == 1
    row = scan_df.iloc[0]
    assert row["scan_pvalue_calibration"] == "full_scan"
    assert row["scan_permutation_success_count"] == 4
    assert row["scan_permutation_failure_count"] == 0
    assert row["scan_permutation_failure_reasons"] == ""
    assert np.isfinite(float(row["p_rate_enrichment_empirical_maxT"]))
    assert 0 < float(row["p_rate_enrichment_empirical_maxT"]) <= 1
    assert row["q_rate_enrichment_empirical"] == pytest.approx(
        row["p_rate_enrichment_empirical"]
    )


def test_full_scan_reuses_static_atomic_events_across_permutations(monkeypatch):
    g, on_tensor = _toy_scan_context()
    g["scan_pvalue_calibration"] = "full_scan"
    g["scan_n_permutations"] = 4
    calls = []
    original = substitution_scan.extract_atomic_events

    def counting_extract(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(substitution_scan, "extract_atomic_events", counting_extract)

    substitution_scan.scan_substitutions(g=g, ON_tensor=on_tensor)

    assert len(calls) == 1


def test_prepare_scan_output_table_formats_only_p_and_q_values_scientifically():
    df = pd.DataFrame(
        {
            "target_event_count": [0.000002],
            "candidate_event_pp_sum": [0.000002],
            "p_rate_enrichment": [0.000002],
            "q_rate_enrichment": [0.00015577],
            "p_rate_enrichment_empirical": [np.nan],
        }
    )

    out = main_scan._prepare_scan_output_table(df)

    assert out.loc[0, "p_rate_enrichment"] == "2.000000e-06"
    assert out.loc[0, "q_rate_enrichment"] == "1.557700e-04"
    assert out.loc[0, "p_rate_enrichment_empirical"] == ""
    assert out.loc[0, "target_event_count"] == pytest.approx(0.000002)
    assert out.loc[0, "candidate_event_pp_sum"] == pytest.approx(0.000002)

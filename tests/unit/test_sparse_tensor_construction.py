from sparse_fixtures import toy_dense_tensor as _toy_dense_tensor
from sparse_fixtures import toy_reducer_tensor as _toy_reducer_tensor

import numpy as np

from csubst import substitution
from csubst import substitution_sparse
from csubst import tree
from csubst import ete






def test_get_substitution_tensor_sparse_asis_matches_manual_values():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    state = np.zeros((3, 2, 2), dtype=float)
    state[labels["R"], :, :] = [[1.0, 0.0], [0.5, 0.5]]
    state[labels["A"], :, :] = [[0.0, 1.0], [1.0, 0.0]]
    state[labels["B"], :, :] = [[1.0, 0.0], [0.0, 1.0]]
    g = {"tree": tr, "ml_anc": "yes", "float_tol": 1e-12}
    observed = substitution.get_substitution_tensor(
        state_tensor=state,
        mode="asis",
        g=g,
        mmap_attr="toy_sparse_asis",
    )
    expected = np.zeros((3, 2, 1, 2, 2), dtype=np.float64)
    for child_name in ["A", "B"]:
        child_id = labels[child_name]
        for site in range(state.shape[1]):
            expected[child_id, site, 0, :, :] = np.outer(
                state[labels["R"], site, :], state[child_id, site, :]
            )
            np.fill_diagonal(expected[child_id, site, 0, :, :], 0.0)
    assert isinstance(observed, substitution_sparse.SparseSubstitutionTensor)
    np.testing.assert_allclose(observed.to_dense(), expected, atol=1e-12)


def test_get_substitution_tensor_returns_sparse_for_small_input():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    state = np.zeros((3, 2, 2), dtype=np.float64)
    state[labels["R"], :, :] = [[1.0, 0.0], [1.0, 0.0]]
    state[labels["A"], :, :] = [[0.0, 1.0], [1.0, 0.0]]
    state[labels["B"], :, :] = [[1.0, 0.0], [0.0, 1.0]]
    g = {
        "tree": tr,
        "ml_anc": "yes",
        "float_tol": 1e-12,
    }

    observed = substitution.get_substitution_tensor(
        state_tensor=state,
        mode="asis",
        g=g,
        mmap_attr="toy_auto_sparse_density",
    )

    assert isinstance(observed, substitution_sparse.SparseSubstitutionTensor)
    assert observed.nnz > 0


def test_get_substitution_tensor_asis_threads_setting_matches_single_thread():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    num_node = max(labels.values()) + 1
    state = np.zeros((num_node, 3, 3), dtype=np.float64)
    state[labels["R"], :, :] = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.2, 0.5]]
    state[labels["N1"], :, :] = [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.4, 0.1, 0.5]]
    state[labels["A"], :, :] = [[0.2, 0.7, 0.1], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6]]
    state[labels["B"], :, :] = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.6, 0.1, 0.3]]
    state[labels["C"], :, :] = [[0.3, 0.6, 0.1], [0.1, 0.2, 0.7], [0.5, 0.2, 0.3]]
    g_single = {
        "tree": tr,
        "ml_anc": "yes",
        "float_tol": 1e-12,
        "threads": 1,
    }
    g_parallel = dict(g_single, threads=2)
    expected = substitution.get_substitution_tensor(
        state_tensor=state,
        mode="asis",
        g=g_single,
        mmap_attr="toy_asis_single",
    )
    observed = substitution.get_substitution_tensor(
        state_tensor=state,
        mode="asis",
        g=g_parallel,
        mmap_attr="toy_asis_parallel",
    )
    assert isinstance(observed, substitution_sparse.SparseSubstitutionTensor)
    np.testing.assert_allclose(observed.to_dense(), expected.to_dense(), atol=1e-12)


def test_get_substitution_tensor_syn_threads_setting_matches_single_thread():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    num_node = max(labels.values()) + 1
    # state order: [AAA, AAG, TTT, TTC]
    state = np.zeros((num_node, 2, 4), dtype=np.float64)
    state[labels["R"], :, :] = [[0.5, 0.5, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]]
    state[labels["N1"], :, :] = [[0.7, 0.3, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0]]
    state[labels["A"], :, :] = [[0.2, 0.8, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0]]
    state[labels["B"], :, :] = [[0.9, 0.1, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0]]
    state[labels["C"], :, :] = [[0.6, 0.4, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]]
    g_single = {
        "tree": tr,
        "ml_anc": "yes",
        "float_tol": 1e-12,
        "threads": 1,
        "amino_acid_orders": np.array(["K", "F"], dtype=object),
        "synonymous_indices": {"K": [0, 1], "F": [2, 3]},
        "max_synonymous_size": 2,
    }
    g_parallel = dict(g_single, threads=2)
    expected = substitution.get_substitution_tensor(
        state_tensor=state,
        mode="syn",
        g=g_single,
        mmap_attr="toy_syn_single",
    )
    observed = substitution.get_substitution_tensor(
        state_tensor=state,
        mode="syn",
        g=g_parallel,
        mmap_attr="toy_syn_parallel",
    )
    assert isinstance(observed, substitution_sparse.SparseSubstitutionTensor)
    np.testing.assert_allclose(observed.to_dense(), expected.to_dense(), atol=1e-12)


def test_get_substitution_tensor_sparse_asis_parallel_matches_single_thread(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,C:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    num_node = max(labels.values()) + 1
    state = np.zeros((num_node, 3, 3), dtype=np.float64)
    state[labels["R"], :, :] = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.2, 0.5]]
    state[labels["N1"], :, :] = [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.4, 0.1, 0.5]]
    state[labels["A"], :, :] = [[0.2, 0.7, 0.1], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6]]
    state[labels["B"], :, :] = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.6, 0.1, 0.3]]
    state[labels["C"], :, :] = [[0.3, 0.6, 0.1], [0.1, 0.2, 0.7], [0.5, 0.2, 0.3]]
    g_single = {
        "tree": tr,
        "ml_anc": "yes",
        "float_tol": 1e-12,
        "threads": 1,
    }
    g_parallel = dict(g_single, threads=2)
    expected = substitution.get_substitution_tensor(
        state_tensor=state,
        mode="asis",
        g=g_single,
        mmap_attr="toy_sparse_asis_single",
    )
    monkeypatch.setattr(
        substitution,
        "_resolve_sub_tensor_parallel_n_jobs",
        lambda num_branch_pairs, g, state_tensor=None, mode="": min(2, int(num_branch_pairs)),
    )
    observed = substitution.get_substitution_tensor(
        state_tensor=state,
        mode="asis",
        g=g_parallel,
        mmap_attr="toy_sparse_asis_parallel",
    )
    np.testing.assert_allclose(observed.to_dense(), expected.to_dense(), atol=1e-12)


def test_get_substitution_tensor_sparse_syn_indexed_child_matches_baseline(monkeypatch):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    state = np.zeros((3, 4, 4), dtype=np.float64)
    state[labels["R"], :, :] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    state[labels["A"], :, :] = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    state[labels["B"], :, :] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.3],
        [0.0, 0.0, 0.2, 0.8],
    ]
    base_g = {
        "tree": tr,
        "ml_anc": "yes",
        "float_tol": 1e-12,
        "amino_acid_orders": np.array(["AA0", "AA1"], dtype=object),
        "synonymous_indices": {"AA0": [0, 1], "AA1": [2, 3]},
        "max_synonymous_size": 2,
        "threads": 1,
    }
    monkeypatch.setattr(substitution, "_SPARSE_SUB_TENSOR_INDEXED_CHILD_MAX_DENSITY", -1.0)
    baseline = substitution.get_substitution_tensor(
        state_tensor=state,
        mode="syn",
        g=dict(base_g),
        mmap_attr="toy_sparse_syn_baseline",
    )
    monkeypatch.setattr(substitution, "_SPARSE_SUB_TENSOR_INDEXED_CHILD_MAX_DENSITY", 1.0)
    optimized = substitution.get_substitution_tensor(
        state_tensor=state,
        mode="syn",
        g=dict(base_g),
        mmap_attr="toy_sparse_syn_indexed",
    )
    np.testing.assert_allclose(optimized.to_dense(), baseline.to_dense(), atol=1e-12)


def test_apply_min_sub_pp_sparse_matches_dense():
    dense = _toy_reducer_tensor()
    sparse = substitution.dense_to_sparse_sub_tensor(dense.copy(), tol=0)
    g = {"min_sub_pp": 0.25, "ml_anc": False}
    dense_out = substitution.apply_min_sub_pp(g, dense.copy())
    sparse_out = substitution.apply_min_sub_pp(g, sparse)
    np.testing.assert_allclose(sparse_out.to_dense(), dense_out, atol=1e-12)


def test_get_group_state_totals_matches_dense_and_sparse():
    dense = _toy_dense_tensor()
    sparse = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    gad_d, ga_d, gd_d = substitution.get_group_state_totals(dense)
    gad_s, ga_s, gd_s = substitution.get_group_state_totals(sparse)
    np.testing.assert_allclose(gad_d, dense.sum(axis=(0, 1)), atol=1e-12)
    np.testing.assert_allclose(ga_d, dense.sum(axis=(0, 1, 4)), atol=1e-12)
    np.testing.assert_allclose(gd_d, dense.sum(axis=(0, 1, 3)), atol=1e-12)
    np.testing.assert_allclose(gad_s, gad_d, atol=1e-12)
    np.testing.assert_allclose(ga_s, ga_d, atol=1e-12)
    np.testing.assert_allclose(gd_s, gd_d, atol=1e-12)

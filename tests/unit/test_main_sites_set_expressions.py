import numpy as np
import pandas as pd
import pytest

from csubst import main_sites
from csubst import substitution_sparse
from csubst import ete


def test_set_expression_requires_parentheses_for_multiple_operators():
    branch_site_bool = {
        1: np.array([True, True, False, False], dtype=bool),
        2: np.array([True, False, True, False], dtype=bool),
        3: np.array([False, True, True, False], dtype=bool),
    }
    tokens = main_sites._tokenize_set_expression("1|2&3")
    with pytest.raises(ValueError, match="Ambiguous --mode set expression"):
        main_sites._evaluate_set_expression_boolean(tokens=tokens, branch_site_bool=branch_site_bool)


def test_set_expression_accepts_same_operator_or_chain_without_parentheses():
    branch_site_bool = {
        1: np.array([True, False, False], dtype=bool),
        2: np.array([True, True, False], dtype=bool),
        3: np.array([True, False, True], dtype=bool),
    }
    tokens = main_sites._tokenize_set_expression("1|2|3")
    out = main_sites._evaluate_set_expression_boolean(tokens=tokens, branch_site_bool=branch_site_bool)
    expected = branch_site_bool[1] | branch_site_bool[2] | branch_site_bool[3]
    np.testing.assert_array_equal(out, expected)


def test_set_expression_accepts_same_operator_and_chain_without_parentheses():
    branch_site_bool = {
        1: np.array([True, False, True, False], dtype=bool),
        2: np.array([True, True, False, False], dtype=bool),
        3: np.array([True, True, True, False], dtype=bool),
    }
    tokens = main_sites._tokenize_set_expression("1&2&3")
    out = main_sites._evaluate_set_expression_boolean(tokens=tokens, branch_site_bool=branch_site_bool)
    expected = branch_site_bool[1] & branch_site_bool[2] & branch_site_bool[3]
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("mode_expression", ["1^2^3", "1-2-3"])
def test_set_expression_rejects_same_operator_chain_for_xor_and_difference(mode_expression):
    branch_site_bool = {
        1: np.array([True, False, False], dtype=bool),
        2: np.array([True, True, False], dtype=bool),
        3: np.array([True, False, True], dtype=bool),
    }
    tokens = main_sites._tokenize_set_expression(mode_expression)
    with pytest.raises(ValueError, match="Ambiguous --mode set expression"):
        main_sites._evaluate_set_expression_boolean(tokens=tokens, branch_site_bool=branch_site_bool)


def test_set_expression_accepts_parenthesized_order():
    branch_site_bool = {
        1: np.array([True, True, False, False], dtype=bool),
        2: np.array([True, False, True, False], dtype=bool),
        3: np.array([False, True, True, False], dtype=bool),
    }
    tokens = main_sites._tokenize_set_expression("1|(2&3)")
    out = main_sites._evaluate_set_expression_boolean(tokens=tokens, branch_site_bool=branch_site_bool)
    expected = branch_site_bool[1] | (branch_site_bool[2] & branch_site_bool[3])
    np.testing.assert_array_equal(out, expected)


def test_add_set_mode_columns_evaluates_set_expression():
    df = pd.DataFrame(
        {
            "N_sub_1": [0.9, 0.2, 0.9, 0.1],
            "N_sub_5": [0.1, 0.9, 0.9, 0.9],
            "N_sub_25": [0.9, 0.9, 0.1, 0.9],
        }
    )
    g = {"mode": "set", "set_stat_type": "any", "mode_expression": "((1|5)-0)&25", "min_single_prob": 0.8}
    out = main_sites.add_set_mode_columns(df=df.copy(), g=g)
    # (1|5) => [T,T,T,T], minus root(0)=same, intersect 25 => [T,T,F,T]
    assert out["N_set_expr"].tolist() == [True, True, False, True]
    # PP is propagated by expression rules: OR=max, minus keeps LHS, AND=min.
    np.testing.assert_allclose(out["N_set_expr_prob"].to_numpy(), [0.9, 0.9, 0.0, 0.9], atol=1e-12)


def test_add_set_mode_columns_supports_xor_and_parentheses():
    df = pd.DataFrame(
        {
            "N_sub_1": [0.9, 0.1, 0.9, 0.1],
            "N_sub_5": [0.1, 0.9, 0.9, 0.1],
            "N_sub_9": [0.1, 0.1, 0.9, 0.9],
        }
    )
    g = {"mode": "set", "set_stat_type": "any", "mode_expression": "(1^5)&9", "min_single_prob": 0.8}
    out = main_sites.add_set_mode_columns(df=df.copy(), g=g)
    # 1^5 => [T,T,F,F], intersect 9 => [F,F,F,F]
    assert out["N_set_expr"].tolist() == [False, False, False, False]
    np.testing.assert_allclose(out["N_set_expr_prob"].to_numpy(), [0.0, 0.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    ("mode_expression", "expected_bool", "expected_prob"),
    [
        ("1|2", [True, True, True, False], [0.9, 0.9, 0.8, 0.0]),
        ("1&2", [False, True, False, False], [0.0, 0.8, 0.0, 0.0]),
        ("1-2", [True, False, False, False], [0.9, 0.0, 0.0, 0.0]),
        ("1^2", [True, False, True, False], [0.9, 0.0, 0.8, 0.0]),
    ],
)
def test_add_set_mode_columns_operator_probability_rules(mode_expression, expected_bool, expected_prob):
    df = pd.DataFrame(
        {
            "N_sub_1": [0.9, 0.9, 0.7, 0.1],
            "N_sub_2": [0.4, 0.8, 0.8, 0.2],
        }
    )
    g = {"mode": "set", "set_stat_type": "any", "mode_expression": mode_expression, "min_single_prob": 0.8}
    out = main_sites.add_set_mode_columns(df=df.copy(), g=g)
    assert out["N_set_expr"].tolist() == expected_bool
    np.testing.assert_allclose(out["N_set_expr_prob"].to_numpy(), expected_prob, atol=1e-12)


@pytest.mark.parametrize(
    ("set_stat_type", "expected_prob"),
    [
        ("any", 1.0),
        ("spe", 0.9),
    ],
)
def test_add_set_mode_columns_set_stat_type_changes_channelwise_aggregation(set_stat_type, expected_prob):
    # Toy example (1 site, 2 states):
    # branch 1: 0->1 is high (0.9), 1->0 is low (0.1)
    # branch 3: 0->1 is low  (0.1), 1->0 is high (0.9)
    # For expression 1|3:
    #   any: (1.0 | 1.0) -> 1.0
    #   spe: channel-wise OR then max channel -> 0.9
    on_tensor = np.zeros((4, 1, 1, 2, 2), dtype=float)
    on_tensor[1, 0, 0, 0, 1] = 0.9
    on_tensor[1, 0, 0, 1, 0] = 0.1
    on_tensor[3, 0, 0, 0, 1] = 0.1
    on_tensor[3, 0, 0, 1, 0] = 0.9
    df = pd.DataFrame(
        {
            "N_sub_1": [1.0],
            "N_sub_3": [1.0],
        }
    )
    g = {
        "mode": "set",
        "set_stat_type": set_stat_type,
        "mode_expression": "1|3",
        "min_single_prob": 0.5,
    }
    out = main_sites.add_set_mode_columns(df=df.copy(), g=g, ON_tensor=on_tensor)
    assert out["N_set_expr"].tolist() == [True]
    np.testing.assert_allclose(out["N_set_expr_prob"].to_numpy(), [expected_prob], atol=1e-12)


def test_add_set_mode_columns_spe_labels_use_nonsyn_state_orders():
    on_tensor = np.zeros((2, 1, 1, 2, 2), dtype=float)
    on_tensor[1, 0, 0, 0, 1] = 0.9
    df = pd.DataFrame({"N_sub_1": [0.9]})
    g = {
        "mode": "set",
        "set_stat_type": "spe",
        "mode_expression": "1",
        "min_single_prob": 0.5,
        "amino_acid_orders": np.array(["A", "V", "T", "I"], dtype=object),
        "nonsyn_state_orders": np.array(["AGPST", "C"], dtype=object),
    }

    out = main_sites.add_set_mode_columns(df=df.copy(), g=g, ON_tensor=on_tensor)

    assert out["N_set_expr_channel_label"].tolist() == ["X→C"]


def test_add_set_mode_columns_accepts_sparse_branch_tensors_for_spe():
    dense = np.zeros((4, 2, 1, 2, 2), dtype=float)
    dense[1, 0, 0, 0, 1] = 0.9
    dense[3, 0, 0, 0, 1] = 0.2
    dense[1, 1, 0, 1, 0] = 0.3
    dense[3, 1, 0, 1, 0] = 0.8
    sparse = substitution_sparse.SparseSubstitutionTensor.from_dense(dense)
    df = pd.DataFrame({"codon_site_alignment": [0, 1]})
    g = {
        "mode": "set",
        "set_stat_type": "spe",
        "mode_expression": "1|3",
        "min_single_prob": 0.5,
        "amino_acid_orders": np.array(["A", "B"], dtype=object),
    }

    dense_out = main_sites.add_set_mode_columns(df=df.copy(), g=g, ON_tensor=dense)
    sparse_out = main_sites.add_set_mode_columns(df=df.copy(), g=g, ON_tensor=sparse)

    assert sparse_out["N_set_expr"].tolist() == dense_out["N_set_expr"].tolist()
    np.testing.assert_allclose(
        sparse_out["N_set_expr_prob"].to_numpy(),
        dense_out["N_set_expr_prob"].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        sparse_out["N_set_expr_channel_index"].to_numpy(),
        dense_out["N_set_expr_channel_index"].to_numpy(),
    )


def test_add_set_mode_columns_supports_all_other_symbol(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    a_id = labels["A"]
    c_id = labels["C"]
    b_id = labels["B"]
    x_id = labels["X"]
    n_site = 3
    max_id = max([int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()])
    on_tensor = np.zeros((max_id + 1, n_site, 1, 1, 1), dtype=float)
    # Site 0: target-only substitution on A branch.
    on_tensor[a_id, 0, 0, 0, 0] = 0.9
    # Site 1: target on A and another-branch substitution on B.
    on_tensor[a_id, 1, 0, 0, 0] = 0.9
    on_tensor[b_id, 1, 0, 0, 0] = 0.9
    # Site 2: target-only substitution on C branch.
    on_tensor[c_id, 2, 0, 0, 0] = 0.9
    # Keep one internal non-target branch explicitly below threshold at all sites.
    on_tensor[x_id, :, 0, 0, 0] = 0.1
    df = pd.DataFrame(
        {
            "N_sub_{}".format(a_id): [0.9, 0.9, 0.0],
            "N_sub_{}".format(c_id): [0.0, 0.0, 0.9],
        }
    )
    g = {
        "mode": "set",
        "set_stat_type": "any",
        "mode_expression": "({}|{})-A".format(a_id, c_id),
        "min_single_prob": 0.8,
        "tree": tiny_tree,
    }
    out = main_sites.add_set_mode_columns(df=df.copy(), g=g, ON_tensor=on_tensor)
    assert out["N_set_expr"].tolist() == [True, False, True]
    np.testing.assert_allclose(out["N_set_expr_prob"].to_numpy(), [0.9, 0.0, 0.9], atol=1e-12)
    assert out["N_set_other"].tolist() == [False, True, False]
    np.testing.assert_allclose(out["N_set_other_prob"].to_numpy(), [0.1, 0.9, 0.1], atol=1e-12)
    np.testing.assert_allclose(out["S_set_other_prob"].to_numpy(), [0.0, 0.0, 0.0], atol=1e-12)
    assert out["N_set_A"].tolist() == [False, True, False]
    np.testing.assert_allclose(out["N_set_A_prob"].to_numpy(), [0.1, 0.9, 0.1], atol=1e-12)
    np.testing.assert_allclose(out["S_set_A_prob"].to_numpy(), [0.0, 0.0, 0.0], atol=1e-12)


def test_add_set_mode_columns_all_other_accepts_sparse_tensors(tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    a_id = labels["A"]
    c_id = labels["C"]
    b_id = labels["B"]
    x_id = labels["X"]
    n_site = 3
    max_id = max([int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()])
    on_dense = np.zeros((max_id + 1, n_site, 1, 1, 1), dtype=float)
    os_dense = np.zeros((max_id + 1, n_site, 1, 1, 1), dtype=float)
    on_dense[a_id, 0, 0, 0, 0] = 0.9
    on_dense[a_id, 1, 0, 0, 0] = 0.9
    on_dense[b_id, 1, 0, 0, 0] = 0.9
    on_dense[c_id, 2, 0, 0, 0] = 0.9
    on_dense[x_id, :, 0, 0, 0] = 0.1
    os_dense[b_id, 1, 0, 0, 0] = 0.4
    on_sparse = substitution_sparse.SparseSubstitutionTensor.from_dense(on_dense)
    os_sparse = substitution_sparse.SparseSubstitutionTensor.from_dense(os_dense)
    df = pd.DataFrame(
        {
            "N_sub_{}".format(a_id): [0.9, 0.9, 0.0],
            "N_sub_{}".format(c_id): [0.0, 0.0, 0.9],
        }
    )
    g = {
        "mode": "set",
        "set_stat_type": "any",
        "mode_expression": "({}|{})-A".format(a_id, c_id),
        "min_single_prob": 0.8,
        "tree": tiny_tree,
    }

    dense_out = main_sites.add_set_mode_columns(df=df.copy(), g=g, ON_tensor=on_dense, OS_tensor=os_dense)
    sparse_out = main_sites.add_set_mode_columns(df=df.copy(), g=g, ON_tensor=on_sparse, OS_tensor=os_sparse)

    assert sparse_out["N_set_expr"].tolist() == dense_out["N_set_expr"].tolist()
    assert sparse_out["N_set_other"].tolist() == dense_out["N_set_other"].tolist()
    np.testing.assert_allclose(
        sparse_out[["N_set_expr_prob", "N_set_other_prob", "S_set_other_prob"]].to_numpy(dtype=float),
        dense_out[["N_set_expr_prob", "N_set_other_prob", "S_set_other_prob"]].to_numpy(dtype=float),
        atol=1e-12,
    )


def test_resolve_site_jobs_intersection_fg_reads_cb_file(tmp_path, tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    cb_path = tmp_path / "cb.tsv"
    pd.DataFrame(
        {
            "branch_id_1": [labels["A"], labels["B"]],
            "branch_id_2": [labels["C"], labels["C"]],
            "is_fg_demo": ["Y", "N"],
        }
    ).to_csv(cb_path, sep="\t", index=False)
    g = {
        "tree": tiny_tree,
        "mode": "intersection",
        "branch_id": "fg",
        "cb_file": str(cb_path),
    }
    out = main_sites.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    np.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], [labels["A"], labels["C"]])


@pytest.mark.parametrize("fg_value", ["y", "yes", "true", "1", True])
def test_resolve_site_jobs_intersection_fg_accepts_truthy_fg_values(tmp_path, tiny_tree, fg_value):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    cb_path = tmp_path / "cb.tsv"
    pd.DataFrame(
        {
            "branch_id_1": [labels["A"], labels["B"]],
            "branch_id_2": [labels["C"], labels["C"]],
            "is_fg_demo": [fg_value, "N"],
        }
    ).to_csv(cb_path, sep="\t", index=False)
    g = {
        "tree": tiny_tree,
        "mode": "intersection",
        "branch_id": "fg",
        "cb_file": str(cb_path),
    }
    out = main_sites.resolve_site_jobs(g)
    assert len(out["site_jobs"]) == 1
    np.testing.assert_array_equal(out["site_jobs"][0]["branch_ids"], [labels["A"], labels["C"]])


def test_resolve_site_jobs_intersection_fg_rejects_non_integer_branch_ids(tmp_path, tiny_tree):
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tiny_tree.traverse()}
    cb_path = tmp_path / "cb.tsv"
    pd.DataFrame(
        {
            "branch_id_1": [float(labels["A"]) + 0.5],
            "branch_id_2": [labels["C"]],
            "is_fg_demo": ["Y"],
        }
    ).to_csv(cb_path, sep="\t", index=False)
    g = {
        "tree": tiny_tree,
        "mode": "intersection",
        "branch_id": "fg",
        "cb_file": str(cb_path),
    }
    with pytest.raises(ValueError, match="integer-like"):
        main_sites.resolve_site_jobs(g)

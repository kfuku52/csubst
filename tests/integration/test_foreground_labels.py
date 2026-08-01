
import numpy as np
import pandas as pd
import pytest

from csubst import foreground
from csubst import ete


def test_normalize_branch_ids_rejects_non_integer_like_values():
    with pytest.raises(ValueError, match="integer-like"):
        foreground._normalize_branch_ids([1.5])
    with pytest.raises(ValueError, match="integer-like"):
        foreground._normalize_branch_ids(["2.5"])
    with pytest.raises(ValueError, match="integer-like"):
        foreground._normalize_branch_ids([True])


def test_get_num_foreground_lineages_uses_compat_props():
    tr = ete.PhyloNode("(A:1,B:1)R;", format=1)
    for node in tr.traverse():
        ete.set_prop(node, "is_lineage_fg_traitA_1", False)
    root = [n for n in tr.traverse() if ete.is_root(n)][0]
    ete.set_prop(root, "is_lineage_fg_traitA_3", True)
    assert foreground.get_num_foreground_lineages(tr, "traitA") == 3


def test_read_foreground_file_rejects_invalid_fg_format1_shape(tmp_path):
    foreground_file = tmp_path / "foreground.tsv"
    foreground_file.write_text("1\tA\tEXTRA\n", encoding="utf-8")
    g = {"foreground": str(foreground_file), "fg_format": 1}
    with pytest.raises(ValueError, match="--fg_format 1"):
        foreground.read_foreground_file(g)


def test_read_foreground_file_rejects_invalid_fg_format2_shape(tmp_path):
    foreground_file = tmp_path / "foreground.tsv"
    foreground_file.write_text("name\nA\n", encoding="utf-8")
    g = {"foreground": str(foreground_file), "fg_format": 2}
    with pytest.raises(ValueError, match="--fg_format 2"):
        foreground.read_foreground_file(g)


def test_build_clade_permutation_mode_accepts_scalar_randomized_branch_id():
    out = foreground._build_clade_permutation_mode(
        trait_name="traitA",
        iteration=1,
        randomized_bids=np.int64(42),
        sample_original_foreground=False,
    )
    assert out == "randomization_traitA_iter1_bid42"


def test_set_target_label_column_accepts_scalar_positive_index():
    df = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 11, 12])
    out = foreground._set_target_label_column(
        df=df.copy(),
        column_name="is_target",
        positive_index=np.int64(11),
    )
    assert out.loc[10, "is_target"] == "N"
    assert out.loc[11, "is_target"] == "Y"
    assert out.loc[12, "is_target"] == "N"


def test_set_target_label_column_prefers_branch_id_column_over_index_labels():
    # index labels intentionally do not match the branch_id values.
    df = pd.DataFrame(
        {"branch_id": [2, 0, 1], "x": [1, 2, 3]},
        index=[0, 1, 2],
    )
    out = foreground._set_target_label_column(
        df=df.copy(),
        column_name="is_target",
        positive_index=np.int64(1),
    )
    assert out.loc[0, "is_target"] == "N"
    assert out.loc[1, "is_target"] == "N"
    assert out.loc[2, "is_target"] == "Y"


def test_count_branch_memberships_accepts_scalar_ids():
    cb = pd.DataFrame({"branch_id_1": [1, 2], "branch_id_2": [3, 4]})
    out = foreground._count_branch_memberships(cb=cb, bid_cols=["branch_id_1", "branch_id_2"], ids=np.int64(3))
    assert out.tolist() == [1, 0]


def test_count_branch_memberships_from_bid_matrix_accepts_scalar_ids():
    bid_matrix = np.array([[1, 3], [2, 4]], dtype=np.int64)
    out = foreground._count_branch_memberships_from_bid_matrix(bid_matrix=bid_matrix, ids=np.int64(3))
    assert out.tolist() == [1, 0]


def test_mark_dependent_foreground_rows_is_order_invariant_for_pairs():
    cb = pd.DataFrame(
        {
            "branch_id_1": [1, 1, 2, 3],
            "branch_id_2": [5, 3, 5, 4],
            "is_fg_traitA": ["Y", "Y", "Y", "Y"],
        }
    )
    dep = np.array([[5, 1], [5, 2]], dtype=np.int64)
    out = foreground._mark_dependent_foreground_rows(
        cb=cb.copy(deep=True),
        bid_cols=["branch_id_1", "branch_id_2"],
        trait_name="traitA",
        dependent_id_combinations=dep,
    )
    assert out.loc[:, "is_fg_traitA"].tolist() == ["N", "Y", "N", "Y"]


def test_mark_dependent_foreground_rows_is_order_invariant_for_higher_arity():
    cb = pd.DataFrame(
        {
            "branch_id_1": [1, 1, 2],
            "branch_id_2": [2, 2, 3],
            "branch_id_3": [3, 4, 4],
            "is_fg_traitA": ["Y", "Y", "Y"],
        }
    )
    dep = np.array([[3, 2, 1], [4, 3, 2]], dtype=np.int64)
    out = foreground._mark_dependent_foreground_rows(
        cb=cb.copy(deep=True),
        bid_cols=["branch_id_1", "branch_id_2", "branch_id_3"],
        trait_name="traitA",
        dependent_id_combinations=dep,
    )
    assert out.loc[:, "is_fg_traitA"].tolist() == ["N", "Y", "N"]


def test_compute_dependent_foreground_mask_is_order_invariant_for_pairs():
    cb = pd.DataFrame(
        {
            "branch_id_1": [1, 1, 2, 3],
            "branch_id_2": [5, 3, 5, 4],
        }
    )
    dep = np.array([[5, 1], [5, 2]], dtype=np.int64)
    out = foreground._compute_dependent_foreground_mask(
        cb=cb,
        bid_cols=["branch_id_1", "branch_id_2"],
        dependent_id_combinations=dep,
    )
    assert out.tolist() == [True, False, True, False]


def test_compute_dependent_foreground_mask_accepts_precomputed_bid_key():
    cb = pd.DataFrame(
        {
            "branch_id_1": [1, 1, 2, 3],
            "branch_id_2": [5, 3, 5, 4],
        }
    )
    dep = np.array([[5, 1], [5, 2]], dtype=np.int64)
    bid_matrix = cb.loc[:, ["branch_id_1", "branch_id_2"]].to_numpy(copy=False)
    bid_key = foreground._build_order_invariant_row_keys(bid_matrix, assume_sorted=False)
    out = foreground._compute_dependent_foreground_mask(
        cb=cb,
        bid_cols=["branch_id_1", "branch_id_2"],
        dependent_id_combinations=dep,
        precomputed_bid_key=bid_key,
    )
    assert out.tolist() == [True, False, True, False]


def test_assign_trait_labels_applies_dependent_mask_to_foreground_only():
    cb = pd.DataFrame(
        {
            "branch_num_fg_traitA": [2, 2, 1, 0],
            "branch_num_mg_traitA": [0, 0, 1, 2],
        }
    )
    out = foreground._assign_trait_labels(
        cb=cb.copy(deep=True),
        trait_name="traitA",
        arity=2,
        is_fg_dependent=np.array([True, False, False, False], dtype=bool),
    )
    assert out.loc[:, "is_fg_traitA"].tolist() == ["N", "Y", "N", "N"]
    assert out.loc[:, "is_mf_traitA"].tolist() == ["N", "N", "Y", "N"]
    assert out.loc[:, "is_mg_traitA"].tolist() == ["N", "N", "N", "Y"]


def test_assign_trait_labels_rejects_mismatched_dependent_mask_length():
    cb = pd.DataFrame(
        {
            "branch_num_fg_traitA": [2, 2],
            "branch_num_mg_traitA": [0, 0],
        }
    )
    with pytest.raises(ValueError, match="did not match cb rows"):
        foreground._assign_trait_labels(
            cb=cb.copy(deep=True),
            trait_name="traitA",
            arity=2,
            is_fg_dependent=np.array([True], dtype=bool),
        )


def test_assign_trait_labels_rejects_mismatched_num_fg_num_mg_length():
    cb = pd.DataFrame(
        {
            "branch_num_fg_traitA": [2, 2],
            "branch_num_mg_traitA": [0, 0],
        }
    )
    with pytest.raises(ValueError, match="num_fg length"):
        foreground._assign_trait_labels(
            cb=cb.copy(deep=True),
            trait_name="traitA",
            arity=2,
            num_fg=np.array([2], dtype=np.int64),
            num_mg=np.array([0, 0], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="num_mg length"):
        foreground._assign_trait_labels(
            cb=cb.copy(deep=True),
            trait_name="traitA",
            arity=2,
            num_fg=np.array([2, 2], dtype=np.int64),
            num_mg=np.array([0], dtype=np.int64),
        )

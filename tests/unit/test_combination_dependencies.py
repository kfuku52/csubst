import numpy as np
import pytest
import itertools

from csubst import combination
from csubst import tree
from csubst import ete


def test_map_row_combinations_to_node_ids_identity_fast_path_returns_row_ids():
    rows = np.array([[0, 2], [1, 3]], dtype=np.int64)
    node_ids = np.array([10, 20, 30, 40], dtype=np.int64)
    out = combination._map_row_combinations_to_node_ids(
        row_combinations=rows,
        node_ids=node_ids,
        row_ids_are_node_ids=True,
    )
    np.testing.assert_array_equal(out, rows)


def test_mark_dependent_row_combinations_flags_pairs_only_when_group_has_two_or_more():
    row_combinations = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    dep_groups = [
        np.array([0, 1], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
        np.array([9], dtype=np.int64),
    ]
    observed = combination._mark_dependent_row_combinations(row_combinations=row_combinations, dep_row_groups=dep_groups)
    expected = np.array([True, True, True, True], dtype=bool)
    np.testing.assert_array_equal(observed, expected)


def test_mark_dependent_row_combinations_bitset_matches_python_randomized():
    if combination.combination_cy is None or not hasattr(
        combination.combination_cy,
        "mark_dependent_combinations_bitset_int64",
    ):
        pytest.skip("Cython dependency-bitset fast path is unavailable")
    rng = np.random.default_rng(91)
    num_row = 80
    row_combinations = np.array(
        [rng.choice(num_row, size=4, replace=False) for _ in range(1000)],
        dtype=np.int64,
    )
    dep_groups = [
        rng.choice(num_row, size=int(size), replace=False).astype(np.int64)
        for size in rng.integers(1, 10, size=30)
    ]
    expected = combination._mark_dependent_row_combinations_python(
        row_combinations=row_combinations,
        dep_row_groups=dep_groups,
    )
    bitset = combination._build_dependency_bitset(dep_groups, num_row=num_row)
    observed = combination.combination_cy.mark_dependent_combinations_bitset_int64(
        row_combinations,
        bitset,
    ).astype(bool)
    np.testing.assert_array_equal(observed, expected)


def test_mark_dependent_row_combinations_falls_back_when_bitset_exceeds_cap(monkeypatch):
    rows = np.array([[0, 1], [0, 2], [1, 3]], dtype=np.int64)
    dep_groups = [np.array([0, 1], dtype=np.int64)]
    monkeypatch.setattr(combination, "_DEPENDENCY_BITSET_MAX_BYTES", 0)
    observed = combination._mark_dependent_row_combinations(rows, dep_groups)
    np.testing.assert_array_equal(observed, np.array([True, False, False]))


def test_dependency_bitset_fast_paths_reject_too_few_columns():
    malformed_bitset = np.zeros(shape=(9, 1), dtype=np.uint8)
    rows = np.array([[0, 8]], dtype=np.int64)
    with pytest.raises(ValueError, match="too few columns"):
        combination._mark_dependent_row_combinations_from_bitset(rows, malformed_bitset)
    with pytest.raises(ValueError, match="too few columns"):
        combination._generate_independent_row_combinations(
            candidate_rows=np.array([0, 8], dtype=np.int64),
            arity=2,
            dependency_bitset=malformed_bitset,
        )
    with pytest.raises(ValueError, match="too few columns"):
        combination._generate_union_candidates_by_shared_subset(
            target_nodes=np.array([[0, 1], [0, 8]], dtype=np.int64),
            arity=3,
            dependency_bitset=malformed_bitset,
        )
    if combination.combination_cy is not None:
        with pytest.raises(ValueError, match="too few columns"):
            combination.combination_cy.mark_dependent_combinations_bitset_int64(
                rows,
                malformed_bitset,
            )
        with pytest.raises(ValueError, match="too few columns"):
            combination.combination_cy.generate_independent_combinations_int64(
                np.array([0, 8], dtype=np.int64),
                malformed_bitset,
                2,
            )
        with pytest.raises(ValueError, match="too few columns"):
            combination.combination_cy.generate_union_candidates_shared_subset_int64(
                np.array([[0, 1], [0, 8]], dtype=np.int64),
                malformed_bitset,
            )


def test_generate_independent_row_combinations_matches_generate_then_filter():
    rng = np.random.default_rng(314)
    num_row = 18
    dep_groups = [
        rng.choice(num_row, size=int(size), replace=False).astype(np.int64)
        for size in [2, 3, 4, 5, 2, 3]
    ]
    dependency_bitset = combination._build_dependency_bitset(dep_groups, num_row=num_row)
    candidate_rows = np.arange(num_row, dtype=np.int64)
    for arity in [2, 3, 4]:
        all_rows = np.array(
            list(itertools.combinations(candidate_rows.tolist(), arity)),
            dtype=np.int64,
        )
        is_dependent = combination._mark_dependent_row_combinations_python(
            row_combinations=all_rows,
            dep_row_groups=dep_groups,
        )
        expected = all_rows[~is_dependent, :]
        observed = combination._generate_independent_row_combinations(
            candidate_rows=candidate_rows,
            arity=arity,
            dependency_bitset=dependency_bitset,
        )
        np.testing.assert_array_equal(observed, expected)


def test_generate_independent_row_combinations_python_fallback_matches_cython(monkeypatch):
    dep_groups = [
        np.array([0, 2, 5], dtype=np.int64),
        np.array([1, 4], dtype=np.int64),
    ]
    dependency_bitset = combination._build_dependency_bitset(dep_groups, num_row=7)
    expected = combination._generate_independent_row_combinations(
        candidate_rows=np.arange(7, dtype=np.int64),
        arity=3,
        dependency_bitset=dependency_bitset,
    )
    monkeypatch.setattr(combination, "combination_cy", None)
    observed = combination._generate_independent_row_combinations(
        candidate_rows=np.arange(7, dtype=np.int64),
        arity=3,
        dependency_bitset=dependency_bitset,
    )
    np.testing.assert_array_equal(observed, expected)


def test_conflict_aware_union_generation_matches_post_filter():
    pair_rows = np.array(
        [
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
        ],
        dtype=np.int64,
    )
    dep_groups = [np.array([2, 3], dtype=np.int64)]
    dependency_bitset = combination._build_dependency_bitset(dep_groups, num_row=4)
    generated = combination._generate_union_candidates_by_shared_subset(
        target_nodes=pair_rows,
        arity=3,
    )
    is_dependent = combination._mark_dependent_row_combinations_python(
        row_combinations=generated,
        dep_row_groups=dep_groups,
    )
    expected = generated[~is_dependent, :]
    observed = combination._generate_union_candidates_by_shared_subset(
        target_nodes=pair_rows,
        arity=3,
        dependency_bitset=dependency_bitset,
    )
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("exclude_sister_pair", [False, True])
def test_analytical_independent_combination_counts_match_bruteforce(exclude_sister_pair):
    tr = tree.add_numerical_node_labels(
        ete.PhyloNode("((A:1,B:1)X:1,(C:1,D:1)Y:1,E:1)R;", format=1)
    )
    branches = [node for node in tr.traverse() if not ete.is_root(node)]
    branch_ids = [int(ete.get_prop(node, "numerical_label")) for node in branches]
    node_by_id = {
        int(ete.get_prop(node, "numerical_label")): node
        for node in branches
    }

    def is_independent(branch_combination):
        selected = set(branch_combination)
        for branch_id in branch_combination:
            ancestors = {
                int(ete.get_prop(ancestor, "numerical_label"))
                for ancestor in ete.iter_ancestors(node_by_id[branch_id])
                if not ete.is_root(ancestor)
            }
            if len(selected & ancestors) > 0:
                return False
        if exclude_sister_pair:
            for node in tr.traverse():
                child_ids = {
                    int(ete.get_prop(child, "numerical_label"))
                    for child in ete.get_children(node)
                }
                if len(selected & child_ids) > 1:
                    return False
        return True

    observed = combination.count_independent_branch_combinations(
        tree_obj=tr,
        max_arity=4,
        exclude_sister_pair=exclude_sister_pair,
    )
    expected = [1]
    for arity in range(1, 5):
        expected.append(
            sum(
                is_independent(branch_combination)
                for branch_combination in itertools.combinations(branch_ids, arity)
            )
        )
    assert observed == expected

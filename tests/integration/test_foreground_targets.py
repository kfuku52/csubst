
import numpy as np
import pandas as pd
import pytest

from csubst import foreground
from csubst import combination
from csubst import ete
from csubst import tree


def test_annotate_foreground_fg_stem_only_keeps_lineage_specific_stem_colors():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((Nep1:1,Nep2:1)N:1,Ceph:1)R;", format=1))
    g = {
        "tree": tr,
        "fg_stem_only": True,
        "fg_df": pd.DataFrame(
            {
                "name": ["Nep1", "Nep2", "Ceph"],
                "traitA": [1, 1, 2],
            }
        ),
    }
    out = foreground.get_foreground_ids(g=g, write=False)
    node_by_name = {n.name: n for n in out["tree"].traverse()}
    trait = "traitA"
    # Stem branch of lineage 1 (Nep1+Nep2 clade) should remain red.
    assert ete.get_prop(node_by_name["N"], "color_" + trait) == "red"
    # Stem branch of lineage 2 (Ceph leaf branch) should be blue.
    assert ete.get_prop(node_by_name["Ceph"], "color_" + trait) == "blue"
    # Tip label colors should match lineage colors.
    assert ete.get_prop(node_by_name["Nep1"], "labelcolor_" + trait) == "red"
    assert ete.get_prop(node_by_name["Nep2"], "labelcolor_" + trait) == "red"
    assert ete.get_prop(node_by_name["Ceph"], "labelcolor_" + trait) == "blue"


def test_get_num_foreground_lineages_reads_tree_properties():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "is_lineage_fg_traitX_1", True)
        ete.set_prop(node, "is_lineage_fg_traitX_3", False)
        # Non-numeric suffix should be ignored.
        ete.set_prop(node, "is_lineage_fg_traitX_extra", True)
    assert foreground.get_num_foreground_lineages(tr, "traitX") == 3


def test_annotate_foreground_keeps_distinct_lineage_colors_for_stem_only():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)N1:1,(C:1,D:1)N2:1)R;", format=1))
    g = {
        "tree": tr,
        "fg_stem_only": True,
        "fg_df": pd.DataFrame(
            {
                "name": ["A", "B", "C", "D"],
                "PLACEHOLDER": [1, 1, 2, 2],
            }
        ),
    }
    g["fg_leaf_names"] = {"PLACEHOLDER": [["A", "B"], ["C", "D"]]}
    g["tree"] = foreground.annotate_lineage_foreground(lineages=np.array([1, 2]), trait_name="PLACEHOLDER", g=g)
    g["tree"] = foreground.annotate_foreground(lineages=np.array([1, 2]), trait_name="PLACEHOLDER", g=g)

    nodes_by_name = {n.name: n for n in g["tree"].traverse() if n.name}
    n1_color = ete.get_prop(nodes_by_name["N1"], "color_PLACEHOLDER")
    n2_color = ete.get_prop(nodes_by_name["N2"], "color_PLACEHOLDER")
    n1_lineage = ete.get_prop(nodes_by_name["N1"], "foreground_lineage_id_PLACEHOLDER")
    n2_lineage = ete.get_prop(nodes_by_name["N2"], "foreground_lineage_id_PLACEHOLDER")
    assert n1_color != "black"
    assert n2_color != "black"
    assert n1_color != n2_color
    assert n1_lineage != n2_lineage
    assert {int(n1_lineage), int(n2_lineage)} == {1, 2}


def test_get_target_ids_excludes_root_even_for_full_clade_foreground():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    root_id = ete.get_prop(ete.get_tree_root(tr), "numerical_label")
    g = {
        "tree": tr,
        "fg_stem_only": False,
        "fg_df": pd.DataFrame({"name": ["A", "B"], "PLACEHOLDER": [1, 1]}),
        "fg_leaf_names": {"PLACEHOLDER": [["A", "B"]]},
    }
    lineages = np.array([1])
    g["tree"] = foreground.annotate_lineage_foreground(lineages=lineages, trait_name="PLACEHOLDER", g=g)
    target_ids = foreground.get_target_ids(lineages=lineages, trait_name="PLACEHOLDER", g=g)
    assert int(root_id) not in set(int(x) for x in target_ids.tolist())


def test_get_df_clade_size_handles_noncontiguous_branch_ids():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,(B:1,C:1)X:1)R;", format=1))
    nodes = {n.name: n for n in tr.traverse() if n.name}
    reassigned = {
        "A": 11,
        "B": 29,
        "C": 41,
        "X": 73,
        "R": 5,
    }
    for name,node in nodes.items():
        ete.set_prop(node, "numerical_label", reassigned[name])
        ete.set_prop(node, "is_fg_traitA", False)
    ete.set_prop(nodes["X"], "is_fg_traitA", True)
    ete.set_prop(nodes["B"], "is_fg_traitA", True)
    g = {"tree": tr}

    out = foreground.get_df_clade_size(g=g, trait_name="traitA")

    expected_ids = {11, 29, 41, 73}
    assert set(out.loc[:, "branch_id"].astype(int).tolist()) == expected_ids
    assert not out.loc[:, "size"].isna().any()
    assert bool(out.loc[73, "is_fg_stem_traitA"])
    assert not bool(out.loc[29, "is_fg_stem_traitA"])


def test_get_marginal_branch_accepts_scalar_target_ids(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,(B:1,C:1)X:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    for node in tr.traverse():
        ete.set_prop(node, "is_fg_traitA", False)
    ete.set_prop(next(n for n in tr.traverse() if n.name == "B"), "is_fg_traitA", True)
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"name": ["A", "B", "C"], "traitA": [0, 1, 0]}),
        "target_ids": {"traitA": np.int64(labels["B"])},
        "mg_parent": False,
        "mg_sister": True,
        "mg_sister_stem_only": True,
        "outdir": str(tmp_path),
        "output_prefix": "margin_run",
    }
    out = foreground.get_marginal_branch(g)
    assert set(out["mg_ids"]["traitA"].tolist()) == {labels["C"]}
    assert set(out["target_ids"]["traitA"].tolist()) == {labels["B"], labels["C"]}
    assert (tmp_path / "margin_run_marginal_branch_traitA.txt").exists()


def test_get_foreground_ids_writes_target_file_under_output_namespace(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,(B:1,C:1)X:1)R;", format=1))
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"name": ["B"], "traitA": [1]}),
        "fg_stem_only": True,
        "outdir": str(tmp_path),
        "output_prefix": "fg_run",
    }

    out = foreground.get_foreground_ids(g, write=True)

    target_path = tmp_path / "fg_run_foreground_branch_traitA.txt"
    assert target_path.exists()
    target_ids = [int(line.strip()) for line in target_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert target_ids == out["target_ids"]["traitA"].tolist()


def test_get_foreground_ids_ignores_string_zero_background_rows():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"name": ["A", "B"], "traitA": ["FG1", "0"]}),
        "fg_stem_only": True,
    }

    out = foreground.get_foreground_ids(g, write=False)

    assert out["fg_leaf_names"]["traitA"] == [["A"]]
    assert len(out["fg_ids"]["traitA"].tolist()) == 1
    node_by_name = {n.name: n for n in out["tree"].traverse() if n.name}
    assert int(ete.get_prop(node_by_name["A"], "foreground_lineage_id_traitA")) == 1
    assert int(ete.get_prop(node_by_name["B"], "foreground_lineage_id_traitA")) == 0


def test_randomize_foreground_flags_without_sample_original_preserves_target_count():
    before = np.array([True, False, False, True, False], dtype=bool)
    out = foreground._randomize_foreground_flags(
        before_randomization=before,
        sample_original_foreground=False,
        rng=np.random.default_rng(0),
    )
    assert int(out.sum()) == int(before.sum())
    assert not out[np.where(before)[0]].any()


def test_randomize_foreground_flags_without_sample_original_raises_when_candidates_insufficient():
    before = np.array([True, True, False], dtype=bool)
    with pytest.raises(ValueError, match="Not enough non-foreground clades"):
        foreground._randomize_foreground_flags(before_randomization=before, sample_original_foreground=False)


def test_randomize_foreground_stems_preserves_bin_counts_without_nested_clades():
    trait_cache = {
        "is_fg_stem": np.array([False, True, False, False, True], dtype=bool),
        "descendant_indices_by_index": [
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([4], dtype=np.int64),
        ],
    }
    randomization_plan = {
        "fg_bins": np.array([1, 2], dtype=np.int64),
        "bin_indices": {
            1: np.array([1, 2, 3], dtype=np.int64),
            2: np.array([0, 4], dtype=np.int64),
        },
    }

    randomized = foreground._randomize_foreground_stem_flags_from_plan(
        trait_cache=trait_cache,
        randomization_plan=randomization_plan,
        sample_original_foreground=False,
        rng=np.random.default_rng(0),
    )

    assert np.where(randomized)[0].tolist() == [0, 3]
    assert not randomized[trait_cache["is_fg_stem"]].any()
    for indices in randomization_plan["bin_indices"].values():
        assert int(randomized[indices].sum()) == int(trait_cache["is_fg_stem"][indices].sum())
    selected = np.where(randomized)[0]
    for i, branch_index in enumerate(selected.tolist()):
        descendants = set(trait_cache["descendant_indices_by_index"][branch_index].tolist())
        assert descendants.isdisjoint(set(selected[i + 1:].tolist()))


def test_randomize_foreground_stems_avoids_nested_candidates_within_one_bin():
    trait_cache = {
        "is_fg_stem": np.array([False, False, False, True, True], dtype=bool),
        "descendant_indices_by_index": [
            np.array([0, 1], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([4], dtype=np.int64),
        ],
    }
    randomization_plan = {
        "fg_bins": np.array([1], dtype=np.int64),
        "bin_indices": {1: np.arange(5, dtype=np.int64)},
    }

    for seed in range(20):
        randomized = foreground._randomize_foreground_stem_flags_from_plan(
            trait_cache=trait_cache,
            randomization_plan=randomization_plan,
            sample_original_foreground=False,
            rng=np.random.default_rng(seed),
        )
        selected = set(np.where(randomized)[0].tolist())
        assert len(selected) == 2
        assert not ({0, 1} <= selected)
        assert selected.isdisjoint({3, 4})


def test_get_randomized_pair_combinations_matches_general_combination_path():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,(C:1,D:1)Y:1)R;", format=1))
    node_by_name = {n.name: n for n in tr.traverse() if n.name}
    a_id = int(ete.get_prop(node_by_name["A"], "numerical_label"))
    b_id = int(ete.get_prop(node_by_name["B"], "numerical_label"))
    c_id = int(ete.get_prop(node_by_name["C"], "numerical_label"))
    d_id = int(ete.get_prop(node_by_name["D"], "numerical_label"))
    g = {
        "tree": tr,
        "fg_df": pd.DataFrame({"name": ["A", "B", "C", "D"], "traitA": [1, 1, 1, 1]}),
        "dep_ids": [np.array([a_id, b_id], dtype=np.int64)],
        "fg_dep_ids": {"traitA": [np.array([c_id, d_id], dtype=np.int64)]},
        "exhaustive_until": 1,
        "current_arity": 2,
        "threads": 1,
        "r_target_ids": {"traitA": np.array([a_id, b_id, c_id, d_id], dtype=np.int64)},
    }
    fast = foreground._get_randomized_pair_combinations(g=g, trait_name="traitA")

    g_general = {
        "tree": tr,
        "dep_ids": [np.array([a_id, b_id], dtype=np.int64)],
        "fg_dep_ids": {"traitA": [np.array([c_id, d_id], dtype=np.int64)]},
        "fg_df": pd.DataFrame({"name": ["A", "B", "C", "D"], "traitA": [1, 1, 1, 1]}),
        "threads": 1,
        "exhaustive_until": 1,
    }
    _, general = combination.get_node_combinations(
        g=g_general,
        target_id_dict={"traitA": np.array([a_id, b_id, c_id, d_id], dtype=np.int64)},
        arity=2,
        check_attr="name",
        verbose=False,
    )

    def _sort_rows(values):
        if values.shape[0] == 0:
            return values
        out = np.sort(values.astype(np.int64, copy=False), axis=1)
        order = np.lexsort((out[:, 1], out[:, 0]))
        return out[order, :]

    assert np.array_equal(_sort_rows(fast), _sort_rows(general))

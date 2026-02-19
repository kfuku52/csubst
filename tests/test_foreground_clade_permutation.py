import numpy
import pandas
import pandas.testing as pdt

from csubst import foreground
from csubst import ete
from csubst import tree


def test_get_num_foreground_lineages_uses_compat_props():
    tr = ete.PhyloNode("(A:1,B:1)R;", format=1)
    for node in tr.traverse():
        ete.set_prop(node, "is_lineage_fg_traitA_1", False)
    root = [n for n in tr.traverse() if ete.is_root(n)][0]
    ete.set_prop(root, "is_lineage_fg_traitA_3", True)
    assert foreground.get_num_foreground_lineages(tr, "traitA") == 3


def test_annotate_foreground_fg_stem_only_keeps_lineage_specific_stem_colors():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((Nep1:1,Nep2:1)N:1,Ceph:1)R;", format=1))
    g = {
        "tree": tr,
        "fg_stem_only": True,
        "fg_df": pandas.DataFrame(
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
        "fg_df": pandas.DataFrame(
            {
                "name": ["A", "B", "C", "D"],
                "PLACEHOLDER": [1, 1, 2, 2],
            }
        ),
    }
    g["fg_leaf_names"] = {"PLACEHOLDER": [["A", "B"], ["C", "D"]]}
    g["tree"] = foreground.annotate_lineage_foreground(lineages=numpy.array([1, 2]), trait_name="PLACEHOLDER", g=g)
    g["tree"] = foreground.annotate_foreground(lineages=numpy.array([1, 2]), trait_name="PLACEHOLDER", g=g)

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
        "fg_df": pandas.DataFrame({"name": ["A", "B"], "PLACEHOLDER": [1, 1]}),
        "fg_leaf_names": {"PLACEHOLDER": [["A", "B"]]},
    }
    lineages = numpy.array([1])
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


def test_clade_permutation_uses_observed_stats_when_main_table_has_only_permutations(monkeypatch):
    observed = pandas.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.5],
            "total_OCNany2spe_fg_traitA": [3.0],
        }
    )
    g = {
        "fg_df": pandas.DataFrame({"name": ["tip1"], "traitA": [1]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pandas.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pandas.DataFrame({"branch_id_1": [1], "branch_id_2": [2]})

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pandas.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name):
        local_g["r_fg_ids"] = {trait_name: numpy.array([10, 11], dtype=numpy.int64)}
        return local_g, numpy.array([[1, 2]], dtype=numpy.int64)

    def fake_add_median_cb_stats(local_g, rcb, current_arity, start, verbose=False):
        local_g["df_cb_stats"].loc[:, "arity"] = current_arity
        local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_traitA"] = 0.9
        local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_traitA"] = 2.0
        return local_g

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)
    monkeypatch.setattr(foreground, "add_median_cb_stats", fake_add_median_cb_stats)

    out = foreground.clade_permutation(cb, g)

    pdt.assert_frame_equal(out["df_cb_stats"], observed)
    assert out["df_cb_stats_main"].shape[0] == 1
    assert out["df_cb_stats_main"].loc[0, "mode"].startswith("randomization_traitA_")


def test_clade_permutation_iterates_all_traits_in_fg_format2(monkeypatch):
    observed = pandas.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.4],
            "total_OCNany2spe_fg_traitA": [2.5],
            "median_omegaCany2spe_fg_traitB": [0.8],
            "total_OCNany2spe_fg_traitB": [4.0],
        }
    )
    g = {
        "fg_df": pandas.DataFrame({"name": ["tip1"], "traitA": [1], "traitB": [0]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pandas.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pandas.DataFrame({"branch_id_1": [1], "branch_id_2": [2]})
    called_traits = []

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pandas.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name, num_trial=100, sample_original_foreground=False):
        called_traits.append(trait_name)
        local_g["r_fg_ids"] = {trait_name: numpy.array([21, 22], dtype=numpy.int64)}
        return local_g, numpy.array([[1, 2]], dtype=numpy.int64)

    def fake_add_median_cb_stats(local_g, rcb, current_arity, start, verbose=False):
        local_g["df_cb_stats"].loc[:, "arity"] = current_arity
        trait_cols = [c for c in rcb.columns if c.startswith("is_fg_") and (rcb[c] == "Y").all()]
        assert len(trait_cols) == 1
        focal_trait = trait_cols[0].replace("is_fg_", "")
        for trait_name in ["traitA", "traitB"]:
            local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_" + trait_name] = numpy.nan
            local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_" + trait_name] = numpy.nan
        local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_" + focal_trait] = 1.0
        local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_" + focal_trait] = 3.0
        return local_g

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)
    monkeypatch.setattr(foreground, "add_median_cb_stats", fake_add_median_cb_stats)

    out = foreground.clade_permutation(cb, g)

    assert called_traits == ["traitA", "traitB"]
    assert out["df_cb_stats_main"].shape[0] == 2
    modes = out["df_cb_stats_main"].loc[:, "mode"].tolist()
    assert any(mode.startswith("randomization_traitA_") for mode in modes)
    assert any(mode.startswith("randomization_traitB_") for mode in modes)


def test_clade_permutation_continues_when_randomization_fails(monkeypatch):
    observed = pandas.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.5],
            "total_OCNany2spe_fg_traitA": [2.0],
        }
    )
    g = {
        "fg_df": pandas.DataFrame({"name": ["tip1"], "traitA": [1]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pandas.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pandas.DataFrame({"branch_id_1": [1], "branch_id_2": [2]})

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pandas.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name, num_trial=100, sample_original_foreground=False):
        raise Exception("no permutation candidates")

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)

    out = foreground.clade_permutation(cb, g)

    pdt.assert_frame_equal(out["df_cb_stats"], observed)
    assert out["df_cb_stats_main"].shape[0] == 1
    mode_value = out["df_cb_stats_main"].loc[0, "mode"]
    assert mode_value.startswith("randomization_traitA_iter0_failed_trial")
    assert out["df_cb_stats_main"].loc[0, "clade_permutation_status_traitA"] == "no permutation candidates"


def test_clade_permutation_retries_with_sample_original_foreground(monkeypatch):
    observed = pandas.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.4],
            "total_OCNany2spe_fg_traitA": [2.5],
        }
    )
    g = {
        "fg_df": pandas.DataFrame({"name": ["tip1"], "traitA": [1]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pandas.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pandas.DataFrame({"branch_id_1": [1], "branch_id_2": [2]})
    sampled_flags = []

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pandas.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name, num_trial=100, sample_original_foreground=False):
        sampled_flags.append(sample_original_foreground)
        if not sample_original_foreground:
            raise Exception("strict mode failed")
        local_g["r_fg_ids"] = {trait_name: numpy.array([1, 2], dtype=numpy.int64)}
        return local_g, numpy.array([[1, 2]], dtype=numpy.int64)

    def fake_add_median_cb_stats(local_g, rcb, current_arity, start, verbose=False):
        local_g["df_cb_stats"].loc[:, "arity"] = current_arity
        local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_traitA"] = 0.9
        local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_traitA"] = 3.0
        return local_g

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)
    monkeypatch.setattr(foreground, "add_median_cb_stats", fake_add_median_cb_stats)

    out = foreground.clade_permutation(cb, g)

    assert sampled_flags == [False, True]
    assert out["df_cb_stats_main"].shape[0] == 1
    assert "_sampleorig_" in out["df_cb_stats_main"].loc[0, "mode"]


def test_clade_permutation_recomputes_missing_randomized_combinations(monkeypatch, capsys):
    observed = pandas.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.5],
            "total_OCNany2spe_fg_traitA": [2.0],
        }
    )
    g = {
        "fg_df": pandas.DataFrame({"name": ["tip1"], "traitA": [1]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pandas.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pandas.DataFrame({"branch_id_1": [1], "branch_id_2": [2], "dummy": [0.0]})
    recompute_calls = []

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pandas.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name, num_trial=100, sample_original_foreground=False):
        local_g["r_fg_ids"] = {trait_name: numpy.array([1, 2], dtype=numpy.int64)}
        # include one combination that is missing from cb to trigger recomputation
        return local_g, numpy.array([[1, 2], [2, 3]], dtype=numpy.int64)

    def fake_recompute_missing(g, missing_id_combinations, OS_tensor_reducer, ON_tensor_reducer):
        recompute_calls.append(missing_id_combinations.tolist())
        cb_missing = pandas.DataFrame({"branch_id_1": [2], "branch_id_2": [3], "dummy": [1.0]})
        return cb_missing, g

    def fake_add_median_cb_stats(local_g, rcb, current_arity, start, verbose=False):
        branch_pairs = {(int(r.branch_id_1), int(r.branch_id_2)) for r in rcb.itertuples(index=False)}
        assert branch_pairs == {(1, 2), (2, 3)}
        local_g["df_cb_stats"].loc[:, "arity"] = current_arity
        local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_traitA"] = 0.9
        local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_traitA"] = 3.0
        return local_g

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)
    monkeypatch.setattr(foreground, "_recompute_missing_permutation_rows", fake_recompute_missing)
    monkeypatch.setattr(foreground, "add_median_cb_stats", fake_add_median_cb_stats)

    out = foreground.clade_permutation(
        cb=cb,
        g=g,
        OS_tensor_reducer=object(),
        ON_tensor_reducer=object(),
    )

    assert recompute_calls == [[[2, 3]]]
    assert out["df_cb_stats_main"].shape[0] == 1
    captured = capsys.readouterr()
    assert "permuted foreground branch combinations were dropped" not in captured.out


def test_clade_permutation_reports_dropped_without_recomputation(monkeypatch, capsys):
    observed = pandas.DataFrame(
        {
            "arity": [2],
            "mode": ["foreground"],
            "median_omegaCany2spe_fg_traitA": [0.5],
            "total_OCNany2spe_fg_traitA": [2.0],
        }
    )
    g = {
        "fg_df": pandas.DataFrame({"name": ["tip1"], "traitA": [1]}),
        "df_cb_stats": observed.copy(deep=True),
        "df_cb_stats_main": pandas.DataFrame(),
        "fg_clade_permutation": 1,
        "current_arity": 2,
        "cutoff_stat": "dist_bl>0",
    }
    cb = pandas.DataFrame({"branch_id_1": [1], "branch_id_2": [2], "dummy": [0.0]})

    def fake_initialize_df_cb_stats(local_g):
        local_g["df_cb_stats"] = pandas.DataFrame({"arity": [local_g["current_arity"]], "mode": [""]})
        return local_g

    def fake_set_random_foreground_branch(local_g, trait_name, num_trial=100, sample_original_foreground=False):
        local_g["r_fg_ids"] = {trait_name: numpy.array([1, 2], dtype=numpy.int64)}
        return local_g, numpy.array([[1, 2], [2, 3]], dtype=numpy.int64)

    def fake_add_median_cb_stats(local_g, rcb, current_arity, start, verbose=False):
        # without recomputation only one row survives from the merge
        assert rcb.shape[0] == 1
        local_g["df_cb_stats"].loc[:, "arity"] = current_arity
        local_g["df_cb_stats"].loc[:, "median_omegaCany2spe_fg_traitA"] = 0.9
        local_g["df_cb_stats"].loc[:, "total_OCNany2spe_fg_traitA"] = 3.0
        return local_g

    monkeypatch.setattr(foreground.param, "initialize_df_cb_stats", fake_initialize_df_cb_stats)
    monkeypatch.setattr(foreground, "set_random_foreground_branch", fake_set_random_foreground_branch)
    monkeypatch.setattr(foreground, "add_median_cb_stats", fake_add_median_cb_stats)

    out = foreground.clade_permutation(cb=cb, g=g)

    assert out["df_cb_stats_main"].shape[0] == 1
    captured = capsys.readouterr()
    assert "permuted foreground branch combinations were dropped" in captured.out

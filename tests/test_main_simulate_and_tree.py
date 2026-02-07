import numpy
import pytest

from csubst import main_simulate
from csubst import tree
from csubst import ete


def test_add_numerical_node_labels_assigns_unique_integers():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(B:1,(A:1,C:1)X:1)R;", format=1))
    labels = [n.numerical_label for n in tr.traverse()]
    assert sorted(labels) == list(range(len(labels)))


def test_is_consistent_tree_checks_leaf_sets():
    t1 = ete.PhyloNode("(A:1,B:1)R;", format=1)
    t2 = ete.PhyloNode("(B:1,A:1)R2;", format=1)
    t3 = ete.PhyloNode("(A:1,C:1)R3;", format=1)
    assert tree.is_consistent_tree(t1, t2)
    assert not tree.is_consistent_tree(t1, t3)


def test_standardize_node_names_removes_suffixes_and_quotes():
    tr = ete.PhyloNode("('A/1':1,'B[abc]':1)'N1[xy]':0;", format=1)
    out = tree.standardize_node_names(tr)
    names = sorted([n.name for n in out.traverse()])
    assert names == ["A", "B", "N1"]


def test_transfer_internal_node_names_copies_labels_by_topology():
    tree_to = ete.PhyloNode("(A:1,(B:1,C:1):1);", format=1)
    tree_from = ete.PhyloNode("(A:2,(B:2,C:2)X:2)R;", format=1)
    out = tree.transfer_internal_node_names(tree_to, tree_from)
    name_by_leafset = {tuple(sorted(ete.get_leaf_names(n))): n.name for n in out.traverse() if not ete.is_leaf(n)}
    assert name_by_leafset[("A", "B", "C")] == "R"
    assert name_by_leafset[("B", "C")] == "X"


def test_is_internal_node_labeled():
    labeled = ete.PhyloNode("(A:1,B:1)R;", format=1)
    unlabeled = ete.PhyloNode("(A:1,(B:1,C:1):1)R;", format=1)
    assert tree.is_internal_node_labeled(labeled)
    assert not tree.is_internal_node_labeled(unlabeled)


def test_get_pyvolve_codon_order_contains_61_sense_codons():
    codons = main_simulate.get_pyvolve_codon_order()
    assert codons.shape == (61,)
    assert set(["TAA", "TAG", "TGA"]).isdisjoint(set(codons))


def test_get_codons_extracts_expected_members():
    codon_table = [["K", "AAA"], ["K", "AAG"], ["N", "AAC"], ["*", "TAA"]]
    out = main_simulate.get_codons("K", codon_table)
    numpy.testing.assert_array_equal(out, ["AAA", "AAG"])


def test_biased_index_helpers_match_manual_pairs():
    codon_table = [["K", "AAA"], ["K", "AAG"], ["N", "AAC"], ["*", "TAA"]]
    codon_order = numpy.array(["AAA", "AAG", "AAC"])
    biased_aas = numpy.array(["K"])
    nsy_idx = main_simulate.get_biased_nonsynonymous_substitution_index(biased_aas, codon_table, codon_order)
    numpy.testing.assert_array_equal(nsy_idx, [[2, 0], [2, 1]])
    cdn_idx = main_simulate.get_biased_codon_index(biased_aas, codon_table, codon_order)
    numpy.testing.assert_array_equal(cdn_idx, [[0], [1]])


def test_get_synonymous_codon_substitution_index_manual_count():
    g = {"codon_table": [["K", "AAA"], ["K", "AAG"], ["N", "AAC"]]}
    codon_order = numpy.array(["AAA", "AAG", "AAC"])
    out = main_simulate.get_synonymous_codon_substitution_index(g, codon_order)
    numpy.testing.assert_array_equal(out, [[0, 1], [1, 0]])


def test_get_total_Q_sums_requested_entries():
    mat = numpy.array([[0.0, 1.0], [2.0, 0.0]])
    idx = numpy.array([[0, 1], [1, 0]])
    assert main_simulate.get_total_Q(mat, idx) == 3.0


def test_rescale_substitution_matrix_preserves_total_and_zero_row_sum():
    mat = numpy.array([[0.0, 2.0, 1.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]])
    idx = numpy.array([[0, 1], [1, 2]])
    out = main_simulate.rescale_substitution_matrix(mat, idx, scaling_factor=2.0)
    offdiag_before = mat[~numpy.eye(mat.shape[0], dtype=bool)].sum()
    offdiag_after = out[~numpy.eye(out.shape[0], dtype=bool)].sum()
    assert pytest.approx(offdiag_after, rel=0, abs=1e-12) == offdiag_before
    numpy.testing.assert_allclose(out.sum(axis=1), [0.0, 0.0, 0.0], atol=1e-12)


def test_bias_eq_freq_scales_and_renormalizes():
    eq = numpy.array([0.2, 0.3, 0.5], dtype=float)
    out = main_simulate.bias_eq_freq(eq, biased_cdn_index=numpy.array([[0], [1]]), convergence_intensity_factor=2.0)
    # pre-normalization: [0.4, 0.6, 0.5] -> normalized by 1.5
    numpy.testing.assert_allclose(out, [4 / 15, 6 / 15, 5 / 15], atol=1e-12)


def test_apply_percent_biased_sub_preserves_foreground_omega():
    mat = numpy.array(
        [[-1.0, 0.5, 0.5], [0.5, -1.0, 0.5], [0.5, 0.5, -1.0]],
        dtype=float,
    )
    all_syn = numpy.array([[0, 1], [1, 0]])
    all_nsy = numpy.array([[0, 2], [1, 2], [2, 0], [2, 1]])
    codon_table = [["K", "AAA"], ["K", "AAG"], ["N", "AAC"]]
    out = main_simulate.apply_percent_biased_sub(
        mat=mat,
        percent_biased_sub=20,
        target_index=numpy.array([[2, 0], [2, 1]]),
        biased_aas=numpy.array(["K"]),
        codon_table=codon_table,
        codon_orders=numpy.array(["AAA", "AAG", "AAC"]),
        all_nsy_cdn_index=all_nsy,
        all_syn_cdn_index=all_syn,
        foreground_omega=2.0,
    )
    dnds = main_simulate.get_total_Q(out, all_nsy) / main_simulate.get_total_Q(out, all_syn)
    assert pytest.approx(dnds, rel=0, abs=1e-12) == 2.0
    numpy.testing.assert_allclose(out.sum(axis=1), [0.0, 0.0, 0.0], atol=1e-12)


def test_get_pyvolve_newick_marks_foreground_without_mutating_distances():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:0.1,B:0.2)R;", format=1))
    for node in tr.traverse():
        ete.add_features(node, **{"is_fg_t": False, "foreground_lineage_id_t": 0})
    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    a_node.is_fg_t = True
    a_node.foreground_lineage_id_t = 1
    out = main_simulate.get_pyvolve_newick(tr, "t")
    assert "#m1" in out
    assert pytest.approx(a_node.dist, rel=0, abs=1e-12) == 0.1


def test_scale_tree_multiplies_every_branch_length():
    tr = ete.PhyloNode("(A:1.0,B:2.0)R;", format=1)
    out = main_simulate.scale_tree(tr, 3.0)
    dists = sorted([n.dist for n in out.traverse()])
    assert dists == [0.0, 3.0, 6.0]

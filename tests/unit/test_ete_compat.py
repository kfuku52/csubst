import gzip

import pytest

from csubst import ete


def test_prop_roundtrip_and_delete():
    tr = ete.PhyloNode("(A:1,B:2)R;", format=1)
    a_node = [n for n in tr.traverse() if n.name == "A"][0]

    assert ete.get_prop(a_node, "custom_key") is None
    assert not ete.has_prop(a_node, "custom_key")

    ete.set_prop(a_node, "custom_key", 123)
    assert ete.get_prop(a_node, "custom_key") == 123
    assert ete.has_prop(a_node, "custom_key")

    ete.del_prop(a_node, "custom_key")
    assert ete.get_prop(a_node, "custom_key") is None
    assert not ete.has_prop(a_node, "custom_key")


def test_tree_relationship_wrappers():
    tr = ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1)
    x_node = [n for n in tr.traverse() if n.name == "X"][0]

    child_names = {n.name for n in ete.get_children(x_node)}
    sister_names = {n.name for n in ete.get_sisters(x_node)}
    descendant_names = {n.name for n in ete.get_descendants(x_node)}

    assert child_names == {"A", "B"}
    assert sister_names == {"C"}
    assert descendant_names == {"A", "B"}


def test_get_distance_wrapper_for_branch_length_and_topology():
    tr = ete.PhyloNode("(A:1,B:2)R;", format=1)
    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    b_node = [n for n in tr.traverse() if n.name == "B"][0]

    assert ete.get_distance(a_node, b_node, topology_only=False) == pytest.approx(3.0)
    assert ete.get_distance(a_node, b_node, topology_only=True) == pytest.approx(2.0)


def test_link_to_alignment_maps_leaf_sequences(tmp_path):
    tr = ete.PhyloNode("(A:1,B:1)R;", format=1)
    aln = tmp_path / "toy.fa"
    aln.write_text(">A description\nAAACCC\n>B extra\nGGGTTT\n", encoding="utf-8")

    ete.link_to_alignment(tr, alignment=str(aln), alg_format="fasta")

    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    b_node = [n for n in tr.traverse() if n.name == "B"][0]
    assert ete.get_prop(a_node, "sequence") == "AAACCC"
    assert ete.get_prop(b_node, "sequence") == "GGGTTT"


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
def test_link_to_alignment_ignores_sequence_whitespace(tmp_path, compressed):
    tr = ete.PhyloNode("(A:1,B:1)R;", format=1)
    aln = tmp_path / ("toy.fa.gz" if compressed else "toy.fa")
    content = b">A\tdescription\r\nAAA \tC\r\nCC\r\n>B extra\r\nGG G\tTTT\r\n"
    aln.write_bytes(gzip.compress(content) if compressed else content)

    ete.link_to_alignment(tr, alignment=aln)

    assert {leaf.name: ete.get_prop(leaf, "sequence") for leaf in ete.iter_leaves(tr)} == {
        "A": "AAACCC", "B": "GGGTTT",
    }


@pytest.mark.parametrize("second_header", ["A first", "A second", "A"])
def test_link_to_alignment_rejects_ambiguous_identifiers(tmp_path, second_header):
    tr = ete.PhyloNode("(A:1,B:1)R;", format=1)
    aln = tmp_path / "duplicate.fa"
    aln.write_text(f">A first\nAAA\n>{second_header}\nCCC\n", encoding="utf-8")

    with pytest.raises(ValueError, match='Duplicate FASTA header "A"'):
        ete.link_to_alignment(tr, alignment=aln)


def test_link_to_alignment_rejects_unsupported_format(tmp_path):
    tr = ete.PhyloNode("(A:1,B:1)R;", format=1)
    aln = tmp_path / "toy.fa"
    aln.write_text(">A\nAAACCC\n>B\nGGGTTT\n", encoding="utf-8")

    if ete.backend_name() == "ete4":
        with pytest.raises(ValueError, match="Unsupported alignment format"):
            ete.link_to_alignment(tr, alignment=str(aln), alg_format="phylip")
    else:
        # ete3 path delegates to native API and may raise a backend-specific exception.
        with pytest.raises(Exception):
            ete.link_to_alignment(tr, alignment=str(aln), alg_format="phylip")

import argparse

from csubst import ete
from csubst import tree


def make_tiny_tree():
    """Return the small labelled tree shared by site-level unit tests."""
    phylogeny = ete.PhyloNode("(B:1,(A:1,C:2)X:3)R;", format=1)
    return tree.add_numerical_node_labels(phylogeny)


def make_args(**kwargs):
    """Build the minimal argparse namespace used by parameter unit tests."""
    defaults = {
        "threads": 1,
        "float_digit": 4,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)

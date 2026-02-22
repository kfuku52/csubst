import numpy as np

from csubst import parser_phylobayes
from csubst import tree
from csubst import ete


def test_get_state_tensor_ignores_unknown_selected_branch_ids_in_ml_mode(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {
        "tree": tr,
        "phylobayes_dir": str(tmp_path),
        "num_input_state": 4,
        "num_input_site": 2,
        "float_type": np.float64,
        "ml_anc": True,
    }
    monkeypatch.setattr(parser_phylobayes.os, "listdir", lambda _path: [])
    out = parser_phylobayes.get_state_tensor(g=g, selected_branch_ids=np.array([9999], dtype=np.int64))
    assert out.shape == (len(list(tr.traverse())), 2, 4)
    assert out.sum() == 0

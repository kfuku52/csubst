import numpy
import pandas

from csubst import omega
from csubst import substitution
from csubst import tree
from csubst import ete


def _toy_sub_tensor():
    # shape = [branch, site, group, from, to]
    sub = numpy.zeros((3, 3, 2, 2, 2), dtype=numpy.float64)
    sub[0, 0, 0, 0, 1] = 0.3
    sub[1, 0, 0, 0, 1] = 0.4
    sub[2, 0, 0, 0, 1] = 0.5
    sub[0, 1, 0, 1, 0] = 0.2
    sub[1, 1, 0, 1, 0] = 0.3
    sub[2, 1, 0, 1, 0] = 0.1
    sub[0, 2, 1, 0, 0] = 0.6
    sub[1, 2, 1, 0, 0] = 0.4
    sub[2, 2, 1, 0, 0] = 0.2
    return sub


def _toy_cb():
    return pandas.DataFrame(
        {
            "branch_id_1": [0, 1],
            "branch_id_2": [1, 2],
        }
    )


def _toy_g(sub_tensor):
    num_branch = sub_tensor.shape[0]
    num_site = sub_tensor.shape[1]
    return {
        "threads": 1,
        "float_type": numpy.float64,
        "asrv": "no",
        "sub_sites": {"no": numpy.ones((num_branch, num_site), dtype=numpy.float64) / num_site},
        "N_ind_nomissing_gad": numpy.where(sub_tensor.sum(axis=(0, 1)) != 0),
        "N_ind_nomissing_ga": numpy.where(sub_tensor.sum(axis=(0, 1, 4)) != 0),
        "N_ind_nomissing_gd": numpy.where(sub_tensor.sum(axis=(0, 1, 3)) != 0),
    }


def test_calc_E_stat_mean_sparse_matches_dense_for_all_modes():
    dense = _toy_sub_tensor()
    sparse = substitution.dense_to_sparse_sub_tensor(dense, tol=0)
    cb = _toy_cb()
    g = _toy_g(dense)
    modes = ["spe2spe", "spe2any", "any2spe", "any2any"]
    for mode in modes:
        out_dense = omega.calc_E_stat(cb=cb, sub_tensor=dense, mode=mode, stat="mean", SN="N", g=g)
        out_sparse = omega.calc_E_stat(cb=cb, sub_tensor=sparse, mode=mode, stat="mean", SN="N", g=g)
        numpy.testing.assert_allclose(out_sparse, out_dense, atol=1e-12)


def test_get_exp_state_uses_branch_distance_props():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    labels = {n.name: ete.get_prop(n, "numerical_label") for n in tr.traverse()}
    num_node = max(labels.values()) + 1

    state = numpy.zeros((num_node, 1, 2), dtype=numpy.float64)
    state[labels["R"], 0, 0] = 1.0

    g = {
        "tree": tr,
        "state_pep": state.copy(),
        "state_cdn": state.copy(),
        "instantaneous_aa_rate_matrix": numpy.array([[-1.0, 1.0], [1.0, -1.0]], dtype=numpy.float64),
        "instantaneous_codon_rate_matrix": numpy.array([[-1.0, 1.0], [1.0, -1.0]], dtype=numpy.float64),
        "iqtree_rate_values": numpy.array([1.0], dtype=numpy.float64),
        "float_type": numpy.float64,
        "float_tol": 1e-12,
    }

    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    ete.set_prop(a_node, "Ndist", 0.5)
    ete.set_prop(a_node, "SNdist", 0.5)
    # B keeps missing Ndist/SNdist to confirm default=0 behavior.

    pep = omega.get_exp_state(g=g, mode="pep")
    cdn = omega.get_exp_state(g=g, mode="cdn")

    assert pep[labels["A"], 0, :].sum() > 0
    assert cdn[labels["A"], 0, :].sum() > 0
    numpy.testing.assert_allclose(pep[labels["B"], 0, :], [0.0, 0.0], atol=1e-12)
    numpy.testing.assert_allclose(cdn[labels["B"], 0, :], [0.0, 0.0], atol=1e-12)


def test_calibrate_dsc_skips_substitution_class_without_finite_pairs():
    combinatorial_substitutions = [
        "any2any",
        "any2spe",
        "any2dif",
        "dif2any",
        "dif2spe",
        "dif2dif",
        "spe2any",
        "spe2spe",
        "spe2dif",
    ]
    row = {"branch_id_1": 0}
    for sub in combinatorial_substitutions:
        row["dNC" + sub] = numpy.nan
        row["dSC" + sub] = numpy.nan
        row["omegaC" + sub] = numpy.nan
    cb = pandas.DataFrame([row])

    out = omega.calibrate_dsc(cb=cb.copy())

    for sub in combinatorial_substitutions:
        assert "dSC" + sub in out.columns
        assert "omegaC" + sub in out.columns
        assert "dSC" + sub + "_nocalib" not in out.columns
        assert "omegaC" + sub + "_nocalib" not in out.columns

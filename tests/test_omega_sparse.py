import numpy
import pandas

from csubst import omega
from csubst import substitution


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


"""Real worker parity for scan permutations across tensor representations."""

import numpy as np
import pandas as pd
import pytest

from csubst import substitution_scan
from scan_fixtures import make_scan_context, make_endpoint_scan_context


@pytest.mark.slow
@pytest.mark.process
@pytest.mark.parametrize('exposure,calibration,tensor_kind', [
    ('state_aware', 'full_scan', 'dense'),
    ('endpoint', 'candidate_fixed', 'sparse'),
    ('endpoint', 'full_scan', 'stored'),
])
def test_scan_parallel_permutation_matches_single_thread_result(exposure, calibration, tensor_kind):
    from csubst import substitution, site_storage

    context = make_endpoint_scan_context if exposure == 'endpoint' else make_scan_context
    serial_g, serial_tensor = context()
    serial_g.update(threads=1, scan_n_permutations=4)
    serial_g["scan_rate_exposure"] = exposure
    serial_g["scan_pvalue_calibration"] = calibration
    serial_g["scan_permutation_seed"] = 3
    parallel_g = dict(serial_g, threads=2)
    parallel_tensor = serial_tensor
    if tensor_kind == 'sparse':
        parallel_tensor = substitution.dense_to_sparse_sub_tensor(serial_tensor)
    elif tensor_kind == 'stored':
        parallel_tensor = site_storage.SiteEventTensor(serial_tensor.shape)
        parallel_tensor.write_block(0, np.moveaxis(serial_tensor, 1, 0))
        parallel_tensor.seal()

    serial_df, _ = substitution_scan.scan_substitutions(g=serial_g, ON_tensor=serial_tensor)
    parallel_df, _ = substitution_scan.scan_substitutions(g=parallel_g, ON_tensor=parallel_tensor)

    assert parallel_df.iloc[0]["scan_permutation_n_jobs"] == 2
    assert parallel_df.iloc[0]["scan_permutation_success_count"] == 4
    cols = [c for c in serial_df if c not in ["scan_permutation_n_jobs", "scan_permutation_backend"]]
    pd.testing.assert_frame_equal(serial_df[cols], parallel_df[cols])

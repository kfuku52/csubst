"""Compare production boundaries directly against the fixed pre-change source."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np  # noqa: E402

from bench_common import benchmark_lock, environment  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scan_checks(baseline):
    from csubst import substitution_scan as scan

    original = load(baseline / 'csubst/substitution_scan.py', 'original_scan_boundaries')
    trial = load(ROOT / 'reports/performance_round2_20261022/proposal02.py', 'rate_boundary_cases')
    current = scan._rate_summary
    conversions = 0

    def numeric(**options):
        nonlocal conversions
        frame = options['candidate_events']
        if 'event_pp' in frame:
            # Posterior extraction supplies one mass per branch. Use pandas'
            # reference aggregation here to prepare that producer-compatible
            # input even when the boundary fixture contains called duplicates.
            frame = frame.groupby('branch_id', as_index=False)['event_pp'].sum()
            try:
                ids = frame['branch_id'].to_numpy(dtype=np.int64)
            except (ValueError, TypeError):
                pass
            else:
                if (ids >= 0).all():
                    options['candidate_events'] = scan._PosteriorBranchValues(frame, max(1, int(ids.max(initial=0)) + 1))
                    conversions += 1
        return current(**options)

    scan._rate_summary = original._rate_summary
    try:
        metadata = trial.edge_checks(current)
        vectors = trial.edge_checks(numeric)
    finally:
        scan._rate_summary = current
    assert conversions > 100
    return dict(metadata=metadata, numeric_vectors=vectors, numeric_path_calls=conversions, exact=True)


def outcome(function, arrays, config):
    try:
        return 'result', function(*arrays, config)
    except Exception as error:
        return 'error', type(error).__name__, str(error)


def cv_checks(baseline):
    from csubst import epistasis

    original = load(baseline / 'csubst/epistasis.py', 'original_crossfit_boundaries')
    rng = np.random.default_rng(22010)
    config = dict(epistasis_beta_auto=True, epistasis_joint_auto=True,
                  epistasis_joint_alpha_grid=[0., .5, 1.], epistasis_clip_auto=True,
                  epistasis_joint_clip_grid=[1.5, 3.], asrv_dirichlet_alpha=1.)
    checks = []
    for case in ('normal', 'zero_context', 'zero_counts', 'missing_row', 'readonly', 'buffer',
                 'two_partitions', 'nan_counts', 'overflow', 'wrong_shape', 'few_clades',
                 'fixed_zero', 'all_undefined', 'no_features', 'ties', 'float32'):
        counts = rng.poisson(.5, size=(25, 11)).astype(float)
        context, features = rng.normal(size=(25, 2)), rng.normal(size=(11, 2))
        masks, groups, partitions = rng.random((25, 11)) > .2, np.arange(25) % 5, np.zeros(25, dtype=int)
        settings = dict(config)
        if case == 'zero_context':
            context[:] = 0
        elif case == 'zero_counts':
            counts[:] = 0
        elif case == 'missing_row':
            masks[0] = False
        elif case == 'buffer':
            groups[0] = -1
        elif case == 'two_partitions':
            partitions = np.arange(25) % 2
        elif case == 'nan_counts':
            counts[0, 0] = np.nan
        elif case == 'overflow':
            context[:] = np.finfo(float).max
            features[:] = 2
        elif case == 'wrong_shape':
            masks = masks[:, :-1]
        elif case == 'few_clades':
            groups %= 2
        elif case == 'fixed_zero':
            settings.update(epistasis_beta_auto=False, epistasis_beta_value=0., epistasis_joint_auto=False,
                            epistasis_clip_auto=False, epistasis_clip_value=3.)
        elif case == 'all_undefined':
            counts[:] = 0
            for index in range(25):
                counts[index, groups[index]] = 1
            settings['epistasis_joint_alpha_grid'] = [0.]
        elif case == 'no_features':
            context, features = context[:, :0], features[:, :0]
        elif case == 'ties':
            counts[:] = 1
            context[:] = 0
            masks[:] = True
        elif case == 'float32':
            counts, context, features = [array.astype(np.float32) for array in (counts, context, features)]
        arrays = [counts, context, features, masks, groups, partitions]
        if case == 'readonly':
            for array in arrays:
                array.flags.writeable = False
        with np.errstate(over='ignore', invalid='ignore'):
            before = outcome(original.crossfit, arrays, settings)
            after = outcome(epistasis.crossfit, arrays, settings)
        np.testing.assert_equal(before, after)
        if before[0] == 'result':
            for name, value in before[1].items():
                if isinstance(value, np.ndarray):
                    assert value.tobytes() == after[1][name].tobytes()
        checks.append(dict(case=case, exact=True, outcome=before[0]))
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    with benchmark_lock():
        record = dict(environment=environment(), scan=scan_checks(args.baseline), cv=cv_checks(args.baseline))
    args.result.write_text(json.dumps(record, indent=2) + '\n')
    print('Exact baseline comparisons passed:', len(record['scan']['metadata']),
          len(record['scan']['numeric_vectors']), len(record['cv']))

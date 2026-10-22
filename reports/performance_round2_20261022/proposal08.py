"""Evaluate compressed unit membership and segmented support maxima.

Run: python reports/performance_round2_20261022/proposal08.py
Only report artifacts are written; product functions are restored after capture.
"""

import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, substitution_scan as scan  # noqa: E402

ORIGINAL = scan._summarize_unit_support
TRACE = None


class SupportPlan(list):
    """Private, assignment-local prepared form; no allocation by maximum ID."""

    def __init__(self, units):
        super().__init__(units)
        self.lengths = np.asarray([len(ids) for _, _, ids in self], dtype=np.int64)
        self.active = np.flatnonzero(self.lengths)
        self.starts = np.cumsum(self.lengths) - self.lengths
        self.flat = (np.concatenate([ids for _, _, ids in self])
                     if len(self) else np.empty(0, dtype=np.int64))
        self.unique, self.inverse = np.unique(self.flat, return_inverse=True)
        self.nbytes = sum(value.nbytes for value in (
            self.lengths, self.active, self.starts, self.flat, self.unique, self.inverse))
        for value in (self.lengths, self.active, self.starts, self.flat, self.unique, self.inverse):
            value.flags.writeable = False


def segmented_summary(candidate_events, units_df, target_class, min_event_pp, unit_plan=None):
    """Specialize only an explicitly prepared private plan; public path stays intact."""
    if not isinstance(unit_plan, SupportPlan) or str(target_class) != 'fg' or len(units_df) == 0:
        return ORIGINAL(candidate_events, units_df, target_class, min_event_pp, unit_plan)
    branch_event = candidate_events.groupby('branch_id')['event_pp'].sum().to_dict()
    plan = unit_plan
    values = np.fromiter((float(branch_event.get(int(bid), 0.0)) for bid in plan.unique),
                         dtype=np.float64, count=len(plan.unique))[plan.inverse]
    positive = values > 0
    supported = positive & (values >= float(min_event_pp)) if positive.any() else positive
    maxima = np.zeros(len(plan), dtype=np.float64)
    if len(plan.active):
        maxima[plan.active] = np.maximum.reduceat(np.where(supported, values, 0.), plan.starts[plan.active])
    if TRACE is not None:
        TRACE['prepared_calls'] += 1
        TRACE['segmented_max_calls'] += int(len(plan.active) > 0)
    support_pp = []
    support_units = []
    analyzable_lineages = {}
    support_lineages = set()
    for index, (unit_id, lineage_value, fg_ids) in enumerate(plan):
        if fg_ids.size > 0:
            analyzable_lineages[lineage_value] = None
        if maxima[index] > 0:
            support_units.append(int(unit_id))
            support_pp.append(float(maxima[index]))
            support_lineages.add(lineage_value)
    unit_total = int(units_df.shape[0])
    support_count = len(support_units)
    lineage_total = len(analyzable_lineages)
    support_lineage_count = len(support_lineages)
    return {
        'unit_total': unit_total,
        'support_unit_count': int(support_count),
        'support_fraction': (support_count / unit_total) if unit_total > 0 else np.nan,
        'support_pp_sum': float(np.sum(support_pp)) if len(support_pp) else 0.0,
        'support_pp_mean': float(np.mean(support_pp)) if len(support_pp) else np.nan,
        'support_unit_ids': ','.join(str(v) for v in support_units),
        'support_branch_ids': ','.join(str(v) for v in np.unique(plan.flat[supported]).tolist()),
        'lineage_total': lineage_total,
        'support_lineage_count': support_lineage_count,
        'support_lineage_fraction': (support_lineage_count / lineage_total) if lineage_total > 0 else np.nan,
        'support_lineage_ids': scan._format_lineage_id_list(v for v in analyzable_lineages if v in support_lineages),
    }


def outcome(function, options):
    try:
        return 'result', function(**options)
    except Exception as exc:
        return 'error', type(exc).__name__, str(exc)


def equal(before, after):
    np.testing.assert_equal(before, after)
    if before[0] == 'result':
        assert before[1].keys() == after[1].keys()
        for key in before[1]:
            assert type(before[1][key]) is type(after[1][key]), key


def edge_checks():
    checks = []
    plans = {
        'empty_units': [],
        'empty_memberships': [(1, 'A', []), (2, 'B', [])],
        'mixed_empty': [(1, 'A', []), (2, 'B', [2]), (3, 'A', []), (4, 'A', [5, 2])],
        'duplicates_and_gaps': [(9, 'Z', [2, 2, -1, 10**12]), (3, 'A', [5, -1, 5])],
        'lineage_csv': [(9, 'a,b', [2]), (3, 'a"b', [5]), (8, '日本\n語', [-1]), (1, 'a,b', [2])],
    }
    event_cases = {
        'ordinary': {'branch_id': [2, 5, -1, 10**12], 'event_pp': [.75, .1, .6, .5]},
        'empty': {'branch_id': [], 'event_pp': []},
        'nan_zero_negative': {'branch_id': [2, 5, -1, 10**12], 'event_pp': [np.nan, 0., -.1, -np.inf]},
        'inf': {'branch_id': [2, 5, -1], 'event_pp': [np.inf, -np.inf, np.nan]},
        'compensated_duplicates': {'branch_id': [2, 2, 2, 5, 5], 'event_pp': [1e16, 1., -1e16, .3, .2]},
    }
    for plan_name, items in plans.items():
        raw = [(unit, lineage, np.asarray(ids, dtype=np.int64)) for unit, lineage, ids in items]
        frame = pd.DataFrame({'unit_id': [i[0] for i in raw], 'lineage_value': [i[1] for i in raw]})
        for event_name, events in event_cases.items():
            for threshold in (0., .5, np.nextafter(.5, 0), np.nextafter(.5, 1), np.nan, np.inf, -np.inf, 'bad'):
                options = dict(candidate_events=pd.DataFrame(events), units_df=frame, target_class='fg',
                               min_event_pp=threshold, unit_plan=raw)
                before = outcome(ORIGINAL, options)
                equal(before, outcome(segmented_summary, dict(options, unit_plan=SupportPlan(raw))))
                # Unprepared public calls preserve the original path.
                equal(before, outcome(segmented_summary, options))
                checks.append(dict(plan=plan_name, event=event_name, threshold=str(threshold),
                                   exact_outcome=True, outcome=before[0]))
    raw = [(1, 'A', np.array([2]))]
    frame = pd.DataFrame({'unit_id': [1], 'lineage_value': ['A'], 'fg_branch_ids': ['2']})
    for mode in ('bad_target', 'bad_event_column', 'bad_unit_id', 'none_plan'):
        options = dict(candidate_events=pd.DataFrame({'branch_id': [2], 'event_pp': [.75]}),
                       units_df=frame, target_class='fg', min_event_pp=.5, unit_plan=raw)
        if mode == 'bad_target':
            options['target_class'] = 'sister'
        elif mode == 'bad_event_column':
            options['candidate_events'] = options['candidate_events'].drop(columns='event_pp')
        elif mode == 'bad_unit_id':
            options['unit_plan'] = [('bad', 'A', np.array([2]))]
        else:
            options['unit_plan'] = None
        before = outcome(ORIGINAL, options)
        after = dict(options)
        if after['unit_plan'] is not None:
            after['unit_plan'] = SupportPlan(after['unit_plan'])
        equal(before, outcome(segmented_summary, after))
        checks.append(dict(case=mode, exact_outcome=True, outcome=before[0]))
    oracle_plan = [(3, 'lineage,a', np.array([9, 2, 2])), (1, 'B', np.array([5])),
                   (2, 'A', np.array([], dtype=np.int64)), (8, 'lineage,a', np.array([7]))]
    oracle_units = pd.DataFrame({'unit_id': [3, 1, 2, 8]})
    oracle_events = pd.DataFrame({'branch_id': [9, 2, 5, 7, 7], 'event_pp': [.25, .75, .49, .5, .1]})
    expected = dict(unit_total=4, support_unit_count=2, support_fraction=.5,
                    support_pp_sum=1.35, support_pp_mean=.675, support_unit_ids='3,8',
                    support_branch_ids='2,7', lineage_total=2, support_lineage_count=1,
                    support_lineage_fraction=.5, support_lineage_ids='"lineage,a"')
    for function, plan in ((ORIGINAL, oracle_plan), (segmented_summary, SupportPlan(oracle_plan))):
        equal(('result', expected), outcome(function, dict(
            candidate_events=oracle_events, units_df=oracle_units, target_class='fg',
            min_event_pp=.5, unit_plan=plan)))
    checks.append(dict(case='independent_hand_calculated_support_oracle', exact_outcome=True))
    return checks


def run_batch(calls, prepared=False, trace=False):
    global TRACE
    TRACE = {'prepared_calls': 0, 'segmented_max_calls': 0} if trace else None
    plans = {}
    for call in calls:
        raw = call['unit_plan']
        if id(raw) not in plans:
            plans[id(raw)] = SupportPlan(raw) if prepared else raw
    outputs = []
    for call in calls:
        options = dict(call, unit_plan=plans[id(call['unit_plan'])])
        outputs.append(outcome(segmented_summary if prepared else ORIGINAL, options))
    stats = dict(TRACE) if trace else {}
    TRACE = None
    stats.update(plan_count=len(plans), extra_retained_array_bytes=sum(
        plan.nbytes for plan in plans.values()) if prepared else 0)
    return outputs, stats


def benchmark(calls):
    before, _ = run_batch(calls)
    after, trace = run_batch(calls, prepared=True, trace=True)
    for left, right in zip(before, after, strict=True):
        equal(left, right)
    assert trace['prepared_calls'] == len(calls)
    runs = {'before': [], 'segmented': []}
    for repeat in range(6):
        for label in (('before', 'segmented') if repeat % 2 == 0 else ('segmented', 'before')):
            result, timing = measure(run_batch, calls, label == 'segmented')
            for left, right in zip(before, result[0], strict=True):
                equal(left, right)
            if repeat:
                runs[label].append(timing)
    return dict(call_count=len(calls), exact_fields_types=True, path_trace=trace,
                runs=runs, summary={name: summarize(times) for name, times in runs.items()})


def real_case(dataset, workdir):
    data = workdir / 'data'
    data.mkdir()
    names = [dataset + '.alignment.fa', dataset + '.tree.nwk', dataset + '.foreground.txt']
    names += [dataset + '.alignment.fa.' + suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')]
    for name in names:
        shutil.copyfile(ROOT / 'csubst/dataset' / name, data / name)
    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')), '--outdir', str(workdir / 'out'),
            '--threads', '1', '--blas_threads', '1', '--scan_site_plot', 'no',
            '--scan_pvalue_calibration', 'none', '--scan_n_permutations', '0',
            '--random_seed', '8', '--float_digit', '12']
    for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + suffix, str(data / (dataset + '.alignment.fa.' + suffix))]
    calls, plans, units = [], {}, {}

    def intercept(*args, **kwargs):
        call = dict(inspect.signature(ORIGINAL).bind(*args, **kwargs).arguments)
        call.setdefault('unit_plan', None)
        original_plan = call['unit_plan']
        if original_plan is None:
            original_plan = scan._scan_unit_support_plan(call['units_df'])
        key = id(original_plan)
        if key not in plans:
            plans[key] = [(unit, lineage, branches.copy()) for unit, lineage, branches in original_plan]
        unit_key = id(call['units_df'])
        if unit_key not in units:
            units[unit_key] = call['units_df'].copy(deep=True)
        copied = dict(call, candidate_events=call['candidate_events'].copy(deep=True),
                      units_df=units[unit_key], unit_plan=plans[key])
        calls.append(copied)
        return ORIGINAL(*args, **kwargs)

    scan._summarize_unit_support = intercept
    try:
        with (workdir / 'capture.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            cli.main(argv)
    finally:
        scan._summarize_unit_support = ORIGINAL
    assert calls, 'real scan never invoked support reducer'
    result = benchmark(calls)
    result.update(unit_counts=[len(plan) for plan in plans.values()],
                  membership_counts=[sum(len(ids) for _, _, ids in plan) for plan in plans.values()],
                  maximum_membership_per_unit=max(len(ids) for plan in plans.values() for _, _, ids in plan),
                  input_sha256={name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in names})
    return result


def synthetic_case():
    rng = np.random.default_rng(29)
    raw = [(unit, 'lineage_' + str(unit % 10), rng.choice(2000, 100, replace=False)) for unit in range(200)]
    frame = pd.DataFrame({'unit_id': np.arange(200), 'lineage_value': ['lineage_' + str(i % 10) for i in range(200)]})
    calls = [dict(candidate_events=pd.DataFrame({'branch_id': np.arange(2000), 'event_pp': rng.random(2000)}),
                  units_df=frame, target_class='fg', min_event_pp=.5, unit_plan=raw) for _ in range(40)]
    result = benchmark(calls)
    result.update(unit_count=200, memberships=20_000, unique_branch_ids=2000,
                  description='artificial large overlapping unit memberships; not a biological replicate')
    return result


def main():
    result = dict(proposal=8, environment=environment(),
                  command='python reports/performance_round2_20261022/proposal08.py',
                  source_function_sha256=hashlib.sha256(inspect.getsource(ORIGINAL).encode()).hexdigest(),
                  scope='support reducer batches; one cold prepared plan construction per batch; not whole CLI',
                  memory_scope='retained array bytes only, excluding Python objects and temporary arrays; RSS not measured',
                  warmups=1, measured_repeats=5)
    with benchmark_lock():
        result['edge_checks'] = edge_checks()
        result['cases'] = {}
        for dataset in ('PGK', 'PEPC'):
            with tempfile.TemporaryDirectory(prefix='csubst-proposal08-') as directory:
                result['cases'][dataset] = real_case(dataset, Path(directory))
        result['synthetic_large_membership'] = synthetic_case()
    (HERE / 'proposal08.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, case in list(result['cases'].items()) + [('synthetic', result['synthetic_large_membership'])]:
        print(name, case['summary'])


if __name__ == '__main__':
    main()

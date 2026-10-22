"""Measure assignment-local numeric metadata/mask reuse without product edits.

Run: python reports/performance_round2_20261022/proposal02.py
"""

import argparse
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
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


class BranchData:
    """Borrow immutable metadata only during one assignment; no global cache."""

    def __init__(self, branch_meta):
        self.branch_meta = branch_meta
        self.branch_ids = None
        self.lengths = None
        self.parent_ids = None
        self.owned_masks = []

    def masks(self, target_branch_ids, other_branch_ids):
        target_set = set(int(v) for v in np.asarray(target_branch_ids, dtype=np.int64).reshape(-1).tolist())
        if self.branch_ids is None:
            self.branch_ids = self.branch_meta['branch_id'].astype(int).to_numpy(copy=False)
        is_target = np.isin(self.branch_ids, np.array(sorted(target_set), dtype=np.int64))
        if other_branch_ids is None:
            other_set = set(int(v) for v in self.branch_ids.tolist()).difference(target_set)
        else:
            other_set = set(int(v) for v in np.asarray(other_branch_ids, dtype=np.int64).reshape(-1).tolist())
            other_set = other_set.difference(target_set)
        is_other = np.isin(self.branch_ids, np.array(sorted(other_set), dtype=np.int64))
        self.owned_masks.extend((is_target, is_other))
        return self, is_target, is_other

    def get_lengths(self, rate_length):
        if self.lengths is None:
            length_col = scan._length_column(rate_length)
            chosen = self.branch_meta[length_col].to_numpy(dtype=np.float64, copy=False)
            raw = self.branch_meta['raw_length'].to_numpy(dtype=np.float64, copy=False)
            sn = self.branch_meta['sn_rescaled_length'].to_numpy(dtype=np.float64, copy=False)
            nonsyn = self.branch_meta['n_rescaled_length'].to_numpy(dtype=np.float64, copy=False)
            self.lengths = chosen, raw, sn, nonsyn
        return self.lengths

    def get_parents(self):
        if self.parent_ids is None:
            self.parent_ids = self.branch_meta['parent_id'].astype(int).to_numpy(copy=False)
        return self.parent_ids

    def array_bytes(self):
        assert self.branch_ids is not None
        return self.branch_ids.nbytes + sum(mask.nbytes for mask in self.owned_masks)


def make_prototype():
    original = inspect.getsource(scan._rate_summary)
    source = original.replace('    score_only=False,\n', '    score_only=False,\n    prepared_branch=None,\n')
    start = source.index('    target_set = ')
    stop = source.index('    event_ids = ')
    prefix = source[start:stop]
    source = source[:start] + ('    if prepared_branch is None:\n' +
                              ''.join('    ' + line for line in prefix.splitlines(keepends=True)) +
                              '    else:\n'
                              '        branch_data, is_target, is_other = prepared_branch\n'
                              '        branch_ids = branch_data.branch_ids\n') + source[stop:]
    start = source.index('    length_col = _length_column(rate_length)')
    stop = source.index('    missing_exposure = ')
    lengths = source[start:stop]
    source = source[:start] + ('    if prepared_branch is None:\n' +
                              ''.join('    ' + line for line in lengths.splitlines(keepends=True)) +
                              '    else:\n'
                              '        chosen_length, raw_length, sn_length, n_length = branch_data.get_lengths(rate_length)\n') + source[stop:]
    source = source.replace('            parent_ids = branch_meta["parent_id"].astype(int).to_numpy(copy=False)',
                            '            parent_ids = (branch_meta["parent_id"].astype(int).to_numpy(copy=False)\n'
                            '                          if prepared_branch is None else branch_data.get_parents())')
    namespace = dict(scan.__dict__, BranchData=BranchData)
    exec(compile(source, '<proposal02-prepared-rate-summary>', 'exec'), namespace)
    reducer = namespace['_rate_summary']
    original_fixed = inspect.getsource(scan._candidate_fixed_permutation_scores)
    fixed = original_fixed.replace('    branch_sets = {}', '    branch_sets = {}\n    branch_data = None\n    branch_masks = {}')
    fixed = fixed.replace('        rate = _rate_summary(\n',
                          '        if branch_data is None:\n'
                          '            branch_data = BranchData(branch_meta)\n'
                          '        if branch_key not in branch_masks:\n'
                          '            branch_masks[branch_key] = branch_data.masks(target_branch_ids, other_branch_ids)\n'
                          '        rate = _rate_summary(\n'
                          '            prepared_branch=branch_masks[branch_key],\n')
    exec(compile(fixed, '<proposal02-prepared-fixed-candidates>', 'exec'), namespace)
    return reducer, namespace['_candidate_fixed_permutation_scores'], {
        'rate_summary': hashlib.sha256(original.encode()).hexdigest(),
        'fixed_candidates': hashlib.sha256(original_fixed.encode()).hexdigest(),
    }


def outcome(function, options):
    try:
        return ('ok', function(**options))
    except Exception as exc:
        return 'error', type(exc).__name__, str(exc)


def edge_checks(reducer):
    branch = pd.DataFrame({'branch_id': [3, 1, 7], 'parent_id': [0, 0, 0],
                           'raw_length': [0.1, 0.2, 0.3],
                           'sn_rescaled_length': [0.2, 0.4, 0.6],
                           'n_rescaled_length': [0.3, 0.6, 0.9]})
    states = np.full((8, 2, 2), 0.5)
    events = pd.DataFrame({'branch_id': [1, 3, 3, 9], 'event_pp': [0.75, 0.1, np.nan, 0.5]})
    base = dict(candidate_events=events, branch_meta=branch, state_nsy=states, site=0,
                from_ids=np.array([0]), to_ids=np.array([1]), target_branch_ids=np.array([1]),
                other_branch_ids=np.array([3, 7]), rate_length='raw', rate_exposure='raw_branch_length',
                q_matrix=np.array([[-1., 1.], [1., -1.]]))
    checks = []
    for exposure in ('raw_branch_length', 'state_aware', 'q_weighted'):
        for length in ('raw', 'sn_rescaled', 'n_rescaled'):
            for mode in ('normal', 'overlap_duplicates', 'other_none', 'empty_groups', 'empty_events',
                         'nan_event', 'negative_event', 'zero_target', 'missing_state'):
                options = dict(base, rate_exposure=exposure, rate_length=length,
                               branch_meta=branch.copy(), state_nsy=states.copy(), candidate_events=events.copy())
                if mode == 'overlap_duplicates':
                    options.update(target_branch_ids=[1, 1, 9], other_branch_ids=[1, 3, 3, 7])
                elif mode == 'other_none':
                    options['other_branch_ids'] = None
                elif mode == 'empty_groups':
                    options.update(target_branch_ids=[], other_branch_ids=[])
                elif mode == 'empty_events':
                    options['candidate_events'] = events.iloc[:0]
                elif mode == 'nan_event':
                    options['candidate_events']['event_pp'] = np.nan
                elif mode == 'negative_event':
                    options['candidate_events'].loc[0, 'event_pp'] = -1
                elif mode == 'zero_target':
                    options['branch_meta'].loc[1, ['raw_length', 'sn_rescaled_length', 'n_rescaled_length']] = 0
                elif mode == 'missing_state':
                    options['state_nsy'][0, 0] = 0
                data = BranchData(options['branch_meta'])
                prepared = data.masks(options['target_branch_ids'], options['other_branch_ids'])
                for score_only in (False, True):
                    options['score_only'] = score_only
                    before = outcome(scan._rate_summary, options)
                    np.testing.assert_equal(before, outcome(reducer, dict(options, prepared_branch=prepared)))
                    np.testing.assert_equal(before, outcome(reducer, options))
                checks.append({'exposure': exposure, 'length': length, 'case': mode})
    original_expected = scan.scan_endpoint.expected_events
    try:
        for mode in ('normal', 'positive_zero', 'missing', 'zero_control', 'bad_order', 'no_context', 'bad_length'):
            vals = np.array([0.1, 0.2, 0.3])
            missing = np.zeros(3, dtype=bool)
            if mode in ('positive_zero', 'missing'):
                vals[1] = 0
                missing[1] = mode == 'missing'
            elif mode == 'zero_control':
                vals[[0, 2]] = 0
            def expected(*args, values=vals, absent=missing):
                return values.copy(), absent.copy(), np.full(3, 'ok', dtype='U24')
            scan.scan_endpoint.expected_events = expected
            context = {'branch_ids': np.array([3, 1, 7]), 'model': 'synthetic'}
            options = dict(base, rate_exposure='endpoint', endpoint_context=context)
            if mode == 'bad_order':
                options['endpoint_context'] = dict(context, branch_ids=np.array([1, 3, 7]))
            elif mode == 'no_context':
                options['endpoint_context'] = None
            elif mode == 'bad_length':
                options['rate_length'] = 'n_rescaled'
            data = BranchData(branch)
            prepared = data.masks(options['target_branch_ids'], options['other_branch_ids'])
            for score_only in (False, True):
                options['score_only'] = score_only
                np.testing.assert_equal(outcome(scan._rate_summary, options),
                                        outcome(reducer, dict(options, prepared_branch=prepared)))
            checks.append({'exposure': 'endpoint_stub', 'case': mode})
    finally:
        scan.scan_endpoint.expected_events = original_expected
    def prepared_call(**options):
        data = BranchData(options['branch_meta'])
        prepared = data.masks(options['target_branch_ids'], options.get('other_branch_ids'))
        return reducer(**options, prepared_branch=prepared)
    for mode in ('invalid_target', 'invalid_branch_id', 'both_invalid', 'missing_event_column',
                 'missing_length_column', 'invalid_length', 'invalid_parent'):
        options = dict(base, branch_meta=branch.copy(), candidate_events=events.copy())
        if mode in ('invalid_target', 'both_invalid'):
            options['target_branch_ids'] = ['bad']
        if mode in ('invalid_branch_id', 'both_invalid'):
            options['branch_meta']['branch_id'] = ['bad', '1', '7']
        elif mode == 'missing_event_column':
            options['candidate_events'] = events.drop(columns='event_pp')
        elif mode == 'missing_length_column':
            options['branch_meta'] = branch.drop(columns='sn_rescaled_length')
        elif mode == 'invalid_length':
            options['rate_length'] = 'bad'
        elif mode == 'invalid_parent':
            options['rate_exposure'] = 'state_aware'
            options['branch_meta']['parent_id'] = ['bad', '0', '0']
        before = outcome(scan._rate_summary, options)
        np.testing.assert_equal(before, outcome(prepared_call, options))
        np.testing.assert_equal(before, outcome(reducer, options))
        checks.append({'exposure': 'failure_order', 'case': mode, 'outcome': before[0]})
    return checks


class Captured(Exception):
    pass


def copy_inputs(dataset, workdir):
    data = workdir / (dataset + '_inputs')
    data.mkdir()
    names = [dataset + '.alignment.fa', dataset + '.tree.nwk', dataset + '.foreground.txt']
    names += [dataset + '.alignment.fa.' + suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')]
    for name in names:
        shutil.copyfile(ROOT / 'csubst/dataset' / name, data / name)
    return data, {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in names}


def argv_for(dataset, data, outdir):
    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')), '--outdir', str(outdir),
            '--threads', '1', '--blas_threads', '1', '--scan_site_plot', 'no',
            '--scan_pvalue_calibration', 'candidate_fixed', '--scan_n_permutations', '50',
            '--random_seed', '8', '--float_digit', '12']
    for key in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + key, str(data / (dataset + '.alignment.fa.' + key))]
    return argv


def real_case(dataset, prototype, workdir):
    original_chunk = scan._run_scan_permutation_chunk
    original_fixed = scan._candidate_fixed_permutation_scores
    result = {}
    def compare(*args, **kwargs):
        values = inspect.signature(original_chunk).bind(*args, **kwargs).arguments
        runs = {'before': [], 'prepared': []}
        for repeat in range(4):
            outputs = {}
            for label in (('before', 'prepared') if repeat % 2 == 0 else ('prepared', 'before')):
                scan._candidate_fixed_permutation_scores = original_fixed if label == 'before' else prototype
                try:
                    outputs[label], timing = measure(original_chunk, *args, **kwargs)
                finally:
                    scan._candidate_fixed_permutation_scores = original_fixed
                if repeat:
                    runs[label].append(timing)
            np.testing.assert_equal(outputs['before'], outputs['prepared'])
        assert outputs['before']['failure_count'] == 0
        empty = dict(values, observed_df=values['observed_df'].iloc[:0], observed_keys=set())
        before = original_chunk(**empty)
        scan._candidate_fixed_permutation_scores = prototype
        try:
            after = original_chunk(**empty)
        finally:
            scan._candidate_fixed_permutation_scores = original_fixed
        np.testing.assert_equal(before, after)
        branch_count = len(values['branch_meta'])
        result.update(candidate_rows=len(values['observed_df']), branch_count=branch_count,
                      permutations=len(values['permutation_indices']),
                      success_count=outputs['before']['success_count'], failure_count=0,
                      exact_scores_keys_diagnostics=True, empty_family_exact_parity=True,
                      metadata_branch_ids_and_two_mask_bytes_per_family=branch_count * 10,
                      runs=runs, summary={key: summarize(samples) for key, samples in runs.items()})
        raise Captured
    data, hashes = copy_inputs(dataset, workdir)
    scan._run_scan_permutation_chunk = compare
    try:
        with (workdir / (dataset + '.log')).open('w') as log, contextlib.redirect_stdout(log):
            try:
                cli.main(argv_for(dataset, data, workdir / dataset))
            except Captured:
                pass
    finally:
        scan._run_scan_permutation_chunk = original_chunk
        scan._candidate_fixed_permutation_scores = original_fixed
    assert result
    result['input_sha256'] = hashes
    return result


def cli_case(dataset, workdir):
    data, hashes = copy_inputs(dataset, workdir)
    runs = {'before': [], 'prepared': []}
    commands = {}
    for repeat in range(4):
        outputs = {}
        for label in (('before', 'prepared') if repeat % 2 == 0 else ('prepared', 'before')):
            outdir = workdir / f'{dataset}_{repeat}_{label}'
            command = ['/usr/bin/time', '-p', '-l', sys.executable, str(Path(__file__).resolve()),
                       '--worker', label, '--dataset', dataset, '--data', str(data), '--outdir', str(outdir)]
            with (workdir / f'{dataset}_{repeat}_{label}.log').open('w') as log:
                completed, timing = measure(subprocess.run, command, stdout=log,
                                            stderr=subprocess.PIPE, text=True, check=True)
            usage = completed.stderr
            timing['cpu_s'] = sum(float(re.search(rf'^{part}\s+([\d.]+)$', usage, re.MULTILINE).group(1))
                                  for part in ('user', 'sys'))
            timing['peak_rss_bytes'] = int(re.search(r'(\d+)\s+maximum resident set size', usage).group(1))
            if repeat:
                runs[label].append(timing)
            outputs[label] = outdir
            commands[label] = ['$PYTHON', 'reports/performance_round2_20261022/proposal02.py',
                               '--worker', label, '--dataset', dataset, '--data', '$COPIED_INPUTS',
                               '--outdir', '$FRESH_OUTDIR']
        before = outputs['before'] / 'csubst_scan.tsv'
        after = outputs['prepared'] / 'csubst_scan.tsv'
        assert before.exists() and after.exists()
        for name in ('csubst_scan.tsv', 'csubst_scan_units.tsv'):
            pd.testing.assert_frame_equal(pd.read_csv(outputs['before'] / name, sep='\t'),
                                          pd.read_csv(outputs['prepared'] / name, sep='\t'), check_exact=True)
        for name in ('csubst_scan_calibration.json', 'csubst_scan_inference.json'):
            before_json = json.loads((outputs['before'] / name).read_text())
            after_json = json.loads((outputs['prepared'] / name).read_text())
            np.testing.assert_equal(before_json, after_json)
    summaries = {label: summarize(samples) for label, samples in runs.items()}
    for label in summaries:
        rss = [sample['peak_rss_bytes'] for sample in runs[label]]
        summaries[label]['peak_rss_bytes'] = {'min': min(rss), 'max': max(rss), 'median': float(np.median(rss))}
    return {'scope': 'whole isolated Python subprocess, imports and wrapper startup included',
            'input_sha256': hashes, 'commands': commands,
            'all_scan_and_unit_table_columns_types_order_exact': True,
            'calibration_and_inference_json_exact': True, 'runs': runs, 'summary': summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('before', 'prepared'))
    parser.add_argument('--dataset', choices=('PGK', 'PEPC'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--outdir', type=Path)
    args = parser.parse_args()
    reducer, prototype, hashes = make_prototype()
    if args.worker:
        if args.worker == 'prepared':
            scan._candidate_fixed_permutation_scores = prototype
        cli.main(argv_for(args.dataset, args.data, args.outdir))
        return
    result = {'proposal': 2, 'environment': environment(), 'source_function_sha256': hashes,
              'scope': 'candidate-fixed worker chunk, including assignment-local metadata preparation; not CLI',
              'warmups': 1, 'measured_repeats': 3, 'thread_control': 'all five native-thread env vars=1 before NumPy import',
              'memory_scope': 'owned branch-id/mask array bytes, excluding small Python objects; peak RSS unmeasured',
              'cases': {}}
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-round2-p02-') as directory:
        result['edge_checks'] = edge_checks(reducer)
        for dataset in ('PGK', 'PEPC'):
            result['cases'][dataset] = real_case(dataset, prototype, Path(directory))
            print(dataset, result['cases'][dataset]['summary'], flush=True)
        result['cli_cases'] = {}
        for dataset in ('PGK', 'PEPC'):
            subdir = Path(directory) / (dataset + '_cli')
            subdir.mkdir()
            result['cli_cases'][dataset] = cli_case(dataset, subdir)
            print(dataset, 'CLI', result['cli_cases'][dataset]['summary'], flush=True)
    result['cli_memory_measurement'] = ('/usr/bin/time -p -l: per-child maximum resident set size '
                                        'in bytes on macOS; CPU=user+sys')
    result['verification'] = {
        'synthetic_scenarios': len(result['edge_checks']),
        'exact_fields': ('all returned dictionaries and exceptions; both score_only modes; '
                         'unprepared fallback also checked'),
        'cli_pairs': 8, 'timed_cli_pairs': 6,
        'note': ('fixed bundled fits and current native modules; no scientific recalibration, '
                 'installed wheel, process parallelism, or fallback validation'),
    }
    (HERE / 'proposal02.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

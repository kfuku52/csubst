"""Compare production changes with a fixed source snapshot, without editing it."""

import argparse
import contextlib
from contextlib import contextmanager
import cProfile
import functools
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import types
import pstats

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  VECLIB_MAXIMUM_THREADS='1', MPLBACKEND='Agg')
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'reports/performance_candidates_20261022'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVIDENCE))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def replace(module, name, function):
    original = getattr(module, name)
    setattr(module, name, function)
    try:
        yield
    finally:
        setattr(module, name, original)


def phases(args):
    from bench_common import benchmark_lock, environment
    from csubst import omega, substitution_scan as scan

    old_scan = load(args.baseline_root / 'csubst/substitution_scan.py', 'reference_scan')
    old_omega = load(args.baseline_root / 'csubst/omega.py', 'reference_omega')
    p6 = load(EVIDENCE / 'proposal06.py', 'candidate_display_harness')
    p7 = load(EVIDENCE / 'proposal07.py', 'fixed_score_harness')
    p9 = load(EVIDENCE / 'proposal09.py', 'fisher_harness')
    result = dict(environment=environment(), scope='production phase comparisons, not CLI',
                  baseline='13c31dc', cases={})
    args.workdir.mkdir(parents=True, exist_ok=False)
    with benchmark_lock():
        current_builder = scan.build_candidates
        with replace(scan, 'build_candidates', old_scan.build_candidates):
            result['display_checks'], _, _ = p6.candidate_checks(current_builder)
            for dataset in ('PGK', 'PEPC'):
                workdir = args.workdir / ('display_' + dataset)
                workdir.mkdir()
                result['cases']['display_' + dataset] = p6.real_case(dataset, current_builder, workdir)
        current_fixed = scan._candidate_fixed_permutation_scores
        # Hold the production numerical core fixed to isolate the score-only
        # output path. Independent complete-source comparisons run in CLI mode.
        reference_fixed = types.FunctionType(
            old_scan._candidate_fixed_permutation_scores.__code__, scan.__dict__,
            'reference_fixed_scores', old_scan._candidate_fixed_permutation_scores.__defaults__)
        reducer = functools.partial(scan._rate_summary, score_only=True)
        result['fixed_score_checks'] = p7.edge_checks(reducer)
        with replace(scan, '_candidate_fixed_permutation_scores', reference_fixed):
            for dataset in ('PGK', 'PEPC'):
                workdir = args.workdir / ('fixed_' + dataset)
                workdir.mkdir()
                result['cases']['fixed_' + dataset] = p7.real_case(dataset, reducer, current_fixed, workdir)
        current_fisher = omega._calc_fisher_inclusion_probabilities
        with replace(omega, '_calc_fisher_inclusion_probabilities', old_omega._calc_fisher_inclusion_probabilities):
            result['fisher_checks'] = p9.check_small(current_fisher)
            for name, weights, draw in p9.cases():
                result['cases']['fisher_' + name] = p9.benchmark_case(current_fisher, weights, draw)
    args.result.write_text(json.dumps(result, indent=2) + '\n')
    print('Production phases and all direct output comparisons passed.', flush=True)


def cli():
    helper = load(ROOT / 'reports/memory_optimization_20260911/benchmark.py', 'cli_harness')
    harness = helper.harness
    harness.output_summary, harness.compare = helper.output_summary, helper.compare_all
    harness.CASES.update({
        'pepc_scan_all': ('PEPC', 'scan', 'defaults', {'scan_match': 'all'}),
        'pepc_fixed10': ('PEPC', 'scan', 'defaults', {
            'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '10'}),
        'pepc_fixed50': ('PEPC', 'scan', 'defaults', {
            'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '50'}),
        'pgk_fixed_parallel20': ('PGK', 'scan', 'defaults', {
            'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '20', 'threads': '2'}),
        'pgk_scan_parallel20': ('PGK', 'scan', 'defaults', {
            'scan_pvalue_calibration': 'full_scan', 'scan_n_permutations': '20', 'threads': '2'}),
        'pgk_fisher': ('PGK', 'search', 'defaults', {
            'expectation_method': 'urn', 'urn_model': 'fisher',
            'output_stat': 'any2spe', 'calibrate_longtail': 'no'}),
    })
    harness.main()


def endpoint_worker(argv):
    sys.path.insert(0, argv[0])
    argv = argv[1:]
    from csubst import cli as command, endpoint

    experiment = load(EVIDENCE / 'proposal10.py', 'endpoint_out_experiment')
    endpoint.EndpointModel.iter_blocks = experiment.PROTOTYPE
    raise SystemExit(command.main(argv))


def production_worker(argv):
    sys.path.insert(0, argv[0])
    from csubst import cli as command

    raise SystemExit(command.main(argv[1:]))


def conditionals():
    helper = load(ROOT / 'reports/memory_optimization_20260911/benchmark.py', 'conditional_harness')
    harness = helper.harness
    harness.output_summary, harness.compare = helper.output_summary, helper.compare_all
    command_for = harness.command_for
    harness.CASES = {key: (dataset, 'scan', 'defaults', {}) for key, dataset in (
        ('block_pgk', 'PGK'), ('block_pepc', 'PEPC'), ('block_pepc4', 'PEPCx4'),
        ('out_pgk', 'PGK'), ('out_pepc', 'PEPC'), ('out_pepc4', 'PEPCx4'))}

    def command(case, label, source, data, outdir):
        values = command_for(case, label, source, data, outdir)
        if label == 'after':
            if case.startswith('block_'):
                values += ['--endpoint_block_size', '128']
        worker = 'endpoint-worker' if label == 'after' and case.startswith('out_') else 'production-worker'
        return [sys.executable, str(Path(__file__).resolve()), worker, str(source), *values[3:]]

    harness.command_for = command
    harness.main()


def check_result(path):
    helper = load(ROOT / 'reports/efficiency_20261015/benchmark.py', 'result_checker')
    helper.check_cli_result(path)


def profile(args):
    from csubst import cli as command

    harness = load(ROOT / 'reports/resource_audit_20260911/benchmark.py', 'profile_harness')
    harness.CASES['fixed50'] = ('PEPC', 'scan', 'defaults', {
        'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '50'})
    args.workdir.mkdir(parents=True, exist_ok=False)
    argv = harness.command_for('fixed50', 'after', ROOT, args.baseline_root / 'csubst/dataset',
                               args.workdir / 'output')
    profiler = cProfile.Profile()
    with (args.workdir / 'stdout.log').open('w') as log, contextlib.redirect_stdout(log):
        result = profiler.runcall(command.main, argv[3:])
    assert result in (0, None)
    rows = []
    for (filename, line, name), (_, calls, self_time, cumulative, _) in pstats.Stats(profiler).stats.items():
        if filename.startswith(str(ROOT) + '/'):
            label = str(Path(filename).relative_to(ROOT))
        elif '/pandas/' in filename:
            label = 'pandas/' + filename.split('/pandas/', 1)[1]
        else:
            continue
        rows.append(dict(function=f'{label}:{line}:{name}', calls=calls,
                         self_seconds=self_time, cumulative_seconds=cumulative))
    rows.sort(key=lambda row: row['cumulative_seconds'], reverse=True)
    args.result.write_text(json.dumps(dict(scope='diagnostic cProfile; not speed comparison',
                                           case='PEPC candidate_fixed 50', functions=rows[:100]), indent=2) + '\n')
    print('Profile captured.', flush=True)


def branch_events(args):
    from bench_common import benchmark_lock
    from csubst import cli as command, substitution_scan as scan

    reference = load(args.baseline_root / 'csubst/substitution_scan.py', 'branch_reference')
    source = inspect.getsource(reference._rate_summary)
    assert 'score_only=False' in source, 'Use the 55fedb0 source snapshot for this mode.'
    old = '    event_by_branch = candidate_events.groupby("branch_id")["event_pp"].sum().to_dict()'
    new = '''    event_ids = candidate_events["branch_id"]
    if event_ids.is_unique:
        event_by_branch = dict(zip(event_ids, candidate_events["event_pp"].fillna(0)))
    else:
        event_by_branch = candidate_events.groupby("branch_id")["event_pp"].sum().to_dict()'''
    assert source.count(old) == 1
    namespace = dict(scan.__dict__)
    exec(compile(source.replace(old, new), '<unique-branch-prototype>', 'exec'), namespace)
    reducer = namespace['_rate_summary']
    fixed = types.FunctionType(scan._candidate_fixed_permutation_scores.__code__,
                               dict(scan.__dict__, _rate_summary=reducer), 'unique_branch_fixed',
                               scan._candidate_fixed_permutation_scores.__defaults__)
    p7 = load(EVIDENCE / 'proposal07.py', 'branch_events_harness')
    p7.ROOT = args.baseline_root
    original_main = command.main

    def fifty(argv):
        return original_main([*argv, '--scan_n_permutations', '50'])

    args.workdir.mkdir(parents=True, exist_ok=False)
    reference_fixed = types.FunctionType(scan._candidate_fixed_permutation_scores.__code__,
                                         scan.__dict__, 'branch_reference_fixed',
                                         scan._candidate_fixed_permutation_scores.__defaults__)
    with (benchmark_lock(), replace(command, 'main', fifty),
          replace(scan, '_rate_summary', reference._rate_summary),
          replace(scan, '_candidate_fixed_permutation_scores', reference_fixed)):
        result = dict(scope='unique branch fast path versus production score-only core',
                      edge_checks=p7.edge_checks(functools.partial(reducer, score_only=True)), cases={})
        for dataset in ('PGK', 'PEPC'):
            workdir = args.workdir / dataset
            workdir.mkdir()
            result['cases'][dataset] = p7.real_case(
                dataset, functools.partial(reducer, score_only=True), fixed, workdir)
    args.result.write_text(json.dumps(result, indent=2) + '\n')
    print('Unique-branch prototype and direct comparisons passed.', flush=True)


def main():
    mode = sys.argv.pop(1)
    if mode in ('cli', 'conditionals'):
        baseline = Path(sys.argv[sys.argv.index('--baseline-root') + 1]).resolve()
        if baseline == ROOT:
            raise ValueError('Use a temporary baseline snapshot: the harness generates repeated-site inputs there.')
        if mode == 'conditionals':
            candidate = Path(sys.argv[sys.argv.index('--candidate-root') + 1]).resolve()
            if candidate == ROOT:
                raise ValueError('Use temporary production snapshots for both conditional roots.')
    if mode == 'cli':
        cli()
    elif mode == 'conditionals':
        conditionals()
    elif mode == 'endpoint-worker':
        endpoint_worker(sys.argv[1:])
    elif mode == 'production-worker':
        production_worker(sys.argv[1:])
    elif mode == 'check-results':
        check_result(Path(sys.argv[1]))
    elif mode in ('phases', 'profile', 'branch-events'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--baseline-root', type=Path, required=True)
        parser.add_argument('--workdir', type=Path, required=True)
        parser.add_argument('--result', type=Path, required=True)
        {'phases': phases, 'profile': profile, 'branch-events': branch_events}[mode](parser.parse_args())
    else:
        raise SystemExit('Use phases, cli, conditionals, check-results or endpoint-worker.')


if __name__ == '__main__':
    main()

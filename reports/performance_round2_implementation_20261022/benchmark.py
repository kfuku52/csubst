"""Compare complete fixed-fit CLI runs from two explicit source trees.

Inputs are copied to a fresh work directory; bundled fixtures are never edited.
"""

import argparse
import cProfile
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pstats
import re
import shutil
import subprocess
import sys

for variable in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['MPLBACKEND'] = 'Agg'

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CASES = {
    'pgk_fixed200': ('PGK', 'scan', {'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '200'}),
    'pepc_fixed200': ('PEPC', 'scan', {'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '200'}),
    'pepc_fixed1000': ('PEPC', 'scan', {'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '1000'}),
    'pgk_fixed_parallel200': ('PGK', 'scan', {'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '200', 'threads': '2'}),
    'pepc_fixed_parallel200': ('PEPC', 'scan', {'scan_pvalue_calibration': 'candidate_fixed', 'scan_n_permutations': '200', 'threads': '2'}),
    'pgk_full20': ('PGK', 'scan', {'scan_pvalue_calibration': 'full_scan', 'scan_n_permutations': '20'}),
    'pgk_full_parallel20': ('PGK', 'scan', {'scan_pvalue_calibration': 'full_scan', 'scan_n_permutations': '20', 'threads': '2'}),
    'pepc_plain': ('PEPC', 'scan', {}),
    'pgk_called50': ('PGK', 'scan', {'substitution_posterior': 'marginal', 'scan_observation': 'marginal',
                                  'scan_rate_exposure': 'raw_branch_length', 'scan_rate_length': 'raw',
                                  'scan_rate_event_mode': 'called', 'scan_pvalue_calibration': 'candidate_fixed',
                                  'scan_n_permutations': '50'}),
    'pgk_analytic': ('PGK', 'scan', {'scan_match': 'any2spe,spe2spe', 'scan_analytic_pvalue': 'endpoint_mixture'}),
    'pepc_analytic': ('PEPC', 'scan', {'scan_match': 'any2spe,spe2spe', 'scan_analytic_pvalue': 'endpoint_mixture'}),
    'pgk_search': ('PGK', 'search', {}),
    'pepc_search': ('PEPC', 'search', {}),
    'pgk_structure': ('PGK', 'search', {'expectation_method': 'urn', 'asrv': 'sn',
                                     'epistasis_beta': 'auto', 'epistasis_clip': 'auto',
                                     'epistasis_joint_auto': 'yes', 'epistasis_apply_to': 'N'}),
    'pepc_structure': ('PEPC', 'search', {'expectation_method': 'urn', 'asrv': 'sn',
                                      'epistasis_beta': 'auto', 'epistasis_clip': 'auto',
                                      'epistasis_joint_auto': 'yes', 'epistasis_apply_to': 'N'}),
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def analytical_prototype():
    from csubst import scan_analytic

    trial = load(ROOT / 'reports/performance_round2_20261022/proposal04.py', 'analytical_trial')
    source = inspect.getsource(scan_analytic.annotate)
    start = source.index('    values, logs = [], []')
    stop = source.index("    total = g['scan_analytic_summary']['family_size']", start)
    loop = '''    values, logs = np.empty(len(out)), np.empty(len(out))
    previous_site, snapshot = None, None
    for position in np.argsort(out['site'].to_numpy(), kind='stable'):
        row = out.iloc[position]
        site = int(row['site'])
        if site != previous_site:
            tips = {}
            for leaf in engine.model.leaves:
                emissions = g.get('scan_tip_emissions', g['state_cdn'])
                obs = np.asarray(emissions[leaf, site], dtype=float)
                tips[leaf] = obs if obs.sum() else np.ones(engine.model.pi.size)
            snapshot = PreparedSite(engine, tips)
            previous_site = site
        branches = substitution_scan._target_branch_ids_from_maps(rate_fg, row['trait'], row['target_class'], valid)
        if "event_eligible" in g:
            branches = branches[g["event_eligible"][branches, site]]
        result = snapshot.log_likelihoods(branches,
            [int(v) for v in str(row['from_state_ids']).split(',') if v],
            [int(v) for v in str(row['to_state_ids']).split(',') if v])
        values[position], logs[position] = test_from_logs(engine, result)
'''
    namespace = dict(scan_analytic.__dict__, PreparedSite=trial.PreparedSite,
                     test_from_logs=trial.test_from_logs)
    exec(compile(source[:start] + loop + source[stop:], '<analytical-one-site>', 'exec'), namespace)
    scan_analytic.annotate = namespace['annotate']


def worker(source, experiment, argv):
    sys.path.insert(0, source)
    from csubst import cli

    if experiment == 'analytic':
        analytical_prototype()
    if experiment == 'profile':
        profiler = cProfile.Profile()
        code = profiler.runcall(cli.main, argv)
        stats = pstats.Stats(profiler)
        rows = [dict(function=f'{Path(file).name}:{line}:{name}', calls=value[1],
                     self_seconds=value[2], cumulative_seconds=value[3])
                for (file, line, name), value in stats.stats.items()]
        Path('profile.json').write_text(json.dumps(sorted(rows, key=lambda r: -r['cumulative_seconds'])[:100], indent=2))
        return code
    return cli.main(argv)


def copy_inputs(dataset, source, work):
    data = work / 'data'
    data.mkdir()
    names = [dataset + suffix for suffix in ('.alignment.fa', '.tree.nwk', '.foreground.txt')]
    names += [dataset + '.alignment.fa.' + suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')]
    for name in names:
        shutil.copyfile(source / 'csubst/dataset' / name, data / name)
    return data, {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in names}


def structure_inputs(data, dataset):
    import numpy as np
    import pandas as pd
    from csubst import epistasis, ete

    tr = ete.PhyloNode((data / (dataset + '.tree.nwk')).read_text(), format=1)
    with (data / 'context.tsv').open('w') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['branch_key', 'context_1'])
        for node in tr.traverse():
            if ete.is_root(node):
                continue
            key = epistasis.branch_key(node)
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
            writer.writerow([key, np.random.default_rng(seed).normal()])
    sites = {'PGK': 417, 'PEPC': 971}[dataset]
    pd.DataFrame({'site': np.arange(sites), 'epistasis_degree_z': np.random.default_rng(7210).normal(size=sites)}).to_csv(
        data / 'features.tsv', sep='\t', index=False)
    return {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in ('context.tsv', 'features.tsv')}


def command(case, source, data, output, experiment):
    dataset, action, options = CASES[case]
    settings = dict(threads='1', blas_threads='1', random_seed='8', float_digit='12')
    if action == 'scan':
        settings.update(scan_site_plot='no', scan_pvalue_calibration='none', scan_n_permutations='0')
    else:
        settings.update(max_arity='2')
    settings.update(options)
    argv = [action, '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')), '--outdir', str(output)]
    for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        settings['iqtree_' + suffix] = str(data / (dataset + '.alignment.fa.' + suffix))
    if case.endswith('_structure'):
        settings.update(epistasis_degree_file=str(data / 'features.tsv'),
                        epistasis_context_file=str(data / 'context.tsv'),
                        epistasis_context_source='Synthetic independent context for software verification only')
    for key, value in settings.items():
        argv.extend(('--' + key, value))
    return [sys.executable, str(Path(__file__).resolve()), 'worker', str(source), experiment, *argv]


def compare_outputs(before, after):
    import pandas as pd

    def tables(directory):
        return {str(p.relative_to(directory)): p for p in directory.rglob('*.tsv')
                if re.fullmatch(r'csubst_(cb_\d+|cb_stats|b|s|bs|cs|cbs|scan|scan_units)\.tsv', p.name)}
    left, right = tables(before), tables(after)
    assert left and left.keys() == right.keys(), (left.keys(), right.keys())
    comparisons = {}
    for name in left:
        first, second = pd.read_csv(left[name], sep='\t'), pd.read_csv(right[name], sep='\t')
        excluded = []
        if Path(name).name == 'csubst_cb_stats.tsv':
            assert first.columns.tolist() == second.columns.tolist()
            excluded = ['elapsed_sec'] if 'elapsed_sec' in first else []
            first, second = first.drop(columns=excluded), second.drop(columns=excluded)
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        comparisons[name] = dict(rows=len(first), columns=len(first.columns), exact=True)
        if excluded:
            comparisons[name]['excluded_dynamic_columns'] = excluded
    json_names = ('csubst_scan_calibration.json', 'csubst_scan_inference.json', 'csubst_epistasis.json')
    for name in json_names:
        first, second = before / name, after / name
        assert first.exists() == second.exists(), name
        if first.exists():
            before_json, after_json = json.loads(first.read_text()), json.loads(second.read_text())
            versions = None
            if name == 'csubst_scan_inference.json':
                versions = (before_json.pop('csubst_version', None), after_json.pop('csubst_version', None))
            assert before_json == after_json, name
            comparisons[name] = (dict(exact_except_csubst_version=True, csubst_versions=versions)
                                 if versions is not None and versions[0] != versions[1] else dict(exact=True))
    return comparisons


def source_hashes(source):
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((source / 'csubst').glob('*.py'))}


def run(args):
    assert args.baseline.resolve() != ROOT, 'Use a fixed baseline snapshot, not the live checkout.'
    args.workdir.mkdir(parents=True, exist_ok=False)
    # Freeze product Python before measuring, so later working-tree edits cannot
    # change a candidate between repetitions. Native binaries stay identical.
    frozen = args.workdir / 'candidate_source'
    shutil.copytree(args.baseline, frozen, symlinks=True)
    for path in (args.candidate / 'csubst').rglob('*.py'):
        destination = frozen / path.relative_to(args.candidate)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    args.candidate = frozen
    result = dict(environment=environment(), baseline_source=source_hashes(args.baseline),
                  candidate_source=source_hashes(args.candidate), warmups=1, repeats=args.repeats,
                  scope='whole CLI subprocess; fixed bundled fits; all primary table columns exact', cases={})
    with benchmark_lock():
        for case in args.cases:
            directory = args.workdir / case
            directory.mkdir()
            data, hashes = copy_inputs(CASES[case][0], args.baseline, directory)
            if case.endswith('_structure'):
                hashes.update(structure_inputs(data, CASES[case][0]))
            runs, comparisons = {'before': [], 'after': []}, []
            for repeat in range(args.repeats + 1):
                outputs = {}
                for label in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
                    output = directory / f'{repeat}_{label}'
                    source = args.baseline if label == 'before' else args.candidate
                    experiment = args.experiment if label == 'after' else 'none'
                    cmd = ['/usr/bin/time', '-p', '-l', *command(case, source, data, output, experiment)]
                    with (directory / f'{repeat}_{label}.log').open('w') as log:
                        completed, timing = measure(subprocess.run, cmd, stdout=log, stderr=subprocess.PIPE,
                                                    text=True, cwd=directory, check=True)
                    usage = completed.stderr
                    timing['cpu_s'] = sum(float(re.search(rf'^{part}\s+([\d.]+)$', usage, re.MULTILINE).group(1))
                                          for part in ('user', 'sys'))
                    timing['peak_rss_bytes'] = int(re.search(r'(\d+)\s+maximum resident set size', usage).group(1))
                    timing['command'] = cmd
                    if repeat:
                        runs[label].append(timing)
                    outputs[label] = output
                comparisons.append(compare_outputs(outputs['before'], outputs['after']))
            summaries = {label: summarize(samples) for label, samples in runs.items()}
            for label, samples in runs.items():
                import statistics
                rss = [entry['peak_rss_bytes'] for entry in samples]
                summaries[label]['peak_rss_bytes'] = dict(median=statistics.median(rss), min=min(rss), max=max(rss))
            result['cases'][case] = dict(input_sha256=hashes, runs=runs, summary=summaries, comparisons=comparisons)
            args.result.write_text(json.dumps(result, indent=2) + '\n')
            print(case, summaries, flush=True)
    print('All primary table columns and scientific JSON matched exactly.', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        raise SystemExit(worker(sys.argv[2], sys.argv[3], sys.argv[4:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=ROOT)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=CASES, required=True)
    parser.add_argument('--experiment', choices=('none', 'analytic', 'profile'), default='none')
    parser.add_argument('--repeats', type=int, default=3)
    run(parser.parse_args())

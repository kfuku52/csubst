"""Compare compact candidate-fixed permutation contexts against 624bb8a.

Run from the repository root with Python 3.14.7. The extracted baseline is
selected by CSUBST_PERF_BASELINE or the session's temporary pointer file.
"""

import argparse
import contextlib
import copy
import difflib
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['MPLBACKEND'] = 'Agg'

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = os.environ.get('CSUBST_PERF_BASELINE')
if not BASELINE:
    BASELINE = Path('/tmp/csubst-round3-baseline-path').read_text().strip()
sys.path.insert(0, BASELINE)
sys.path.insert(0, str(ROOT / 'tests/support'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, ete, scan_permutation, substitution_scan as scan  # noqa: E402
from scan_fixtures import make_scan_context  # noqa: E402


def prototype():
    original = (Path(BASELINE) / 'csubst/substitution_scan.py').read_text()
    source = original

    def replace(before, after, count=1):
        nonlocal source
        assert source.count(before) == count, (before, source.count(before))
        source = source.replace(before, after)

    replace('    configuration=None,\n):\n    if not sample_original_foreground:',
            '    configuration=None,\n    include_units=True,\n):\n    if not sample_original_foreground:')
    replace('    unit_specs: list[dict[str, Any]] = []\n',
            '    unit_specs: list[dict[str, Any]] = []\n    sampled_components = []\n')
    replace('                "leaf_names": list(trait_cache["leaf_names_by_index"][stem_index]),',
            '                "leaf_names": list(trait_cache["leaf_names_by_index"][stem_index]) if include_units else [],')
    replace('        # Lineage groups (including the number of components in each bin) are\n',
            '        if not include_units:\n            sampled_components.extend(components)\n            continue\n'
            '        # Lineage groups (including the number of components in each bin) are\n')
    replace('    all_fg_ids = _component_ids(unit_specs, "fg_branch_ids")\n',
            '''    if not include_units:
        # Fixed candidates need branch unions, not per-unit display metadata.
        # All unit sister sets exclude the same complete foreground union;
        # their union therefore equals the sister set of that union.
        all_fg_ids = _component_ids(sampled_components, "fg_branch_ids")
        all_rate_fg_ids = _component_ids(sampled_components, "clade_branch_ids")
        sister_ids = np.array([], dtype=np.int64)
        if normalize_scan_other_scope(g.get("scan_other_scope", "all")) == "sister":
            sister_ids = _unit_sister_branch_ids(
                g=g, fg_branch_ids=all_fg_ids, all_fg_branch_ids=all_fg_ids,
                node_by_id=_node_by_branch_id(g),
            )
            sister_ids = np.array([int(v) for v in sister_ids if int(v) in valid_set], dtype=np.int64)
        return {
            "units": None,
            "fg_ids": all_fg_ids,
            "rate_fg_ids": all_rate_fg_ids,
            "sister_ids": sister_ids,
            "fg_leaf_names": [],
            "sampling_attempts": attempts,
            "configuration": [[int(trait_cache["branch_ids"][i]) for i in group] for group in configuration],
        }
    all_fg_ids = _component_ids(unit_specs, "fg_branch_ids")
''')
    replace('    configurations=None,\n):\n    if rng is None:',
            '    configurations=None,\n    include_units=True,\n):\n    if rng is None:')
    replace('    sampled_configurations = {}\n', '    sampled_configurations = {}\n    sister_ids = {}\n')
    replace('            configuration=None if configurations is None else configurations[trait_name],\n',
            '            configuration=None if configurations is None else configurations[trait_name],\n'
            '            include_units=include_units,\n')
    replace('        units.append(trait_context["units"])\n',
            '        if include_units:\n            units.append(trait_context["units"])\n'
            '        else:\n            sister_ids[trait_name] = trait_context["sister_ids"]\n')
    replace('    return {\n        "units": pd.concat(units, ignore_index=True) if units else pd.DataFrame(columns=list(SCAN_UNIT_COLUMNS)),',
            '    units_df = None\n    if include_units:\n'
            '        units_df = pd.concat(units, ignore_index=True) if units else pd.DataFrame(columns=list(SCAN_UNIT_COLUMNS))\n'
            '    context = {\n        "units": units_df,')
    replace('        "configuration_id": _scan_configuration_id(sampled_configurations),\n    }\n\n\ndef _candidate_state_sets',
            '        "configuration_id": _scan_configuration_id(sampled_configurations),\n    }\n'
            '    if not include_units:\n        context["sister_ids"] = sister_ids\n'
            '    return context\n\n\ndef _candidate_state_sets')
    replace('def _build_permuted_context_with_seed(g, trait_names, valid_branch_ids, permutation_index):',
            'def _build_permuted_context_with_seed(g, trait_names, valid_branch_ids, permutation_index, include_units=True):')
    replace('        sample_original_foreground=True, rng=rng, configurations=configurations,\n',
            '        sample_original_foreground=True, rng=rng, configurations=configurations,\n'
            '        include_units=include_units,\n')
    replace('''            units_trait = scan_context['units'].loc[scan_context['units']['trait'] == trait_name, :].reset_index(drop=True)
            branch_sets[branch_key] = (
                _target_branch_ids_from_maps(fg_ids, trait_name, target_class, valid_branch_ids),
                _other_branch_ids_from_maps(fg_ids, trait_name, target_class, valid_branch_ids,
                                           scan_other_scope, units_df=units_trait))''',
            '''            target_ids = _target_branch_ids_from_maps(fg_ids, trait_name, target_class, valid_branch_ids)
            if scan_context['units'] is None:
                if scan_other_scope == 'all':
                    other_ids = valid_branch_ids
                elif scan_other_scope == 'sister':
                    other_ids = scan_context['sister_ids'][trait_name]
                else:
                    raise ValueError("Unsupported fixed-candidate control scope: " + str(scan_other_scope))
                other_ids = np.setdiff1d(other_ids, target_ids)
            else:
                units_trait = scan_context['units'].loc[scan_context['units']['trait'] == trait_name, :].reset_index(drop=True)
                other_ids = _other_branch_ids_from_maps(
                    fg_ids, trait_name, target_class, valid_branch_ids, scan_other_scope, units_df=units_trait)
            branch_sets[branch_key] = (target_ids, other_ids)''')
    replace('            permutation_index=permutation_index,\n        )\n        for key in',
            '            permutation_index=permutation_index,\n'
            '            include_units=calibration != "candidate_fixed",\n        )\n        for key in')
    (HERE / 'proposal05.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), source.splitlines(keepends=True),
        fromfile='a/csubst/substitution_scan.py', tofile='b/csubst/substitution_scan.py', n=0)))
    namespace = dict(scan.__dict__)
    exec(compile(source, '<proposal05>', 'exec'), namespace)
    return namespace


def equal(first, second):
    if isinstance(first, pd.DataFrame):
        pd.testing.assert_frame_equal(first, second, check_exact=True)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for one, two in zip(first, second):
            equal(one, two)
    else:
        np.testing.assert_equal(first, second)


def assign_groups(g, internal_clade=False):
    groups = [['A', 'B'], ['C']] if internal_clade else [['A', 'D'], ['C']]
    labels = {n.name: int(ete.get_prop(n, 'numerical_label')) for n in g['tree'].traverse()}
    names = [name for group in groups for name in group]
    g['fg_df'] = pd.DataFrame({'name': names, 'trait': [1, 1, 2]})
    g['fg_leaf_names'] = {'trait': groups}
    g['fg_ids'] = {'trait': np.array([labels[name] for name in names])}
    for node in g['tree'].traverse():
        tips = set(ete.get_leaf_names(node))
        flags = [tips.issubset(set(group)) for group in groups]
        ete.set_prop(node, 'is_fg_trait', any(flags))
        for index, flag in enumerate(flags, 1):
            ete.set_prop(node, 'is_lineage_fg_trait_' + str(index), flag)


def correctness(candidate):
    cases = []
    for mode, scope, stem_only, sister_stem, internal in itertools.product(
            ('lineage', 'stem', 'clade'), ('all', 'sister'),
            (False, True), (False, True), (False, True)):
        g, tensor = make_scan_context()
        assign_groups(g, internal_clade=internal)
        g.update(scan_unit_mode=mode, scan_other_scope=scope, fg_stem_only=stem_only,
                 scan_sister_stem_only=sister_stem, scan_pvalue_calibration='candidate_fixed',
                 scan_n_permutations=99)
        static = scan._build_scan_static_context(g, tensor)
        valid = static['valid_branch_ids']
        plan = scan._get_scan_trait_plan(g, 'trait', valid)
        configurations = plan['sampler'].configurations
        if configurations is None:
            configurations = [plan['sampler'].observed]
        for configuration in configurations:
            arguments = dict(g=g, trait_names=['trait'], valid_branch_ids=valid,
                             configurations={'trait': configuration})
            full = scan._build_permuted_scan_context(**arguments)
            equal(full, candidate['_build_permuted_scan_context'](**arguments))
            compact = candidate['_build_permuted_scan_context'](**arguments, include_units=False)
            for key in ('fg_ids', 'rate_fg_ids', 'configuration_id', 'configuration', 'sampling_attempts'):
                equal(full[key], compact[key])
            assert compact['units'] is None
            target = scan._target_branch_ids_from_maps(full['rate_fg_ids'], 'trait', 'fg', valid)
            other = scan._other_branch_ids_from_maps(
                full['rate_fg_ids'], 'trait', 'fg', valid, scope, full['units'])
            direct = np.setdiff1d(valid if scope == 'all' else compact['sister_ids']['trait'], target)
            equal(other, direct)
        # Compare entire successful exact scan and per-trial diagnostics.
        old_g, new_g = copy.deepcopy(g), copy.deepcopy(g)
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
            old = scan.scan_substitutions(old_g, tensor)
            new = candidate['scan_substitutions'](new_g, tensor)
        equal(old, new)
        equal(old_g['scan_calibration_diagnostics'], new_g['scan_calibration_diagnostics'])
        cases.append(dict(mode=mode, scope=scope, fg_stem_only=stem_only,
                          sister_stem_only=sister_stem, internal_clade=internal,
                          configurations=len(configurations)))
    edge = []
    for scenario in ('empty', 'zero_exposure', 'invalid_observed', 'invalid_null',
                     'failure', 'sampling_error', 'monte_carlo', 'full_scan'):
        outputs = []
        for module in (scan.__dict__, candidate):
            g, tensor = make_scan_context()
            g.update(scan_pvalue_calibration='candidate_fixed', scan_n_permutations=99)
            if scenario == 'empty':
                g['scan_min_support'] = '3'
            elif scenario == 'zero_exposure':
                for node in g['tree'].traverse():
                    if node.name in ('B', 'D'):
                        node.dist = 0.
            elif scenario == 'invalid_observed':
                tensor[0, 0, 0, 0, 1] = np.nan
            elif scenario == 'monte_carlo':
                g['scan_n_permutations'] = 3
            elif scenario == 'full_scan':
                g['scan_pvalue_calibration'] = 'full_scan'
            original = module['_build_permuted_context_with_seed']
            original_score = module['_candidate_fixed_permutation_scores']
            if scenario in ('failure', 'sampling_error'):
                def failed(**kwargs):
                    if scenario == 'sampling_error':
                        raise scan_permutation.SamplingError('injected proposal exhaustion', 37)
                    raise RuntimeError('injected context failure')
                module['_build_permuted_context_with_seed'] = failed
            if scenario == 'invalid_null':
                def invalid(**kwargs):
                    scores = original_score(**kwargs)
                    if scores:
                        scores[next(iter(scores))] = np.nan
                    return scores
                module['_candidate_fixed_permutation_scores'] = invalid
            try:
                with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
                    try:
                        values = module['scan_substitutions'](g, tensor)
                        outputs.append(('ok', values, g['scan_calibration_diagnostics']))
                    except Exception as exc:
                        outputs.append(('error', type(exc).__name__, str(exc)))
            finally:
                module['_build_permuted_context_with_seed'] = original
                module['_candidate_fixed_permutation_scores'] = original_score
        equal(*outputs)
        edge.append(dict(scenario=scenario, equivalent=True,
                         status=outputs[0][2]['status'] if outputs[0][0] == 'ok' else outputs[0][1]))
        if scenario in ('failure', 'sampling_error', 'invalid_null'):
            assert outputs[0][0] == 'ok'
            assert outputs[0][2]['status'] == 'unavailable_failed_trials'
            if scenario == 'sampling_error':
                assert all(trial['sampling_attempts'] == 37 for trial in outputs[0][2]['trials'])
    # Invalid explicit component and exclusion flag must reject with exact errors.
    errors = []
    g, tensor = make_scan_context()
    static = scan._build_scan_static_context(g, tensor)
    valid = static['valid_branch_ids']
    for sample_original in (False, True):
        results = []
        for module in (scan.__dict__, candidate):
            try:
                module['_build_permuted_trait_context'](
                    g=g, trait_name='trait', valid_branch_ids=valid,
                    sample_original_foreground=sample_original, configuration=((99999,),),
                    **({'include_units': False} if module is candidate else {}))
            except Exception as exc:
                results.append((type(exc).__name__, str(exc)))
        equal(*results)
        errors.append(results[0])
    return dict(scope_cases=cases, edge_cases=edge, invalid_explicit_configuration=errors,
                exact_outputs_and_diagnostics=True)


class Captured(Exception):
    pass


def phase_case(dataset, candidate, workdir, warmups, repeats, permutations):
    original = scan._run_scan_permutation_chunk
    result = {}

    def capture(*args, **kwargs):
        values = inspect.signature(original).bind(*args, **kwargs).arguments
        assert values['calibration'] == 'candidate_fixed'
        runs = {'baseline': [], 'compact': []}
        functions = {'baseline': original, 'compact': candidate['_run_scan_permutation_chunk']}
        for iteration in range(warmups + repeats):
            labels = ('baseline', 'compact') if iteration % 2 == 0 else ('compact', 'baseline')
            outputs = {}
            for label in labels:
                outputs[label], timing = measure(functions[label], *args, **kwargs)
                if iteration >= warmups:
                    runs[label].append(timing)
            equal(outputs['baseline'], outputs['compact'])
        result.update(runs=runs, summary={key: summarize(samples) for key, samples in runs.items()},
                      exact_outputs=True, candidates=len(values['observed_df']),
                      permutations=len(values['permutation_indices']),
                      success_count=outputs['baseline']['success_count'],
                      failure_count=outputs['baseline']['failure_count'],
                      scope='candidate_fixed permutation chunk with fresh worker caches; not full CLI')
        raise Captured

    data = workdir / (dataset + '_inputs')
    data.mkdir()
    paths = [dataset + suffix for suffix in ('.alignment.fa', '.tree.nwk', '.foreground.txt')]
    paths.extend(dataset + '.alignment.fa.' + suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'))
    hashes = {}
    for name in paths:
        shutil.copyfile(Path(BASELINE) / 'csubst/dataset' / name, data / name)
        hashes[name] = hashlib.sha256((data / name).read_bytes()).hexdigest()
    argv = ['scan', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')),
            '--foreground', str(data / (dataset + '.foreground.txt')),
            '--outdir', str(workdir / (dataset + '_output')),
            '--threads', '1', '--blas_threads', '1', '--scan_site_plot', 'no',
            '--scan_pvalue_calibration', 'candidate_fixed', '--scan_n_permutations', str(permutations),
            '--random_seed', '8', '--float_digit', '12']
    for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv.extend(['--iqtree_' + suffix, str(data / (dataset + '.alignment.fa.' + suffix))])
    scan._run_scan_permutation_chunk = capture
    try:
        with (workdir / (dataset + '.log')).open('w') as log, contextlib.redirect_stdout(log):
            try:
                cli.main(argv)
            except Captured:
                pass
    finally:
        scan._run_scan_permutation_chunk = original
    assert result
    return dict(result, input_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--permutations', type=int, default=50)
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    candidate = prototype()
    result = dict(proposal=5, baseline_commit='624bb8a6dae536d0fa10891f666684ef1ad3dab0',
                  environment=environment(), warmups=args.warmups, repeats=args.repeats,
                  command='python reports/performance_round3_20261023/proposal05.py',
                  source_sha256=hashlib.sha256((Path(BASELINE) / 'csubst/substitution_scan.py').read_bytes()).hexdigest(),
                  memory='Eliminates unit DataFrames and leaf-name formatting per fixed-candidate trial; peak RSS not measured.',
                  cases={})
    print('Proposal05 waiting for shared benchmark lock.', flush=True)
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-p05-') as temporary:
        result['correctness'] = correctness(candidate)
        if args.correctness_only:
            saved = json.loads((HERE / 'proposal05.json').read_text())
            saved['correctness'] = result['correctness']
            (HERE / 'proposal05.json').write_text(json.dumps(saved, indent=2) + '\n')
            print('Proposal05 extended correctness passed.', flush=True)
            return
        for dataset in ('PGK', 'PEPC'):
            result['cases'][dataset] = phase_case(dataset, candidate, Path(temporary),
                                                  args.warmups, args.repeats, args.permutations)
            (HERE / 'proposal05.json').write_text(json.dumps(result, indent=2) + '\n')
            print('Proposal05', dataset, result['cases'][dataset]['summary'], flush=True)
    (HERE / 'proposal05.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

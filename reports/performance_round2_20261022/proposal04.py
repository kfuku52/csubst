"""Evaluate bounded, per-run reuse of unaffected analytical subtrees.

Run: python reports/performance_round2_20261022/proposal04.py
Product code is not modified. PreparedSite receives private immutable tips.
"""

from collections import OrderedDict
import contextlib
import hashlib
import io
import inspect
from itertools import product
import json
import math
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
from scipy.special import logsumexp  # noqa: E402

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402
from csubst import cli, endpoint, scan_analytic, substitution_scan as scan  # noqa: E402


class PreparedSite:
    """Keep unmodified subtree partials only for one fixed model/site/profile."""

    def __init__(self, engine, tips):
        self.engine = engine
        self.tips = {node: np.array(value, dtype=float, copy=True) for node, value in tips.items()}
        for value in self.tips.values():
            value.flags.writeable = False
        model = engine.model
        if set(tips) != model.leaves:
            raise ValueError('Every leaf needs an observation likelihood.')
        self.factors = np.array([1.] + [a['multiplier'] or 1. for a in engine.atoms])
        self.participation = np.array([0.] + [a['participation'] for a in engine.atoms])
        self.partial, self.scales = [], []
        for category in range(len(model.weights)):
            partial, scales = {}, {}
            for node in reversed(model.order):
                value = np.ones((len(self.factors), model.pi.size))
                scale = np.zeros(len(self.factors))
                if node in model.leaves:
                    tip = self.tips[node]
                    if tip.shape != model.pi.shape or not np.isfinite(tip).all() or np.any(tip < 0):
                        raise ValueError('Invalid leaf observation likelihood.')
                    value *= tip
                for child in model.children[node]:
                    value *= partial[child] @ model.transition(child, category).T
                    scale += scales[child]
                    normalizer = value.max(axis=1)
                    positive = normalizer > 0
                    value[positive] /= normalizer[positive, None]
                    with np.errstate(divide='ignore'):
                        scale += np.log(normalizer)
                partial[node], scales[node] = value, scale
            self.partial.append(partial)
            self.scales.append(scales)
        self.nbytes = sum(v.nbytes for part in self.partial + self.scales for v in part.values())
        self.nbytes += sum(v.nbytes for v in self.tips.values())
        self.nbytes += self.factors.nbytes + self.participation.nbytes

    def log_likelihoods(self, branches=(), from_ids=(), to_ids=()):
        engine, model = self.engine, self.engine.model
        branches = set(map(int, branches))
        if not branches <= set(model.order) - {model.root}:
            raise ValueError('Target branches must be non-root tree nodes.')
        event = (np.isin(engine.mapping, from_ids)[:, None]
                 & np.isin(engine.mapping, to_ids)[None, :]
                 & (engine.mapping[:, None] != engine.mapping[None, :]))
        affected = set()
        for branch in branches:
            node = int(model.parents[branch])
            while node >= 0:
                affected.add(node)
                node = int(model.parents[node])
        likelihoods = []
        for category, weight in enumerate(model.weights):
            partial, scales = {}, {}
            for node in reversed(model.order):
                if node not in affected:
                    continue
                value = np.ones((len(self.factors), model.pi.size))
                scale = np.zeros(len(self.factors))
                for child in model.children[node]:
                    transition = model.transition(child, category)
                    child_partial = partial.get(child, self.partial[category][child])
                    if child in branches:
                        tilted = transition[None, :, :] * np.where(event, self.factors[:, None, None], 1.)
                        tilted /= tilted.sum(axis=2, keepdims=True)
                        selected = transition * event
                        mass = selected.sum(axis=1, keepdims=True)
                        for index, atom in enumerate(engine.atoms, 1):
                            if atom['multiplier'] is None:
                                np.divide(selected, mass, out=tilted[index], where=mass > 0)
                        tilted = (self.participation[:, None, None] * tilted
                                  + (1 - self.participation[:, None, None]) * transition)
                        value *= np.einsum('aij,aj->ai', tilted, child_partial)
                    else:
                        value *= child_partial @ transition.T
                    scale += scales.get(child, self.scales[category][child])
                    normalizer = value.max(axis=1)
                    positive = normalizer > 0
                    value[positive] /= normalizer[positive, None]
                    with np.errstate(divide='ignore'):
                        scale += np.log(normalizer)
                partial[node], scales[node] = value, scale
            root_partial = partial.get(model.root, self.partial[category][model.root])
            root_scales = scales.get(model.root, self.scales[category][model.root])
            with np.errstate(divide='ignore'):
                likelihoods.append(np.log(root_partial @ model.pi) + root_scales + math.log(weight))
        return logsumexp(likelihoods, axis=0)


def test_from_logs(engine, logs):
    if not np.isfinite(logs[0]):
        raise ValueError('Observed site has zero likelihood under the declared null.')
    log_e = float(logsumexp(logs[1:], b=[a['weight'] for a in engine.atoms]) - logs[0])
    return min(1., math.exp(min(0., -log_e))), log_e


def run_batch(engine, cases, prepared=False):
    # cases own the emissions and cannot be changed during this call. The cache
    # is private to one annotation run, not indexed by public object identity.
    cache, cached_bytes, peak_bytes, builds = OrderedDict(), 0, 0, 0
    output = [None] * len(cases)
    ordered = list(enumerate(cases))
    if prepared == 2:
        ordered.sort(key=lambda item: item[1][0])
    for index, (site, tips, branches, from_ids, to_ids) in ordered:
        if prepared:
            if site not in cache:
                snapshot = PreparedSite(engine, tips)
                builds += 1
                if snapshot.nbytes <= 32 * 1024**2:
                    while cache and cached_bytes + snapshot.nbytes > 32 * 1024**2:
                        cached_bytes -= cache.popitem(last=False)[1].nbytes
                    cache[site] = snapshot
                    cached_bytes += snapshot.nbytes
                    peak_bytes = max(peak_bytes, cached_bytes)
            else:
                snapshot = cache[site]
                cache.move_to_end(site)
            logs = snapshot.log_likelihoods(branches, from_ids, to_ids)
        else:
            logs = engine.log_likelihoods(tips, branches, from_ids, to_ids)
        output[index] = np.r_[logs, test_from_logs(engine, logs)]
    return np.asarray(output), {'peak_retained_array_bytes': peak_bytes, 'snapshot_builds': builds}


def compare(engine, cases, variant='prepared'):
    runs = {'before': [], variant: []}
    exact = True
    maximum = 0.
    stats = None
    for repeat in range(4):
        result = {}
        for label in (('before', variant) if repeat % 2 == 0 else (variant, 'before')):
            mode = (2 if variant == 'grouped' else 1) if label == variant else 0
            result[label], timing = measure(run_batch, engine, cases, mode)
            if repeat:
                runs[label].append(timing)
        a, b = result['before'][0], result[variant][0]
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13)
        exact &= a.tobytes() == b.tobytes()
        with np.errstate(invalid='ignore'):
            difference = np.abs(a - b)
        maximum = max(maximum, float(difference[np.isfinite(difference)].max(initial=0)))
        stats = result[variant][1]
    return {'candidate_count': len(cases), 'site_count': len({case[0] for case in cases}),
            'branch_count': len(engine.model.order), 'rate_categories': len(engine.model.weights),
            'exact_bytes': exact, 'maximum_absolute_difference': maximum,
            'comparison_rtol': 1e-13, 'comparison_atol': 1e-13,
            'cache': stats, 'runs': runs,
            'summary': {label: summarize(samples) for label, samples in runs.items()}}


class Captured(BaseException):
    pass


def real_case(dataset, workdir, grouped_only=False):
    data = workdir / (dataset + '_inputs')
    data.mkdir()
    names = [dataset + '.alignment.fa', dataset + '.tree.nwk', dataset + '.foreground.txt']
    names += [dataset + '.alignment.fa.' + key for key in ('treefile', 'state', 'rate', 'iqtree', 'log')]
    for name in names:
        shutil.copyfile(ROOT / 'csubst/dataset' / name, data / name)
    argv = ['scan', '--alignment_file', str(data / names[0]), '--rooted_tree_file', str(data / names[1]),
            '--foreground', str(data / names[2]), '--outdir', str(workdir / dataset),
            '--threads', '1', '--blas_threads', '1', '--scan_site_plot', 'no',
            '--scan_match', 'any2spe,spe2spe', '--scan_pvalue_calibration', 'none',
            '--scan_analytic_pvalue', 'endpoint_mixture', '--random_seed', '8', '--float_digit', '12']
    for key in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        argv += ['--iqtree_' + key, str(data / (dataset + '.alignment.fa.' + key))]
    saved = scan_analytic.annotate
    result = {}

    def capture(g, frame, units, engine):
        valid = np.array([n for n in engine.model.order if n != engine.model.root], dtype=int)
        units = scan.build_scan_units(g, pd.DataFrame({'branch_id': valid}))
        traits = list(g['fg_df'].columns[1:])
        fg = scan._fg_ids_map_from_units(units, traits, 'fg_branch_ids', g.get('fg_ids', {}))
        rate_fg = scan._rate_fg_ids_map_from_units(units, fg, traits)
        cases = []
        for _, row in frame.iterrows():
            site = int(row['site'])
            emissions = g.get('scan_tip_emissions', g['state_cdn'])
            tips = {}
            for leaf in engine.model.leaves:
                obs = np.asarray(emissions[leaf, site], dtype=float)
                tips[leaf] = obs.copy() if obs.sum() else np.ones(engine.model.pi.size)
            branches = scan._target_branch_ids_from_maps(rate_fg, row['trait'], row['target_class'], valid)
            if 'event_eligible' in g:
                branches = branches[g['event_eligible'][branches, site]]
            cases.append((site, tips, branches,
                          [int(v) for v in str(row['from_state_ids']).split(',') if v],
                          [int(v) for v in str(row['to_state_ids']).split(',') if v]))
        assert cases
        result['actual_candidates_grouped'] = compare(engine, cases, 'grouped')
        if grouped_only:
            raise Captured
        result['actual_candidates'] = compare(engine, cases)
        # Repeated contrast stress is explicitly synthetic, with the SAME tips
        # and target branches, to expose the amortization condition.
        base = cases[0]
        contrasts = [(base[0], base[1], base[2], list(range(int(engine.mapping.max()) + 1)), [to])
                     for to in range(int(engine.mapping.max()) + 1)]
        result['same_site_20_contrasts_synthetic'] = compare(engine, contrasts)
        raise Captured

    scan_analytic.annotate = capture
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                cli.main(argv)
            except Captured:
                pass
    finally:
        scan_analytic.annotate = saved
    result['input_sha256'] = {f'csubst/dataset/{name}': hashlib.sha256((data / name).read_bytes()).hexdigest()
                              for name in names}
    return result


def edge_checks():
    model = endpoint.EndpointModel(np.array([-1, 0, 0, 1, 1, 2, 2]),
                                   [0, .2, .4, 0, .7, .05, .1],
                                   [[-1, 1], [1, -1]], [.5, .5], [0, .3, 2], [.1, .3, .6])
    profiles = [None, dict(version=1, atoms=[dict(multiplier=7., participation=.3, weight=.7),
                                            dict(multiplier=None, participation=.8, weight=.3)])]
    checked, exact, max_diff = 0, True, 0.
    for profile in profiles:
        engine = scan_analytic.EndpointEnrichment(model, np.arange(2), profile)
        for pattern in product(range(3), repeat=4):
            tips = {leaf: np.eye(2)[v] if v < 2 else np.ones(2)
                    for leaf, v in zip(sorted(model.leaves), pattern)}
            prepared = PreparedSite(engine, tips)
            for branches in ([], [3], [3, 5], [1, 2, 3, 4, 5, 6]):
                a = engine.log_likelihoods(tips, branches, [0], [1])
                b = prepared.log_likelihoods(branches, [0], [1])
                np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13)
                exact &= a.tobytes() == b.tobytes()
                with np.errstate(invalid='ignore'):
                    diff = np.abs(a - b)
                max_diff = max(max_diff, float(diff[np.isfinite(diff)].max(initial=0)))
                checked += 1
    # Independent enumeration includes shared hidden states, mixture, infinite
    # tilt and zero branch lengths; the stable scaled log-space result is
    # compared with direct full assignment summation on this tiny model.
    engine = scan_analytic.EndpointEnrichment(model, np.arange(2))
    enumerated = 0
    for pattern in product(range(2), repeat=4):
        tips = {leaf: np.eye(2)[v] for leaf, v in zip(sorted(model.leaves), pattern)}
        actual = np.exp(PreparedSite(engine, tips).log_likelihoods([3, 5], [0], [1]))
        explicit = np.zeros(5)
        for category, weight in enumerate(model.weights):
            for hidden in product(range(2), repeat=3):
                assignment = dict(enumerate(hidden)) | dict(zip(sorted(model.leaves), pattern))
                for alt, factor in enumerate([1, 2, 10, 100, None]):
                    likelihood = model.pi[assignment[0]] * weight
                    for child in range(1, 7):
                        trans = model.transition(child, category).copy()
                        if child in (3, 5):
                            if factor is None:
                                if trans[0, 1] > 0:
                                    trans[0] = [0, 1]
                            else:
                                trans[0, 1] *= factor
                                trans /= trans.sum(axis=1, keepdims=True)
                        likelihood *= trans[assignment[model.parents[child]], assignment[child]]
                    explicit[alt] += likelihood
        np.testing.assert_allclose(actual, explicit, rtol=2e-13, atol=1e-16)
        enumerated += 1
    tips = {leaf: np.ones(2) for leaf in model.leaves}
    errors = []
    invalid_cases = [(dict(tips), [model.root])]
    missing = dict(tips)
    missing.pop(next(iter(missing)))
    invalid_cases.append((missing, []))
    for value in (np.array([-1., 2.]), np.array([np.nan, 1.]), np.ones(3)):
        invalid_cases.append((dict(tips) | {min(model.leaves): value}, []))
    for bad_tips, bad_branches in invalid_cases:
        outcomes = []
        for prepared in (False, True):
            try:
                if prepared:
                    PreparedSite(engine, bad_tips).log_likelihoods(bad_branches, [0], [1])
                else:
                    engine.log_likelihoods(bad_tips, bad_branches, [0], [1])
            except ValueError as exc:
                outcomes.append((type(exc).__name__, str(exc)))
        assert len(outcomes) == 2 and outcomes[0] == outcomes[1]
        errors.append(outcomes[0])
    return {'scaled_parity_cases': checked, 'exact_bytes': exact, 'maximum_absolute_difference': max_diff,
            'independent_enumerated_tip_patterns': enumerated, 'failure_parity_cases': errors}


def main():
    result = {'proposal': 4, 'environment': environment(), 'warmups': 1, 'measured_repeats': 3,
              'scope': 'analytical likelihood/test stage including private snapshot/cache preparation; not whole CLI',
              'memory_scope': 'retained numerical array bytes; Python container overhead and process RSS unmeasured',
              'cache_limit_bytes': 32 * 1024**2,
              'source_function_sha256': hashlib.sha256(inspect.getsource(scan_analytic.EndpointEnrichment.log_likelihoods).encode()).hexdigest(),
              'cases': {}}
    grouped_only = '--grouped-only' in sys.argv
    if grouped_only:
        result = json.loads((HERE / 'proposal04.json').read_text())
    with benchmark_lock(), tempfile.TemporaryDirectory(prefix='csubst-round2-p04-') as directory:
        result['edge_checks'] = edge_checks()
        for dataset in ('PGK', 'PEPC'):
            update = real_case(dataset, Path(directory), grouped_only)
            if grouped_only:
                result['cases'][dataset].update(update)
            else:
                result['cases'][dataset] = update
            print(dataset, {name: {label: value['wall_s']['median'] for label, value in case['summary'].items()}
                            for name, case in result['cases'][dataset].items() if 'summary' in case}, flush=True)
    result['source_function_sha256'] = hashlib.sha256(inspect.getsource(scan_analytic.EndpointEnrichment.log_likelihoods).encode()).hexdigest()
    (HERE / 'proposal04.json').write_text(json.dumps(result, indent=2) + '\n')


def regression_check():
    """Run existing analytical contracts against the private numerical backend."""
    import pytest

    original = scan_analytic.EndpointEnrichment.log_likelihoods

    def prepared_method(self, tips, branches=(), from_ids=(), to_ids=()):
        # Preserve the existing entrypoint's key/branch validation order.
        if set(tips) != self.model.leaves:
            raise ValueError('Every leaf needs an observation likelihood.')
        branches = set(map(int, branches))
        if not branches <= set(self.model.order) - {self.model.root}:
            raise ValueError('Target branches must be non-root tree nodes.')
        return PreparedSite(self, tips).log_likelihoods(branches, from_ids, to_ids)

    with benchmark_lock():
        scan_analytic.EndpointEnrichment.log_likelihoods = prepared_method
        try:
            return pytest.main(['-q', 'tests/unit/test_scan_analytic.py'])
        finally:
            scan_analytic.EndpointEnrichment.log_likelihoods = original


if __name__ == '__main__':
    if '--verify-only' in sys.argv:
        raise SystemExit(regression_check())
    else:
        main()

"""Investigate spectral transitions without changing the endpoint implementation.

Run from the repository root: python reports/performance_candidates_20261022/proposal03.py
"""

# ruff: noqa: E402 -- Thread limits must precede imports of numerical libraries.

import os

for name in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
             'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

from contextlib import redirect_stdout
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from csubst import cli, endpoint
from bench_common import benchmark_lock, environment, measure, summarize


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class SpectralModel(endpoint.EndpointModel):
    """Guarded reversible decomposition; retain validation and transition cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectral = None
        if np.any(self.pi <= 0):
            return
        root = np.sqrt(self.pi)
        symmetric = root[:, None] * self.q / root[None, :]
        if not np.allclose(symmetric, symmetric.T, rtol=0, atol=1e-12):
            return
        values, vectors = eigh((symmetric + symmetric.T) * .5)
        if values.max() > 1e-12:
            return
        self.spectral = (np.minimum(values, 0), vectors, root)

    def _uniform_transition(self, t):
        if self.spectral is None:
            return super()._uniform_transition(t)
        if t == 0:
            return np.eye(len(self.pi))
        values, vectors, root = self.spectral
        return ((vectors * np.exp(values * t)) @ vectors.T) / root[:, None] * root[None, :]


class Captured(Exception):
    pass


def load_real(name):
    original = endpoint.EndpointModel.iter_blocks
    captured = {}

    def capture(model, tips, *args, **kwargs):
        captured.update(args=(model.parents.copy(), model.lengths.copy(), model.q.copy(),
                              model.pi.copy(), model.rates.copy(), model.weights.copy()),
                        tips={k: v.copy() for k, v in tips.items()})
        raise Captured

    base = ROOT / 'csubst/dataset' / (name + '.alignment.fa')
    with tempfile.TemporaryDirectory(prefix='csubst-proposal03-') as work:
        command = ['search', '--alignment_file', str(base), '--rooted_tree_file',
                   str(ROOT / 'csubst/dataset' / (name + '.tree.nwk')),
                   '--foreground', str(ROOT / 'csubst/dataset' / (name + '.foreground.txt')),
                   '--outdir', str(Path(work) / 'output'), '--threads', '1', '--blas_threads', '1',
                   '--drop_invariant_tip_sites', 'no', '--calibrate_longtail', 'no']
        for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
            command += ['--iqtree_' + suffix, str(base) + '.' + suffix]
        endpoint.EndpointModel.iter_blocks = capture
        try:
            with open(Path(work) / 'capture.log', 'w') as log, redirect_stdout(log):
                try:
                    cli.main(command)
                except Captured:
                    pass
        finally:
            endpoint.EndpointModel.iter_blocks = original
    assert captured
    return captured


def transitions(model_args, cls):
    model = cls(*model_args)
    return np.stack([model.transition(child, category)
                     for child in model.order[1:] for category in range(len(model.rates))])


def decimal_exponential(q, length):
    with localcontext() as context:
        context.prec = 110
        k = len(q)
        a = [[Decimal(str(q[i, j])) * Decimal(str(length)) for j in range(k)] for i in range(k)]
        term = [[Decimal(int(i == j)) for j in range(k)] for i in range(k)]
        total = [row.copy() for row in term]
        for n in range(1, 100):
            term = [[sum(term[i][z] * a[z][j] for z in range(k)) / n
                     for j in range(k)] for i in range(k)]
            total = [[total[i][j] + term[i][j] for j in range(k)] for i in range(k)]
            if max(abs(x) for row in term for x in row) < Decimal('1e-100'):
                break
        return np.array(total, dtype=float)


def accuracy_tests():
    k = 8
    q = np.zeros((k, k))
    for i in range(k - 1):
        q[i, i + 1] = q[i + 1, i] = .7
    np.fill_diagonal(q, -q.sum(axis=1))
    rows = []
    for length in (0, 1e-10, 1e-5, .1, 1, 10, 100):
        args = ([-1, 0], [0, length], q, np.ones(k) / k)
        before, after = endpoint.EndpointModel(*args), SpectralModel(*args)
        actual = before.transition(1, 0)
        prototype = after.transition(1, 0)
        reference = decimal_exponential(q, length) if 0 < length < .01 else actual
        np.testing.assert_allclose(actual, reference, rtol=3e-14, atol=1e-300)
        try:
            np.testing.assert_allclose(prototype, reference, rtol=3e-14, atol=1e-300)
            passes = True
        except AssertionError:
            passes = False
        positive = reference > 0
        rows.append(dict(length=length, production_passes=True, spectral_passes=passes,
                         max_absolute_error=float(np.max(abs(prototype-reference))),
                         max_positive_relative_error=float(np.max(abs(prototype[positive]-reference[positive])
                                                                  / reference[positive])),
                         positive_transitions_lost=int(np.sum(positive & (prototype == 0))),
                         rare_reference=float(reference[0, -1]), rare_spectral=float(prototype[0, -1])))
    # Evidence at opposing rare tip states amplifies tiny transition errors.
    args = ([-1, 0, 0], [0, 1e-5, 1e-5], q, np.ones(k) / k)
    tips = {1: np.eye(k)[[0]], 2: np.eye(k)[[-1]]}
    left = list(endpoint.EndpointModel(*args).iter_blocks(tips))
    right = list(SpectralModel(*args).iter_blocks(tips))
    posterior_error = max(float(np.max(abs(a.node-b.node))) for a, b in zip(left, right))
    q_nonrev = np.array([[-1., 1, 0], [0, -1., 1], [1, 0, -1.]])
    fallback = SpectralModel([-1, 0], [0, .3], q_nonrev, np.ones(3)/3)
    assert fallback.spectral is None
    np.testing.assert_array_equal(fallback.transition(1, 0),
                                  endpoint.EndpointModel([-1, 0], [0, .3], q_nonrev, np.ones(3)/3).transition(1, 0))
    zero_pi = SpectralModel([-1, 0], [0, .3], np.zeros((3, 3)), [0, .5, .5])
    assert zero_pi.spectral is None
    rejected = 0
    for q_bad, pi_bad in ((np.ones((3, 3)), np.ones(3)/3),
                           (np.zeros((3, 3)), [1., 1., 1.])):
        try:
            SpectralModel([-1, 0], [0, .3], q_bad, pi_bad)
        except ValueError:
            rejected += 1
    assert rejected == 2
    return dict(chain=rows, rare_evidence_max_node_absolute_error=posterior_error,
                nonreversible_fallback_matches=True, zero_pi_fallback=True, invalid_inputs_rejected=rejected)


def main():
    result = dict(environment=environment(), scope='cold transition construction, not CLI runtime',
                  native_thread_control='environment set before NumPy/SciPy import; no threadpoolctl installed',
                  accuracy=accuracy_tests(), datasets={})
    with benchmark_lock():
        for name in ('PGK', 'PEPC'):
            real = load_real(name)
            samples = {'production': [], 'spectral': []}
            outputs = {}
            for rep in range(4):
                order = [('production', endpoint.EndpointModel), ('spectral', SpectralModel)]
                if rep % 2:
                    order.reverse()
                for label, cls in order:
                    output, sample = measure(transitions, real['args'], cls)
                    outputs[label] = output
                    if rep:
                        samples[label].append(sample)
            before, after = outputs['production'], outputs['spectral']
            positive = before > 0
            actual = endpoint.EndpointModel(*real['args'])
            spectral = SpectralModel(*real['args'])
            # Directly compare every node, category-node and endpoint-pair element
            # on 16 original sites. Full alignment storage is unnecessary.
            tip_subset = {node: values[:16] for node, values in real['tips'].items()}
            differences = {'node': 0., 'joint': 0., 'predictive': 0., 'category_node': 0.}
            allclose = True
            for a, b in zip(actual.iter_blocks(tip_subset, predictive=True, category_nodes=True),
                            spectral.iter_blocks(tip_subset, predictive=True, category_nodes=True)):
                for field in differences:
                    x, y = getattr(a, field), getattr(b, field)
                    if x is not None:
                        differences[field] = max(differences[field], float(np.max(abs(x-y))))
                        try:
                            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-13)
                        except AssertionError:
                            allclose = False
            result['datasets'][name] = dict(
                nodes=len(actual.parents), sites=next(iter(real['tips'].values())).shape[0],
                states=len(actual.pi), categories=len(actual.rates),
                q_sha256=hashlib.sha256(actual.q.tobytes()).hexdigest(),
                lengths_sha256=hashlib.sha256(actual.lengths.tobytes()).hexdigest(),
                spectral_applicable=spectral.spectral is not None, samples=samples,
                summary={key: summarize(value) for key, value in samples.items()},
                transition_max_absolute_error=float(np.max(abs(before-after))),
                transition_max_positive_relative_error=float(np.max(abs(before[positive]-after[positive])/before[positive])),
                transition_positive_entries_lost=int(np.sum(positive & (after == 0))),
                posterior_16_sites_allclose=allclose, posterior_max_absolute_errors=differences,
                production_cache_capacity_bytes=actual._uniform_powers.nbytes,
                spectral_extra_array_bytes=sum(x.nbytes for x in spectral.spectral) if spectral.spectral else 0)
            print(name, result['datasets'][name]['summary'], flush=True)
    result['decision'] = 'reject: spectral reconstruction violates existing entrywise rare-transition accuracy'
    (HERE / 'proposal03.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

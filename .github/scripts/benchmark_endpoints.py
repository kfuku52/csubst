#!/usr/bin/env python3
"""Reproduce endpoint accuracy and resource comparisons in isolated processes.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Work files and CLI outputs go
under --workdir; only the compact JSON result is intended for version control.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def cli_args(dataset, mode, block, outdir):
    data = ROOT / 'csubst' / 'dataset'
    args = ['csubst', 'search', '--alignment_file', str(data / (dataset + '.alignment.fa')),
            '--rooted_tree_file', str(data / (dataset + '.tree.nwk')), '--outdir', str(outdir),
            '--threads', '1', '--blas_threads', '1', '--max_arity', '2', '--calibrate_longtail', 'no',
            '--b', 'no', '--s', 'no', '--cs', 'no', '--bs', 'no', '--cbs', 'no', '--random_seed', '8',
            '--substitution_posterior', mode, '--endpoint_block_size', str(block)]
    for key in ('treefile', 'state', 'rate', 'iqtree', 'log'):
        args += ['--iqtree_' + key, str(data / (dataset + '.alignment.fa.' + key))]
    return args


def make_fixture(path, tips_count, sites, states, seed):
    import numpy as np
    from csubst.endpoint import EndpointModel
    from csubst import genetic_code
    rng = np.random.default_rng(seed)
    n = 2 * tips_count - 1
    parents = np.r_[-1, (np.arange(1, n) - 1) // 2]
    lengths = rng.uniform(.01, .1, n)
    lengths[0] = 0
    # Include genuine zero and short internal edges, not just uncertain tips.
    lengths[1] = 0
    lengths[2] = 1e-6
    pi = rng.dirichlet(np.full(states, 5.))
    exchange = np.ones((states, states))
    if states == 61:
        codons = [c for aa, c in genetic_code.get_codon_table(1) if aa != '*']
        exchange = np.array([[sum(a != b for a, b in zip(x, y)) == 1 for y in codons] for x in codons], dtype=float)
    q = exchange * pi
    np.fill_diagonal(q, 0)
    np.fill_diagonal(q, -q.sum(1))
    q /= -pi @ q.diagonal()
    rates, weights = np.array([.1, .5, 1., 2.4]), np.full(4, .25)
    model = EndpointModel(parents, lengths, q, pi, rates, weights)
    category = rng.choice(4, size=sites, p=weights)
    truth = np.zeros((n, sites), dtype=np.int16)
    truth[0] = rng.choice(states, size=sites, p=pi)
    for node in range(1, n):
        for c in range(4):
            at = np.flatnonzero(category == c)
            transition = model.transition(node, c)
            prob = transition[truth[parents[node], at]]
            u = rng.random(at.size)
            truth[node, at] = np.minimum((u[:, None] > prob.cumsum(1)).sum(1), states - 1)
    leaf_ids = np.array(sorted(model.leaves))
    tips = np.eye(states)[truth[leaf_ids]]
    # 10% ambiguous observations represented as observation likelihoods.
    for li in range(len(leaf_ids)):
        ambiguous = rng.random(sites) < .1
        other = rng.integers(states, size=int(ambiguous.sum()))
        tips[li, np.flatnonzero(ambiguous), other] = 1
    marginals = np.zeros((n, sites, states))
    for rec in model.iter_blocks(dict(zip(leaf_ids, tips)), joint=False):
        marginals[rec.child, rec.start:rec.stop] = rec.node
    np.savez_compressed(path, parents=parents, lengths=lengths, q=q, pi=pi, rates=rates, weights=weights,
                        leaf_ids=leaf_ids, tips=tips, marginals=marginals, truth=truth)


def kernel_worker(args):
    import numpy as np
    # Match module import overhead for the two workers.
    from csubst import substitution  # noqa: F401
    from csubst.endpoint import EndpointModel
    with np.load(args.fixture) as f:
        parents, truth = f['parents'], f['truth']
        if args.mode == 'joint':
            model = EndpointModel(parents, f['lengths'], f['q'], f['pi'], f['rates'], f['weights'])
            tips = dict(zip(f['leaf_ids'], f['tips']))
            marginals = None
        else:
            marginals = f['marginals']
            model = tips = None
    n, size = truth.shape
    loss = np.zeros(size)
    event_loss = np.zeros(size)
    mass = 0.
    zero_mass = 0.
    start = time.perf_counter()

    def collect(child, begin, end, edge):
        nonlocal mass, zero_mass
        diagonal = np.arange(edge.shape[1])
        edge[:, diagonal, diagonal] = 0
        prob = edge.sum((1, 2))
        a, d = truth[parents[child], begin:end], truth[child, begin:end]
        changed = a != d
        loss[begin:end] += (prob - changed) ** 2
        event_loss[begin:end] += (edge ** 2).sum((1, 2)) - 2 * edge[np.arange(end - begin), a, d] + changed
        mass += float(prob.sum())
        if child == 1:
            zero_mass += float(prob.sum())

    if args.mode == 'joint':
        for rec in model.iter_blocks(tips, block_size=args.block):
            if rec.parent >= 0:
                collect(rec.child, rec.start, rec.stop, rec.joint)
    else:
        for begin in range(0, size, args.block):
            end = min(size, begin + args.block)
            for child in range(1, n):
                edge = (marginals[parents[child], begin:end, :, None]
                        * marginals[child, begin:end, None, :])
                collect(child, begin, end, edge)
    elapsed = time.perf_counter() - start
    np.savez_compressed(args.result + '.loss.npz', change=loss / (n - 1), event=event_loss / (n - 1))
    return dict(kernel_seconds=elapsed, nodes=n, sites=size, change_brier=float(loss.mean() / (n - 1)),
                event_brier=float(event_loss.mean() / (n - 1)), total_change_mass=mass,
                zero_branch_mean_mass=zero_mass / size)


def worker(args):
    start = time.perf_counter()
    if args.kind == 'pipeline':
        sys.argv = cli_args(args.dataset, args.mode, args.block, args.outdir)
        runpy.run_module('csubst', run_name='__main__')
        result = {'command': sys.argv}
    else:
        result = kernel_worker(args)
    result['worker_seconds'] = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_rss_mib'] = rss / (1024**2 if sys.platform == 'darwin' else 1024)
    result.update(mode=args.mode, block_size=args.block)
    Path(args.result).write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, default=Path('/tmp/csubst-endpoint-benchmark'))
    parser.add_argument('--result', default='endpoint_benchmark.json')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--kind', choices=['pipeline', 'kernel'])
    parser.add_argument('--dataset', default='PGK')
    parser.add_argument('--mode', default='joint')
    parser.add_argument('--block', type=int, default=64)
    parser.add_argument('--fixture')
    parser.add_argument('--outdir')
    parser.add_argument('--pipelines-only', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    import numpy as np
    args.workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    scenarios = [('PGK', 'pipeline', []), ('PEPC', 'pipeline', [])]
    if not args.pipelines_only:
        for name, tips, sites, states in [('structural', 32, 2000, 20), ('codon', 32, 1000, 61),
                                         ('large_structural', 128, 2000, 20)]:
            fixture = args.workdir / (name + '.npz')
            if not fixture.exists():
                make_fixture(fixture, tips, sites, states, 810)
            scenarios.append((name, 'kernel', ['--fixture', str(fixture)]))
    results = {}
    for name, kind, extra in scenarios:
        runs = []
        configs = [('marginal', 64), ('joint', 64), ('joint', 16)]
        for repeat in range(args.repeats + 1):
            for mode, block in configs:
                tag = '{}-{}-b{}-{}'.format(name, mode, block, repeat)
                path = args.workdir / (tag + '.json')
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--kind', kind,
                           '--dataset', name, '--mode', mode, '--block', str(block), '--result', str(path),
                           '--outdir', str(args.workdir / tag)] + extra
                with open(args.workdir / (tag + '.log'), 'w') as log:
                    subprocess.run(command, cwd=args.workdir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads(path.read_text())
                result.update(repeat=repeat, warmup=(repeat == 0), result_file=str(path))
                runs.append(result)
                print(tag, round(result['worker_seconds'], 3), round(result['peak_rss_mib'], 1), flush=True)
        summary = {}
        for mode, block in configs:
            selected = [r for r in runs if not r['warmup'] and r['mode'] == mode and r['block_size'] == block]
            summary[mode + '_b' + str(block)] = {
                key: {'median': float(np.median([r[key] for r in selected])),
                      'min': min(r[key] for r in selected), 'max': max(r[key] for r in selected)}
                for key in ['worker_seconds', 'peak_rss_mib'] + (['kernel_seconds'] if kind == 'kernel' else [])}
        results[name] = {'kind': kind, 'summary': summary, 'runs': runs}
        if kind == 'pipeline':
            import pandas as pd
            reference = pd.read_csv(args.workdir / (name + '-joint-b64-1') / 'csubst_cb_2.tsv', sep='\t')
            other = pd.read_csv(args.workdir / (name + '-joint-b16-1') / 'csubst_cb_2.tsv', sep='\t')
            pd.testing.assert_frame_equal(reference, other, check_exact=False, rtol=1e-10, atol=1e-10)
            results[name]['block_size_output_parity'] = True
        if kind == 'kernel':
            baseline = next(r for r in runs if r['mode'] == 'marginal')
            joint = next(r for r in runs if r['mode'] == 'joint' and r['block_size'] == 64)
            with np.load(baseline['result_file'] + '.loss.npz') as a, np.load(joint['result_file'] + '.loss.npz') as b:
                delta = b['change'] - a['change']
                smaller = next(r for r in runs if r['mode'] == 'joint' and r['block_size'] == 16)
                with np.load(smaller['result_file'] + '.loss.npz') as small:
                    np.testing.assert_allclose(b['change'], small['change'], atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(b['event'], small['event'], atol=1e-12, rtol=1e-12)
            results[name]['block_size_output_parity'] = True
            results[name]['paired_change_brier_delta'] = float(delta.mean())
            results[name]['paired_change_brier_delta_95ci'] = (delta.mean() + np.array([-1, 1]) * 1.96
                                                             * delta.std(ddof=1) / np.sqrt(delta.size)).tolist()
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / 'csubst' / 'dataset').glob('*') if p.is_file() and p.name.startswith(('PGK.', 'PEPC.'))}
    code_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in list((ROOT / 'csubst').glob('*.py')) + [Path(__file__).resolve()]}
    output = {'environment': {'platform': platform.platform(), 'python': sys.version, 'numpy': np.__version__,
                              'threads': 1, 'warmups': 1, 'repeats': args.repeats},
              'input_sha256': hashes, 'source_sha256': code_hashes, 'scenarios': results}
    Path(args.result).write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()

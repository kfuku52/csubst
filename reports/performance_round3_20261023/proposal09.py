"""Measure releasing consumed tree-count DP states against baseline 624bb8a.

Run from the repository root:
python reports/performance_round3_20261023/proposal09.py
Only proposal09 report files are written; all workloads use the shared lock.
"""

import ast
import difflib
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tracemalloc

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

from bench_common import benchmark_lock, environment, measure, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
BASELINE = os.environ.get('CSUBST_PERF_BASELINE')
if not BASELINE:
    BASELINE = Path('/tmp/csubst-round3-baseline-path').read_text().strip()


def prototype(ete):
    source = (Path(BASELINE) / 'csubst/combination.py').read_text()
    names = ('_add_truncated_polynomials', '_multiply_truncated_polynomials',
             'count_independent_branch_combinations')
    functions = {
        node.name: '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno]) + '\n'
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    original = functions[names[-1]]
    candidate = original.replace(
        '        child_not_selected = [\n',
        '        # Child states are no longer needed after this parent consumes them.\n'
        '        child_not_selected = [\n').replace(
        'not_selected_polynomial[id(child)]',
        'not_selected_polynomial.pop(id(child))').replace(
        'all_polynomial[id(child)]', 'all_polynomial.pop(id(child))')
    assert candidate != original
    changed = source.replace(original, candidate)
    (HERE / 'proposal09.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile='a/csubst/combination.py', tofile='b/csubst/combination.py', n=0)))
    functions[names[-1]] = original
    old_namespace = {'ete': ete}
    exec(compile('\n'.join(functions[name] for name in names), '<proposal09-baseline>', 'exec'), old_namespace)
    functions[names[-1]] = candidate
    new_namespace = {'ete': ete}
    exec(compile('\n'.join(functions[name] for name in names), '<proposal09-candidate>', 'exec'), new_namespace)
    return (old_namespace[names[-1]], new_namespace[names[-1]],
            hashlib.sha256(source.encode()).hexdigest())


def newick(shape, tips, fanout=4):
    nodes = [f'L{i}:1' for i in range(tips)]
    if tips == 1:
        return nodes[0] + ';'
    if shape == 'star':
        return '(' + ','.join(nodes) + ')R;'
    if shape == 'pectinate':
        subtree = '(' + ','.join(nodes[:2]) + '):1'
        for leaf in nodes[2:]:
            subtree = '(' + subtree + ',' + leaf + '):1'
        return subtree[:-2] + 'R;'
    width = 2 if shape == 'balanced' else fanout
    while len(nodes) > 1:
        nodes = ['(' + ','.join(nodes[index:index + width]) + '):1'
                 if len(nodes[index:index + width]) > 1 else nodes[index]
                 for index in range(0, len(nodes), width)]
    return nodes[0][:-2] + 'R;'


def brute_count(ete, tr, max_arity, exclude):
    branches = [node for node in tr.traverse() if not ete.is_root(node)]
    forbidden = set()
    for branch in branches:
        for ancestor in ete.iter_ancestors(branch):
            if not ete.is_root(ancestor):
                forbidden.add(frozenset((id(branch), id(ancestor))))
    if exclude:
        for node in tr.traverse():
            for left, right in itertools.combinations(ete.get_children(node), 2):
                forbidden.add(frozenset((id(left), id(right))))
    counts = [0] * (max_arity + 1)
    for selected in itertools.chain.from_iterable(
            itertools.combinations(branches, arity)
            for arity in range(min(max_arity, len(branches)) + 1)):
        if all(frozenset((id(left), id(right))) not in forbidden
               for left, right in itertools.combinations(selected, 2)):
            counts[len(selected)] += 1
    return counts


def exactness(ete, old, new):
    checks, brute_checks, max_bits = 0, 0, 0
    texts = ['R;', '(A:1)R;', '((A:1)X:1)R;',
             '((A:1,B:1)X:1,(C:1,D:1)Y:1,E:1)R;']
    texts += [newick(shape, tips) for shape in ('balanced', 'pectinate', 'star', 'polytomous')
              for tips in (2, 3, 4, 5, 6)]
    rng = random.Random(9023)
    for _ in range(16):
        groups = [f'R{i}:1' for i in range(rng.randrange(2, 7))]
        while len(groups) > 1:
            size = rng.randrange(2, len(groups) + 1)
            indices = sorted(rng.sample(range(len(groups)), size), reverse=True)
            selected = [groups.pop(index) for index in indices]
            groups.append('(' + ','.join(selected) + '):1')
        texts.append(groups[0][:-2] + 'R;')
    for text in texts:
        tr = ete.PhyloNode(text, format=1)
        for exclude in (False, True):
            for arity in (0, 1, 2, 4, 16):
                expected = brute_count(ete, tr, arity, exclude)
                assert old(tr, arity, exclude) == expected
                assert new(tr, arity, exclude) == expected
                assert all(type(count) is int for count in new(tr, arity, exclude))
                checks += 1
                brute_checks += 1
            for arity in (-1, -10):
                for function in (old, new):
                    try:
                        function(tr, arity, exclude)
                    except ValueError as exc:
                        assert str(exc) == 'max_arity should be >= 0.'
                    else:
                        raise AssertionError('negative arity accepted')
                checks += 1
    for shape, tips, arity in (('balanced', 512, 64), ('pectinate', 257, 64),
                               ('star', 128, 64), ('polytomous', 256, 64)):
        tr = ete.PhyloNode(newick(shape, tips), format=1)
        for exclude in (False, True):
            before, after = old(tr, arity, exclude), new(tr, arity, exclude)
            assert before == after
            assert all(type(count) is int for count in after)
            max_bits = max(max_bits, max(count.bit_length() for count in after))
            if shape == 'star':
                expected = ([1, tips] + [0] * (arity - 1) if exclude
                            else [math.comb(tips, k) for k in range(arity + 1)])
                assert before == expected
            checks += 1
    assert max_bits > 64
    return {'equal_cases': checks, 'independent_bruteforce_cases': brute_checks,
            'largest_integer_bits': max_bits,
            'covered': ['both sister-pair modes', 'root-only', 'unary', 'balanced',
                        'pectinate', 'star', 'polytomous', 'random small topologies',
                        'arity 0', 'negative arity', 'arity beyond branch count',
                        'Python integers exceeding 64 bits']}


def timed_case(ete, old, new, tr, arity, exclude):
    args = (tr, arity, exclude)
    before, warmup = measure(old, *args)
    after = new(*args)
    assert before == after
    batch = max(1, min(40, math.ceil(0.04 / max(warmup['cpu_s'], 1e-6))))
    samples = {'baseline': [], 'candidate': []}
    functions = {'baseline': old, 'candidate': new}
    def batched(function):
        result = None
        for _ in range(batch):
            result = function(*args)
        return result
    for repeat in range(7):
        order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
        for name in order:
            gc.collect()
            value, timing = measure(batched, functions[name])
            assert value == before
            samples[name].append({key: value / batch for key, value in timing.items()})
    memory_samples = {'baseline': [], 'candidate': []}
    for repeat in range(3):
        order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
        for name in order:
            gc.collect()
            tracemalloc.start()
            value = functions[name](*args)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert value == before
            memory_samples[name].append({'current_bytes': current, 'peak_bytes': peak})
            del value
    summary = {name: summarize(values) for name, values in samples.items()}
    memory = {name: {'median_peak_bytes': statistics.median(sample['peak_bytes'] for sample in values),
                     'samples': values} for name, values in memory_samples.items()}
    return {'tips': sum(1 for _ in ete.iter_leaves(tr)), 'nodes': sum(1 for _ in tr.traverse()),
            'arity': arity, 'exclude_sister_pair': exclude, 'calls_per_repeat': batch,
            'largest_integer_bits': max(count.bit_length() for count in before),
            'counts_sha256': hashlib.sha256(str(before).encode()).hexdigest(),
            'samples': samples, 'summary': summary, 'memory': memory,
            'candidate_to_baseline_wall': summary['candidate']['wall_s']['median'] / summary['baseline']['wall_s']['median'],
            'candidate_to_baseline_cpu': summary['candidate']['cpu_s']['median'] / summary['baseline']['cpu_s']['median'],
            'candidate_to_baseline_peak_traced': memory['candidate']['median_peak_bytes'] / memory['baseline']['median_peak_bytes']}


def main():
    with benchmark_lock():
        from csubst import ete
        old, new, source_sha = prototype(ete)
        result = {'proposal': 9, 'baseline': '624bb8a', 'environment': environment(),
                  'source_sha256': source_sha, 'exactness': exactness(ete, old, new),
                  'recommendation': 'Adopt as a memory reduction; CPU/wall timing varies by shape, so no universal runtime speedup is claimed',
                  'method': 'One warmup per implementation; seven alternating timing repeats batched toward 40 ms CPU (1-40 calls); three alternating tracemalloc measurements; input tree generation outside timing; all preparation/tests/benchmarks hold shared lock; BLAS thread counts fixed to 1 before imports',
                  'memory_scope': 'Python tracemalloc peak per call; preexisting input tree and imported modules excluded; native allocator and RSS overhead excluded',
                  'benchmarks': {}}
        for dataset in ('PGK', 'PEPC'):
            tr = ete.PhyloNode(str(Path(BASELINE) / f'csubst/dataset/{dataset}.tree.nwk'), format=1)
            for exclude in (False, True):
                key = dataset + ('_exclude_sister' if exclude else '_allow_sister')
                result['benchmarks'][key] = timed_case(ete, old, new, tr, 10, exclude)
        for shape in ('balanced', 'pectinate', 'star', 'polytomous'):
            tr = ete.PhyloNode(newick(shape, 4096), format=1)
            for exclude in (False, True):
                key = shape + ('_exclude_sister' if exclude else '_allow_sister')
                result['benchmarks'][key] = timed_case(ete, old, new, tr, 16, exclude)
        (HERE / 'proposal09.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'exactness': result['exactness'], 'benchmarks': {
            key: {'wall_ratio': value['candidate_to_baseline_wall'],
                  'cpu_ratio': value['candidate_to_baseline_cpu'],
                  'baseline_peak_bytes': value['memory']['baseline']['median_peak_bytes'],
                  'candidate_peak_bytes': value['memory']['candidate']['median_peak_bytes']}
            for key, value in result['benchmarks'].items()}}, indent=2))


if __name__ == '__main__':
    main()

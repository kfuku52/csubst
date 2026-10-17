"""Measure fold-local reuse in optional structure-weighting parameter selection."""

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path

from audit import paired
import numpy as np

from csubst import epistasis


def cached_select(counts, context, features, masks, groups, rows, candidates, counters):
    original_base, original_predict = epistasis.base_probabilities, epistasis.predict
    bases, scores = OrderedDict(), {}
    retained = 0

    def base(values, train, mask, alpha):
        nonlocal retained
        key = (float(alpha), train.tobytes(), mask.shape, mask.tobytes())
        counters['base_calls'] += 1
        if key in bases:
            counters['base_hits'] += 1
            bases.move_to_end(key)
            return bases[key]
        value = original_base(values, train, mask, alpha)
        size = value.nbytes
        while bases and retained + size > 32 * 1024**2:
            retained -= bases.popitem(last=False)[1].nbytes
        if size <= 32 * 1024**2:
            bases[key] = value
            retained += size
        counters['base_retained_bytes'] = retained
        return value

    def predict(base, selected_context, features, beta, clip, mask):
        key = (selected_context.shape, selected_context.tobytes())
        if key not in scores:
            scores[key] = (selected_context @ features.T) / features.shape[1]
        score = scores[key]
        if not np.isfinite(score).all():
            raise ValueError('Nonfinite structure interaction score.')
        logweight = np.clip(beta * score, -clip, clip)
        logbase = np.full_like(base, -np.inf)
        np.log(base, out=logbase, where=(base > 0) & mask)
        logmass = logbase + logweight
        maximum = np.max(logmass, axis=1, keepdims=True)
        maximum = np.where(np.isfinite(maximum), maximum, 0.)
        mass = np.exp(logmass - maximum)
        totals = mass.sum(axis=1, keepdims=True)
        if np.any(mask.any(axis=1) & (totals[:, 0] <= 0)):
            raise ValueError('Structure weighting requires positive base probability on at least one valid site.')
        return np.divide(mass, totals, out=np.zeros_like(mass), where=totals > 0)

    epistasis.base_probabilities, epistasis.predict = base, predict
    try:
        return epistasis._select(counts, context, features, masks, groups, rows, candidates)
    finally:
        epistasis.base_probabilities, epistasis.predict = original_base, original_predict
        counters['score_retained_bytes'] = sum(value.nbytes for value in scores.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(43)
    n, sites, dimensions = 80, 256, 2
    counts = rng.poisson(.25, size=(n, sites)).astype(float)
    context = rng.normal(size=(n, dimensions))
    features = rng.normal(size=(sites, dimensions))
    masks = rng.random((n, sites)) > .1
    counts[~masks] = 0
    groups = np.repeat(np.arange(5), n // 5)
    rows = np.arange(n)
    candidates = epistasis.candidate_grid(dict(epistasis_beta_auto=True, epistasis_joint_auto=True,
                                              epistasis_clip_auto=True))
    counters = []
    def prototype():
        stats = Counter()
        value = cached_select(counts, context, features, masks, groups, rows, candidates, stats)
        counters.append(dict(stats))
        return value
    def check(a, b):
        assert a == b
    result = paired(lambda: epistasis._select(counts, context, features, masks, groups, rows, candidates),
                    prototype, check)
    result.update(branches=n, sites=sites, dimensions=dimensions, folds=5, candidates=len(candidates),
                  cache_counters=counters, scope='Inner selection only; synthetic valid inputs, optional feature.')
    args.result.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()

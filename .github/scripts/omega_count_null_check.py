#!/usr/bin/env python3
"""Independent known-mean count experiments for issue #46 (not ASR calibration)."""

import argparse
import contextlib
import hashlib
import io
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest, poisson

from _installed_package import configure_package_imports

configure_package_imports(Path(__file__).resolve().parents[2])
from csubst import omega  # noqa: E402


def run_case(setting, regime, trials=512, draws=999):
    """One independent count pair per row; no shared fitted prior or map."""
    if setting not in ('raw', 'symmetric', 'independent_null'):
        raise ValueError('Unknown setting')
    if regime not in ('sparse_S', 'dense', 'enriched'):
        raise ValueError('Unknown regime')
    if trials < 1 or draws < 19:
        raise ValueError('Require positive trials and at least 19 null draws')
    sub = 'any2spe'
    # All settings see the same observations; inference uses another stream.
    rng = np.random.default_rng(460910)
    # IID exposures make rows replicate draws from one mixture population,
    # so binomial intervals refer to its unconditional rejection probability.
    exposure = np.exp(rng.uniform(np.log(.5), np.log(2.), trials))
    exp_n = 8 * exposure
    exp_s = (0.1 if regime == 'sparse_S' else 8) / exposure
    cb = pd.DataFrame({'branch_id_1': np.arange(trials),
                       'branch_id_2': np.arange(trials) + trials})
    # Independent observations use explicit atom sums, not CSUBST's sampler.
    categories = dict(spe2spe=[0], spe2any=[0, 1], any2spe=[0, 2],
                      any2any=[0, 1, 2, 3], any2dif=[1, 3])
    for channel, expected in [('N', exp_n), ('S', exp_s)]:
        means = np.repeat(expected[:, None] / 2, 4, axis=1)
        factor = 4 if channel == 'N' and regime == 'enriched' else 1
        observed = poisson.rvs(means * factor, random_state=rng)
        for category, indices in categories.items():
            cb['EC' + channel + category] = means[:, indices].sum(axis=1)
            cb['OC' + channel + category] = observed[:, indices].sum(axis=1)
    # Including dif selects the production joint Poisson count engine.
    g = dict(calc_omega_pvalue=True, expectation_method='urn', output_stats=[sub, 'any2dif'],
             omega_pvalue_null_model='poisson', omega_pvalue_niter_schedule=[draws],
             pseudocount_mode='none' if setting == 'raw' else 'symmetric',
             pseudocount_alpha=0. if setting == 'raw' else 1.,
             calibrate_longtail=setting == 'independent_null',
             longtail_method='independent_null', longtail_null_niter=1000,
             random_seed=460911, float_tol=1e-12)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        cb = omega.get_omega(cb, g)
        if g['calibrate_longtail']:
            # The joint count engine only uses EC columns. Sentinels satisfy
            # the generic API's presence check and fail if tensors are read.
            cb = omega.calibrate_dsc(cb, g=g, ON_tensor=object(), OS_tensor=object())
        cb = omega.add_omega_empirical_pvalues(cb, None, None, g)
    seconds = time.perf_counter() - started
    p = cb['pomegaC' + sub].to_numpy()
    if not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise AssertionError('Missing or invalid P values')
    rejected = int((p <= .05).sum())
    ci = binomtest(rejected, trials).proportion_ci()
    return dict(setting=setting, regime=regime, trials=trials, draws=draws,
                rejected=rejected, rate=rejected / trials,
                interval_95=[ci.low, ci.high],
                excess_rejection_p=binomtest(rejected, trials, .05, alternative='greater').pvalue,
                pvalue_sha256=hashlib.sha256(p.astype('<f8').tobytes()).hexdigest(),
                seconds=seconds)


def validate_case(result):
    if result['regime'] == 'enriched':
        # Also reject a broken implementation returning P=1 for everything.
        if result['rate'] < .5:
            raise AssertionError('Less than 50% power for the fixed fourfold count alternative')
    elif result['excess_rejection_p'] < .001 / 6:
        # Six prespecified null cases; conservative tests are allowed.
        raise AssertionError('Excess rejection under the known-mean count null')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=512)
    parser.add_argument('--draws', type=int, default=999)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    # Warm up imports and numerical paths before timing repeated experiments.
    run_case('symmetric', 'dense', trials=8, draws=19)
    results = []
    for setting in ('raw', 'symmetric', 'independent_null'):
        for regime in ('sparse_S', 'dense', 'enriched'):
            samples = [run_case(setting, regime, args.trials, args.draws)
                       for _ in range(args.repeats)]
            result = samples[0]
            for sample in samples[1:]:
                if sample['pvalue_sha256'] != result['pvalue_sha256']:
                    raise AssertionError('Seeded repetition changed P values')
            result['seconds_repeats'] = [s['seconds'] for s in samples]
            validate_case(result)
            results.append(result)
            print(setting, regime, result['rate'], flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = dict(
        scope='Known-mean independent Poisson atoms, joint any2spe/any2dif inference; rejection diagnostics use any2spe. No ASR, fractional posterior mass, fitted-mean uncertainty, search selection, or q-value guarantee.',
        independent_unit='One IID log-uniform exposure and independently generated Poisson atoms with disjoint branch IDs; fixed smoothing and independent per-row calibration maps. Intervals average over the exposure distribution.',
        seeds=dict(observed=460910, inference=460911),
        platform=platform.platform(), python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
        peak_rss_bytes=int(rss * (1 if sys.platform == 'darwin' else 1024)),
        memory_scope='Peak of entire warmed-up benchmark process, not per setting.',
        results=results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

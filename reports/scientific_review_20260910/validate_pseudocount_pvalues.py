"""Independent conditional-count calibration experiment; no phylogenetic claim.

Run from the repository root with its Python environment. Each trial generates
an independent eight-row observed dataset and independent test null draws.
Only row zero contributes to each binomial interval, so shared prior/calibration
fits across rows are not miscounted as independent experimental replicates.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest, poisson

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from csubst import omega  # noqa: E402

WEIGHTS = {
    'spe2spe': [1, 0, 0, 0], 'spe2dif': [0, 1, 0, 0],
    'dif2spe': [0, 0, 1, 0], 'dif2dif': [0, 0, 0, 1],
    'any2any': [1, 1, 1, 1], 'any2spe': [1, 0, 1, 0],
    'any2dif': [0, 1, 0, 1], 'spe2any': [1, 1, 0, 0],
    'dif2any': [0, 0, 1, 1],
}
STATS = ['any2any', 'any2spe', 'any2dif']


def run(trials, draws):
    results = []
    started = time.time()
    for regime, scale_s in [('sparse_S', .04), ('heterogeneous_exposure', 1.)]:
        row_scales = np.geomspace(.5, 3., 8)[:, None]
        means = {'N': row_scales * np.array([1., .4, 2., .7]),
                 'S': row_scales[::-1] * np.array([.3, .6, .7, .4]) * scale_s}
        settings = [('none', 0., False), ('symmetric', 1., False),
                    ('empirical', 1., False), ('empirical', 'auto', False),
                    ('symmetric', 1., True)]
        for setting_id, (mode, alpha, calibrated) in enumerate(settings):
            pvalues = {s: [] for s in STATS}
            for trial in range(trials):
                # SciPy's observed sampler uses a separate seed from CSUBST.
                observed_rng = np.random.default_rng(np.random.SeedSequence([923, setting_id, trial, int(scale_s*100)]))
                data = {'branch_id_1': np.arange(8), 'branch_id_2': np.arange(8)+8}
                for ch in ['N', 'S']:
                    observed = poisson.rvs(means[ch], random_state=observed_rng)
                    for stat, weights in WEIGHTS.items():
                        data['OC'+ch+stat] = (observed*np.array(weights)).sum(axis=1)
                        data['EC'+ch+stat] = (means[ch]*np.array(weights)).sum(axis=1)
                cb = pd.DataFrame(data)
                g = dict(calc_omega_pvalue=True, expectation_method='urn',
                         output_stats=STATS, omega_pvalue_null_model='poisson',
                         omega_pvalue_niter_schedule=[draws], float_tol=1e-12,
                         pseudocount_mode=mode, pseudocount_alpha=alpha,
                         longtail_method='empirical', calibrate_longtail=calibrated,
                         random_seed=int(np.random.SeedSequence([671, setting_id, trial, int(scale_s*100)]).generate_state(1)[0]))
                with contextlib.redirect_stdout(io.StringIO()):
                    cb = omega.get_omega(cb, g)
                    if calibrated:
                        cb = omega.calibrate_dsc(cb, g=g)
                    cb = omega.add_omega_empirical_pvalues(cb, None, None, g)
                for stat in STATS:
                    pvalues[stat].append(float(cb['pomegaC'+stat].iloc[0]))
            for stat in STATS:
                p = np.asarray(pvalues[stat])
                for level in [.01, .05, .1]:
                    valid = np.isfinite(p)
                    rejected = int((p[valid] <= level).sum())
                    interval = binomtest(rejected, int(valid.sum())).proportion_ci()
                    results.append(dict(regime=regime, mode=mode, alpha=alpha,
                                        longtail='empirical' if calibrated else 'none', stat=stat,
                                        nominal=level, trials=trials, draws=draws,
                                        valid=int(valid.sum()), rejections=rejected,
                                        rejection_rate=rejected/int(valid.sum()),
                                        interval_95=[interval.low, interval.high]))
            print(regime, mode, alpha, calibrated, 'done; elapsed', round(time.time()-started, 1), flush=True)
    return dict(scope='Known-mean, independent-row and N/S Poisson atom count null. No ASR, fitted-parameter uncertainty, biological false-positive calibration, or BH guarantee.',
                observed_sampler='scipy.stats.poisson; independent seed per dataset',
                inference='production get_omega / calibrate_dsc / add_omega_empirical_pvalues',
                independent_unit='row zero of each independently generated eight-row dataset',
                results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=250)
    parser.add_argument('--draws', type=int, default=199)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('pseudocount_null_calibration.json'))
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.trials, args.draws), indent=2)+'\n')

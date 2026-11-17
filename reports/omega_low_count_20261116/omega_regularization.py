"""Compare point-estimate regularization to simulated omegaCany2spe."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pilot import EXPECTED, KEY, NULL, OBS, TRUTH


OMEGA = 'omegaCany2spe'
GENERATING = OMEGA + '_expected'


def regularized_estimate(obs_n, exp_n, obs_s, exp_s, prior_count):
    valid = (np.isfinite(obs_n) & (obs_n >= 0.) & np.isfinite(obs_s) & (obs_s >= 0.)
             & np.isfinite(exp_n) & (exp_n > 0.) & np.isfinite(exp_s) & (exp_s > 0.))
    out = np.full(valid.shape, np.nan)
    log_prior = np.log(prior_count)
    # Normalize S in log space so tiny expectations and OCS=0 stay defined.
    with np.errstate(divide='ignore', over='ignore', under='ignore'):
        log_s = np.log(obs_s[valid]) + np.log(exp_n[valid]) - np.log(exp_s[valid])
        log_num = np.logaddexp(np.log(obs_n[valid]), log_prior)
        log_den = np.logaddexp(log_s, log_prior)
        out[valid] = np.exp(log_num - log_den)
    return out


def compare(path, reference):
    frame = pd.read_csv(path, sep='\t', usecols=KEY + OBS + EXPECTED + [OMEGA]).merge(
        pd.read_csv(reference, sep='\t', usecols=KEY + NULL + TRUTH + [GENERATING]),
        on=KEY, how='outer', validate='one_to_one', indicator=True,
    )
    if not frame['_merge'].eq('both').all():
        raise ValueError(f'Pair populations differ for {path}.')
    obs_n, obs_s = frame[OBS].to_numpy().T
    exp_n, exp_s = frame[EXPECTED].to_numpy().T
    raw = frame[OMEGA].to_numpy()
    truth = frame[GENERATING].to_numpy()
    available = (np.isfinite(truth) & (truth >= 0.) & (exp_n > 0.) & (exp_s > 0.)
                 & np.isfinite(exp_n) & np.isfinite(exp_s) & np.isfinite(obs_n) & np.isfinite(obs_s)
                 & (obs_n >= 0.) & (obs_s >= 0.))
    support_s = frame[NULL[1]].ge(.5).to_numpy()
    enriched = truth >= 1.5
    subsets = {'all': available, 'near_null': np.abs(truth - 1.) < .01,
               'enriched': enriched, 'enriched_supported_s': enriched & support_s,
               'enriched_true_n': enriched & frame[TRUTH[0]].gt(0).to_numpy(), 'supported_s': support_s}
    with np.errstate(invalid='ignore', divide='ignore'):
        results = {'raw': raw, 'center_one': np.ones_like(raw)}
        for prior in (.01, .03, .1):
            results[f'regularized_{prior}'] = regularized_estimate(obs_n, exp_n, obs_s, exp_s, prior)
        for alpha in (.1, .3):
            results[f'symmetric_{alpha}'] = ((obs_n + alpha) / (exp_n + alpha)) / (
                (obs_s + alpha) / (exp_s + alpha)
            )
        for subset, group in subsets.items():
            supported = available & group
            include = supported & np.isfinite(raw)
            if not include.any():
                continue
            for method, estimate in results.items():
                prediction = np.log1p(estimate[include])
                target = np.log1p(truth[include])
                error = prediction - target
                yield dict(subset=subset, method=method, pairs=int(include.sum()),
                           raw_nonfinite=int((supported & ~np.isfinite(raw)).sum()),
                           nonfinite=int((supported & ~np.isfinite(estimate)).sum()),
                           log_estimate_sum=prediction.sum(), log_truth_sum=target.sum(),
                           squared_error_sum=np.square(error).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('recovered_pgk', type=Path)
    parser.add_argument('pepc_runs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = []
    for dataset in ('PGK', 'PEPC'):
        datasets = sorted((args.panel / dataset).glob('*/sim/sim_expected_omega.tsv'))
        for reference in datasets:
            run = reference.parents[1]
            scenario, seed = run.name.rsplit('-', 1)
            replay, fit = ((args.recovered_pgk / run.name / 'csubst_search', 'recovered') if dataset == 'PGK'
                           else (args.pepc_runs / run.name / 'codon_model', 'replay'))
            if not (replay / 'csubst_cb_2.tsv').is_file():
                replay, fit = run / 'default', 'original'
            rows.extend(dict(dataset=dataset, scenario=scenario, seed=int(seed), fit=fit, **row)
                        for row in compare(replay / 'csubst_cb_2.tsv', reference))
    pd.DataFrame(rows).to_csv(args.output, sep='\t', index=False, float_format='%.12g')


if __name__ == '__main__':
    main()

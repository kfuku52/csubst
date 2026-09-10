"""Independent-replicate validation of a predeclared pipeline statistic.

This module does not replace the conditional omega P-value engine. A fixed
calibration sample of complete null datasets supplies an empirical reference;
separate validation datasets measure its operating characteristics.
"""

import numpy as np
from scipy.stats import beta


def selected_statistic(frame, specification):
    """Maximum over the declared eligible family; an empty family is -inf."""
    statistic = specification['statistic']
    if statistic not in frame:
        raise ValueError('Missing selected statistic: {}'.format(statistic))
    selected = np.ones(len(frame), dtype=bool)
    excluded = specification.get('exclude_branch_ids', [])
    if excluded:
        columns = [c for c in frame.columns if c.startswith('branch_id_')]
        if not columns:
            raise ValueError('Missing branch columns for prespecified exclusions.')
        selected &= ~frame[columns].isin(excluded).any(axis=1).to_numpy()
    branch_ids = specification.get('branch_ids')
    if branch_ids is not None:
        columns = ['branch_id_{}'.format(i + 1) for i in range(len(branch_ids))]
        if not set(columns).issubset(frame):
            raise ValueError('Missing branch columns for the prespecified combination.')
        matches = (np.sort(frame[columns].to_numpy(dtype=int), axis=1) == np.sort(branch_ids)).all(axis=1)
        if not matches.any():
            raise ValueError('The prespecified branch combination is absent from the output family.')
        selected &= matches
    for column, threshold in specification.get('minimum_counts', {}).items():
        if column not in frame:
            raise ValueError('Missing eligibility count: {}'.format(column))
        counts = frame[column].to_numpy(dtype=float)
        if not np.isfinite(counts).all():
            raise ValueError('Nonfinite eligibility counts cannot be silently omitted.')
        selected &= counts >= float(threshold)
    values = frame.loc[selected, statistic].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError('Undefined statistic in the eligible family; revise the declared eligibility rule.')
    return float(values.max()) if values.size else -np.inf


def upper_tail_reference_pvalue(observed, calibration):
    reference = np.asarray(calibration, dtype=float)
    if reference.ndim != 1 or reference.size == 0 or np.isnan(reference).any() or np.isnan(observed):
        raise ValueError('A nonempty, defined independent calibration sample is required.')
    # Inclusive ties, including +/- infinity and the empty-family -infinity.
    return float((1 + np.count_nonzero(reference >= observed)) / (1 + reference.size))


def binomial_interval(hits, total, confidence=.95):
    if total < 1 or hits < 0 or hits > total:
        raise ValueError('Invalid independent-replicate binomial counts.')
    tail = (1 - confidence) / 2
    low = 0. if hits == 0 else float(beta.ppf(tail, hits, total - hits + 1))
    high = 1. if hits == total else float(beta.ppf(1 - tail, hits + 1, total - hits))
    return low, high


def summarize_validation(records, level=.05, fpr_limit=.06):
    if not 0 < level < 1 or not level <= fpr_limit < 1:
        raise ValueError('Require 0 < level <= fpr_limit < 1.')
    calibration = [r['selected_statistic'] for r in records if r['role'] == 'calibration']
    if not calibration:
        raise ValueError('No independent calibration datasets.')
    if any(r['truth'] != 'null' for r in records if r['role'] == 'calibration'):
        raise ValueError('Calibration datasets must be null.')
    output = []
    for record in records:
        if record['role'] != 'validation':
            continue
        p = upper_tail_reference_pvalue(record['selected_statistic'], calibration)
        output.append(dict(record, pipeline_p=p, rejected=p <= level))
    summary = {}
    for truth in ('null', 'alternative'):
        rows = [r for r in output if r['truth'] == truth]
        if not rows:
            continue
        hits = sum(r['rejected'] for r in rows)
        low, high = binomial_interval(hits, len(rows))
        summary[truth] = {'independent_datasets': len(rows), 'rejections': hits,
                          'rate': hits / len(rows), 'ci95': [low, high],
                          'empty_families': sum(r['selected_statistic'] == -np.inf for r in rows)}
    return output, {
        'inference_scope': 'independent_simulator_reference',
        'level': level, 'fpr_limit': fpr_limit,
        'calibration_datasets': len(calibration), 'minimum_p': 1 / (len(calibration) + 1),
        'validation': summary,
        'fpr_criterion_met': ('null' in summary and summary['null']['ci95'][1] <= fpr_limit
                              and 1 / (len(calibration) + 1) <= level),
        'scope_note': 'Conditional on the simulator and fixed reference sample; not a universal calibration claim.',
    }

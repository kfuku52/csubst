"""Exploratory structure weighting with external context and blocked cross-fitting.

The held-out response is used only for scoring. This is conditional count
prediction, not independent validation of an ASR pipeline or an epistasis test.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from csubst import ete


BETA_GRID = np.round(np.arange(0., 3.01, 0.1), 10)
CLIP_GRID = (1.5, 2., 2.5, 3., 4., 5.)


def validate_options(g):
    for name in ('epistasis_context_file', 'epistasis_context_source'):
        g[name] = str(g.get(name, '') or '').strip()
    g['epistasis_cv_clades'] = int(g.get('epistasis_cv_clades', 5))
    if g['epistasis_cv_clades'] < 3:
        raise ValueError('--epistasis_cv_clades must be at least 3 for nested CV.')
    if not g.get('epistasis_requested', False):
        return
    if not g['epistasis_context_file'] or not g['epistasis_context_source']:
        raise ValueError('Structure weighting requires --epistasis_context_file and '
                         '--epistasis_context_source describing independently obtained context; '
                         'evaluation substitutions must not be used to construct it (review ID 3).')
    if g.get('expectation_method') != 'urn' or g.get('asrv', 'each') != 'sn':
        raise ValueError('Structure weighting currently requires --expectation_method urn --asrv sn; '
                         'other ASRV modes do not share its cross-fitted prediction contract.')
    if g.get('asrv_report', False):
        raise ValueError('--asrv_report cannot describe clade-specific priors; use the automatic epistasis.json report.')
    if g.get('calc_omega_pvalue', False):
        raise ValueError('Structure weighting with --calc_omega_pvalue is not supported: '
                         'joint null generation and refitting are not calibrated (review IDs 2/3/4).')


def branch_key(node):
    """A rooted clade identity independent of row order and internal node names."""
    return json.dumps(sorted(str(n.name) for n in ete.iter_leaves(node)), separators=(',', ':'))


def tree_layout(tree, num_branch, target_clades=5):
    leaves = [str(n.name) for n in ete.iter_leaves(tree)]
    if len(set(leaves)) != len(leaves) or any(not name for name in leaves):
        raise ValueError('Structure weighting requires unique nonempty tip names.')
    nodes = list(tree.traverse())
    ids = [int(ete.get_prop(n, 'numerical_label')) for n in nodes]
    if len(set(ids)) != len(ids) or any(i < 0 or i >= num_branch for i in ids):
        raise ValueError('Invalid structure-weighting tree branch IDs.')
    keys = {int(ete.get_prop(n, 'numerical_label')): branch_key(n) for n in nodes if not ete.is_root(n)}
    if len(set(keys.values())) != len(keys):
        raise ValueError('Unary nodes give ambiguous clade identities; contract them before weighting.')
    frontier = list(tree.children)
    while len(frontier) < target_clades:
        candidates = [n for n in frontier if len(n.children) > 1]
        if not candidates:
            break
        split = min(candidates, key=lambda n: (-len(list(ete.iter_leaves(n))), branch_key(n)))
        frontier.remove(split)
        frontier.extend(split.children)
    frontier.sort(key=branch_key)
    # Ancestral connecting branches are a buffer: prediction only, never train
    # or score on them. Their predictions use all terminal clade blocks.
    groups = np.full(num_branch, -1, dtype=np.int64)
    for group, node in enumerate(frontier):
        for child in node.traverse():
            groups[int(ete.get_prop(child, 'numerical_label'))] = group
    depths = np.zeros(num_branch, dtype=np.float64)
    for node in tree.traverse('preorder'):
        if ete.is_root(node):
            continue
        length = float(node.dist or 0.)
        if not np.isfinite(length) or length < 0:
            raise ValueError('Structure weighting requires finite nonnegative branch lengths.')
        i = int(ete.get_prop(node, 'numerical_label'))
        parent = int(ete.get_prop(node.up, 'numerical_label'))
        depths[i] = depths[parent] + length
    return keys, groups, depths


def load_context(path, keys, num_branch, num_feature, source):
    if not str(source).strip():
        raise ValueError('Independent context needs a nonempty source description.')
    columns = ['context_{}'.format(i + 1) for i in range(num_feature)]
    raw = Path(path).read_bytes()
    rows = csv.DictReader(raw.decode('utf-8-sig').splitlines(), delimiter='\t')
    if rows.fieldnames != ['branch_key'] + columns:
        raise ValueError('Context TSV columns must be: branch_key, ' + ', '.join(columns))
    values = {}
    for row in rows:
        try:
            taxa = json.loads(row['branch_key'])
            if (not isinstance(taxa, list) or not taxa or
                    any(not isinstance(t, str) or not t for t in taxa) or len(set(taxa)) != len(taxa)):
                raise ValueError('Invalid taxa')
            key = json.dumps(sorted(taxa), separators=(',', ':'))
            vector = np.asarray([float(row[c]) for c in columns])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError('Invalid independent context row.') from exc
        if key in values or not np.isfinite(vector).all() or None in row:
            raise ValueError('Duplicate, nonfinite or malformed independent context row.')
        values[key] = vector
    if set(values) != set(keys.values()):
        raise ValueError('Independent context clades must match every non-root branch of the analysis tree.')
    matrix = np.zeros((num_branch, num_feature), dtype=np.float64)
    for i, key in keys.items():
        matrix[i] = values[key]
    return matrix, {'source': str(source), 'sha256': hashlib.sha256(raw).hexdigest(),
                    'independence': 'user_declared_external_not_statistically_verified',
                    'columns': columns}


def base_probabilities(counts, train_rows, mask, alpha):
    prior = counts[train_rows].sum(axis=0, dtype=np.float64) + alpha
    mass = np.where(mask, prior[None, :], 0.)
    totals = mass.sum(axis=1, keepdims=True)
    uniform = mask.astype(np.float64)
    np.divide(uniform, uniform.sum(axis=1, keepdims=True), out=uniform,
              where=uniform.sum(axis=1, keepdims=True) > 0)
    return np.divide(mass, totals, out=uniform, where=totals > 0)


def predict(base, context, features, beta, clip, mask):
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(clip) or clip <= 0:
        raise ValueError('beta must be finite and nonnegative; clip must be finite and positive.')
    score = (context @ features.T) / features.shape[1]
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



def log_score(counts, probabilities):
    positive = counts > 0
    if np.any(probabilities[positive] <= 0):
        return -np.inf
    return float(np.sum(counts[positive] * np.log(probabilities[positive])))


def candidate_grid(g):
    betas = BETA_GRID if g.get('epistasis_beta_auto', False) else [g.get('epistasis_beta_value', 0.)]
    joint = g.get('epistasis_joint_auto', False)
    alphas = g.get('epistasis_joint_alpha_grid', [0., 0.5, 1., 2.]) if joint else [g.get('asrv_dirichlet_alpha', 1.)]
    clips = (g.get('epistasis_joint_clip_grid', CLIP_GRID) if joint else CLIP_GRID) if g.get('epistasis_clip_auto', False) else [g.get('epistasis_clip_value', 3.)]
    candidates = sorted(set((float(b), float(a), float(c)) for b in betas for a in alphas for c in clips))
    if (not candidates or any(not np.isfinite(v).all() or v[0] < 0 or v[1] < 0 or v[2] <= 0 for v in candidates)):
        raise ValueError('Invalid structure-weighting beta/alpha/clip candidates.')
    return candidates


def _select(counts, context, features, masks, groups, rows, candidates):
    blocks = np.unique(groups[rows])
    if len(blocks) < 2 and len(candidates) > 1:
        raise ValueError('Insufficient independent clades for inner CV; reduce depth bins or use fixed parameters.')
    if len(candidates) == 1:
        return candidates[0], {'selection': 'fixed', 'inner_score': None}
    best, best_score, best_folds = candidates[0], -np.inf, []
    for beta, alpha, clip in candidates:
        scores = []
        for group in blocks:
            test = rows[groups[rows] == group]
            train = rows[groups[rows] != group]
            base = base_probabilities(counts, train, masks[test], alpha)
            p = predict(base, context[test], features, beta, clip, masks[test])
            total = counts[test].sum()
            # All branches (including zero-event rows) remain in the folds.
            scores.append(log_score(counts[test], p) / total if total > 0 else 0.)
        mean = float(np.mean(scores))
        # Deterministic ordering: on ties prefer beta=0, then less smoothing,
        # then a smaller clip. No negative beta can alias the null model.
        if mean > best_score + 1e-12:
            best, best_score, best_folds = (beta, alpha, clip), mean, scores
    if not np.isfinite(best_score):
        raise ValueError('All inner structure-weighting scores are undefined; use positive ASRV alpha or more training data.')
    return best, {'selection': 'argmax_mean_clade_log_score_ties_smallest_beta',
                  'inner_score': best_score if np.isfinite(best_score) else None,
                  'inner_fold_scores': [s if np.isfinite(s) else None for s in best_folds],
                  'candidate_count': len(candidates), 'all_scores_undefined': not np.isfinite(best_score)}


def crossfit(counts, context, features, masks, groups, partitions, g):
    counts, context, features = [np.asarray(v, dtype=np.float64) for v in (counts, context, features)]
    masks = np.asarray(masks, dtype=bool)
    groups, partitions = np.asarray(groups), np.asarray(partitions)
    if (counts.ndim != 2 or counts.shape[1] == 0 or masks.shape != counts.shape or context.ndim != 2 or features.ndim != 2 or
            context.shape != (counts.shape[0], features.shape[1]) or features.shape[0] != counts.shape[1] or
            features.shape[1] < 1 or groups.shape != (counts.shape[0],) or partitions.shape != groups.shape):
        raise ValueError('Incompatible structure-weighting array dimensions.')
    if (not all(np.isfinite(v).all() for v in (counts, context, features)) or np.any(counts < 0)):
        raise ValueError('Structure weighting requires finite features/context and nonnegative finite counts.')
    counts = np.where(masks, counts, 0.)
    candidates = candidate_grid(g)
    null_candidates = sorted(set((0., a, c) for _, a, c in candidates))
    result = np.zeros_like(counts)
    null_result = np.zeros_like(counts)
    parameters = np.zeros((counts.shape[0], 3))
    diagnostics = []
    for partition in sorted(set(partitions) - {-1}):
        eligible = (partitions == partition) & masks.any(axis=1)
        blocks = sorted(set(groups[eligible]) - {-1})
        required = 3 if len(candidates) > 1 or len(null_candidates) > 1 else 2
        if len(blocks) < required:
            raise ValueError('Structure weighting needs at least {} nonempty clades per depth bin; '
                             'reduce depth bins or use a larger tree.'.format(required))
        # Buffer branches (-1) are predictions only, with no response fed back
        # into training. They are excluded from reported outer scores.
        for group in blocks + ([-1] if np.any(eligible & (groups == -1)) else []):
            test = np.flatnonzero(eligible & (groups == group))
            train = np.flatnonzero(eligible & (groups >= 0) & (groups != group))
            selected, diag = _select(counts, context, features, masks, groups, train, candidates)
            baseline, _ = _select(counts, context, features, masks, groups, train, null_candidates)
            beta, alpha, clip = selected
            base = base_probabilities(counts, train, masks[test], alpha)
            p = predict(base, context[test], features, beta, clip, masks[test])
            null = base_probabilities(counts, train, masks[test], baseline[1])
            result[test], null_result[test], parameters[test] = p, null, selected
            ll, ll0 = log_score(counts[test], p), log_score(counts[test], null)
            diagnostics.append(dict(diag, partition=int(partition), clade=int(group),
                                    beta=beta, alpha=alpha, clip=clip, null_alpha=baseline[1],
                                    at_beta_grid_max=bool(g.get('epistasis_beta_auto') and beta == max(v[0] for v in candidates)),
                                    prediction_branch_ids=test.tolist(), training_branch_ids=train.tolist(),
                                    scored=bool(group >= 0), events=float(counts[test].sum()),
                                    outer_log_score=ll if group >= 0 and np.isfinite(ll) else None,
                                    outer_null_log_score=ll0 if group >= 0 and np.isfinite(ll0) else None))
    return {'probabilities': result, 'null_probabilities': null_result,
            'beta_by_branch': parameters[:, 0], 'alpha_by_branch': parameters[:, 1],
            'clip_by_branch': parameters[:, 2], 'beta_diag': {'candidates_beta_alpha_clip': candidates, 'outer_folds': diagnostics}}


def prepare_layout(g, num_branch, num_feature):
    keys, groups, depths = tree_layout(g['tree'], num_branch, g.get('epistasis_cv_clades', 5))
    context, provenance = load_context(g['epistasis_context_file'], keys, num_branch, num_feature,
                                       g['epistasis_context_source'])
    partitions = np.full(num_branch, -1, dtype=np.int64)
    eligible = np.asarray(sorted(keys), dtype=np.int64)
    if g.get('epistasis_beta_partition', 'global') == 'branch_depth':
        # Equal depths cannot be broken by numerical row order.
        bins = g.get('epistasis_branch_depth_bins', 3)
        edges = np.unique(np.quantile(depths[eligible], np.linspace(0, 1, bins + 1)[1:-1]))
        partitions[eligible] = np.searchsorted(edges, depths[eligible], side='right')
    else:
        partitions[eligible] = 0
    g['_epistasis_provenance'] = dict(provenance, schema_version=1,
                                     scope='exploratory_conditional_count_prediction',
                                     pipeline_calibrated=False, asrv='sn',
                                     counts_conditioned_on='supplied_counts; ASR not recomputed inside folds',
                                     fold_rule='deterministic_monophyletic_frontier_with_ancestral_buffer',
                                     branch_keys={str(i): key for i, key in keys.items()},
                                     clade_by_branch=groups.tolist(), partition_by_branch=partitions.tolist())
    return context, groups, partitions


def write_report(g, path):
    report = dict(g['_epistasis_provenance'])
    report['channels'] = {c: state['beta_diag'] for c, state in g['_epistasis_state'].items()}
    Path(path).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

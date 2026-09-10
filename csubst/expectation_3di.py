"""Independent fitted 3Di context for model-based nonsynonymous expectations.

IQ-TREE 2 ModelMorph checkpoints store reversible exchangeabilities in upper
triangle order (i, j), i < j. Read those ~10-digit values, not rounded report Qs.
The initial implementation deliberately supports uniform GTR/GTRX (+FQ) only.
"""
import gzip
import re

import numpy as np

from csubst import ete


CONTEXT_KEYS = ('3di_q', '3di_pi', '3di_rates', '3di_branch_lengths', '3di_orders', '3di_tip_invariant_mask')


def required(g):
    return (g.get('subcommand', 'search') in ('search', 'analyze', 'benchmark')
            and g.get('nonsyn_recode') == '3di20'
            and str(g.get('expectation_method', 'codon_model')).lower() == 'codon_model')


def validate_options(g):
    if not required(g):
        return
    if str(g.get('sa_asr_mode', 'direct')).lower() != 'direct':
        raise ValueError('Model-based 3Di expectations require --sa_asr_mode direct; '
                         'use --expectation_method urn explicitly for translate ASR.')
    model = str(g.get('sa_iqtree_model', 'GTR')).strip().upper()
    if model not in ('GTR', 'GTRX', 'GTR20', 'GTR+FQ', 'GTRX+FQ', 'GTR20+FQ'):
        raise ValueError('Model-based 3Di expectations currently require uniform GTR '
                         '(--sa_iqtree_model GTR or GTR+FQ). Rate mixtures and other '
                         '3Di models are not yet supported; --expectation_method urn '
                         'remains available explicitly.')


def validate_context(g, num_node=None, num_site=None):
    missing = [key for key in CONTEXT_KEYS if key not in g]
    if missing:
        raise ValueError('Missing fitted 3Di expectation context: ' + ', '.join(missing))
    q = np.asarray(g['3di_q'], dtype=float)
    pi = np.asarray(g['3di_pi'], dtype=float)
    rates = np.asarray(g['3di_rates'], dtype=float)
    lengths = np.asarray(g['3di_branch_lengths'], dtype=float)
    orders = np.asarray(g['3di_orders']).astype(str)
    invariant_mask = np.asarray(g['3di_tip_invariant_mask'])
    if invariant_mask.dtype != np.bool_ or invariant_mask.shape != rates.shape:
        raise ValueError('Invalid fitted 3Di tip-invariant mask.')
    if not np.array_equal(orders, list('ACDEFGHIKLMNPQRSTVWY')):
        raise ValueError('Invalid fitted 3Di state order.')
    if q.shape != (20, 20) or pi.shape != (20,):
        raise ValueError('Invalid fitted 3Di Q/frequency dimensions.')
    if not all(np.isfinite(a).all() for a in (q, pi, rates, lengths)):
        raise ValueError('Non-finite fitted 3Di model parameters.')
    off = q.copy()
    np.fill_diagonal(off, 0)
    if (np.any(off < 0) or np.any(pi < 0) or np.any(lengths < 0)
            or rates.ndim != 1 or lengths.ndim != 1 or not np.all(rates == 1)):
        raise ValueError('Invalid fitted 3Di rates or branch lengths.')
    if (not np.allclose(q.sum(axis=1), 0, atol=1e-10)
            or not np.isclose(pi.sum(), 1)
            or not np.allclose(pi @ q, 0, atol=1e-10)
            or not np.isclose(-pi @ np.diag(q), 1)):
        raise ValueError('Fitted 3Di Q must be stationary and normalized to one substitution/site.')
    if num_node is not None and lengths.shape != (num_node,):
        raise ValueError('Fitted 3Di branch axis mismatch.')
    if num_site is not None and rates.shape != (num_site,):
        raise ValueError('Fitted 3Di site axis mismatch.')


def read_context(g, paths, direct_tree, state_columns, num_site, tip_invariant_mask):
    validate_options(g)
    with open(paths['iqtree']) as handle:
        report = handle.read()
    if not re.search(r'Model of substitution:\s*GTRX\+FQ\s*\n', report):
        raise ValueError('Unexpected fitted 3Di model: expected GTRX+FQ.')
    if not re.search(r'Model of rate heterogeneity:\s*Uniform', report):
        raise ValueError('Expected uniform fitted 3Di site rates.')
    symbols = [str(col).removeprefix('p_') for col in state_columns]
    morph = '0123456789ABCDEFGHIJ'
    if symbols != sorted(set(symbols), key=morph.index):
        raise ValueError('Unexpected IQ-TREE 3Di probability-state ordering.')
    indices = np.array([morph.index(s) for s in symbols])
    n = len(indices)
    with gzip.open(paths['checkpoint'], 'rt') as handle:
        checkpoint = handle.read()
    match = re.search(r'^ModelMorph:\n((?:[ \t].*\n)*)', checkpoint, re.M)
    rate_match = None if match is None else re.search(r'^\s+rates:\s*([^\n]+)', match[1], re.M)
    if rate_match is None:
        raise ValueError('IQ-TREE checkpoint has no fitted ModelMorph exchangeabilities.')
    exchangeabilities = np.array([float(x.strip()) for x in rate_match[1].split(',')])
    if (exchangeabilities.size != n * (n - 1) // 2
            or not np.isfinite(exchangeabilities).all() or np.any(exchangeabilities < 0)):
        raise ValueError('Invalid IQ-TREE ModelMorph exchangeabilities.')
    small_q = np.zeros((n, n))
    small_q[np.triu_indices(n, 1)] = exchangeabilities / n
    small_q += small_q.T
    np.fill_diagonal(small_q, -small_q.sum(axis=1))
    scale = -np.diag(small_q).mean()
    if scale <= 0:
        raise ValueError('Fitted 3Di model has zero substitution rate.')
    q = np.zeros((20, 20))
    q[np.ix_(indices, indices)] = small_q / scale
    pi = np.zeros(20)
    pi[indices] = 1 / n
    reference = g['rooted_tree']
    # Match clades explicitly; numerical labels alone cannot establish identity.
    clade_ids = {frozenset(ete.get_leaf_names(node)): int(ete.get_prop(node, 'numerical_label'))
                 for node in reference.traverse()}
    lengths = np.zeros(len(clade_ids))
    for node in direct_tree.traverse():
        key = frozenset(ete.get_leaf_names(node))
        if key not in clade_ids or clade_ids[key] != int(ete.get_prop(node, 'numerical_label')):
            raise ValueError('Fitted 3Di tree branch identity mismatch.')
        lengths[clade_ids[key]] = 0 if ete.is_root(node) else float(node.dist)
    context = dict(zip(CONTEXT_KEYS, (q, pi, np.ones(num_site), lengths,
                                    np.array(list('ACDEFGHIKLMNPQRSTVWY')),
                                    np.asarray(tip_invariant_mask, dtype=bool))))
    validate_context(context, len(clade_ids), num_site)
    g.update(context)
    print("Model expectations: N=3Di GTRX+FQ (uniform); S=codon.", flush=True)


def root_posterior(g, tree_obj, tip_state):
    """Pruning likelihood for the inserted root, conditional on the fitted Q.

    Missing tip states contribute likelihood one; wholly missing columns retain
    zero posterior mass to match the state-tensor missing-data convention.
    """
    from scipy.linalg import expm
    q = np.asarray(g['3di_q'], dtype=np.float64)
    pi = np.asarray(g['3di_pi'], dtype=np.float64)
    likelihood = {}
    any_observed = np.zeros(tip_state.shape[1], dtype=bool)
    for node in tree_obj.traverse('postorder'):
        idx = int(ete.get_prop(node, 'numerical_label'))
        if ete.is_leaf(node):
            block = np.array(tip_state[idx], dtype=np.float64, copy=True)
            observed = block.sum(axis=1) > 0
            any_observed |= observed
            block[~observed] = 1
        else:
            block = np.ones((tip_state.shape[1], 20))
            for child in ete.get_children(node):
                child_id = int(ete.get_prop(child, 'numerical_label'))
                transition = expm(q * float(child.dist))
                block *= likelihood.pop(child_id) @ transition.T
                scale = block.max(axis=1)
                if np.any(scale <= 0):
                    raise ValueError('Zero likelihood in fitted 3Di root reconstruction.')
                block /= scale[:, None]
        likelihood[idx] = block
    root_id = int(ete.get_prop(tree_obj, 'numerical_label'))
    posterior = likelihood[root_id] * pi
    posterior /= posterior.sum(axis=1, keepdims=True)
    posterior[~any_observed] = 0
    return posterior

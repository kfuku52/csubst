"""Shared fitted codon model precision and observation eligibility."""
import gzip
from decimal import Decimal
import hashlib
from pathlib import Path
import re

import numpy as np

from csubst import ete, parser_misc


def _fitted_generator(g, codons):
    state_path = Path(g['path_iqtree_state'])
    checkpoint = Path(str(state_path).removesuffix('.state') + '.ckp.gz')
    with gzip.open(checkpoint, 'rt') as handle:
        text = handle.read()
    block = re.search(r'^ModelCodon:\n((?:[ \t].*\n)+)', text, flags=re.MULTILINE)
    if block is None:
        raise ValueError('Missing ModelCodon checkpoint for the fitted codon model.')
    parameters = {}
    parameter_tokens = {}
    for name in ('omega', 'kappa'):
        value = re.search(r'^\s+' + name + r':\s+(\S+)', block[1], flags=re.MULTILINE)
        if value is None or not np.isfinite(float(value[1])) or float(value[1]) <= 0:
            raise ValueError('Invalid checkpoint parameter: ' + name)
        parameters[name] = float(value[1])
        parameter_tokens[name] = value[1]
    if str(g['iqtree_model']).upper() == 'GY+FQ':
        pi = np.full(len(codons), 1/len(codons))
    else:
        log = Path(g['path_iqtree_log']).read_text()
        matches = re.findall(r'^Empirical state frequencies:\s*([^\n]+)', log, flags=re.MULTILINE)
        if not matches:
            raise ValueError('Missing full-precision empirical frequencies for the fitted codon model.')
        frequencies = np.array([float(v) for v in matches[-1].split()])
        if frequencies.size != len(codons):
            raise ValueError('IQ-TREE empirical frequency order/size does not match the sense codon alphabet.')
        # IQ-TREE's sense-codon alphabet is lexicographic A,C,G,T. Explicitly
        # remap it to the state-file axis; never assume an incidental row order.
        mapping = dict(zip(sorted(codons.tolist()), frequencies.tolist()))
        pi = np.array([mapping[c] for c in codons])
        if not np.isfinite(pi).all() or (pi < 0).any() or not np.isclose(pi.sum(), 1., rtol=0, atol=1e-8):
            raise ValueError('Invalid full-precision fitted codon frequencies.')
        pi /= pi.sum()  # log serialization roundoff only
    local = dict(g, **parameters, equilibrium_frequency=pi, float_type=np.float64)
    q = parser_misc.get_mechanistic_instantaneous_rate_matrix(local)
    return q, pi, dict(parameters, checkpoint=str(checkpoint), parameter_tokens=parameter_tokens, frequency_source='verbose_log_or_FQ')



def prepare(g, strict=False):
    """Select identical available parameter precision for every command.

    A supplied report-only fit remains usable for conditional inference. Fitted
    bootstrap explicitly requires the precise generator and its likelihood check.
    """
    model = str(g.get('substitution_model', g.get('iqtree_model', '')))
    base = re.sub(r'\+(?:G\d*|R\d*|I)(?=\+|$)', '', model)
    state = g.get('path_iqtree_state')
    checkpoint = Path(str(state).removesuffix('.state') + '.ckp.gz') if state else None
    log = Path(g['path_iqtree_log']) if g.get('path_iqtree_log') else None
    precise = (base in ('GY+F', 'GY+FQ') and checkpoint is not None and checkpoint.is_file()
               and (base == 'GY+FQ' or (log is not None and log.is_file()
                    and 'Empirical state frequencies:' in log.read_text())))
    if strict and not precise:
        raise ValueError('Fitted bootstrap requires a matching .ckp.gz and full-precision frequencies.')
    provenance = {'precision': 'reported_parameters', 'model': model}
    if precise:
        q, pi, source = _fitted_generator(dict(g, iqtree_model=base), np.asarray(g['codon_orders']))
        # Compare against serialization intervals, not a permissive fit tolerance.
        log_text = log.read_text() if log is not None and log.is_file() else ''
        def half_unit(token):
            exponent = Decimal(str(token)).as_tuple().exponent
            if not isinstance(exponent, int):
                raise ValueError('Nonfinite fitted parameter serialization.')
            return .5 * 10. ** exponent
        for name in ('omega', 'kappa'):
            if g.get(name) is None:
                continue
            tokens = re.findall(r'\(' + name + r'\)\s*:\s*([-+0-9.eE]+)', log_text)
            token = tokens[-1] if tokens else str(g[name])
            precise_token = source.get('parameter_tokens', {}).get(name, str(source[name]))
            tolerance = half_unit(token) + half_unit(precise_token) + 1e-13 * max(1., abs(source[name]))
            if abs(float(g[name]) - source[name]) > tolerance:
                raise ValueError('Checkpoint/report fitted parameter mismatch: ' + name)
        report_path = g.get('path_iqtree_iqtree')
        report = Path(report_path).read_text() if report_path else ''
        entries = dict(re.findall(r'pi\(\s*([A-Z]+)\s*\)\s*=\s*([-+0-9.eE]+)', report))
        previous = np.asarray(g['equilibrium_frequency'])
        for i, codon in enumerate(g['codon_orders']):
            token = entries.get(codon, str(previous[i]))
            # Precise frequency log itself is rounded at ten decimal places.
            if abs(float(token) - pi[i]) > half_unit(token) + 1e-9:
                raise ValueError('Precise log/report fitted frequency mismatch.')
        g.update(instantaneous_codon_rate_matrix=q, equilibrium_frequency=pi,
                 omega=source['omega'], kappa=source['kappa'])
        provenance.update(source, precision='checkpoint_and_precise_frequencies')
    digest = hashlib.sha256()
    for key in ('instantaneous_codon_rate_matrix', 'equilibrium_frequency'):
        digest.update(np.asarray(g[key], dtype='<f8').tobytes())
    provenance['generator_sha256'] = digest.hexdigest()
    g['fitted_model_provenance'] = provenance
    g['scan_ctmc_model_precision'] = provenance


def observation_masks(tree, states):
    """Observed descendant support and eligible edges from original emissions.

    Partial ambiguity is observed; zero and constant likelihood rows carry no
    observation. Root rows are never edges. This does not change pruning evidence.
    """
    observed = np.zeros(states.shape[:2], dtype=bool)
    eligible = np.zeros_like(observed)
    for node in tree.traverse('postorder'):
        i = int(ete.get_prop(node, 'numerical_label'))
        children = ete.get_children(node)
        if children:
            for child in children:
                observed[i] |= observed[int(ete.get_prop(child, 'numerical_label'))]
        else:
            rows = np.asarray(states[i])
            observed[i] = (rows.sum(axis=1) > 0) & (np.ptp(rows, axis=1) > 0)
    for node in tree.traverse():
        if not ete.is_root(node):
            i = int(ete.get_prop(node, 'numerical_label'))
            eligible[i] = observed[i] & observed[int(ete.get_prop(node.up, 'numerical_label'))]
    return observed, eligible

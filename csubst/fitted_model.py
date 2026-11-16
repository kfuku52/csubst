"""Shared fitted codon model precision and observation eligibility."""
import gzip
from decimal import Decimal
import hashlib
from pathlib import Path
import re

import numpy as np

from csubst import ete, parser_misc


def checkpoint_path(state_path):
    return Path(str(state_path).removesuffix('.state') + '.ckp.gz')


def checkpoint_block(text, name):
    match = re.search(r'^' + re.escape(name) + r':\n((?:[ \t].*\n)+)', text, re.MULTILINE)
    if match is None:
        raise ValueError('Missing {} checkpoint block for the fitted codon model.'.format(name))
    return match[1]


def _fitted_generator(g, codons):
    checkpoint = checkpoint_path(g['path_iqtree_state'])
    with gzip.open(checkpoint, 'rt') as handle:
        block = checkpoint_block(handle.read(), 'ModelCodon')
    parameters = {}
    parameter_tokens = {}
    for name in ('omega', 'kappa'):
        value = re.search(r'^\s+' + name + r':\s+(\S+)', block, flags=re.MULTILINE)
        if value is None or not np.isfinite(float(value[1])) or float(value[1]) <= 0:
            raise ValueError('Invalid checkpoint parameter: ' + name)
        parameters[name] = float(value[1])
        parameter_tokens[name] = value[1]
    model = str(g['iqtree_model'])
    hybrid = model.startswith('ECMK07_GY0K+')
    checkpoint_freq = re.search(r'^[ \t]+state_freq:[ \t]*(.*)$', block, re.MULTILINE) if hybrid else None
    frequency_source = 'verbose_log_or_FQ'
    if model.endswith('+FQ'):
        pi = np.full(len(codons), 1/len(codons))
    else:
        if checkpoint_freq is not None:
            frequencies = np.array([float(v) for v in checkpoint_freq[1].split(',')])
            frequency_source = 'checkpoint_state_freq'
        else:
            log_path = g.get('path_iqtree_log')
            log = Path(log_path).read_text() if log_path and Path(log_path).is_file() else ''
            matches = re.findall(r'^Empirical state frequencies:\s*([^\n]+)', log, flags=re.MULTILINE)
            if not matches:
                raise ValueError('Missing full-precision codon frequencies in the IQ-TREE checkpoint and verbose log.')
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
    if hybrid:
        q = parser_misc.get_ecm_omega_rate_matrix(local)
    else:
        q = parser_misc.get_mechanistic_instantaneous_rate_matrix(local)
    return q, pi, dict(parameters, checkpoint=str(checkpoint), parameter_tokens=parameter_tokens, frequency_source=frequency_source)


def _frequency_reference(previous, codons, log_text):
    """Reproduce the requested IQ-TREE frequency floor on the raw reference."""
    command = re.search(r'^Command:[ \t]+([^\n]+)', log_text, re.MULTILINE)
    if command is None or re.search(r'(?:^|\s)--inc-zero-freq(?:\s|$)', command[1]) is None:
        return previous
    match = re.search(r'(?:^|\s)--min-freq\s+([\d.eE+-]+)(?:\s|$)', command[1])
    if match is None:
        return previous
    floor = float(match[1])
    if not np.isfinite(floor) or not 0 < floor < 1 or previous.min() >= floor:
        return previous
    # IQ-TREE takes the residual from the first maximum in lexical codon order.
    max_codon = min(np.asarray(codons)[previous == previous.max()])
    adjusted = np.maximum(previous, floor)
    adjusted[np.asarray(codons) == max_codon] += 1 - adjusted.sum()
    return adjusted



def prepare(g, strict=False):
    """Select identical available parameter precision for every command.

    A supplied report-only fit remains usable for conditional inference. Fitted
    bootstrap explicitly requires the precise generator and its likelihood check.
    """
    model = str(g.get('substitution_model', g.get('iqtree_model', '')))
    base = re.sub(r'\+(?:G\d*|R\d*|I)(?=\+|$)', '', model)
    state = g.get('path_iqtree_state')
    checkpoint = checkpoint_path(state) if state else None
    log = Path(g['path_iqtree_log']) if g.get('path_iqtree_log') else None
    hybrid = base in ('ECMK07_GY0K+F', 'ECMK07_GY0K+FQ')
    precise = (base in ('GY+F', 'GY+FQ', 'ECMK07_GY0K+F', 'ECMK07_GY0K+FQ')
               and checkpoint is not None and checkpoint.is_file()
               and (hybrid or base.endswith('+FQ') or (log is not None and log.is_file()
                    and 'Empirical state frequencies:' in log.read_text())))
    if strict and (not precise or base not in ('GY+F', 'GY+FQ')):
        raise ValueError('Fitted bootstrap requires a matching .ckp.gz and full-precision frequencies.')
    if base.startswith('ECMK07_GY0K+') and not precise:
        raise ValueError('ECMK07_GY0K requires the matching IQ-TREE .ckp.gz. Refit with --iqtree_redo yes.')
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
        frequency_atol = 1e-9 + np.finfo(previous.dtype).eps * previous.max()
        if hybrid and source.get('frequency_source') == 'checkpoint_state_freq':
            # IQ-TREE 3 can print unadjusted empirical frequencies to the log
            # and omit the final vector from the report. Check the explicit
            # frequency adjustment against the raw alignment in that case.
            if not np.allclose(previous, pi, rtol=0, atol=frequency_atol):
                previous = _frequency_reference(previous.astype(float), g['codon_orders'], log_text)
        for i, codon in enumerate(g['codon_orders']):
            token = entries.get(codon)
            if token is None:
                differs = abs(previous[i] - pi[i]) > frequency_atol
            else:
                differs = abs(float(token) - pi[i]) > half_unit(token) + 1e-9
            if differs:
                raise ValueError('Precise log/report fitted frequency mismatch.')
        if base.startswith('ECMK07_GY0K+'):
            # IQ-TREE's printed omega_E is proportional to the stationary
            # N/S flux ratio. Use it to detect mismatched sidecars, not as
            # the multiplier used to build Q.
            match = re.search(r'\(omega_E\)\s*:\s*([-+0-9.eE]+)', report)
            if match:
                nonsyn = parser_misc.codon_nonsynonymous_mask(g)
                flux = pi[:, None] * q
                n = flux[nonsyn].sum()
                reconstructed = (n / (-pi @ q.diagonal() - n)) * (0.21 / 0.79)
                if abs(reconstructed - float(match[1])) > half_unit(match[1]) + 1e-9:
                    raise ValueError('Checkpoint/report ECMK07_GY0K omega_E mismatch.')
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

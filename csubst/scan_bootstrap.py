"""Fitted uniform-codon parametric bootstrap for the complete scan pipeline.

The generating Q and branch lengths are fitted once under a homogeneous null.
Each independent pseudo-alignment is fitted again on the supplied topology by
IQ-TREE, then passes through recoding, ASR, site filtering and scan discovery.
This is model-conditional bootstrap inference, not finite-sample exact testing.
"""

import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import expm

from csubst import __version__, cli_io, ete, runtime, scan_statistics, sequence


def capture_options(args, parser):
    """Keep the original CLI values, including automatic recoding choices.

    Resolved configuration contains fitted/data-dependent state and must never
    be recycled as if it were the original analysis procedure.
    """
    options = {key: value for key, value in vars(args).items()
               if key not in ('handler', 'subcommand') and value is not None}
    for action in cli_io._path_actions(parser):
        value = options.get(action.dest)
        if isinstance(value, str) and value and Path(value).exists():
            options[action.dest] = str(Path(value).resolve())
    return options


def validate_options(g):
    if g.get('scan_pvalue_calibration') != 'parametric_bootstrap':
        return
    if int(g.get('scan_permutation_seed', 1)) < 0:
        raise ValueError('Scan parametric_bootstrap requires a nonnegative --scan_permutation_seed.')
    if g.get('nonsyn_recode', 'no') == '3di20':
        raise ValueError('Scan parametric_bootstrap currently supports codon/AA recoding, not 3Di observation models.')
    model = str(g.get('iqtree_model', ''))
    if model.upper() not in ('GY+F', 'GY+FQ'):
        raise ValueError('Scan parametric_bootstrap requires --iqtree_model GY+F or GY+FQ; '
                         'other model families, rate mixtures and model selection are not supported.')
    if not isinstance(g.get('scan_cli_options'), dict):
        raise ValueError('Scan parametric_bootstrap requires the original scan_cli_options (provided by the CLI).')
    for suffix in ('state', 'treefile', 'rate', 'iqtree', 'log'):
        if g['scan_cli_options'].get('iqtree_' + suffix, 'infer') not in ('infer', '', None):
            raise ValueError('Scan parametric_bootstrap requires fresh ASR on the supplied topology; '
                             'explicit IQ-TREE intermediate inputs are not supported.')


def requires_precise_model(g):
    return g.get('scan_pvalue_calibration') == 'parametric_bootstrap'


def require_precise_fit(g):
    if not requires_precise_model(g):
        return
    explicit_state = g.get('scan_cli_options', g).get('iqtree_state', 'infer')
    prefix = (str(explicit_state).removesuffix('.state') if explicit_state not in ('infer', '', None)
              else runtime.infer_iqtree_output_prefix(g['alignment_file'], g['iqtree_outdir']))
    checkpoint = Path(prefix + '.ckp.gz')
    log = Path(g['iqtree_log']) if g.get('iqtree_log', 'infer') not in ('infer', '', None) else Path(prefix + '.log')
    needs_frequencies = str(g.get('iqtree_model', '')).upper() != 'GY+FQ'
    if (not checkpoint.is_file() or not log.is_file()
            or (needs_frequencies and 'Empirical state frequencies:' not in log.read_text())):
        if explicit_state not in ('infer', '', None):
            raise ValueError('Joint/bridge GY scans require a matching .ckp.gz and precise log beside the supplied state; '
                             'omit explicit IQ-TREE intermediates to refit.')
        print('Scan bootstrap requires a checkpoint and full-precision frequencies; refitting IQ-TREE.', flush=True)
        g['iqtree_redo'] = True


def prepare_observation_model(g):
    """Compatibility entry point; all commands use the shared fitted loader."""
    from csubst import fitted_model
    fitted_model.prepare(g, strict=g.get('scan_pvalue_calibration') == 'parametric_bootstrap')


def _precise_newick(tr):
    if ete._backend == 'ete4':
        from ete4.parser.newick import PARSERS
        parser = copy.deepcopy(PARSERS[1])
        for group in ('leaf', 'internal'):
            parser[group][1]['write'] = lambda value: format(value, '.17g')
        return tr.write(parser=parser, format_root_node=True)
    return tr.write(format=1, dist_formatter='%.17g', format_root_node=True)


def _fitted_generator(g, codons):
    from csubst import fitted_model
    return fitted_model._fitted_generator(g, codons)


def alignment_loglikelihood(model, sequences):
    """Independent scaled pruning check of the reconstructed fitted generator."""
    tr = ete.PhyloNode(model['topology'], format=1)
    lookup = {c: i for i,c in enumerate(model['codons'])}
    values: dict[int, np.ndarray] = {}
    logscale = np.zeros(model['sites'])
    matrices = {}
    for node in tr.traverse('postorder'):
        if ete.is_leaf(node):
            seq = sequences[node.name].upper().replace('U', 'T')
            value = np.ones((model['sites'], len(lookup)))
            for site in range(model['sites']):
                token = seq[site*3:site*3+3]
                if token in lookup:
                    value[site] = 0
                    value[site, lookup[token]] = 1
        else:
            value = np.ones((model['sites'], len(lookup)))
            for child in ete.get_children(node):
                if child.dist not in matrices:
                    matrices[child.dist] = expm(model['q'] * child.dist)
                value *= values[id(child)] @ matrices[child.dist].T
                scale = value.sum(axis=1)
                if (scale <= 0).any():
                    raise ValueError('Zero likelihood under the reconstructed fitted codon model.')
                logscale += np.log(scale)
                value /= scale[:, None]
        values[id(node)] = value
    return float(np.sum(np.log(values[id(tr)] @ model['pi']) + logscale))


def prepare_model(g):
    """Snapshot the fitted model before any analysis site filtering/rescaling."""
    validate_options(g)
    codons = np.asarray(g['codon_orders'], dtype=str)
    q, pi, provenance = _fitted_generator(g, codons)
    n = len(codons)
    if q.shape != (n, n) or pi.shape != (n,) or not np.isfinite(q).all() or not np.isfinite(pi).all():
        raise ValueError('Invalid fitted codon generator dimensions/values.')
    off_diagonal = q.copy()
    np.fill_diagonal(off_diagonal, 0.)
    if (off_diagonal < 0).any() or (pi < 0).any() or not np.isclose(pi.sum(), 1., atol=1e-8):
        raise ValueError('Invalid fitted codon generator or frequencies.')
    if not np.allclose(q.sum(axis=1), 0., atol=1e-8) or not np.allclose(pi @ q, 0., atol=1e-8):
        raise ValueError('Fitted codon generator must be conservative and stationary.')
    rates = np.asarray(g['iqtree_rate_values'], dtype=float)
    if rates.ndim != 1 or not rates.size or not np.allclose(rates, 1., atol=1e-8):
        raise ValueError('Scan parametric_bootstrap requires uniform unit site rates before filtering.')
    sequences = sequence.read_fasta(g['alignment_file'])
    topology = _precise_newick(g['tree'])
    tip_names = ete.get_leaf_names(g['tree'])
    if len(tip_names) != len(set(tip_names)) or set(tip_names) != set(sequences):
        raise ValueError('Bootstrap alignment and fitted tree require identical unique tip names.')
    missing = {}
    for name, seq in sequences.items():
        if len(seq) != 3 * rates.size:
            raise ValueError('Bootstrap must use the complete, unfiltered codon alignment.')
        tokens = [seq[i:i + 3].upper().replace('U', 'T') for i in range(0, len(seq), 3)]
        allowed_missing = {'---', 'NNN', '???'}
        if any(c not in codons and c not in allowed_missing for c in tokens):
            raise ValueError('Bootstrap supports unambiguous sense codons or fully missing codons (---/NNN/???); '
                             'partial ambiguity requires an explicit observation model.')
        missing[name] = np.array([c in allowed_missing for c in tokens], dtype=bool)
    for node in g['tree'].traverse():
        if not ete.is_root(node) and (not np.isfinite(node.dist) or node.dist < 0):
            raise ValueError('Bootstrap fitted branch lengths must be finite and nonnegative.')
    model = dict(q=q, pi=pi, codons=codons, topology=topology, sites=int(rates.size), missing=missing)
    report = Path(g['path_iqtree_iqtree']).read_text()
    match = re.search(r'Log-likelihood of the tree:\s*([-+\d.eE]+)', report)
    if match is None:
        raise ValueError('Missing IQ-TREE likelihood to validate the bootstrap generator.')
    reported = float(match[1])
    reproduced = alignment_loglikelihood(model, sequences)
    if not np.isfinite(reported) or not np.isfinite(reproduced):
        raise ValueError('Bootstrap likelihoods must be finite.')
    tolerance = max(.001, 1e-8*model['sites'])
    difference = abs(reported - reproduced)
    mismatch = difference > tolerance
    if mismatch:
        warnings.warn(
            'Bootstrap generator does not reproduce the fitted IQ-TREE likelihood: {} versus {}. '
            'Continuing bootstrap with the reconstructed generator; likelihood mismatch is recorded in provenance.'.format(reproduced, reported),
            RuntimeWarning, stacklevel=2,
        )
    model['provenance'] = dict(provenance, reported_loglik=reported, reproduced_loglik=reproduced,
                              likelihood_check='warning' if mismatch else 'matched',
                              likelihood_absolute_difference=difference, likelihood_tolerance=tolerance)
    return model


def transition_matrices(model):
    """Compute the complete CTMC transition, including multiple substitutions."""
    tr = ete.PhyloNode(model['topology'], format=1)
    matrices = {}
    for node in tr.traverse():
        if ete.is_root(node) or node.dist in matrices:
            continue
        p = expm(model['q'] * node.dist)
        if not np.isfinite(p).all() or p.min() < -1e-12 or not np.allclose(p.sum(axis=1), 1., atol=1e-10):
            raise ValueError('Invalid CTMC transition matrix in scan bootstrap.')
        # Only machine-roundoff correction after validation, not a model fallback.
        p = np.maximum(p, 0.)
        p /= p.sum(axis=1, keepdims=True)
        cumulative = p.cumsum(axis=1)
        cumulative[:, -1] = 1.
        matrices[node.dist] = cumulative
    return matrices


def simulate_alignment(model, rng, transitions=None):
    if transitions is None:
        transitions = transition_matrices(model)
    tr = ete.PhyloNode(model['topology'], format=1)
    states = {}
    sequences = {}
    for node in tr.traverse('preorder'):
        if ete.is_root(node):
            states[id(node)] = rng.choice(len(model['pi']), model['sites'], p=model['pi'])
        else:
            cumulative = transitions[node.dist][states[id(node.up)]]
            states[id(node)] = (rng.random(model['sites'])[:, None] > cumulative).sum(axis=1)
        if ete.is_leaf(node):
            tokens = model['codons'][states[id(node)]].copy()
            tokens[model['missing'][node.name]] = '---'
            sequences[node.name] = ''.join(tokens)
    return sequences


def replicate_seed(seed, index):
    return 1 + int(np.random.SeedSequence([int(seed), int(index), 5105]).generate_state(1)[0] % (2**31 - 2))


def run_replicate(options, model, directory, seed, transitions):
    """Run in a fresh directory: no observed ASR, filters or learned groups reused."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sequences = simulate_alignment(model, np.random.default_rng(seed), transitions)
    (directory / 'input.fa').write_text(''.join('>{}\n{}\n'.format(k, v) for k, v in sequences.items()))
    shutil.copyfile(options['rooted_tree_file'], directory / 'input.nwk')
    shutil.copyfile(options['foreground'], directory / 'foreground.tsv')
    local = dict(options)
    local.update(alignment_file=str(directory / 'input.fa'), rooted_tree_file=str(directory / 'input.nwk'),
                 foreground=str(directory / 'foreground.tsv'), outdir=str(directory), output_prefix='csubst',
                 iqtree_outdir=str(directory / 'iqtree'), log_file='',
                 scan_pvalue_calibration='none', scan_n_permutations=0, scan_site_plot=False,
                 scan_site_plot_filter='all', random_seed=int(seed))
    for suffix in ('state', 'treefile', 'rate', 'iqtree', 'log'):
        if 'iqtree_' + suffix in local:
            local['iqtree_' + suffix] = 'infer'
    command = [sys.executable, '-m', 'csubst', 'scan']
    for key, value in local.items():
        if value is None or value == '':
            continue
        if not re.fullmatch(r'[a-z][a-z0-9_]*', key) or not isinstance(value, (str, int, float, bool)):
            raise ValueError('Invalid original CLI option: {}'.format(key))
        command.extend(['--' + key, ('yes' if value else 'no') if isinstance(value, bool) else str(value)])
    (directory / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
    with (directory / 'process.log').open('w') as log:
        subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    frame = pd.read_csv(directory / 'csubst_scan.tsv', sep='\t')
    return scan_statistics.maximum_score(frame)


def calibrate(g, observed, model):
    requested = int(g['scan_n_permutations'])
    out = observed.copy()
    # A repeated API call must never retain P values from an earlier successful
    # calibration when the new run fails or is empty.
    out['p_rate_enrichment_bootstrap_maxT'] = np.nan
    if observed.empty:
        g['scan_bootstrap_summary'] = {'status': 'no_observed_candidates', 'requested': requested,
                                     'success_count': 0, 'failure_count': 0}
        return out
    directory = Path(tempfile.mkdtemp(prefix=str(g.get('output_prefix', 'csubst')) + '_scan_bootstrap_',
                                     dir=g['outdir'])).resolve()
    np.savez_compressed(directory / 'model.npz', q=model['q'], pi=model['pi'], codons=model['codons'],
                        topology=np.array(model['topology']),
                        tip_names=np.array(list(model['missing'])), missing=np.array(list(model['missing'].values())))
    summary = {'schema_version': 1, 'csubst_version': __version__, 'status': 'running',
               'null': 'fitted_uniform_codon_ctmc', 'requested': requested,
               'seed': int(g.get('scan_permutation_seed', 1)), 'success_count': 0, 'failure_count': 0,
               'fixed': ['topology', 'root_position', 'foreground', 'alignment_length', 'missing_codon_mask'],
               'refitted': ['codon_model_parameters', 'branch_lengths', 'ASR', 'automatic_recoding_if_requested'],
               'repeated': ['site_filtering', 'exposure', 'candidate_discovery', 'support_filter', 'maximum_score'],
               'observation': g.get('scan_observation', 'marginal'),
               'score_method': scan_statistics.SCORE_METHOD, 'maximum_scope': scan_statistics.MAXIMUM_SCOPE,
               'fitted_model_validation': model.get('provenance'),
               'inference': 'model_conditional_parametric_bootstrap_not_exact', 'replicates': []}
    manifest = directory / 'manifest.json'
    manifest.write_text(json.dumps(summary, indent=2) + '\n')
    transitions = transition_matrices(model)
    scores = []
    for index in range(requested):
        seed = replicate_seed(summary['seed'], index)
        print('Scan parametric bootstrap: replicate {}/{} (seed={}).'.format(index + 1, requested, seed), flush=True)
        record = {'index': index, 'seed': seed}
        try:
            value = run_replicate(g['scan_cli_options'], model, directory / 'rep{:06d}'.format(index), seed, transitions)
            if np.isnan(value) or value == np.inf:
                raise ValueError('Undefined replicate maximum score.')
            scores.append(value)
            record.update(status='success', empty=bool(value == -np.inf),
                          maximum_score=None if value == -np.inf else value)
            summary['success_count'] += 1
        except Exception as exc:
            record.update(status='failed', reason='{}: {}'.format(type(exc).__name__, exc))
            summary['failure_count'] += 1
        summary['replicates'].append(record)
        manifest.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        if summary['failure_count']:
            # Stop and expose the failure; neither retry-until-success nor remove it from the denominator.
            break
    if summary['failure_count']:
        summary['status'] = 'failed_null_replicates'
        print('Scan bootstrap failed; calibrated P values are undefined. See {}.'.format(manifest), flush=True)
    else:
        summary['status'] = 'model_conditional_parametric_bootstrap'
        out['p_rate_enrichment_bootstrap_maxT'] = [
            scan_statistics.empirical_pvalue(value, scores, requested)
            for value in out['score_rate_enrichment'].to_numpy(dtype=float)
        ]
        summary['minimum_p'] = 1 / (requested + 1)
    out['scan_inference_status'] = summary['status']
    out['scan_bootstrap_success_count'] = summary['success_count']
    out['scan_bootstrap_failure_count'] = summary['failure_count']
    out['scan_bootstrap_directory'] = str(directory)
    manifest.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    g['scan_bootstrap_summary'] = dict(summary, directory=str(directory))
    return out


def write_inference_report(g, frame):
    calibration = g.get('scan_pvalue_calibration', 'full_scan')
    report = {'schema_version': 1, 'csubst_version': __version__,
              'score_method': scan_statistics.SCORE_METHOD,
              'event_measure': 'posterior_mean_jump_count' if g.get('scan_observation') == 'bridge' else 'posterior_mass_in_both_event_modes',
              'observation': g.get('scan_observation', 'marginal'),
              'ctmc_model_precision': g.get('scan_ctmc_model_precision'),
              'asymptotic_and_bh_interpretation': 'exploratory_diagnostics_not_selection_adjusted_FDR',
              'bh_scope': scan_statistics.BH_SCOPE, 'calibration': calibration,
              'requested_replicates': int(g.get('scan_n_permutations', 1000)),
              'seed': int(g.get('scan_permutation_seed', 1)),
              'candidate_count': len(frame), 'bh_families': [],
              'status': ['no_observed_candidates'] if frame.empty else frame['scan_inference_status'].unique().tolist(),
              'maximum_scope': scan_statistics.MAXIMUM_SCOPE if calibration in ('full_scan', 'parametric_bootstrap', 'parametric') else 'none',
              'no_test_reason': g.get('scan_no_test_reason'),
              'clade_resampling_note': 'Requires exchangeability; not a universal FWER guarantee. '
                                       'Candidate-fixed inference does not adjust discovery selection.'}
    if not frame.empty:
        report['bh_families'] = frame[['scan_bh_family_id', 'scan_bh_family_size',
                                      'scan_bh_valid_asymptotic_count', 'scan_bh_valid_empirical_count']].drop_duplicates().to_dict('records')
        report['clade_replicates'] = {
            'success_count': int(frame['scan_permutation_success_count'].iloc[0]),
            'failure_count': int(frame['scan_permutation_failure_count'].iloc[0]),
            'failure_reasons': str(frame['scan_permutation_failure_reasons'].iloc[0]),
        }
    if 'scan_analytic_summary' in g:
        report['analytical_endpoint'] = g['scan_analytic_summary']
    if 'scan_bootstrap_summary' in g:
        report['bootstrap'] = g['scan_bootstrap_summary']
    Path(runtime.output_path(g, 'scan_inference.json')).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Scan asymptotic P and trait-by-match BH values are exploratory diagnostics; '
          'they do not correct candidate discovery selection.', flush=True)

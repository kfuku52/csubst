import numpy as np
import pandas as pd

import os
import platform
import re
import sys
try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata

from csubst.__init__ import __version__
from csubst import output_stat
from csubst import pseudocount
from csubst import recoding
from csubst import runtime
from csubst import table

DEPENDENCY_DISTRIBUTIONS = (
    'ete4',
    'numpy',
    'scipy',
    'pandas',
    'cython',
    'matplotlib',
)

MAX_QUANTILE_NITER = 10000


def _default_prostt5_cache_file():
    return 'csubst_prostt5_cache.tsv'


def _default_sa_state_cache_file():
    return 'csubst_3di_state_cache.npz'


def _infer_iqtree_output_prefix_from_alignment(alignment_file):
    return runtime.infer_iqtree_output_prefix(alignment_file=alignment_file)


def _parse_bool_like(value, param_name):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ['1', 'true', 'yes', 'y', 'on']:
            return True
        if normalized in ['0', 'false', 'no', 'n', 'off']:
            return False
        txt = '{} should be boolean-like (yes/no, true/false, 1/0).'
        raise ValueError(txt.format(param_name))
    return bool(value)


def _require_finite_float(value, param_name):
    if not np.isfinite(value):
        raise ValueError('{} should be a finite number.'.format(param_name))
    return value


def _parse_auto_or_float(value, param_name, min_value=None, strict_min=False):
    if isinstance(value, str):
        token = value.strip().lower()
        if token == 'auto':
            return True, np.nan
        if token == '':
            raise ValueError('{} should be "auto" or a number.'.format(param_name))
        try:
            numeric = float(token)
        except ValueError as exc:
            raise ValueError('{} should be "auto" or a number.'.format(param_name)) from exc
    else:
        numeric = float(value)
    numeric = _require_finite_float(numeric, param_name=param_name)
    if min_value is not None:
        min_value = float(min_value)
        if strict_min:
            if numeric <= min_value:
                raise ValueError('{} should be > {}.'.format(param_name, min_value))
        else:
            if numeric < min_value:
                raise ValueError('{} should be >= {}.'.format(param_name, min_value))
    return False, float(numeric)


def _parse_float_grid(value, param_name, min_value=None, strict_min=False):
    if isinstance(value, (list, tuple, np.ndarray)):
        raw_tokens = [str(v).strip() for v in value]
    else:
        raw_txt = '' if value is None else str(value).strip()
        raw_tokens = [token.strip() for token in raw_txt.split(',') if token.strip() != '']
    if len(raw_tokens) == 0:
        raise ValueError('{} should contain one or more numeric values.'.format(param_name))
    out = list()
    for token in raw_tokens:
        try:
            numeric = float(token)
        except ValueError as exc:
            raise ValueError('{} should contain numeric values.'.format(param_name)) from exc
        numeric = _require_finite_float(numeric, param_name=param_name)
        if min_value is not None:
            min_value = float(min_value)
            if strict_min:
                if numeric <= min_value:
                    raise ValueError('{} should contain values > {}.'.format(param_name, min_value))
            else:
                if numeric < min_value:
                    raise ValueError('{} should contain values >= {}.'.format(param_name, min_value))
        out.append(float(numeric))
    out = sorted(list(set(out)))
    return out


def _parse_omega_pvalue_niter_schedule(value, max_niter=MAX_QUANTILE_NITER):
    if isinstance(value, (list, tuple, np.ndarray)):
        raw_tokens = [str(v).strip() for v in value]
    else:
        raw_txt = '' if value is None else str(value).strip()
        if raw_txt.lower() in ['', '0', 'auto']:
            return None
        raw_tokens = [token.strip() for token in raw_txt.split(',') if token.strip() != '']
    if len(raw_tokens) == 0:
        raise ValueError('--omega_pvalue_niter_schedule should not be empty.')
    schedule = list()
    for token in raw_tokens:
        if not re.fullmatch(r'[0-9]+', token):
            raise ValueError('--omega_pvalue_niter_schedule should contain integers.')
        niter = int(token)
        if niter <= 0:
            raise ValueError('--omega_pvalue_niter_schedule should contain positive integers.')
        schedule.append(niter)
    for prev, curr in zip(schedule, schedule[1:]):
        if curr <= prev:
            raise ValueError('--omega_pvalue_niter_schedule should be strictly increasing.')
    if max(schedule) > int(max_niter):
        txt = '--omega_pvalue_niter_schedule upper bound should be <= {}.'
        raise ValueError(txt.format(int(max_niter)))
    return schedule


def _normalize_expectation_method(value, param_name='--expectation_method'):
    normalized = str(value).strip().lower()
    if normalized in ['codon_model', 'submodel']:
        return 'codon_model'
    if normalized in ['urn', 'modelfree']:
        return 'urn'
    txt = '{} should be one of codon_model, urn.'
    raise ValueError(txt.format(param_name))


def _normalize_urn_model(value, param_name='--urn_model'):
    normalized = str(value).strip().lower()
    if normalized in ['wallenius', 'fisher']:
        return normalized
    if normalized in ['factorized_approx', 'factorized', 'legacy_factorized', 'approx']:
        return 'factorized_approx'
    txt = '{} should be one of wallenius, fisher, factorized_approx.'
    raise ValueError(txt.format(param_name))


def _get_dependency_version(distribution_name):
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return 'not installed'


def _format_dependency_versions():
    dep_versions = list()
    missing_packages = list()
    for distribution_name in DEPENDENCY_DISTRIBUTIONS:
        version = _get_dependency_version(distribution_name)
        dep_versions.append('{}={}'.format(distribution_name, version))
        if version == 'not installed':
            missing_packages.append(distribution_name)
    return ', '.join(dep_versions), missing_packages


def get_global_parameters(args):
    print('OS: {}'.format(platform.platform()), flush=True)
    print('Python version: {}'.format(sys.version.replace('\n', ' ')), flush=True)
    print('CSUBST version: {}'.format(__version__), flush=True)
    dep_versions, missing_packages = _format_dependency_versions()
    print('CSUBST dependency versions: {}'.format(dep_versions), flush=True)
    if len(missing_packages):
        print('CSUBST missing dependency packages: {}'.format(', '.join(missing_packages)), flush=True)
    else:
        print('CSUBST missing dependency packages: none', flush=True)
    print('CSUBST command: {}'.format(' '.join(sys.argv)), flush=True)
    print('CSUBST working directory: {}'.format(os.getcwd()), flush=True)
    print('CSUBST bug report: https://github.com/kfuku52/csubst/issues', flush=True)
    g = dict()
    for attr in [a for a in dir(args) if not a.startswith('_')]:
        g[attr] = getattr(args, attr)
    if 'full_cds_alignment_file' in g.keys():
        if g['full_cds_alignment_file'] is None:
            g['full_cds_alignment_file'] = ''
        else:
            g['full_cds_alignment_file'] = str(g['full_cds_alignment_file']).strip()
    else:
        g['full_cds_alignment_file'] = ''
    if 'alignment_file' in g.keys():
        if g['alignment_file'] is None:
            g['alignment_file'] = ''
        else:
            g['alignment_file'] = str(g['alignment_file']).strip()
    raw_recode_token = str(g.get('nonsyn_recode', 'no')).strip()
    try:
        raw_recode_normalized = recoding.normalize_nonsyn_recode(raw_recode_token)
    except ValueError:
        raw_recode_normalized = None
    if raw_recode_normalized == '3di20':
        full_path = str(g.get('full_cds_alignment_file', '')).strip()
        if full_path != '':
            if g.get('alignment_file', '') not in ['', full_path]:
                txt = '--alignment_file is disabled when --nonsyn_recode is 3di20. Use --full_cds_alignment_file.'
                raise ValueError(txt)
            g['alignment_file'] = full_path
    if 'calc_omega_pvalue' in g.keys():
        g['calc_omega_pvalue'] = _parse_bool_like(g['calc_omega_pvalue'], '--calc_omega_pvalue')
    else:
        g['calc_omega_pvalue'] = False
    expectation_method = None
    raw_expectation_method = g.get('expectation_method', None)
    if (raw_expectation_method is not None) and (str(raw_expectation_method).strip() != ''):
        expectation_method = _normalize_expectation_method(raw_expectation_method, param_name='--expectation_method')
    raw_urn_model = g.get('urn_model', None)
    urn_model = None
    if (raw_urn_model is not None) and (str(raw_urn_model).strip() != ''):
        urn_model = _normalize_urn_model(raw_urn_model, param_name='--urn_model')
    raw_legacy_omega_method = g.get('omegaC_method', None)
    if (raw_legacy_omega_method is not None) and (str(raw_legacy_omega_method).strip() != ''):
        legacy_expectation_method = _normalize_expectation_method(raw_legacy_omega_method, param_name='--omegaC_method')
        if (expectation_method is not None) and (expectation_method != legacy_expectation_method):
            raise ValueError('--omegaC_method conflicts with --expectation_method.')
        expectation_method = legacy_expectation_method
        sys.stderr.write('Deprecated option --omegaC_method detected. Use --expectation_method instead.\n')
    if expectation_method is None:
        expectation_method = 'codon_model'
    if urn_model is None:
        urn_model = 'wallenius'
    g['expectation_method'] = expectation_method
    g['urn_model'] = urn_model
    g['omegaC_method'] = 'modelfree' if (expectation_method == 'urn') else 'submodel'
    if g['calc_omega_pvalue'] and (g['expectation_method'] != 'urn'):
        txt = '--calc_omega_pvalue "yes" should be used with --expectation_method "urn" '
        txt += '(legacy alias: --omegaC_method "modelfree").'
        raise ValueError(txt)
    if 'omega_pvalue_null_model' in g.keys():
        g['omega_pvalue_null_model'] = str(g['omega_pvalue_null_model']).strip().lower()
    else:
        g['omega_pvalue_null_model'] = 'hypergeom'
    if g['omega_pvalue_null_model'] not in ['hypergeom', 'poisson', 'poisson_full', 'nbinom']:
        raise ValueError('--omega_pvalue_null_model should be one of hypergeom, poisson, poisson_full, nbinom.')
    if 'omega_pvalue_nbinom_alpha' in g.keys():
        token = g['omega_pvalue_nbinom_alpha']
    else:
        token = 'auto'
    if isinstance(token, str) and (token.strip().lower() == 'auto'):
        g['omega_pvalue_nbinom_alpha'] = 'auto'
    else:
        g['omega_pvalue_nbinom_alpha'] = _require_finite_float(
            value=float(token),
            param_name='--omega_pvalue_nbinom_alpha',
        )
        if g['omega_pvalue_nbinom_alpha'] < 0:
            raise ValueError('--omega_pvalue_nbinom_alpha should be >= 0.')
    if 'omega_pvalue_safe_min_sub_pp' in g.keys():
        raise ValueError(
            '--omega_pvalue_safe_min_sub_pp was removed. '
            'Use --min_sub_pp explicitly when running --calc_omega_pvalue.'
        )
    if 'ml_anc' in g.keys():
        g['ml_anc'] = _parse_bool_like(g['ml_anc'], '--ml_anc')
    else:
        g['ml_anc'] = False
    if 'min_sub_pp' in g.keys():
        g['min_sub_pp'] = _require_finite_float(
            value=float(g['min_sub_pp']),
            param_name='--min_sub_pp',
        )
    else:
        g['min_sub_pp'] = 0.0
    if g['min_sub_pp'] < 0:
        raise ValueError('--min_sub_pp should be >= 0.')
    if g['min_sub_pp'] > 1:
        raise ValueError('--min_sub_pp should be <= 1.')
    if 'omega_pvalue_niter' in g.keys():
        raise ValueError(
            '--omega_pvalue_niter was removed. '
            'Use --omega_pvalue_niter_schedule.'
        )
    if 'omega_pvalue_niter_schedule' in g.keys():
        g['omega_pvalue_niter_schedule'] = _parse_omega_pvalue_niter_schedule(g['omega_pvalue_niter_schedule'])
    else:
        g['omega_pvalue_niter_schedule'] = None
    if 'omega_pvalue_refine_threshold' in g.keys():
        raise ValueError(
            '--omega_pvalue_refine_threshold was removed. '
            'Use --omega_pvalue_refine_upper_edge_bins.'
        )
    if 'omega_pvalue_refine_ci_alpha' in g.keys():
        raise ValueError(
            '--omega_pvalue_refine_ci_alpha was removed. '
            'Use --omega_pvalue_refine_upper_edge_bins.'
        )
    if 'omega_pvalue_refine_upper_edge_bins' in g.keys():
        g['omega_pvalue_refine_upper_edge_bins'] = int(g['omega_pvalue_refine_upper_edge_bins'])
    else:
        g['omega_pvalue_refine_upper_edge_bins'] = 2
    if g['omega_pvalue_refine_upper_edge_bins'] < 0:
        raise ValueError('--omega_pvalue_refine_upper_edge_bins should be >= 0.')
    if 'omega_pvalue_rounding' in g.keys():
        g['omega_pvalue_rounding'] = str(g['omega_pvalue_rounding']).strip().lower()
    else:
        g['omega_pvalue_rounding'] = 'stochastic'
    if g['omega_pvalue_rounding'] not in ['round', 'stochastic', 'floor', 'ceil']:
        raise ValueError('--omega_pvalue_rounding should be one of round, stochastic, floor, ceil.')
    if 'asrv' in g.keys():
        g['asrv'] = str(g['asrv']).strip().lower()
    else:
        g['asrv'] = 'each'
    if g['asrv'] not in ['no', 'pool', 'sn', 'each', 'file', 'file_each']:
        raise ValueError('--asrv should be one of no, pool, sn, each, file, file_each.')
    if 'asrv_dirichlet_alpha' in g.keys():
        g['asrv_dirichlet_alpha'] = _require_finite_float(
            value=float(g['asrv_dirichlet_alpha']),
            param_name='--asrv_dirichlet_alpha',
        )
    else:
        g['asrv_dirichlet_alpha'] = 1.0
    if g['asrv_dirichlet_alpha'] < 0:
        raise ValueError('--asrv_dirichlet_alpha should be >= 0.')
    if 'epistasis_apply_to' in g.keys():
        g['epistasis_apply_to'] = str(g['epistasis_apply_to']).strip().upper()
    else:
        g['epistasis_apply_to'] = 'N'
    if g['epistasis_apply_to'] not in ['N', 'S', 'NS']:
        raise ValueError('--epistasis_apply_to should be one of N, S, NS.')
    if ('epistasis_beta' not in g.keys()) or (g['epistasis_beta'] is None):
        g['epistasis_beta'] = 'off'
    epistasis_beta_token = None
    if isinstance(g['epistasis_beta'], str):
        epistasis_beta_token = g['epistasis_beta'].strip().lower()
    if epistasis_beta_token in ['off', 'none', 'false', 'no']:
        epistasis_beta_auto = False
        epistasis_beta_value = 0.0
    else:
        epistasis_beta_auto, epistasis_beta_value = _parse_auto_or_float(
            value=g['epistasis_beta'],
            param_name='--epistasis_beta',
            min_value=0.0,
            strict_min=False,
        )
    g['epistasis_beta_auto'] = bool(epistasis_beta_auto)
    g['epistasis_beta_value'] = float(epistasis_beta_value) if (not epistasis_beta_auto) else np.nan
    if ('epistasis_site_metric' not in g.keys()) or (g['epistasis_site_metric'] is None):
        g['epistasis_site_metric'] = 'off'
    else:
        g['epistasis_site_metric'] = str(g['epistasis_site_metric']).strip().lower()
    if g['epistasis_site_metric'] not in ['off', 'auto', 'degree', 'proximity', 'hybrid']:
        raise ValueError('--epistasis_site_metric should be one of off, auto, degree, proximity, hybrid.')
    if (g['epistasis_site_metric'] == 'off') and (g['epistasis_beta_auto'] or (g['epistasis_beta_value'] > 0)):
        g['epistasis_site_metric'] = 'auto'
    if 'epistasis_clip' not in g.keys():
        g['epistasis_clip'] = '3.0'
    epistasis_clip_auto, epistasis_clip_value = _parse_auto_or_float(
        value=g['epistasis_clip'],
        param_name='--epistasis_clip',
        min_value=0.0,
        strict_min=True,
    )
    g['epistasis_clip_auto'] = bool(epistasis_clip_auto)
    g['epistasis_clip_value'] = float(epistasis_clip_value) if (not epistasis_clip_auto) else np.nan
    if 'epistasis_beta_partition' not in g.keys():
        g['epistasis_beta_partition'] = 'global'
    g['epistasis_beta_partition'] = str(g['epistasis_beta_partition']).strip().lower()
    if g['epistasis_beta_partition'] not in ['global', 'branch_depth']:
        raise ValueError('--epistasis_beta_partition should be one of global, branch_depth.')
    if 'epistasis_branch_depth_bins' not in g.keys():
        g['epistasis_branch_depth_bins'] = 3
    g['epistasis_branch_depth_bins'] = int(g['epistasis_branch_depth_bins'])
    if g['epistasis_branch_depth_bins'] < 1:
        raise ValueError('--epistasis_branch_depth_bins should be >= 1.')
    if 'epistasis_feature_mode' not in g.keys():
        g['epistasis_feature_mode'] = 'single'
    g['epistasis_feature_mode'] = str(g['epistasis_feature_mode']).strip().lower()
    if g['epistasis_feature_mode'] not in ['single', 'paired']:
        raise ValueError('--epistasis_feature_mode should be one of single, paired.')
    if 'epistasis_joint_auto' not in g.keys():
        g['epistasis_joint_auto'] = False
    g['epistasis_joint_auto'] = _parse_bool_like(g['epistasis_joint_auto'], '--epistasis_joint_auto')
    if 'epistasis_joint_alpha_grid' not in g.keys():
        g['epistasis_joint_alpha_grid'] = '0,0.5,1,2'
    g['epistasis_joint_alpha_grid'] = _parse_float_grid(
        value=g['epistasis_joint_alpha_grid'],
        param_name='--epistasis_joint_alpha_grid',
        min_value=0.0,
        strict_min=False,
    )
    if 'epistasis_joint_clip_grid' not in g.keys():
        g['epistasis_joint_clip_grid'] = '1.5,2,2.5,3,4,5'
    g['epistasis_joint_clip_grid'] = _parse_float_grid(
        value=g['epistasis_joint_clip_grid'],
        param_name='--epistasis_joint_clip_grid',
        min_value=0.0,
        strict_min=True,
    )
    if 'epistasis_degree_file' not in g.keys():
        g['epistasis_degree_file'] = ''
    if g['epistasis_degree_file'] is None:
        g['epistasis_degree_file'] = ''
    g['epistasis_degree_file'] = str(g['epistasis_degree_file']).strip()
    if 'epistasis_pdb' not in g.keys():
        g['epistasis_pdb'] = ''
    if g['epistasis_pdb'] is None:
        g['epistasis_pdb'] = ''
    g['epistasis_pdb'] = str(g['epistasis_pdb']).strip()
    if 'epistasis_database' not in g.keys():
        g['epistasis_database'] = 'pdb,alphafill,alphafold'
    g['epistasis_database'] = str(g['epistasis_database']).strip().lower()
    if g['epistasis_database'] == '':
        g['epistasis_database'] = 'pdb,alphafill,alphafold'
    epistasis_database_names = [db.strip().lower() for db in g['epistasis_database'].split(',') if db.strip() != '']
    if len(epistasis_database_names) == 0:
        raise ValueError('--epistasis_database should include one or more of pdb,alphafill,alphafold.')
    epistasis_allowed_database_names = {'pdb', 'alphafill', 'alphafold'}
    epistasis_unknown_database_names = sorted(set(epistasis_database_names).difference(epistasis_allowed_database_names))
    if len(epistasis_unknown_database_names):
        txt = '--epistasis_database includes unknown values: {}. Supported: {}.'
        raise ValueError(txt.format(','.join(epistasis_unknown_database_names), ','.join(sorted(epistasis_allowed_database_names))))
    g['epistasis_database'] = ','.join(epistasis_database_names)
    if 'epistasis_database_timeout' in g.keys():
        g['epistasis_database_timeout'] = _require_finite_float(
            value=float(g['epistasis_database_timeout']),
            param_name='--epistasis_database_timeout',
        )
    else:
        g['epistasis_database_timeout'] = 30.0
    if g['epistasis_database_timeout'] <= 0:
        raise ValueError('--epistasis_database_timeout should be > 0.')
    if 'epistasis_database_evalue_cutoff' in g.keys():
        g['epistasis_database_evalue_cutoff'] = _require_finite_float(
            value=float(g['epistasis_database_evalue_cutoff']),
            param_name='--epistasis_database_evalue_cutoff',
        )
    else:
        g['epistasis_database_evalue_cutoff'] = 1.0
    if g['epistasis_database_evalue_cutoff'] <= 0:
        raise ValueError('--epistasis_database_evalue_cutoff should be > 0.')
    if 'epistasis_database_minimum_identity' in g.keys():
        g['epistasis_database_minimum_identity'] = _require_finite_float(
            value=float(g['epistasis_database_minimum_identity']),
            param_name='--epistasis_database_minimum_identity',
        )
    else:
        g['epistasis_database_minimum_identity'] = 0.25
    if (g['epistasis_database_minimum_identity'] < 0) or (g['epistasis_database_minimum_identity'] > 1):
        raise ValueError('--epistasis_database_minimum_identity should satisfy 0 <= value <= 1.')
    if 'epistasis_user_alignment' not in g.keys():
        g['epistasis_user_alignment'] = ''
    if g['epistasis_user_alignment'] is None:
        g['epistasis_user_alignment'] = ''
    g['epistasis_user_alignment'] = str(g['epistasis_user_alignment']).strip()
    if 'epistasis_contact_distance' in g.keys():
        g['epistasis_contact_distance'] = _require_finite_float(
            value=float(g['epistasis_contact_distance']),
            param_name='--epistasis_contact_distance',
        )
    else:
        g['epistasis_contact_distance'] = 8.0
    if g['epistasis_contact_distance'] <= 0:
        raise ValueError('--epistasis_contact_distance should be > 0.')
    if 'epistasis_mafft_exe' not in g.keys():
        g['epistasis_mafft_exe'] = 'mafft'
    g['epistasis_mafft_exe'] = str(g['epistasis_mafft_exe']).strip()
    if g['epistasis_mafft_exe'] == '':
        raise ValueError('--epistasis_mafft_exe should be non-empty.')
    if 'epistasis_mafft_op' in g.keys():
        g['epistasis_mafft_op'] = _require_finite_float(
            value=float(g['epistasis_mafft_op']),
            param_name='--epistasis_mafft_op',
        )
    else:
        g['epistasis_mafft_op'] = -1.0
    if (g['epistasis_mafft_op'] != -1) and (g['epistasis_mafft_op'] < 0):
        raise ValueError('--epistasis_mafft_op should be -1 or >= 0.')
    if 'epistasis_mafft_ep' in g.keys():
        g['epistasis_mafft_ep'] = _require_finite_float(
            value=float(g['epistasis_mafft_ep']),
            param_name='--epistasis_mafft_ep',
        )
    else:
        g['epistasis_mafft_ep'] = -1.0
    if (g['epistasis_mafft_ep'] != -1) and (g['epistasis_mafft_ep'] < 0):
        raise ValueError('--epistasis_mafft_ep should be -1 or >= 0.')
    if 'epistasis_pymol_max_num_chain' in g.keys():
        g['epistasis_pymol_max_num_chain'] = int(g['epistasis_pymol_max_num_chain'])
    else:
        g['epistasis_pymol_max_num_chain'] = 20
    if g['epistasis_pymol_max_num_chain'] < 1:
        raise ValueError('--epistasis_pymol_max_num_chain should be >= 1.')
    if 'epistasis_degree_outfile' not in g.keys():
        g['epistasis_degree_outfile'] = 'csubst_epistasis_structure_degree.tsv'
    if g['epistasis_degree_outfile'] is None:
        g['epistasis_degree_outfile'] = 'csubst_epistasis_structure_degree.tsv'
    g['epistasis_degree_outfile'] = str(g['epistasis_degree_outfile']).strip()
    if g['epistasis_degree_outfile'] == '':
        raise ValueError('--epistasis_degree_outfile should be non-empty.')
    g['epistasis_requested'] = bool(g['epistasis_beta_auto'] or (g['epistasis_beta_value'] > 0))
    pseudocount_config = pseudocount.validate_args(g)
    g.update(pseudocount_config)
    if 'exhaustive_until' in g.keys():
        if (g['exhaustive_until']==1)&(g['foreground'] is None):
            raise ValueError('To enable --exhaustive_until 1, use --foreground')
    if 'fg_clade_permutation' in g.keys():
        if g['fg_clade_permutation']>0:
            if (g['foreground'] is None):
                raise ValueError('To enable --fg_clade_permutation, use --foreground')
    g = runtime.ensure_iqtree_layout(g, create_dir=False)
    alignment_file_txt = str(g.get('alignment_file', '')).strip()
    if alignment_file_txt != '':
        if 'iqtree_treefile' in g.keys():
            if (g['iqtree_treefile']=='infer'):
                iqtree_prefix = runtime.infer_iqtree_output_prefix(
                    alignment_file=g['alignment_file'],
                    iqtree_outdir=g['iqtree_outdir'],
                )
                g['iqtree_treefile'] = iqtree_prefix+'.treefile'
        if 'iqtree_state' in g.keys():
            if (g['iqtree_state']=='infer'):
                iqtree_prefix = runtime.infer_iqtree_output_prefix(
                    alignment_file=g['alignment_file'],
                    iqtree_outdir=g['iqtree_outdir'],
                )
                g['iqtree_state'] = iqtree_prefix+'.state'
        if 'iqtree_rate' in g.keys():
            if (g['iqtree_rate']=='infer'):
                iqtree_prefix = runtime.infer_iqtree_output_prefix(
                    alignment_file=g['alignment_file'],
                    iqtree_outdir=g['iqtree_outdir'],
                )
                g['iqtree_rate'] = iqtree_prefix+'.rate'
        if 'iqtree_iqtree' in g.keys():
            if (g['iqtree_iqtree']=='infer'):
                iqtree_prefix = runtime.infer_iqtree_output_prefix(
                    alignment_file=g['alignment_file'],
                    iqtree_outdir=g['iqtree_outdir'],
                )
                g['iqtree_iqtree'] = iqtree_prefix+'.iqtree'
    if 'float_type' in g.keys():
        g['float_type'] = int(g['float_type'])
        if (g['float_type']==16):
            g['float_type'] = np.float16
            g['float_tol'] = 10**-1
        elif (g['float_type']==32):
            g['float_type'] = np.float32
            g['float_tol'] = 10**-3
        elif (g['float_type']==64):
            g['float_type'] = np.float64
            g['float_tol'] = 10**-9
        else:
            raise ValueError('--float_type should be one of 16, 32, 64.')
    if 'sub_tensor_backend' in g.keys():
        g['sub_tensor_backend'] = str(g['sub_tensor_backend']).lower()
    else:
        g['sub_tensor_backend'] = 'auto'
    if g['sub_tensor_backend'] not in ['auto', 'dense', 'sparse']:
        raise ValueError('--sub_tensor_backend should be one of auto, dense, sparse.')
    if 'sub_tensor_sparse_density_cutoff' in g.keys():
        g['sub_tensor_sparse_density_cutoff'] = _require_finite_float(
            value=float(g['sub_tensor_sparse_density_cutoff']),
            param_name='--sub_tensor_sparse_density_cutoff',
        )
    else:
        g['sub_tensor_sparse_density_cutoff'] = 0.15
    if (g['sub_tensor_sparse_density_cutoff'] < 0) or (g['sub_tensor_sparse_density_cutoff'] > 1):
        raise ValueError('--sub_tensor_sparse_density_cutoff should be between 0 and 1.')
    if 'sub_tensor_auto_sparse_min_elements' in g.keys():
        g['sub_tensor_auto_sparse_min_elements'] = int(g['sub_tensor_auto_sparse_min_elements'])
    else:
        g['sub_tensor_auto_sparse_min_elements'] = 100000000
    if g['sub_tensor_auto_sparse_min_elements'] < 0:
        raise ValueError('--sub_tensor_auto_sparse_min_elements should be >= 0.')
    if 'parallel_backend' in g.keys():
        g['parallel_backend'] = str(g['parallel_backend']).lower()
    else:
        g['parallel_backend'] = 'auto'
    if g['parallel_backend'] not in ['auto', 'multiprocessing', 'threading']:
        raise ValueError('--parallel_backend should be one of auto, multiprocessing, threading.')
    if 'parallel_chunk_factor' in g.keys():
        g['parallel_chunk_factor'] = int(g['parallel_chunk_factor'])
    else:
        g['parallel_chunk_factor'] = 1
    if g['parallel_chunk_factor'] < 1:
        raise ValueError('--parallel_chunk_factor should be >= 1.')
    if 'parallel_chunk_factor_reducer' in g.keys():
        g['parallel_chunk_factor_reducer'] = int(g['parallel_chunk_factor_reducer'])
    else:
        g['parallel_chunk_factor_reducer'] = 4
    if g['parallel_chunk_factor_reducer'] < 1:
        raise ValueError('--parallel_chunk_factor_reducer should be >= 1.')
    parallel_min_items_defaults = {
        'parallel_min_items_sub_tensor': 256,
        'parallel_min_items_node_union': 20000,
        'parallel_min_items_nc_matrix': 100000,
        'parallel_min_items_cb': 20000,
        'parallel_min_rows_cbs': 200000,
        'parallel_min_items_branch_dist': 20000,
        'parallel_min_items_expected_state': 50000000,
    }
    for key, default_value in parallel_min_items_defaults.items():
        if key in g.keys():
            g[key] = int(g[key])
        else:
            g[key] = int(default_value)
        if g[key] < 0:
            raise ValueError('--{} should be >= 0.'.format(key))
    parallel_min_per_job_defaults = {
        'parallel_min_items_per_job_sub_tensor': 64,
        'parallel_min_items_per_job_node_union': 5000,
        'parallel_min_items_per_job_nc_matrix': 25000,
        'parallel_min_items_per_job_cb': 5000,
        'parallel_min_rows_per_job_cbs': 50000,
        'parallel_min_items_per_job_branch_dist': 5000,
        'parallel_min_items_per_job_expected_state': 10000000,
    }
    for key, default_value in parallel_min_per_job_defaults.items():
        if key in g.keys():
            g[key] = int(g[key])
        else:
            g[key] = int(default_value)
        if g[key] < 1:
            raise ValueError('--{} should be >= 1.'.format(key))
    if 'pdb' in g.keys():
        if g['pdb']=='besthit':
            g['run_pdb_sequence_search'] = True
        else:
            g['run_pdb_sequence_search'] = False
    if 'uniprot_feature_types' in g.keys():
        value = g['uniprot_feature_types']
        if value is None:
            g['uniprot_feature_types'] = None
        else:
            value = str(value).strip()
            if (value=='')|(value.lower() in ['all','*']):
                g['uniprot_feature_types'] = None
            else:
                g['uniprot_feature_types'] = [ v.strip() for v in value.split(',') if v.strip()!='' ]
    if 'uniprot_include_redundant' in g.keys():
        g['uniprot_include_redundant'] = _parse_bool_like(
            value=g['uniprot_include_redundant'],
            param_name='--uniprot_include_redundant',
        )
    if 'export2chimera' in g.keys():
        export2chimera = _parse_bool_like(
            value=g['export2chimera'],
            param_name='--export2chimera',
        )
        g['export2chimera'] = export2chimera
        if export2chimera and (g.get('untrimmed_cds', None) in [None, '']):
            raise ValueError('--export2chimera "yes" requires --untrimmed_cds.')
    drop_mode_token = str(g.get('drop_invariant_tip_sites', 'tip_invariant')).strip().lower()
    if drop_mode_token == 'no':
        g['drop_invariant_tip_sites'] = False
        g['drop_invariant_tip_sites_mode'] = 'tip_invariant'
    elif drop_mode_token in ['tip_invariant', 'zero_sub_mass']:
        g['drop_invariant_tip_sites'] = True
        g['drop_invariant_tip_sites_mode'] = drop_mode_token
    else:
        txt = '--drop_invariant_tip_sites should be one of no, tip_invariant, zero_sub_mass.'
        raise ValueError(txt)
    if 'num_simulated_site' in g.keys():
        g['num_simulated_site'] = int(g['num_simulated_site'])
        if (g['num_simulated_site'] != -1) and (g['num_simulated_site'] < 1):
            raise ValueError('--num_simulated_site should be -1 or >= 1.')
    if 'percent_convergent_site' in g.keys():
        g['percent_convergent_site'] = _require_finite_float(
            value=float(g['percent_convergent_site']),
            param_name='--percent_convergent_site',
        )
        if (g['percent_convergent_site'] < 0) or (g['percent_convergent_site'] > 100):
            raise ValueError('--percent_convergent_site should be between 0 and 100.')
    if 'simulate_seed' in g.keys():
        g['simulate_seed'] = int(g['simulate_seed'])
        if (g['simulate_seed'] < -1):
            raise ValueError('--simulate_seed should be -1 or >= 0.')
    if 'simulate_asrv' in g.keys():
        g['simulate_asrv'] = str(g['simulate_asrv']).strip().lower()
    else:
        g['simulate_asrv'] = 'no'
    if g['simulate_asrv'] not in ['no', 'file']:
        raise ValueError('--simulate_asrv should be one of no, file.')
    if 'export_true_asr' in g.keys():
        g['export_true_asr'] = _parse_bool_like(
            value=g['export_true_asr'],
            param_name='--export_true_asr',
        )
    else:
        g['export_true_asr'] = True
    if 'true_asr_prefix' in g.keys():
        if g['true_asr_prefix'] is None:
            g['true_asr_prefix'] = ''
        g['true_asr_prefix'] = str(g['true_asr_prefix']).strip()
    else:
        g['true_asr_prefix'] = ''
    if 'tree_scaling_factor' in g.keys():
        g['tree_scaling_factor'] = _require_finite_float(
            value=float(g['tree_scaling_factor']),
            param_name='--tree_scaling_factor',
        )
        if g['tree_scaling_factor'] < 0:
            raise ValueError('--tree_scaling_factor should be >= 0.')
    if 'foreground_scaling_factor' in g.keys():
        g['foreground_scaling_factor'] = _require_finite_float(
            value=float(g['foreground_scaling_factor']),
            param_name='--foreground_scaling_factor',
        )
        if g['foreground_scaling_factor'] < 0:
            raise ValueError('--foreground_scaling_factor should be >= 0.')
    if 'background_omega' in g.keys():
        raw_background_omega = g['background_omega']
        if raw_background_omega is None:
            g['background_omega'] = None
        else:
            omega_txt = str(raw_background_omega).strip().lower()
            if omega_txt in ['', 'none', 'auto', 'infer', 'iqtree']:
                g['background_omega'] = None
            else:
                g['background_omega'] = _require_finite_float(
                    value=float(raw_background_omega),
                    param_name='--background_omega',
                )
                if g['background_omega'] < 0:
                    raise ValueError('--background_omega should be >= 0.')
    if 'foreground_omega' in g.keys():
        g['foreground_omega'] = _require_finite_float(
            value=float(g['foreground_omega']),
            param_name='--foreground_omega',
        )
        if g['foreground_omega'] < 0:
            raise ValueError('--foreground_omega should be >= 0.')
    if 'simulate_eq_freq' in g.keys():
        g['simulate_eq_freq'] = str(g['simulate_eq_freq']).strip().lower()
    else:
        g['simulate_eq_freq'] = 'auto'
    if g['simulate_eq_freq'] not in ['auto', 'iqtree', 'alignment']:
        raise ValueError('--simulate_eq_freq should be one of auto, iqtree, alignment.')
    if 'convergent_amino_acids' in g.keys():
        conv_aa = str(g['convergent_amino_acids']).strip()
        if conv_aa == '':
            raise ValueError('--convergent_amino_acids should be non-empty.')
        if conv_aa.startswith('random'):
            suffix = conv_aa.replace('random', '', 1)
            if not re.fullmatch(r'\d+', suffix):
                raise ValueError('--convergent_amino_acids random mode should be randomN (N is an integer).')
            num_random_aa = int(suffix)
            if num_random_aa > 20:
                raise ValueError('--convergent_amino_acids randomN should satisfy 0 <= N <= 20.')
        else:
            from csubst import genetic_code
            codon_table = genetic_code.get_codon_table(ncbi_id=g.get('genetic_code', 1))
            valid_aas = set([item[0] for item in codon_table if item[0] != '*'])
            invalid_aas = sorted(set([aa for aa in conv_aa if aa not in valid_aas]))
            if len(invalid_aas):
                raise ValueError('--convergent_amino_acids contains unsupported amino acids: {}.'.format(','.join(invalid_aas)))
        g['convergent_amino_acids'] = conv_aa
    if 'percent_biased_sub' in g.keys():
        g['percent_biased_sub'] = _require_finite_float(
            value=float(g['percent_biased_sub']),
            param_name='--percent_biased_sub',
        )
        if (g['percent_biased_sub'] < 0) or (g['percent_biased_sub'] >= 100):
            raise ValueError('--percent_biased_sub should be between 0 and <100.')
    if 'tree_site_plot_max_sites' in g.keys():
        g['tree_site_plot_max_sites'] = int(g['tree_site_plot_max_sites'])
        if g['tree_site_plot_max_sites'] < 1:
            raise ValueError('--tree_site_plot_max_sites should be >= 1.')
    if 'species_regex' in g.keys():
        if g['species_regex'] is None:
            g['species_regex'] = ''
        else:
            g['species_regex'] = str(g['species_regex']).strip()
        if g['species_regex'] != '':
            try:
                re.compile(g['species_regex'])
            except re.error as exc:
                txt = '--species_regex is not a valid regular expression: {}'
                raise ValueError(txt.format(exc))
    if 'species_overlap_node_plot' in g.keys():
        g['species_overlap_node_plot'] = str(g['species_overlap_node_plot']).strip().lower()
    else:
        g['species_overlap_node_plot'] = 'auto'
    if g['species_overlap_node_plot'] not in ['yes', 'no', 'auto']:
        raise ValueError('--species_overlap_node_plot should be one of yes, no, auto.')
    if 'tree_site_plot_min_prob' in g.keys():
        g['tree_site_plot_min_prob'] = _require_finite_float(
            value=float(g['tree_site_plot_min_prob']),
            param_name='--tree_site_plot_min_prob',
        )
        if (g['tree_site_plot_min_prob'] < 0) or (g['tree_site_plot_min_prob'] > 1):
            raise ValueError('--tree_site_plot_min_prob should satisfy 0 <= value <= 1.')
    if 'min_single_prob' in g.keys():
        g['min_single_prob'] = _require_finite_float(
            value=float(g['min_single_prob']),
            param_name='--min_single_prob',
        )
        if (g['min_single_prob'] < 0) or (g['min_single_prob'] > 1):
            raise ValueError('--min_single_prob should satisfy 0 <= value <= 1.')
    if 'min_combinat_prob' in g.keys():
        g['min_combinat_prob'] = _require_finite_float(
            value=float(g['min_combinat_prob']),
            param_name='--min_combinat_prob',
        )
        if (g['min_combinat_prob'] < 0) or (g['min_combinat_prob'] > 1):
            raise ValueError('--min_combinat_prob should satisfy 0 <= value <= 1.')
    if 'database_timeout' in g.keys():
        g['database_timeout'] = _require_finite_float(
            value=float(g['database_timeout']),
            param_name='--database_timeout',
        )
        if g['database_timeout'] <= 0:
            raise ValueError('--database_timeout should be > 0.')
    if 'database_evalue_cutoff' in g.keys():
        g['database_evalue_cutoff'] = _require_finite_float(
            value=float(g['database_evalue_cutoff']),
            param_name='--database_evalue_cutoff',
        )
        if g['database_evalue_cutoff'] <= 0:
            raise ValueError('--database_evalue_cutoff should be > 0.')
    if 'database_minimum_identity' in g.keys():
        g['database_minimum_identity'] = _require_finite_float(
            value=float(g['database_minimum_identity']),
            param_name='--database_minimum_identity',
        )
        if (g['database_minimum_identity'] < 0) or (g['database_minimum_identity'] > 1):
            raise ValueError('--database_minimum_identity should satisfy 0 <= value <= 1.')
    if 'mafft_op' in g.keys():
        g['mafft_op'] = _require_finite_float(
            value=float(g['mafft_op']),
            param_name='--mafft_op',
        )
        if (g['mafft_op'] != -1) and (g['mafft_op'] < 0):
            raise ValueError('--mafft_op should be -1 or >= 0.')
    if 'mafft_ep' in g.keys():
        g['mafft_ep'] = _require_finite_float(
            value=float(g['mafft_ep']),
            param_name='--mafft_ep',
        )
        if (g['mafft_ep'] != -1) and (g['mafft_ep'] < 0):
            raise ValueError('--mafft_ep should be -1 or >= 0.')
    if 'pymol_gray' in g.keys():
        g['pymol_gray'] = int(g['pymol_gray'])
        if (g['pymol_gray'] < 0) or (g['pymol_gray'] > 100):
            raise ValueError('--pymol_gray should satisfy 0 <= value <= 100.')
    if 'pymol_transparency' in g.keys():
        g['pymol_transparency'] = _require_finite_float(
            value=float(g['pymol_transparency']),
            param_name='--pymol_transparency',
        )
        if (g['pymol_transparency'] < 0) or (g['pymol_transparency'] > 1):
            raise ValueError('--pymol_transparency should satisfy 0 <= value <= 1.')
    if 'pymol_max_num_chain' in g.keys():
        g['pymol_max_num_chain'] = int(g['pymol_max_num_chain'])
        if g['pymol_max_num_chain'] < 1:
            raise ValueError('--pymol_max_num_chain should be >= 1.')
    if 'float_digit' in g.keys():
        g['float_format'] = '%.'+str(g['float_digit'])+'f'
    if 'threads' in g.keys():
        g['threads'] = int(g['threads'])
        if g['threads'] < 1:
            raise ValueError('--threads should be >= 1.')
        set_num_thread_variables(num_thread=g['threads'])
    if 'nonsyn_recode' in g.keys():
        g['nonsyn_recode'] = recoding.normalize_nonsyn_recode(g['nonsyn_recode'])
    else:
        g['nonsyn_recode'] = 'no'
    if 'sa_asr_mode' in g.keys():
        g['sa_asr_mode'] = str(g['sa_asr_mode']).strip().lower()
    else:
        g['sa_asr_mode'] = 'direct'
    if g['sa_asr_mode'] not in ['translate', 'direct']:
        raise ValueError('--sa_asr_mode should be one of translate, direct.')
    if 'prostt5_model' not in g.keys():
        g['prostt5_model'] = 'Rostlab/ProstT5'
    g['prostt5_model'] = str(g['prostt5_model']).strip()
    if g['prostt5_model'] == '':
        raise ValueError('--prostt5_model should be non-empty.')
    if 'prostt5_local_dir' not in g.keys():
        g['prostt5_local_dir'] = ''
    if g['prostt5_local_dir'] is None:
        g['prostt5_local_dir'] = ''
    g['prostt5_local_dir'] = str(g['prostt5_local_dir']).strip()
    if 'prostt5_no_download' not in g.keys():
        g['prostt5_no_download'] = False
    g['prostt5_no_download'] = _parse_bool_like(
        value=g['prostt5_no_download'],
        param_name='--prostt5_no_download',
    )
    if 'download_prostt5' not in g.keys():
        g['download_prostt5'] = False
    g['download_prostt5'] = _parse_bool_like(
        value=g['download_prostt5'],
        param_name='--download_prostt5',
    )
    if 'write_instantaneous_rate_matrix' not in g.keys():
        g['write_instantaneous_rate_matrix'] = False
    g['write_instantaneous_rate_matrix'] = _parse_bool_like(
        value=g['write_instantaneous_rate_matrix'],
        param_name='--write_instantaneous_rate_matrix',
    )
    if 'prostt5_device' not in g.keys():
        g['prostt5_device'] = 'auto'
    g['prostt5_device'] = str(g['prostt5_device']).strip().lower()
    if g['prostt5_device'] not in ['auto', 'cpu', 'cuda', 'mps']:
        raise ValueError('--prostt5_device should be one of auto, cpu, cuda, mps.')
    if 'prostt5_cache' not in g.keys():
        g['prostt5_cache'] = True
    g['prostt5_cache'] = _parse_bool_like(
        value=g['prostt5_cache'],
        param_name='--prostt5_cache',
    )
    if 'prostt5_cache_file' not in g.keys():
        g['prostt5_cache_file'] = _default_prostt5_cache_file()
    if g['prostt5_cache_file'] is None:
        g['prostt5_cache_file'] = _default_prostt5_cache_file()
    g['prostt5_cache_file'] = str(g['prostt5_cache_file']).strip()
    if g['prostt5_cache_file'] == '':
        raise ValueError('--prostt5_cache_file should be non-empty.')
    if 'sa_iqtree_model' not in g.keys():
        g['sa_iqtree_model'] = 'GTR'
    g['sa_iqtree_model'] = str(g['sa_iqtree_model']).strip()
    if g['sa_iqtree_model'] == '':
        raise ValueError('--sa_iqtree_model should be non-empty.')
    if 'sa_state_cache' not in g.keys():
        g['sa_state_cache'] = 'auto'
    g['sa_state_cache'] = str(g['sa_state_cache']).strip().lower()
    if g['sa_state_cache'] not in ['auto', 'yes', 'no']:
        raise ValueError('--sa_state_cache should be one of auto, yes, no.')
    if 'sa_state_cache_file' not in g.keys():
        g['sa_state_cache_file'] = _default_sa_state_cache_file()
    if g['sa_state_cache_file'] is None:
        g['sa_state_cache_file'] = _default_sa_state_cache_file()
    g['sa_state_cache_file'] = str(g['sa_state_cache_file']).strip()
    if g['sa_state_cache_file'] == '':
        raise ValueError('--sa_state_cache_file should be non-empty.')
    if 'sa_smoke_max_branches' in g.keys():
        g['sa_smoke_max_branches'] = int(g['sa_smoke_max_branches'])
    else:
        g['sa_smoke_max_branches'] = 0
    if g['sa_smoke_max_branches'] < 0:
        raise ValueError('--sa_smoke_max_branches should be >= 0.')
    if 'plot_nonsyn_recode_pca_3di20' not in g.keys():
        g['plot_nonsyn_recode_pca_3di20'] = False
    g['plot_nonsyn_recode_pca_3di20'] = _parse_bool_like(
        value=g['plot_nonsyn_recode_pca_3di20'],
        param_name='--plot_nonsyn_recode_pca_3di20',
    )
    if g['nonsyn_recode'] == '3di20':
        full_path = str(g.get('full_cds_alignment_file', '')).strip()
        if full_path == '':
            raise ValueError('--nonsyn_recode 3di20 requires --full_cds_alignment_file.')
        if g.get('alignment_file', '') not in ['', full_path]:
            txt = '--alignment_file is disabled when --nonsyn_recode is 3di20. Use --full_cds_alignment_file.'
            raise ValueError(txt)
        g['alignment_file'] = full_path
    if 'output_stat' in g.keys():
        g['output_stats'] = output_stat.parse_output_stats(g['output_stat'])
    else:
        g['output_stats'] = list(output_stat.ALL_OUTPUT_STATS)
    g['output_stat'] = ','.join(g['output_stats'])
    g['output_base_stats'] = output_stat.get_required_base_stats(g['output_stats'])
    g['output_dif_stats'] = output_stat.get_required_dif_stats(g['output_stats'])
    if 'cutoff_stat' in g.keys():
        if str(g['cutoff_stat']) == output_stat.DEFAULT_CUTOFF_STAT:
            adjusted = output_stat.get_default_cutoff_stat_for_output_stats(g['output_stats'])
            if adjusted != g['cutoff_stat']:
                txt = 'Default --cutoff_stat was adjusted to "{}" to match --output_stat.'
                print(txt.format(adjusted), flush=True)
                g['cutoff_stat'] = adjusted
        output_stat.validate_cutoff_stat_compatibility(g['cutoff_stat'], g['output_stats'])
        # Fail fast on malformed cutoff tokens/regex/value ranges.
        table.parse_cutoff_stat(cutoff_stat_str=g['cutoff_stat'])
    g = runtime.ensure_output_layout(g, create_dir=False)
    return runtime.ensure_run_context(g)

def initialize_df_cb_stats(g):
    cols = ['arity','elapsed_sec','fg_enrichment_factor','mode','dSC_calibration',]
    g['df_cb_stats'] = pd.DataFrame(index=[0,], columns=cols)
    g['df_cb_stats']['arity'] = [g['current_arity'],]
    g['df_cb_stats']['cutoff_stat'] = g['cutoff_stat']
    return(g)

def set_num_thread_variables(num_thread=1):
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
    # TODO Not sure yet if this really accelerate the computation.
    os.environ["OMP_NUM_THREADS"] = str(num_thread)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_thread)
    os.environ["MKL_NUM_THREADS"] = str(num_thread)
    os.environ["VECLIB_NUM_THREADS"] = str(num_thread)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_thread)
    return None

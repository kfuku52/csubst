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
from csubst import table

DEPENDENCY_DISTRIBUTIONS = (
    'ete4',
    'numpy',
    'scipy',
    'pandas',
    'cython',
    'matplotlib',
)


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
    if 'calc_quantile' in g.keys():
        if g['calc_quantile']:
            if g['omegaC_method'] != 'modelfree':
                raise ValueError('--calc_quantile "yes" should be used with --omegaC_method "modelfree".')
    if 'exhaustive_until' in g.keys():
        if (g['exhaustive_until']==1)&(g['foreground'] is None):
            raise ValueError('To enable --exhaustive_until 1, use --foreground')
    if 'fg_clade_permutation' in g.keys():
        if g['fg_clade_permutation']>0:
            if (g['foreground'] is None):
                raise ValueError('To enable --fg_clade_permutation, use --foreground')
    if 'iqtree_treefile' in g.keys():
        if (g['iqtree_treefile']=='infer'):
            g['iqtree_treefile'] = g['alignment_file']+'.treefile'
    if 'iqtree_state' in g.keys():
        if (g['iqtree_state']=='infer'):
            g['iqtree_state'] = g['alignment_file']+'.state'
    if 'iqtree_rate' in g.keys():
        if (g['iqtree_rate']=='infer'):
            g['iqtree_rate'] = g['alignment_file']+'.rate'
    if 'iqtree_iqtree' in g.keys():
        if (g['iqtree_iqtree']=='infer'):
            g['iqtree_iqtree'] = g['alignment_file']+'.iqtree'
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
    if 'parallel_backend' in g.keys():
        g['parallel_backend'] = str(g['parallel_backend']).lower()
    else:
        g['parallel_backend'] = 'auto'
    if g['parallel_backend'] not in ['auto', 'multiprocessing', 'threading', 'loky']:
        raise ValueError('--parallel_backend should be one of auto, multiprocessing, threading, loky.')
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
        g['background_omega'] = _require_finite_float(
            value=float(g['background_omega']),
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
    return g

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

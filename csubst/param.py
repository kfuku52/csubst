import numpy
import pandas

import os
import platform
import re
import sys

from csubst.__init__ import __version__

def get_global_parameters(args):
    print('OS: {}'.format(platform.platform()), flush=True)
    print('Python version: {}'.format(sys.version.replace('\n', ' ')), flush=True)
    print('CSUBST version: {}'.format(__version__), flush=True)
    print('CSUBST command: {}'.format(' '.join(sys.argv)), flush=True)
    print('CSUBST working directory: {}'.format(os.getcwd()), flush=True)
    print('CSUBST bug report: https://github.com/kfuku52/csubst/issues', flush=True)
    g = dict()
    for attr in [a for a in dir(args) if not a.startswith('_')]:
        g[attr] = getattr(args, attr)
    if 'calc_quantile' in g.keys():
        if g['calc_quantile']:
            assert g['omegaC_method']=='modelfree', '--calc_quantile "yes" should be used with --omegaC_method "modelfree".'
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
        if (g['float_type']==16):
            g['float_type'] = numpy.float16
            g['float_tol'] = 10**-1
        elif (g['float_type']==32):
            g['float_type'] = numpy.float32
            g['float_tol'] = 10**-3
        elif (g['float_type']==64):
            g['float_type'] = numpy.float64
            g['float_tol'] = 10**-9
    if 'sub_tensor_backend' in g.keys():
        g['sub_tensor_backend'] = str(g['sub_tensor_backend']).lower()
    else:
        g['sub_tensor_backend'] = 'auto'
    if g['sub_tensor_backend'] not in ['auto', 'dense', 'sparse']:
        raise ValueError('--sub_tensor_backend should be one of auto, dense, sparse.')
    if 'sub_tensor_sparse_density_cutoff' in g.keys():
        g['sub_tensor_sparse_density_cutoff'] = float(g['sub_tensor_sparse_density_cutoff'])
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
        g['uniprot_include_redundant'] = bool(g['uniprot_include_redundant'])
    if 'num_simulated_site' in g.keys():
        g['num_simulated_site'] = int(g['num_simulated_site'])
        if (g['num_simulated_site'] != -1) and (g['num_simulated_site'] < 1):
            raise ValueError('--num_simulated_site should be -1 or >= 1.')
    if 'percent_convergent_site' in g.keys():
        g['percent_convergent_site'] = float(g['percent_convergent_site'])
        if (g['percent_convergent_site'] < 0) or (g['percent_convergent_site'] > 100):
            raise ValueError('--percent_convergent_site should be between 0 and 100.')
    if 'tree_scaling_factor' in g.keys():
        g['tree_scaling_factor'] = float(g['tree_scaling_factor'])
        if g['tree_scaling_factor'] < 0:
            raise ValueError('--tree_scaling_factor should be >= 0.')
    if 'foreground_scaling_factor' in g.keys():
        g['foreground_scaling_factor'] = float(g['foreground_scaling_factor'])
        if g['foreground_scaling_factor'] < 0:
            raise ValueError('--foreground_scaling_factor should be >= 0.')
    if 'background_omega' in g.keys():
        g['background_omega'] = float(g['background_omega'])
        if g['background_omega'] < 0:
            raise ValueError('--background_omega should be >= 0.')
    if 'foreground_omega' in g.keys():
        g['foreground_omega'] = float(g['foreground_omega'])
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
        g['percent_biased_sub'] = float(g['percent_biased_sub'])
        if (g['percent_biased_sub'] < 0) or (g['percent_biased_sub'] >= 100):
            raise ValueError('--percent_biased_sub should be between 0 and <100.')
    if 'tree_site_plot_max_sites' in g.keys():
        g['tree_site_plot_max_sites'] = int(g['tree_site_plot_max_sites'])
        if g['tree_site_plot_max_sites'] < 1:
            raise ValueError('--tree_site_plot_max_sites should be >= 1.')
    if 'tree_site_plot_min_prob' in g.keys():
        g['tree_site_plot_min_prob'] = float(g['tree_site_plot_min_prob'])
        if (g['tree_site_plot_min_prob'] >= 0) and (g['tree_site_plot_min_prob'] > 1):
            raise ValueError('--tree_site_plot_min_prob should be <= 1 when non-negative.')
    if 'float_digit' in g.keys():
        g['float_format'] = '%.'+str(g['float_digit'])+'f'
    if 'threads' in g.keys():
        g['threads'] = int(g['threads'])
        if g['threads'] < 1:
            raise ValueError('--threads should be >= 1.')
        set_num_thread_variables(num_thread=g['threads'])
    return g

def initialize_df_cb_stats(g):
    cols = ['arity','elapsed_sec','fg_enrichment_factor','mode','dSC_calibration',]
    g['df_cb_stats'] = pandas.DataFrame(index=[0,], columns=cols)
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

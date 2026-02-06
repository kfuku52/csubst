import numpy
import pandas

import os
import platform
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
            raise Exception('To enable --exhaustive_until 1, use --foreground')
    if 'fg_clade_permutation' in g.keys():
        if g['fg_clade_permutation']>0:
            if (g['foreground'] is None):
                raise Exception('To enable --fg_clade_permutation, use --foreground')
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
    if 'percent_biased_sub' in g.keys():
        assert (g['percent_biased_sub']<100), '--percent_biased_sub should be <100.'
    if 'float_digit' in g.keys():
        g['float_format'] = '%.'+str(g['float_digit'])+'f'
    if 'threads' in g.keys():
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

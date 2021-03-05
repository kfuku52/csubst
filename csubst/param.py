import numpy
import pandas

def get_global_parameters(args):
    g = dict()
    for attr in [a for a in dir(args) if not a.startswith('_')]:
        g[attr] = getattr(args, attr)
    if 'calc_quantile' in g.keys():
        if g['calc_quantile']:
            assert g['omega_method']=='modelfree', '--calc_quantile "yes" should be used with --omega_method "modelfree".'
    if g['fg_random']>0:
        if g['force_exhaustive']:
            raise Exception('To enable --fg_random, set --force_exhaustive "no"')
        if (g['foreground'] is not None):
            raise Exception('To enable --fg_random, set --foreground')
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
    if 'percent_biased_sub' in g.keys():
        assert (g['percent_biased_sub']<100), '--percent_biased_sub should be <100.'
    return g

def initialize_df_cb_stats(g):
    ind = numpy.arange(0, g['max_arity'])
    cols = ['arity','elapsed_sec','fg_enrichment_factor',]
    g['df_cb_stats'] = pandas.DataFrame(index=ind, columns=cols)
    g['df_cb_stats'].loc[:,'arity'] = ind + 1
    g['df_cb_stats'].loc[:,'cutoff_stat'] = g['cutoff_stat']
    g['df_cb_stats'].loc[:,'cutoff_stat_min'] = g['cutoff_stat_min']
    return(g)

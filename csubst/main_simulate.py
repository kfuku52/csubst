import numpy as np

import copy
import itertools
import os
import re
from collections import OrderedDict

from csubst import foreground
from csubst import genetic_code
from csubst import parser_misc
from csubst import tree
from csubst import ete

_PYVOLVE = None


def _require_pyvolve():
    global _PYVOLVE
    if _PYVOLVE is not None:
        return _PYVOLVE
    try:
        import pyvolve as _pyvolve
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'pyvolve is an optional dependency for `csubst simulate`. '
            'Install it with `pip install "csubst[simulate]"` (or `pip install pyvolve`).'
        ) from exc
    _PYVOLVE = _pyvolve
    return _PYVOLVE


class suppress_stdout_stderr(object):
    # https://www.semicolonworld.com/question/57657/suppress-stdout-stderr-print-from-python-functions
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def scale_tree(tree, scaling_factor):
    for node in tree.traverse():
        node.dist = (node.dist or 0.0) * scaling_factor
    return tree

def get_pyvolve_newick(tree, trait_name):
    for node in tree.traverse():
        ete.set_prop(node, 'dist2', node.dist)
        node.dist = ete.get_prop(node, "numerical_label")
    newick_txt = ete.write_tree(tree, format=1)
    for node in tree.traverse():
        sub_from = str(ete.get_prop(node, "numerical_label"))
        is_fg = ete.get_prop(node, 'is_fg_' + trait_name, False)
        if is_fg:
            fg_lineage = ete.get_prop(node, 'foreground_lineage_id_' + trait_name, 0)
            sub_to = str(ete.get_prop(node, 'dist2')) + '#m' + str(fg_lineage)
        else:
            sub_to = str(ete.get_prop(node, 'dist2'))
        newick_txt = re.sub(r':{}(?=,)'.format(re.escape(sub_from)), ':'+sub_to, newick_txt)
        newick_txt = re.sub(r':{}(?=\))'.format(re.escape(sub_from)), ':'+sub_to, newick_txt)
    for node in tree.traverse():
        node.dist = ete.get_prop(node, 'dist2')
        ete.del_prop(node, 'dist2')
    return newick_txt


def ensure_internal_node_names(tree_obj, prefix='N'):
    leaf_names = set([leaf.name for leaf in ete.get_leaves(tree_obj)])
    used_names = set([name for name in leaf_names if name not in [None, '']])
    for node in tree_obj.traverse():
        if ete.is_leaf(node):
            continue
        raw_name = '' if node.name is None else str(node.name).strip()
        if (raw_name == '') or (raw_name in used_names):
            raw_name = '{}{}'.format(prefix, int(ete.get_prop(node, 'numerical_label')))
        candidate = raw_name
        suffix = 1
        while candidate in used_names:
            candidate = '{}_{}'.format(raw_name, suffix)
            suffix += 1
        node.name = candidate
        used_names.add(candidate)
    return tree_obj

def get_pyvolve_codon_order():
    codons = [ ''.join(a) for a in itertools.product('ACGT', repeat=3) ]
    stops = ['TAA', 'TAG', 'TGA']
    codons = [ c for c in codons if c not in stops ]
    codons = np.array(codons)
    return codons

def get_codons(amino_acids, codon_table):
    cdn = list()
    for ct in codon_table:
        if ct[0] in amino_acids:
            cdn.append(ct[1])
    cdn = np.array(cdn)
    return cdn

def get_biased_nonsynonymous_substitution_index(biased_aas, codon_table, codon_order):
    num_codon = codon_order.shape[0]
    row_index = np.arange(num_codon)
    num_conv_codons = 0
    biased_cdn_index = list()
    for conv_aa in biased_aas:
        conv_cdn = get_codons(amino_acids=conv_aa, codon_table=codon_table)
        index_bool = np.array([ pco in conv_cdn for pco in codon_order ])
        biased_cdn_index0 = np.argwhere(index_bool==True)
        biased_cdn_index0 = biased_cdn_index0.reshape(biased_cdn_index0.shape[0])
        conv_cdn_iter = itertools.product(row_index, biased_cdn_index0)
        biased_cdn_index1 = [ cci for cci in conv_cdn_iter if not ((cci[0] in biased_cdn_index0)&(cci[1] in biased_cdn_index0)) ]
        biased_cdn_index = biased_cdn_index + biased_cdn_index1
        num_conv_codons += biased_cdn_index0.shape[0]
    biased_cdn_index = np.array(biased_cdn_index)
    return biased_cdn_index

def get_biased_amino_acids(convergent_amino_acids, codon_table):
    amino_acids = np.array(sorted(list(set([ item[0] for item in codon_table if item[0] != '*' ]))))
    if convergent_amino_acids.startswith('random'):
        num_random_aa = int(convergent_amino_acids.replace('random',''))
        if num_random_aa < 0:
            raise ValueError('--convergent_amino_acids randomN requires N >= 0.')
        if num_random_aa > amino_acids.shape[0]:
            msg = '--convergent_amino_acids random{} exceeds available amino acids ({}).'
            raise ValueError(msg.format(num_random_aa, amino_acids.shape[0]))
        if num_random_aa == 0:
            return np.array([], dtype=amino_acids.dtype)
        aa_index = np.random.choice(a=np.arange(amino_acids.shape[0]), size=num_random_aa, replace=False)
        biased_aas = amino_acids[aa_index]
    else:
        biased_aas = np.array(list(convergent_amino_acids))
        invalid = sorted(list(set([aa for aa in biased_aas.tolist() if aa not in amino_acids])))
        if len(invalid) > 0:
            invalid_txt = ','.join(invalid)
            raise ValueError('Unknown amino acid(s) in --convergent_amino_acids: {}'.format(invalid_txt))
    return biased_aas

def get_biased_codon_index(biased_aas, codon_table, codon_order):
    biased_cdn_index = list()
    for conv_aa in biased_aas:
        conv_cdn = get_codons(amino_acids=conv_aa, codon_table=codon_table)
        index_bool = np.array([ pco in conv_cdn for pco in codon_order ])
        biased_cdn_index0 = np.argwhere(index_bool==True).tolist()
        biased_cdn_index = biased_cdn_index + biased_cdn_index0
    biased_cdn_index = np.array(biased_cdn_index)
    return biased_cdn_index

def get_synonymous_codon_substitution_index(g, codon_order):
    amino_acids = np.array(list(set([ item[0] for item in g['codon_table'] if item[0]!='*' ])))
    num_codon = codon_order.shape[0]
    num_syn_codons = 0
    cdn_index = list()
    for aa in amino_acids:
        codons = get_codons(amino_acids=aa, codon_table=g['codon_table'])
        if len(codons)==1:
            continue
        index_bool = np.array([ pco in codons for pco in codon_order ])
        cdn_index0 = np.argwhere(index_bool==True)
        cdn_index0 = cdn_index0.reshape(cdn_index0.shape[0])
        cdn_index1 = list(itertools.permutations(cdn_index0, 2))
        cdn_index = cdn_index + cdn_index1
        num_syn_codons += cdn_index0.shape[0]
    cdn_index = np.array(cdn_index)
    txt = '# of all synonymous codon substitutions = {}/{}'
    print(txt.format(cdn_index.shape[0], num_codon*(num_codon-1)), flush=True)
    return cdn_index

def get_nonsynonymous_codon_substitution_index(all_syn_cdn_index):
    pyvolve_codon_order = get_pyvolve_codon_order()
    num_codon = pyvolve_codon_order.shape[0]
    row_index = np.arange(num_codon)
    cdn_index = np.array(list(itertools.permutations(row_index, 2)))
    nsy_index_list = list()
    for ci in np.arange(cdn_index.shape[0]):
        flag = True
        for si in np.arange(all_syn_cdn_index.shape[0]):
            if (cdn_index[ci,0]==all_syn_cdn_index[si,0])&(cdn_index[ci,1]==all_syn_cdn_index[si,1]):
                flag = False
                break
        if flag:
            nsy_index_list.append(ci)
    nsy_index = cdn_index[nsy_index_list,:]
    txt = '# of all nonsynonymous codon substitutions = {}/{}'
    print(txt.format(nsy_index.shape[0], num_codon*(num_codon-1)), flush=True)
    return nsy_index

def get_total_Q(mat, cdn_index):
    total = 0
    for ci in np.arange(cdn_index.shape[0]):
        row = cdn_index[ci,0]
        col = cdn_index[ci,1]
        total += mat[row,col]
    return total

def _normalize_matrix_by_expected_rate(mat, eq_freq):
    eq_freq = np.asarray(eq_freq, dtype=float).reshape(-1)
    mat = np.array(mat, copy=True, dtype=float)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Rate matrix should be square.')
    if mat.shape[0] != eq_freq.shape[0]:
        txt = 'Equilibrium frequency length ({}) should match matrix dimension ({}).'
        raise ValueError(txt.format(eq_freq.shape[0], mat.shape[0]))
    if not np.isfinite(eq_freq).all():
        raise ValueError('Equilibrium frequencies should be finite.')
    if (eq_freq < 0).any():
        raise ValueError('Equilibrium frequencies should be >= 0.')
    total_eq = eq_freq.sum()
    if not (total_eq > 0):
        raise ValueError('Equilibrium frequencies should sum to a positive value.')
    eq_freq = eq_freq / total_eq
    diag_bool = np.eye(mat.shape[0], dtype=bool)
    mat[diag_bool] = 0
    mat[diag_bool] = -mat.sum(axis=1)
    expected_rate = float(np.sum(eq_freq * (-np.diag(mat))))
    if not np.isfinite(expected_rate):
        raise ValueError('Expected substitution rate should be finite.')
    if expected_rate <= 0:
        raise ValueError('Expected substitution rate should be positive.')
    mat /= expected_rate
    mat[diag_bool] = 0
    mat[diag_bool] = -mat.sum(axis=1)
    return mat


def rescale_substitution_matrix(mat, target_index, scaling_factor, eq_freq=None):
    mat = copy.copy(mat)
    diag_bool = np.eye(mat.shape[0], dtype=bool)
    mat[diag_bool] = 0
    sum_before = mat.sum()
    for i in np.arange(target_index.shape[0]):
        mat[target_index[i,0],target_index[i,1]] *= scaling_factor
    if eq_freq is None:
        sum_after = mat.sum()
        mat = mat / sum_after * sum_before
    else:
        mat[diag_bool] = -mat.sum(axis=1)
        mat = _normalize_matrix_by_expected_rate(mat=mat, eq_freq=eq_freq)
    mat[diag_bool] = 0
    mat[diag_bool] = -mat.sum(axis=1)
    return mat

def generate_Q_matrix(eq_freq, omega, all_nsy_cdn_index, all_syn_cdn_index):
    pyvolve = _require_pyvolve()
    all_cdn_index = np.concatenate([all_syn_cdn_index, all_nsy_cdn_index])
    cmp = {'omega':omega, 'k_ti':1, 'k_tv':1} # background_omega have to be readjusted.
    model = pyvolve.Model(model_type='ECMunrest', name='placeholder', parameters=cmp, state_freqs=eq_freq)
    mat = model.matrix
    dnds = get_total_Q(mat, all_nsy_cdn_index) / get_total_Q(mat, all_syn_cdn_index)
    mat = rescale_substitution_matrix(mat, all_nsy_cdn_index, scaling_factor=omega/dnds, eq_freq=eq_freq)
    return mat

def bias_eq_freq(eq_freq, biased_cdn_index, convergence_intensity_factor):
    eq_freq = np.array(eq_freq)
    eq_freq[biased_cdn_index] *= convergence_intensity_factor
    eq_freq /= eq_freq.sum()
    return eq_freq

def get_total_biased_Q(mat, biased_aas, codon_table, codon_order):
    biased_nsy_cdn_index = get_biased_nonsynonymous_substitution_index(biased_aas, codon_table, codon_order)
    total_biased_Q = 0
    for i in np.arange(biased_nsy_cdn_index.shape[0]):
        total_biased_Q += mat[biased_nsy_cdn_index[i,0],biased_nsy_cdn_index[i,1]]
    return total_biased_Q

def apply_percent_biased_sub(mat, percent_biased_sub, target_index, biased_aas, codon_table, codon_orders,
                             all_nsy_cdn_index, all_syn_cdn_index, foreground_omega, eq_freq=None):
    mat = copy.copy(mat)
    diag_bool = np.eye(mat.shape[0], dtype=bool)
    mat[diag_bool] = 0
    if (target_index is None) or (np.asarray(target_index).shape[0] == 0) or (len(biased_aas) == 0):
        dnds = get_total_Q(mat, all_nsy_cdn_index) / get_total_Q(mat, all_syn_cdn_index)
        if dnds == 0:
            raise ValueError('Cannot rescale nonsynonymous rates: background dN/dS is zero.')
        omega_scaling_factor = foreground_omega / dnds
        mat = rescale_substitution_matrix(mat, all_nsy_cdn_index, omega_scaling_factor, eq_freq=eq_freq)
        return mat
    total_biased_Q_before = get_total_biased_Q(mat, biased_aas, codon_table, codon_orders)
    if total_biased_Q_before <= 0:
        raise ValueError('No target-biased nonsynonymous substitutions are available for the selected amino acids.')
    total_nsy_Q_before = get_total_Q(mat, all_nsy_cdn_index)
    scaling_factor = total_nsy_Q_before / total_biased_Q_before / (1-(percent_biased_sub/100))
    mat = rescale_substitution_matrix(mat, target_index, scaling_factor, eq_freq=eq_freq)
    dnds = get_total_Q(mat, all_nsy_cdn_index) / get_total_Q(mat, all_syn_cdn_index)
    if dnds == 0:
        raise ValueError('Cannot rescale nonsynonymous rates: current dN/dS is zero.')
    omega_scaling_factor = foreground_omega/dnds
    mat = rescale_substitution_matrix(mat, all_nsy_cdn_index, omega_scaling_factor, eq_freq=eq_freq)
    return mat


def _validate_simulate_params(g):
    if int(g['num_simulated_site']) <= 0:
        raise ValueError('--num_simulated_site must be a positive integer.')
    if (g['percent_convergent_site'] < 0) or (g['percent_convergent_site'] > 100):
        raise ValueError('--percent_convergent_site must be within [0, 100].')
    if (g['percent_biased_sub'] < 0) or (g['percent_biased_sub'] >= 100):
        raise ValueError('--percent_biased_sub must be within [0, 100).')
    return None


def _scale_rate_matrix(matrix, rate_factor):
    rate_factor = float(rate_factor)
    if not np.isfinite(rate_factor):
        raise ValueError('Simulation site-rate factors should be finite.')
    if rate_factor < 0:
        raise ValueError('Simulation site-rate factors should be >= 0.')
    scaled = np.array(matrix, copy=True)
    if rate_factor != 1.0:
        scaled *= rate_factor
    return scaled


def _resolve_simulation_site_rates(g):
    num_site = int(g['num_simulated_site'])
    mode = str(g.get('simulate_asrv', 'no')).strip().lower()
    if mode == 'no':
        return np.ones(num_site, dtype=float)
    if mode != 'file':
        raise ValueError('Unsupported --simulate_asrv mode: {}'.format(mode))
    if ('iqtree_rate_values' not in g) or (g['iqtree_rate_values'] is None):
        raise ValueError('--simulate_asrv file requires IQ-TREE site rates.')
    source_rates = np.asarray(g['iqtree_rate_values'], dtype=float).reshape(-1)
    if source_rates.shape[0] == 0:
        raise ValueError('IQ-TREE site rates are empty for --simulate_asrv file.')
    if not np.isfinite(source_rates).all():
        raise ValueError('IQ-TREE site rates include non-finite values.')
    if (source_rates < 0).any():
        raise ValueError('IQ-TREE site rates include negative values.')
    if num_site <= source_rates.shape[0]:
        return source_rates[:num_site].copy()
    txt = '--num_simulated_site ({}) exceeded available IQ-TREE site rates ({}). Reusing rates cyclically.'
    print(txt.format(num_site, source_rates.shape[0]), flush=True)
    return np.resize(source_rates, num_site).astype(float)


def evolve_convergent_partitions(g):
    pyvolve = _require_pyvolve()
    num_fl = foreground.get_num_foreground_lineages(tree=g['tree'], trait_name=g['trait_name'])
    model_names = ['root',] + [ 'm'+str(i+1) for i in range(num_fl) ]
    num_convergent_partition = g['num_convergent_site']
    site_rates = np.asarray(g.get('simulate_rate_convergent', np.ones(num_convergent_partition)), dtype=float)
    if site_rates.shape[0] != num_convergent_partition:
        txt = 'simulate_rate_convergent length ({}) did not match num_convergent_site ({}).'
        raise ValueError(txt.format(site_rates.shape[0], num_convergent_partition))
    convergent_partitions = list()
    biased_substitution_fractions = list()
    current_site = 0
    for partition_index in np.arange(num_convergent_partition):
        current_site += 1
        site_rate = float(site_rates[partition_index])
        biased_aas = get_biased_amino_acids(g['convergent_amino_acids'], g['codon_table'])
        print('Codon site {}; Biased amino acids = {}; '.format(current_site, ''.join(biased_aas)), end='')
        biased_nsy_sub_index = get_biased_nonsynonymous_substitution_index(biased_aas,
                                                                           g['codon_table'],
                                                                           g['pyvolve_codon_orders'])
        biased_Q = apply_percent_biased_sub(mat=g['background_Q'],
                                            percent_biased_sub=g['percent_biased_sub'],
                                            target_index=biased_nsy_sub_index,
                                            biased_aas=biased_aas,
                                            codon_table=g['codon_table'],
                                            codon_orders=g['pyvolve_codon_orders'],
                                            all_nsy_cdn_index=g['all_nsy_cdn_index'],
                                            all_syn_cdn_index=g['all_syn_cdn_index'],
                                            foreground_omega=g['foreground_omega'],
                                            eq_freq=g['eq_freq'],
                                            )
        total_nsy_Q = get_total_Q(biased_Q, g['all_nsy_cdn_index'])
        total_biased_Q = get_total_biased_Q(biased_Q, biased_aas, g['codon_table'], g['pyvolve_codon_orders'])
        fraction_biased_Q = total_biased_Q / total_nsy_Q
        bg_total_nsy_Q = get_total_Q(g['background_Q'], g['all_nsy_cdn_index'])
        bg_total_biased_Q = get_total_biased_Q(g['background_Q'], biased_aas, g['codon_table'], g['pyvolve_codon_orders'])
        bg_fraction_biased_Q = bg_total_biased_Q / bg_total_nsy_Q
        txt = 'Total in Q toward the codons before and after the bias introduction = ' \
              '{:,.1f}% ({:,.1f}/{:,.1f}) and {:,.1f}% ({:,.1f}/{:,.1f})'
        print(txt.format(bg_fraction_biased_Q*100, bg_total_biased_Q, bg_total_nsy_Q,
                         fraction_biased_Q*100, total_biased_Q, total_nsy_Q))
        biased_substitution_fractions.append(fraction_biased_Q)
        models = list()
        for model_name in model_names:
            is_nonroot_model = (model_name!='root')
            if (is_nonroot_model):
                q_matrix = _scale_rate_matrix(biased_Q, site_rate)
            else:
                q_matrix = _scale_rate_matrix(g['background_Q'], site_rate)
            with suppress_stdout_stderr():
                model = pyvolve.Model(model_type='custom', name=model_name, parameters={'matrix':q_matrix})
            models.append(model)
        partition = pyvolve.Partition(models=models, size=1,  root_model_name='root')
        convergent_partitions.append(partition)
    if len(biased_substitution_fractions):
        mean_biased_substitution_fraction = np.array(biased_substitution_fractions).mean()
    else:
        mean_biased_substitution_fraction = 0
    txt = '{:,.2f}% of substitutions in {} sites in the foreground branches are ' \
          'expected to result from the introduced bias in Q matrix.'
    fraction_convergent_site = g['num_convergent_site'] / g['num_simulated_site']
    print(txt.format(mean_biased_substitution_fraction*fraction_convergent_site*100, g['num_simulated_site']))
    txt = '{:,.2f}% of substitutions in {} convergent sites in the foreground branches are ' \
          'expected to result from the introduced bias in Q matrix.'
    print(txt.format(mean_biased_substitution_fraction*100, g['num_convergent_site']))
    evolver = pyvolve.Evolver(partitions=convergent_partitions, tree=g['foreground_tree'])
    kwargs = dict(
        ratefile='tmp.csubst.simulate_convergent_ratefile.txt',
        infofile='tmp.csubst.simulate_convergent_infofile.txt',
        seqfile='tmp.csubst.simulate_convergent.fa',
        write_anc=bool(g.get('export_true_asr', True)),
    )
    if g.get('simulate_seed_convergent', None) is not None:
        kwargs['seed'] = int(g['simulate_seed_convergent'])
    evolver(**kwargs)

def evolve_nonconvergent_partition(g):
    pyvolve = _require_pyvolve()
    if (g['num_convergent_site']==0):
        site_start = 1
    else:
        site_start = g['num_simulated_site'] - g['num_convergent_site'] + 1
    site_end = g['num_simulated_site']
    print('Codon site {}-{}; Non-convergent codons'.format(site_start, site_end))
    num_nonconvergent_site = g['num_simulated_site'] - g['num_convergent_site']
    site_rates = np.asarray(g.get('simulate_rate_nonconvergent', np.ones(num_nonconvergent_site)), dtype=float)
    if site_rates.shape[0] != num_nonconvergent_site:
        txt = 'simulate_rate_nonconvergent length ({}) did not match non-convergent site count ({}).'
        raise ValueError(txt.format(site_rates.shape[0], num_nonconvergent_site))
    if (str(g.get('simulate_asrv', 'no')).strip().lower() == 'file'):
        partitions = list()
        for site_rate in site_rates:
            q_matrix = _scale_rate_matrix(g['background_Q'], site_rate)
            with suppress_stdout_stderr():
                model = pyvolve.Model(model_type='custom', name='root', parameters={'matrix':q_matrix})
            partitions.append(pyvolve.Partition(models=model, size=1))
        evolver = pyvolve.Evolver(partitions=partitions, tree=g['background_tree'])
    else:
        q_matrix = _scale_rate_matrix(g['background_Q'], 1.0)
        with suppress_stdout_stderr():
            model = pyvolve.Model(model_type='custom', name='root', parameters={'matrix':q_matrix})
        partition = pyvolve.Partition(models=model, size=num_nonconvergent_site)
        evolver = pyvolve.Evolver(partitions=partition, tree=g['background_tree'])
    kwargs = dict(
        ratefile='tmp.csubst.simulate_nonconvergent_ratefile.txt',
        infofile='tmp.csubst.simulate_nonconvergent_infofile.txt',
        seqfile='tmp.csubst.simulate_nonconvergent.fa',
        write_anc=bool(g.get('export_true_asr', True)),
    )
    if g.get('simulate_seed_nonconvergent', None) is not None:
        kwargs['seed'] = int(g['simulate_seed_nonconvergent'])
    evolver(**kwargs)

def get_pyvolve_tree(tree, foreground_scaling_factor, trait_name):
    pyvolve = _require_pyvolve()
    if (foreground_scaling_factor!=1):
        print('Foreground branches are rescaled by {}.'.format(foreground_scaling_factor))
    for node in tree.traverse():
        if ete.get_prop(node, 'is_fg_' + trait_name, False):
            node.dist *= foreground_scaling_factor
    newick_txt = get_pyvolve_newick(tree=tree, trait_name=trait_name)
    pyvolve_tree = pyvolve.read_tree(tree=newick_txt)
    return pyvolve_tree


def _get_synonymous_indices(codon_order, codon_table):
    codon_order = np.asarray(codon_order, dtype=object).reshape(-1)
    aa_order = sorted(list(set([item[0] for item in codon_table if item[0] != '*'])))
    synonymous_indices = dict()
    codon_to_index = {str(codon): idx for idx, codon in enumerate(codon_order.tolist())}
    for aa in aa_order:
        synonymous_codons = [item[1] for item in codon_table if item[0] == aa]
        synonymous_indices[aa] = [codon_to_index[codon] for codon in synonymous_codons if codon in codon_to_index]
    return np.asarray(aa_order, dtype=object), synonymous_indices


def _get_model_base_name(model_txt):
    model_txt = str(model_txt).strip()
    if model_txt == '':
        return ''
    return re.sub(r'\+.*$', '', model_txt)


def _build_mechanistic_background_Q(g):
    aa_order, synonymous_indices = _get_synonymous_indices(
        codon_order=g['pyvolve_codon_orders'],
        codon_table=g['codon_table'],
    )
    kappa = g.get('kappa', None)
    if kappa is not None:
        kappa = float(kappa)
    local_g = {
        'codon_orders': np.asarray(g['pyvolve_codon_orders'], dtype=object),
        'amino_acid_orders': aa_order,
        'synonymous_indices': synonymous_indices,
        'omega': 1.0,
        'kappa': kappa,
        'equilibrium_frequency': np.asarray(g['eq_freq'], dtype=float),
        'float_type': g.get('float_type', np.float64),
    }
    return parser_misc.get_mechanistic_instantaneous_rate_matrix(local_g)


def _resolve_simulation_background_omega(g):
    user_background_omega = g.get('background_omega', None)
    if user_background_omega is not None:
        return float(user_background_omega)
    iqtree_omega = g.get('omega', None)
    if iqtree_omega is None:
        default_omega = 0.2
        txt = 'IQ-TREE omega was unavailable. Falling back to background omega={}.'
        print(txt.format(default_omega), flush=True)
        return float(default_omega)
    iqtree_omega = float(iqtree_omega)
    if not np.isfinite(iqtree_omega):
        raise ValueError('IQ-TREE omega should be finite when used as simulation background omega.')
    if iqtree_omega < 0:
        raise ValueError('IQ-TREE omega should be >= 0 when used for simulation.')
    txt = 'Using IQ-TREE estimated omega for simulation background: {:.6f}'
    print(txt.format(iqtree_omega), flush=True)
    return float(iqtree_omega)


def _resolve_simulation_eq_freq(g):
    mode = str(g.get('simulate_eq_freq', 'auto')).strip().lower()
    if mode not in ['auto', 'iqtree', 'alignment']:
        raise ValueError('Unsupported --simulate_eq_freq mode: {}'.format(mode))
    eq_freq_iqtree = None
    if ('equilibrium_frequency' in g) and (g['equilibrium_frequency'] is not None):
        eq_freq_iqtree = np.asarray(g['equilibrium_frequency'], dtype=float).reshape(-1)
        input_codon_order = np.asarray(g.get('codon_orders', g['pyvolve_codon_orders']), dtype=object).reshape(-1)
        if eq_freq_iqtree.shape[0] != input_codon_order.shape[0]:
            txt = 'Ignoring IQ-TREE equilibrium frequency because its length ({}) did not match codon order ({}).'
            print(txt.format(eq_freq_iqtree.shape[0], input_codon_order.shape[0]), flush=True)
            eq_freq_iqtree = None
        else:
            codon_order_index = parser_misc.get_codon_order_index(
                order_from=g['pyvolve_codon_orders'],
                order_to=input_codon_order,
            )
            eq_freq_iqtree = eq_freq_iqtree[codon_order_index]
            if not np.isfinite(eq_freq_iqtree).all():
                raise ValueError('IQ-TREE equilibrium frequencies should be finite.')
            if (eq_freq_iqtree < 0).any():
                raise ValueError('IQ-TREE equilibrium frequencies should be >= 0.')
            total = eq_freq_iqtree.sum()
            if not (total > 0):
                raise ValueError('IQ-TREE equilibrium frequencies should sum to a positive value.')
            eq_freq_iqtree = eq_freq_iqtree / total
    if (mode in ['auto', 'iqtree']) and (eq_freq_iqtree is not None):
        txt = 'Simulation codon equilibrium frequencies: using IQ-TREE output ({} mode).'
        print(txt.format(mode), flush=True)
        return eq_freq_iqtree
    if mode == 'iqtree':
        raise ValueError('--simulate_eq_freq iqtree requires parsable IQ-TREE equilibrium frequencies.')
    pyvolve = _require_pyvolve()
    f = pyvolve.ReadFrequencies('codon', file=g['alignment_file'])
    eq_freq_alignment = np.asarray(f.compute_frequencies(), dtype=float).reshape(-1)
    if not np.isfinite(eq_freq_alignment).all():
        raise ValueError('Alignment-derived equilibrium frequencies should be finite.')
    if (eq_freq_alignment < 0).any():
        raise ValueError('Alignment-derived equilibrium frequencies should be >= 0.')
    total = eq_freq_alignment.sum()
    if not (total > 0):
        raise ValueError('Alignment-derived equilibrium frequencies should sum to a positive value.')
    eq_freq_alignment /= total
    if mode == 'alignment':
        print('Simulation codon equilibrium frequencies: using alignment frequencies (--simulate_eq_freq alignment).', flush=True)
    else:
        print('Simulation codon equilibrium frequencies: IQ-TREE unavailable, falling back to alignment frequencies.', flush=True)
    return eq_freq_alignment


def get_background_Q(g, Q_method):
    q_method = str(Q_method).strip().lower()
    if (q_method == 'auto'):
        base_model = _get_model_base_name(g.get('substitution_model', ''))
        if base_model.startswith('ECMrest'):
            matrix_file = 'substitution_matrix/ECMrest.dat'
            g['exchangeability_matrix'] = parser_misc.read_exchangeability_matrix(matrix_file, g['pyvolve_codon_orders'])
            unscaled_mat = parser_misc.exchangeability2Q(g['exchangeability_matrix'], g['eq_freq'])
        elif base_model.startswith('GY') or base_model.startswith('MG'):
            unscaled_mat = _build_mechanistic_background_Q(g)
        else:
            if base_model.startswith('ECMK07'):
                matrix_file = 'substitution_matrix/ECMunrest.dat'
            else:
                matrix_file = 'substitution_matrix/ECMunrest.dat'
                txt = 'Unsupported/unknown model "{}" in simulate. Falling back to {}.'
                print(txt.format(base_model, matrix_file), flush=True)
            g['exchangeability_matrix'] = parser_misc.read_exchangeability_matrix(matrix_file, g['pyvolve_codon_orders'])
            unscaled_mat = parser_misc.exchangeability2Q(g['exchangeability_matrix'], g['eq_freq'])
        dnds = get_total_Q(unscaled_mat, g['all_nsy_cdn_index']) / get_total_Q(unscaled_mat, g['all_syn_cdn_index'])
        scaling_factor = g['background_omega'] / dnds
        background_Q = rescale_substitution_matrix(
            unscaled_mat,
            g['all_nsy_cdn_index'],
            scaling_factor,
            eq_freq=g['eq_freq'],
        )
    elif (q_method == 'csubst'):
        matrix_file = 'substitution_matrix/ECMunrest.dat'
        g['exchangeability_matrix'] = parser_misc.read_exchangeability_matrix(matrix_file, g['pyvolve_codon_orders'])
        unscaled_mat = parser_misc.exchangeability2Q(g['exchangeability_matrix'], g['eq_freq'])
        dnds = get_total_Q(unscaled_mat, g['all_nsy_cdn_index']) / get_total_Q(unscaled_mat, g['all_syn_cdn_index'])
        scaling_factor = g['background_omega'] / dnds
        background_Q = rescale_substitution_matrix(
            unscaled_mat,
            g['all_nsy_cdn_index'],
            scaling_factor,
            eq_freq=g['eq_freq'],
        )
    elif (q_method == 'pyvolve'):
        background_Q = generate_Q_matrix(g['eq_freq'], g['background_omega'], g['all_nsy_cdn_index'], g['all_syn_cdn_index'])
        background_Q = _normalize_matrix_by_expected_rate(background_Q, g['eq_freq'])
    else:
        raise ValueError('Unsupported Q matrix method: {}'.format(Q_method))
    return background_Q

def concatenate_alignment(in1, in2, out):
    seqs1 = read_fasta(in1)
    seqs2 = read_fasta(in2)
    if set(seqs1.keys()) != set(seqs2.keys()):
        missing_in_second = sorted(list(set(seqs1.keys()) - set(seqs2.keys())))
        missing_in_first = sorted(list(set(seqs2.keys()) - set(seqs1.keys())))
        txt = 'FASTA headers differ between files. Missing in second: {}; Missing in first: {}'
        raise ValueError(txt.format(','.join(missing_in_second), ','.join(missing_in_first)))
    with open(out, 'w') as f:
        for key in seqs1.keys():
            f.write('>' + key + '\n' + seqs1[key] + seqs2[key] + '\n')
    return None


def read_fasta(path):
    seqs = OrderedDict()
    current_name = None
    with open(path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == '':
                continue
            if line.startswith('>'):
                current_name = line[1:].strip()
                if current_name == '':
                    raise ValueError('Invalid FASTA header in {}.'.format(path))
                if current_name in seqs:
                    raise ValueError('Duplicate FASTA header in {}: {}'.format(path, current_name))
                seqs[current_name] = ''
                continue
            if current_name is None:
                raise ValueError('Invalid FASTA format in {}: sequence line appeared before header.'.format(path))
            seqs[current_name] += line.upper().replace('U', 'T')
    return seqs


def write_fasta(seqs, path):
    with open(path, 'w') as f:
        for name, seq in seqs.items():
            f.write('>{}\n{}\n'.format(name, seq))
    return None


def split_tip_and_ancestor_alignment(in_fasta, tip_out, anc_out, tip_names):
    tip_names = set([str(name) for name in tip_names])
    seqs = read_fasta(in_fasta)
    tip = OrderedDict()
    anc = OrderedDict()
    for name, seq in seqs.items():
        if name in tip_names:
            tip[name] = seq
        else:
            anc[name] = seq
    if len(tip) != len(tip_names):
        missing_tip = sorted(list(set(tip_names) - set(tip.keys())))
        txt = 'Some tip sequences were not found in simulated FASTA: {}'
        raise ValueError(txt.format(','.join(missing_tip[:10])))
    write_fasta(tip, tip_out)
    write_fasta(anc, anc_out)
    return len(tip), len(anc)


def write_true_asr_bundle(g, anc_fasta, prefix):
    prefix = str(prefix).strip()
    if prefix == '':
        raise ValueError('--true_asr_prefix should be a non-empty path prefix.')
    anc_seqs = read_fasta(anc_fasta)
    if len(anc_seqs) == 0:
        raise ValueError('No ancestral sequences were found in {}.'.format(anc_fasta))
    codon_order = get_pyvolve_codon_order()
    codon_to_index = {codon: i for i, codon in enumerate(codon_order.tolist())}
    internal_nodes = [node for node in g['tree'].traverse() if not ete.is_leaf(node)]
    internal_names = [str(node.name) for node in internal_nodes]
    missing_nodes = [name for name in internal_names if name not in anc_seqs]
    if len(missing_nodes):
        txt = 'Ancestral FASTA did not contain all internal nodes. Missing examples: {}'
        raise ValueError(txt.format(','.join(missing_nodes[:10])))
    seq_len = len(next(iter(anc_seqs.values())))
    if seq_len % 3 != 0:
        raise ValueError('Ancestral sequence length should be a multiple of 3.')
    num_site = seq_len // 3
    for name, seq in anc_seqs.items():
        if len(seq) != seq_len:
            raise ValueError('Ancestral sequence lengths are inconsistent: {}'.format(name))
    state_file = prefix + '.state'
    tree_file = prefix + '.treefile'
    rate_file = prefix + '.rate'
    iqtree_file = prefix + '.iqtree'
    log_file = prefix + '.log'
    anc_file = prefix + '.anc.fa'
    out_tree = copy.deepcopy(g['tree'])
    ete.write_tree(out_tree, format=1, outfile=tree_file)
    write_fasta(anc_seqs, anc_file)
    with open(state_file, 'w') as f:
        header = ['Node', 'Site', 'State'] + ['p_' + c for c in codon_order.tolist()]
        f.write('\t'.join(header) + '\n')
        for node in internal_nodes:
            node_name = str(node.name)
            seq = anc_seqs[node_name]
            for site in range(num_site):
                codon = seq[(site * 3):((site + 1) * 3)]
                state_vector = ['0'] * codon_order.shape[0]
                codon_index = codon_to_index.get(codon, None)
                if codon_index is not None:
                    state_txt = codon
                    state_vector[codon_index] = '1'
                else:
                    state_txt = '???'
                row = [node_name, str(site + 1), state_txt] + state_vector
                f.write('\t'.join(row) + '\n')
    with open(rate_file, 'w') as f:
        f.write('Site\tC_Rate\n')
        for site in range(num_site):
            f.write('{}\t1.000000\n'.format(site + 1))
    model_txt = str(g.get('iqtree_model', 'ECMK07+F+R4'))
    eq_freq = np.asarray(g['eq_freq'], dtype=float)
    with open(iqtree_file, 'w') as f:
        f.write('IQ-TREE multicore version 2.2.6\n')
        f.write('Model of substitution: {}\n'.format(model_txt))
        for codon, freq in zip(codon_order.tolist(), eq_freq.tolist()):
            f.write('pi({}) = {:.10f}\n'.format(codon, float(freq)))
    with open(log_file, 'w') as f:
        f.write('IQ-TREE multicore version 2.2.6\n')
        f.write('Converting to codon sequences with genetic code {} ...\n'.format(int(g['genetic_code'])))
        f.write('Nonsynonymous/synonymous ratio (omega): {:.6f}\n'.format(float(g['background_omega'])))
        f.write('Transition/transversion ratio (kappa): 1.000000\n')
    return {
        'state': os.path.abspath(state_file),
        'treefile': os.path.abspath(tree_file),
        'rate': os.path.abspath(rate_file),
        'iqtree': os.path.abspath(iqtree_file),
        'log': os.path.abspath(log_file),
        'anc_fasta': os.path.abspath(anc_file),
    }

def main_simulate(g, Q_method='auto'):
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['genetic_code'])
    g = tree.read_treefile(g)
    g = parser_misc.generate_intermediate_files(g, force_notree_run=True)
    g = parser_misc.annotate_tree(g, ignore_tree_inconsistency=True)
    g = parser_misc.read_input(g)
    g['simulate_asrv'] = str(g.get('simulate_asrv', 'no')).strip().lower()
    g['export_true_asr'] = bool(g.get('export_true_asr', True))
    g['true_asr_prefix'] = str(g.get('true_asr_prefix', 'simulate_true_asr')).strip()
    if g['export_true_asr'] and (g['true_asr_prefix'] == ''):
        raise ValueError('--true_asr_prefix should be a non-empty path prefix.')
    g['trait_name'] = 'PLACEHOLDER'
    g['tree'] = ensure_internal_node_names(g['tree'])
    g['pyvolve_codon_orders'] = get_pyvolve_codon_order()
    g['all_syn_cdn_index'] = get_synonymous_codon_substitution_index(g, g['pyvolve_codon_orders'])
    g['all_nsy_cdn_index'] = get_nonsynonymous_codon_substitution_index(g['all_syn_cdn_index'])
    if (g['num_simulated_site']==-1):
        g['num_simulated_site'] = g['num_input_site']
    _validate_simulate_params(g)
    if int(g.get('simulate_seed', -1)) >= 0:
        base_seed = int(g['simulate_seed'])
        g['simulate_seed_convergent'] = base_seed
        g['simulate_seed_nonconvergent'] = base_seed + 1
        txt = 'Simulation seeds: convergent={}, nonconvergent={}'
        print(txt.format(g['simulate_seed_convergent'], g['simulate_seed_nonconvergent']), flush=True)
    else:
        g['simulate_seed_convergent'] = None
        g['simulate_seed_nonconvergent'] = None
    if g['optimized_branch_length']:
        g['tree'] = g['tree']
    else:
        g['tree2'] = g['tree']
        g['tree'] = g['rooted_tree']
    g['tree'] = scale_tree(tree=g['tree'], scaling_factor=g['tree_scaling_factor'])
    g['tree'] = ensure_internal_node_names(g['tree'])
    # Re-annotate the tree chosen for simulation/plotting so lineage colors and
    # lineage IDs in --foreground are reflected in simulate_branch_id.pdf and
    # pyvolve model tags.
    g = foreground.get_foreground_branch(g, simulate=True)
    tree.plot_branch_category(g, file_base='simulate_branch_id', label='all')
    g['background_tree'] = get_pyvolve_tree(g['tree'], foreground_scaling_factor=1, trait_name=g['trait_name'])
    g['foreground_tree'] = get_pyvolve_tree(g['tree'], foreground_scaling_factor=g['foreground_scaling_factor'], trait_name=g['trait_name'])
    g['background_omega'] = _resolve_simulation_background_omega(g)
    g['eq_freq'] = _resolve_simulation_eq_freq(g)
    g['background_Q'] = get_background_Q(g, Q_method)
    if g['foreground'] is None:
        print('--foreground was not provided. --percent_convergent_site will be set to 0.')
        g['num_convergent_site'] = 0
    else:
        g['num_convergent_site'] = int(g['num_simulated_site'] * g['percent_convergent_site'] / 100)
    g['num_no_convergent_site'] = int(g['num_simulated_site'] - g['num_convergent_site'])
    g['simulate_site_rates'] = _resolve_simulation_site_rates(g)
    g['simulate_rate_convergent'] = g['simulate_site_rates'][:g['num_convergent_site']]
    g['simulate_rate_nonconvergent'] = g['simulate_site_rates'][g['num_convergent_site']:]
    if g['simulate_asrv'] == 'file':
        txt = 'Simulation ASRV (--simulate_asrv file): min={:.4f}, median={:.4f}, max={:.4f}, mean={:.4f}'
        print(txt.format(
            float(np.min(g['simulate_site_rates'])),
            float(np.median(g['simulate_site_rates'])),
            float(np.max(g['simulate_site_rates'])),
            float(np.mean(g['simulate_site_rates'])),
        ), flush=True)
    txt = '{:,} out of {:,} sites ({:.1f}%) will evolve convergently in the foreground lineages.'
    print(txt.format(g['num_convergent_site'], g['num_simulated_site'], g['percent_convergent_site']))
    if (g['num_convergent_site'] > 0):
        evolve_convergent_partitions(g)
    if (g['num_no_convergent_site'] > 0):
        evolve_nonconvergent_partition(g)
    file_conv = 'tmp.csubst.simulate_convergent.fa'
    file_noconv = 'tmp.csubst.simulate_nonconvergent.fa'
    file_out = 'simulate.fa'
    file_all = 'tmp.csubst.simulate_all.fa'
    file_anc = 'tmp.csubst.simulate_anc.fa'
    if (os.path.exists(file_conv))&(not os.path.exists(file_noconv)):
        os.rename(file_conv, file_out if (not g['export_true_asr']) else file_all)
    elif (not os.path.exists(file_conv))&(os.path.exists(file_noconv)):
        os.rename(file_noconv, file_out if (not g['export_true_asr']) else file_all)
    else:
        concatenate_alignment(
            in1=file_conv,
            in2=file_noconv,
            out=file_out if (not g['export_true_asr']) else file_all,
        )
    if g['export_true_asr']:
        tip_names = [leaf.name for leaf in ete.get_leaves(g['tree'])]
        num_tip, num_anc = split_tip_and_ancestor_alignment(
            in_fasta=file_all,
            tip_out=file_out,
            anc_out=file_anc,
            tip_names=tip_names,
        )
        txt = 'Split simulated sequences into tips ({}) and ancestors ({}) for true-ASR export.'
        print(txt.format(num_tip, num_anc), flush=True)
        true_asr_files = write_true_asr_bundle(
            g=g,
            anc_fasta=file_anc,
            prefix=g['true_asr_prefix'],
        )
        print('True-ASR bundle prefix: {}'.format(os.path.abspath(g['true_asr_prefix'])), flush=True)
        print('  state: {}'.format(true_asr_files['state']), flush=True)
        print('  tree : {}'.format(true_asr_files['treefile']), flush=True)
        print('  rate : {}'.format(true_asr_files['rate']), flush=True)
        print('  iqtree: {}'.format(true_asr_files['iqtree']), flush=True)
        print('  log  : {}'.format(true_asr_files['log']), flush=True)
        print('  anc  : {}'.format(true_asr_files['anc_fasta']), flush=True)

    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]

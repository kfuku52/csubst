import pyvolve
import numpy

import contextlib
import copy
import io
import itertools
import os
import re
import sys

from csubst import foreground
from csubst import genetic_code
from csubst import parser_misc
from csubst.tree import plot_branch_category # This cannot be "from csubst import tree" because of conflicting name "tree"

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
        node.dist = node.dist * scaling_factor
    return tree

def get_pyvolve_tree(tree):
    for node in tree.traverse():
        node.dist2 = node.dist
        node.dist = node.numerical_label
    newick_txt = tree.write(format=1)
    for node in tree.traverse():
        sub_from = str(node.numerical_label)
        if node.is_foreground:
            sub_to = str(node.dist2)+'#m'+str(node.foreground_lineage_id)
        else:
            sub_to = str(node.dist2)
        newick_txt = re.sub(':'+sub_from+',', ':'+sub_to+',', newick_txt)
        newick_txt = re.sub(':'+sub_from+'\)', ':'+sub_to+')', newick_txt)
    for node in tree.traverse():
        node.dist = node.dist2
        node.dist2 = None
    return newick_txt

def get_pyvolve_codon_order():
    codons = [ ''.join(a) for a in itertools.product('ACGT', repeat=3) ]
    stops = ['TAA', 'TAG', 'TGA']
    codons = [ c for c in codons if c not in stops ]
    codons = numpy.array(codons)
    return codons

def get_codons(amino_acids, codon_table):
    cdn = list()
    for ct in codon_table:
        if ct[0] in amino_acids:
            cdn.append(ct[1])
    cdn = numpy.array(cdn)
    return cdn

def get_biased_nonsynonymous_substitution_index(biased_aas, codon_table, codon_order):
    num_codon = codon_order.shape[0]
    row_index = numpy.arange(num_codon)
    num_conv_codons = 0
    biased_cdn_index = list()
    for conv_aa in biased_aas:
        conv_cdn = get_codons(amino_acids=conv_aa, codon_table=codon_table)
        index_bool = numpy.array([ pco in conv_cdn for pco in codon_order ])
        biased_cdn_index0 = numpy.argwhere(index_bool==True)
        biased_cdn_index0 = biased_cdn_index0.reshape(biased_cdn_index0.shape[0])
        conv_cdn_iter = itertools.product(row_index, biased_cdn_index0)
        biased_cdn_index1 = [ cci for cci in conv_cdn_iter if not ((cci[0] in biased_cdn_index0)&(cci[1] in biased_cdn_index0)) ]
        biased_cdn_index = biased_cdn_index + biased_cdn_index1
        num_conv_codons += biased_cdn_index0.shape[0]
    biased_cdn_index = numpy.array(biased_cdn_index)
    return biased_cdn_index

def get_biased_amino_acids(convergent_amino_acids, codon_table):
    if convergent_amino_acids.startswith('random'):
        amino_acids = numpy.array(list(set([ item[0] for item in codon_table if item[0] is not '*' ])))
        num_random_aa = int(convergent_amino_acids.replace('random',''))
        aa_index = numpy.random.choice(a=numpy.arange(amino_acids.shape[0]), size=num_random_aa, replace=False)
        biased_aas = amino_acids[aa_index]
    else:
        biased_aas = numpy.array(list(convergent_amino_acids))
    return biased_aas

def get_biased_codon_index(biased_aas, codon_table, codon_order):
    biased_cdn_index = list()
    for conv_aa in biased_aas:
        conv_cdn = get_codons(amino_acids=conv_aa, codon_table=codon_table)
        index_bool = numpy.array([ pco in conv_cdn for pco in codon_order ])
        biased_cdn_index0 = numpy.argwhere(index_bool==True).tolist()
        biased_cdn_index = biased_cdn_index + biased_cdn_index0
    biased_cdn_index = numpy.array(biased_cdn_index)
    return biased_cdn_index

def get_synonymous_codon_substitution_index(g, codon_order):
    amino_acids = numpy.array(list(set([ item[0] for item in g['codon_table'] if item[0] is not '*' ])))
    num_codon = codon_order.shape[0]
    num_syn_codons = 0
    cdn_index = list()
    for aa in amino_acids:
        codons = get_codons(amino_acids=aa, codon_table=g['codon_table'])
        if len(codons)==1:
            continue
        index_bool = numpy.array([ pco in codons for pco in codon_order ])
        cdn_index0 = numpy.argwhere(index_bool==True)
        cdn_index0 = cdn_index0.reshape(cdn_index0.shape[0])
        cdn_index1 = list(itertools.permutations(cdn_index0, 2))
        cdn_index = cdn_index + cdn_index1
        num_syn_codons += cdn_index0.shape[0]
    cdn_index = numpy.array(cdn_index)
    txt = '# of all synonymous codon substitutions = {}/{}'
    print(txt.format(cdn_index.shape[0], num_codon*(num_codon-1)), flush=True)
    return cdn_index

def get_nonsynonymous_codon_substitution_index(all_syn_cdn_index):
    pyvolve_codon_order = get_pyvolve_codon_order()
    num_codon = pyvolve_codon_order.shape[0]
    row_index = numpy.arange(num_codon)
    cdn_index = numpy.array(list(itertools.permutations(row_index, 2)))
    nsy_index_list = list()
    for ci in numpy.arange(cdn_index.shape[0]):
        flag = True
        for si in numpy.arange(all_syn_cdn_index.shape[0]):
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
    for ci in numpy.arange(cdn_index.shape[0]):
        row = cdn_index[ci,0]
        col = cdn_index[ci,1]
        total += mat[row,col]
    return total

def rescale_substitution_matrix(mat, target_index, scaling_factor):
    mat = copy.copy(mat)
    diag_bool = numpy.eye(mat.shape[0], dtype=bool)
    mat[diag_bool] = 0
    sum_before = mat.sum()
    for i in numpy.arange(target_index.shape[0]):
        mat[target_index[i,0],target_index[i,1]] *= scaling_factor
    sum_after = mat.sum()
    mat = mat / sum_after * sum_before
    mat[diag_bool] = -mat.sum(axis=1)
    return mat

def generate_Q_matrix(eq_freq, omega, all_nsy_cdn_index, all_syn_cdn_index):
    all_cdn_index = numpy.concatenate([all_syn_cdn_index, all_nsy_cdn_index])
    cmp = {'omega':omega, 'k_ti':1, 'k_tv':1} # background_omega have to be readjusted.
    model = pyvolve.Model(model_type='ECMunrest', name='placeholder', parameters=cmp, state_freqs=eq_freq)
    mat = model.matrix
    dnds = get_total_Q(mat, all_nsy_cdn_index) / get_total_Q(mat, all_syn_cdn_index)
    mat = rescale_substitution_matrix(mat, all_nsy_cdn_index, scaling_factor=omega/dnds)
    return mat

def bias_eq_freq(eq_freq, biased_cdn_index, convergence_intensity_factor):
    eq_freq = numpy.array(eq_freq)
    eq_freq[biased_cdn_index] *= convergence_intensity_factor
    eq_freq /= eq_freq.sum()
    return eq_freq

def get_total_biased_Q(mat, biased_aas, codon_table, codon_order):
    biased_nsy_cdn_index = get_biased_nonsynonymous_substitution_index(biased_aas, codon_table, codon_order)
    total_biased_Q = 0
    for i in numpy.arange(biased_nsy_cdn_index.shape[0]):
        total_biased_Q += mat[biased_nsy_cdn_index[i,0],biased_nsy_cdn_index[i,1]]
    return total_biased_Q

def apply_percent_biased_sub(mat, percent_biased_sub, target_index, biased_aas, codon_table, codon_orders,
                             all_nsy_cdn_index, all_syn_cdn_index, foreground_omega):
    mat = copy.copy(mat)
    diag_bool = numpy.eye(mat.shape[0], dtype=bool)
    mat[diag_bool] = 0
    total_biased_Q_before = get_total_biased_Q(mat, biased_aas, codon_table, codon_orders)
    total_nsy_Q_before = get_total_Q(mat, all_nsy_cdn_index)
    scaling_factor = total_nsy_Q_before / total_biased_Q_before / (1-(percent_biased_sub/100))
    mat = rescale_substitution_matrix(mat, target_index, scaling_factor)
    dnds = get_total_Q(mat, all_nsy_cdn_index) / get_total_Q(mat, all_syn_cdn_index)
    omega_scaling_factor = foreground_omega/dnds
    mat = rescale_substitution_matrix(mat, all_nsy_cdn_index, omega_scaling_factor)
    return mat

def main_simulate(g, Q_method='csubst'):
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['genetic_code'])
    g = parser_misc.generate_intermediate_files(g)
    g = parser_misc.read_input(g)
    g = foreground.get_foreground_branch(g)
    num_fl = foreground.get_num_foreground_lineages(tree=g['tree'])
    all_syn_cdn_index = get_synonymous_codon_substitution_index(g, get_pyvolve_codon_order())
    all_nsy_cdn_index = get_nonsynonymous_codon_substitution_index(all_syn_cdn_index)
    if (g['num_simulated_site']==-1):
        g['num_simulated_site'] = g['num_input_site']
    if g['optimized_branch_length']:
        g['tree'] = scale_tree(tree=g['tree'], scaling_factor=g['tree_scaling_factor'])
        newick_txt = get_pyvolve_tree(tree=g['tree'])
        plot_branch_category(g['tree'], file_name='simulate_branch_category.pdf')
    else:
        g['rooted_tree'] = scale_tree(tree=g['rooted_tree'], scaling_factor=g['tree_scaling_factor'])
        g['rooted_tree'] = foreground.initialize_foreground_annotation(tree=g['rooted_tree'])
        g['rooted_tree'],_,_ = foreground.annotate_foreground_branch(g['rooted_tree'], g['fg_df'], g['fg_stem_only'])
        newick_txt = get_pyvolve_tree(tree=g['rooted_tree'])
        plot_branch_category(g['rooted_tree'], file_name='simulate_branch_category.pdf')
    tree = pyvolve.read_tree(tree=newick_txt)
    f = pyvolve.ReadFrequencies('codon', file=g['alignment_file'])
    eq_freq = f.compute_frequencies()
    pyvolve_codon_orders = get_pyvolve_codon_order()
    if (Q_method=='csubst'):
        matrix_file = 'substitution_matrix/ECMunrest.dat'
        g['exchangeability_matrix'] = parser_misc.read_exchangeability_matrix(matrix_file, pyvolve_codon_orders)
        unscaled_mat = parser_misc.exchangeability2Q(g['exchangeability_matrix'], eq_freq, g['float_type'])
        dnds = get_total_Q(unscaled_mat, all_nsy_cdn_index) / get_total_Q(unscaled_mat, all_syn_cdn_index)
        scaling_factor = g['background_omega']/dnds
        background_Q = rescale_substitution_matrix(unscaled_mat, all_nsy_cdn_index, scaling_factor)
    elif (Q_method=='pyvolve'):
        background_Q = generate_Q_matrix(eq_freq, g['background_omega'], all_nsy_cdn_index, all_syn_cdn_index)
    model_names = ['root',] + [ 'm'+str(i+1) for i in range(num_fl) ]
    if g['foreground'] is None:
        num_convergent_site = 0
    else:
        num_convergent_site = int(g['num_simulated_site'] * g['percent_convergent_site'] / 100)
    num_no_conv_site = int(g['num_simulated_site'] - num_convergent_site)
    txt = '{:,} out of {:,} sites ({:.1f}%) will evolve convergently in the foreground lineages.'
    print(txt.format(num_convergent_site, g['num_simulated_site'], g['percent_convergent_site']))
    num_partition = num_convergent_site + 1 if (g['percent_convergent_site']!=100) else num_convergent_site
    partitions = list()
    biased_substitution_fractions = list()
    current_site = 0
    for partition_index in numpy.arange(num_partition):
        is_convergent_partition = (partition_index<num_convergent_site)
        size = num_no_conv_site if not is_convergent_partition else 1
        prev_site = current_site
        current_site += size
        if is_convergent_partition:
            biased_aas = get_biased_amino_acids(g['convergent_amino_acids'], g['codon_table'])
            print('Codon site {}; Biased amino acids = {}; '.format(current_site, ''.join(biased_aas)), end='')
            biased_nsy_sub_index = get_biased_nonsynonymous_substitution_index(biased_aas, g['codon_table'], pyvolve_codon_orders)
            biased_Q = apply_percent_biased_sub(mat=background_Q,
                                                percent_biased_sub=g['percent_biased_sub'],
                                                target_index=biased_nsy_sub_index,
                                                biased_aas=biased_aas,
                                                codon_table=g['codon_table'],
                                                codon_orders=pyvolve_codon_orders,
                                                all_nsy_cdn_index=all_nsy_cdn_index,
                                                all_syn_cdn_index=all_syn_cdn_index,
                                                foreground_omega=g['foreground_omega'],
                                                )
            total_nsy_Q = get_total_Q(biased_Q, all_nsy_cdn_index)
            total_biased_Q = get_total_biased_Q(biased_Q, biased_aas, g['codon_table'], pyvolve_codon_orders)
            fraction_biased_Q = total_biased_Q / total_nsy_Q
            bg_total_nsy_Q = get_total_Q(background_Q, all_nsy_cdn_index)
            bg_total_biased_Q = get_total_biased_Q(background_Q, biased_aas, g['codon_table'], pyvolve_codon_orders)
            bg_fraction_biased_Q = bg_total_biased_Q / bg_total_nsy_Q
            txt = 'Total in Q toward the codons before and after the bias introduction = ' \
                  '{:,.1f}% ({:,.1f}/{:,.1f}) and {:,.1f}% ({:,.1f}/{:,.1f})'
            print(txt.format(bg_fraction_biased_Q*100, bg_total_biased_Q, bg_total_nsy_Q,
                             fraction_biased_Q*100, total_biased_Q, total_nsy_Q))
            biased_substitution_fractions.append(fraction_biased_Q)
        else:
            print('Codon site {}-{}; No convergent amino acid'.format(prev_site+1, current_site))
        models = list()
        for model_name in model_names:
            is_nonroot_model = (model_name!='root')
            if (is_nonroot_model & is_convergent_partition):
                q_matrix = copy.copy(biased_Q)
            else:
                q_matrix = copy.copy(background_Q)
            with suppress_stdout_stderr():
                model = pyvolve.Model(model_type='custom', name=model_name, parameters={'matrix':q_matrix})
            models.append(model)
        partition = pyvolve.Partition(models=models, size=size,  root_model_name='root')
        partitions.append(partition)
    mean_biased_substitution_fraction = numpy.array(biased_substitution_fractions).mean() if len(biased_substitution_fractions) else 0
    txt = '{:,.2f}% of substitutions in {} sites in the foreground branches are ' \
          'expected to result from the introduced bias in Q matrix.'
    fraction_convergent_site = num_convergent_site / g['num_simulated_site']
    print(txt.format(mean_biased_substitution_fraction*fraction_convergent_site*100, g['num_simulated_site']))
    txt = '{:,.2f}% of substitutions in {} convergent sites in the foreground branches are ' \
          'expected to result from the introduced bias in Q matrix.'
    print(txt.format(mean_biased_substitution_fraction*100, num_convergent_site))
    evolver = pyvolve.Evolver(partitions=partitions, tree=tree)
    evolver(
        ratefile='simulate_ratefile.txt',
        infofile='simulate_infofile.txt',
        seqfile='simulate.fa',
        write_anc=False
    )

    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]
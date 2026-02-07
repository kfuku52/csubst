import pyvolve
import numpy

import copy
import itertools
import os
import re

from csubst import foreground
from csubst import genetic_code
from csubst import parser_misc
from csubst import tree
from csubst import ete

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
        newick_txt = re.sub(':'+sub_from+',', ':'+sub_to+',', newick_txt)
        newick_txt = re.sub(':'+sub_from+'\)', ':'+sub_to+')', newick_txt)
    for node in tree.traverse():
        node.dist = ete.get_prop(node, 'dist2')
        ete.del_prop(node, 'dist2')
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
        amino_acids = numpy.array(list(set([ item[0] for item in codon_table if item[0]!='*' ])))
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
    amino_acids = numpy.array(list(set([ item[0] for item in g['codon_table'] if item[0]!='*' ])))
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

def evolve_convergent_partitions(g):
    num_fl = foreground.get_num_foreground_lineages(tree=g['tree'], trait_name=g['trait_name'])
    model_names = ['root',] + [ 'm'+str(i+1) for i in range(num_fl) ]
    num_convergent_partition = g['num_convergent_site']
    convergent_partitions = list()
    biased_substitution_fractions = list()
    current_site = 0
    for partition_index in numpy.arange(num_convergent_partition):
        current_site += 1
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
                q_matrix = copy.copy(biased_Q)
            else:
                q_matrix = copy.copy(g['background_Q'])
            with suppress_stdout_stderr():
                model = pyvolve.Model(model_type='custom', name=model_name, parameters={'matrix':q_matrix})
            models.append(model)
        partition = pyvolve.Partition(models=models, size=1,  root_model_name='root')
        convergent_partitions.append(partition)
    if len(biased_substitution_fractions):
        mean_biased_substitution_fraction = numpy.array(biased_substitution_fractions).mean()
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
    evolver(
        ratefile='tmp.csubst.simulate_convergent_ratefile.txt',
        infofile='tmp.csubst.simulate_convergent_infofile.txt',
        seqfile='tmp.csubst.simulate_convergent.fa',
        write_anc=False
    )

def evolve_nonconvergent_partition(g):
    if (g['num_convergent_site']==0):
        site_start = 1
    else:
        site_start = g['num_simulated_site'] - g['num_convergent_site'] + 1
    site_end = g['num_simulated_site']
    print('Codon site {}-{}; Non-convergent codons'.format(site_start, site_end))
    num_nonconvergent_site = g['num_simulated_site'] - g['num_convergent_site']
    q_matrix = copy.copy(g['background_Q'])
    with suppress_stdout_stderr():
        model = pyvolve.Model(model_type='custom', name='root', parameters={'matrix':q_matrix})
    partition = pyvolve.Partition(models=model, size=num_nonconvergent_site)
    evolver = pyvolve.Evolver(partitions=partition, tree=g['background_tree'])
    evolver(
        ratefile='tmp.csubst.simulate_nonconvergent_ratefile.txt',
        infofile='tmp.csubst.simulate_nonconvergent_infofile.txt',
        seqfile='tmp.csubst.simulate_nonconvergent.fa',
        write_anc=False
    )

def get_pyvolve_tree(tree, foreground_scaling_factor, trait_name):
    if (foreground_scaling_factor!=1):
        print('Foreground branches are rescaled by {}.'.format(foreground_scaling_factor))
    for node in tree.traverse():
        if ete.get_prop(node, 'is_fg_' + trait_name, False):
            node.dist *= foreground_scaling_factor
    newick_txt = get_pyvolve_newick(tree=tree, trait_name=trait_name)
    pyvolve_tree = pyvolve.read_tree(tree=newick_txt)
    return pyvolve_tree

def get_background_Q(g, Q_method):
    if (Q_method=='csubst'):
        matrix_file = 'substitution_matrix/ECMunrest.dat'
        g['exchangeability_matrix'] = parser_misc.read_exchangeability_matrix(matrix_file, g['pyvolve_codon_orders'])
        unscaled_mat = parser_misc.exchangeability2Q(g['exchangeability_matrix'], g['eq_freq'])
        dnds = get_total_Q(unscaled_mat, g['all_nsy_cdn_index']) / get_total_Q(unscaled_mat, g['all_syn_cdn_index'])
        scaling_factor = g['background_omega']/dnds
        background_Q = rescale_substitution_matrix(unscaled_mat, g['all_nsy_cdn_index'], scaling_factor)
    elif (Q_method=='pyvolve'):
        background_Q = generate_Q_matrix(g['eq_freq'], g['background_omega'], g['all_nsy_cdn_index'], g['all_syn_cdn_index'])
    return background_Q

def concatenate_alignment(in1, in2, out):
    seqs = dict()
    with open(in1, 'r') as f:
        txt_in1 = f.read()
    for line in txt_in1.split('\n'):
        if line.startswith('>'):
            current_name = line
            seqs[current_name] = ''
        else:
            seqs[current_name] += line
    with open(in2, 'r') as f:
        txt_in2 = f.read()
    for line in txt_in2.split('\n'):
        if line.startswith('>'):
            current_name = line
        else:
            seqs[current_name] += line
    with open(out, 'w') as f:
        for key in seqs.keys():
            txt = key+'\n'+seqs[key]+'\n'
            f.write(txt)
    return None

def main_simulate(g, Q_method='csubst'):
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['genetic_code'])
    g = tree.read_treefile(g)
    g = parser_misc.generate_intermediate_files(g, force_notree_run=True)
    g = parser_misc.annotate_tree(g, ignore_tree_inconsistency=True)
    g = parser_misc.read_input(g)
    g = foreground.get_foreground_branch(g, simulate=True)
    g['trait_name'] = 'PLACEHOLDER'
    g['all_syn_cdn_index'] = get_synonymous_codon_substitution_index(g, get_pyvolve_codon_order())
    g['all_nsy_cdn_index'] = get_nonsynonymous_codon_substitution_index(g['all_syn_cdn_index'])
    if (g['num_simulated_site']==-1):
        g['num_simulated_site'] = g['num_input_site']
    if g['optimized_branch_length']:
        g['tree'] = g['tree']
    else:
        g['tree2'] = g['tree']
        g['tree'] = g['rooted_tree']
    g['tree'] = scale_tree(tree=g['tree'], scaling_factor=g['tree_scaling_factor'])
    g['tree'] = foreground.dummy_foreground_annotation(tree=g['tree'], trait_name=g['trait_name'])
    tree.plot_branch_category(g, file_base='simulate_branch_id', label='all')
    g['background_tree'] = get_pyvolve_tree(g['tree'], foreground_scaling_factor=1, trait_name=g['trait_name'])
    g['foreground_tree'] = get_pyvolve_tree(g['tree'], foreground_scaling_factor=g['foreground_scaling_factor'], trait_name=g['trait_name'])
    f = pyvolve.ReadFrequencies('codon', file=g['alignment_file'])
    g['eq_freq'] = f.compute_frequencies()
    g['pyvolve_codon_orders'] = get_pyvolve_codon_order()
    g['background_Q'] = get_background_Q(g, Q_method)
    if g['foreground'] is None:
        print('--foreground was not provided. --percent_convergent_site will be set to 0.')
        g['num_convergent_site'] = 0
    else:
        g['num_convergent_site'] = int(g['num_simulated_site'] * g['percent_convergent_site'] / 100)
    g['num_no_convergent_site'] = int(g['num_simulated_site'] - g['num_convergent_site'])
    txt = '{:,} out of {:,} sites ({:.1f}%) will evolve convergently in the foreground lineages.'
    print(txt.format(g['num_convergent_site'], g['num_simulated_site'], g['percent_convergent_site']))
    if (g['percent_convergent_site']>0):
        evolve_convergent_partitions(g)
    if (g['percent_convergent_site']<100):
        evolve_nonconvergent_partition(g)
    file_conv = 'tmp.csubst.simulate_convergent.fa'
    file_noconv = 'tmp.csubst.simulate_nonconvergent.fa'
    file_out = 'simulate.fa'
    if (os.path.exists(file_conv))&(not os.path.exists(file_noconv)):
        os.rename(file_conv, file_out)
    elif (not os.path.exists(file_conv))&(os.path.exists(file_noconv)):
        os.rename(file_noconv, file_out)
    else:
        concatenate_alignment(in1=file_conv, in2=file_noconv, out=file_out)

    tmp_files = [f for f in os.listdir() if f.startswith('tmp.csubst.')]
    _ = [os.remove(ts) for ts in tmp_files]

import pyvolve
import numpy
import itertools
import re

from csubst import parser_misc
from csubst import genetic_code
from csubst import foreground

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
    return newick_txt

def get_pyvolve_codon_order():
    codons = [ ''.join(a) for a in itertools.product('ACGT', repeat=3) ]
    stops = ['TAA', 'TAG', 'TGA']
    codons = [ c for c in codons if c not in stops ]
    codons = numpy.array(codons)
    return codons

def get_convergent_codons(conv_aa, codon_table):
    conv_cdn = list()
    for ct in codon_table:
        if ct[0] in conv_aa:
            conv_cdn.append(ct[1])
    conv_cdn = numpy.array(conv_cdn)
    return conv_cdn

def get_convergent_codon_index(g):
    if g['convergent_amino_acids'].startswith('random'):
        amino_acids = numpy.array(list(set([ item[0] for item in g['codon_table'] ])))
        num_random_aa = int(g['convergent_amino_acids'].replace('random',''))
        aa_index = numpy.random.choice(a=numpy.arange(amino_acids.shape[0]), size=num_random_aa)
        conv_aa = amino_acids[aa_index]
    else:
        conv_aa = numpy.array(list(g['convergent_amino_acids']))
    print('Convergent amino acids:', conv_aa, flush=True)
    conv_cdn = get_convergent_codons(conv_aa=conv_aa, codon_table=g['codon_table'])
    print('Convergent codons:', conv_cdn, flush=True)
    pyvolve_codon_order = get_pyvolve_codon_order()
    index_bool = numpy.array([ pco in conv_cdn for pco in pyvolve_codon_order ])
    conv_cdn_index = numpy.argwhere(index_bool==True)
    return conv_cdn_index

def rescale_substitution_matrix(mat, conv_cdn_index, scaling_factor):
    diag_bool = numpy.eye(mat.shape[0], dtype=bool)
    mat[diag_bool] = 0
    sum_before = mat.sum()
    mat[:,conv_cdn_index] *= scaling_factor
    sum_after = mat.sum()
    mat = mat / sum_after * sum_before
    mat[diag_bool] = -mat.sum(axis=1)
    return mat

def main_simulate(g):
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['ncbi_codon_table'])
    g = parser_misc.read_input(g)
    g = foreground.get_foreground_branch(g)
    num_fl = foreground.get_num_foreground_lineages(tree=g['tree'])
    conv_cdn_index = get_convergent_codon_index(g)
    g['tree'] = scale_tree(tree=g['tree'], scaling_factor=g['tree_scaling_factor'])
    newick_txt = get_pyvolve_tree(tree=g['tree'])
    tree = pyvolve.read_tree(tree=newick_txt)
    cmp = {'omega':g['omega'], 'k_ti':1, 'k_tv':1} # TODO: Read IQ-TREE inputs
    f = pyvolve.ReadFrequencies('codon', file=g['aln_file'])
    sf = f.compute_frequencies()
    model_names = ['root',] + [ 'm'+str(i+1) for i in range(num_fl) ]
    models = list()
    for model_name in model_names:
        if (model_name=='root'):
            model = pyvolve.Model(model_type='ECMrest', name=model_name, parameters=cmp, state_freqs=sf)
        else:
            model_tmp = pyvolve.Model(model_type='ECMrest', name=model_name, parameters=cmp, state_freqs=sf)
            mat = model_tmp.matrix
            mat = rescale_substitution_matrix(mat, conv_cdn_index, scaling_factor=g['convergence_intensity_factor'])
            cmp2 = {'matrix':mat}
            model = pyvolve.Model(model_type='custom', name=model_name, parameters=cmp2)
        models.append(model)
    del model,model_tmp,cmp,cmp2
    if g['num_simulated_site']==0:
        num_simulated_site = g['num_input_site']
    else:
        num_simulated_site = g['num_simulated_site']
    partition = pyvolve.Partition(models=models, size=num_simulated_site,  root_model_name='root')
    evolver = pyvolve.Evolver(partition=partition, tree=tree)
    evolver(
        ratefile='simulate_ratefile.txt',
        infofile='simulate_infofile.txt',
        seqfile='simulate.fa',
        write_anc=False
    )
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

def get_codons(amino_acids, codon_table):
    cdn = list()
    for ct in codon_table:
        if ct[0] in amino_acids:
            cdn.append(ct[1])
    cdn = numpy.array(cdn)
    return cdn

def get_convergent_nsyonymous_codon_substitution_index(g):
    if g['convergent_amino_acids'].startswith('random'):
        amino_acids = numpy.array(list(set([ item[0] for item in g['codon_table'] if item[0] is not '*' ])))
        num_random_aa = int(g['convergent_amino_acids'].replace('random',''))
        aa_index = numpy.random.choice(a=numpy.arange(amino_acids.shape[0]), size=num_random_aa, replace=False)
        conv_aas = amino_acids[aa_index]
    else:
        conv_aas = numpy.array(list(g['convergent_amino_acids']))
    pyvolve_codon_order = get_pyvolve_codon_order()
    num_codon = pyvolve_codon_order.shape[0]
    row_index = numpy.arange(num_codon)
    num_conv_codons = 0
    conv_cdn_index = list()
    for conv_aa in conv_aas:
        conv_cdn = get_codons(amino_acids=conv_aa, codon_table=g['codon_table'])
        index_bool = numpy.array([ pco in conv_cdn for pco in pyvolve_codon_order ])
        conv_cdn_index0 = numpy.argwhere(index_bool==True)
        conv_cdn_index0 = conv_cdn_index0.reshape(conv_cdn_index0.shape[0])
        conv_cdn_iter = itertools.product(row_index, conv_cdn_index0)
        conv_cdn_index1 = [ cci for cci in conv_cdn_iter if not ((cci[0] in conv_cdn_index0)&(cci[1] in conv_cdn_index0)) ]
        conv_cdn_index = conv_cdn_index + conv_cdn_index1
        num_conv_codons += conv_cdn_index0.shape[0]
    conv_cdn_index = numpy.array(conv_cdn_index)
    txt = 'Convergent amino acids = {}; # of codons = {}; # of nsyonymous codon substitutions = {}/{}'
    print(txt.format(conv_aas, num_conv_codons, conv_cdn_index.shape[0], num_codon*(num_codon-1)), flush=True)
    return conv_cdn_index

def get_synonymous_codon_substitution_index(g):
    amino_acids = numpy.array(list(set([ item[0] for item in g['codon_table'] if item[0] is not '*' ])))
    pyvolve_codon_order = get_pyvolve_codon_order()
    num_codon = pyvolve_codon_order.shape[0]
    num_syn_codons = 0
    cdn_index = list()
    for aa in amino_acids:
        codons = get_codons(amino_acids=aa, codon_table=g['codon_table'])
        if len(codons)==1:
            continue
        index_bool = numpy.array([ pco in codons for pco in pyvolve_codon_order ])
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

def get_total_freq(mat, cdn_index):
    total = 0
    for ci in numpy.arange(cdn_index.shape[0]):
        row = cdn_index[ci,0]
        col = cdn_index[ci,1]
        total += mat[row,col]
    return total

def rescale_substitution_matrix(mat, target_index, all_index, scaling_factor):
    diag_bool = numpy.eye(mat.shape[0], dtype=bool)
    mat[diag_bool] = 0
    sum_before = get_total_freq(mat, all_index)
    for i in numpy.arange(target_index.shape[0]):
        mat[target_index[i,0],target_index[i,1]] *= scaling_factor
    sum_after = get_total_freq(mat, all_index)
    for ai in numpy.arange(all_index.shape[0]):
        row = all_index[ai,0]
        col = all_index[ai,1]
        mat[row,col] = mat[row,col] / sum_after * sum_before
    mat[diag_bool] = -mat.sum(axis=1)
    return mat

def main_simulate(g):
    g['codon_table'] = genetic_code.get_codon_table(ncbi_id=g['ncbi_codon_table'])
    g = parser_misc.read_input(g)
    g = foreground.get_foreground_branch(g)
    num_fl = foreground.get_num_foreground_lineages(tree=g['tree'])
    all_syn_cdn_index = get_synonymous_codon_substitution_index(g)
    all_nsy_cdn_index = get_nonsynonymous_codon_substitution_index(all_syn_cdn_index)
    conv_nsy_cdn_index = get_convergent_nsyonymous_codon_substitution_index(g)
    #cdn_index = numpy.concatenate(conv_nsy_cdn_index, all_syn_cdn_index) # Rescale both to maintain dN/dS
    g['tree'] = scale_tree(tree=g['tree'], scaling_factor=g['tree_scaling_factor'])
    newick_txt = get_pyvolve_tree(tree=g['tree'])
    tree = pyvolve.read_tree(tree=newick_txt)
    cmp = {'omega':g['simulated_omega'], 'k_ti':1, 'k_tv':1} # TODO: Read IQ-TREE inputs
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
            dnds = get_total_freq(mat, all_nsy_cdn_index) / get_total_freq(mat, all_syn_cdn_index)
            print('N/S freq before rescaling convergent nonsynonymous changes = {}'.format(dnds))
            mat = rescale_substitution_matrix(mat, conv_nsy_cdn_index, all_nsy_cdn_index,
                                              scaling_factor=g['convergence_intensity_factor'])
            dnds = get_total_freq(mat, all_nsy_cdn_index) / get_total_freq(mat, all_syn_cdn_index)
            print('N/S freq after rescaling convergent nonsynonymous changes = {}'.format(dnds))
            cmp2 = {'matrix':mat}
            model = pyvolve.Model(model_type='custom', name=model_name, parameters=cmp2)
        models.append(model)
    del model,model_tmp,cmp,cmp2
    if g['num_simulated_site']==-1:
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
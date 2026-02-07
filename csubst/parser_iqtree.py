import numpy
import pandas

import os
import re
import subprocess
import sys
from collections import OrderedDict
from distutils.version import LooseVersion

from csubst import genetic_code
from csubst import sequence
from csubst import tree
from csubst import ete

def _parse_iqtree_version_text(txt):
    # IQ-TREE 2 and 3 both expose a semantic version string in startup banners.
    patterns = [
        r'IQ-TREE(?: multicore)? version\s+([0-9]+(?:\.[0-9]+)*)',
        r'IQ-TREE\s+([0-9]+(?:\.[0-9]+)*)',
    ]
    for pattern in patterns:
        m = re.search(pattern, txt)
        if m is not None:
            version = m.group(1)
            major = int(version.split('.')[0])
            return version, major
    return None, None

def _read_text(path):
    with open(path) as f:
        return f.read()

def detect_iqtree_output_version(g):
    g['iqtree_output_version'] = None
    g['iqtree_output_version_major'] = None
    for key in ['path_iqtree_iqtree', 'path_iqtree_log']:
        path = g.get(key, None)
        if (path is None) or (not os.path.exists(path)):
            continue
        txt = _read_text(path)
        version, major = _parse_iqtree_version_text(txt)
        if version is not None:
            g['iqtree_output_version'] = version
            g['iqtree_output_version_major'] = major
            break
    return g

def _parse_substitution_model(iqtree_txt):
    model = None
    for line in iqtree_txt.splitlines():
        m = re.match(r'\s*Model of substitution:\s*(.+?)\s*$', line)
        if m is not None:
            model = m.group(1)
            break
    return model

def _parse_equilibrium_frequency(iqtree_txt, codon_orders, parser_name, float_type):
    # IQ-TREE 2 and 3 share similar labels but differ slightly in spacing/number formatting.
    pattern_iqtree2 = r'pi\(([A-Z]+)\)\s*=\s*([0-9.]+)(?![eE][+-]?[0-9]+)'
    pattern_iqtree3 = r'pi\(\s*([A-Z]+)\s*\)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    if parser_name == 'iqtree3':
        patterns = [pattern_iqtree3, pattern_iqtree2]
    else:
        patterns = [pattern_iqtree2, pattern_iqtree3]
    eq_map = dict()
    for pattern in patterns:
        for codon,freq in re.findall(pattern, iqtree_txt):
            codon = codon.upper().replace('U', 'T')
            try:
                eq_map[codon] = float(freq)
            except ValueError:
                continue
    values = list()
    missing_codons = list()
    for codon in codon_orders:
        freq = eq_map.get(codon, numpy.nan)
        if not numpy.isfinite(freq):
            missing_codons.append(codon)
        values.append(freq)
    if len(missing_codons)>0:
        txt = 'Failed to parse equilibrium frequencies from {} output. '
        txt += 'Missing codon(s): {}'
        missing_txt = ','.join(missing_codons[0:10])
        if len(missing_codons)>10:
            missing_txt += ',...'
        raise AssertionError(txt.format(parser_name, missing_txt))
    equilibrium_frequency = numpy.array(values, dtype=float_type)
    total = equilibrium_frequency.sum()
    assert total>0, 'Failed to parse equilibrium frequencies: sum should be positive.'
    equilibrium_frequency /= total
    return equilibrium_frequency

def _parse_float_from_log_line(line, label):
    pattern = r'\s*{}\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'.format(re.escape(label))
    m = re.match(pattern, line)
    if m is not None:
        return float(m.group(1))
    return None

def _estimate_empirical_codon_frequency_from_alignment(alignment_file, codon_orders, float_type):
    seq_dict = sequence.read_fasta(alignment_file)
    codon_orders = numpy.array(codon_orders)
    counts = numpy.zeros(shape=(codon_orders.shape[0],), dtype=float_type)
    for seq in seq_dict.values():
        seq = seq.upper().replace('U', 'T')
        assert len(seq)%3==0, 'Sequence length is not multiple of 3 in alignment file.'
        for s in numpy.arange(0, len(seq), 3):
            codon = seq[s:s+3]
            codon_idx = sequence.get_state_index(codon, codon_orders, genetic_code.ambiguous_table)
            if len(codon_idx)==0:
                continue
            counts[codon_idx] += 1 / len(codon_idx)
    total = counts.sum()
    assert total>0, 'Failed to estimate codon frequencies from alignment file.'
    counts /= total
    return counts

def check_iqtree_dependency(g):
    test_iqtree = subprocess.run([g['iqtree_exe'], '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert (test_iqtree.returncode==0), "iqtree PATH cannot be found: "+g['iqtree_exe']
    txt = test_iqtree.stdout.decode('utf8') + '\n' + test_iqtree.stderr.decode('utf8')
    version_iqtree,major_iqtree = _parse_iqtree_version_text(txt)
    if version_iqtree is None:
        version_iqtree = txt.split('\n')[0].strip()
    g['iqtree_version'] = version_iqtree
    g['iqtree_version_major'] = major_iqtree
    is_satisfied_version = LooseVersion(version_iqtree) >= LooseVersion('2.0.0')
    assert is_satisfied_version, 'IQ-TREE version ({}) should be 2.0.0 or greater.'.format(version_iqtree)
    print("IQ-TREE's version: {}, PATH: {}".format(version_iqtree, g['iqtree_exe']), flush=True)
    return None

def check_intermediate_files(g):
    all_exist = True
    extensions = ['iqtree','log','rate','state','treefile']
    for ext in extensions:
        if (g['iqtree_'+ext]=='infer'):
            g['path_iqtree_'+ext] = g['alignment_file']+'.'+ext
        else:
            g['path_iqtree_'+ext] = g['iqtree_'+ext]
        if not os.path.exists(g['path_iqtree_'+ext]):
            print('Intermediate file is missing: {}'.format(g['path_iqtree_'+ext]))
            all_exist = False
    return g,all_exist

def run_iqtree_ancestral(g, force_notree_run=False):
    file_tree = 'tmp.csubst.nwk'
    tree.write_tree(g['rooted_tree'], outfile=file_tree, add_numerical_label=False)
    is_consistent = tree.is_consistent_tree_and_aln(g=g)
    if is_consistent:
        command = [g['iqtree_exe'], '-s', g['alignment_file'], '-te', file_tree,
                   '-m', g['iqtree_model'], '--seqtype', 'CODON'+str(g['genetic_code']),
                   '--threads-max', str(g['threads']), '-T', 'AUTO', '--ancestral', '--rate', '--redo']
    else:
        sys.stderr.write('--rooted_tree and --alignment_file are not consistent.\n')
        if force_notree_run:
            sys.stderr.write('Running iqtree without tree input to estimate simulation parameters.\n')
            command = [g['iqtree_exe'], '-s', g['alignment_file'],
                       '-m', g['iqtree_model'], '--seqtype', 'CODON'+str(g['genetic_code']),
                       '--threads-max', str(g['threads']), '-T', 'AUTO', '--ancestral', '--rate', '--redo']
        else:
            sys.stderr.write('Exiting.\n')
            sys.exit(1)
    run_iqtree = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr)
    assert (run_iqtree.returncode==0), "IQ-TREE did not finish safely: {}".format(run_iqtree.stdout.decode('utf8'))
    if os.path.exists(g['alignment_file']+'.ckp.gz'):
        os.remove(g['alignment_file']+'.ckp.gz')
    os.remove(file_tree)
    return None

def read_state(g):
    print('Reading the state file:', g['iqtree_state'])
    state_table = pandas.read_csv(g['iqtree_state'], sep="\t", index_col=False, header=0, comment='#')
    g['num_input_site'] = state_table['Site'].unique().shape[0]
    g['num_input_state'] = state_table.shape[1] - 3
    g['input_state'] = state_table.columns[3:].str.replace('p_','').tolist()
    if g['num_input_state']==4:
        g['input_data_type'] = 'nuc'
    elif g['num_input_state']==20:
        g['input_data_type'] = 'pep'
    elif g['num_input_state'] > 20:
        g['input_data_type'] = 'cdn'
        g['codon_orders'] = state_table.columns[3:].str.replace('p_','').values
    if (g['input_data_type']=='cdn'):
        g['amino_acid_orders'] = sorted(list(set([ c[0] for c in g['codon_table'] if c[0]!='*' ])))
        matrix_groups = OrderedDict()
        for aa in g['amino_acid_orders']:
            matrix_groups[aa] = [ c[1] for c in g['codon_table'] if c[0]==aa ]
        g['matrix_groups'] = matrix_groups
        synonymous_indices = dict()
        for aa in matrix_groups.keys():
            synonymous_indices[aa] = []
        for i,c in enumerate(g['codon_orders']):
            for aa in matrix_groups.keys():
                if c in matrix_groups[aa]:
                    synonymous_indices[aa].append(i)
                    break
        g['synonymous_indices'] = synonymous_indices
        g['max_synonymous_size'] = max([ len(si) for si in synonymous_indices.values() ])
    print('')
    return g

def read_rate(g):
    rate_sites = pandas.read_csv(g['path_iqtree_rate'], sep='\t', header=0, comment='#')
    rate_sites = rate_sites.loc[:,'C_Rate'].values
    if rate_sites.shape[0]==0:
        rate_sites = numpy.ones(g['num_input_site'])
    return rate_sites

def read_iqtree(g, eq=True):
    iqtree_txt = _read_text(g['path_iqtree_iqtree'])
    g = detect_iqtree_output_version(g)
    if g['iqtree_output_version_major'] is None:
        # Fall back to executable version if .iqtree file lacks a version banner.
        g['iqtree_output_version'] = g.get('iqtree_version', None)
        g['iqtree_output_version_major'] = g.get('iqtree_version_major', None)
    if (g['iqtree_output_version_major'] is not None) and (g['iqtree_output_version_major']>=3):
        parser_name = 'iqtree3'
    else:
        parser_name = 'iqtree2'
    g['iqtree_parser'] = parser_name
    g['substitution_model'] = _parse_substitution_model(iqtree_txt)
    assert g['substitution_model'] is not None, 'Failed to parse substitution model from IQ-TREE output.'
    if eq:
        try:
            g['equilibrium_frequency'] = _parse_equilibrium_frequency(
                iqtree_txt=iqtree_txt,
                codon_orders=g['codon_orders'],
                parser_name=parser_name,
                float_type=g['float_type'],
            )
        except AssertionError as e:
            # IQ-TREE 3 may omit codon pi(...) entries from .iqtree outputs for codon models.
            if parser_name == 'iqtree3':
                txt = 'Could not parse codon frequencies from IQ-TREE 3 output. '
                txt += 'Estimating empirical codon frequencies from alignment: {}'
                print(txt.format(g['alignment_file']), flush=True)
                g['equilibrium_frequency'] = _estimate_empirical_codon_frequency_from_alignment(
                    alignment_file=g['alignment_file'],
                    codon_orders=g['codon_orders'],
                    float_type=g['float_type'],
                )
            else:
                raise e
    return g

def read_log(g):
    g = detect_iqtree_output_version(g)
    g['omega'] = None
    g['kappa'] = None
    g['reconstruction_codon_table'] = None
    with open(g['path_iqtree_log']) as f:
        lines = f.readlines()
    for line in lines:
        omega = _parse_float_from_log_line(line, 'Nonsynonymous/synonymous ratio (omega)')
        if omega is not None:
            g['omega'] = omega
        kappa = _parse_float_from_log_line(line, 'Transition/transversion ratio (kappa)')
        if kappa is not None:
            g['kappa'] = kappa
        rgc = re.match(r'\s*Converting to codon sequences with genetic code\s+([0-9]+)\s*\.\.\.', line)
        if rgc is not None:
            g['reconstruction_codon_table'] = int(rgc.group(1))
    return g

def mask_missing_sites(state_tensor, tree):
    for node in tree.traverse():
        if (ete.is_root(node)) | (ete.is_leaf(node)):
            continue
        nl = ete.get_prop(node, "numerical_label")
        child0_leaf_nls = numpy.array([ete.get_prop(l, "numerical_label") for l in ete.get_leaves(node.get_children()[0])], dtype=int)
        child1_leaf_nls = numpy.array([ete.get_prop(l, "numerical_label") for l in ete.get_leaves(node.get_children()[1])], dtype=int)
        sister_leaf_nls = numpy.array([ete.get_prop(l, "numerical_label") for l in ete.get_leaves(node.get_sisters()[0])], dtype=int)
        c0 = (state_tensor[child0_leaf_nls,:,:].sum(axis=(0,2))!=0) # is_child0_leaf_nonzero
        c1 = (state_tensor[child1_leaf_nls,:,:].sum(axis=(0,2))!=0) # is_child1_leaf_nonzero
        s = (state_tensor[sister_leaf_nls,:,:].sum(axis=(0,2))!=0) # is_sister_leaf_nonzero
        is_nonzero = (c0&c1)|(c0&s)|(c1&s)
        state_tensor[nl,:,:] = numpy.einsum('ij,i->ij', state_tensor[nl,:,:], is_nonzero)
    return state_tensor

def get_state_tensor(g):
    g['tree'].link_to_alignment(alignment=g['alignment_file'], alg_format='fasta')
    num_codon_alignment = int(len(ete.get_leaves(g['tree'])[0].sequence)/3)
    err_txt = 'The number of codon sites did not match between the alignment and ancestral states. ' \
              'Delete intermediate files and rerun.'
    assert num_codon_alignment==g['num_input_site'], err_txt
    num_node = len(list(g['tree'].traverse()))
    state_table = pandas.read_csv(g['path_iqtree_state'], sep="\t", index_col=False, header=0, comment='#')
    axis = [num_node, g['num_input_site'], g['num_input_state']]
    state_tensor = numpy.zeros(axis, dtype=g['float_type'])
    for node in g['tree'].traverse():
        if ete.is_root(node):
            continue
        elif ete.is_leaf(node):
            seq = node.sequence.upper()
            state_matrix = numpy.zeros([g['num_input_site'], g['num_input_state']], dtype=g['float_type'])
            if g['input_data_type']=='cdn':
                assert len(seq)%3==0, 'Sequence length is not multiple of 3. Node name = '+node.name
                for s in numpy.arange(int(len(seq)/3)):
                    codon = seq[(s*3):((s+1)*3)]
                    codon_index = sequence.get_state_index(codon, g['codon_orders'], genetic_code.ambiguous_table)
                    for ci in codon_index:
                        state_matrix[s,ci] = 1/len(codon_index)
            elif g['input_data_type']=='nuc':
                for s in numpy.arange(len(seq)):
                    nuc_index = sequence.get_state_index(seq[s], g['input_state'], genetic_code.ambiguous_table)
                    for ni in nuc_index:
                        state_matrix[s, ni] = 1/len(nuc_index)
            state_tensor[ete.get_prop(node, "numerical_label"),:,:] = state_matrix
        else: # Internal nodes
            state_matrix = state_table.loc[(state_table['Node']==node.name),:].iloc[:,3:]
            is_missing = (state_table.loc[:,'State']=='???') | (state_table.loc[:,'State']=='?')
            state_matrix.loc[is_missing,:] = 0
            if state_matrix.shape[0]==0:
                print('Node name not found in .state file:', node.name)
            else:
                state_tensor[ete.get_prop(node, "numerical_label"),:,:] = state_matrix
    state_tensor = numpy.nan_to_num(state_tensor)
    state_tensor = mask_missing_sites(state_tensor, g['tree'])
    if (g['ml_anc']):
        print('Ancestral state frequency is converted to the ML-like binary states.')
        idxmax = numpy.argmax(state_tensor, axis=2)
        state_tensor2 = numpy.zeros(state_tensor.shape, dtype=bool)
        for b in numpy.arange(state_tensor2.shape[0]):
            for s in numpy.arange(state_tensor2.shape[1]):
                if state_tensor[b,s,:].sum()!=0:
                    state_tensor2[b,s,idxmax[b,s]] = 1
        state_tensor = state_tensor2
        del state_tensor2
    return(state_tensor)

def get_input_information(g):
    g = read_state(g)
    g = read_iqtree(g)
    g = read_log(g)
    g['iqtree_rate_values'] = read_rate(g)
    return g

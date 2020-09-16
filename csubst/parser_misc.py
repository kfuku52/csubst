import numpy

def read_substitution_matrix(file):
    import pkg_resources
    from scipy.spatial.distance import squareform
    txt = pkg_resources.resource_string(__name__, file)
    arr = numpy.fromstring(txt, sep=' ')
    if (arr.shape[0]==1891): # for codon substitution matrix
        num_state = 61
    else:
        raise Error('This is not a codon substitution matrix.')
    sub_matrix = numpy.zeros(shape=(num_state,num_state))
    ind = numpy.tril_indices_from(sub_matrix, k=0)
    sub_matrix[ind] = arr
    #sub_matrix += sub_matrix.T


    codon_order = [
        'TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG',
        'TAT', 'TAC', 'TGT', 'TGC', 'TGG', 'CTT', 'CTC', 'CTA',
        'CTG', 'CCT', 'CCC', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA',
        'CAG', 'CGT', 'CGC', 'CGA', 'CGG', 'ATT', 'ATC', 'ATA',
        'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAA',
        'AAG', 'AGT', 'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA',
        'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA',
        'GAG', 'GGT', 'GGC', 'GGA', 'GGG'
    ]


    return sub_matrix
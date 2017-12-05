import os

def get_global_parameters(args):
    g = dict()
    for attr in [a for a in dir(args) if not a.startswith('_')]:
        g[attr] = getattr(args, attr)
    if not g['infile_dir'].endswith('/'):
        g['infile_dir'] = g['infile_dir'] + '/'
    if (g['aln_file'] == '') & (g['infile_type'] in ['iqtree', ]):
        files = os.listdir(g['infile_dir'])
        aln_files = list()
        for ext in ['.fasta','.fa','.fas']:
            aln_files = aln_files + [ f for f in files if f.endswith(ext) ]
        if len(aln_files)==1:
            g['aln_file'] = g['infile_dir']+aln_files[0]
            print('Alignment file found:', g['aln_file'])
        else:
            print('Alignment file not found. Use --aln option.', aln_files)
    if (g['tre_file'] == '') & (g['infile_type'] in ['iqtree', ]):
        files = os.listdir(g['infile_dir'])
        tree_files = list()
        for ext in ['.r.nwk']:
            tree_files = tree_files + [ f for f in files if f.endswith(ext) ]
        if len(tree_files)==1:
            g['tre_file'] = g['infile_dir']+tree_files[0]
            print('Tree file found:', g['tre_file'])
        else:
            print('The rooted tree file not found. Use --tre option.', tree_files)
    return g





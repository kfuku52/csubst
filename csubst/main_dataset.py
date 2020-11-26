import pkg_resources

def main_dataset(g):
    dataset_base = 'dataset/'+g['name']
    extensions = ['.fa','.nwk','.txt']
    outfiles = ['alignment.fa','tree.nwk','foreground.txt']
    for ext,outfile in zip(extensions,outfiles):
        print('Writing {} for the dataset {}'.format(outfile, g['name']))
        file_path = dataset_base+ext
        binary_obj = pkg_resources.resource_string(__name__, file_path)
        txt = str(bytes)
        with open(outfile, mode='wb') as f:
            f.write(binary_obj)

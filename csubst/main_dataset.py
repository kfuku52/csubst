import pkg_resources

def main_dataset(g):
    dataset_base = 'dataset/'+g['name']
    extensions = ['.fa','.nwk','.txt','.untrimmed_cds.fa']
    outfiles = ['alignment.fa','tree.nwk','foreground.txt','untrimmed_cds.fa']
    for ext,outfile in zip(extensions,outfiles):
        file_path = dataset_base+ext
        try:
            binary_obj = pkg_resources.resource_string(__name__, file_path)
        except:
            continue
        print('Writing {} for the dataset {}'.format(outfile, g['name']))
        with open(outfile, mode='wb') as f:
            f.write(binary_obj)

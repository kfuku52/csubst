import numpy
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

import os
import re
import urllib

from csubst import sequence

def get_top_hit_id(my_hits):
    top_hit_title = my_hits.descriptions[0].title
    top_hit_id = re.findall('\|.*\|', top_hit_title)[0]
    top_hit_id = re.sub('\|', '', top_hit_id)
    top_hit_id = re.sub('\..*', '', top_hit_id)
    return top_hit_id

def run_qblast(aa_query, num_display=10, evalue_cutoff=10):
    print('Running NCBI BLAST against UniProtKB/SwissProt. '
          'This step should finish within minutes but may take hours depending on the NCBI QBLAST server conditions.')
    my_search = NCBIWWW.qblast(program='blastp', database='swissprot', sequence=aa_query, expect=evalue_cutoff)
    my_hits = NCBIXML.read(my_search)
    my_search.close()
    if my_hits.descriptions is None:
        print('No hit found.')
        pdb_id = None
        return pdb_id
    print('Top hits (up to {:,} displayed)'.format(num_display))
    for i, description in enumerate(my_hits.descriptions):
        if i >= num_display:
            break
        print(description.title)
    top_hit_id = get_top_hit_id(my_hits)
    return top_hit_id

def get_representative_leaf(node, size='median'):
    leaves = node.get_leaves()
    leaf_seqlens = [ len(l.sequence.replace('-', '')) for l in leaves ]
    if size=='median':
        ind = numpy.argsort(leaf_seqlens)[len(leaf_seqlens) // 2]
    representative_leaf = leaves[ind]
    return representative_leaf

def is_url_valid(url):
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'
    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False

def pdb_sequence_search(g):
    from pypdb import Query
    print('')
    representative_branch_id = g['branch_ids'][0]
    for node in g['tree'].traverse():
        if (node.numerical_label==representative_branch_id):
            representative_leaf = get_representative_leaf(node, size='median')
            nlabel = representative_leaf.numerical_label
            aa_query = sequence.translate_state(nlabel=nlabel, mode='aa', g=g)
            aa_query = aa_query.replace('-', '')
            break
    pdb_id = None
    top_hit_id = None
    database_names = g['database'].split(',')
    for database_name in database_names:
        if pdb_id is not None:
            break
        print('Starting the sequence similarity search against protein structure database: {}'.format(database_name))
        if (database_name=='pdb'):
            try:
                print('MMseqs2 search against PDB: Query = {}'.format(representative_leaf.name))
                print('MMseqs2 search against PDB: Query sequence = {}'.format(aa_query))
                q = Query(aa_query, query_type='sequence', return_type='polymer_entity')
                mmseqs2_out = q.search()
                best_hit = mmseqs2_out['result_set'][0]
                best_hit_mc = best_hit['services'][0]['nodes'][0]['match_context'][0]
                print('MMseqs2 search against PDB: Best hit identifier = {}'.format(best_hit['identifier']))
                for key in best_hit_mc.keys():
                    print('MMseqs2 search against PDB: Best hit {} = {}'.format(key, best_hit_mc[key]))
                print('')
                pdb_id = re.sub('_.*', '', best_hit['identifier'])
                g['selected_database'] = 'pdb'
            except:
                print('MMseqs2 search against PDB was unsuccessful.')
                pdb_id = None
        elif (database_name=='alphafill')|(database_name=='alphafold'):
            if top_hit_id is None:
                top_hit_id = run_qblast(aa_query, num_display=10, evalue_cutoff=10)
            if (database_name=='alphafill'):
                download_url = 'https://alphafill.eu/v1/aff/'+top_hit_id
            elif (database_name=='alphafold'):
                download_url = 'https://alphafold.ebi.ac.uk/files/AF-' + top_hit_id + '-F1-model_v2.pdb'
            if (top_hit_id is None):
                print('There is no suitable QBLAST hit.')
            elif is_url_valid(url=download_url):
                alphafold_pdb = urllib.request.urlopen(download_url).read()
                if (database_name == 'alphafill'):
                    alphafold_pdb_path = os.path.basename(download_url)+'.cif'
                elif (database_name=='alphafold'):
                    alphafold_pdb_path = os.path.basename(download_url)
                with open(alphafold_pdb_path, mode='wb') as f:
                    f.write(alphafold_pdb)
                pdb_id = alphafold_pdb_path
                g['selected_database'] = database_name
            else:
                print('Download URL not found: {}'.format(download_url))
                pdb_id = None
    g['pdb'] = pdb_id
    if g['pdb'] is not None:
        print('Selected database and ID: {} and {}'.format(g['selected_database'], g['pdb']))
    else:
        txt = 'All specified databases ({}) were searched but no suitable structure was found. '
        txt += 'Continuing without protein structure.'
        print(txt.format(g['database']))
    return g

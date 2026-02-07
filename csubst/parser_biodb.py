import numpy
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

import json
import os
import re
import requests
import time
import urllib

from csubst import sequence
from csubst import parser_pymol
from csubst import ete

def get_top_hit_ids(my_hits):
    if len(my_hits.descriptions)==0:
        return []
    top_hit_ids = []
    for i in range(len(my_hits.descriptions)):
        top_hit_title = my_hits.descriptions[i].title
        top_hit_id = re.findall(r'\|.*\|', top_hit_title)[0]
        top_hit_id = re.sub(r'\|', '', top_hit_id)
        top_hit_id = re.sub(r'\..*', '', top_hit_id)
        top_hit_ids.append(top_hit_id)
    return top_hit_ids

def run_qblast(aa_query, num_display=10, evalue_cutoff=10):
    print('Running NCBI BLAST against UniProtKB/SwissProt. '
          'This step should finish within minutes but may take hours depending on the NCBI QBLAST server conditions.')
    start = time.time()
    my_search = NCBIWWW.qblast(program='blastp', database='swissprot', sequence=aa_query, expect=evalue_cutoff)
    my_hits = NCBIXML.read(my_search)
    my_search.close()
    print('Time elapsed for NCBI QBLAST: {:,} sec'.format(int(time.time() - start)))
    if my_hits.descriptions is None:
        print('No hit found.')
        pdb_id = None
        return pdb_id
    print('Top hits (up to {:,} displayed)'.format(num_display))
    for i, description in enumerate(my_hits.descriptions):
        if i >= num_display:
            break
        print(description.title)
    top_hit_ids = get_top_hit_ids(my_hits)
    return top_hit_ids

def get_representative_leaf(node, size='median'):
    leaves = ete.get_leaves(node)
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
    top_hit_ids = []
    database_names = g['database'].split(',')
    for database_name in database_names:
        if pdb_id is not None:
            break
        print('Starting the sequence similarity search against protein structure database: {}'.format(database_name))
        if (database_name=='pdb'):
            print('MMseqs2 search against PDB: Query = {}'.format(representative_leaf.name))
            print('MMseqs2 search against PDB: Query sequence = {}'.format(aa_query))
            try:
                endpoint_url = 'https://search.rcsb.org/rcsbsearch/v2/query'
                headers = {'Content-Type':'application/json'}
                param_dict = { # https://search.rcsb.org/index.html#building-search-request
                    'evalue_cutoff':g['database_evalue_cutoff'],
                    'identity_cutoff':g['database_minimum_identity'],
                    'sequence_type':'protein',
                    'value':aa_query,
                }
                query_dict = {
                    'query':{'type':'terminal', 'service':'sequence', 'parameters':param_dict,},
                    'request_options':{'scoring_strategy':'sequence',},
                    'return_type':'polymer_entity',
                }
                query_json = json.dumps(query_dict)
                start = time.time()
                response = requests.post(url=endpoint_url, data=query_json, headers=headers)
                print('Time elapsed for MMseqs2 search: {:,} sec'.format(int(time.time() - start)))
                mmseqs2_out = response.json()
                print('Top hits (up to 10 displayed)')
                best_hit = None
                for hit in mmseqs2_out['result_set']:
                    txt = 'PDB identifier: {}  RCSB PDB Search Score: {}'
                    print(txt.format(hit['identifier'], hit['score']))
                for hit in mmseqs2_out['result_set']:
                    if best_hit is None:
                        hit_pdb_id = re.sub('_.*', '', hit['identifier'])
                        parser_pymol.initialize_pymol(pdb_id=hit_pdb_id)
                        num_chain = parser_pymol.get_num_chain()
                        if num_chain <= g['pymol_max_num_chain']:
                            best_hit = hit
                        else:
                            print(f'Number of chains in {hit_pdb_id} ({num_chain}) is larger than the maximum number of chains allowed (--pymol_max_num_chain {g["pymol_max_num_chain"]}). Unsuitable.', flush=True)
                if best_hit is None:
                    print('No suitable hit found in the PDB database.')
                    pdb_id = None
                else:
                    print('MMseqs2 search against PDB: Best hit identifier = {}'.format(best_hit['identifier']))
                    pdb_id = re.sub('_.*', '', best_hit['identifier'])
                    g['selected_database'] = 'pdb'
            except Exception as e:
                print(e)
                print('MMseqs2 search against PDB was unsuccessful.')
                pdb_id = None
        elif (database_name=='alphafill')|(database_name=='alphafold'):
            if len(top_hit_ids)==0:
                top_hit_ids = run_qblast(aa_query, num_display=10, evalue_cutoff=g['database_evalue_cutoff'])
            if len(top_hit_ids)==0:
                print('No QBLAST hit with the E-value threshold of {}.'.format(g['database_evalue_cutoff']))
                pdb_id = None
            else:
                for i in range(len(top_hit_ids)):
                    print('Retrieving protein structure: {}'.format(top_hit_ids[i]), flush=True)
                    if (database_name=='alphafill'):
                        download_url = 'https://alphafill.eu/v1/aff/'+top_hit_ids[i]
                    elif (database_name=='alphafold'):
                        download_url = 'https://alphafold.ebi.ac.uk/files/AF-' + top_hit_ids[i] + '-F1-model_v2.pdb'
                    if is_url_valid(url=download_url):
                        try:
                            alphafold_pdb = urllib.request.urlopen(download_url).read()
                            if (database_name == 'alphafill'):
                                alphafold_pdb_path = os.path.basename(download_url)+'.cif'
                            elif (database_name=='alphafold'):
                                alphafold_pdb_path = os.path.basename(download_url)
                            with open(alphafold_pdb_path, mode='wb') as f:
                                f.write(alphafold_pdb)
                            print('Download succeeded at: {}'.format(download_url), flush=True)
                            pdb_id = alphafold_pdb_path
                            g['selected_database'] = database_name
                            break
                        except:
                            print('Download failed at: {}'.format(download_url), flush=True)
                            pdb_id = None
                    else:
                        print('Download URL not found: {}'.format(download_url), flush=True)
                        pdb_id = None
    g['pdb'] = pdb_id
    if g['pdb'] is not None:
        print('Selected database and ID: {} and {}'.format(g['selected_database'], g['pdb']))
    else:
        txt = 'All specified databases ({}) were searched but no suitable structure was found. '
        txt += 'Continuing without protein structure.'
        print(txt.format(g['database']))
    return g

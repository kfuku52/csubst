import numpy as np
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
    descriptions = getattr(my_hits, 'descriptions', None)
    if not descriptions:
        return []
    top_hit_ids = []
    for description in descriptions:
        top_hit_title = str(getattr(description, 'title', '') or '')
        hit_id_match = re.search(r'\|([^|]+)\|', top_hit_title)
        if hit_id_match is not None:
            top_hit_id = hit_id_match.group(1)
        else:
            tokens = top_hit_title.split()
            top_hit_id = tokens[0] if len(tokens) > 0 else ''
        top_hit_id = re.sub(r'\..*', '', top_hit_id)
        top_hit_id = top_hit_id.strip()
        if top_hit_id != '':
            top_hit_ids.append(top_hit_id)
    return top_hit_ids

def run_qblast(aa_query, num_display=10, evalue_cutoff=10):
    print('Running NCBI BLAST against UniProtKB/SwissProt. '
          'This step should finish within minutes but may take hours depending on the NCBI QBLAST server conditions.')
    start = time.time()
    my_search = NCBIWWW.qblast(program='blastp', database='swissprot', sequence=aa_query, expect=evalue_cutoff)
    try:
        my_hits = NCBIXML.read(my_search)
    finally:
        my_search.close()
    print('Time elapsed for NCBI QBLAST: {:,} sec'.format(int(time.time() - start)))
    if (my_hits.descriptions is None) or (len(my_hits.descriptions) == 0):
        print('No hit found.')
        return []
    print('Top hits (up to {:,} displayed)'.format(num_display))
    for i, description in enumerate(my_hits.descriptions):
        if i >= num_display:
            break
        print(description.title)
    top_hit_ids = get_top_hit_ids(my_hits)
    return top_hit_ids

def _resolve_network_timeout(g, default=30):
    timeout = g.get('database_timeout', default)
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        return float(default)
    if timeout <= 0:
        return float(default)
    return timeout

def get_representative_leaf(node, size='median'):
    leaves = ete.get_leaves(node)
    if len(leaves) == 0:
        raise ValueError('No leaves were found to select a representative sequence.')
    leaf_seqlens = [ len(ete.get_prop(l, 'sequence', '').replace('-', '')) for l in leaves ]
    if size == 'median':
        ind = np.argsort(leaf_seqlens)[len(leaf_seqlens) // 2]
    else:
        raise ValueError('Unsupported representative leaf size mode: {}'.format(size))
    representative_leaf = leaves[ind]
    return representative_leaf

def is_url_valid(url, timeout=30):
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            pass
        return True
    except urllib.error.HTTPError as exc:
        try:
            exc.close()
        except Exception:
            pass
        # Some servers reject HEAD even when GET is available.
        if int(getattr(exc, 'code', 0)) in [403, 405, 501]:
            get_request = urllib.request.Request(url)
            get_request.get_method = lambda: 'GET'
            try:
                with urllib.request.urlopen(get_request, timeout=timeout):
                    pass
                return True
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                return False
        return False
    except (urllib.error.URLError, TimeoutError):
        return False


def _normalize_branch_ids(branch_ids):
    if branch_ids is None:
        return list()
    values = np.asarray(branch_ids, dtype=object)
    flat_values = np.atleast_1d(values).reshape(-1)
    if flat_values.size == 0:
        return list()
    normalized = []
    for value in flat_values.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('branch_ids should be integer-like.')
        if isinstance(value, (int, np.integer)):
            normalized.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            if (not np.isfinite(value)) or (not float(value).is_integer()):
                raise ValueError('branch_ids should be integer-like.')
            normalized.append(int(value))
            continue
        value_txt = str(value).strip()
        if (value_txt == '') or (not bool(re.fullmatch(r'[+-]?[0-9]+(?:\.0+)?', value_txt))):
            raise ValueError('branch_ids should be integer-like.')
        normalized.append(int(float(value_txt)))
    return normalized


def pdb_sequence_search(g):
    print('')
    branch_ids = _normalize_branch_ids(g.get('branch_ids', []))
    if len(branch_ids) == 0:
        raise ValueError('No branch IDs were provided for selecting a representative sequence.')
    representative_branch_id = branch_ids[0]
    representative_leaf = None
    for node in g['tree'].traverse():
        if (ete.get_prop(node, "numerical_label")==representative_branch_id):
            representative_leaf = get_representative_leaf(node, size='median')
            nlabel = ete.get_prop(representative_leaf, "numerical_label")
            aa_query = sequence.translate_state(nlabel=nlabel, mode='aa', g=g)
            aa_query = aa_query.replace('-', '')
            break
    if representative_leaf is None:
        raise ValueError('Representative branch ID {} was not found in the tree.'.format(representative_branch_id))
    if aa_query == '':
        raise ValueError('Representative amino acid query was empty.')
    pdb_id = None
    top_hit_ids = []
    database_names = [db.strip().lower() for db in g['database'].split(',') if db.strip()]
    allowed_database_names = {'pdb', 'alphafill', 'alphafold'}
    if len(database_names) == 0:
        raise ValueError('No database was specified. Use --database with one or more of pdb,alphafill,alphafold.')
    unknown_database_names = [db for db in database_names if db not in allowed_database_names]
    if len(unknown_database_names) > 0:
        txt = 'Unknown database name(s) in --database: {}. Supported values: {}.'
        raise ValueError(txt.format(','.join(unknown_database_names), ','.join(sorted(allowed_database_names))))
    network_timeout = _resolve_network_timeout(g=g)
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
                response = requests.post(
                    url=endpoint_url,
                    data=query_json,
                    headers=headers,
                    timeout=network_timeout,
                )
                response.raise_for_status()
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
        elif database_name in ['alphafill', 'alphafold']:
            if len(top_hit_ids)==0:
                try:
                    top_hit_ids = run_qblast(aa_query, num_display=10, evalue_cutoff=g['database_evalue_cutoff'])
                except Exception as e:
                    print(e)
                    print('QBLAST search was unsuccessful.')
                    top_hit_ids = []
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
                    if is_url_valid(url=download_url, timeout=network_timeout):
                        try:
                            with urllib.request.urlopen(download_url, timeout=network_timeout) as response:
                                alphafold_pdb = response.read()
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
                        except Exception as exc:
                            print('Download failed at: {} ({})'.format(download_url, exc), flush=True)
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

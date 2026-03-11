import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymol

from Bio import SeqIO

from io import StringIO
import copy
import os
import re
import subprocess
import sys
import time

from csubst import sequence
from csubst import runtime
from csubst import utility
from csubst import ete

def _cmd_delete_all():
    delete = getattr(pymol.cmd, 'delete', None)
    if callable(delete):
        return delete('all')
    return pymol.cmd.do('delete all')


def _cmd_fetch(pdb_id):
    fetch = getattr(pymol.cmd, 'fetch', None)
    if callable(fetch):
        return fetch(pdb_id)
    return pymol.cmd.do('fetch {}'.format(pdb_id))


def _cmd_remove(selection):
    remove = getattr(pymol.cmd, 'remove', None)
    if callable(remove):
        return remove(selection)
    return pymol.cmd.do('remove {}'.format(selection))


def _cmd_hide(representation, selection='all'):
    hide = getattr(pymol.cmd, 'hide', None)
    if callable(hide):
        return hide(representation, selection)
    return pymol.cmd.do('hide {}'.format(representation))


def _cmd_show(representation, selection='all'):
    show = getattr(pymol.cmd, 'show', None)
    if callable(show):
        return show(representation, selection)
    return pymol.cmd.do('show {}'.format(representation))


def _cmd_set(setting_name, value, selection=''):
    set_cmd = getattr(pymol.cmd, 'set', None)
    if callable(set_cmd):
        return set_cmd(setting_name, value, selection)
    if selection != '':
        return pymol.cmd.do('set {}, {}, {}'.format(setting_name, value, selection))
    return pymol.cmd.do('set {}, {}'.format(setting_name, value))


def _cmd_color(color_name, selection):
    color = getattr(pymol.cmd, 'color', None)
    if callable(color):
        return color(color_name, selection)
    return pymol.cmd.do('color {}, {}'.format(color_name, selection))


def _cmd_count_atoms(selection):
    count_atoms = getattr(pymol.cmd, 'count_atoms', None)
    if callable(count_atoms):
        return int(count_atoms(selection))
    return 0


def _cmd_zoom(selection='all'):
    zoom = getattr(pymol.cmd, 'zoom', None)
    if callable(zoom):
        return zoom(selection)
    return pymol.cmd.do('zoom')


def _cmd_preset_ligand_sites_trans_hq(selection='all'):
    preset_module = getattr(pymol, 'preset', None)
    preset_func = getattr(preset_module, 'ligand_sites_trans_hq', None)
    if callable(preset_func):
        return preset_func(selection=selection)
    return pymol.cmd.do("preset.ligand_sites_trans_hq(selection='{}')".format(selection))


def _cmd_color_organic():
    util_module = getattr(pymol, 'util', None)
    cbag = getattr(util_module, 'cbag', None)
    if callable(cbag):
        return cbag('organic')
    return pymol.cmd.do('util.cbag organic')


def _has_organic_ligand():
    return (_cmd_count_atoms('organic') > 0)


def _extract_positive_site_array(series):
    values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64, copy=True)
    valid_mask = np.isfinite(values)
    if valid_mask.any():
        valid_mask &= np.isclose(values, np.rint(values))
    site_values = np.zeros(values.shape[0], dtype=np.int64)
    if valid_mask.any():
        rounded = np.rint(values[valid_mask]).astype(np.int64, copy=False)
        positive_mask = rounded > 0
        valid_positions = np.flatnonzero(valid_mask)
        invalid_positions = valid_positions[~positive_mask]
        valid_mask[invalid_positions] = False
        site_values[valid_positions[positive_mask]] = rounded[positive_mask]
    return site_values, valid_mask


def initialize_pymol(pdb_id):
    #pymol.pymol_argv = ['pymol','-qc']
    #pymol.finish_launching()
    _cmd_delete_all()
    is_old_pdb_code = bool(re.fullmatch('[0-9][A-Za-z0-9]{3}', pdb_id))
    is_new_pdb_code = bool(re.fullmatch('pdb_[0-9]{5}[A-Za-z0-9]{3}', pdb_id))
    if is_old_pdb_code|is_new_pdb_code:
        print('Fetching PDB code {}. Internet connection is needed.'.format(pdb_id), flush=True)
        _cmd_fetch(pdb_id)
    else:
        print('Loading PDB file: {}'.format(pdb_id), flush=True)
        pymol.cmd.load(pdb_id)

def write_mafft_alignment(g):
    tmp_pdb_fasta = runtime.temp_path('tmp.csubst.pdb_seq.fa')
    mafft_workdir = os.path.dirname(os.path.abspath(tmp_pdb_fasta)) or os.getcwd()
    mafft_map_file = tmp_pdb_fasta+'.map'
    if os.path.exists(mafft_map_file):
        os.remove(mafft_map_file)
    pdb_seq = pymol.cmd.get_fastastr(selection='polymer.protein', state=-1, quiet=1)
    with open(tmp_pdb_fasta, 'w') as f:
        f.write(pdb_seq)
    leaf_aa_fasta = runtime.temp_path('tmp.csubst.leaf.aa.fa')
    sequence.write_alignment(outfile=leaf_aa_fasta, mode='aa', g=g, leaf_only=True)
    cmd_mafft = [g['mafft_exe'], '--keeplength', '--mapout', '--quiet',
                 '--thread', '1',]
    if g['mafft_op'] >= 0:
        cmd_mafft += ['--op', str(g['mafft_op']),]
    if g['mafft_ep'] >= 0:
        cmd_mafft += ['--ep', str(g['mafft_ep']),]
    cmd_mafft += ['--add', os.path.basename(tmp_pdb_fasta), os.path.basename(leaf_aa_fasta)]
    print('Running MAFFT to align the PDB sequence with the input alignment.', flush=True)
    print('Command: {}'.format(' '.join(cmd_mafft)), flush=True)
    try:
        out_mafft = subprocess.run(cmd_mafft, cwd=mafft_workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise AssertionError('mafft PATH cannot be found: {}'.format(g['mafft_exe'])) from exc
    if out_mafft.returncode != 0:
        stderr_text = out_mafft.stderr.decode('utf8', errors='replace').strip()
        if stderr_text == '':
            stderr_text = 'unknown error'
        txt = 'MAFFT failed with exit code {}: {}'
        raise RuntimeError(txt.format(out_mafft.returncode, stderr_text))
    with open(g['mafft_add_fasta'], 'w') as f:
        f.write(out_mafft.stdout.decode('utf8'))
    if os.path.getsize(g['mafft_add_fasta'])==0:
        txt = 'File size of {} is 0. A wrong ID might be specified in --pdb.'
        raise ValueError(txt.format(g['mafft_add_fasta']))
    print('')
    map_generated = False
    for i in range(10):
        if os.path.exists(mafft_map_file):
            print('MAFFT alignment file was generated: {}'.format(g['mafft_add_fasta']), flush=True)
            map_generated = True
            break
        else:
            print('MAFFT alignment file not detected. Waiting {:} sec'.format(i+1), flush=True)
            time.sleep(1)
    if not map_generated:
        txt = 'MAFFT map output file was not generated: {}'
        raise RuntimeError(txt.format(mafft_map_file))
    print(
        "Since CSUBST does not automatically exclude poorly aligned regions, "
        "please carefully check the MAFFT alignment file before interpreting substitution events.",
        flush=True
    )
    print(
        "If manual adjustments are necessary, please correct the amino acid positions of the "
        "database-derived sequences and use the updated MAFFT alignment file as input with --user_alignment.",
        flush=True
    )
    print(
        "When manually editing the alignment, do not disturb the amino acid positions. "
        "If excess amino acid sites are present in the database-derived sequences, remove them, "
        "but be mindful of amino acid site numbering because CSUBST cannot account for numbering shifts "
        "introduced by manual removal.",
        flush=True
    )
    print(
        "If you choose to rerun CSUBST with --user_alignment, please use the same --alignment_file "
        "that was used in this run.",
        flush=True
    )
    print('', flush=True)

def get_residue_numberings():
    out = dict()
    object_names = pymol.cmd.get_names()
    object_names = [ on for on in object_names if not on.endswith('_pol_conts') ]
    print('Detected protein structure objects: {}'.format(', '.join(object_names)))
    for object_name in object_names:
        for ch in pymol.cmd.get_chains(object_name):
            pymol.stored.residues = []
            txt_selection = '{} and chain {} and name ca'.format(object_name, ch)
            pymol.cmd.iterate(selection=txt_selection, expression='stored.residues.append(resi)')
            residue_numbers = [ int(x) for x in pymol.stored.residues if not bool(re.search('[a-zA-Z]', x)) ]
            residue_numbers = sorted(list(set(residue_numbers))) # Drop occasionally observed duplications
            residue_iloc = np.arange(len(residue_numbers)) + 1
            col1 = 'codon_site_'+object_name+'_'+ch
            col2 = 'codon_site_pdb_'+object_name+'_'+ch
            dict_tmp = {col1:residue_iloc, col2:residue_numbers}
            df_tmp = pd.DataFrame(dict_tmp)
            out[object_name+'_'+ch] = df_tmp
    return out

def add_pdb_residue_numbering(df):
    residue_numberings = get_residue_numberings()
    object_names = [on for on in pymol.cmd.get_names() if not on.endswith('_pol_conts')]
    for object_name in object_names:
        for ch in pymol.cmd.get_chains(object_name):
            key = object_name+'_'+ch
            if key not in residue_numberings:
                continue
            if residue_numberings[key].shape[0]==0:
                continue
            if 'codon_site_'+key in df.columns:
                df = pd.merge(df, residue_numberings[key], on='codon_site_'+key, how='left')
                df['codon_site_pdb_'+key] = df['codon_site_pdb_'+key].fillna(0).astype(int)
    print('The column "codon_site_**ID**" indicates the positions of codons/amino acids in the sequence "**ID**" in the input alignment. 0 = missing site.')
    print('The column "codon_site_pdb_**ID**" indicates the positions of codons/amino acids in the sequence "**ID**" in the PDB file. 0 = missing site.')
    return df


def _select_best_pdb_chain_column(df):
    pdb_cols = [col for col in df.columns.tolist() if str(col).startswith('codon_site_pdb_')]
    if len(pdb_cols) == 0:
        return None
    best_col = None
    best_score = None
    for col in sorted(pdb_cols):
        values = pd.to_numeric(df.loc[:, col], errors='coerce').fillna(0).to_numpy(dtype=np.int64, copy=False)
        is_mapped = values > 0
        mapped_count = int(is_mapped.sum())
        unique_count = int(np.unique(values[is_mapped]).shape[0]) if mapped_count > 0 else 0
        score = (mapped_count, unique_count, col)
        if (best_score is None) or (score > best_score):
            best_col = col
            best_score = score
    return best_col


def _parse_pdb_residue_number(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        residue = int(value)
    elif isinstance(value, (float, np.floating)):
        if (not np.isfinite(value)) or (not float(value).is_integer()):
            return None
        residue = int(value)
    else:
        txt = str(value).strip()
        if txt == '':
            return None
        if not bool(re.fullmatch(r'[+-]?[0-9]+(?:\.0+)?', txt)):
            return None
        residue = int(float(txt))
    if residue <= 0:
        return None
    return residue


def _get_chain_ca_coordinates():
    out = dict()
    object_names = [on for on in pymol.cmd.get_names() if not str(on).endswith('_pol_conts')]
    for object_name in object_names:
        for ch in pymol.cmd.get_chains(object_name):
            key = object_name + '_' + ch
            model = pymol.cmd.get_model('{} and chain {} and name ca'.format(object_name, ch))
            residue_coord = dict()
            for atom in model.atom:
                residue = _parse_pdb_residue_number(atom.resi)
                if residue is None:
                    continue
                residue_coord[int(residue)] = np.array(atom.coord, dtype=np.float64)
            out[key] = residue_coord
    return out


def add_contact_degree_from_structure(df, distance_cutoff=8.0, chain_col=None):
    distance_cutoff = float(distance_cutoff)
    if distance_cutoff <= 0:
        raise ValueError('distance_cutoff should be > 0.')
    out = df.copy(deep=True)
    if chain_col is None:
        chain_col = _select_best_pdb_chain_column(df=out)
    if chain_col is None:
        out.loc[:, 'epistasis_contact_chain'] = ''
        out.loc[:, 'epistasis_contact_residue'] = 0
        out.loc[:, 'epistasis_contact_degree'] = np.nan
        out.loc[:, 'epistasis_contact_degree_z'] = 0.0
        out.loc[:, 'epistasis_contact_proximity'] = np.nan
        out.loc[:, 'epistasis_contact_proximity_z'] = 0.0
        return out
    chain_name = str(chain_col).replace('codon_site_pdb_', '', 1)
    residue_values = pd.to_numeric(out.loc[:, chain_col], errors='coerce').fillna(0).astype(np.int64).to_numpy(copy=False)
    coord_by_chain = _get_chain_ca_coordinates()
    chain_coord = coord_by_chain.get(chain_name, dict())
    mapped_residues = sorted(list(set([int(v) for v in residue_values.tolist() if (int(v) > 0) and (int(v) in chain_coord)])))
    residue_degree = dict()
    residue_proximity = dict()
    if len(mapped_residues) > 0:
        xyz = np.array([chain_coord[resi] for resi in mapped_residues], dtype=np.float64)
        delta = xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :]
        dist2 = np.einsum('ijk,ijk->ij', delta, delta)
        cutoff2 = distance_cutoff ** 2
        degree = (dist2 <= cutoff2).sum(axis=1).astype(np.int64) - 1
        dist = np.sqrt(np.clip(dist2, a_min=0.0, a_max=None))
        proximity_weight = np.clip((distance_cutoff - dist) / distance_cutoff, a_min=0.0, a_max=None)
        np.fill_diagonal(proximity_weight, 0.0)
        proximity = proximity_weight.sum(axis=1, dtype=np.float64)
        residue_degree = {int(resi): int(deg) for resi, deg in zip(mapped_residues, degree.tolist())}
        residue_proximity = {int(resi): float(prox) for resi, prox in zip(mapped_residues, proximity.tolist())}
    degree_values = np.full(shape=(out.shape[0],), fill_value=np.nan, dtype=np.float64)
    proximity_values = np.full(shape=(out.shape[0],), fill_value=np.nan, dtype=np.float64)
    selected_residue = np.zeros(shape=(out.shape[0],), dtype=np.int64)
    for i, residue in enumerate(residue_values.tolist()):
        if residue <= 0:
            continue
        selected_residue[i] = int(residue)
        if int(residue) in residue_degree:
            degree_values[i] = float(residue_degree[int(residue)])
        if int(residue) in residue_proximity:
            proximity_values[i] = float(residue_proximity[int(residue)])
    out.loc[:, 'epistasis_contact_chain'] = chain_name
    out.loc[:, 'epistasis_contact_residue'] = selected_residue
    out.loc[:, 'epistasis_contact_degree'] = degree_values
    out.loc[:, 'epistasis_contact_proximity'] = proximity_values
    degree_z_values = np.zeros(shape=(out.shape[0],), dtype=np.float64)
    is_finite = np.isfinite(degree_values)
    if is_finite.any():
        finite_values = degree_values[is_finite]
        mean = finite_values.mean()
        std = finite_values.std(ddof=0)
        if std > 0:
            degree_z_values[is_finite] = (finite_values - mean) / std
        else:
            degree_z_values[is_finite] = 0.0
    out.loc[:, 'epistasis_contact_degree_z'] = degree_z_values
    proximity_z_values = np.zeros(shape=(out.shape[0],), dtype=np.float64)
    is_finite = np.isfinite(proximity_values)
    if is_finite.any():
        finite_values = proximity_values[is_finite]
        mean = finite_values.mean()
        std = finite_values.std(ddof=0)
        if std > 0:
            proximity_z_values[is_finite] = (finite_values - mean) / std
        else:
            proximity_z_values[is_finite] = 0.0
    out.loc[:, 'epistasis_contact_proximity_z'] = proximity_z_values
    return out


def add_coordinate_from_mafft_map(df, mafft_map_file='tmp.csubst.pdb_seq.fa.map'):
    mafft_map_file = runtime.temp_path(mafft_map_file)
    print('Loading amino acid coordinates from: {}'.format(mafft_map_file), flush=True)
    with open(mafft_map_file, 'r') as f:
        map_str = f.read()
    map_list = map_str.split('>')[1:]
    for map_item in map_list:
        map_lines = map_item.splitlines()
        if len(map_lines)==0:
            continue
        seq_name = map_lines[0]
        seq_csv = '\n'.join(map_lines[1:])
        if seq_csv.strip()=='': # empty data
            df.loc[:,'codon_site_'+seq_name] = 0
            df.loc[:,'aa_'+seq_name] = ''
        else:
            try:
                df_tmp = pd.read_csv(StringIO(seq_csv), comment='#', header=None)
            except pd.errors.EmptyDataError:
                df.loc[:, 'codon_site_'+seq_name] = 0
                df.loc[:, 'aa_'+seq_name] = ''
                continue
            df_tmp.columns = ['aa_'+seq_name, 'codon_site_'+seq_name, 'codon_site_alignment']
            aln_site_str = df_tmp['codon_site_alignment'].astype(str).str.strip()
            is_missing_in_aln = aln_site_str.isin(['', '-'])
            df_tmp = df_tmp.loc[~is_missing_in_aln, :].copy()
            if df_tmp.shape[0] == 0:
                df.loc[:, 'codon_site_'+seq_name] = 0
                df.loc[:, 'aa_'+seq_name] = ''
                continue
            aln_sites = pd.to_numeric(df_tmp['codon_site_alignment'], errors='coerce')
            if aln_sites.isna().any():
                invalid_values = df_tmp.loc[aln_sites.isna(), 'codon_site_alignment'].astype(str).tolist()
                invalid_txt = ','.join(invalid_values[:5])
                if len(invalid_values) > 5:
                    invalid_txt += ',...'
                txt = 'Invalid codon_site_alignment value(s) in {} for sequence {}: {}'
                raise ValueError(txt.format(mafft_map_file, seq_name, invalid_txt))
            df_tmp = df_tmp.drop(columns=['codon_site_alignment']).assign(
                codon_site_alignment=aln_sites.astype(np.int64).to_numpy(copy=False)
            )
            df = pd.merge(df, df_tmp, on='codon_site_alignment', how='left')
            codon_col = 'codon_site_' + seq_name
            aa_col = 'aa_' + seq_name
            codon_raw = df[codon_col]
            codon_text = codon_raw.astype(str).str.strip()
            is_missing_codon = codon_raw.isna() | codon_text.isin(['', '-'])
            codon_numeric = pd.to_numeric(codon_raw, errors='coerce')
            is_invalid_codon = (~is_missing_codon) & codon_numeric.isna()
            if is_invalid_codon.any():
                invalid_values = df.loc[is_invalid_codon, codon_col].astype(str).tolist()
                invalid_txt = ','.join(invalid_values[:5])
                if len(invalid_values) > 5:
                    invalid_txt += ',...'
                txt = 'Invalid codon_site value(s) in {} for sequence {}: {}'
                raise ValueError(txt.format(mafft_map_file, seq_name, invalid_txt))
            df[codon_col] = codon_numeric.fillna(0).astype(int)
            df[aa_col] = df.loc[:, aa_col].fillna('')
            df.loc[df[codon_col] == 0, aa_col] = ''
    return df

def add_coordinate_from_user_alignment(df, user_alignment):
    print('Loading amino acid coordinates from: {}'.format(user_alignment), flush=True)
    pdb_fasta = pymol.cmd.get_fastastr(selection='polymer.protein', state=-1, quiet=1)
    tmp_pdb_fasta = runtime.temp_path('tmp.csubst.pdb_seq.fa')
    with open(tmp_pdb_fasta, 'w') as f:
        f.write(pdb_fasta)
    with open(tmp_pdb_fasta, 'r') as f:
        pdb_seqs = list(SeqIO.parse(f, 'fasta'))
    with open(user_alignment, 'r') as f:
        user_seqs = list(SeqIO.parse(f, 'fasta'))
    num_matched_sequences = 0
    for user_seq in user_seqs:
        for pdb_seq in pdb_seqs:
            if user_seq.name!=pdb_seq.name:
                continue
            num_matched_sequences += 1
            user_seq_str = str(user_seq.seq).replace('\n', '').upper()
            pdb_seq_str = str(pdb_seq.seq).replace('\n', '').upper()
            user_seq_counter = 0
            pdb_seq_counter = 0
            row_labels = df.index.to_list()
            txt = 'The alignment length should match between --alignment_file ({} sites) and --user_alignment ({} sites)'
            if len(user_seq_str) != len(row_labels):
                raise ValueError(txt.format(len(row_labels), len(user_seq_str)))
            df['aa_' + user_seq.name] = ''
            df['codon_site_' + user_seq.name] = 0
            while user_seq_counter <= (len(row_labels) - 1):
                if user_seq_str[user_seq_counter]=='-':
                    user_seq_counter += 1
                    continue
                if pdb_seq_counter >= len(pdb_seq_str):
                    txt = 'Unable to map --user_alignment residue at position {} for sequence "{}" to the PDB sequence "{}".'
                    raise ValueError(txt.format(user_seq_counter + 1, user_seq.name, pdb_seq.name))
                if user_seq_str[user_seq_counter]==pdb_seq_str[pdb_seq_counter]:
                    row_label = row_labels[user_seq_counter]
                    df.at[row_label, 'aa_' + user_seq.name] = user_seq_str[user_seq_counter]
                    df.at[row_label, 'codon_site_' + user_seq.name] = pdb_seq_counter + 1
                    user_seq_counter += 1
                    pdb_seq_counter += 1
                else:
                    pdb_seq_counter += 1
    if num_matched_sequences == 0:
        pdb_names = sorted([seq.name for seq in pdb_seqs])
        user_names = sorted([seq.name for seq in user_seqs])
        txt = 'No sequence name overlap was found between PDB-derived sequences ({}) and --user_alignment sequences ({}).'
        raise ValueError(txt.format(','.join(pdb_names), ','.join(user_names)))
    return df

def calc_aa_identity(g):
    seqs = sequence.read_fasta(path=g['mafft_add_fasta'])
    seqnames = list(seqs.keys())
    pdb_base = re.sub(r'\..*', '', os.path.basename(g['pdb']))
    pdb_seqnames = [ sn for sn in seqnames if sn.startswith(pdb_base) ]
    other_seqnames = [ sn for sn in seqnames if not sn.startswith(pdb_base) ]
    aa_identity_values = dict()
    for pdb_seqname in pdb_seqnames:
        aa_identity_values[pdb_seqname] = []
        for other_seqname in other_seqnames:
            aa_identity = sequence.calc_identity(seq1=seqs[pdb_seqname], seq2=seqs[other_seqname])
            aa_identity_values[pdb_seqname].append(aa_identity)
        aa_identity_values[pdb_seqname] = np.array(aa_identity_values[pdb_seqname])
    aa_identity_means = dict()
    for pdb_seqname in pdb_seqnames:
        if aa_identity_values[pdb_seqname].shape[0] == 0:
            aa_identity_means[pdb_seqname] = np.nan
        else:
            aa_identity_means[pdb_seqname] = aa_identity_values[pdb_seqname].mean()
    aa_ranges = dict()
    for pdb_seqname in pdb_seqnames:
        alphabet_sites = [ m.start() for m in re.finditer('[a-zA-Z]', seqs[pdb_seqname]) ]
        if len(alphabet_sites)==0:
            aa_start = 0
            aa_end = 0
        else:
            aa_start = min(alphabet_sites)
            aa_end = max(alphabet_sites)
        aa_ranges[pdb_seqname] = [aa_start, aa_end]
    g['aa_identity_values'] = aa_identity_values
    g['aa_identity_means'] = aa_identity_means
    g['aa_spans'] = aa_ranges
    return g

def mask_subunit(g):
    protein_chains = pymol.cmd.get_chains('polymer.protein')
    nucleic_chains = pymol.cmd.get_chains('polymer.nucleic')
    colors = ['wheat','slate','salmon','brightorange','violet','olive',
              'firebrick','pink','marine','density','cyan','chocolate','teal',]
    num_chains = len(protein_chains) + len(nucleic_chains)
    if num_chains > len(colors):
        colors *= int(num_chains/len(colors)) + 1 # for supercomplex
    for nucleotide in ['DG','DT','DA','DC']: # DNA
        _cmd_color('pink', 'resn ' + nucleotide)
    if len(protein_chains) == 1:
        return None
    g = calc_aa_identity(g)
    pdb_seqnames = list(g['aa_identity_means'].keys())
    if len(pdb_seqnames) == 0:
        txt = 'No sequence with PDB basename prefix was found in {}. Skipping subunit masking.'
        print(txt.format(g['mafft_add_fasta']), flush=True)
        return None
    finite_identity_items = [item for item in g['aa_identity_means'].items() if np.isfinite(item[1])]
    if len(finite_identity_items) > 0:
        max_pdb_seqname = max(finite_identity_items, key=lambda item: item[1])[0]
    else:
        max_pdb_seqname = pdb_seqnames[0]
    max_spans = g['aa_spans'][max_pdb_seqname]
    i = 0
    for pdb_seqname in pdb_seqnames:
        if pdb_seqname==max_pdb_seqname:
            continue
        spans = g['aa_spans'][pdb_seqname]
        is_nonoverlapping_N_side = (max_spans[1] < spans[0])
        is_nonoverlapping_C_side = (max_spans[0] > spans[1])
        if (is_nonoverlapping_N_side|is_nonoverlapping_C_side):
            continue
        chain = pdb_seqname.rsplit('_', 1)[-1]
        print('Masking chain {}'.format(chain), flush=True)
        if spans[1]==0: # End position = 0 if no protein in the chain
            continue
        _cmd_color(colors[i], 'chain {} and polymer.protein'.format(chain))
        i += 1
    for chain in nucleic_chains:
        print('Masking chain {}'.format(chain), flush=True)
        _cmd_color(colors[i], 'chain {} and polymer.nucleic'.format(chain))
        i += 1

def set_color_gray(object_names, residue_numberings, gray_value):
    for object_name in object_names:
        if object_name.endswith('_pol_conts'):
            continue
        for ch in pymol.cmd.get_chains(object_name):
            key = object_name+'_'+ch
            if key not in residue_numberings:
                continue
            codon_site_pdb = residue_numberings[key].loc[:,'codon_site_pdb_'+key]
            is_nonzero = (codon_site_pdb!=0)
            if not is_nonzero.any():
                continue
            residue_start = codon_site_pdb.loc[is_nonzero].min()
            residue_end = codon_site_pdb.loc[is_nonzero].max()
            if pd.isna(residue_start) or pd.isna(residue_end):
                continue
            _cmd_color(
                'gray{}'.format(gray_value),
                '{} and chain {} and resi {:}-{:}'.format(object_name, ch, int(residue_start), int(residue_end)),
            )


def _paint_sites_with_hex(object_name, chain_id, sites, hex_value, label):
    if len(sites)==0:
        print('Skipping site painting. No amino acid substitutions: {}'.format(label), flush=True)
        return None
    sites = sorted(list(set([int(site) for site in sites])))
    print('Amino acid sites with {} will be painted with {}.'.format(label, hex_value), flush=True)
    txt_resi = '+'.join([str(site) for site in sites])
    _cmd_color(hex_value, '{} and chain {} and resi {}'.format(object_name, chain_id, txt_resi))
    return None


def _get_intensity_bin_hex(bin_index, num_bin):
    if num_bin <= 1:
        frac = 1.0
    else:
        frac = (bin_index + 1) / num_bin
    return utility.rgb_to_hex(r=1.0, g=1.0-0.8*frac, b=1.0-0.8*frac)


def _paint_intensity_sites(object_name, chain_id, site_values, label):
    if len(site_values)==0:
        print('Skipping site painting. No amino acid substitutions: {}'.format(label), flush=True)
        return None
    max_value = max(site_values.values())
    if max_value <= 0:
        print('Skipping site painting. Non-positive substitution values: {}'.format(label), flush=True)
        return None
    num_bin = 10
    bins = {i: [] for i in range(num_bin)}
    for site,value in site_values.items():
        frac = min(max(float(value)/max_value, 0.0), 1.0)
        bin_index = int(round(frac * (num_bin-1)))
        bins[bin_index].append(int(site))
    txt = '{}: max summed substitution probability = {:.4f}'
    print(txt.format(label, max_value), flush=True)
    for bin_index,sites in bins.items():
        if len(sites)==0:
            continue
        hex_value = _get_intensity_bin_hex(bin_index=bin_index, num_bin=num_bin)
        sub_label = '{} (bin {}/{})'.format(label, bin_index+1, num_bin)
        _paint_sites_with_hex(object_name=object_name, chain_id=chain_id, sites=sites, hex_value=hex_value, label=sub_label)
    return None


def _oldness_frac_to_hex(frac):
    frac = min(max(float(frac), 0.0), 1.0)
    if frac <= 0.5:
        t = frac / 0.5
        r,g,b = t,t,1.0-t  # blue -> yellow
    else:
        t = (frac - 0.5) / 0.5
        r,g,b = 1.0,1.0-t,0.0  # yellow -> red
    return utility.rgb_to_hex(r=r, g=g, b=b)


def _get_lineage_hex_by_branch(branch_ids, g):
    if len(branch_ids)==0:
        return dict()
    if len(branch_ids)==1:
        return {int(branch_ids[0]): _oldness_frac_to_hex(1.0)}

    node_by_id = dict()
    for node in g['tree'].traverse():
        node_by_id[int(ete.get_prop(node, "numerical_label"))] = node

    lengths = []
    for branch_id in branch_ids:
        node = node_by_id.get(int(branch_id), None)
        bl = float(getattr(node, 'dist', 0.0)) if (node is not None) else 0.0
        lengths.append(max(bl, 0.0))

    total_len = float(sum(lengths))
    out = dict()
    if total_len <= 0:
        for i,branch_id in enumerate(branch_ids):
            frac = i / (len(branch_ids)-1)
            out[int(branch_id)] = _oldness_frac_to_hex(frac)
        return out

    mid_fracs = []
    cumul = 0.0
    for bl in lengths:
        mid_fracs.append((cumul + bl*0.5) / total_len)
        cumul += bl
    min_mid = min(mid_fracs)
    max_mid = max(mid_fracs)
    span = max_mid - min_mid
    if span <= 0:
        for i,branch_id in enumerate(branch_ids):
            frac = i / (len(branch_ids)-1)
            out[int(branch_id)] = _oldness_frac_to_hex(frac)
        return out
    for branch_id,mid_frac in zip(branch_ids, mid_fracs):
        frac = (mid_frac - min_mid) / span
        out[int(branch_id)] = _oldness_frac_to_hex(frac)
    return out


def _normalize_branch_ids(branch_ids):
    if branch_ids is None:
        return []
    arr = np.asarray(branch_ids, dtype=object)
    arr = np.atleast_1d(arr).reshape(-1)
    if arr.size == 0:
        return []
    normalized = []
    for value in arr.tolist():
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


def _parse_positive_site(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        site = int(value)
    elif isinstance(value, (float, np.floating)):
        if (not np.isfinite(value)) or (not float(value).is_integer()):
            return None
        site = int(value)
    else:
        value_txt = str(value).strip()
        if value_txt == '':
            return None
        if not bool(re.fullmatch(r'[+-]?[0-9]+(?:\.0+)?', value_txt)):
            return None
        site = int(float(value_txt))
    if site <= 0:
        return None
    return site


def _parse_bool_like(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        if not np.isfinite(float(value)):
            return False
        return bool(float(value) != 0.0)
    txt = str(value).strip().lower()
    if txt in ['1', 'true', 'yes', 'y', 'on']:
        return True
    if txt in ['0', 'false', 'no', 'n', 'off', '', 'none', 'nan', 'na']:
        return False
    return False


def set_substitution_colors(df, g, object_names, N_sub_cols):
    single_branch_mode = bool(g.get('single_branch_mode', False))
    mode = str(g.get('mode', 'intersection')).lower()
    need_single_prob = (mode not in ['set', 'lineage', 'total'])
    need_total_prob = (mode == 'total')
    need_any2 = (mode not in ['set', 'lineage', 'total']) and (not single_branch_mode)
    single_prob_values = np.zeros(df.shape[0], dtype=np.float64)
    total_prob_values = np.zeros(df.shape[0], dtype=np.float64)
    if len(N_sub_cols) != 0:
        n_sub_frame = df.loc[:, N_sub_cols]
        if need_single_prob:
            single_prob_values = pd.to_numeric(
                n_sub_frame.max(axis=1),
                errors='coerce',
            ).fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        if need_total_prob:
            total_prob_values = pd.to_numeric(
                n_sub_frame.sum(axis=1),
                errors='coerce',
            ).fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    if need_any2 and ('OCNany2spe' in df.columns):
        any2spe_values = pd.to_numeric(df['OCNany2spe'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    else:
        any2spe_values = np.zeros(df.shape[0], dtype=np.float64)
    if need_any2 and ('OCNany2dif' in df.columns):
        any2dif_values = pd.to_numeric(df['OCNany2dif'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    else:
        any2dif_values = np.zeros(df.shape[0], dtype=np.float64)
    for object_name in object_names:
        if object_name.endswith('_pol_conts'):
            continue
        for ch in pymol.cmd.get_chains(object_name):
            codon_site_col = 'codon_site_pdb_'+object_name+'_'+ch
            if not codon_site_col in df.columns:
                continue
            site_values, valid_site_mask = _extract_positive_site_array(df.loc[:, codon_site_col])
            if mode=='set':
                if 'N_set_expr' not in df.columns:
                    print('Skipping site painting. N_set_expr column was not found for --mode set.', flush=True)
                    continue
                selected_sites = []
                for i in df.index:
                    codon_site = _parse_positive_site(df.at[i, codon_site_col])
                    if codon_site is None:
                        continue
                    if _parse_bool_like(df.at[i, 'N_set_expr']):
                        selected_sites.append(codon_site)
                _paint_sites_with_hex(
                    object_name=object_name,
                    chain_id=ch,
                    sites=selected_sites,
                    hex_value=utility.rgb_to_hex(r=1.0, g=0.0, b=0.0),
                    label='set_expr_N',
                )
                continue
            if mode=='total':
                site_values = dict()
                qualifying_mask = valid_site_mask & (total_prob_values >= float(g.get('min_single_prob', 0.8)))
                for codon_site, total_sub in zip(site_values[qualifying_mask], total_prob_values[qualifying_mask]):
                    site_values[int(codon_site)] = max(site_values.get(int(codon_site), 0.0), float(total_sub))
                _paint_intensity_sites(
                    object_name=object_name,
                    chain_id=ch,
                    site_values=site_values,
                    label='total_N',
                )
                continue
            if mode=='lineage':
                lineage_branch_ids = _normalize_branch_ids(g.get('branch_ids', []))
                if len(lineage_branch_ids) == 0:
                    print('Skipping site painting. No branch IDs were provided for lineage mode.', flush=True)
                    continue
                lineage_pairs = [(bid, 'N_sub_' + str(bid)) for bid in lineage_branch_ids if ('N_sub_' + str(bid)) in df.columns]
                if len(lineage_pairs) == 0:
                    print('Skipping site painting. No N_sub_* columns matched lineage branch IDs.', flush=True)
                    continue
                lineage_branch_ids = [bid for bid,_ in lineage_pairs]
                lineage_cols = [col for _,col in lineage_pairs]
                branch_hex = _get_lineage_hex_by_branch(lineage_branch_ids, g=g)
                site_by_branch = {bid: [] for bid in lineage_branch_ids}
                lineage_min_prob = float(g.get('min_single_prob', 0.8))
                for i in df.index:
                    codon_site = _parse_positive_site(df.at[i, codon_site_col])
                    if codon_site is None:
                        continue
                    vals = df.loc[i, lineage_cols].to_numpy(dtype=float)
                    if vals.shape[0]==0:
                        continue
                    qualifying = np.where(vals >= lineage_min_prob)[0]
                    if qualifying.shape[0]==0:
                        continue
                    # Stratigraphy mode: assign a site to the oldest lineage branch
                    # with sufficiently high substitution probability.
                    selected_index = int(qualifying.min())
                    branch_id = lineage_branch_ids[selected_index]
                    site_by_branch[branch_id].append(codon_site)
                for branch_id in lineage_branch_ids:
                    label = 'lineage_branch_{}'.format(branch_id)
                    _paint_sites_with_hex(
                        object_name=object_name,
                        chain_id=ch,
                        sites=site_by_branch[branch_id],
                        hex_value=branch_hex[branch_id],
                        label=label,
                    )
                continue
            color_sites = dict()
            color_sites['OCNany2spe'] = []
            color_sites['OCNany2dif'] = []
            color_sites['single_sub'] = []
            has_any2spe = ('OCNany2spe' in df.columns) and (not single_branch_mode)
            has_any2dif = ('OCNany2dif' in df.columns) and (not single_branch_mode)
            min_combinat_prob = float(g.get('min_combinat_prob', 0.5))
            min_single_prob = float(g.get('min_single_prob', 0.8))
            selected_site_values = site_values[valid_site_mask]
            if has_any2spe:
                spe_mask = valid_site_mask & (any2spe_values >= min_combinat_prob) & (any2dif_values <= any2spe_values)
                color_sites['OCNany2spe'] = selected_site_values[np.flatnonzero(spe_mask[valid_site_mask])].tolist()
            if has_any2dif:
                dif_mask = valid_site_mask & (any2dif_values >= min_combinat_prob) & (any2dif_values > any2spe_values)
                color_sites['OCNany2dif'] = selected_site_values[np.flatnonzero(dif_mask[valid_site_mask])].tolist()
            occupied_mask = np.zeros(df.shape[0], dtype=bool)
            if has_any2spe:
                occupied_mask |= spe_mask
            if has_any2dif:
                occupied_mask |= dif_mask
            single_mask = valid_site_mask & (~occupied_mask) & (single_prob_values >= min_single_prob)
            color_sites['single_sub'] = selected_site_values[np.flatnonzero(single_mask[valid_site_mask])].tolist()
            if single_branch_mode:
                color_sites['single_branch_N'] = copy.deepcopy(color_sites['single_sub'])
                del color_sites['OCNany2spe']
                del color_sites['OCNany2dif']
                del color_sites['single_sub']
            for key in color_sites.keys():
                if key=='OCNany2spe':
                    hex_value = utility.rgb_to_hex(r=1, g=0, b=0)
                elif key=='OCNany2dif':
                    hex_value = utility.rgb_to_hex(r=0, g=0, b=1)
                elif key=='single_sub':
                    hex_value = utility.rgb_to_hex(r=0.4, g=0.4, b=0.4)
                elif key=='single_branch_N':
                    hex_value = utility.rgb_to_hex(r=0.5, g=0, b=0.5)
                else:
                    continue
                _paint_sites_with_hex(
                    object_name=object_name,
                    chain_id=ch,
                    sites=color_sites[key],
                    hex_value=hex_value,
                    label=key,
                )
                if key in ['OCNany2spe','OCNany2dif'] and len(color_sites[key])>0:
                    txt_resi = '+'.join([str(site) for site in sorted(list(set(color_sites[key])))])
                    _cmd_set('transparency', 0.3, '{} and chain {} and resi {}'.format(object_name, ch, txt_resi))

def write_pymol_session(df, g):
    df = df.reset_index(drop=True)
    _cmd_set('seq_view', 1)
    _cmd_set('surface_quality', int(g.get('pymol_surface_quality', -1)))
    if g['remove_solvent']:
        _cmd_remove('solvent')
    if g['remove_ligand']:
        molecule_codes = g['remove_ligand'].split(',')
        for molecule_code in molecule_codes:
            _cmd_remove('resn '+molecule_code)
    has_organic_ligand = _has_organic_ligand()
    if has_organic_ligand:
        _cmd_preset_ligand_sites_trans_hq(selection='all')
    _cmd_hide('wire')
    _cmd_hide('ribbon')
    _cmd_show('cartoon')
    _cmd_set('transparency', g['pymol_transparency'], 'polymer.protein')
    _cmd_show('surface')
    object_names = pymol.cmd.get_names()
    #residue_numberings = get_residue_numberings()
    #set_color_gray(object_names, residue_numberings, gray_value=g['pymol_gray'])
    _cmd_color('gray{}'.format(g['pymol_gray']), 'polymer.protein')
    if has_organic_ligand:
        _cmd_color_organic()
    N_sub_cols = df.columns[df.columns.str.startswith('N_sub_')]
    set_substitution_colors(df, g, object_names, N_sub_cols)
    if g['mask_subunit']:
        mask_subunit(g)
    _cmd_zoom()
    pymol.cmd.deselect()
    pymol.cmd.save(g['session_file_path'])

def quit_pymol():
    pymol.cmd.quit(code=0)

def save_six_views(selection='all', 
                   prefix='tmp.csubst.pymol', 
                   width=900, 
                   height=900, 
                   dpi=300, 
                   ray=True):
    """
    Saves six images of the selected object from +X, -X, +Y, -Y, +Z, -Z directions.
    """
    
    # Each direction is 18 floats:
    #   1) 3x3 rotation matrix = 9 floats
    #   2) camera position (3 floats)
    #   3) front, back (2 floats)
    #   4) perspective (1 float)
    #   5) origin shift (3 floats)
    directions = {
        'pos_x': [
            # Rotation: point the camera along +X => +X points "out of screen"
            0.0,  0.0, -1.0,   # row1
            0.0,  1.0,  0.0,   # row2
            1.0,  0.0,  0.0,   # row3
            
            # Camera position (move camera +X):
            100.0, 0.0, 0.0,
            
            # front plane, back plane:
            0.0, 200.0,
            
            # perspective (1.0 => perspective on, -1 or 0 => orthoscopic):
            1.0,
            
            # origin translation:
            0.0, 0.0, 0.0
        ],
        'neg_x': [
            0.0,  0.0,  1.0,
            0.0,  1.0,  0.0,
           -1.0,  0.0,  0.0,
           
           -100.0, 0.0, 0.0,
           0.0, 200.0,
           1.0,
           0.0, 0.0, 0.0
        ],
        'pos_y': [
            # Camera along +Y => +Y out of screen
            1.0,  0.0,  0.0,
            0.0,  0.0, -1.0,
            0.0,  1.0,  0.0,
            
            0.0, 100.0, 0.0,
            0.0, 200.0,
            1.0,
            0.0, 0.0, 0.0
        ],
        'neg_y': [
            1.0,  0.0,  0.0,
            0.0,  0.0,  1.0,
            0.0, -1.0,  0.0,
            
            0.0, -100.0, 0.0,
            0.0, 200.0,
            1.0,
            0.0, 0.0, 0.0
        ],
        'pos_z': [
            # Camera along +Z => +Z out of screen
            1.0,  0.0,  0.0,
            0.0,  1.0,  0.0,
            0.0,  0.0,  1.0,
            
            0.0, 0.0, 100.0,
            0.0, 200.0,
            1.0,
            0.0, 0.0, 0.0
        ],
        'neg_z': [
            1.0,  0.0,  0.0,
            0.0,  1.0,  0.0,
            0.0,  0.0, -1.0,
            
            0.0, 0.0, -100.0,
            0.0, 200.0,
            1.0,
            0.0, 0.0, 0.0
        ],
    }

    prefix = runtime.temp_path(prefix)
    for direction, view in directions.items():
        pymol.cmd.set_view(view)
        pymol.cmd.zoom(selection, buffer=0.5)  
        
        filename = f"{prefix}_{direction}.png"
        pymol.cmd.png(filename, width=width, height=height, dpi=dpi, ray=ray)
        print(f"Saved {filename}")
    return None

def save_6view_pdf(image_prefix='tmp.csubst.pymol', 
                   directions=None, 
                   pdf_filename='6view.pdf'):
    """
    Combines the 6 saved view images into a single PDF with 2 columns and 3 rows.

    Parameters
    ----------
    image_prefix : str
        Common prefix used when saving the 6 PNG images with save_six_views.
    directions : list of str
        List of the view directions in the order you want them arranged.
        Default is ['pos_x','neg_x','pos_y','neg_y','pos_z','neg_z'].
    pdf_filename : str
        Name of the output PDF file.
    """
    image_prefix = runtime.temp_path(image_prefix)
    if directions is None:
        directions = ['pos_x','neg_x','pos_y','neg_y','pos_z','neg_z']

    # Create a figure with 3 rows & 2 columns
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7.2, 9.7))

    for idx, direction in enumerate(directions):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Construct filename for each view image
        img_file = f"{image_prefix}_{direction}.png"
        
        if not os.path.isfile(img_file):
            print(f"Warning: {img_file} not found. Skipping.")
            ax.axis('off')
            ax.set_title(f"{direction} (missing)")
            continue
        
        # Read and show image
        img = plt.imread(img_file)
        ax.imshow(img)
        
        # Remove axes ticks
        ax.axis('off')
        
        # Optionally label each subplot
        ax.set_title(direction)

    plt.tight_layout()
    plt.savefig(pdf_filename)
    plt.close(fig)
    print(f"Saved 6-view PDF as {pdf_filename}")
    return None

def get_num_chain():
    """
    Get the number of chains in the PDB file.
    """
    chains = pymol.cmd.get_chains('all')
    num_chain = len(chains)
    return num_chain

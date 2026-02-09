import numpy
import matplotlib.pyplot
import pandas
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
from csubst import utility

def initialize_pymol(pdb_id):
    #pymol.pymol_argv = ['pymol','-qc']
    #pymol.finish_launching()
    pymol.cmd.do('delete all')
    is_old_pdb_code = bool(re.fullmatch('[0-9][A-Za-z0-9]{3}', pdb_id))
    is_new_pdb_code = bool(re.fullmatch('pdb_[0-9]{5}[A-Za-z0-9]{3}', pdb_id))
    if is_old_pdb_code|is_new_pdb_code:
        print('Fetching PDB code {}. Internet connection is needed.'.format(pdb_id), flush=True)
        pymol.cmd.do('fetch {}'.format(pdb_id))
    else:
        print('Loading PDB file: {}'.format(pdb_id), flush=True)
        pymol.cmd.load(pdb_id)

def write_mafft_alignment(g):
    tmp_pdb_fasta = 'tmp.csubst.pdb_seq.fa'
    mafft_map_file = tmp_pdb_fasta+'.map'
    if os.path.exists(mafft_map_file):
        os.remove(mafft_map_file)
    pdb_seq = pymol.cmd.get_fastastr(selection='polymer.protein', state=-1, quiet=1)
    with open(tmp_pdb_fasta, 'w') as f:
        f.write(pdb_seq)
    sequence.write_alignment(outfile='tmp.csubst.leaf.aa.fa', mode='aa', g=g, leaf_only=True)
    cmd_mafft = [g['mafft_exe'], '--keeplength', '--mapout', '--quiet',
                 '--thread', '1',]
    if g['mafft_op'] >= 0:
        cmd_mafft += ['--op', str(g['mafft_op']),]
    if g['mafft_ep'] >= 0:
        cmd_mafft += ['--ep', str(g['mafft_ep']),]
    cmd_mafft += ['--add', tmp_pdb_fasta, 'tmp.csubst.leaf.aa.fa',]
    print('Running MAFFT to align the PDB sequence with the input alignment.', flush=True)
    print('Command: {}'.format(' '.join(cmd_mafft)), flush=True)
    out_mafft = subprocess.run(cmd_mafft, stdout=subprocess.PIPE)
    with open(g['mafft_add_fasta'], 'w') as f:
        f.write(out_mafft.stdout.decode('utf8'))
    print('')
    for i in range(10):
        if os.path.exists(mafft_map_file):
            print('MAFFT alignment file was generated: {}'.format(g['mafft_add_fasta']), flush=True)
            break
        else:
            print('MAFFT alignment file not detected. Waiting {:} sec'.format(i+1), flush=True)
            time.sleep(1)
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
    if os.path.getsize(g['mafft_add_fasta'])==0:
        sys.stderr.write('File size of {} is 0. A wrong ID might be specified in --pdb.\n'.format(g['mafft_add_fasta']))
        sys.stderr.write('Exiting.\n')
        sys.exit(1)

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
            residue_iloc = numpy.arange(len(residue_numbers)) + 1
            col1 = 'codon_site_'+object_name+'_'+ch
            col2 = 'codon_site_pdb_'+object_name+'_'+ch
            dict_tmp = {col1:residue_iloc, col2:residue_numbers}
            df_tmp = pandas.DataFrame(dict_tmp)
            out[object_name+'_'+ch] = df_tmp
    return out

def add_pdb_residue_numbering(df):
    residue_numberings = get_residue_numberings()
    object_names = pymol.cmd.get_names()
    for object_name in object_names:
        for ch in pymol.cmd.get_chains(object_name):
            key = object_name+'_'+ch
            if residue_numberings[key].shape[0]==0:
                continue
            if 'codon_site_'+key in df.columns:
                df = pandas.merge(df, residue_numberings[key], on='codon_site_'+key, how='left')
                df['codon_site_pdb_'+key] = df['codon_site_pdb_'+key].fillna(0).astype(int)
    print('The column "codon_site_**ID**" indicates the positions of codons/amino acids in the sequence "**ID**" in the input alignment. 0 = missing site.')
    print('The column "codon_site_pdb_**ID**" indicates the positions of codons/amino acids in the sequence "**ID**" in the PDB file. 0 = missing site.')
    return df

def add_coordinate_from_mafft_map(df, mafft_map_file='tmp.csubst.pdb_seq.fa.map'):
    print('Loading amino acid coordinates from: {}'.format(mafft_map_file), flush=True)
    with open(mafft_map_file, 'r') as f:
        map_str = f.read()
    map_list = map_str.split('>')[1:]
    for map_item in map_list:
        seq_name = re.sub('\n.*', '', map_item)
        seq_csv = re.sub(seq_name+'\n', '', map_item)
        if seq_csv.count('\n')==1: # empty data
            df.loc[:,'codon_site_'+seq_name] = 0
            df.loc[:,'aa_'+seq_name] = 0
        else:
            df_tmp = pandas.read_csv(StringIO(seq_csv), comment='#', header=None)
            df_tmp.columns = ['aa_'+seq_name, 'codon_site_'+seq_name, 'codon_site_alignment']
            is_missing_in_aln = (df_tmp['codon_site_alignment']==' -')
            df_tmp = df_tmp.loc[~is_missing_in_aln,:]
            df_tmp['codon_site_alignment'] = df_tmp['codon_site_alignment'].astype(int)
            df = pandas.merge(df, df_tmp, on='codon_site_alignment', how='left')
            df['codon_site_'+seq_name] = df['codon_site_'+seq_name].fillna(0).astype(int)
            df['aa_'+seq_name] = df.loc[:,'aa_'+seq_name].fillna('')
    return df

def add_coordinate_from_user_alignment(df, user_alignment):
    print('Loading amino acid coordinates from: {}'.format(user_alignment), flush=True)
    pdb_fasta = pymol.cmd.get_fastastr(selection='polymer.protein', state=-1, quiet=1)
    tmp_pdb_fasta = 'tmp.csubst.pdb_seq.fa'
    with open(tmp_pdb_fasta, 'w') as f:
        f.write(pdb_fasta)
    pdb_seqs = list(SeqIO.parse(open(tmp_pdb_fasta, 'r'), 'fasta'))
    user_seqs = list(SeqIO.parse(open(user_alignment, 'r'), 'fasta'))
    for user_seq in user_seqs:
        for pdb_seq in pdb_seqs:
            if user_seq.name!=pdb_seq.name:
                continue
            user_seq_str = str(user_seq.seq).replace('\n', '')
            pdb_seq_str = str(pdb_seq.seq).replace('\n', '')
            user_seq_counter = 0
            pdb_seq_counter = 0
            txt = 'The alignment length should match between --alignment_file ({} sites) and --user_alignment ({} sites)'
            assert len(user_seq_str)==df.shape[0], txt.format(df.shape[0], len(user_seq_str))
            df['aa_' + user_seq.name] = ''
            df['codon_site_' + user_seq.name] = 0
            while user_seq_counter <= df.shape[0]-1:
                if user_seq_str[user_seq_counter]=='-':
                    user_seq_counter += 1
                    continue
                if user_seq_str[user_seq_counter]==pdb_seq_str[pdb_seq_counter]:
                    df.at[user_seq_counter, 'aa_' + user_seq.name] = user_seq_str[user_seq_counter]
                    df.at[user_seq_counter, 'codon_site_' + user_seq.name] = pdb_seq_counter + 1
                    user_seq_counter += 1
                    pdb_seq_counter += 1
                else:
                    pdb_seq_counter += 1
    return df

def calc_aa_identity(g):
    seqs = sequence.read_fasta(path=g['mafft_add_fasta'])
    seqnames = list(seqs.keys())
    pdb_base = re.sub('\..*', '', os.path.basename(g['pdb']))
    pdb_seqnames = [ sn for sn in seqnames if sn.startswith(pdb_base) ]
    other_seqnames = [ sn for sn in seqnames if not sn.startswith(pdb_base) ]
    aa_identity_values = dict()
    for pdb_seqname in pdb_seqnames:
        aa_identity_values[pdb_seqname] = []
        for other_seqname in other_seqnames:
            aa_identity = sequence.calc_identity(seq1=seqs[pdb_seqname], seq2=seqs[other_seqname])
            aa_identity_values[pdb_seqname].append(aa_identity)
        aa_identity_values[pdb_seqname] = numpy.array(aa_identity_values[pdb_seqname])
    aa_identity_means = dict()
    for pdb_seqname in pdb_seqnames:
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
    g = calc_aa_identity(g)
    pdb_seqnames = list(g['aa_identity_means'].keys())
    nucleic_chains = pymol.cmd.get_chains('polymer.nucleic')
    colors = ['wheat','slate','salmon','brightorange','violet','olive',
              'firebrick','pink','marine','density','cyan','chocolate','teal',]
    num_chains = len(pdb_seqnames) + len(nucleic_chains)
    if num_chains > len(colors):
        colors *= int(num_chains/len(colors)) + 1 # for supercomplex
    for nucleotide in ['DG','DT','DA','DC']: # DNA
        pymol.cmd.do('color pink, resn '+nucleotide)
    if len(pdb_seqnames)==1:
        return None
    max_aa_identity_mean = max(g['aa_identity_means'].values())
    for pdb_seqname in pdb_seqnames:
        aa_identity_mean = g['aa_identity_means'][pdb_seqname]
        if abs(max_aa_identity_mean-aa_identity_mean)<g['float_tol']:
            max_pdb_seqname = pdb_seqname
            max_spans = g['aa_spans'][pdb_seqname]
            break
    i = 0
    for pdb_seqname in pdb_seqnames:
        if pdb_seqname==max_pdb_seqname:
            continue
        spans = g['aa_spans'][pdb_seqname]
        is_nonoverlapping_N_side = (max_spans[1] < spans[0])
        is_nonoverlapping_C_side = (max_spans[0] > spans[1])
        if (is_nonoverlapping_N_side|is_nonoverlapping_C_side):
            continue
        chain = pdb_seqname.replace(g['pdb']+'_', '')
        print('Masking chain {}'.format(chain), flush=True)
        if spans[1]==0: # End position = 0 if no protein in the chain
            continue
        pymol.cmd.do('color {}, chain {} and polymer.protein'.format(colors[i], chain))
        i += 1
    for chain in nucleic_chains:
        print('Masking chain {}'.format(chain), flush=True)
        pymol.cmd.do('color {}, chain {} and polymer.nucleic'.format(colors[i], chain))
        i += 1

def set_color_gray(object_names, residue_numberings, gray_value):
    for object_name in object_names:
        if object_name.endswith('_pol_conts'):
            continue
        for ch in pymol.cmd.get_chains(object_name):
            key = object_name+'_'+ch
            codon_site_pdb = residue_numberings[key].loc[:,'codon_site_pdb_'+key]
            is_nonzero = (codon_site_pdb!=0)
            residue_start = codon_site_pdb.loc[is_nonzero].min()
            residue_end = codon_site_pdb.loc[is_nonzero].max()
            cmd_color = "color gray{}, {} and chain {} and resi {:}-{:}"
            pymol.cmd.do(cmd_color.format(gray_value, object_name, ch, residue_start, residue_end))


def _paint_sites_with_hex(object_name, chain_id, sites, hex_value, label):
    if len(sites)==0:
        print('Skipping site painting. No amino acid substitutions: {}'.format(label), flush=True)
        return None
    sites = sorted(list(set([int(site) for site in sites])))
    print('Amino acid sites with {} will be painted with {}.'.format(label, hex_value), flush=True)
    txt_resi = '+'.join([str(site) for site in sites])
    cmd_color = "color {}, {} and chain {} and resi {}"
    pymol.cmd.do(cmd_color.format(hex_value, object_name, chain_id, txt_resi))
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


def _get_lineage_hex_by_branch(branch_ids):
    if len(branch_ids)==1:
        return {int(branch_ids[0]): utility.rgb_to_hex(r=1.0, g=0.0, b=0.0)}
    cmap = matplotlib.pyplot.get_cmap('rainbow')
    out = dict()
    for i,branch_id in enumerate(branch_ids):
        rgba = cmap(i/(len(branch_ids)-1))
        out[int(branch_id)] = utility.rgb_to_hex(r=rgba[0], g=rgba[1], b=rgba[2])
    return out


def set_substitution_colors(df, g, object_names, N_sub_cols):
    for object_name in object_names:
        if object_name.endswith('_pol_conts'):
            continue
        for ch in pymol.cmd.get_chains(object_name):
            codon_site_col = 'codon_site_pdb_'+object_name+'_'+ch
            if not codon_site_col in df.columns:
                continue
            mode = str(g.get('mode', 'intersection')).lower()
            if mode=='set':
                if 'N_set_expr' not in df.columns:
                    print('Skipping site painting. N_set_expr column was not found for --mode set.', flush=True)
                    continue
                selected_sites = []
                for i in df.index:
                    codon_site = int(df.at[i,codon_site_col])
                    if codon_site==0:
                        continue
                    if bool(df.at[i,'N_set_expr']):
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
                for i in df.index:
                    codon_site = int(df.at[i,codon_site_col])
                    if codon_site==0:
                        continue
                    total_sub = float(df.loc[i,N_sub_cols].sum())
                    if total_sub < g['pymol_min_single_prob']:
                        continue
                    site_values[codon_site] = max(site_values.get(codon_site, 0.0), total_sub)
                _paint_intensity_sites(
                    object_name=object_name,
                    chain_id=ch,
                    site_values=site_values,
                    label='total_N',
                )
                continue
            if mode=='lineage':
                color_policy = str(g.get('color', 'stratigraphy')).lower()
                if color_policy=='count':
                    site_values = dict()
                    for i in df.index:
                        codon_site = int(df.at[i,codon_site_col])
                        if codon_site==0:
                            continue
                        total_sub = float(df.loc[i,N_sub_cols].sum())
                        if total_sub < g['pymol_min_single_prob']:
                            continue
                        site_values[codon_site] = max(site_values.get(codon_site, 0.0), total_sub)
                    _paint_intensity_sites(
                        object_name=object_name,
                        chain_id=ch,
                        site_values=site_values,
                        label='lineage_count_N',
                    )
                else:
                    lineage_branch_ids = [int(bid) for bid in numpy.asarray(g['branch_ids']).tolist()]
                    branch_hex = _get_lineage_hex_by_branch(lineage_branch_ids)
                    site_by_branch = {bid: [] for bid in lineage_branch_ids}
                    for i in df.index:
                        codon_site = int(df.at[i,codon_site_col])
                        if codon_site==0:
                            continue
                        vals = df.loc[i,N_sub_cols].to_numpy(dtype=float)
                        if vals.shape[0]==0:
                            continue
                        max_value = vals.max()
                        if max_value < g['pymol_min_single_prob']:
                            continue
                        max_index = int(vals.argmax())
                        branch_id = lineage_branch_ids[min(max_index, len(lineage_branch_ids)-1)]
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
            for i in df.index:
                codon_site = int(df.at[i,codon_site_col])
                prob_Nany2spe = df.at[i,'OCNany2spe']
                prob_Nany2dif = df.at[i,'OCNany2dif']
                prob_single_sub = df.loc[i,N_sub_cols].max()
                if codon_site==0:
                    continue
                elif (prob_Nany2spe>=g['pymol_min_combinat_prob'])&(prob_Nany2dif<=prob_Nany2spe):
                    color_sites['OCNany2spe'].append(codon_site)
                elif (prob_Nany2dif>=g['pymol_min_combinat_prob'])&(prob_Nany2dif>prob_Nany2spe):
                    color_sites['OCNany2dif'].append(codon_site)
                elif (prob_single_sub>=g['pymol_min_single_prob']):
                    color_sites['single_sub'].append(codon_site)
            if g['single_branch_mode']:
                color_sites['single_branch_N'] = copy.deepcopy(color_sites['OCNany2spe'])
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
                    cmd_tp = "set transparency, 0.3, {} and chain {} and resi {:}"
                    pymol.cmd.do(cmd_tp.format(object_name, ch, txt_resi))

def write_pymol_session(df, g):
    df = df.reset_index(drop=True)
    pymol.cmd.do('set seq_view, 1')
    if g['remove_solvent']:
        pymol.cmd.do("remove solvent")
    if g['remove_ligand']:
        molecule_codes = g['remove_ligand'].split(',')
        for molecule_code in molecule_codes:
            pymol.cmd.do("remove resn "+molecule_code)
    pymol.cmd.do("preset.ligand_sites_trans_hq(selection='all')")
    pymol.cmd.do("hide wire")
    pymol.cmd.do("hide ribbon")
    pymol.cmd.do("show cartoon")
    pymol.cmd.do("show surface")
    pymol.cmd.do("set transparency, {}, polymer.protein".format(g['pymol_transparency']))
    object_names = pymol.cmd.get_names()
    #residue_numberings = get_residue_numberings()
    #set_color_gray(object_names, residue_numberings, gray_value=g['pymol_gray'])
    pymol.cmd.do("color gray{}, polymer.protein".format(g['pymol_gray']))
    pymol.cmd.do('util.cbag organic')
    N_sub_cols = df.columns[df.columns.str.startswith('N_sub_')]
    set_substitution_colors(df, g, object_names, N_sub_cols)
    if g['mask_subunit']:
        mask_subunit(g)
    pymol.cmd.do('zoom')
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
    if directions is None:
        directions = ['pos_x','neg_x','pos_y','neg_y','pos_z','neg_z']

    # Create a figure with 3 rows & 2 columns
    fig, axes = matplotlib.pyplot.subplots(nrows=3, ncols=2, figsize=(7.2, 9.7))

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
        img = matplotlib.image.imread(img_file)
        ax.imshow(img)
        
        # Remove axes ticks
        ax.axis('off')
        
        # Optionally label each subplot
        ax.set_title(direction)

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(pdf_filename)
    matplotlib.pyplot.close(fig)
    print(f"Saved 6-view PDF as {pdf_filename}")
    return None

def get_num_chain():
    """
    Get the number of chains in the PDB file.
    """
    chains = pymol.cmd.get_chains('all')
    num_chain = len(chains)
    return num_chain

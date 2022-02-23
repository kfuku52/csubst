import numpy
import pandas
import pymol

from io import StringIO
#import itertools
import os
import re
import subprocess
import sys
import time

from csubst import sequence
from csubst import utility

def initialize_pymol(g):
    #pymol.pymol_argv = ['pymol','-qc']
    #pymol.finish_launching()
    pymol.cmd.do('delete all')
    is_old_pdb_code = bool(re.fullmatch('[0-9][A-Za-z0-9]{3}', g['pdb']))
    is_new_pdb_code = bool(re.fullmatch('pdb_[0-9]{5}[A-Za-z0-9]{3}', g['pdb']))
    if is_old_pdb_code|is_new_pdb_code:
        print('Fetching PDB code {}. Internet connection is needed.'.format(g['pdb']), flush=True)
        pymol.cmd.do('fetch {}'.format(g['pdb']))
    else:
        print('Loading PDB file: {}'.format(g['pdb']), flush=True)
        pymol.cmd.load(g['pdb'])

def write_mafft_map(g):
    tmp_pdb_fasta = 'tmp.csubst.pdb_seq.fa'
    mafft_map_file = tmp_pdb_fasta+'.map'
    if os.path.exists(mafft_map_file):
        os.remove(mafft_map_file)
    pdb_seq = pymol.cmd.get_fastastr(selection='polymer.protein', state=-1, quiet=1)
    with open(tmp_pdb_fasta, 'w') as f:
        f.write(pdb_seq)
    sequence.write_alignment(outfile='tmp.csubst.leaf.aa.fa', mode='aa', g=g, leaf_only=True)
    cmd_mafft = [g['mafft_exe'], '--keeplength', '--mapout', '--quiet',
                 '--thread', '1',
                 '--op', str(g['mafft_op']),
                 '--ep', str(g['mafft_ep']),
                 '--add', tmp_pdb_fasta,
                 'tmp.csubst.leaf.aa.fa',
                 ]
    out_mafft = subprocess.run(cmd_mafft, stdout=subprocess.PIPE)
    with open(g['mafft_add_fasta'], 'w') as f:
        f.write(out_mafft.stdout.decode('utf8'))
    for i in range(10):
        if os.path.exists(mafft_map_file):
            print('mafft map file was generated.', flush=True)
            break
        else:
            print('mafft map file not detected. Waiting {:} sec'.format(i+1), flush=True)
            time.sleep(1)
    txt = 'CSUBST does not exclude poorly aligned regions. ' \
          'Please carefully check {} before biological interpretation of substitution events.'
    print(txt.format(g['mafft_add_fasta']), flush=True)
    if os.path.getsize(g['mafft_add_fasta'])==0:
        sys.stderr.write('File size of {} is 0. A wrong ID might be specified in --pdb.\n'.format(g['mafft_add_fasta']))
        sys.stderr.write('Exiting.\n')
        sys.exit(1)

def get_residue_numberings():
    out = dict()
    object_names = pymol.cmd.get_names()
    for object_name in object_names:
        if object_name.endswith('_pol_conts'):
            continue
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
                sys.stderr.write('PDB protein sequence is unavailable: {}\n'.format(key))
                continue
            df = pandas.merge(df, residue_numberings[key], on='codon_site_'+key, how='left')
            df.loc[:,'codon_site_pdb_'+key] = df.loc[:,'codon_site_pdb_'+key].fillna(0).astype(int)
    return df

def add_mafft_map(df, mafft_map_file='tmp.csubst.pdb_seq.fa.map'):
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
            is_missing_in_aln = (df_tmp.loc[:,'codon_site_alignment']==' -')
            df_tmp = df_tmp.loc[~is_missing_in_aln,:]
            df_tmp.loc[:,'codon_site_alignment'] = df_tmp.loc[:,'codon_site_alignment'].astype(int)
            df = pandas.merge(df, df_tmp, on='codon_site_alignment', how='left')
            df.loc[:,'codon_site_'+seq_name] = df.loc[:,'codon_site_'+seq_name].fillna(0).astype(int)
            df.loc[:,'aa_'+seq_name] = df.loc[:,'aa_'+seq_name].fillna('')
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
        aa_start = min(alphabet_sites)
        aa_end = max(alphabet_sites)
        aa_ranges[pdb_seqname] = [aa_start, aa_end]
    g['aa_identity_values'] = aa_identity_values
    g['aa_identity_means'] = aa_identity_means
    g['aa_spans'] = aa_ranges
    return g

def mask_subunit(g):
    colors = ['wheat','slate','salmon','brightorange','violet','olive',
              'firebrick','pink','marine','density','cyan','chocolate','teal',]
    colors *= 10 # for supercomplex
    g = calc_aa_identity(g)
    pdb_seqnames = list(g['aa_identity_means'].keys())
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
        pymol.cmd.do('color {}, chain {} and polymer.protein'.format(colors[i], chain))
        i += 1
    for chain in pymol.cmd.get_chains(selection='polymer.nucleic'):
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

def set_substitution_colors(df, g, object_names, N_sub_cols):
    for object_name in object_names:
        if object_name.endswith('_pol_conts'):
            continue
        for ch in pymol.cmd.get_chains(object_name):
            codon_site_col = 'codon_site_pdb_'+object_name+'_'+ch
            if not codon_site_col in df.columns:
                continue
            color_sites = dict()
            color_sites['Nany2spe'] = []
            color_sites['Nany2dif'] = []
            color_sites['single_sub'] = []
            for i in df.index:
                codon_site = df.at[i,codon_site_col]
                prob_Nany2spe = df.at[i,'Nany2spe']
                prob_Nany2dif = df.at[i,'Nany2dif']
                prob_single_sub = df.loc[i,N_sub_cols].max()
                if codon_site==0:
                    continue
                elif (prob_Nany2spe>=g['pymol_min_combinat_prob'])&(prob_Nany2dif<=prob_Nany2spe):
                    color_sites['Nany2spe'].append(codon_site)
                elif (prob_Nany2dif>=g['pymol_min_combinat_prob'])&(prob_Nany2dif>prob_Nany2spe):
                    color_sites['Nany2dif'].append(codon_site)
                elif (prob_single_sub>=g['pymol_min_single_prob']):
                    color_sites['single_sub'].append(codon_site)
            for key in color_sites.keys():
                if key=='Nany2spe':
                    hex_value = utility.rgb_to_hex(r=1, g=0, b=0)
                elif key=='Nany2dif':
                    hex_value = utility.rgb_to_hex(r=0, g=0, b=1)
                elif key=='single_sub':
                    hex_value = utility.rgb_to_hex(r=0.4, g=0.4, b=0.4)
                print('Amino acid sites with {} will be painted with {}.'.format(key, hex_value), flush=True)
                txt_resi = '+'.join([str(site) for site in color_sites[key]])
                cmd_color = "color {}, {} and chain {} and resi {}"
                pymol.cmd.do(cmd_color.format(hex_value, object_name, ch, txt_resi))
                if key in ['Nany2spe','Nany2dif']:
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
    pymol.cmd.do("set transparency, {}".format(g['pymol_transparency']))
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

import numpy
import pandas
import pymol

from io import StringIO
import os
import re
import subprocess
import sys
import time

from csubst import sequence

def initialize_pymol(g):
    pymol.pymol_argv = ['pymol','-qc']
    pymol.finish_launching()
    pymol.cmd.do('delete all')
    is_old_pdb_code = bool(re.search('[0-9][A-Za-z0-9]{3}', g['pdb']))
    is_new_pdb_code = bool(re.search('pdb_[0-9]{5}[A-Za-z0-9]{3}', g['pdb']))
    if is_old_pdb_code|is_new_pdb_code:
        print('Fetching PDB code {}. Internet connection is needed.'.format(g['pdb']), flush=True)
        pymol.cmd.do('fetch {}'.format(g['pdb']))
    else:
        print('Loading PDB file: {}'.format(g['pdb']), flush=True)
        pymol.cmd.load(g['pdb'])

def write_mafft_map(g, mafft_add_fasta='csubst_site.pdb_add.aa.fa'):
    tmp_pdb_fasta = 'tmp.csubst.pdb_seq.fa'
    mafft_map_file = tmp_pdb_fasta+'.map'
    if os.path.exists(mafft_map_file):
        os.remove(mafft_map_file)
    pdb_seq = pymol.cmd.get_fastastr(selection='all', state=-1, quiet=1)
    with open(tmp_pdb_fasta, 'w') as f:
        f.write(pdb_seq)
    sequence.write_alignment(outfile='tmp.csubst.leaf.aa.fa', mode='aa', g=g, leaf_only=True)
    cmd_mafft = [g['mafft_exe'], '--keeplength', '--mapout', '--quiet',
                 '--thread', '1',
                 '--add', tmp_pdb_fasta,
                 'tmp.csubst.leaf.aa.fa',
                 ]
    out_mafft = subprocess.run(cmd_mafft, stdout=subprocess.PIPE)
    with open(mafft_add_fasta, 'w') as f:
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
    print(txt.format(mafft_add_fasta), flush=True)

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
            residue_numbers = [ int(x) for x in pymol.stored.residues ]
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
        df_tmp = pandas.read_csv(StringIO(seq_csv), comment='#', header=None)
        df_tmp.columns = ['aa_'+seq_name, 'codon_site_'+seq_name, 'codon_site_alignment']
        is_missing_in_aln = (df_tmp.loc[:,'codon_site_alignment']==' -')
        df_tmp = df_tmp.loc[~is_missing_in_aln,:]
        df_tmp.loc[:,'codon_site_alignment'] = df_tmp.loc[:,'codon_site_alignment'].astype(int)
        df = pandas.merge(df, df_tmp, on='codon_site_alignment', how='left')
        df.loc[:,'codon_site_'+seq_name] = df.loc[:,'codon_site_'+seq_name].fillna(0).astype(int)
        df.loc[:,'aa_'+seq_name] = df.loc[:,'aa_'+seq_name].fillna('')
    return df

def rgb_to_hex(r,g,b):
    rgb = [r,g,b]
    for i in range(len(rgb)):
        assert rgb[i]<=1
        rgb[i] = int(numpy.round(rgb[i]*255, decimals=0))
    hex_color = '0x%02X%02X%02X' % (rgb[0],rgb[1],rgb[2])
    return hex_color

def write_pymol_session(df, session_file, g):
    pymol.cmd.do('set seq_view, 1')
    if g['remove_solvent']:
        pymol.cmd.do("remove solvent")
        pymol.cmd.do("remove resn so4")
        pymol.cmd.do("remove resn po4")
        pymol.cmd.do("remove resn bme")
        pymol.cmd.do("remove resn edo")
        pymol.cmd.do("remove resn dyd")
    pymol.cmd.do("preset.ligand_sites_trans_hq(selection='all')")
    pymol.cmd.do("hide wire")
    pymol.cmd.do("hide ribbon")
    pymol.cmd.do("show cartoon")
    pymol.cmd.do("show surface")
    pymol.cmd.do("set transparency, 0.65")
    residue_numberings = get_residue_numberings()
    object_names = pymol.cmd.get_names()
    for object_name in object_names:
        if object_name.endswith('_pol_conts'):
            continue
        for ch in pymol.cmd.get_chains(object_name):
            key = object_name+'_'+ch
            codon_site_pdb = residue_numberings[key].loc[:,'codon_site_pdb_'+key]
            is_nonzero = (codon_site_pdb!=0)
            residue_start = codon_site_pdb.loc[is_nonzero].min()
            residue_end = codon_site_pdb.loc[is_nonzero].max()
            cmd_color = "color gray80, {} and chain {} and resi {:}-{:}"
            pymol.cmd.do(cmd_color.format(object_name, ch, residue_start, residue_end))
    df = df.reset_index(drop=True)
    N_sub_cols = df.columns[df.columns.str.startswith('N_sub_')]
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
                    hex_value = rgb_to_hex(r=1, g=0, b=0)
                elif key=='Nany2dif':
                    hex_value = rgb_to_hex(r=0, g=0, b=1)
                elif key=='single_sub':
                    hex_value = rgb_to_hex(r=0.4, g=0.4, b=0.4)
                print('Amino acid sites with {} will be painted with {}.'.format(key, hex_value), flush=True)
                txt_resi = '+'.join([str(site) for site in color_sites[key]])
                cmd_color = "color {}, {} and chain {} and resi {}"
                pymol.cmd.do(cmd_color.format(hex_value, object_name, ch, txt_resi))
                if key in ['Nany2spe','Nany2dif']:
                    cmd_tp = "set transparency, 0.3, {} and chain {} and resi {:}"
                    pymol.cmd.do(cmd_tp.format(object_name, ch, txt_resi))
    pymol.cmd.do('zoom')
    pymol.cmd.deselect()
    pymol.cmd.save(session_file)

def quit_pymol():
    pymol.cmd.quit(code=0)
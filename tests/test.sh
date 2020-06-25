#!/usr/bin/env bash

pip install '/Users/kef74yk/Dropbox_w/repos/csubst'

tree_name='OG0002332'
dir_script="/Users/kef74yk/Dropbox_w/script"
dir_iqtree="/Users/kef74yk/Dropbox_w/downloaded_programs/iqtree-1.6.10-MacOSX/bin"
dir_work="/Users/kef74yk/Dropbox_w/repos/csubst/tests"
dir_data="/Users/kef74yk/Dropbox_w/repos/csubst/data"
file_aln=${dir_data}/${tree_name}/${tree_name}".cds.trimal.fasta"
file_tree=${dir_data}/${tree_name}/${tree_name}".root.nosupport.nwk"
file_fg=${dir_data}/${tree_name}/${tree_name}".foreground.txt"
NSLOTS=4
cd ${dir_work}

mkdir -p ${dir_work}/csubst_out; cd $_

csubst \
--max_arity 2 \
--nslots ${NSLOTS} \
--ncbi_codon_table 1 \
--infile_dir ${dir_data}/${tree_name} \
--infile_type iqtree \
--aln_file ${file_aln} \
--tre_file ${file_tree} \
--calc_omega yes \
--ml_anc yes \
--b yes \
--s yes \
--cs no \
--cb yes \
--bs yes \
--cbs no \
--target_stat omega_any2spe_asrv \
--min_stat 3 \
--min_branch_sub 1 \
--min_Nany2spe 5 \
--exclude_sisters yes \
--num_subsample 10000 \
--omega_method permutation \
--fg_exclude_wg no \
--foreground ${file_fg}




#python ${dir_script}/delete_internal_node_name_and_support.py \
#${dir_work}/${file_tree} \
#${dir_work}/${file_tree}.nosupport

#${dir_iqtree}/iqtree \
#-s ${file_aln} \
#-te ${file_tree}.nosupport \
#-m MGK+F3X4+R2 \
#-st CODON1 \
#-nt ${NSLOTS} \
#-asr \
#-redo
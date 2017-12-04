#!/usr/bin/env python

# omega_asrv can be larger than omega_flat when,
# for example, ASRV is less biased in nonsynonymous substitutions
# for example, # nonsynonymous substs are largely different between branch 1 and branch 2

import os, ete3, numpy, pandas, sys, time, itertools, multiprocessing, joblib, collections, argparse
pandas.options.display.max_rows=10
pandas.options.display.max_columns=100

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='%(prog)s 0.1') # version
parser.add_argument('--arity', metavar='INTEGER', default=2, type=int, help='The combinatorial number of branches. Set 2 for paired substitutions.')
parser.add_argument('--nslots', metavar='INTEGER',default=1, type=int, help='The number of processors for parallel computations.')
parser.add_argument('--codon_file', metavar='PATH',type=str, required=True, help='PATH to the codon table file.')
parser.add_argument('--infile_dir', metavar='PATH',type=str, required=True, help='PATH to the input file directory.')
parser.add_argument('--infile_type', metavar='[phylobayes,foo,bar]', default='phylobayes', type=str, help='The input file format. Only PhyloBayes is supported currently.')
parser.add_argument('--min_pp', metavar='FLOAT', default=0, type=float, help='The minimum posterior probability of single substitutions to count. Set 0 for a full Bayesian counting without binarization.')
args = parser.parse_args()

if not args.infile_dir.endswith('/'):
    args.infile_dir = args.infile_dir+'/'

def get_codon_table(codon_file):
    f = open(codon_file)
    lines = f.readlines()
    f.close()
    codon_table = []
    for line in lines:
        line_split = line.replace("\n", "").split(" ")
        aa = line_split[0]
        codon = line_split[1]
        codon_table.append([aa, codon])
    return(codon_table)

def get_tree(infile_dir, infile_type):
    print("Reading the input tree...", flush=True)
    if infile_type=='phylobayes':
        phylobayes_files = os.listdir(infile_dir)
        sample_labels = [ file for file in phylobayes_files if "_sample.labels" in file ][0]
        tree = ete3.PhyloNode(infile_dir + sample_labels, format=1)
    else:
        print('--infile_type', infile_type, 'is not currently supported.')
        sys.exit()
    return(tree)

def add_numerical_node_labels(tree):
    all_leaf_names = tree.get_leaf_names()
    all_leaf_names.sort()
    leaf_numerical_labels = dict()
    power = 0
    for i in range(0, len(all_leaf_names)):
        leaf_numerical_labels[all_leaf_names[i]] = 2**i
    numerical_labels = list()
    for node in tree.traverse():
        leaf_names = node.get_leaf_names()
        numerical_labels.append(sum([leaf_numerical_labels[leaf_name] for leaf_name in leaf_names]))
    argsort_labels = numpy.argsort(numerical_labels)
    short_labels = numpy.arange(len(argsort_labels))
    i=0
    for node in tree.traverse():
        node.numerical_label = short_labels[argsort_labels==i][0]
        i+=1
    return(tree)

def get_node_phylobayes_out(node, files):
    if node.is_leaf():
        pp_file = [ file for file in files if file.find(node.name+"_"+node.name+".ancstatepostprob") > -1 ]
    else:
        pp_file = [ file for file in files if file.find(".ancstatepostprob") > -1 ]
        pp_file = [ file for file in pp_file if file.find("_sample_"+node.name+"_") > -1 ]
    return(pp_file)

def get_pp_nuc(phylobayes_dir, pp_file):
    pp_nuc = pandas.read_csv(phylobayes_dir+pp_file, sep="\t", index_col=False, header=0)
    pp_nuc = pp_nuc.iloc[:,2:]
    if len(pp_nuc) % 3 != 0:
        print('The nucleotide sequence is not multiple of 3.', flush=True)
        sys.exit()
    return(pp_nuc)

def get_pp_cdn(pp_nuc, codon_table):
    codon_columns = pp_nuc.columns.values.reshape((-1,1,1)) + pp_nuc.columns.values.reshape((1,-1,1)) + pp_nuc.columns.values.reshape((1,1,-1))
    codon_columns = codon_columns.ravel()
    aa_columns = []
    for codon_column in codon_columns:
        aa_columns.append([ codon[0] for codon in codon_table if codon[1]==codon_column ][0])
    num_codon = int(len(pp_nuc.index)/3)
    columns = pandas.MultiIndex.from_arrays([codon_columns, aa_columns], names=["codon", "aa"])
    pp_cdn = pandas.DataFrame(0, index=range(0, num_codon), columns=columns)
    for i in pp_cdn.index:
        pp_codon = pp_nuc.loc[i*3:i*3, : ].values.reshape((-1,1,1)) * pp_nuc.loc[i*3+1:i*3+1, : ].values.reshape((1,-1,1)) * pp_nuc.loc[i*3+2:i*3+2, : ].values.reshape((1,1,-1))
        pp_cdn.loc[i,:] = pp_codon.ravel()
    is_stop = pp_cdn.columns.get_level_values(1) == "*"
    pp_cdn = pp_cdn.loc[:,~is_stop]
    return(pp_cdn)

def get_pp_pep(pp_cdn):
    aa_index = pp_cdn.columns.get_level_values(1)
    pp_pep = pp_cdn.groupby(by=aa_index, axis=1).sum()
    return(pp_pep)

def attach_node_posterior(tree, infile_dir, infile_type, codon_table):
    print("Attaching phylobayes posterior probabilities to nodes...", flush=True)
    start = time.time()
    files = os.listdir(infile_dir)
    if infile_type=='phylobayes':
        for node in tree.traverse():
            pp_file = get_node_phylobayes_out(node=node, files=files)
            if len(pp_file) == 1:
                pp_file = pp_file[0]
                node.pp_nuc = get_pp_nuc(infile_dir, pp_file)
                node.pp_cdn = get_pp_cdn(node.pp_nuc, codon_table)
                node.pp_pep = get_pp_pep(node.pp_cdn)
    else:
        print('--infile_type', infile_type, 'is not currently supported.')
        sys.exit()
    elapsed_time = int(time.time()-start)
    print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]", flush=True)
    return(tree)

def prepare_node_combinations(tree, arity, check_attr):
    print("Preparing node combinations...", flush=True)
    start = time.time()
    print("arity:", arity, flush=True)
    all_nodes = list(tree.traverse())
    print("all nodes:", len(all_nodes), flush=True)
    target_nodes = list()
    for node in all_nodes:
        if check_attr in dir(node):
            target_nodes.append(node)
    print("nodes with", check_attr, ":", len(target_nodes), flush=True)
    node_combinations = list(itertools.combinations(target_nodes, arity))
    print("all node combinations: ", len(node_combinations), flush=True)
    for leaf in tree.iter_leaves():
        dep_nodes = [leaf.numerical_label]
        dep_node_labels = set(dep_nodes + [ ancestor.numerical_label for ancestor in leaf.get_ancestors() ])
        identified_duplicates = list()
        for nodes in node_combinations:
            node_labels = set([ node.numerical_label for node in nodes ])
            if len(node_labels.intersection(dep_node_labels)) >= 2:
                identified_duplicates.append(nodes)
        for dups in identified_duplicates:
            node_combinations.remove(dups)
    print("independent node combinations: ", len(node_combinations), flush=True)
    elapsed_time = int(time.time()-start); print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]", flush=True)
    return(node_combinations)

def node2id_combinations(node_combinations):
    start = time.time()
    print("Converting node combinations to id combinations", flush=True)
    arity = len(node_combinations[0])
    id_combinations = numpy.zeros((len(node_combinations), arity), dtype=numpy.int64)
    for c in range(len(node_combinations)):
        for a in range(arity):
            id_combinations[c,a] = node_combinations[c][a].numerical_label
    elapsed_time = int(time.time()-start); print(("elapsed_time for sorting: {0}".format(elapsed_time)) + "[sec]", flush=True)
    return(id_combinations)

def get_branch_table(tree, sub_tensor, attr):
    start = time.time()
    print("Making branch table...", flush=True)
    column_names=['branch_name','branch_id',attr+'_sub']
    df = pandas.DataFrame(numpy.nan, index=range(0,len(list(tree.traverse()))), columns=column_names)
    i=0
    for node in tree.traverse():
        df.loc[i,'branch_name'] = getattr(node, 'name')
        df.loc[i,'branch_id'] = getattr(node, 'numerical_label')
        df.loc[i,attr+'_sub'] = sub_tensor[node.numerical_label,:,:,:,:].sum()
        i+=1
    df = df.dropna(axis=0)
    df['branch_id'] = df['branch_id'].astype(int)
    df = df.sort_values(by='branch_id')
    elapsed_time = int(time.time()-start); print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]", flush=True)
    return(df)

def get_branch_site_table(tree):
    start = time.time()
    print("Making branch-site table...", flush=True)
    attr='pp_cdn'
    for node in tree.traverse():
        if attr in dir(node):
            num_site = getattr(node, attr).shape[0]
            break
    column_names=['branch_id','site','cdn_sub','syn_sub','pep_sub']+list(getattr(node, attr).columns.get_level_values(0))
    df = pandas.DataFrame(numpy.nan, index=numpy.arange(0,len(list(tree.traverse()))*num_site), columns=column_names)
    i=0
    for node in tree.traverse():
        if attr in dir(node):
            ind = numpy.arange(i*num_site,(i+1)*num_site)
            df.loc[ind,list(getattr(node, attr).columns.get_level_values(0))] = getattr(node, attr).values
            df.loc[ind,'site'] = numpy.arange(0, num_site)
            df.loc[ind,'branch_id'] = node.numerical_label
            i+=1
    df = df.sort_values(by=['branch_id','site']).reset_index(drop=True)
    elapsed_time = int(time.time()-start); print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]", flush=True)
    return(df)

def calc_combinat_branch_site(id_combinations, sub_tensor, attr):
    arity = id_combinations.shape[1]
    num_site = sub_tensor.shape[2]
    df = numpy.zeros([id_combinations.shape[0]*num_site, arity+5])
    node=0
    start = time.time()
    for i in range(id_combinations.shape[0]):
        row_start = node*num_site
        row_end = (node+1)*num_site
        df[row_start:row_end,:arity] = id_combinations[node,:] # branch_ids
        df[row_start:row_end,arity] = numpy.arange(num_site)# site
        for sg in range(sub_tensor.shape[1]):
            df[row_start:row_end,arity+1] += sub_tensor[id_combinations[i,:],sg,:,:,:].sum(axis=(2,3)).prod(axis=0) #any2any
            df[row_start:row_end,arity+2] += sub_tensor[id_combinations[i,:],sg,:,:,:].sum(axis=3).prod(axis=0).sum(axis=1) #spe2any
            df[row_start:row_end,arity+3] += sub_tensor[id_combinations[i,:],sg,:,:,:].sum(axis=2).prod(axis=0).sum(axis=1) #any2spe
            df[row_start:row_end,arity+4] += sub_tensor[id_combinations[i,:],sg,:,:,:].prod(axis=0).sum(axis=(1,2)) #spe2spe
        if node%1000 ==0:
            print(node, int(time.time()-start), '[sec]', flush=True)
        node += 1
    return(df)

def get_substitution_tensor(tree, mode, min_pp, codon_table):
    start = time.time()
    print("Preparing substitution tensor:", mode, flush=True)
    if mode=='cdn':
        max_matrix_size = len([ codon[0] for codon in codon_table if codon[0]!='*' ])
        num_synonymous_group = 1
        attr_pp = 'pp_'+mode
    elif mode=='syn':
        count_dict = collections.Counter([ codon[0] for codon in codon_table ])
        max_matrix_size = max(count_dict.values())
        num_synonymous_group = len(set([ codon[0] for codon in codon_table if codon[0]!='*' ]))
        attr_pp = 'pp_cdn'
    if mode=='pep':
        max_matrix_size = len(set([ codon[0] for codon in codon_table if codon[0]!='*' ]))
        num_synonymous_group = 1
        attr_pp = 'pp_'+mode
    for node in tree.traverse():
        if attr_pp in dir(node):
            num_site = len(getattr(node, attr_pp).values)
    num_branch = len(list(tree.traverse()))
    # axis = [branch,synonymous_group,site,from,to]
    axis = [num_branch, num_synonymous_group, num_site, max_matrix_size, max_matrix_size]
    sub_tensor = numpy.zeros(axis, dtype=numpy.float64)
    for node in tree.traverse():
        if not node.is_root():
            if attr_pp in dir(node.up):
                pp_child = getattr(node, attr_pp).values
                pp_parent = numpy.transpose(getattr(node.up, attr_pp).values)
                diag_zero = numpy.diag([-1]*pp_child.shape[1])+1
                # sub_matrix.shape = [site,from,to]
                sub_matrix = numpy.einsum("ij,jk,ik->jik", pp_parent, pp_child, diag_zero)
                if min_pp!=0:
                    sub_matrix = (sub_matrix>=min_pp).astype(numpy.float64)
                if mode=='syn':
                    aa_redundant = getattr(node, "pp_cdn").columns.get_level_values(1)
                    aa_unique = aa_redundant.unique()
                    for i in range(len(aa_unique)):
                        is_target_aa = numpy.array(aa_redundant==aa_unique[i], dtype=bool)
                        matrix_size = sum(is_target_aa)
                        sub_tensor[node.numerical_label,i,:,:matrix_size,:matrix_size] = sub_matrix[:,is_target_aa,:][:,:,is_target_aa]
                else:
                    sub_tensor[node.numerical_label,0,:,:,:] = sub_matrix
            else:
                sub_tensor[node.numerical_label,:,:,:] = numpy.nan,
        else:
            sub_tensor[node.numerical_label,:,:,:] = numpy.nan,
    print(mode, ': size of substitution tensor :', int(sys.getsizeof(sub_tensor)/(1024*1024)), 'MB', flush=True)
    return(sub_tensor)

def get_combinat_branch_site_table(id_combinations, sub_tensor, attr, nslots, min_pp):
    start = time.time()
    print("Calculating combinatorial substitutions: attr =", attr, flush=True)
    arity = id_combinations.shape[1]
    cn1 = [ "branch_id_" + str(num+1) for num in range(0,arity) ]
    cn2 = ["site",]
    cn3 = [ attr+'_'+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    if nslots==1:
        df = calc_combinat_branch_site(id_combinations, sub_tensor, attr)
    else:
        chunks = [ (id_combinations.shape[0]+i)//nslots for i in range(nslots) ]
        id_chunks = list()
        i = 0
        for c in chunks:
            id_chunks.append(id_combinations[i:i+c,:])
            i+= c
        results = joblib.Parallel(n_jobs=nslots)( joblib.delayed(calc_combinat_branch_site)(ids, sub_tensor, attr) for ids in id_chunks )
        del sub_tensor
        df = numpy.concatenate(results, axis=0)
        del results
    df = pandas.DataFrame(df, columns=cn1+cn2+cn3)
    df = df.dropna()
    if min_pp!=0:
        df = df.astype(numpy.int64)
    for cn in cn1+cn2:
        df[cn] = df[cn].astype(int)
    print(type(df), flush=True)
    elapsed_time = int(time.time()-start); print(("elapsed_time: {0}".format(elapsed_time)) + "[sec]", flush=True)
    return(df)

def combinat_branch_table(df_combinat_branch_site):
    start = time.time()
    print("Making branch-site table...", flush=True)
    df = df_combinat_branch_site.drop('site', axis=1)
    by_keywords = list(df.columns[df.columns.str.startswith('branch_id')])
    df = df.groupby(by_keywords, axis=0).sum()
    df_id = pandas.DataFrame()
    df_id['branch_id_1'] = df.index.get_level_values('branch_id_1')
    df_id['branch_id_2'] = df.index.get_level_values('branch_id_2')
    df = pandas.concat([df_id, df.reset_index(drop=True)], axis=1, ignore_index=False)
    elapsed_time = int(time.time()-start); print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]", flush=True)
    return(df)

def get_site_table(sub_tensor, attr):
    start = time.time()
    print("Making site table...", flush=True)
    column_names=['site',attr+'_sub']
    num_site = sub_tensor.shape[2]
    df = pandas.DataFrame(0, index=numpy.arange(0,num_site), columns=column_names)
    df['site'] = numpy.arange(0, num_site)
    df[attr+'_sub'] = numpy.nan_to_num(sub_tensor).sum(axis=4).sum(axis=3).sum(axis=1).sum(axis=0)
    df['site'] = df['site'].astype(int)
    df = df.sort_values(by='site')
    elapsed_time = int(time.time()-start); print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]", flush=True)
    return(df)

def combinat_site_table(df_combinat_branch_site):
    start = time.time()
    print("Making combinat-site table...", flush=True)
    df = df_combinat_branch_site.drop('branch_id_1', axis=1).drop('branch_id_2', axis=1)
    by_keywords = 'site'
    df = df.groupby(by_keywords, axis=0).sum()
    colnames = list(df.columns)
    df['site'] = df.index
    df = df.loc[:,['site',]+colnames].reset_index(drop=True)
    elapsed_time = int(time.time()-start); print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]", flush=True)
    return(df)

def get_num_site(tree):
    for node in tree.traverse():
        if 'pp_pep' in dir(node):
            num_site = node.pp_pep.shape[0]
            break
    return(num_site)

codon_table = get_codon_table(codon_file=args.codon_file)
tree = get_tree(infile_dir=args.infile_dir, infile_type=args.infile_type)
tree = add_numerical_node_labels(tree)
tree = attach_node_posterior(tree=tree, infile_dir=args.infile_dir, infile_type=args.infile_type, codon_table=codon_table)
num_site = get_num_site(tree)

def calc_combinat_branch_omega(cb, b, s):
    num_site = s.shape[0]
    b1 = b.loc[:,['branch_id','cdn_sub','syn_sub','pep_sub']]
    b1.columns = [ c+'_1' for c in b1.columns ]
    cb = pandas.merge(cb, b1, on='branch_id_1', how='left')
    del b1
    b2 = b.loc[:,['branch_id','cdn_sub','syn_sub','pep_sub']]
    b2.columns = [ c+'_2' for c in b2.columns ]
    cb = pandas.merge(cb, b2, on='branch_id_2', how='left')
    del b2
    del b
    rhoNconv = cb['pep_any2spe'].sum() / cb['pep_any2any'].sum()
    rhoNdiv = 1 - rhoNconv
    rhoSconv = cb['syn_any2spe'].sum() / cb['syn_any2any'].sum()
    rhoSdiv = 1 - rhoSconv
    print('rhoNconv =', numpy.round(rhoNconv, decimals=3), 'rhoNdiv =', numpy.round(rhoNdiv, decimals=3))
    print('rhoSconv =', numpy.round(rhoSconv, decimals=3), 'rhoSdiv =', numpy.round(rhoSdiv, decimals=3))
    cb['EN_pair_unif'] = ((cb['pep_sub_1'] / num_site) * (cb['pep_sub_2'] / num_site)) * num_site
    cb['EN_conv_unif'] = cb['EN_pair_unif'] * rhoNconv
    cb['EN_div_unif'] = cb['EN_pair_unif'] * rhoNdiv
    cb['ES_pair_unif'] = ((cb['syn_sub_1'] / num_site) * (cb['syn_sub_2'] / num_site)) * num_site
    cb['ES_conv_unif'] = cb['ES_pair_unif'] * rhoSconv
    cb['ES_div_unif'] = cb['ES_pair_unif'] * rhoSdiv
    N_asrv = numpy.reshape((s['pep_sub'] / s['pep_sub'].sum()).values, newshape=(1,s.shape[0]))
    S_asrv = numpy.reshape((s['syn_sub'] / s['syn_sub'].sum()).values, newshape=(1,s.shape[0]))
    cb['EN_pair_asrv'] = ((1 - ((1 - N_asrv) ** numpy.expand_dims(cb['pep_sub_1'], axis=1))) * (1 - ((1 - N_asrv) ** numpy.expand_dims(cb['pep_sub_2'], axis=1)))).sum(axis=1)
    cb['EN_conv_asrv'] = cb['EN_pair_asrv'] * rhoNconv
    cb['EN_div_asrv'] = cb['EN_pair_asrv'] * rhoNdiv
    cb['ES_pair_asrv'] = ((1 - ((1 - S_asrv) ** numpy.expand_dims(cb['syn_sub_1'], axis=1))) * (1 - ((1 - S_asrv) ** numpy.expand_dims(cb['syn_sub_2'], axis=1)))).sum(axis=1)
    cb['ES_conv_asrv'] = cb['ES_pair_asrv'] * rhoSconv
    cb['ES_div_asrv'] = cb['ES_pair_asrv'] * rhoSdiv
    cb['omega_pair_unif'] = (cb['pep_any2any'] / cb['EN_pair_unif']) / (cb['syn_any2any'] / cb['ES_pair_unif'])
    cb['omega_conv_unif'] = (cb['pep_any2spe'] / cb['EN_conv_unif']) / (cb['syn_any2spe'] / cb['ES_conv_unif'])
    cb['omega_div_unif'] = ((cb['pep_any2any']-cb['pep_any2spe']) / cb['EN_div_unif']) / ((cb['syn_any2any']-cb['syn_any2spe']) / cb['ES_div_unif'])
    cb['omega_pair_asrvNS'] = (cb['pep_any2any'] / cb['EN_pair_asrv']) / (cb['syn_any2any'] / cb['ES_pair_asrv'])
    cb['omega_conv_asrvNS'] = (cb['pep_any2spe'] / cb['EN_conv_asrv']) / (cb['syn_any2spe'] / cb['ES_conv_asrv'])
    cb['omega_div_asrvNS'] = ((cb['pep_any2any']-cb['pep_any2spe']) / cb['EN_div_asrv']) / ((cb['syn_any2any']-cb['syn_any2spe']) / cb['ES_div_asrv'])
    cb['omega_pair_asrvN'] = (cb['pep_any2any'] / cb['EN_pair_asrv']) / (cb['syn_any2any'] / cb['ES_pair_unif'])
    cb['omega_conv_asrvN'] = (cb['pep_any2spe'] / cb['EN_conv_asrv']) / (cb['syn_any2spe'] / cb['ES_conv_unif'])
    cb['omega_div_asrvN'] = ((cb['pep_any2any']-cb['pep_any2spe']) / cb['EN_div_asrv']) / ((cb['syn_any2any']-cb['syn_any2spe']) / cb['ES_div_unif'])
    return(cb)

nc = prepare_node_combinations(tree=tree, arity=args.arity, check_attr="name")
id_combinations = node2id_combinations(node_combinations=nc)
del nc

# Calculate combinatorial substitutions
df_branch_site = get_branch_site_table(tree=tree)
cdn_tensor = get_substitution_tensor(tree=tree, mode='cdn', min_pp=args.min_pp, codon_table=codon_table)
df_branch_site['cdn_sub'] = cdn_tensor.sum(axis=4).sum(axis=3).sum(axis=1).reshape(cdn_tensor.shape[0]*cdn_tensor.shape[2])
cbs_cdn = get_combinat_branch_site_table(id_combinations=id_combinations, sub_tensor=cdn_tensor, attr='cdn', nslots=args.nslots, min_pp=args.min_pp)
b_cdn = get_branch_table(tree, cdn_tensor, attr='cdn')
s_cdn = get_site_table(cdn_tensor, attr='cdn')
del cdn_tensor

syn_tensor = get_substitution_tensor(tree=tree, mode='syn', min_pp=args.min_pp, codon_table=codon_table)
df_branch_site['syn_sub'] = syn_tensor.sum(axis=4).sum(axis=3).sum(axis=1).reshape(syn_tensor.shape[0]*syn_tensor.shape[2])
cbs_syn = get_combinat_branch_site_table(id_combinations=id_combinations, sub_tensor=syn_tensor, attr='syn', nslots=args.nslots, min_pp=args.min_pp)
b_syn = get_branch_table(tree, syn_tensor, attr='syn')
s_syn = get_site_table(syn_tensor, attr='syn')
del syn_tensor

df_combinat_branch_site = pandas.merge(cbs_cdn, cbs_syn, on=['branch_id_1','branch_id_2','site'])
del cbs_cdn
del cbs_syn

pep_tensor = get_substitution_tensor(tree=tree, mode='pep', min_pp=args.min_pp, codon_table=codon_table)
df_branch_site['pep_sub'] = pep_tensor.sum(axis=4).sum(axis=3).sum(axis=1).reshape(pep_tensor.shape[0]*pep_tensor.shape[2])
df_branch_site = df_branch_site.dropna(axis=0)
if args.min_pp==0:
    df_branch_site['branch_id'] = df_branch_site['branch_id'].astype(int)
    df_branch_site['site'] = df_branch_site['site'].astype(int)
else:
    df_branch_site = df_branch_site.astype(numpy.int64)
print('\ndf_branch_site', flush=True)
print(df_branch_site.info(), flush=True)
df_branch_site.to_csv("subst_branch_site.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
del df_branch_site

cbs_pep = get_combinat_branch_site_table(id_combinations=id_combinations, sub_tensor=pep_tensor, attr='pep', nslots=args.nslots, min_pp=args.min_pp)
b_pep = get_branch_table(tree, pep_tensor, attr='pep')
s_pep = get_site_table(pep_tensor, attr='pep')
del pep_tensor

df_combinat_branch_site = pandas.merge(df_combinat_branch_site, cbs_pep, on=['branch_id_1','branch_id_2','site'])
del cbs_pep
df_combinat_branch_site = df_combinat_branch_site.reset_index(drop=True)

# combinat_branch_site table
start = time.time()
swap_combinat = ['branch_id_1','branch_id_2']
is_swap = (df_combinat_branch_site[swap_combinat[0]] > df_combinat_branch_site[swap_combinat[1]])
if is_swap.sum():
    swap_to_0 = df_combinat_branch_site.loc[is_swap,swap_combinat[1]]
    swap_to_1 = df_combinat_branch_site.loc[is_swap,swap_combinat[0]]
    df_combinat_branch_site.loc[is_swap,swap_combinat[0]] = swap_to_0
    df_combinat_branch_site.loc[is_swap,swap_combinat[1]] = swap_to_1
df_combinat_branch_site = df_combinat_branch_site.sort_values(by=['branch_id_1','branch_id_2','site'])
for cn in ['branch_id_1','branch_id_2','site']:
    df_combinat_branch_site[cn] = df_combinat_branch_site[cn].astype(int)
elapsed_time = int(time.time()-start); print(("elapsed_time for sorting cbs table: {0}".format(elapsed_time)) + "[sec]", flush=True)
print('\ndf_combinat_branch_site', flush=True)
print(df_combinat_branch_site.info(), flush=True)
df_combinat_branch_site.to_csv("subst_combinat_branch_site.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)

# combinat_branch table
df_combinat_branch = combinat_branch_table(df_combinat_branch_site=df_combinat_branch_site)

# site table
df_combinat_site = combinat_site_table(df_combinat_branch_site=df_combinat_branch_site)
df_site = s_cdn.merge(s_syn, on=['site']).merge(s_pep, on=['site']).merge(df_combinat_site, on=['site'])
del df_combinat_site
if args.min_pp!=0:
    colnames = ['cdn_sub','syn_sub','pep_sub']
    df_site.loc[:,colnames] = df_site.loc[:,colnames].astype(numpy.int64)
print('\ndf_site', flush=True)
print(df_site.info(), flush=True)
df_site.to_csv("subst_site.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
#del df_site
del df_combinat_branch_site

# branch table
df_branch = b_cdn.merge(b_syn, on=['branch_name','branch_id']).merge(b_pep, on=['branch_name','branch_id'])
if args.min_pp!=0:
    colnames = ['cdn_sub','syn_sub','pep_sub']
    df_branch.loc[:,colnames] = df_branch.loc[:,colnames].astype(numpy.int64)
print('\ndf_branch', flush=True)
print(df_branch.info(), flush=True)
df_branch.to_csv("subst_branch.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
#del df_branch

# omega calculation
df_combinat_branch = calc_combinat_branch_omega(cb=df_combinat_branch, b=df_branch, s=df_site)
print('\ndf_combinat_branch', flush=True)
print(df_combinat_branch.info(), flush=True)
print('median omega_pair_unif =', numpy.round(df_combinat_branch['omega_pair_unif'].median(),decimals=3))
print('median omega_conv_unif =', numpy.round(df_combinat_branch['omega_conv_unif'].median(),decimals=3))
print('median omega_div_unif  =', numpy.round(df_combinat_branch['omega_div_unif'].median(),decimals=3))
print('median omega_pair_asrvNS =', numpy.round(df_combinat_branch['omega_pair_asrvNS'].median(),decimals=3))
print('median omega_conv_asrvNS =', numpy.round(df_combinat_branch['omega_conv_asrvNS'].median(),decimals=3))
print('median omega_div_asrvNS  =', numpy.round(df_combinat_branch['omega_div_asrvNS'].median(),decimals=3))
print('median omega_pair_asrvN =', numpy.round(df_combinat_branch['omega_pair_asrvN'].median(),decimals=3))
print('median omega_conv_asrvN =', numpy.round(df_combinat_branch['omega_conv_asrvN'].median(),decimals=3))
print('median omega_div_asrvN  =', numpy.round(df_combinat_branch['omega_div_asrvN'].median(),decimals=3))
df_combinat_branch.to_csv("subst_combinat_branch.tsv", sep="\t", index=False, float_format='%.4f', chunksize=10000)
#del df_combinat_branch

print("Completed!", flush=True)


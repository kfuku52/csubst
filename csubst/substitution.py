import time
import numpy
import pandas
import joblib
import os
from csubst.table import *
from csubst.parallel import *

def initialize_substitution_tensor(state_tensor, mode, g, mmap_attr, dtype=None):
    if dtype is None:
        dtype = state_tensor.dtype
    num_branch = state_tensor.shape[0]
    num_site = state_tensor.shape[1]
    if mode=='asis':
        num_syngroup = 1
        num_state = state_tensor.shape[2]
    elif mode=='syn':
        num_syngroup = len(g['amino_acid_orders'])
        num_state = g['max_synonymous_size']
    axis = (num_branch,num_site,num_syngroup,num_state,num_state) # axis = [branch,site,matrix_group,state_from,state_to]
    mmap_tensor = os.path.join(os.getcwd(), 'tmp.csubst.sub_tensor.'+mmap_attr+'.mmap')
    if os.path.exists(mmap_tensor): os.unlink(mmap_tensor)
    txt = 'Memory map is generated. dtype={}, axis={}, path={}'
    print(txt.format(state_tensor.dtype, axis, mmap_tensor), flush=True)
    sub_tensor = numpy.memmap(mmap_tensor, dtype=dtype, shape=axis, mode='w+')
    return sub_tensor

def get_substitution_tensor(state_tensor, state_tensor_anc=None, mode='', g={}, mmap_attr=''):
    sub_tensor = initialize_substitution_tensor(state_tensor, mode, g, mmap_attr)
    if state_tensor_anc is None:
        state_tensor_anc = state_tensor
    if (g['ml_anc']=='no'):
        sub_tensor[:,:,:,:,:] = numpy.nan
    if mode=='asis':
        num_state = state_tensor.shape[2]
        diag_zero = numpy.diag([-1] * num_state) + 1
    for node in g['tree'].traverse():
        if node.is_root():
            continue
        child = node.numerical_label
        parent = node.up.numerical_label
        if state_tensor_anc[parent, :, :].sum()==0:
            continue
        if mode=='asis':
            sub_matrix = numpy.einsum("sa,sd,ad->sad", state_tensor_anc[parent,:,:], state_tensor[child,:,:], diag_zero)
            # s=site, a=ancestral, d=derived
            sub_tensor[child, :, 0, :, :] = sub_matrix
        elif mode=='syn':
            for s,aa in enumerate(g['amino_acid_orders']):
                ind = numpy.array(g['synonymous_indices'][aa])
                size = len(ind)
                diag_zero = numpy.diag([-1] * size) + 1
                parent_matrix = state_tensor_anc[parent, :, ind] # axis is swapped, shape=[state,site]
                child_matrix = state_tensor[child, :, ind] # axis is swapped, shape=[state,site]
                sub_matrix = numpy.einsum("as,ds,ad->sad", parent_matrix, child_matrix, diag_zero)
                sub_tensor[child, :, s, :size, :size] = sub_matrix
    if g['min_sub_pp']!=0:
        if (g['ml_anc']):
            print('--ml_anc is set. --min_sub_pp is not applied.')
        else:
            sub_tensor = (numpy.nan_to_num(sub_tensor)>=g['min_sub_pp'])
    return sub_tensor

def get_b(g, sub_tensor, attr):
    column_names=['branch_name','branch_id',attr+'_sub']
    df = pandas.DataFrame(numpy.nan, index=range(0, g['num_node']), columns=column_names)
    i=0
    for node in g['tree'].traverse():
        df.loc[i,'branch_name'] = getattr(node, 'name')
        df.loc[i,'branch_id'] = getattr(node, 'numerical_label')
        df.loc[i,attr+'_sub'] = sub_tensor[node.numerical_label,:,:,:,:].sum()
        i+=1
    df = df.dropna(axis=0)
    df['branch_id'] = df['branch_id'].astype(int)
    df = df.sort_values(by='branch_id')
    df = set_substitution_dtype(df=df)
    return(df)

def get_s(sub_tensor, attr):
    column_names=['site',attr+'_sub']
    num_site = sub_tensor.shape[1]
    df = pandas.DataFrame(0, index=numpy.arange(0,num_site), columns=column_names)
    df['site'] = numpy.arange(0, num_site)
    df[attr+'_sub'] = numpy.nan_to_num(sub_tensor).sum(axis=4).sum(axis=3).sum(axis=2).sum(axis=0)
    df['site'] = df['site'].astype(int)
    df = df.sort_values(by='site')
    df = set_substitution_dtype(df=df)
    return(df)

def get_cs(id_combinations, sub_tensor, attr):
    num_site = sub_tensor.shape[1]
    df = numpy.zeros([num_site, 5])
    df[:, 0] = numpy.arange(num_site)
    for i in numpy.arange(id_combinations.shape[0]):
        for sg in numpy.arange(sub_tensor.shape[2]): # TODO: Couldn't this sg included in the matrix calc using .sum()?
            df[:, 1] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=(2, 3)).prod(axis=0))  # any2any
            df[:, 2] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=3).prod(axis=0).sum(axis=1))  # spe2any
            df[:, 3] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=2).prod(axis=0).sum(axis=1))  # any2spe
            df[:, 4] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].prod(axis=0).sum(axis=(1, 2)))  # spe2spe
    cn = ['site',] + [ attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    df = pandas.DataFrame(df, columns=cn)
    df = set_substitution_dtype(df=df)
    return (df)

def get_bs(S_tensor, N_tensor):
    num_site = S_tensor.shape[1]
    num_branch = S_tensor.shape[0]
    column_names=['branch_id','site','S_sub','N_sub']
    df = pandas.DataFrame(numpy.nan, index=numpy.arange(0,num_branch*num_site), columns=column_names)
    for i in numpy.arange(num_branch):
        ind = numpy.arange(i*num_site, (i+1)*num_site)
        df.loc[ind, 'site'] = numpy.arange(0, num_site)
        df.loc[ind, 'branch_id'] = i
        df.loc[ind, 'S_sub'] = numpy.nan_to_num(S_tensor[i, :, :, :, :]).sum(axis=(1,2,3))
        df.loc[ind, 'N_sub'] = numpy.nan_to_num(N_tensor[i, :, :, :, :]).sum(axis=(1,2,3))
    df = set_substitution_dtype(df=df)
    return(df)

def sub_tensor2cb(id_combinations, sub_tensor, mmap=False, df_mmap=None, mmap_start=0, float_type=numpy.float64):
    arity = id_combinations.shape[1]
    if mmap:
        df = df_mmap
    else:
        if (sub_tensor.dtype==numpy.bool):
            data_type = numpy.int32
        else:
            data_type = float_type
        df = numpy.zeros([id_combinations.shape[0], arity+4], dtype=data_type)
    start = mmap_start
    end = mmap_start + id_combinations.shape[0]
    df[start:end, :arity] = id_combinations[:, :]  # branch_ids
    start_time = time.time()
    for i,j in zip(numpy.arange(start, end),numpy.arange(id_combinations.shape[0])):
        for sg in numpy.arange(sub_tensor.shape[2]):
            df[i, arity+0] += sub_tensor[id_combinations[j,:], :, sg, :, :].sum(axis=(2, 3)).prod(axis=0).sum(axis=0)  # any2any
            df[i, arity+1] += sub_tensor[id_combinations[j,:], :, sg, :, :].sum(axis=3).prod(axis=0).sum(axis=1).sum(axis=0)  # spe2any
            df[i, arity+2] += sub_tensor[id_combinations[j,:], :, sg, :, :].sum(axis=2).prod(axis=0).sum(axis=1).sum(axis=0)  # any2spe
            df[i, arity+3] += sub_tensor[id_combinations[j,:], :, sg, :, :].prod(axis=0).sum(axis=(1, 2)).sum(axis=0)  # spe2spe
        if j % 1000 == 0:
            mmap_end = mmap_start + id_combinations.shape[0]
            txt = 'cb: {:,}th in the id range {:,}-{:,}: {:,} [sec]'
            print(txt.format(j, mmap_start, mmap_end, int(time.time() - start_time)), flush=True)
    if not mmap:
        return (df)

def get_cb(id_combinations, sub_tensor, g, attr):
    arity = id_combinations.shape[1]
    cn = [ "branch_id_" + str(num+1) for num in range(0,arity) ]
    cn = cn + [ attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    if g['nslots']==1:
        df = sub_tensor2cb(id_combinations, sub_tensor, float_type=g['float_type'])
        df = pandas.DataFrame(df, columns=cn)
    else:
        id_chunks,mmap_starts = get_chunks(id_combinations, g['nslots'])
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.cb.out.mmap')
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        axis = (id_combinations.shape[0], arity+4)
        my_dtype = sub_tensor.dtype
        if 'bool' in str(my_dtype):
            my_dtype = numpy.int32
        df_mmap = numpy.memmap(mmap_out, dtype=my_dtype, shape=axis, mode='w+')
        joblib.Parallel(n_jobs=g['nslots'], max_nbytes=None, backend='multiprocessing')(
            joblib.delayed(sub_tensor2cb)
            (ids, sub_tensor, True, df_mmap, ms, g['float_type']) for ids, ms in zip(id_chunks, mmap_starts)
        )
        df = pandas.DataFrame(df_mmap, columns=cn)
        if os.path.exists(mmap_out): os.unlink(mmap_out)
    df = sort_labels(df)
    df = df.dropna()
    if not attr.startswith('E'):
        df = set_substitution_dtype(df=df)
    return df

def sub_tensor2cbs(id_combinations, sub_tensor, mmap=False, df_mmap=None, mmap_start=0):
    arity = id_combinations.shape[1]
    num_site = sub_tensor.shape[1]
    sites = numpy.arange(num_site)
    if mmap:
        df = df_mmap
    else:
        shape = (int(id_combinations.shape[0]*num_site), arity+5)
        my_dtype = sub_tensor.dtype
        if 'bool' in str(my_dtype):
            my_dtype = numpy.int32
        df = numpy.zeros(shape=shape, dtype=my_dtype)
    node=0
    start_time = time.time()
    for i in numpy.arange(id_combinations.shape[0]):
        row_start = (node*num_site)+(mmap_start*num_site)
        row_end = ((node+1)*num_site)+(mmap_start*num_site)
        df[row_start:row_end,:arity] = id_combinations[node,:] # branch_ids
        df[row_start:row_end,arity] = sites # site
        ic = id_combinations[i,:]
        for sg in range(sub_tensor.shape[2]):
            df[row_start:row_end,arity+1] += sub_tensor[ic,:,sg,:,:].sum(axis=(2,3)).prod(axis=0) #any2any
            df[row_start:row_end,arity+2] += sub_tensor[ic,:,sg,:,:].sum(axis=3).prod(axis=0).sum(axis=1) #spe2any
            df[row_start:row_end,arity+3] += sub_tensor[ic,:,sg,:,:].sum(axis=2).prod(axis=0).sum(axis=1) #any2spe
            df[row_start:row_end,arity+4] += sub_tensor[ic,:,sg,:,:].prod(axis=0).sum(axis=(1,2)) #spe2spe
        if (node%1000==0):
            mmap_start = mmap_start
            mmap_end = mmap_start+id_combinations.shape[0]
            txt = 'cbs: {:,}th in the id range {:,}-{:,}: {:,} [sec]'
            print(txt.format(node, mmap_start, mmap_end, int(time.time() - start_time)), flush=True)
        node += 1
    if not mmap:
        return df

def get_cbs(id_combinations, sub_tensor, attr, g):
    print("Calculating combinatorial substitutions: attr =", attr, flush=True)
    arity = id_combinations.shape[1]
    cn1 = [ "branch_id_" + str(num+1) for num in range(0,arity) ]
    cn2 = ["site",]
    cn3 = [ attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    if g['nslots']==1:
        df = sub_tensor2cbs(id_combinations, sub_tensor)
        df = pandas.DataFrame(df, columns=cn1 + cn2 + cn3)
    else:
        id_chunks,mmap_starts = get_chunks(id_combinations, g['nslots'])
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.cbs.out.mmap')
        if os.path.exists(mmap_out): os.remove(mmap_out)
        axis = (id_combinations.shape[0]*sub_tensor.shape[1], arity+5)
        my_dtype = sub_tensor.dtype
        if 'bool' in str(my_dtype):
            my_dtype = numpy.int32
        df_mmap = numpy.memmap(mmap_out, dtype=my_dtype, shape=axis, mode='w+')
        joblib.Parallel(n_jobs=g['nslots'], max_nbytes=None, backend='multiprocessing')(
            joblib.delayed(sub_tensor2cbs)
            (ids, sub_tensor, True, df_mmap, ms) for ids,ms in zip(id_chunks,mmap_starts)
        )
        df = pandas.DataFrame(df_mmap, columns=cn1 + cn2 + cn3)
        if os.path.exists(mmap_out): os.remove(mmap_out)
    df = df.dropna()
    df = sort_labels(df)
    df = set_substitution_dtype(df=df)
    return(df)

def get_sub_sites(g, sS, sN, state_tensor):
    num_site = sS.shape[0]
    num_branch = len(list(g['tree'].traverse()))
    g['is_site_nonmissing'] = numpy.zeros(shape=[num_branch, num_site], dtype=numpy.bool)
    for node in g['tree'].traverse():
        nl = node.numerical_label
        g['is_site_nonmissing'][nl,:] = (state_tensor[nl,:,:].sum(axis=1)!=0)
    g['sub_sites'] = dict()
    g['sub_sites'][g['asrv']] = numpy.zeros(shape=[num_branch, num_site], dtype=g['float_type'])
    if (g['asrv']=='no'):
        sub_sites = numpy.ones(shape=[num_site,]) / num_site
    elif (g['asrv']=='pool'):
        sub_sites = sS['S_sub'].values + sN['N_sub'].values
    elif (g['asrv']=='file'):
        from csubst.parser_iqtree import read_rate
        sub_sites = g['iqtree_rate_values']
    if (g['asrv']=='sn'):
        for SN,df in zip(['S','N'],[sS,sN]):
            g['sub_sites'][SN] = numpy.zeros(shape=[num_branch, num_site], dtype=g['float_type'])
            sub_sites = df[SN+'_sub'].values
            for node in g['tree'].traverse():
                nl = node.numerical_label
                adjusted_sub_sites = sub_sites * g['is_site_nonmissing'][nl,:]
                total_sub_sites = adjusted_sub_sites.sum()
                total_sub_sites = 1 if (total_sub_sites==0) else total_sub_sites
                adjusted_sub_sites = adjusted_sub_sites/total_sub_sites
                g['sub_sites'][SN][nl,:] = adjusted_sub_sites
    elif (g['asrv']!='each'):
        for node in g['tree'].traverse():
            nl = node.numerical_label
            is_site_nonmissing = (state_tensor[nl,:,:].sum(axis=1)!=0)
            adjusted_sub_sites = sub_sites * is_site_nonmissing
            total_sub_sites = adjusted_sub_sites.sum()
            total_sub_sites = 1 if (total_sub_sites==0) else total_sub_sites
            adjusted_sub_sites = adjusted_sub_sites/total_sub_sites
            g['sub_sites'][g['asrv']][nl,:] = adjusted_sub_sites
    return g

def get_each_sub_sites(sub_sad, mode, sg, a, d, g):
    sub_sites = numpy.zeros(shape=g['is_site_nonmissing'].shape, dtype=g['float_type'])
    if mode == 'spe2spe':
        nonadjusted_sub_sites = sub_sad[:, sg, a, d]
    elif mode == 'spe2any':
        nonadjusted_sub_sites = sub_sad[:, sg, a]
    elif mode == 'any2spe':
        nonadjusted_sub_sites = sub_sad[:, sg, d]
    elif mode == 'any2any':
        nonadjusted_sub_sites = sub_sad[:, sg]
    for node in g['tree'].traverse():
        nl = node.numerical_label
        sub_sites[nl,:] = nonadjusted_sub_sites * g['is_site_nonmissing'][nl,:]
        total_sub_sites = sub_sites[nl,:].sum()
        total_sub_sites = 1 if (total_sub_sites==0) else total_sub_sites
        sub_sites[nl,:] = sub_sites[nl,:] / total_sub_sites
    return sub_sites

def get_sub_branches(sub_bad, mode, sg, a, d):
    if mode == 'spe2spe':
        sub_branches = sub_bad[:, sg, a, d]
    elif mode == 'spe2any':
        sub_branches = sub_bad[:, sg, a]
    elif mode == 'any2spe':
        sub_branches = sub_bad[:, sg, d]
    elif mode == 'any2any':
        sub_branches = sub_bad[:, sg]
    return sub_branches

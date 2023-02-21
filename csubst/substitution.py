import joblib
import numpy
import pandas

import os
import time

from csubst import table
from csubst import parallel
from csubst import substitution_cy

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
        if state_tensor_anc[parent, :, :].sum()<g['float_tol']:
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
    sub_tensor = numpy.nan_to_num(sub_tensor, nan=0, copy=False)
    #numpy.nan_to_num(sub_tensor, nan=0, copy=False)
    return sub_tensor

def apply_min_sub_pp(g, sub_tensor):
    if g['min_sub_pp']==0:
        return sub_tensor
    if (g['ml_anc']):
        print('--ml_anc is set. --min_sub_pp will not be applied.')
    else:
        sub_tensor[(sub_tensor<g['min_sub_pp'])] = 0
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
    df = table.set_substitution_dtype(df=df)
    return(df)

def get_s(sub_tensor, attr):
    column_names=['site',attr+'_sub']
    num_site = sub_tensor.shape[1]
    df = pandas.DataFrame(0, index=numpy.arange(0,num_site), columns=column_names)
    df['site'] = numpy.arange(0, num_site)
    df[attr+'_sub'] = numpy.nan_to_num(sub_tensor).sum(axis=4).sum(axis=3).sum(axis=2).sum(axis=0)
    df['site'] = df['site'].astype(int)
    df = df.sort_values(by='site')
    df = table.set_substitution_dtype(df=df)
    return(df)

def get_cs(id_combinations, sub_tensor, attr):
    num_site = sub_tensor.shape[1]
    df = numpy.zeros([num_site, 5])
    df[:, 0] = numpy.arange(num_site)
    for i in numpy.arange(id_combinations.shape[0]):
        for sg in numpy.arange(sub_tensor.shape[2]): # Couldn't this sg included in the matrix calc using .sum()?
            df[:, 1] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=(2, 3)).prod(axis=0))  # any2any
            df[:, 2] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=3).prod(axis=0).sum(axis=1))  # spe2any
            df[:, 3] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].sum(axis=2).prod(axis=0).sum(axis=1))  # any2spe
            df[:, 4] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], :, sg, :, :].prod(axis=0).sum(axis=(1, 2)))  # spe2spe
    cn = ['site',] + [ 'OC'+attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    df = pandas.DataFrame(df, columns=cn)
    df = table.set_substitution_dtype(df=df)
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
    df = table.set_substitution_dtype(df=df)
    return(df)

def sub_tensor2cb(id_combinations, sub_tensor, mmap=False, df_mmap=None, mmap_start=0, float_type=numpy.float64):
    if False:
        # Experimental. Currently this function does not speed up the analysis.
        df = substitution_cy.calc_combinatorial_sub_float32(id_combinations, mmap_start, sub_tensor, mmap, df_mmap)
    else:
        arity = id_combinations.shape[1]
        if mmap:
            df = df_mmap
        else:
            if (sub_tensor.dtype == numpy.bool):
                data_type = numpy.int32
            else:
                data_type = float_type
            df = numpy.zeros([id_combinations.shape[0], arity + 4], dtype=data_type)
        start_time = time.time()
        start = mmap_start
        end = mmap_start + id_combinations.shape[0]
        df[start:end, :arity] = id_combinations[:, :]  # branch_ids
        for i,j in zip(numpy.arange(start, end),numpy.arange(id_combinations.shape[0])):
            for sg in numpy.arange(sub_tensor.shape[2]):
                df[i, arity+0] += sub_tensor[id_combinations[j,:], :, sg, :, :].sum(axis=(2, 3)).prod(axis=0).sum(axis=0) # any2any
                df[i, arity+1] += sub_tensor[id_combinations[j,:], :, sg, :, :].sum(axis=3).prod(axis=0).sum(axis=1).sum(axis=0) # spe2any
                df[i, arity+2] += sub_tensor[id_combinations[j,:], :, sg, :, :].sum(axis=2).prod(axis=0).sum(axis=1).sum(axis=0) # any2spe
                df[i, arity+3] += sub_tensor[id_combinations[j,:], :, sg, :, :].prod(axis=0).sum(axis=(1, 2)).sum(axis=0) # spe2spe
            if j % 10000 == 0:
                mmap_end = mmap_start + id_combinations.shape[0]
                txt = 'cb: {:,}th in the id range {:,}-{:,}: {:,} [sec]'
                print(txt.format(j, mmap_start, mmap_end, int(time.time() - start_time)), flush=True)
    if not mmap:
        return (df)

def get_cb(id_combinations, sub_tensor, g, attr):
    arity = id_combinations.shape[1]
    cn = [ "branch_id_" + str(num+1) for num in range(0,arity) ]
    cn = cn + [ attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    if (g['threads']==1):
        df = sub_tensor2cb(id_combinations, sub_tensor, mmap=False, df_mmap=None,
                           mmap_start=0, float_type=g['float_type'])
        df = pandas.DataFrame(df, columns=cn)
    else:
        id_chunks,mmap_starts = parallel.get_chunks(id_combinations, g['threads'])
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.cb.out.mmap')
        if os.path.exists(mmap_out): os.unlink(mmap_out)
        axis = (id_combinations.shape[0], arity+4)
        my_dtype = sub_tensor.dtype
        if 'bool' in str(my_dtype):
            my_dtype = numpy.int32
        df_mmap = numpy.memmap(mmap_out, dtype=my_dtype, shape=axis, mode='w+')
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api='blas'):
            joblib.Parallel(n_jobs=g['threads'], max_nbytes=None, backend='multiprocessing')(
                joblib.delayed(sub_tensor2cb)
                (ids, sub_tensor, True, df_mmap, ms, g['float_type']) for ids,ms in zip(id_chunks, mmap_starts)
            )
        df = pandas.DataFrame(df_mmap, columns=cn)
        if os.path.exists(mmap_out): os.unlink(mmap_out)
    df = table.sort_branch_ids(df)
    df = df.dropna()
    if not attr.startswith('EC'):
        df = table.set_substitution_dtype(df=df)
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
        if (node%10000==0):
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
    cn3 = [ 'OC'+attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    if (g['threads']==1):
        df = sub_tensor2cbs(id_combinations, sub_tensor)
        df = pandas.DataFrame(df, columns=cn1 + cn2 + cn3)
    else:
        id_chunks,mmap_starts = parallel.get_chunks(id_combinations, g['threads'])
        mmap_out = os.path.join(os.getcwd(), 'tmp.csubst.cbs.out.mmap')
        if os.path.exists(mmap_out): os.remove(mmap_out)
        axis = (id_combinations.shape[0]*sub_tensor.shape[1], arity+5)
        my_dtype = sub_tensor.dtype
        if 'bool' in str(my_dtype):
            my_dtype = numpy.int32
        df_mmap = numpy.memmap(mmap_out, dtype=my_dtype, shape=axis, mode='w+')
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api='blas'):
            joblib.Parallel(n_jobs=g['threads'], max_nbytes=None, backend='multiprocessing')(
                joblib.delayed(sub_tensor2cbs)
                (ids, sub_tensor, True, df_mmap, ms) for ids,ms in zip(id_chunks,mmap_starts)
            )
        df = pandas.DataFrame(df_mmap, columns=cn1 + cn2 + cn3)
        if os.path.exists(mmap_out): os.remove(mmap_out)
    df = df.dropna()
    df = table.sort_branch_ids(df)
    df = table.set_substitution_dtype(df=df)
    return(df)

def get_sub_sites(g, sS, sN, state_tensor):
    num_site = sS.shape[0]
    num_branch = len(list(g['tree'].traverse()))
    g['is_site_nonmissing'] = numpy.zeros(shape=[num_branch, num_site], dtype=bool)
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
    elif (g['asrv']!='each'): # if 'each', Defined later in get_each_sub_sites()
        for node in g['tree'].traverse():
            nl = node.numerical_label
            is_site_nonmissing = (state_tensor[nl,:,:].sum(axis=1)!=0)
            adjusted_sub_sites = sub_sites * is_site_nonmissing
            total_sub_sites = adjusted_sub_sites.sum()
            total_sub_sites = 1 if (total_sub_sites==0) else total_sub_sites
            adjusted_sub_sites = adjusted_sub_sites/total_sub_sites
            g['sub_sites'][g['asrv']][nl,:] = adjusted_sub_sites
    return g

def get_each_sub_sites(sub_sg, mode, sg, a, d, g): # sub_sites for each "sg" group
    sub_sites = numpy.zeros(shape=g['is_site_nonmissing'].shape, dtype=g['float_type'])
    if mode == 'spe2spe':
        nonadjusted_sub_sites = sub_sg[:, sg, a, d]
    elif mode == 'spe2any':
        nonadjusted_sub_sites = sub_sg[:, sg, a]
    elif mode == 'any2spe':
        nonadjusted_sub_sites = sub_sg[:, sg, d]
    elif mode == 'any2any':
        nonadjusted_sub_sites = sub_sg[:, sg]
    for node in g['tree'].traverse():
        nl = node.numerical_label
        sub_sites[nl,:] = nonadjusted_sub_sites * g['is_site_nonmissing'][nl,:]
        total_sub_sites = sub_sites[nl,:].sum()
        total_sub_sites = 1 if (total_sub_sites==0) else total_sub_sites
        sub_sites[nl,:] = sub_sites[nl,:] / total_sub_sites
    return sub_sites

def get_sub_branches(sub_bg, mode, sg, a, d):
    if mode == 'spe2spe':
        sub_branches = sub_bg[:, sg, a, d]
    elif mode == 'spe2any':
        sub_branches = sub_bg[:, sg, a]
    elif mode == 'any2spe':
        sub_branches = sub_bg[:, sg, d]
    elif mode == 'any2any':
        sub_branches = sub_bg[:, sg]
    return sub_branches

def get_substitutions_per_branch(cb, b, g):
    for a in numpy.arange(g['current_arity']):
        b_tmp = b.loc[:,['branch_id','S_sub','N_sub']]
        b_tmp.columns = [ c+'_'+str(a+1) for c in b_tmp.columns ]
        cb = pandas.merge(cb, b_tmp, on='branch_id_'+str(a+1), how='left')
        del b_tmp
    return(cb)

def add_dif_column(cb, col_dif, col_any, col_spe, tol):
    if ((col_any in cb.columns) & (col_spe in cb.columns)):
        cb.loc[:, col_dif] = cb[col_any] - cb[col_spe]
        is_negative = (cb[col_dif] < -tol)
        is_almost_zero = (~is_negative)&(cb[col_dif] < tol)
        cb.loc[is_negative, col_dif] = numpy.nan
        cb.loc[is_almost_zero, col_dif] = 0
    return cb

def add_dif_stats(cb, tol, prefix):
    for SN in ['S','N']:
        for anc in ['any','spe']:
            col_any = prefix+SN+anc+'2any'
            col_spe = prefix+SN+anc+'2spe'
            col_dif = prefix+SN+anc+'2dif'
            cb = add_dif_column(cb, col_dif, col_any, col_spe, tol)
        for des in ['any','spe','dif']:
            col_any = prefix+SN+'any2'+des
            col_spe = prefix+SN+'spe2'+des
            col_dif = prefix+SN+'dif2'+des
            cb = add_dif_column(cb, col_dif, col_any, col_spe, tol)
    return cb
import time
import numpy
import pandas
import joblib
from util.util import *

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
    return(df)

def get_s(sub_tensor, attr):
    column_names=['site',attr+'_sub']
    num_site = sub_tensor.shape[2]
    df = pandas.DataFrame(0, index=numpy.arange(0,num_site), columns=column_names)
    df['site'] = numpy.arange(0, num_site)
    df[attr+'_sub'] = numpy.nan_to_num(sub_tensor).sum(axis=4).sum(axis=3).sum(axis=1).sum(axis=0)
    df['site'] = df['site'].astype(int)
    df = df.sort_values(by='site')
    return(df)

def get_cs(id_combinations, sub_tensor, attr):
    num_site = sub_tensor.shape[2]
    df = numpy.zeros([num_site, 5])
    df[:, 0] = numpy.arange(num_site)
    for i in numpy.arange(id_combinations.shape[0]):
        for sg in numpy.arange(sub_tensor.shape[1]):
            df[:, 1] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], sg, :, :, :].sum(axis=(2, 3)).prod(axis=0))  # any2any
            df[:, 2] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], sg, :, :, :].sum(axis=3).prod(axis=0).sum(axis=1))  # spe2any
            df[:, 3] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], sg, :, :, :].sum(axis=2).prod(axis=0).sum(axis=1))  # any2spe
            df[:, 4] += numpy.nan_to_num(sub_tensor[id_combinations[i,:], sg, :, :, :].prod(axis=0).sum(axis=(1, 2)))  # spe2spe
    cn = ['site',] + [ attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    df = pandas.DataFrame(df, columns=cn)
    return (df)

def get_bs(S_tensor, N_tensor):
    num_site = S_tensor.shape[2]
    num_branch = S_tensor.shape[0]
    column_names=['branch_id','site','S_sub','N_sub']
    df = pandas.DataFrame(numpy.nan, index=numpy.arange(0,num_branch*num_site), columns=column_names)
    for i in numpy.arange(num_branch):
        ind = numpy.arange(i*num_site, (i+1)*num_site)
        df.loc[ind, 'site'] = numpy.arange(0, num_site)
        df.loc[ind, 'branch_id'] = i
        df.loc[ind, 'S_sub'] = numpy.nan_to_num(S_tensor[i, :, :, :, :]).sum(axis=(0,2,3))
        df.loc[ind, 'N_sub'] = numpy.nan_to_num(N_tensor[i, :, :, :, :]).sum(axis=(0,2,3))
    return(df)

def sub_tensor2cb_obsolete(id_combinations, sub_tensor): # always slower than the other
    arity = id_combinations.shape[1]
    df = numpy.zeros([id_combinations.shape[0], arity+4])
    df[:, :arity] = id_combinations[:, :]  # branch_ids
    chunk = 10000
    for i in numpy.arange(numpy.ceil(id_combinations.shape[0]/chunk)):
        start = int(i*chunk)
        end = int(((i+1)*chunk)-1) if (id_combinations.shape[0] >= int(((i+1)*chunk)-1)) else id_combinations.shape[0]
        any2any_shape = (end-start,sub_tensor.shape[2])
        any2any = numpy.zeros(shape=any2any_shape, dtype=numpy.float64)
        for sg in numpy.arange(sub_tensor.shape[1]):
            any2any_sg = numpy.ones(shape=any2any_shape, dtype=numpy.float64)
            for a in numpy.arange(arity):
                any2any_sg *= sub_tensor[id_combinations[start:end,a], sg, :, :, :].sum(axis=(2, 3))
            any2any += any2any_sg
        df[start:end,arity+0] = any2any.sum(axis=1)
        del any2any, any2any_sg, any2any_shape
        spe2any_shape = (end-start,sub_tensor.shape[2],sub_tensor.shape[4])
        spe2any = numpy.zeros(shape=spe2any_shape, dtype=numpy.float64)
        for sg in numpy.arange(sub_tensor.shape[1]):
            spe2any_sg = numpy.ones(shape=spe2any_shape, dtype=numpy.float64)
            for a in numpy.arange(arity):
                spe2any_sg *= sub_tensor[id_combinations[start:end,a], sg, :, :, :].sum(axis=3)
            spe2any += spe2any_sg
        df[start:end,arity+1] = spe2any.sum(axis=2).sum(axis=1)
        del spe2any, spe2any_sg, spe2any_shape
        any2spe_shape = (end-start, sub_tensor.shape[2], sub_tensor.shape[3])
        any2spe = numpy.zeros(shape=any2spe_shape, dtype=numpy.float64)
        for sg in numpy.arange(sub_tensor.shape[1]):
            any2spe_sg = numpy.ones(shape=any2spe_shape, dtype=numpy.float64)
            for a in numpy.arange(arity):
                any2spe_sg *= sub_tensor[id_combinations[start:end, a], sg, :, :, :].sum(axis=2)
            any2spe += any2spe_sg
        df[start:end, arity + 2] = any2spe.sum(axis=2).sum(axis=1)
        del any2spe, any2spe_sg, any2spe_shape
        spe2spe_shape = (end-start,sub_tensor.shape[2],sub_tensor.shape[3],sub_tensor.shape[4])
        spe2spe = numpy.zeros(shape=spe2spe_shape, dtype=numpy.float64)
        for sg in numpy.arange(sub_tensor.shape[1]):
            spe2spe_sg = numpy.ones(shape=spe2spe_shape, dtype=numpy.float64)
            for a in numpy.arange(arity):
                spe2spe_sg *= sub_tensor[id_combinations[start:end, a], sg, :, :, :]
            spe2spe += spe2spe_sg
        df[start:end, arity + 3] = spe2spe.sum(axis=(2, 3)).sum(axis=1)
        del spe2spe, spe2spe_sg, spe2spe_shape
    return (df)

def sub_tensor2cb(id_combinations, sub_tensor):
    arity = id_combinations.shape[1]
    df = numpy.zeros([id_combinations.shape[0], arity+4])
    df[:, :arity] = id_combinations[:, :]  # branch_ids
    for i in numpy.arange(id_combinations.shape[0]):
        for sg in numpy.arange(sub_tensor.shape[1]):
            df[i, arity+0] += sub_tensor[id_combinations[i,:], sg, :, :, :].sum(axis=(2, 3)).prod(axis=0).sum(axis=0)  # any2any
            df[i, arity+1] += sub_tensor[id_combinations[i,:], sg, :, :, :].sum(axis=3).prod(axis=0).sum(axis=1).sum(axis=0)  # spe2any
            df[i, arity+2] += sub_tensor[id_combinations[i,:], sg, :, :, :].sum(axis=2).prod(axis=0).sum(axis=1).sum(axis=0)  # any2spe
            df[i, arity+3] += sub_tensor[id_combinations[i,:], sg, :, :, :].prod(axis=0).sum(axis=(1, 2)).sum(axis=0)  # spe2spe
    return (df)

def get_cb(id_combinations, sub_tensor, g, attr):
    arity = id_combinations.shape[1]
    if g['nslots']==1:
        df = sub_tensor2cb(id_combinations, sub_tensor)
    else:
        chunks = [ (id_combinations.shape[0]+i)//g['nslots'] for i in range(g['nslots']) ]
        id_chunks = list()
        i = 0
        for c in chunks:
            id_chunks.append(id_combinations[i:i+c,:])
            i+= c
        results = joblib.Parallel(n_jobs=g['nslots'])( joblib.delayed(sub_tensor2cb)(ids, sub_tensor) for ids in id_chunks )
        del sub_tensor
        df = numpy.concatenate(results, axis=0)
        del results
    cn = [ "branch_id_" + str(num+1) for num in range(0,arity) ]
    cn = cn + [ attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    df = pandas.DataFrame(df, columns=cn)
    df = sort_labels(df)
    df = df.dropna()
    return df

def sub_tensor2cbs(id_combinations, sub_tensor):
    arity = id_combinations.shape[1]
    num_site = sub_tensor.shape[2]
    df = numpy.zeros([id_combinations.shape[0]*num_site, arity+5])
    node=0
    start = time.time()
    for i in numpy.arange(id_combinations.shape[0]):
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

def get_cbs(id_combinations, sub_tensor, attr, nslots):
    print("Calculating combinatorial substitutions: attr =", attr, flush=True)
    arity = id_combinations.shape[1]
    cn1 = [ "branch_id_" + str(num+1) for num in range(0,arity) ]
    cn2 = ["site",]
    cn3 = [ attr+subs for subs in ["any2any","spe2any","any2spe","spe2spe"] ]
    if nslots==1:
        df = sub_tensor2cbs(id_combinations, sub_tensor)
    else:
        chunks = [ (id_combinations.shape[0]+i)//nslots for i in range(nslots) ]
        id_chunks = list()
        i = 0
        for c in chunks:
            id_chunks.append(id_combinations[i:i+c,:])
            i+= c
        results = joblib.Parallel(n_jobs=nslots)( joblib.delayed(sub_tensor2cbs)(ids, sub_tensor) for ids in id_chunks )
        del sub_tensor
        df = numpy.concatenate(results, axis=0)
        del results
    df = pandas.DataFrame(df, columns=cn1+cn2+cn3)
    df = df.dropna()
    df = sort_labels(df)
    print(type(df), flush=True)
    return(df)
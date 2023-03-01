import numpy
cimport numpy
cimport cython

import time

@cython.nonecheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
cpdef calc_combinatorial_sub_float32(long[:,:] id_combinations, int mmap_start,
                                     numpy.ndarray[numpy.float32_t, ndim=5, cast=True] sub_tensor, mmap, df_mmap):
    cdef int start_time = int(time.time())
    cdef int arity = id_combinations.shape[1]
    cdef int start = mmap_start
    cdef int end = mmap_start + id_combinations.shape[0]
    cdef Py_ssize_t i,j,sg
    cdef int mmap_end
    cdef int elapsed_sec
    if mmap:
        df = df_mmap
    else:
        df = numpy.zeros([id_combinations.shape[0], arity + 4], dtype=numpy.float32)
    df[start:end, :arity] = id_combinations[:, :]  # branch_ids
    for i,j in zip(numpy.arange(start, end),numpy.arange(id_combinations.shape[0])):
        for sg in numpy.arange(sub_tensor.shape[2]):
            df[i, arity+0] += sub_tensor[id_combinations[j,:], :, sg, :, :].sum(axis=(2, 3)).prod(axis=0).sum(axis=0) # any2any
            df[i, arity+1] += sub_tensor[id_combinations[j,:], :, sg, :, :].sum(axis=3).prod(axis=0).sum(axis=1).sum(axis=0) # spe2any
            df[i, arity+2] += sub_tensor[id_combinations[j,:], :, sg, :, :].sum(axis=2).prod(axis=0).sum(axis=1).sum(axis=0) # any2spe
            df[i, arity+3] += sub_tensor[id_combinations[j,:], :, sg, :, :].prod(axis=0).sum(axis=(1, 2)).sum(axis=0) # spe2spe
        if j % 1000 == 0:
            mmap_end = mmap_start + id_combinations.shape[0]
            elapsed_sec = int(time.time()) - start_time
            txt = 'cb: {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(j, mmap_start, mmap_end, elapsed_sec))
    return numpy.asarray(df)
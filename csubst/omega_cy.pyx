import numpy
cimport numpy as cnumpy
cimport cython
from cython.parallel import prange


@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef get_permutations(long[:,:] cb_ids, long[:] sites, long[:] sub_branches, double[:] p, long niter):
    cdef cnumpy.ndarray[cnumpy.uint8_t, ndim = 3, cast=True] ps
    cdef cnumpy.ndarray[cnumpy.uint8_t, ndim = 2, cast=True] sub_bool_array
    cdef long[:,:] num_shared_sub = numpy.zeros(shape=(cb_ids.shape[0], niter), dtype=numpy.long)
    cdef long[:] num_shared_sub2 = numpy.zeros(shape=(niter), dtype=numpy.long)
    cdef long size
    cdef long prev
    cdef long[:] site_indices
    cdef Py_ssize_t i,j,s,b # Py_ssize_t is the proper C type for Python array indices.
    ps = numpy.zeros(shape=(sub_branches.shape[0], niter, sites.shape[0]), dtype=numpy.bool_)
    for i in range(sub_branches.shape[0]):
        size = sub_branches[i]
        if size!=0:
            if size in sub_branches[:i]:
                prev = numpy.arange(i)[numpy.equal(sub_branches[:i], size)][0]
                ps[i,:,:] = ps[prev,numpy.random.permutation(numpy.arange(ps.shape[1])),:]
            else:
                for j in range(niter):
                    site_indices = numpy.random.choice(a=sites, size=size, replace=False, p=p)
                    ps[i,j,site_indices] = True
    for i in range(cb_ids.shape[0]):
        for b in range(cb_ids[i,:].shape[0]):
            if cb_ids[i,b]==cb_ids[i,0]:
                sub_bool_array = ps[cb_ids[i,b],:,:].copy()
            else:
                sub_bool_array *= ps[cb_ids[i,b],:,:]
        num_shared_sub2 = sub_bool_array.sum(axis=1)
        num_shared_sub[i,:] = num_shared_sub2
    return numpy.asarray(num_shared_sub)










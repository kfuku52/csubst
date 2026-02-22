import numpy
cimport numpy
cimport cython


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_permutations(long[:,:] cb_ids, long[:] sites, long[:] sub_branches, double[:] p, long niter):
    cdef numpy.ndarray[numpy.uint8_t, ndim=3, cast=True] ps
    cdef numpy.ndarray[numpy.uint8_t, ndim=2, cast=True] sub_bool_array
    cdef long[:,:] num_shared_sub = numpy.zeros(shape=(cb_ids.shape[0], niter), dtype=numpy.int64)
    cdef long[:] num_shared_sub2 = numpy.zeros(shape=(niter), dtype=numpy.int64)
    cdef long size
    cdef long prev
    cdef long[:] site_indices
    cdef Py_ssize_t i, j, b
    ps = numpy.zeros(shape=(sub_branches.shape[0], niter, sites.shape[0]), dtype=bool)
    for i in range(sub_branches.shape[0]):
        size = sub_branches[i]
        if size != 0:
            if size in sub_branches[:i]:
                prev = numpy.arange(i)[numpy.equal(sub_branches[:i], size)][0]
                ps[i, :, :] = ps[prev, numpy.random.permutation(numpy.arange(ps.shape[1])), :]
            else:
                for j in range(niter):
                    site_indices = numpy.random.choice(a=sites, size=size, replace=False, p=p)
                    ps[i, j, site_indices] = True
    for i in range(cb_ids.shape[0]):
        for b in range(cb_ids[i, :].shape[0]):
            if cb_ids[i, b] == cb_ids[i, 0]:
                sub_bool_array = ps[cb_ids[i, b], :, :].copy()
            else:
                sub_bool_array *= ps[cb_ids[i, b], :, :]
        num_shared_sub2 = sub_bool_array.sum(axis=1)
        num_shared_sub[i, :] = num_shared_sub2
    return numpy.asarray(num_shared_sub)


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef project_expected_state_block_double(
    numpy.ndarray[numpy.float64_t, ndim=2] parent_state_block,
    numpy.ndarray[numpy.float64_t, ndim=2] transition_prob,
    double float_tol,
):
    cdef Py_ssize_t num_site = parent_state_block.shape[0]
    cdef Py_ssize_t num_state = parent_state_block.shape[1]
    cdef numpy.ndarray[numpy.float64_t, ndim=2] expected_state_block
    cdef double[:, :] parent_mv
    cdef double[:, :] trans_mv
    cdef double[:, :] expected_mv
    cdef Py_ssize_t i, a, d
    cdef double acc
    cdef double row_sum

    if transition_prob.shape[0] != transition_prob.shape[1]:
        raise ValueError('transition_prob should be a square matrix.')
    if transition_prob.shape[0] != num_state:
        raise ValueError('State size mismatch between parent_state_block and transition_prob.')

    expected_state_block = numpy.zeros((num_site, num_state), dtype=numpy.float64)
    parent_mv = parent_state_block
    trans_mv = transition_prob
    expected_mv = expected_state_block
    for i in range(num_site):
        for d in range(num_state):
            acc = 0.0
            for a in range(num_state):
                acc += parent_mv[i, a] * trans_mv[a, d]
            expected_mv[i, d] = acc
        row_sum = 0.0
        for d in range(num_state):
            row_sum += expected_mv[i, d]
        if (row_sum > 0.0) and ((row_sum - 1.0) > float_tol):
            for d in range(num_state):
                expected_mv[i, d] /= row_sum
    return expected_state_block

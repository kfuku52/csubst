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


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calc_tmp_E_sum_double(
    numpy.ndarray[numpy.int64_t, ndim=2] cb_ids,
    numpy.ndarray[numpy.float64_t, ndim=2] sub_sites,
    numpy.ndarray[numpy.float64_t, ndim=1] sub_branches,
):
    cdef Py_ssize_t num_cb = cb_ids.shape[0]
    cdef Py_ssize_t arity = cb_ids.shape[1]
    cdef Py_ssize_t num_branch = sub_sites.shape[0]
    cdef Py_ssize_t num_site = sub_sites.shape[1]
    cdef numpy.ndarray[numpy.float64_t, ndim=1] out
    cdef long[:, :] cb_mv
    cdef double[:, :] site_mv
    cdef double[:] branch_mv
    cdef double[:] out_mv
    cdef Py_ssize_t i, j, b
    cdef long bid
    cdef double total, prod

    if sub_branches.shape[0] != num_branch:
        raise ValueError('sub_sites and sub_branches shape mismatch.')
    out = numpy.zeros((num_cb,), dtype=numpy.float64)
    if num_cb == 0:
        return out
    if arity <= 0:
        raise ValueError('cb_ids should have at least one column.')

    cb_mv = cb_ids
    site_mv = sub_sites
    branch_mv = sub_branches
    out_mv = out
    for i in range(num_cb):
        total = 0.0
        for j in range(num_site):
            prod = 1.0
            for b in range(arity):
                bid = cb_mv[i, b]
                if (bid < 0) or (bid >= num_branch):
                    raise ValueError('cb_ids contain out-of-range branch ID.')
                prod *= site_mv[bid, j] * branch_mv[bid]
            total += prod
        out_mv[i] = total
    return out


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calc_shared_counts_packed_uint8(
    numpy.ndarray[numpy.uint8_t, ndim=3] packed_masks,
    numpy.ndarray[numpy.int64_t, ndim=2] remapped_cb_ids,
):
    cdef Py_ssize_t num_branch = packed_masks.shape[0]
    cdef Py_ssize_t niter = packed_masks.shape[1]
    cdef Py_ssize_t num_packed_site = packed_masks.shape[2]
    cdef Py_ssize_t num_cb = remapped_cb_ids.shape[0]
    cdef Py_ssize_t arity = remapped_cb_ids.shape[1]
    cdef numpy.ndarray[numpy.int32_t, ndim=2] out
    cdef unsigned char[:, :, :] mask_mv
    cdef long[:, :] cb_mv
    cdef int[:, :] out_mv
    cdef Py_ssize_t i, j, col, b
    cdef long bid
    cdef unsigned int shared
    cdef unsigned int x
    cdef int count

    if arity <= 0:
        raise ValueError('remapped_cb_ids should have at least one column.')
    out = numpy.zeros((num_cb, niter), dtype=numpy.int32)
    if num_cb == 0:
        return out

    mask_mv = packed_masks
    cb_mv = remapped_cb_ids
    out_mv = out
    for i in range(num_cb):
        for col in range(arity):
            bid = cb_mv[i, col]
            if (bid < 0) or (bid >= num_branch):
                raise ValueError('remapped_cb_ids contain out-of-range branch IDs.')

    for i in range(num_cb):
        for j in range(niter):
            count = 0
            for b in range(num_packed_site):
                shared = <unsigned int>mask_mv[cb_mv[i, 0], j, b]
                for col in range(1, arity):
                    shared &= <unsigned int>mask_mv[cb_mv[i, col], j, b]
                x = shared
                x = x - ((x >> 1) & 0x55)
                x = (x & 0x33) + ((x >> 2) & 0x33)
                count += <int>((x + (x >> 4)) & 0x0F)
            out_mv[i, j] = count
    return out


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pack_sampled_site_indices_uint8(
    numpy.ndarray[numpy.int64_t, ndim=2] sampled_site_indices,
    long num_site,
):
    cdef Py_ssize_t niter = sampled_site_indices.shape[0]
    cdef Py_ssize_t size = sampled_site_indices.shape[1]
    cdef Py_ssize_t num_packed_site
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] out
    cdef long[:, :] site_mv
    cdef unsigned char[:, :] out_mv
    cdef Py_ssize_t i, j
    cdef long site_id
    cdef Py_ssize_t byte_index
    cdef unsigned char bit_mask

    if num_site < 0:
        raise ValueError('num_site should be >= 0.')
    num_packed_site = (num_site + 7) // 8
    out = numpy.zeros((niter, num_packed_site), dtype=numpy.uint8)
    if (niter == 0) or (size == 0) or (num_site == 0):
        return out

    site_mv = sampled_site_indices
    out_mv = out
    for i in range(niter):
        for j in range(size):
            site_id = site_mv[i, j]
            if (site_id < 0) or (site_id >= num_site):
                raise ValueError('sampled_site_indices contain out-of-range site IDs.')
            byte_index = site_id >> 3
            bit_mask = <unsigned char>(1 << (7 - (site_id & 7)))
            out_mv[i, byte_index] |= bit_mask
    return out

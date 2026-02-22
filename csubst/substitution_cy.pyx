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


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calc_combinatorial_sub_double_arity2(
    long[:, :] id_combinations,
    int mmap_start,
    numpy.ndarray[numpy.float64_t, ndim=5] sub_tensor,
    bint mmap,
    object df_mmap,
):
    cdef Py_ssize_t n_comb = id_combinations.shape[0]
    cdef Py_ssize_t n_site = sub_tensor.shape[1]
    cdef Py_ssize_t n_group = sub_tensor.shape[2]
    cdef Py_ssize_t n_from = sub_tensor.shape[3]
    cdef Py_ssize_t n_to = sub_tensor.shape[4]
    cdef numpy.ndarray[numpy.float64_t, ndim=2] df
    cdef double[:, :] dfv
    cdef Py_ssize_t i, j, s, sg, a, d
    cdef Py_ssize_t b0, b1
    cdef Py_ssize_t row
    cdef double any2any, spe2any, any2spe, spe2spe
    cdef double sum0, sum1, row0, row1, v0, v1
    cdef double col0[128]
    cdef double col1[128]
    cdef int mmap_end
    cdef int elapsed_sec
    cdef int start_time = int(time.time())
    if id_combinations.shape[1] != 2:
        raise ValueError('Cython fast path requires arity=2')
    if n_to > 128:
        raise ValueError('Cython fast path supports up to 128 derived states')
    if mmap:
        df = df_mmap
    else:
        df = numpy.zeros((n_comb, 6), dtype=numpy.float64)
    dfv = df
    for j in range(n_comb):
        b0 = <Py_ssize_t>id_combinations[j, 0]
        b1 = <Py_ssize_t>id_combinations[j, 1]
        if mmap:
            row = <Py_ssize_t>mmap_start + j
        else:
            row = j
        dfv[row, 0] = b0
        dfv[row, 1] = b1
        any2any = 0.0
        spe2any = 0.0
        any2spe = 0.0
        spe2spe = 0.0
        for s in range(n_site):
            for sg in range(n_group):
                for d in range(n_to):
                    col0[d] = 0.0
                    col1[d] = 0.0
                sum0 = 0.0
                sum1 = 0.0
                for a in range(n_from):
                    row0 = 0.0
                    row1 = 0.0
                    for d in range(n_to):
                        v0 = sub_tensor[b0, s, sg, a, d]
                        v1 = sub_tensor[b1, s, sg, a, d]
                        row0 += v0
                        row1 += v1
                        col0[d] += v0
                        col1[d] += v1
                        spe2spe += v0 * v1
                    sum0 += row0
                    sum1 += row1
                    spe2any += row0 * row1
                any2any += sum0 * sum1
                for d in range(n_to):
                    any2spe += col0[d] * col1[d]
        dfv[row, 2] += any2any
        dfv[row, 3] += spe2any
        dfv[row, 4] += any2spe
        dfv[row, 5] += spe2spe
        if j % 10000 == 0:
            mmap_end = mmap_start + n_comb
            elapsed_sec = int(time.time()) - start_time
            txt = 'cb(cython): {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(j, mmap_start, mmap_end, elapsed_sec), flush=True)
    if mmap:
        return None
    return numpy.asarray(df)


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calc_combinatorial_sub_by_site_double_arity2(
    long[:, :] id_combinations,
    int mmap_start,
    numpy.ndarray[numpy.float64_t, ndim=5] sub_tensor,
    bint mmap,
    object df_mmap,
):
    cdef Py_ssize_t n_comb = id_combinations.shape[0]
    cdef Py_ssize_t n_site = sub_tensor.shape[1]
    cdef Py_ssize_t n_group = sub_tensor.shape[2]
    cdef Py_ssize_t n_from = sub_tensor.shape[3]
    cdef Py_ssize_t n_to = sub_tensor.shape[4]
    cdef numpy.ndarray[numpy.float64_t, ndim=2] df
    cdef double[:, :] dfv
    cdef Py_ssize_t j, s, sg, a, d
    cdef Py_ssize_t b0, b1
    cdef Py_ssize_t row, row_start
    cdef double any2any, spe2any, any2spe, spe2spe
    cdef double sum0, sum1, row0, row1, v0, v1
    cdef double col0[128]
    cdef double col1[128]
    cdef int mmap_end
    cdef int elapsed_sec
    cdef int start_time = int(time.time())
    if id_combinations.shape[1] != 2:
        raise ValueError('Cython fast path requires arity=2')
    if n_to > 128:
        raise ValueError('Cython fast path supports up to 128 derived states')
    if mmap:
        df = df_mmap
    else:
        df = numpy.zeros((n_comb * n_site, 7), dtype=numpy.float64)
    dfv = df
    for j in range(n_comb):
        b0 = <Py_ssize_t>id_combinations[j, 0]
        b1 = <Py_ssize_t>id_combinations[j, 1]
        if mmap:
            row_start = (<Py_ssize_t>mmap_start + j) * n_site
        else:
            row_start = j * n_site
        for s in range(n_site):
            row = row_start + s
            dfv[row, 0] = b0
            dfv[row, 1] = b1
            dfv[row, 2] = s
            any2any = 0.0
            spe2any = 0.0
            any2spe = 0.0
            spe2spe = 0.0
            for sg in range(n_group):
                for d in range(n_to):
                    col0[d] = 0.0
                    col1[d] = 0.0
                sum0 = 0.0
                sum1 = 0.0
                for a in range(n_from):
                    row0 = 0.0
                    row1 = 0.0
                    for d in range(n_to):
                        v0 = sub_tensor[b0, s, sg, a, d]
                        v1 = sub_tensor[b1, s, sg, a, d]
                        row0 += v0
                        row1 += v1
                        col0[d] += v0
                        col1[d] += v1
                        spe2spe += v0 * v1
                    sum0 += row0
                    sum1 += row1
                    spe2any += row0 * row1
                any2any += sum0 * sum1
                for d in range(n_to):
                    any2spe += col0[d] * col1[d]
            dfv[row, 3] += any2any
            dfv[row, 4] += spe2any
            dfv[row, 5] += any2spe
            dfv[row, 6] += spe2spe
        if j % 10000 == 0:
            mmap_end = mmap_start + n_comb
            elapsed_sec = int(time.time()) - start_time
            txt = 'cbs(cython): {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(j, mmap_start, mmap_end, elapsed_sec), flush=True)
    if mmap:
        return None
    return numpy.asarray(df)


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calc_combinatorial_sub_sparse_summary_double_arity2(
    long[:, :] id_combinations,
    int mmap_start,
    object branch_group_site_total_obj,
    object branch_group_from_site_obj,
    object branch_group_to_site_obj,
    object branch_group_pair_site_obj,
    bint mmap,
    object df_mmap,
    bint calc_any2any,
    bint calc_spe2any,
    bint calc_any2spe,
    bint calc_spe2spe,
):
    cdef Py_ssize_t n_comb = id_combinations.shape[0]
    cdef numpy.ndarray[numpy.float64_t, ndim=2] df
    cdef double[:, :] dfv
    cdef numpy.ndarray[numpy.float64_t, ndim=3] branch_group_site_total
    cdef numpy.ndarray[numpy.float64_t, ndim=4] branch_group_from_site
    cdef numpy.ndarray[numpy.float64_t, ndim=4] branch_group_to_site
    cdef numpy.ndarray[numpy.float64_t, ndim=4] branch_group_pair_site
    cdef numpy.ndarray[numpy.float64_t, ndim=2] total2d
    cdef numpy.ndarray[numpy.float64_t, ndim=2] from2d
    cdef numpy.ndarray[numpy.float64_t, ndim=2] to2d
    cdef numpy.ndarray[numpy.float64_t, ndim=2] pair2d
    cdef double[:, :] totalv
    cdef double[:, :] fromv
    cdef double[:, :] tov
    cdef double[:, :] pairv
    cdef Py_ssize_t n_total = 0
    cdef Py_ssize_t n_fromsite = 0
    cdef Py_ssize_t n_tosite = 0
    cdef Py_ssize_t n_pairsite = 0
    cdef Py_ssize_t j, k, row
    cdef Py_ssize_t b0, b1
    cdef double any2any, spe2any, any2spe, spe2spe
    cdef int mmap_end
    cdef int elapsed_sec
    cdef int start_time = int(time.time())

    if id_combinations.shape[1] != 2:
        raise ValueError('Cython sparse-summary fast path requires arity=2')
    if not (calc_any2any or calc_spe2any or calc_any2spe or calc_spe2spe):
        raise ValueError('At least one sparse summary statistic should be enabled.')

    if calc_any2any:
        if branch_group_site_total_obj is None:
            raise ValueError('branch_group_site_total is required when calc_any2any is True.')
        branch_group_site_total = branch_group_site_total_obj
        total2d = branch_group_site_total.reshape((branch_group_site_total.shape[0], -1))
        totalv = total2d
        n_total = total2d.shape[1]
    if calc_spe2any:
        if branch_group_from_site_obj is None:
            raise ValueError('branch_group_from_site is required when calc_spe2any is True.')
        branch_group_from_site = branch_group_from_site_obj
        from2d = branch_group_from_site.reshape((branch_group_from_site.shape[0], -1))
        fromv = from2d
        n_fromsite = from2d.shape[1]
    if calc_any2spe:
        if branch_group_to_site_obj is None:
            raise ValueError('branch_group_to_site is required when calc_any2spe is True.')
        branch_group_to_site = branch_group_to_site_obj
        to2d = branch_group_to_site.reshape((branch_group_to_site.shape[0], -1))
        tov = to2d
        n_tosite = to2d.shape[1]
    if calc_spe2spe:
        if branch_group_pair_site_obj is None:
            raise ValueError('branch_group_pair_site is required when calc_spe2spe is True.')
        branch_group_pair_site = branch_group_pair_site_obj
        pair2d = branch_group_pair_site.reshape((branch_group_pair_site.shape[0], -1))
        pairv = pair2d
        n_pairsite = pair2d.shape[1]

    if mmap:
        df = df_mmap
    else:
        df = numpy.zeros((n_comb, 6), dtype=numpy.float64)
    dfv = df
    for j in range(n_comb):
        b0 = <Py_ssize_t>id_combinations[j, 0]
        b1 = <Py_ssize_t>id_combinations[j, 1]
        if mmap:
            row = <Py_ssize_t>mmap_start + j
        else:
            row = j
        dfv[row, 0] = b0
        dfv[row, 1] = b1
        any2any = 0.0
        spe2any = 0.0
        any2spe = 0.0
        spe2spe = 0.0
        if calc_any2any:
            for k in range(n_total):
                any2any += totalv[b0, k] * totalv[b1, k]
        if calc_spe2any:
            for k in range(n_fromsite):
                spe2any += fromv[b0, k] * fromv[b1, k]
        if calc_any2spe:
            for k in range(n_tosite):
                any2spe += tov[b0, k] * tov[b1, k]
        if calc_spe2spe:
            for k in range(n_pairsite):
                spe2spe += pairv[b0, k] * pairv[b1, k]
        dfv[row, 2] += any2any
        dfv[row, 3] += spe2any
        dfv[row, 4] += any2spe
        dfv[row, 5] += spe2spe
        if j % 10000 == 0:
            mmap_end = mmap_start + n_comb
            elapsed_sec = int(time.time()) - start_time
            txt = 'cb(sparse cython): {:,}th in the id range {:,}-{:,}: {:,} sec'
            print(txt.format(j, mmap_start, mmap_end, elapsed_sec), flush=True)
    if mmap:
        return None
    return numpy.asarray(df)


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fill_sub_tensor_asis_branch_double(
    numpy.ndarray[numpy.float64_t, ndim=2] parent_matrix,
    numpy.ndarray[numpy.float64_t, ndim=2] child_matrix,
    numpy.ndarray[numpy.float64_t, ndim=3] out_branch_group,
):
    cdef Py_ssize_t n_site = parent_matrix.shape[0]
    cdef Py_ssize_t n_state = parent_matrix.shape[1]
    cdef double[:, :] parent_mv = parent_matrix
    cdef double[:, :] child_mv = child_matrix
    cdef double[:, :, :] out_mv = out_branch_group
    cdef Py_ssize_t s, a, d
    cdef double pa
    cdef double cd
    if child_matrix.shape[0] != n_site or child_matrix.shape[1] != n_state:
        raise ValueError('child_matrix shape should match parent_matrix shape.')
    if out_branch_group.shape[0] != n_site or out_branch_group.shape[1] != n_state or out_branch_group.shape[2] != n_state:
        raise ValueError('out_branch_group shape should be [site,state,state] with matching state/site sizes.')
    for s in range(n_site):
        for a in range(n_state):
            pa = parent_mv[s, a]
            for d in range(n_state):
                if a == d:
                    out_mv[s, a, d] = 0.0
                else:
                    cd = child_mv[s, d]
                    out_mv[s, a, d] = pa * cd


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fill_sub_tensor_syn_branch_double(
    numpy.ndarray[numpy.float64_t, ndim=2] parent_matrix,
    numpy.ndarray[numpy.float64_t, ndim=2] child_matrix,
    numpy.ndarray[numpy.int64_t, ndim=2] syn_index_matrix,
    numpy.ndarray[numpy.int64_t, ndim=1] syn_group_sizes,
    numpy.ndarray[numpy.float64_t, ndim=4] out_branch_tensor,
):
    cdef Py_ssize_t n_site = parent_matrix.shape[0]
    cdef Py_ssize_t n_state = parent_matrix.shape[1]
    cdef Py_ssize_t n_group = syn_group_sizes.shape[0]
    cdef Py_ssize_t max_syn = syn_index_matrix.shape[1]
    cdef double[:, :] parent_mv = parent_matrix
    cdef double[:, :] child_mv = child_matrix
    cdef long[:, :] idx_mv = syn_index_matrix
    cdef long[:] size_mv = syn_group_sizes
    cdef double[:, :, :, :] out_mv = out_branch_tensor
    cdef Py_ssize_t sg, s, ai, di
    cdef Py_ssize_t size
    cdef Py_ssize_t anc_state, der_state
    cdef double pa

    if child_matrix.shape[0] != n_site or child_matrix.shape[1] != n_state:
        raise ValueError('child_matrix shape should match parent_matrix shape.')
    if out_branch_tensor.shape[0] != n_site or out_branch_tensor.shape[1] != n_group:
        raise ValueError('out_branch_tensor first two dimensions should be [site,group].')
    if out_branch_tensor.shape[2] < max_syn or out_branch_tensor.shape[3] < max_syn:
        raise ValueError('out_branch_tensor synonymous dimensions should cover max syn group size.')

    for sg in range(n_group):
        size = size_mv[sg]
        if size <= 1:
            continue
        for s in range(n_site):
            for ai in range(size):
                anc_state = idx_mv[sg, ai]
                pa = parent_mv[s, anc_state]
                for di in range(size):
                    if ai == di:
                        out_mv[s, sg, ai, di] = 0.0
                    else:
                        der_state = idx_mv[sg, di]
                        out_mv[s, sg, ai, di] = pa * child_mv[s, der_state]

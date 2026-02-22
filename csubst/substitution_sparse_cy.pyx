import numpy
cimport numpy
cimport cython
from libc.math cimport fabs, isfinite

ctypedef fused index_t:
    numpy.int32_t
    numpy.int64_t


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dense_block_to_csr_arrays_double(
    numpy.ndarray[numpy.float64_t, ndim=2] block,
    double tol,
):
    cdef Py_ssize_t n_row = block.shape[0]
    cdef Py_ssize_t n_col = block.shape[1]
    cdef numpy.ndarray[numpy.int64_t, ndim=1] indptr
    cdef numpy.ndarray[numpy.int64_t, ndim=1] indices
    cdef numpy.ndarray[numpy.float64_t, ndim=1] data
    cdef double[:, :] block_mv = block
    cdef double[:] data_mv
    cdef long[:] indices_mv
    cdef long[:] indptr_mv
    cdef Py_ssize_t i, j, k, nnz_row, nnz_total
    cdef double v
    cdef bint use_tol = tol > 0.0

    indptr = numpy.zeros(n_row + 1, dtype=numpy.int64)
    indptr_mv = indptr
    nnz_total = 0
    for i in range(n_row):
        nnz_row = 0
        for j in range(n_col):
            v = block_mv[i, j]
            if use_tol:
                if not (fabs(v) <= tol):
                    nnz_row += 1
            else:
                if v != 0.0:
                    nnz_row += 1
        nnz_total += nnz_row
        indptr_mv[i + 1] = nnz_total

    data = numpy.empty(nnz_total, dtype=numpy.float64)
    indices = numpy.empty(nnz_total, dtype=numpy.int64)
    data_mv = data
    indices_mv = indices
    k = 0
    for i in range(n_row):
        for j in range(n_col):
            v = block_mv[i, j]
            if use_tol:
                if not (fabs(v) <= tol):
                    data_mv[k] = v
                    indices_mv[k] = j
                    k += 1
            else:
                if v != 0.0:
                    data_mv[k] = v
                    indices_mv[k] = j
                    k += 1
    return data, indices, indptr


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef accumulate_sparse_summary_block_double(
    numpy.ndarray[numpy.int64_t, ndim=1] rows,
    numpy.ndarray[numpy.int64_t, ndim=1] cols,
    numpy.ndarray[numpy.float64_t, ndim=1] vals,
    long sg,
    long a,
    long d,
    long num_group,
    long num_site,
    long num_state_from,
    long num_state_to,
    object total_flat_obj,
    object from_flat_obj,
    object to_flat_obj,
    object pair_flat_obj,
    bint need_any2any,
    bint need_spe2any,
    bint need_any2spe,
    bint need_spe2spe,
):
    cdef Py_ssize_t i
    cdef Py_ssize_t nnz = rows.shape[0]
    cdef long row, col, pair_index, num_state_pair
    cdef double v
    cdef double[:] total_flat
    cdef double[:] from_flat
    cdef double[:] to_flat
    cdef double[:] pair_flat
    cdef Py_ssize_t idx
    if cols.shape[0] != nnz or vals.shape[0] != nnz:
        raise ValueError('rows/cols/vals should have identical length.')
    if need_any2any:
        total_flat = total_flat_obj
    if need_spe2any:
        from_flat = from_flat_obj
    if need_any2spe:
        to_flat = to_flat_obj
    if need_spe2spe:
        pair_flat = pair_flat_obj
        num_state_pair = num_state_from * num_state_to
        pair_index = a * num_state_to + d
    for i in range(nnz):
        row = rows[i]
        col = cols[i]
        v = vals[i]
        if need_any2any:
            idx = ((row * num_group) + sg) * num_site + col
            total_flat[idx] += v
        if need_spe2any:
            idx = (((row * num_group) + sg) * num_state_from + a) * num_site + col
            from_flat[idx] += v
        if need_any2spe:
            idx = (((row * num_group) + sg) * num_state_to + d) * num_site + col
            to_flat[idx] += v
        if need_spe2spe:
            idx = (((row * num_group) + sg) * num_state_pair + pair_index) * num_site + col
            pair_flat[idx] += v


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef accumulate_branch_sub_counts_csr_double(
    numpy.ndarray[index_t, ndim=1] indptr,
    numpy.ndarray[numpy.float64_t, ndim=1] data,
    numpy.ndarray[numpy.float64_t, ndim=1] out,
):
    cdef Py_ssize_t n_branch = out.shape[0]
    cdef Py_ssize_t row, k
    cdef Py_ssize_t start, end
    cdef double total
    if indptr.shape[0] != (n_branch + 1):
        raise ValueError('indptr length should be out.shape[0] + 1.')
    for row in range(n_branch):
        start = indptr[row]
        end = indptr[row + 1]
        total = 0.0
        for k in range(start, end):
            total += data[k]
        out[row] += total


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef accumulate_site_sub_counts_csr_double(
    numpy.ndarray[index_t, ndim=1] indices,
    numpy.ndarray[numpy.float64_t, ndim=1] data,
    numpy.ndarray[numpy.float64_t, ndim=1] out,
):
    cdef Py_ssize_t nnz = data.shape[0]
    cdef Py_ssize_t k
    cdef Py_ssize_t col
    if indices.shape[0] != nnz:
        raise ValueError('indices and data should have identical length.')
    for k in range(nnz):
        col = indices[k]
        out[col] += data[k]


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef accumulate_branch_site_row_csr_double(
    numpy.ndarray[index_t, ndim=1] indptr,
    numpy.ndarray[index_t, ndim=1] indices,
    numpy.ndarray[numpy.float64_t, ndim=1] data,
    long branch_id,
    numpy.ndarray[numpy.float64_t, ndim=1] out,
):
    cdef Py_ssize_t n_branch = indptr.shape[0] - 1
    cdef Py_ssize_t start, end, k
    cdef Py_ssize_t col
    if branch_id < 0 or branch_id >= n_branch:
        raise ValueError('branch_id is out of range.')
    start = indptr[branch_id]
    end = indptr[branch_id + 1]
    for k in range(start, end):
        col = indices[k]
        out[col] += data[k]


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef scatter_branch_row_to_tensor_double(
    numpy.ndarray[index_t, ndim=1] indptr,
    numpy.ndarray[index_t, ndim=1] indices,
    numpy.ndarray[numpy.float64_t, ndim=1] data,
    long branch_id,
    long sg,
    long a,
    long d,
    numpy.ndarray[numpy.float64_t, ndim=4] out,
):
    cdef Py_ssize_t n_branch = indptr.shape[0] - 1
    cdef Py_ssize_t start, end, k
    cdef Py_ssize_t site
    if branch_id < 0 or branch_id >= n_branch:
        raise ValueError('branch_id is out of range.')
    if sg < 0 or sg >= out.shape[1]:
        raise ValueError('sg is out of range.')
    if a < 0 or a >= out.shape[2]:
        raise ValueError('a is out of range.')
    if d < 0 or d >= out.shape[3]:
        raise ValueError('d is out of range.')
    start = indptr[branch_id]
    end = indptr[branch_id + 1]
    for k in range(start, end):
        site = indices[k]
        out[site, sg, a, d] = data[k]


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef scan_sitewise_max_indices_double(
    numpy.ndarray[numpy.float64_t, ndim=4] branch_tensor,
    double min_sitewise_pp,
):
    cdef Py_ssize_t n_site = branch_tensor.shape[0]
    cdef Py_ssize_t n_group = branch_tensor.shape[1]
    cdef Py_ssize_t n_from = branch_tensor.shape[2]
    cdef Py_ssize_t n_to = branch_tensor.shape[3]
    cdef numpy.ndarray[numpy.int64_t, ndim=1] site_idx
    cdef numpy.ndarray[numpy.int64_t, ndim=1] anc_idx
    cdef numpy.ndarray[numpy.int64_t, ndim=1] der_idx
    cdef long[:] site_mv
    cdef long[:] anc_mv
    cdef long[:] der_mv
    cdef double[:, :, :, :] tv = branch_tensor
    cdef Py_ssize_t s, sg, a, d, out_count
    cdef double v
    cdef double max_val
    cdef bint has_finite
    cdef long best_a, best_d

    site_idx = numpy.empty(n_site, dtype=numpy.int64)
    anc_idx = numpy.empty(n_site, dtype=numpy.int64)
    der_idx = numpy.empty(n_site, dtype=numpy.int64)
    site_mv = site_idx
    anc_mv = anc_idx
    der_mv = der_idx
    out_count = 0

    for s in range(n_site):
        has_finite = False
        max_val = 0.0
        best_a = -1
        best_d = -1
        for sg in range(n_group):
            for a in range(n_from):
                for d in range(n_to):
                    v = tv[s, sg, a, d]
                    if not isfinite(v):
                        continue
                    if (not has_finite) or (v > max_val):
                        has_finite = True
                        max_val = v
                        best_a = <long>a
                        best_d = <long>d
        if has_finite and isfinite(max_val) and (max_val >= min_sitewise_pp):
            site_mv[out_count] = <long>s
            anc_mv[out_count] = best_a
            der_mv[out_count] = best_d
            out_count += 1

    return (
        numpy.asarray(site_idx[:out_count]),
        numpy.asarray(anc_idx[:out_count]),
        numpy.asarray(der_idx[:out_count]),
    )

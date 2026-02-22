import numpy
cimport numpy
cimport cython
from libc.math cimport fabs


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

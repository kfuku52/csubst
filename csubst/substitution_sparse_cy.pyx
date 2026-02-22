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

import numpy
cimport numpy
cimport cython

@cython.nonecheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
cpdef where_equal_1d(long[:] data, long val):
    cdef int xmax = data.shape[0]
    cdef Py_ssize_t x
    cdef int count = 0
    cdef Py_ssize_t[:] xind = numpy.zeros(xmax, dtype=numpy.long)
    for x in range(xmax):
        if (data[x] == val):
            xind[count] = x
            count += 1
    return xind[0:count]

@cython.nonecheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
cpdef nc_matrix2id_combinations(numpy.uint8_t[:,:] nc_matrix, int arity):
    cdef long[:] rows,cols
    rows, cols = numpy.where(numpy.equal(nc_matrix, 1))
    cdef long[:] unique_cols = numpy.unique(cols)
    cdef Py_ssize_t[:] ind = numpy.arange(unique_cols.shape[0], dtype=numpy.long)
    cdef Py_ssize_t[:] ind2  = numpy.arange(arity, dtype=numpy.long)
    cdef Py_ssize_t[:] row_ind
    cdef numpy.uint8_t[:] is_match
    cdef Py_ssize_t i,j
    cdef long[:,:] id_combinations = numpy.zeros(shape=(unique_cols.shape[0], arity), dtype=numpy.long)
    for i in ind:
        row_ind = where_equal_1d(cols, unique_cols[i])
        for j in ind2:
            id_combinations[i,j] = rows[row_ind[j]]
    return numpy.asarray(id_combinations)

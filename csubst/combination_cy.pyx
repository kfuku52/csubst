import numpy
cimport numpy
cimport cython

@cython.nonecheck(False)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
cdef where_equal_1d(long[:] data, long val):
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
cpdef generate_id_chunk(long[:,:] chunk, int start, int arity, Py_ssize_t[:] ind2, long[:] rows, long[:] cols, long[:] unique_cols):
    cdef Py_ssize_t[:] row_ind
    cdef Py_ssize_t i,j
    for i in range(chunk.shape[0]):
        row_ind = where_equal_1d(cols, unique_cols[i+start])
        for j in ind2:
            chunk[i,j] = rows[row_ind[j]]
    return numpy.asarray(chunk)
import numpy
cimport numpy
cimport cython


cdef inline int _nuc_code(unsigned char c):
    if c == 65:   # A
        return 0
    if c == 67:   # C
        return 1
    if c == 71:   # G
        return 2
    if c == 84:   # T
        return 3
    return -1


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fill_leaf_state_matrix_codon_unambiguous(
    bytes seq_bytes,
    numpy.ndarray[numpy.float64_t, ndim=2] state_matrix,
    numpy.ndarray[numpy.int64_t, ndim=1] codon_lookup,
):
    cdef Py_ssize_t num_codon_site = state_matrix.shape[0]
    cdef Py_ssize_t num_state = state_matrix.shape[1]
    cdef numpy.ndarray[numpy.uint8_t, ndim=1] unresolved
    cdef const unsigned char[:] seq_mv
    cdef unsigned char[:] unresolved_mv
    cdef long[:] lookup_mv
    cdef Py_ssize_t s, base
    cdef int c0, c1, c2
    cdef int code
    cdef long state_idx

    if codon_lookup.shape[0] != 64:
        raise ValueError('codon_lookup should have 64 entries.')
    if len(seq_bytes) != (num_codon_site * 3):
        raise ValueError('Sequence length should match state_matrix codon-site count.')

    unresolved = numpy.zeros(num_codon_site, dtype=numpy.uint8)
    seq_mv = seq_bytes
    unresolved_mv = unresolved
    lookup_mv = codon_lookup
    for s in range(num_codon_site):
        base = s * 3
        c0 = _nuc_code(seq_mv[base + 0])
        c1 = _nuc_code(seq_mv[base + 1])
        c2 = _nuc_code(seq_mv[base + 2])
        if (c0 < 0) or (c1 < 0) or (c2 < 0):
            unresolved_mv[s] = 1
            continue
        code = (c0 * 16) + (c1 * 4) + c2
        state_idx = lookup_mv[code]
        if (state_idx < 0) or (state_idx >= num_state):
            unresolved_mv[s] = 1
            continue
        state_matrix[s, state_idx] = 1.0
    return unresolved

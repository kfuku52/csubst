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
cpdef build_packed_sub_tensor_asis_double(
    numpy.ndarray[numpy.float64_t, ndim=3] state_tensor,
    numpy.ndarray[numpy.float64_t, ndim=3] state_tensor_anc,
    numpy.ndarray[numpy.int64_t, ndim=2] branch_pairs,
    long num_branch,
):
    """Build branch x (event*site) CSR arrays without Python entry maps."""
    cdef Py_ssize_t num_pair = branch_pairs.shape[0]
    cdef Py_ssize_t num_site = state_tensor.shape[1]
    cdef Py_ssize_t num_state = state_tensor.shape[2]
    cdef numpy.ndarray[numpy.int64_t, ndim=1] indptr = numpy.zeros(num_branch + 1, dtype=numpy.int64)
    cdef numpy.ndarray[numpy.int64_t, ndim=1] row_counts = numpy.zeros(num_branch, dtype=numpy.int64)
    cdef numpy.ndarray[numpy.int64_t, ndim=1] cursor
    cdef numpy.ndarray[numpy.int64_t, ndim=1] indices
    cdef numpy.ndarray[numpy.float64_t, ndim=1] data
    cdef const double[:, :, :] child_mv = state_tensor
    cdef const double[:, :, :] parent_mv = state_tensor_anc
    cdef const long[:, :] pairs_mv = branch_pairs
    cdef long[:] indptr_mv = indptr
    cdef long[:] counts_mv = row_counts
    cdef long[:] cursor_mv
    cdef long[:] indices_mv
    cdef double[:] data_mv
    cdef Py_ssize_t p, site, a, d, pos, nnz
    cdef long child, parent, event_id
    cdef double value

    for p in range(num_pair):
        child = pairs_mv[p, 0]
        parent = pairs_mv[p, 1]
        for a in range(num_state):
            for d in range(num_state):
                if a == d:
                    continue
                for site in range(num_site):
                    value = parent_mv[parent, site, a] * child_mv[child, site, d]
                    if value != 0.0:
                        counts_mv[child] += 1
    for child in range(num_branch):
        indptr_mv[child + 1] = indptr_mv[child] + counts_mv[child]
    nnz = indptr_mv[num_branch]
    indices = numpy.empty(nnz, dtype=numpy.int64)
    data = numpy.empty(nnz, dtype=numpy.float64)
    cursor = indptr[0:num_branch].copy()
    indices_mv = indices
    data_mv = data
    cursor_mv = cursor
    for p in range(num_pair):
        child = pairs_mv[p, 0]
        parent = pairs_mv[p, 1]
        for a in range(num_state):
            for d in range(num_state):
                if a == d:
                    continue
                event_id = a * num_state + d
                for site in range(num_site):
                    value = parent_mv[parent, site, a] * child_mv[child, site, d]
                    if value != 0.0:
                        pos = cursor_mv[child]
                        indices_mv[pos] = event_id * num_site + site
                        data_mv[pos] = value
                        cursor_mv[child] = pos + 1
    return data, indices, indptr


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef build_packed_sub_tensor_syn_double(
    numpy.ndarray[numpy.float64_t, ndim=3] state_tensor,
    numpy.ndarray[numpy.float64_t, ndim=3] state_tensor_anc,
    numpy.ndarray[numpy.int64_t, ndim=2] branch_pairs,
    numpy.ndarray[numpy.int64_t, ndim=2] state_indices,
    long num_branch,
):
    """Two-pass packed CSR builder for synonymous substitution groups."""
    cdef Py_ssize_t num_pair = branch_pairs.shape[0]
    cdef Py_ssize_t num_site = state_tensor.shape[1]
    cdef Py_ssize_t num_group = state_indices.shape[0]
    cdef Py_ssize_t num_state = state_indices.shape[1]
    cdef numpy.ndarray[numpy.int64_t, ndim=1] indptr = numpy.zeros(num_branch + 1, dtype=numpy.int64)
    cdef numpy.ndarray[numpy.int64_t, ndim=1] row_counts = numpy.zeros(num_branch, dtype=numpy.int64)
    cdef numpy.ndarray[numpy.int64_t, ndim=1] cursor
    cdef numpy.ndarray[numpy.int64_t, ndim=1] indices
    cdef numpy.ndarray[numpy.float64_t, ndim=1] data
    cdef const double[:, :, :] child_mv = state_tensor
    cdef const double[:, :, :] parent_mv = state_tensor_anc
    cdef const long[:, :] pairs_mv = branch_pairs
    cdef const long[:, :] states_mv = state_indices
    cdef long[:] indptr_mv = indptr
    cdef long[:] counts_mv = row_counts
    cdef long[:] cursor_mv
    cdef long[:] indices_mv
    cdef double[:] data_mv
    cdef Py_ssize_t p, site, sg, a, d, pos, nnz
    cdef long child, parent, source_a, source_d, event_id
    cdef double value

    for p in range(num_pair):
        child = pairs_mv[p, 0]
        parent = pairs_mv[p, 1]
        for sg in range(num_group):
            for a in range(num_state):
                source_a = states_mv[sg, a]
                if source_a < 0:
                    continue
                for d in range(num_state):
                    if a == d:
                        continue
                    source_d = states_mv[sg, d]
                    if source_d < 0:
                        continue
                    for site in range(num_site):
                        value = parent_mv[parent, site, source_a] * child_mv[child, site, source_d]
                        if value != 0.0:
                            counts_mv[child] += 1
    for child in range(num_branch):
        indptr_mv[child + 1] = indptr_mv[child] + counts_mv[child]
    nnz = indptr_mv[num_branch]
    indices = numpy.empty(nnz, dtype=numpy.int64)
    data = numpy.empty(nnz, dtype=numpy.float64)
    cursor = indptr[0:num_branch].copy()
    indices_mv = indices
    data_mv = data
    cursor_mv = cursor
    for p in range(num_pair):
        child = pairs_mv[p, 0]
        parent = pairs_mv[p, 1]
        for sg in range(num_group):
            for a in range(num_state):
                source_a = states_mv[sg, a]
                if source_a < 0:
                    continue
                for d in range(num_state):
                    if a == d:
                        continue
                    source_d = states_mv[sg, d]
                    if source_d < 0:
                        continue
                    event_id = (sg * num_state + a) * num_state + d
                    for site in range(num_site):
                        value = parent_mv[parent, site, source_a] * child_mv[child, site, source_d]
                        if value != 0.0:
                            pos = cursor_mv[child]
                            indices_mv[pos] = event_id * num_site + site
                            data_mv[pos] = value
                            cursor_mv[child] = pos + 1
    return data, indices, indptr


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
    cdef const double[:, :] block_mv = block
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
cpdef calc_sparse_projection_product_double(
    numpy.ndarray[numpy.int64_t, ndim=2] id_combinations,
    numpy.ndarray[index_t, ndim=1] indptr,
    numpy.ndarray[index_t, ndim=1] indices,
    numpy.ndarray[numpy.float64_t, ndim=1] data,
):
    cdef Py_ssize_t num_combination = id_combinations.shape[0]
    cdef Py_ssize_t arity = id_combinations.shape[1]
    cdef Py_ssize_t num_branch = indptr.shape[0] - 1
    cdef numpy.ndarray[numpy.float64_t, ndim=1] out
    cdef numpy.ndarray[numpy.int64_t, ndim=1] starts
    cdef numpy.ndarray[numpy.int64_t, ndim=1] ends
    cdef numpy.ndarray[numpy.int64_t, ndim=1] positions
    cdef double[:] out_mv
    cdef long[:] starts_mv
    cdef long[:] ends_mv
    cdef long[:] positions_mv
    cdef Py_ssize_t combo, slot, candidate, pos
    cdef Py_ssize_t branch_id, min_slot, min_count
    cdef Py_ssize_t col, current_count
    cdef double product_value, total
    cdef bint matched
    if arity < 1:
        raise ValueError('id_combinations should contain at least one branch column.')
    if indices.shape[0] != data.shape[0]:
        raise ValueError('indices and data should have identical length.')
    out = numpy.zeros(num_combination, dtype=numpy.float64)
    starts = numpy.empty(arity, dtype=numpy.int64)
    ends = numpy.empty(arity, dtype=numpy.int64)
    positions = numpy.empty(arity, dtype=numpy.int64)
    out_mv = out
    starts_mv = starts
    ends_mv = ends
    positions_mv = positions
    for combo in range(num_combination):
        min_slot = 0
        min_count = -1
        for slot in range(arity):
            branch_id = id_combinations[combo, slot]
            if branch_id < 0 or branch_id >= num_branch:
                raise IndexError('branch ID is out of range for sparse projection.')
            starts_mv[slot] = indptr[branch_id]
            ends_mv[slot] = indptr[branch_id + 1]
            positions_mv[slot] = starts_mv[slot]
            current_count = ends_mv[slot] - starts_mv[slot]
            if min_count < 0 or current_count < min_count:
                min_count = current_count
                min_slot = slot
        total = 0.0
        for candidate in range(starts_mv[min_slot], ends_mv[min_slot]):
            col = indices[candidate]
            product_value = data[candidate]
            matched = True
            for slot in range(arity):
                if slot == min_slot:
                    continue
                pos = positions_mv[slot]
                while pos < ends_mv[slot] and indices[pos] < col:
                    pos += 1
                positions_mv[slot] = pos
                if pos >= ends_mv[slot] or indices[pos] != col:
                    matched = False
                    break
                product_value *= data[pos]
            if matched:
                total += product_value
        out_mv[combo] = total
    return out


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calc_sparse_projection_product_by_site_double(
    numpy.ndarray[numpy.int64_t, ndim=2] id_combinations,
    numpy.ndarray[index_t, ndim=1] indptr,
    numpy.ndarray[index_t, ndim=1] indices,
    numpy.ndarray[numpy.float64_t, ndim=1] data,
    long num_site,
):
    cdef Py_ssize_t num_combination = id_combinations.shape[0]
    cdef Py_ssize_t arity = id_combinations.shape[1]
    cdef Py_ssize_t num_branch = indptr.shape[0] - 1
    cdef numpy.ndarray[numpy.float64_t, ndim=2] out
    cdef numpy.ndarray[numpy.int64_t, ndim=1] starts
    cdef numpy.ndarray[numpy.int64_t, ndim=1] ends
    cdef numpy.ndarray[numpy.int64_t, ndim=1] positions
    cdef double[:, :] out_mv
    cdef long[:] starts_mv
    cdef long[:] ends_mv
    cdef long[:] positions_mv
    cdef Py_ssize_t combo, slot, candidate, pos
    cdef Py_ssize_t branch_id, min_slot, min_count
    cdef Py_ssize_t col, site, current_count
    cdef double product_value
    cdef bint matched
    if arity < 1:
        raise ValueError('id_combinations should contain at least one branch column.')
    if num_site < 1:
        raise ValueError('num_site should be >= 1.')
    if indices.shape[0] != data.shape[0]:
        raise ValueError('indices and data should have identical length.')
    out = numpy.zeros((num_combination, num_site), dtype=numpy.float64)
    starts = numpy.empty(arity, dtype=numpy.int64)
    ends = numpy.empty(arity, dtype=numpy.int64)
    positions = numpy.empty(arity, dtype=numpy.int64)
    out_mv = out
    starts_mv = starts
    ends_mv = ends
    positions_mv = positions
    for combo in range(num_combination):
        min_slot = 0
        min_count = -1
        for slot in range(arity):
            branch_id = id_combinations[combo, slot]
            if branch_id < 0 or branch_id >= num_branch:
                raise IndexError('branch ID is out of range for sparse projection.')
            starts_mv[slot] = indptr[branch_id]
            ends_mv[slot] = indptr[branch_id + 1]
            positions_mv[slot] = starts_mv[slot]
            current_count = ends_mv[slot] - starts_mv[slot]
            if min_count < 0 or current_count < min_count:
                min_count = current_count
                min_slot = slot
        for candidate in range(starts_mv[min_slot], ends_mv[min_slot]):
            col = indices[candidate]
            product_value = data[candidate]
            matched = True
            for slot in range(arity):
                if slot == min_slot:
                    continue
                pos = positions_mv[slot]
                while pos < ends_mv[slot] and indices[pos] < col:
                    pos += 1
                positions_mv[slot] = pos
                if pos >= ends_mv[slot] or indices[pos] != col:
                    matched = False
                    break
                product_value *= data[pos]
            if matched:
                site = col % num_site
                out_mv[combo, site] += product_value
    return out


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
cpdef accumulate_sparse_summary_block_csr_double(
    numpy.ndarray[index_t, ndim=1] indptr,
    numpy.ndarray[index_t, ndim=1] indices,
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
    cdef Py_ssize_t n_branch = indptr.shape[0] - 1
    cdef Py_ssize_t row, k
    cdef Py_ssize_t start, end
    cdef Py_ssize_t col
    cdef double v
    cdef double[:] total_flat
    cdef double[:] from_flat
    cdef double[:] to_flat
    cdef double[:] pair_flat
    cdef Py_ssize_t idx
    cdef long pair_index, num_state_pair
    if indices.shape[0] != vals.shape[0]:
        raise ValueError('indices and vals should have identical length.')
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
    for row in range(n_branch):
        start = indptr[row]
        end = indptr[row + 1]
        for k in range(start, end):
            col = indices[k]
            v = vals[k]
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
cpdef update_sitewise_max_from_csr_row_double(
    numpy.ndarray[index_t, ndim=1] indptr,
    numpy.ndarray[index_t, ndim=1] indices,
    numpy.ndarray[numpy.float64_t, ndim=1] data,
    long branch_id,
    long a,
    long d,
    numpy.ndarray[numpy.float64_t, ndim=1] max_prob,
    numpy.ndarray[numpy.int64_t, ndim=1] max_a,
    numpy.ndarray[numpy.int64_t, ndim=1] max_d,
    numpy.ndarray[numpy.uint8_t, ndim=1] seen,
):
    cdef Py_ssize_t n_branch = indptr.shape[0] - 1
    cdef Py_ssize_t start, end, k
    cdef Py_ssize_t site
    cdef double v
    if branch_id < 0 or branch_id >= n_branch:
        raise ValueError('branch_id is out of range.')
    if indices.shape[0] != data.shape[0]:
        raise ValueError('indices and data should have identical length.')
    if max_prob.shape[0] != max_a.shape[0] or max_prob.shape[0] != max_d.shape[0] or max_prob.shape[0] != seen.shape[0]:
        raise ValueError('max_prob/max_a/max_d/seen should have identical length.')
    start = indptr[branch_id]
    end = indptr[branch_id + 1]
    for k in range(start, end):
        site = indices[k]
        v = data[k]
        if not isfinite(v):
            continue
        if (seen[site] == 0) or (v > max_prob[site]):
            max_prob[site] = v
            max_a[site] = a
            max_d[site] = d
        seen[site] = 1


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef scan_packed_sitewise_max_row_double(
    numpy.ndarray[index_t, ndim=1] packed_indices,
    numpy.ndarray[numpy.float64_t, ndim=1] data,
    long num_site,
    long num_state_from,
    long num_state_to,
    double min_sitewise_pp,
):
    cdef Py_ssize_t nnz = packed_indices.shape[0]
    cdef numpy.ndarray[numpy.float64_t, ndim=1] max_prob
    cdef numpy.ndarray[numpy.int64_t, ndim=1] max_a
    cdef numpy.ndarray[numpy.int64_t, ndim=1] max_d
    cdef numpy.ndarray[numpy.uint8_t, ndim=1] seen
    cdef numpy.ndarray[numpy.int64_t, ndim=1] site_idx
    cdef numpy.ndarray[numpy.int64_t, ndim=1] anc_idx
    cdef numpy.ndarray[numpy.int64_t, ndim=1] der_idx
    cdef double[:] prob_mv
    cdef long[:] max_a_mv
    cdef long[:] max_d_mv
    cdef unsigned char[:] seen_mv
    cdef long[:] site_mv
    cdef long[:] anc_mv
    cdef long[:] der_mv
    cdef Py_ssize_t k, out_count
    cdef long packed_col, event_id, site, a, d
    cdef double value
    if data.shape[0] != nnz:
        raise ValueError('packed_indices and data should have identical length.')
    if num_site < 1 or num_state_from < 1 or num_state_to < 1:
        raise ValueError('Packed tensor dimensions should be positive.')
    max_prob = numpy.zeros(num_site, dtype=numpy.float64)
    max_a = numpy.full(num_site, -1, dtype=numpy.int64)
    max_d = numpy.full(num_site, -1, dtype=numpy.int64)
    seen = numpy.zeros(num_site, dtype=numpy.uint8)
    prob_mv = max_prob
    max_a_mv = max_a
    max_d_mv = max_d
    seen_mv = seen
    for k in range(nnz):
        packed_col = packed_indices[k]
        if packed_col < 0:
            raise IndexError('Packed substitution column should be nonnegative.')
        site = packed_col % num_site
        event_id = packed_col // num_site
        d = event_id % num_state_to
        a = (event_id // num_state_to) % num_state_from
        value = data[k]
        if not isfinite(value):
            continue
        if (seen_mv[site] == 0) or (value > prob_mv[site]):
            prob_mv[site] = value
            max_a_mv[site] = a
            max_d_mv[site] = d
        seen_mv[site] = 1
    site_idx = numpy.empty(num_site, dtype=numpy.int64)
    anc_idx = numpy.empty(num_site, dtype=numpy.int64)
    der_idx = numpy.empty(num_site, dtype=numpy.int64)
    site_mv = site_idx
    anc_mv = anc_idx
    der_mv = der_idx
    out_count = 0
    for site in range(num_site):
        if (seen_mv[site] != 0) and (prob_mv[site] >= min_sitewise_pp):
            site_mv[out_count] = site
            anc_mv[out_count] = max_a_mv[site]
            der_mv[out_count] = max_d_mv[site]
            out_count += 1
    return (
        numpy.asarray(site_idx[:out_count]),
        numpy.asarray(anc_idx[:out_count]),
        numpy.asarray(der_idx[:out_count]),
    )


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

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
    cdef Py_ssize_t[:] xind = numpy.zeros(xmax, dtype=numpy.int64)
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


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef generate_all_triples_from_sorted_nodes_int64(numpy.ndarray[numpy.int64_t, ndim=1] unique_nodes):
    cdef Py_ssize_t n = unique_nodes.shape[0]
    cdef unsigned long long total_ull
    cdef Py_ssize_t total
    cdef numpy.ndarray[numpy.int64_t, ndim=2] out
    cdef numpy.int64_t[:, :] out_view
    cdef numpy.int64_t[:] nodes_view = unique_nodes
    cdef Py_ssize_t i, j, k, pos
    if n < 3:
        return numpy.zeros(shape=(0, 3), dtype=numpy.int64)
    total_ull = (<unsigned long long> n) * (<unsigned long long> (n - 1)) * (<unsigned long long> (n - 2)) // 6
    total = <Py_ssize_t> total_ull
    out = numpy.empty(shape=(total, 3), dtype=numpy.int64)
    out_view = out
    pos = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                out_view[pos, 0] = nodes_view[i]
                out_view[pos, 1] = nodes_view[j]
                out_view[pos, 2] = nodes_view[k]
                pos += 1
    return out


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef generate_all_k_combinations_from_sorted_nodes_int64(
    numpy.ndarray[numpy.int64_t, ndim=1] unique_nodes,
    int k,
):
    cdef Py_ssize_t n = unique_nodes.shape[0]
    cdef Py_ssize_t kk = <Py_ssize_t> k
    cdef Py_ssize_t i, j, pos
    cdef unsigned long long total_ull = 1
    cdef Py_ssize_t total
    cdef numpy.ndarray[numpy.int64_t, ndim=2] out
    cdef numpy.int64_t[:, :] out_view
    cdef numpy.int64_t[:] nodes_view = unique_nodes
    cdef numpy.ndarray[numpy.int64_t, ndim=1] idx
    cdef numpy.int64_t[:] idx_view
    cdef bint found

    if kk <= 0:
        return numpy.zeros(shape=(0, 0), dtype=numpy.int64)
    if n < kk:
        return numpy.zeros(shape=(0, kk), dtype=numpy.int64)
    if kk == 1:
        out = numpy.empty(shape=(n, 1), dtype=numpy.int64)
        out_view = out
        for i in range(n):
            out_view[i, 0] = nodes_view[i]
        return out
    if kk == 2:
        total_ull = (<unsigned long long> n) * (<unsigned long long> (n - 1)) // 2
        total = <Py_ssize_t> total_ull
        out = numpy.empty(shape=(total, 2), dtype=numpy.int64)
        out_view = out
        pos = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                out_view[pos, 0] = nodes_view[i]
                out_view[pos, 1] = nodes_view[j]
                pos += 1
        return out
    if kk == 3:
        return generate_all_triples_from_sorted_nodes_int64(unique_nodes)

    total_ull = 1
    for i in range(1, kk + 1):
        total_ull = (total_ull * <unsigned long long> (n - kk + i)) // <unsigned long long> i
    total = <Py_ssize_t> total_ull
    if total <= 0:
        return numpy.zeros(shape=(0, kk), dtype=numpy.int64)

    out = numpy.empty(shape=(total, kk), dtype=numpy.int64)
    out_view = out
    idx = numpy.empty(shape=(kk,), dtype=numpy.int64)
    idx_view = idx
    for i in range(kk):
        idx_view[i] = i

    pos = 0
    while True:
        for j in range(kk):
            out_view[pos, j] = nodes_view[idx_view[j]]
        pos += 1
        found = False
        for i in range(kk - 1, -1, -1):
            if idx_view[i] < (i + n - kk):
                idx_view[i] += 1
                for j in range(i + 1, kk):
                    idx_view[j] = idx_view[j - 1] + 1
                found = True
                break
        if not found:
            break
    return out


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef generate_union_encoded_arity3_dense_int64(
    numpy.ndarray[numpy.int64_t, ndim=2] remapped_pairs,
    Py_ssize_t num_nodes,
):
    cdef Py_ssize_t num_edges = remapped_pairs.shape[0]
    cdef numpy.int64_t[:, :] pairs_view = remapped_pairs
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] adj
    cdef numpy.uint8_t[:, :] adj_view
    cdef numpy.ndarray[numpy.int64_t, ndim=1] degree
    cdef numpy.int64_t[:] degree_view
    cdef numpy.ndarray[numpy.int64_t, ndim=1] offsets
    cdef numpy.int64_t[:] offsets_view
    cdef numpy.ndarray[numpy.int64_t, ndim=1] neighbors
    cdef numpy.int64_t[:] neighbors_view
    cdef numpy.ndarray[numpy.int64_t, ndim=1] cursor
    cdef numpy.int64_t[:] cursor_view
    cdef numpy.ndarray[numpy.int64_t, ndim=1] encoded
    cdef numpy.int64_t[:] encoded_view
    cdef Py_ssize_t i, a, b
    cdef Py_ssize_t pos, start, end, center, ii, jj
    cdef Py_ssize_t u, v, uv_min, uv_max, low, high, mid
    cdef Py_ssize_t neighbor_len, max_candidates
    cdef Py_ssize_t write_pos = 0
    cdef unsigned long long deg_ull
    cdef unsigned long long max_candidates_ull = 0
    if (num_nodes < 3) or (num_edges < 2):
        return numpy.zeros(shape=(0,), dtype=numpy.int64)

    adj = numpy.zeros(shape=(num_nodes, num_nodes), dtype=numpy.uint8)
    adj_view = adj
    degree = numpy.zeros(shape=(num_nodes,), dtype=numpy.int64)
    degree_view = degree
    for i in range(num_edges):
        a = pairs_view[i, 0]
        b = pairs_view[i, 1]
        adj_view[a, b] = 1
        adj_view[b, a] = 1
        degree_view[a] += 1
        degree_view[b] += 1

    offsets = numpy.empty(shape=(num_nodes + 1,), dtype=numpy.int64)
    offsets_view = offsets
    offsets_view[0] = 0
    for i in range(num_nodes):
        offsets_view[i + 1] = offsets_view[i] + degree_view[i]
        deg_ull = <unsigned long long> degree_view[i]
        if deg_ull >= 2:
            max_candidates_ull += (deg_ull * (deg_ull - 1)) // 2

    neighbor_len = offsets_view[num_nodes]
    neighbors = numpy.empty(shape=(neighbor_len,), dtype=numpy.int64)
    neighbors_view = neighbors
    cursor = numpy.empty(shape=(num_nodes,), dtype=numpy.int64)
    cursor_view = cursor
    for i in range(num_nodes):
        cursor_view[i] = offsets_view[i]
    for i in range(num_edges):
        a = pairs_view[i, 0]
        b = pairs_view[i, 1]
        pos = cursor_view[a]
        neighbors_view[pos] = b
        cursor_view[a] = pos + 1
        pos = cursor_view[b]
        neighbors_view[pos] = a
        cursor_view[b] = pos + 1

    max_candidates = <Py_ssize_t> max_candidates_ull
    if max_candidates <= 0:
        return numpy.zeros(shape=(0,), dtype=numpy.int64)
    encoded = numpy.empty(shape=(max_candidates,), dtype=numpy.int64)
    encoded_view = encoded
    for center in range(num_nodes):
        start = offsets_view[center]
        end = offsets_view[center + 1]
        if (end - start) < 2:
            continue
        for ii in range(start, end - 1):
            u = neighbors_view[ii]
            for jj in range(ii + 1, end):
                v = neighbors_view[jj]
                if u < v:
                    uv_min = u
                    uv_max = v
                else:
                    uv_min = v
                    uv_max = u
                if (adj_view[uv_min, uv_max] != 0) and (center > uv_min):
                    continue
                if center < uv_min:
                    low = center
                else:
                    low = uv_min
                if center > uv_max:
                    high = center
                else:
                    high = uv_max
                mid = center + u + v - low - high
                encoded_view[write_pos] = ((low * num_nodes) + mid) * num_nodes + high
                write_pos += 1
    return encoded[:write_pos]


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef generate_union_candidates_arity4_from_triples_int64(
    numpy.ndarray[numpy.int64_t, ndim=2] sorted_triples,
):
    cdef Py_ssize_t num_rows = sorted_triples.shape[0]
    cdef numpy.int64_t[:, :] triples_view = sorted_triples
    cdef Py_ssize_t num_entries = num_rows * 3
    cdef numpy.ndarray[numpy.int64_t, ndim=1] key_a
    cdef numpy.ndarray[numpy.int64_t, ndim=1] key_b
    cdef numpy.ndarray[numpy.int64_t, ndim=1] vals
    cdef numpy.int64_t[:] key_a_view
    cdef numpy.int64_t[:] key_b_view
    cdef numpy.int64_t[:] vals_view
    cdef numpy.ndarray[numpy.int64_t, ndim=1] order
    cdef numpy.int64_t[:] order_view
    cdef Py_ssize_t i, j, t, idx
    cdef Py_ssize_t start, end, run_len
    cdef Py_ssize_t write_pos = 0
    cdef unsigned long long total_ull = 0
    cdef Py_ssize_t total_out
    cdef numpy.ndarray[numpy.int64_t, ndim=2] out
    cdef numpy.int64_t[:, :] out_view
    cdef numpy.int64_t cur_a, cur_b
    cdef numpy.int64_t a, b, c, d, tmp
    cdef numpy.int64_t val_i
    if num_rows < 2:
        return numpy.zeros(shape=(0, 4), dtype=numpy.int64)

    key_a = numpy.empty(shape=(num_entries,), dtype=numpy.int64)
    key_b = numpy.empty(shape=(num_entries,), dtype=numpy.int64)
    vals = numpy.empty(shape=(num_entries,), dtype=numpy.int64)
    key_a_view = key_a
    key_b_view = key_b
    vals_view = vals

    for i in range(num_rows):
        a = triples_view[i, 0]
        b = triples_view[i, 1]
        c = triples_view[i, 2]
        idx = i * 3
        key_a_view[idx] = b
        key_b_view[idx] = c
        vals_view[idx] = a
        idx += 1
        key_a_view[idx] = a
        key_b_view[idx] = c
        vals_view[idx] = b
        idx += 1
        key_a_view[idx] = a
        key_b_view[idx] = b
        vals_view[idx] = c

    order = numpy.lexsort((key_b, key_a))
    order_view = order
    if order_view.shape[0] < 2:
        return numpy.zeros(shape=(0, 4), dtype=numpy.int64)

    cur_a = key_a_view[order_view[0]]
    cur_b = key_b_view[order_view[0]]
    run_len = 1
    for t in range(1, order_view.shape[0]):
        idx = order_view[t]
        if (key_a_view[idx] == cur_a) and (key_b_view[idx] == cur_b):
            run_len += 1
            continue
        if run_len >= 2:
            total_ull += (<unsigned long long> run_len) * (<unsigned long long> (run_len - 1)) // 2
        cur_a = key_a_view[idx]
        cur_b = key_b_view[idx]
        run_len = 1
    if run_len >= 2:
        total_ull += (<unsigned long long> run_len) * (<unsigned long long> (run_len - 1)) // 2
    total_out = <Py_ssize_t> total_ull
    if total_out <= 0:
        return numpy.zeros(shape=(0, 4), dtype=numpy.int64)

    out = numpy.empty(shape=(total_out, 4), dtype=numpy.int64)
    out_view = out
    start = 0
    while start < order_view.shape[0]:
        idx = order_view[start]
        cur_a = key_a_view[idx]
        cur_b = key_b_view[idx]
        end = start + 1
        while end < order_view.shape[0]:
            idx = order_view[end]
            if (key_a_view[idx] != cur_a) or (key_b_view[idx] != cur_b):
                break
            end += 1
        run_len = end - start
        if run_len >= 2:
            for i in range(start, end - 1):
                val_i = vals_view[order_view[i]]
                for j in range(i + 1, end):
                    c = val_i
                    d = vals_view[order_view[j]]
                    a = cur_a
                    b = cur_b
                    # Sorting network for 4 values
                    if a > b:
                        tmp = a
                        a = b
                        b = tmp
                    if c > d:
                        tmp = c
                        c = d
                        d = tmp
                    if a > c:
                        tmp = a
                        a = c
                        c = tmp
                    if b > d:
                        tmp = b
                        b = d
                        d = tmp
                    if b > c:
                        tmp = b
                        b = c
                        c = tmp
                    out_view[write_pos, 0] = a
                    out_view[write_pos, 1] = b
                    out_view[write_pos, 2] = c
                    out_view[write_pos, 3] = d
                    write_pos += 1
        start = end
    return out[:write_pos, :]


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef generate_union_candidates_shared_subset_int64(
    numpy.ndarray[numpy.int64_t, ndim=2] sorted_nodes,
):
    cdef Py_ssize_t num_rows = sorted_nodes.shape[0]
    cdef Py_ssize_t width = sorted_nodes.shape[1]
    cdef Py_ssize_t key_len = width - 1
    cdef Py_ssize_t out_width = width + 1
    cdef numpy.int64_t[:, :] nodes_view = sorted_nodes
    cdef Py_ssize_t num_entries
    cdef numpy.ndarray[numpy.int64_t, ndim=2] keys
    cdef numpy.int64_t[:, :] keys_view
    cdef numpy.ndarray[numpy.int64_t, ndim=1] vals
    cdef numpy.int64_t[:] vals_view
    cdef numpy.ndarray[numpy.int64_t, ndim=1] order
    cdef numpy.int64_t[:] order_view
    cdef numpy.ndarray[numpy.int64_t, ndim=2] out
    cdef numpy.int64_t[:, :] out_view
    cdef Py_ssize_t i, j, d, col
    cdef Py_ssize_t idx, idx2
    cdef Py_ssize_t start, end, run_len
    cdef Py_ssize_t p, r
    cdef Py_ssize_t write_pos = 0
    cdef unsigned long long total_ull = 0
    cdef Py_ssize_t total_out
    cdef numpy.int64_t a, b, lo, hi
    cdef bint differs

    if num_rows < 2:
        return numpy.zeros(shape=(0, out_width), dtype=numpy.int64)
    if width < 2:
        return numpy.zeros(shape=(0, out_width), dtype=numpy.int64)

    num_entries = num_rows * width
    keys = numpy.empty(shape=(num_entries, key_len), dtype=numpy.int64)
    vals = numpy.empty(shape=(num_entries,), dtype=numpy.int64)
    keys_view = keys
    vals_view = vals

    idx = 0
    for i in range(num_rows):
        for d in range(width):
            vals_view[idx] = nodes_view[i, d]
            for col in range(d):
                keys_view[idx, col] = nodes_view[i, col]
            for col in range(d, key_len):
                keys_view[idx, col] = nodes_view[i, col + 1]
            idx += 1

    order = numpy.lexsort(keys.T[::-1])
    order_view = order
    if order_view.shape[0] < 2:
        return numpy.zeros(shape=(0, out_width), dtype=numpy.int64)

    start = 0
    while start < order_view.shape[0]:
        idx = order_view[start]
        end = start + 1
        while end < order_view.shape[0]:
            idx2 = order_view[end]
            differs = False
            for col in range(key_len):
                if keys_view[idx2, col] != keys_view[idx, col]:
                    differs = True
                    break
            if differs:
                break
            end += 1
        run_len = end - start
        if run_len >= 2:
            total_ull += (<unsigned long long> run_len) * (<unsigned long long> (run_len - 1)) // 2
        start = end

    total_out = <Py_ssize_t> total_ull
    if total_out <= 0:
        return numpy.zeros(shape=(0, out_width), dtype=numpy.int64)

    out = numpy.empty(shape=(total_out, out_width), dtype=numpy.int64)
    out_view = out

    start = 0
    while start < order_view.shape[0]:
        idx = order_view[start]
        end = start + 1
        while end < order_view.shape[0]:
            idx2 = order_view[end]
            differs = False
            for col in range(key_len):
                if keys_view[idx2, col] != keys_view[idx, col]:
                    differs = True
                    break
            if differs:
                break
            end += 1
        run_len = end - start
        if run_len >= 2:
            for i in range(start, end - 1):
                a = vals_view[order_view[i]]
                for j in range(i + 1, end):
                    b = vals_view[order_view[j]]
                    if a < b:
                        lo = a
                        hi = b
                    else:
                        lo = b
                        hi = a
                    if lo == hi:
                        continue
                    p = 0
                    r = 0
                    while (p < key_len) and (keys_view[idx, p] < lo):
                        out_view[write_pos, r] = keys_view[idx, p]
                        p += 1
                        r += 1
                    out_view[write_pos, r] = lo
                    r += 1
                    while (p < key_len) and (keys_view[idx, p] < hi):
                        out_view[write_pos, r] = keys_view[idx, p]
                        p += 1
                        r += 1
                    out_view[write_pos, r] = hi
                    r += 1
                    while p < key_len:
                        out_view[write_pos, r] = keys_view[idx, p]
                        p += 1
                        r += 1
                    write_pos += 1
        start = end
    return out[:write_pos, :]

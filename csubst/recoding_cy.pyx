import numpy
cimport numpy
cimport cython
from libc.math cimport fabs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef numpy.ndarray[numpy.int64_t, ndim=2] random_bin_assignments_int64(
    int num_item,
    int num_bin,
    object rng,
    int n_random,
):
    cdef:
        numpy.ndarray[numpy.int64_t, ndim=2] out
        numpy.ndarray[numpy.int64_t, ndim=1] relabel
        long[:, :] out_mv
        long[:] relabel_mv
        Py_ssize_t row, i, old, nb
        long l1, l2, last_label
        long next_label

    if num_item < 1:
        raise ValueError("num_item should be >= 1.")
    if n_random < 1:
        raise ValueError("n_random should be >= 1.")
    if num_bin < 2 or num_bin > num_item:
        raise ValueError("Invalid number of bins for random bin assignment.")
    if not hasattr(rng, "integers"):
        raise ValueError("rng should provide integers().")

    out = numpy.empty((n_random, num_item), dtype=numpy.int64)
    relabel = numpy.empty((num_item,), dtype=numpy.int64)
    out_mv = out
    relabel_mv = relabel

    for row in range(n_random):
        for i in range(num_item):
            out_mv[row, i] = i
        nb = num_item
        while nb > num_bin:
            l1 = int(rng.integers(low=0, high=nb))
            l2 = int(rng.integers(low=0, high=(nb - 1)))
            if l2 >= l1:
                l2 += 1
            if l2 < l1:
                l1, l2 = l2, l1
            last_label = nb - 1
            for i in range(num_item):
                if out_mv[row, i] == l2:
                    out_mv[row, i] = l1
                elif (l2 != last_label) and (out_mv[row, i] == last_label):
                    out_mv[row, i] = l2
            nb -= 1

        for i in range(num_item):
            relabel_mv[i] = -1
        next_label = 0
        for old in range(num_item):
            for i in range(num_item):
                if out_mv[row, i] == old:
                    relabel_mv[old] = next_label
                    next_label += 1
                    break
        for i in range(num_item):
            out_mv[row, i] = relabel_mv[out_mv[row, i]]

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef tuple search_initial_bins_chunk_chisq_double(
    numpy.ndarray[numpy.int64_t, ndim=2] initial_bins_chunk,
    int num_bin,
    numpy.ndarray[numpy.float64_t, ndim=2] fmat,
    numpy.ndarray[numpy.float64_t, ndim=1] fr,
    numpy.ndarray[numpy.float64_t, ndim=1] nsitev,
    double obj_eps=1e-8,
):
    cdef:
        Py_ssize_t n_start = initial_bins_chunk.shape[0]
        Py_ssize_t i
        numpy.ndarray[numpy.int64_t, ndim=1] bins
        numpy.ndarray[numpy.int64_t, ndim=1] best_bins = None
        double crit
        double best_crit = float("inf")
        Py_ssize_t best_offset = -1

    if n_start < 1:
        raise ValueError("initial_bins_chunk should include at least one start.")

    for i in range(n_start):
        bins, crit = hill_climb_bins_chisq_double(
            initial_bins=initial_bins_chunk[i, :],
            num_bin=num_bin,
            fmat=fmat,
            fr=fr,
            nsitev=nsitev,
            obj_eps=obj_eps,
        )
        if (crit < (best_crit - obj_eps)) or ((fabs(crit - best_crit) <= obj_eps) and ((best_offset < 0) or (i < best_offset))):
            best_bins = bins.copy()
            best_crit = crit
            best_offset = i
            if best_crit <= 0.0:
                break

    if best_bins is None:
        raise ValueError("Failed to optimize auto recoding bins.")
    return best_bins, float(best_crit), int(best_offset)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef tuple search_initial_bins_chunk_conductance_double(
    numpy.ndarray[numpy.int64_t, ndim=2] initial_bins_chunk,
    int num_bin,
    numpy.ndarray[numpy.float64_t, ndim=1] pi,
    numpy.ndarray[numpy.float64_t, ndim=2] weighted_q,
    double obj_eps=1e-8,
):
    cdef:
        Py_ssize_t n_start = initial_bins_chunk.shape[0]
        Py_ssize_t i
        numpy.ndarray[numpy.int64_t, ndim=1] bins
        numpy.ndarray[numpy.int64_t, ndim=1] best_bins = None
        double crit
        double best_crit = float("inf")
        Py_ssize_t best_offset = -1

    if n_start < 1:
        raise ValueError("initial_bins_chunk should include at least one start.")

    for i in range(n_start):
        bins, crit = hill_climb_bins_conductance_double(
            initial_bins=initial_bins_chunk[i, :],
            num_bin=num_bin,
            pi=pi,
            weighted_q=weighted_q,
            obj_eps=obj_eps,
        )
        if (crit < (best_crit - obj_eps)) or ((fabs(crit - best_crit) <= obj_eps) and ((best_offset < 0) or (i < best_offset))):
            best_bins = bins.copy()
            best_crit = crit
            best_offset = i
            if best_crit <= 0.0:
                break

    if best_bins is None:
        raise ValueError("Failed to optimize auto recoding bins.")
    return best_bins, float(best_crit), int(best_offset)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef tuple hill_climb_bins_chisq_double(
    numpy.ndarray[numpy.int64_t, ndim=1] initial_bins,
    int num_bin,
    numpy.ndarray[numpy.float64_t, ndim=2] fmat,
    numpy.ndarray[numpy.float64_t, ndim=1] fr,
    numpy.ndarray[numpy.float64_t, ndim=1] nsitev,
    double obj_eps=1e-8,
):
    cdef:
        Py_ssize_t num_state = initial_bins.shape[0]
        Py_ssize_t num_taxa = fmat.shape[0]
        numpy.ndarray[numpy.int64_t, ndim=1] bins = initial_bins.copy()
        numpy.ndarray[numpy.int64_t, ndim=1] counts = numpy.zeros((num_bin,), dtype=numpy.int64)
        numpy.ndarray[numpy.float64_t, ndim=1] frb = numpy.zeros((num_bin,), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim=2] frt = numpy.zeros((num_taxa, num_bin), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim=2] term = numpy.zeros((num_taxa, num_bin), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim=1] taxon_sum = numpy.zeros((num_taxa,), dtype=numpy.float64)
        long[:] bins_mv
        long[:] counts_mv
        double[:] frb_mv
        double[:, :] frt_mv
        double[:, :] term_mv
        double[:] taxon_sum_mv
        double[:, :] fmat_mv
        double[:] fr_mv
        double[:] nsitev_mv
        Py_ssize_t i, b, t, el, src, dst, argmax_idx
        double fr_el
        double frb_src_new, frb_dst_new
        double frt_src_new, frt_dst_new
        double diff_src, diff_dst
        double new_src, new_dst
        double old_src, old_dst
        double weighted
        double crit = -1.0
        double crit_new
        double taxon_argmax_new
        double src_argmax_new, dst_argmax_new
        bint improved, reject

    if num_bin < 2:
        raise ValueError("num_bin should be >= 2.")
    if fmat.shape[1] != num_state:
        raise ValueError("fmat column size should match initial_bins length.")
    if fr.shape[0] != num_state:
        raise ValueError("fr length should match initial_bins length.")
    if nsitev.shape[0] != num_taxa:
        raise ValueError("nsitev length should match fmat row size.")

    bins_mv = bins
    counts_mv = counts
    frb_mv = frb
    frt_mv = frt
    term_mv = term
    taxon_sum_mv = taxon_sum
    fmat_mv = fmat
    fr_mv = fr
    nsitev_mv = nsitev

    # Initialize counts and grouped frequencies.
    for i in range(num_state):
        b = bins_mv[i]
        counts_mv[b] += 1
        frb_mv[b] += fr_mv[i]
        for t in range(num_taxa):
            frt_mv[t, b] += fmat_mv[t, i]

    # Initialize term, taxon-wise sums, and current objective.
    argmax_idx = 0
    for t in range(num_taxa):
        taxon_sum_mv[t] = 0.0
        for b in range(num_bin):
            diff_src = frt_mv[t, b] - frb_mv[b]
            term_mv[t, b] = (diff_src * diff_src) / frb_mv[b]
            taxon_sum_mv[t] += term_mv[t, b]
        weighted = taxon_sum_mv[t] * nsitev_mv[t]
        if weighted > crit:
            crit = weighted
            argmax_idx = t

    with nogil:
        while True:
            improved = False
            for el in range(num_state):
                src = bins_mv[el]
                if counts_mv[src] <= 1:
                    continue
                fr_el = fr_mv[el]
                for dst in range(num_bin):
                    if dst == src:
                        continue
                    frb_src_new = frb_mv[src] - fr_el
                    frb_dst_new = frb_mv[dst] + fr_el
                    if (frb_src_new <= 0.0) or (frb_dst_new <= 0.0):
                        continue

                    # Lower bound at current argmax taxon. If this does not improve,
                    # global max cannot improve.
                    frt_src_new = frt_mv[argmax_idx, src] - fmat_mv[argmax_idx, el]
                    frt_dst_new = frt_mv[argmax_idx, dst] + fmat_mv[argmax_idx, el]
                    diff_src = frt_src_new - frb_src_new
                    diff_dst = frt_dst_new - frb_dst_new
                    src_argmax_new = (diff_src * diff_src) / frb_src_new
                    dst_argmax_new = (diff_dst * diff_dst) / frb_dst_new
                    taxon_argmax_new = (
                        taxon_sum_mv[argmax_idx]
                        - term_mv[argmax_idx, src]
                        - term_mv[argmax_idx, dst]
                        + src_argmax_new
                        + dst_argmax_new
                    )
                    if (taxon_argmax_new * nsitev_mv[argmax_idx]) >= (crit - obj_eps):
                        continue

                    # Full objective scan with early reject.
                    crit_new = -1.0
                    reject = False
                    for t in range(num_taxa):
                        frt_src_new = frt_mv[t, src] - fmat_mv[t, el]
                        frt_dst_new = frt_mv[t, dst] + fmat_mv[t, el]
                        diff_src = frt_src_new - frb_src_new
                        diff_dst = frt_dst_new - frb_dst_new
                        new_src = (diff_src * diff_src) / frb_src_new
                        new_dst = (diff_dst * diff_dst) / frb_dst_new
                        weighted = (taxon_sum_mv[t] - term_mv[t, src] - term_mv[t, dst] + new_src + new_dst) * nsitev_mv[t]
                        if weighted > crit_new:
                            crit_new = weighted
                        if crit_new >= (crit - obj_eps):
                            reject = True
                            break
                    if reject:
                        continue

                    # Accept move and update state.
                    bins_mv[el] = dst
                    counts_mv[src] -= 1
                    counts_mv[dst] += 1
                    frb_mv[src] = frb_src_new
                    frb_mv[dst] = frb_dst_new

                    crit = -1.0
                    argmax_idx = 0
                    for t in range(num_taxa):
                        old_src = term_mv[t, src]
                        old_dst = term_mv[t, dst]
                        frt_src_new = frt_mv[t, src] - fmat_mv[t, el]
                        frt_dst_new = frt_mv[t, dst] + fmat_mv[t, el]
                        frt_mv[t, src] = frt_src_new
                        frt_mv[t, dst] = frt_dst_new

                        diff_src = frt_src_new - frb_src_new
                        diff_dst = frt_dst_new - frb_dst_new
                        new_src = (diff_src * diff_src) / frb_src_new
                        new_dst = (diff_dst * diff_dst) / frb_dst_new
                        term_mv[t, src] = new_src
                        term_mv[t, dst] = new_dst
                        taxon_sum_mv[t] = taxon_sum_mv[t] - old_src - old_dst + new_src + new_dst

                        weighted = taxon_sum_mv[t] * nsitev_mv[t]
                        if weighted > crit:
                            crit = weighted
                            argmax_idx = t

                    improved = True
                    break
                if improved:
                    break
            if not improved:
                break

    return bins, float(crit)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef tuple hill_climb_bins_conductance_double(
    numpy.ndarray[numpy.int64_t, ndim=1] initial_bins,
    int num_bin,
    numpy.ndarray[numpy.float64_t, ndim=1] pi,
    numpy.ndarray[numpy.float64_t, ndim=2] weighted_q,
    double obj_eps=1e-8,
):
    cdef:
        Py_ssize_t num_state = initial_bins.shape[0]
        numpy.ndarray[numpy.int64_t, ndim=1] bins = initial_bins.copy()
        numpy.ndarray[numpy.int64_t, ndim=1] counts = numpy.zeros((num_bin,), dtype=numpy.int64)
        numpy.ndarray[numpy.float64_t, ndim=1] cap = numpy.zeros((num_bin,), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim=1] out = numpy.zeros((num_bin,), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim=2] flow = numpy.zeros((num_bin, num_bin), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim=1] row_bin_sum = numpy.zeros((num_bin,), dtype=numpy.float64)
        numpy.ndarray[numpy.float64_t, ndim=1] col_bin_sum = numpy.zeros((num_bin,), dtype=numpy.float64)
        long[:] bins_mv
        long[:] counts_mv
        double[:] cap_mv
        double[:] out_mv
        double[:, :] flow_mv
        double[:] row_bin_sum_mv
        double[:] col_bin_sum_mv
        double[:] pi_mv
        double[:, :] weighted_q_mv
        Py_ssize_t i, j, b, el, src, dst
        double row_total, pi_el
        double cap_src_new, cap_dst_new
        double out_src_new, out_dst_new
        double old_src_term, old_dst_term
        double new_src_term, new_dst_term
        double crit = 0.0
        double crit_new
        bint improved

    if num_bin < 2:
        raise ValueError("num_bin should be >= 2.")
    if pi.shape[0] != num_state:
        raise ValueError("pi length should match initial_bins length.")
    if weighted_q.shape[0] != num_state or weighted_q.shape[1] != num_state:
        raise ValueError("weighted_q should be square and match initial_bins length.")

    bins_mv = bins
    counts_mv = counts
    cap_mv = cap
    out_mv = out
    flow_mv = flow
    row_bin_sum_mv = row_bin_sum
    col_bin_sum_mv = col_bin_sum
    pi_mv = pi
    weighted_q_mv = weighted_q

    for i in range(num_state):
        b = bins_mv[i]
        counts_mv[b] += 1
        cap_mv[b] += pi_mv[i]

    for i in range(num_state):
        for j in range(num_state):
            flow_mv[bins_mv[i], bins_mv[j]] += weighted_q_mv[i, j]

    for i in range(num_bin):
        out_mv[i] = 0.0
        for j in range(num_bin):
            out_mv[i] += flow_mv[i, j]
        out_mv[i] -= flow_mv[i, i]
        if cap_mv[i] <= 0.0:
            return bins, float("inf")
        crit += out_mv[i] / cap_mv[i]

    with nogil:
        while True:
            improved = False
            for el in range(num_state):
                src = bins_mv[el]
                if counts_mv[src] <= 1:
                    continue
                pi_el = pi_mv[el]

                for b in range(num_bin):
                    row_bin_sum_mv[b] = 0.0
                    col_bin_sum_mv[b] = 0.0
                for j in range(num_state):
                    b = bins_mv[j]
                    row_bin_sum_mv[b] += weighted_q_mv[el, j]
                    col_bin_sum_mv[b] += weighted_q_mv[j, el]
                row_total = 0.0
                for b in range(num_bin):
                    row_total += row_bin_sum_mv[b]

                old_src_term = out_mv[src] / cap_mv[src]
                for dst in range(num_bin):
                    if dst == src:
                        continue
                    cap_src_new = cap_mv[src] - pi_el
                    cap_dst_new = cap_mv[dst] + pi_el
                    if cap_src_new <= 0.0:
                        continue
                    out_src_new = out_mv[src] - (row_total - row_bin_sum_mv[src]) + col_bin_sum_mv[src]
                    out_dst_new = out_mv[dst] + (row_total - row_bin_sum_mv[dst]) - col_bin_sum_mv[dst]
                    old_dst_term = out_mv[dst] / cap_mv[dst]
                    new_src_term = out_src_new / cap_src_new
                    new_dst_term = out_dst_new / cap_dst_new
                    crit_new = crit - old_src_term - old_dst_term + new_src_term + new_dst_term
                    if crit_new < (crit - obj_eps):
                        bins_mv[el] = dst
                        counts_mv[src] -= 1
                        counts_mv[dst] += 1
                        cap_mv[src] = cap_src_new
                        cap_mv[dst] = cap_dst_new
                        out_mv[src] = out_src_new
                        out_mv[dst] = out_dst_new
                        crit = crit_new
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    return bins, float(crit)

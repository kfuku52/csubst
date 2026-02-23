from collections import OrderedDict

import numpy as np

from csubst import parallel
from csubst import sequence
try:
    from csubst import recoding_cy
except Exception:  # pragma: no cover - Cython extension is optional
    recoding_cy = None


_CANONICAL_AA = tuple("ACDEFGHIKLMNPQRSTVWY")
_CANONICAL_AA_SET = frozenset(_CANONICAL_AA)
_DEFAULT_AUTO_RANDOM_STARTS = 1000
_DEFAULT_AUTO_RANDOM_SEED = 42
_AA_ALIGNMENT_CACHE_KEY = "_nonsyn_recode_alignment_cache"
_AA_PSEUDOCOUNT = 1e-12
_OBJ_EPS = 1e-8

RECODING_SCHEMES = OrderedDict(
    [
        ("dayhoff6", ("AGPST", "DENQ", "HKR", "ILMV", "FWY", "C")),
        ("sr6", ("APST", "DEGN", "KQR", "ILMV", "CW", "FHY")),
        ("kgb6", ("AGPS", "DENQHKRT", "MIL", "W", "FY", "CV")),
        ("sr4", ("AGNPST", "CHWY", "DEKQR", "FILMV")),
        ("dayhoff9", ("DEHNQ", "ILMV", "FY", "AST", "KR", "G", "P", "C", "W")),
        ("dayhoff12", ("DEQ", "MLIV", "FY", "KHR", "G", "A", "P", "S", "T", "N", "W", "C")),
        ("dayhoff15", ("DEQ", "ML", "IV", "FY", "G", "A", "P", "S", "T", "N", "K", "H", "R", "W", "C")),
        ("dayhoff18", ("ML", "FY", "I", "V", "G", "A", "P", "S", "T", "D", "E", "Q", "N", "H", "K", "R", "W", "C")),
    ]
)

AUTO_RECODING_SCHEMES = OrderedDict(
    [
        ("srchisq6", {"family": "srchisq", "n_bins": 6}),
        ("kgbauto6", {"family": "kgbauto", "n_bins": 6}),
    ]
)

_RECODING_ALIASES = {
    "none": "none",
    "off": "none",
    "20": "none",
    "dayhoff6": "dayhoff6",
    "dayhoff-6": "dayhoff6",
    "dayhoff_6": "dayhoff6",
    "d6": "dayhoff6",
    "sr6": "sr6",
    "sr-6": "sr6",
    "sr_6": "sr6",
    "kgb6": "kgb6",
    "kgb-6": "kgb6",
    "kgb_6": "kgb6",
    "sr4": "sr4",
    "sr-4": "sr4",
    "sr_4": "sr4",
    "dayhoff9": "dayhoff9",
    "dayhoff-9": "dayhoff9",
    "dayhoff_9": "dayhoff9",
    "dayhoff12": "dayhoff12",
    "dayhoff-12": "dayhoff12",
    "dayhoff_12": "dayhoff12",
    "dayhoff15": "dayhoff15",
    "dayhoff-15": "dayhoff15",
    "dayhoff_15": "dayhoff15",
    "dayhoff18": "dayhoff18",
    "dayhoff-18": "dayhoff18",
    "dayhoff_18": "dayhoff18",
    "srchisq": "srchisq6",
    "sr-chisq": "srchisq6",
    "sr_chisq": "srchisq6",
    "sr-chi-sq": "srchisq6",
    "sr_chi_sq": "srchisq6",
    "srchisq6": "srchisq6",
    "kgbauto": "kgbauto6",
    "kgb-auto": "kgbauto6",
    "kgb_auto": "kgbauto6",
    "kgbauto6": "kgbauto6",
    "ais6": "kgbauto6",
}

SUPPORTED_RECODINGS = tuple(["none"] + list(RECODING_SCHEMES.keys()) + list(AUTO_RECODING_SCHEMES.keys()))


def _validate_scheme(name, groups):
    seen = set()
    for group in groups:
        letters = list(str(group))
        for aa in letters:
            if aa not in _CANONICAL_AA_SET:
                txt = 'Recoding scheme "{}" contains unsupported amino acid symbol: "{}".'
                raise ValueError(txt.format(name, aa))
            if aa in seen:
                txt = 'Recoding scheme "{}" contains duplicate amino acid assignment: "{}".'
                raise ValueError(txt.format(name, aa))
            seen.add(aa)
    if seen != _CANONICAL_AA_SET:
        missing = sorted(_CANONICAL_AA_SET.difference(seen))
        extra = sorted(seen.difference(_CANONICAL_AA_SET))
        txt = 'Recoding scheme "{}" must cover 20 amino acids exactly once. missing={}, extra={}'
        raise ValueError(txt.format(name, ",".join(missing), ",".join(extra)))


for _scheme_name, _groups in RECODING_SCHEMES.items():
    _validate_scheme(_scheme_name, _groups)


def normalize_nonsyn_recode(value):
    if value is None:
        return "none"
    value_txt = str(value).strip().lower()
    if value_txt in _RECODING_ALIASES:
        normalized = _RECODING_ALIASES[value_txt]
    else:
        normalized = value_txt.replace("-", "").replace("_", "")
    if normalized not in SUPPORTED_RECODINGS:
        txt = '--nonsyn_recode should be one of {}.'
        raise ValueError(txt.format(", ".join(SUPPORTED_RECODINGS)))
    return normalized


def _copy_nonsyn_groups_from_amino_acids(g):
    state_orders = [str(aa) for aa in g["amino_acid_orders"]]
    index_map = OrderedDict()
    codon_map = OrderedDict()
    members = OrderedDict()
    aa_to_state = OrderedDict()
    for aa in state_orders:
        indices = np.asarray(g["synonymous_indices"][aa], dtype=np.int64).reshape(-1)
        index_map[aa] = indices.tolist()
        codon_map[aa] = [str(c) for c in g["matrix_groups"][aa]]
        members[aa] = (aa,)
        aa_to_state[aa] = aa
    return state_orders, index_map, codon_map, members, aa_to_state


def _validate_auto_recode_amino_acids(aa_orders):
    aa_set = set(aa_orders)
    missing = sorted(_CANONICAL_AA_SET.difference(aa_set))
    extra = sorted(aa_set.difference(_CANONICAL_AA_SET))
    if (len(missing) > 0) or (len(extra) > 0):
        txt = 'Auto recoding expects canonical 20 amino acids. missing={}, extra={}'
        raise ValueError(txt.format(",".join(missing), ",".join(extra)))


def _build_codon_to_aa_map(g, aa_to_index):
    if "codon_table" not in g:
        raise ValueError('Auto recoding requires "codon_table".')
    codon_to_aa = dict()
    for aa, codon in g["codon_table"]:
        aa = str(aa)
        codon = str(codon).upper().replace("U", "T")
        if (aa == "*") or (aa not in aa_to_index):
            continue
        codon_to_aa[codon] = aa
    if len(codon_to_aa) == 0:
        raise ValueError("No coding codons were found in codon_table.")
    return codon_to_aa


def _encode_alignment_as_aa_matrix(g, aa_orders):
    alignment_file = g.get("alignment_file", "")
    if alignment_file is None or str(alignment_file).strip() == "":
        raise ValueError(
            'Auto recoding requires "alignment_file". '
            "Provide a codon FASTA alignment or use a fixed recoding scheme."
        )
    seq_dict = sequence.read_fasta(alignment_file)
    if len(seq_dict) == 0:
        raise ValueError("Auto recoding requires a non-empty alignment file.")
    aa_to_index = {aa: i for i, aa in enumerate(aa_orders)}
    codon_to_aa = _build_codon_to_aa_map(g=g, aa_to_index=aa_to_index)
    n_state = len(aa_orders)
    counts = np.zeros((len(seq_dict), n_state), dtype=np.float64)
    nsitev = np.zeros((len(seq_dict),), dtype=np.int64)
    aa_rows = []
    codon_site_count = None
    for row_index, seq_txt in enumerate(seq_dict.values()):
        seq_txt = str(seq_txt).upper().replace("U", "T")
        if (len(seq_txt) % 3) != 0:
            raise ValueError("Sequence length is not multiple of 3 in alignment file.")
        this_site_count = len(seq_txt) // 3
        if codon_site_count is None:
            codon_site_count = this_site_count
        elif this_site_count != codon_site_count:
            raise ValueError("Alignment sequences must have identical lengths.")
        aa_index_row = np.full((codon_site_count,), -1, dtype=np.int16)
        valid_site_count = 0
        for site in np.arange(codon_site_count):
            codon = seq_txt[(3 * site) : ((3 * site) + 3)]
            if any([base not in {"A", "C", "G", "T"} for base in codon]):
                continue
            aa = codon_to_aa.get(codon, None)
            if aa is None:
                continue
            aa_index = aa_to_index.get(aa, None)
            if aa_index is None:
                continue
            aa_index_row[site] = aa_index
            counts[row_index, aa_index] += 1.0
            valid_site_count += 1
        aa_rows.append(aa_index_row)
        nsitev[row_index] = int(valid_site_count)
    if np.all(nsitev == 0):
        raise ValueError("Auto recoding failed: no valid amino-acid states were parsed from alignment.")
    aa_matrix = np.vstack(aa_rows)
    fmat = np.zeros_like(counts, dtype=np.float64)
    valid_taxa = nsitev > 0
    fmat[valid_taxa, :] = counts[valid_taxa, :] / nsitev[valid_taxa, np.newaxis]
    fr = counts.sum(axis=0) + _AA_PSEUDOCOUNT
    fr = fr / fr.sum()
    return aa_matrix, fmat, fr, nsitev


def _get_alignment_aa_statistics(g, aa_orders):
    cache = g.get(_AA_ALIGNMENT_CACHE_KEY, None)
    aa_orders_tuple = tuple(aa_orders)
    alignment_file = str(g.get("alignment_file", ""))
    if isinstance(cache, dict):
        if (cache.get("alignment_file", None) == alignment_file) and (cache.get("aa_orders", None) == aa_orders_tuple):
            return cache["aa_matrix"], cache["fmat"], cache["fr"], cache["nsitev"]
    aa_matrix, fmat, fr, nsitev = _encode_alignment_as_aa_matrix(g=g, aa_orders=aa_orders)
    g[_AA_ALIGNMENT_CACHE_KEY] = {
        "alignment_file": alignment_file,
        "aa_orders": aa_orders_tuple,
        "aa_matrix": aa_matrix,
        "fmat": fmat,
        "fr": fr,
        "nsitev": nsitev,
    }
    return aa_matrix, fmat, fr, nsitev


def _get_auto_search_settings(g):
    n_random = int(g.get("nonsyn_recode_random_starts", _DEFAULT_AUTO_RANDOM_STARTS))
    if n_random < 1:
        raise ValueError("nonsyn_recode_random_starts should be >= 1.")
    seed = int(g.get("nonsyn_recode_seed", _DEFAULT_AUTO_RANDOM_SEED))
    return seed, n_random


def _random_bin_assignment(num_item, num_bin, rng):
    if (num_bin < 2) or (num_bin > num_item):
        raise ValueError("Invalid number of bins for random bin assignment.")
    bins = list(range(int(num_item)))
    nb = int(num_item)
    while nb > num_bin:
        l1 = int(rng.integers(low=0, high=nb))
        l2 = int(rng.integers(low=0, high=(nb - 1)))
        if l2 >= l1:
            l2 += 1
        if l2 < l1:
            l1, l2 = l2, l1
        last_label = int(nb - 1)
        for i in range(int(num_item)):
            v = bins[i]
            if v == l2:
                bins[i] = l1
            elif (l2 != last_label) and (v == last_label):
                bins[i] = l2
        nb -= 1
    unique_labels = sorted(set(bins))
    relabel = {old: new for new, old in enumerate(unique_labels)}
    return np.array([relabel[v] for v in bins], dtype=np.int64)


def _can_use_cython_random_bin_assignments(num_item, num_bin, rng, n_random):
    if recoding_cy is None:
        return False
    func = getattr(recoding_cy, "random_bin_assignments_int64", None)
    if func is None:
        return False
    if int(n_random) < 1:
        return False
    if int(num_item) < 1:
        return False
    if (int(num_bin) < 2) or (int(num_bin) > int(num_item)):
        return False
    return hasattr(rng, "integers")


def _random_bin_assignments(num_item, num_bin, rng, n_random):
    num_item = int(num_item)
    num_bin = int(num_bin)
    n_random = int(n_random)
    if n_random < 1:
        raise ValueError("n_random should be >= 1.")
    use_cython = _can_use_cython_random_bin_assignments(
        num_item=num_item,
        num_bin=num_bin,
        rng=rng,
        n_random=n_random,
    )
    if use_cython:
        try:
            out = recoding_cy.random_bin_assignments_int64(
                num_item=num_item,
                num_bin=num_bin,
                rng=rng,
                n_random=n_random,
            )
            out = np.ascontiguousarray(out, dtype=np.int64)
            if out.shape != (n_random, num_item):
                raise ValueError("Unexpected shape from random_bin_assignments_int64.")
            return out
        except Exception:
            pass
    out = np.empty((n_random, num_item), dtype=np.int64)
    for i in range(n_random):
        out[i, :] = _random_bin_assignment(num_item=num_item, num_bin=num_bin, rng=rng)
    return out


def _hill_climb_bins(initial_bins, num_bin, objective_fn):
    bins = np.asarray(initial_bins, dtype=np.int64).copy()
    crit = float(objective_fn(bins))
    counts = np.bincount(bins, minlength=num_bin).astype(np.int64, copy=False)
    while True:
        improved = False
        for el in range(int(bins.shape[0])):
            src = int(bins[el])
            if counts[src] <= 1:
                continue
            for dst in range(int(num_bin)):
                if dst == src:
                    continue
                bins[el] = dst
                crit_new = float(objective_fn(bins))
                if crit_new < (crit - _OBJ_EPS):
                    counts[src] -= 1
                    counts[dst] += 1
                    crit = crit_new
                    improved = True
                    break
                bins[el] = src
            if improved:
                break
        if not improved:
            break
    return bins, crit


def _optimize_bins_random_start(num_item, num_bin, objective_fn, rng, n_random):
    best_bins = None
    best_crit = np.inf
    for _ in range(int(n_random)):
        initial_bins = _random_bin_assignment(num_item=num_item, num_bin=num_bin, rng=rng)
        bins, crit = _hill_climb_bins(initial_bins=initial_bins, num_bin=num_bin, objective_fn=objective_fn)
        if crit < best_crit:
            best_bins = bins.copy()
            best_crit = float(crit)
            if best_crit <= 0.0:
                break
    if best_bins is None:
        raise ValueError("Failed to optimize auto recoding bins.")
    return best_bins, best_crit


def _hill_climb_bins_chisq(initial_bins, num_bin, fmat, fr, nsitev):
    bins = np.asarray(initial_bins, dtype=np.int64).copy()
    num_bin = int(num_bin)
    num_state = int(bins.shape[0])
    counts = np.bincount(bins, minlength=num_bin).astype(np.int64, copy=False)
    frb = np.bincount(bins, weights=fr, minlength=num_bin).astype(np.float64, copy=False)
    num_taxa = int(fmat.shape[0])
    frt = np.zeros((num_taxa, num_bin), dtype=np.float64)
    for b in range(num_bin):
        mask = bins == b
        if np.any(mask):
            frt[:, b] = fmat[:, mask].sum(axis=1)
    term = ((frt - frb[np.newaxis, :]) ** 2) / frb[np.newaxis, :]
    taxon_sum = term.sum(axis=1)
    weighted_sum = taxon_sum * nsitev
    argmax_idx = int(weighted_sum.argmax())
    crit = float(weighted_sum[argmax_idx])
    while True:
        improved = False
        for el in range(num_state):
            src = int(bins[el])
            if counts[src] <= 1:
                continue
            fr_el = float(fr[el])
            fvec = fmat[:, el]
            for dst in range(num_bin):
                if dst == src:
                    continue
                frb_src_new = float(frb[src] - fr_el)
                frb_dst_new = float(frb[dst] + fr_el)
                if (frb_src_new <= 0.0) or (frb_dst_new <= 0.0):
                    continue
                # Lower bound at the current argmax taxon:
                # if this single taxon's updated score does not decrease,
                # the global max cannot improve.
                frt_src_argmax_new = float(frt[argmax_idx, src] - fvec[argmax_idx])
                frt_dst_argmax_new = float(frt[argmax_idx, dst] + fvec[argmax_idx])
                src_argmax_new = ((frt_src_argmax_new - frb_src_new) ** 2) / frb_src_new
                dst_argmax_new = ((frt_dst_argmax_new - frb_dst_new) ** 2) / frb_dst_new
                taxon_argmax_new = (
                    taxon_sum[argmax_idx]
                    - term[argmax_idx, src]
                    - term[argmax_idx, dst]
                    + src_argmax_new
                    + dst_argmax_new
                )
                if (taxon_argmax_new * nsitev[argmax_idx]) >= (crit - _OBJ_EPS):
                    continue
                old_src = term[:, src]
                old_dst = term[:, dst]
                frt_src_new = frt[:, src] - fvec
                frt_dst_new = frt[:, dst] + fvec
                new_src = ((frt_src_new - frb_src_new) ** 2) / frb_src_new
                new_dst = ((frt_dst_new - frb_dst_new) ** 2) / frb_dst_new
                taxon_sum_new = taxon_sum - old_src - old_dst + new_src + new_dst
                crit_new = float((taxon_sum_new * nsitev).max())
                if crit_new < (crit - _OBJ_EPS):
                    bins[el] = dst
                    counts[src] -= 1
                    counts[dst] += 1
                    frb[src] = frb_src_new
                    frb[dst] = frb_dst_new
                    frt[:, src] = frt_src_new
                    frt[:, dst] = frt_dst_new
                    term[:, src] = new_src
                    term[:, dst] = new_dst
                    taxon_sum = taxon_sum_new
                    weighted_sum = taxon_sum * nsitev
                    argmax_idx = int(weighted_sum.argmax())
                    crit = float(weighted_sum[argmax_idx])
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return bins, crit


def _can_use_cython_hill_climb_bins_chisq(fmat, fr, nsitev):
    if recoding_cy is None:
        return False
    func = getattr(recoding_cy, "hill_climb_bins_chisq_double", None)
    if func is None:
        return False
    if not isinstance(fmat, np.ndarray):
        return False
    if not isinstance(fr, np.ndarray):
        return False
    if not isinstance(nsitev, np.ndarray):
        return False
    if fmat.dtype != np.float64:
        return False
    if fr.dtype != np.float64:
        return False
    if nsitev.dtype != np.float64:
        return False
    if fmat.ndim != 2:
        return False
    if fr.ndim != 1:
        return False
    if nsitev.ndim != 1:
        return False
    if fmat.shape[1] != fr.shape[0]:
        return False
    if fmat.shape[0] != nsitev.shape[0]:
        return False
    if not fmat.flags["C_CONTIGUOUS"]:
        return False
    if not fr.flags["C_CONTIGUOUS"]:
        return False
    if not nsitev.flags["C_CONTIGUOUS"]:
        return False
    return True


def _can_use_cython_search_chunk_chisq(initial_bins_chunk, num_bin, fmat, fr, nsitev):
    if recoding_cy is None:
        return False
    func = getattr(recoding_cy, "search_initial_bins_chunk_chisq_double", None)
    if func is None:
        return False
    if not isinstance(initial_bins_chunk, np.ndarray):
        return False
    if initial_bins_chunk.dtype != np.int64:
        return False
    if initial_bins_chunk.ndim != 2:
        return False
    if initial_bins_chunk.shape[1] != int(fr.shape[0]):
        return False
    if not initial_bins_chunk.flags["C_CONTIGUOUS"]:
        return False
    if int(num_bin) < 2:
        return False
    return _can_use_cython_hill_climb_bins_chisq(fmat=fmat, fr=fr, nsitev=nsitev)


def _can_use_cython_hill_climb_bins_conductance(pi, weighted_q):
    if recoding_cy is None:
        return False
    func = getattr(recoding_cy, "hill_climb_bins_conductance_double", None)
    if func is None:
        return False
    if not isinstance(pi, np.ndarray):
        return False
    if not isinstance(weighted_q, np.ndarray):
        return False
    if pi.dtype != np.float64:
        return False
    if weighted_q.dtype != np.float64:
        return False
    if pi.ndim != 1:
        return False
    if weighted_q.ndim != 2:
        return False
    if weighted_q.shape[0] != weighted_q.shape[1]:
        return False
    if weighted_q.shape[0] != pi.shape[0]:
        return False
    if not pi.flags["C_CONTIGUOUS"]:
        return False
    if not weighted_q.flags["C_CONTIGUOUS"]:
        return False
    return True


def _can_use_cython_search_chunk_conductance(initial_bins_chunk, num_bin, pi, weighted_q):
    if recoding_cy is None:
        return False
    func = getattr(recoding_cy, "search_initial_bins_chunk_conductance_double", None)
    if func is None:
        return False
    if not isinstance(initial_bins_chunk, np.ndarray):
        return False
    if initial_bins_chunk.dtype != np.int64:
        return False
    if initial_bins_chunk.ndim != 2:
        return False
    if initial_bins_chunk.shape[1] != int(pi.shape[0]):
        return False
    if not initial_bins_chunk.flags["C_CONTIGUOUS"]:
        return False
    if int(num_bin) < 2:
        return False
    return _can_use_cython_hill_climb_bins_conductance(pi=pi, weighted_q=weighted_q)


def _conductance_score(cap, out):
    if np.any(cap <= 0):
        return np.inf
    return float((out / cap).sum())


def _hill_climb_bins_conductance(initial_bins, num_bin, pi, weighted_q):
    bins = np.asarray(initial_bins, dtype=np.int64).copy()
    num_bin = int(num_bin)
    num_state = int(bins.shape[0])
    counts = np.bincount(bins, minlength=num_bin).astype(np.int64, copy=False)
    cap = np.bincount(bins, weights=pi, minlength=num_bin).astype(np.float64, copy=False)
    membership = np.zeros((num_state, num_bin), dtype=np.float64)
    membership[np.arange(num_state), bins] = 1.0
    flow = membership.T @ weighted_q @ membership
    out = flow.sum(axis=1) - np.diag(flow)
    crit = _conductance_score(cap=cap, out=out)
    while True:
        improved = False
        for el in range(num_state):
            src = int(bins[el])
            if counts[src] <= 1:
                continue
            pi_el = float(pi[el])
            row_bin_sum = np.bincount(bins, weights=weighted_q[el, :], minlength=num_bin).astype(np.float64, copy=False)
            col_bin_sum = np.bincount(bins, weights=weighted_q[:, el], minlength=num_bin).astype(np.float64, copy=False)
            row_total = float(row_bin_sum.sum())
            old_src_term = float(out[src] / cap[src])
            for dst in range(num_bin):
                if dst == src:
                    continue
                cap_src_new = float(cap[src] - pi_el)
                cap_dst_new = float(cap[dst] + pi_el)
                if cap_src_new <= 0.0:
                    continue
                out_src_new = float(out[src] - (row_total - row_bin_sum[src]) + col_bin_sum[src])
                out_dst_new = float(out[dst] + (row_total - row_bin_sum[dst]) - col_bin_sum[dst])
                old_dst_term = float(out[dst] / cap[dst])
                new_src_term = float(out_src_new / cap_src_new)
                new_dst_term = float(out_dst_new / cap_dst_new)
                crit_new = float(crit - old_src_term - old_dst_term + new_src_term + new_dst_term)
                if crit_new < (crit - _OBJ_EPS):
                    bins[el] = dst
                    counts[src] -= 1
                    counts[dst] += 1
                    cap[src] = cap_src_new
                    cap[dst] = cap_dst_new
                    out[src] = out_src_new
                    out[dst] = out_dst_new
                    crit = crit_new
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return bins, crit


def _is_better_auto_candidate(crit, start_index, best_crit, best_start_index):
    if crit < (best_crit - _OBJ_EPS):
        return True
    if (abs(crit - best_crit) <= _OBJ_EPS) and (start_index < best_start_index):
        return True
    return False


def _resolve_auto_recode_parallel_n_jobs(g, n_random, work_scale=1):
    if g is None:
        return 1
    n_random = int(n_random)
    work_scale = max(1, int(work_scale))
    effective_starts = int(n_random * work_scale)
    requested_backend = str(g.get("parallel_backend", "auto")).lower()
    if requested_backend == "auto":
        min_total_starts = int(g.get("nonsyn_recode_parallel_min_total_starts", 30000))
        if min_total_starts < 1:
            raise ValueError("nonsyn_recode_parallel_min_total_starts should be >= 1.")
        if effective_starts < min_total_starts:
            return 1
    threads = int(g.get("threads", 1))
    n_jobs = parallel.resolve_n_jobs(num_items=n_random, threads=threads)
    if n_jobs <= 1:
        return 1
    min_starts = int(g.get("nonsyn_recode_parallel_min_starts_per_job", 5000))
    if min_starts < 1:
        raise ValueError("nonsyn_recode_parallel_min_starts_per_job should be >= 1.")
    max_jobs_by_work = max(1, effective_starts // min_starts)
    return max(1, min(n_jobs, max_jobs_by_work))


def _resolve_auto_recode_parallel_backend(g, prefer_threading=False):
    if g is None:
        if prefer_threading:
            return "threading"
        return "multiprocessing"
    resolved = parallel.resolve_parallel_backend(g=g, task="general")
    requested_backend = str(g.get("parallel_backend", "auto")).lower()
    if prefer_threading and (requested_backend == "auto"):
        return "threading"
    return resolved


def _resolve_auto_recode_chunk_factor(g, total_work_units):
    base_chunk_factor = parallel.resolve_chunk_factor(g=g, task="general")
    if g is None:
        return int(base_chunk_factor)
    if "parallel_chunk_factor" in g:
        return int(base_chunk_factor)
    total_work_units = int(total_work_units)
    if total_work_units >= 10_000_000:
        return max(int(base_chunk_factor), 8)
    if total_work_units >= 6_000_000:
        return max(int(base_chunk_factor), 4)
    return int(base_chunk_factor)


def _search_initial_bins_chunk_chisq(initial_bins_chunk, start_index, num_bin, fmat, fr, nsitev, use_cython=None):
    best_bins = None
    best_crit = np.inf
    best_start = np.iinfo(np.int64).max
    initial_bins_chunk = np.asarray(initial_bins_chunk, dtype=np.int64)
    if initial_bins_chunk.ndim == 1:
        initial_bins_chunk = initial_bins_chunk[np.newaxis, :]
    if initial_bins_chunk.ndim != 2:
        raise ValueError("initial_bins_chunk should be 2-dimensional.")
    if use_cython is None:
        use_cython = _can_use_cython_hill_climb_bins_chisq(fmat=fmat, fr=fr, nsitev=nsitev)
    else:
        use_cython = bool(use_cython)
    if use_cython and _can_use_cython_search_chunk_chisq(
        initial_bins_chunk=initial_bins_chunk,
        num_bin=num_bin,
        fmat=fmat,
        fr=fr,
        nsitev=nsitev,
    ):
        try:
            bins, crit, best_offset = recoding_cy.search_initial_bins_chunk_chisq_double(
                initial_bins_chunk=initial_bins_chunk,
                num_bin=int(num_bin),
                fmat=fmat,
                fr=fr,
                nsitev=nsitev,
                obj_eps=float(_OBJ_EPS),
            )
            return bins.copy(), float(crit), int(start_index + int(best_offset))
        except Exception:
            use_cython = False
    cython_fn = None
    if use_cython:
        cython_fn = recoding_cy.hill_climb_bins_chisq_double
    for offset in range(int(initial_bins_chunk.shape[0])):
        initial_bins = initial_bins_chunk[offset, :]
        if cython_fn is not None:
            bins, crit = cython_fn(
                initial_bins=initial_bins,
                num_bin=int(num_bin),
                fmat=fmat,
                fr=fr,
                nsitev=nsitev,
                obj_eps=float(_OBJ_EPS),
            )
        else:
            bins, crit = _hill_climb_bins_chisq(
                initial_bins=initial_bins,
                num_bin=num_bin,
                fmat=fmat,
                fr=fr,
                nsitev=nsitev,
            )
        global_start = int(start_index + offset)
        if _is_better_auto_candidate(
            crit=float(crit),
            start_index=global_start,
            best_crit=best_crit,
            best_start_index=best_start,
        ):
            best_bins = bins.copy()
            best_crit = float(crit)
            best_start = int(global_start)
            if best_crit <= 0.0:
                break
    return best_bins, float(best_crit), int(best_start)


def _search_initial_bins_chunk_conductance(initial_bins_chunk, start_index, num_bin, pi, weighted_q, use_cython=None):
    best_bins = None
    best_crit = np.inf
    best_start = np.iinfo(np.int64).max
    initial_bins_chunk = np.asarray(initial_bins_chunk, dtype=np.int64)
    if initial_bins_chunk.ndim == 1:
        initial_bins_chunk = initial_bins_chunk[np.newaxis, :]
    if initial_bins_chunk.ndim != 2:
        raise ValueError("initial_bins_chunk should be 2-dimensional.")
    if use_cython is None:
        use_cython = _can_use_cython_hill_climb_bins_conductance(pi=pi, weighted_q=weighted_q)
    else:
        use_cython = bool(use_cython)
    if use_cython and _can_use_cython_search_chunk_conductance(
        initial_bins_chunk=initial_bins_chunk,
        num_bin=num_bin,
        pi=pi,
        weighted_q=weighted_q,
    ):
        try:
            bins, crit, best_offset = recoding_cy.search_initial_bins_chunk_conductance_double(
                initial_bins_chunk=initial_bins_chunk,
                num_bin=int(num_bin),
                pi=pi,
                weighted_q=weighted_q,
                obj_eps=float(_OBJ_EPS),
            )
            return bins.copy(), float(crit), int(start_index + int(best_offset))
        except Exception:
            use_cython = False
    cython_fn = None
    if use_cython:
        cython_fn = recoding_cy.hill_climb_bins_conductance_double
    for offset in range(int(initial_bins_chunk.shape[0])):
        initial_bins = initial_bins_chunk[offset, :]
        if cython_fn is not None:
            bins, crit = cython_fn(
                initial_bins=initial_bins,
                num_bin=int(num_bin),
                pi=pi,
                weighted_q=weighted_q,
                obj_eps=float(_OBJ_EPS),
            )
        else:
            bins, crit = _hill_climb_bins_conductance(
                initial_bins=initial_bins,
                num_bin=num_bin,
                pi=pi,
                weighted_q=weighted_q,
            )
        global_start = int(start_index + offset)
        if _is_better_auto_candidate(
            crit=float(crit),
            start_index=global_start,
            best_crit=best_crit,
            best_start_index=best_start,
        ):
            best_bins = bins.copy()
            best_crit = float(crit)
            best_start = int(global_start)
            if best_crit <= 0.0:
                break
    return best_bins, float(best_crit), int(best_start)


def _optimize_bins_random_start_chisq(num_item, num_bin, fmat, fr, nsitev, rng, n_random, g=None):
    best_bins = None
    best_crit = np.inf
    best_start = np.iinfo(np.int64).max
    n_random = int(n_random)
    fmat = np.ascontiguousarray(fmat, dtype=np.float64)
    fr = np.ascontiguousarray(fr, dtype=np.float64)
    nsitev = np.ascontiguousarray(nsitev, dtype=np.float64)
    use_cython = _can_use_cython_hill_climb_bins_chisq(fmat=fmat, fr=fr, nsitev=nsitev)
    taxa_scale = max(1, int((float(fmat.shape[0]) + 31.0) // 32.0))
    n_jobs = _resolve_auto_recode_parallel_n_jobs(g=g, n_random=n_random, work_scale=taxa_scale)
    initial_bins_all = _random_bin_assignments(num_item=num_item, num_bin=num_bin, rng=rng, n_random=n_random)
    if n_jobs == 1:
        results = [
            _search_initial_bins_chunk_chisq(
                initial_bins_chunk=initial_bins_all,
                start_index=0,
                num_bin=num_bin,
                fmat=fmat,
                fr=fr,
                nsitev=nsitev,
                use_cython=use_cython,
            )
        ]
    else:
        backend = _resolve_auto_recode_parallel_backend(g=g, prefer_threading=use_cython)
        total_work_units = int(n_random * int(fmat.shape[0]))
        chunk_factor = _resolve_auto_recode_chunk_factor(g=g, total_work_units=total_work_units)
        initial_chunks, starts = parallel.get_chunks(input_data=initial_bins_all, threads=n_jobs, chunk_factor=chunk_factor)
        args_iterable = [
            (initial_chunk, int(start), num_bin, fmat, fr, nsitev, use_cython)
            for initial_chunk, start in zip(initial_chunks, starts)
        ]
        results = parallel.run_starmap(
            func=_search_initial_bins_chunk_chisq,
            args_iterable=args_iterable,
            n_jobs=n_jobs,
            backend=backend,
        )
    for bins, crit, start_idx in results:
        if bins is None:
            continue
        if _is_better_auto_candidate(
            crit=float(crit),
            start_index=int(start_idx),
            best_crit=best_crit,
            best_start_index=best_start,
        ):
            best_bins = bins.copy()
            best_crit = float(crit)
            best_start = int(start_idx)
    if best_bins is None:
        raise ValueError("Failed to optimize auto recoding bins.")
    return best_bins, best_crit


def _optimize_bins_random_start_conductance(num_item, num_bin, pi, weighted_q, rng, n_random, g=None):
    best_bins = None
    best_crit = np.inf
    best_start = np.iinfo(np.int64).max
    n_random = int(n_random)
    pi = np.ascontiguousarray(pi, dtype=np.float64)
    weighted_q = np.ascontiguousarray(weighted_q, dtype=np.float64)
    use_cython = _can_use_cython_hill_climb_bins_conductance(pi=pi, weighted_q=weighted_q)
    n_jobs = _resolve_auto_recode_parallel_n_jobs(g=g, n_random=n_random, work_scale=1)
    initial_bins_all = _random_bin_assignments(num_item=num_item, num_bin=num_bin, rng=rng, n_random=n_random)
    if n_jobs == 1:
        results = [
            _search_initial_bins_chunk_conductance(
                initial_bins_chunk=initial_bins_all,
                start_index=0,
                num_bin=num_bin,
                pi=pi,
                weighted_q=weighted_q,
                use_cython=use_cython,
            )
        ]
    else:
        backend = _resolve_auto_recode_parallel_backend(g=g, prefer_threading=use_cython)
        chunk_factor = parallel.resolve_chunk_factor(g=g, task="general")
        initial_chunks, starts = parallel.get_chunks(input_data=initial_bins_all, threads=n_jobs, chunk_factor=chunk_factor)
        args_iterable = [
            (initial_chunk, int(start), num_bin, pi, weighted_q, use_cython)
            for initial_chunk, start in zip(initial_chunks, starts)
        ]
        results = parallel.run_starmap(
            func=_search_initial_bins_chunk_conductance,
            args_iterable=args_iterable,
            n_jobs=n_jobs,
            backend=backend,
        )
    for bins, crit, start_idx in results:
        if bins is None:
            continue
        if _is_better_auto_candidate(
            crit=float(crit),
            start_index=int(start_idx),
            best_crit=best_crit,
            best_start_index=best_start,
        ):
            best_bins = bins.copy()
            best_crit = float(crit)
            best_start = int(start_idx)
    if best_bins is None:
        raise ValueError("Failed to optimize auto recoding bins.")
    return best_bins, best_crit


def _chisq_max_criterion(bin_assignment, fmat, fr, nsitev, num_bin):
    bin_assignment = np.asarray(bin_assignment, dtype=np.int64).reshape(-1)
    membership = np.zeros((bin_assignment.shape[0], int(num_bin)), dtype=np.float64)
    membership[np.arange(bin_assignment.shape[0]), bin_assignment] = 1.0
    frb = fr @ membership
    if np.any(frb <= 0):
        return np.inf
    frt = fmat @ membership
    chisq = (((frt - frb[np.newaxis, :]) ** 2) / frb[np.newaxis, :]).sum(axis=1)
    chisq = chisq * nsitev
    return float(chisq.max())


def _estimate_empirical_transition_matrix(aa_matrix, num_state):
    pair_counts = np.zeros((num_state, num_state), dtype=np.float64)
    num_site = int(aa_matrix.shape[1])
    for site in range(num_site):
        col = aa_matrix[:, site]
        valid = col >= 0
        if not np.any(valid):
            continue
        counts = np.bincount(col[valid].astype(np.int64, copy=False), minlength=int(num_state)).astype(
            np.float64,
            copy=False,
        )
        pair_counts += np.outer(counts, counts)
    np.fill_diagonal(pair_counts, 0.0)
    pair_counts = pair_counts + _AA_PSEUDOCOUNT
    np.fill_diagonal(pair_counts, 0.0)
    q = np.zeros_like(pair_counts)
    row_sum = pair_counts.sum(axis=1)
    valid_row = row_sum > 0
    q[valid_row, :] = pair_counts[valid_row, :] / row_sum[valid_row, np.newaxis]
    return q


def _conductance_criterion(bin_assignment, pi, weighted_q, num_bin):
    bin_assignment = np.asarray(bin_assignment, dtype=np.int64).reshape(-1)
    membership = np.zeros((bin_assignment.shape[0], int(num_bin)), dtype=np.float64)
    membership[np.arange(bin_assignment.shape[0]), bin_assignment] = 1.0
    cap = pi @ membership
    if np.any(cap <= 0):
        return np.inf
    flow = membership.T @ weighted_q @ membership
    phi = flow / cap[:, np.newaxis]
    np.fill_diagonal(phi, 0.0)
    return float(phi.sum())


def _groups_from_bin_assignment(aa_orders, bin_assignment, num_bin):
    aa_pos = {aa: i for i, aa in enumerate(aa_orders)}
    grouped = []
    for b in np.arange(num_bin):
        members = [aa_orders[i] for i in np.arange(len(aa_orders)) if int(bin_assignment[i]) == int(b)]
        if len(members) == 0:
            continue
        grouped.append(tuple(members))
    if len(grouped) != num_bin:
        raise ValueError("Auto recoding generated empty bin(s).")
    grouped = sorted(grouped, key=lambda members: min([aa_pos[aa] for aa in members]))
    return tuple(["".join(members) for members in grouped])


def _build_auto_recoded_groups(g, scheme_name):
    spec = AUTO_RECODING_SCHEMES[scheme_name]
    family = spec["family"]
    num_bin = int(spec["n_bins"])
    aa_orders = [str(aa) for aa in g["amino_acid_orders"]]
    _validate_auto_recode_amino_acids(aa_orders=aa_orders)
    aa_matrix, fmat, fr, nsitev = _get_alignment_aa_statistics(g=g, aa_orders=aa_orders)
    seed, n_random = _get_auto_search_settings(g=g)
    rng = np.random.default_rng(seed=seed)
    if family == "srchisq":
        best_bins, best_crit = _optimize_bins_random_start_chisq(
            num_item=len(aa_orders),
            num_bin=num_bin,
            fmat=fmat,
            fr=fr,
            nsitev=nsitev,
            rng=rng,
            n_random=n_random,
            g=g,
        )
    elif family == "kgbauto":
        q = _estimate_empirical_transition_matrix(aa_matrix=aa_matrix, num_state=len(aa_orders))
        weighted_q = fr[:, np.newaxis] * q
        best_bins, best_crit = _optimize_bins_random_start_conductance(
            num_item=len(aa_orders),
            num_bin=num_bin,
            pi=fr,
            weighted_q=weighted_q,
            rng=rng,
            n_random=n_random,
            g=g,
        )
    else:
        raise ValueError('Unsupported auto-recoding family "{}".'.format(family))
    g["nonsyn_recode_auto_score"] = float(best_crit)
    g["nonsyn_recode_auto_random_starts"] = int(n_random)
    g["nonsyn_recode_auto_seed"] = int(seed)
    groups = _groups_from_bin_assignment(aa_orders=aa_orders, bin_assignment=best_bins, num_bin=num_bin)
    _validate_scheme(scheme_name, groups)
    return groups


def _build_recoded_groups(g, scheme_name):
    if scheme_name in RECODING_SCHEMES:
        groups = RECODING_SCHEMES[scheme_name]
    elif scheme_name in AUTO_RECODING_SCHEMES:
        groups = _build_auto_recoded_groups(g=g, scheme_name=scheme_name)
    else:
        raise ValueError('Unsupported recoding scheme "{}".'.format(scheme_name))
    aa_orders = [str(aa) for aa in g["amino_acid_orders"]]
    aa_set = set(aa_orders)
    unsupported_aa = sorted(aa_set.difference(_CANONICAL_AA_SET))
    if len(unsupported_aa) > 0:
        txt = "Unsupported amino acid(s) found in input state orders for recoding: {}"
        raise ValueError(txt.format(",".join(unsupported_aa)))
    aa_to_state = dict()
    for group in groups:
        for aa in group:
            aa_to_state[aa] = group
    missing = sorted([aa for aa in aa_orders if aa not in aa_to_state])
    if len(missing) > 0:
        txt = 'Recoding scheme "{}" does not define class membership for: {}'
        raise ValueError(txt.format(scheme_name, ",".join(missing)))
    state_orders = []
    index_map = OrderedDict()
    codon_map = OrderedDict()
    members = OrderedDict()
    codon_orders = [str(c) for c in g.get("codon_orders", [])]
    for group in groups:
        group_members = [aa for aa in aa_orders if aa_to_state[aa] == group]
        if len(group_members) == 0:
            continue
        group_indices = []
        for aa in group_members:
            group_indices.extend(np.asarray(g["synonymous_indices"][aa], dtype=np.int64).reshape(-1).tolist())
        if len(group_indices) == 0:
            continue
        group_indices = sorted(list(set([int(i) for i in group_indices])))
        if len(codon_orders) > 0:
            group_codons = [codon_orders[i] for i in group_indices]
        else:
            group_codons = []
            for aa in group_members:
                group_codons.extend([str(c) for c in g["matrix_groups"].get(aa, [])])
            group_codons = list(dict.fromkeys(group_codons))
        label = "".join(group_members)
        state_orders.append(label)
        index_map[label] = group_indices
        codon_map[label] = group_codons
        members[label] = tuple(group_members)
    aa_to_state_out = OrderedDict()
    for state_label, state_members in members.items():
        for aa in state_members:
            aa_to_state_out[aa] = state_label
    return state_orders, index_map, codon_map, members, aa_to_state_out


def initialize_nonsyn_groups(g):
    recode = normalize_nonsyn_recode(g.get("nonsyn_recode", "none"))
    g["nonsyn_recode"] = recode
    required_keys = ["amino_acid_orders", "synonymous_indices", "matrix_groups"]
    missing_keys = [key for key in required_keys if key not in g]
    if len(missing_keys) > 0:
        txt = "Missing required key(s) for nonsynonymous recoding initialization: {}"
        raise ValueError(txt.format(", ".join(missing_keys)))
    if recode == "none":
        state_orders, index_map, codon_map, members, aa_to_state = _copy_nonsyn_groups_from_amino_acids(g)
    else:
        state_orders, index_map, codon_map, members, aa_to_state = _build_recoded_groups(g, scheme_name=recode)
    if len(state_orders) == 0:
        raise ValueError("No nonsynonymous recoding states were generated.")
    max_size = max([len(index_map[state]) for state in state_orders])
    g["nonsyn_state_orders"] = np.array(state_orders, dtype=object)
    g["nonsynonymous_indices"] = index_map
    g["nonsyn_matrix_groups"] = codon_map
    g["nonsyn_state_members"] = members
    g["nonsyn_aa_to_state"] = aa_to_state
    g["max_nonsynonymous_size"] = int(max_size)
    return g


def _join_values(values):
    return ",".join([str(v) for v in values])


def write_nonsyn_recoding_table(g, output_path="csubst_nonsyn_recoding.tsv"):
    recode = normalize_nonsyn_recode(g.get("nonsyn_recode", "none"))
    if recode == "none":
        return None
    required_keys = [
        "amino_acid_orders",
        "nonsyn_state_orders",
        "nonsyn_state_members",
        "nonsyn_aa_to_state",
        "synonymous_indices",
        "matrix_groups",
        "nonsynonymous_indices",
        "nonsyn_matrix_groups",
    ]
    missing_keys = [key for key in required_keys if key not in g]
    if len(missing_keys) > 0:
        txt = "Missing required key(s) for nonsynonymous recoding output: {}"
        raise ValueError(txt.format(", ".join(missing_keys)))
    aa_orders = [str(aa) for aa in g["amino_acid_orders"]]
    state_orders = [str(state) for state in g["nonsyn_state_orders"]]
    state_index = {state: (i + 1) for i, state in enumerate(state_orders)}
    lines = [
        "\t".join(
            [
                "recode",
                "state_id",
                "state_label",
                "state_members",
                "amino_acid",
                "aa_codon_indices",
                "aa_codons",
                "state_codon_indices",
                "state_codons",
            ]
        )
    ]
    for aa in aa_orders:
        state = str(g["nonsyn_aa_to_state"][aa])
        state_members = [str(v) for v in g["nonsyn_state_members"][state]]
        aa_indices = sorted(np.asarray(g["synonymous_indices"][aa], dtype=np.int64).reshape(-1).tolist())
        aa_codons = [str(c) for c in g["matrix_groups"][aa]]
        state_indices = sorted(np.asarray(g["nonsynonymous_indices"][state], dtype=np.int64).reshape(-1).tolist())
        state_codons = [str(c) for c in g["nonsyn_matrix_groups"][state]]
        lines.append(
            "\t".join(
                [
                    str(recode),
                    str(state_index[state]),
                    str(state),
                    _join_values(state_members),
                    str(aa),
                    _join_values(aa_indices),
                    _join_values(aa_codons),
                    _join_values(state_indices),
                    _join_values(state_codons),
                ]
            )
        )
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return output_path


def _get_scheme_groups_for_pca(g):
    groups_by_scheme = OrderedDict()
    groups_by_scheme["none"] = tuple([str(aa) for aa in _CANONICAL_AA])
    for scheme_name, groups in RECODING_SCHEMES.items():
        groups_by_scheme[str(scheme_name)] = tuple([str(group) for group in groups])
    recode = normalize_nonsyn_recode(g.get("nonsyn_recode", "none"))
    if (recode in AUTO_RECODING_SCHEMES) and ("nonsyn_state_orders" not in g):
        raise ValueError('Missing "nonsyn_state_orders" for auto recoding PCA output.')
    g_for_auto = dict(g)
    for auto_name in AUTO_RECODING_SCHEMES.keys():
        if (recode == auto_name) and ("nonsyn_state_orders" in g):
            groups_by_scheme[auto_name] = tuple([str(state) for state in g["nonsyn_state_orders"].tolist()])
            continue
        try:
            groups = _build_auto_recoded_groups(g=g_for_auto, scheme_name=auto_name)
        except (KeyError, TypeError, ValueError):
            continue
        groups_by_scheme[auto_name] = tuple([str(group) for group in groups])
    return groups_by_scheme


def _validate_scheme_aa_membership(groups):
    aa_to_group = dict()
    for i,group in enumerate(groups):
        group_txt = str(group)
        for aa in list(group_txt):
            if aa not in _CANONICAL_AA_SET:
                txt = 'Unsupported amino acid "{}" in recoding group "{}".'
                raise ValueError(txt.format(aa, group_txt))
            if aa in aa_to_group:
                txt = 'Duplicate amino acid "{}" across recoding groups.'
                raise ValueError(txt.format(aa))
            aa_to_group[aa] = int(i)
    missing = sorted([aa for aa in _CANONICAL_AA if aa not in aa_to_group])
    if len(missing) > 0:
        txt = 'Recoding groups should cover all 20 amino acids exactly once. Missing: {}'
        raise ValueError(txt.format(",".join(missing)))
    return aa_to_group


def _build_co_cluster_feature_vector(groups):
    aa_to_group = _validate_scheme_aa_membership(groups=groups)
    values = []
    for i,aa1 in enumerate(_CANONICAL_AA):
        for aa2 in _CANONICAL_AA[(i + 1) :]:
            values.append(1.0 if aa_to_group[aa1] == aa_to_group[aa2] else 0.0)
    return np.array(values, dtype=np.float64)


def _project_pca_2d(feature_matrix):
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix should be 2D.")
    if feature_matrix.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((2,), dtype=np.float64)
    centered = feature_matrix - feature_matrix.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    coords = np.zeros((feature_matrix.shape[0], 2), dtype=np.float64)
    max_dim = min(2, s.shape[0])
    if max_dim > 0:
        coords[:, :max_dim] = u[:, :max_dim] * s[:max_dim]
    if s.shape[0] == 0:
        return coords, np.zeros((2,), dtype=np.float64)
    denom = max(1, feature_matrix.shape[0] - 1)
    variance = (s ** 2) / float(denom)
    total_variance = float(variance.sum())
    explained = np.zeros((2,), dtype=np.float64)
    if total_variance > 0:
        explained[:max_dim] = variance[:max_dim] / total_variance
    return coords, explained


def write_nonsyn_recoding_pca_plot(g, output_path="csubst_nonsyn_recoding_pca.png"):
    recode = normalize_nonsyn_recode(g.get("nonsyn_recode", "none"))
    groups_by_scheme = _get_scheme_groups_for_pca(g=g)
    scheme_names = list(groups_by_scheme.keys())
    feature_matrix = np.vstack(
        [_build_co_cluster_feature_vector(groups_by_scheme[name]) for name in scheme_names]
    )
    coords, explained = _project_pca_2d(feature_matrix=feature_matrix)

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
    with mpl.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        x = coords[:, 0].astype(np.float64, copy=False)
        y = coords[:, 1].astype(np.float64, copy=False)
        x_span = float(x.max() - x.min()) if x.shape[0] > 0 else 1.0
        y_span = float(y.max() - y.min()) if y.shape[0] > 0 else 1.0
        if x_span <= 0:
            x_span = 1.0
        if y_span <= 0:
            y_span = 1.0

        # Lightweight ggrepel-like label adjustment in y-direction for nearby points.
        label_x = x + (0.012 * x_span)
        label_y = y.copy()
        y_sep = 0.08 * y_span
        x_neighbor = 0.45 * x_span
        for _ in np.arange(200):
            moved = False
            for i in np.arange(label_y.shape[0] - 1):
                for j in np.arange(i + 1, label_y.shape[0]):
                    if abs(label_x[i] - label_x[j]) > x_neighbor:
                        continue
                    dy = label_y[j] - label_y[i]
                    if abs(dy) >= y_sep:
                        continue
                    shift = 0.5 * (y_sep - abs(dy))
                    if dy >= 0:
                        label_y[i] -= shift
                        label_y[j] += shift
                    else:
                        label_y[i] += shift
                        label_y[j] -= shift
                    moved = True
            if not moved:
                break

        has_auto_note = ("srchisq6" in scheme_names) or ("kgbauto6" in scheme_names)
        y_all = np.concatenate([y, label_y]) if y.shape[0] > 0 else y
        y_pad = 0.08 * y_span
        y_note_pad = (0.34 * y_span) if has_auto_note else 0.0
        y_min = float(y_all.min() - y_pad - y_note_pad)
        y_max = float(y_all.max() + y_pad)
        x_min = float(x.min() - (0.08 * x_span))
        x_max = float(label_x.max() + (0.10 * x_span))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        for i, scheme_name in enumerate(scheme_names):
            marker_size = 24
            ax.scatter(
                [float(x[i])],
                [float(y[i])],
                color="black",
                s=marker_size,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )
            ax.text(
                float(label_x[i]),
                float(label_y[i]),
                str(scheme_name),
                color="black",
                fontsize=8,
                fontfamily="Helvetica",
                va="center",
                ha="left",
            )

        pc1_txt = "PC1 ({:.1f}%)".format(float(explained[0] * 100.0))
        pc2_txt = "PC2 ({:.1f}%)".format(float(explained[1] * 100.0))
        ax.set_xlabel(pc1_txt, fontfamily="Helvetica", fontsize=8, color="black")
        ax.set_ylabel(pc2_txt, fontfamily="Helvetica", fontsize=8, color="black")
        ax.set_title("Amino acid recoding PCA", fontfamily="Helvetica", fontsize=8, color="black")
        ax.tick_params(axis="both", which="both", labelsize=8, colors="black")
        if has_auto_note:
            note_txt = "Note: srchisq6 and kgbauto6\nare inferred from the input dataset."
            ax.text(
                0.99,
                0.01,
                note_txt,
                transform=ax.transAxes,
                fontsize=8,
                fontfamily="Helvetica",
                color="black",
                ha="right",
                va="bottom",
            )
        ax.grid(True, linewidth=0.4, color="#d9d9d9", zorder=0)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    return output_path

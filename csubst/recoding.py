from collections import OrderedDict

import numpy as np


_CANONICAL_AA = tuple("ACDEFGHIKLMNPQRSTVWY")
_CANONICAL_AA_SET = frozenset(_CANONICAL_AA)

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
}

SUPPORTED_RECODINGS = tuple(["none"] + list(RECODING_SCHEMES.keys()))


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


def _build_recoded_groups(g, scheme_name):
    groups = RECODING_SCHEMES[scheme_name]
    aa_orders = [str(aa) for aa in g["amino_acid_orders"]]
    aa_set = set(aa_orders)
    unsupported_aa = sorted(aa_set.difference(_CANONICAL_AA_SET))
    if len(unsupported_aa) > 0:
        txt = 'Unsupported amino acid(s) found in input state orders for recoding: {}'
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
        txt = 'Missing required key(s) for nonsynonymous recoding initialization: {}'
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

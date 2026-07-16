"""Lightweight nonsynonymous-recoding option definitions.

Keep argument normalization independent of the numerical recoding module so
ordinary (unrecoded) runs do not load its compiled extension during startup.
"""

from collections import OrderedDict


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
    "no": "no",
    "3di": "3di20",
    "3di20": "3di20",
    "threedi": "3di20",
    "threedi20": "3di20",
    "structuralalphabet": "3di20",
    "structural_alphabet": "3di20",
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

SUPPORTED_RECODINGS = tuple(
    ["no", "3di20"] + list(RECODING_SCHEMES.keys()) + list(AUTO_RECODING_SCHEMES.keys())
)


def normalize_nonsyn_recode(value):
    if value is None:
        return "no"
    value_txt = str(value).strip().lower()
    if value_txt in _RECODING_ALIASES:
        normalized = _RECODING_ALIASES[value_txt]
    else:
        normalized = value_txt.replace("-", "").replace("_", "")
    if normalized not in SUPPORTED_RECODINGS:
        txt = '--nonsyn_recode should be one of {}.'
        raise ValueError(txt.format(", ".join(SUPPORTED_RECODINGS)))
    return normalized

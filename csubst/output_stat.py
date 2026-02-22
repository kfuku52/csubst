import re


ALL_OUTPUT_STATS = (
    "any2any",
    "any2spe",
    "any2dif",
    "dif2any",
    "dif2spe",
    "dif2dif",
    "spe2any",
    "spe2spe",
    "spe2dif",
)

STAT_COLUMN_PREFIXES = (
    "OCN",
    "OCS",
    "ECN",
    "ECS",
    "dNC",
    "dSC",
    "omegaC",
    "QCN",
    "QCS",
)

DEFAULT_CUTOFF_STAT = "OCNany2spe,2.0|omegaCany2spe,5.0"

DEFAULT_OUTPUT_STATS = (
    "any2any",
    "any2dif",
    "any2spe",
)

BASE_OUTPUT_STATS = (
    "any2any",
    "spe2any",
    "any2spe",
    "spe2spe",
)

_REQUIRED_BASE_STATS_BY_OUTPUT = {
    "any2any": ("any2any",),
    "any2spe": ("any2spe",),
    "spe2any": ("spe2any",),
    "spe2spe": ("spe2spe",),
    "any2dif": ("any2any", "any2spe"),
    "dif2any": ("any2any", "spe2any"),
    "dif2spe": ("any2spe", "spe2spe"),
    "spe2dif": ("spe2any", "spe2spe"),
    "dif2dif": ("any2any", "any2spe", "spe2any", "spe2spe"),
}

_REQUIRED_DIF_STATS_BY_OUTPUT = {
    "any2dif": ("any2dif",),
    "dif2any": ("dif2any",),
    "dif2spe": ("dif2spe",),
    "spe2dif": ("spe2dif",),
    # dif2dif is derived from any2dif and spe2dif in add_dif_stats.
    "dif2dif": ("any2dif", "spe2dif", "dif2dif"),
}


def _deduplicate_preserve_order(values):
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def parse_output_stats(value, default=None):
    if default is None:
        default = DEFAULT_OUTPUT_STATS
    if value is None:
        tokens = [str(v).strip().lower() for v in default]
    elif isinstance(value, str):
        tokens = [v.strip().lower() for v in value.split(",") if v.strip() != ""]
    else:
        tokens = [str(v).strip().lower() for v in value if str(v).strip() != ""]
    if len(tokens) == 0:
        raise ValueError("--output_stat should specify at least one statistic.")
    supported = set(ALL_OUTPUT_STATS)
    invalid = sorted(set(tokens).difference(supported))
    if len(invalid):
        txt = "--output_stat contains unsupported statistics: {}. Supported statistics: {}."
        raise ValueError(txt.format(", ".join(invalid), ", ".join(ALL_OUTPUT_STATS)))
    return _deduplicate_preserve_order(tokens)


def get_required_base_stats(output_stats):
    stats = parse_output_stats(output_stats, default=DEFAULT_OUTPUT_STATS)
    required = set()
    for stat in stats:
        required.update(_REQUIRED_BASE_STATS_BY_OUTPUT[stat])
    return [s for s in BASE_OUTPUT_STATS if s in required]


def get_required_dif_stats(output_stats):
    stats = parse_output_stats(output_stats, default=DEFAULT_OUTPUT_STATS)
    required = set()
    for stat in stats:
        required.update(_REQUIRED_DIF_STATS_BY_OUTPUT.get(stat, ()))
    return [s for s in ALL_OUTPUT_STATS if s in required]


def get_default_cutoff_stat_for_output_stats(output_stats):
    stats = parse_output_stats(output_stats, default=DEFAULT_OUTPUT_STATS)
    priority = [
        "any2spe",
        "any2dif",
        "any2any",
        "spe2spe",
        "spe2any",
        "spe2dif",
        "dif2spe",
        "dif2any",
        "dif2dif",
    ]
    chosen = None
    for stat in priority:
        if stat in stats:
            chosen = stat
            break
    if chosen is None:
        chosen = stats[0]
    return "OCN{},2.0|omegaC{},5.0".format(chosen, chosen)


def validate_cutoff_stat_compatibility(cutoff_stat, output_stats):
    stats = set(parse_output_stats(output_stats, default=DEFAULT_OUTPUT_STATS))
    known_suffixes = set(ALL_OUTPUT_STATS)
    known_prefixes = ("OCN", "OCS", "ECN", "ECS", "omegaC", "dNC", "dSC", "QCN", "QCS")
    for token in _split_cutoff_stat_tokens(cutoff_stat):
        token = token.strip()
        if token == "":
            continue
        parts = token.rsplit(",", 1)
        if len(parts) != 2:
            continue
        stat_exp = parts[0].strip()
        if stat_exp == "":
            continue
        try:
            re.compile(stat_exp)
        except re.error:
            continue
        matched_suffixes = set()
        for prefix in known_prefixes:
            for suffix in known_suffixes:
                stat_col = prefix + suffix
                if re.fullmatch(stat_exp, stat_col):
                    matched_suffixes.add(suffix)
        missing_suffixes = sorted([suffix for suffix in matched_suffixes if suffix not in stats])
        if len(missing_suffixes) > 0:
            txt = '--cutoff_stat "{}" requires --output_stat to include "{}".'
            raise ValueError(txt.format(stat_exp, ",".join(missing_suffixes)))


def _split_cutoff_stat_tokens(cutoff_stat_str):
    text = str(cutoff_stat_str)
    tokens = []
    current = []
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    escaped = False
    for ch in text:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == '\\':
            current.append(ch)
            escaped = True
            continue
        if ch == '(':
            depth_paren += 1
        elif ch == ')' and depth_paren > 0:
            depth_paren -= 1
        elif ch == '[':
            depth_bracket += 1
        elif ch == ']' and depth_bracket > 0:
            depth_bracket -= 1
        elif ch == '{':
            depth_brace += 1
        elif ch == '}' and depth_brace > 0:
            depth_brace -= 1
        if (ch == '|') and (depth_paren == 0) and (depth_bracket == 0) and (depth_brace == 0):
            tokens.append(''.join(current).strip())
            current = []
            continue
        current.append(ch)
    tokens.append(''.join(current).strip())
    return tokens


def drop_unrequested_stat_columns(df, output_stats):
    requested = set(parse_output_stats(output_stats, default=DEFAULT_OUTPUT_STATS))
    drop_cols = []
    for col in df.columns:
        col_str = str(col)
        for prefix in STAT_COLUMN_PREFIXES:
            if not col_str.startswith(prefix):
                continue
            suffix = col_str[len(prefix):]
            if suffix.endswith("_nocalib"):
                suffix = suffix[:-8]
            if (suffix in ALL_OUTPUT_STATS) and (suffix not in requested):
                drop_cols.append(col)
            break
    if len(drop_cols) == 0:
        return df
    return df.drop(columns=drop_cols)

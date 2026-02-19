import pandas

import re


def _http_get_json(url, timeout=30):
    import requests

    response = requests.get(url=url, timeout=timeout, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def _normalize_feature_types(feature_types):
    if feature_types is None:
        return None
    if isinstance(feature_types, str):
        value = feature_types.strip()
        if (value == "") or (value.lower() in ["all", "*"]):
            return None
        feature_types = [ft.strip() for ft in value.split(",")]
    out = [ft for ft in feature_types if ft != ""]
    if len(out) == 0:
        return None
    return out


def _extract_accession_from_seq_name(seq_name):
    patterns = [
        r"AF-([A-Z0-9]+)-F[0-9]+",  # AlphaFold
        r"(^|[^A-Z0-9])([A-Z][0-9][A-Z0-9]{3}[0-9])([^A-Z0-9]|$)",  # common UniProt accession
    ]
    for pattern in patterns:
        match = re.search(pattern, seq_name, flags=re.IGNORECASE)
        if match is None:
            continue
        if len(match.groups()) == 1:
            return match.group(1).upper()
        elif len(match.groups()) >= 2:
            return match.group(2).upper()
    return None


def _resolve_uniprot_accession_from_rcsb(pdb_code, chain_id):
    try:
        instance_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{pdb_code}/{chain_id}"
        instance = _http_get_json(instance_url)
        ids = instance.get("rcsb_polymer_entity_instance_container_identifiers", {})
        entity_id = ids.get("entity_id")
        if entity_id is None:
            return None
        entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_code}/{entity_id}"
        entity = _http_get_json(entity_url)
    except Exception as e:
        print(f"RCSB query failed for {pdb_code}:{chain_id}: {e}", flush=True)
        return None

    entity_ids = entity.get("rcsb_polymer_entity_container_identifiers", {})
    uniprot_ids = entity_ids.get("uniprot_ids")
    if isinstance(uniprot_ids, list) and len(uniprot_ids) > 0:
        return uniprot_ids[0]
    ref_ids = entity_ids.get("reference_sequence_identifiers", [])
    for ref in ref_ids:
        if ref.get("database_name", "").lower() == "uniprot":
            accession = ref.get("database_accession")
            if accession:
                return accession
    return None


def resolve_uniprot_accession(seq_name, pdb_id=None):
    accession = _extract_accession_from_seq_name(seq_name=seq_name)
    if accession is not None:
        return accession
    if pdb_id is None:
        return None
    chain_id = seq_name.rsplit("_", 1)[-1] if "_" in seq_name else None
    if chain_id is None:
        return None
    pdb_code = seq_name.split("_", 1)[0]
    is_pdb_code = bool(re.fullmatch(r"[0-9][A-Za-z0-9]{3}", pdb_code))
    if not is_pdb_code:
        return None
    return _resolve_uniprot_accession_from_rcsb(pdb_code=pdb_code, chain_id=chain_id)


def get_uniprot_features(accession):
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        uni = _http_get_json(url=url)
    except Exception as e:
        print(f"UniProt query failed for {accession}: {e}", flush=True)
        return []
    features = []
    for feature in uni.get("features", []):
        feature_type = feature.get("type", "")
        desc = feature.get("description", "")
        location = feature.get("location", {})
        start = location.get("start", {}).get("value")
        end = location.get("end", {}).get("value")
        try:
            start = int(start)
            end = int(end)
        except Exception:
            continue
        if (start <= 0) or (end <= 0):
            continue
        if end < start:
            continue
        features.append(
            {
                "type": feature_type,
                "description": desc,
                "start": start,
                "end": end,
            }
        )
    return features


def _filter_features(features, feature_types):
    feature_types = _normalize_feature_types(feature_types=feature_types)
    if feature_types is None:
        return features
    allowed = set([ft.lower() for ft in feature_types])
    return [f for f in features if f["type"].lower() in allowed]


def _get_mapped_sites(df, col_site):
    mapped_sites = set()
    for value in df.loc[:, col_site].tolist():
        if pandas.isna(value):
            continue
        site = int(value)
        if site <= 0:
            continue
        mapped_sites.add(site)
    return mapped_sites


def _filter_redundant_features(features, mapped_sites):
    if len(mapped_sites) == 0:
        return features
    non_redundant = []
    for feature in features:
        covers_all = True
        for site in mapped_sites:
            if (site < feature["start"]) or (site > feature["end"]):
                covers_all = False
                break
        if not covers_all:
            non_redundant.append(feature)
    return non_redundant


def _is_constant_column(series):
    if series.shape[0] == 0:
        return True
    return series.fillna("").astype(str).nunique(dropna=False) == 1


def add_uniprot_site_annotations(df, g):
    seq_cols = [c for c in df.columns if c.startswith("codon_site_pdb_")]
    if len(seq_cols) == 0:
        return df
    if "_uniprot_feature_cache" not in g:
        g["_uniprot_feature_cache"] = {}
    feature_types = g.get("uniprot_feature_types", None)
    include_redundant = bool(g.get("uniprot_include_redundant", False))
    for seq_col in seq_cols:
        seq_name = seq_col.replace("codon_site_pdb_", "")
        col_site = "codon_site_" + seq_name
        if col_site not in df.columns:
            continue
        accession = resolve_uniprot_accession(seq_name=seq_name, pdb_id=g.get("pdb"))
        if accession is None:
            print(f"UniProt accession could not be resolved for sequence {seq_name}.", flush=True)
            continue
        if accession in g["_uniprot_feature_cache"]:
            features = g["_uniprot_feature_cache"][accession]
        else:
            features = get_uniprot_features(accession=accession)
            g["_uniprot_feature_cache"][accession] = features
        features = _filter_features(features=features, feature_types=feature_types)
        if not include_redundant:
            mapped_sites = _get_mapped_sites(df=df, col_site=col_site)
            before_filter = len(features)
            features = _filter_redundant_features(features=features, mapped_sites=mapped_sites)
            removed = before_filter - len(features)
            if removed > 0:
                print(
                    f'Removed {removed} redundant UniProt feature(s) for sequence {seq_name}.',
                    flush=True,
                )
        feature_map = {}
        for feature in features:
            for pos in range(feature["start"], feature["end"] + 1):
                if pos not in feature_map:
                    feature_map[pos] = []
                feature_map[pos].append(feature)

        col_acc = "uniprot_acc_" + seq_name
        col_count = "uniprot_feature_count_" + seq_name
        col_type = "uniprot_feature_types_" + seq_name
        col_desc = "uniprot_feature_descriptions_" + seq_name
        df.loc[:, col_acc] = accession
        df.loc[:, col_count] = 0
        df.loc[:, col_type] = ""
        df.loc[:, col_desc] = ""

        for i in df.index:
            site = df.at[i, col_site]
            if pandas.isna(site):
                continue
            site = int(site)
            if site <= 0:
                continue
            site_features = feature_map.get(site, [])
            if len(site_features) == 0:
                continue
            types = sorted(set([f["type"] for f in site_features]))
            labels = []
            for feature in site_features:
                label = feature["type"]
                if feature["description"] != "":
                    label += ": " + feature["description"]
                if feature["start"] == feature["end"]:
                    label += " ({})".format(feature["start"])
                else:
                    label += " ({}-{})".format(feature["start"], feature["end"])
                if label not in labels:
                    labels.append(label)
            df.at[i, col_count] = len(site_features)
            df.at[i, col_type] = "|".join(types)
            df.at[i, col_desc] = "|".join(labels)
        df.loc[:, col_count] = df[col_count].astype(int)
        if not include_redundant:
            output_cols = [col_acc, col_count, col_type, col_desc]
            redundant_cols = [c for c in output_cols if _is_constant_column(df[c])]
            if col_type not in redundant_cols:
                redundant_cols.append(col_type)
            if len(redundant_cols) > 0:
                df = df.drop(columns=redundant_cols)
                print(
                    f'Excluded redundant UniProt column(s) for sequence {seq_name}: '
                    + ", ".join(redundant_cols),
                    flush=True,
                )
        txt = 'Added UniProt site annotations from accession {} to columns with suffix "{}".'
        print(txt.format(accession, seq_name), flush=True)
    return df

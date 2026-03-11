import datetime
import os
import re

import numpy as np
import pandas as pd


def _normalize_branch_ids(branch_ids):
    if branch_ids is None:
        return np.array([], dtype=np.int64)
    values = np.asarray(branch_ids, dtype=object)
    if values.ndim == 0:
        scalar = values.item()
        if isinstance(scalar, (list, tuple, set, np.ndarray)):
            values = np.asarray(list(scalar), dtype=object)
        else:
            values = np.asarray([scalar], dtype=object)
    flat_values = np.atleast_1d(values).reshape(-1)
    if flat_values.size == 0:
        return np.array([], dtype=np.int64)
    normalized = list()
    for value in flat_values.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('branch_ids should be integer-like.')
        if isinstance(value, (int, np.integer)):
            normalized.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            if (not np.isfinite(value)) or (not float(value).is_integer()):
                raise ValueError('branch_ids should be integer-like.')
            normalized.append(int(value))
            continue
        value_txt = str(value).strip()
        if value_txt == '':
            raise ValueError('branch_ids should be integer-like.')
        if not bool(re.fullmatch(r'[+-]?[0-9]+(?:\.0+)?', value_txt)):
            raise ValueError('branch_ids should be integer-like.')
        normalized.append(int(float(value_txt)))
    return np.asarray(normalized, dtype=np.int64)


def add_output_manifest_row(
    manifest_rows,
    output_path,
    output_kind,
    note='',
    base_dir=None,
    branch_ids=None,
    extra_fields=None,
):
    output_path_abs = os.path.abspath(str(output_path))
    exists = os.path.exists(output_path_abs)
    size_bytes = os.path.getsize(output_path_abs) if exists else -1
    if base_dir is None:
        output_file = output_path_abs
    else:
        base_dir_abs = os.path.abspath(str(base_dir))
        if output_path_abs.startswith(base_dir_abs + os.sep):
            output_file = os.path.relpath(output_path_abs, start=base_dir_abs)
        elif output_path_abs == base_dir_abs:
            output_file = '.'
        else:
            output_file = output_path_abs
    row = {
        'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'output_kind': str(output_kind),
        'output_file': str(output_file),
        'output_path': output_path_abs,
        'file_exists': 'Y' if exists else 'N',
        'file_size_bytes': int(size_bytes),
        'note': str(note),
    }
    if branch_ids is not None:
        normalized_branch_ids = _normalize_branch_ids(branch_ids)
        row['branch_ids'] = ','.join([str(int(bid)) for bid in normalized_branch_ids.tolist()])
        row['branch_count'] = int(normalized_branch_ids.shape[0])
    if extra_fields is not None:
        row.update(dict(extra_fields))
    manifest_rows.append(row)
    return manifest_rows


def write_output_manifest(
    manifest_rows,
    manifest_path,
    note='manifest_self_row',
    base_dir=None,
    branch_ids=None,
    extra_fields=None,
    sort_by=None,
):
    manifest_path_abs = os.path.abspath(str(manifest_path))
    parent_dir = os.path.dirname(manifest_path_abs)
    if parent_dir != '':
        os.makedirs(parent_dir, exist_ok=True)
    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_df.shape[0] > 0:
        sort_columns = ['output_kind', 'output_file'] if sort_by is None else list(sort_by)
        sort_columns = [col for col in sort_columns if col in manifest_df.columns]
        if len(sort_columns) > 0:
            manifest_df = manifest_df.sort_values(by=sort_columns).reset_index(drop=True)
    manifest_df.to_csv(manifest_path_abs, sep='\t', index=False, chunksize=10000)
    add_output_manifest_row(
        manifest_rows=manifest_rows,
        output_path=manifest_path_abs,
        output_kind='output_manifest',
        note=note,
        base_dir=base_dir,
        branch_ids=branch_ids,
        extra_fields=extra_fields,
    )
    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_df.shape[0] > 0:
        sort_columns = ['output_kind', 'output_file'] if sort_by is None else list(sort_by)
        sort_columns = [col for col in sort_columns if col in manifest_df.columns]
        if len(sort_columns) > 0:
            manifest_df = manifest_df.sort_values(by=sort_columns).reset_index(drop=True)
    manifest_df.to_csv(manifest_path_abs, sep='\t', index=False, chunksize=10000)
    return manifest_path_abs

import numpy as np
import pandas as pd

import re
import sys
import time

from scipy.stats import chi2_contingency


def _normalize_integer_like(values, column_name):
    arr = np.asarray(values, dtype=object).reshape(-1)
    normalized = []
    for value in arr.tolist():
        if pd.isna(value):
            raise ValueError('Column "{}" contains missing values.'.format(column_name))
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('Column "{}" should be integer-like.'.format(column_name))
        if isinstance(value, (int, np.integer)):
            normalized.append(int(value))
            continue
        if isinstance(value, (float, np.floating)):
            if (not np.isfinite(value)) or (not float(value).is_integer()):
                raise ValueError('Column "{}" should be integer-like.'.format(column_name))
            normalized.append(int(value))
            continue
        value_txt = str(value).strip()
        if value_txt == '':
            raise ValueError('Column "{}" contains blank values.'.format(column_name))
        if not bool(re.fullmatch(r'[+-]?[0-9]+(?:\.0+)?', value_txt)):
            raise ValueError('Column "{}" should be integer-like.'.format(column_name))
        normalized.append(int(float(value_txt)))
    return np.array(normalized, dtype=np.int64)


def sort_branch_ids(df):
    swap_columns = df.columns[df.columns.str.startswith('branch_id')].tolist()
    if len(swap_columns)>1:
        swap_values = df.loc[:,swap_columns].to_numpy(copy=True)
        swap_values.sort(axis=1)
        df.loc[:,swap_columns] = swap_values
    if 'site' in df.columns:
        swap_columns.append('site')
    if len(swap_columns) == 0:
        return df
    df = df.sort_values(by=swap_columns)
    for cn in swap_columns:
        df[cn] = _normalize_integer_like(values=df.loc[:, cn].to_numpy(copy=False), column_name=cn)
    return df

def sort_cb(cb):
    start = time.time()
    is_omega = cb.columns.str.contains('^omegaC')
    is_d = cb.columns.str.contains('^d[NS]C')
    is_nocalib = cb.columns.str.contains('_nocalib$')
    num_branch_id_cols = cb.columns.str.contains('^branch_id_').sum()
    col_order = []
    col_order += [ 'branch_id_'+str(i+1) for i in np.arange(num_branch_id_cols) ] # https://github.com/kfuku52/csubst/issues/20
    col_order += cb.columns[cb.columns.str.contains('^dist_')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^branch_num_')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^is_')].sort_values().tolist()
    col_order += cb.columns[(is_omega)&(~is_nocalib)].sort_values().tolist()
    col_order += cb.columns[(is_d)&(~is_nocalib)].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^OC[NS]CoD$')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^OC[NS]_linreg_residual$')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^[NS]_sub_')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains(r'^OC[NS](?:any|dif|spe)')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains(r'^EC[NS](?:any|dif|spe)')].sort_values().tolist()
    col_order += cb.columns[(is_omega)&(is_nocalib)].sort_values().tolist()
    col_order += cb.columns[(is_d)&(is_nocalib)].sort_values().tolist()
    if (len(col_order) < cb.columns.shape[0]):
        col_order += [ col for col in cb.columns if col not in col_order ]
    cb = cb.loc[:,col_order]
    print('Time elapsed for sorting cb table: {:,} sec'.format(int(time.time() - start)))
    return cb

def sort_cb_stats(cb_stats):
    col_order = ['arity', 'elapsed_sec', 'cutoff_stat', 'fg_enrichment_factor', 'mode', 'dSC_calibration', ]
    if cb_stats is None:
        return pd.DataFrame(columns=col_order)
    if cb_stats.shape[1] == 0:
        return pd.DataFrame(columns=col_order)
    str_columns = pd.Index([str(c) for c in cb_stats.columns])
    col_order += cb_stats.columns[str_columns.str.contains('^num_')].tolist()
    col_order = [col for col in col_order if col in cb_stats.columns]
    if (len(col_order) < cb_stats.columns.shape[0]):
        col_order += [ col for col in cb_stats.columns if col not in col_order ]
    cb_stats = cb_stats.loc[:,col_order]
    return cb_stats

def merge_tables(df1, df2):
    start = time.time()
    columns = []
    columns = columns + df1.columns[df1.columns.str.startswith('branch_name')].tolist()
    columns = columns + df1.columns[df1.columns.str.startswith('branch_id')].tolist()
    columns = columns + df1.columns[df1.columns.str.startswith('site')].tolist()
    df = pd.merge(df1, df2, on=columns)
    df = sort_branch_ids(df=df)
    print('Time elapsed for merging tables: {:,} sec'.format(int(time.time() - start)))
    return df

def set_substitution_dtype(df):
    col_exts = ['_sub', '2any', '2spe']
    sub_cols = list()
    for ck in col_exts:
        sub_cols = sub_cols + df.columns[df.columns.str.endswith(ck)].tolist()
    for sc in sub_cols:
        if (df[sc]%1).sum()==0:
            df[sc] = df[sc].astype(int)
    return df

def get_linear_regression(cb):
    start = time.time()
    for prefix in ['OCS','OCN']:
        col_x = prefix + 'any2any'
        col_y = prefix + 'any2spe'
        if not all([col in cb.columns for col in [col_x, col_y]]):
            continue
        x = cb.loc[:,col_x].values
        y = cb.loc[:,col_y].values
        x = x[:,np.newaxis]
        coef,residuals,rank,s = np.linalg.lstsq(x, y, rcond=None)
        cb.loc[:,prefix+'_linreg_residual'] = y - (x[:,0]*coef[0])
    print('Time elapsed for the linear regression of C ~ D: {:,} sec'.format(int(time.time() - start)))
    return cb

def chisq_test(x, total_S, total_N):
    obs = x.loc[['OCSany2spe','OCNany2spe']].values
    if obs.sum()==0:
        return 1
    else:
        contingency_table = np.array([obs, [total_S, total_N]])
        out = chi2_contingency(contingency_table, lambda_="log-likelihood")
        return out[1]

def get_cutoff_stat_bool_array(cb, cutoff_stat_str):
    cutoff_stat_entries = parse_cutoff_stat(cutoff_stat_str=cutoff_stat_str)
    is_enough_stat = True
    for cutoff_stat_exp,cutoff_stat_value in cutoff_stat_entries:
        is_col = cb.columns.str.fullmatch(cutoff_stat_exp, na=False)
        if is_col.sum()==0:
            txt = 'The column "{}" was not found in the cb table. '
            txt += 'Check the format of the --cutoff_stat specification ("{}") carefully.'
            raise ValueError(txt.format(cutoff_stat_exp, cutoff_stat_str))
        cutoff_stat_cols = cb.columns[is_col]
        for cutoff_stat_col in cutoff_stat_cols:
            is_enough_stat &= (cb.loc[:,cutoff_stat_col] >= cutoff_stat_value).fillna(False)
    return is_enough_stat


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


def parse_cutoff_stat(cutoff_stat_str):
    cutoff_stat_entries = []
    cutoff_stat_list = [s.replace('\'', '').replace('\"', '').strip() for s in _split_cutoff_stat_tokens(cutoff_stat_str)]
    for cutoff_stat in cutoff_stat_list:
        if cutoff_stat == '':
            continue
        cutoff_stat_list2 = cutoff_stat.rsplit(',', 1)
        if len(cutoff_stat_list2) != 2:
            txt = 'Invalid --cutoff_stat token "{}". Expected "COLUMN_OR_REGEX,VALUE".'
            raise ValueError(txt.format(cutoff_stat))
        cutoff_stat_exp = cutoff_stat_list2[0].strip()
        cutoff_stat_value_txt = cutoff_stat_list2[1].strip()
        if (cutoff_stat_exp == '') or (cutoff_stat_value_txt == ''):
            txt = 'Invalid --cutoff_stat token "{}". Empty column/regex or value is not allowed.'
            raise ValueError(txt.format(cutoff_stat))
        try:
            re.compile(cutoff_stat_exp)
        except re.error:
            txt = 'Invalid cutoff regex "{}" in token "{}".'
            raise ValueError(txt.format(cutoff_stat_exp, cutoff_stat))
        try:
            cutoff_stat_value = float(cutoff_stat_value_txt)
        except ValueError:
            txt = 'Invalid cutoff value "{}" in token "{}".'
            raise ValueError(txt.format(cutoff_stat_value_txt, cutoff_stat))
        if not np.isfinite(cutoff_stat_value):
            txt = 'Invalid cutoff value "{}" in token "{}". A finite numeric value is required.'
            raise ValueError(txt.format(cutoff_stat_value_txt, cutoff_stat))
        cutoff_stat_entries.append((cutoff_stat_exp, cutoff_stat_value))
    if len(cutoff_stat_entries) == 0:
        txt = 'No valid --cutoff_stat token was found in "{}".'
        raise ValueError(txt.format(cutoff_stat_str))
    return cutoff_stat_entries

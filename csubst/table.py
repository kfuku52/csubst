import numpy
import pandas

import re
import sys
import time

def sort_branch_ids(df):
    swap_columns = df.columns[df.columns.str.startswith('branch_id')].tolist()
    if len(swap_columns)>1:
        swap_values = df.loc[:,swap_columns].to_numpy(copy=True)
        swap_values.sort(axis=1)
        df.loc[:,swap_columns] = swap_values
    if 'site' in df.columns:
        swap_columns.append('site')
    df = df.sort_values(by=swap_columns)
    for cn in swap_columns:
        df[cn] = df[cn].astype(int)
    return df

def sort_cb(cb):
    start = time.time()
    is_omega = cb.columns.str.contains('^omegaC')
    is_d = cb.columns.str.contains('^d[NS]C')
    is_nocalib = cb.columns.str.contains('_nocalib$')
    num_branch_id_cols = cb.columns.str.contains('^branch_id_').sum()
    col_order = []
    col_order += [ 'branch_id_'+str(i+1) for i in numpy.arange(num_branch_id_cols) ] # https://github.com/kfuku52/csubst/issues/20
    col_order += cb.columns[cb.columns.str.contains('^dist_')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^branch_num_')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^is_')].sort_values().tolist()
    col_order += cb.columns[(is_omega)&(~is_nocalib)].sort_values().tolist()
    col_order += cb.columns[(is_d)&(~is_nocalib)].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^OC[NS]CoD$')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^OC[NS]_linreg_residual$')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^[NS]_sub_')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^OC[NS][any|dif|spe]')].sort_values().tolist()
    col_order += cb.columns[cb.columns.str.contains('^EC[NS][any|dif|spe]')].sort_values().tolist()
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
        return pandas.DataFrame(columns=col_order)
    if cb_stats.shape[1] == 0:
        return pandas.DataFrame(columns=col_order)
    str_columns = pandas.Index([str(c) for c in cb_stats.columns])
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
    df = pandas.merge(df1, df2, on=columns)
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
        x = cb.loc[:,prefix+'any2any'].values
        y = cb.loc[:,prefix+'any2spe'].values
        x = x[:,numpy.newaxis]
        coef,residuals,rank,s = numpy.linalg.lstsq(x, y, rcond=None)
        cb.loc[:,prefix+'_linreg_residual'] = y - (x[:,0]*coef[0])
    print('Time elapsed for the linear regression of C ~ D: {:,} sec'.format(int(time.time() - start)))
    return cb

def chisq_test(x, total_S, total_N):
    obs = x.loc[['OCSany2spe','OCNany2spe']].values
    if obs.sum()==0:
        return 1
    else:
        contingency_table = numpy.array([obs, [total_S, total_N]])
        out = chi2_contingency(contingency_table, lambda_="log-likelihood")
        return out[1]

def get_cutoff_stat_bool_array(cb, cutoff_stat_str):
    cutoff_stat_entries = parse_cutoff_stat(cutoff_stat_str=cutoff_stat_str)
    is_enough_stat = True
    for cutoff_stat_exp,cutoff_stat_value in cutoff_stat_entries:
        is_col = cb.columns.str.fullmatch(cutoff_stat_exp, na=False)
        if is_col.sum()==0:
            txt = 'The column "{}" was not found in the cb table. '
            txt += 'Check the format of the --cutoff_stat specification ("{}") carefully. Exiting.\n'
            sys.stderr.write(txt.format(cutoff_stat_exp, cutoff_stat_str))
            sys.exit(1)
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
            txt = 'Invalid --cutoff_stat token "{}". Expected "COLUMN_OR_REGEX,VALUE". Exiting.\n'
            sys.stderr.write(txt.format(cutoff_stat))
            sys.exit(1)
        cutoff_stat_exp = cutoff_stat_list2[0].strip()
        cutoff_stat_value_txt = cutoff_stat_list2[1].strip()
        if (cutoff_stat_exp == '') or (cutoff_stat_value_txt == ''):
            txt = 'Invalid --cutoff_stat token "{}". Empty column/regex or value is not allowed. Exiting.\n'
            sys.stderr.write(txt.format(cutoff_stat))
            sys.exit(1)
        try:
            re.compile(cutoff_stat_exp)
        except re.error:
            txt = 'Invalid cutoff regex "{}" in token "{}". Exiting.\n'
            sys.stderr.write(txt.format(cutoff_stat_exp, cutoff_stat))
            sys.exit(1)
        try:
            cutoff_stat_value = float(cutoff_stat_value_txt)
        except ValueError:
            txt = 'Invalid cutoff value "{}" in token "{}". Exiting.\n'
            sys.stderr.write(txt.format(cutoff_stat_value_txt, cutoff_stat))
            sys.exit(1)
        cutoff_stat_entries.append((cutoff_stat_exp, cutoff_stat_value))
    if len(cutoff_stat_entries) == 0:
        txt = 'No valid --cutoff_stat token was found in "{}". Exiting.\n'
        sys.stderr.write(txt.format(cutoff_stat_str))
        sys.exit(1)
    return cutoff_stat_entries

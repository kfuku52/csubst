import pandas

def sort_labels(df):
    swap_columns = df.columns[df.columns.str.startswith('branch_id')].tolist()
    if len(swap_columns)>1:
        swap_values = df.loc[:,swap_columns].values
        swap_values.sort(axis=1)
        df.loc[:,swap_columns] = swap_values
    if 'site' in df.columns:
        swap_columns.append('site')
    df = df.sort_values(by=swap_columns)
    for cn in swap_columns:
        df.loc[:,cn] = df[cn].astype(int)
    return df

def merge_tables(df1, df2):
    columns = []
    columns = columns + df1.columns[df1.columns.str.startswith('branch_name')].tolist()
    columns = columns + df1.columns[df1.columns.str.startswith('branch_id')].tolist()
    columns = columns + df1.columns[df1.columns.str.startswith('site')].tolist()
    df = pandas.merge(df1, df2, on=columns)
    df = sort_labels(df=df)
    return df

def set_substitution_dtype(df):
    col_exts = ['_sub', '2any', '2spe']
    sub_cols = list()
    for ck in col_exts:
        sub_cols = sub_cols + df.columns[df.columns.str.endswith(ck)].tolist()
    for sc in sub_cols:
        if (df[sc]%1).sum()==0:
            df.loc[:,sc] = df[sc].astype(int)
    return df
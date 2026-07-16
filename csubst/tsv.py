"""Bounded-memory TSV output with vectorized numeric formatting."""

import csv

import numpy as np
import pandas as pd


def _format_column(series, float_format):
    if pd.api.types.is_float_dtype(series.dtype):
        values = series.to_numpy(dtype=np.float64, copy=False)
        if float_format is None:
            formatted = values.astype(str)
        else:
            formatted = np.char.mod(float_format, values)
        missing = pd.isna(values)
        if missing.any():
            formatted = np.asarray(formatted, dtype=object)
            formatted[missing] = ''
        return formatted
    if pd.api.types.is_integer_dtype(series.dtype) and not pd.api.types.is_extension_array_dtype(series.dtype):
        return series.to_numpy(copy=False).astype(str)
    if pd.api.types.is_bool_dtype(series.dtype) and not pd.api.types.is_extension_array_dtype(series.dtype):
        return series.to_numpy(copy=False).astype(str)

    values = series.to_numpy(dtype=object, copy=False)
    return ['' if pd.isna(value) else str(value) for value in values]


def write_dataframe(
    dataframe,
    output_path,
    *,
    float_format=None,
    chunksize=10000,
    header=True,
    mode='w',
):
    """Write a DataFrame as pandas-compatible UTF-8 TSV.

    Numeric columns are formatted a column at a time and rows are emitted by
    the C-backed csv writer.  Chunking bounds temporary string storage.
    """
    if chunksize is None:
        chunksize = max(1, int(dataframe.shape[0]))
    chunksize = max(1, int(chunksize))
    with open(output_path, mode=mode, encoding='utf-8', newline='') as handle:
        writer = csv.writer(
            handle,
            delimiter='\t',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            lineterminator='\n',
        )
        if header:
            writer.writerow([str(column) for column in dataframe.columns])
        for start in range(0, dataframe.shape[0], chunksize):
            chunk = dataframe.iloc[start:start + chunksize, :]
            columns = [
                _format_column(chunk.iloc[:, column_index], float_format)
                for column_index in range(chunk.shape[1])
            ]
            writer.writerows(zip(*columns))
    return None

"""Bounded-memory TSV output using pandas' compiled CSV writer."""


def write_dataframe(
    dataframe,
    output_path,
    *,
    float_format=None,
    chunksize=10000,
    header=True,
    mode='w',
):
    """Write a DataFrame as a pandas-compatible, bounded-memory UTF-8 TSV."""
    if chunksize is None:
        chunksize = max(1, int(dataframe.shape[0]))
    chunksize = max(1, int(chunksize))
    if dataframe.shape[1] == 0:
        # Preserve the historical csv.writer behavior: an empty header is one
        # newline and zero-column data rows emit nothing.
        with open(output_path, mode=mode, encoding='utf-8', newline='') as handle:
            if header:
                handle.write('\n')
        return None
    dataframe.to_csv(
        output_path,
        sep='\t',
        index=False,
        float_format=float_format,
        lineterminator='\n',
        chunksize=chunksize,
        header=header,
        mode=mode,
        encoding='utf-8',
    )
    return None

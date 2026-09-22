"""Bounded-memory TSV output using pandas' compiled CSV writer."""


def write_dataframe(
    dataframe,
    output_path,
    *,
    float_format=None,
    chunksize=10000,
    header=True,
    mode='w',
    report_context=None,
):
    """Write a DataFrame as a pandas-compatible, bounded-memory UTF-8 TSV."""
    if report_context is not None:
        from csubst import event_reporting
        dataframe = event_reporting.annotate(dataframe, report_context)
    from csubst import output_safety
    output_safety.validate_destination(output_path)
    if chunksize is None:
        chunksize = max(1, int(dataframe.shape[0]))
    chunksize = max(1, int(chunksize))
    if dataframe.shape[1] == 0:
        # Preserve the historical csv.writer behavior: an empty header is one
        # newline and zero-column data rows emit nothing. Keep the shared
        # pandas writer so compression and caller-owned streams still work.
        dataframe = dataframe.iloc[:0]
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
        na_rep='NA' if report_context is not None else '',
    )
    return None

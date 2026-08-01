import numpy as np
import pandas as pd

from csubst import tsv


def test_write_dataframe_matches_pandas_for_analysis_types(tmp_path):
    frame = pd.DataFrame(
        {
            "branch_id": np.array([1, 2, 3], dtype=np.int64),
            "score": np.array([1.23456, np.nan, np.inf], dtype=np.float64),
            "flag": [True, False, True],
            "label": ["plain", "tab\tvalue", None],
        }
    )
    expected_path = tmp_path / "pandas.tsv"
    observed_path = tmp_path / "csubst.tsv"
    frame.to_csv(expected_path, sep="\t", index=False, float_format="%.4f", lineterminator="\n")

    tsv.write_dataframe(frame, observed_path, float_format="%.4f", chunksize=2)

    assert observed_path.read_bytes() == expected_path.read_bytes()


def test_write_dataframe_supports_append_without_duplicate_header(tmp_path):
    frame = pd.DataFrame({"x": [1.0, 2.0], "name": ["a", "b"]})
    output_path = tmp_path / "chunks.tsv"
    tsv.write_dataframe(frame.iloc[:1], output_path, float_format="%.2f")
    tsv.write_dataframe(
        frame.iloc[1:],
        output_path,
        float_format="%.2f",
        header=False,
        mode="a",
    )
    assert output_path.read_text(encoding="utf-8") == "x\tname\n1.00\ta\n2.00\tb\n"

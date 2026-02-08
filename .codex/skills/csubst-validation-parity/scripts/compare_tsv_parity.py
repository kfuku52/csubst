#!/usr/bin/env python3
import argparse
import re
import sys

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two TSV files for parity with optional ignored columns."
    )
    parser.add_argument("left", help="Left TSV path")
    parser.add_argument("right", help="Right TSV path")
    parser.add_argument(
        "--ignore-col-regex",
        default=None,
        help="Regex for column names to ignore (example: '^elapsed_sec$').",
    )
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for numeric columns.")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for numeric columns.")
    return parser.parse_args()


def load_tsv(path):
    return pd.read_csv(path, sep="\t")


def select_columns(df_left, df_right, ignore_pattern):
    common = [c for c in df_left.columns if c in df_right.columns]
    ignored = []
    if ignore_pattern is not None:
        rgx = re.compile(ignore_pattern)
        ignored = [c for c in common if rgx.search(c)]
    keep = [c for c in common if c not in ignored]
    return keep, ignored


def compare_columns(df_left, df_right, columns, rtol, atol):
    differences = []
    for col in columns:
        s_left = df_left[col]
        s_right = df_right[col]
        if s_left.shape[0] != s_right.shape[0]:
            differences.append((col, "row_count_diff"))
            continue
        if np.issubdtype(s_left.dtype, np.number) and np.issubdtype(s_right.dtype, np.number):
            left = s_left.to_numpy(dtype=float)
            right = s_right.to_numpy(dtype=float)
            equal = np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
            if not np.all(equal):
                diff = np.abs(left - right)
                diff[~np.isfinite(diff)] = 0.0
                max_diff = float(np.max(diff))
                differences.append((col, f"numeric_diff:max_abs={max_diff}"))
        else:
            left = s_left.fillna("__NA__").astype(str).to_numpy()
            right = s_right.fillna("__NA__").astype(str).to_numpy()
            if not np.array_equal(left, right):
                differences.append((col, "categorical_diff"))
    return differences


def main():
    args = parse_args()
    left_df = load_tsv(args.left)
    right_df = load_tsv(args.right)

    columns, ignored = select_columns(left_df, right_df, args.ignore_col_regex)
    if len(columns) == 0:
        print("No common comparable columns found.")
        return 1

    differences = compare_columns(
        df_left=left_df,
        df_right=right_df,
        columns=columns,
        rtol=args.rtol,
        atol=args.atol,
    )

    print(f"left_rows={left_df.shape[0]} right_rows={right_df.shape[0]}")
    print(f"compared_columns={len(columns)} ignored_columns={len(ignored)}")
    if ignored:
        print("ignored:", ",".join(ignored))

    if differences:
        print("DIFF")
        for col, reason in differences:
            print(f"{col}\t{reason}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

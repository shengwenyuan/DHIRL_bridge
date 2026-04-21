#!/usr/bin/env python3
"""Report mean ± std of train_ll and test_ll for one or more CSV files."""

import sys
import argparse
import numpy as np
import pandas as pd


def summarize(path: str) -> None:
    df = pd.read_csv(path)
    for col in ("train_ll", "test_ll"):
        if col not in df.columns:
            print(f"  [{col}] column not found")
            continue
        vals = df[col].dropna().values
        print(f"  {col}: mean={np.mean(vals):.6f}  std={np.std(vals, ddof=1):.6f}  (n={len(vals)})")


def main():
    parser = argparse.ArgumentParser(description="Mean ± std of train/test LL from CSV files.")
    parser.add_argument("csvfiles", nargs="+", metavar="CSV", help="Path(s) to CSV file(s)")
    args = parser.parse_args()

    for path in args.csvfiles:
        print(path)
        try:
            summarize(path)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()

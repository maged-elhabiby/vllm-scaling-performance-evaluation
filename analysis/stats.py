"""
Statistical analysis utilities
Computes per-cell aggregates (mean ± CI) across the 3 iterations, detects outliers,
and produces a tidy DataFrame ready for visualisation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as st


def load_results(results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    records = []
    for path in sorted(results_dir.glob("pl*_c*_i*.json")):
        with open(path) as f:
            d = json.load(f)
        # raw_results holds per-request records and can be several MB per file — strip it
        # before building the DataFrame to avoid loading gigabytes for large result sets.
        d.pop("raw_results", None)
        records.append(d)

    if not records:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    df = pd.DataFrame(records)
    df = df.sort_values(["prompt_len", "concurrency", "iteration"]).reset_index(drop=True)
    return df


def mean_ci(values: list[float] | np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    # t-distribution CI because n=3 is too small for z - the t interval is wider and more honest
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]  # TTFT can be NaN when no tokens came back
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(arr))
    if n < 2:
        # can't estimate variance with one sample, just return the point estimate
        return m, m, m
    se = st.sem(arr)
    t = st.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t * se
    return m, m - margin, m + margin


METRIC_COLS = [
    "mean_latency", "p95_latency", "p99_latency",
    "mean_ttft", "p95_ttft", "p99_ttft",
    "requests_per_second", "tokens_per_second",
    "error_rate", "timeout_rate",
]


def aggregate_iterations(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    # Collapses the 3 iterations per (prompt_len, concurrency) cell into mean ± CI columns.
    # Output schema: prompt_len, concurrency, <metric>_mean, <metric>_lo, <metric>_hi.
    # The _lo and _hi bounds are consumed by visualize.py to shade confidence bands on
    # line plots. A wide band signals that the 3 iterations disagreed, so those cells
    # should be interpreted cautiously rather than treated as stable point estimates.
    rows = []
    for (pl, cl), grp in df.groupby(["prompt_len", "concurrency"]):
        row: dict = {"prompt_len": pl, "concurrency": cl, "n_iterations": len(grp)}
        for col in METRIC_COLS:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna().tolist()
            m, lo, hi = mean_ci(vals, confidence)
            row[f"{col}_mean"] = m
            row[f"{col}_lo"] = lo
            row[f"{col}_hi"] = hi
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["prompt_len", "concurrency"]).reset_index(drop=True)


def flag_outliers(df: pd.DataFrame, metric: str = "mean_latency", z_thresh: float = 3.0) -> pd.DataFrame:
    # modified Z-score (Iglewicz & Hoaglin) - uses median + MAD instead of mean + std
    # so the outliers themselves don't inflate the detection threshold
    # 0.6745 = 1 / (75th percentile of standard normal) - scales MAD to match std
    df = df.copy()
    df["is_outlier"] = False

    for (pl, cl), grp in df.groupby(["prompt_len", "concurrency"]):
        vals = grp[metric].values
        if len(vals) < 3:
            # can't reliably flag outliers with fewer than 3 points
            continue
        median = np.median(vals)
        mad = np.median(np.abs(vals - median))
        if mad == 0:
            # all values identical - nothing to flag
            continue
        modified_z = 0.6745 * (vals - median) / mad
        outlier_mask = np.abs(modified_z) > z_thresh
        df.loc[grp.index[outlier_mask], "is_outlier"] = True

    return df


def outlier_report(df: pd.DataFrame, metric: str = "mean_latency") -> pd.DataFrame:
    flagged = flag_outliers(df, metric)
    return flagged[flagged["is_outlier"]][["prompt_len", "concurrency", "iteration", metric]]


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Statistical analysis of benchmark results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/summary.csv",
                        help="Path for the aggregated summary CSV")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...")
    df = load_results(args.results_dir)
    print(f"  {len(df)} run records loaded.")

    outliers = outlier_report(df)
    if not outliers.empty:
        print(f"\n[OUTLIERS] {len(outliers)} outlier runs detected:")
        print(outliers.to_string(index=False))

    summary = aggregate_iterations(df, confidence=args.confidence)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nAggregated summary saved to {out_path}")
    print(summary[["prompt_len", "concurrency",
                    "mean_latency_mean", "p99_latency_mean",
                    "requests_per_second_mean", "tokens_per_second_mean",
                    "error_rate_mean"]].to_string(index=False))

"""
monitoring/drift.py

Simple statistical drift detection: compares a reference distribution
(the training data) against a recent production sample using the
population stability index (PSI) — a common, dependency-light drift
metric. PSI > 0.2 is generally considered significant drift.
"""

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    breakpoints = np.linspace(0, 100, buckets + 1)
    bucket_edges = np.percentile(reference, breakpoints)
    bucket_edges[0] -= 1e-6
    bucket_edges[-1] += 1e-6

    ref_counts, _ = np.histogram(reference, bins=bucket_edges)
    cur_counts, _ = np.histogram(current, bins=bucket_edges)

    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / len(current))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_missing_values(df) -> dict:
    return df.isnull().mean().round(4).to_dict()


def drift_report(reference_df, current_df, columns=("latitude", "longitude", "depth")) -> dict:
    report = {}
    for col in columns:
        ref_col = reference_df.get(col.title(), reference_df.get(col))
        cur_col = current_df.get(col.title(), current_df.get(col))
        if ref_col is None or cur_col is None:
            continue
        score = psi(ref_col.dropna().values, cur_col.dropna().values)
        report[col] = {
            "psi": round(score, 4),
            "drift_detected": score > 0.2,
        }
    return report

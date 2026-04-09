"""Statistical summaries and paired significance tests for D1 reporting."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def summary_with_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    """Summarise values with mean/std/min/max and a t-based confidence interval."""
    arr = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if arr.size == 0:
        raise ValueError("summary_with_ci requires at least one numeric value.")

    mean = float(arr.mean())
    std = float(arr.std())
    if arr.size == 1:
        ci_low = ci_high = mean
    else:
        sem = stats.sem(arr)
        ci_low, ci_high = stats.t.interval(confidence, df=arr.size - 1, loc=mean, scale=sem)
        ci_low = float(ci_low)
        ci_high = float(ci_high)

    return {
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def paired_ttest(
    left: Sequence[float],
    right: Sequence[float],
) -> Dict[str, Optional[float]]:
    """Run a paired t-test on two matched samples."""
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.size != right_arr.size:
        raise ValueError("Paired t-test inputs must have identical lengths.")
    if left_arr.size < 2:
        return {
            "n": int(left_arr.size),
            "mean_left": float(left_arr.mean()) if left_arr.size else None,
            "mean_right": float(right_arr.mean()) if right_arr.size else None,
            "mean_diff": float((right_arr - left_arr).mean()) if left_arr.size else None,
            "t_stat": None,
            "p_value": None,
        }

    t_stat, p_value = stats.ttest_rel(left_arr, right_arr, nan_policy="omit")
    return {
        "n": int(left_arr.size),
        "mean_left": float(left_arr.mean()),
        "mean_right": float(right_arr.mean()),
        "mean_diff": float((right_arr - left_arr).mean()),
        "t_stat": float(t_stat) if t_stat is not None and not np.isnan(t_stat) else None,
        "p_value": float(p_value) if p_value is not None and not np.isnan(p_value) else None,
    }


def paired_ttests_from_case_metrics(
    case_metrics_by_label: Dict[str, Path],
    metrics: Iterable[str],
    id_col: str = "case_id",
) -> pd.DataFrame:
    """Compute pairwise paired t-tests for shared case metrics across result files."""
    loaded: Dict[str, pd.DataFrame] = {}
    for label, path in case_metrics_by_label.items():
        if path.exists():
            loaded[label] = pd.read_csv(path)

    rows: List[Dict[str, object]] = []
    for (left_label, left_df), (right_label, right_df) in combinations(loaded.items(), 2):
        merged = left_df.merge(right_df, on=id_col, suffixes=("_left", "_right"))
        if merged.empty:
            continue
        for metric in metrics:
            left_col = f"{metric}_left"
            right_col = f"{metric}_right"
            if left_col not in merged.columns or right_col not in merged.columns:
                continue
            pair_df = merged[[left_col, right_col]].dropna()
            if pair_df.empty:
                continue
            stats_row = paired_ttest(pair_df[left_col].to_numpy(), pair_df[right_col].to_numpy())
            rows.append(
                {
                    "comparison": f"{left_label} vs {right_label}",
                    "metric": metric,
                    **stats_row,
                }
            )
    return pd.DataFrame(rows)

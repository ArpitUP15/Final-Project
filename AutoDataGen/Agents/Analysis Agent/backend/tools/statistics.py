"""
statistical_analysis_tool
--------------------------
Step 9: mean, median, std, min, max, quartiles for numerical columns.
"""

import pandas as pd


def statistical_analysis_tool(df: pd.DataFrame, numerical_columns: list) -> dict:
    stats = {}
    for col in numerical_columns:
        series = df[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "q1": round(float(series.quantile(0.25)), 2),
            "q3": round(float(series.quantile(0.75)), 2),
        }
    return stats

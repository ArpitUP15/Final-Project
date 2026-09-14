"""
missing_value_tool
-------------------
Step 7: missing value count + percentage per column.
Step 8: duplicate row detection.
"""

import pandas as pd


def missing_value_tool(df: pd.DataFrame) -> dict:
    counts = df.isnull().sum()
    percentages = (df.isnull().mean() * 100).round(2)

    missing = {}
    for col in df.columns:
        if counts[col] > 0:
            missing[col] = {
                "count": int(counts[col]),
                "percentage": float(percentages[col]),
            }

    return missing


def duplicate_tool(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())

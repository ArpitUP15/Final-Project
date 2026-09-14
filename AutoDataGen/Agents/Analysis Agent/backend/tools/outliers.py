"""
outlier_analysis_tool
-----------------------
Step 10: IQR method.

Lower bound = Q1 - 1.5 * IQR
Upper bound = Q3 + 1.5 * IQR
"""

import pandas as pd


def outlier_analysis_tool(df: pd.DataFrame, numerical_columns: list) -> dict:
    outliers = {}
    for col in numerical_columns:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outliers[col] = 0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(((series < lower) | (series > upper)).sum())
        if count > 0:
            outliers[col] = count

    return outliers

"""
column_analysis_tool
---------------------
Step 5 + 6: logical data-type classification (not just pandas dtype)
and unique value / class counts per column.

Classification buckets:
    numerical   -> int / float
    binary      -> categorical column with exactly 2 unique values
    categorical -> object/category column with <= CATEGORICAL_THRESHOLD unique values
    datetime    -> datetime64 or column that parses cleanly as dates
    text        -> object column with too many unique values to be categorical
"""

import warnings
import pandas as pd

CATEGORICAL_THRESHOLD = 20  # beyond this, treat an object column as free text


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.9
    except Exception:
        return False


def _classify_column(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    if _looks_like_datetime(series):
        return "datetime"

    n_unique = series.nunique(dropna=True)
    if n_unique == 2:
        return "binary"
    if n_unique <= CATEGORICAL_THRESHOLD:
        return "categorical"
    return "text"


def column_analysis_tool(df: pd.DataFrame) -> dict:
    result = {
        "numerical": [],
        "binary": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "unique_values": {},
    }

    for col in df.columns:
        col_type = _classify_column(df[col])
        result[col_type].append(col)
        result["unique_values"][col] = int(df[col].nunique(dropna=True))

    return result

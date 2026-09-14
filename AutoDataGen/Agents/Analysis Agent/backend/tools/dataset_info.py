"""
dataset_info_tool
------------------
Step 4: rows, columns, column names.
"""

import pandas as pd


def dataset_info_tool(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
    }

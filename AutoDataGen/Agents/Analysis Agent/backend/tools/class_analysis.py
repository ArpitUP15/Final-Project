"""
class_distribution_tool
-------------------------
Step 11: suggest candidate target columns (don't blindly assume last column).
Step 12: class distribution + imbalance flag for the chosen/likely target.
"""

import pandas as pd

# common target-ish names seen in real datasets
TARGET_NAME_HINTS = [
    "target", "label", "class", "churn", "outcome",
    "y", "result", "output", "diagnosis", "default",
]

IMBALANCE_THRESHOLD = 0.65  # majority class share above this => imbalanced


def suggest_target_columns(df: pd.DataFrame, binary_cols: list, categorical_cols: list) -> list:
    candidates = []

    # 1. name-based hints
    for col in df.columns:
        col_lower = col.lower()
        for hint in TARGET_NAME_HINTS:
            if len(hint) == 1:
                # single-char hints (like "y") must match the whole name,
                # not just appear somewhere inside it
                if col_lower == hint:
                    candidates.append(col)
                    break
            elif hint in col_lower:
                candidates.append(col)
                break

    # 2. binary columns are strong target candidates
    for col in binary_cols:
        if col not in candidates:
            candidates.append(col)

    # 3. low-cardinality categorical columns as weaker candidates
    for col in categorical_cols:
        if col not in candidates and df[col].nunique(dropna=True) <= 10:
            candidates.append(col)

    return candidates[:5]


def class_distribution_tool(df: pd.DataFrame, target_column: str) -> dict:
    if target_column not in df.columns:
        return {}

    counts = df[target_column].value_counts(dropna=True)
    total = counts.sum()
    if total == 0:
        return {}

    distribution = {str(k): int(v) for k, v in counts.items()}
    majority_share = counts.max() / total

    return {
        "column": target_column,
        "classes": int(counts.shape[0]),
        "distribution": distribution,
        "imbalanced": bool(majority_share >= IMBALANCE_THRESHOLD),
        "majority_share_percent": round(float(majority_share * 100), 2),
    }

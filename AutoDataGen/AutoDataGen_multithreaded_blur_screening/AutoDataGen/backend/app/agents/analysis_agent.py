import numpy as np
import pandas as pd

def _num(x):
    return None if pd.isna(x) or not np.isfinite(x) else float(x)

def analyze_dataframe(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    missing = df.isna().sum()
    stats, outliers, cat_stats = {}, {}, {}
    for col in numeric:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty: continue
        q1, q3 = s.quantile([.25, .75]).tolist()
        iqr = q3-q1
        outliers[str(col)] = int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
        stats[str(col)] = {"mean": _num(s.mean()), "std": _num(s.std()),
                           "min": _num(s.min()), "median": _num(s.median()), "max": _num(s.max())}
    for col in categorical:
        top = df[col].astype("string").value_counts(dropna=False).head(8)
        cat_stats[str(col)] = {"unique_values": int(df[col].nunique(dropna=True)),
                               "top_values": {str(k): int(v) for k,v in top.items()}}
    miss_rate = int(missing.sum()) / max(rows*cols, 1)
    dup = int(df.duplicated().sum())
    dup_rate = dup / max(rows, 1)
    score = round(max(0, min(100, 100*(1-.65*miss_rate-.35*dup_rate))), 2)
    result = {
        "shape": {"rows": int(rows), "columns": int(cols)},
        "dataset_type": "tabular",
        "column_types": {"numeric": list(map(str,numeric)), "categorical": list(map(str,categorical))},
        "missing_values": {"total_missing_cells": int(missing.sum()),
            "by_column": {str(k):int(v) for k,v in missing.items()}, "missing_rate": round(miss_rate,4)},
        "duplicate_rows": dup, "duplicate_rate": round(dup_rate,4),
        "numeric_statistics": stats, "numeric_outlier_counts_iqr": outliers,
        "categorical_statistics": cat_stats, "quality_score": score,
        "quality_score_note": "Heuristic based on missingness and duplicates only; not a universal quality measure."
    }
    # LLM analysis is advisory. Send aggregate statistics, not raw records or target values.
    from app.agents.ollama_agent import ask_json
    compact_profile = {
        "shape": result["shape"],
        "numeric_statistics": result["numeric_statistics"],
        "categorical_statistics": {
            col: {"unique_values": stats.get("unique_values", 0)}
            for col, stats in result["categorical_statistics"].items()
        },
        "missing_values": result["missing_values"],
        "duplicate_rows": result["duplicate_rows"],
        "quality_score": result["quality_score"],
    }
    insights = ask_json(
        "You are a cautious tabular-data quality analyst. Inspect only the supplied aggregate profile. "
        "Return JSON with keys: risks (array of short strings), recommended_checks (array of short strings), "
        "summary (string). Do not infer causal relationships, claim model accuracy, or invent columns.",
        {"profile": compact_profile}, num_predict=220,
    )
    result["llm_insights"] = insights or {
        "summary": "LLM insights unavailable; deterministic profiling is available.",
        "risks": [], "recommended_checks": [], "source": "fallback",
    }
    result["llm_insights"]["source"] = "ollama" if insights else "fallback"
    return result

"""
analysis_agent.py
------------------
Agent 1 = Dataset Analysis Agent.

Deterministic part (Pandas/NumPy) builds the JSON report.
LLM is only used at the very end to turn the JSON into a human-readable
explanation. LLM never touches raw numbers/calculations.

    Dataset -> Pandas/NumPy tools -> JSON report -> LLM -> explanation
"""

import os
from pathlib import Path
import shutil
import pandas as pd

from tools.dataset_info import dataset_info_tool
from tools.column_analysis import column_analysis_tool
from tools.missing_values import missing_value_tool, duplicate_tool
from tools.statistics import statistical_analysis_tool
from tools.outliers import outlier_analysis_tool
from tools.class_analysis import suggest_target_columns, class_distribution_tool
from tools.content_analysis import image_analysis_tool, image_dataset_analysis_tool, text_analysis_tool
from utils.file_handler import extract_image_archive


def compute_quality_score(rows: int, missing_values: dict, duplicates: int,
                           outliers: dict, target: dict | None) -> int:
    """Simple heuristic score out of 100. NOT a scientifically validated metric."""
    score = 100

    if missing_values:
        avg_missing_pct = sum(v["percentage"] for v in missing_values.values()) / len(missing_values)
        score -= min(30, round(avg_missing_pct))

    if rows > 0:
        dup_pct = (duplicates / rows) * 100
        score -= min(20, round(dup_pct))

    if outliers:
        total_outliers = sum(outliers.values())
        outlier_pct = (total_outliers / rows) * 100 if rows else 0
        score -= min(20, round(outlier_pct))

    if target and target.get("imbalanced"):
        score -= 15

    return max(0, min(100, score))


def _fallback_summary(report: dict) -> str:
    """Used when no LLM API key is configured — a plain templated summary."""
    d = report["dataset"]
    lines = [
        f"The dataset has {d['rows']} rows and {d['columns']} columns.",
    ]
    if report["duplicates"]:
        lines.append(f"{report['duplicates']} duplicate rows were found.")
    if report["missing_values"]:
        worst = max(report["missing_values"].items(), key=lambda kv: kv[1]["percentage"])
        lines.append(f"'{worst[0]}' has the most missing data at {worst[1]['percentage']}%.")
    if report["outliers"]:
        lines.append(f"Outliers were detected in: {', '.join(report['outliers'].keys())}.")
    if report.get("target"):
        t = report["target"]
        if t.get("imbalanced"):
            lines.append(
                f"Target column '{t['column']}' appears imbalanced "
                f"(majority class = {t['majority_share_percent']}%)."
            )
    lines.append(f"Overall heuristic quality score: {report['quality_score']}/100.")
    return " ".join(lines)


def _llm_summary(report: dict) -> str | None:
    """Call Claude to turn the JSON report into a readable explanation.
    Returns None if no API key is set (caller should fall back)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "You are summarizing a dataset quality report for a data scientist. "
            "Only explain and interpret the numbers below in plain English — "
            "do not invent any numbers that aren't present. Keep it under 150 words.\n\n"
            f"{report}"
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
    except Exception:
        return None


def run_analysis(df: pd.DataFrame, target_column: str | None = None) -> dict:
    """Runs all 6 tools and returns the final structured report."""

    info = dataset_info_tool(df)
    columns = column_analysis_tool(df)
    missing = missing_value_tool(df)
    duplicates = duplicate_tool(df)

    numerical_cols = columns["numerical"]
    stats = statistical_analysis_tool(df, numerical_cols)
    outliers = outlier_analysis_tool(df, numerical_cols)

    target_candidates = suggest_target_columns(
        df, columns["binary"], columns["categorical"]
    )

    chosen_target = target_column or (target_candidates[0] if target_candidates else None)
    target_info = class_distribution_tool(df, chosen_target) if chosen_target else None

    quality_score = compute_quality_score(
        rows=info["rows"],
        missing_values=missing,
        duplicates=duplicates,
        outliers=outliers,
        target=target_info,
    )

    report = {
        "dataset": info,
        "columns": columns,
        "missing_values": missing,
        "duplicates": duplicates,
        "statistics": stats,
        "outliers": outliers,
        "target_suggestions": target_candidates,
        "target": target_info,
        "quality_score": quality_score,
    }

    summary = _llm_summary(report)
    report["llm_summary"] = summary if summary else _fallback_summary(report)

    return report


def run_file_analysis(path: str, target_column: str | None = None) -> tuple[str, dict]:
    """Analyze a supported upload and return its dataset type and report."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "tabular", run_analysis(pd.read_csv(path), target_column=target_column)
    if suffix in {".txt", ".md", ".log"}:
        text_stats = text_analysis_tool(path)
        word_count = text_stats["words"]
        quality_score = 100 if word_count else 0
        return "text", {
            "dataset": {"rows": text_stats["lines"], "columns": 1, "column_names": ["text"]},
            "text": text_stats,
            "quality_score": quality_score,
            "llm_summary": f"Text dataset contains {word_count} words and {text_stats['unique_words']} unique words.",
        }
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        image_stats = image_analysis_tool(path)
        quality_score = 100 if image_stats["valid"] and image_stats["opencv_readable"] else 0
        return "image", {
            "dataset": {"rows": 1, "columns": 1, "column_names": ["image"]},
            "image": image_stats,
            "quality_score": quality_score,
            "llm_summary": "Image integrity checks completed with Pillow and OpenCV.",
        }
    if suffix == ".zip":
        extraction_dir, image_paths = extract_image_archive(path)
        try:
            image_stats = image_dataset_analysis_tool(image_paths, extraction_dir)
        finally:
            shutil.rmtree(extraction_dir, ignore_errors=True)
        total = image_stats["total_images"]
        valid = image_stats["valid_images"]
        quality_score = round((valid / total) * 100) if total else 0
        if image_stats["duplicate_images"] and total:
            quality_score = max(0, quality_score - round(image_stats["duplicate_images"] / total * 20))
        if image_stats["imbalanced"]:
            quality_score = max(0, quality_score - 15)
        return "image_dataset", {
            "dataset": {"rows": total, "columns": 1, "column_names": ["image"]},
            "image_dataset": image_stats,
            "quality_score": quality_score,
            "llm_summary": f"Image dataset contains {total} images; {valid} passed integrity checks.",
        }
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

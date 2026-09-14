"""
state.py
--------
Shape of the structured JSON report that Agent 1 produces.
This is also what gets handed to Agent 2 (Strategy Selection Agent) later.
"""

from typing import Optional
from pydantic import BaseModel


class DatasetInfo(BaseModel):
    rows: int
    columns: int
    column_names: list[str]


class AnalysisReport(BaseModel):
    dataset: DatasetInfo
    columns: dict
    missing_values: dict
    duplicates: int
    statistics: dict
    outliers: dict
    target_suggestions: list[str]
    target: Optional[dict] = None
    quality_score: int
    text: Optional[dict] = None
    image: Optional[dict] = None
    llm_summary: Optional[str] = None

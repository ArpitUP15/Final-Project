"""PostgreSQL storage for analysis reports."""

import json
import os
from typing import Any


CREATE_REPORTS_TABLE = """
CREATE TABLE IF NOT EXISTS dataset_analysis_reports (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    dataset_type TEXT NOT NULL,
    report JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def _connection():
    import psycopg2

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg2.connect(database_url)


def save_report(filename: str, dataset_type: str, report: dict[str, Any]) -> int:
    """Create the audit table if needed and save one immutable report."""
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_REPORTS_TABLE)
            cursor.execute(
                """
                INSERT INTO dataset_analysis_reports (filename, dataset_type, report)
                VALUES (%s, %s, %s::jsonb)
                RETURNING id
                """,
                (filename, dataset_type, json.dumps(report, allow_nan=False)),
            )
            report_id = cursor.fetchone()[0]
        connection.commit()
    return int(report_id)


def get_report(report_id: int) -> dict[str, Any] | None:
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, filename, dataset_type, report, created_at
                FROM dataset_analysis_reports
                WHERE id = %s
                """,
                (report_id,),
            )
            row = cursor.fetchone()

    if row is None:
        return None
    return {
        "id": row[0],
        "filename": row[1],
        "dataset_type": row[2],
        "report": row[3],
        "created_at": row[4].isoformat(),
    }

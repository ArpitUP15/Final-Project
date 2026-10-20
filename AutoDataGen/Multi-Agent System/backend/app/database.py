"""Persistent SQLite storage for input/output datasets and run reports.

SQLite is built into Python. Dataset payloads are stored as BLOBs in the database,
so runs survive backend restarts. Actual capacity is bounded by available disk and
SQLite's limits, not by an application-level upload-size setting.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path(__import__("os").getenv(
    "AUTODATAGEN_DB_PATH", str(BACKEND_DIR / "data" / "autodatagen.sqlite3")
)).expanduser().resolve()


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                target_column TEXT,
                output_format TEXT,
                status TEXT NOT NULL,
                report_json TEXT,
                input_dataset BLOB NOT NULL,
                output_dataset BLOB,
                output_suffix TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # Additive migrations keep existing local databases usable.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        if "cancel_requested" not in columns:
            conn.execute("ALTER TABLE runs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")


def mark_interrupted_runs_on_startup():
    """Mark jobs that were in-process when the API restarted; local thread jobs are not durable."""
    now = datetime.now(timezone.utc).isoformat()
    message = "Backend restarted before this in-process job completed. Submit the dataset again."
    with _connect() as conn:
        conn.execute(
            "UPDATE runs SET status='failed', report_json=?, updated_at=? WHERE status IN ('queued','processing')",
            (json.dumps({"status": "failed", "error": message, "restart_recovery": True}), now),
        )


def create_run_record(run_id: str, filename: str, target_column: str,
                      output_format: str, input_bytes: bytes):
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute("""
            INSERT INTO runs (
                run_id, filename, target_column, output_format, status,
                input_dataset, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)
        """, (run_id, filename, target_column, output_format,
              sqlite3.Binary(input_bytes), now, now))


def update_run_progress(run_id: str, progress: dict[str, Any]):
    """Persist lightweight job progress so clients can poll without a long HTTP request."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE runs SET status='processing', report_json=?, updated_at=? WHERE run_id=? AND status IN ('queued','processing')",
            (json.dumps({"run_id": run_id, "status": "processing", "progress": progress}, default=str), now, run_id),
        )


def request_run_cancel(run_id: str) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE runs SET cancel_requested=1, updated_at=? WHERE run_id=? AND status IN ('queued','processing')",
            (now, run_id),
        )
        return cursor.rowcount > 0


def is_run_cancel_requested(run_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT cancel_requested FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return bool(row and row[0])


def mark_run_cancelled(run_id: str, message: str = "Cancelled by user"):
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "UPDATE runs SET status='cancelled', report_json=?, updated_at=? WHERE run_id=?",
            (json.dumps({"run_id": run_id, "status": "cancelled", "error": message}), now, run_id),
        )


def complete_run(run_id: str, report: dict[str, Any],
                 output_bytes: bytes, output_suffix: str):
    now = datetime.now(timezone.utc).isoformat()
    safe_report = {k: v for k, v in report.items()
                   if k not in {"dataset_bytes", "input_dataset", "output_dataset"}}
    with _connect() as conn:
        cursor = conn.execute("""
            UPDATE runs SET status='completed', report_json=?,
                output_dataset=?, output_suffix=?, updated_at=?
            WHERE run_id=?
        """, (json.dumps(safe_report, default=str), sqlite3.Binary(output_bytes),
              output_suffix, now, run_id))
        if cursor.rowcount == 0:
            raise KeyError(f"Run {run_id} not found in database")


def fail_run(run_id: str, error: str):
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute("""
            UPDATE runs SET status='failed', report_json=?, updated_at=?
            WHERE run_id=?
        """, (json.dumps({"run_id": run_id, "status": "failed", "error": error}), now, run_id))


def get_run_record(run_id: str):
    with _connect() as conn:
        row = conn.execute("""
            SELECT run_id, filename, target_column, output_format, status,
                   report_json, output_suffix, created_at, updated_at
            FROM runs WHERE run_id=?
        """, (run_id,)).fetchone()
    if not row:
        return None
    keys = ["run_id", "filename", "target_column", "output_format", "status",
            "report_json", "output_suffix", "created_at", "updated_at"]
    result = dict(zip(keys, row))
    report = json.loads(result.pop("report_json") or "{}")
    result.update(report)
    result.setdefault("run_id", run_id)
    result["status"] = row[4]
    result["created_at"] = row[7]
    result["updated_at"] = row[8]
    return result


def get_dataset_blob(run_id: str, kind: str):
    column = {"input": "input_dataset", "output": "output_dataset"}.get(kind)
    if not column:
        raise ValueError("kind must be 'input' or 'output'")
    with _connect() as conn:
        row = conn.execute(f"SELECT {column} FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return bytes(row[0]) if row and row[0] is not None else None


def get_output_suffix(run_id: str):
    with _connect() as conn:
        row = conn.execute("SELECT output_suffix FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return row[0] if row else None


def list_runs(limit: int = 100, offset: int = 0):
    """Paginated metadata only; dataset blobs are never loaded by this endpoint."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT run_id, filename, target_column, output_format, status, created_at, updated_at
            FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()
    columns = ["run_id", "filename", "target_column", "output_format", "status", "created_at", "updated_at"]
    return [dict(zip(columns, row)) for row in rows]

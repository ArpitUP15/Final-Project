"""
main.py
-------
FastAPI entrypoint for Agent 1 (Dataset Analysis Agent).

Run with:
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000 in your browser, or POST a CSV to /upload.
"""

import sys
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from utils.file_handler import save_upload, load_and_validate, DatasetValidationError
from agents.analysis_agent import run_file_analysis
from utils.report_store import get_report, save_report

app = FastAPI(title="AutoDataGen - Agent 1: Dataset Analysis Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the user-friendly web interface."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "agent": "Dataset Analysis Agent"}


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str | None = Query(default=None, description="Optional: name of the target/label column"),
):
    file_bytes = await file.read()

    try:
        saved_path = save_upload(file_bytes, file.filename)
        if Path(file.filename).suffix.lower() == ".csv":
            load_and_validate(saved_path)
    except DatasetValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        dataset_type, report = run_file_analysis(saved_path, target_column=target_column)
    except (ValueError, UnicodeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    report_id = None
    persistence_error = None
    try:
        report_id = save_report(file.filename, dataset_type, report)
    except Exception as e:
        persistence_error = str(e)

    return {
        "filename": file.filename,
        "dataset_type": dataset_type,
        "report_id": report_id,
        "persistence": {"saved": report_id is not None, "error": persistence_error},
        "report": report,
    }


@app.get("/reports/{report_id}")
def report(report_id: int):
    try:
        saved_report = get_report(report_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Report database unavailable: {e}")
    if saved_report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return saved_report

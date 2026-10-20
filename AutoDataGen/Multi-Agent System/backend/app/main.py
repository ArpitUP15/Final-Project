import io, json, os, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.workflow.graph import run_workflow
from app.config import (HARD_MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS, DEFAULT_TARGET_ACCURACY,
                        DEFAULT_STRETCH_TARGET, DEFAULT_TOTAL_TIME_BUDGET_SECONDS, DEFAULT_WORKERS)
from app.agents.analysis_agent import analyze_dataframe
from app.agents.ollama_agent import ollama_status
from app.database import (
    init_db, create_run_record, complete_run, fail_run,
    get_run_record, get_dataset_blob, get_output_suffix, list_runs,
    update_run_progress, is_run_cancel_requested, request_run_cancel, mark_run_cancelled,
    mark_interrupted_runs_on_startup,
)

load_dotenv()
app = FastAPI(title="AutoDataGen API", version="0.4.0")
# Training is moved off the request/ASGI event loop. A small worker pool prevents local
# development from launching many CPU-heavy jobs at once; production deployments should
# replace this with a durable process/task queue.
JOB_POOL = ThreadPoolExecutor(max_workers=DEFAULT_WORKERS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    mark_interrupted_runs_on_startup()


def read_dataset(filename, content):
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        if suffix in {".xlsx", ".xlsm"}:
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        if suffix == ".xls":
            raise HTTPException(400, "Legacy .xls is not enabled; save as .xlsx or .csv.")
        raise HTTPException(400, "Upload a .csv, .xlsx, or .xlsm file.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Could not read dataset: {exc}") from exc


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AutoDataGen API", "database": "sqlite",
            "application_upload_limit": None, "ollama": ollama_status()}


@app.get("/api/ollama/status")
def get_ollama_status():
    """Report whether the configured local Ollama service and model are available."""
    return ollama_status()


@app.post("/api/analyze")
async def analyze(file: Annotated[UploadFile, File(...)]):
    content = await file.read()
    df = read_dataset(file.filename or "dataset.csv", content)
    if df.empty or len(df.columns) < 2:
        raise HTTPException(400, "Dataset must have rows and at least two columns.")
    return {
        "filename": file.filename, "rows": len(df), "columns": list(map(str, df.columns)),
        "preview": df.head(8).astype(object).where(pd.notna(df.head(8)), None).to_dict("records"),
        "analysis": analyze_dataframe(df),
    }


def _run_pipeline_job(run_id, df, filename, target_column, output_format,
                      max_iterations, target_accuracy, stretch_target, total_time_budget_seconds):
    try:
        if is_run_cancel_requested(run_id):
            mark_run_cancelled(run_id, "Cancelled before worker start.")
            return
        update_run_progress(run_id, {
            "status": "processing", "stage": "starting", "baseline_score": None,
            "target_score": target_accuracy, "stretch_target": stretch_target,
            "max_iterations": min(HARD_MAX_ITERATIONS, max(1, int(max_iterations))),
            "planned_iterations": None, "completed_iterations": 0,
            "remaining_iterations": None, "best_iteration": None,
            "best_validation_score": None, "elapsed_seconds": 0,
        })
        def progress_callback(progress):
            update_run_progress(run_id, progress)
        def cancel_callback():
            return is_run_cancel_requested(run_id)

        result = run_workflow(
            df, target_column, run_id, filename, output_format,
            progress_callback=progress_callback, cancel_callback=cancel_callback,
            max_iterations=min(HARD_MAX_ITERATIONS, max(1, int(max_iterations))),
            target_accuracy=min(0.99, max(0.50, float(target_accuracy))),
            stretch_target=min(0.99, max(0.50, float(stretch_target))),
            total_time_budget_seconds=min(3600.0, max(30.0, float(total_time_budget_seconds))),
        )
        if is_run_cancel_requested(run_id):
            mark_run_cancelled(run_id, "Cancellation requested; the job stopped at a safe candidate boundary.")
            return
        update_run_progress(run_id, {
            "status": "finalizing", "stage": "finalizing", "completed_iterations": result.get("iteration_search", {}).get("completed_iterations", 0),
            "planned_iterations": result.get("iteration_search", {}).get("planned_iterations"),
            "best_validation_score": result.get("iteration_search", {}).get("best_validation_score"),
            "elapsed_seconds": result.get("iteration_search", {}).get("elapsed_seconds", 0),
        })
        complete_run(run_id, result, result["dataset_bytes"], result["download_suffix"])
    except Exception as exc:
        if is_run_cancel_requested(run_id):
            mark_run_cancelled(run_id, "Cancelled while processing.")
        else:
            fail_run(run_id, f"{type(exc).__name__}: {exc}")


@app.post("/api/runs", status_code=202)
async def create_run(
    file: Annotated[UploadFile, File(...)],
    target_column: Annotated[str, Form(...)],
    output_format: Annotated[str, Form()] = "original",
    max_iterations: Annotated[int, Form()] = DEFAULT_MAX_ITERATIONS,
    target_accuracy: Annotated[float, Form()] = DEFAULT_TARGET_ACCURACY,
    stretch_target: Annotated[float, Form()] = DEFAULT_STRETCH_TARGET,
    total_time_budget_seconds: Annotated[float, Form()] = DEFAULT_TOTAL_TIME_BUDGET_SECONDS,
):
    content = await file.read()
    filename = file.filename or "dataset.csv"
    df = read_dataset(filename, content)
    if target_column not in df.columns:
        raise HTTPException(400, f"Target column '{target_column}' not found.")
    if output_format not in {"original", "csv", "xlsx"}:
        raise HTTPException(400, "output_format must be original, csv, or xlsx.")
    if df.empty or len(df.columns) < 2:
        raise HTTPException(400, "Dataset must have rows and at least two columns.")
    # Server-side clamping is authoritative, regardless of client-supplied values.
    max_iterations = min(HARD_MAX_ITERATIONS, max(1, int(max_iterations)))
    target_accuracy = min(0.99, max(0.50, float(target_accuracy)))
    stretch_target = min(0.99, max(target_accuracy, float(stretch_target)))
    total_time_budget_seconds = min(3600.0, max(30.0, float(total_time_budget_seconds)))

    run_id = str(uuid.uuid4())
    create_run_record(run_id, filename, target_column, output_format, content)
    update_run_progress(run_id, {
        "status": "queued", "stage": "queued", "max_iterations": max_iterations,
        "target_score": target_accuracy, "stretch_target": stretch_target,
        "completed_iterations": 0, "elapsed_seconds": 0,
    })
    JOB_POOL.submit(_run_pipeline_job, run_id, df, filename, target_column, output_format,
                    max_iterations, target_accuracy, stretch_target, total_time_budget_seconds)
    return JSONResponse(status_code=202, content={
        "run_id": run_id, "status": "queued", "filename": filename,
        "progress": {"stage": "queued", "max_iterations": max_iterations,
                     "target_score": target_accuracy, "stretch_target": stretch_target,
                     "completed_iterations": 0},
    })


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    result = get_run_record(run_id)
    if result is None:
        raise HTTPException(404, "Run not found.")
    if result.get("status") in {"completed", "failed", "cancelled"}:
        raise HTTPException(409, f"Cannot cancel a run with status '{result.get('status')}'.")
    if not request_run_cancel(run_id):
        raise HTTPException(409, "The run is no longer cancellable.")
    return {"run_id": run_id, "status": "cancellation_requested"}


@app.get("/api/runs")
def runs(limit: int = 100, offset: int = 0):
    if limit < 1 or limit > 1000 or offset < 0:
        raise HTTPException(400, "limit must be 1..1000 and offset must be non-negative.")
    return {"items": list_runs(limit, offset), "limit": limit, "offset": offset}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    result = get_run_record(run_id)
    if result is None:
        raise HTTPException(404, "Run not found.")
    return result


@app.get("/api/runs/{run_id}/download")
def download(run_id: str):
    result = get_run_record(run_id)
    if result is None:
        raise HTTPException(404, "Run not found.")
    payload = get_dataset_blob(run_id, "output")
    if payload is None:
        raise HTTPException(409, "Output dataset is not available; the run may have failed.")
    suffix = get_output_suffix(run_id) or ".csv"
    media = "text/csv" if suffix == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return StreamingResponse(io.BytesIO(payload), media_type=media,
        headers={"Content-Disposition": f'attachment; filename="autodatagen_{run_id[:8]}{suffix}"'})


@app.get("/api/runs/{run_id}/input")
def download_input(run_id: str):
    result = get_run_record(run_id)
    if result is None:
        raise HTTPException(404, "Run not found.")
    payload = get_dataset_blob(run_id, "input")
    if payload is None:
        raise HTTPException(404, "Input dataset not found.")
    suffix = Path(result.get("filename", "input.csv")).suffix.lower() or ".csv"
    media = "text/csv" if suffix == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return StreamingResponse(io.BytesIO(payload), media_type=media,
        headers={"Content-Disposition": f'attachment; filename="input_{run_id[:8]}{suffix}"'})


@app.get("/api/runs/{run_id}/report")
def report_download(run_id: str):
    result = get_run_record(run_id)
    if result is None:
        raise HTTPException(404, "Run not found.")
    report = {k: v for k, v in result.items()
              if k not in {"dataset_bytes", "input_dataset", "output_dataset"}}
    return StreamingResponse(io.BytesIO(json.dumps(report, indent=2, default=str).encode()),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="autodatagen_{run_id[:8]}_report.json"'})

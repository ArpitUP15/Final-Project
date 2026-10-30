import io, json, os, uuid, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.workflow.graph import run_workflow
from app.image_augmentation import augment_image_zip
from app.config import (HARD_MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS, DEFAULT_TARGET_ACCURACY,
                        DEFAULT_STRETCH_TARGET, DEFAULT_TOTAL_TIME_BUDGET_SECONDS, DEFAULT_WORKERS)
from app.agents.analysis_agent import analyze_dataframe
from app.agents.multimodal_analysis_agent import analyze_upload
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
IMAGE_EXPORTS = {}
UNIFIED_EXPORTS = {}
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
        if suffix == ".txt":
            # Plain text files are interpreted as one text example per non-empty line.
            text = content.decode("utf-8-sig")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                raise HTTPException(400, "The .txt file contains no non-empty text lines.")
            return pd.DataFrame({"text": lines})
        if suffix in {".xlsx", ".xlsm"}:
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        if suffix == ".xls":
            raise HTTPException(400, "Legacy .xls is not enabled; save as .xlsx or .csv.")
        raise HTTPException(400, "Upload a .txt, .csv, .xlsx, or .xlsm file.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Could not read dataset: {exc}") from exc


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AutoDataGen API", "database": "sqlite",
            "application_upload_limit": None, "ollama": ollama_status()}




@app.post("/api/image-augmentation")
async def image_augmentation(
    file: Annotated[UploadFile, File(...)],
    task_type: Annotated[str, Form()] = "classification",
    mode: Annotated[str, Form()] = "offline",
    variants_per_image: Annotated[int, Form()] = 2,
    seed: Annotated[int, Form()] = 42,
    blur_filter: Annotated[bool, Form()] = True,
    blur_threshold: Annotated[float, Form()] = 100.0,
):
    """Augment an image dataset ZIP. Supports class folders, YOLO labels, or segmentation masks."""
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "Upload a ZIP archive containing the image dataset and annotations.")
    # Read in bounded chunks so oversized uploads are rejected early rather than
    # allocating an unbounded byte string in one operation. Configure the cap with
    # AUTODATAGEN_MAX_IMAGE_ZIP_MIB (default 2048 MiB).
    max_upload_mib = max(1, int(os.getenv("AUTODATAGEN_MAX_IMAGE_ZIP_MIB", "2048")))
    max_upload_bytes = max_upload_mib * 1024 * 1024
    chunks = []
    total_bytes = 0
    while True:
        chunk = await file.read(8 * 1024 * 1024)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_upload_bytes:
            raise HTTPException(
                413,
                f"Image dataset ZIP exceeds the configured {max_upload_mib} MiB limit. "
                "Set AUTODATAGEN_MAX_IMAGE_ZIP_MIB to raise it if sufficient RAM and disk are available.",
            )
        chunks.append(chunk)
    content = b"".join(chunks)
    await file.close()
    if not content:
        raise HTTPException(400, "The uploaded ZIP is empty.")
    try:
        # PIL transforms and scikit-learn evaluation are CPU-bound; do not block
        # FastAPI's event loop while the dataset is being processed.
        output, report = await run_in_threadpool(
            augment_image_zip, content, task_type, mode, variants_per_image, seed, None, blur_filter, blur_threshold
        )
    except (ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(400, str(exc)) from exc
    # Make evaluation scope explicit: annotation integrity is not model accuracy.
    report["model_evaluation_scope"] = (
        "Classification model comparison is performed only when a compatible train/validation class-folder layout is available."
        if task_type == "classification" else
        "No task-specific model training/evaluation is performed for this detection/segmentation run; annotation checks are integrity checks only."
    )
    report["model_evaluation_limitations"] = (
        ["Object detection needs a detection training/evaluation adapter and metrics such as mAP/IoU.",
         "Segmentation needs a segmentation training/evaluation adapter and metrics such as mIoU/Dice.",
         "Valid bounding boxes or masks do not prove model accuracy."]
        if task_type in {"detection", "segmentation"} else
        ["Classification evaluation is conditional on a compatible validation split.",
         "A validation accuracy comparison does not prove every generated image is semantically correct."]
    )
    run_id = str(uuid.uuid4())
    IMAGE_EXPORTS[run_id] = output
    # Keep in-memory export cache bounded for local development.
    while len(IMAGE_EXPORTS) > 8:
        IMAGE_EXPORTS.pop(next(iter(IMAGE_EXPORTS)))
    return {"run_id": run_id, "status": "completed", "report": report,
            "download_url": f"/api/image-augmentation/{run_id}/download"}


@app.get("/api/image-augmentation/{run_id}/download")
def download_image_augmentation(run_id: str):
    from fastapi.responses import Response
    output = IMAGE_EXPORTS.get(run_id)
    if output is None:
        raise HTTPException(404, "Image augmentation export not found; run the augmentation again.")
    return Response(content=output, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="autodatagen-images-{run_id[:8]}.zip"'})


@app.get("/api/ollama/status")
def get_ollama_status():
    """Report whether the configured local Ollama service and model are available."""
    return ollama_status()


@app.post("/api/analyze")
async def analyze(file: Annotated[UploadFile, File(...)]):
    """Unified Analysis Agent entry point: automatically routes image ZIPs, text, and tabular data."""
    filename = file.filename or "dataset.csv"
    content = await file.read()
    if not content:
        raise HTTPException(400, "The uploaded dataset is empty.")
    try:
        profile = analyze_upload(filename, content, read_dataset)
        if profile.get("modality") == "image":
            return {"filename": filename, "rows": None, "columns": [], "preview": [],
                    "dataset_type": "image", "analysis": profile}
        df = read_dataset(filename, content)
        if df.empty:
            raise HTTPException(400, "Dataset contains no rows.")
        return {"filename": filename, "rows": len(df), "columns": list(map(str, df.columns)),
                "preview": df.head(8).astype(object).where(pd.notna(df.head(8)), None).to_dict("records"),
                "dataset_type": profile.get("modality", "tabular"), "analysis": profile}
    except HTTPException:
        raise
    except (ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/datasets/augment")
async def unified_augment(
    file: Annotated[UploadFile, File(...)],
    target_column: Annotated[str | None, Form()] = None,
    output_format: Annotated[str, Form()] = "original",
    image_task_type: Annotated[str, Form()] = "classification",
    image_mode: Annotated[str, Form()] = "offline",
    variants_per_image: Annotated[int, Form()] = 2,
    seed: Annotated[int, Form()] = 42,
    max_iterations: Annotated[int, Form()] = DEFAULT_MAX_ITERATIONS,
    target_accuracy: Annotated[float, Form()] = DEFAULT_TARGET_ACCURACY,
    stretch_target: Annotated[float, Form()] = DEFAULT_STRETCH_TARGET,
    total_time_budget_seconds: Annotated[float, Form()] = DEFAULT_TOTAL_TIME_BUDGET_SECONDS,
    blur_filter: Annotated[bool, Form()] = True,
    blur_threshold: Annotated[float, Form()] = 100.0,
):
    """Unified modality router. Image ZIPs use the image agent; text/tabular files use the existing LangGraph agents."""
    filename = file.filename or "dataset.csv"
    content = await file.read()
    if not content:
        raise HTTPException(400, "The uploaded dataset is empty.")
    run_id = str(uuid.uuid4())
    try:
        modality_profile = analyze_upload(filename, content, read_dataset)
    except (ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(400, str(exc)) from exc
    if Path(filename).suffix.lower() == ".zip":
        try:
            output, report = augment_image_zip(content, image_task_type, image_mode,
                                               max(1, min(20, int(variants_per_image))), int(seed), None, blur_filter, blur_threshold)
        except (ValueError, zipfile.BadZipFile) as exc:
            raise HTTPException(400, str(exc)) from exc
        evaluation_scope = (
            "Classification model comparison is performed only when a compatible train/validation class-folder layout is available."
            if image_task_type == "classification" else
            "No task-specific model training/evaluation is performed for this detection/segmentation run; annotation checks are integrity checks only."
        )
        evaluation_limitations = (
            ["Object detection needs a detection training/evaluation adapter and metrics such as mAP/IoU.",
             "Segmentation needs a segmentation training/evaluation adapter and metrics such as mIoU/Dice.",
             "Valid bounding boxes or masks do not prove model accuracy."]
            if image_task_type in {"detection", "segmentation"} else
            ["Classification evaluation is conditional on a compatible validation split.",
             "A validation accuracy comparison does not prove every generated image is semantically correct."]
        )
        report["model_evaluation_scope"] = evaluation_scope
        report["model_evaluation_limitations"] = evaluation_limitations
        UNIFIED_EXPORTS[run_id] = {"bytes": output, "suffix": ".zip", "report": {
            "run_id": run_id, "filename": filename, "modality": "image", "status": "completed", "analysis": modality_profile,
            "task_type": image_task_type, "augmentation": report,
            "model_evaluation_scope": evaluation_scope,
            "model_evaluation_limitations": evaluation_limitations,
            "pipeline_agents": ["AnalysisAgent", "ImageStrategyAgent", "ImageExecutionAgent", "ImageValidationAgent", "ExportAgent"]}}
        return {"run_id": run_id, "status": "completed", "modality": "image", "report": UNIFIED_EXPORTS[run_id]["report"],
                "download_url": f"/api/datasets/augment/{run_id}/download"}

    try:
        df = read_dataset(filename, content)
    except HTTPException:
        raise
    if df.empty:
        raise HTTPException(400, "Dataset contains no rows.")
    suffix = Path(filename).suffix.lower()
    if target_column is None and (suffix == ".txt" or (len(df.columns) == 1 and str(df.columns[0]).lower() in {"text", "content", "sentence", "document"})):
        target_column = str(df.columns[0])
        result = run_workflow(df, target_column, run_id, filename, output_format)
    elif target_column is None:
        raise HTTPException(422, "A target_column is required for supervised tabular or labeled text datasets. Analyze the dataset first and select its label/target column.")
    else:
        if target_column not in df.columns:
            raise HTTPException(400, f"Target column '{target_column}' not found.")
        if output_format not in {"original", "csv", "xlsx"}:
            raise HTTPException(400, "output_format must be original, csv, or xlsx.")
        result = run_workflow(df, target_column, run_id, filename, output_format,
            max_iterations=min(HARD_MAX_ITERATIONS, max(1, int(max_iterations))),
            target_accuracy=min(.99, max(.50, float(target_accuracy))),
            stretch_target=min(.99, max(float(target_accuracy), float(stretch_target))),
            total_time_budget_seconds=min(3600.0, max(30.0, float(total_time_budget_seconds))))
    report = {k: v for k, v in result.items() if k not in {"dataset_bytes", "download_suffix"}}
    modality = "text" if result.get("task_type") == "text_augmentation" or "text" in str(result.get("task_type", "")).lower() else "tabular"
    report["modality"] = modality
    report["analysis"] = report.get("analysis") or modality_profile
    report["pipeline_agents"] = ["AnalysisAgent", "StrategyAgent", "ExecutionAgent", "EvaluationAgent", "OrchestratorAgent", "ExportAgent"]
    UNIFIED_EXPORTS[run_id] = {"bytes": result["dataset_bytes"], "suffix": result["download_suffix"], "report": report}
    while len(UNIFIED_EXPORTS) > 12:
        UNIFIED_EXPORTS.pop(next(iter(UNIFIED_EXPORTS)))
    return {"run_id": run_id, "status": "completed", "modality": modality, "report": report,
            "download_url": f"/api/datasets/augment/{run_id}/download"}


@app.get("/api/datasets/augment/{run_id}/download")
def download_unified_augmentation(run_id: str):
    from fastapi.responses import Response
    item = UNIFIED_EXPORTS.get(run_id)
    if item is None:
        raise HTTPException(404, "Augmentation export not found; rerun the augmentation.")
    suffix = item["suffix"]
    media_type = "application/zip" if suffix == ".zip" else ("text/plain; charset=utf-8" if suffix == ".txt" else "application/octet-stream")
    return Response(content=item["bytes"], media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="autodatagen-{item["report"].get("modality", "dataset")}-{run_id[:8]}{suffix}"'})


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
    is_txt = Path(filename).suffix.lower() == ".txt"
    is_single_text_column = len(df.columns) == 1 and not pd.api.types.is_numeric_dtype(df.iloc[:, 0])
    if df.empty or (len(df.columns) < 2 and not is_txt and not is_single_text_column):
        raise HTTPException(400, "Tabular datasets must have rows and at least two columns; text corpora need at least one non-empty text column or TXT line.")
    if is_txt and target_column != "text":
        raise HTTPException(400, "For a .txt file, target_column must be 'text'.")
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
    media = ("text/plain; charset=utf-8" if suffix == ".txt" else
             "text/csv; charset=utf-8" if suffix == ".csv" else
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
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
    media = ("text/plain; charset=utf-8" if suffix == ".txt" else
             "text/csv; charset=utf-8" if suffix == ".csv" else
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
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

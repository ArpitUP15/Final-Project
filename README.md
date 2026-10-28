# AutoDataGen

Local-first prototype for tabular CSV/XLSX dataset profiling, strategy selection, augmentation, and classification evaluation using FastAPI, React, LangGraph, scikit-learn, imbalanced-learn, and optional Ollama.

## Requirements
- Python **3.14.3** (managed/pinned for the backend with uv)
- Node.js **22.2.0** (pinned in `frontend/.nvmrc`)
- [uv](https://docs.astral.sh/uv/) for Python dependency and environment management
- Ollama local model runtime (recommended for LLM-assisted analysis, strategy, augmentation proposals, search ordering, and evaluation summaries). Deterministic fallbacks remain available if Ollama is disabled or offline.

## Backend (Windows PowerShell)

Install uv if it is not already installed, then run:

```powershell
cd backend
uv python install 3.14.3
uv sync
Copy-Item .env.example .env
uv run python -m uvicorn app.main:app --reload
```

`uv sync` creates/synchronizes the project's `.venv` and resolves dependencies from `pyproject.toml`. `uv run` executes commands inside that environment; do not use `pip install` or `requirements.txt`.

API docs: http://127.0.0.1:8000/docs

### Backend (macOS/Linux)

```bash
cd backend
uv python install 3.14.3
uv sync
cp .env.example .env
uv run python -m uvicorn app.main:app --reload
```

## Ollama setup (Windows / macOS / Linux)

AutoDataGen uses Ollama through LangChain's `ChatOllama`. Ollama runs locally by default at `http://localhost:11434`; download/install it from [the official Ollama download page](https://ollama.com/download). Windows installs a background app; on Linux, start the service with `ollama serve` if it is not already running. See the [official Windows notes](https://github.com/ollama/ollama/blob/main/docs/windows.mdx) and [LangChain Ollama integration guide](https://docs.langchain.com/oss/python/integrations/providers/ollama/).

1. Install Ollama and start the Ollama app/service.
2. Open a separate terminal and download the configured model:

   ```powershell
   ollama pull llama3.2:3b
   ollama list
   ```

   On macOS/Linux, use the same commands in Terminal. The model name must match `OLLAMA_MODEL` in `backend/.env`.
3. In a separate terminal, verify the local service responds:

   ```powershell
   ollama run llama3.2:3b
   ```

   Enter a short prompt, then type `/bye` to exit. Ollama's local API normally listens on `http://localhost:11434`.
4. In `backend/`, copy `.env.example` to `.env` if you have not already, then confirm these settings:

   ```dotenv
   OLLAMA_ENABLED=true
   OLLAMA_MODEL=llama3.2:3b
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_TIMEOUT_SECONDS=20
   ```
5. Start/restart the backend so it loads the `.env` file:

   ```powershell
   uv run python -m uvicorn app.main:app --reload
   ```

6. Open [http://127.0.0.1:8000/api/ollama/status](http://127.0.0.1:8000/api/ollama/status). Verify `enabled`, `available`, and `model_available` are all `true`. The frontend header also shows Ollama status; use **Check Ollama** to refresh it.
7. Start the frontend and run a dataset. The report shows whether each agent used Ollama or a deterministic fallback.

If Ollama is installed on a different host or port, update `OLLAMA_BASE_URL`. If the configured model is missing, run `ollama pull <model-name>`. If the model is too large or slow for your hardware, choose a smaller model and update `.env`.

### What uses Ollama, and what remains deterministic

- **Analysis agent:** asks for quality risks and recommended checks from aggregate statistics; it does not send raw dataset rows.
- **Strategy agent:** asks for a preferred supported strategy and rationale; strategy names are checked against an allow-list and unsupported recommendations are ignored.
- **Execution/augmentation agent:** asks for candidate unseen categorical values from observed unique-value summaries. Candidates are checked for type, length, duplicates, and whether they already exist. Generated values are still candidates and require domain validation.
- **Orchestrator agent:** asks Ollama to prioritize already-supported strategy/model candidates. It may reorder the search only; it cannot invent a model configuration or choose a winner without validation.
- **Evaluation agent:** asks for an explanation of the actual computed metrics. Ollama cannot set or modify scores, and the test set remains excluded from model selection.
- **Actual ML training, predictions, metrics, row counts, schema checks, the 60-iteration cap, and API/database operations** remain normal Python/scikit-learn/FastAPI code. An LLM must not fabricate these results.

All LLM calls use the configured local Ollama endpoint, with bounded timeouts and deterministic fallbacks. If Ollama is offline, disabled, times out, or returns invalid JSON, the pipeline continues with rule-based behavior and marks the relevant metadata as `fallback`. Only aggregate profiles, unique category values, candidate configurations, and evaluation metrics are sent to prompts—not full row records or the target column's values. The LLM's proposed category values may still be incorrect; review them before relying on them in a real domain.

## Frontend

Use Node.js **22.2.0** (for example, `nvm use` in the frontend directory), then:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Test
Upload a CSV/XLSX with a target column and preferably at least 100 rows. Categorical/low-cardinality targets use classification; numeric targets with many distinct values use regression. Choose the target column and run the workflow.

## Workflow and limitations
LangGraph: analyze -> choose strategy -> execute augmentation -> evaluate -> export.
The original data is split before augmentation; the holdout test partition is never augmented. The export combines the augmented training partition with the untouched test partition and labels rows in `__autodatagen_source`.

This version focuses on tabular classification and baseline regression model comparison. It does not yet implement image/text augmentation, CTGAN/TVAE, GAN/diffusion, auth, a durable external task queue, or production-grade upload isolation. Input and output dataset bytes and run reports are persisted in a SQLite database at `backend/data/autodatagen.sqlite3` by default. Configure `AUTODATAGEN_DB_PATH` to change the location. The application does not impose a configured upload-size cap; practical limits still depend on available RAM, disk space, SQLite's BLOB limits, and any reverse proxy/server limits. For multi-gigabyte production datasets, use PostgreSQL metadata plus object storage and streaming/chunked processing rather than loading the entire dataset into pandas memory. Use only non-sensitive test data. Synthetic data is not guaranteed to be realistic, private, or beneficial.


### Adaptive optimization and asynchronous jobs

- The optimizer uses a server-enforced **hard cap of 60 candidate evaluations**. User configuration is clamped to `1..60` in the API and again in the orchestrator.
- If baseline validation accuracy is at least **80%**, the optimizer evaluates at most one candidate. Below 80%, it plans a larger budget (at least 5 and up to the configured cap), adjusted for row count, feature count, imbalance, missingness, duplicates, and baseline validation instability. The default maximum is 60.
- Classification candidates include Random Forest, Extra Trees, Decision Trees, AdaBoost, linear/SVM models and KNN when the dataset is small enough, alongside compatible class weighting and resampling strategies. Every candidate is evaluated on the same fixed validation split; baseline stability is measured over three splits. The untouched test partition is used only for final reporting, never model selection.
- Balanced classification uses accuracy as its primary score. Imbalanced classification uses balanced accuracy as the primary score while requiring raw accuracy not to regress on validation. Macro-F1 and per-class metrics are also reported. The default target is 80%, with 90% as a stretch goal; neither is guaranteed.
- Numeric targets with many distinct values are treated as regression and compared with Random Forest, Extra Trees, Decision Tree, Gradient Boosting, HistGradientBoosting, and Ridge using validation R². Regression reports MAE/RMSE/R², not accuracy. Synthetic regression rows are not fabricated; a target-aware regression synthesizer would need to be configured separately.
- The `SimilarityOverSampler` creates up to 20% additional rows from similar same-class parent records. Numeric features are interpolated locally and categorical features mostly reuse parent values. Optional unseen categorical candidates are inferred from observed column patterns and recorded in metadata; validate them against domain rules.
- Candidate augmentation is selected using validation scores only. The untouched test set may show lower performance than validation; this is reported honestly rather than using the test set to switch models after the fact.
- `POST /api/runs` now returns HTTP 202 and a `run_id` quickly. The frontend polls `GET /api/runs/{run_id}` every two seconds for progress, current candidate, best validation score, elapsed time, and status. `POST /api/runs/{run_id}/cancel` requests cancellation at a candidate boundary. CPU training runs in a bounded in-process worker pool rather than holding a long HTTP request open.
- The in-process worker is intended for local development. A process restart marks queued/running jobs as interrupted; durable recovery and hard process-level candidate timeouts require a persistent task queue/worker deployment. The total time budget is checked between candidate fits, since scikit-learn estimator fits are not safely interruptible from a Python thread.
- The report includes baseline score, best validation score, planned/completed/remaining iterations, best iteration, stop reason, candidate history, selected model, class diagnostics, and final untouched test metrics.


### Balanced and imbalanced dataset handling

- The strategy agent classifies the target distribution using the majority/minority ratio (>= 1.5 is treated as imbalanced).
- Balanced datasets start with no resampling; class weighting is evaluated as a comparison.
- Imbalanced datasets evaluate class-weighted learning first, followed by no-resampling and compatible oversampling strategies.
- For imbalanced classification, balanced accuracy is the selection objective and raw validation accuracy is a non-regression guardrail.
- Evaluation reports accuracy, balanced accuracy, macro precision/recall/F1, Matthews correlation coefficient, ROC-AUC where supported, average precision for binary classification, per-class classification report, and confusion matrices.
- Resampling occurs only on the training partition. Validation and final test partitions retain their original class distribution.
- The 90% threshold is a model-selection target, not a guarantee and not a claim about dataset quality.


### Persistent dataset storage

- SQLite stores each uploaded input dataset and generated output dataset as BLOBs, along with run status, report JSON, file name, target column, format, and timestamps.
- Database location defaults to `backend/data/autodatagen.sqlite3`; override it with `AUTODATAGEN_DB_PATH`.
- `GET /api/runs` lists run metadata with pagination.
- `GET /api/runs/{run_id}/input` downloads the original uploaded dataset.
- `GET /api/runs/{run_id}/download` downloads the generated dataset.
- `GET /api/runs/{run_id}/report` downloads the report.
- Dataset/report records persist across backend restarts. In-process jobs cannot resume after a restart; queued/running records are marked failed with a recovery message. Back up the SQLite file to preserve datasets and reports.
- No application-level upload-size limit is configured. This does not mean infinite capacity: hardware, available memory/disk, SQLite limits, and infrastructure may constrain very large datasets.


### Generated-row behavior

The orchestrator separates dataset generation from model selection. If no classification candidate improves the validation objective while preserving raw validation accuracy, it still applies `SimilarityOverSampler` to create a bounded synthetic training batch (up to 20% of the training partition), while marking the baseline model as selected. The untouched test set is never used to switch models. Lower final test performance is reported honestly and does not trigger post-hoc model selection.

### Central configuration

`backend/app/config.py` centralizes the defaults. Optional environment variables include `AUTODATAGEN_MAX_ITERATIONS` (always clamped to 60), `AUTODATAGEN_TARGET_ACCURACY` (0.80), `AUTODATAGEN_STRETCH_TARGET` (0.90), `AUTODATAGEN_TIME_BUDGET_SECONDS` (420), `AUTODATAGEN_CANDIDATE_TIMEOUT_SECONDS` (45; over-budget candidates are rejected after their fit completes because Python-thread estimator fits cannot be forcefully interrupted), and `AUTODATAGEN_WORKERS` (1). The frontend offers maximum iteration choices up to 60.

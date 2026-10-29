# AutoDataGen

Local-first prototype for text, table, and image dataset analysis and augmentation using FastAPI, React, LangGraph, scikit-learn, imbalanced-learn, and optional Ollama.

The main frontend URL now opens a **dataset-type chooser**. Select **Text**, **Table**, or **Image** to open its separate augmentation page. See [FRONTEND_PAGES.md](FRONTEND_PAGES.md) for routes, file structure, and login redirect integration.

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
uv run uvicorn app.main:app --reload
```

`uv sync` creates/synchronizes the project's `.venv` and resolves dependencies from `pyproject.toml`. `uv run` executes commands inside that environment; do not use `pip install` or `requirements.txt`.

API docs: http://127.0.0.1:8000/docs

### Backend (macOS/Linux)

```bash
cd backend
uv python install 3.14.3
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload
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
   uv run uvicorn app.main:app --reload
   ```

6. Open [http://127.0.0.1:8000/api/ollama/status](http://127.0.0.1:8000/api/ollama/status). Verify `enabled`, `available`, and `model_available` are all `true`. The frontend header also shows Ollama status; use **Check connection** to refresh it.
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
npm ci
npm run dev
```

Open http://127.0.0.1:5173. The first page asks which dataset type you want to augment.

- Text: `http://127.0.0.1:5173/#/augment/text`
- Table: `http://127.0.0.1:5173/#/augment/table`
- Image: `http://127.0.0.1:5173/#/augment/image`

The frontend proxies `/api` to the backend on port 8000 for development and preview. Optional overrides are documented in `frontend/.env.example`. Browser refresh and Back/Forward work on every workspace link. The separate `frontend.zip` is not required.

## Test
Upload a CSV/XLSX with a target column and preferably at least 100 rows. Categorical/low-cardinality targets use classification; numeric targets with many distinct values use regression. Choose the target column and run the workflow.

## Workflow and limitations
LangGraph: analyze -> choose strategy -> execute augmentation -> evaluate -> export.
The original data is split before augmentation; the holdout test partition is never augmented. The export combines the augmented training partition with the untouched test partition and labels rows in `__autodatagen_source`.

This version supports tabular classification/regression, local-model text augmentation, and task-aware image augmentation. Evaluation coverage differs by task, as documented below. It does not implement CTGAN/TVAE, GAN/diffusion, auth, a durable external task queue, or production-grade upload isolation. Input and output dataset bytes and run reports are persisted in a SQLite database at `backend/data/autodatagen.sqlite3` by default. Configure `AUTODATAGEN_DB_PATH` to change the location. The application does not impose a configured upload-size cap; practical limits still depend on available RAM, disk space, SQLite's BLOB limits, and any reverse proxy/server limits. For multi-gigabyte production datasets, use PostgreSQL metadata plus object storage and streaming/chunked processing rather than loading the entire dataset into pandas memory. Use only non-sensitive test data. Synthetic data is not guaranteed to be realistic, private, or beneficial.


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

## Ollama-powered text augmentation

For text-based classification datasets (for example spam detection, sentiment, or text labels), AutoDataGen detects likely natural-language columns and adds two candidate strategies to the validation search when Ollama is enabled:

- `TextSynonymReplacement`: asks the configured Ollama model to replace a few non-critical words/phrases while preserving meaning, sentiment, negation, and entities.
- `TextBackTranslation`: asks Ollama for a back-translation-style paraphrase while preserving the original intent.

The augmentation runs on the training partition only. Each generated row copies the source row's target label and non-text features; only the detected text column is changed. The default batch is capped at 20 rows and about 10% of the training partition to avoid excessive LLM latency. Duplicate, unchanged, and implausibly sized variants are rejected. Generated rows are marked in `__autodatagen_source` and reported under `augmentation.generated_rows` and `augmentation.generated_text_examples`.

To enable it, start Ollama and pull the model configured in `backend/.env` (default `llama3.2:3b`). Verify `GET /api/ollama/status` reports the model available. Text augmentation is a candidate strategy, not a guarantee of better accuracy; validation metrics determine model selection. If Ollama is unavailable, the system records that no text rows were generated instead of fabricating paraphrases.


## Fix: unlabeled text datasets (`sentences.csv`)

If a selected target column has a very high unique-value ratio and looks like free text (for example, `translation_extra` has 720 unique values across 720 rows), it is not a classification label. The workflow now detects this case and runs **text-only Ollama augmentation** on the selected column instead of failing while calculating classification metrics.

- Uses `TextBackTranslation` by default for text-only runs.
- Keeps the original rows and appends valid, non-duplicate variants returned by Ollama.
- Preserves the other columns of the source row.
- Reports generated-row counts and examples.
- Does not invent classification accuracy: `evaluation` is `null` because no ground-truth labels were supplied.
- If you want supervised classification metrics, provide a separate `label`/`class` column and select that as the target; choose the sentence column as the text feature.

## Plain-text (`.txt`) augmentation

AutoDataGen accepts `.txt` files in addition to CSV and Excel. Each non-empty line is treated as one text example and loaded into a single `text` column. Choose **Same as input** to export an augmented `.txt` file with one text example per line; CSV and Excel exports are also available. TXT inputs are routed directly to the existing Ollama text-only augmentation workflow, including short/repeated lines. Because a plain text corpus has no label column by default, classification accuracy is not computed.

Example `sentences.txt`:

```text
Saving money helps people prepare for emergencies.
Artificial intelligence is changing how businesses work.
Reading books improves vocabulary and knowledge.
```

Blank lines are ignored. Use UTF-8 text encoding. Generated variants depend on Ollama being enabled and a configured model being available; if generation is unavailable, the app should report zero generated variants rather than claiming success for text that was not generated.


## Image dataset augmentation (Milestones 1–5)

The image augmentation workflow is exposed on the dedicated **Image augmentation** page of the frontend and through `POST /api/image-augmentation`.

### Supported tasks and ZIP layouts

- **Classification:** `train/<class_name>/*.jpg` (class folders are required). Validation/test folders are preserved and not augmented.
- **Object detection:** `images/train/*.jpg` and matching `labels/train/*.txt` in YOLO normalized `class x_center y_center width height` format. The current annotation-safe transform is horizontal flipping; box centers are updated in the same operation. Rotation/cropping are deliberately not applied to detection because they require exact box clipping/recalculation.
- **Segmentation:** matching `images/train/name.png` and `masks/train/name.png` pairs with identical dimensions and stems. Spatial transforms are synchronized; masks use nearest-neighbour interpolation so class IDs remain discrete.

### What the five milestones deliver

1. **Classification augmentation:** shared transform engine supports flip, small rotation, crop/zoom, brightness/contrast, noise and light blur. Offline mode saves variants. `ImageAugmentationDataset` exposes dynamic per-access transforms for training loaders without writing files.
2. **Evaluation and selection:** every generated image is reopened/verified; detection labels are parsed and bounds-checked; segmentation image/mask dimensions are checked. The image-specific report records generated/skipped items and annotation checks. For classification datasets with `train/<class>/` and `val/<class>/`, a deterministic baseline and augmented-data RandomForest are compared on the same validation split; variants are rejected if validation accuracy regresses. This lightweight guard is a safety baseline, not a guarantee of production model quality.
3. **Object detection:** image/YOLO-label pairs are transformed together with label integrity checks.
4. **Segmentation:** images and masks receive matching spatial decisions and mask-safe interpolation.
5. **Frontend integration:** choose task and mode, set variants, inspect the report/previews, and download the output ZIP. The ZIP includes `autodatagen_image_report.json`.

### API example

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/image-augmentation `
  -F "file=@images-dataset.zip" -F "task_type=classification" `
  -F "mode=offline" -F "variants_per_image=2"
```

The response includes a `download_url` for the generated ZIP. Supported task values are `classification`, `detection`, and `segmentation`. Modes are `offline` and `training`. Training mode returns the source data plus an explicit augmentation manifest; use `ImageAugmentationDataset` in a Python training loop to apply dynamic classification transforms. The classification guard trains a lightweight validation model when a proper validation split is present; it does not claim that this model is suitable for production or guarantee accuracy improvements.

## Unified multimodal dataset pipeline

AutoDataGen exposes a unified Analysis Agent router for image, text, and tabular uploads:

- `POST /api/analyze` accepts `.zip` image datasets and `.txt`, `.csv`, `.xlsx`, or `.xlsm` data. It returns a modality decision, evidence, detailed profile, quality checks, and the recommended pipeline.
- `POST /api/datasets/augment` dispatches image ZIPs to the task-aware image augmentation engine (classification, YOLO detection, or segmentation) and text/tabular files to the existing LangGraph agent workflow. For supervised datasets, supply `target_column`; unlabeled `.txt`/single-text-column corpora can be augmented without a target.
- `GET /api/datasets/augment/{run_id}/download` downloads the output produced by the unified endpoint.

The modality router is deterministic and uses file/container structure plus content profiling; an LLM may add advisory insights but cannot override file-safety checks. CSV files containing long natural-language columns are identified as text-oriented tabular datasets, so they can retain labels and use the existing strategy/evaluation agents. Image ZIPs are profiled for decodable images, dimensions, folder/split distribution, and likely annotation type before dispatch.

Important evaluation boundary: the current image engine provides image/annotation validation and classification validation metrics where a compatible train/validation class-folder layout exists. Object-detection and segmentation model-specific metrics require a task-specific training/evaluation adapter and are not inferred from annotation integrity alone.


## Evaluation coverage and limitations

AutoDataGen detects and routes tabular, text, and image datasets, but evaluation coverage differs by task. Tabular evaluation uses the existing model-candidate validation workflow. Text evaluation depends on the existing labeled-text workflow. Image classification accuracy checks require a compatible train/validation class-folder split. Object detection and segmentation currently perform task-aware augmentation and annotation-integrity checks; they do **not** yet train/evaluate dedicated models. Detection needs task metrics such as mAP/IoU, while segmentation needs mIoU/Dice. Valid annotations do not prove model accuracy. These distinctions are also included in image reports and the frontend.

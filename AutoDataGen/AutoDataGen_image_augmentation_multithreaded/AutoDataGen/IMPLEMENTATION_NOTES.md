# AutoDataGen adaptive optimizer and timeout fix

## Root-cause analysis

1. **Accuracy search was too restrictive.** The previous orchestrator capped the search at 20 candidates and accepted a candidate only when raw validation accuracy strictly improved. For imbalanced datasets this rejected candidates that improved balanced accuracy without changing raw accuracy. Candidate order also concentrated on a narrow set of configurations.
2. **Long request timeout.** `POST /api/runs` ran the entire LangGraph workflow synchronously before responding. The browser had to keep an HTTP request open during repeated model fits, which could produce `timeout of 600000ms exceeded`.
3. **Slow synthetic neighbor search.** Similarity oversampling used repeated pandas `.loc` operations inside the nearest-neighbor loop. This caused avoidable overhead on every augmentation candidate.
4. **Test-set selection leakage.** The earlier workflow could fall back to the baseline based on final test accuracy. That used the holdout test set to select a model, contrary to a leakage-safe evaluation protocol.

## Changes made

- `backend/app/config.py`: central defaults, hard cap 60, target 80%, stretch target 90%, time budget, candidate timeout diagnostic, and worker count.
- `backend/app/agents/orchestrator_agent.py`: adaptive planned iteration budget; hard cap enforced again inside the optimizer; richer classifier/strategy search; fixed validation split per candidate; baseline stability across three splits; progress and cancellation callbacks; early stop at target/time budget; classification selection now improves the task-specific validation score while raw validation accuracy must not regress. Regression targets are compared with R²/MAE/RMSE and are not labeled accuracy.
- `backend/app/agents/execution_agent.py`: vectorized numeric/categorical distance computation for faster similarity augmentation.
- `backend/app/agents/evaluation_agent.py`: single-worker KNN instead of `n_jobs=-1`, plus regression target detection, regression pipelines, and regression metrics.
- `backend/app/agents/strategy_agent.py`: detects numeric high-cardinality regression targets before class-imbalance logic.
- `backend/app/workflow/graph.py`: passes configuration and progress callbacks through LangGraph; test set is used only for final metrics, never candidate selection.
- `backend/app/main.py`: existing `POST /api/runs` now returns HTTP 202 and a run ID; bounded background worker; lightweight status polling through `GET /api/runs/{run_id}`; cancellation endpoint; request iteration cap is clamped server-side.
- `backend/app/database.py`: persists job progress and cancellation requests, and marks interrupted in-process jobs after a server restart.
- `frontend/src/App.jsx`: polls every two seconds, displays live baseline/best score/iteration/time progress, exposes max iterations up to 60, and handles completed/failed/cancelled jobs and regression metrics.
- `backend/tests/`: adaptive planning, hard cap, improved/rejected candidate, regression task, augmentation, and asynchronous API contract tests.

## Supplied dataset benchmark

Dataset: `train-00000-of-00001.csv`, target `Risk`, 1,000 rows, 9 features.

- Baseline mean validation accuracy: **67.33%**.
- Best validation accuracy among searched candidates: **69.33%**.
- Primary metric: balanced accuracy, because the target is imbalanced; best validation balanced accuracy: **67.30%**.
- Planned/completed candidate evaluations: **37/37** (adaptive plan, below hard cap 60).
- Selected model/strategy: **ExtraTreesLeaf2 + SMOTENC**.
- Generated training rows: **300**.
- Baseline untouched-test accuracy: **70.00%**.
- Selected model untouched-test accuracy: **66.80%** (delta **-3.20 percentage points**).
- Baseline untouched-test balanced accuracy: **58.00%**.
- Selected model untouched-test balanced accuracy: **61.43%**.

These are actual local-run results, not a claim that the model achieved the target. The test set was evaluated after candidate selection and did not influence model choice. The validation improvement did not generalize to raw test accuracy on this dataset; that outcome is reported transparently rather than using test data to choose a different model. Improving this dataset to 80–90% is not guaranteed by resampling and may require more predictive features, better labels, domain-specific feature engineering, or more representative data.

## Verification

- `python -m compileall -q app tests`: passed.
- `PYTHONPATH=. pytest -q`: **16 passed**, with two FastAPI `on_event` deprecation warnings.
- Supplied dataset optimizer run: completed 37 candidate evaluations in approximately 11.7 seconds after vectorizing similarity search.
- Frontend production build: **not run successfully** because `vite` is not installed in the execution environment (`npm run build` returned `vite: not found`). Run `npm install` in `frontend/` and then `npm run build` locally.
- Full LangGraph end-to-end API run: not available in this environment because `langgraph` and `langchain_core` are not installed. The async API contract test stubs those graph dependencies and verifies HTTP 202 submission, polling, output download, and server-side clamping. Run `uv sync` and perform a full local smoke test to verify the installed integration.

## Operational limitation

The background worker is in-process for the local app. API requests no longer wait for model fitting, but a process restart cannot resume an active fit; interrupted jobs are marked failed. Candidate duration is recorded and the overall budget is checked between fits. Hard-killing a currently executing scikit-learn fit and durable recovery require a process-based worker or external task queue.


## Ollama integration update

- `backend/app/agents/ollama_agent.py`: added a shared LangChain `ChatOllama` adapter, JSON parsing, bounded call timeouts, local `/api/tags` health check, and deterministic fallback behavior.
- `backend/app/agents/analysis_agent.py`: Ollama reviews aggregate profiling statistics and returns advisory data-quality risks/checks; raw dataset rows are not sent.
- `backend/app/agents/strategy_agent.py`: Ollama can recommend a preferred strategy from an allow-list. Unsupported names are ignored; regression remains on `NoResampling` because a target-aware regression synthesizer is not implemented.
- `backend/app/agents/execution_agent.py`: Ollama may propose up to three unseen categorical values per feature from unique-value summaries. Candidates are validated, cached, and recorded in augmentation metadata. It does not set target labels or generate numeric model metrics.
- `backend/app/agents/orchestrator_agent.py`: Ollama can reorder existing model/strategy candidates. Actual selection still uses measured validation scores and the hard iteration/time limits.
- `backend/app/agents/evaluation_agent.py`: Ollama adds an advisory interpretation of the computed classification/regression metrics. The metric values are never generated or modified by the LLM.
- `backend/app/main.py`: added `GET /api/ollama/status`; `/api/health` includes Ollama status.
- `frontend/src/App.jsx` and `frontend/src/styles.css`: added Ollama connection/model status in the header and surfaced LLM-assisted profile/strategy/category/search information.
- `backend/.env.example` and `README.md`: enabled Ollama integration in the example configuration and added setup/troubleshooting steps.
- `backend/tests/test_ollama_agent.py` and `backend/tests/test_ollama_agent_integration.py`: cover disabled/offline fallbacks and strategy allow-list validation.

### Ollama verification in this environment

- Focused unit tests for deterministic fallback, category generation, adaptive iteration limits, and allow-listed strategy recommendations: **21 passed**.
- Python compile checks: passed.
- Full Ollama-backed live calls were not executed because this environment does not have `langchain_ollama`, `langchain_core`, or an Ollama server installed. The project dependency manifest already includes `langchain-ollama` and `langchain-core`; run `uv sync`, install Ollama, pull the configured model, and check `/api/ollama/status` to verify live integration locally.
- Frontend production build was not run because frontend dependencies/Vite are not installed in this execution environment.

Ollama is an advisory component in every agent stage, not a replacement for numerical ML. Scikit-learn still trains and predicts; metrics, schema checks, safety allow-lists, target labels, iteration caps, and final model selection remain validated by code. This is intentional: an LLM must not invent accuracy scores or bypass evaluation safeguards.

## Ollama-powered text augmentation (added)

- Added `TextSynonymReplacement` and `TextBackTranslation` strategies for datasets with detected natural-language columns.
- Ollama generates variants in a bounded batch (maximum 20 rows and about 10% of the training partition by default); the target label and non-text features are copied from each source record.
- The orchestrator includes text augmentation candidates only when Ollama is enabled and a likely text column is detected.
- The frontend displays detected text columns, generated row count, and a few source/variant examples.
- Text generation occurs on training data passed into each candidate only; validation and test partitions are not augmented. Variants are still evaluated and are not guaranteed to improve accuracy.
- Automated tests mock Ollama responses to verify target preservation and output row counts without requiring a live model.


## Image dataset milestones

- Added `backend/app/image_augmentation.py` with classification transforms, offline ZIP augmentation, a dynamic `ImageAugmentationDataset`, paired detection/YOLO transforms, and paired segmentation image/mask transforms.
- Added `POST /api/image-augmentation` and `GET /api/image-augmentation/{run_id}/download`; generated ZIPs include a JSON validation/report file.
- Added a frontend panel for image ZIP upload, task/mode selection, augmentation counts, preview manifest, report status, and export download.
- Added Pillow as a backend dependency.
- Scope note: the image accuracy guard is a lightweight RGB-pixel RandomForest baseline, not a domain-specialized production classifier. If the archive lacks a proper validation split, the report clearly says model-based acceptance could not be run; integrity checks still run.


### Multimodal model-evaluation boundary (explicit)

Unified modality detection and dispatch do not mean all modalities have equivalent model-evaluation support. Tabular evaluation follows the existing candidate-training/validation workflow; text evaluation depends on the supported labeled-data workflow; image classification evaluation is conditional on a compatible train/validation class-folder split. Object detection and segmentation currently validate transformed annotations but do not train/evaluate task-specific models. Detection needs an evaluation adapter (for example mAP/IoU), and segmentation needs an adapter (for example mIoU/Dice). Annotation integrity must never be presented as model accuracy. The image API response, embedded image report, and frontend now expose these limitations. Training-time image mode currently provides a manifest rather than an automatically connected training loop.

## Image augmentation performance fix

- Classification's diagnostic pixel-RandomForest evaluation is bounded to at most 300 training and 200 validation images, uses 16x16 RGB features and 30 trees, and reports that it is a sampled lightweight baseline. This avoids model fitting becoming the dominant cost for large archives.
- Detection and segmentation pair matching now use an image-stem index instead of repeatedly scanning every archive entry for every label/mask (avoids quadratic lookup behavior).
- ZIP export uses DEFLATE level 1 for a faster compression/output tradeoff.
- Image augmentation CPU work runs in Starlette's thread pool so it does not block the FastAPI event loop.
- The former hard-coded 500 MiB upload rejection is replaced with `AUTODATAGEN_MAX_IMAGE_ZIP_MIB` (default 2048 MiB). Large archives still require sufficient RAM because the current augmentation API and export are in-memory; this is not an unlimited/streaming upload implementation.
- Verification after changes: `python -m compileall -q app tests` passed; `pytest -q` passed (40 tests).

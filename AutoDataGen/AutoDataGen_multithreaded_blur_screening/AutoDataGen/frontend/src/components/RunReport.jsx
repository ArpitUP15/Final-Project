import { CheckCircle2, Download, FileText } from "lucide-react";
import Metric from "./Metric.jsx";
import { apiUrl } from "../lib/api.js";
const fmt = (value) =>
  value == null
    ? "N/A"
    : typeof value === "number"
      ? Number.isInteger(value)
        ? value
        : value.toFixed(3)
      : String(value);
export default function RunReport({
  run,
  iterationCap,
  cancelRun,
  cancelRequested,
}) {
  if (!run) return null;
  const progress = run.progress || {};
  const download = (path) => {
    window.location.href = apiUrl(`/api/runs/${run.run_id}/${path}`);
  };
  return (
    <>
      {run &&
        run.status !== "completed" &&
        run.status !== "failed" &&
        run.status !== "cancelled" && (
          <section className="panel">
            <div className="heading">
              <span>03</span>
              <div>
                <h2>Pipeline progress</h2>
                <p>
                  Job {run.run_id} · {run.status || "queued"}
                </p>
              </div>
            </div>
            <div className="metrics">
              <Metric
                label="Baseline score"
                value={
                  progress.baseline_score == null
                    ? "Analyzing…"
                    : `${(progress.baseline_score * 100).toFixed(2)}%`
                }
              />
              <Metric
                label="Best validation score"
                value={
                  progress.best_validation_score == null
                    ? "Searching…"
                    : `${(progress.best_validation_score * 100).toFixed(2)}%`
                }
              />
              <Metric
                label="Iterations"
                value={`${progress.completed_iterations ?? 0} / ${progress.planned_iterations ?? progress.max_iterations ?? iterationCap}`}
              />
              <Metric
                label="Elapsed seconds"
                value={progress.elapsed_seconds ?? 0}
              />
            </div>
            <p className="notice">
              Stage: {progress.stage || "queued"} · Current model:{" "}
              {progress.current_model || "—"} · Strategy:{" "}
              {progress.current_strategy || "—"} · Target:{" "}
              {progress.target_score == null
                ? "80%"
                : `${(progress.target_score * 100).toFixed(0)}%`}{" "}
              · Stretch:{" "}
              {progress.stretch_target == null
                ? "90%"
                : `${(progress.stretch_target * 100).toFixed(0)}%`}
            </p>
            <div className="actions">
              <button
                className="secondary"
                disabled={run.status === "finalizing" || cancelRequested}
                onClick={cancelRun}
              >
                {cancelRequested ? "Cancellation requested" : "Cancel pipeline"}
              </button>
            </div>
            <small>
              Progress polls every 2 seconds. Model training runs in the backend
              worker, so the browser request does not stay open for the full
              optimization.
            </small>
          </section>
        )}
      {run && run.status === "failed" && (
        <section className="panel">
          <div className="error">
            Pipeline failed: {run.error || "Unknown error"}
          </div>
          <button className="secondary" onClick={() => download("report")}>
            Download error report
          </button>
        </section>
      )}
      {run && run.status === "cancelled" && (
        <section className="panel">
          <div className="notice">
            Pipeline cancelled: {run.error || "Cancelled by user"}
          </div>
        </section>
      )}
      {run &&
        run.status === "completed" &&
        run.task_type === "text_augmentation" && (
          <section className="panel">
            <div className="heading">
              <span>03</span>
              <div>
                <h2>Text augmentation results</h2>
                <p>Run {run.run_id} · Unlabeled text augmentation</p>
              </div>
              <b className="complete">
                <CheckCircle2 size={15} /> Completed
              </b>
            </div>
            <div className="metrics">
              <Metric
                label="Original rows"
                value={fmt(run.augmentation?.original_rows)}
              />
              <Metric
                label="Generated rows"
                value={fmt(run.augmentation?.generated_rows)}
              />
              <Metric
                label="Augmented rows"
                value={fmt(run.augmentation?.augmented_rows)}
              />
              <Metric
                label="Method"
                value={run.augmentation?.method || "Text augmentation"}
              />
            </div>
            <div className="card">
              <p>{run.augmentation?.note || "Text augmentation completed."}</p>
              <p>
                <b>Text column:</b>{" "}
                {(run.augmentation?.text_columns || []).join(", ")}
              </p>
              {(run.augmentation?.generated_text_examples || []).map(
                (example, i) => (
                  <div key={i}>
                    <small>Original: {example.source}</small>
                    <p>Generated: {example.variant}</p>
                  </div>
                ),
              )}
            </div>
            <p className="notice">
              No classification accuracy is reported because this dataset has no
              label column. Download the generated dataset to inspect the
              augmented rows.
            </p>
            <div className="actions">
              <button onClick={() => download("download")}>
                <Download size={16} /> Download dataset
              </button>
              <button className="secondary" onClick={() => download("report")}>
                <FileText size={16} /> Download report
              </button>
            </div>
          </section>
        )}
      {run &&
        run.status === "completed" &&
        run.evaluation &&
        run.task_type === "regression" && (
          <section className="panel">
            <div className="heading">
              <span>03</span>
              <div>
                <h2>Regression results</h2>
                <p>
                  Run {run.run_id} · Regression scores are not classification
                  accuracy.
                </p>
              </div>
              <b className="complete">
                <CheckCircle2 size={15} /> Completed
              </b>
            </div>
            <div className="metrics">
              <Metric
                label="Baseline MAE"
                value={fmt(run.evaluation.baseline.mae)}
              />
              <Metric
                label="Selected MAE"
                value={fmt(run.evaluation.augmented.mae)}
              />
              <Metric
                label="Baseline RMSE"
                value={fmt(run.evaluation.baseline.rmse)}
              />
              <Metric
                label="Selected RMSE"
                value={fmt(run.evaluation.augmented.rmse)}
              />
              <Metric
                label="Baseline R²"
                value={fmt(run.evaluation.baseline.r2)}
              />
              <Metric
                label="Selected R²"
                value={fmt(run.evaluation.augmented.r2)}
              />
              <Metric
                label="Iterations"
                value={`${run.iteration_search?.completed_iterations ?? 0} / ${run.iteration_search?.planned_iterations ?? run.iteration_search?.max_iterations ?? 60}`}
              />
            </div>
            <div className="card">
              <h3>Regression model search</h3>
              <p>
                Selected model: <b>{run.selected_model}</b>
              </p>
              <p>
                Target score: R² {fmt(run.iteration_search?.target_score)} ·
                Stretch target: R² {fmt(run.iteration_search?.stretch_target)}
              </p>
              <p>Stop reason: {run.iteration_search?.stop_reason || "N/A"}</p>
              <div className="tablewrap">
                <table>
                  <thead>
                    <tr>
                      <th>Iteration</th>
                      <th>Model</th>
                      <th>Validation R²</th>
                      <th>Validation MAE</th>
                      <th>Validation RMSE</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(run.iteration_search?.history || []).map((item) => (
                      <tr key={item.iteration}>
                        <td>{item.iteration}</td>
                        <td>{item.model_name}</td>
                        <td>{fmt(item.validation_r2)}</td>
                        <td>{fmt(item.validation_mae)}</td>
                        <td>{fmt(item.validation_rmse)}</td>
                        <td>
                          {item.accepted
                            ? "Best so far"
                            : item.error || "Evaluated"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <p className="notice">{run.evaluation.interpretation}</p>
            <div className="actions">
              <button onClick={() => download("download")}>
                <Download size={16} /> Download dataset
              </button>
              <button className="secondary" onClick={() => download("report")}>
                <FileText size={16} /> Download JSON report
              </button>
            </div>
          </section>
        )}
      {run &&
        run.status === "completed" &&
        run.evaluation &&
        run.task_type !== "regression" && (
          <section className="panel">
            <div className="heading">
              <span>03</span>
              <div>
                <h2>Workflow results</h2>
                <p>Run {run.run_id}</p>
              </div>
              <b className="complete">
                <CheckCircle2 size={15} /> Completed
              </b>
            </div>
            <div className="agents">
              {[
                "Analysis",
                "Strategy",
                "Execution",
                "Evaluation",
                "Orchestrator",
              ].map((x, i) => (
                <div key={x}>
                  <small>0{i + 1}</small>
                  <b>{x} Agent</b>
                  <CheckCircle2 size={15} />
                </div>
              ))}
            </div>
            <div className="twocol">
              <div className="card">
                <h3>Selected strategy</h3>
                <h2 className="strategy">{run.strategy.strategy}</h2>
                <p>{run.strategy.reason}</p>
                <p>Training rows: {run.augmentation.original_training_rows}</p>
                <p>Generated/resampled: {run.augmentation.generated_rows}</p>
                <p>
                  Selected model:{" "}
                  {run.selected_model ||
                    run.evaluation.augmented_model ||
                    "RandomForest"}
                </p>
                <p>
                  Class weighting: {run.augmentation.class_weight || "None"}
                </p>
                <p>
                  Dataset balance:{" "}
                  {run.iteration_search?.dataset_balance || "N/A"}
                </p>
                {run.strategy.llm_rationale && (
                  <small>
                    Ollama strategy rationale: {run.strategy.llm_rationale}
                  </small>
                )}
                {run.augmentation.llm_proposed_categorical_candidates &&
                  Object.keys(
                    run.augmentation.llm_proposed_categorical_candidates,
                  ).length > 0 && (
                    <p>
                      <b>Ollama-proposed new categories:</b>{" "}
                      {JSON.stringify(
                        run.augmentation.llm_proposed_categorical_candidates,
                      )}
                    </p>
                  )}
                {run.augmentation.text_columns?.length > 0 && (
                  <div className="notice">
                    <b>Ollama text augmentation</b>
                    <p>
                      Method:{" "}
                      {run.augmentation.method ||
                        run.augmentation.used_strategy}{" "}
                      · Text columns: {run.augmentation.text_columns.join(", ")}
                    </p>
                    <p>
                      Generated text rows: {run.augmentation.generated_rows}
                    </p>
                    {(run.augmentation.generated_text_examples || []).map(
                      (example, i) => (
                        <div key={i}>
                          <small>Original: {example.source}</small>
                          <p>Variant: {example.variant}</p>
                        </div>
                      ),
                    )}
                    <small>{run.augmentation.note}</small>
                  </div>
                )}
              </div>
              <div className="card">
                <h3>Model evaluation</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Baseline</th>
                      <th>Augmented</th>
                      <th>Δ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      "accuracy",
                      "balanced_accuracy",
                      "precision_macro",
                      "recall_macro",
                      "f1_macro",
                      "matthews_corrcoef",
                      "roc_auc",
                      "average_precision",
                    ].map((k) => (
                      <tr key={k}>
                        <td>{k}</td>
                        <td>{fmt(run.evaluation.baseline[k])}</td>
                        <td>{fmt(run.evaluation.augmented[k])}</td>
                        <td>{fmt(run.evaluation.delta[k])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="twocol">
              <div className="card">
                <h3>Baseline confusion matrix</h3>
                <p>Rows = actual class · Columns = predicted class</p>
                <div className="tablewrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Actual / Predicted</th>
                        {(run.evaluation.baseline.labels || []).map((label) => (
                          <th key={label}>{label}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(run.evaluation.baseline.confusion_matrix || []).map(
                        (row, i) => (
                          <tr key={i}>
                            <th>{run.evaluation.baseline.labels[i]}</th>
                            {row.map((value, j) => (
                              <td key={j}>{value}</td>
                            ))}
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
              <div className="card">
                <h3>Augmented-model confusion matrix</h3>
                <p>Rows = actual class · Columns = predicted class</p>
                <div className="tablewrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Actual / Predicted</th>
                        {(run.evaluation.augmented.labels || []).map(
                          (label) => (
                            <th key={label}>{label}</th>
                          ),
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {(run.evaluation.augmented.confusion_matrix || []).map(
                        (row, i) => (
                          <tr key={i}>
                            <th>{run.evaluation.augmented.labels[i]}</th>
                            {row.map((value, j) => (
                              <td key={j}>{value}</td>
                            ))}
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="card">
              <h3>Per-class performance · augmented model</h3>
              <div className="tablewrap">
                <table>
                  <thead>
                    <tr>
                      <th>Class</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1</th>
                      <th>Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(run.evaluation.augmented.labels || []).map((label) => {
                      const m =
                        run.evaluation.augmented.classification_report?.[
                          label
                        ] || {};
                      return (
                        <tr key={label}>
                          <td>{label}</td>
                          <td>{fmt(m.precision)}</td>
                          <td>{fmt(m.recall)}</td>
                          <td>{fmt(m["f1-score"])}</td>
                          <td>{fmt(m.support)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
            <p className="notice">
              Dataset balance: {run.iteration_search?.dataset_balance || "N/A"}{" "}
              · Selection metric:{" "}
              {run.iteration_search?.selection_metric || "accuracy"} · Imbalance
              ratio: {fmt(run.iteration_search?.imbalance_ratio)}. For rare
              classes, review recall, macro-F1 and average precision rather than
              accuracy alone.
            </p>
            <div className="metrics">
              <Metric
                label="Untouched test rows"
                value={run.evaluation.split.test_rows_untouched}
              />
              <Metric
                label="Augmented train rows"
                value={run.augmentation.augmented_training_rows}
              />
              <Metric label="Export rows" value={run.export.rows} />
            </div>
            <p className="notice">{run.evaluation.interpretation}</p>
            <div className="actions">
              <button onClick={() => download("download")}>
                <Download size={16} /> Download dataset
              </button>
              <button className="secondary" onClick={() => download("report")}>
                <FileText size={16} /> Download JSON report
              </button>
            </div>
            <p className="warning">
              Synthetic data is not guaranteed realistic, unique, private, or
              better. Validate the results for your domain.
            </p>
          </section>
        )}
      {run &&
        run.status === "completed" &&
        run.evaluation &&
        run.task_type !== "regression" && (
          <section className="panel">
            <div className="heading">
              <span>04</span>
              <div>
                <h2>Before & after accuracy</h2>
                <p>
                  Baseline versus augmented training, measured on the same
                  untouched test set. The Orchestrator compares multiple
                  classifiers across repeated validation splits, targets 80%
                  with 90% as a stretch goal when feasible, and keeps the
                  untouched test set out of model selection.
                </p>
              </div>
            </div>
            <div className="metrics accuracy-metrics">
              <Metric
                label="Before · baseline accuracy"
                value={`${(run.evaluation.baseline.accuracy * 100).toFixed(2)}%`}
              />
              <Metric
                label="After · augmented accuracy"
                value={`${(run.evaluation.augmented.accuracy * 100).toFixed(2)}%`}
              />
              <Metric
                label="Accuracy change"
                value={`${run.evaluation.delta.accuracy >= 0 ? "+" : ""}${(run.evaluation.delta.accuracy * 100).toFixed(2)} pp`}
              />
              <Metric
                label="Iterations performed"
                value={`${run.iteration_search?.completed_iterations ?? run.iteration_search?.iterations_run ?? 0} / ${run.iteration_search?.planned_iterations ?? run.iteration_search?.iteration_budget ?? run.iteration_search?.max_iterations ?? 60}`}
              />
              <Metric
                label="80% validation target"
                value={
                  run.iteration_search?.target_reached
                    ? "Reached"
                    : "Not reached"
                }
              />
            </div>
            <div className="card">
              <h3>Pipeline iteration history</h3>
              <div className="tablewrap">
                <table>
                  <thead>
                    <tr>
                      <th>Iteration</th>
                      <th>Strategy</th>
                      <th>Model</th>
                      <th>Config size</th>
                      <th>Validation accuracy</th>
                      <th>Selection score</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(run.iteration_search?.history || []).map((item) => (
                      <tr key={item.iteration}>
                        <td>{item.iteration}</td>
                        <td>{item.strategy}</td>
                        <td>{item.model_name || "RandomForest"}</td>
                        <td>{item.n_estimators}</td>
                        <td>
                          {item.validation_accuracy == null
                            ? "Failed"
                            : `${(item.validation_accuracy * 100).toFixed(2)}%`}
                        </td>
                        <td>
                          {item.selection_score == null
                            ? "N/A"
                            : `${(item.selection_score * 100).toFixed(2)}% ${item.selection_metric || ""}`}
                        </td>
                        <td>
                          {item.status === "target_reached"
                            ? "Target reached"
                            : item.error
                              ? "Error"
                              : "Evaluated"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="notice">
                {run.iteration_search?.note ||
                  "The 80% target and 90% stretch goal are not guaranteed. Final test metrics are reported only after model selection."}
              </p>
              {run.iteration_search?.llm_search_guidance?.rationale && (
                <p className="notice">
                  Ollama search guidance:{" "}
                  {run.iteration_search.llm_search_guidance.rationale}
                </p>
              )}
            </div>
          </section>
        )}
    </>
  );
}

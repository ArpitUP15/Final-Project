import { useEffect, useRef, useState } from "react";
import { Activity, Upload, Workflow } from "lucide-react";
import PageHeading from "./PageHeading.jsx";
import DatasetProfile from "./DatasetProfile.jsx";
import RunReport from "./RunReport.jsx";
import useAugmentationRun from "../hooks/useAugmentationRun.js";
import { api, errorMessage, validateFile } from "../lib/api.js";

export default function DataAugmentationWorkspace({ type }) {
  const isText = type === "text";
  const title = isText ? "Text" : "Table";
  const extensions = isText
    ? [".txt", ".csv", ".xlsx", ".xlsm"]
    : [".csv", ".xlsx", ".xlsm"];
  const [file, setFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [columns, setColumns] = useState([]);
  const [preview, setPreview] = useState([]);
  const [target, setTarget] = useState("");
  const [format, setFormat] = useState("original");
  const [iterationCap, setIterationCap] = useState("60");
  const [inspecting, setInspecting] = useState(false);
  const [error, setError] = useState("");
  const analysisController = useRef(null);
  const job = useAugmentationRun(type);
  const busy = inspecting || job.busy;
  const plainText = file?.name.toLowerCase().endsWith(".txt");
  const corpus =
    isText &&
    (plainText ||
      columns.length === 1 ||
      analysis?.task_type === "text_corpus");

  useEffect(() => () => analysisController.current?.abort(), []);

  function selectFile(event) {
    const selected = event.target.files?.[0] || null;
    const validationError = validateFile(selected, extensions);
    setFile(validationError ? null : selected);
    setAnalysis(null);
    setColumns([]);
    setPreview([]);
    setTarget("");
    setFormat("original");
    setError(validationError);
    job.resetRun();
    if (validationError) event.target.value = "";
  }

  async function inspect() {
    if (!file || busy) return;
    const controller = new AbortController();
    analysisController.current = controller;
    setInspecting(true);
    setError("");
    setAnalysis(null);
    job.resetRun();
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await api.post("/api/analyze", form, {
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      if (!isText && data.columns.length < 2) {
        throw new Error(
          "A table workflow needs features and a target column. For a single text column, open the Text workspace.",
        );
      }
      if (isText && data.dataset_type !== "text") {
        throw new Error(
          "No text corpus or natural-language column was detected. Use the Table workspace for structured numeric or categorical data.",
        );
      }
      setAnalysis(data.analysis);
      setColumns(data.columns);
      setPreview(data.preview);
      const label = data.columns.find((column) =>
        /^(label|target|class|sentiment|category)$/i.test(column),
      );
      setTarget(label || data.columns.at(-1) || "");
    } catch (failure) {
      if (!controller.signal.aborted) setError(errorMessage(failure));
    } finally {
      if (!controller.signal.aborted) setInspecting(false);
    }
  }

  function execute() {
    if (!file || !target || !analysis || busy) return;
    setError("");
    const form = new FormData();
    form.append("file", file);
    form.append("target_column", target);
    form.append("output_format", format);
    form.append("max_iterations", iterationCap);
    form.append("target_accuracy", "0.80");
    form.append("stretch_target", "0.90");
    form.append("total_time_budget_seconds", "420");
    job.startRun(form);
  }

  return (
    <>
      <PageHeading type={title}>
        {isText
          ? "Upload a text corpus or labeled text data. Review its profile, generate text variants, and export the result."
          : "Upload structured data. Review its quality, select your target column, and compare augmentation and model strategies."}
      </PageHeading>
      <section className="panel" aria-busy={inspecting}>
        <div className="heading">
          <span>01</span>
          <div>
            <h2>Upload {type} dataset</h2>
            <p>
              {isText
                ? "Use one example per line in TXT, or a text column in CSV or Excel."
                : "Choose a CSV or Excel file containing features and a target column."}
            </p>
          </div>
        </div>
        <label className="drop">
          <Upload size={25} aria-hidden="true" />
          <b>{file?.name || `Select a ${type} dataset`}</b>
          <small>
            {extensions.map((value) => value.slice(1).toUpperCase()).join(", ")}
          </small>
          <input
            aria-label={`Upload ${type} dataset`}
            type="file"
            accept={extensions.join(",")}
            disabled={busy}
            onChange={selectFile}
          />
        </label>
        <div className="actions">
          <button disabled={!file || busy} onClick={inspect}>
            <Activity size={16} aria-hidden="true" />
            {inspecting ? "Analyzing dataset..." : "Analyze dataset"}
          </button>
        </div>
        {(error || job.error) && (
          <div className="error" role="alert">
            {error || job.error}
          </div>
        )}
        {job.restoring && !job.pollError && (
          <p role="status">Restoring your previous run...</p>
        )}
        {job.pollError && (
          <div className="error" role="alert">
            {job.pollError}
            <div className="actions">
              <button className="secondary" onClick={job.resumePolling}>
                Resume status updates
              </button>
            </div>
          </div>
        )}
      </section>

      {analysis && (
        <section className="panel">
          <div className="heading">
            <span>02</span>
            <div>
              <h2>{title} dataset analysis</h2>
              <p>Review the profile and configure your augmentation run.</p>
            </div>
          </div>
          <DatasetProfile
            analysis={analysis}
            columns={columns}
            preview={preview}
          />
          <fieldset disabled={busy} className="workspace-settings">
            <legend className="sr-only">{title} augmentation settings</legend>
            <div className="formrow">
              <label>
                {corpus
                  ? "Text column to augment"
                  : isText
                    ? "Label column, or text column for an unlabeled corpus"
                    : "Target / label column"}
                <select
                  value={target}
                  onChange={(event) => setTarget(event.target.value)}
                >
                  {columns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Output format
                <select
                  value={format}
                  onChange={(event) => setFormat(event.target.value)}
                >
                  <option value="original">Same as input</option>
                  <option value="csv">CSV</option>
                  <option value="xlsx">Excel (.xlsx)</option>
                </select>
              </label>
              {!corpus && (
                <label>
                  Maximum iterations
                  <select
                    value={iterationCap}
                    onChange={(event) => setIterationCap(event.target.value)}
                  >
                    <option value="10">10</option>
                    <option value="20">20</option>
                    <option value="40">40</option>
                    <option value="60">60 (maximum)</option>
                  </select>
                </label>
              )}
            </div>
          </fieldset>
          <p className="notice">
            {corpus
              ? "Text variants use the configured local language model. Without ground-truth labels, the report shows generated text counts and examples instead of classification accuracy."
              : isText
                ? "For supervised evaluation, choose the class/label column. For an unlabeled multi-column corpus, select the text column you want to augment. Text variants require the configured local language model."
                : "Classification and numeric-target regression use an untouched test partition. The search adapts to the dataset, up to the selected iteration limit. Accuracy improvement is not guaranteed."}
          </p>
          {analysis.modality === "text" && !isText && (
            <p className="warning">
              This table includes text content. For a text-focused workflow,{" "}
              <a href="#/augment/text">open Text augmentation</a>.
            </p>
          )}
          <div className="actions">
            <button disabled={!target || busy} onClick={execute}>
              <Workflow size={16} aria-hidden="true" />
              {job.busy ? "Running agents..." : `Run ${type} augmentation`}
            </button>
          </div>
        </section>
      )}

      <RunReport
        run={job.run}
        iterationCap={iterationCap}
        cancelRun={job.cancelRun}
        cancelRequested={job.cancelRequested}
      />
    </>
  );
}

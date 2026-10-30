import { useEffect, useRef, useState } from "react";
import { Activity, Download, Upload } from "lucide-react";
import PageHeading from "../components/PageHeading.jsx";
import Metric from "../components/Metric.jsx";
import BlurScreeningSummary from "../components/BlurScreeningSummary.jsx";
import ImageReport from "../components/ImageReport.jsx";
import { api, apiUrl, errorMessage, validateFile } from "../lib/api.js";

export default function ImageAugmentationPage() {
  const [file, setFile] = useState(null);
  const [task, setTask] = useState("classification");
  const [mode, setMode] = useState("offline");
  const [variants, setVariants] = useState("2");
  const [analysis, setAnalysis] = useState(null);
  const [imageReport, setImageReport] = useState(null);
  const [stage, setStage] = useState("");
  const [error, setError] = useState("");
  const requestController = useRef(null);
  const busy = Boolean(stage);

  useEffect(() => () => requestController.current?.abort(), []);

  function selectFile(event) {
    const selected = event.target.files?.[0] || null;
    const validationError = validateFile(selected, [".zip"]);
    setFile(validationError ? null : selected);
    setAnalysis(null);
    setImageReport(null);
    setError(validationError);
    if (validationError) event.target.value = "";
  }

  async function inspect() {
    if (!file || busy) return;
    const controller = new AbortController();
    requestController.current = controller;
    setStage("analyzing");
    setError("");
    setAnalysis(null);
    setImageReport(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await api.post("/api/analyze", form, {
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      setAnalysis(data.analysis);
      if (
        ["classification", "detection", "segmentation"].includes(
          data.analysis.task_type,
        )
      )
        setTask(data.analysis.task_type);
    } catch (failure) {
      if (!controller.signal.aborted) setError(errorMessage(failure));
    } finally {
      if (!controller.signal.aborted) setStage("");
    }
  }

  async function execute() {
    if (!file || !analysis || busy) return;
    const controller = new AbortController();
    requestController.current = controller;
    setStage("augmenting");
    setError("");
    setImageReport(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("task_type", task);
      form.append("mode", mode);
      form.append("variants_per_image", variants);
      form.append("seed", "42");
      const { data } = await api.post("/api/image-augmentation", form, {
        timeout: 600000,
        signal: controller.signal,
      });
      if (!controller.signal.aborted) setImageReport(data);
    } catch (failure) {
      if (!controller.signal.aborted) setError(errorMessage(failure));
    } finally {
      if (!controller.signal.aborted) setStage("");
    }
  }

  return (
    <>
      <PageHeading type="Image">
        Upload a ZIP of your image dataset. Inspect its structure, choose
        task-aware transforms, and download the augmented images and report.
      </PageHeading>
      <section className="panel" aria-busy={busy}>
        <div className="heading">
          <span>01</span>
          <div>
            <h2>Upload image dataset</h2>
            <p>
              Keep your training images, class folders, and annotations together
              in a ZIP.
            </p>
          </div>
        </div>
        <label className="drop">
          <Upload size={25} aria-hidden="true" />
          <b>{file?.name || "Select an image dataset ZIP"}</b>
          <small>ZIP containing JPG, PNG, or other supported image files</small>
          <input
            type="file"
            aria-label="Upload image dataset"
            accept=".zip,application/zip"
            disabled={busy}
            onChange={selectFile}
          />
        </label>
        <div className="actions">
          <button disabled={!file || busy} onClick={inspect}>
            <Activity size={16} aria-hidden="true" />
            {stage === "analyzing" ? "Analyzing images..." : "Analyze dataset"}
          </button>
        </div>
        {error && (
          <div className="error" role="alert">
            {error}
          </div>
        )}
        <details className="dataset-layouts">
          <summary>Supported ZIP layouts</summary>
          <ul>
            <li>
              Classification: <code>train/class_name/image.jpg</code>; optional{" "}
              <code>val/class_name/</code> and <code>test/class_name/</code>.
            </li>
            <li>
              YOLO detection: <code>images/train/name.jpg</code> with{" "}
              <code>labels/train/name.txt</code>.
            </li>
            <li>
              Segmentation: <code>images/train/name.png</code> with{" "}
              <code>masks/train/name.png</code>.
            </li>
          </ul>
        </details>
      </section>

      {analysis && (
        <section className="panel">
          <div className="heading">
            <span>02</span>
            <div>
              <h2>Image dataset analysis</h2>
              <p>
                Review the profile and choose the task that matches your
                annotations.
              </p>
            </div>
          </div>
          <div className="metrics">
            <Metric label="Detected modality" value={analysis.modality} />
            <Metric label="Likely task" value={analysis.task_type} />
            <Metric label="Image files" value={analysis.image_count} />
            <Metric label="Valid images" value={analysis.valid_image_count} />
            <Metric
              label="Annotation files"
              value={analysis.annotation_file_count}
            />
          </div>
          <p className="notice">{analysis.summary}</p>
          <BlurScreeningSummary blurScreening={analysis.blur_screening} />
          {analysis.corrupt_images?.length > 0 && (
            <p className="warning">
              Corrupt or unreadable images: {analysis.corrupt_images.length}.
              Check the export report for skipped files.
            </p>
          )}
          <fieldset className="workspace-settings" disabled={busy}>
            <legend className="sr-only">Image augmentation settings</legend>
            <div className="image-controls">
              <label>
                Task type
                <select
                  value={task}
                  onChange={(event) => {
                    setTask(event.target.value);
                    setImageReport(null);
                  }}
                >
                  <option value="classification">Image classification</option>
                  <option value="detection">
                    Object detection (YOLO labels)
                  </option>
                  <option value="segmentation">
                    Segmentation (pixel masks)
                  </option>
                </select>
              </label>
              <label>
                Execution mode
                <select
                  value={mode}
                  onChange={(event) => {
                    setMode(event.target.value);
                    setImageReport(null);
                  }}
                >
                  <option value="offline">
                    Offline augmentation: save images
                  </option>
                  <option value="training">
                    Training-time: loader manifest
                  </option>
                </select>
              </label>
              <label>
                Variants per source
                <select
                  value={variants}
                  onChange={(event) => {
                    setVariants(event.target.value);
                    setImageReport(null);
                  }}
                >
                  {[1, 2, 3, 5, 10].map((count) => (
                    <option key={count} value={count}>
                      {count}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </fieldset>
          <p className="notice">
            {task === "classification"
              ? "Classification uses flips, small rotations, crop/zoom, brightness/contrast, noise, and light blur. Validation accuracy is compared only when a compatible train/validation class-folder split is available."
              : task === "detection"
                ? "Detection synchronizes horizontal image flips with YOLO box coordinates. Annotation validation does not measure detector accuracy; mAP and IoU require a model adapter."
                : "Segmentation synchronizes image and mask transforms using nearest-neighbour mask interpolation. Annotation validation does not measure model accuracy; mIoU and Dice require a model adapter."}
          </p>
          <div className="actions">
            <button disabled={busy} onClick={execute}>
              <Activity size={16} aria-hidden="true" />
              {stage === "augmenting"
                ? "Augmenting images..."
                : "Run image augmentation"}
            </button>
          </div>
        </section>
      )}

      {imageReport && (
        <section className="panel">
          <div className="heading">
            <span>03</span>
            <div>
              <h2>Image augmentation results</h2>
              <p>
                Review generated variants and validation details before using
                your export.
              </p>
            </div>
          </div>
          <a className="button" href={apiUrl(imageReport.download_url)}>
            <Download size={16} aria-hidden="true" />
            Download augmented ZIP
          </a>
          <ImageReport imageReport={imageReport} />
        </section>
      )}
    </>
  );
}

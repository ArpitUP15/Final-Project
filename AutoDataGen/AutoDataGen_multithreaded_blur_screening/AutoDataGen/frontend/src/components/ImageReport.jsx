import Metric from "./Metric.jsx";
import BlurScreeningSummary from "./BlurScreeningSummary.jsx";
export default function ImageReport({ imageReport }) {
  return (
    <>
      {imageReport && (
        <div className="card image-report">
          <h3>Image augmentation report</h3>
          <div className="metrics">
            <Metric
              label="Source images"
              value={imageReport.report.source_images}
            />
            <Metric
              label="Generated images"
              value={imageReport.report.generated_images}
            />
            <Metric
              label="Output files"
              value={imageReport.report.output_file_count}
            />
            <Metric label="Validation" value={imageReport.report.validation} />
          </div>
          <BlurScreeningSummary
            blurScreening={imageReport.report.blur_screening}
          />
          <p className="notice">
            Task: {imageReport.report.task_type} · Mode:{" "}
            {imageReport.report.mode} · Seed: {imageReport.report.seed}
          </p>
          {imageReport.report.note && (
            <p className="notice">{imageReport.report.note}</p>
          )}
          {imageReport.report.annotation_integrity && (
            <p className="notice">
              Annotation integrity: {imageReport.report.annotation_integrity}
            </p>
          )}
          {imageReport.report.selection && (
            <p className="notice">
              Selection decision: {imageReport.report.selection}
            </p>
          )}
          {imageReport.report.model_evaluation_scope && (
            <div className="notice">
              <b>Model evaluation scope</b>
              <p>{imageReport.report.model_evaluation_scope}</p>
              {(imageReport.report.model_evaluation_limitations || []).map(
                (item, i) => (
                  <p key={i}>• {item}</p>
                ),
              )}
            </div>
          )}
          {imageReport.report.model_evaluation?.baseline && (
            <div className="metrics">
              <Metric
                label="Baseline validation accuracy"
                value={`${(imageReport.report.model_evaluation.baseline.accuracy * 100).toFixed(2)}%`}
              />
              <Metric
                label="Candidate validation accuracy"
                value={`${(imageReport.report.model_evaluation.candidate.accuracy * 100).toFixed(2)}%`}
              />
              <Metric
                label="Augmentation accepted"
                value={
                  imageReport.report.model_evaluation.accepted ? "Yes" : "No"
                }
              />
            </div>
          )}
          {imageReport.report.skipped?.length > 0 && (
            <p className="warning">
              Skipped files: {imageReport.report.skipped.length}. Review
              file-level reasons in autodatagen_image_report.json inside the
              export.
            </p>
          )}
          {imageReport.report.preview_images?.length > 0 && (
            <>
              <h3>Generated image previews</h3>
              <div className="image-preview-grid">
                {imageReport.report.preview_images.map((p, i) => (
                  <div className="image-preview" key={p.file}>
                    <img src={p.data_url} alt={`Generated preview ${i + 1}`} />
                    <small>{p.file}</small>
                  </div>
                ))}
              </div>
            </>
          )}
          {imageReport.report.previews?.length > 0 && (
            <>
              <h3>Generated image manifest</h3>
              <ul>
                {imageReport.report.previews.map((p, i) => (
                  <li key={i}>
                    {p.source} → {p.generated}
                    {p.annotation ? ` · ${p.annotation}` : ""}
                    {p.mask ? ` · mask: ${p.mask}` : ""}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </>
  );
}

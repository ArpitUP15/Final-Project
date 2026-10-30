import Metric from "./Metric.jsx";

export default function BlurScreeningSummary({ blurScreening }) {
  if (!blurScreening) return null;

  return (
    <section aria-label="Visual blur screening summary">
      <h3>Visual blur screening</h3>
      <div className="metrics">
        <Metric label="Images in dataset" value={blurScreening.total_images} />
        <Metric label="Images rejected by screening" value={blurScreening.flagged_count} />
        <Metric
          label="Images removed from dataset"
          value={blurScreening.removed_from_dataset ?? blurScreening.removed_from_augmentation}
        />
        <Metric label="Unreadable images retained" value={blurScreening.unreadable_count ?? 0} />
      </div>
      <p className="notice">
        {blurScreening.enabled
          ? `Readable non-mask images with a grayscale Laplacian variance below ${blurScreening.threshold_score} are removed. Large images are downscaled to at most 256×256 for scoring. Unreadable images and masks are retained. Low-texture images may also receive a low score.`
          : "Visual blur screening has not run yet."}
      </p>
    </section>
  );
}

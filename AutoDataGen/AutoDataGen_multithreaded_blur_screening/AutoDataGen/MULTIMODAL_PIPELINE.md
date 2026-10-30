# Unified multimodal pipeline: implementation guide

## Routing flow

1. **Analysis Agent / modality router** inspects the extension and container, parses safe file structures, and produces a modality-specific profile.
2. **Strategy Agent** chooses the appropriate existing tabular/text strategy or task-specific image transform path.
3. **Execution Agent** performs tabular/text augmentation through LangGraph, or the image augmentation engine handles classification/detection/segmentation pairs.
4. **Validation and Evaluation** validates generated records/images and annotations before any model evaluation. The tabular workflow evaluates candidates on validation splits and uses an untouched test split only for final reporting. The image classification engine evaluates only when it has a compatible train/validation class-folder layout.
5. **Export Agent** returns a downloadable dataset and a report with modality, profile, strategy, augmentation details, and validation results.

## Unified API

- `POST /api/analyze` — automatic modality detection and profile.
- `POST /api/datasets/augment` — automatic dispatch for `.zip`, `.txt`, `.csv`, `.xlsx`, and `.xlsm`.
- `GET /api/datasets/augment/{run_id}/download` — fetch the result.

For supervised tabular or labeled text datasets, submit `target_column`. For an unlabeled TXT corpus or a one-column text corpus, omit it. For image ZIPs, use `image_task_type` (`classification`, `detection`, `segmentation`), `image_mode` (`offline`, `training`), and `variants_per_image`.

## Modality cues

- **Image:** ZIP with supported image files. Folder names and label/mask paths help infer classification, detection, or segmentation. The image path validates image decoding and task annotations.
- **Text:** plain-text document files, one-column text corpora, or structured files containing long natural-language columns. A labeled CSV with text features stays structured and can use a target column.
- **Tabular:** CSV/Excel with structured numeric/categorical columns and no dominant long-text features.

For ambiguous CSVs, the router returns its decision evidence and a medium-confidence text decision rather than claiming certainty.

## Validation boundary

Image decoding and annotation checks are not equivalent to model quality. Classification validation accuracy is available only when a compatible validation split exists. Detection/segmentation task-specific model metrics need their own training/evaluation adapter; annotation integrity alone is not reported as model accuracy. Synthetic outputs should always be domain-reviewed.


## Model evaluation scope and limitations

The system routes tabular, text, and image datasets through one modality-aware workflow, but model evaluation is not yet equivalent across all three modalities.

- **Tabular:** the existing workflow trains candidate models and compares supported validation metrics; final test metrics are reported separately.
- **Text:** the existing text/tabular agent workflow retains its supported strategy and evaluation behavior. Results depend on the dataset schema and whether a valid target/label is supplied.
- **Image classification:** the accuracy guard runs only when the archive contains a compatible `train/<class>/` and `val/<class>/` folder structure with at least two classes. If that split is absent, augmentation can still run but model-level acceptance is not established.
- **Object detection:** the current pipeline transforms YOLO boxes and checks annotation integrity, but does not train/evaluate a detector. A dedicated detector adapter and metrics such as mAP and IoU are still required.
- **Segmentation:** the current pipeline transforms images and masks together and checks alignment, but does not train/evaluate a segmentation model. A dedicated segmentation adapter and metrics such as mIoU or Dice are still required.

**Important:** successful image decoding, valid bounding boxes, and aligned segmentation masks are data-integrity checks—not proof of model accuracy. The exported image report and UI now state this distinction explicitly. Training-time mode currently exports a configuration manifest; it does not automatically wire a dynamic transform loader into a model's training loop.

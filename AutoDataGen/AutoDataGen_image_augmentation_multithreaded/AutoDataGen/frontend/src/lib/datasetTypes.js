export const DATASET_TYPES = [
  {
    id: "text",
    label: "Text",
    title: "Text augmentation",
    description:
      "Expand text corpora and labeled language datasets while preserving their meaning.",
    formats: "TXT, CSV, XLSX, XLSM",
    details: [
      "One text example per line in TXT files",
      "Unlabeled corpora or text with a label column",
      "Generated examples and a downloadable report",
    ],
    href: "#/augment/text",
  },
  {
    id: "table",
    label: "Table",
    title: "Table augmentation",
    description:
      "Profile structured data, compare augmentation strategies, and evaluate model performance.",
    formats: "CSV, XLSX, XLSM",
    details: [
      "Numeric and categorical columns",
      "Classification and regression workflows",
      "Adaptive search with up to 60 iterations",
    ],
    href: "#/augment/table",
  },
  {
    id: "image",
    label: "Image",
    title: "Image augmentation",
    description:
      "Create image variants with transforms that respect your dataset's annotations.",
    formats: "ZIP containing images and annotations",
    details: [
      "Classification, YOLO detection, segmentation",
      "Offline exports or training loader manifests",
      "Image previews and annotation validation",
    ],
    href: "#/augment/image",
  },
];

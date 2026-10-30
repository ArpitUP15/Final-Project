"""First-stage modality router and profiler for tabular, text, and image datasets.

The router uses file/container evidence and deterministic content inspection. It never
uses an LLM to decide whether an upload is safe to parse or which execution path to run.
"""
from __future__ import annotations
import io, json, zipfile
from pathlib import Path, PurePosixPath
from collections import Counter
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TEXT_EXTS = {".txt", ".jsonl", ".json", ".md", ".html", ".xml"}

def _text_profile(df: pd.DataFrame) -> dict:
    columns = []
    for col in df.columns:
        s = df[col].dropna().astype(str).str.strip()
        if s.empty:
            continue
        avg_len = float(s.str.len().mean())
        unique_ratio = float(s.nunique() / max(len(s), 1))
        if avg_len >= 20 or (avg_len >= 8 and unique_ratio >= .5):
            columns.append({"column": str(col), "non_empty_values": int(len(s)),
                            "average_characters": round(avg_len, 2),
                            "unique_ratio": round(unique_ratio, 4),
                            "sample_values": s.head(3).tolist()})
    return {"text_columns": columns, "text_column_names": [x["column"] for x in columns]}

def _analyze_image_zip(content: bytes) -> dict:
    from PIL import Image
    from app.image_augmentation import screen_blurry_images

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        entries = [i for i in zf.infolist() if not i.is_dir()]
        unsafe = [i.filename for i in entries if PurePosixPath(i.filename).is_absolute() or ".." in PurePosixPath(i.filename).parts]
        if unsafe:
            raise ValueError(f"Unsafe paths in ZIP: {unsafe[:3]}")
        image_entries = [i for i in entries if Path(i.filename).suffix.lower() in IMAGE_EXTS]
        if not image_entries:
            raise ValueError("ZIP archive contains no supported image files.")
        _, blur_screening = screen_blurry_images(
            {item.filename: zf.read(item) for item in image_entries}
        )
        splits, classes, image_details, corrupt = Counter(), Counter(), [], []
        annotation_files = []
        for item in entries:
            p = PurePosixPath(item.filename)
            parts = [x.lower() for x in p.parts]
            if p.suffix.lower() == ".txt" and ("labels" in parts or "label" in parts): annotation_files.append(item.filename)
            if Path(item.filename).suffix.lower() not in IMAGE_EXTS: continue
            if parts:
                splits[parts[0]] += 1
                if len(parts) >= 2 and parts[0] in {"train", "training", "val", "valid", "validation", "test"}:
                    classes[p.parts[1]] += 1
            try:
                with zf.open(item) as stream, Image.open(stream) as im:
                    im.verify()
                with zf.open(item) as stream, Image.open(stream) as im:
                    image_details.append({"file": item.filename, "width": int(im.width), "height": int(im.height), "mode": im.mode})
            except Exception as exc:
                corrupt.append({"file": item.filename, "reason": str(exc)[:180]})
        task = "detection" if annotation_files else ("segmentation" if any("mask" in PurePosixPath(n).parts for n in (i.filename for i in entries)) else "classification")
        dims = Counter(f"{x['width']}x{x['height']}" for x in image_details)
        return {"dataset_type": "image", "modality": "image", "task_type": task,
                "file_count": len(entries), "image_count": len(image_entries), "valid_image_count": len(image_details),
                "blur_screening": blur_screening,
                "corrupt_images": corrupt, "image_formats": dict(Counter(Path(i.filename).suffix.lower() for i in image_entries)),
                "dimensions_distribution": dict(dims.most_common(12)), "split_or_root_distribution": dict(splits),
                "class_or_folder_distribution": dict(classes), "annotation_file_count": len(annotation_files),
                "annotation_files_sample": annotation_files[:20], "sample_images": image_details[:12],
                "quality_checks": {"unsafe_paths": "passed", "image_decode": "passed" if not corrupt else "some images failed decoding",
                                   "annotation_pairing": "requires task-specific validation during augmentation"},
                "recommended_pipeline": ["image_analysis", "task_aware_augmentation", "image_annotation_validation", "model_evaluation_when_supported", "export"]}

def analyze_upload(filename: str, content: bytes, dataframe_reader) -> dict:
    """Return modality + detailed profile; dataframe_reader is the app's safe file reader."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        profile = _analyze_image_zip(content)
        profile["router"] = {"decision": "image", "confidence": "high", "evidence": ["ZIP container", "image files inspected"]}
        profile["summary"] = f"Detected an image dataset with {profile['image_count']} image files; likely task: {profile['task_type']}."
        return profile
    df = dataframe_reader(filename, content)
    if suffix in TEXT_EXTS or (len(df.columns) == 1 and str(df.columns[0]).lower() in {"text", "content", "sentence", "document"}):
        text_col = str(df.columns[0]) if len(df.columns) else "text"
        values = df[text_col].dropna().astype(str)
        lengths = values.str.len()
        profile = {"dataset_type": "text", "modality": "text", "task_type": "text_corpus",
                   "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
                   "text_column": text_col, "non_empty_documents": int(len(values)),
                   "empty_documents": int(df[text_col].isna().sum() + (df[text_col].astype(str).str.strip() == "").sum()),
                   "unique_document_ratio": round(float(values.nunique()/max(len(values), 1)), 4),
                   "document_length_characters": {"min": int(lengths.min()) if len(lengths) else 0,
                       "mean": round(float(lengths.mean()), 2) if len(lengths) else 0,
                       "median": float(lengths.median()) if len(lengths) else 0,
                       "max": int(lengths.max()) if len(lengths) else 0},
                   "sample_documents": values.head(5).tolist(),
                   "recommended_pipeline": ["text_analysis", "semantic_and_label_preserving_augmentation", "text_quality_validation", "evaluation_if_labels_exist", "export"]}
        profile["router"] = {"decision": "text", "confidence": "high", "evidence": [f"file extension {suffix}", "documents parsed as text"]}
        profile["summary"] = f"Detected a text corpus with {len(values)} non-empty documents."
        return profile
    from app.agents.analysis_agent import analyze_dataframe
    analysis = analyze_dataframe(df)
    text_info = _text_profile(df)
    is_text_dataset = len(df.columns) == 1 or bool(text_info["text_column_names"])
    modality = "text" if is_text_dataset else "tabular"
    analysis.update({"modality": modality, "dataset_type": modality,
                     "task_type": "supervised_text" if modality == "text" and len(df.columns) > 1 else ("text_corpus" if modality == "text" else "tabular_ml"),
                     **text_info,
                     "router": {"decision": modality, "confidence": "medium" if modality == "text" else "high",
                                "evidence": (["one-column dataset"] if len(df.columns) == 1 else (text_info["text_column_names"] and ["long/high-uniqueness natural-language column(s) detected"] or ["structured column profile detected"]))},
                     "recommended_pipeline": (["text_analysis", "strategy_selection", "text_augmentation", "semantic_validation", "evaluation", "export"] if modality == "text" else ["tabular_analysis", "strategy_selection", "tabular_augmentation", "candidate_evaluation", "quality_guard", "export"])})
    analysis["summary"] = f"Detected {modality} data with {len(df)} rows and {len(df.columns)} columns."
    return analysis

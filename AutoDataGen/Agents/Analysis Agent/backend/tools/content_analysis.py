"""Diagnostics for image and plain-text datasets."""

from collections import Counter
import hashlib
from pathlib import Path
import re


def image_analysis_tool(path: str) -> dict:
    """Inspect image integrity with Pillow and OpenCV when available."""
    result = {
        "path": Path(path).name,
        "valid": False,
        "format": None,
        "width": None,
        "height": None,
        "channels": None,
        "mode": None,
        "opencv_readable": False,
        "errors": [],
    }

    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            result.update({
                "valid": True,
                "format": image.format,
                "width": int(image.width),
                "height": int(image.height),
                "mode": image.mode,
            })
    except Exception as exc:
        result["errors"].append(f"Pillow: {exc}")

    try:
        import cv2

        cv_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if cv_image is not None:
            result["opencv_readable"] = True
            if result["channels"] is None:
                result["channels"] = 1 if len(cv_image.shape) == 2 else int(cv_image.shape[2])
    except Exception as exc:
        result["errors"].append(f"OpenCV: {exc}")

    if result["valid"] and result["channels"] is None:
        result["channels"] = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}.get(result["mode"])
    return result


def image_dataset_analysis_tool(paths: list[str], root: str) -> dict:
    """Analyze an extracted image dataset and infer labels from folders."""
    supported = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    files = [path for path in paths if Path(path).suffix.lower() in supported]
    results = [image_analysis_tool(path) for path in files]
    valid_results = [result for result in results if result["valid"] and result["opencv_readable"]]
    hashes = {}
    duplicate_count = 0
    for path, result in zip(files, results):
        if not result["valid"]:
            continue
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest in hashes:
            duplicate_count += 1
        hashes[digest] = path

    labels = []
    for path in files:
        relative_parts = Path(path).relative_to(root).parts
        labels.append(relative_parts[0] if len(relative_parts) > 1 else "unlabelled")
    label_counts = dict(Counter(labels))
    valid_label_counts = dict(Counter(
        label for label, result in zip(labels, results)
        if result["valid"] and result["opencv_readable"]
    ))
    dimensions = Counter(
        f"{result['width']}x{result['height']}"
        for result in valid_results
    )
    formats = dict(Counter(result["format"] for result in valid_results))
    channels = dict(Counter(result["channels"] for result in valid_results))
    majority_share = max(label_counts.values()) / len(labels) if labels else 0
    corrupt_files = [result["path"] for result in results if not result["valid"]]

    return {
        "total_images": len(files),
        "valid_images": len(valid_results),
        "corrupt_images": len(files) - len(valid_results),
        "duplicate_images": duplicate_count,
        "labels": label_counts,
        "valid_labels": valid_label_counts,
        "label_count": len(label_counts),
        "imbalanced": bool(label_counts and majority_share >= 0.65),
        "majority_label_share_percent": round(majority_share * 100, 2),
        "formats": formats,
        "channels": channels,
        "dimensions": dict(dimensions),
        "corrupt_files": corrupt_files[:100],
    }


def text_analysis_tool(path: str) -> dict:
    """Calculate size, length, vocabulary, and empty-line diagnostics."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    words = re.findall(r"\b[\w']+\b", text.lower(), flags=re.UNICODE)
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    vocabulary = set(words)
    frequencies = Counter(words)

    return {
        "characters": len(text),
        "lines": len(text.splitlines()),
        "non_empty_lines": len(non_empty_lines),
        "empty_lines": len(text.splitlines()) - len(non_empty_lines),
        "words": len(words),
        "unique_words": len(vocabulary),
        "vocabulary_ratio": round(len(vocabulary) / len(words), 4) if words else 0.0,
        "average_word_length": round(sum(map(len, words)) / len(words), 2) if words else 0.0,
        "average_line_length": round(len(text) / len(non_empty_lines), 2) if non_empty_lines else 0.0,
        "top_words": dict(frequencies.most_common(20)),
    }

"""
file_handler.py
----------------
Handles:
    - Saving uploaded datasets to disk
  - Validating it before we let Pandas anywhere near it

Step 2 (Validation) + Step 3 (Load) from the Agent 1 spec.
"""

import os
import pandas as pd
from pathlib import Path
import shutil
import tempfile
import zipfile

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".md", ".log", ".zip", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_ARCHIVE_FILES = 10_000
MAX_EXTRACTED_SIZE_MB = 500

MIN_ROWS = 5
MIN_COLUMNS = 1
MAX_FILE_SIZE_MB = 100


class DatasetValidationError(Exception):
    """Raised when the uploaded file fails validation."""
    pass


def save_upload(file_bytes: bytes, filename: str) -> str:
    """Save raw uploaded bytes to the uploads/ folder and return the path."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DatasetValidationError("Supported files: CSV, text, ZIP image datasets, PNG, JPG, WEBP, BMP, and TIFF.")

    # Sanitize filename to prevent path traversal
    safe_filename = Path(filename).name
    if safe_filename != filename:
        raise DatasetValidationError("Invalid filename.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise DatasetValidationError(
            f"File too large ({size_mb:.1f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB."
        )

    if len(file_bytes) == 0:
        raise DatasetValidationError("Uploaded file is empty.")

    save_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    return save_path


def extract_image_archive(path: str) -> tuple[str, list[str]]:
    """Safely extract a ZIP and return its root plus image files."""
    extraction_dir = tempfile.mkdtemp(prefix="image_dataset_", dir=UPLOAD_DIR)
    try:
        with zipfile.ZipFile(path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) > MAX_ARCHIVE_FILES:
                raise DatasetValidationError(f"ZIP contains too many files. Maximum: {MAX_ARCHIVE_FILES}.")
            extracted_size = sum(member.file_size for member in members)
            if extracted_size > MAX_EXTRACTED_SIZE_MB * 1024 * 1024:
                raise DatasetValidationError("ZIP expands beyond the 500 MB safety limit.")
            root = Path(extraction_dir).resolve()
            safe_members = []
            for member in members:
                target = (root / member.filename).resolve()
                if os.path.commonpath([str(root), str(target)]) != str(root):
                    raise DatasetValidationError("ZIP contains an unsafe path.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, open(target, "wb") as destination:
                    shutil.copyfileobj(source, destination)
                if target.suffix.lower() in IMAGE_EXTENSIONS:
                    safe_members.append(str(target))
    except zipfile.BadZipFile as exc:
        shutil.rmtree(extraction_dir, ignore_errors=True)
        raise DatasetValidationError(f"Invalid ZIP archive: {exc}")
    except Exception:
        shutil.rmtree(extraction_dir, ignore_errors=True)
        raise
    if not safe_members:
        shutil.rmtree(extraction_dir, ignore_errors=True)
        raise DatasetValidationError("ZIP does not contain supported image files.")
    return extraction_dir, safe_members


def load_and_validate(path: str) -> pd.DataFrame:
    """
    Load CSV with Pandas and run structural validation.
    Raises DatasetValidationError on any failure.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DatasetValidationError(f"Pandas could not read this CSV: {e}")

    if df.shape[0] < MIN_ROWS:
        raise DatasetValidationError(
            f"Dataset has too few rows ({df.shape[0]}). Minimum required: {MIN_ROWS}."
        )

    if df.shape[1] < MIN_COLUMNS:
        raise DatasetValidationError("Dataset has no usable columns.")

    return df

import io
import zipfile

from PIL import Image, ImageDraw, ImageFilter

import pytest

from app.image_augmentation import (
    ImageAugmentationDataset,
    augment_image_zip,
    screen_blurry_images,
)


def _image(color, mode="RGB", size=(24, 24), textured=True):
    buffer = io.BytesIO()
    image = Image.new(mode, size, color)
    if mode == "RGB" and textured:
        draw = ImageDraw.Draw(image)
        darker = tuple(max(channel - 48, 0) for channel in color)
        for y in range(0, size[1], 4):
            for x in range(0, size[0], 4):
                if (x // 4 + y // 4) % 2:
                    draw.rectangle((x, y, x + 3, y + 3), fill=darker)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _blurred_image(color, size=(24, 24)):
    with Image.open(io.BytesIO(_image(color, size=size))) as image:
        buffer = io.BytesIO()
        image.filter(ImageFilter.GaussianBlur(radius=6)).save(buffer, format="PNG")
        return buffer.getvalue()


def _zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_classification_generates_and_evaluates_variants():
    entries = {}
    for split in ("train", "val"):
        for label, color in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
            for i in range(2 if split == "train" else 1):
                entries[f"{split}/{label}/{i}.png"] = _image(color, textured=False)
    output, report = augment_image_zip(
        _zip(entries), "classification", "offline", 1, 7, blur_threshold=0
    )
    with zipfile.ZipFile(io.BytesIO(output)) as archive:
        assert len([n for n in archive.namelist() if "__aug" in n and n.endswith(".png")]) == 4
        assert "autodatagen_image_report.json" in archive.namelist()
    assert report["model_evaluation"]["metric"] == "validation_accuracy"
    assert report["generated_images"] in (0, 4)


def test_detection_keeps_yolo_labels_paired():
    output, report = augment_image_zip(_zip({
        "images/train/a.png": _image((50, 80, 120)),
        "labels/train/a.txt": b"0 0.5 0.5 0.25 0.25\n",
    }), "detection", "offline", 1, 3)
    with zipfile.ZipFile(io.BytesIO(output)) as archive:
        assert "images/train/a__aug1.png" in archive.namelist()
        label = archive.read("labels/train/a__aug1.txt").decode().strip().split()
        assert len(label) == 5
        assert 0 <= float(label[1]) <= 1
    assert report["annotation_integrity"].startswith("validated")


def test_segmentation_keeps_matching_mask_and_image():
    output, report = augment_image_zip(_zip({
        "images/train/a.png": _image((50, 80, 120)),
        "masks/train/a.png": _image(1, mode="L"),
    }), "segmentation", "offline", 1, 3)
    with zipfile.ZipFile(io.BytesIO(output)) as archive:
        assert "images/train/a__aug1.png" in archive.namelist()
        assert "masks/train/a__aug1.png" in archive.namelist()
    assert report["generated_images"] == 1


def test_training_dataset_augments_dynamically():
    dataset = ImageAugmentationDataset([(_image((10, 20, 30)), "class-a")], seed=2)
    image, label = dataset[0]
    assert image.size == (24, 24)
    assert label == "class-a"


def test_visual_blur_screening_removes_blurred_images_and_retains_sharp_unreadable_and_masks():
    files = {
        "sharp.png": _image((100, 100, 100), size=(64, 64)),
        "blurred.png": _blurred_image((100, 100, 100), size=(64, 64)),
        "broken.png": b"not a decodable image",
        "masks/mask.png": _image(1, mode="L", size=(64, 64)),
    }

    kept, report = screen_blurry_images(files, threshold_score=100)

    assert "sharp.png" in kept
    assert "blurred.png" not in kept
    assert "broken.png" in kept
    assert "masks/mask.png" in kept
    assert report["method"] == "grayscale_laplacian_variance_max_256"
    assert report["images_scored"] == 2
    assert report["flagged_count"] == 1
    assert report["unreadable_count"] == 1
    assert report["flagged_images"][0]["file"] == "blurred.png"
    assert report["flagged_images"][0]["sharpness_score"] < 100


def test_visual_blur_screening_removes_blurred_images_and_reports_scores():
    entries = {
        "train/class-a/sharp-a.png": _image((20, 20, 20), size=(96, 96)),
        "train/class-a/sharp-b.png": _image((20, 20, 20), size=(96, 96)),
        "train/class-a/blurred.png": _blurred_image((20, 20, 20), size=(96, 96)),
        "train/class-a/broken.png": b"not a decodable image",
        "notes.txt": b"keep unrelated dataset files",
    }
    output, report = augment_image_zip(_zip(entries), "classification", "offline", 1, 9,
                                        blur_filter=True, blur_threshold=100.0)
    with zipfile.ZipFile(io.BytesIO(output)) as archive:
        names = set(archive.namelist())
        assert "train/class-a/blurred.png" not in names
        assert "train/class-a/broken.png" in names
        assert "train/class-a/sharp-a.png" in names
        assert "train/class-a/sharp-b.png" in names
        assert "train/class-a/sharp-a__aug1.png" in names
        assert "train/class-a/sharp-b__aug1.png" in names
        assert archive.read("notes.txt") == b"keep unrelated dataset files"
        archived_report = __import__("json").loads(archive.read("autodatagen_image_report.json"))
        assert archived_report["blur_screening"]["method"] == "grayscale_laplacian_variance_max_256"
        assert archived_report["blur_screening"]["threshold_score"] == 100
        assert archived_report["blur_screening"]["flagged_count"] == 1
        assert archived_report["blur_screening"]["unreadable_count"] == 1
        assert archived_report["blur_screening"]["removed_from_augmentation"] == archived_report["blur_screening"]["flagged_count"]
        assert archived_report["blur_screening"]["removed_from_dataset"] == archived_report["blur_screening"]["flagged_count"]


def test_image_augmentation_cannot_bypass_blur_screening():
    with pytest.raises(ValueError, match="Visual blur screening is required"):
        augment_image_zip(_zip({"train/class-a/image.png": _image((20, 40, 60))}),
                          "classification", "offline", 1, 9, blur_filter=False)


def test_training_mode_exports_only_visually_sharp_screened_images():
    output, report = augment_image_zip(_zip({
        "train/class-a/sharp-a.png": _image((10, 20, 30), size=(96, 96)),
        "train/class-a/sharp-b.png": _image((10, 20, 30), size=(96, 96)),
        "train/class-a/blurred.png": _blurred_image((10, 20, 30), size=(96, 96)),
    }), "classification", "training", 1, 9)

    with zipfile.ZipFile(io.BytesIO(output)) as archive:
        names = set(archive.namelist())
        assert "train/class-a/sharp-a.png" in names
        assert "train/class-a/sharp-b.png" in names
        assert "train/class-a/blurred.png" not in names
        assert "autodatagen_augmentation_manifest.json" in names
        assert "train/class-a/sharp-a__aug1.png" not in names
    assert report["blur_screening"]["removed_from_dataset"] == 1


def test_visual_blur_screening_removes_paired_detection_label_only():
    output, _ = augment_image_zip(_zip({
        "images/train/standard.png": _image((50, 80, 120), size=(96, 96)),
        "labels/train/standard.txt": b"0 0.5 0.5 0.25 0.25\n",
        "images/train/blurred.png": _blurred_image((50, 80, 120), size=(96, 96)),
        "labels/train/blurred.txt": b"0 0.5 0.5 0.25 0.25\n",
        "metadata.txt": b"keep",
    }), "detection", "offline", 1, 9)

    with zipfile.ZipFile(io.BytesIO(output)) as archive:
        names = set(archive.namelist())
        assert "images/train/blurred.png" not in names
        assert "labels/train/blurred.txt" not in names
        assert "images/train/standard.png" in names
        assert "labels/train/standard.txt" in names
        assert "metadata.txt" in names


def test_visual_blur_screening_removes_paired_segmentation_mask_only():
    output, _ = augment_image_zip(_zip({
        "images/train/standard.png": _image((50, 80, 120), size=(96, 96)),
        "masks/train/standard.png": _image(1, mode="L", size=(96, 96)),
        "images/train/blurred.png": _blurred_image((50, 80, 120), size=(96, 96)),
        "masks/train/blurred.png": _image(2, mode="L", size=(96, 96)),
    }), "segmentation", "offline", 1, 9)

    with zipfile.ZipFile(io.BytesIO(output)) as archive:
        names = set(archive.namelist())
        assert "images/train/blurred.png" not in names
        assert "masks/train/blurred.png" not in names
        assert "images/train/standard.png" in names
        assert "masks/train/standard.png" in names


def test_image_analysis_includes_visual_blur_screening_counts():
    from app.agents.multimodal_analysis_agent import _analyze_image_zip

    profile = _analyze_image_zip(_zip({
        "train/class-a/sharp.png": _image((10, 20, 30), size=(96, 96)),
        "train/class-a/blurred.png": _blurred_image((10, 20, 30), size=(96, 96)),
    }))

    assert profile["blur_screening"]["total_images"] == 2
    assert profile["blur_screening"]["flagged_count"] == 1
    assert profile["blur_screening"]["removed_from_augmentation"] == profile["blur_screening"]["flagged_count"]

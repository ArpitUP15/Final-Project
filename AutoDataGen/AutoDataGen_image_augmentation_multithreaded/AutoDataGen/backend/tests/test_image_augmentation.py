import io
import zipfile

from PIL import Image

from app.image_augmentation import augment_image_zip, ImageAugmentationDataset


def _image(color, mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, (24, 24), color).save(buffer, format="PNG")
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
                entries[f"{split}/{label}/{i}.png"] = _image(color)
    output, report = augment_image_zip(_zip(entries), "classification", "offline", 1, 7)
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

"""Task-aware image augmentation for classification, YOLO detection, and segmentation.

ZIP layout:
 classification: train/<class>/*.jpg and val/<class>/*.jpg (or <class>/*.jpg)
 detection: images/{train,val}/* plus labels/{train,val}/*.txt (YOLO normalized xywh)
 segmentation: images/{train,val}/* plus masks/{train,val}/* (same stem; mask pixels are class IDs)
"""
from __future__ import annotations
import io, json, random, zipfile, math, base64
from pathlib import Path, PurePosixPath
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def _safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts

def _transform(img: Image.Image, rng: random.Random, *, mask=False):
    """Shared spatial transforms; masks use nearest-neighbour interpolation."""
    original = img
    w, h = img.size
    # Random horizontal flip is exact for all task types.
    flipped = rng.random() < 0.5
    if flipped:
        img = ImageOps.mirror(img)
    # Small rotations preserve canvas size. For masks use nearest interpolation.
    angle = rng.uniform(-12, 12)
    if abs(angle) > 2:
        img = img.rotate(angle, resample=Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC,
                         expand=False, fillcolor=0 if mask else (0, 0, 0))
    # Small crop/zoom with resize back to original dimensions.
    if rng.random() < 0.55 and w > 8 and h > 8:
        scale = rng.uniform(0.82, 0.98)
        cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
        left, top = rng.randint(0, w-cw), rng.randint(0, h-ch)
        img = img.crop((left, top, left+cw, top+ch)).resize(
            (w, h), Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC)
    if not mask:
        if rng.random() < 0.75:
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.78, 1.22))
        if rng.random() < 0.65:
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.82, 1.18))
        if rng.random() < 0.25:
            arr = np.asarray(img.convert("RGB")).astype(np.int16)
            noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 5, arr.shape)
            img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
        if rng.random() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 0.6)))
    return img, flipped

def _read_zip(data: bytes):
    zf = zipfile.ZipFile(io.BytesIO(data))
    files = {}
    for info in zf.infolist():
        if info.is_dir(): continue
        if not _safe_name(info.filename):
            raise ValueError(f"Unsafe path in ZIP: {info.filename}")
        files[info.filename] = zf.read(info.filename)
    return files

def _is_image(name): return Path(name).suffix.lower() in IMAGE_EXTS

def _image_bytes(img, suffix):
    buf = io.BytesIO()
    fmt = {".jpg":"JPEG",".jpeg":"JPEG",".webp":"WEBP",".bmp":"BMP",".tif":"TIFF",".tiff":"TIFF"}.get(suffix.lower(),"PNG")
    if fmt == "JPEG" and img.mode not in ("RGB","L"): img = img.convert("RGB")
    img.save(buf, format=fmt, quality=94 if fmt == "JPEG" else None)
    return buf.getvalue()


def _classification_metrics(files, *, augmented: bool):
    """Fast, bounded RGB-pixel baseline used only as a lightweight validation guard.

    Evaluation is deliberately capped so a large ZIP does not spend most of its
    runtime fitting a diagnostic model. The report records when sampling occurs.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    max_train = 300
    max_val = 200
    train_rows=[]; val_rows=[]
    for name, content in files.items():
        if not _is_image(name): continue
        parts=PurePosixPath(name).parts
        if len(parts)<3: continue
        split=parts[0].lower()
        if split in {"train","training"}:
            if "__aug" in Path(name).stem and not augmented: continue
            train_rows.append((name, content, parts[1]))
        elif split in {"val","valid","validation"}:
            val_rows.append((name, content, parts[1]))
    if not train_rows or not val_rows:
        return None
    # Stable sampling keeps evaluation cost bounded and reproducible.
    sample_rng = random.Random(2026)
    if len(train_rows) > max_train:
        train_rows = sample_rng.sample(train_rows, max_train)
    if len(val_rows) > max_val:
        val_rows = sample_rng.sample(val_rows, max_val)
    classes=sorted({label for _, _, label in train_rows})
    if len(classes)<2 or not {label for _, _, label in val_rows}.issubset(set(classes)):
        return None

    def feature(content):
        with Image.open(io.BytesIO(content)) as im:
            im=im.convert("RGB").resize((16,16),Image.Resampling.BILINEAR)
            return np.asarray(im,dtype=np.uint8).reshape(-1)
    x_train=np.stack([feature(content) for _,content,_ in train_rows])
    y_train=[label for _,_,label in train_rows]
    x_val=np.stack([feature(content) for _,content,_ in val_rows])
    y_val=[label for _,_,label in val_rows]
    model=RandomForestClassifier(n_estimators=30, max_depth=16, random_state=2026,
                                 n_jobs=_worker_count(), class_weight="balanced")
    model.fit(x_train,y_train)
    return {"accuracy":float(accuracy_score(y_val,model.predict(x_val))),
            "validation_images":len(val_rows),"training_images":len(train_rows),
            "evaluation_sampled":True,"feature_size":"16x16 RGB",
            "note":"Fast sampled pixel baseline; not a domain-specialized model."}


def _worker_count(requested=None):
    """Keep parallel image transforms bounded to avoid memory/CPU oversubscription."""
    import os
    cpu_count = os.cpu_count() or 2
    if requested is None:
        requested = min(8, cpu_count)
    return max(1, min(int(requested), 8, cpu_count))


def _augment_classification(files, count, seed, max_workers=None):
    from concurrent.futures import ThreadPoolExecutor
    out = dict(files); previews=[]; skipped=[]
    images = [n for n in files if _is_image(n) and len(PurePosixPath(n).parts) >= 2]
    train = [n for n in images if PurePosixPath(n).parts[0].lower() in {"train","training"}]
    if not train:
        train = [n for n in images if PurePosixPath(n).parts[0].lower() not in {"val","valid","validation","test","holdout"}]
    if not train: raise ValueError("Classification ZIP needs class folders, e.g. train/cats/img.jpg.")

    def process(item):
        index, name = item
        try:
            with Image.open(io.BytesIO(files[name])) as im:
                im = im.convert("RGB")
            generated = []
            rng = random.Random((int(seed) + index * 1_000_003) % (2**32))
            for i in range(count):
                aug, _ = _transform(im.copy(), rng)
                target = str(PurePosixPath(name).with_name(f"{Path(name).stem}__aug{i+1}{Path(name).suffix or '.png'}"))
                generated.append((target, _image_bytes(aug, Path(target).suffix)))
            return name, generated, None
        except Exception as exc:
            return name, [], str(exc)

    with ThreadPoolExecutor(max_workers=_worker_count(max_workers), thread_name_prefix="autodatagen-img") as pool:
        # executor.map preserves source ordering for reproducible reports.
        for name, generated, error in pool.map(process, enumerate(train)):
            if error:
                skipped.append({"file": name, "reason": error})
                continue
            for target, content in generated:
                out[target] = content
                if len(previews) < 6: previews.append({"source": name, "generated": target})
    generated_count = sum(1 for name in out if "__aug" in Path(name).stem and _is_image(name))
    return out, {"generated_images":generated_count,"source_images":len(train),"skipped":skipped,"previews":previews,
                 "parallel_workers":_worker_count(max_workers)}


def _augment_detection(files, count, seed, max_workers=None):
    from concurrent.futures import ThreadPoolExecutor
    out=dict(files); skipped=[]; previews=[]
    label_paths=[n for n in files if n.lower().endswith(".txt") and PurePosixPath(n).parts[0].lower() not in {"."}]
    image_index={}
    for image_path in files:
        if _is_image(image_path):
            image_index.setdefault(Path(image_path).stem, []).append(image_path)
    pairs=[]
    for label in label_paths:
        p=PurePosixPath(label)
        if p.parent.name.lower() not in {"labels","label"} and "labels" not in p.parts: continue
        candidates=[n for n in image_index.get(p.stem, [])
                    if ("images" in PurePosixPath(n).parts or PurePosixPath(n).parent==p.parent.parent)]
        if candidates: pairs.append((label,candidates[0]))

    def process(item):
        index, (label, image_name) = item
        p=PurePosixPath(label)
        try:
            rows=[]
            for line in files[label].decode("utf-8").splitlines():
                bits=line.split()
                if not bits: continue
                if len(bits)!=5: raise ValueError("Expected YOLO rows: class x_center y_center width height")
                cls=int(float(bits[0])); x,y,w,h=map(float,bits[1:])
                if not all(math.isfinite(v) for v in (x,y,w,h)) or min(x,y,w,h)<0 or max(x,y,w,h)>1 or w<=0 or h<=0 or x-w/2<0 or x+w/2>1 or y-h/2<0 or y+h/2>1:
                    raise ValueError("Invalid/out-of-bounds YOLO box")
                rows.append([cls,x,y,w,h])
            with Image.open(io.BytesIO(files[image_name])) as im: im=im.convert("RGB")
            generated=[]
            for i in range(count):
                aug, transformed_rows=transform_detection_pair(im, rows, seed=(int(seed)+index*1_000_003+i*97)%(2**32))
                newrows=[f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls,x,y,w,h in transformed_rows]
                image_target=str(PurePosixPath(image_name).with_name(f"{Path(image_name).stem}__aug{i+1}{Path(image_name).suffix}"))
                label_target=str(PurePosixPath(label).with_name(f"{p.stem}__aug{i+1}.txt"))
                generated.append((image_target,_image_bytes(aug,Path(image_target).suffix)))
                generated.append((label_target,("\n".join(newrows)+("\n" if newrows else "")).encode()))
            return image_name, generated, None, label
        except Exception as exc:
            return image_name, [], str(exc), label

    with ThreadPoolExecutor(max_workers=_worker_count(max_workers), thread_name_prefix="autodatagen-img") as pool:
        for image_name, generated, error, label in pool.map(process, enumerate(pairs)):
            if error:
                skipped.append({"file": image_name, "reason": error})
                continue
            for target, content in generated: out[target]=content
            if len(previews)<6:
                previews.append({"source":image_name,"generated":str(PurePosixPath(image_name).with_name(f"{Path(image_name).stem}__aug1{Path(image_name).suffix}")),"annotation":str(PurePosixPath(label).with_name(f"{Path(label).stem}__aug1.txt"))})
    generated_count=sum(1 for n in out if "__aug" in Path(n).stem and _is_image(n))
    if generated_count==0: raise ValueError("No matching image/YOLO-label pairs found. Expected images/train/x.jpg and labels/train/x.txt.")
    return out, {"generated_images":generated_count,"source_images":len(pairs),"skipped":skipped,"previews":previews,
                 "annotation_format":"YOLO normalized xywh","annotation_integrity":"validated; horizontal flips update x_center",
                 "parallel_workers":_worker_count(max_workers)}


def _augment_segmentation(files, count, seed, max_workers=None):
    from concurrent.futures import ThreadPoolExecutor
    out=dict(files); skipped=[]; previews=[]
    masks=[n for n in files if _is_image(n) and ("masks" in PurePosixPath(n).parts or "mask" in PurePosixPath(n).parts)]
    image_index={}
    for image_path in files:
        if _is_image(image_path): image_index.setdefault(Path(image_path).stem, []).append(image_path)
    pairs=[]
    for mask_name in masks:
        p=PurePosixPath(mask_name)
        candidates=[n for n in image_index.get(p.stem, [])
                    if n != mask_name and ("images" in PurePosixPath(n).parts or "image" in PurePosixPath(n).parts)]
        if candidates: pairs.append((mask_name,candidates[0]))

    def process(item):
        index, (mask_name,image_name)=item
        p=PurePosixPath(mask_name)
        try:
            with Image.open(io.BytesIO(files[image_name])) as im: im=im.convert("RGB")
            with Image.open(io.BytesIO(files[mask_name])) as m: mask=m.copy()
            if im.size != mask.size: raise ValueError(f"Image/mask dimensions differ: {im.size} vs {mask.size}")
            generated=[]
            for i in range(count):
                shared_seed=(int(seed)+index*1_000_003+i*97)%(2**32)
                aug, aug_mask=transform_segmentation_pair(im,mask,seed=shared_seed)
                if aug.size!=aug_mask.size: raise ValueError("Augmented image/mask shape mismatch")
                image_target=str(PurePosixPath(image_name).with_name(f"{Path(image_name).stem}__aug{i+1}{Path(image_name).suffix}"))
                mask_target=str(PurePosixPath(mask_name).with_name(f"{p.stem}__aug{i+1}{Path(mask_name).suffix}"))
                generated.extend([(image_target,_image_bytes(aug,Path(image_target).suffix)),
                                  (mask_target,_image_bytes(aug_mask,Path(mask_target).suffix))])
            return image_name,generated,None,mask_name
        except Exception as exc:
            return image_name,[],str(exc),mask_name

    with ThreadPoolExecutor(max_workers=_worker_count(max_workers), thread_name_prefix="autodatagen-img") as pool:
        for image_name,generated,error,mask_name in pool.map(process,enumerate(pairs)):
            if error:
                skipped.append({"file":image_name,"reason":error})
                continue
            for target,content in generated: out[target]=content
            if len(previews)<6:
                previews.append({"source":image_name,"generated":str(PurePosixPath(image_name).with_name(f"{Path(image_name).stem}__aug1{Path(image_name).suffix}")),"mask":str(PurePosixPath(mask_name).with_name(f"{Path(mask_name).stem}__aug1{Path(mask_name).suffix}"))})
    generated_count=sum(1 for n in out if "__aug" in Path(n).stem and _is_image(n) and "mask" not in PurePosixPath(n).parts and "masks" not in PurePosixPath(n).parts)
    if generated_count==0: raise ValueError("No matching image/mask pairs found. Expected images/train/x.png and masks/train/x.png with identical stems and dimensions.")
    return out, {"generated_images":generated_count,"source_images":len(pairs),"skipped":skipped,"previews":previews,
                 "annotation_integrity":"validated; masks use nearest-neighbour interpolation and remain pixel-aligned",
                 "parallel_workers":_worker_count(max_workers)}


class ImageAugmentationDataset:
    """Minimal indexable classification dataset that augments dynamically on access.

    Pass records as [(image_path_or_bytes, class_label), ...]. Each __getitem__ call
    applies a fresh transform; no augmented files are written to disk.
    """
    def __init__(self, records, seed=42, enabled=True):
        self.records = list(records)
        self.rng = random.Random(seed)
        self.enabled = enabled

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        source, label = self.records[index]
        if isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        else:
            raw = Path(source).read_bytes()
        with Image.open(io.BytesIO(raw)) as im:
            image = im.convert("RGB")
        if self.enabled:
            image, _ = _transform(image, self.rng)
        return image.copy(), label


def transform_detection_pair(image: Image.Image, yolo_rows, seed=42):
    """Apply a label-safe horizontal flip and return updated normalized YOLO boxes."""
    rng = random.Random(seed)
    flipped = rng.random() < 0.5
    transformed = ImageOps.mirror(image) if flipped else image.copy()
    rows = []
    for cls, x, y, w, h in yolo_rows:
        rows.append((int(cls), 1.0-float(x) if flipped else float(x), float(y), float(w), float(h)))
    return transformed, rows


def transform_segmentation_pair(image: Image.Image, mask: Image.Image, seed=42):
    """Apply identical geometry to image and mask; mask labels stay discrete."""
    if image.size != mask.size:
        raise ValueError("Image and segmentation mask must have identical dimensions.")
    # The same seed guarantees identical flip/rotation/crop decisions.
    seed = int(seed)
    transformed_image, _ = _transform(image.copy(), random.Random(seed), mask=False)
    transformed_mask, _ = _transform(mask.copy(), random.Random(seed), mask=True)
    if transformed_image.size != transformed_mask.size:
        raise ValueError("Image/mask alignment failed after augmentation.")
    return transformed_image, transformed_mask

def augment_image_zip(data: bytes, task_type="classification", mode="offline", variants_per_image=2, seed=42, max_workers=None):
    if task_type not in {"classification","detection","segmentation"}: raise ValueError("task_type must be classification, detection, or segmentation")
    if mode not in {"offline","training"}: raise ValueError("mode must be offline or training")
    if not 1 <= int(variants_per_image) <= 10: raise ValueError("variants_per_image must be between 1 and 10")
    files=_read_zip(data)
    source_file_count = len(files)
    if not any(_is_image(n) for n in files): raise ValueError("ZIP contains no supported images.")
    # Training-time mode records augmentation configuration and leaves source data intact.
    # The same engine/config can be invoked dynamically by a training data loader.
    if mode=="training":
        manifest={"task_type":task_type,"mode":"training-time","seed":int(seed),"transforms":["horizontal_flip","small_rotation","random_crop_zoom","brightness","contrast","gaussian_noise","light_blur"],"note":"Dynamic transforms are applied by the training loader; no derived image files are saved in this mode."}
        files["autodatagen_augmentation_manifest.json"]=json.dumps(manifest,indent=2).encode()
        stats={"generated_images":0,"source_images":sum(_is_image(n) for n in files),"mode":"training-time","manifest":"autodatagen_augmentation_manifest.json","note":"This ZIP contains the original data plus a reproducible augmentation manifest. Use AutoDataGen's training loader to apply transforms dynamically."}
    elif task_type=="classification":
        original_files=dict(files)
        baseline=_classification_metrics(original_files,augmented=False)
        files,stats=_augment_classification(files,int(variants_per_image),int(seed),max_workers)
        candidate=_classification_metrics(files,augmented=True)
        if baseline is not None and candidate is not None:
            accepted=candidate["accuracy"] >= baseline["accuracy"]
            stats["model_evaluation"]={"metric":"validation_accuracy","baseline":baseline,"candidate":candidate,
                                       "accepted":accepted,"rule":"candidate validation accuracy must not regress"}
            if not accepted:
                files={name:content for name,content in files.items() if "__aug" not in Path(name).stem}
                stats["rejected_images"]=stats.get("generated_images",0)
                stats["accepted_images"]=0
                stats["generated_images"]=0
                stats["selection"]="rejected: validation accuracy regressed; original dataset retained"
                stats["previews"]=[]
            else:
                stats["accepted_images"]=stats.get("generated_images",0)
                stats["rejected_images"]=0
                stats["selection"]="accepted: validation accuracy did not regress"
        else:
            stats["model_evaluation"]={"available":False,"reason":"A train/<class>/ and val/<class>/ split with at least two classes is required for the image accuracy guard."}
            stats["selection"]="image integrity validated; model accuracy guard not run (validation split unavailable)"
    elif task_type=="detection": files,stats=_augment_detection(files,int(variants_per_image),int(seed),max_workers)
    else: files,stats=_augment_segmentation(files,int(variants_per_image),int(seed),max_workers)
    # Validate image bytes and dimensions for every generated image.
    for name in files:
        if "__aug" in name and _is_image(name):
            with Image.open(io.BytesIO(files[name])) as im: im.verify()
    stats.update({"task_type":task_type,"mode":mode,"parallel_processing": mode == "offline","validation":"passed" if not stats.get("skipped") else "passed_with_skips","source_file_count":source_file_count,"output_file_count":len(files),"seed":int(seed)})
    # Explicitly separate data/annotation validation from model-level evaluation.
    if task_type == "classification":
        stats["model_evaluation_scope"] = "Classification validation accuracy is compared only when a compatible train/validation class-folder layout is available."
        stats["model_evaluation_limitations"] = [
            "Without a compatible validation split, the accuracy guard is not run.",
            "A validation accuracy comparison does not prove every generated image is semantically correct."
        ]
    else:
        stats["model_evaluation_scope"] = "Task-specific model training/evaluation is not performed for detection or segmentation; annotation checks are integrity checks only."
        stats["model_evaluation_limitations"] = [
            "Object detection needs a detection training/evaluation adapter and metrics such as mAP/IoU.",
            "Segmentation needs a segmentation training/evaluation adapter and metrics such as mIoU/Dice.",
            "Valid bounding boxes or masks do not prove model accuracy."
        ]
    if mode == "training":
        stats["training_mode_limitation"] = "This mode exports a configuration manifest; it does not itself connect a dynamic transform pipeline to a model training loop."
    files["autodatagen_image_report.json"]=json.dumps(stats,indent=2).encode()
    buf=io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name,content in files.items(): zf.writestr(name,content)
    # Small inline previews let the frontend show actual generated samples without
    # requiring the user to download/extract the entire dataset first.
    preview_images=[]
    for item in stats.get("previews",[])[:4]:
        name=item.get("generated")
        if name in files and _is_image(name):
            try:
                with Image.open(io.BytesIO(files[name])) as im:
                    im=im.convert("RGB"); im.thumbnail((256,256))
                    pb=io.BytesIO(); im.save(pb,format="JPEG",quality=75)
                    preview_images.append({"file":name,"data_url":"data:image/jpeg;base64,"+base64.b64encode(pb.getvalue()).decode("ascii")})
            except Exception:
                pass
    stats["preview_images"]=preview_images
    return buf.getvalue(),stats

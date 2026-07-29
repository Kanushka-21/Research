"""
Offline class-balancing augmentation for the 'defect' class only.

Why this exists (2026-07-29): after deleting the irrelevant/mismatched defect
images and adding newly self-captured, whole-tomato-box-annotated defect
photos, the v6 export (tomato_project.v6-version_05.yolov8) is cleaner but
much smaller for 'defect' than the other 4 classes:

    train instances:  breaker 1137  defect 448  green 1380  red 1434  turning 1138

Ultralytics doesn't do per-class balanced sampling -- it samples per IMAGE, so
a class that appears in far fewer images just gets seen far less often per
epoch, independent of the on-the-fly hsv/rotate/flip/mosaic/mixup augmentation
already configured in train_local.py (that augmentation diversifies each
sample, it doesn't change how often a class is sampled).

This script generates extra offline copies of the defect-only training images
(every defect image in this export contains ONLY defect boxes, confirmed by
inspection) using bbox-aware geometric + light color transforms, and writes
them to a SEPARATE sibling folder -- it never touches the v6 export itself,
and never touches valid/ or test/ (touching those would leak augmented
near-duplicates into the evaluation split and inflate reported metrics).

Output: dataset/tomato_defect_augmented_v6/{images,labels}/, referenced by
tomato_project.v6-version_05.yolov8/data_balanced.yaml as a second train path
alongside the untouched original train/images.

Usage:
    python augment_defect_class.py --copies 2
"""

import argparse
import os
from pathlib import Path

import albumentations as A
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFECT_CLASS_ID = 1  # matches EXPECTED_CLASS_ORDER in train_local.py: breaker=0, defect=1, ...

V6_DIR = Path(r"D:\Research\dataset\tomato_project.v6-version_05.yolov8")
OUT_DIR = Path(r"D:\Research\dataset\tomato_defect_augmented_v6")


def build_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-20, 20), scale=(0.85, 1.15), translate_percent=(0.0, 0.08), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.5),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(std_range=(0.04, 0.1), p=1.0),
                ],
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def load_yolo_labels(label_path: Path):
    boxes, classes = [], []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        boxes.append([x, y, w, h])
        classes.append(cls_id)
    return boxes, classes


def write_yolo_labels(label_path: Path, boxes, classes):
    lines = []
    for (x, y, w, h), cls_id in zip(boxes, classes):
        lines.append(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=2,
                         help="Augmented copies to generate per original defect image")
    args = parser.parse_args()

    train_img_dir = V6_DIR / "train" / "images"
    train_label_dir = V6_DIR / "train" / "labels"

    out_img_dir = OUT_DIR / "images"
    out_label_dir = OUT_DIR / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    # Clean out any previous run's generated files so re-running with a different
    # --copies value doesn't leave stale extras mixed in.
    for existing in list(out_img_dir.iterdir()) + list(out_label_dir.iterdir()):
        existing.unlink()

    transform = build_transform()

    defect_images = []
    for img_path in train_img_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = train_label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        boxes, classes = load_yolo_labels(label_path)
        if classes and all(c == DEFECT_CLASS_ID for c in classes):
            defect_images.append((img_path, label_path, boxes, classes))

    print(f"Found {len(defect_images)} defect-only training images in {train_img_dir}")
    print(f"Generating {args.copies} augmented copies each -> up to {len(defect_images) * args.copies} new images")

    written = 0
    skipped_empty = 0
    for img_path, label_path, boxes, classes in defect_images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] could not read {img_path.name}, skipping")
            continue

        for copy_idx in range(args.copies):
            result = transform(image=image, bboxes=boxes, class_labels=classes)
            new_boxes, new_classes = result["bboxes"], result["class_labels"]
            if not new_boxes:
                # augmentation pushed the tomato fully out of frame -- discard, don't
                # write an image with an empty label (that would silently become a
                # negative/background example instead of a defect example)
                skipped_empty += 1
                continue

            out_stem = f"{img_path.stem}_aug{copy_idx}"
            out_img_path = out_img_dir / f"{out_stem}{img_path.suffix}"
            out_label_path = out_label_dir / f"{out_stem}.txt"

            cv2.imwrite(str(out_img_path), result["image"])
            write_yolo_labels(out_label_path, new_boxes, new_classes)
            written += 1

    print(f"\nWrote {written} augmented images to {out_img_dir}")
    if skipped_empty:
        print(f"Skipped {skipped_empty} augmentations that pushed the box out of frame")

    new_instance_count = sum(
        len(load_yolo_labels(p)[1]) for p in out_label_dir.iterdir() if p.suffix == ".txt"
    )
    print(f"New defect instances added: {new_instance_count}")
    print("(train/valid/test in the v6 export itself were NOT modified)")


if __name__ == "__main__":
    main()

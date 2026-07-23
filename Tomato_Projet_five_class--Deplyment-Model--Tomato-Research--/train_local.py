"""
Local YOLOv8m training script for the 5-class tomato model (breaker, defect, green, red, turning).

Rewritten from kaggle_unified_5class_.ipynb to fix three problems found in the Kaggle pipeline:
  1. No pre-training dataset audit (class-order mismatch and cross-split duplicates go undetected).
  2. No test-set evaluation -- the Kaggle notebook only ever reports validation-split metrics,
     which is the same split used for early-stopping/checkpoint selection (optimistic bias).
  3. Kaggle-specific config (dual-GPU, batch=32, /kaggle/input path autodetect) that doesn't
     transfer to a single 8GB local GPU.

Usage (PowerShell):
    python train_local.py --data "C:\path\to\new_roboflow_export\data.yaml"

Run the audit alone first, before committing to a multi-hour training run:
    python train_local.py --data "C:\path\to\data.yaml" --audit-only
"""

import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# Must match the CLASS_NAMES mapping hardcoded in every deployment script
# (realtime_classifier_*.py, streamlit_*.py, database.py). If your new Roboflow
# export does not produce this exact order, either fix it in Roboflow's class
# list or you WILL silently mislabel every prediction after deployment.
EXPECTED_CLASS_ORDER = ["breaker", "defect", "green", "red", "turning"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_data_yaml(data_yaml_path: Path) -> dict:
    with open(data_yaml_path, "r") as f:
        return yaml.safe_load(f)


def resolve_split_dirs(data_yaml_path: Path, data_cfg: dict) -> dict:
    """Roboflow YOLO exports use relative paths (../train/images etc.) resolved
    relative to the data.yaml location, not the CWD."""
    base = data_yaml_path.parent
    splits = {}
    for split in ("train", "val", "test"):
        rel = data_cfg.get(split)
        if rel is None:
            continue
        img_dir = (base / rel).resolve()
        splits[split] = img_dir
    return splits


def file_md5(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(data_yaml_path: Path) -> bool:
    """Returns True if the dataset passes all checks, False if it should not be trained on yet."""
    print("=" * 70)
    print("DATASET AUDIT")
    print("=" * 70)

    if not data_yaml_path.exists():
        print(f"[FAIL] data.yaml not found at {data_yaml_path}")
        return False

    data_cfg = load_data_yaml(data_yaml_path)
    names = data_cfg.get("names")
    if isinstance(names, dict):
        # Roboflow sometimes exports names as {0: 'breaker', 1: 'defect', ...}
        names_list = [names[i] for i in sorted(names.keys())]
    else:
        names_list = list(names)

    print(f"\nnc (num classes) in data.yaml: {data_cfg.get('nc')}")
    print(f"Class order in data.yaml:      {names_list}")
    print(f"Expected class order:          {EXPECTED_CLASS_ORDER}")

    ok = True
    if names_list != EXPECTED_CLASS_ORDER:
        print("[FAIL] Class order does NOT match the deployment scripts' CLASS_NAMES mapping.")
        print("       Fix the class order in Roboflow before exporting, or every prediction")
        print("       from a model trained on this export will be mislabeled downstream.")
        ok = False
    else:
        print("[PASS] Class order matches deployment scripts.")

    splits = resolve_split_dirs(data_yaml_path, data_cfg)
    if "test" not in splits:
        print("\n[FAIL] No 'test' split found in data.yaml. Your own methodology (70/10/30) "
              "requires a held-out test set for the officially reported numbers -- "
              "training will proceed but you must add a test split before publishing results.")
        ok = False

    print("\n--- Per-split, per-class instance counts ---")
    split_image_hashes = {}
    box_areas_by_class = defaultdict(list)  # relative area (w*h, normalized 0-1) per class, across all splits
    for split, img_dir in splits.items():
        label_dir = Path(str(img_dir).replace("images", "labels"))
        if not img_dir.exists():
            print(f"[FAIL] {split}: image dir missing -> {img_dir}")
            ok = False
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        class_counts = Counter()
        orphan_images = []
        empty_labels = []
        hashes = {}

        for img_path in images:
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                orphan_images.append(img_path.name)
                continue
            lines = [l for l in label_path.read_text().splitlines() if l.strip()]
            if not lines:
                empty_labels.append(img_path.name)
            for line in lines:
                parts = line.split()
                cls_id = int(parts[0])
                class_name = names_list[cls_id] if cls_id < len(names_list) else f"UNKNOWN({cls_id})"
                class_counts[class_name] += 1
                if len(parts) >= 5:
                    # YOLO format: class x_center y_center width height (all normalized 0-1)
                    box_w, box_h = float(parts[3]), float(parts[4])
                    box_areas_by_class[class_name].append(box_w * box_h)
            hashes[img_path.name] = file_md5(img_path)

        split_image_hashes[split] = hashes

        print(f"\n{split.upper()} ({len(images)} images):")
        for cls in EXPECTED_CLASS_ORDER:
            print(f"    {cls:<10} {class_counts.get(cls, 0)}")
        if orphan_images:
            print(f"  [WARN] {len(orphan_images)} images with no label file "
                  f"(e.g. {orphan_images[:3]})")
        if empty_labels:
            print(f"  [INFO] {len(empty_labels)} images with empty label files (background/negatives)")

    # Bounding-box scale consistency check across classes. Catches the specific bug where
    # one class was annotated with a tight box around a small region (e.g. only the blemish
    # on a defective tomato) while every other class was boxed around the whole fruit --
    # this mismatch is very hard to spot visually but cripples that class's detection
    # performance, because YOLO's detection heads assume roughly consistent object-scale
    # statistics within a class.
    print("\n--- Bounding-box scale consistency (mean relative box area per class) ---")
    mean_areas = {}
    for cls in EXPECTED_CLASS_ORDER:
        areas = box_areas_by_class.get(cls, [])
        if areas:
            mean_areas[cls] = sum(areas) / len(areas)
            print(f"    {cls:<10} mean area = {mean_areas[cls]:.4f}  (n={len(areas)})")
        else:
            print(f"    {cls:<10} no boxes found")

    if len(mean_areas) >= 2:
        sorted_areas = sorted(mean_areas.values())
        median_area = sorted_areas[len(sorted_areas) // 2]
        for cls, area in mean_areas.items():
            if area < 0.35 * median_area:
                print(f"  [FAIL] '{cls}' boxes average {area:.4f} relative area, far smaller than "
                      f"the median class ({median_area:.4f}). This strongly suggests inconsistent "
                      f"annotation convention (e.g. blemish-only polygon vs whole-object box) rather "
                      f"than a genuine size difference -- re-check these annotations before training.")
                ok = False

    # Exact-duplicate leakage check across splits (catches copy-paste leakage;
    # does NOT catch near-duplicate augmented variants of the same source photo --
    # verify in Roboflow that augmentation was generated AFTER the train/val/test split,
    # not before, or check visually for near-duplicates across splits).
    print("\n--- Cross-split exact-duplicate check ---")
    all_hashes = defaultdict(list)
    for split, hashes in split_image_hashes.items():
        for fname, h in hashes.items():
            all_hashes[h].append((split, fname))

    leaks = {h: locs for h, locs in all_hashes.items() if len({s for s, _ in locs}) > 1}
    if leaks:
        print(f"[FAIL] {len(leaks)} exact-duplicate images found across multiple splits:")
        for h, locs in list(leaks.items())[:5]:
            print(f"    {locs}")
        ok = False
    else:
        print("[PASS] No exact-duplicate images across splits (near-duplicates from "
              "pre-split augmentation are NOT checked here -- verify in Roboflow).")

    print("\n" + "=" * 70)
    print("AUDIT RESULT:", "PASS" if ok else "FAIL -- fix the issues above before training")
    print("=" * 70)
    return ok


def train(data_yaml_path: Path, epochs: int, batch, device: str):
    from ultralytics import YOLO
    import torch

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = YOLO("yolov8m.pt")

    # Baseline hyperparameters carried over from the known-good Kaggle run
    # (Output/kaggle/TOMATO_MODEL_RESULTS/args.yaml -> 87% mAP50), adjusted only
    # for single-GPU/8GB hardware. Everything else is left as-is deliberately --
    # change one variable at a time from here, don't re-tune blind.
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=batch,           # -1 = Ultralytics auto-batch, targets ~60% VRAM use
        device=device,          # single GPU: "0"; CPU fallback: "cpu"
        workers=4,              # lower if you see RAM pressure / dataloader stalls
        cache=False,            # avoid RAM cache on a laptop; switch to 'ram' only if you
                                 # have plenty of free system RAM and confirmed dataset fits
        patience=25,
        save=True,
        save_period=5,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        erasing=0.4,
        project="runs_local",
        name="tomato_5class_local",
        exist_ok=True,
        verbose=True,
    )
    return results


def evaluate_test_set(data_yaml_path: Path, weights_path: Path):
    """This is the step the Kaggle notebook never did. Run this and report THESE
    numbers as the official results -- not the validation-split confusion matrix
    that training produces automatically."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST-SET EVALUATION (held-out, unbiased -- report these numbers)")
    print("=" * 70)
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml_path), split="test")
    print(f"\nTest mAP50:    {metrics.box.map50:.4f}")
    print(f"Test mAP50-95: {metrics.box.map:.4f}")
    print(f"Test Precision (mean): {metrics.box.mp:.4f}")
    print(f"Test Recall (mean):    {metrics.box.mr:.4f}")
    print("\nPer-class mAP50:")
    for i, cls_name in enumerate(EXPECTED_CLASS_ORDER):
        try:
            print(f"    {cls_name:<10} {metrics.box.ap50[i]:.4f}")
        except (IndexError, AttributeError):
            pass
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the Roboflow-exported data.yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", default=-1, help="-1 for auto-batch (recommended on an unknown 8GB GPU)")
    parser.add_argument("--device", default="0", help='"0" for single GPU, "cpu" for CPU-only')
    parser.add_argument("--audit-only", action="store_true", help="Run the dataset audit and exit, don't train")
    args = parser.parse_args()

    data_yaml_path = Path(args.data).resolve()

    passed = audit_dataset(data_yaml_path)
    if args.audit_only:
        sys.exit(0 if passed else 1)

    if not passed:
        print("\nAudit failed -- refusing to start a multi-hour training run on a flagged "
              "dataset. Fix the issues above, or re-run with --audit-only to iterate, "
              "then remove that flag to train.")
        sys.exit(1)

    train(data_yaml_path, epochs=args.epochs, batch=args.batch, device=args.device)

    best_weights = Path("runs_local/tomato_5class_local/weights/best.pt")
    if best_weights.exists():
        evaluate_test_set(data_yaml_path, best_weights)
    else:
        print(f"\n[WARN] Expected weights not found at {best_weights} -- run test evaluation manually.")

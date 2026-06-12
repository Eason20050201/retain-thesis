"""
Train with local retain.v6 train images and evaluate on local NTU test images.

This follows the same training/evaluation pattern as dentist/NTU_test_trainV4.3.py:
  - build merged_train.txt from the local retain train folder
  - train yaml sets train/val/test to merged_train.txt
  - eval yaml sets train/val/test to the NTU test folder
  - train output folder is train_all
  - eval output folder is eval_ntu_test

Default data layout:
  data/retain.v6-traing-data/train/images        101 images
  data/NTU retain.v4-2025-12-25-retain_yolo12_test/test/images  26 images

Example:
    python -m src.ntu_train101_eval26
"""

from __future__ import annotations

import argparse
import csv
import gc
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

from src import ROOT


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

DEFAULT_TRAIN_ROOT = ROOT / "data" / "retain.v6-traing-data"
DEFAULT_TEST_ROOT = ROOT / "data" / "NTU retain.v4-2025-12-25-retain_yolo12_test"
DEFAULT_CLAHE_ROOT = ROOT / "data" / "retain.v6-traing-data-clahe"
DEFAULT_RUNS_ROOT = ROOT / "experiments" / "ntu_train101_eval26"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def list_images(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in IMG_EXTS:
        files.extend(sorted(image_dir.glob(f"*{ext}")))
    return sorted(files, key=lambda p: p.name)


def apply_clahe_bgr(img_bgr, clip_limit: float, tile_grid_size: int):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def prepare_clahe_train_dataset(
    source_root: Path,
    output_root: Path,
    clip_limit: float,
    tile_grid_size: int,
) -> Path:
    src_img_dir = source_root / "train" / "images"
    src_lbl_dir = source_root / "train" / "labels"
    dst_img_dir = output_root / "train" / "images"
    dst_lbl_dir = output_root / "train" / "labels"

    if not src_img_dir.exists():
        raise FileNotFoundError(f"Missing train images folder: {src_img_dir}")

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in list_images(src_img_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] cannot read image: {img_path}")
            continue

        enhanced = apply_clahe_bgr(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        out_img = dst_img_dir / img_path.name
        if not cv2.imwrite(str(out_img), enhanced):
            raise RuntimeError(f"Failed to write image: {out_img}")

        src_label = src_lbl_dir / f"{img_path.stem}.txt"
        if src_label.exists():
            shutil.copy2(src_label, dst_lbl_dir / src_label.name)
        count += 1

    src_yaml = source_root / "data.yaml"
    cfg = load_yaml(src_yaml) if src_yaml.exists() else {"nc": 1, "names": ["Retained"]}
    cfg["path"] = str(output_root.resolve())
    cfg["train"] = "train/images"
    cfg["val"] = "train/images"
    cfg["test"] = "train/images"
    out_yaml = output_root / "data.yaml"
    dump_yaml(cfg, out_yaml)
    print(f"[OK] CLAHE train images = {count} -> {output_root}")
    return output_root


def build_merged_train_txt(train_roots: list[Path], out_txt: Path) -> Path:
    images: list[Path] = []
    for train_root in train_roots:
        train_img_dir = train_root / "train" / "images"
        if not train_img_dir.exists():
            raise FileNotFoundError(f"Missing train images folder: {train_img_dir}")
        images.extend(list_images(train_img_dir))

    if not images:
        raise FileNotFoundError(f"No train images found in: {train_roots}")
    images = sorted(images, key=lambda p: p.name)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(str(p.resolve()) for p in images), encoding="utf-8")
    print(f"[OK] train images = {len(images)} -> {out_txt}")
    return out_txt


def make_train_yaml(train_root: Path, train_data_yaml: Path, merged_txt: Path, out_yaml: Path) -> Path:
    cfg = load_yaml(train_data_yaml)
    cfg["path"] = str(train_root.resolve())
    cfg["train"] = str(merged_txt.resolve())
    cfg["val"] = str(merged_txt.resolve())
    cfg["test"] = str(merged_txt.resolve())
    dump_yaml(cfg, out_yaml)
    return out_yaml


def make_eval_yaml(test_root: Path, test_data_yaml: Path, out_yaml: Path) -> Path:
    cfg = load_yaml(test_data_yaml)
    test_img_dir = test_root / "test" / "images"
    if not test_img_dir.exists():
        raise FileNotFoundError(f"Missing NTU test images folder: {test_img_dir}")

    cfg["path"] = str(test_root.resolve())
    cfg["train"] = "test/images"
    cfg["val"] = "test/images"
    cfg["test"] = "test/images"
    dump_yaml(cfg, out_yaml)
    print(f"[OK] NTU test images = {len(list_images(test_img_dir))} -> {out_yaml}")
    return out_yaml


def save_one_row_csv(out_csv: Path, metrics_dict: dict) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"]
    row = [
        metrics_dict.get("metrics/precision(B)"),
        metrics_dict.get("metrics/recall(B)"),
        metrics_dict.get("metrics/mAP50(B)"),
        metrics_dict.get("metrics/mAP50-95(B)"),
        metrics_dict.get("fitness"),
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)


def clear_cuda() -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "0,1" if torch.cuda.device_count() > 1 else "0"
    return "cpu"


def run_experiment(args, run_out: Path, train_yaml: Path, eval_yaml: Path) -> dict:
    exp_name = f"img{args.imgsz}_b{args.batch}_{args.optimizer}_lr{args.lr0}"
    exp_dir = run_out / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "exp_name": exp_name,
        "status": "OK",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "ntu_recall": None,
        "ntu_precision": None,
        "ntu_mAP50": None,
        "ntu_mAP50_95": None,
        "fitness": None,
        "best_pt": "",
        "ntu_csv": "",
    }

    device = resolve_device(args.device)

    try:
        model = YOLO(args.weights)
        model.train(
            data=str(train_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer=args.optimizer,
            lr0=args.lr0,
            epochs=args.epochs,
            patience=args.patience,
            workers=args.workers,
            device=device,
            project=str(exp_dir),
            name="train_all",
            pretrained=True,
            verbose=False,
        )
    except RuntimeError as e:
        msg = str(e).lower()
        row["status"] = "OOM_TRAIN" if ("out of memory" in msg or "cuda out of memory" in msg) else "RUNTIME_ERROR_TRAIN"
        clear_cuda()
        return row
    except Exception as e:
        row["status"] = f"ERROR_TRAIN: {type(e).__name__}"
        clear_cuda()
        return row

    best_pt = exp_dir / "train_all" / "weights" / "best.pt"
    row["best_pt"] = str(best_pt) if best_pt.exists() else ""
    if not best_pt.exists():
        row["status"] = "NO_BEST_PT"
        clear_cuda()
        return row

    eval_dir = exp_dir / "eval_ntu_test"
    eval_csv = eval_dir / "ntu_test_results.csv"
    row["ntu_csv"] = str(eval_csv)

    try:
        test_model = YOLO(str(best_pt))
        metrics = test_model.val(
            data=str(eval_yaml),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            project=str(exp_dir),
            name="eval_ntu_test",
            verbose=False,
            hide_labels=True,
            hide_conf=True,
        )

        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "ntu_test_results.txt").write_text(str(metrics), encoding="utf-8")

        if hasattr(metrics, "results_dict") and isinstance(metrics.results_dict, dict):
            d = metrics.results_dict
            save_one_row_csv(eval_csv, d)
            row["ntu_precision"] = d.get("metrics/precision(B)")
            row["ntu_recall"] = d.get("metrics/recall(B)")
            row["ntu_mAP50"] = d.get("metrics/mAP50(B)")
            row["ntu_mAP50_95"] = d.get("metrics/mAP50-95(B)")
            row["fitness"] = d.get("fitness")
    except RuntimeError as e:
        msg = str(e).lower()
        row["status"] += "|OOM_EVAL" if ("out of memory" in msg or "cuda out of memory" in msg) else "|RUNTIME_ERROR_EVAL"
    except Exception as e:
        row["status"] += f"|ERROR_EVAL: {type(e).__name__}"

    clear_cuda()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["original", "clahe", "mixed"], default="original")
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--clahe-root", type=Path, default=DEFAULT_CLAHE_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--weights", default="yolo12n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0014)
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid-size", type=int, default=8)
    args = parser.parse_args()

    train_yaml_src = args.train_root / "data.yaml"
    test_yaml_src = args.test_root / "data.yaml"
    if not train_yaml_src.exists():
        raise FileNotFoundError(f"Missing train data.yaml: {train_yaml_src}")
    if not test_yaml_src.exists():
        raise FileNotFoundError(f"Missing NTU test data.yaml: {test_yaml_src}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out = args.runs_root / f"{args.dataset}_train101_eval26_{stamp}"
    run_out.mkdir(parents=True, exist_ok=True)
    print("[INFO] output:", run_out)

    if args.dataset in {"clahe", "mixed"}:
        prepare_clahe_train_dataset(
            source_root=args.train_root,
            output_root=args.clahe_root,
            clip_limit=args.clip_limit,
            tile_grid_size=args.tile_grid_size,
        )

    if args.dataset == "original":
        train_roots = [args.train_root]
        template_root = args.train_root
        template_yaml = train_yaml_src
    elif args.dataset == "clahe":
        train_roots = [args.clahe_root]
        template_root = args.clahe_root
        template_yaml = args.clahe_root / "data.yaml"
    else:
        train_roots = [args.train_root, args.clahe_root]
        template_root = args.train_root
        template_yaml = train_yaml_src

    merged_txt = build_merged_train_txt(train_roots, run_out / "merged_train.txt")
    train_yaml = make_train_yaml(template_root, template_yaml, merged_txt, run_out / "retain_all_as_train.yaml")
    eval_yaml = make_eval_yaml(args.test_root, test_yaml_src, run_out / "ntu_eval.yaml")

    row = run_experiment(args, run_out, train_yaml, eval_yaml)

    summary_csv = run_out / "summary_all_results.csv"
    fieldnames = [
        "exp_name", "status", "imgsz", "batch", "optimizer", "lr0",
        "ntu_recall", "ntu_precision", "ntu_mAP50", "ntu_mAP50_95", "fitness",
        "best_pt", "ntu_csv",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})

    print("[OK] summary:", summary_csv)
    print("[OK] status:", row["status"])


if __name__ == "__main__":
    main()

"""
Standalone CLAHE experiment runner for retained-root detection.

This file intentionally does not modify the existing training scripts or dataset files.
It can:
  1. create a CLAHE-enhanced copy of a YOLO dataset,
  2. train YOLO12n on the original dataset,
  3. train YOLO12n on the CLAHE dataset,
  4. train YOLO12n on original + CLAHE images together,
  5. evaluate all experiments on the same test set.

Examples:
    python -m src.clahe_retain_experiment --mode prepare
    python -m src.clahe_retain_experiment --mode baseline
    python -m src.clahe_retain_experiment --mode clahe
    python -m src.clahe_retain_experiment --mode mixed
    python -m src.clahe_retain_experiment --mode compare
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

from src import ROOT


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_SOURCE_YAML = ROOT / "data" / "dataset.yaml"
DEFAULT_TEST_YAML = DEFAULT_SOURCE_YAML
DEFAULT_CLAHE_ROOT = ROOT / "data" / "retain_v6_clahe"
DEFAULT_RUNS_ROOT = ROOT / "experiments" / "clahe_retain"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def dataset_base(yaml_path: Path, cfg: dict) -> Path:
    base = Path(cfg.get("path", yaml_path.parent))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    return base


def resolve_split_path(split_value: str, base: Path) -> Path:
    path = Path(split_value)
    if path.is_absolute():
        return path

    resolved = base / path
    if resolved.exists():
        return resolved

    # Some local dataset.yaml files use paths relative to the repo root while
    # their "path" field is relative to the yaml location. Keep this runner
    # tolerant so we do not need to edit the dataset file itself.
    repo_relative = ROOT / path
    if repo_relative.exists():
        return repo_relative

    return resolved


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    if image_dir.is_file():
        return [
            Path(line.strip())
            for line in image_dir.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: p.name,
    )


def write_image_list(image_dirs: list[Path], output_txt: Path) -> Path:
    images: list[Path] = []
    for image_dir in image_dirs:
        images.extend(list_images(image_dir))
    images = sorted(images, key=lambda p: p.name)
    if not images:
        raise FileNotFoundError(f"No images found for merged train list: {image_dirs}")

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(str(p.resolve()) for p in images), encoding="utf-8")
    print(f"[OK] merged train+valid images: {len(images)} -> {output_txt}")
    return output_txt


def split_to_dir_name(split: str) -> str:
    return "valid" if split == "val" else split


def apply_clahe_bgr(img_bgr, clip_limit: float, tile_grid_size: int):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def process_split(
    split: str,
    source_image_dir: Path,
    output_root: Path,
    clip_limit: float,
    tile_grid_size: int,
) -> int:
    output_split = split_to_dir_name(split)
    output_image_dir = output_root / output_split / "images"
    output_label_dir = output_root / output_split / "labels"
    source_label_dir = source_image_dir.parent / "labels"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in list_images(source_image_dir):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[WARN] cannot read image: {image_path}")
            continue

        enhanced = apply_clahe_bgr(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        output_image_path = output_image_dir / image_path.name
        if not cv2.imwrite(str(output_image_path), enhanced):
            raise RuntimeError(f"Failed to write image: {output_image_path}")

        label_path = source_label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, output_label_dir / label_path.name)
        else:
            print(f"[WARN] missing label: {label_path}")
        count += 1

    return count


def make_eval_test_path(test_yaml: Path) -> Path:
    cfg = load_yaml(test_yaml)
    base = dataset_base(test_yaml, cfg)
    test_value = cfg.get("test") or cfg.get("val") or cfg.get("train")
    if not test_value:
        raise ValueError(f"No train/val/test split found in {test_yaml}")
    return resolve_split_path(test_value, base).resolve()


def make_original_train_yaml(source_yaml: Path, test_yaml: Path, output_yaml: Path) -> Path:
    cfg = load_yaml(source_yaml)
    base = dataset_base(source_yaml, cfg)
    train_path = resolve_split_path(cfg["train"], base).resolve()
    val_path = resolve_split_path(cfg.get("val", cfg["train"]), base).resolve()
    test_path = make_eval_test_path(test_yaml)
    merged_txt = write_image_list(
        [p for p in (train_path, val_path) if p.exists()],
        output_yaml.parent / "retain_v6_train_valid.txt",
    )

    out = {
        "path": str(base),
        "train": str(merged_txt),
        "val": str(merged_txt),
        "test": str(test_path),
        "nc": cfg.get("nc", 1),
        "names": cfg.get("names", ["Retained"]),
    }
    dump_yaml(out, output_yaml)
    return output_yaml


def prepare_clahe_dataset(
    source_yaml: Path,
    test_yaml: Path,
    output_root: Path,
    clip_limit: float,
    tile_grid_size: int,
) -> Path:
    cfg = load_yaml(source_yaml)
    base = dataset_base(source_yaml, cfg)
    output_root.mkdir(parents=True, exist_ok=True)

    written: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_value = cfg.get(split)
        if not split_value:
            continue
        source_image_dir = resolve_split_path(split_value, base)
        if not source_image_dir.exists():
            print(f"[WARN] split path does not exist, skipped: {source_image_dir}")
            continue
        written[split] = process_split(
            split=split,
            source_image_dir=source_image_dir,
            output_root=output_root,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )

    if "train" not in written:
        raise FileNotFoundError(f"No train images found from {source_yaml}")

    train_dirs = [output_root / "train" / "images"]
    if "val" in written:
        train_dirs.append(output_root / "valid" / "images")
    merged_txt = write_image_list(train_dirs, output_root / "train_valid.txt")

    out_cfg = {
        "path": str(output_root.resolve()),
        "train": str(merged_txt.resolve()),
        "val": str(merged_txt.resolve()),
        "test": "test/images" if "test" in written else str(make_eval_test_path(test_yaml)),
        "nc": cfg.get("nc", 1),
        "names": cfg.get("names", ["Retained"]),
    }

    output_yaml = output_root / "data.yaml"
    dump_yaml(out_cfg, output_yaml)
    print(f"[OK] CLAHE dataset yaml: {output_yaml}")
    print(f"[OK] processed splits: {written}")
    return output_yaml


def make_mixed_train_yaml(
    original_yaml: Path,
    clahe_yaml: Path,
    output_yaml: Path,
) -> Path:
    original_cfg = load_yaml(original_yaml)
    clahe_cfg = load_yaml(clahe_yaml)
    original_base = dataset_base(original_yaml, original_cfg)
    clahe_base = dataset_base(clahe_yaml, clahe_cfg)

    train_dirs = [
        resolve_split_path(original_cfg["train"], original_base),
        resolve_split_path(clahe_cfg["train"], clahe_base),
    ]
    original_val = resolve_split_path(original_cfg.get("val", original_cfg["train"]), original_base)
    clahe_val = resolve_split_path(clahe_cfg.get("val", clahe_cfg["train"]), clahe_base)
    train_dirs.extend([p for p in (original_val, clahe_val) if p.exists()])

    merged_txt = write_image_list(
        train_dirs,
        output_yaml.parent / "mixed_original_clahe_train_valid.txt",
    )

    out = {
        "path": str(ROOT),
        "train": str(merged_txt),
        "val": str(merged_txt),
        "test": original_cfg["test"],
        "nc": original_cfg.get("nc", 1),
        "names": original_cfg.get("names", ["Retained"]),
    }
    dump_yaml(out, output_yaml)
    return output_yaml


def choose_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "0,1" if torch.cuda.device_count() > 1 else "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def metrics_to_row(metrics) -> dict:
    box = metrics.box
    results = getattr(metrics, "results_dict", {}) or {}
    precision = float(box.mp)
    recall = float(box.mr)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "mAP50": round(float(box.map50), 6),
        "mAP50_95": round(float(box.map), 6),
        "f1": round(f1, 6),
        "fitness": results.get("fitness", ""),
    }


def save_scores(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "precision", "recall", "mAP50", "mAP50_95", "f1", "fitness", "best_pt"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_and_eval(
    experiment_name: str,
    data_yaml: Path,
    runs_root: Path,
    weights: str,
    epochs: int,
    batch: int,
    imgsz: int,
    lr0: float,
    optimizer: str,
    patience: int,
    workers: int,
    device: str,
) -> dict:
    run_dir = runs_root / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    train_device = choose_device(device)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        optimizer=optimizer,
        patience=patience,
        workers=workers,
        device=train_device,
        project=str(run_dir),
        name="train",
        exist_ok=True,
    )

    best_pt = run_dir / "train" / "weights" / "best.pt"
    eval_model = YOLO(str(best_pt))
    metrics = eval_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=train_device,
        project=str(run_dir),
        name="eval_test",
        exist_ok=True,
        verbose=False,
    )

    row = {"experiment": experiment_name, **metrics_to_row(metrics), "best_pt": str(best_pt)}
    save_scores(run_dir / "scores.csv", [row])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "baseline", "clahe", "mixed", "compare"], default="prepare")
    parser.add_argument("--source-yaml", type=Path, default=DEFAULT_SOURCE_YAML)
    parser.add_argument("--test-yaml", type=Path, default=DEFAULT_TEST_YAML)
    parser.add_argument("--clahe-root", type=Path, default=DEFAULT_CLAHE_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--weights", default="yolo12n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--lr0", type=float, default=0.0014)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid-size", type=int, default=8)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.runs_root.mkdir(parents=True, exist_ok=True)

    original_yaml = args.runs_root / "retain_v6_original_eval_ntu.yaml"
    make_original_train_yaml(args.source_yaml, args.test_yaml, original_yaml)

    if args.mode in {"prepare", "clahe", "mixed", "compare"}:
        clahe_yaml = prepare_clahe_dataset(
            source_yaml=args.source_yaml,
            test_yaml=args.test_yaml,
            output_root=args.clahe_root,
            clip_limit=args.clip_limit,
            tile_grid_size=args.tile_grid_size,
        )
    else:
        clahe_yaml = args.clahe_root / "data.yaml"

    mixed_yaml = args.runs_root / "retain_v6_original_plus_clahe_eval_ntu.yaml"
    if args.mode in {"mixed", "compare"}:
        make_mixed_train_yaml(original_yaml, clahe_yaml, mixed_yaml)

    if args.mode == "prepare":
        print("[OK] prepare only; no training was started.")
        return

    rows = []
    common = dict(
        runs_root=args.runs_root,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        optimizer=args.optimizer,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
    )

    if args.mode in {"baseline", "compare"}:
        rows.append(
            train_and_eval(
                experiment_name=f"baseline_original_{timestamp}",
                data_yaml=original_yaml,
                **common,
            )
        )

    if args.mode in {"clahe", "compare"}:
        rows.append(
            train_and_eval(
                experiment_name=f"clahe_clip{args.clip_limit}_tile{args.tile_grid_size}_{timestamp}",
                data_yaml=clahe_yaml,
                **common,
            )
        )

    if args.mode in {"mixed", "compare"}:
        rows.append(
            train_and_eval(
                experiment_name=f"mixed_original_clahe_clip{args.clip_limit}_tile{args.tile_grid_size}_{timestamp}",
                data_yaml=mixed_yaml,
                **common,
            )
        )

    summary_path = args.runs_root / f"summary_{timestamp}.csv"
    save_scores(summary_path, rows)
    print(f"[OK] summary: {summary_path}")


if __name__ == "__main__":
    main()

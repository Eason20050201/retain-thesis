"""
Create the formal ROI-crop dataset variant used by YOLO CV experiments.

This is the promoted version of ``roi_center_crop_preview.py``. It applies the
same energy-based center crop to train/valid/test images and remaps YOLO labels
to the cropped image coordinates.

Usage:
    python -m src.preprocess.roi_crop
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from src import ROOT
from src.preprocess.roi_center_crop_preview import expand_box, find_roi


DEFAULT_SRC_ROOT = ROOT / "data"
DEFAULT_DST_ROOT = ROOT / "data_roi_crop"
SPLITS = ["train", "valid", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def image_paths(src_root: Path, split: str) -> list[Path]:
    image_dir = src_root / split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"missing source image directory: {image_dir}")
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_for_image(image_path: Path, root: Path) -> Path:
    return root / image_path.parent.parent.name / "labels" / image_path.with_suffix(".txt").name


def parse_label_line(line: str, image_w: int, image_h: int) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0]))
    values = [float(v) for v in parts[1:]]
    if len(values) == 4:
        cx, cy, w, h = values
        return (
            class_id,
            (cx - w / 2.0) * image_w,
            (cy - h / 2.0) * image_h,
            (cx + w / 2.0) * image_w,
            (cy + h / 2.0) * image_h,
        )
    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        return class_id, min(xs) * image_w, min(ys) * image_h, max(xs) * image_w, max(ys) * image_h
    return None


def remap_label_line(
    line: str,
    image_w: int,
    image_h: int,
    crop_box: tuple[int, int, int, int],
    min_box_size: float,
) -> str | None:
    parsed = parse_label_line(line, image_w, image_h)
    if parsed is None:
        return None
    class_id, x1, y1, x2, y2 = parsed
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    new_w = crop_x2 - crop_x1
    new_h = crop_y2 - crop_y1

    inter_x1 = max(x1, crop_x1)
    inter_y1 = max(y1, crop_y1)
    inter_x2 = min(x2, crop_x2)
    inter_y2 = min(y2, crop_y2)
    if inter_x2 - inter_x1 < min_box_size or inter_y2 - inter_y1 < min_box_size:
        return None

    rel_x1 = inter_x1 - crop_x1
    rel_y1 = inter_y1 - crop_y1
    rel_x2 = inter_x2 - crop_x1
    rel_y2 = inter_y2 - crop_y1

    cx = ((rel_x1 + rel_x2) / 2.0) / new_w
    cy = ((rel_y1 + rel_y2) / 2.0) / new_h
    bw = (rel_x2 - rel_x1) / new_w
    bh = (rel_y2 - rel_y1) / new_h
    return f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def remap_labels(
    src_label: Path,
    dst_label: Path,
    image_w: int,
    image_h: int,
    crop_box: tuple[int, int, int, int],
    min_box_size: float,
) -> tuple[int, int]:
    lines = src_label.read_text().splitlines() if src_label.exists() else []
    remapped = [
        line
        for line in (
            remap_label_line(raw_line, image_w, image_h, crop_box, min_box_size)
            for raw_line in lines
            if raw_line.strip()
        )
        if line is not None
    ]
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(remapped) + ("\n" if remapped else ""))
    return len(lines), len(remapped)


def process_image(
    image_path: Path,
    split: str,
    args: argparse.Namespace,
) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_h, image_w = gray.shape[:2]
    roi_box, _, debug = find_roi(
        gray,
        args.method,
        args.min_area_frac,
        args.x_low,
        args.x_high,
        args.y_low,
        args.y_high,
    )
    x, y, w, h = roi_box
    crop_box = expand_box(x, y, w, h, image_w, image_h, args.margin)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

    dst_root = Path(args.dst_root)
    src_root = Path(args.src_root)

    dst_image = dst_root / split / "images" / image_path.name
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_image), crop)

    src_label = label_for_image(image_path, src_root)
    dst_label = label_for_image(image_path, dst_root)
    labels_in, labels_out = remap_labels(src_label, dst_label, image_w, image_h, crop_box, args.min_box_size)

    return {
        "split": split,
        "image": image_path.name,
        "method": debug["method"],
        "orig_w": image_w,
        "orig_h": image_h,
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "crop_x2": crop_x2,
        "crop_y2": crop_y2,
        "crop_w": crop_x2 - crop_x1,
        "crop_h": crop_y2 - crop_y1,
        "labels_in": labels_in,
        "labels_out": labels_out,
        "labels_dropped": labels_in - labels_out,
        "fallback": debug["fallback"],
    }


def write_metadata(rows: list[dict], args: argparse.Namespace) -> None:
    dst_root = Path(args.dst_root)
    metadata = dst_root / "roi_metadata.csv"
    if rows:
        with metadata.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    dropped = sum(int(row["labels_dropped"]) for row in rows)
    fallback = sum(1 for row in rows if row["fallback"])
    readme = f"""# ROI Crop Dataset

Formal ROI-crop dataset variant generated from `data/`.

- Source: `{Path(args.src_root).relative_to(ROOT).as_posix() if Path(args.src_root).is_relative_to(ROOT) else args.src_root}`
- Output: `{dst_root.relative_to(ROOT).as_posix() if dst_root.is_relative_to(ROOT) else dst_root}`
- Splits: `{', '.join(args.splits)}`
- Images: `{len(rows)}`
- ROI method: `{args.method}`
- Crop margin: `{args.margin}`
- Energy quantiles: `x={args.x_low}-{args.x_high}`, `y={args.y_low}-{args.y_high}`
- Min box size: `{args.min_box_size}` pixels
- Fallback full-image cases: `{fallback}`
- Dropped labels after crop clipping: `{dropped}`

Labels are remapped to cropped image coordinates and written as YOLO bbox labels.
"""
    (dst_root / "README.md").write_text(readme)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROI-cropped images and remapped YOLO labels.")
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--src-root", default=str(DEFAULT_SRC_ROOT))
    parser.add_argument("--dst-root", default=str(DEFAULT_DST_ROOT))
    parser.add_argument("--method", choices=["energy", "component"], default="energy")
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--x-low", type=float, default=0.10)
    parser.add_argument("--x-high", type=float, default=0.90)
    parser.add_argument("--y-low", type=float, default=0.08)
    parser.add_argument("--y-high", type=float, default=0.94)
    parser.add_argument("--min-area-frac", type=float, default=0.004)
    parser.add_argument("--min-box-size", type=float, default=2.0)
    parser.add_argument("--fail-on-dropped", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.src_root = str(Path(args.src_root).resolve())
    args.dst_root = str(Path(args.dst_root).resolve())
    Path(args.dst_root).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for split in args.splits:
        for image_path in image_paths(Path(args.src_root), split):
            rows.append(process_image(image_path, split, args))
    write_metadata(rows, args)
    dropped = sum(int(row["labels_dropped"]) for row in rows)
    if args.fail_on_dropped and dropped:
        raise RuntimeError(f"ROI crop dropped {dropped} labels; see {Path(args.dst_root) / 'roi_metadata.csv'}")
    print(f"Wrote {len(rows)} ROI-cropped images to {Path(args.dst_root)}")
    print(f"Metadata: {Path(args.dst_root) / 'roi_metadata.csv'}")


if __name__ == "__main__":
    main()

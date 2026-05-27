from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src import ROOT


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "valid", "test")
TARGET_ASPECT = 1.8896


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def image_paths(src_root: Path, split: str) -> list[Path]:
    image_dir = src_root / split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"missing image directory: {image_dir}")
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_for_image(image_path: Path, root: Path) -> Path:
    split = image_path.parent.parent.name
    return root / split / "labels" / f"{image_path.stem}.txt"


def parse_label_line(line: str, width: int, height: int) -> Box | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0]))
    values = [float(v) for v in parts[1:]]
    if len(values) == 4:
        cx, cy, bw, bh = values
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        x1, x2 = min(xs) * width, max(xs) * width
        y1, y2 = min(ys) * height, max(ys) * height
    else:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(class_id, x1, y1, x2, y2)


def read_boxes(label_path: Path, width: int, height: int) -> tuple[int, list[Box]]:
    if not label_path.exists():
        return 0, []
    lines = [line for line in label_path.read_text().splitlines() if line.strip()]
    boxes = [box for box in (parse_label_line(line, width, height) for line in lines) if box is not None]
    return len(lines), boxes


def box_to_yolo_line(box: Box, width: int, height: int) -> str:
    cx = ((box.x1 + box.x2) / 2.0) / width
    cy = ((box.y1 + box.y2) / 2.0) / height
    bw = (box.x2 - box.x1) / width
    bh = (box.y2 - box.y1) / height
    return f"{box.class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def border_fill_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    color = np.median(border, axis=0).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def pad_to_aspect(
    image: Image.Image,
    boxes: list[Box],
    target_aspect: float = TARGET_ASPECT,
) -> tuple[Image.Image, list[Box], dict]:
    image = image.convert("RGB")
    width, height = image.size
    current_aspect = width / height
    pad_left = pad_right = pad_top = pad_bottom = 0
    if current_aspect < target_aspect:
        new_width = int(round(height * target_aspect))
        total_pad = max(0, new_width - width)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
    elif current_aspect > target_aspect:
        new_height = int(round(width / target_aspect))
        total_pad = max(0, new_height - height)
        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top

    padded = ImageOps.expand(
        image,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=border_fill_color(image),
    )
    remapped = [
        Box(box.class_id, box.x1 + pad_left, box.y1 + pad_top, box.x2 + pad_left, box.y2 + pad_top)
        for box in boxes
    ]
    meta = {
        "orig_w": width,
        "orig_h": height,
        "out_w": padded.width,
        "out_h": padded.height,
        "orig_aspect": current_aspect,
        "out_aspect": padded.width / padded.height,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
    }
    return padded, remapped, meta


def percentile_normalize_array(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    arr_f = arr.astype(np.float32)
    if arr_f.ndim == 2:
        lo = float(np.percentile(arr_f, low))
        hi = float(np.percentile(arr_f, high))
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.uint8)
        out = (np.clip(arr_f, lo, hi) - lo) * (255.0 / (hi - lo))
        return out.round().astype(np.uint8)

    out = np.empty_like(arr_f)
    for channel in range(arr_f.shape[2]):
        ch = arr_f[:, :, channel]
        lo = float(np.percentile(ch, low))
        hi = float(np.percentile(ch, high))
        if hi <= lo:
            out[:, :, channel] = 0
        else:
            out[:, :, channel] = (np.clip(ch, lo, hi) - lo) * (255.0 / (hi - lo))
    return out.round().astype(np.uint8)


def percentile_normalize_image(image: Image.Image, low: float = 1.0, high: float = 99.0) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    return Image.fromarray(percentile_normalize_array(arr, low=low, high=high), mode="RGB")


def write_label(path: Path, boxes: list[Box], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [box_to_yolo_line(box, width, height) for box in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def process_image(image_path: Path, split: str, src_root: Path, dst_root: Path, mode: str, target_aspect: float) -> dict:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        label_count, boxes = read_boxes(label_for_image(image_path, src_root), width, height)

        meta = {
            "orig_w": width,
            "orig_h": height,
            "out_w": width,
            "out_h": height,
            "orig_aspect": width / height,
            "out_aspect": width / height,
            "pad_left": 0,
            "pad_right": 0,
            "pad_top": 0,
            "pad_bottom": 0,
        }
        out_image = image
        out_boxes = boxes
        if mode in {"aspect", "aspect_pnorm"}:
            out_image, out_boxes, meta = pad_to_aspect(out_image, out_boxes, target_aspect=target_aspect)
        if mode in {"pnorm", "aspect_pnorm"}:
            out_image = percentile_normalize_image(out_image)

    dst_image = dst_root / split / "images" / image_path.name
    dst_label = dst_root / split / "labels" / f"{image_path.stem}.txt"
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    out_image.save(dst_image)
    write_label(dst_label, out_boxes, out_image.width, out_image.height)
    return {
        "split": split,
        "image": image_path.name,
        "mode": mode,
        **meta,
        "labels_in": label_count,
        "labels_out": len(out_boxes),
        "labels_dropped": label_count - len(out_boxes),
    }


def write_metadata(dst_root: Path, rows: list[dict], mode: str, target_aspect: float) -> None:
    metadata = dst_root / "domain_alignment_metadata.csv"
    if rows:
        with metadata.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    dropped = sum(int(row["labels_dropped"]) for row in rows)
    readme = f"""# Domain Alignment Dataset

Image-only domain alignment dataset generated from `data/`.

- Mode: `{mode}`
- Target aspect: `{target_aspect}`
- Images: `{len(rows)}`
- Dropped labels: `{dropped}`

Labels are converted to YOLO bounding boxes and remapped only when geometry changes.
No test labels are used to choose preprocessing parameters.
"""
    (dst_root / "README.md").write_text(readme)


def generate_variant(
    src_root: Path | str = ROOT / "data",
    dst_root: Path | str = ROOT / "data_aspect188",
    mode: str = "aspect",
    splits: list[str] | tuple[str, ...] = SPLITS,
    target_aspect: float = TARGET_ASPECT,
    fail_on_dropped: bool = False,
) -> list[dict]:
    src_root = repo_path(src_root)
    dst_root = repo_path(dst_root)
    if mode not in {"aspect", "pnorm", "aspect_pnorm"}:
        raise ValueError(f"unknown domain alignment mode: {mode}")
    rows: list[dict] = []
    dst_root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        for image_path in image_paths(src_root, split):
            rows.append(process_image(image_path, split, src_root, dst_root, mode, target_aspect))
    write_metadata(dst_root, rows, mode, target_aspect)
    dropped = sum(int(row["labels_dropped"]) for row in rows)
    if fail_on_dropped and dropped:
        raise RuntimeError(f"domain alignment dropped {dropped} labels; see {dst_root / 'domain_alignment_metadata.csv'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image-only domain alignment dataset variants.")
    parser.add_argument("--src-root", default="data")
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--mode", choices=["aspect", "pnorm", "aspect_pnorm"], required=True)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--target-aspect", type=float, default=TARGET_ASPECT)
    parser.add_argument("--fail-on-dropped", action="store_true")
    args = parser.parse_args()

    rows = generate_variant(
        src_root=args.src_root,
        dst_root=args.dst_root,
        mode=args.mode,
        splits=args.splits,
        target_aspect=args.target_aspect,
        fail_on_dropped=args.fail_on_dropped,
    )
    print(f"Wrote {len(rows)} images to {repo_path(args.dst_root)}")
    print(f"Metadata: {repo_path(args.dst_root) / 'domain_alignment_metadata.csv'}")


if __name__ == "__main__":
    main()

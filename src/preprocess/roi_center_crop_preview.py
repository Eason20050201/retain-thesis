"""
Create ROI center-crop preview images for panoramic dental X-rays.

This script is intentionally preview-only: it writes visual inspection assets
under reports/ and does not create a training dataset or touch locked test data
unless explicitly requested.

Usage:
    python -m src.preprocess.roi_center_crop_preview
    python -m src.preprocess.roi_center_crop_preview --splits train valid --max-images 40
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src import ROOT


SRC_ROOT = ROOT / "data"
DEFAULT_OUT = ROOT / "reports" / "roi_center_crop_preview"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def image_paths(split: str) -> list[Path]:
    image_dir = SRC_ROOT / split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"missing image directory: {image_dir}")
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def expand_box(
    x: int,
    y: int,
    w: int,
    h: int,
    image_w: int,
    image_h: int,
    margin: float,
) -> tuple[int, int, int, int]:
    pad_x = int(round(w * margin))
    pad_y = int(round(h * margin))
    x1 = clamp(x - pad_x, 0, image_w - 1)
    y1 = clamp(y - pad_y, 0, image_h - 1)
    x2 = clamp(x + w + pad_x, 1, image_w)
    y2 = clamp(y + h + pad_y, 1, image_h)
    if x2 <= x1:
        x1, x2 = 0, image_w
    if y2 <= y1:
        y1, y2 = 0, image_h
    return x1, y1, x2, y2


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def weighted_bounds(weights: np.ndarray, low_q: float, high_q: float) -> tuple[int, int, float]:
    weights = np.maximum(weights.astype(np.float64), 0.0)
    total = float(weights.sum())
    if total <= 1e-8:
        return 0, len(weights), len(weights) / 2.0

    cdf = np.cumsum(weights) / total
    low = int(np.searchsorted(cdf, low_q, side="left"))
    high = int(np.searchsorted(cdf, high_q, side="right")) + 1
    coords = np.arange(len(weights), dtype=np.float64)
    center = float((coords * weights).sum() / total)
    return clamp(low, 0, len(weights) - 1), clamp(high, 1, len(weights)), center


def enforce_min_size(
    start: int,
    end: int,
    center: float,
    min_size: int,
    limit: int,
) -> tuple[int, int]:
    if end - start >= min_size:
        return start, end

    half = min_size / 2.0
    start = int(round(center - half))
    end = start + min_size
    if start < 0:
        start = 0
        end = min(limit, min_size)
    if end > limit:
        end = limit
        start = max(0, limit - min_size)
    return start, end


def find_roi_component(gray: np.ndarray, min_area_frac: float) -> tuple[tuple[int, int, int, int], np.ndarray, dict]:
    image_h, image_w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    p_low = float(np.percentile(blurred, 2))
    p_high = float(np.percentile(blurred, 98))
    threshold = max(8.0, p_low + 0.08 * max(1.0, p_high - p_low))
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    kernel_size = max(9, int(round(min(image_w, image_h) / 55)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = int(round(image_w * image_h * min_area_frac))
    kept = np.zeros_like(mask)

    kept_labels = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            kept[labels == label] = 255
            kept_labels.append(label)

    if not kept_labels:
        return (0, 0, image_w, image_h), mask, {
            "threshold": threshold,
            "method": "component",
            "kernel_size": kernel_size,
            "components_kept": 0,
            "fallback": True,
        }

    points = cv2.findNonZero(kept)
    x, y, w, h = cv2.boundingRect(points)
    return (x, y, w, h), kept, {
        "threshold": threshold,
        "method": "component",
        "kernel_size": kernel_size,
        "components_kept": len(kept_labels),
        "fallback": False,
    }


def find_roi_energy(
    gray: np.ndarray,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
) -> tuple[tuple[int, int, int, int], np.ndarray, dict]:
    image_h, image_w = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)

    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    lap = np.abs(cv2.Laplacian(blurred, cv2.CV_32F, ksize=3))

    local_sigma = max(5.0, min(image_w, image_h) / 70.0)
    local_mean = cv2.GaussianBlur(clahe, (0, 0), sigmaX=local_sigma, sigmaY=local_sigma)
    local_contrast = np.abs(clahe.astype(np.float32) - local_mean.astype(np.float32))

    energy = 0.55 * grad + 0.25 * lap + 0.20 * local_contrast
    p99 = float(np.percentile(energy, 99))
    if p99 <= 1e-8:
        return (0, 0, image_w, image_h), np.zeros_like(gray), {
            "method": "energy",
            "threshold": 0.0,
            "kernel_size": 0,
            "components_kept": 0,
            "fallback": True,
        }
    energy = np.clip(energy / p99, 0.0, 1.0)

    border_x = max(1, int(round(image_w * 0.02)))
    border_y = max(1, int(round(image_h * 0.025)))
    energy[:, :border_x] = 0.0
    energy[:, image_w - border_x :] = 0.0
    energy[:border_y, :] = 0.0
    energy[image_h - border_y :, :] = 0.0

    x_axis = np.linspace(-1.0, 1.0, image_w, dtype=np.float32)
    y_axis = np.linspace(-1.0, 1.0, image_h, dtype=np.float32)
    center_prior = np.exp(
        -0.5
        * (
            (x_axis[None, :] / 0.72) ** 2
            + (y_axis[:, None] / 0.82) ** 2
        )
    )
    weighted = energy * (0.35 + 0.65 * center_prior)

    smooth_sigma = max(3.0, min(image_w, image_h) / 160.0)
    weighted = cv2.GaussianBlur(weighted, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)

    col_energy = smooth_1d(weighted.sum(axis=0), max(9, image_w // 90))
    row_energy = smooth_1d(weighted.sum(axis=1), max(9, image_h // 90))

    x1, x2, center_x = weighted_bounds(col_energy, x_low, x_high)
    y1, y2, center_y = weighted_bounds(row_energy, y_low, y_high)
    x1, x2 = enforce_min_size(x1, x2, center_x, int(round(image_w * 0.62)), image_w)
    y1, y2 = enforce_min_size(y1, y2, center_y, int(round(image_h * 0.58)), image_h)

    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)
    energy_vis = np.clip(weighted * 255.0, 0, 255).astype(np.uint8)
    return (x1, y1, roi_w, roi_h), energy_vis, {
        "method": "energy",
        "threshold": p99,
        "kernel_size": int(round(smooth_sigma)),
        "components_kept": 0,
        "fallback": False,
        "center_x": center_x,
        "center_y": center_y,
    }


def find_roi(
    gray: np.ndarray,
    method: str,
    min_area_frac: float,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
) -> tuple[tuple[int, int, int, int], np.ndarray, dict]:
    if method == "component":
        return find_roi_component(gray, min_area_frac)
    if method == "energy":
        return find_roi_energy(gray, x_low, x_high, y_low, y_high)
    raise ValueError(f"unknown ROI method: {method}")


def draw_preview(
    image: np.ndarray,
    roi_box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    image_name: str,
    method: str,
) -> np.ndarray:
    overlay = image.copy()
    x, y, w, h = roi_box
    cx = x + w // 2
    cy = y + h // 2
    c1, r1, c2, r2 = crop_box

    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 3)
    cv2.rectangle(overlay, (c1, r1), (c2, r2), (0, 255, 0), 3)
    cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=34, thickness=3)
    cv2.circle(overlay, (cx, cy), 8, (0, 0, 255), -1)

    label = f"{image_name} | {method} | yellow=ROI green=crop red=center"
    cv2.rectangle(overlay, (8, 8), (min(8 + 12 * len(label), overlay.shape[1] - 8), 42), (0, 0, 0), -1)
    cv2.putText(overlay, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def process_image(
    image_path: Path,
    split: str,
    output_root: Path,
    margin: float,
    min_area_frac: float,
    method: str,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_h, image_w = gray.shape[:2]
    roi_box, mask, debug = find_roi(gray, method, min_area_frac, x_low, x_high, y_low, y_high)
    x, y, w, h = roi_box
    crop_box = expand_box(x, y, w, h, image_w, image_h, margin)
    c1, r1, c2, r2 = crop_box

    crop = image[r1:r2, c1:c2]
    overlay = draw_preview(image, roi_box, crop_box, image_path.name, method)

    crop_dir = output_root / "crops" / split
    overlay_dir = output_root / "overlays" / split
    mask_dir = output_root / "masks" / split
    crop_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(crop_dir / image_path.name), crop)
    cv2.imwrite(str(overlay_dir / image_path.name), overlay)
    cv2.imwrite(str(mask_dir / image_path.name), mask)

    return {
        "split": split,
        "image": image_path.name,
        "method": debug["method"],
        "width": image_w,
        "height": image_h,
        "roi_x": x,
        "roi_y": y,
        "roi_w": w,
        "roi_h": h,
        "center_x": x + w / 2,
        "center_y": y + h / 2,
        "crop_x1": c1,
        "crop_y1": r1,
        "crop_x2": c2,
        "crop_y2": r2,
        "crop_w": c2 - c1,
        "crop_h": r2 - r1,
        "threshold": round(float(debug["threshold"]), 4),
        "kernel_size": debug["kernel_size"],
        "components_kept": debug["components_kept"],
        "fallback": debug["fallback"],
        "x_low": x_low,
        "x_high": x_high,
        "y_low": y_low,
        "y_high": y_high,
    }


def write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    path: Path,
    rows: list[dict],
    splits: list[str],
    margin: float,
    min_area_frac: float,
    method: str,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
) -> None:
    fallback_count = sum(1 for row in rows if row["fallback"])
    text = f"""# ROI Center Crop Preview

This folder is a visual preview only. It does not create a training dataset and
does not modify labels.

- Splits: `{', '.join(splits)}`
- Images: `{len(rows)}`
- ROI method: `{method}`
- Crop margin: `{margin}`
- Energy quantiles: `x={x_low}-{x_high}`, `y={y_low}-{y_high}`
- Minimum connected-component area fraction: `{min_area_frac}`
- Fallback full-image cases: `{fallback_count}`

Folders:

- `overlays/`: original image with yellow ROI bbox, green crop box, red center.
- `crops/`: cropped image preview.
- `masks/`: foreground mask or energy map used to estimate the ROI.
- `metadata.csv`: ROI/crop coordinates and debug values.

Review `overlays/` first. If the red center and green crop box look stable, this
can be promoted into a real dataset variant with label remapping.
"""
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROI center-crop preview images.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], help="Dataset splits to preview.")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Preview output directory.")
    parser.add_argument("--method", choices=["energy", "component"], default="energy", help="ROI center detection method.")
    parser.add_argument("--margin", type=float, default=0.08, help="Crop margin as a fraction of ROI bbox size.")
    parser.add_argument("--min-area-frac", type=float, default=0.004, help="Minimum foreground component area fraction.")
    parser.add_argument("--x-low", type=float, default=0.06, help="Low weighted quantile for energy ROI columns.")
    parser.add_argument("--x-high", type=float, default=0.94, help="High weighted quantile for energy ROI columns.")
    parser.add_argument("--y-low", type=float, default=0.05, help="Low weighted quantile for energy ROI rows.")
    parser.add_argument("--y-high", type=float, default=0.96, help="High weighted quantile for energy ROI rows.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images per split; 0 means all images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for split in args.splits:
        paths = image_paths(split)
        if args.max_images > 0:
            paths = paths[: args.max_images]
        for image_path in paths:
            rows.append(
                process_image(
                    image_path,
                    split,
                    output_root,
                    args.margin,
                    args.min_area_frac,
                    args.method,
                    args.x_low,
                    args.x_high,
                    args.y_low,
                    args.y_high,
                )
            )

    write_metadata(output_root / "metadata.csv", rows)
    write_readme(
        output_root / "README.md",
        rows,
        args.splits,
        args.margin,
        args.min_area_frac,
        args.method,
        args.x_low,
        args.x_high,
        args.y_low,
        args.y_high,
    )
    print(f"Wrote {len(rows)} ROI preview images to {output_root}")


if __name__ == "__main__":
    main()

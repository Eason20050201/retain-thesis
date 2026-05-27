from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from PIL import Image, ImageDraw

from src import ROOT


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "valid", "test")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def image_paths(split: str, data_root: Path | None = None) -> list[Path]:
    root = data_root or ROOT / "data"
    image_dir = root / split / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"missing image directory: {image_dir}")
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path_for_image(image_path: Path, data_root: Path | None = None) -> Path:
    root = data_root or ROOT / "data"
    split = image_path.parent.parent.name
    return root / split / "labels" / f"{image_path.stem}.txt"


def compute_image_features(image: Image.Image, black_threshold: int = 12) -> dict:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    height, width = arr.shape
    mask = arr > black_threshold
    ys, xs = np.where(mask)
    if len(xs):
        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1
        mass_cx = float(xs.mean() / width)
        mass_cy = float(ys.mean() / height)
    else:
        x1, y1, x2, y2 = 0, 0, width, height
        mass_cx, mass_cy = 0.5, 0.5

    content_w = x2 - x1
    content_h = y2 - y1
    return {
        "width": width,
        "height": height,
        "aspect": width / height,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "black_fraction": float((arr <= black_threshold).mean()),
        "content_x1": x1 / width,
        "content_y1": y1 / height,
        "content_x2": x2 / width,
        "content_y2": y2 / height,
        "content_w": content_w / width,
        "content_h": content_h / height,
        "content_area": (content_w * content_h) / (width * height),
        "mass_cx": mass_cx,
        "mass_cy": mass_cy,
    }


def parse_label_boxes(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
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
            continue
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def collect_image_rows(data_root: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    for split in SPLITS:
        for image_path in image_paths(split, data_root):
            with Image.open(image_path) as image:
                features = compute_image_features(image)
            rows.append({"split": split, "image": rel(image_path), **features})
    return rows


def collect_label_rows(data_root: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    for split in SPLITS:
        for image_path in image_paths(split, data_root):
            with Image.open(image_path) as image:
                width, height = image.size
            boxes = parse_label_boxes(label_path_for_image(image_path, data_root), width, height)
            widths = [(x2 - x1) / width for x1, _, x2, _ in boxes]
            heights = [(y2 - y1) / height for _, y1, _, y2 in boxes]
            rows.append(
                {
                    "split": split,
                    "image": rel(image_path),
                    "target_count": len(boxes),
                    "bbox_w_mean": mean(widths) if widths else 0.0,
                    "bbox_h_mean": mean(heights) if heights else 0.0,
                    "bbox_w_median": median(widths) if widths else 0.0,
                    "bbox_h_median": median(heights) if heights else 0.0,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_group(rows: list[dict], key: str) -> dict[str, list[float]]:
    groups = {"train+valid": [], "test": []}
    for row in rows:
        name = "test" if row["split"] == "test" else "train+valid"
        groups[name].append(float(row[key]))
    return groups


def summary_value(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "median": None}
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
    }


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def draw_histogram(path: Path, values_by_group: dict[str, list[float]], title: str) -> None:
    all_values = [v for values in values_by_group.values() for v in values]
    if not all_values:
        return
    width, height = 720, 360
    margin = 48
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 12), title, fill=(20, 20, 20))
    lo, hi = min(all_values), max(all_values)
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    bins = np.linspace(lo, hi, 18)
    colors = {"train+valid": (31, 119, 180), "test": (214, 39, 40)}
    max_count = 1
    histograms = {}
    for group, values in values_by_group.items():
        counts, _ = np.histogram(values, bins=bins)
        histograms[group] = counts
        max_count = max(max_count, int(counts.max()))
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    base_y = height - margin
    bar_w = plot_w / (len(bins) - 1)
    for group_idx, (group, counts) in enumerate(histograms.items()):
        color = colors.get(group, (80, 80, 80))
        offset = group_idx * bar_w * 0.38
        for idx, count in enumerate(counts):
            x1 = margin + idx * bar_w + offset
            x2 = x1 + bar_w * 0.35
            y1 = base_y - (count / max_count) * plot_h
            draw.rectangle((x1, y1, x2, base_y), fill=color)
        draw.rectangle((width - 170, 42 + group_idx * 20, width - 158, 54 + group_idx * 20), fill=color)
        draw.text((width - 152, 38 + group_idx * 20), group, fill=(20, 20, 20))
    draw.line((margin, base_y, width - margin, base_y), fill=(0, 0, 0))
    draw.line((margin, margin, margin, base_y), fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_montage(path: Path, data_root: Path | None = None, per_split: int = 6) -> None:
    thumbs = []
    for split in SPLITS:
        for image_path in image_paths(split, data_root)[:per_split]:
            with Image.open(image_path).convert("RGB") as image:
                image.thumbnail((180, 95))
                canvas = Image.new("RGB", (190, 125), "white")
                canvas.paste(image, ((190 - image.width) // 2, 22))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 4), split, fill=(20, 20, 20))
                thumbs.append(canvas)
    if not thumbs:
        return
    cols = per_split
    rows = math.ceil(len(thumbs) / cols)
    montage = Image.new("RGB", (cols * 190, rows * 125), "white")
    for idx, thumb in enumerate(thumbs):
        montage.paste(thumb, ((idx % cols) * 190, (idx // cols) * 125))
    path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(path)


def write_report(path: Path, image_rows: list[dict], label_rows: list[dict] | None, out_dir: Path) -> None:
    image_metrics = [
        ("width", "image width"),
        ("height", "image height"),
        ("aspect", "aspect ratio"),
        ("mean", "brightness mean"),
        ("std", "contrast std"),
        ("black_fraction", "black pixel fraction"),
        ("content_area", "content bbox area"),
        ("mass_cx", "content mass center x"),
        ("mass_cy", "content mass center y"),
    ]
    lines = [
        "# Train-Test Domain Gap",
        "",
        "## Image-Only Domain Gap",
        "",
        "Main analysis uses images only. Test labels are not used to choose alignment or preprocessing parameters.",
        "",
        "| metric | train+valid mean | train+valid std | test mean | test std | delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in image_metrics:
        groups = split_group(image_rows, key)
        dev = summary_value(groups["train+valid"])
        test = summary_value(groups["test"])
        delta = None if dev["mean"] is None or test["mean"] is None else test["mean"] - dev["mean"]
        lines.append(
            f"| {label} | {fmt(dev['mean'])} | {fmt(dev['std'])} | {fmt(test['mean'])} | {fmt(test['std'])} | {fmt(delta)} |"
        )

    lines.extend(
        [
            "",
            "## Visual Outputs",
            "",
            f"- Image features CSV: `{rel(out_dir / 'image_features.csv')}`",
            f"- Montage: `{rel(out_dir / 'montage.jpg')}`",
            f"- Histograms: `{rel(out_dir)}`",
            "",
            "## Label Diagnostic Appendix",
            "",
        ]
    )
    if label_rows is None:
        lines.append("Diagnostic appendix disabled; main report did not read labels.")
    else:
        lines.extend(
            [
                "This appendix is diagnostic only. These label statistics must not be used to select preprocessing parameters.",
                "",
                "| metric | train+valid mean | test mean | delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for key, label in [
            ("target_count", "targets per image"),
            ("bbox_w_mean", "bbox width ratio"),
            ("bbox_h_mean", "bbox height ratio"),
        ]:
            groups = split_group(label_rows, key)
            dev = summary_value(groups["train+valid"])
            test = summary_value(groups["test"])
            delta = None if dev["mean"] is None or test["mean"] is None else test["mean"] - dev["mean"]
            lines.append(f"| {label} | {fmt(dev['mean'])} | {fmt(test['mean'])} | {fmt(delta)} |")
        lines.append(f"\nDiagnostic CSV: `{rel(out_dir / 'label_diagnostic_features.csv')}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_analysis(
    out_dir: Path | str = ROOT / "reports" / "train_test_domain_gap",
    include_label_appendix: bool = True,
    data_root: Path | None = None,
) -> None:
    out_dir = repo_path(out_dir)
    image_rows = collect_image_rows(data_root)
    write_csv(out_dir / "image_features.csv", image_rows)
    for metric in ["aspect", "mean", "std", "black_fraction", "content_area", "mass_cy"]:
        draw_histogram(out_dir / f"hist_{metric}.png", split_group(image_rows, metric), metric)
    draw_montage(out_dir / "montage.jpg", data_root)

    label_rows = None
    if include_label_appendix:
        label_rows = collect_label_rows(data_root)
        write_csv(out_dir / "label_diagnostic_features.csv", label_rows)

    write_report(ROOT / "reports" / "TRAIN_TEST_DOMAIN_GAP.md", image_rows, label_rows, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze train/valid vs external test image-domain gap.")
    parser.add_argument("--out-dir", default="reports/train_test_domain_gap")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-label-appendix", action="store_true")
    args = parser.parse_args()

    run_analysis(
        out_dir=repo_path(args.out_dir),
        include_label_appendix=not args.no_label_appendix,
        data_root=repo_path(args.data_root),
    )
    print(f"Wrote reports/TRAIN_TEST_DOMAIN_GAP.md and {args.out_dir}")


if __name__ == "__main__":
    main()

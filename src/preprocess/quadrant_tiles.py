from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from src import ROOT
from src.tiling import assign_boxes_to_tiles, label_path_for_image, make_grid_tiles, parse_yolo_label


DEFAULT_SRC_ROOT = ROOT / "data"
DEFAULT_DST_ROOT = ROOT / "data_quadrant"
SPLITS = ["train", "valid", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


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


def read_boxes(label_path: Path, image_w: int, image_h: int):
    lines = label_path.read_text().splitlines() if label_path.exists() else []
    boxes = []
    for line in lines:
        if not line.strip():
            continue
        box = parse_yolo_label(line, image_w, image_h)
        if box is not None:
            boxes.append(box)
    return lines, boxes


def process_image(image_path: Path, split: str, args: argparse.Namespace, manifest: dict, rows: list[dict]) -> None:
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_w, image_h = image.size
        tiles = make_grid_tiles(image_w, image_h, rows=args.rows, cols=args.cols, overlap_frac=args.overlap_frac)
        src_label = label_path_for_image(image_path, src_root)
        label_lines, boxes = read_boxes(src_label, image_w, image_h)
        assigned, assigned_box_indices = assign_boxes_to_tiles(
            boxes,
            tiles,
            min_visibility=args.min_visibility,
            min_box_size=args.min_box_size,
        )

        source_rel = rel(image_path)
        manifest["tiles_by_source"][source_rel] = []
        for tile_idx, tile in enumerate(tiles):
            tile_name = f"{image_path.stem}__r{tile.row}c{tile.col}{image_path.suffix}"
            dst_image = dst_root / split / "images" / tile_name
            dst_label = dst_root / split / "labels" / Path(tile_name).with_suffix(".txt").name
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            dst_label.parent.mkdir(parents=True, exist_ok=True)

            crop = image.crop((tile.x1, tile.y1, tile.x2, tile.y2))
            crop.save(dst_image)
            labels = assigned.get(tile_idx, [])
            dst_label.write_text("\n".join(labels) + ("\n" if labels else ""))

            manifest["tiles_by_source"][source_rel].append(
                {
                    "image": rel(dst_image),
                    "crop_x1": tile.x1,
                    "crop_y1": tile.y1,
                    "crop_x2": tile.x2,
                    "crop_y2": tile.y2,
                    "orig_w": image_w,
                    "orig_h": image_h,
                    "row": tile.row,
                    "col": tile.col,
                }
            )
            rows.append(
                {
                    "split": split,
                    "source_image": source_rel,
                    "tile_image": rel(dst_image),
                    "row": tile.row,
                    "col": tile.col,
                    "crop_x1": tile.x1,
                    "crop_y1": tile.y1,
                    "crop_x2": tile.x2,
                    "crop_y2": tile.y2,
                    "orig_w": image_w,
                    "orig_h": image_h,
                    "tile_w": tile.width,
                    "tile_h": tile.height,
                    "source_labels": len(label_lines),
                    "parsed_boxes": len(boxes),
                    "tile_labels": len(labels),
                    "source_boxes_assigned": len(assigned_box_indices),
                    "source_boxes_dropped": len(boxes) - len(assigned_box_indices),
                }
            )


def write_outputs(manifest: dict, rows: list[dict], args: argparse.Namespace) -> None:
    dst_root = Path(args.dst_root)
    manifest_path = dst_root / "tile_manifest.json"
    metadata_path = dst_root / "tile_metadata.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    if rows:
        with metadata_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    source_images = len(manifest["tiles_by_source"])
    tile_images = sum(len(v) for v in manifest["tiles_by_source"].values())
    dropped = sum(int(row["source_boxes_dropped"]) for row in rows)
    readme = f"""# Quadrant Tiling Dataset

Formal 2x2 overlapping tile dataset generated from `data/`.

- Source: `{rel(Path(args.src_root))}`
- Output: `{rel(dst_root)}`
- Source images: `{source_images}`
- Tile images: `{tile_images}`
- Grid: `{args.rows}x{args.cols}`
- Overlap fraction: `{args.overlap_frac}`
- Min visibility: `{args.min_visibility}`
- Min box size: `{args.min_box_size}` pixels
- Dropped source boxes: `{dropped}`

Labels are clipped to tile coordinates and written as YOLO bbox labels. Inference
must merge tile predictions back to the original image coordinates before scoring
against `data/test`.
"""
    (dst_root / "README.md").write_text(readme)
    if args.fail_on_dropped and dropped:
        raise RuntimeError(f"quadrant tiling dropped {dropped} source boxes; see {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2x2 overlapping quadrant tiles with remapped labels.")
    parser.add_argument("--src-root", default=str(DEFAULT_SRC_ROOT))
    parser.add_argument("--dst-root", default=str(DEFAULT_DST_ROOT))
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--overlap-frac", type=float, default=0.20)
    parser.add_argument("--min-visibility", type=float, default=0.25)
    parser.add_argument("--min-box-size", type=float, default=2.0)
    parser.add_argument("--fail-on-dropped", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.src_root = str(Path(args.src_root).resolve())
    args.dst_root = str(Path(args.dst_root).resolve())
    manifest = {
        "source_root": rel(Path(args.src_root)),
        "tile_root": rel(Path(args.dst_root)),
        "rows": args.rows,
        "cols": args.cols,
        "overlap_frac": args.overlap_frac,
        "min_visibility": args.min_visibility,
        "min_box_size": args.min_box_size,
        "tiles_by_source": {},
    }
    rows: list[dict] = []
    for split in args.splits:
        for image_path in image_paths(Path(args.src_root), split):
            process_image(image_path, split, args, manifest, rows)
    write_outputs(manifest, rows, args)
    print(f"Wrote {sum(len(v) for v in manifest['tiles_by_source'].values())} tile images to {Path(args.dst_root)}")
    print(f"Manifest: {Path(args.dst_root) / 'tile_manifest.json'}")


if __name__ == "__main__":
    main()

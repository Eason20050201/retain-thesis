from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class LabeledBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_grid_tiles(width: int, height: int, rows: int = 2, cols: int = 2, overlap_frac: float = 0.20) -> list[Tile]:
    if rows != 2 or cols != 2:
        raise ValueError("v1 quadrant tiling supports rows=2 and cols=2 only")
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if not 0 <= overlap_frac < 0.8:
        raise ValueError("overlap_frac must be in [0, 0.8)")

    half_overlap_x = int(round(width * overlap_frac / 2.0))
    half_overlap_y = int(round(height * overlap_frac / 2.0))
    mid_x = width // 2
    mid_y = height // 2
    x_ranges = [(0, min(width, mid_x + half_overlap_x)), (max(0, mid_x - half_overlap_x), width)]
    y_ranges = [(0, min(height, mid_y + half_overlap_y)), (max(0, mid_y - half_overlap_y), height)]

    tiles: list[Tile] = []
    for row, (y1, y2) in enumerate(y_ranges):
        for col, (x1, x2) in enumerate(x_ranges):
            tiles.append(Tile(row=row, col=col, x1=x1, y1=y1, x2=x2, y2=y2))
    return tiles


def parse_yolo_label(line: str, image_w: int, image_h: int) -> LabeledBox | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0]))
    values = [float(v) for v in parts[1:]]
    if len(values) == 4:
        cx, cy, w, h = values
        return LabeledBox(
            class_id,
            (cx - w / 2.0) * image_w,
            (cy - h / 2.0) * image_h,
            (cx + w / 2.0) * image_w,
            (cy + h / 2.0) * image_h,
        )
    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        return LabeledBox(class_id, min(xs) * image_w, min(ys) * image_h, max(xs) * image_w, max(ys) * image_h)
    return None


def intersect_box(box: LabeledBox, tile: Tile) -> LabeledBox | None:
    x1 = max(box.x1, tile.x1)
    y1 = max(box.y1, tile.y1)
    x2 = min(box.x2, tile.x2)
    y2 = min(box.y2, tile.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return LabeledBox(box.class_id, x1, y1, x2, y2)


def box_to_tile_label(box: LabeledBox, tile: Tile, min_box_size: float) -> str | None:
    rel_w = box.x2 - box.x1
    rel_h = box.y2 - box.y1
    if rel_w < min_box_size or rel_h < min_box_size:
        return None
    x1 = box.x1 - tile.x1
    y1 = box.y1 - tile.y1
    x2 = box.x2 - tile.x1
    y2 = box.y2 - tile.y1
    cx = ((x1 + x2) / 2.0) / tile.width
    cy = ((y1 + y2) / 2.0) / tile.height
    w = (x2 - x1) / tile.width
    h = (y2 - y1) / tile.height
    return f"{box.class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"


def assign_boxes_to_tiles(
    boxes: list[LabeledBox],
    tiles: list[Tile],
    min_visibility: float,
    min_box_size: float,
) -> tuple[dict[int, list[str]], set[int]]:
    assigned: dict[int, list[str]] = defaultdict(list)
    assigned_box_indices: set[int] = set()
    for box_idx, box in enumerate(boxes):
        candidates: list[tuple[float, int, LabeledBox]] = []
        for tile_idx, tile in enumerate(tiles):
            clipped = intersect_box(box, tile)
            if clipped is None:
                continue
            visibility = clipped.area / box.area if box.area > 0 else 0.0
            candidates.append((visibility, tile_idx, clipped))
            if visibility >= min_visibility:
                label = box_to_tile_label(clipped, tile, min_box_size)
                if label is not None:
                    assigned[tile_idx].append(label)
                    assigned_box_indices.add(box_idx)

        if box_idx not in assigned_box_indices and candidates:
            _, tile_idx, clipped = max(candidates, key=lambda item: item[0])
            label = box_to_tile_label(clipped, tiles[tile_idx], min_box_size)
            if label is not None:
                assigned[tile_idx].append(label)
                assigned_box_indices.add(box_idx)
    return dict(assigned), assigned_box_indices


def iou_xyxy(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
    bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_prediction_rows(rows: Iterable[dict], nms_iou: float, max_det: int) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["image_path"]), int(row["class_id"]))].append(row)

    kept: list[dict] = []
    for _, group_rows in sorted(groups.items()):
        ordered = sorted(group_rows, key=lambda row: float(row["confidence"]), reverse=True)
        group_kept: list[dict] = []
        for row in ordered:
            if all(iou_xyxy(row, prev) < nms_iou for prev in group_kept):
                group_kept.append(row)
            if len(group_kept) >= max_det:
                break
        kept.extend(group_kept)
    return sorted(kept, key=lambda row: (str(row["image_path"]), -float(row["confidence"])))


def merge_tile_predictions(prediction_rows: list[dict], tile_lookup: dict[str, dict], nms_iou: float, max_det: int) -> list[dict]:
    restored: list[dict] = []
    for row in prediction_rows:
        tile_image = str(row["image_path"])
        info = tile_lookup.get(tile_image)
        if info is None:
            raise KeyError(f"missing tile mapping for prediction image: {tile_image}")
        orig_w = float(info["orig_w"])
        orig_h = float(info["orig_h"])
        x1 = clamp(float(row["x1"]) + float(info["crop_x1"]), 0.0, orig_w)
        y1 = clamp(float(row["y1"]) + float(info["crop_y1"]), 0.0, orig_h)
        x2 = clamp(float(row["x2"]) + float(info["crop_x1"]), 0.0, orig_w)
        y2 = clamp(float(row["y2"]) + float(info["crop_y1"]), 0.0, orig_h)
        if x2 <= x1 or y2 <= y1:
            continue
        restored.append(
            {
                "image_path": info["source_image"],
                "class_id": int(float(row["class_id"])),
                "confidence": round(float(row["confidence"]), 8),
                "x1": round(x1, 3),
                "y1": round(y1, 3),
                "x2": round(x2, 3),
                "y2": round(y2, 3),
            }
        )
    return nms_prediction_rows(restored, nms_iou, max_det)


def tile_lookup_from_manifest(manifest: dict) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for source_image, tiles in manifest["tiles_by_source"].items():
        for tile in tiles:
            lookup[tile["image"]] = {
                **tile,
                "source_image": source_image,
            }
    return lookup


def label_path_for_image(image: Path, root: Path) -> Path:
    return root / image.parent.parent.name / "labels" / image.with_suffix(".txt").name

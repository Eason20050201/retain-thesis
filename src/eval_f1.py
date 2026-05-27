import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src import ROOT


IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]


@dataclass(frozen=True)
class Box:
    image_path: str
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def yolo_label_to_xyxy(parts: list[str], width: int, height: int) -> tuple[int, float, float, float, float] | None:
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0]))
    values = list(map(float, parts[1:]))
    if len(values) == 4:
        cx, cy, w, h = values
        return (
            class_id,
            (cx - w / 2) * width,
            (cy - h / 2) * height,
            (cx + w / 2) * width,
            (cy + h / 2) * height,
        )
    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        return class_id, min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height
    return None


def label_for_image(image: Path) -> Path:
    parts = list(image.relative_to(ROOT).parts)
    if "images" not in parts:
        raise ValueError(f"cannot infer label path for {image}")
    parts[parts.index("images")] = "labels"
    return ROOT.joinpath(*parts).with_suffix(".txt")


def read_image_list(path: str | Path) -> list[Path]:
    list_path = repo_path(path)
    images: list[Path] = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        images.append(repo_path(line))
    return images


def read_ground_truth(image_list: str | Path) -> dict[str, list[Box]]:
    ground_truth: dict[str, list[Box]] = {}
    for image in read_image_list(image_list):
        width, height = image_size(image)
        label_path = label_for_image(image)
        boxes: list[Box] = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                parsed = yolo_label_to_xyxy(parts, width, height)
                if parsed is None:
                    continue
                class_id, x1, y1, x2, y2 = parsed
                boxes.append(Box(rel(image), class_id, x1, y1, x2, y2))
        ground_truth[rel(image)] = boxes
    return ground_truth


def read_predictions(path: str | Path) -> list[Box]:
    predictions: list[Box] = []
    with repo_path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(
                Box(
                    image_path=row["image_path"],
                    class_id=int(float(row["class_id"])),
                    confidence=float(row["confidence"]),
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                )
            )
    return predictions


def iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(
    predictions: list[Box],
    ground_truth: dict[str, list[Box]],
    iou_threshold: float,
    conf_threshold: float,
) -> tuple[dict, list[dict], list[dict]]:
    filtered = [p for p in predictions if p.confidence >= conf_threshold]
    filtered.sort(key=lambda p: p.confidence, reverse=True)

    matched: dict[str, set[int]] = {image: set() for image in ground_truth}
    rows: list[dict] = []
    error_rows: list[dict] = []
    tp = fp = 0

    for pred in filtered:
        gts = ground_truth.get(pred.image_path, [])
        best_idx = None
        best_iou = 0.0
        best_any_iou = 0.0
        for idx, gt in enumerate(gts):
            if gt.class_id != pred.class_id:
                continue
            current_iou = iou(pred, gt)
            best_any_iou = max(best_any_iou, current_iou)
            if idx in matched[pred.image_path]:
                continue
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx

        if best_idx is not None and best_iou >= iou_threshold:
            matched[pred.image_path].add(best_idx)
            tp += 1
            rows.append(match_row("TP", pred, best_iou, "matched"))
        else:
            fp += 1
            reason = "localization_error" if best_any_iou > 0 else "false_positive"
            rows.append(match_row("FP", pred, best_any_iou, reason))
            error_rows.append(error_row("FP", pred.image_path, reason, pred.confidence, best_any_iou))

    fn = 0
    for image_path, gts in ground_truth.items():
        for idx, gt in enumerate(gts):
            if idx not in matched[image_path]:
                fn += 1
                rows.append(match_row("FN", gt, 0.0, "missed_ground_truth"))
                error_rows.append(error_row("FN", image_path, "false_negative", 0.0, 0.0))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "confidence_threshold": round(conf_threshold, 6),
        "iou_threshold": iou_threshold,
    }
    return metrics, rows, error_rows


def match_row(status: str, box: Box, matched_iou: float, reason: str) -> dict:
    return {
        "status": status,
        "image_path": box.image_path,
        "class_id": box.class_id,
        "confidence": round(box.confidence, 6),
        "x1": round(box.x1, 3),
        "y1": round(box.y1, 3),
        "x2": round(box.x2, 3),
        "y2": round(box.y2, 3),
        "iou": round(matched_iou, 6),
        "reason": reason,
    }


def error_row(status: str, image_path: str, error_type: str, confidence: float, matched_iou: float) -> dict:
    return {
        "status": status,
        "image_path": image_path,
        "error_type": error_type,
        "confidence": round(confidence, 6),
        "iou": round(matched_iou, 6),
        "notes": "",
    }


def ap_for_threshold(predictions: list[Box], ground_truth: dict[str, list[Box]], iou_threshold: float) -> float:
    total_gt = sum(len(v) for v in ground_truth.values())
    if total_gt == 0:
        return 0.0

    ordered = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    matched: dict[str, set[int]] = {image: set() for image in ground_truth}
    tps: list[int] = []
    fps: list[int] = []

    for pred in ordered:
        gts = ground_truth.get(pred.image_path, [])
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gts):
            if gt.class_id != pred.class_id or idx in matched[pred.image_path]:
                continue
            current_iou = iou(pred, gt)
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched[pred.image_path].add(best_idx)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    cum_tp = 0
    cum_fp = 0
    points: list[tuple[float, float]] = []
    for tp, fp in zip(tps, fps):
        cum_tp += tp
        cum_fp += fp
        recall = cum_tp / total_gt
        precision = cum_tp / (cum_tp + cum_fp) if cum_tp + cum_fp else 0.0
        points.append((recall, precision))

    ap = 0.0
    for recall_level in [x / 100 for x in range(101)]:
        precisions = [p for r, p in points if r >= recall_level]
        ap += max(precisions) if precisions else 0.0
    return ap / 101


def compute_ap_metrics(predictions: list[Box], ground_truth: dict[str, list[Box]]) -> dict:
    ap50 = ap_for_threshold(predictions, ground_truth, 0.50)
    ap5095 = sum(ap_for_threshold(predictions, ground_truth, t) for t in IOU_THRESHOLDS) / len(IOU_THRESHOLDS)
    return {"AP50": round(ap50, 6), "AP50_95": round(ap5095, 6)}


def choose_threshold(predictions: list[Box], ground_truth: dict[str, list[Box]], iou_threshold: float) -> float:
    if not predictions:
        return 1.0
    candidates = sorted({0.0, *[p.confidence for p in predictions]})
    best_threshold = candidates[0]
    best_metrics = {"f1": -1.0, "precision": -1.0}
    for threshold in candidates:
        metrics, _, _ = match_predictions(predictions, ground_truth, iou_threshold, threshold)
        key = (metrics["f1"], metrics["precision"], threshold)
        best_key = (best_metrics["f1"], best_metrics["precision"], best_threshold)
        if key > best_key:
            best_threshold = threshold
            best_metrics = metrics
    return float(best_threshold)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("")


def evaluate(
    predictions_csv: str | Path,
    image_list: str | Path,
    output_dir: str | Path,
    iou_threshold: float = 0.5,
    conf_mode: str = "auto",
    conf_threshold: float | None = None,
) -> dict:
    output = repo_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ground_truth = read_ground_truth(image_list)
    predictions = read_predictions(predictions_csv)

    if conf_mode == "auto":
        selected_conf = choose_threshold(predictions, ground_truth, iou_threshold)
    elif conf_mode == "fixed":
        if conf_threshold is None:
            raise ValueError("conf_threshold is required when conf_mode=fixed")
        selected_conf = float(conf_threshold)
    else:
        raise ValueError(f"unknown conf_mode: {conf_mode}")

    metrics, matches, errors = match_predictions(predictions, ground_truth, iou_threshold, selected_conf)
    metrics.update(compute_ap_metrics(predictions, ground_truth))
    metrics["prediction_count"] = len(predictions)
    metrics["ground_truth_count"] = sum(len(v) for v in ground_truth.values())
    metrics["image_count"] = len(ground_truth)
    metrics["conf_mode"] = conf_mode

    (output / "scores.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    write_csv(output / "matches.csv", matches)
    write_csv(output / "error_cases.csv", errors)
    predictions_src = repo_path(predictions_csv).resolve()
    predictions_dst = (output / "predictions.csv").resolve()
    if predictions_src != predictions_dst:
        shutil.copyfile(predictions_src, predictions_dst)

    with (output / "scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in ["precision", "recall", "f1", "AP50", "AP50_95", "confidence_threshold", "tp", "fp", "fn"]:
            writer.writerow([key, metrics[key]])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detection F1 with fixed IoU and confidence policy.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--images", required=True, help="Image-list txt used as ground truth source.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--conf", choices=["auto", "fixed"], default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    metrics = evaluate(args.predictions, args.images, args.out, args.iou, args.conf, args.threshold)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

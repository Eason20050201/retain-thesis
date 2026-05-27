import argparse
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path

import yaml

from src import ROOT
from src.eval_f1 import evaluate, read_image_list
from src.models import DentalDetector
from src.models.detector import resolve_device


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_yaml(path: str | Path) -> dict:
    with repo_path(path).open() as f:
        return yaml.safe_load(f)


def parse_folds(value: str | None, fold_count: int) -> list[int]:
    if not value:
        return list(range(fold_count))
    folds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            folds.append(int(item))
    invalid = [f for f in folds if f < 0 or f >= fold_count]
    if invalid:
        raise ValueError(f"invalid folds {invalid}; expected 0..{fold_count - 1}")
    return folds


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict:
    versions = {}
    for module_name in ["torch", "torchvision", "ultralytics"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = f"unavailable: {exc}"
    return versions


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def copy_config_snapshot(config_path: Path, fold_dir: Path) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, fold_dir / "experiment_config.yaml")


def prediction_rows_to_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "class_id", "confidence", "x1", "y1", "x2", "y2"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict_yolo_to_csv(detector: DentalDetector, image_list: Path, output_csv: Path, train_cfg: dict, eval_cfg: dict) -> None:
    image_paths = read_image_list(image_list)
    images = [str(p) for p in image_paths]
    results = detector.model.predict(
        source=images,
        conf=float(eval_cfg.get("predict_conf", 0.001)),
        iou=float(eval_cfg.get("nms_iou", 0.45)),
        max_det=int(eval_cfg.get("max_det", 300)),
        imgsz=int(train_cfg.get("img_size", 1024)),
        device=resolve_device(train_cfg.get("device", "auto")),
        verbose=False,
        save=False,
    )

    rows: list[dict] = []
    for result, source_path in zip(results, image_paths, strict=True):
        image_path = rel(source_path)
        boxes = result.boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        for box, conf, cls in zip(xyxy, confs, classes):
            rows.append(
                {
                    "image_path": image_path,
                    "class_id": int(cls),
                    "confidence": round(float(conf), 8),
                    "x1": round(float(box[0]), 3),
                    "y1": round(float(box[1]), 3),
                    "x2": round(float(box[2]), 3),
                    "y2": round(float(box[3]), 3),
                }
            )
    prediction_rows_to_csv(output_csv, rows)


def torch_device(device_value: str):
    import torch

    if device_value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if str(device_value).isdigit():
        return torch.device(f"cuda:{device_value}" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def yolo_label_to_xyxy(parts: list[str], width: int, height: int) -> tuple[int, float, float, float, float] | None:
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0])) + 1
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


class YoloDetectionDataset:
    def __init__(self, image_list: Path):
        from PIL import Image
        import torch
        from torchvision.transforms import functional as F

        self.Image = Image
        self.torch = torch
        self.F = F
        self.images = read_image_list(image_list)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image = self.Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = label_for_image(image_path)

        boxes = []
        labels = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                parsed = yolo_label_to_xyxy(parts, width, height)
                if parsed is None:
                    continue
                class_id, x1, y1, x2, y2 = parsed
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        target = {
            "boxes": self.torch.tensor(boxes, dtype=self.torch.float32).reshape(-1, 4),
            "labels": self.torch.tensor(labels, dtype=self.torch.int64),
            "image_id": self.torch.tensor([idx]),
        }
        return self.F.to_tensor(image), target, rel(image_path)


def label_for_image(image: Path) -> Path:
    parts = list(image.relative_to(ROOT).parts)
    parts[parts.index("images")] = "labels"
    return ROOT.joinpath(*parts).with_suffix(".txt")


def collate_fn(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)


def build_torchvision_model(model_name: str, pretrained_backbone: bool):
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn

    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
    if model_name == "fasterrcnn_resnet50_fpn":
        return fasterrcnn_resnet50_fpn(weights=None, weights_backbone=weights_backbone, num_classes=2)
    if model_name == "retinanet_resnet50_fpn":
        return retinanet_resnet50_fpn(weights=None, weights_backbone=weights_backbone, num_classes=2)
    raise ValueError(f"unsupported torchvision model_name: {model_name}")


def predict_torchvision_to_csv(model, image_list: Path, output_csv: Path, device, eval_cfg: dict) -> None:
    import torch
    from torch.utils.data import DataLoader

    dataset = YoloDetectionDataset(image_list)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    model.eval()
    rows: list[dict] = []
    conf_floor = float(eval_cfg.get("predict_conf", 0.001))
    max_det = int(eval_cfg.get("max_det", 300))
    with torch.no_grad():
        for images, _, paths in loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            for output, image_path in zip(outputs, paths):
                boxes = output["boxes"].detach().cpu()
                scores = output["scores"].detach().cpu()
                labels = output["labels"].detach().cpu()
                order = scores.argsort(descending=True)[:max_det]
                for idx in order:
                    score = float(scores[idx])
                    if score < conf_floor:
                        continue
                    box = boxes[idx].tolist()
                    rows.append(
                        {
                            "image_path": image_path,
                            "class_id": int(labels[idx]) - 1,
                            "confidence": round(score, 8),
                            "x1": round(float(box[0]), 3),
                            "y1": round(float(box[1]), 3),
                            "x2": round(float(box[2]), 3),
                            "y2": round(float(box[3]), 3),
                        }
                    )
    prediction_rows_to_csv(output_csv, rows)


def train_torchvision_fold(exp_cfg: dict, train_list: Path, val_list: Path, fold_dir: Path) -> Path:
    import torch
    from torch.utils.data import DataLoader

    train_cfg = exp_cfg["train"]
    eval_cfg = exp_cfg.get("evaluation", {})
    seed = int(train_cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch_device(str(train_cfg.get("device", "auto")))
    model = build_torchvision_model(exp_cfg["model_name"], bool(train_cfg.get("pretrained_backbone", True)))
    model.to(device)

    dataset = YoloDetectionDataset(train_list)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(train_cfg.get("workers", 0)),
        collate_fn=collate_fn,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = train_cfg.get("optimizer", "SGD")
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=float(train_cfg["learning_rate"]),
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0005)),
        )

    train_log = []
    best_f1 = -1.0
    best_path = fold_dir / "train" / "weights" / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    eval_every = max(1, int(train_cfg.get("eval_every", 1)))

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        for images, targets, _ in loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        row = {"epoch": epoch, "loss": round(epoch_loss / max(1, len(loader)), 6)}
        if epoch % eval_every == 0:
            epoch_dir = fold_dir / "train" / f"epoch_{epoch:03d}"
            pred_csv = epoch_dir / "predictions.csv"
            predict_torchvision_to_csv(model, val_list, pred_csv, device, eval_cfg)
            metrics = evaluate(
                pred_csv,
                val_list,
                epoch_dir,
                float(eval_cfg.get("iou_threshold", 0.5)),
                eval_cfg.get("conf_mode", "auto"),
                eval_cfg.get("conf_threshold"),
            )
            row.update({"val_f1": metrics["f1"], "val_precision": metrics["precision"], "val_recall": metrics["recall"]})
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, best_path)
        train_log.append(row)

    if not best_path.exists():
        torch.save({"model": model.state_dict(), "epoch": int(train_cfg["epochs"])}, best_path)

    with (fold_dir / "train" / "log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in train_log for k in row.keys()}))
        writer.writeheader()
        writer.writerows(train_log)
    return best_path


def load_torchvision_checkpoint(exp_cfg: dict, weights_path: Path, device):
    import torch

    model = build_torchvision_model(exp_cfg["model_name"], bool(exp_cfg["train"].get("pretrained_backbone", True)))
    checkpoint = torch.load(weights_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    return model


def run_fold(exp_cfg: dict, config_path: Path, protocol: dict, fold_idx: int, dry_run: bool) -> dict:
    name = exp_cfg.get("name", config_path.stem)
    dataset_variant = exp_cfg["dataset_variant"]
    split_dir = repo_path(protocol["splits"]["output_dir"]) / dataset_variant
    fold_yaml = split_dir / f"fold{fold_idx}.yaml"
    train_list = split_dir / f"fold{fold_idx}_train.txt"
    val_list = split_dir / f"fold{fold_idx}_val.txt"
    default_output_root = ROOT / "experiments" / "official" / protocol["name"]
    output_root = repo_path(exp_cfg.get("output_root", default_output_root))
    fold_dir = output_root / name / f"fold{fold_idx}"

    info = {
        "fold": fold_idx,
        "engine": exp_cfg["engine"],
        "dataset_variant": dataset_variant,
        "fold_yaml": rel(fold_yaml),
        "train_list": rel(train_list),
        "val_list": rel(val_list),
        "output_dir": rel(fold_dir),
    }
    if dry_run:
        return info

    if not fold_yaml.exists():
        raise FileNotFoundError(f"missing fold dataset yaml: {fold_yaml}; run python -m src.split --write first")
    fold_dir.mkdir(parents=True, exist_ok=True)
    copy_config_snapshot(config_path, fold_dir)

    eval_cfg = exp_cfg.get("evaluation", {})
    if exp_cfg["engine"] == "yolo":
        model_cfg_path = str(ROOT / "configs" / exp_cfg["model"])
        detector = DentalDetector(model_cfg_path)
        best_pt = detector.train(str(fold_yaml), exp_cfg["train"], str(fold_dir))
        detector.load_weights(str(best_pt))
        pred_csv = fold_dir / "eval" / "predictions.csv"
        predict_yolo_to_csv(detector, val_list, pred_csv, exp_cfg["train"], eval_cfg)
    elif exp_cfg["engine"] == "torchvision":
        best_pt = train_torchvision_fold(exp_cfg, train_list, val_list, fold_dir)
        device = torch_device(str(exp_cfg["train"].get("device", "auto")))
        model = load_torchvision_checkpoint(exp_cfg, best_pt, device)
        pred_csv = fold_dir / "eval" / "predictions.csv"
        predict_torchvision_to_csv(model, val_list, pred_csv, device, eval_cfg)
    else:
        raise ValueError(f"unsupported engine: {exp_cfg['engine']}")

    metrics = evaluate(
        pred_csv,
        val_list,
        fold_dir / "eval",
        float(eval_cfg.get("iou_threshold", protocol.get("iou_threshold", 0.5))),
        eval_cfg.get("conf_mode", "auto"),
        eval_cfg.get("conf_threshold"),
    )
    metadata = {
        **info,
        "weights": rel(best_pt),
        "metrics": metrics,
        "git_commit": git_commit(),
        "package_versions": package_versions(),
    }
    write_json(fold_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official 5-fold CV experiments.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--folds", default=None, help="Comma-separated folds. Default: all folds.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = repo_path(args.experiment)
    exp_cfg = load_yaml(config_path)
    protocol = load_yaml(exp_cfg["protocol"])
    folds = parse_folds(args.folds, int(protocol["folds"]))

    results = [run_fold(exp_cfg, config_path, protocol, fold_idx, args.dry_run) for fold_idx in folds]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("\nDry run only. No training, prediction, or evaluation was executed.")


if __name__ == "__main__":
    main()

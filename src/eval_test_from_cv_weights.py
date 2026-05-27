from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml

from src import ROOT
from src.eval_f1 import evaluate
from src.models import DentalDetector
from src.run_cv import (
    load_torchvision_checkpoint,
    prediction_rows_to_csv,
    predict_torchvision_to_csv,
    predict_yolo_to_csv,
    repo_path,
    torch_device,
)
from src.eval_f1 import read_predictions
from src.tiling import merge_tile_predictions, tile_lookup_from_manifest


TOP5_EXPERIMENTS = [
    "configs/experiment/official/fasterrcnn_resnet50_fpn_raw_cv5.yaml",
    "configs/experiment/official/yolo12n_clahe_cv5.yaml",
    "configs/experiment/official/yolo12n_raw_cv5.yaml",
    "configs/experiment/official/yolo12n_kde_cv5.yaml",
    "configs/experiment/official/yolo12n_affine_aug_cv5.yaml",
]

METRICS = ["f1", "precision", "recall", "AP50", "AP50_95", "confidence_threshold"]


def load_yaml(path: str | Path) -> dict:
    with repo_path(path).open() as f:
        return yaml.safe_load(f)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_folds(value: str | None, fold_count: int) -> list[int]:
    if not value:
        return list(range(fold_count))
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    invalid = [fold for fold in folds if fold < 0 or fold >= fold_count]
    if invalid:
        raise ValueError(f"invalid folds {invalid}; expected 0..{fold_count - 1}")
    return folds


def experiment_root(exp_cfg: dict, protocol: dict) -> Path:
    default_root = ROOT / "experiments" / "official" / protocol["name"]
    return repo_path(exp_cfg.get("output_root", default_root))


def split_dir(exp_cfg: dict, protocol: dict) -> Path:
    return repo_path(protocol["splits"]["output_dir"]) / exp_cfg["dataset_variant"]


def dataset_config(exp_cfg: dict, protocol: dict) -> dict:
    return protocol["datasets"][exp_cfg["dataset_variant"]]


def is_tiled_experiment(exp_cfg: dict, protocol: dict) -> bool:
    return "tile_manifest" in dataset_config(exp_cfg, protocol)


def original_locked_test(protocol: dict) -> Path:
    return repo_path(protocol["splits"]["locked_test_file"])


def load_tile_lookup(exp_cfg: dict, protocol: dict) -> dict[str, dict]:
    manifest_path = repo_path(dataset_config(exp_cfg, protocol)["tile_manifest"])
    with manifest_path.open() as f:
        manifest = json.load(f)
    return tile_lookup_from_manifest(manifest)


def cv_threshold(exp_name: str, protocol_name: str) -> float:
    summary_path = ROOT / "experiments" / "official" / protocol_name / exp_name / "cv_summary.json"
    with summary_path.open() as f:
        summary = json.load(f)
    return float(summary["confidence_threshold_mean"])


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(mean, 6), round(std, 6)


def summarize(rows: list[dict], mode: str) -> dict:
    mode_rows = [row for row in rows if row["mode"] == mode]
    out = {"folds_completed": len(mode_rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in mode_rows if row.get(metric) is not None]
        mean, std = mean_std(values)
        out[f"{metric}_mean"] = mean
        out[f"{metric}_std"] = std
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def predict_fold(exp_cfg: dict, protocol: dict, fold: int, test_list: Path, fold_dir: Path) -> Path:
    exp_name = exp_cfg["name"]
    weights_path = experiment_root(exp_cfg, protocol) / exp_name / f"fold{fold}" / "train" / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"missing weights: {weights_path}")

    pred_csv = fold_dir / "predictions.csv"
    eval_cfg = exp_cfg.get("evaluation", {})
    raw_pred_csv = fold_dir / "tile_predictions.csv" if is_tiled_experiment(exp_cfg, protocol) else pred_csv
    if exp_cfg["engine"] == "yolo":
        detector = DentalDetector(str(ROOT / "configs" / exp_cfg["model"]))
        detector.load_weights(str(weights_path))
        predict_yolo_to_csv(detector, test_list, raw_pred_csv, exp_cfg["train"], eval_cfg)
    elif exp_cfg["engine"] == "torchvision":
        device = torch_device(str(exp_cfg["train"].get("device", "auto")))
        model = load_torchvision_checkpoint(exp_cfg, weights_path, device)
        predict_torchvision_to_csv(model, test_list, raw_pred_csv, device, eval_cfg)
    else:
        raise ValueError(f"unsupported engine: {exp_cfg['engine']}")

    if is_tiled_experiment(exp_cfg, protocol):
        tile_rows = [row.__dict__ for row in read_predictions(raw_pred_csv)]
        merged = merge_tile_predictions(
            tile_rows,
            load_tile_lookup(exp_cfg, protocol),
            float(eval_cfg.get("merge_nms_iou", eval_cfg.get("nms_iou", 0.45))),
            int(eval_cfg.get("max_det", 300)),
        )
        prediction_rows_to_csv(pred_csv, merged)

    return pred_csv


def evaluate_fold(
    exp_cfg: dict,
    protocol: dict,
    fold: int,
    test_list: Path,
    output_root: Path,
) -> list[dict]:
    exp_name = exp_cfg["name"]
    fold_dir = output_root / exp_name / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = predict_fold(exp_cfg, protocol, fold, test_list, fold_dir)

    eval_cfg = exp_cfg.get("evaluation", {})
    iou = float(eval_cfg.get("iou_threshold", protocol.get("iou_threshold", 0.5)))
    fixed_threshold = cv_threshold(exp_name, protocol["name"])
    eval_list = original_locked_test(protocol) if is_tiled_experiment(exp_cfg, protocol) else test_list
    fixed = evaluate(pred_csv, eval_list, fold_dir / "eval_fixed_cv_threshold", iou, "fixed", fixed_threshold)
    auto = evaluate(pred_csv, eval_list, fold_dir / "eval_auto_test_threshold", iou, "auto", None)

    rows = []
    for mode, metrics in [("fixed_cv_threshold", fixed), ("auto_test_threshold", auto)]:
        rows.append(
            {
                "experiment": exp_name,
                "fold": fold,
                "mode": mode,
                "dataset_variant": exp_cfg["dataset_variant"],
                "engine": exp_cfg["engine"],
                "test_list": rel(test_list),
                "eval_list": rel(eval_list),
                "predictions": rel(pred_csv),
                **metrics,
            }
        )
    return rows


def write_report(path: Path, summaries: list[dict], rows: list[dict]) -> None:
    ordered = sorted(summaries, key=lambda item: item["fixed_f1_mean"], reverse=True)
    lines = [
        "# Test Scores From CV Fold Weights",
        "",
        "`data/test` is treated here as an external-source test set. Models are the already generated CV fold weights; no retraining is done in this run.",
        "",
        "Primary ranking uses fixed CV-mean confidence threshold. Auto-test threshold is shown only as a threshold-sensitivity reference.",
        "",
        "| rank | experiment | folds | fixed test F1 | fixed P | fixed R | fixed AP50 | fixed AP50-95 | fixed conf | auto test F1 | auto conf |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            "| {rank} | `{experiment}` | {folds} | {fixed_f1_mean:.4f} +/- {fixed_f1_std:.4f} | "
            "{fixed_precision_mean:.4f} | {fixed_recall_mean:.4f} | {fixed_AP50_mean:.4f} | "
            "{fixed_AP50_95_mean:.4f} | {fixed_confidence_threshold_mean:.4f} | "
            "{auto_f1_mean:.4f} +/- {auto_f1_std:.4f} | {auto_confidence_threshold_mean:.4f} |".format(
                rank=rank,
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## Per-Fold Fixed-Threshold Scores",
            "",
            "| experiment | fold | F1 | precision | recall | AP50 | AP50-95 | conf | TP | FP | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted([r for r in rows if r["mode"] == "fixed_cv_threshold"], key=lambda r: (r["experiment"], r["fold"])):
        lines.append(
            "| `{experiment}` | {fold} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {AP50:.4f} | "
            "{AP50_95:.4f} | {confidence_threshold:.4f} | {tp} | {fp} | {fn} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate data/test with already generated CV fold weights.")
    parser.add_argument("--experiments", nargs="*", default=TOP5_EXPERIMENTS)
    parser.add_argument("--folds", default=None)
    parser.add_argument("--output-root", default="experiments/external_test_from_cv_weights/top5")
    parser.add_argument("--report", default="reports/TEST_FROM_CV_WEIGHTS_TOP5.md")
    parser.add_argument("--summary-csv", default="experiments/external_test_from_cv_weights/top5_summary.csv")
    parser.add_argument("--rows-csv", default="experiments/external_test_from_cv_weights/top5_fold_scores.csv")
    args = parser.parse_args()

    output_root = repo_path(args.output_root)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for exp_path in args.experiments:
        exp_cfg = load_yaml(exp_path)
        protocol = load_yaml(exp_cfg["protocol"])
        folds = parse_folds(args.folds, int(protocol["folds"]))
        test_list = split_dir(exp_cfg, protocol) / "locked_test.txt"
        exp_rows: list[dict] = []
        for fold in folds:
            fold_rows = evaluate_fold(exp_cfg, protocol, fold, test_list, output_root)
            exp_rows.extend(fold_rows)
            all_rows.extend(fold_rows)

        fixed = summarize(exp_rows, "fixed_cv_threshold")
        auto = summarize(exp_rows, "auto_test_threshold")
        summary = {"experiment": exp_cfg["name"], "folds": fixed["folds_completed"]}
        for key, value in fixed.items():
            if key != "folds_completed":
                summary[f"fixed_{key}"] = value
        for key, value in auto.items():
            if key != "folds_completed":
                summary[f"auto_{key}"] = value
        summaries.append(summary)

        exp_summary_path = output_root / exp_cfg["name"] / "test_from_cv_weights_summary.json"
        exp_summary_path.parent.mkdir(parents=True, exist_ok=True)
        exp_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    write_csv(repo_path(args.rows_csv), all_rows)
    write_csv(repo_path(args.summary_csv), summaries)
    write_report(repo_path(args.report), summaries, all_rows)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()

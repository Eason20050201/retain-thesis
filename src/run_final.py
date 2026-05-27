import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml

from src import ROOT
from src.eval_f1 import evaluate
from src.run_cv import (
    load_torchvision_checkpoint,
    predict_torchvision_to_csv,
    predict_yolo_to_csv,
    repo_path,
    torch_device,
    train_torchvision_fold,
)
from src.models import DentalDetector


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_yaml(path: str | Path) -> dict:
    with repo_path(path).open() as f:
        return yaml.safe_load(f)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_final_dataset(protocol: dict, dataset_variant: str, final_dir: Path) -> tuple[Path, Path, Path]:
    split_dir = repo_path(protocol["splits"]["output_dir"]) / dataset_variant
    train_lines = set()
    val_lines = set()
    for fold in range(int(protocol["folds"])):
        train_lines.update((split_dir / f"fold{fold}_train.txt").read_text().splitlines())
        val_lines.update((split_dir / f"fold{fold}_val.txt").read_text().splitlines())
    dev_all = sorted(train_lines | val_lines)
    if len(dev_all) != sum(1 for _ in dev_all):
        raise RuntimeError("unexpected duplicate handling failure while building final dev list")

    dev_file = final_dir / "dev_all.txt"
    locked_test_file = split_dir / "locked_test.txt"
    dataset_yaml = final_dir / "dataset_final.yaml"
    write_lines(dev_file, dev_all)
    with dataset_yaml.open("w") as f:
        yaml.safe_dump(
            {
                "path": ".",
                "train": rel(dev_file),
                "val": rel(dev_file),
                "test": rel(locked_test_file),
                "nc": len(protocol.get("class_names", ["tooth"])),
                "names": protocol.get("class_names", ["tooth"]),
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )
    return dataset_yaml, dev_file, locked_test_file


def cv_threshold(exp_name: str, protocol_name: str) -> float:
    summary_path = ROOT / "experiments" / "official" / protocol_name / exp_name / "cv_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"missing {summary_path}; run python -m src.summarize_cv after completing CV, "
            "or pass --threshold explicitly"
        )
    with summary_path.open() as f:
        summary = json.load(f)
    value = summary.get("confidence_threshold_mean")
    if value is None:
        raise ValueError(f"confidence_threshold_mean is missing in {summary_path}")
    return float(value)


def write_final_report(path: Path, exp_name: str, threshold: float, metrics: dict) -> None:
    lines = [
        "# Final Locked Test Result",
        "",
        "此報告只應在方法已由 5-fold CV 固定後產生；不得用本結果回頭調方法。",
        "",
        f"- experiment: `{exp_name}`",
        f"- confidence threshold: `{threshold:.6f}`",
        f"- IoU threshold: `{metrics['iou_threshold']}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in ["precision", "recall", "f1", "AP50", "AP50_95", "tp", "fp", "fn"]:
        lines.append(f"| {key} | {metrics[key]} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final model on dev-all and evaluate locked test once.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-from-cv", action="store_true", default=True)
    args = parser.parse_args()

    exp_cfg = load_yaml(args.experiment)
    protocol = load_yaml(exp_cfg["protocol"])
    exp_name = exp_cfg.get("name", Path(args.experiment).stem)
    final_dir = ROOT / "experiments" / "official" / protocol["name"] / exp_name / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo_path(args.experiment), final_dir / "experiment_config.yaml")

    dataset_yaml, dev_file, locked_test_file = build_final_dataset(protocol, exp_cfg["dataset_variant"], final_dir)
    threshold = args.threshold if args.threshold is not None else cv_threshold(exp_name, protocol["name"])

    eval_cfg = dict(exp_cfg.get("evaluation", {}))
    eval_cfg["conf_mode"] = "fixed"
    eval_cfg["conf_threshold"] = threshold

    if exp_cfg["engine"] == "yolo":
        detector = DentalDetector(str(ROOT / "configs" / exp_cfg["model"]))
        best_pt = detector.train(str(dataset_yaml), exp_cfg["train"], str(final_dir))
        detector.load_weights(str(best_pt))
        pred_csv = final_dir / "locked_test_eval" / "predictions.csv"
        predict_yolo_to_csv(detector, locked_test_file, pred_csv, exp_cfg["train"], eval_cfg)
    elif exp_cfg["engine"] == "torchvision":
        best_pt = train_torchvision_fold(exp_cfg, dev_file, dev_file, final_dir)
        device = torch_device(str(exp_cfg["train"].get("device", "auto")))
        model = load_torchvision_checkpoint(exp_cfg, best_pt, device)
        pred_csv = final_dir / "locked_test_eval" / "predictions.csv"
        predict_torchvision_to_csv(model, locked_test_file, pred_csv, device, eval_cfg)
    else:
        raise ValueError(f"unsupported engine: {exp_cfg['engine']}")

    metrics = evaluate(
        pred_csv,
        locked_test_file,
        final_dir / "locked_test_eval",
        float(eval_cfg.get("iou_threshold", protocol.get("iou_threshold", 0.5))),
        "fixed",
        threshold,
    )
    write_final_report(ROOT / "reports" / "FINAL_TEST_RESULT.md", exp_name, threshold, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


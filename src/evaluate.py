"""
評分工具：從 metrics dict 產生 scores.csv。
"""

import csv
from pathlib import Path


METRICS = ["mAP50", "mAP50_95", "precision", "recall", "f1"]


def add_f1(metrics: dict) -> dict:
    p = metrics.get("precision", 0)
    r = metrics.get("recall", 0)
    metrics["f1"] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
    return metrics


def save_scores_csv(
    output_dir: Path,
    test_metrics: dict,
    val_metrics: dict | None = None,
) -> Path:
    test_metrics = add_f1(test_metrics)

    path = output_dir / "scores.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if val_metrics is not None:
            val_metrics = add_f1(val_metrics)
            writer.writerow(["metric", "val_best", "test"])
            for key in METRICS:
                writer.writerow([key, val_metrics.get(key, ""), test_metrics.get(key, "")])
        else:
            writer.writerow(["metric", "test"])
            for key in METRICS:
                writer.writerow([key, test_metrics.get(key, "")])
    return path


def print_scores(test_metrics: dict, val_metrics: dict | None = None) -> None:
    test_metrics = add_f1(test_metrics)

    if val_metrics is not None:
        val_metrics = add_f1(val_metrics)
        header = f"{'metric':<14} {'val_best':>10} {'test':>10}"
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for key in METRICS:
            print(f"{key:<14} {val_metrics.get(key, '-'):>10} {test_metrics.get(key, '-'):>10}")
    else:
        header = f"{'metric':<14} {'test':>10}"
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for key in METRICS:
            print(f"{key:<14} {test_metrics.get(key, '-'):>10}")
    print(sep)

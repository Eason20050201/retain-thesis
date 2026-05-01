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
    val_metrics: dict,
    test_metrics: dict,
) -> Path:
    """
    將 val best 與 test 分數合併寫入 scores.csv。

    output:
        metric,val_best,test
        mAP50,0.8700,0.8300
        ...
    """
    val_metrics = add_f1(val_metrics)
    test_metrics = add_f1(test_metrics)

    path = output_dir / "scores.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "val_best", "test"])
        for key in METRICS:
            writer.writerow([key, val_metrics.get(key, ""), test_metrics.get(key, "")])
    return path


def print_scores(val_metrics: dict, test_metrics: dict) -> None:
    val_metrics = add_f1(val_metrics)
    test_metrics = add_f1(test_metrics)

    header = f"{'metric':<14} {'val_best':>10} {'test':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for key in METRICS:
        print(f"{key:<14} {val_metrics.get(key, '-'):>10} {test_metrics.get(key, '-'):>10}")
    print(sep)

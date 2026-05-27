from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src import ROOT


BASELINE = {
    "experiment": "yolo12n_raw_cv5",
    "test_f1": 0.6951,
    "test_recall": 0.6475,
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def metric(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def load_cv(exp_name: str, protocol: str) -> dict:
    path = ROOT / "experiments" / "official" / protocol / exp_name / "cv_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_report(path: Path, exp_name: str, test_summary: dict, protocol: str) -> None:
    cv = load_cv(exp_name, protocol)
    f1 = metric(test_summary, "fixed_f1_mean")
    recall = metric(test_summary, "fixed_recall_mean")
    f1_delta = None if f1 is None else f1 - BASELINE["test_f1"]
    recall_delta = None if recall is None else recall - BASELINE["test_recall"]

    decision = "keep as candidate" if f1 is not None and f1 > BASELINE["test_f1"] else "do not replace raw baseline yet"
    if recall is not None and recall > BASELINE["test_recall"] and (f1 is None or f1 <= BASELINE["test_f1"]):
        decision = "recall improved, inspect FP before deciding"

    lines = [
        "# Tiling Decision",
        "",
        "This report evaluates the 2x2 overlapping quadrant tiling line against the current raw YOLO baseline.",
        "",
        "| experiment | CV F1 | CV recall | test F1 | test recall | test precision | AP50 | AP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `{exp_name}` | {fmt(cv.get('f1_mean'))} | {fmt(cv.get('recall_mean'))} | "
            f"{fmt(f1)} | {fmt(recall)} | {fmt(metric(test_summary, 'fixed_precision_mean'))} | "
            f"{fmt(metric(test_summary, 'fixed_AP50_mean'))} | {fmt(metric(test_summary, 'fixed_AP50_95_mean'))} |"
        ),
        "",
        "## Baseline Comparison",
        "",
        f"- baseline: `{BASELINE['experiment']}` test F1 `{BASELINE['test_f1']:.4f}`, recall `{BASELINE['test_recall']:.4f}`",
        f"- F1 delta: `{fmt(f1_delta)}`",
        f"- recall delta: `{fmt(recall_delta)}`",
        f"- decision: `{decision}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize quadrant tiling result against raw baseline.")
    parser.add_argument("--experiment", default="yolo12n_quadrant_cv5")
    parser.add_argument("--test-summary", default="experiments/external_test_from_cv_weights/quadrant_summary.csv")
    parser.add_argument("--protocol", default="cv5_locked_test")
    parser.add_argument("--out", default="reports/TILING_DECISION.md")
    args = parser.parse_args()

    rows = read_csv(repo_path(args.test_summary))
    row = next((item for item in rows if item.get("experiment") == args.experiment), {})
    write_report(repo_path(args.out), args.experiment, row, args.protocol)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

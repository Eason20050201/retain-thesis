from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src import ROOT


ROUND3_EXPERIMENTS = [
    "yolo12n_raw_1280_cv5",
    "yolo12s_raw_1024_cv5",
    "yolo12s_raw_1280_cv5",
    "yolo12n_conservative_roi_cv5",
]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def cv_summary(exp: str, protocol: str) -> dict:
    path = ROOT / "experiments" / "official" / protocol / exp / "cv_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def f(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def verdict(rows: list[dict]) -> list[str]:
    by_exp = {row["experiment"]: row for row in rows}
    raw1280 = f(by_exp.get("yolo12n_raw_1280_cv5", {}), "fixed_f1_mean")
    s1024 = f(by_exp.get("yolo12s_raw_1024_cv5", {}), "fixed_f1_mean")
    s1280 = f(by_exp.get("yolo12s_raw_1280_cv5", {}), "fixed_f1_mean")
    roi = f(by_exp.get("yolo12n_conservative_roi_cv5", {}), "fixed_f1_mean")
    baseline = 0.6951
    lines = []
    lines.append("- Higher resolution helps if `yolo12n_raw_1280_cv5` beats baseline `0.6951` and improves recall.")
    lines.append("- YOLOv12s helps if either `yolo12s_raw_1024_cv5` or `yolo12s_raw_1280_cv5` beats the matching YOLOv12n setting.")
    lines.append("- Conservative ROI helps if it beats raw baseline without dropping test recall.")
    if raw1280 is not None and raw1280 >= baseline + 0.01:
        lines.append("- Current decision: keep high-resolution YOLO in the short-list.")
    if s1024 is not None and s1280 is not None and max(s1024, s1280) >= baseline + 0.01:
        lines.append("- Current decision: keep YOLOv12s in the short-list.")
    if roi is not None and roi >= baseline + 0.01:
        lines.append("- Current decision: keep conservative ROI in the short-list.")
    if rows and max(f(row, "fixed_f1_mean") or 0.0 for row in rows) < baseline + 0.01:
        lines.append("- Current decision: promote quadrant/tiling to the next formal experiment line.")
    return lines


def write_report(path: Path, test_rows: list[dict], protocol: str) -> None:
    lines = [
        "# Round 3 Decision",
        "",
        "Round 3 asks whether the recall bottleneck is better addressed by input resolution, model capacity, or conservative ROI.",
        "",
        "| experiment | CV F1 | CV recall | test F1 | test precision | test recall | test AP50 | test AP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exp in ROUND3_EXPERIMENTS:
        test = next((row for row in test_rows if row.get("experiment") == exp), {})
        cv = cv_summary(exp, protocol)
        lines.append(
            f"| `{exp}` | {fmt(cv.get('f1_mean'))} | {fmt(cv.get('recall_mean'))} | "
            f"{fmt(f(test, 'fixed_f1_mean'))} | {fmt(f(test, 'fixed_precision_mean'))} | "
            f"{fmt(f(test, 'fixed_recall_mean'))} | {fmt(f(test, 'fixed_AP50_mean'))} | {fmt(f(test, 'fixed_AP50_95_mean'))} |"
        )

    lines.extend(["", "## Decision Rules", ""])
    lines.extend(verdict(test_rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Round 3 CV/test results into a decision report.")
    parser.add_argument("--test-summary", default="experiments/external_test_from_cv_weights/round3_summary.csv")
    parser.add_argument("--protocol", default="cv5_locked_test")
    parser.add_argument("--out", default="reports/ROUND3_DECISION.md")
    args = parser.parse_args()

    rows = read_csv(repo_path(args.test_summary))
    rows = [row for row in rows if row.get("experiment") in ROUND3_EXPERIMENTS]
    write_report(repo_path(args.out), rows, args.protocol)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src import ROOT


ROUND4_EXPERIMENTS = [
    "yolo12n_aspect188_cv5",
    "yolo12n_pnorm_cv5",
    "yolo12n_aspect188_pnorm_cv5",
]

BASELINE = {
    "experiment": "yolo12n_raw_cv5",
    "test_f1": 0.6951,
    "test_recall": 0.6475,
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path | str) -> list[dict]:
    path = repo_path(path)
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
    return "-" if value is None else f"{value:.4f}"


def cv_summary(exp: str, protocol: str) -> dict:
    path = ROOT / "experiments" / "official" / protocol / exp / "cv_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def decision_for(row: dict) -> str:
    f1 = metric(row, "fixed_f1_mean")
    recall = metric(row, "fixed_recall_mean")
    if f1 is None or recall is None:
        return "pending"
    f1_delta = f1 - BASELINE["test_f1"]
    recall_delta = recall - BASELINE["test_recall"]
    if f1_delta > 0 and recall_delta > 0:
        return "candidate final method"
    if recall >= 0.70 and f1 >= BASELINE["test_f1"] - 0.01:
        return "strong recall candidate"
    if recall_delta >= 0.04 and f1 >= 0.675:
        return "recall-oriented candidate"
    if recall_delta > 0:
        return "recall gain but F1 trade-off"
    return "do not replace raw baseline"


def write_report(path: Path | str, test_rows: list[dict], protocol: str = "cv5_locked_test") -> None:
    path = repo_path(path)
    by_exp = {row.get("experiment"): row for row in test_rows}
    lines = [
        "# Round 4 Domain Alignment Decision",
        "",
        "Round 4 asks whether image-only train-test alignment improves external test recall without using test labels to choose preprocessing parameters.",
        "",
        f"Baseline: `{BASELINE['experiment']}` fixed test F1 `{BASELINE['test_f1']:.4f}`, recall `{BASELINE['test_recall']:.4f}`.",
        "",
        "| experiment | CV F1 | CV recall | test F1 | test precision | test recall | test AP50 | test AP50-95 | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for exp in ROUND4_EXPERIMENTS:
        row = by_exp.get(exp, {})
        cv = cv_summary(exp, protocol)
        lines.append(
            f"| `{exp}` | {fmt(cv.get('f1_mean'))} | {fmt(cv.get('recall_mean'))} | "
            f"{fmt(metric(row, 'fixed_f1_mean'))} | {fmt(metric(row, 'fixed_precision_mean'))} | "
            f"{fmt(metric(row, 'fixed_recall_mean'))} | {fmt(metric(row, 'fixed_AP50_mean'))} | "
            f"{fmt(metric(row, 'fixed_AP50_95_mean'))} | {decision_for(row)} |"
        )

    completed = [row for row in test_rows if row.get("experiment") in ROUND4_EXPERIMENTS]
    lines.extend(["", "## Decision Rules", ""])
    lines.append("- Strong success: test recall `>= 0.70` and test F1 close to or above raw baseline.")
    lines.append("- Acceptable recall-oriented result: recall improves by at least `+0.04` and test F1 remains `>= 0.675`.")
    lines.append("- Otherwise, do not replace `yolo12n_raw_cv5`.")
    if completed:
        best_recall = max(completed, key=lambda row: metric(row, "fixed_recall_mean") or -1)
        best_f1 = max(completed, key=lambda row: metric(row, "fixed_f1_mean") or -1)
        lines.extend(
            [
                "",
                "## Current Reading",
                "",
                f"- Best recall: `{best_recall['experiment']}` recall `{fmt(metric(best_recall, 'fixed_recall_mean'))}`.",
                f"- Best F1: `{best_f1['experiment']}` F1 `{fmt(metric(best_f1, 'fixed_f1_mean'))}`.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Round 4 domain alignment CV/test results.")
    parser.add_argument("--test-summary", default="experiments/external_test_from_cv_weights/round4_domain_alignment_summary.csv")
    parser.add_argument("--protocol", default="cv5_locked_test")
    parser.add_argument("--out", default="reports/ROUND4_DOMAIN_ALIGNMENT_DECISION.md")
    args = parser.parse_args()

    rows = [row for row in read_csv(args.test_summary) if row.get("experiment") in ROUND4_EXPERIMENTS]
    write_report(args.out, rows, args.protocol)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

import argparse
import csv
from collections import Counter
from pathlib import Path

from src import ROOT


ERROR_LABELS = {
    "FP": ["implant", "bridge", "artifact", "jaw-edge", "duplicate tooth", "other"],
    "FN": ["low contrast", "blur", "overlap", "partial tooth", "posterior", "cropped boundary", "other"],
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_errors(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize_errors(rows: list[dict]) -> Counter:
    counter = Counter()
    for row in rows:
        key = row.get("notes") or row.get("error_type") or row.get("status") or "unlabeled"
        counter[key] += 1
    return counter


def write_report(path: Path, experiment_dir: Path, rows: list[dict]) -> None:
    counts = summarize_errors(rows)
    lines = [
        "# Error Analysis",
        "",
        f"Source: `{experiment_dir}`",
        "",
        "本報告由 evaluator 的 `error_cases.csv` 產生。`notes` 欄位可人工補上固定錯誤分類後重跑本腳本。",
        "",
        "固定錯誤分類：",
        "",
        "- FP: " + ", ".join(ERROR_LABELS["FP"]),
        "- FN: " + ", ".join(ERROR_LABELS["FN"]),
        "",
        "| error type / note | count |",
        "| --- | ---: |",
    ]
    for key, count in counts.most_common():
        lines.append(f"| {key} | {count} |")
    if not counts:
        lines.append("| pending | 0 |")
    lines.extend(
        [
            "",
            "下一步：針對 final method 與主要 baseline，人工打開對應影像與 `matches.csv`，",
            "在 `error_cases.csv` 的 `notes` 欄補上 FP/FN 類型，再重跑此腳本生成正式錯誤分析表。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize evaluator error_cases.csv into ERROR_ANALYSIS.md.")
    parser.add_argument("--experiment-dir", required=True, help="Directory containing eval/error_cases.csv or locked_test_eval/error_cases.csv.")
    parser.add_argument("--out", default="reports/ERROR_ANALYSIS.md")
    args = parser.parse_args()

    experiment_dir = repo_path(args.experiment_dir)
    candidates = [
        experiment_dir / "locked_test_eval" / "error_cases.csv",
        experiment_dir / "eval" / "error_cases.csv",
        experiment_dir / "error_cases.csv",
    ]
    rows = []
    for candidate in candidates:
        rows = read_errors(candidate)
        if rows:
            break
    write_report(repo_path(args.out), experiment_dir, rows)
    print(f"Wrote {args.out} from {len(rows)} error rows")


if __name__ == "__main__":
    main()


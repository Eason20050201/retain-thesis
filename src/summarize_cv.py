import argparse
import csv
import json
import statistics
from pathlib import Path

from src import ROOT


METRICS = ["f1", "precision", "recall", "AP50", "AP50_95", "confidence_threshold"]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_scores(experiment_dir: Path) -> list[dict]:
    rows = []
    for scores_path in sorted(experiment_dir.glob("fold*/eval/scores.json")):
        with scores_path.open() as f:
            scores = json.load(f)
        fold = int(scores_path.parts[-3].replace("fold", ""))
        rows.append({"fold": fold, **scores})
    return rows


def summarize(rows: list[dict]) -> dict:
    summary = {"folds_completed": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if metric in row]
        if values:
            summary[f"{metric}_mean"] = round(statistics.mean(values), 6)
            summary[f"{metric}_std"] = round(statistics.stdev(values), 6) if len(values) > 1 else 0.0
        else:
            summary[f"{metric}_mean"] = None
            summary[f"{metric}_std"] = None
    return summary


def write_summary_csv(path: Path, summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "folds_completed"]
    for metric in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def write_report(path: Path, summaries: list[dict], protocol_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        summaries,
        key=lambda row: (
            row.get("f1_mean") is not None,
            row.get("f1_mean") if row.get("f1_mean") is not None else -1,
        ),
        reverse=True,
    )
    is_overnight = protocol_name.startswith("yolo_preprocessing")
    lines = [
        "# Overnight YOLO Preprocessing Results" if is_overnight else "# CV5 Results",
        "",
        f"Protocol: `{protocol_name}`",
        "",
        "Fold0 screening only. These are not final paper results; locked test was not used."
        if is_overnight
        else "正式排名以 validation 5-fold mean F1 為準；locked test 不在本表使用。",
        "",
        "| rank | experiment | folds | mean F1 | std F1 | mean precision | mean recall | mean AP50 | mean AP50-95 | mean conf |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            "| {rank} | `{experiment}` | {folds_completed} | {f1_mean} | {f1_std} | "
            "{precision_mean} | {recall_mean} | {AP50_mean} | {AP50_95_mean} | {confidence_threshold_mean} |".format(
                rank=rank,
                **{k: format_value(v) for k, v in row.items()},
            )
        )
    if not ordered:
        lines.append("| - | pending | 0 | - | - | - | - | - | - | - |")
    lines.append("")
    lines.append("Note: missing experiments are pending until all five folds produce `eval/scores.json`.")
    path.write_text("\n".join(lines) + "\n")


def format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def collect(root: Path) -> list[dict]:
    summaries = []
    for experiment_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if experiment_dir.name.startswith("smoke_") or "_recheck" in experiment_dir.name:
            continue
        rows = load_scores(experiment_dir)
        summary = summarize(rows)
        summary["experiment"] = experiment_dir.name
        summaries.append(summary)
        if rows:
            with (experiment_dir / "cv_summary.json").open("w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize official CV results.")
    parser.add_argument("--root", default="experiments/official/cv5_locked_test")
    parser.add_argument("--report", default="reports/CV5_RESULTS.md")
    parser.add_argument("--csv", default="experiments/official/cv5_locked_test/summary.csv")
    args = parser.parse_args()

    root = repo_path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    summaries = collect(root)
    write_summary_csv(repo_path(args.csv), summaries)
    write_report(repo_path(args.report), summaries, root.name)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"Wrote {args.report} and {args.csv}")


if __name__ == "__main__":
    main()

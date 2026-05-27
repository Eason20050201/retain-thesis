from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from src import ROOT


STATUS_COLORS = {
    "TP": (40, 180, 40),
    "FP": (230, 30, 30),
    "FN": (255, 170, 0),
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fold_dirs(input_root: Path, experiment: str) -> list[Path]:
    exp_root = input_root / experiment
    folds = sorted(p for p in exp_root.glob("fold*") if p.is_dir())
    if not folds:
        raise FileNotFoundError(f"no fold directories found under {exp_root}")
    return folds


def load_fold_scores(folds: list[Path]) -> list[dict]:
    rows = []
    for fold in folds:
        scores_path = fold / "eval_fixed_cv_threshold" / "scores.json"
        if not scores_path.exists():
            raise FileNotFoundError(f"missing scores.json: {scores_path}")
        score = json.loads(scores_path.read_text())
        score["fold"] = fold.name
        score["fold_dir"] = fold
        rows.append(score)
    return rows


def best_fold(scores: list[dict]) -> dict:
    return max(scores, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])))


def aggregate_matches(folds: list[Path]) -> tuple[list[dict], list[dict]]:
    all_matches = []
    per_image = defaultdict(Counter)
    for fold in folds:
        matches_path = fold / "eval_fixed_cv_threshold" / "matches.csv"
        for row in read_csv(matches_path):
            row["fold"] = fold.name
            all_matches.append(row)
            image = Path(row["image_path"]).name
            per_image[image][row["status"]] += 1
            if row["status"] == "FP" and row.get("reason") == "localization_error":
                per_image[image]["localization_error"] += 1
    rows = []
    for image, counts in per_image.items():
        rows.append(
            {
                "image": image,
                "tp": counts["TP"],
                "fp": counts["FP"],
                "fn": counts["FN"],
                "localization_error": counts["localization_error"],
                "total_errors": counts["FP"] + counts["FN"],
            }
        )
    rows.sort(key=lambda row: (row["fn"], row["fp"], row["localization_error"], row["image"]), reverse=True)
    return all_matches, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("")


def draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    y0 = max(0, y - 18)
    draw.rectangle((x, y0, x + 88, y), fill=color)
    draw.text((x + 3, y0 + 2), text, fill=(255, 255, 255))


def draw_visualizations(matches: list[dict], out_dir: Path) -> list[Path]:
    by_image = defaultdict(list)
    for row in matches:
        by_image[row["image_path"]].append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for image_rel, rows in sorted(by_image.items(), key=lambda item: Path(item[0]).name):
        image_path = repo_path(image_rel)
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for row in rows:
            status = row["status"]
            color = STATUS_COLORS.get(status, (255, 255, 255))
            x1 = int(round(float(row["x1"])))
            y1 = int(round(float(row["y1"])))
            x2 = int(round(float(row["x2"])))
            y2 = int(round(float(row["y2"])))
            for offset in range(3):
                draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
            suffix = ""
            if status == "FP" and row.get("reason") == "localization_error":
                suffix = " loc"
            draw_label(draw, status + suffix, x1, y1, color)
        out_path = out_dir / Path(image_rel).name
        image.save(out_path)
        written.append(out_path)
    return written


def write_report(
    path: Path,
    experiment: str,
    input_root: Path,
    scores: list[dict],
    per_image: list[dict],
    visual_paths: list[Path],
    per_image_csv: Path,
) -> None:
    best = best_fold(scores)
    totals = Counter()
    for row in per_image:
        totals["tp"] += int(row["tp"])
        totals["fp"] += int(row["fp"])
        totals["fn"] += int(row["fn"])
        totals["localization_error"] += int(row["localization_error"])

    lines = [
        "# Round 3 Error Analysis",
        "",
        f"Experiment: `{experiment}`",
        f"Source: `{rel(input_root)}`",
        "",
        "This report summarizes fixed-CV-threshold test errors across the five CV fold weights.",
        "",
        "## Fold Scores",
        "",
        "| fold | F1 | precision | recall | TP | FP | FN | conf |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for score in sorted(scores, key=lambda row: row["fold"]):
        lines.append(
            "| {fold} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {tp} | {fp} | {fn} | {confidence_threshold:.4f} |".format(
                **score
            )
        )

    lines.extend(
        [
            "",
            "## Best Fold For Visualization",
            "",
            f"- best fold: `{best['fold']}`",
            f"- F1: `{best['f1']:.4f}`",
            f"- precision: `{best['precision']:.4f}`",
            f"- recall: `{best['recall']:.4f}`",
            "",
            "Color legend in visualizations: green = TP, red = FP, orange = FN.",
            "",
            "## Aggregate Error Counts",
            "",
            f"- TP across folds: `{totals['tp']}`",
            f"- FP across folds: `{totals['fp']}`",
            f"- FN across folds: `{totals['fn']}`",
            f"- Localization-error FP across folds: `{totals['localization_error']}`",
            f"- per-image CSV: `{rel(per_image_csv)}`",
            f"- visualization directory: `{rel(visual_paths[0].parent) if visual_paths else 'not generated'}`",
            "",
            "## Top Error Images",
            "",
            "| image | TP | FP | FN | localization error | total errors | visualization |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    visual_by_name = {p.name: p for p in visual_paths}
    for row in per_image[:12]:
        vis = visual_by_name.get(row["image"])
        vis_link = rel(vis) if vis else ""
        lines.append(
            f"| `{row['image']}` | {row['tp']} | {row['fp']} | {row['fn']} | {row['localization_error']} | {row['total_errors']} | `{vis_link}` |"
        )

    lines.extend(
        [
            "",
            "## Initial Reading",
            "",
            "- Current failure is recall-heavy: the best baseline repeatedly leaves many FN on external test images.",
            "- Localization errors are present but smaller than pure missed detections, so Round 3 prioritizes resolution/model capacity/ROI before threshold tricks.",
            "- This report does not manually classify medical causes yet; the visualization folder is the review surface for low contrast, overlap, posterior, and boundary categories.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 3 error report and fixed-threshold test visualizations.")
    parser.add_argument("--experiment", default="yolo12n_raw_cv5")
    parser.add_argument("--input-root", default="experiments/external_test_from_cv_weights/top5")
    parser.add_argument("--out", default="reports/ROUND3_ERROR_ANALYSIS.md")
    parser.add_argument("--visual-root", default=None)
    args = parser.parse_args()

    input_root = repo_path(args.input_root)
    folds = fold_dirs(input_root, args.experiment)
    scores = load_fold_scores(folds)
    _, per_image = aggregate_matches(folds)

    visual_root = repo_path(args.visual_root) if args.visual_root else ROOT / "reports" / "round3_error_analysis" / args.experiment
    best = best_fold(scores)
    best_matches = read_csv(Path(best["fold_dir"]) / "eval_fixed_cv_threshold" / "matches.csv")
    visual_paths = draw_visualizations(best_matches, visual_root / best["fold"])

    per_image_csv = visual_root / "per_image_error_counts.csv"
    write_csv(per_image_csv, per_image)
    write_report(repo_path(args.out), args.experiment, input_root, scores, per_image, visual_paths, per_image_csv)
    print(f"Wrote {args.out}")
    print(f"Wrote {per_image_csv}")
    print(f"Wrote {len(visual_paths)} visualizations")


if __name__ == "__main__":
    main()

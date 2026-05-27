from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src import ROOT


ROUND2_METHODS = [
    {
        "experiment": "yolo12n_photo_aug_cv5",
        "method": "Photometric augmentation",
        "literature": "Yuksel/DENTECT; DENTEX sequential framework",
        "level": "inspired/proxy",
        "claim": "YOLO train-time proxy for brightness/contrast-style augmentation; not a full reproduction.",
    },
    {
        "experiment": "yolo12n_cutout_aug_cv5",
        "method": "Random erasing / cutout proxy",
        "literature": "DENTEX sequential framework",
        "level": "inspired/proxy",
        "claim": "Ultralytics random erasing used as a cutout proxy; not identical to the original pipeline.",
    },
    {
        "experiment": "yolo12n_dentex_mild_combo_cv5",
        "method": "Mild DENTEX-style combined augmentation",
        "literature": "DENTEX sequential framework; Yuksel/DENTECT",
        "level": "project combination",
        "claim": "Combines mild photometric, affine, and cutout-style augmentation for an ablation-style literature-inspired setting.",
    },
    {
        "experiment": "yolo12n_gamma05_cv5",
        "method": "Gamma correction 0.5",
        "literature": "PS-KDE / radiograph enhancement context",
        "level": "control/inspired",
        "claim": "Common radiograph intensity transform baseline; not a method uniquely proposed by a dental detection paper.",
    },
]


def load_test_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["experiment"]: row for row in csv.DictReader(f)}


def load_cv_summary(exp: str, protocol: str) -> dict:
    path = ROOT / "experiments" / "official" / protocol / exp / "cv_summary.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def fmt(value: object, digits: int = 4) -> str:
    if value in (None, ""):
        return "pending"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(output: Path, protocol: str, test_summary_path: Path) -> None:
    test_rows = load_test_summary(test_summary_path)
    lines = [
        "# Round 2 Literature-First Results",
        "",
        "第二輪目標是補齊高度相關牙科影像 / B-level 會議論文中可直接落地的方法，並清楚區分 direct reproduction、inspired proxy 與 control baseline。",
        "",
        "Primary CV metric: `mean F1@IoU=0.5` under `cv5_locked_test`.",
        "Test score policy: evaluate `data/test` with generated CV fold weights and fixed CV-mean confidence threshold.",
        "",
        "## Method Sources",
        "",
        "| experiment | method | literature source | source level | wording guard |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in ROUND2_METHODS:
        lines.append(
            "| `{experiment}` | {method} | {literature} | {level} | {claim} |".format(**item)
        )

    lines.extend(
        [
            "",
            "## CV And Test Results",
            "",
            "| experiment | CV mean F1 | CV std F1 | CV precision | CV recall | test F1 | test precision | test recall | test AP50 | test AP50-95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in ROUND2_METHODS:
        exp = item["experiment"]
        cv = load_cv_summary(exp, protocol)
        test = test_rows.get(exp, {})
        lines.append(
            "| `{exp}` | {cv_f1} | {cv_std} | {cv_p} | {cv_r} | {test_f1} | {test_p} | {test_r} | {test_ap50} | {test_ap5095} |".format(
                exp=exp,
                cv_f1=fmt(cv.get("f1_mean")),
                cv_std=fmt(cv.get("f1_std")),
                cv_p=fmt(cv.get("precision_mean")),
                cv_r=fmt(cv.get("recall_mean")),
                test_f1=fmt(test.get("fixed_f1_mean")),
                test_p=fmt(test.get("fixed_precision_mean")),
                test_r=fmt(test.get("fixed_recall_mean")),
                test_ap50=fmt(test.get("fixed_AP50_mean")),
                test_ap5095=fmt(test.get("fixed_AP50_95_mean")),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Do not describe these methods as exact reproductions unless the implementation matches the original paper pipeline.",
            "- Compare round 2 against `yolo12n_raw_cv5`, because first-round test-from-CV-weights currently favors raw YOLO.",
            "- Prefer methods that improve test F1 or recall without collapsing precision; CV-only improvement is not enough for final method selection.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="cv5_locked_test")
    parser.add_argument("--test-summary", default="experiments/external_test_from_cv_weights/round2_summary.csv")
    parser.add_argument("--output", default="reports/ROUND2_LITERATURE_SUMMARY.md")
    args = parser.parse_args()
    write_report(ROOT / args.output, args.protocol, ROOT / args.test_summary)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

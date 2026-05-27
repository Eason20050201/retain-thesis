from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src import ROOT


GROUPS = {
    "fasterrcnn_resnet50_fpn_raw_cv5": "R1 detector baseline",
    "retinanet_resnet50_fpn_raw_cv5": "R1 detector baseline",
    "yolo12n_raw_cv5": "R1 YOLO baseline",
    "yolo12n_clahe_cv5": "R1 preprocessing",
    "yolo12n_kde_cv5": "R1 preprocessing",
    "yolo12n_clahe_abc_cv5": "R1 preprocessing",
    "yolo12n_affine_aug_cv5": "R1 augmentation",
    "yolo12n_photo_aug_cv5": "R2 literature augmentation",
    "yolo12n_cutout_aug_cv5": "R2 literature augmentation",
    "yolo12n_dentex_mild_combo_cv5": "R2 literature augmentation",
    "yolo12n_gamma05_cv5": "R2 intensity baseline",
    "yolo12n_raw_1280_cv5": "R3 recall rescue",
    "yolo12s_raw_1024_cv5": "R3 capacity",
    "yolo12s_raw_1280_cv5": "R3 capacity+resolution",
    "yolo12n_conservative_roi_cv5": "R3 ROI",
    "yolo12n_roi_crop_cv5": "ROI side experiment",
    "yolo12n_quadrant_cv5": "Tiling side experiment",
    "yolo12n_aspect188_cv5": "R4 domain alignment",
    "yolo12n_pnorm_cv5": "R4 domain alignment",
    "yolo12n_aspect188_pnorm_cv5": "R4 domain alignment",
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def load_cv_rows(protocol: str) -> dict[str, dict]:
    root = ROOT / "experiments" / "official" / protocol
    rows: dict[str, dict] = {}
    for row in read_csv(root / "summary.csv"):
        rows[row["experiment"]] = row
    for path in root.glob("*/cv_summary.json"):
        data = json.loads(path.read_text())
        exp = data.get("experiment", path.parent.name)
        rows[exp] = {
            "experiment": exp,
            "folds_completed": data.get("folds_completed"),
            "f1_mean": data.get("f1_mean"),
            "f1_std": data.get("f1_std"),
            "precision_mean": data.get("precision_mean"),
            "recall_mean": data.get("recall_mean"),
            "AP50_mean": data.get("AP50_mean"),
            "AP50_95_mean": data.get("AP50_95_mean"),
            "confidence_threshold_mean": data.get("confidence_threshold_mean"),
        }
    return rows


def load_test_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    root = ROOT / "experiments" / "external_test_from_cv_weights"
    for path in sorted(root.glob("*_summary.csv")):
        for row in read_csv(path):
            rows[row["experiment"]] = row
    return rows


def row_record(exp: str, cv: dict, test: dict | None) -> dict:
    return {
        "experiment": exp,
        "group": GROUPS.get(exp, "other"),
        "cv_f1": f(cv, "f1_mean"),
        "cv_f1_std": f(cv, "f1_std"),
        "cv_p": f(cv, "precision_mean"),
        "cv_r": f(cv, "recall_mean"),
        "test_f1": f(test, "fixed_f1_mean"),
        "test_f1_std": f(test, "fixed_f1_std"),
        "test_p": f(test, "fixed_precision_mean"),
        "test_r": f(test, "fixed_recall_mean"),
        "test_ap50": f(test, "fixed_AP50_mean"),
        "test_ap5095": f(test, "fixed_AP50_95_mean"),
        "conf": f(test, "fixed_confidence_threshold_mean"),
    }


def write_scoreboard(out: Path | str = "reports/CURRENT_EXPERIMENT_SCOREBOARD.md", protocol: str = "cv5_locked_test") -> None:
    out = repo_path(out)
    cv_rows = load_cv_rows(protocol)
    test_rows = load_test_rows()
    records = [row_record(exp, cv, test_rows.get(exp)) for exp, cv in cv_rows.items()]
    with_test = sorted(
        [record for record in records if record["test_f1"] is not None],
        key=lambda record: record["test_f1"],
        reverse=True,
    )
    missing = sorted(
        [record for record in records if record["test_f1"] is None],
        key=lambda record: record["cv_f1"] if record["cv_f1"] is not None else -1,
        reverse=True,
    )

    lines = [
        "# Current Experiment Scoreboard",
        "",
        "這份檔案把目前已完成的正式實驗集中成一張表，避免 CV 分數、外部 test 分數、各輪報告分散在不同檔案。",
        "",
        "## 分數讀法",
        "",
        "- **CV F1**：`cv5_locked_test` protocol 下，`train+valid` 共 101 張做 5-fold validation 的 mean F1@IoU=0.5。",
        "- **test F1**：使用每個 fold 已訓練出的 best weight 推 `data/test` 26 張，再用該方法的 CV mean confidence threshold 評估。",
        "- **test 分數不是 final retrain result**：目前是 test-from-CV-weights，用來看外部來源泛化狀況。",
        "",
        "## 主表：同時有 CV 與 Test 的方法",
        "",
        "| rank(test) | experiment | group | CV F1 | CV R | test F1 | test P | test R | test AP50 | test AP50-95 | conf |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, record in enumerate(with_test, start=1):
        lines.append(
            f"| {rank} | `{record['experiment']}` | {record['group']} | "
            f"{fmt(record['cv_f1'])} +/- {fmt(record['cv_f1_std'])} | {fmt(record['cv_r'])} | "
            f"{fmt(record['test_f1'])} +/- {fmt(record['test_f1_std'])} | {fmt(record['test_p'])} | "
            f"{fmt(record['test_r'])} | {fmt(record['test_ap50'])} | {fmt(record['test_ap5095'])} | {fmt(record['conf'])} |"
        )

    lines.extend(
        [
            "",
            "## CV 已完成但尚未有 Test Summary",
            "",
            "| experiment | group | CV F1 | CV P | CV R | status |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for record in missing:
        lines.append(
            f"| `{record['experiment']}` | {record['group']} | {fmt(record['cv_f1'])} +/- {fmt(record['cv_f1_std'])} | "
            f"{fmt(record['cv_p'])} | {fmt(record['cv_r'])} | CV done, test-from-CV not yet run/synced |"
        )

    if with_test:
        best = with_test[0]
        lines.extend(
            [
                "",
                "## 目前結論",
                "",
                f"1. 目前外部 test F1 最高是 `{best['experiment']}`，test F1 `{fmt(best['test_f1'])}`，test recall `{fmt(best['test_r'])}`。",
                "2. CV 高不等於外部 test 高，因此 final method 仍需看泛化與錯誤分析。",
                "3. 若 Round4 domain alignment 能提升 recall，需同時檢查 F1 是否維持在可接受範圍。",
            ]
        )

    lines.extend(
        [
            "",
            "## 原始資料來源",
            "",
            "- CV summary：`reports/CV5_RESULTS.md`",
            "- Test summaries：`experiments/external_test_from_cv_weights/*_summary.csv`",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the current CV/test experiment scoreboard.")
    parser.add_argument("--protocol", default="cv5_locked_test")
    parser.add_argument("--out", default="reports/CURRENT_EXPERIMENT_SCOREBOARD.md")
    args = parser.parse_args()
    write_scoreboard(args.out, args.protocol)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

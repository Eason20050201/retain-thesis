"""
超參數搜尋：使用 Optuna TPE 自動找最佳 lr / batch_size / optimizer / img_size。

用法：
    python -m src.tune

結果會存在 experiments/tune/results.csv，並印出最佳參數。
"""

import csv
from pathlib import Path

import optuna
import torch

from src import ROOT
from src.models.detector import DentalDetector

DATA_YAML    = str(ROOT / "data" / "dataset.yaml")
MODEL_CFG    = str(ROOT / "configs" / "model" / "yolo12n.yaml")
TUNE_DIR     = ROOT / "experiments" / "tune_v2"

# ── 搜尋空間 ──────────────────────────────────────────
LR_LOW           = 0.0001
LR_HIGH          = 0.01
BATCH_CHOICES    = [8, 16]
OPTIMIZER_CHOICES = ["AdamW", "SGD"]
IMGSZ_CHOICES    = [640, 1024]

# ── 每次試驗的訓練設定（短跑，節省時間）────────────────
EPOCHS_PER_TRIAL = 80
PATIENCE         = 25
DEVICE           = "0"
N_TRIALS         = 50


def objective(trial: optuna.Trial) -> float:
    lr         = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
    batch_size = trial.suggest_categorical("batch_size", BATCH_CHOICES)
    optimizer  = trial.suggest_categorical("optimizer", OPTIMIZER_CHOICES)
    img_size   = trial.suggest_categorical("img_size", IMGSZ_CHOICES)

    trial_dir = TUNE_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = {
        "epochs":        EPOCHS_PER_TRIAL,
        "batch_size":    batch_size,
        "learning_rate": lr,
        "optimizer":     optimizer,
        "patience":      PATIENCE,
        "img_size":      img_size,
        "device":        DEVICE,
        "seed":          0,
        "workers":       2,
        "plots":         False,
    }

    try:
        detector = DentalDetector(MODEL_CFG)
        best_pt  = detector.train(DATA_YAML, train_cfg, str(trial_dir))
        detector.load_weights(str(best_pt))
        metrics  = detector.validate(DATA_YAML, split="val", device=DEVICE)
        map50    = metrics["mAP50"]
        p, r     = metrics["precision"], metrics["recall"]
        f1       = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
        trial.set_user_attr("mAP50_95",  metrics["mAP50_95"])
        trial.set_user_attr("precision", p)
        trial.set_user_attr("recall",    r)
        trial.set_user_attr("f1",        f1)
    except torch.cuda.OutOfMemoryError:
        print(f"Trial {trial.number:>3} | OOM — batch={batch_size}, imgsz={img_size}，跳過")
        return 0.0

    print(
        f"Trial {trial.number:>3} | mAP50={map50:.4f}  f1={f1:.4f} | "
        f"lr={lr:.5f}  batch={batch_size}  opt={optimizer}  imgsz={img_size}"
    )
    return map50


def save_results(study: optuna.Study) -> Path:
    results_path = TUNE_DIR / "results.csv"
    trials = sorted(study.trials, key=lambda t: t.value or 0.0, reverse=True)

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "trial", "mAP50", "mAP50_95", "precision", "recall", "f1", "lr", "batch_size", "optimizer", "img_size"])
        for rank, t in enumerate(trials, 1):
            writer.writerow([
                rank,
                t.number,
                round(t.value, 4) if t.value else "",
                t.user_attrs.get("mAP50_95",  ""),
                t.user_attrs.get("precision", ""),
                t.user_attrs.get("recall",    ""),
                t.user_attrs.get("f1",        ""),
                round(t.params.get("lr", 0), 6),
                t.params.get("batch_size", ""),
                t.params.get("optimizer", ""),
                t.params.get("img_size", ""),
            ])

    return results_path


def main():
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path = TUNE_DIR / "study.db"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="yolo12n_tune",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    done = len(study.trials)
    remaining = N_TRIALS - done
    if remaining <= 0:
        print(f"已完成全部 {N_TRIALS} 次試驗，直接輸出結果。")
    else:
        if done > 0:
            print(f"繼續上次進度：已完成 {done} / {N_TRIALS}，剩餘 {remaining} 次。")
        study.optimize(objective, n_trials=remaining)

    results_path = save_results(study)

    best = study.best_trial
    sep = "=" * 42
    print(f"\n{sep}")
    print("  最佳結果")
    print(sep)
    print(f"  mAP50      : {best.value:.4f}")
    print(f"  lr         : {best.params['lr']:.6f}")
    print(f"  batch_size : {best.params['batch_size']}")
    print(f"  optimizer  : {best.params['optimizer']}")
    print(f"  img_size   : {best.params['img_size']}")
    print(sep)
    print(f"\n完整結果已儲存至 {results_path}")
    print("下一步：將最佳參數填入 configs/experiment/ 並用完整 epochs 訓練。")


if __name__ == "__main__":
    main()

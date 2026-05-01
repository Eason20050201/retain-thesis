"""
實驗主流程入口：訓練 + 推論一次完成。

用法：
    # 完整流程（訓練 → val 評估 → test 評估 → scores.csv）
    python -m src.predict --experiment configs/experiment/baseline.yaml

    # 只做推論（已有訓練好的權重）
    python -m src.predict --experiment configs/experiment/baseline.yaml \\
        --weights experiments/baseline_xxx/weights/best.pt
"""

import argparse
from pathlib import Path

import yaml

from src import ROOT
from src.evaluate import print_scores, save_scores_csv
from src.models import DentalDetector


def load_experiment(exp_path: str) -> dict:
    with open(exp_path) as f:
        return yaml.safe_load(f)


def make_output_dir(exp_path: str) -> Path:
    name = Path(exp_path).stem          # 從檔名取，例如 exp001_yolo12n_adamw
    out = ROOT / "experiments" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_pipeline(exp_cfg: dict, output_dir: Path, weights: str | None) -> None:
    model_cfg_path = str(ROOT / "configs" / exp_cfg["model"])
    data_yaml = str(ROOT / "data" / "dataset.yaml")
    device = exp_cfg["train"].get("device", "mps")

    detector = DentalDetector(model_cfg_path)

    # ── 1. 訓練（若沒有指定既有權重）────────────────────────────────
    if weights is None:
        print("\n[1/3] 開始訓練...")
        best_pt = detector.train(data_yaml, exp_cfg["train"], str(output_dir))
        print(f"      訓練完成，best weights: {best_pt}")
        detector.load_weights(str(best_pt))
    else:
        best_pt = Path(weights)
        print(f"\n[1/3] 跳過訓練，載入既有權重: {best_pt}")
        detector.load_weights(str(best_pt))

    # ── 2. Val 評估（best.pt 對 val 集）─────────────────────────────
    print("\n[2/3] 對 val 集評估...")
    val_metrics = detector.validate(data_yaml, split="val", device=device)

    # ── 3. Test 評估（best.pt 對 test 集）───────────────────────────
    print("\n[3/3] 對 test 集評估...")
    test_metrics = detector.validate(data_yaml, split="test", device=device)

    # ── 結果輸出 ─────────────────────────────────────────────────────
    scores_path = save_scores_csv(output_dir, val_metrics, test_metrics)

    print(f"\n結果已儲存至 {scores_path}")
    print_scores(val_metrics, test_metrics)


def main():
    parser = argparse.ArgumentParser(description="Dental X-ray Pipeline")
    parser.add_argument("--experiment", required=True, help="實驗設定檔路徑")
    parser.add_argument(
        "--weights", default=None,
        help="跳過訓練，直接用此權重做評估（例如 experiments/xxx/weights/best.pt）"
    )
    args = parser.parse_args()

    exp_cfg = load_experiment(args.experiment)
    output_dir = make_output_dir(args.experiment)
    (output_dir / "experiment.yaml").write_text(yaml.dump(exp_cfg, allow_unicode=True))

    run_pipeline(exp_cfg, output_dir, args.weights)


if __name__ == "__main__":
    main()

"""
實驗主流程入口：訓練 + 推論一次完成。

用法：
    # 完整流程（訓練 → val 評估 → test 評估）
    python -m src.predict --experiment configs/experiment/exp001_yolo12n_adamw.yaml

    # 只做評估（已有訓練好的權重）
    python -m src.predict --experiment configs/experiment/exp001_yolo12n_adamw.yaml \
        --weights experiments/exp001_yolo12n_adamw/train/weights/best.pt
"""

import argparse
from pathlib import Path

import yaml

from src import ROOT
from src.evaluate import print_scores, save_scores_csv
from src.models import DentalDetector
from src.preprocess import PreprocessPipeline


def load_experiment(exp_path: str) -> dict:
    with open(exp_path) as f:
        return yaml.safe_load(f)


def make_output_dir(exp_path: str) -> Path:
    name = Path(exp_path).stem
    out = ROOT / "experiments" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_merged_yaml(data_yaml: str, output_dir: Path) -> str:
    """
    train + val 合併成訓練資料，val 也指向同一份（讓 ultralytics 決定 best.pt），
    test 保持獨立，不參與任何訓練過程。
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    src_root = Path(data_yaml).parent

    # 收集 train + val 所有圖片路徑寫成 txt
    merged_txt = output_dir / "merged_train_val.txt"
    imgs = []
    for split in ("train", "valid"):
        img_dir = src_root / split / "images"
        if img_dir.exists():
            imgs += sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    merged_txt.write_text("\n".join(str(p) for p in imgs))
    print(f"      merged train+val: {len(imgs)} 張圖片")

    merged_cfg = dict(cfg)
    merged_cfg["train"] = str(merged_txt)
    merged_cfg["val"]   = str(merged_txt)   # early stopping 依訓練資料表現，test 完全獨立
    merged_cfg["test"]  = str(src_root / "test" / "images")

    merged_yaml_path = output_dir / "dataset_merged.yaml"
    with open(merged_yaml_path, "w") as f:
        yaml.dump(merged_cfg, f, allow_unicode=True)

    return str(merged_yaml_path)


def run_pipeline(exp_cfg: dict, output_dir: Path, weights: str | None) -> None:
    model_cfg_path = str(ROOT / "configs" / exp_cfg["model"])
    data_yaml = str(ROOT / "data" / "dataset.yaml")
    device = exp_cfg["train"].get("device", "auto")
    merge_val = exp_cfg.get("merge_val_to_train", False)

    # ── 0. 前處理（若有設定 preprocess）─────────────────────────────
    if "preprocess" in exp_cfg:
        preprocess_cfg_path = str(ROOT / "configs" / exp_cfg["preprocess"])
        print("\n[0/3] 執行前處理...")
        pipeline = PreprocessPipeline(preprocess_cfg_path)
        data_yaml = pipeline.run(data_yaml)

    # ── merge val into train ──────────────────────────────────────
    if merge_val:
        print("\n train + val 合併訓練，僅對 test 評估")
        data_yaml = build_merged_yaml(data_yaml, output_dir)

    detector = DentalDetector(model_cfg_path)

    # ── 1. 訓練 ───────────────────────────────────────────────────
    if weights is None:
        print("\n[1/3] 開始訓練...")
        best_pt = detector.train(data_yaml, exp_cfg["train"], str(output_dir))
        print(f"      訓練完成，best weights: {best_pt}")
        detector.load_weights(str(best_pt))
    else:
        best_pt = Path(weights)
        print(f"\n[1/3] 跳過訓練，載入既有權重: {best_pt}")
        detector.load_weights(str(best_pt))

    # ── 2. Val 評估（merge 模式下跳過）───────────────────────────
    val_metrics = None
    if not merge_val:
        print("\n[2/3] 對 val 集評估...")
        val_metrics = detector.validate(data_yaml, split="val", device=device)
    else:
        print("\n[2/3] 跳過 val 評估（train+val 已合併入訓練）")

    # ── 3. Test 評估 ──────────────────────────────────────────────
    print("\n[3/3] 對 test 集評估...")
    test_metrics = detector.validate(data_yaml, split="test", device=device)

    # ── 結果輸出 ──────────────────────────────────────────────────
    scores_path = save_scores_csv(output_dir, test_metrics, val_metrics)
    print(f"\n結果已儲存至 {scores_path}")
    print_scores(test_metrics, val_metrics)


def main():
    parser = argparse.ArgumentParser(description="Dental X-ray Pipeline")
    parser.add_argument("--experiment", required=True, help="實驗設定檔路徑")
    parser.add_argument(
        "--weights", default=None,
        help="跳過訓練，直接用此權重做評估"
    )
    args = parser.parse_args()

    exp_cfg = load_experiment(args.experiment)
    output_dir = make_output_dir(args.experiment)
    (output_dir / "experiment.yaml").write_text(yaml.dump(exp_cfg, allow_unicode=True))

    run_pipeline(exp_cfg, output_dir, args.weights)


if __name__ == "__main__":
    main()

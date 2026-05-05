"""
用法：
    python -m src.train --experiment configs/experiment/exp001_yolo12n_adamw.yaml
    python -m src.train --experiment configs/experiment/exp001_yolo12n_adamw.yaml \
        --weights experiments/exp001_yolo12n_adamw/train/weights/best.pt
"""

import argparse
from pathlib import Path

import yaml

from src import ROOT
from src.evaluate import print_scores, save_scores_csv
from src.models import DentalDetector


def build_merged_yaml(data_yaml: str, output_dir: Path) -> str:
    """把 train + valid 合併成一份 list，產生新的 dataset yaml：
    train = val = merged_txt（讓 ultralytics 用 train 自己當 val 挑 best.pt）
    test 保持原路徑。
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    base_path = Path(cfg.get("path", Path(data_yaml).parent))
    if not base_path.is_absolute():
        base_path = (Path(data_yaml).parent / base_path).resolve()

    imgs = []
    for split_key in ("train", "val"):
        split_rel = cfg.get(split_key)
        if not split_rel:
            continue
        split_dir = Path(split_rel)
        if not split_dir.is_absolute():
            split_dir = base_path / split_rel
        if split_dir.exists():
            imgs += list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))

    # 全域依檔名排序，跟 NTU 腳本一致（先收齊再排，避免 per-folder sort 造成順序不同）
    imgs = sorted(imgs, key=lambda p: p.name)

    merged_txt = output_dir / "merged_train_val.txt"
    merged_txt.write_text("\n".join(str(p) for p in imgs))
    print(f"merge_val_to_train: 合併 {len(imgs)} 張圖片 → {merged_txt}")

    merged_cfg = dict(cfg)
    merged_cfg["train"] = str(merged_txt)
    merged_cfg["val"]   = str(merged_txt)
    # test 保持原樣

    merged_yaml_path = output_dir / "dataset_merged.yaml"
    with open(merged_yaml_path, "w") as f:
        yaml.dump(merged_cfg, f, allow_unicode=True)

    return str(merged_yaml_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    with open(args.experiment) as f:
        exp_cfg = yaml.safe_load(f)

    train_cfg = exp_cfg["train"]
    model_cfg_path = str(ROOT / "configs" / exp_cfg["model"])
    data_yaml = str(ROOT / "data" / exp_cfg.get("dataset", "dataset.yaml"))
    device = train_cfg.get("device", "auto")
    merge_val = exp_cfg.get("merge_val_to_train", False)

    name = Path(args.experiment).stem
    output_dir = ROOT / "experiments" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    if merge_val:
        data_yaml = build_merged_yaml(data_yaml, output_dir)

    detector = DentalDetector(model_cfg_path)

    if args.weights is None:
        best_pt = detector.train(data_yaml, train_cfg, str(output_dir))
        detector.load_weights(str(best_pt))
    else:
        detector.load_weights(args.weights)

    val_metrics = None
    if not merge_val:
        val_metrics = detector.validate(data_yaml, split="val", device=device)

    test_metrics = detector.validate(data_yaml, split="test", device=device)

    scores_path = save_scores_csv(output_dir, test_metrics, val_metrics)
    print(f"\n結果已儲存至 {scores_path}")
    print_scores(test_metrics, val_metrics)


if __name__ == "__main__":
    main()

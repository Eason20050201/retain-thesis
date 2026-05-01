from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def resolve_device(device: str) -> str:
    """auto → 依序選 cuda > mps > cpu。"""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DentalDetector:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        weights = self.cfg.get("weights", "yolov8n-seg.pt")
        self.model = YOLO(weights)
        self.inf_cfg = self.cfg.get("inference", {})

    def train(self, data_yaml: str, train_cfg: dict, output_dir: str) -> Path:
        """訓練並回傳 best.pt 的路徑。"""
        kwargs = {
            "data":     data_yaml,
            "epochs":   train_cfg.get("epochs", 100),
            "batch":    train_cfg.get("batch_size", 16),
            "imgsz":    train_cfg.get("img_size", self.inf_cfg.get("img_size", 640)),
            "lr0":      train_cfg.get("learning_rate", 0.01),
            "optimizer": train_cfg.get("optimizer", "SGD"),
            "patience": train_cfg.get("patience", 50),
            "device":   resolve_device(train_cfg.get("device", "auto")),
            "project":  output_dir,
            "name":     "train",
            "exist_ok": True,
        }
        # 只有 yaml 有設才傳給 ultralytics，其餘用 ultralytics 預設
        for key, cfg_key in [("weight_decay", "weight_decay"),
                              ("warmup_epochs", "warmup_epochs"),
                              ("save_period", "save_period"),
                              ("workers", "workers"),
                              ("seed", "seed")]:
            if cfg_key in train_cfg:
                kwargs[key] = train_cfg[cfg_key]

        self.model.train(**kwargs)
        best_path = Path(output_dir) / "train" / "weights" / "best.pt"
        return best_path

    def load_weights(self, weights_path: str) -> None:
        self.model = YOLO(weights_path)

    def validate(self, data_yaml: str, split: str, device: str) -> dict:
        """對指定 split 執行 validation，回傳 metrics dict。"""
        metrics = self.model.val(
            data=data_yaml,
            split=split,
            device=resolve_device(device),
            verbose=False,
        )
        task = self.cfg.get("task", "segment")
        m = metrics.seg if task == "segment" else metrics.box
        return {
            "mAP50":     round(float(m.map50), 4),
            "mAP50_95":  round(float(m.map),   4),
            "precision": round(float(m.mp),     4),
            "recall":    round(float(m.mr),     4),
        }

    def predict(self, source: str) -> list:
        return self.model.predict(
            source=source,
            conf=self.inf_cfg.get("conf_threshold", 0.25),
            iou=self.inf_cfg.get("iou_threshold", 0.45),
            max_det=self.inf_cfg.get("max_det", 50),
            imgsz=self.inf_cfg.get("img_size", 640),
            device=resolve_device(self.inf_cfg.get("device", "auto")),
        )

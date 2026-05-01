import shutil
from pathlib import Path

import albumentations as A
import cv2
import yaml
from tqdm import tqdm


class PreprocessPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        img_cfg = self.cfg["image"]
        h, w = img_cfg["size"]
        self.output_dir = Path(self.cfg["output_dir"])

        aug_cfg = self.cfg["augmentation"]
        transforms = [A.Resize(h, w)]

        if aug_cfg.get("enabled"):
            clahe = aug_cfg.get("clahe", {})
            if clahe.get("p", 0) > 0:
                transforms.append(
                    A.CLAHE(
                        clip_limit=clahe["clip_limit"],
                        tile_grid_size=clahe["tile_grid_size"],
                        p=clahe["p"],
                    )
                )
            rbc = aug_cfg.get("random_brightness_contrast", {})
            if rbc.get("p", 0) > 0:
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=rbc["brightness_limit"],
                        contrast_limit=rbc["contrast_limit"],
                        p=rbc["p"],
                    )
                )

        # 只套用不影響 bounding box 的 transform（無幾何變換）
        self.transform = A.Compose(transforms)

    def run(self, data_yaml_path: str) -> str:
        """
        讀取原始 dataset.yaml，對每個 split 做前處理，
        輸出新的 dataset_processed.yaml。
        """
        with open(data_yaml_path) as f:
            dataset_cfg = yaml.safe_load(f)

        src_root = Path(data_yaml_path).parent

        for split in ("train", "val", "test"):
            split_img_dir = src_root / f"{split}" / "images"
            split_lbl_dir = src_root / f"{split}" / "labels"
            if not split_img_dir.exists():
                continue
            self._process_split(split_img_dir, split_lbl_dir, split)

        # 產生新的 dataset yaml 指向 processed 資料
        processed_yaml_path = self.output_dir / "dataset_processed.yaml"
        processed_cfg = dict(dataset_cfg)
        processed_cfg["path"] = str(self.output_dir)
        processed_cfg["train"] = "train/images"
        processed_cfg["val"]   = "val/images"
        processed_cfg["test"]  = "test/images"

        processed_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(processed_yaml_path, "w") as f:
            yaml.dump(processed_cfg, f, allow_unicode=True)

        print(f"前處理完成，dataset yaml: {processed_yaml_path}")
        return str(processed_yaml_path)

    def _process_split(self, img_dir: Path, lbl_dir: Path, split: str) -> None:
        out_img_dir = self.output_dir / split / "images"
        out_lbl_dir = self.output_dir / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        for img_path in tqdm(img_paths, desc=f"preprocess [{split}]"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.transform(image=img)["image"]
            out_img = out_img_dir / img_path.name
            cv2.imwrite(str(out_img), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

            # label 直接複製（無幾何變換，座標不變）
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy(lbl_path, out_lbl_dir / lbl_path.name)

import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class PreprocessPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.img_size = tuple(self.cfg["image"]["size"])
        self.output_dir = Path(self.cfg["output_dir"])
        self.splits_dir = Path(self.cfg["splits_dir"])

        aug_cfg = self.cfg["augmentation"]
        transforms = [A.Resize(*self.img_size)]
        if aug_cfg.get("enabled"):
            if aug_cfg.get("horizontal_flip", 0) > 0:
                transforms.append(A.HorizontalFlip(p=aug_cfg["horizontal_flip"]))
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
        self.transform = A.Compose(transforms)

    def run(self, raw_dir: str) -> None:
        raw_dir = Path(raw_dir)
        image_paths = sorted(raw_dir.rglob("*.jpg")) + sorted(raw_dir.rglob("*.png"))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        processed = []
        for img_path in tqdm(image_paths, desc="Preprocessing"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.transform(image=img)["image"]
            out_path = self.output_dir / img_path.name
            cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            processed.append(str(out_path))

        self._make_splits(processed)

    def _make_splits(self, paths: list[str]) -> None:
        split_cfg = self.cfg["split"]
        seed = split_cfg["seed"]
        val_ratio = split_cfg["val_ratio"]
        test_ratio = split_cfg["test_ratio"]

        train_paths, test_paths = train_test_split(
            paths, test_size=test_ratio, random_state=seed
        )
        adjusted_val = val_ratio / (1 - test_ratio)
        train_paths, val_paths = train_test_split(
            train_paths, test_size=adjusted_val, random_state=seed
        )

        for split_name, split_paths in [
            ("train", train_paths),
            ("val", val_paths),
            ("test", test_paths),
        ]:
            out_file = self.splits_dir / f"{split_name}.txt"
            out_file.write_text("\n".join(split_paths))
            print(f"{split_name}: {len(split_paths)} images → {out_file}")

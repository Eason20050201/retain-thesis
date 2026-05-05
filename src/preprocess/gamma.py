"""
Gamma Correction 前處理：output = (input / 255) ^ gamma * 255

gamma < 1：影像變亮（暗部細節更明顯，適合 X 光）
gamma > 1：影像變暗

用法：
    python -m src.preprocess.gamma
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

from src import ROOT

SRC_ROOT = ROOT / "data"
DST_ROOT = ROOT / "data_gamma"
SPLITS   = ["train", "valid", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

GAMMA = 0.5

# 預先建 LUT 加速（只算一次）
_lut = np.array([((i / 255.0) ** GAMMA) * 255 for i in range(256)], dtype=np.uint8)


def process_one(img_bgr: np.ndarray) -> np.ndarray:
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corrected = cv2.LUT(gray, _lut)
    return cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)


def convert_split(split: str) -> tuple[int, int]:
    src_img = SRC_ROOT / split / "images"
    src_lbl = SRC_ROOT / split / "labels"
    dst_img = DST_ROOT / split / "images"
    dst_lbl = DST_ROOT / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n_img = 0
    for p in sorted(src_img.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cv2.imwrite(str(dst_img / p.name), process_one(img))
        n_img += 1

    n_lbl = 0
    for p in sorted(src_lbl.iterdir()):
        if p.suffix.lower() != ".txt":
            continue
        shutil.copy2(p, dst_lbl / p.name)
        n_lbl += 1

    return n_img, n_lbl


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        n_img, n_lbl = convert_split(split)
        print(f"{split:5s}: {n_img} 張影像 / {n_lbl} 個標註 → {DST_ROOT / split}")
    print(f"\n完成。新資料集根目錄：{DST_ROOT} (gamma={GAMMA})")


if __name__ == "__main__":
    main()

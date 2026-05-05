"""
CLAHE 前處理：固定 clipLimit=2.0, tileGridSize=(8,8)。

用法：
    python -m src.preprocess.clahe
"""

import shutil
from pathlib import Path

import cv2

from src import ROOT

SRC_ROOT = ROOT / "data"
DST_ROOT = ROOT / "data_clahe"
SPLITS   = ["train", "valid", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def process_one(img_bgr):
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = CLAHE.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def convert_split(split):
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
    print(f"\n完成。新資料集根目錄：{DST_ROOT}")


if __name__ == "__main__":
    main()

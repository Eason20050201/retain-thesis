"""
將 data/{train,valid,test}/images 轉灰階後存到 data_gray/ 對應位置。
標註檔（labels）以硬連結複製，bbox 不變。

用法：
    python -m src.preprocess.grayscale
"""

import shutil
from pathlib import Path

import cv2

from src import ROOT

SRC_ROOT = ROOT / "data"
DST_ROOT = ROOT / "data_gray"
SPLITS = ["train", "valid", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def convert_split(split: str) -> tuple[int, int]:
    src_img_dir = SRC_ROOT / split / "images"
    src_lbl_dir = SRC_ROOT / split / "labels"
    dst_img_dir = DST_ROOT / split / "images"
    dst_lbl_dir = DST_ROOT / split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_count = 0
    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  跳過讀取失敗: {img_path.name}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(dst_img_dir / img_path.name), gray_3ch)
        img_count += 1

    lbl_count = 0
    for lbl_path in sorted(src_lbl_dir.iterdir()):
        if lbl_path.suffix.lower() != ".txt":
            continue
        shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)
        lbl_count += 1

    return img_count, lbl_count


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        imgs, lbls = convert_split(split)
        print(f"{split:5s}: {imgs} 張影像 / {lbls} 個標註 → {DST_ROOT / split}")
    print(f"\n完成。新資料集根目錄：{DST_ROOT}")


if __name__ == "__main__":
    main()

"""
PS-KDE (Pixel Substitution via Kernel Density Estimation)

1. 計算灰階直方圖（256 bins）
2. 用 Gaussian smoothing 近似 KDE（數學等價，速度遠快於 scipy gaussian_kde）
   - 頻寬以 Silverman rule 決定：h = 1.06 × σ × n^(-1/5)
3. 由平滑後 PDF 積分出 CDF，以 CDF×255 建立 LUT
4. 用 LUT 重映射像素值（等效於 KDE-based CDF 均衡化）

gamma<1 的 X 光影像版本：此法讓分布均勻化，強化中間灰階對比

用法：
    python -m src.preprocess.kde
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src import ROOT

SRC_ROOT = ROOT / "data"
DST_ROOT = ROOT / "data_kde"
SPLITS   = ["train", "valid", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def process_one(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    flat = gray.ravel().astype(float)
    n = flat.size

    # histogram as discrete PDF
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)

    # Silverman bandwidth (in pixel-value units 0-255)
    sigma_data = float(np.std(flat))
    h = max(1.06 * sigma_data * (n ** (-0.2)), 0.5)   # clamp to ≥0.5

    # KDE via Gaussian-smoothed histogram
    pdf = gaussian_filter1d(hist, sigma=h)
    pdf = np.maximum(pdf, 0.0)
    pdf /= pdf.sum()

    # CDF → LUT
    cdf = np.cumsum(pdf)
    lut = np.round(cdf * 255).clip(0, 255).astype(np.uint8)

    enhanced = cv2.LUT(gray, lut)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


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
    print(f"\n完成。新資料集根目錄：{DST_ROOT}")


if __name__ == "__main__":
    main()

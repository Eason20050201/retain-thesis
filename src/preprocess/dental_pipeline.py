"""
牙科 X 光多階段前處理 pipeline。

順序：
  1. Grayscale conversion
  2. Morphological Opening (cv2.MORPH_OPEN：標準定義是 erosion → dilation)
  3. ABC-based adaptive thresholding (Otsu between-class variance 為適應度)
  4. Adaptive Gaussian blurring (kernel size 由 Shannon entropy 決定)
  5. Image merging：binary==0 → original，binary!=0 → blurred
  6. Histogram equalization
  7. Morphological Closing (cv2.MORPH_CLOSE：dilation → erosion)
  8. Weighted summation：0.5 * original - 0.5 * closed + 128（避免負值溢位）

輸出到 data_pipeline/{train,valid,test}/。標籤直接複製。

用法：
    python -m src.preprocess.dental_pipeline
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

from src import ROOT

SRC_ROOT = ROOT / "data"
DST_ROOT = ROOT / "data_pipeline2"
SPLITS = ["train", "valid", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SEED = 0

KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


# ─── 基礎運算 ──────────────────────────────────────────

def shannon_entropy(gray: np.ndarray) -> float:
    h = np.bincount(gray.ravel(), minlength=256).astype(float)
    p = h / h.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def adaptive_blur(gray: np.ndarray) -> np.ndarray:
    """以 Shannon entropy 推導 Gaussian kernel size，影像細節越多 kernel 越大。"""
    e = shannon_entropy(gray)            # 自然影像通常 4 ~ 8
    ksize = int(round(e * 2)) | 1        # 取奇數
    ksize = max(3, min(ksize, 15))
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


# ─── ABC 閾值搜尋 ──────────────────────────────────────

def make_otsu_fitness(gray: np.ndarray):
    """預先算累積直方圖，讓 ABC 內部每次評估只要 O(1)。"""
    h = np.bincount(gray.ravel(), minlength=256).astype(float)
    p = h / h.sum()
    cum_p = np.cumsum(p)
    cum_xp = np.cumsum(np.arange(256) * p)
    total_mu = cum_xp[-1]

    def fitness(t: int) -> float:
        w0 = cum_p[t]
        w1 = 1.0 - w0
        if w0 <= 0 or w1 <= 0:
            return 0.0
        mu0 = cum_xp[t] / w0
        mu1 = (total_mu - cum_xp[t]) / w1
        return float(w0 * w1 * (mu0 - mu1) ** 2)

    return fitness


def abc_threshold(gray: np.ndarray, rng: np.random.RandomState,
                  n_bees: int = 20, n_iter: int = 20, limit: int = 5) -> int:
    """Artificial Bee Colony 找最佳單一 threshold (1-254)，目標：Otsu between-class variance。"""
    fitness = make_otsu_fitness(gray)
    sources = rng.randint(1, 255, size=n_bees)
    fits = np.array([fitness(int(t)) for t in sources])
    trials = np.zeros(n_bees, dtype=int)

    def neighbor(i: int) -> int:
        k = int(rng.choice(np.delete(np.arange(n_bees), i)))
        phi = rng.uniform(-1.0, 1.0)
        new_t = int(round(sources[i] + phi * (sources[i] - sources[k])))
        return int(np.clip(new_t, 1, 254))

    for _ in range(n_iter):
        # Employed bees
        for i in range(n_bees):
            new_t = neighbor(i)
            new_fit = fitness(new_t)
            if new_fit > fits[i]:
                sources[i], fits[i], trials[i] = new_t, new_fit, 0
            else:
                trials[i] += 1
        # Onlooker bees
        total = fits.sum()
        probs = (fits / total) if total > 0 else np.full(n_bees, 1 / n_bees)
        for _ in range(n_bees):
            i = int(rng.choice(n_bees, p=probs))
            new_t = neighbor(i)
            new_fit = fitness(new_t)
            if new_fit > fits[i]:
                sources[i], fits[i], trials[i] = new_t, new_fit, 0
            else:
                trials[i] += 1
        # Scout bees
        for i in range(n_bees):
            if trials[i] > limit:
                sources[i] = rng.randint(1, 255)
                fits[i] = fitness(int(sources[i]))
                trials[i] = 0

    return int(sources[int(fits.argmax())])


# ─── Pipeline ─────────────────────────────────────────

def process_one(img_bgr: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)                 # 1
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, KERNEL)          # 2
    t = abc_threshold(opened, rng=rng)                               # 3
    _, binary = cv2.threshold(opened, t, 255, cv2.THRESH_BINARY)
    blurred = adaptive_blur(gray)                                    # 4
    merged = np.where(binary == 0, gray, blurred).astype(np.uint8)   # 5
    eq = cv2.equalizeHist(merged)                                    # 6
    closed = cv2.morphologyEx(eq, cv2.MORPH_CLOSE, KERNEL)           # 7
    final = cv2.addWeighted(gray, 1.5, closed, -0.5, 0)              # 8 unsharp mask
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)


def convert_split(split: str) -> tuple[int, int]:
    src_img_dir = SRC_ROOT / split / "images"
    src_lbl_dir = SRC_ROOT / split / "labels"
    dst_img_dir = DST_ROOT / split / "images"
    dst_lbl_dir = DST_ROOT / split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(SEED)

    img_count = 0
    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  跳過讀取失敗: {img_path.name}")
            continue
        out = process_one(img, rng)
        cv2.imwrite(str(dst_img_dir / img_path.name), out)
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

"""
ABC-CLAHE 前處理：用 Artificial Bee Colony 對每張影像自適應找最佳 CLAHE clipLimit。

適應度：Shannon entropy（entropy 越高 → 對比越豐富 → 越好）
搜尋空間：clipLimit ∈ [1.0, 8.0]，tileGridSize 固定 (8, 8)

用法：
    python -m src.preprocess.clahe_abc
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

from src import ROOT

SRC_ROOT = ROOT / "data"
DST_ROOT = ROOT / "data_clahe_abc"
SPLITS   = ["train", "valid", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SEED     = 0

TILE_GRID  = (8, 8)
CL_LOW, CL_HIGH = 1.0, 8.0


# ─── 適應度 ────────────────────────────────────────────

def shannon_entropy(gray: np.ndarray) -> float:
    h = np.bincount(gray.ravel(), minlength=256).astype(float)
    p = h / h.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def clahe_entropy(gray: np.ndarray, clip: float) -> float:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=TILE_GRID)
    return shannon_entropy(clahe.apply(gray))


# ─── ABC ───────────────────────────────────────────────

def abc_clahe_clip(gray: np.ndarray, rng: np.random.RandomState,
                   n_bees: int = 20, n_iter: int = 20, limit: int = 5) -> float:
    """回傳令 CLAHE entropy 最大的 clipLimit。"""
    sources = rng.uniform(CL_LOW, CL_HIGH, size=n_bees)
    fits    = np.array([clahe_entropy(gray, c) for c in sources])
    trials  = np.zeros(n_bees, dtype=int)

    def neighbor(i: int) -> float:
        k   = int(rng.choice(np.delete(np.arange(n_bees), i)))
        phi = rng.uniform(-1.0, 1.0)
        new_c = sources[i] + phi * (sources[i] - sources[k])
        return float(np.clip(new_c, CL_LOW, CL_HIGH))

    for _ in range(n_iter):
        # Employed
        for i in range(n_bees):
            nc = neighbor(i)
            nf = clahe_entropy(gray, nc)
            if nf > fits[i]:
                sources[i], fits[i], trials[i] = nc, nf, 0
            else:
                trials[i] += 1
        # Onlooker
        total = fits.sum()
        probs = (fits / total) if total > 0 else np.full(n_bees, 1 / n_bees)
        for _ in range(n_bees):
            i  = int(rng.choice(n_bees, p=probs))
            nc = neighbor(i)
            nf = clahe_entropy(gray, nc)
            if nf > fits[i]:
                sources[i], fits[i], trials[i] = nc, nf, 0
            else:
                trials[i] += 1
        # Scout
        for i in range(n_bees):
            if trials[i] > limit:
                sources[i] = rng.uniform(CL_LOW, CL_HIGH)
                fits[i]    = clahe_entropy(gray, sources[i])
                trials[i]  = 0

    return float(sources[int(fits.argmax())])


# ─── 前處理主流程 ──────────────────────────────────────

def process_one(img_bgr: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    best_cl  = abc_clahe_clip(gray, rng)
    clahe    = cv2.createCLAHE(clipLimit=best_cl, tileGridSize=TILE_GRID)
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def convert_split(split: str) -> tuple[int, int]:
    src_img = SRC_ROOT / split / "images"
    src_lbl = SRC_ROOT / split / "labels"
    dst_img = DST_ROOT / split / "images"
    dst_lbl = DST_ROOT / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(SEED)

    n_img = 0
    for p in sorted(src_img.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skip: {p.name}")
            continue
        cv2.imwrite(str(dst_img / p.name), process_one(img, rng))
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

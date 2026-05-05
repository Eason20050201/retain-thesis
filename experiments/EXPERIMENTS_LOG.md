# 實驗結果 Log

YOLOv12n 牙齒偵測（detect 任務，單類別 `tooth`）
資料集：train 81 / valid 20 / test 26 張
最後更新：2026-05-04

---

## 1. tune_v2 超參數搜尋（80 epoch × 50 trials）

腳本：[src/tune.py](src/tune.py)
搜尋空間：lr ∈ [1e-4, 1e-2] log，batch ∈ {8,16}，optimizer ∈ {AdamW,SGD}，imgsz ∈ {640,1024}
固定：epochs=80, patience=25, seed=0, workers=2

完整結果：[experiments/tune_v2/results.csv](experiments/tune_v2/results.csv)

**Top 10（依 val mAP50 排序）**

| rank | trial | val mAP50 | mAP50_95 | precision | recall | f1 | lr | batch | optimizer | imgsz |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 46 | 0.8487 | 0.3469 | 0.7777 | 0.8387 | 0.807 | 0.000438 | 16 | AdamW | 1024 |
| 2 | 40 | 0.7935 | 0.2928 | 0.7199 | 0.7419 | 0.7307 | 0.000676 | 16 | AdamW | 1024 |
| 3 | 48 | 0.7869 | 0.3402 | 0.7152 | 0.8065 | 0.7581 | 0.000534 | 16 | AdamW | 1024 |
| 4 | 32 | 0.7837 | 0.2832 | 0.7872 | 0.7161 | 0.75   | 0.001090 | 16 | AdamW | 1024 |
| 5 | 3  | 0.7836 | 0.3094 | 0.6365 | 0.8065 | 0.7115 | 0.000231 | 16 | AdamW | 1024 |
| 6 | 25 | 0.7762 | 0.3262 | 0.7227 | 0.8065 | 0.7623 | 0.000259 | 16 | AdamW | 1024 |
| 7 | 45 | 0.7604 | 0.3234 | 0.7003 | 0.8387 | 0.7633 | 0.000304 | 16 | AdamW | 1024 |
| 8 | 29 | 0.7563 | 0.3525 | 0.7406 | 0.7370 | 0.7388 | 0.000195 | 16 | AdamW | 1024 |
| 9 | 26 | 0.7530 | 0.3313 | 0.7268 | 0.7742 | 0.7498 | 0.000373 | 16 | AdamW | 1024 |
| 10 | 1 | 0.7514 | 0.2662 | 0.8611 | 0.6001 | 0.7073 | 0.000561 | 8  | AdamW | 640 |

**穩定區觀察**：top-9 全是 `AdamW + batch=16 + imgsz=1024 + lr ∈ [0.0002, 0.0011]`。SGD 在 80 epoch 內表現不佳（最高排名 23，mAP50=0.7095）。

---

## 2. 各正式實驗 scores.csv 彙整

| exp | 來源 | 實際模式 | val_best mAP50 | test mAP50 | test precision | test recall | test f1 |
|---|---|---|---|---|---|---|---|
| baseline | YOLOv12n 預訓練未微調 | 直接推論 | 0.0023 | 0.0025 | 0.0066 | 0.3898 | 0.013 |
| exp001 | 早期 AdamW | 重訓 | 0.5161 | 0.3153 | 0.6000 | 0.0508 | 0.0937 |
| exp002 | merged train+val | 重訓 | — | 0.6329 | 0.5638 | 0.6573 | 0.6070 |
| exp003 | SGD baseline | 重訓 | 0.6761 | 0.6037 | 0.6257 | 0.5384 | 0.5788 |
| exp004 | tune_v2 SGD trial_031 (lr=0.002445, b=8, 200ep) | 重訓 | 0.7718 | 0.6549 | 0.8418 | 0.5593 | 0.6721 |
| exp005 | tune_v2 trial_046 best.pt（rank 1） | --weights 評估 | 0.8487 | **0.6373** | 0.7150 | 0.5593 | 0.6276 |
| exp006 | tune_v2 trial_040 best.pt（rank 2） | --weights 評估 | 0.7935 | **0.6620** | 0.8592 | 0.5763 | 0.6899 |
| exp007 | tune_v2 trial_048 best.pt（rank 3） | --weights 評估 | 0.7869 | **0.6386** | 0.7120 | 0.5593 | 0.6265 |

`val_best` = 該模型在 valid 20 張上的 mAP50；`test` = 該模型在 test 26 張上的 mAP50。

---

## 3. 重訓 vs tune 結果差異（exp005 / exp006 案例）

問題：相同超參數寫進 config 重訓，val 結果跟 tune trial 不一致。

**exp005 重訓 vs tune trial_046**（兩者參數相同：lr=0.000438, AdamW, b=16, imgsz=1024, ep=80, seed=0）

| 項目 | trial_046 | exp005 重訓 |
|---|---|---|
| 訓練過程 best epoch | 33 | 25 |
| 該 epoch val mAP50 | 0.87315 | 0.78556 |
| best.pt 檔案大小 | 5,538,586 bytes | 5,537,178 bytes |
| 觸發 patience 停止於 | epoch 58 | epoch 47 |

**exp006 重訓 vs tune trial_040**（lr=0.000676, AdamW, b=16, imgsz=1024）

| 項目 | trial_040 | exp006 重訓 |
|---|---|---|
| 訓練過程 best mAP50 | 0.7935 | 0.7411 |
| best.pt 大小 | 5,538,330 bytes | 5,541,082 bytes |

**根因**：

1. **CUDA / cuDNN 非確定性** — AMP 浮點累加、cuDNN backward atomic add，`seed=0 + deterministic=true` 控不住。
2. **DataLoader workers=2** 引入 batch 順序非確定性。
3. **lr 精度截斷** — Optuna 原始 lr `0.00043822823657896506`，config 寫 `0.000438`，差約 0.05%，80 epoch 累積會放大。
4. **val set 只有 20 張** — 1–2 張預測翻轉就會讓 mAP50 變動 0.05–0.1。

**結論**：tune 結果不能用「同參數重訓」驗證。要評估 tune 找到的模型，直接用該 trial 的 `best.pt` 跑 test 才是有效做法（exp005/006/007 的 scores.csv 即為此模式）。

---

## 4. 關鍵發現

1. **AdamW @ 80ep 是目前最佳組合**。tune_v2 top-9 全部是 AdamW + batch=16 + imgsz=1024，SGD 在短訓練下落後。
2. **lr 穩定區 ≈ 0.0002 – 0.0011**。trial 第一名（0.000438）的領先程度在 val 噪音範圍內，不一定真的勝過 trial_040 / trial_048。
3. **test 分數彼此接近**（0.6373 / 0.6620 / 0.6386），實際差距遠小於 val。
   - 目前 test mAP50 最佳：**exp006（trial_040 best.pt）= 0.6620**
   - test precision 最佳：**exp006 = 0.8592**
4. **epoch / optimizer 耦合**：先前已知 80ep AdamW 勝、200ep SGD 較穩。tune_v2 全為 80ep，因此偏向 AdamW。若要做 long-run 比較需另跑 200ep tune。
5. **val 僅 20 張**對排名雜訊極大，未來若可能應擴增 val 或改用 cross-validation。

---

## 5. 已知問題 / TODO

- [ ] tune.py 寫入 results.csv 時 lr 截斷到 6 位小數，重訓會跟原 trial 不完全一致 → 改成不 round 或寫完整 float
- [ ] exp005/006 重訓的 train/ 資料夾與 scores.csv 來源不一致（scores.csv 是 --weights 跑的，但 train/ 是另一次重訓的痕跡）→ 後續清理
- [ ] tune trial_022 / trial_034 / trial_000 缺 metrics（中途 OOM 或中斷）
- [ ] 目前無 final 模型 — 若以 test mAP50 為準，候選為 exp006（trial_040）

---

## 6. 怎麼選 final 模型

兩條路：

**A. 直接定 exp006（test mAP50 最佳）作為 final** — 省時、就是 tune 出來最好的 test 表現。
**B. 對 top-3 候選各跑 3 個 seed，取平均** — 真正避開 val 噪音、reportable for thesis。指令模板：
```bash
for seed in 0 1 2; do
  python -m src.train --experiment configs/experiment/exp006_yolo12n_adamw_80ep_t040.yaml
done
```
（需先在 config 加 seed 欄或改 src/tune.py 接受 seed 參數）

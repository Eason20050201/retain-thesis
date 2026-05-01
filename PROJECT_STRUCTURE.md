# 專案結構說明

## 目錄總覽

```
retain-thesis/
├── configs/
│   ├── preprocess/
│   ├── model/
│   └── experiment/
├── data/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── dataset.yaml
├── src/
│   ├── preprocess/
│   ├── models/
│   ├── evaluate.py
│   └── predict.py
├── baselines/
├── experiments/
├── notebooks/
├── submissions/
├── requirements.txt
├── README.md
└── PROJECT_STRUCTURE.md
```

---

## configs/

實驗相關的所有參數設定，拆成三層讓各層可以獨立替換。

### configs/preprocess/

前處理流程的設定，由 `src/preprocess/pipeline.py` 讀取。

| 欄位 | 說明 |
|------|------|
| `image.size` | Resize 目標尺寸 `[H, W]` |
| `image.normalize` | 是否正規化 |
| `augmentation` | 各項資料增強的開關與強度（CLAHE、翻轉、亮度對比等） |
| `split` | train / val / test 比例與隨機種子 |
| `output_dir` | 前處理結果輸出目錄 |

### configs/model/

模型架構與推論參數，由 `src/models/detector.py` 讀取。

| 欄位 | 說明 |
|------|------|
| `name` | 模型名稱（`yolov8n` / `yolov8s` / `yolov8m` 等） |
| `task` | 任務類型（`segment`：輪廓分割） |
| `weights` | 預訓練權重檔名（`yolov8n-seg.pt` 會自動下載） |
| `num_classes` | 類別數（目前為 `1`，僅有 `tooth`） |
| `inference.conf_threshold` | 推論時的信心分數門檻 |
| `inference.iou_threshold` | NMS 的 IoU 門檻 |
| `inference.img_size` | 推論輸入尺寸 |

### configs/experiment/

實驗組合設定，**一個檔案代表一次實驗**。由 `src/predict.py` 讀取，輸出目錄名稱直接取自此檔案的檔名。

| 欄位 | 說明 |
|------|------|
| `preprocess` | 引用的前處理設定檔（相對於 `configs/`） |
| `model` | 引用的模型設定檔（相對於 `configs/`） |
| `train.*` | 訓練超參數（epochs、batch、lr、optimizer、device 等） |

---

## data/

| 目錄 / 檔案 | 說明 |
|-------------|------|
| `train/` | 訓練集（images/ + labels/），YOLO segmentation 格式，唯讀 |
| `valid/` | 驗證集，同上 |
| `test/` | 測試集，同上 |
| `raw/` | 未處理的原始來源資料，放入後不再修改 |
| `processed/` | 經 `PreprocessPipeline` 輸出的處理後影像 |
| `splits/` | train.txt / val.txt / test.txt，每行一個影像路徑 |
| `dataset.yaml` | YOLO 訓練設定，指定 train / val / test 路徑與類別數 |

---

## src/

主要程式碼，以模組方式執行（`python -m src.predict`）。

### src/predict.py — 主流程入口

整個實驗的執行起點，負責：
1. 讀取 experiment config
2. 建立 `experiments/<實驗名稱>_<時間戳>/` 輸出目錄
3. 呼叫 `DentalDetector.train()` 執行訓練
4. 訓練完成後載入 `best.pt`，對 val 和 test 集執行評估
5. 呼叫 `save_scores_csv()` 產生 `scores.csv`

### src/evaluate.py — 評分計算

| 函式 | 說明 |
|------|------|
| `add_f1(metrics)` | 由 precision 與 recall 計算 F1，注入到 metrics dict |
| `save_scores_csv(output_dir, val, test)` | 寫出 `scores.csv` |
| `print_scores(val, test)` | 在終端機印出對齊的分數表格 |

評估指標：`mAP50`、`mAP50_95`、`precision`、`recall`、`f1`

### src/models/detector.py — 模型封裝

`DentalDetector` 封裝 ultralytics YOLO，對外提供三個方法：

| 方法 | 說明 |
|------|------|
| `train(data_yaml, train_cfg, output_dir)` | 執行訓練，回傳 `best.pt` 的路徑 |
| `validate(data_yaml, split, device)` | 對指定 split 執行評估，回傳 metrics dict |
| `predict(source)` | 對影像來源執行推論，回傳 ultralytics Results list |
| `load_weights(path)` | 載入指定權重檔 |

### src/preprocess/pipeline.py — 前處理 Pipeline

`PreprocessPipeline` 讀取 preprocess config，提供：

| 方法 | 說明 |
|------|------|
| `run(raw_dir)` | 對 raw_dir 內的影像執行 resize + augmentation，輸出至 `processed/`，並自動產生 train / val / test splits |

---

## experiments/

每次執行 `src/predict.py` 自動產生的結果目錄，命名規則：`<設定檔檔名>_<YYYYMMDD_HHMMSS>`。

```
experiments/baseline_20260501_120000/
├── experiment.yaml       # 此次實驗的完整設定快照（供重現）
├── scores.csv            # val 與 test 的最終評估分數
└── weights/              # ultralytics 訓練輸出
    ├── best.pt           # val fitness 最高的 epoch 權重
    ├── last.pt           # 最後一個 epoch 的權重
    ├── results.csv       # 每個 epoch 的 loss / mAP 數值
    ├── results.png       # 訓練曲線圖
    ├── confusion_matrix.png
    └── PR_curve.png
```

> `weights/` 內的 `.pt` 檔與視覺化圖表由 ultralytics 自動產生，不進版控（見 `.gitignore`）。

---

## baselines/

存放外部提供的 baseline 程式，僅供參考對照，不整合進主流程。

## notebooks/

Jupyter Notebook，用於 EDA（探索性資料分析）與錯誤分析。

## submissions/

上傳平台用的預測結果檔案，由推論流程自動產生。

---

## 設定檔與程式碼的對應關係

```
configs/experiment/exp002.yaml
        │
        ├─ preprocess: → configs/preprocess/default.yaml → src/preprocess/pipeline.py
        └─ model:      → configs/model/yolo_default.yaml → src/models/detector.py
                                                            ↑
                                               src/predict.py 統一呼叫
```

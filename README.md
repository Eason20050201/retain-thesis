# 全口牙齒 X 光影像分割專案

全景牙科 X 光影像（Panoramic Dental X-ray）的實例分割研究專案，基於 YOLOv8-seg 模型，針對牙齒輪廓進行偵測與分割。

## 環境需求

- macOS（Apple Silicon，支援 MPS 加速）
- Python 3.12

## 安裝

```bash
# 建立並啟動虛擬環境
python3.12 -m venv .venv
source .venv/bin/activate

# 安裝套件
pip install -r requirements.txt
```

之後每次開新的 terminal，都需要先啟動虛擬環境：

```bash
source .venv/bin/activate
```

---

## 資料格式

資料集使用 **YOLO Segmentation 格式**，標注檔每行為一個物件：

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

目前只有一個類別（`tooth`，class id = `0`），座標為正規化後的多邊形輪廓點。

資料集設定檔位於 [data/dataset.yaml](data/dataset.yaml)，指定 train / val / test 路徑與類別數。

---

## 執行方式

### 完整流程（訓練 + val 評估 + test 評估）

```bash
python -m src.predict --experiment configs/experiment/baseline.yaml
```

流程：
1. 訓練模型，每個 epoch 結束後對 val 集評估，儲存 fitness 最高的 epoch 為 `best.pt`
2. 用 `best.pt` 對 val 集重新評估，取得最終 val 分數
3. 用 `best.pt` 對 test 集評估，取得 test 分數
4. 輸出 `scores.csv`

### 只做評估（已有訓練好的權重）

```bash
python -m src.predict \
  --experiment configs/experiment/baseline.yaml \
  --weights experiments/baseline_20260501_120000/weights/best.pt
```

---

## 實驗設定

每個實驗對應 `configs/experiment/` 下的一個 YAML 檔。**輸出目錄名稱由設定檔的檔名決定**。

例如 `configs/experiment/exp002_yolov8s.yaml` → 輸出至 `experiments/exp002_yolov8s_YYYYMMDD_HHMMSS/`

設定檔結構：

```yaml
preprocess: preprocess/default.yaml   # 前處理設定
model:      model/yolo_default.yaml   # 模型設定

train:
  epochs:       100
  batch_size:   16
  learning_rate: 0.01
  optimizer:    SGD          # SGD / Adam / AdamW
  device:       mps          # cpu / cuda / mps
  patience:     20           # early stopping
```

模型設定位於 `configs/model/`，可切換模型大小（`yolov8n-seg` / `yolov8s-seg` / `yolov8m-seg`）與推論參數。

---

## 輸出結果

訓練完成後，結果儲存在 `experiments/<實驗名稱>_<時間戳>/`：

```
experiments/baseline_20260501_120000/
├── experiment.yaml       # 此次實驗的完整設定（供重現）
├── scores.csv            # val 與 test 評估分數
└── weights/
    ├── best.pt           # 最佳模型權重
    ├── last.pt           # 最後一個 epoch 的權重
    ├── results.csv       # 每個 epoch 的詳細訓練曲線數值
    ├── results.png       # 訓練曲線圖
    ├── confusion_matrix.png
    └── PR_curve.png
```

`scores.csv` 格式：

```
metric,val_best,test
mAP50,0.8700,0.8300
mAP50_95,0.6100,0.5800
precision,0.9200,0.8800
recall,0.8500,0.8100
f1,0.8836,0.8444
```

- **val_best**：`best.pt` 對 val 集的評估分數
- **test**：`best.pt` 對 test 集的評估分數
- **F1**：由 precision 與 recall 計算（`2PR / (P+R)`）

---

## 新增實驗

1. 複製現有設定檔並修改參數：

```bash
cp configs/experiment/baseline.yaml configs/experiment/exp002_yolov8s.yaml
```

2. 修改 `exp002_yolov8s.yaml` 內容（例如換模型、調整 batch size）

3. 執行：

```bash
python -m src.predict --experiment configs/experiment/exp002_yolov8s.yaml
```

---

## 主要套件版本

| 套件 | 版本 |
|------|------|
| Python | 3.12 |
| torch | 2.4.1 |
| torchvision | 0.19.1 |
| ultralytics | 8.2.87 |
| albumentations | 1.4.21 |
| numpy | 1.26.4 |
| opencv-python | 4.10.0.84 |

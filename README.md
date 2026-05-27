# retain-thesis

全景牙科 X 光影像（panoramic dental X-ray）的牙齒偵測實驗專案。現行主線使用 Ultralytics YOLOv12n，任務為 **detect**，單一類別 `tooth`。

> 注意：本專案目前不是 YOLO segmentation 流程。標註與評估皆以 bounding box detection 為主。

## 目前狀態

- 目標專案位置：`taiyu` 的 WSL Ubuntu 內 `/home/taiyu/retain-thesis`
- 主要入口：`python -m src.train`
- 模型設定：`configs/model/yolo12n.yaml`
- Pilot 實驗設定：`configs/experiment/*.yaml`
- 正式實驗設定：`configs/experiment/official/*.yaml`
- 正式 protocol：`configs/protocol/cv5_locked_test.yaml`
- 實驗紀錄：`experiments/EXPERIMENTS_LOG.md`
- 資料切分：train 81 張、valid 20 張、test 26 張
- 正式比較指標：5-fold CV validation mean `F1@IoU=0.5`
- Locked final test：目前 `data/test/images` 的 26 張，後續只做 final generalization，不再用來選方法
- Pilot F1 最佳候選：`exp012_yolo12n_adamw_80ep_t040_clahe`
- Overnight fold0 preprocessing screening：已完成 10 組 YOLOv12n screening，top 3 為 `ABC-CLAHE`、`affine augmentation`、`PS-KDE-inspired KDE-CDF`

Pilot F1 最佳候選的 recorded score：

| metric | val_best | test |
|---|---:|---:|
| mAP50 | 0.8630 | 0.6836 |
| mAP50_95 | 0.3321 | 0.2387 |
| precision | 0.7992 | 0.8230 |
| recall | 0.8065 | 0.6102 |
| f1 | 0.8028 | 0.7008 |

注意：`exp001` 到 `exp014` 都定位為 **pilot experiments**。這些結果用來形成研究假設，不作為最終論文主結果；正式方法選擇改用 `train+valid` 的 5-fold CV，locked test 不再用來調參、選 preprocessing、選 confidence threshold 或選模型。

## 可信實驗 Protocol

正式論文實驗採用 `cv5_locked_test`：

```bash
# 產生 deterministic split 與 list-based dataset YAML
python -m src.split --protocol configs/protocol/cv5_locked_test.yaml --write

# 檢查 official CV 實驗會跑哪些 fold，不訓練
python -m src.run_cv --experiment configs/experiment/official/yolo12n_raw_cv5.yaml --dry-run

# 跑單一 fold smoke / 正式實驗片段
python -m src.run_cv --experiment configs/experiment/official/yolo12n_raw_cv5.yaml --folds 0

# 聚合已完成的 fold 結果
python -m src.summarize_cv
```

正式規則：

- `data/test/images` 的 26 張是 locked final test，只能在方法固定後評一次。
- `data/train/images + data/valid/images` 的 101 張用於 5-fold CV。
- Primary metric 是 `F1@IoU=0.5`，TP 需 class match 且 IoU >= 0.5。
- confidence threshold 只能從 validation fold 選，不能用 test set 調。
- 第一輪 official methods：YOLOv12n raw / CLAHE / PS-KDE-inspired KDE-CDF、Faster R-CNN ResNet50-FPN raw、RetinaNet ResNet50-FPN raw。

## 遠端連線

`ssh taiyu` 目前會先進 Windows OpenSSH，再用 `wsl.exe` 進 Ubuntu。互動式進專案：

```bash
ssh -t taiyu 'wsl.exe --exec /bin/bash -lc "cd /home/taiyu/retain-thesis && exec bash -l"'
```

非互動式執行：

```bash
ssh taiyu 'wsl.exe --exec /bin/bash -lc "cd /home/taiyu/retain-thesis && git status --short --branch"'
```

若需要從本機自動化操作，請在本機另外放 helper script；不要把本機連線腳本放進此專案 repo。

## 環境

目前已在 taiyu WSL 驗證：

- Ubuntu 24.04 on WSL2
- Python 3.12.3
- CUDA GPU：NVIDIA GeForce RTX 5070 Ti 16GB
- Ultralytics 8.4.46
- torch 2.11.0+cu128

啟用既有虛擬環境：

```bash
source ./venv/bin/activate
```

若要重建環境：

```bash
python3.12 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## 資料格式

資料集每張影像在 `images/`，對應標註在 `labels/`。目前標註檔實際上是 YOLO polygon / segmentation-style label；訓練與正式 F1 evaluator 會把 polygon 轉成 enclosing bounding box 來做 tooth detection。

若標註是一般 YOLO bbox，格式為：

```text
<class_id> <x_center> <y_center> <width> <height>
```

若標註是 polygon，格式為：

```text
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

座標皆為 0 到 1 的正規化值。目前只有一個類別：

```yaml
nc: 1
names:
  - tooth
```

主要資料集設定：

- `data/dataset.yaml`：原始資料
- `data/dataset_gray.yaml`：灰階前處理
- `data/dataset_clahe.yaml`：CLAHE 前處理
- `data/dataset_gamma.yaml`：Gamma correction
- `data/dataset_kde.yaml`：PS-KDE 前處理，目前 test mAP50 最高
- `data/dataset_pipeline*.yaml`：多階段 pipeline 實驗

前處理資料夾（例如 `data_kde/`）由 `src/preprocess/*.py` 產生，標註檔會直接複製，bbox 不改。

## 專案結構

```text
retain-thesis/
├── configs/
│   ├── experiment/        # pilot 實驗
│   ├── experiment/official/ # official CV 實驗
│   ├── model/             # YOLO 模型與 inference 設定
│   └── protocol/          # locked-test / CV protocol
├── data/                  # 原始 YOLO dataset，未進 git
├── data/splits/           # official split files，未進 git
├── data_kde/              # PS-KDE 前處理資料，未進 git
├── experiments/
│   ├── EXPERIMENTS_LOG.md # 手動整理的實驗結論
│   ├── exp*/              # pilot 實驗 artifacts
│   ├── official/          # official CV artifacts
│   └── overnight/         # fold0 preprocessing screening artifacts
├── src/
│   ├── train.py           # 訓練與評估主入口
│   ├── split.py           # official split / dataset adapter
│   ├── run_cv.py          # official 5-fold CV runner
│   ├── eval_f1.py         # F1@IoU evaluator
│   ├── summarize_cv.py    # CV 結果聚合
│   ├── tune.py            # Optuna 超參數搜尋
│   ├── evaluate.py        # scores.csv 與 F1 計算
│   ├── models/detector.py # Ultralytics YOLO 封裝
│   └── preprocess/        # 前處理腳本
└── requirements.txt
```

## Pilot 實驗如何製作

舊式 pilot 實驗由三個部分組成：資料集 YAML、模型 YAML、實驗 YAML。投稿用正式結果請優先使用上一節的 `cv5_locked_test` protocol。

1. 選擇或產生資料集

   原始資料使用 `data/dataset.yaml`。如果要做前處理，先跑對應腳本：

   ```bash
   python -m src.preprocess.kde
   ```

   然後使用對應的 `data/dataset_kde.yaml`。

2. 檢查模型設定

   目前主線是 `configs/model/yolo12n.yaml`：

   ```yaml
   name: yolo12n
   task: detect
   weights: yolo12n.pt
   num_classes: 1
   ```

3. 新增實驗設定

   在 `configs/experiment/` 新增一個 YAML。檔名會決定輸出資料夾名稱。

   ```yaml
   description: "YOLO12n, AdamW, imgsz=1024, PS-KDE"

   model: model/yolo12n.yaml
   dataset: dataset_kde.yaml

   train:
     epochs:        80
     batch_size:    16
     learning_rate: 0.000676
     optimizer:     AdamW
     patience:      25
     img_size:      1024
     device:        "0"
     seed:          0
     workers:       2
   ```

   若省略 `dataset`，預設使用 `data/dataset.yaml`。

4. 執行完整訓練

   ```bash
   python -m src.train --experiment configs/experiment/exp014_yolo12n_adamw_80ep_t040_kde.yaml
   ```

   `src.train` 會：

   - 讀取 experiment YAML
   - 讀取 `configs/model/...`
   - 用 Ultralytics YOLO 訓練
   - 儲存 best.pt 到 `experiments/<experiment_name>/train/weights/best.pt`
   - 重新載入 best.pt
   - 對 valid 與 test 評估
   - 寫出 `experiments/<experiment_name>/scores.csv`

   注意：`project=experiments/<experiment_name>` 且 `name=train`、`exist_ok=True`，所以同名實驗重跑會覆蓋或延續同一個 `train/` 目錄。若要保留舊結果，先複製新 config 名稱。

5. 只做評估

   已有權重時用 `--weights`，不重訓，只跑 valid/test 並重寫 `scores.csv`：

   ```bash
   python -m src.train \
     --experiment configs/experiment/exp014_yolo12n_adamw_80ep_t040_kde.yaml \
     --weights experiments/exp014_yolo12n_adamw_80ep_t040_kde/train/weights/best.pt
   ```

6. 記錄結果

   每次正式實驗後，更新：

   - `experiments/<experiment_name>/scores.csv`
   - `experiments/EXPERIMENTS_LOG.md`

   Ultralytics 的 validation 圖表會另外輸出到 `runs/detect/val-*`。

## 超參數搜尋

Optuna 搜尋入口：

```bash
python -m src.tune
```

目前 `src/tune.py` 的搜尋空間：

- learning rate：`1e-4` 到 `1e-2`，log scale
- batch size：`8` 或 `16`
- optimizer：`AdamW` 或 `SGD`
- image size：`640` 或 `1024`
- 每個 trial：80 epochs、patience 25、seed 0、workers 2

結果輸出：

- `experiments/tune_v2/study.db`
- `experiments/tune_v2/results.csv`
- 每個 trial 的訓練結果在 `experiments/tune_v2/trial_*/`

目前觀察：top trials 幾乎都落在 `AdamW + batch=16 + imgsz=1024 + lr 約 0.0002 到 0.0011`。

## 目前實驗結論

詳見 `experiments/EXPERIMENTS_LOG.md`。摘要：

主要排名以 test F1 score 為準：

| rank | experiment | preprocessing | test F1 | test mAP50 | note |
|---:|---|---|---:|---:|---|
| 1 | `exp012` | CLAHE | 0.7008 | 0.6836 | 目前 F1 最高 |
| 2 | `exp014` | PS-KDE | 0.6932 | 0.7334 | test mAP50 最高 |
| 3 | `exp006` | 無前處理 / trial_040 best.pt | 0.6899 | 0.6620 | 無前處理強 baseline |
| 4 | `exp008` | Grayscale | 0.6885 | 0.7265 | 與 exp006 F1 很接近 |

PS-KDE 的 mAP50 最高，但若 final selection 以 F1 為主，CLAHE 的 `exp012` 目前排第一。

目前最值得繼續的方向：

- 以 `exp012` 當 F1-first final candidate
- 對 `exp012` / `exp014` / `exp006` / `exp008` 跑多個 seed，取平均與標準差
- 若仍想追 mAP50，可在 `data_kde` 上重新做 tune

## 常用指令

```bash
# 進入遠端專案
ssh -t taiyu 'wsl.exe --exec /bin/bash -lc "cd /home/taiyu/retain-thesis && exec bash -l"'

# 查看遠端 git 狀態
ssh taiyu 'wsl.exe --exec /bin/bash -lc "cd /home/taiyu/retain-thesis && git status --short --branch"'

# 重跑目前 F1 最佳候選的 evaluation-only
ssh taiyu 'wsl.exe --exec /bin/bash -lc "cd /home/taiyu/retain-thesis && ./venv/bin/python -m src.train --experiment configs/experiment/exp012_yolo12n_adamw_80ep_t040_clahe.yaml --weights experiments/exp012_yolo12n_adamw_80ep_t040_clahe/train/weights/best.pt"'

# 在遠端專案中新增 PS-KDE 資料
ssh taiyu 'wsl.exe --exec /bin/bash -lc "cd /home/taiyu/retain-thesis && ./venv/bin/python -m src.preprocess.kde"'
```

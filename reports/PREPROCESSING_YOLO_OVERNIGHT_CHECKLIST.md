# YOLO 前處理 Overnight Checklist

最後整理日期：2026-05-25

這份 checklist 用來執行約 8 小時的 YOLOv12n 前處理初篩。所有命令都從本機執行，透過專案 helper 連到遠端 WSL：

```bash
cd "/Users/wangyixin/Documents/Undergraduate Project"
./retain-thesis-remote.sh run <remote-command>
```

## 0. 原則確認

- [ ] 今晚只做 screening，不產生 final paper result。
- [ ] 今晚只跑 YOLOv12n，不加入 Faster R-CNN / RetinaNet / DETR。
- [ ] 只使用 official `cv5_locked_test` 的 `fold0` validation F1 排名。
- [ ] 不讀取 `data/test/images`。
- [ ] 不讀取或評估 `data/splits/locked_test.txt`。
- [ ] 不根據 locked test 修改 preprocessing、threshold、epochs、augmentation。
- [ ] 隔天只把 top 3 推進正式 5-fold CV。

## 1. 開跑前檢查

### 1.1 遠端狀態

```bash
./retain-thesis-remote.sh status
```

- [ ] 確認目前 branch。
- [ ] 確認本輪新增文件與 config 的變動可以追蹤。
- [ ] 確認沒有正在跑的舊訓練佔用 GPU。

### 1.2 split 與資料數量

```bash
./retain-thesis-remote.sh run bash -lc 'wc -l data/splits/locked_test.txt; for f in data/splits/cv5_seed20260525/raw/fold*_val.txt; do echo "$f $(wc -l < "$f")"; done'
```

- [ ] `locked_test.txt` 是 26 張。
- [ ] fold0 validation 是 21 張。
- [ ] fold1-fold4 validation 各 20 張。
- [ ] 今晚 runner 只會使用 fold0 train/val。

```bash
./retain-thesis-remote.sh run bash -lc 'for d in data data_clahe data_kde data_gray data_gamma data_clahe_abc data_pipeline data_pipeline2; do printf "%s images=" "$d"; find "$d" -path "*/images/*" -type f | wc -l; printf "%s labels=" "$d"; find "$d" -path "*/labels/*" -type f | wc -l; done'
```

- [ ] 每個要跑的 dataset variant 都是 127 images。
- [ ] 每個要跑的 dataset variant 都是 127 labels。
- [ ] 若有任一 variant 缺檔，該方法跳過並記錄，不自動改用別的資料。

### 1.3 evaluator 快速測試

```bash
./retain-thesis-remote.sh run ./venv/bin/python -m unittest tests.test_eval_f1 tests.test_split
```

- [ ] evaluator tests 通過。
- [ ] split tests 通過。
- [ ] 若失敗，停止 overnight，先修 evaluator / split。

## 2. Smoke run

先做一個 1 epoch smoke run，確認輸出目錄、fold0、evaluation pipeline 都正常。

```bash
./retain-thesis-remote.sh run ./venv/bin/python -m src.run_cv --experiment configs/experiment/overnight/smoke_raw_1ep.yaml --folds 0
```

- [ ] 可以完成 training。
- [ ] 可以完成 prediction。
- [ ] 可以完成 F1 evaluation。
- [ ] 輸出包含 `scores.json`。
- [ ] 輸出包含 `scores.csv`。
- [ ] 輸出包含 `matches.csv`。
- [ ] 輸出包含 `predictions.csv`。
- [ ] 沒有讀取 locked test。

若 smoke run 失敗，今晚不要直接開長訓練。

## 3. Priority A queue

建議順序如下。若 8 小時內跑不完，優先保留前面項目。

| done | order | experiment | config | priority |
| --- | ---: | --- | --- | --- |
| [ ] | 1 | raw baseline | `configs/experiment/overnight/yolo12n_raw_fold0.yaml` | 必跑 |
| [ ] | 2 | CLAHE fixed | `configs/experiment/overnight/yolo12n_clahe_fixed_fold0.yaml` | 必跑 |
| [ ] | 3 | KDE-CDF | `configs/experiment/overnight/yolo12n_kde_cdf_fold0.yaml` | 必跑 |
| [ ] | 4 | grayscale | `configs/experiment/overnight/yolo12n_gray_fold0.yaml` | 必跑 |
| [ ] | 5 | gamma 0.5 | `configs/experiment/overnight/yolo12n_gamma05_fold0.yaml` | 必跑 |
| [ ] | 6 | ABC-CLAHE | `configs/experiment/overnight/yolo12n_clahe_abc_fold0.yaml` | 必跑 |
| [ ] | 7 | photometric augmentation | `configs/experiment/overnight/yolo12n_photo_aug_fold0.yaml` | 必跑 |
| [ ] | 8 | affine augmentation | `configs/experiment/overnight/yolo12n_affine_aug_fold0.yaml` | 必跑 |
| [ ] | 9 | cutout augmentation | `configs/experiment/overnight/yolo12n_cutout_aug_fold0.yaml` | 若時間足夠 |
| [ ] | 10 | pipeline2 | `configs/experiment/overnight/yolo12n_pipeline2_fold0.yaml` | 若時間足夠 |

單組執行命令範例：

```bash
./retain-thesis-remote.sh run ./venv/bin/python -m src.run_cv --experiment configs/experiment/overnight/yolo12n_clahe_fixed_fold0.yaml --folds 0
```

每跑完一組就記錄：

- [ ] experiment id：
- [ ] start time：
- [ ] end time：
- [ ] completed / failed：
- [ ] best epoch：
- [ ] selected confidence threshold：
- [ ] precision：
- [ ] recall：
- [ ] F1：
- [ ] AP50：
- [ ] AP50-95：
- [ ] notes：

## 4. Priority B queue

只有在 Priority A 提早完成時才跑。

| done | experiment | config | 目的 |
| --- | --- | --- | --- |
| [ ] | CLAHE lite | `configs/experiment/overnight/yolo12n_clahe_lite_fold0.yaml` | 測較弱 contrast enhancement。 |
| [ ] | CLAHE strong | `configs/experiment/overnight/yolo12n_clahe_strong_fold0.yaml` | 測較強 contrast enhancement。 |
| [ ] | CLAHE tile16 | `configs/experiment/overnight/yolo12n_clahe_tile16_fold0.yaml` | 測較大 tile normalization。 |
| [ ] | gamma 0.7 | `configs/experiment/overnight/yolo12n_gamma07_fold0.yaml` | 測較溫和暗部提亮。 |
| [ ] | KDE mild | `configs/experiment/overnight/yolo12n_kde_mild_fold0.yaml` | 測較平滑 KDE-CDF。 |
| [ ] | KDE strong | `configs/experiment/overnight/yolo12n_kde_strong_fold0.yaml` | 測較強 KDE-CDF。 |

## 5. 中途監控

每 1-2 小時檢查一次：

```bash
./retain-thesis-remote.sh run bash -lc 'nvidia-smi; find experiments/overnight -maxdepth 4 -name scores.json | sort'
```

- [ ] GPU 正常使用。
- [ ] 沒有卡在資料讀取錯誤。
- [ ] 沒有因 out-of-memory 中斷。
- [ ] `scores.json` 數量逐步增加。
- [ ] 若某組失敗，記錄原因後跑下一組，不臨時修改 locked test 或切分。

## 6. 隔天彙整

```bash
./retain-thesis-remote.sh run ./venv/bin/python -m src.summarize_cv --root experiments/overnight/yolo_preprocessing_YYYYMMDD --report reports/OVERNIGHT_YOLO_PREPROCESSING_RESULTS.md --csv experiments/overnight/yolo_preprocessing_YYYYMMDD/summary.csv
```

- [ ] 報告標題明確寫 `fold0 screening only`。
- [ ] 報告明確寫沒有使用 locked test。
- [ ] 表格包含 F1、precision、recall、AP50、AP50-95、selected threshold。
- [ ] 依 F1 排名。
- [ ] 標出 top 3。
- [ ] 標出 failed / skipped experiments。

## 7. Top 3 決策

每個方法照下面規則分類：

| decision | 條件 | 後續 |
| --- | --- | --- |
| promote | F1 高於 raw，且 AP50/precision/recall 沒有明顯崩 | 進正式 5-fold CV。 |
| hold | 分數接近 raw，或只改善單一 secondary metric | 暫存，必要時補 fold1。 |
| drop | F1 低於 raw，或 FP/FN 明顯變差 | 不進正式 CV。 |

Top 3 寫入：

- [ ] `reports/OVERNIGHT_YOLO_PREPROCESSING_RESULTS.md`
- [ ] `reports/CV5_RESULTS.md` 的待跑清單
- [ ] README 的 current experiment status

## 8. 正式 CV 交接

promote 的方法下一步才可以進：

```bash
./retain-thesis-remote.sh run ./venv/bin/python -m src.run_cv --experiment <official-cv5-config> --folds 0,1,2,3,4
```

- [ ] 正式 CV 用 5 folds。
- [ ] 正式 CV 仍然不碰 locked test。
- [ ] 正式 CV 完成後才選 final method。
- [ ] final method 決定後，才用 train+valid 重訓並評 locked test 一次。

## 9. 今晚停止條件

遇到以下任一狀況，停止 overnight，不硬跑：

- [ ] evaluator tests 失敗。
- [ ] split tests 失敗。
- [ ] smoke run 無法產生 `scores.json`。
- [ ] runner 需要讀 `data/test` 才能完成。
- [ ] 任一 config 把 locked test 當 validation。
- [ ] GPU 持續 OOM，且無法在不改研究設定的情況下降 batch。
- [ ] 發現 fold0 list 和 labels basename 對不上。

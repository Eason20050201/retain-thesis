# YOLO 前處理 Overnight 實驗設計

最後整理日期：2026-05-25

這份文件整理目前 `retain-thesis` 下一輪要做的 YOLO 前處理實驗。目標不是馬上產生正式論文表格，而是用一個晚上約 8 小時，先把「哪些前處理值得進正式 5-fold CV」篩出來。

本輪結果只能稱為 screening / pilot。正式論文結果仍必須回到 `cv5_locked_test` protocol：train+valid 的 5-fold CV 做方法選擇，`data/test` 的 26 張 locked test 只在最後評估一次。

## 核心定位

目前專案最有機會形成 B 級會議投稿敘事的主軸是：

> 在 limited-data panoramic dental X-ray tooth detection 中，比較不同 radiograph preprocessing 對 YOLOv12n 的 F1 影響，並用 locked-test + 5-fold CV 防止 test leakage。

所以 overnight 不應該被設計成「亂跑很多模型」，而是應該固定主模型，只改前處理與訓練增強：

- 主模型：YOLOv12n。
- 初篩資料：official split 的 `fold0`。
- 初篩指標：`F1@IoU=0.5`。
- confidence threshold：只能由 validation fold 自動選，不能用 locked test。
- locked test：完全不碰。
- 隔天決策：依 fold0 validation F1 挑 top 3，進正式 5-fold CV。

## 文獻前處理線索

| 來源 | 文獻 / 任務 | 前處理或增強做法 | 對本專案的意義 |
| --- | --- | --- | --- |
| PS-KDE paper | Chest X-ray semantic segmentation | adaptive histogram equalization / CLAHE、KDE-based pixel substitution、CDF remapping | 直接支撐 `CLAHE` 與 `PS-KDE-inspired KDE-CDF` 作為主要研究變因。雖然不是牙科，但同屬 radiograph contrast enhancement。 |
| Yuksel et al. / DENTECT | Panoramic X-ray tooth enumeration and treatment detection | photometric distortion、random flipping、brightness / contrast / saturation transformation、quadrant-level processing | 支撐小資料 panoramic X-ray 下需要 photometric augmentation；quadrant crop 可列第二輪，因為需要 bbox remapping。 |
| DENTEX sequential framework | Abnormal teeth detection/classification | horizontal flipping、brightness / contrast、affine transformations、random cutouts | 支撐今晚加入 photo augmentation、affine augmentation、cutout augmentation。 |
| Tuzoff et al. 2019 | Tooth detection and numbering | tooth crop with neighboring context、anchor / NMS / score threshold tuning、augmentation | 支撐後續做 context-aware crop 或 threshold analysis；但 crop 需要新資料管線，今晚先不放主線。 |
| DENTEX benchmark / Diffusion hierarchical detection | Panoramic dental X-ray object detection | random crop / resize、multi-detector baselines、AP-family metrics | 支撐我們同時報 F1、AP50、AP50-95，也提醒 preprocessing comparison 必須使用固定 split。 |

## 目前專案已有的資料版本

已在遠端檢查過，以下資料版本都有 127 張 images 與 127 個 labels：

| variant | 資料夾 | 來源 | 今晚定位 |
| --- | --- | --- | --- |
| `raw` | `data` | 原始 Roboflow / YOLO dataset | 必跑 baseline。 |
| `clahe_fixed` | `data_clahe` | `src/preprocess/clahe.py`，clipLimit=2.0, tileGrid=8x8 | 必跑，pilot F1 目前最佳。 |
| `kde_cdf` | `data_kde` | `src/preprocess/kde.py`，KDE-smoothed histogram + CDF LUT | 必跑，對應 PS-KDE-inspired 方法。 |
| `gray` | `data_gray` | `src/preprocess/grayscale.py` | 必跑，檢查移除無意義 RGB 色彩是否穩定。 |
| `gamma_0.5` | `data_gamma` | `src/preprocess/gamma.py`，gamma=0.5 | 必跑，檢查 X-ray 暗部提亮是否改善低對比牙齒。 |
| `clahe_abc` | `data_clahe_abc` | `src/preprocess/clahe_abc.py`，entropy 搜尋 clipLimit | 必跑，檢查自適應 CLAHE 是否優於固定 CLAHE。 |
| `pipeline` | `data_pipeline` | 舊 dental pipeline | 後跑，僅作為 aggressive preprocessing 對照。 |
| `pipeline2` | `data_pipeline2` | `src/preprocess/dental_pipeline.py` | 後跑，方法複雜，若表現差也能當作 negative ablation。 |

注意：official protocol 目前只註冊 `raw`、`clahe`、`kde`。若要把 `gray/gamma/clahe_abc/pipeline/pipeline2` 納入 official runner，需要先在 protocol 或 overnight config 補 dataset variant。這是工程準備，不是結果調參。

## Overnight 實驗規則

### 固定項目

| 項目 | 設定 |
| --- | --- |
| model | YOLOv12n |
| split | `cv5_locked_test` 的 `fold0` |
| seed | `20260525`，若現有 train entrypoint 只接受 seed `0`，需要在 config snapshot 中明確記錄 |
| epochs | 50 |
| patience | 15 |
| batch | 16 |
| image size | 1024 |
| optimizer | AdamW |
| lr0 | 0.000676 |
| primary metric | F1@IoU=0.5 |
| secondary metrics | precision, recall, AP50, AP50-95, selected confidence threshold |
| output root | `experiments/overnight/yolo_preprocessing_YYYYMMDD/` |

### 禁止項目

- 不讀取 `data/test/images`。
- 不讀取 `data/splits/locked_test.txt`。
- 不用 locked test 選 preprocessing。
- 不用 locked test 選 confidence threshold。
- 不在今晚改模型架構。
- 不把 fold0 結果寫成 final paper result。

## Priority A：今晚主跑清單

| 順序 | experiment id | 類型 | 方法 | 目的 | 成功訊號 | 風險 |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | `smoke_raw_1ep` | sanity | raw, 1 epoch | 確認 runner、fold0、evaluator、輸出路徑正常 | 可產生 scores/matches/predictions | 若失敗，不能開 overnight |
| 1 | `yolo12n_raw_fold0` | baseline | 原始影像 | 建立所有前處理的比較底線 | F1 接近或高於舊 exp006 的方向 | 單 fold 波動大 |
| 2 | `yolo12n_clahe_fixed_fold0` | contrast | CLAHE clipLimit=2.0 | 驗證 pilot 最佳方法是否在 fold0 仍強 | F1 或 recall 高於 raw | 可能放大雜訊造成 FP |
| 3 | `yolo12n_kde_cdf_fold0` | contrast | KDE-CDF 灰階重映射 | 驗證 PS-KDE-inspired 方法是否穩定 | AP50 或 F1 高於 raw | 可能使局部邊界過度平滑 |
| 4 | `yolo12n_gray_fold0` | color ablation | grayscale 3-channel | 檢查 YOLO 是否需要原始 RGB | 接近 raw 或優於 raw | 若低於 raw，代表原影像色彩/壓縮資訊仍有用 |
| 5 | `yolo12n_gamma05_fold0` | intensity | gamma=0.5 | 強化暗部，測低對比區域 | recall 提升 | FP 可能增加 |
| 6 | `yolo12n_clahe_abc_fold0` | adaptive contrast | entropy-based adaptive CLAHE | 比固定 CLAHE 更個別化 | F1 優於 fixed CLAHE | 每圖最佳 entropy 不一定等於 detection 最佳 |
| 7 | `yolo12n_photo_aug_fold0` | train augmentation | brightness / contrast / saturation | 對應 DENTECT / SEQ 的 photometric distortion | validation F1 更穩，threshold 不極端 | augmentation 太強會破壞 X-ray intensity |
| 8 | `yolo12n_affine_aug_fold0` | train augmentation | small rotation / scale / translation | 對應 SEQ affine transformations | recall 提升，FN 減少 | 牙齒位置幾何被過度扭曲 |
| 9 | `yolo12n_cutout_aug_fold0` | train augmentation | random cutout / erasing | 對應 SEQ random cutouts，模擬遮擋 | overlap / partial tooth FN 減少 | 小資料下可能讓訓練變難 |
| 10 | `yolo12n_pipeline2_fold0` | aggressive preprocessing | morphology + threshold + equalize + unsharp | 檢查複雜 dental pipeline 是否真的有用 | 若低於 CLAHE/KDE，可作為 negative ablation | 容易過度處理 |

Priority A 的核心是「先把最有論文意義的方向跑完」。如果 8 小時跑不完，保留前 1-8，`cutout` 和 `pipeline2` 可以隔天補。

## Priority B：如果還有時間

| experiment id | 方法 | 為什麼暫列 B |
| --- | --- | --- |
| `yolo12n_clahe_lite_fold0` | CLAHE clipLimit=1.0, tileGrid=8x8 | 需要新增資料版本；可測較弱對比增強是否更少 FP。 |
| `yolo12n_clahe_strong_fold0` | CLAHE clipLimit=3.0, tileGrid=8x8 | 需要新增資料版本；可測強化是否犧牲 precision。 |
| `yolo12n_clahe_tile16_fold0` | CLAHE clipLimit=2.0, tileGrid=16x16 | 需要新增資料版本；測較大區塊 contrast normalization。 |
| `yolo12n_gamma07_fold0` | gamma=0.7 | 需要新增資料版本；比 gamma=0.5 溫和。 |
| `yolo12n_kde_mild_fold0` | KDE bandwidth multiplier > 1 | 需要新增資料版本；減少 CDF 映射強度。 |
| `yolo12n_kde_strong_fold0` | KDE bandwidth multiplier < 1 | 需要新增資料版本；增加灰階分布拉伸。 |

Priority B 不能擠掉 Priority A。它們比較像參數細化，不是主要文獻敘事。

## 第二輪才做的高成本方法

| 方法 | 來源 | 延後原因 | 後續價值 |
| --- | --- | --- | --- |
| Quadrant crop / tiling | Yuksel / DENTECT | 需要把 bounding boxes 轉到 tile 座標，推論時還要合併回全景圖 | 可能大幅降低單張圖牙齒數量，讓 detector 更容易學。 |
| Context crop | Tuzoff | 需要以 tooth candidates 或 GT 產生 crop，不適合在 detection 第一階段直接做 | 有機會改善相鄰牙齒 overlap 和 numbering 類任務。 |
| WBF / detector ensemble | DENTEX challenge methods | 需要多模型或多尺度推論，今晚主線是 preprocessing | 若 top methods 分別擅長 precision/recall，可後續測。 |
| Segmentation-to-box | DENTEX top solutions | 需要 segmentation labels 或 pseudo masks | 對目前只有 bbox 的資料不划算。 |
| DETR / RT-DETR / DiffusionDet | MICCAI high-bar papers | 工程成本高，8 小時不適合 | 等 YOLO preprocessing CV 穩定後再決定。 |

## 結果判讀規則

隔天不只看最高 F1，還要看它是不是可被寫成論文：

1. Primary ranking：`fold0 validation F1@IoU=0.5`。
2. Tie-break 1：AP50 較高者優先。
3. Tie-break 2：recall 較高但 precision 不崩者優先。
4. Tie-break 3：selected confidence threshold 不應極端低；若 threshold 很低才贏，可能代表 FP 風險高。
5. Tie-break 4：error cases 是否合理；例如 CLAHE 若多出 jaw-edge FP，要記錄。

建議隔天把方法分成三類：

| 分類 | 標準 | 下一步 |
| --- | --- | --- |
| promote | F1 明顯高於 raw，且 AP50/precision/recall 沒有崩 | 進正式 5-fold CV。 |
| hold | 接近 raw 或只改善單一 secondary metric | 保留，但不優先。 |
| drop | F1 低於 raw，或 FP/FN 型態明顯變差 | 不進正式 CV；可作 negative ablation。 |

## 預期正式 CV 候選

若 overnight 結果符合 pilot 方向，最可能進正式 5-fold CV 的 shortlist 是：

1. `YOLOv12n + raw`
2. `YOLOv12n + CLAHE fixed`
3. `YOLOv12n + KDE-CDF`
4. `YOLOv12n + best augmentation variant`
5. `YOLOv12n + best adaptive/intensity variant`

正式論文不需要把今晚所有方法都放入主表。主表應該乾淨，附錄或 pilot 表再列完整篩選。

## 建議輸出檔案

每個 overnight experiment 至少要有：

- `config_snapshot.yaml`
- `scores.json`
- `scores.csv`
- `matches.csv`
- `predictions.csv`
- `error_cases.csv`
- `train.log`
- `weights/best.pt`

彙整報告：

- `reports/OVERNIGHT_YOLO_PREPROCESSING_RESULTS.md`

該報告必須在標題或摘要中寫清楚：

> These are fold0 screening results only. They are not final paper results and were not evaluated on the locked test set.


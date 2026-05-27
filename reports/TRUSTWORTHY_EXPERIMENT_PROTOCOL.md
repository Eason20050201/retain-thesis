# 可信實驗流程規劃

最後整理日期：2026-05-25

這份文件的目的，是把 `retain-thesis` 從目前的 exploratory / pilot experiments，整理成比較接近 B 級國際會議論文會接受的實驗流程。重點不是馬上追最高分，而是先讓實驗設計、比較方式、評估指標、資料切分都能被信任。

## 參考論文的實驗流程

### DENTEX benchmark / challenge

參考 PDF：`reference/07_dentex_abnormal_tooth_detection_enumeration_diagnosis_benchmark.pdf`

DENTEX 的流程最值得參考，因為它直接是 panoramic X-ray abnormal tooth detection / enumeration / diagnosis benchmark。

它的可信點：

- 明確定義任務輸出：每個 abnormal tooth 要輸出 bounding box，以及 quadrant、enumeration、diagnosis 三種 label。
- 明確切資料：1005 張 fully annotated images 分成 training 705、validation 50、testing 250。
- validation 用於模型開發與 leaderboard；test 是 hidden test，由 organizer 執行 submitted Docker container。
- test set 不給參賽者看，避免用 test set 調模型。
- 評估採用 object detection 標準指標：AR、AP、AP50、AP75，並用不同 task 的多個 metric 做總排名。
- 排名使用 Wilcoxon signed-rank test 比較方法之間的顯著差異。
- Baseline 不只一個模型，而是同時比較 challenge submissions、HierarchicalDet、DiffusionDet、DETR、Faster R-CNN、RetinaNet。
- 會列出 qualitative examples，包含成功案例與困難案例。

對我們的啟示：

- 我們目前沒有 hidden test，但至少要建立 locked final test。
- 現在不應再用 test F1 選方法；test 只能留到最後一次報 final generalization。
- 必須把 old experiments 定位成 pilot，正式結果要從新的 protocol 開始。
- 如果要投 B 級，至少要有一組 non-YOLO baseline，例如 Faster R-CNN 或 RetinaNet，否則比較容易被看成只是在調 YOLO。

### Diffusion-Based Hierarchical Multi-Label Object Detection

參考 PDF：`reference/06_miccai2023_diffusion_hierarchical_multilabel_detection_panoramic_dental_xrays.pdf`

這篇是高標版本，但它的實驗結構很清楚。

它的可信點：

- 資料分層很明確：quadrant-only、quadrant-enumeration、quadrant-enumeration-diagnosis。
- 只用 fully labeled quadrant-enumeration-diagnosis data 做 final testing。
- split 明確列出：quadrant 590/103、quadrant-enumeration 539/95、quadrant-enumeration-diagnosis 705/50/250。
- 和 SOTA object detectors 比較：RetinaNet、Faster R-CNN、DETR、DiffusionDet。
- 有 ablation study：移除 transfer、移除 box manipulation、移除 manipulation + transfer。
- 明確列訓練設定：backbone、iterations / epochs、learning rate、batch size、GPU。
- 評估使用 AR、AP、AP50、AP75、APm、APl。

對我們的啟示：

- 我們可以保留 YOLOv12n 作為主方法，但要設計公平 baseline，而不是只比較 preprocessing。
- 如果我們主張 preprocessing 有效果，必須有 ablation：raw、CLAHE、PS-KDE-inspired KDE-CDF、gray / gamma，而不是只挑最高結果。
- 每個實驗要固定資料切分、固定 training budget、固定 metric script。

### A Sequential Framework for Detection and Classification of Abnormal Teeth

參考 PDF：`reference/08_miccai2023_dentex_sequential_framework_detection_classification_abnormal_teeth.pdf`

這篇是 DENTEX challenge solution，規模較短，但有清楚的 pipeline。

它的可信點：

- 明確分成三階段：teeth detection / numbering、healthy filtering、abnormal classification。
- detection 用 Faster R-CNN。
- classification 報 F1 score；detection 報 AP、AP50、AP75。
- 使用 DENTEX subset：634 張 tooth-number labels、705 張 abnormal-tooth labels。
- 訓練時使用 on-the-fly augmentation：horizontal flipping、brightness / contrast、affine transformations、random cutouts。

對我們的啟示：

- F1 score 在相似任務中可以作為 classification / decision metric，但 object detection 本身仍要報 AP / AP50。
- 如果我們以 F1 為主，需要清楚定義 F1 的計算方式，不能只說「看 F1」。
- 可把 Faster R-CNN 當作 non-YOLO baseline。

### Tuzoff et al. 2019

參考 PDF：`reference/12_tuzoff2019_tooth_detection_numbering_panoramic_radiographs_cnn.pdf`

這篇和我們最接近的是 tooth detection / numbering。

它的可信點：

- 資料來源明確：1574 張 anonymized panoramic radiographs。
- 明確分成 training 1352、testing 222。
- test set 沒有在 training 中出現。
- 五位 radiology experts 提供 ground truth annotations。
- 直接和 expert performance 比較。
- detection metric 使用 sensitivity 與 precision。
- numbering metric 使用 sensitivity 與 specificity。
- 有錯誤分析：false negatives 常見於 root remnants、orthopaedic appliances、impacted / overlapped teeth；false positives 可能來自 implants、bridges、multi-rooted teeth 等。
- 討論限制與後續：更大的資料集、更先進 augmentation、更近期 CNN architectures、延伸到 pathology detection。

對我們的啟示：

- 我們雖然沒有 expert comparison，但至少要有 error taxonomy。
- 最終論文不能只放表格，要列出 FP / FN 的常見原因。
- 對 tooth detection 來說，precision、recall、F1 比單純 mAP 更容易對應臨床可讀性。

### Yuksel et al. 2021 / DENTECT

參考 PDF：`reference/13_yuksel2021_dental_enumeration_treatment_detection_panoramic_xrays.pdf`

這篇是 panoramic X-ray 上的 enumeration / treatment detection。

它的可信點：

- 標註流程清楚：intern dental students label，professional endodontist validate。
- 1005 張影像；85% training、15% testing。
- 明確說明不同任務使用不同標註數量：500 張 tooth regions、1005 張 treatment labels、600 張 enumeration labels。
- Pipeline 分段：quadrant segmentation、enumeration detection、treatment detection。
- 對小資料集採用 data augmentation：photometric distortion、random flipping、brightness / contrast / saturation transformation。
- 對 class imbalance 使用 class weights、rare-class augmentation、negative sampling。
- 評估分任務：segmentation 用 IoU / dice / pixel accuracy；detection 用 AP / AR at IoU ranges。
- 明確討論小資料集限制：醫學影像資料少、每個 data point 影響很大、影像品質與雜訊會影響模型。

對我們的啟示：

- 我們現在資料量比這些論文更小，所以更需要 cross-validation 或 repeated seeds。
- Preprocessing 不是小技巧，而是可以被包成正式變因；但必須固定流程並避免用 test set 選。
- 小資料集限制要正面寫，不要假裝結果已經非常穩。

## 目前專案的實驗可信度問題

目前 `retain-thesis` 的舊實驗可以保留，但應該定位成 pilot experiments，不應直接當最終投稿主結果。

主要問題：

| 問題 | 現況 | 風險 |
| --- | --- | --- |
| 用 test F1 做排名 | README 與實驗 log 目前以 test F1 / test mAP50 排方法 | 容易被視為用 test set 選模型，造成 test leakage |
| validation 太小 | valid 只有 20 張 | 1-2 張影像變動就會造成排名大幅波動 |
| 單一 split | train 81 / valid 20 / test 26 | 結果受 split 運氣影響很大 |
| 重訓不可完全重現 | 目前 log 已記錄 CUDA / DataLoader / lr precision 等造成差異 | 單次 seed 結果不夠穩 |
| F1 定義不夠正式 | 目前 F1 由 Ultralytics precision / recall 後處理得到 | 論文需要固定 IoU、confidence、matching rule |
| 實驗命名與 artifacts 混雜 | 部分 `scores.csv` 來源是 `--weights`，但 train folder 是另一次重訓痕跡 | 審稿或後續複查時難追溯 |
| 缺少 non-YOLO baseline | 目前主線幾乎都是 YOLOv12n + preprocessing | 很難證明選 YOLO 是合理決策 |
| 缺少錯誤分析 | 目前主要是分數表 | 不足以支撐 medical imaging paper 的可信性 |

## 建議正式 protocol

### 原則 1：舊實驗全部降級為 pilot

目前 `exp001` 到 `exp014` 的功能是幫我們找研究方向：

- AdamW / imgsz=1024 / batch=16 是合理主線。
- CLAHE 與 PS-KDE-inspired KDE-CDF 是值得保留的 preprocessing 候選。
- test set 曾經被用來看排名，因此不應再把這批結果當 final paper result。

論文寫法應該是：

> Pilot experiments suggested that radiograph preprocessing, especially CLAHE and KDE-CDF mapping, may improve F1-oriented tooth detection. We therefore conducted a new controlled evaluation under a locked-test and cross-validation protocol.

### 原則 2：建立 locked final test

建議保留目前 `data/test/images` 的 26 張作為 locked final test。

規則：

- 不能再用 test set 選 preprocessing。
- 不能再用 test set 選 confidence threshold。
- 不能再用 test set 選 YOLO / Faster R-CNN / RetinaNet。
- 最後只在 protocol 完成、方法固定後，重訓 final model，評一次 test。

這是我們模擬 DENTEX hidden test 的最小可行版本。

### 原則 3：在 train + valid 上做 5-fold cross-validation

目前可用於開發的方法池是原 train 81 + valid 20，共 101 張。

建議：

- 將 101 張建立 deterministic 5-fold split。
- 每 fold 約 80 張 train、20/21 張 validation。
- 所有方法使用同一組 fold。
- 所有 preprocessing dataset 都要按同一組 fold 產生對應 list，不要每個 preprocessing 自己切。
- 主要排名使用 5-fold validation mean F1。
- 同時回報 std，必要時加 bootstrap CI。

正式報告表格應該長這樣：

| method | preprocessing | mean F1 | std F1 | mean precision | mean recall | mean AP50 | mean AP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv12n | raw | | | | | | |
| YOLOv12n | CLAHE | | | | | | |
| YOLOv12n | PS-KDE-inspired KDE-CDF | | | | | | |
| Faster R-CNN ResNet-FPN | raw | | | | | | |
| RetinaNet ResNet-FPN | raw | | | | | | |

### 原則 4：F1 必須重新定義成固定規則

如果我們論文主指標要看 F1，應該避免只用 Ultralytics validation object 裡的 aggregate precision / recall 推出 F1，而是明確定義：

- detection counted as TP if IoU >= 0.5 and class matches。
- 每個 ground-truth box 最多 matching 一個 prediction。
- predictions 先經過 NMS。
- confidence threshold 只能在 validation fold 上決定。
- 每 fold 選出 threshold 後，用該 threshold 計算 fold validation F1。
- final test threshold 由 5-fold validation 的平均最佳 threshold 或預先固定值決定，不能用 test 調。

建議報告：

- Primary：F1@IoU=0.5。
- Secondary：precision@IoU=0.5、recall@IoU=0.5、AP50、AP50-95。
- Optional：不同 confidence threshold 的 PR curve。

### 原則 5：模型與 preprocessing 的比較要公平

所有方法要固定：

- 同一組 fold。
- 同一批 train / val list。
- 同一個 image size 或明確說明差異。
- 同一個 training budget，例如 YOLO 80 epochs；Faster R-CNN / RetinaNet 用 equivalent iterations 或 early stopping rule。
- 同一套 F1 evaluator。
- 同一套 artifact layout。

如果 Faster R-CNN / RetinaNet 因框架不同不能完全同 budget，至少要在表格註明：

- backbone。
- pretrained weights。
- batch size。
- epochs / iterations。
- learning rate。
- augmentation。
- GPU。

### 原則 6：要有 ablation，不只是 model zoo

我們的主題不是「哪個模型最大」，而是 limited-data panoramic dental X-ray tooth detection 中 preprocessing 對 F1 的影響。

最低限度 ablation：

| ablation group | 必跑項目 | 目的 |
| --- | --- | --- |
| preprocessing | raw / CLAHE / PS-KDE-inspired KDE-CDF | 驗證 preprocessing 是否穩定改善 F1 |
| detector family | YOLO / Faster R-CNN / RetinaNet | 驗證 YOLO 是否合理 |
| threshold | fixed threshold vs validation-tuned threshold | 確認 F1 不是靠 test threshold 調出來 |
| seed stability | 至少 top methods 跑 3 seeds | 確認不是單次重訓運氣 |

### 原則 7：要做錯誤分析

參考 Tuzoff 的做法，正式結果至少要整理：

- False positives：例如 implant / bridge / artifact / jaw edge / overlapping root 被誤判為 tooth。
- False negatives：例如 low contrast、blur、crowded / overlapped teeth、partial tooth、posterior region、cropped boundary。
- 定位錯誤：IoU 不足但大致位置正確。
- Preprocessing failure：CLAHE 是否放大噪聲；KDE-CDF 是否造成局部 contrast 過度平滑。

建議每個 final method 至少輸出：

- 10 張 TP/FP/FN visualization。
- 每張圖標示 confidence、IoU、是否 matched。
- 一份 `error_cases.csv`，欄位包含 image、method、fold、error_type、confidence、iou、notes。

## 建議專案結構調整

目前建議新增一層 `protocols/` 與 `data/splits/`，把正式實驗和舊實驗分開。

```text
retain-thesis/
├── data/
│   ├── dataset.yaml
│   └── splits/
│       ├── locked_test.txt
│       └── cv5_seed20260525/
│           ├── fold0_train.txt
│           ├── fold0_val.txt
│           ├── fold1_train.txt
│           ├── fold1_val.txt
│           └── ...
├── configs/
│   ├── protocol/
│   │   └── cv5_locked_test.yaml
│   └── experiment/
│       └── official/
│           ├── yolo12n_raw_cv5.yaml
│           ├── yolo12n_clahe_cv5.yaml
│           ├── yolo12n_kde_cv5.yaml
│           ├── fasterrcnn_resnet50_fpn_raw_cv5.yaml
│           └── retinanet_resnet50_fpn_raw_cv5.yaml
├── experiments/
│   ├── pilot/
│   └── official/
│       └── cv5_locked_test/
├── reports/
│   ├── TRUSTWORTHY_EXPERIMENT_PROTOCOL.md
│   ├── CV5_RESULTS.md
│   └── ERROR_ANALYSIS.md
└── src/
    ├── split.py
    ├── run_cv.py
    ├── eval_f1.py
    └── summarize_cv.py
```

## 第一階段落地順序

### Step 1：凍結 protocol

產出：

- `configs/protocol/cv5_locked_test.yaml`
- `data/splits/locked_test.txt`
- `data/splits/cv5_seed20260525/fold*_train.txt`
- `data/splits/cv5_seed20260525/fold*_val.txt`

驗證：

- 每張影像只出現在一個 fold 的 validation。
- locked test 不出現在任何 train / validation fold。
- 如果有 patient id，必須 patient-level split；如果沒有 patient id，至少做 filename-level duplicate check。

### Step 2：建立正式 evaluator

產出：

- `src/eval_f1.py`

必須支援：

- fixed IoU threshold，預設 0.5。
- fixed confidence threshold 或 validation-tuned threshold。
- one-to-one matching。
- per-image TP / FP / FN。
- aggregate precision、recall、F1。
- 輸出 `predictions.csv`、`matches.csv`、`scores.json`。

### Step 3：建立 CV runner

產出：

- `src/run_cv.py`

必須支援：

- 給一個 official experiment config。
- 自動跑 fold0 到 fold4。
- 每 fold 輸出到獨立資料夾。
- 每 fold 記錄 seed、dataset yaml、weights、metrics、commit hash。
- 不碰 locked test。

### Step 4：重跑 YOLO preprocessing ablation

先跑最重要的三組：

1. `YOLOv12n + raw`
2. `YOLOv12n + CLAHE`
3. `YOLOv12n + PS-KDE-inspired KDE-CDF`

這一步完成後，我們才知道 preprocessing 的改善是否跨 fold 穩定。

### Step 5：補 non-YOLO baselines

先跑：

1. `Faster R-CNN + ResNet50-FPN`
2. `RetinaNet + ResNet50-FPN`

這兩個對應 DENTEX / DIFF 常見 baseline。DETR / RT-DETR 可以先列為 optional，因為小資料集可能不穩且工程成本更高。

### Step 6：final locked test

只有在 Step 4 / Step 5 完成後才做：

- 按 CV 結果選定 final method。
- 用 train + valid 101 張重訓 final model。
- 用 locked test 26 張評估一次。
- 寫入 `reports/FINAL_TEST_RESULT.md`。

## 論文結果呈現建議

正式論文應該有四張核心表：

1. Dataset table：train / validation / test 或 CV fold 統計。
2. CV main result：mean F1 ± std，並列 precision / recall / AP50。
3. Ablation table：raw vs CLAHE vs PS-KDE-inspired KDE-CDF。
4. Final locked test table：只列最終模型和主要 baselines，不做重新選擇。

也應該有兩種圖：

1. PR curve 或 F1-confidence curve。
2. Qualitative error analysis：TP / FP / FN 範例。

## 對目前專案的結論

目前專案已經有好的 pilot signal：CLAHE 的 F1 最高，PS-KDE-inspired KDE-CDF 的 mAP50 最高，raw YOLO 也有強 baseline。但如果要往 B 級會議前進，下一步不是直接追加模型，而是先把實驗 protocol 鎖住。

最重要的一句話：

> 從現在開始，test set 不能再用來做決策；所有方法選擇都要移到 5-fold validation，最後 test 只當一次性 generalization check。


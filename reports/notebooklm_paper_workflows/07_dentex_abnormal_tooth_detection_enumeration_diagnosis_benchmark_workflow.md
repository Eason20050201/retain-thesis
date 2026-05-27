# DENTEX 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Notebook：`retain-thesis dental X-ray references`
- Notebook ID：`fb7f7397-8ef5-4164-8b54-9ef61cf6b82a`
- Source：`07_dentex_abnormal_tooth_detection_enumeration_diagnosis_benchmark.pdf`
- Source ID：`0d354af9-969a-4187-8c30-6ba2abd2f830`

整理時採用逐段提問，不一次要求總結全文。已完成的提問面向包含：論文身份確認、研究背景與缺口、資料集與標註、方法與訓練流程、評估指標與結果、錯誤分析與限制。NotebookLM 回答中的 citations 均指向上述 DENTEX source ID。

## 1. 論文身份

- 論文名稱：DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-rays
- 任務類型：panoramic dental X-ray 上的 abnormal tooth multi-label detection benchmark
- 主要任務層級：
  - quadrant detection
  - tooth enumeration detection
  - diagnosis detection
- 診斷類別：
  - caries
  - deep caries
  - periapical lesion
  - impacted tooth
- 論文定位：benchmark / challenge paper，而不是只提出單一新模型。
- 發表脈絡：DENTEX challenge 與 MICCAI 2023 相關。

## 2. 研究背景與問題

DENTEX 的背景是 panoramic dental X-ray 在牙科診斷與治療規劃中很重要，但人工判讀耗時、費力，而且會受到臨床經驗與影像品質影響。若能建立可信的 AI benchmark，就可以更系統性地比較不同 dental detection 方法。

本文要解決的核心問題不是單純「偵測牙齒」，而是讓模型同時做到：

- 定位異常牙齒
- 判斷牙齒所在象限
- 判斷牙齒編號
- 判斷具體病理類別

這比我們目前 `retain-thesis` 的單病徵 single-class detection 更複雜，但它對我們有很高參考價值，因為它示範了 dental detection benchmark 應該如何定義任務、資料切分、評估與排名。

## 3. 文獻缺口與研究動機

本文指出既有公開 dental dataset 有幾個不足：

- 有些資料集提供牙齒或顎骨 segmentation，但沒有具體病理診斷標籤。
- 有些資料集的 abnormality label 太粗，不能直接對應臨床診斷需求。
- 異常牙齒偵測、牙齒編號與診斷之間仍缺少完整 benchmark。
- 真實世界中常有 partial annotation 或資料稀缺問題，因此 benchmark 需要反映這種情況。

對我們的啟發是：即使我們現在是小型、單病徵資料集，也要把「可信 benchmark」這件事講清楚。也就是說，研究價值不只來自模型新穎，而是來自嚴格實驗 protocol 與可重現評估。

## 4. 資料集與標註方式

DENTEX 使用來自土耳其三家醫院的 panoramic X-ray，影像由同型設備拍攝，對象為 12 歲以上病患。

資料規模與標註：

| 類型 | 數量 | 說明 |
| --- | ---: | --- |
| 完整標註影像 | 1005 | 包含 quadrant、enumeration、diagnosis |
| quadrant-only | 693 | 只有象限標註 |
| quadrant + enumeration | 634 | 有象限與牙齒編號 |
| unlabeled images | 1571 | 可用於預訓練 |
| 總影像數 | 3903 | 含完整、部分與無標註資料 |

完整標註資料切分：

| Split | 數量 |
| --- | ---: |
| training | 705 |
| validation | 50 |
| hidden testing | 250 |

標註設計採三階層：

1. quadrant
2. quadrant + tooth enumeration
3. quadrant + tooth enumeration + diagnosis

對我們的啟發：

- 我們目前沒有 patient ID，因此只能做 image-level 5-fold CV，這點要明確寫出。
- 我們雖然沒有 hidden server，但已經用 `data/test/images` 建立 locked final test，概念上對應 DENTEX 的 hidden test 精神。
- DENTEX 的 partial annotation 不適用於我們目前資料，但它提醒我們：資料與標註限制應該獨立寫清楚。

## 5. 研究方法

DENTEX 是 benchmark paper，因此它不只描述一個模型，而是整理官方 baseline、參賽方法與 SOTA detector。

官方 baseline：

- HierarchicalDet
- diffusion-inspired multi-stage hierarchical object detection
- backbone 使用 Swin-Transformer
- 使用 FPN 與 decoder refinement
- 能同時利用完整標註與部分標註資料
- 對 unlabeled images 使用 SimMIM 自監督預訓練

高分方法的主要方向：

- 第一名 He L.：sequential hybrid pipeline，先用 U-Net / SE U-Net 做 instance segmentation，再用 DINO / YOLOv8 / DiffusionDet 等模組做 detection 與 diagnosis。
- 第二名 Mei S.：YOLOrtho，基於 YOLOv8 修改，加入 CoordConv、疾病二元分類頭、Hungarian post-processing。
- 第三名 Choi K.：DETDet，使用 Mask R-CNN 做 enumeration，並 ensemble DiffusionDet 與 DINO 做 diagnosis。

與我們相關的模型線索：

- YOLOv8 被高分方法採用，而且不是只原樣使用，而是加上牙齒任務特化設計。
- Faster R-CNN 與 RetinaNet 在 DENTEX 中被列為比較 baseline，但不是最強方法。
- DiffusionDet / DINO / Transformer 類方法在 diagnosis 或 subtle lesion detection 上有優勢，但工程成本較高。

## 6. 影像前處理與資料增強

DENTEX 不是一篇專門討論影像前處理的論文，因此常規影像增強細節並不完整。

明確提到或可整理出的做法：

- 部分方法使用 ROI crop，並保留邊界上下文。
- 有方法使用 pseudo-label 處理 healthy teeth 或 unlabeled data。
- 有方法使用 copy-paste 類 augmentation，例如將牙齒貼到對側位置。
- 高分方法重點不在 CLAHE / gamma / KDE 這類 intensity preprocessing，而在 task-specific architecture、segmentation-guided detection、ensemble、pseudo-labeling。

對我們的啟發：

- 我們目前的 preprocessing ablation 可以成立，但不能把它包裝成 DENTEX 本文主張的核心方法。
- DENTEX 更支持我們後續加入 error analysis、ROI/context、pseudo-label 或 segmentation-guided detection 作為第二階段研究方向。
- 若只用 YOLO，我們可以先嘗試 task-specific post-processing 或 ROI/context 設計，而不是一開始就跳到大型 Transformer / diffusion detector。

## 7. 訓練流程與實驗設定

官方 baseline 的訓練資訊：

- backbone 先用 1571 張 unlabeled panoramic X-ray 透過 SimMIM 預訓練。
- 完整模型 end-to-end train 40000 iterations。
- 使用 705 張完整標註影像處理完整 detection task。
- 同時利用 693 張 quadrant-only 與 634 張 quadrant-enumeration partial labels。
- batch size：16
- optimizer：AdamW
- learning rate：2.5e-5
- hardware：single NVIDIA RTX A6000
- baseline 不使用 cross-validation，也不使用 ensemble。

對我們的啟發：

- 我們目前資料量遠小於 DENTEX，因此 5-fold CV 比單次 train/valid split 更必要。
- DENTEX 使用 hidden test；我們使用 locked test，目的相同：避免 model selection 污染 final test。
- DENTEX baseline 不用 cross-validation 是因為它有較大資料與 challenge hidden test；我們不能照抄這點。

## 8. 評估指標

DENTEX 使用 object detection 標準指標：

- AP
- AP50
- AP75
- AR

並在三個子任務上計算指標：

- quadrant
- enumeration
- diagnosis

總共形成 12 個評估指標，最後用 mean rank 做 challenge ranking，並使用 Wilcoxon signed-rank test 檢查方法間差異是否顯著。

對我們的啟發：

- 我們目前 primary metric 是 `F1@IoU=0.5`，這和 DENTEX 的 AP/AR 排名不同，但並不衝突。
- 論文中仍應同時報告 AP50 / AP50-95，避免審稿人覺得只看單一 threshold。
- 我們可以在 discussion 裡說明：因資料量小、單病徵任務與 clinical false positive / false negative trade-off，本研究以 validation-selected `F1@IoU=0.5` 作為主要比較指標。

## 9. 實驗結果

主要結果：

- He L. 排名第一，使用 segmentation-guided sequential hybrid pipeline。
- Mei S. 排名第二，使用 modified YOLOv8 / YOLOrtho。
- HierarchicalDet baseline 排名第三附近，顯示官方 baseline 已有很強競爭力。
- Choi K. 使用 DiffusionDet + DINO 的互補 ensemble，在 recall 方面有優勢。
- Faster R-CNN 與 RetinaNet 這類傳統 detector 整體弱於 modern Transformer / diffusion / heavily modified YOLO 方法。

對我們的啟發：

- 我們把 Faster R-CNN 與 RetinaNet 放進 baseline 是合理的，但不應預期它們必然贏 YOLO。
- 若 YOLO + preprocessing 的提升有限，第二階段可以考慮 segmentation-guided detection 或 DINO / RT-DETR 類方法。
- 對 single-class small-data 任務，模型排名可能更不穩，因此我們更需要 mean/std 的 5-fold CV。

## 10. 錯誤分析與研究限制

DENTEX 的 discussion / limitation 提供幾個重要觀察：

- 顯著病灶如 impacted tooth 相對容易偵測。
- 細微病灶如 periapical lesion 更能拉開方法差距。
- 純 detection pipeline 可能在密集牙齒 instance separation 上出錯。
- segmentation-guided detection 可以降低 instance merging 問題。
- 不同臨床用途需要不同 precision / recall trade-off。
- panoramic X-ray 對某些疾病不是最終診斷依據，可能需要 intraoral X-ray。
- AP 等 aggregate metric 可能掩蓋特定牙位或特定錯誤類型。
- 未來應加入更細緻的 confusion matrix 或錯誤類型分析。

對我們的啟發：

- 我們的 `ERROR_ANALYSIS.md` 不能只列分數，要區分 false positive、false negative、localization error。
- 我們應該把 low contrast、overlap、partial tooth、posterior、cropped boundary 等錯誤類別固定下來。
- 由於我們目前是單病徵，沒有 enumeration confusion matrix，但可以做 per-image TP/FP/FN 與錯誤原因分類。

## 11. 對 retain-thesis 的直接啟發

### 可直接採用

- locked test 概念：DENTEX 使用 hidden test；我們使用 `data/test/images` 作為 locked final test。
- 主要模型選擇只看 validation / CV，不看 final test。
- 報告 AP50、AP50-95、precision、recall 與 F1，而不是只報單一數字。
- 增加 error analysis，說明模型錯在哪裡。
- 清楚標記資料來源、標註格式、split 與限制。

### 可作為第二階段方向

- segmentation-guided detection：先分離 tooth/ROI，再偵測病徵。
- pseudo-labeling：若未來有 unlabeled panoramic X-ray，可以考慮。
- ROI/context crop：病徵周圍保留一定上下文。
- task-specific post-processing：處理 duplicate、overlap、牙齒相鄰位置混淆。
- Transformer / diffusion detector：可列 optional，不一定放第一輪正式實驗。

### 不適合直接照抄

- DENTEX 是 multi-label / multi-task benchmark，我們目前是 single-class detection。
- DENTEX 資料量比我們大很多，我們不能只用單次 validation 排名。
- DENTEX 的 mean rank 是 challenge ranking，我們論文不需要照做。
- DENTEX 的 partial annotation learning 不適用於目前資料狀態。

## 12. 本專案檢核表

### 論文敘事

- [ ] 說明我們不是提出大型 benchmark，而是在小型資料上建立可信 single-class detection protocol。
- [ ] 說明 dental panoramic X-ray detection 的背景。
- [ ] 說明 DENTEX 顯示 hidden test / benchmark design 對可信度的重要性。
- [ ] 說明我們採用 locked test + 5-fold CV 對應小資料集情境。

### 實驗設計

- [x] 固定 locked test。
- [x] 建立 5-fold CV。
- [x] primary metric 設為 `F1@IoU=0.5`。
- [ ] 正式報告 AP50 / AP50-95。
- [ ] 完成 YOLO preprocessing ablation 的完整 5-fold CV。
- [ ] 完成 Faster R-CNN / RetinaNet baseline。
- [ ] 不根據 locked test 回頭調整方法。

### 方法延伸

- [ ] 若 YOLO preprocessing 結果穩定，評估 ROI/context crop。
- [ ] 若仍有 instance overlap 或 localization error，考慮 segmentation-guided detection。
- [ ] 若需要更強 baseline，再考慮 DINO / RT-DETR / DiffusionDet。

### 錯誤分析

- [ ] 將 FP 分成 implant / bridge / artifact / jaw-edge / duplicate tooth / other。
- [ ] 將 FN 分成 low contrast / blur / overlap / partial tooth / posterior / cropped boundary / other。
- [ ] 將 localization error 獨立列出。
- [ ] 比較 raw baseline 與 final method 的錯誤分布。

## 13. 可以放進本論文的引用位置

- Introduction：引用 DENTEX 說明 panoramic dental X-ray benchmark 與 abnormal tooth detection 的重要性。
- Related Work：引用 DENTEX 作為 dental enumeration / pathosis detection benchmark。
- Methodology：引用 DENTEX 的 hidden test / challenge protocol，對照我們的 locked test。
- Experiments：引用 DENTEX 的 AP / AR detection metric 設計，說明我們同時報 AP50 / AP50-95。
- Discussion：引用 DENTEX 對 aggregate metric 與錯誤類型分析的限制討論，支撐我們加入 error analysis。

## 14. 一句話總結

DENTEX 對 `retain-thesis` 最重要的價值不是告訴我們要直接換成某個模型，而是示範 dental X-ray detection 論文要如何建立可信 benchmark：清楚資料切分、hidden/locked test、標準 detection metrics、強 baseline、方法比較、以及比單一分數更細的錯誤分析。

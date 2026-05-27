# DENTEX Sequential Framework 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Notebook：`retain-thesis dental X-ray references`
- Notebook ID：`fb7f7397-8ef5-4164-8b54-9ef61cf6b82a`
- Source：`08_miccai2023_dentex_sequential_framework_detection_classification_abnormal_teeth.pdf`
- Source ID：`5123e712-ff63-48cf-9a72-f107be618b3c`

本文件採逐段提問整理，包含身份確認、研究背景與缺口、資料與標註、前處理與資料增強、模型架構、訓練流程、評估結果、限制，以及對 `retain-thesis` 的啟發。

## 1. 論文身份

- 論文名稱：A Sequential Framework for Detection and Classification of Abnormal Teeth in Panoramic X-rays
- 作者：Tudor Dascalu, Shaqayeq Ramezanzade, Azam Bakhshandeh, Lars Bjorndal, Bulat Ibragimov
- 發表脈絡：MICCAI 2023 DENTEX Challenge solution report
- 任務：全景 X 光片上的異常牙齒偵測與分類
- 方法類型：sequential pipeline / object detection + cropped-tooth classification

## 2. 研究背景與問題

本文目標是解決 DENTEX challenge 中的 abnormal tooth detection and classification。它不是提出大型通用 benchmark，而是一個針對競賽任務的多階段技術解法。

本文將任務拆成三個連續階段：

1. 偵測牙齒實例與牙齒編號
2. 過濾健康牙齒
3. 對異常牙齒做多標籤疾病分類

這種設計的重點是將複雜的 end-to-end multi-label detection 問題拆成較容易訓練與分析的子問題。

## 3. 文獻缺口與研究動機

NotebookLM 回答顯示本文是短篇 challenge technical report，沒有完整展開傳統 related work gap。

可從方法設計看出的動機是：

- 單一 end-to-end 模型同時處理定位、編號、健康/異常判斷與疾病分類會很困難。
- 裁切出牙齒區域後再分類，可以讓分類器聚焦在局部病灶。
- 分割模型的 encoder 可能捕捉到齲齒、根尖病變等細部特徵，能輔助後續分類。

## 4. 資料集與標註方式

本文使用 DENTEX challenge dataset 的子集：

| 資料類型 | 數量 | 標註內容 |
| --- | ---: | --- |
| tooth detection subset | 634 | bounding box、tooth number、quadrant |
| abnormal diagnosis subset | 705 | abnormal tooth bounding box、tooth number、quadrant、disease labels |
| 總子集 | 1039 | challenge subset |

疾病類別包含：

- impacted
- periapical lesion
- caries
- deep caries

對我們的啟發：

- 我們目前不是 multi-label classification，而是 single-class detection，因此不能照抄它的三階段分類輸出。
- 但它支持「先定位，再做局部分析」的研究方向。

## 5. 影像前處理與資料增強

本文對我們最直接有用的是 augmentation 設計。

明確使用的資料增強：

- horizontal flipping
- brightness adjustment
- contrast adjustment
- affine transformations
- random cutouts，遮擋區域最高約 80 x 80 pixels

前處理方式：

- 第一階段偵測後，將預測的牙齒區域裁切成 cropped tooth images。
- cropped tooth images 作為第二、第三階段分類器輸入。

對我們的啟發：

- 昨天 overnight 的 `photo_aug`、`affine_aug`、`cutout_aug` 可以明確連回這篇。
- 若要繼續做 YOLO 主線，可以先把 brightness/contrast、affine、cutout 視為 DENTEX sequential framework inspired augmentation。
- ROI crop / cropped-tooth analysis 更適合放在第二階段，不必一開始打斷目前 YOLO detection 主線。

## 6. 模型架構

本文使用三階段循序架構：

### Stage 1：Tooth Instance Detection

- 使用 Faster R-CNN
- 輸出 tooth bounding boxes
- 同時輸出 confidence score、tooth number、quadrant

### Stage 2：Healthy Tooth Filtering

- 對 stage 1 裁切出的 tooth crop 做 binary classification
- 判斷健康或異常
- 模型結合 U-Net encoder 與 VGG16 feature path

### Stage 3：Abnormal Tooth Classification

- 對異常牙齒做 multi-label classification
- 分類 impacted、periapical lesion、caries、deep caries
- 架構同樣結合 pretrained U-Net encoder 與 VGG16

額外設計：

- 先訓練 U-Net 做 caries / periapical lesion segmentation。
- 保留 U-Net encoder 權重，作為分類階段的病灶特徵來源。

## 7. 訓練流程與實驗設定

已知訓練設定：

- U-Net segmentation model 使用 BCE loss 與 Dice loss 的平均。
- 第二、三階段 classification model 使用 BCE loss。
- 有使用 on-the-fly augmentation。

文中未明確提供：

- learning rate
- optimizer
- batch size
- epochs
- hardware

對我們的啟發：

- 這篇的訓練細節不完整，因此不適合直接照抄超參數。
- 它比較適合作為 pipeline 設計與 augmentation source，而不是作為完整 reproducible training recipe。

## 8. 評估指標

本文依任務使用不同 metrics：

- detection：AP、AP50、AP75
- filtering / classification：F1 score

主要結果：

| 階段 | 任務 | 指標 |
| --- | --- | --- |
| Stage 1 | Faster R-CNN tooth detection | AP 0.49、AP50 0.91、AP75 0.46 |
| Stage 2 | healthy / abnormal filtering | F1 0.71 |
| Stage 3 | abnormal multi-label classification | F1 0.76 |

對我們的啟發：

- 對 detection 報 AP/AP50/AP75 是合理的。
- 對 binary / multi-label classification 報 F1 是合理的。
- 我們目前是 detection task，因此 primary metric 可維持 `F1@IoU=0.5`，同時補 AP50 / AP50-95。

## 9. 錯誤分析與研究限制

NotebookLM 回答指出本文沒有完整的 error analysis 或 limitations。

可明確列出的限制：

- 文章篇幅短，缺少細緻錯誤分析。
- 缺少完整超參數。
- Faster R-CNN 與 VGG16 架構相對舊。
- Stage 1 AP 僅 0.49，表示整體 pipeline 可能受第一階段 detector 影響。

對我們的啟發：

- 我們不能只學 pipeline，不補 error analysis。
- 若我們做第二階段 crop classifier，必須同時分析錯誤傳遞：第一階段漏偵測會讓第二階段無法補救。

## 10. 對 retain-thesis 的直接啟發

### 可直接採用

- brightness / contrast adjustment
- affine augmentation
- random cutout augmentation
- detection metric 補充 AP50 / AP75 / AP50-95
- classification task 可使用 F1 score

### 可作第二階段方向

- YOLO 先做 tooth/finding detection，再裁切 ROI 做細分類或 confidence refinement。
- 加入 cropped-tooth classifier 檢查 YOLO 的 false positive。
- 使用 segmentation encoder 或 U-Net-style auxiliary feature 做病灶特徵提取。
- 設計健康/異常二元過濾器，降低 false positives。

### 不適合直接照抄

- 直接把 Faster R-CNN + VGG16 當主線，因為方法較舊且 stage 1 AP 不高。
- 直接照抄超參數，因為文中未提供完整訓練細節。
- 將三階段 multi-label pipeline 套到目前 single-class detection，會讓第一輪正式實驗工程量過大。

## 11. 本專案檢核表

- [ ] 在 preprocessing source 文件中將 `photo_aug`、`affine_aug`、`cutout_aug` 連到此文。
- [ ] 正式 CV 中保留 `affine_aug`，因為 fold0 screening 表現進前三。
- [ ] `cutout_aug` 雖可作文獻支持，但若 5-fold 不穩定，不應放進 final method。
- [ ] 未來若增加 second-stage model，可參考 cropped tooth / ROI crop pipeline。
- [ ] 若採用 second-stage classifier，必須額外報告 error propagation。

## 12. 一句話總結

這篇對 `retain-thesis` 的核心價值是提供 DENTEX challenge 中可引用的 augmentation 與 sequential pipeline 依據：短期可支撐 brightness/contrast、affine、cutout 實驗；中期可延伸成 YOLO detection 後的 ROI crop / second-stage classification。

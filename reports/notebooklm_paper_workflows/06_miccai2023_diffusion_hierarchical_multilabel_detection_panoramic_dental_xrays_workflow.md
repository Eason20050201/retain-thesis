# Diffusion-Based Hierarchical Dental Detection 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Notebook：`retain-thesis dental X-ray references`
- Notebook ID：`fb7f7397-8ef5-4164-8b54-9ef61cf6b82a`
- Source：`06_miccai2023_diffusion_hierarchical_multilabel_detection_panoramic_dental_xrays.pdf`
- Source ID：`1fe4eeae-92dd-439f-8ac1-85673a61dde5`

## 1. 論文身份

- 論文名稱：Diffusion-Based Hierarchical Multi-Label Object Detection to Analyze Panoramic Dental X-rays
- 作者：Ibrahim Ethem Hamamci, Sezgin Er, Enis Simsar 等
- 發表：MICCAI 2023 / DENTEX challenge baseline context
- 任務：panoramic dental X-ray 上的 hierarchical multi-label object detection
- 方法類型：DiffusionDet-inspired hierarchical detector + Swin-Transformer + FPN

## 2. 研究背景與問題

本文目標是自動分析 panoramic dental X-rays，讓模型同時定位異常牙齒並輸出：

- quadrant
- FDI tooth number
- diagnosis

這是一個比 `retain-thesis` 更複雜的 multi-label / multi-task detection 任務。本文的核心價值在於示範如何處理 DENTEX 這種階層式標註資料，而不是只展示單一 YOLO baseline。

## 3. 文獻缺口與研究動機

本文指出：

- 仍缺少能同時做異常牙齒定位、編號與多重診斷的 end-to-end model。
- 醫療資料常缺少完整標註，傳統 detector 不容易利用 partial labels。
- 多任務模型表現通常不如單一任務模型，因此需要能利用階層資訊的架構。

對我們的啟發：

- 我們目前是 single-class detection，不需要直接採用 hierarchical multi-label head。
- 但我們可以引用本文說明 dental X-ray detection 正往 multi-task、partial-label、strong benchmark 發展。
- 這也支持我們把 DETR / DiffusionDet 類方法列為 optional，而不是第一輪主線。

## 4. 資料集與標註方式

本文使用 DENTEX dataset。

| 資料類型 | 數量 | 標註內容 |
| --- | ---: | --- |
| quadrant-only | 693 | 象限 |
| quadrant + enumeration | 634 | 象限、FDI 編號 |
| fully annotated | 1005 | 象限、編號、診斷 |
| unlabeled | 1571 | 用於預訓練 |

診斷類別：

- caries
- deep caries
- periapical lesion
- impacted

對我們的啟發：

- 若未來取得更多未標註 panoramic X-ray，可考慮 self-supervised pretraining。
- 若未來標註不完整，可參考獨立 head / dynamic loss 的思路。

## 5. 前處理與資料增強

本文明確提到：

- training images random crop
- resize
- 使用 1571 張 unlabeled images 做 SimMIM self-supervised pretraining

未明確提到：

- CLAHE
- brightness / contrast
- gamma
- KDE
- flip 等常規 augmentation 細節

對我們的啟發：

- 這篇不是 preprocessing paper。
- 不應把我們的 CLAHE/KDE 實驗直接說成來自本文。
- 它更適合支撐 self-supervised pretraining 或 hierarchical model design。

## 6. 模型架構

本文改良自 DiffusionDet，將 object detection 視為從 noisy boxes 到 real boxes 的 denoising diffusion process。

主要組件：

- Swin-Transformer image encoder
- FPN
- detection decoder
- 三個獨立 classification heads：
  - quadrant head
  - enumeration head
  - diagnosis head

關鍵設計：

- 對 partial labels，凍結未標註類別對應的 classification head。
- 使用 bounding box manipulation。
- 將上一階層模型輸出的高信心 boxes 與 noisy boxes concat，作為下一階段的空間先驗。

對我們的啟發：

- 若未來做 multi-stage / multi-task，可以學它的「上一階段位置先驗」。
- 目前 YOLO 主線無法直接套用 noisy box denoising，但可學「高信心預測框引導第二階段」概念。

## 7. 訓練流程與實驗設定

已知設定：

- 使用 1571 張 unlabeled images 透過 SimMIM 預訓練 image encoder。
- backbone 使用 ImageNet-22k pretrained Swin-Transformer。
- learning rate：2.5e-5
- batch size：16
- iterations：40000
- hardware：single NVIDIA RTX A6000 48GB
- 階層式訓練，後續任務使用前一任務的輸出輔助。

對我們的啟發：

- 這類方法成本明顯高於 YOLO preprocessing ablation。
- 第一輪正式實驗先跑 YOLO、Faster R-CNN、RetinaNet 是合理的。
- 若後續要挑戰更高會議或更強結果，再評估 RT-DETR / DINO / DiffusionDet。

## 8. 評估指標與結果

評估指標：

- AP
- AP50
- AP75
- APm
- APl
- AR

主要結果：

- 本文方法在 quadrant、enumeration、diagnosis 任務上超越 RetinaNet、Faster R-CNN、DETR、基礎 DiffusionDet。
- bounding box manipulation 是消融實驗中最重要的提升來源。
- 傳統 weight transfer 在複雜任務上沒有明顯幫助。

對我們的啟發：

- 正式報告除了 F1，也應補 AP50 / AP50-95。
- 如果只比較 YOLO preprocessing，不應宣稱超越這類 hierarchical detector。
- 可在 future work 中討論 diffusion / transformer detector。

## 9. 錯誤分析與研究限制

NotebookLM 回答顯示本文未提供細緻的混淆矩陣或具體預測錯誤案例。

明確限制：

- 資料量有限。
- 標註不完整，存在 partial labels。
- 方法依賴較複雜的 hierarchical architecture。

對我們的啟發：

- 我們必須補足自己的 error analysis，而不是只引用高階 detector。
- 若資料量小，不應把複雜模型結果過度解讀。

## 10. 對 retain-thesis 的直接啟發

### 可直接採用

- AP / AP50 / AP75 / AR 的報告方式。
- self-supervised pretraining 可作未來方向。
- 對 partial labels 設計獨立 heads / dynamic loss 的想法。

### 可作第二階段方向

- 使用第一階段高信心 boxes 引導第二階段 classifier。
- 嘗試 RT-DETR / DINO / DiffusionDet 作為 high-cost baseline。
- 若未來有多標籤資料，改成 multi-head detector。

### 不適合照抄

- DiffusionDet noisy box denoising 不能直接套到 YOLO。
- 不應在第一輪正式實驗就加入此類工程量高的方法。
- 不應盲目採用 weight transfer，本文消融顯示其效果不一定正面。

## 11. 本專案檢核表

- [ ] 在 paper future work 中列出 DiffusionDet / DINO / RT-DETR。
- [ ] 第一輪正式 CV 保持 YOLO preprocessing + Faster R-CNN / RetinaNet。
- [ ] 若第二階段做 ROI classifier，可嘗試用第一階段高信心 boxes 作為先驗。
- [ ] 不把 DiffusionDet 類方法放進目前 overnight preprocessing 主線。

## 12. 一句話總結

這篇對 `retain-thesis` 的價值在於提供 dental panoramic X-ray 上的 high-cost hierarchical detector 參考，證明現代 diffusion / transformer 方法值得列為未來方向，但第一輪可信實驗仍應先完成 YOLO preprocessing 與基本 detection baselines。

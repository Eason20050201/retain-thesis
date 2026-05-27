# YOLO 原始論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Source：`14_yolo_cvpr2016_you_only_look_once_unified_realtime_object_detection.pdf`
- Source ID：`585d07c3-7dfd-404d-9bfe-3c05cd12d35e`

## 論文身份

- 論文名稱：You Only Look Once: Unified, Real-Time Object Detection
- 作者：Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- 發表：CVPR 2016
- 任務：real-time object detection

## 核心方法

YOLO 將 object detection 視為單一 regression problem，直接由整張影像預測 bounding boxes 與 class probabilities。它捨棄 sliding window / region proposal pipeline，使用單一 CNN end-to-end 完成偵測。

主要設計：

- 將影像分成 `S x S` grid。
- 每個 grid 預測 bounding boxes、confidence 與 class probabilities。
- 使用單一 network 一次推論完成 detection。
- 利用全圖上下文降低背景 false positive。

## 評估與限制

主要優勢：

- 即時速度快。
- Fast YOLO 可達很高 FPS。
- 對背景 false positives 較少。

主要限制：

- 對密集小物件不利。
- 每個 grid 的預測數與類別限制會影響群聚目標。
- localization error 是主要錯誤來源。
- 對罕見長寬比或不常見 configuration 泛化較弱。

## 對 retain-thesis 的引用用途

### 可直接引用

- 作為 one-stage detector / YOLO family 的源頭。
- 解釋為什麼 YOLO 適合 real-time / efficient detection。
- 說明 YOLO 使用全圖 context，可能降低背景 false positives。

### 對我們的限制提醒

- 牙齒與病灶可能密集、重疊、小尺寸，因此 YOLO 類模型需要特別檢查 localization error。
- 我們的 `F1@IoU=0.5` 與 per-image TP/FP/FN 可以直接對應 YOLO 常見錯誤。
- 若 YOLO 對小病灶不穩，後續可加入 FPN、ROI crop、segmentation-guided detection 或更高解析度。

## 本專案檢核表

- [ ] 在 related work 中引用 YOLO 作為 one-stage detector 基礎。
- [ ] 不引用 YOLO 原文來宣稱 YOLOv12n 的最新細節。
- [ ] 在 error analysis 中特別檢查 localization error 與密集/重疊目標。
- [ ] 若小病灶表現差，討論 one-stage detector 的空間解析度限制。

## 一句話總結

YOLO 原始論文對 `retain-thesis` 的作用是支撐「為何使用 YOLO 類 one-stage detector」，並提供理解 localization error、密集小物件與全圖 context 的基礎框架。

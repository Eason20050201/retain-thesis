# U-Net 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Source：`20_unet_miccai2015_biomedical_image_segmentation.pdf`
- Source ID：`9ac1102a-68de-4f26-865c-b4979ab4f742`

## 論文身份

- 論文名稱：U-Net: Convolutional Networks for Biomedical Image Segmentation
- 作者：Olaf Ronneberger, Philipp Fischer, Thomas Brox
- 發表：MICCAI 2015
- 方法類型：biomedical image segmentation network

## 核心方法

U-Net 使用 U-shaped encoder-decoder architecture，結合 contracting path 與 expanding path，並用 skip connections 將高解析度局部資訊傳到解碼端，達成精細 pixel-level segmentation。

主要設計：

- contracting path 提取 context。
- expanding path 恢復空間解析度。
- skip connections 保留局部細節。
- overlap-tile strategy 處理高解析度影像。
- elastic deformation augmentation 支援小樣本訓練。
- weighted loss 強化相鄰物體邊界。

## 評估與限制

優勢：

- 在 biomedical segmentation 任務取得強結果。
- 對小樣本醫學影像很有代表性。
- 推論速度快。

限制：

- 原文未獨立提供 limitations。
- U-Net 是 segmentation model，不是 detection model。

## 對 retain-thesis 的引用用途

### 可直接引用

- 支撐 segmentation-guided detection 的 future work。
- 支撐 DENTEX sequential framework 中使用 U-Net 作前處理/輔助特徵。
- 說明醫學影像小樣本中資料增強與 skip connections 的價值。

### 對我們的實驗意義

- 目前第一輪 YOLO detection 不需要 U-Net。
- 若後續要先 segment tooth / ROI，再做 detection 或 classification，U-Net 是基礎文獻。

## 本專案檢核表

- [ ] 在 future work 中列出 segmentation-guided detection。
- [ ] 若後續建立 tooth segmentation 前處理，引用 U-Net。
- [ ] 不把 U-Net 放入第一輪可信實驗主線。

## 一句話總結

U-Net 對 `retain-thesis` 的作用是支撐未來的 segmentation-guided second-stage pipeline，而不是目前 YOLO preprocessing ablation 的直接方法。

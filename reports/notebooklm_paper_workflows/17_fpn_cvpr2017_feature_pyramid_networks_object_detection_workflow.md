# FPN 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Source：`17_fpn_cvpr2017_feature_pyramid_networks_object_detection.pdf`
- Source ID：`d66204d6-919a-49ce-97f7-3f1190e65da2`

## 論文身份

- 論文名稱：Feature Pyramid Networks for Object Detection
- 作者：Tsung-Yi Lin, Piotr Dollar, Ross Girshick 等
- 發表：CVPR 2017
- 方法類型：multi-scale feature pyramid for object detection

## 核心方法

FPN 利用 CNN 中自然形成的多尺度特徵，透過 top-down pathway 與 lateral connections，把低解析度但語意強的高層特徵，和高解析度但語意弱的低層特徵融合。

主要設計：

- bottom-up pathway 取 ResNet stages 的 feature maps。
- top-down pathway 做 upsampling。
- lateral 1x1 conv 對齊 channel。
- element-wise addition 融合特徵。
- 產生 `P2-P5` 等多尺度 feature maps。

## 評估與限制

優勢：

- 在 COCO detection 上提升表現。
- 對小物件 recall 特別有幫助。
- 比傳統 image pyramid 更有效率。

限制：

- 原文未獨立提供 limitations。
- FPN 本身不解決資料不足、標註錯誤或 domain shift。

## 對 retain-thesis 的引用用途

### 可直接引用

- 支撐 `Faster R-CNN + ResNet50-FPN` 與 `RetinaNet + ResNet50-FPN` baseline。
- 說明 multi-scale features 對 dental X-ray 小病灶可能重要。
- 解釋為什麼小目標偵測不能只依賴單一深層 feature map。

### 對我們的限制提醒

- 若 YOLOv12n 對小病灶或邊界模糊區域不穩，FPN-based detector 是合理比較對象。
- FPN 不等於解決所有小資料問題，仍需 CV 與 error analysis。

## 本專案檢核表

- [ ] 在 baseline method 中引用 FPN。
- [ ] 對小病灶 FN 做 error analysis。
- [ ] 若 Faster R-CNN/RetinaNet 表現較好，檢查是否與 multi-scale feature 有關。

## 一句話總結

FPN 對 `retain-thesis` 的作用是支撐多尺度 detection baseline，特別是面對 panoramic X-ray 中小而模糊的病灶時，提供比單尺度特徵更合理的比較基礎。

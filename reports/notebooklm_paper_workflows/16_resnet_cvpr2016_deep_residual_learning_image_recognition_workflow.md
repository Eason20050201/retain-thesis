# ResNet 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Source：`16_resnet_cvpr2016_deep_residual_learning_image_recognition.pdf`
- Source ID：`20a4430d-2bb9-49ae-8bd2-c3889f934169`

## 論文身份

- 論文名稱：Deep Residual Learning for Image Recognition
- 作者：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- 發表：CVPR 2016
- 方法類型：image recognition backbone / residual learning

## 核心方法

ResNet 解決深網路訓練中的 degradation problem。它不直接學習 `H(x)`，而是學習 residual function `F(x) = H(x) - x`，並透過 shortcut connection 得到 `F(x) + x`。

主要設計：

- residual block
- identity shortcut connection
- projection shortcut for dimension matching
- bottleneck block for deeper networks

## 評估與限制

優勢：

- 成功訓練 50/101/152 層深網路。
- 在 ImageNet 與 COCO 等任務取得強表現。
- ResNet features 可遷移到 detection / segmentation。

限制：

- 在小資料集上極深模型可能過擬合。
- 原文為了專注最佳化問題，未大量使用 dropout 等正則化方法。

## 對 retain-thesis 的引用用途

### 可直接引用

- 作為 ResNet50 backbone 的基礎文獻。
- 支撐 Faster R-CNN / RetinaNet + ResNet50-FPN baseline。
- 說明 residual learning 為何適合作為醫學影像 feature extractor。

### 對我們的限制提醒

- 我們資料很小，不應盲目使用過深模型。
- ResNet50 baseline 是合理比較基準，但未必勝過 YOLOv12n。

## 本專案檢核表

- [ ] 在 baseline description 中引用 ResNet。
- [ ] Faster R-CNN / RetinaNet baseline 寫清楚 backbone 為 ResNet50-FPN。
- [ ] 若使用更深 backbone，須注意小資料過擬合。

## 一句話總結

ResNet 對 `retain-thesis` 的作用是支撐 non-YOLO baseline 的 feature backbone，並提醒我們在小資料集下不要盲目追求更深模型。

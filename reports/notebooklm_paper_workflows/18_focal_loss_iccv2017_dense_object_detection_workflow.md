# Focal Loss / RetinaNet 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Source：`18_focal_loss_iccv2017_dense_object_detection.pdf`
- Source ID：`0e2493ee-3cfb-4286-8bb6-c2229d2d456a`

## 論文身份

- 論文名稱：Focal Loss for Dense Object Detection
- 作者：Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar
- 發表：ICCV 2017
- 方法類型：dense one-stage detector / loss function / RetinaNet

## 核心方法

本文指出 one-stage detector 訓練困難的主要原因是 foreground-background class imbalance。Focal Loss 在 cross entropy 上加入調節因子 `(1 - p_t)^gamma`，降低 easy negatives 的權重，讓模型專注 hard examples。

RetinaNet 架構：

- ResNet backbone
- FPN
- classification subnet
- box regression subnet
- focal loss 處理 dense anchors 的 imbalance

## 評估與限制

優勢：

- RetinaNet 在 COCO 上達到當時強表現。
- Focal Loss 比 OHEM 更有效處理 easy negatives。
- 讓 one-stage detector 能接近或超越 two-stage detector。

限制：

- 原文未獨立提供 limitations。
- 對極高 FPS 場景仍需更輕量 backbone。
- loss 設計與超參數會影響穩定性。

## 對 retain-thesis 的引用用途

### 可直接引用

- 支撐 `RetinaNet + ResNet50-FPN + raw` baseline。
- 說明 class imbalance 對 dense detection 的影響。
- 支撐 focal loss 對大量背景 / 少量病灶的合理性。

### 對我們的實驗意義

- 如果 RetinaNet 表現不錯，可能與 focal loss 處理背景不平衡有關。
- 如果表現不佳，也能反映小型 dental dataset 與泛化限制。

## 本專案檢核表

- [ ] 正式 CV 中保留 RetinaNet baseline。
- [ ] 討論 false positive / false negative 時提到 foreground-background imbalance。
- [ ] 不把 focal loss 當作我們的新方法，而是 baseline 的核心設計。

## 一句話總結

Focal Loss / RetinaNet 對 `retain-thesis` 的作用是支撐 dense one-stage non-YOLO baseline，特別適合用來比較 class imbalance 下的牙科病灶偵測。

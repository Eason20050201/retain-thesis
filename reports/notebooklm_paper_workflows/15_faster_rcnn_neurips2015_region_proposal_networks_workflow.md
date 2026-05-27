# Faster R-CNN 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Source：`15_faster_rcnn_neurips2015_region_proposal_networks.pdf`
- Source ID：`905ca36b-6019-41c0-8347-c5ef100493a9`

## 論文身份

- 論文名稱：Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- 作者：Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
- 發表：NeurIPS 2015
- 方法類型：two-stage object detector

## 核心方法

Faster R-CNN 的核心是提出 Region Proposal Network (RPN)，用神經網路取代傳統 selective search，並讓 RPN 與 Fast R-CNN detection head 共享 convolutional features。

主要設計：

- RPN 產生 candidate regions。
- detection head 對 regions 做 classification 與 bounding box regression。
- 使用 anchors 表示不同 scales 與 aspect ratios。
- 透過 shared feature map 提升速度與效率。

## 評估與限制

優勢：

- Two-stage detection 的經典架構。
- 對定位與候選區域品質有較細緻控制。
- 在 PASCAL VOC 等資料集取得當時 SOTA。

限制：

- 推論流程比 one-stage detector 複雜。
- 對小物件與細部定位仍受 feature stride 影響。
- 原文未獨立討論 limitations，但 anchor 邊界、scale 與訓練策略會影響收斂與結果。

## 對 retain-thesis 的引用用途

### 可直接引用

- 作為 two-stage baseline 的基礎文獻。
- 支撐我們正式實驗中的 `Faster R-CNN + ResNet50-FPN + raw`。
- 和 YOLO one-stage detector 做方法族群對照。

### 對我們的實驗意義

- 如果 Faster R-CNN 表現優於 YOLO，可能表示 region proposal 對牙科小資料更穩。
- 如果表現不如 YOLO，也能作為合理 baseline，說明 YOLO 主線不是沒有比較對象。

## 本專案檢核表

- [ ] 正式 CV 中保留 Faster R-CNN baseline。
- [ ] 報告與 YOLOv12n raw 的 F1 / precision / recall / AP50 / AP50-95 對比。
- [ ] 在 paper related work 中把 Faster R-CNN 定位為 two-stage detector baseline。
- [ ] 不把 Faster R-CNN 當作新方法，而是比較基準。

## 一句話總結

Faster R-CNN 對 `retain-thesis` 的作用是建立可信的 non-YOLO two-stage baseline，讓 YOLO preprocessing improvement 有更清楚的比較座標。

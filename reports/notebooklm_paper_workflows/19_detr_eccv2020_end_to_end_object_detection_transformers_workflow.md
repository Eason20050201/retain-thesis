# DETR 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Source：`19_detr_eccv2020_end_to_end_object_detection_transformers.pdf`
- Source ID：`fa2ec6e5-4a93-49d0-96fe-871e9d6ca745`

## 論文身份

- 論文名稱：End-to-End Object Detection with Transformers
- 作者：Nicolas Carion, Francisco Massa 等
- 發表：ECCV 2020
- 方法類型：Transformer-based end-to-end object detection

## 核心方法

DETR 將 object detection 定義為 direct set prediction problem，移除 anchors 與 NMS，使用 Transformer encoder-decoder 與 Hungarian matching 做一對一預測。

主要設計：

- CNN backbone 提取 image features。
- 加入 positional encoding。
- Transformer encoder 做 global context modeling。
- object queries 透過 decoder 平行輸出物件集合。
- Hungarian matching loss 包含 classification、L1 box loss、GIoU loss。

## 評估與限制

優勢：

- 架構簡潔，end-to-end。
- 不需要 NMS。
- 在 COCO 上可達到與 Faster R-CNN 相近的 AP。
- 對大型物件與全域關係建模有優勢。

限制：

- 小物件表現弱於 Faster R-CNN。
- 訓練週期長，需 300-500 epochs 等級。
- 第一代 DETR 工程成本與訓練成本高。

## 對 retain-thesis 的引用用途

### 可直接引用

- 作為 Transformer-based detector 的基礎文獻。
- 支撐我們把 DETR / RT-DETR / DINO 類方法列為 optional。
- 說明為何目前第一輪不把 DETR 放進可信實驗主線。

### 對我們的限制提醒

- 我們的病灶可能是小物件，原始 DETR 對小物件不占優。
- 小資料集與有限訓練時間下，DETR 不一定是最好的第一輪 baseline。

## 本專案檢核表

- [ ] 在 optional future work 提到 DETR / transformer detector。
- [ ] 第一輪正式 CV 不放 DETR，以避免工程成本拖慢主線。
- [ ] 若後續 YOLO / FPN baseline 穩定後，再評估 RT-DETR 或 DINO。

## 一句話總結

DETR 對 `retain-thesis` 的作用是作為 transformer detector 的理論背景與 future-work 方向，但因小物件與長訓練成本限制，不適合作為第一輪可信實驗的主線。

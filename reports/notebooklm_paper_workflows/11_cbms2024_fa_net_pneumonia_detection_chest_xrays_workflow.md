# FA-Net 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Notebook：`retain-thesis dental X-ray references`
- Notebook ID：`fb7f7397-8ef5-4164-8b54-9ef61cf6b82a`
- Source：`11_cbms2024_fa_net_pneumonia_detection_chest_xrays.pdf`
- Source ID：`0dfceb7e-313d-4c3a-b8d5-99cda9df4556`

## 1. 論文身份

- 論文名稱：FA-Net: A Fuzzy Attention-aided Deep Neural Network for Pneumonia Detection in Chest X-Rays
- 發表：CBMS 2024
- 任務：chest X-ray pneumonia detection
- 方法類型：attention-aided classification network
- 注意：本文不是 dental paper，也不是 object detection paper。

## 2. 研究背景與問題

本文目標是建立胸部 X 光 pneumonia CAD 系統，處理：

- binary classification：normal vs pneumonia
- multi-class classification：normal vs bacterial pneumonia vs viral pneumonia

對 `retain-thesis` 的關係：

- 可作為醫學 X 光 classification / attention model 的 cross-domain reference。
- 不適合作為 dental detection 或 YOLO preprocessing 的核心文獻。

## 3. 資料集與訓練

資料集：

- Kermany chest X-ray dataset
- 5856 張影像
- pneumonia 3883
- normal 1349
- train 5232，其中 10% validation
- test 624

前處理與 augmentation：

- resize to 256 x 256 x 3
- random rotation
- translation
- flip
- zoom / scaling

訓練：

- Adam optimizer
- learning rate：0.0001
- batch size：48
- cross-entropy loss
- epochs：50
- hardware：NVIDIA TESLA P100

## 4. 方法架構

FA-Net 主要設計：

- binary classification 使用 DenseNet 類 backbone。
- multi-class classification 使用 ResNet50。
- 提出 FCSSAM：Fuzzy Channel Selective Spatial Attention Module。
- 結合 channel attention 與 spatial attention。
- 使用 Richards sigmoid 為特徵通道打分，保留 top-k% 重要 channels。

對我們的啟發：

- 如果未來做 second-stage classifier，可考慮 attention module。
- 若全景 X 光背景雜訊太多，channel/spatial attention 可作為降噪策略。
- 不必在第一輪 YOLO detection 實驗中加入。

## 5. 評估指標與結果

評估指標：

- accuracy
- precision
- recall
- F1-score
- AUC

主要結果：

- binary classification accuracy 97.15%
- multi-class classification accuracy 79.79%
- 相較 CNN baseline，多元分類 accuracy +3.51%、F1 +2.89%

對我們的啟發：

- 醫學影像分類 paper 常同時報 precision、recall、F1、AUC。
- 我們 detection paper 不必報 AUC，但 F1/precision/recall 的溝通方式可借鏡。

## 6. 研究限制

本文限制：

- class imbalance 影響 multi-class performance。
- 部分影像品質不佳。
- feature pruning 還可優化。
- attention mechanism 尚未充分整合 global semantic features。

對我們的啟發：

- 小資料與類別不平衡必須寫入 limitations。
- 如果未來做 classifier，應同時觀察 local lesion feature 與 global context。

## 7. 對 retain-thesis 的直接啟發

### 可直接採用

- 在 discussion 中引用醫學 X 光分類常用的 precision / recall / F1。
- 若做 second-stage classifier，可考慮 attention-based feature filtering。

### 可作第二階段方向

- YOLO detection 後對 ROI crop 做 binary classifier。
- 加入 channel attention / spatial attention 降低背景干擾。
- 對二元過濾與多類分類使用不同 backbone。

### 不適合照抄

- 不適合當作 dental detection baseline。
- 不適合支撐 YOLO preprocessing ablation。
- 不適合將胸部 X 光 classification 結果直接外推到牙科 X 光 detection。

## 8. 本專案檢核表

- [ ] Related work 中若提及 attention-based medical image model，可簡短引用。
- [ ] 第一輪正式實驗不加入 FA-Net。
- [ ] 若未來做 ROI classifier，再考慮 attention module。

## 9. 一句話總結

FA-Net 對 `retain-thesis` 的價值是 cross-domain medical X-ray attention-classification 參考；它可啟發 second-stage ROI classifier，但不是本輪 YOLO detection 或 preprocessing 實驗的核心來源。

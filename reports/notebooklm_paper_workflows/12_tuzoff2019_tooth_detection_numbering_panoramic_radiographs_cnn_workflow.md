# Tuzoff 2019 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Notebook：`retain-thesis dental X-ray references`
- Notebook ID：`fb7f7397-8ef5-4164-8b54-9ef61cf6b82a`
- Source：`12_tuzoff2019_tooth_detection_numbering_panoramic_radiographs_cnn.pdf`
- Source ID：`2aad7ac4-8a23-47bb-abfb-2670f7253bbf`

## 1. 論文身份

- 論文名稱：Tooth detection and numbering in panoramic radiographs using convolutional neural networks
- 作者：Dmitry V. Tuzoff, Lyudmila N. Tuzova 等
- 發表：Dentomaxillofacial Radiology, 2019
- 任務：panoramic radiographs 上的 tooth detection 與 FDI tooth numbering
- 方法類型：Faster R-CNN + VGG-16 classification + heuristic post-processing

## 2. 研究背景與問題

本文主張牙齒偵測與編號是牙科診斷、數位病歷、健康追蹤與治療計畫的重要前置任務。自動化系統可減少人工判讀時間，也可作為後續病理偵測的前處理。

對 `retain-thesis` 的關係：

- 我們不是做 tooth numbering，但 tooth detection / localization 的基礎問題相似。
- 本文的錯誤分析與 context-aware crop 對我們後續 error analysis 與 second-stage ROI 設計有參考價值。

## 3. 文獻缺口與研究動機

本文指出過去方法多依賴手工特徵、傳統影像處理與 SVM，且常用 bitewing 或 CT，而非 panoramic radiographs。本文發表時，CNN object detection 架構在牙科領域尚未成熟應用。

研究缺口可整理為：

- 傳統特徵工程對影像變異敏感。
- 既有研究較少處理 panoramic radiograph。
- 缺少同時完成 tooth localization 與 FDI numbering 的完整 CNN pipeline。

對我們的啟發：

- Related work 可將此文放在「CNN-based panoramic tooth detection / numbering」脈絡中。
- 我們的貢獻不應說是首次使用 detection，而應強調小資料集、單病徵與可信 evaluation protocol。

## 4. 資料集與標註方式

資料集：

- 1574 張成人 panoramic radiographs
- 隨機選自俄羅斯診所
- 1352 張 training
- 222 張 testing

標註：

- 5 位放射科專家提供標註
- 每顆牙齒繪製 bounding box
- 標示 FDI tooth number

排除條件：

- 僅處理具自然牙根的牙齒
- 排除人工植牙、固定牙橋等人工構造

對我們的啟發：

- 我們的 error analysis 中應單獨列出 implant / bridge / artifact，因為這些物件在牙科 X 光中容易造成誤判。
- 如果我們沒有排除人工構造，必須在 FP 類別中明確分析。

## 5. 前處理與資料增強

前處理：

- detection 後將 tooth crop 裁切出來。
- 裁切時會擴大範圍，保留鄰近結構作為 context。
- context-aware cropping 對 tooth numbering sensitivity 有明顯幫助。

資料增強：

- 文中提到使用影像擴增，約提升 2% sensitivity。
- 具體 augmentation 種類未完整列出。

對我們的啟發：

- 如果未來做 second-stage crop classifier，crop 不應只切 tight box，應保留一定周邊 context。
- 對目前 YOLO 主線，可把「context」轉化為不要過度裁切原圖，或做 ROI crop 時保留 margin。

## 6. 模型架構

本文系統包含兩個模組：

1. Tooth detection module
   - Faster R-CNN
   - VGG-16 backbone
   - RPN 產生 region proposals
   - 輸出牙齒 bounding boxes

2. Tooth numbering module
   - VGG-16 classifier
   - 將 cropped tooth image 分成 32 個 FDI classes
   - 輸出 confidence scores

後處理：

- 使用牙齒空間排列規則的 heuristic algorithm。
- 確保每個 tooth number 的排列合理。
- 選擇總 confidence 最高的編號組合。

對我們的啟發：

- 即使模型輸出合理，也可以用 clinical/domain rules 修正不合理結果。
- 我們目前是 single-class detection，沒有 tooth number uniqueness，但仍可用後處理處理 duplicate predictions。

## 7. 訓練流程與實驗設定

已知設定：

- detection 與 numbering CNN 都用 ImageNet pretrained weights 初始化。
- 所有層都允許 fine-tuning。
- detection module initial learning rate = 0.001，並使用 exponential decay。
- classification module batch size = 64。
- detection 使用客製 Faster R-CNN Python implementation。
- classification 使用 Keras / TensorFlow。

對我們的啟發：

- 使用 pretrained weights 是牙科小資料集常見策略。
- 我們正式實驗需要比本文更完整記錄 epochs、batch、imgsz、optimizer、seed、device。

## 8. 評估指標與結果

評估指標：

- tooth detection：sensitivity、precision
- tooth numbering：sensitivity、specificity
- 32 類 FDI 編號採 one-against-all 方式計算

主要結果：

| 任務 | 結果 |
| --- | --- |
| Tooth detection | sensitivity 99.41%、precision 99.45% |
| Human expert reference | sensitivity 99.80%、precision 99.99% |
| Tooth numbering | sensitivity 98.00%、specificity 99.94% |

輔助發現：

- context-aware crop 讓 numbering sensitivity 提升約 6 個百分點。
- augmentation 讓 sensitivity 提升約 2 個百分點。
- heuristic post-processing 讓 sensitivity 進一步提升約 0.5 個百分點。

對我們的啟發：

- 除 AP 與 F1 外，medical/dental 文獻也常使用 sensitivity / precision / specificity。
- 我們若面向牙科審稿人，可在結果中補 recall/sensitivity 的解讀。

## 9. 錯誤分析與研究限制

False negatives：

- root remnants
- orthodontic / orthopedic devices 遮擋
- severely impacted teeth
- highly overlapped teeth

False positives：

- implant 被誤認為自然牙
- bridge 被誤認為自然牙
- multi-root teeth 或大型修復物導致多餘 bounding boxes

Numbering errors：

- 缺牙時，模型容易把牙齒誤認成缺失的相鄰牙。
- 臼齒區域較容易錯。
- 孤立牙或牙冠嚴重損壞會降低分類依據。

研究限制：

- 排除 implants 與 fixed bridges，因此真實世界泛化有限。
- 使用較舊的 Faster R-CNN / VGG-16 架構。

對我們的啟發：

- 我們的 error taxonomy 應包含 implant、bridge、overlap、duplicate tooth、artifact。
- 不應把 implant/bridge 當成「不管」的資料，而要列入 FP 分析。

## 10. 對 retain-thesis 的直接啟發

### 可直接採用

- context-aware cropping 概念
- sensitivity / precision / specificity 的臨床溝通方式
- duplicate prediction / implant / bridge / overlap 的錯誤類別
- domain-rule post-processing 的概念

### 可作第二階段方向

- YOLO detection 後保留 margin 做 ROI crop，再做二階段判斷。
- 對 duplicate predictions 加入後處理規則。
- 對 false positives 做 hard negative mining，特別是 implant / bridge / artifact。

### 不適合直接照抄

- 不應排除人工構造物來簡化任務，除非論文中明確定義 exclusion criteria。
- 不應以 VGG-16 / older Faster R-CNN 作為主力模型，除非只是 baseline。
- 不應用單次 split 替代我們的 5-fold CV。

## 11. 本專案檢核表

- [ ] 在 `ERROR_ANALYSIS.md` 中加入 implant / bridge / overlap / duplicate tooth 類別。
- [ ] 在 discussion 中說明 sensitivity/recall 的 clinical meaning。
- [ ] 若做 ROI crop，保留 context margin。
- [ ] 對 duplicate boxes 檢查 one-to-one matching 與後處理策略。
- [ ] 不把舊式 two-stage model 當作 final claim，只作為文獻背景或 baseline。

## 12. 一句話總結

Tuzoff 2019 對 `retain-thesis` 最重要的價值在於提供 panoramic tooth detection 的經典 CNN pipeline、context-aware crop 的實驗證據，以及可直接轉化成我們錯誤分析分類的 FP/FN 來源。

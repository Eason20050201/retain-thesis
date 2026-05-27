# DENTECT 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Notebook：`retain-thesis dental X-ray references`
- Notebook ID：`fb7f7397-8ef5-4164-8b54-9ef61cf6b82a`
- Source：`13_yuksel2021_dental_enumeration_treatment_detection_panoramic_xrays.pdf`
- Source ID：`14a94e7a-6c09-447f-9d6f-6812d5e9bbee`

## 1. 論文身份

- 論文名稱：Dental enumeration and multiple treatment detection on panoramic X-rays using deep learning
- 作者：Atif Emre Yuksel, Sadullah Gultekin, Enis Simsar 等
- 發表：Scientific Reports, 2021
- 方法名稱：DENTECT
- 任務：panoramic X-ray 上的牙齒編號與多種治療偵測
- 方法類型：semantic segmentation + YOLO-based object detection 的三階段系統

## 2. 研究背景與問題

本文指出 panoramic dental X-ray 在臨床診斷與治療計畫中重要，但人工判讀會受到影像重疊、變形、疲勞與視覺誤差影響。研究目標是透過 AI 自動完成牙齒編號，並標示需要治療介入的牙齒。

本文處理的任務比 `retain-thesis` 更複雜：

- 象限分割
- FDI tooth enumeration
- 多種 treatment detection

我們目前是 single-class detection，但這篇可以支撐「panoramic X-ray + YOLO + detection metric + augmentation」的文獻脈絡。

## 3. 文獻缺口與研究動機

本文指出：

- 過去許多 dental X-ray object detection 研究只聚焦單一疾病。
- 既有方法不一定能直接提供治療處置建議。
- 缺少能將每顆牙齒位置與多種 treatment labels 配對的公開資料。
- 臨床上需要自動編號與 treatment detection，而不只是單一 lesion detection。

對我們的啟發：

- 我們的研究缺口可以更謹慎地寫成：在小型資料與單病徵情境下，現有 dental detection 文獻的方法與評估流程未必直接適用，因此需要建立可信的 validation protocol。
- 不宜宣稱我們解決完整臨床 treatment planning；我們只做其中一個 detection subtask。

## 4. 資料集與標註方式

資料來源：

- 1005 張 panoramic X-ray
- 來自土耳其 Medipol Mega 大學醫院
- 排除 12 歲以下病患，因混合齒列會造成影像複雜
- 排除嚴重模糊、重疊或失真的影像

標註層級：

| 任務 | 影像數 | 標註內容 |
| --- | ---: | --- |
| 象限分割 | 500 | 四象限 segmentation |
| 牙齒編號 | 600 | FDI tooth number |
| 治療偵測 | 1005 | 五種 treatment labels |

治療類別：

- apical lesion treatment
- filling
- root canal treatment
- extraction
- surgical extraction

資料切分：

- 85% training
- 15% testing

對我們的啟發：

- DENTECT 用單次 85/15 split；我們資料更小，因此採用 5-fold CV 是合理補強。
- 它排除低品質影像；我們若沒有排除，應在 limitations 與 error analysis 中說清楚。

## 5. 影像前處理與資料增強

本文的前處理與 augmentation 對我們很有用。

前處理：

- 先將全景影像分成四個象限。
- 切象限時保留邊緣餘裕。
- 牙齒編號模型前，將各象限旋轉對齊到第一象限方向，以降低上下顎與方向差異。
- 治療偵測模型使用未旋轉的象限影像。

資料增強：

- 50% random horizontal flip
- random brightness
- random contrast
- random saturation
- photometric distortion

資料不平衡處理：

- 對稀有 treatment classes 加入額外 augmentation images。
- 使用 negative sampling，加入沒有 treatment labels 的樣本。
- 在分類 loss 中使用與類別頻率成反比的 class weights。

對我們的啟發：

- `photo_aug` 可以明確連到 DENTECT。
- 若我們的病徵正負樣本不平衡，negative sampling 與 class weighting 可以列為後續方法。
- 象限/局部 crop 是 second-stage 的好方向，但第一輪不必加入，以免工程量過大。

## 6. 模型架構

DENTECT 是三階段架構：

1. 象限分割模型
   - 使用 semantic segmentation。
   - 透過 KNN 與 convex hull 平滑邊界。

2. 牙齒編號模型
   - 使用 YOLO 架構。
   - 輸入旋轉對齊後的象限影像。
   - 預測 FDI tooth numbers。

3. 治療偵測模型
   - 使用 YOLO 架構。
   - 輸入未旋轉的象限影像。
   - 預測五種 treatment labels。

後處理：

- 使用 IoU 將牙齒編號模型與治療偵測模型的 bounding boxes 配對。
- 產生同時包含 tooth number 與 treatment suggestion 的結果。

對我們的啟發：

- YOLO 在 dental panoramic X-ray task 中已有前例。
- 先分區再偵測可以降低單張全景影像中的物件數。
- 對我們目前單病徵而言，可以先以 full image YOLO 為主，再把 quadrant crop 放入後續 ablation。

## 7. 訓練流程與實驗設定

已知訓練設計：

- loss 包含 classification、confidence、bounding box size、initial point 等項目。
- 治療偵測使用 class weights 來處理類別不平衡。
- 使用 photometric augmentation 與 random flip。

文中未明確提供：

- learning rate
- batch size
- epochs
- optimizer 細節

對我們的啟發：

- 可引用其 augmentation 與不平衡處理，但不能照抄訓練超參數。
- 我們應把自己的 training config、seed、fold、package versions 寫清楚，補足 reproducibility。

## 8. 評估指標與結果

評估指標：

- 象限分割：IoU、Dice coefficient、pixel accuracy
- 牙齒編號與治療偵測：AP、AR
- 偵測細部：AP0.50、AP0.50:0.95、AR0.50:0.95

主要結果：

| 任務 | 主要結果 |
| --- | --- |
| 象限分割 | mean IoU 86.2%、Dice 92.5%、pixel accuracy 96.4% |
| 牙齒編號 | AP0.50 最高 89.4% |
| 治療偵測 | AP0.50 最高 59.0% |

本文也指出 class weights 與 negative sampling 有助於改善稀有類別，例如根尖病變相關 treatment。

對我們的啟發：

- 我們正式報告可以同時提供 F1、AP50、AP50-95。
- 若 class imbalance 明顯，應在方法或 limitation 裡說明處理策略。

## 9. 錯誤分析與研究限制

本文可整理出的限制：

- 資料量只有 1005 張。
- 類別極度不平衡，例如 filling 遠多於 extraction。
- 排除兒童與嚴重瑕疵影像，限制真實世界泛化能力。
- 模型對同一顆牙齒只給單一 treatment suggestion，未反映臨床上可能有多種處置選擇。

對我們的啟發：

- 我們也要誠實說明資料量小與資料來源限制。
- 若資料沒有涵蓋多機構、多設備，不能過度宣稱泛化能力。
- 若未來做多病徵或多處置，應改成 multi-label 或 probability distribution，而不是單一強制標籤。

## 10. 對 retain-thesis 的直接啟發

### 可直接採用

- photometric augmentation：brightness、contrast、saturation
- AP50 / AP50-95 / AR 作為 F1 以外的輔助指標
- class weighting 與 negative sampling 作為類別不平衡處理方向
- 在論文中明確描述排除條件或資料品質問題

### 可作第二階段方向

- 先做 quadrant / ROI crop，再執行 YOLO detection。
- 將 full panoramic image task 拆成 smaller local tasks。
- 對 YOLO false positives 做 negative sampling 或 hard negative mining。
- 若後續取得 tooth-level labels，可做 tooth-number-aware detection。

### 不適合直接照抄

- 不應照抄 85/15 single split 作為正式結果，因我們資料更少。
- 不應把 treatment detection 敘事套到我們目前的 single-class finding detection。
- 不應主動排除困難影像來拉高分數，除非明確定義 exclusion criteria 並在論文中說明。

## 11. 本專案檢核表

- [ ] 在 preprocessing source 文件中將 `photo_aug` 連到 DENTECT。
- [ ] 正式 CV 報告除 F1 外，同時列 AP50 / AP50-95。
- [ ] 若資料有嚴重不平衡，加入 class imbalance discussion。
- [ ] 若後續做 second-stage，優先考慮 quadrant/ROI crop，而非直接換大型模型。
- [ ] 在 limitations 中說明小資料集、單病徵、單來源或有限來源的泛化限制。

## 12. 一句話總結

DENTECT 對 `retain-thesis` 的核心價值是證明 YOLO 類架構可用於 panoramic dental X-ray 的牙齒與 treatment detection，並提供 photometric augmentation、局部/象限處理、class imbalance handling 與 AP 系列指標作為我們正式實驗與論文敘事的依據。

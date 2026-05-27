# PS-KDE 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Notebook：`retain-thesis dental X-ray references`
- Notebook ID：`fb7f7397-8ef5-4164-8b54-9ef61cf6b82a`
- Source：`00_ps_kde_pixel_substitution_kernel_density_estimation_plosone2024.pdf`
- Source ID：`c0c0c749-803b-41d6-9377-0916c846893c`

## 1. 論文身份

- 論文名稱：Enhancing semantic segmentation in chest X-ray images through image preprocessing: ps-KDE for pixel-wise substitution by kernel density estimation
- 作者：Yuanchen Wang, Yujie Guo, Ziqi Wang, Linzi Yu, Yujie Yan, Zifan Gu
- 發表：PLOS ONE, 2024
- 任務：chest X-ray semantic segmentation
- 方法類型：Kernel Density Estimation based image preprocessing + U-Net / ResNet34 segmentation

## 2. 研究背景與問題

本文關注醫學 X 光影像中，影像前處理對 deep learning segmentation 的影響。作者指出對比度增強對人類判讀與電腦輔助演算法都很重要，但現有方法如 CLAHE 在不同解剖結構與不同資料異質性下不一定穩定。

本文不是 dental paper，也不是 detection paper；它是 chest X-ray segmentation preprocessing paper。

對 `retain-thesis` 的意義：

- 可作為 KDE-based contrast enhancement 的主要文獻來源。
- 不能直接宣稱它證明 KDE 對 dental object detection 有效。
- 我們必須透過自己的 5-fold CV 驗證 KDE-CDF 是否真的改善 F1。

## 3. 文獻缺口與研究動機

本文指出：

- CLAHE 等對比增強方法雖有效，但不同結構與資料分佈下表現不一。
- 臨床資料常是小資料集，深度學習模型需要更穩定的前處理。
- 需要高效率、可解釋、可整合進深度學習流程的影像增強方法。

對我們的啟發：

- 小資料集下，前處理不是單純美化影像，而是可能影響模型能否學到穩定特徵。
- 但前處理是否有效必須以 validation F1 / AP 驗證，不能只靠視覺上看起來更清楚。

## 4. 資料集與標註方式

資料：

- JSRT chest X-ray dataset
- 247 張 PA chest X-rays
- 154 張有 lung nodules
- 93 張無 nodules

標註：

- 使用 SCR dataset 的 binary masks
- 由專業放射科醫師標註
- 分割 5 個結構：
  - heart
  - left lung
  - right lung
  - left clavicle
  - right clavicle

切分：

- 50% training，124 張
- 50% validation/testing，123 張

對我們的啟發：

- 這篇資料量也小，因此可支持 small medical dataset context。
- 但它使用 segmentation masks，我們使用 detection boxes / polygon-to-box evaluation，任務不同。

## 5. PS-KDE 前處理方法

PS-KDE 全名為 Pixel-wise substitution by Kernel Density Estimation。

核心概念：

- 先從 training set 的像素分佈建立 KDE-based probability density function。
- 將影像中的原始 pixel values 替換成該 pixel value 對應的 normalized frequency / density。
- 再做 min-max normalization 到 0-1。

流程：

1. 在訓練資料中估計器官區域 pixel intensity 的 KDE / PDF。
2. 儲存這些 density distribution 作為 prior knowledge。
3. 對新影像逐像素替換：原始 intensity -> 對應 density value。
4. 將替換後影像 normalize 到 0-1。

比較 baseline：

- raw image
- CLAHE

共通處理：

- resize to 256 x 256
- value scale to 0-1
- random augmentation：rotation、horizontal flip、vertical flip、scaling

## 6. 模型架構與訓練流程

模型：

- U-Net
- ResNet34 backbone
- ImageNet pretrained weights
- batch normalization
- zero padding
- 分別為 5 個 organs 訓練 5 個獨立模型

訓練：

- Bayesian optimization
- 5-fold cross-validation for hyperparameter search
- optimizer candidates：RMSprop、Adam、SGD
- loss candidates：BCE、BCE + Jaccard loss、Dice loss
- final best loss：BCE + Jaccard loss
- epochs：50
- cross-validation best batch size：1
- 實際因載入 augmentation images，使用相當於 5 倍的 batch

對我們的啟發：

- 它支持以 cross-validation 選方法，而不是單次 split。
- 它也提醒我們所有 preprocessing 應共用同一組 folds，避免資料切分造成假改善。

## 7. 評估指標與結果

評估指標：

- IoU / Jaccard
- Dice score / F1-score
- precision
- recall
- accuracy

主要結果：

- PS-KDE 在左肺與心臟等較大區域表現較好。
- 左肺 Dice score：PS-KDE 0.780，CLAHE 0.717。
- CLAHE 在較小結構如鎖骨與部分右肺上表現更好。
- accuracy 對不同方法差異不敏感，因此鑑別力有限。
- precision / recall 可幫助分析是過度偵測還是漏偵測。

對我們的啟發：

- 前處理效果可能依目標尺寸與結構不同而變。
- PS-KDE/KDE-CDF 未必適合小病灶；CLAHE 也可能在局部微小目標更有效。
- 這支持我們同時比較 raw、CLAHE、KDE-CDF、ABC-CLAHE，而不是只選一個。

## 8. 錯誤分析與研究限制

本文限制：

- 資料集小，只有 247 張。
- 使用 PNG 影像，缺少 DICOM metadata。
- 模型只訓練 50 epochs，受運算資源限制。
- 缺乏外部驗證。
- 由 training set 建立的 density prior 是否能泛化到外部影像仍未知。
- 在小結構上 PS-KDE 可能造成 precision 下降，表示過度擴張或假陽性。
- 小資料集上的 random flip / rotation 可能帶來副作用。

對我們的啟發：

- 我們的 KDE-CDF 必須在 CV 中檢查 precision / recall trade-off。
- 若 KDE-CDF 提高 recall 但降低 precision，要在 discussion 裡解釋。
- 不應只看 visual enhancement。
- 若未來使用 DICOM 原始資料，前處理應保留灰階深度與 metadata。

## 9. 與 retain-thesis 的關係

### 重要澄清

我們目前實作的 `KDE-CDF` 是 **PS-KDE-inspired**，不是完整 reproduction。

差異：

- PS-KDE 原文針對 chest X-ray segmentation。
- PS-KDE 使用 organ-mask-related pixel distributions。
- 我們目前針對 dental X-ray detection，做的是 KDE/CDF-style intensity remapping。
- 我們沒有完整重現原文五器官 segmentation pipeline。

論文中應寫：

> We implemented a PS-KDE-inspired KDE-CDF intensity remapping preprocessing method, motivated by KDE-based pixel substitution for X-ray image enhancement, and evaluated it under the same 5-fold detection protocol.

不應寫：

> We reproduced PS-KDE.

## 10. 對 retain-thesis 的直接啟發

### 可直接採用

- raw / CLAHE / KDE-based preprocessing comparison
- precision / recall 分析前處理 trade-off
- cross-validation-based preprocessing selection
- 小資料集與缺乏外部驗證的 limitation 寫法

### 可作第二階段方向

- 根據不同病灶大小動態選擇 preprocessing。
- 對大範圍結構使用 KDE-like enhancement，對小型病灶使用 CLAHE 或局部增強。
- 產生 probability heatmap / confidence visualization 觀察前處理影響。

### 不適合照抄

- 不應照抄 chest organ segmentation 設定。
- 不應使用 horizontal flip 破壞 dental quadrant / FDI orientation，除非任務不依賴左右方向。
- 不應忽略 DICOM / original bit depth 對醫學影像前處理的重要性。

## 11. 本專案檢核表

- [ ] 在方法章節把 `KDE-CDF` 寫成 PS-KDE-inspired。
- [ ] 不宣稱完整 reproduction。
- [ ] 在正式 CV 中比較 raw、CLAHE、KDE-CDF。
- [ ] 報告 F1 同時拆 precision / recall。
- [ ] 若 KDE-CDF 只在某些 folds 有效，不能過度宣稱。
- [ ] 在 limitation 中說明目前使用 PNG/JPG 或轉換影像而非完整 DICOM pipeline。

## 12. 一句話總結

PS-KDE 對 `retain-thesis` 的核心價值是提供 KDE-based X-ray contrast enhancement 的文獻依據；但我們目前只能誠實稱為 PS-KDE-inspired KDE-CDF，必須用自己的 5-fold F1 結果證明它是否真的適合 panoramic dental object detection。

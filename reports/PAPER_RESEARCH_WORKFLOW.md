# 論文研究流程與專案檢核表

本文件把一般論文寫作流程，改寫成適合本專案「全景牙科 X 光單病徵 tooth detection」的研究流程。目標不是只把文章寫完，而是讓方法、實驗、結果與討論都能支撐 B 級國際會議投稿。

## 這份流程是否涵蓋所有步驟？

社群貼文中的流程大致涵蓋了論文從研究想法到排版完成的主線：

1. 研究想法
2. 研究背景與依據
3. 文獻蒐集
4. 文獻比較與研究缺口
5. 研究問題
6. 研究架構
7. 研究方法
8. 資料分析
9. 研究結果
10. 討論
11. 結論
12. 前言
13. 摘要
14. 關鍵詞
15. 論文名稱
16. 參考文獻整理
17. 排版與目次生成
18. 附錄

但對本專案這種 AI detection experimental paper 來說，還不夠完整。它缺少幾個會直接影響審稿信任度的部分：

- 研究貢獻定義
- 資料集與標註說明
- locked test 與 5-fold cross validation protocol
- baseline 與 ablation 設計
- metric 定義，尤其是 `F1@IoU=0.5`
- 錯誤分析
- limitations / threats to validity
- reproducibility artifacts

因此，本專案不應只照一般寫作順序走，而要加入可信實驗流程。

## 本專案建議流程

### 1. 研究想法

本專案的研究核心是：在小型全景牙科 X 光資料集上，評估不同影像前處理方法與 detection model 對單一牙科病徵偵測的影響，並以可信實驗流程確認方法是否真的改善 `F1@IoU=0.5`。

檢核項目：

- [ ] 明確描述偵測任務：single-class dental finding detection
- [ ] 明確描述影像來源：panoramic dental X-ray
- [ ] 明確描述主要模型：YOLOv12n
- [ ] 明確描述第一輪比較重點：preprocessing / augmentation
- [ ] 明確標註舊 `exp001-exp014` 為 pilot experiments

### 2. 研究背景與依據

背景要說明為什麼牙科 X 光偵測重要，也要說明為什麼小資料集下的前處理、資料切分、F1 評估會影響可信度。

檢核項目：

- [ ] 說明 panoramic dental X-ray 在臨床與診斷輔助上的價值
- [ ] 說明 tooth / abnormal tooth / dental finding detection 的研究脈絡
- [ ] 說明小資料集容易高估結果，因此需要 locked test 與 cross validation
- [ ] 說明本專案以 F1 為 primary metric 的原因

### 3. 文獻蒐集

文獻蒐集要分成三類：牙科 X 光 detection、醫學影像前處理、通用 object detection baseline。

檢核項目：

- [ ] 牙科 X 光 detection：DENTEX、DENTECT、Tuzoff 等
- [ ] 醫學影像前處理：PS-KDE、AHE/CLAHE、brightness/contrast、gamma 等
- [ ] Object detection baseline：YOLO、Faster R-CNN、RetinaNet、FPN、Focal Loss
- [ ] 記錄每篇文獻與本專案的關係，不只列 citation

### 4. 文獻比較與研究缺口

這一步要把「別人做了什麼」轉成「本專案為什麼還值得做」。

可能研究缺口：

- 多數 dental detection work 以大資料集或多類別任務為主，本專案面對的是小資料集、單病徵、limited annotations。
- 不同前處理在 dental panoramic X-ray single-class detection 上未必有一致結論。
- 只看 mAP 或單次 train/valid split 容易讓小資料集結果不穩定。
- 很多方法沒有嚴格區分 validation-based model selection 與 final locked-test evaluation。

檢核項目：

- [ ] 每個 gap 都要對應至少一組文獻或實驗需求
- [ ] gap 不能只寫「沒有人做過」，而要說明為什麼現有方法不足
- [ ] gap 要導向本專案可以回答的研究問題

### 5. 研究問題

本專案可以設定為以下問題：

RQ1：在小型 panoramic dental X-ray 單病徵偵測任務中，YOLOv12n baseline 的可信 5-fold validation F1 表現為何？

RQ2：文獻啟發的影像前處理方法，如 CLAHE、PS-KDE-inspired KDE-CDF、ABC-CLAHE、photometric augmentation，是否能穩定提升 `F1@IoU=0.5`？

RQ3：相較 YOLOv12n，Faster R-CNN 與 RetinaNet 這類 two-stage / dense detector baseline 是否能提供更好的 generalization？

RQ4：主要錯誤來源是 false positive、false negative，還是 localization error？

檢核項目：

- [ ] 每個 RQ 都要能用現有實驗 protocol 回答
- [ ] 不設定目前資料無法回答的問題，例如 patient-level generalization
- [ ] 明確說明目前是 image-level split，若取得 patient id 才能升級 patient-level split

### 6. 研究貢獻

本專案的貢獻應該務實，不要過度宣稱新模型。

建議貢獻：

- 建立一套適合小型 dental X-ray detection dataset 的 locked-test + 5-fold CV evaluation protocol。
- 系統性比較多種文獻啟發的影像前處理方法在 YOLOv12n 上的 F1 表現。
- 加入 Faster R-CNN 與 RetinaNet 作為 non-YOLO baseline，檢查改善是否只來自 YOLO 特性。
- 提供 per-image TP/FP/FN 與 error analysis，分析錯誤來源。

檢核項目：

- [ ] 貢獻不能寫成「提出全新 SOTA 模型」
- [ ] 每個 contribution 都要能被實驗或文件支撐
- [ ] 貢獻要和 B 級會議的 expected rigor 對齊

### 7. 資料集與標註說明

資料集章節必須清楚說明資料數量、split 來源、標註格式、限制。

目前設定：

- `data/train/images`：81 張
- `data/valid/images`：20 張
- `data/test/images`：26 張 locked final test
- dev pool：train + valid，共 101 張
- official test：只在 final method 決定後使用一次
- label：YOLO polygon / segmentation-style label，evaluation 時轉成 enclosing bounding box

檢核項目：

- [ ] 說明影像總數與 split 數量
- [ ] 說明 locked test 從何時開始凍結
- [ ] 說明沒有 patient id，因此第一版只能做 image-level split
- [ ] 說明 polygon label 轉 bounding box 的規則
- [ ] 說明 duplicate check：basename 與 SHA256

### 8. 可信實驗 Protocol

正式 protocol 為 `cv5_locked_test`。

核心規則：

- seed 固定為 `20260525`
- 使用 5-fold cross validation
- method selection 只看 dev pool 的 validation F1
- primary metric 為 `F1@IoU=0.5`
- locked test 不參與前處理選擇、threshold 選擇、模型選擇
- final test 只在最終方法選定後評估一次

檢核項目：

- [ ] `locked_test.txt` 已固定
- [ ] `fold0` 到 `fold4` deterministic
- [ ] 每張 dev image 恰好出現在一個 validation fold
- [ ] locked test 不出現在任何 fold train/val
- [ ] 所有 preprocessing 與所有模型共用同一組 folds

### 9. 研究方法

方法章節要分清楚：資料前處理、模型、訓練設定、評估方法。

第一輪正式方法：

- YOLOv12n + raw
- YOLOv12n + CLAHE
- YOLOv12n + PS-KDE-inspired KDE-CDF
- YOLOv12n + ABC-CLAHE
- YOLOv12n + affine augmentation
- Faster R-CNN + ResNet50-FPN + raw
- RetinaNet + ResNet50-FPN + raw

檢核項目：

- [ ] 每個 preprocessing 都要有來源或合理動機
- [ ] PS-KDE-inspired KDE-CDF 要註明是 inspired，不是完整 reproduction
- [ ] augmentation 與 preprocessing 要分開描述
- [ ] non-YOLO baseline 要說明為何選 Faster R-CNN 與 RetinaNet
- [ ] DETR / RT-DETR / DiffusionDet 暫列 optional，不放第一輪主線

### 10. Baseline 與 Ablation 設計

Baseline 是讓審稿人相信 improvement 不是偶然的關鍵。

建議設計：

- raw YOLOv12n 作為主 baseline
- CLAHE / KDE-CDF / ABC-CLAHE 檢查前處理貢獻
- affine augmentation 檢查 geometric augmentation 貢獻
- Faster R-CNN / RetinaNet 檢查 architecture baseline
- 最終只把 5-fold mean F1 穩定提升的方法推進 final test

檢核項目：

- [ ] 不用單一 fold 結果直接宣稱 final improvement
- [ ] 不根據 locked test 結果回頭修改方法
- [ ] 報告 mean / std，而不是只報最高 fold
- [ ] raw baseline 必須在所有比較中存在

### 11. 評估指標

本專案 primary metric 是 `F1@IoU=0.5`。

其他輔助指標：

- precision
- recall
- AP50
- AP50-95
- selected confidence threshold
- per-image TP / FP / FN

TP 定義：

- class match
- IoU >= 0.5
- one-to-one greedy matching
- predictions 依 confidence 由高到低配對

檢核項目：

- [ ] confidence threshold 只由 validation fold 決定
- [ ] auto threshold 不讀 locked test
- [ ] duplicate prediction 只能一個算 TP，其餘算 FP
- [ ] missed ground truth 算 FN
- [ ] IoU < 0.5 算 FP + FN，不算 TP

### 12. 資料分析

資料分析不是只列分數，而是解釋分數的可靠性與變異。

檢核項目：

- [ ] 每個正式方法都有 5 folds
- [ ] 報告 mean F1 與 std F1
- [ ] 檢查 precision / recall trade-off
- [ ] 檢查 selected confidence threshold 是否異常
- [ ] 檢查某方法是否只在單一 fold 表現特別好

### 13. 研究結果

結果章節要分成 pilot、CV result、final locked test。

檢核項目：

- [ ] `exp001-exp014` 只放 pilot 或 preliminary observation
- [ ] overnight fold0 preprocessing 結果只當 screening，不當正式結果
- [ ] 正式排名只看 5-fold mean F1
- [ ] final locked test 只放已選定 final method 與少量主要 baseline
- [ ] 明確避免把 test score 當作 model selection 依據

### 14. 錯誤分析

錯誤分析要把模型失敗原因分類，而不是只說準確率不夠。

固定分類：

- false positive：implant / bridge / artifact / jaw-edge / duplicate tooth / other
- false negative：low contrast / blur / overlap / partial tooth / posterior / cropped boundary / other
- localization error：detected but IoU < 0.5

檢核項目：

- [ ] final method 輸出 TP/FP/FN visualization
- [ ] raw baseline 輸出 TP/FP/FN visualization
- [ ] 比較 preprocessing 是否改善 low contrast 或 overlap cases
- [ ] 討論錯誤是否和資料量、標註品質、影像品質有關

### 15. 討論

Discussion 要回答：結果代表什麼、為什麼會這樣、限制在哪裡。

檢核項目：

- [ ] 解釋最有效 preprocessing 的可能原因
- [ ] 解釋 precision / recall 變化
- [ ] 解釋為什麼某些文獻方法在本資料集不一定有效
- [ ] 說明小資料集下 std 的重要性
- [ ] 說明 clinical deployment 不是本研究目標

### 16. Limitations / Threats to Validity

這一節對 B 級會議很重要，因為它讓論文看起來誠實而可信。

建議限制：

- 資料量小
- 單一病徵 / single-class detection
- 目前無 patient id，只能 image-level split
- 標註由 polygon 轉 bounding box，可能影響 localization 評估
- 未做 external validation
- final locked test 只有 26 張，因此只能視為初步 generalization check

檢核項目：

- [ ] 不把 limitations 藏起來
- [ ] 每個限制都說明可能影響哪一種結論
- [ ] 提出合理 future work，但不把 future work 寫成本文已完成

### 17. 結論

Conclusion 要收斂，不要新增新結果。

檢核項目：

- [ ] 回答 RQ1-RQ4
- [ ] 總結最佳方法與其 5-fold mean F1
- [ ] 總結 final locked-test 表現
- [ ] 總結最主要錯誤來源
- [ ] 避免過度宣稱 clinical readiness

### 18. 前言

Introduction 通常最後寫，因為它需要整合整篇論文的動機、gap、方法與貢獻。

建議結構：

1. Panoramic dental X-ray detection 的重要性
2. 小資料集 detection 的挑戰
3. 現有文獻與缺口
4. 本研究問題與方法
5. 本文貢獻

檢核項目：

- [ ] 第一段讓讀者知道問題重要
- [ ] 第二段讓讀者知道問題困難
- [ ] 第三段讓讀者知道現有方法不足
- [ ] 第四段明確說本研究做了什麼
- [ ] 第五段列出貢獻

### 19. 摘要、關鍵詞、標題

這些應該最後寫，因為它們要反映最終實驗結果。

檢核項目：

- [ ] Abstract 包含背景、方法、資料、結果、結論
- [ ] Abstract 不放 pilot-only 結果
- [ ] Keywords 包含 dental panoramic radiograph、object detection、YOLO、cross validation、preprocessing
- [ ] Title 具體，不誇大

可能標題方向：

- Reliable Evaluation of Preprocessing Strategies for Single-Class Dental Finding Detection in Panoramic Radiographs
- Cross-Validated YOLO-Based Dental Finding Detection on Limited Panoramic X-Ray Data
- Preprocessing-Aware Tooth Finding Detection in Panoramic Dental Radiographs with Locked-Test Evaluation

### 20. 參考文獻整理與排版

參考文獻不只是格式，要確認每篇文獻在論文中真的有用途。

檢核項目：

- [ ] 每篇 reference 都至少在 introduction、related work、method、discussion 中被使用
- [ ] 不放與本文無關的 reference 湊數
- [ ] 同一種方法來源只保留最必要的核心文獻
- [ ] 格式依目標會議 template 調整
- [ ] appendix 放補充表格、額外 fold 結果、error case examples

## 總進度 Checklist

### 已完成

- [x] 舊實驗定位為 pilot experiments
- [x] 建立 `cv5_locked_test` protocol
- [x] 固定 locked test
- [x] 建立 5-fold deterministic splits
- [x] 建立 F1 evaluator
- [x] 建立 CV runner 與 summary report
- [x] 建立 preprocessing overnight plan
- [x] 完成 first-round fold0 preprocessing screening
- [x] 整理 preprocessing method sources

### 下一階段

- [ ] 將 fold0 screening 的前幾名推進完整 5-fold CV
- [ ] 完成 raw / CLAHE / KDE-CDF / ABC-CLAHE / affine augmentation 的正式 CV
- [ ] 完成 Faster R-CNN 與 RetinaNet baseline
- [ ] 產生正式 `CV5_RESULTS.md`
- [ ] 選定 final method
- [ ] 只對 locked test 執行一次 final evaluation
- [ ] 產生 `FINAL_TEST_RESULT.md`
- [ ] 產生 `ERROR_ANALYSIS.md`
- [ ] 開始撰寫 paper skeleton

## 寫作順序建議

實際寫論文時，不建議從 Introduction 開始。建議順序如下：

1. Dataset and annotation
2. Experimental protocol
3. Methodology
4. Evaluation metrics
5. Results
6. Error analysis
7. Discussion and limitations
8. Related work
9. Introduction
10. Conclusion
11. Abstract
12. Title and keywords
13. References formatting
14. Appendix

這個順序比較適合本專案，因為目前最大的核心不是文字包裝，而是讓實驗流程可信、結果可重現、結論不超過資料能支撐的範圍。

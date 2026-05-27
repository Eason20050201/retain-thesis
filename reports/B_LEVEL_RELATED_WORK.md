# B 級會議相關論文整理

最後檢查日期：2026-05-25

這份文件聚焦在目前 `retain-thesis` 的方向：以 YOLO 為核心，在牙科全景 X 光影像上做 tooth detection，並且以 F1 score 作為主要評估指標。這裡的目的不是蒐集所有牙科 AI 論文，而是找出「有官方 B 級依據」且和我們投稿目標最接近的論文。

## 官方分級依據

| 會議 | 官方依據 | 本文件採用等級 | 使用方式 |
| --- | --- | --- | --- |
| CBMS | CORE/ICORE Conference Portal：`ICORE2026 Rank: B` | CORE/ICORE B | 最乾淨的 CORE A/B/C 系統下 B 級目標。 |
| BIBM | CCF `交叉/綜合/新興` 類別，B 類 | CCF B | 對 bioinformatics / biomedicine 類型論文很適合，而且已有牙科全景 X 光相關論文。 |
| MICCAI | CCF `交叉/綜合/新興` 類別為 B 類；CORE/ICORE 則列為 A | CCF B / CORE A | 形式上可算 CCF B，但實際難度明顯高於一般 B 級，應視為高標參考。 |

CCF 特別注意事項：CCF 官方說明中，會議論文只計 `Full paper` 或 `Regular paper`；Short paper、Demo paper、Technical Brief、Summary、Findings，以及附屬 Workshop 論文，都不計入目錄評價。因此我們在看會議等級時，必須確認目標是主會議正式長文。

官方來源：

- CORE/ICORE CBMS 頁面：https://portal.core.edu.au/conf-ranks/2350/
- CCF 目錄發布與論文類型說明：https://www.ccf.org.cn/Academic_Evaluation/By_category/2023-03-08/787209.shtml
- CCF 交叉/綜合/新興會議列表：https://www.ccf.org.cn/Academic_Evaluation/Cross_Compre_Emerging/

## 論文篩選規則

相似度依照下列順序判斷：

1. 同影像模態、同大方向任務：牙科全景 X 光 object detection。
2. 同影像模態、相鄰任務：tooth segmentation、dental disease detection、tooth numbering、diagnosis、periodontitis analysis。
3. 同 B 級會議、相鄰方法：醫學影像 detection / segmentation，使用 YOLO、FPN、U-Net、SAM、半監督學習、小物件偵測等方法。

## 優先目標會議：BIBM

BIBM 是目前最重要的目標，因為它是 CCF B，而且已經有多篇與牙科全景 X 光高度相關的論文。

### 1. ACPNet: Enhancing Small-Scale Dieases Detection in Panoramic X-rays

- 會議：BIBM 2024。
- 官方等級：BIBM 為 CCF B。
- 來源：
  - DBLP 紀錄：https://dblp.org/rec/conf/bibm/GaoZCTJL24
  - BIBM 2024 accepted paper list：https://ieeebibm.org/BIBM2024/documents/Accepted%20Papers.pdf
  - BIBM 2024 program：https://ieeebibm.org/BIBM2024/documents/BIBM2024-ProgramNov-30.pdf
- 與我們的相似度：非常高。
- 為什麼重要：
  - 同樣是 panoramic dental X-rays。
  - 同樣屬於 object detection / disease localization 任務。
  - 主題聚焦在 small-scale disease detection，與牙科 X 光低對比、小目標、難定位的問題很接近。
- 他們的貢獻形式：
  - 提出一個具名方法 ACPNet。
  - 任務是 disease-level detection，而不是單純 tooth localization。
  - 這篇能進 BIBM，表示牙科全景 X 光 detection 是可以放進 CCF B 會議的。
- 對我們的啟示：
  - 單純寫「YOLO 可以偵測牙齒」會偏弱。
  - 我們比較有力的角度應該是：在 limited-label dental panoramic X-ray detection 中，radiograph preprocessing 會顯著影響 F1，而且 F1 最好的方法不一定是 mAP50 最好的方法。
  - 官方標題裡 `Dieases` 應是拼字錯誤，但引用時要照原標題保留。

### 2. LeFUNet: UNet with Learnable Feature Connections for Teeth Identification and Segmentation in Dental Panoramic X-ray Images

- 會議：BIBM 2022。
- 官方等級：BIBM 為 CCF B。
- 來源：
  - DBLP 紀錄：https://dblp.org/rec/conf/bibm/RaoNZAHS22
  - 作者頁與 DOI 資訊：https://www.raoyunbo.cn/
- 與我們的相似度：高。
- 為什麼重要：
  - 同樣是 dental panoramic X-ray。
  - 同樣以牙齒為主要分析目標。
  - 任務是 segmentation / identification，不是 bounding-box detection。
- 他們做了什麼：
  - 提出 U-Net 家族模型，加入 learnable feature connections。
  - 論文以方法貢獻為主，不只是套用現成模型。
- 對我們的啟示：
  - BIBM 確實接受 tooth-level panoramic X-ray analysis。
  - 但這篇有明確架構貢獻；如果我們不做新模型，就必須把 preprocessing、F1 行為分析、可重現流程寫得很清楚。

### 3. Utilizing Deep Learning to Opportunistically Screen for Osteoporosis from Dental Panoramic Radiographs

- 會議：BIBM 2022。
- 官方等級：BIBM 為 CCF B。
- 來源：
  - DBLP 紀錄：https://dblp.org/rec/conf/bibm/AnantharamanBNK22
  - BIBM 2022 program：https://ieeebibm.org/BIBM2022/documents/BIBM2022_Program_LasVegas.pdf
- 與我們的相似度：中。
- 為什麼重要：
  - 同樣使用 dental panoramic radiographs。
  - 臨床輸出不同：osteoporosis screening，而不是 tooth detection。
  - 表示 BIBM 接受以牙科全景影像為基礎的臨床應用型研究。
- 對我們的啟示：
  - 臨床動機很重要，不一定只靠模型複雜度。
  - 我們需要說清楚 F1-oriented tooth detection 在有限標註情境下的實際價值。

### 4. GC-UNet: Enhance UNet with GCN for Periodontitis Segmentation

- 會議：BIBM 2024。
- 官方等級：BIBM 為 CCF B。
- 來源：
  - DBLP / BIBM listing：https://dblp.org/db/conf/bibm/bibm2024
  - Proceedings 目錄：https://www.proceedings.com/content/078/078510webtoc.pdf
- 與我們的相似度：中高。
- 為什麼重要：
  - Dental disease segmentation 與我們的 dental X-ray detection 任務相鄰。
  - 再次顯示 BIBM 的 biomedical image analysis 論文常以「具名方法 + 特定臨床目標」包裝。
- 對我們的啟示：
  - 如果我們維持 single-class tooth detection，貢獻必須非常清楚。
  - 若加入 disease / condition layer，會更接近近年 BIBM dental work，但這會是較大的研究範圍改動。

### 5. SegPath-YOLO: A High-Speed, High-Accuracy Pathology Image Segmentation and Tumor Microenvironment Feature Extraction Tool

- 會議：BIBM 2024。
- 官方等級：BIBM 為 CCF B。
- 來源：
  - DBLP / BIBM listing：https://dblp.org/db/conf/bibm/bibm2024
- 與我們的相似度：中低。
- 為什麼重要：
  - 不是牙科，但顯示 YOLO-style medical image tools 可以出現在 BIBM。
  - 這類論文通常會被包裝成實用 biomedical tool，而不是單純報一組 benchmark 分數。
- 對我們的啟示：
  - 若投 BIBM，稿件應該像完整 medical image analysis workflow：dataset、annotation、preprocessing、model、metrics、error analysis、clinical relevance。

## 高標參考：MICCAI

MICCAI 在 CCF `交叉/綜合/新興` 類別中是 B 類，但 CORE/ICORE 把它列為 A。對我們來說，MICCAI 應該當作高標參考，而不是最實際的第一投稿目標。

### 6. Diffusion-Based Hierarchical Multi-Label Object Detection to Analyze Panoramic Dental X-rays

- 會議：MICCAI 2023。
- 官方等級：CCF B，CORE/ICORE A。
- 來源：https://conferences.miccai.org/2023/papers/205-Paper2550.html
- 與我們的相似度：非常高，但研究完整度遠高於目前專案。
- 他們做了什麼：
  - Panoramic dental X-ray object detection。
  - 使用 FDI notation 的 hierarchical annotation。
  - Multi-label abnormal tooth detection，包含 enumeration 與 diagnosis。
  - 將 diffusion-based object detection 改造成適合 hierarchical / partial annotations 的方法。
  - 與 RetinaNet、Faster R-CNN、DETR、DiffusionDet 等主要 detection baseline 比較。
- 對我們的啟示：
  - 這是我們主題方向的高標版本。
  - 若要到 MICCAI 等級，單純 tooth boxes 不夠；需要更豐富標籤、更困難臨床終點，或明確方法創新。

### 7. DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays

- 會議背景：MICCAI 2023 challenge / benchmark。
- 官方等級：MICCAI 背景是 CCF B / CORE A，但 challenge paper 不一定等於主會議 full paper。
- 來源：
  - arXiv：https://arxiv.org/abs/2305.19112
  - Challenge site：https://dentex.grand-challenge.org/
- 與我們的相似度：非常高。
- 他們做了什麼：
  - 定義 panoramic X-rays 上的 abnormal tooth detection、enumeration、diagnosis。
  - 使用 hierarchical labels 與 multi-institutional challenge framing。
- 對我們的啟示：
  - DENTEX 是目前這個領域對「強 panoramic dental X-ray task」的 benchmark-style 參考。
  - 我們目前的 single-class tooth detection 範圍比較窄，因此引用時要明確說明我們的研究範圍與它不同。

### 8. HC-Net: Hybrid Classification Network for Automatic Periodontal Disease Diagnosis

- 會議：MICCAI 2023。
- 官方等級：CCF B，CORE/ICORE A。
- 來源：https://conferences.miccai.org/2023/papers/316-Paper0540.html
- 與我們的相似度：中高。
- 他們做了什麼：
  - 從 panoramic X-ray images 做 periodontal disease classification。
  - 結合 tooth-level 與 patient-level classification，並加入 adaptive noisy-OR mechanism。
  - 針對 radiographic annotations 與 clinical probing ground truth 之間的不一致問題。
- 對我們的啟示：
  - 強 dental AI paper 不只是偵測結構，還會把模型輸出連回 clinical diagnosis 與 label reliability。

### 9. 3D Teeth Reconstruction from Panoramic Radiographs using Neural Implicit Functions

- 會議：MICCAI 2023。
- 官方等級：CCF B，CORE/ICORE A。
- 來源：https://conferences.miccai.org/2023/papers/005-Paper2035.html
- 與我們的相似度：中。
- 他們做了什麼：
  - 使用 panoramic radiographs 作為輸入，但解的是 3D tooth reconstruction。
  - 任務不是 detection，但仍是 dental panoramic radiography 在頂級 medical imaging venue 的例子。
- 對我們的啟示：
  - Dental panoramic X-ray 本身可以成為嚴肅 medical imaging 題材。
  - 但 MICCAI 會期待明確且較強的技術貢獻。

## CORE-B 目標：CBMS

CBMS 是 CORE/ICORE 官方 B。這次沒有找到像 BIBM 的 ACPNet / LeFUNet 那麼貼近牙科全景 X 光的 CBMS 論文，但 CBMS 有不少 medical image detection / segmentation 論文，因此仍是合理的 B 級應用型 medical systems 目標。

### 10. Integrating YOLO and 3D U-Net for COVID-19 Diagnosis on Chest CT Scans

- 會議：CBMS 2024。
- 官方等級：CORE/ICORE B。
- 來源：
  - DBLP CBMS 2024 listing：https://dblp.org/db/conf/cbms/cbms2024.html
  - CORE/ICORE CBMS rank page：https://portal.core.edu.au/conf-ranks/2350/
- 與我們的相似度：中。
- 為什麼重要：
  - 使用 YOLO 作為醫學影像流程的一部分。
  - 將 detection / localization 邏輯與 segmentation / diagnosis component 結合。
- 對我們的啟示：
  - CBMS 可以接受基於既有 deep-learning components 的實用 medical-imaging system。
  - 若投 CBMS，我們應更強調 system design、reproducibility、實際臨床工作流程，而不只強調 detector 架構新穎性。

### 11. FA-Net: A Fuzzy Attention-aided Deep Neural Network for Pneumonia Detection in Chest X-Rays

- 會議：CBMS 2024。
- 官方等級：CORE/ICORE B。
- 來源：https://dblp.org/db/conf/cbms/cbms2024.html
- 與我們的相似度：中低。
- 為什麼重要：
  - 同樣是 X-ray image analysis，但影像是 chest X-ray，不是 dental panoramic X-ray。
  - 任務偏 disease detection / classification，不是 tooth box detection。
- 對我們的啟示：
  - 如果投 CBMS，radiograph-specific preprocessing 這條線是合理的，但需要更強的臨床評估敘事。

### 12. Convolutional Neural Network Architectures and Aggregation Information Models for Pulmonary X-Ray Segmentation

- 會議：CBMS 2024。
- 官方等級：CORE/ICORE B。
- 來源：https://dblp.org/db/conf/cbms/cbms2024.html
- 與我們的相似度：中低。
- 為什麼重要：
  - X-ray segmentation 與 CNN architecture comparison。
  - 支持「在 radiographic images 上做控制變因比較」可以成為 CBMS 型論文。
- 對我們的啟示：
  - 我們的 preprocessing ablation 可以寫成 conference paper，但前提是研究設計清楚、error analysis 充分。

## 非 B 級牙科 YOLO 論文：只能當背景參考

這些論文對 related work 有幫助，但不能用來證明題目已達 B 級。

| 論文 | 會議 | 為什麼有用 | 為什麼不是 B 級證據 |
| --- | --- | --- | --- |
| Tooth Detection and Numbering in Panoramic Radiographs Using YOLOv8-Based Approach | MobiHealth 2023 | 任務很像：YOLO + tooth detection / numbering | 目前未找到官方 B 級證據。 |
| Automated Tooth Detection and Numbering in Panoramic Radiographs Using YOLO | CENTERIS / HCist 2024, Procedia Computer Science | 類似的應用型牙科流程 | 目前在 CORE/CCF 查核下，不能當 B 級證據。 |
| Third Molar Angle Detection in Dental X-Ray Panoramic Radiographs Using YOLO and GoogleNet | ARTIIS 2024 | 窄任務 dental panoramic paper | 目前在 CORE/CCF 查核下，不能當 B 級證據。 |

## B 級相似論文通常具備什麼

從最接近的 BIBM / MICCAI 例子來看，B 級論文通常至少要有下列其中幾點：

- 比單純 object localization 更困難的 clinical endpoint，例如 disease detection、tooth numbering、diagnosis、periodontitis stage、osteoporosis screening。
- 具名方法貢獻，例如 ACPNet、LeFUNet、GC-UNet，或 hierarchical detection framework。
- 清楚說明任務困難點：小目標、低對比、class imbalance、有限標註、partial labels、noisy labels、clinical label mismatch。
- 與強 baseline 做控制比較，而不是只報單一模型結果。
- 不只看一種指標：mAP / AP、precision / recall、F1，若是 segmentation 則還要 Dice / IoU，並搭配 qualitative error analysis。
- 可重現細節完整：dataset split、preprocessing、annotation type、training recipe、thresholding rule、model selection rule。

## 對 `retain-thesis` 的意義

最實際的 B 級路線：

1. 第一目標：BIBM，因為它是官方 CCF B，而且已有最接近的 dental panoramic X-ray 論文。
2. 第二目標：CBMS，因為它是官方 CORE/ICORE B，且接受應用型 medical image systems。
3. 高標參考：MICCAI，因為它雖是 CCF B，但 CORE/ICORE 是 A，實際要求明顯更高。

目前專案的適配情況：

- 我們最強的故事不是「YOLOv12n 可以偵測牙齒」；這對 B 級來說太薄。
- 比較好的故事是：在 limited-data panoramic dental X-ray tooth detection 中，影像前處理會明顯影響 F1，而且 F1 最好的前處理方法不等於 mAP50 最好的前處理方法。
- 這個故事可以連到 B 級常見主題：小型醫學資料集、radiograph contrast、臨床上 precision / recall tradeoff、可重現 model-selection rule。

建議工作標題：

`Preprocessing Matters: An F1-Centered Study of YOLO-Based Tooth Detection in Limited Panoramic Dental X-rays`

論文 related work 最低限度應錨定這幾篇：

- ACPNet, BIBM 2024：最接近的 B 級 panoramic X-ray detection 論文。
- LeFUNet, BIBM 2022：最接近的 B 級 tooth-level panoramic X-ray 論文。
- DENTEX / MICCAI 2023：abnormal tooth detection、enumeration、diagnosis 的 benchmark / high-bar reference。
- Diffusion-Based Hierarchical Multi-Label Object Detection, MICCAI 2023：panoramic dental object detection 的高標方法參考。
- CBMS 的 YOLO + medical imaging examples：支持 CBMS 作為應用型 medical systems 目標。


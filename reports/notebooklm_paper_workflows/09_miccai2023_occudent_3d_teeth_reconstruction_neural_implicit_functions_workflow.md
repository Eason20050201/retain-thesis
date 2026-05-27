# OccuDent 論文研究流程與專案檢核表

本文整理自 NotebookLM 中單一鎖定來源：

- Notebook：`retain-thesis dental X-ray references`
- Notebook ID：`fb7f7397-8ef5-4164-8b54-9ef61cf6b82a`
- Source：`09_miccai2023_occudent_3d_teeth_reconstruction_neural_implicit_functions.pdf`
- Source ID：`a4b3244d-807a-4548-ba20-184c6ac5f743`

## 1. 論文身份

- 論文名稱：3D Teeth Reconstruction from Panoramic Radiographs using Neural Implicit Functions
- 作者：Sihwa Park, Seongjun Kim, In-Seok Song, Seung Jun Baek
- 發表：MICCAI 2023
- 方法名稱：OccuDent
- 任務：從 2D panoramic radiographs 重建 3D teeth shape
- 方法類型：UNet++ tooth segmentation + neural implicit function 3D reconstruction

## 2. 研究背景與問題

本文不是 object detection 論文，而是 2D panoramic X-ray 到 3D teeth reconstruction 的研究。它的核心問題是：如何從單張全景 X 光片中推估牙齒的 3D 幾何結構。

對 `retain-thesis` 的關係：

- 可作為 dental panoramic X-ray deep learning 的 related work。
- 不適合作為我們第一輪 YOLO detection 實驗的直接方法來源。
- 對後續 tooth-level feature conditioning 或 second-stage ROI analysis 有啟發。

## 3. 資料集與標註方式

資料分兩部分：

| 用途 | 資料 | 數量 |
| --- | --- | ---: |
| segmentation pretraining | South Korea AI-Hub panoramic X-rays | 4000 |
| 3D reconstruction | Korea University Anam Hospital PX-CBCT paired data | 39 組 |

3D reconstruction split：

- training：30 組
- validation：2 組
- testing：7 組

對我們的啟發：

- 真正高品質 paired medical data 很稀少，即使 MICCAI paper 也可能只有數十組配對資料。
- 小資料不是不能做研究，但 protocol、限制與問題定義必須誠實。

## 4. 方法架構

OccuDent 是兩階段：

1. 2D tooth segmentation
   - 使用 UNet++
   - 對 panoramic image 做 multi-label segmentation
   - 區分 32 顆牙齒與背景
   - 將個別牙齒裁切成 patches

2. 3D reconstruction
   - 使用 neural implicit functions
   - 對 3D 空間點預測 occupancy
   - 使用 ResNet18 提取 tooth patch embedding
   - 加入 tooth class embedding
   - 提出 Conditional eXcitation (CX) module，根據 tooth type 動態調整特徵

對我們的啟發：

- 如果未來有 tooth number 或 tooth region，牙位資訊可以轉成 embedding，作為疾病判斷的先驗。
- 目前 single-class detection 不需要 3D reconstruction。

## 5. 訓練流程與評估

訓練：

- segmentation 與 reconstruction 分開訓練。
- reconstruction 階段在正規化 3D 空間中採樣 100000 points。
- optimizer：Adam
- learning rate：1e-4
- batch size：10
- epochs：200-300

評估指標：

- Volumetric IoU
- Chamfer-L1 distance
- Normal Consistency
- Volumetric Precision

對我們的啟發：

- 這些指標不適用於我們目前 2D detection 任務。
- 可在 related work 中輕描淡寫，不需要放入 method comparison。

## 6. 實驗結果與限制

主要結果：

- OccuDent 在 Volumetric IoU、Chamfer-L1、Normal Consistency、Volumetric Precision 上優於 3D-R2N2、PSGN、Pix2Vox、X2Teeth。
- 生成 mesh 較平滑，能較好還原臼齒牙根等複雜結構。
- 消融實驗顯示 tooth class embedding 與 CX module 都有幫助。

限制：

- 原文沒有獨立 limitations section。
- 可從設計推論：paired PX-CBCT data 只有 39 組，泛化可能受限。
- 任務依賴 CBCT 3D ground truth，與我們目前資料不同。

## 7. 對 retain-thesis 的直接啟發

### 可直接採用

- 小資料醫學影像研究必須清楚寫資料限制。
- 可以在 related work 中提到 panoramic X-ray 不只用於 2D detection，也可延伸至 3D reconstruction。

### 可作第二階段方向

- tooth class / tooth position embedding
- tooth-aware second-stage classifier
- ROI crop 後加入牙位先驗
- attention / channel weighting 類似 CX module，用牙齒類別或位置動態調整特徵

### 不適合照抄

- 不適合加入第一輪 YOLO detection 實驗。
- 不適合用 3D reconstruction metrics 評估我們的 2D detection。
- 不適合把 OccuDent 寫成我們前處理方法來源。

## 8. 本專案檢核表

- [ ] Related work 可簡短列為 panoramic dental X-ray 進階方向。
- [ ] 第一輪正式實驗不加入 3D reconstruction。
- [ ] 若未來有 tooth numbering labels，可考慮 tooth-position-conditioned classifier。
- [ ] 不把本文作為 preprocessing / YOLO augmentation 來源。

## 9. 一句話總結

OccuDent 對 `retain-thesis` 的價值主要在 related work 與未來 tooth-aware second-stage design；它不是我們目前 YOLO preprocessing 或 2D detection protocol 的直接方法來源。
